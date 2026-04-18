"""GNN-based stroke grouping + CNN classification.

Uses the GNN for edge prediction (which strokes belong together),
then runs the CNN classifier on each group for symbol labeling.
"""

from __future__ import annotations

import math

import torch

from engine.stroke import Stroke, compute_bbox
from engine.renderer import render_strokes
from engine.checkpoint import load_checkpoint
from classifier.inference import SymbolClassifier
from grouper_gnn.model import StrokeGNN
from grouper_gnn.features import (
    compute_node_features,
    compute_edge_features,
    compute_adjacency_mask,
)
from engine.grouper import DetectedSymbol, _postprocess, _DEFAULTS
import config


class GNNGrouper:
    """Stroke grouper: GNN edges → connected components → CNN classify."""

    def __init__(
        self,
        gnn_run: str = "v7",
        classifier_run: str = "v8_32",
        device: str | None = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Load GNN
        ckpt = load_checkpoint("grouper_gnn", gnn_run, device=self.device)
        cfg = ckpt["config"]
        self.gnn = StrokeGNN(**cfg).to(self.device)
        self.gnn.load_state_dict(ckpt["model_state_dict"])
        self.gnn.eval()

        # Load CNN classifier
        self.classifier = SymbolClassifier(run=classifier_run, device=self.device)

    @torch.no_grad()
    def group_and_classify(
        self,
        strokes: list[Stroke],
        stroke_width: float = config.RENDER_STROKE_WIDTH,
        source_size: float | None = None,
        edge_threshold: float = 0.0,
        debug: bool = False,
    ) -> list[list[DetectedSymbol]]:
        """Group strokes via GNN, classify groups via CNN.

        Returns a list with one partition (list of DetectedSymbol).
        """
        if not strokes:
            return [[]]

        # 1. GNN edge prediction
        stroke_dicts = [
            [{"x": p.x, "y": p.y} for p in s.points]
            for s in strokes
        ]

        renders, geo = compute_node_features(stroke_dicts, stroke_width=stroke_width)
        edge_feats = compute_edge_features(stroke_dicts)
        adj_mask = compute_adjacency_mask(stroke_dicts)

        renders = renders.unsqueeze(0).to(self.device)
        geo = geo.unsqueeze(0).to(self.device)
        edge_feats = edge_feats.unsqueeze(0).to(self.device)
        adj_mask = adj_mask.unsqueeze(0).to(self.device)

        edge_scores, _ = self.gnn(renders, geo, edge_feats, adj_mask=adj_mask)
        edge_scores = edge_scores[0]  # (N, N)

        # 2. Dense agglomerative grouping
        groups = self._dense_agglomerative(edge_scores, len(strokes),
                                           threshold=edge_threshold, max_size=4)

        if debug:
            print(f"[grouper_v2] {len(strokes)} strokes → {len(groups)} groups")
            for g in groups:
                scores_str = ""
                if len(g) > 1:
                    pairs = [(edge_scores[i, j].item(), i, j)
                             for idx_a, i in enumerate(g) for j in g[idx_a+1:]]
                    scores_str = " edges=" + ",".join(f"{s:.1f}" for s, _, _ in pairs)
                print(f"  group {g}{scores_str}")

        # 3. Render + classify each group with CNN
        images = []
        size_feats = []
        for group_indices in groups:
            group_strokes = [strokes[i] for i in group_indices]
            images.append(render_strokes(
                group_strokes,
                canvas_size=self.classifier.canvas_size,
                stroke_width=stroke_width,
                source_size=source_size,
            ))
            if self.classifier.use_size_feat:
                all_x = [p.x for s in group_strokes for p in s.points]
                all_y = [p.y for s in group_strokes for p in s.points]
                if all_x:
                    bw = max(all_x) - min(all_x)
                    bh = max(all_y) - min(all_y)
                    sym_diag = math.sqrt(bw * bw + bh * bh)
                    size_feats.append(sym_diag / max(source_size or 800, 1.0))
                else:
                    size_feats.append(0.5)
            else:
                size_feats.append(0.5)

        sf = size_feats if self.classifier.use_size_feat else None
        results = self.classifier.classify_batch(images, size_feats=sf)

        # 4. Build DetectedSymbol list
        symbols = []
        for group_indices, result in zip(groups, results):
            group_strokes = [strokes[i] for i in group_indices]
            bbox = compute_bbox(group_strokes)

            sym_name = result.symbol or "?"
            sym_name = _DEFAULTS.get(sym_name, sym_name)

            symbols.append(DetectedSymbol(
                stroke_indices=sorted(group_indices),
                bbox=bbox,
                symbol=sym_name,
                confidence=result.confidence,
                prototype_distance=result.prototype_distance,
                alternatives=result.alternatives,
            ))

            if debug:
                print(f"  {sorted(group_indices)} → '{sym_name}' "
                      f"conf={result.confidence:.3f} dist={result.prototype_distance:.1f}")

        # 5. Post-process (merge composite symbols like =, div, etc.)
        symbols = _postprocess(symbols)

        return [symbols]

    @staticmethod
    def _dense_agglomerative(
        edge_scores: torch.Tensor,
        n: int,
        threshold: float = 0.0,
        max_size: int = 4,
    ) -> list[list[int]]:
        """Agglomerative grouping with dense clique requirement.

        Starts with each stroke as its own group. Iterates edges from
        strongest to weakest. Two groups merge only if:
        1. The triggering edge score > threshold
        2. ALL cross-edges between the two groups are > threshold
        3. Merged group size <= max_size

        This prevents a single false-positive edge from bridging
        unrelated groups.
        """
        # Collect all candidate edges above threshold, sorted descending
        candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                s = edge_scores[i, j].item()
                if s > threshold:
                    candidates.append((s, i, j))
        candidates.sort(reverse=True)

        # Union-Find with group tracking
        parent = list(range(n))
        groups: dict[int, list[int]] = {i: [i] for i in range(n)}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for _, i, j in candidates:
            ri, rj = find(i), find(j)
            if ri == rj:
                continue

            gi, gj = groups[ri], groups[rj]

            # Size check
            if len(gi) + len(gj) > max_size:
                continue

            # Dense check: ALL cross-edges must be above threshold
            all_dense = True
            for si in gi:
                for sj in gj:
                    if edge_scores[si, sj].item() <= threshold:
                        all_dense = False
                        break
                if not all_dense:
                    break

            if not all_dense:
                continue

            # Merge smaller into larger
            if len(gi) < len(gj):
                ri, rj = rj, ri
                gi, gj = gj, gi
            parent[rj] = ri
            gi.extend(gj)
            del groups[rj]

        return [sorted(g) for g in groups.values()]
