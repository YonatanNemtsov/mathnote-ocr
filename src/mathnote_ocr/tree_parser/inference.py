"""Tree parser inference pipeline.

Architecture:
    TreeParser (abstract)
    ├── SubsetTreeParser  — evidence → cost strategy → Edmonds'
    └── GNNTreeParser     — evidence → GNN refinement → Edmonds'

Shared pipeline (in base class):
1. Sample spatial subsets of symbols
2. Subset model predicts partial tree for each
3. Aggregate predictions into evidence graph
4. Iteratively resolve SEQ conflicts with targeted subsets
5. _evidence_to_tree (subclass-specific) → final tree
6. Tree → LaTeX
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import replace

import torch

from mathnote_ocr.bbox import BBox
from mathnote_ocr.engine.checkpoint import load_checkpoint
from mathnote_ocr.engine.grouper import DetectedSymbol
from mathnote_ocr.latex_utils.relations import compute_features_from_bbox_list
from mathnote_ocr.tree_parser.evidence import (
    aggregate_evidence_soft,
    evidence_to_features,
    jitter_bboxes,
)
from mathnote_ocr.tree_parser.score_tree import score_tree
from mathnote_ocr.tree_parser.subset_model import load_subset_model
from mathnote_ocr.tree_parser.subset_selection import (
    _bbox_edge_dist,
    make_neighborhood_subsets,
    make_spatial_subsets,
    make_xaxis_subsets,
)
from mathnote_ocr.tree_parser.tree_builder import (
    build_tree_from_evidence,
    build_tree_from_scores,
    find_seq_conflicts,
)
from mathnote_ocr.tree_parser.tree_latex import tree_to_latex
from mathnote_ocr.tree_parser.tree_v2 import ROOT_ID, Edge, Node, Symbol, Tree

# A bar is a fraction bar or a minus by its place in the expression; both are
# tried when the classifier gives the other at least this (a minus under a
# fraction's numerator was read 0.52 fraction bar, 0.42 minus) — for the most uncertain
# MAX_BAR_CHOICES bars of an expression (2^n parses)
BARS = ("frac_bar", "-")
MIN_BAR_ALTERNATIVE = 0.1
MAX_BAR_CHOICES = 3

logger = logging.getLogger(__name__)


class TreeParser(ABC):
    """Abstract base: subsets → evidence → _evidence_to_tree → LaTeX."""

    def __init__(
        self,
        subset_run: str = "default",
        device: torch.device | None = None,
        max_subset: int = 8,
        radius_mult: float = 4.0,
        max_iters: int = 3,
        scoring: str = "full_spatial",
        subset_strategy: str = "spatial",
        k_neighbors: int = 5,
        subset_min_size: int = 2,
        subset_max_size: int = 5,
        xaxis_width_mults: list[float] | None = None,
        tree_strategy: str = "edmonds",
        seq_threshold: float = 0.7,
        tta_runs: int = 1,
        tta_dx: float = 0.05,
        tta_dy: float = 0.15,
        tta_size: float = 0.05,
        spatial_penalty: float = 3.0,
        root_discount: float = 0.2,
        weights_dir: str | None = None,
        promote: bool = True,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # promote_symbols before parsing (dot/cdot by position, '-' -> frac_bar by the subset
        # model); False: the labels as given (a grouper that labels in context decides them)
        self.promote = promote
        self.weights_dir = weights_dir
        self.max_subset = max_subset
        self.radius_mult = radius_mult
        self.max_iters = max_iters
        self.scoring = scoring
        self.subset_strategy = subset_strategy
        self.k_neighbors = k_neighbors
        self.subset_min_size = subset_min_size
        self.subset_max_size = subset_max_size
        self.xaxis_width_mults = xaxis_width_mults
        self.tree_strategy = tree_strategy
        self.seq_threshold = seq_threshold
        self.tta_runs = tta_runs
        self.tta_dx = tta_dx
        self.tta_dy = tta_dy
        self.tta_size = tta_size
        self.spatial_penalty = spatial_penalty
        self.root_discount = root_discount

        subset_ckpt = load_checkpoint(
            "tree_subset", subset_run, device=self.device, weights_dir=weights_dir
        )
        self.symbol_vocab: dict[str, int] = subset_ckpt["symbol_vocab"]

        self.subset_model = load_subset_model(subset_ckpt, device=self.device)

        self._subset_cache: dict[tuple, list[tuple]] = {}

    def _make_subsets(self, bboxes: list[list[float]]) -> list[list[int]]:
        """Generate subsets using the configured strategy."""
        if self.subset_strategy == "neighborhood":
            return make_neighborhood_subsets(
                bboxes,
                k_neighbors=self.k_neighbors,
                min_size=self.subset_min_size,
                max_size=self.subset_max_size,
            )
        if self.subset_strategy == "xaxis":
            return make_xaxis_subsets(
                bboxes,
                min_size=self.subset_min_size,
                max_size=self.max_subset,
            )
        return make_spatial_subsets(bboxes, self.max_subset, self.radius_mult)

    @property
    @abstractmethod
    def mode(self) -> str:
        """Short label for logging (e.g. "iter", "gnn")."""

    def _make_symbols(self, detected: list[DetectedSymbol]) -> list[Symbol]:
        """Create tree_v2 Symbol list from DetectedSymbols, carrying stroke_ids."""
        return [
            Symbol(
                id=i,
                name=d.name,
                bbox=BBox(d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h),
                stroke_ids=tuple(s.id for s in d.strokes),
            )
            for i, d in enumerate(detected)
        ]

    @abstractmethod
    def _evidence_to_tree(
        self,
        evidence: dict[str, torch.Tensor],
        symbols: list[Symbol],
    ) -> Tree:
        """Convert aggregated evidence into a Tree."""

    # ── Shared machinery ────────────────────────────────────────────

    def _symbol_id(self, name: str) -> int:
        return self.symbol_vocab.get(name, self.symbol_vocab.get("<unk>", 1))

    def _run_subsets(
        self,
        names: list[str],
        bboxes: list[list[float]],
        subsets: list[list[int]],
    ) -> list[tuple[list[int], dict[str, torch.Tensor], int]]:
        """Run subset model on a list of subsets (batched, cached)."""
        if not subsets:
            return []

        sub_S = self.max_subset

        cached: dict[int, tuple[dict[str, torch.Tensor], int]] = {}
        to_run: list[tuple[int, list[int], tuple]] = []

        v2 = getattr(self.subset_model, "encoding", "v1") != "v1"
        scale = None
        if v2:
            from mathnote_ocr.tree_parser.geometry import writing_scale
            scale = writing_scale(names, bboxes)          # the whole expression's
        for b, subset_indices in enumerate(subsets):
            key = tuple((names[gi], *bboxes[gi]) for gi in subset_indices) + ((scale,) if v2 else ())
            if key in self._subset_cache:
                cached[b] = self._subset_cache[key]
            else:
                to_run.append((b, subset_indices, key))

        if to_run:
            B = len(to_run)
            all_ids = torch.zeros(B, sub_S, dtype=torch.long, device=self.device)
            all_pad = torch.ones(B, sub_S, dtype=torch.bool, device=self.device)
            geo_list = []
            size_list = []
            n_reals = []

            for batch_idx, (_, subset_indices, _) in enumerate(to_run):
                n_sub = len(subset_indices)
                n_reals.append(n_sub)
                for i, gi in enumerate(subset_indices):
                    all_ids[batch_idx, i] = self._symbol_id(names[gi])
                all_pad[batch_idx, :n_sub] = False

                geo, size = self.subset_model.inputs(names, bboxes, subset_indices, scale=scale, S=sub_S)
                geo_list.append(geo)
                size_list.append(size)

            all_geo = torch.stack(geo_list).to(self.device)
            all_size = torch.stack(size_list).to(self.device)
            out = self.subset_model.forward(all_ids, all_geo, all_pad, all_size)

            for batch_idx, (orig_b, subset_indices, key) in enumerate(to_run):
                n_sub = n_reals[batch_idx]
                out_cpu = {k: v[batch_idx].cpu() for k, v in out.items()}
                cached[orig_b] = (out_cpu, n_sub)
                self._subset_cache[key] = (out_cpu, n_sub)

        partial_outputs = []
        for b, subset_indices in enumerate(subsets):
            out_cpu, n_sub = cached[b]
            partial_outputs.append((subset_indices, out_cpu, n_sub))
        return partial_outputs

    @torch.no_grad()
    def promote_symbols(
        self,
        symbols: list[DetectedSymbol],
        confidence_threshold: float = 0.5,
        keep=frozenset(),
    ) -> list[DetectedSymbol]:
        """Use the subset model to disambiguate context-dependent symbols.

        Skipped for backtrack strategy — the builder handles promotion internally.

        Currently handles:
          - `-` → `frac_bar`: if the subset model predicts NUM and DEN
            children for a minus with high confidence, it's a fraction bar.

        Symbols at the positions in *keep* (the pinned ones: the user said
        what they are) keep their names.
        """
        before = [s.name for s in symbols]
        symbols = self._promote(symbols, confidence_threshold)
        for i in keep:
            if symbols[i].name != before[i]:
                symbols[i] = replace(symbols[i], name=before[i])
        return symbols

    def _promote(self, symbols: list[DetectedSymbol], confidence_threshold: float) -> list[DetectedSymbol]:
        N = len(symbols)
        names = [s.name for s in symbols]
        bboxes = [[s.bbox.x, s.bbox.y, s.bbox.w, s.bbox.h] for s in symbols]

        # Dot/cdot promotion runs for all strategies
        self._promote_dot_cdot(symbols, names, bboxes, N, confidence_threshold)
        # Clear subset cache — names may have changed
        self._subset_cache.clear()

        if self.tree_strategy in ("backtrack", "backtrack_collapse"):
            return symbols  # builder handles frac_bar promotion via resolve_frac_bars

        # Rebuild names after dot/cdot promotion may have changed some
        names = [s.name for s in symbols]
        self._promote_frac_bars(symbols, names, bboxes, N, confidence_threshold)

        return symbols

    def _promote_frac_bars(
        self,
        symbols: list[DetectedSymbol],
        names: list[str],
        bboxes: list[list[float]],
        N: int,
        confidence_threshold: float,
    ) -> None:
        """Disambiguate `-` → `frac_bar` using subset model."""
        minus_indices = [i for i, s in enumerate(symbols) if s.name == "-"]
        if not minus_indices:
            return

        names_as_frac = list(names)
        for mi in minus_indices:
            names_as_frac[mi] = "frac_bar"

        subsets = []
        minus_map = []
        for mi in minus_indices:
            neighbors = []
            for j in range(N):
                if j == mi:
                    continue
                d = _bbox_edge_dist(bboxes[mi], bboxes[j])
                neighbors.append((j, d))
            neighbors.sort(key=lambda x: x[1])
            k = min(self.max_subset - 1, len(neighbors), max(5, N - 1))
            subset = sorted([mi] + [n[0] for n in neighbors[:k]])
            subsets.append(subset)
            minus_map.append(mi)

        partials = self._run_subsets(names_as_frac, bboxes, subsets)

        for (subset_indices, out, n_sub), mi in zip(partials, minus_map):
            mi_pos = subset_indices.index(mi)
            parent_probs = torch.softmax(out["parent_scores"][:n_sub], dim=-1)

            has_num = False
            has_den = False

            for pos in range(n_sub):
                if pos == mi_pos:
                    continue
                if parent_probs[pos, mi_pos].item() < confidence_threshold:
                    continue
                edge_pred = out["edge_type_scores"][pos, mi_pos].argmax().item()
                if edge_pred == Edge.NUM:
                    has_num = True
                elif edge_pred == Edge.DEN:
                    has_den = True

            bar_w = bboxes[mi][2]
            neighbor_widths = [bboxes[j][2] for j in subset_indices if j != mi]
            if neighbor_widths:
                median_w = sorted(neighbor_widths)[len(neighbor_widths) // 2]
                wide_bar = bar_w > median_w * 1.7
            else:
                wide_bar = False

            if (has_num and has_den) or (wide_bar and (has_num or has_den)):
                symbols[mi] = replace(symbols[mi], name="frac_bar")
                logger.debug(
                    "promote: '-' (idx %d) → 'frac_bar' (num=%s den=%s wide=%s "
                    "bar_w=%.0f med_w=%.0f)",
                    mi, has_num, has_den, wide_bar, bar_w, median_w,
                )

    def _promote_dot_cdot(
        self,
        symbols: list[DetectedSymbol],
        names: list[str],
        bboxes: list[list[float]],
        N: int,
        confidence_threshold: float,
    ) -> None:
        """Disambiguate dot/cdot by checking nearest horizontal neighbors
        in the same vertical strip.

        Find the closest symbols to the left and right of the dot whose
        vertical extent overlaps the dot's y-range. Compare the dot's
        vertical center to theirs:
        - Below neighbors → dot (decimal point)
        - Centered or above → cdot (multiply)
        No model calls — pure geometry.
        """
        dot_indices = [i for i, s in enumerate(symbols) if s.name in ("dot", "cdot")]
        if not dot_indices:
            return

        for di in dot_indices:
            dot_x, dot_y, dot_w, dot_h = bboxes[di]
            dot_cx = dot_x + dot_w / 2
            dot_cy = dot_y + dot_h / 2
            dot_y1 = dot_y
            dot_y2 = dot_y + dot_h

            # Find neighbors in the same vertical strip (y-range overlaps dot)
            # and pick the closest left and closest right
            left = None  # (index, distance)
            right = None
            for j in range(N):
                if j == di or names[j] in ("dot", "cdot", "prime"):
                    continue
                jx, jy, jw, jh = bboxes[j]
                jcx = jx + jw / 2
                jy1 = jy
                jy2 = jy + jh

                # Check vertical overlap with some tolerance
                tol = max(dot_h, jh) * 0.2
                if jy1 > dot_y2 + tol or jy2 < dot_y1 - tol:
                    continue

                dx = jcx - dot_cx
                if dx < 0:  # left
                    if left is None or abs(dx) < left[1]:
                        left = (j, abs(dx))
                else:  # right
                    if right is None or dx < right[1]:
                        right = (j, dx)

            # Need at least one neighbor
            ref = []
            if left:
                ref.append(left[0])
            if right:
                ref.append(right[0])
            if not ref:
                continue

            ref_cys = [bboxes[r][1] + bboxes[r][3] / 2 for r in ref]
            ref_hs = [bboxes[r][3] for r in ref]
            median_cy = sum(ref_cys) / len(ref_cys)
            median_h = max(ref_hs)
            if median_h < 1e-6:
                continue

            dy = (dot_cy - median_cy) / median_h

            # Below neighbors → dot (decimal point)
            # Centered or above → cdot (multiply)
            if dy > 0.3:
                new_name = "dot"
            else:
                new_name = "cdot"

            if new_name != symbols[di].name:
                logger.debug(
                    "promote: '%s' (idx %d) → '%s' (dy=%.2f, ref=%s)",
                    symbols[di].name, di, new_name, dy, [names[r] for r in ref],
                )
                symbols[di] = replace(symbols[di], name=new_name)

    @torch.no_grad()
    def parse_with_tree(
        self,
        symbols: list[DetectedSymbol],
        pins: list[Tree] | None = None,
        relations=None,
    ) -> tuple[str, float, Tree, dict | None]:
        """Like parse(), but also returns (latex, confidence, tree, evidence).

        A bar's role — fraction bar or minus — is the expression's, not the
        stroke's: the classifier can't tell them apart from one stroke (a
        minus under a fraction's numerator read 0.52 fraction bar, 0.42
        minus) and the label steers the tree. So bars whose other reading is
        possible are read both ways and the labelling whose tree the model
        verifies best is kept (_label_search). Pinned bars keep their label.
        """
        if self.tree_strategy in ("backtrack", "backtrack_collapse") and len(symbols) > 1:
            return self._label_search(symbols, pins, relations)
        return self._parse_labelled(symbols, pins, relations)

    def _label_search(self, symbols, pins, relations):
        """Every labelling of the uncertain bars (at most MAX_BAR_CHOICES of
        them, the most uncertain), parsed once without jitter, and scored by: the classifier's
        probability of the labels x the subset model's verification of each
        parent's children x the GNN's agreement with the tree (as logs)."""
        import itertools
        import math

        from mathnote_ocr.tree_parser.bottomup_v2 import _verify_score_gnn, _verify_score_subset

        pinned = set()
        for pin in pins or []:
            for node in getattr(pin, "nodes", {}).values():
                sym = getattr(node, "symbol", None)
                if sym is not None:
                    pinned.update(sym.stroke_ids)
        choices = []                                     # (position, [(name, p), ...])
        for i, s in enumerate(symbols):
            if s.name not in BARS or any(st.id in pinned for st in s.strokes):
                continue
            # both labels' probabilities from the classifier's reading of the strokes
            probs = dict(s.alternatives)
            other = next(n for n in BARS if n != s.name)
            p_own, p_other = probs.get(s.name, s.confidence), probs.get(other, 0.0)
            if p_other >= MIN_BAR_ALTERNATIVE:
                choices.append((i, [(s.name, max(p_own, 1e-6)), (other, p_other)]))
        if not choices:
            return self._parse_labelled(symbols, pins, relations)
        choices = sorted(choices, key=lambda c: -c[1][1][1])[:MAX_BAR_CHOICES]
        # the labellings are compared on one parse each, without the jittered
        # runs; only the winner is read with them
        best = None
        runs, self.tta_runs = self.tta_runs, 1
        try:
            for combo in itertools.product(*(opts for _i, opts in choices)):
                labelled = list(symbols)
                log_p = 0.0
                for (i, _opts), (name, p) in zip(choices, combo):
                    if name != symbols[i].name:
                        labelled[i] = replace(symbols[i], name=name)
                    log_p += math.log(p)
                tree = self._parse_labelled(labelled, pins, relations)[2]
                v2 = self._make_symbols(labelled)
                score = log_p + math.log(max(_verify_score_subset(tree, v2, self._run_subsets, self._make_subsets), 1e-3))
                gnn = getattr(self, "gnn_model", None)
                if gnn is not None:
                    score += math.log(max(_verify_score_gnn(tree, v2, gnn, self.symbol_vocab, self.device,
                                                            self._run_subsets, self._make_subsets), 1e-3))
                if best is None or score > best[0]:
                    best = (score, labelled)
        finally:
            self.tta_runs = runs
        return self._parse_labelled(best[1], pins, relations)

    def _parse_labelled(
        self,
        symbols: list[DetectedSymbol],
        pins: list[Tree] | None = None,
        relations=None,
    ) -> tuple[str, float, Tree, dict | None]:
        """parse_with_tree for the symbols as labelled.

        When *pins* are provided, the tree builder enforces each pin's internal
        edges. The pin's root attaches to the surrounding tree as the model
        chooses; only the internal subtree structure is fixed.

        *relations* (mathnote_ocr.relations.Relations): relations and links
        the tree may not have.
        """
        if not symbols:
            return "", 1.0, Tree(()), None

        # Promote dot/cdot (and frac bars for non-backtrack strategies) — not a pinned symbol
        pinned = {i for p in pins or [] for node in getattr(p, "nodes", {}).values()
                  if getattr(node, "symbol", None) is not None for i in node.symbol.stroke_ids}
        if self.promote:
            symbols = self.promote_symbols(symbols, keep={k for k, s in enumerate(symbols)
                                                          if any(st.id in pinned for st in s.strokes)})

        if len(symbols) == 1:
            from mathnote_ocr.latex_utils.glyphs import SYMBOL_TO_LATEX

            name = symbols[0].name
            sym = Symbol(
                id=0,
                name=name,
                bbox=BBox(
                    symbols[0].bbox.x, symbols[0].bbox.y, symbols[0].bbox.w, symbols[0].bbox.h
                ),
                stroke_ids=tuple(s.id for s in symbols[0].strokes),
            )
            tree = Tree((Node(sym, ROOT_ID, -1, 0),))
            return SYMBOL_TO_LATEX.get(name, name), 1.0, tree, None

        N = len(symbols)
        names = [s.name for s in symbols]
        bboxes = [[s.bbox.x, s.bbox.y, s.bbox.w, s.bbox.h] for s in symbols]

        if self.tree_strategy in ("backtrack", "backtrack_collapse"):
            from mathnote_ocr.tree_parser.bottomup_v2 import build as build_v2
            from mathnote_ocr.tree_parser.bottomup_v2 import build_with_collapse

            v2_syms = self._make_symbols(symbols)
            gnn_model = getattr(self, "gnn_model", None)
            symbol_vocab = getattr(self, "symbol_vocab", None)
            build_fn = (
                build_with_collapse if self.tree_strategy == "backtrack_collapse" else build_v2
            )
            tree = build_fn(
                v2_syms,
                self._run_subsets,
                self._make_subsets,
                root_discount=self.root_discount,
                tta_runs=self.tta_runs,
                tta_dx=self.tta_dx,
                tta_dy=self.tta_dy,
                tta_size=self.tta_size,
                gnn_model=gnn_model,
                symbol_vocab=symbol_vocab,
                device=self.device if gnn_model else None,
                pins=pins,
                relations=relations,
            )
            return tree_to_latex(tree), 1.0, tree, None

        v2_syms = self._make_symbols(symbols)
        # Pins as in the backtrack search: each pinned group is read as one
        # unit (bottomup_v2._collapse_pins) and its subtree put back after
        stored_pin_subtrees = {}
        units = None                       # the collapsed list's symbols, by search position
        if pins:
            from mathnote_ocr.tree_parser.bottomup_v2 import _collapse_pins

            units, stored_pin_subtrees = _collapse_pins(
                v2_syms, pins, next_id_start=max(s.id for s in v2_syms) + 1
            )
            # the search below takes symbol ids for positions: renumber
            v2_syms = [replace(u, id=i) for i, u in enumerate(units)]
            N = len(v2_syms)
            names = [s.name for s in v2_syms]
            bboxes = [s.bbox.to_list() for s in v2_syms]
            if N == 1:
                tree = self._expand_pins(
                    Tree((Node(v2_syms[0], ROOT_ID, -1, 0),)), units, stored_pin_subtrees
                )
                return tree_to_latex(tree), 1.0, tree, None
        all_partial = []
        seen_subsets: set[tuple] = set()
        for tta_i in range(self.tta_runs):
            bb = (
                bboxes
                if tta_i == 0
                else jitter_bboxes(bboxes, self.tta_dx, self.tta_dy, self.tta_size)
            )
            subsets = self._make_subsets(bb)
            all_partial.extend(self._run_subsets(names, bb, subsets))
            for s in subsets:
                seen_subsets.add(tuple(s))

        def restricted(ev):
            return relations.restrict(ev, v2_syms) if relations else ev

        for _ in range(self.max_iters):
            evidence = restricted(aggregate_evidence_soft(N, all_partial))
            tree = self._evidence_to_tree(evidence, v2_syms)
            targets = find_seq_conflicts(
                evidence,
                tree,
                seq_threshold=2.0,
                max_subset_size=min(self.max_subset, N),
            )
            new_targets = [t for t in targets if tuple(t) not in seen_subsets]
            if not new_targets:
                break
            for t in new_targets:
                seen_subsets.add(tuple(t))
            all_partial.extend(self._run_subsets(names, bboxes, new_targets))

        evidence = restricted(aggregate_evidence_soft(N, all_partial))
        tree = self._evidence_to_tree(evidence, v2_syms)
        confidence = score_tree(self.scoring, evidence, tree, N)
        if units is not None:
            tree = self._expand_pins(tree, units, stored_pin_subtrees)
        return tree_to_latex(tree), confidence, tree, evidence

    @staticmethod
    def _expand_pins(tree: Tree, units: list[Symbol], stored_pin_subtrees: dict) -> Tree:
        """The tree over the renumbered units back on the units' own symbols,
        each pinned group's subtree put back in place of its unit."""
        from mathnote_ocr.tree_parser.tree_ops import expand, reorder_siblings

        tree = Tree(tuple(
            Node(units[n.symbol.id], ROOT_ID if n.parent_id == ROOT_ID else units[n.parent_id].id,
                 n.edge_type, n.order)
            for sid, n in tree.nodes.items() if sid != tree.root
        ))

        for atom_id, sub in stored_pin_subtrees.items():
            if atom_id in tree.nodes:
                tree = expand(tree, atom_id, sub)
        return reorder_siblings(tree)

    @torch.no_grad()
    def parse_with_diagnostics(
        self,
        symbols: list[DetectedSymbol],
    ) -> dict:
        """Full parse with subset-level diagnostics.

        Returns dict with keys:
            latex, confidence, roots, evidence, subsets
        where subsets is a list of dicts per subset:
            {indices, symbols, pred_parent, pred_edge, pred_seq}
        """
        if not symbols:
            return {
                "latex": "",
                "confidence": 1.0,
                "tree": Tree(()),
                "evidence": None,
                "subsets": [],
            }

        if len(symbols) == 1:
            from mathnote_ocr.latex_utils.glyphs import SYMBOL_TO_LATEX

            name = symbols[0].name
            sym = Symbol(
                id=0,
                name=name,
                bbox=BBox(
                    symbols[0].bbox.x, symbols[0].bbox.y, symbols[0].bbox.w, symbols[0].bbox.h
                ),
                stroke_ids=tuple(s.id for s in symbols[0].strokes),
            )
            tree = Tree((Node(sym, ROOT_ID, -1, 0),))
            return {
                "latex": SYMBOL_TO_LATEX.get(name, name),
                "confidence": 1.0,
                "tree": tree,
                "evidence": None,
                "subsets": [],
            }

        N = len(symbols)
        names = [s.name for s in symbols]
        bboxes = [[s.bbox.x, s.bbox.y, s.bbox.w, s.bbox.h] for s in symbols]
        v2_syms = self._make_symbols(symbols)

        subsets = self._make_subsets(bboxes)
        all_partial = self._run_subsets(names, bboxes, subsets)
        seen_subsets = {tuple(s) for s in subsets}

        for _ in range(self.max_iters):
            evidence = aggregate_evidence_soft(N, all_partial)
            tree = self._evidence_to_tree(evidence, v2_syms)
            targets = find_seq_conflicts(
                evidence,
                tree,
                seq_threshold=2.0,
                max_subset_size=min(self.max_subset, N),
            )
            new_targets = [t for t in targets if tuple(t) not in seen_subsets]
            if not new_targets:
                break
            for t in new_targets:
                seen_subsets.add(tuple(t))
            all_partial.extend(self._run_subsets(names, bboxes, new_targets))

        evidence = aggregate_evidence_soft(N, all_partial)
        tree = self._evidence_to_tree(evidence, v2_syms)
        latex = tree_to_latex(tree)
        confidence = score_tree(self.scoring, evidence, tree, N)

        # Build per-subset diagnostics
        subset_diags = []
        for subset_indices, out_cpu, n_sub in all_partial:
            S = out_cpu["parent_scores"].shape[0]
            diag = {
                "indices": list(subset_indices),
                "symbols": [names[gi] for gi in subset_indices],
                "predictions": [],
            }
            parent_scores = out_cpu["parent_scores"]
            edge_scores = out_cpu["edge_type_scores"]
            seq_scores = out_cpu.get("seq_scores")

            for i in range(n_sub):
                pred_parent_local = parent_scores[i].argmax().item()
                if pred_parent_local == S:
                    pred_parent_global = -1
                    pred_edge = -1
                else:
                    pred_parent_global = subset_indices[pred_parent_local]
                    pred_edge = edge_scores[i, pred_parent_local].argmax().item()

                pred_seq_global = -1
                if seq_scores is not None:
                    seq_local = seq_scores[i].argmax().item()
                    pred_seq_global = -1 if seq_local == S else subset_indices[seq_local]

                parent_conf = torch.softmax(parent_scores[i], dim=0).max().item()

                diag["predictions"].append(
                    {
                        "symbol": names[subset_indices[i]],
                        "global_idx": subset_indices[i],
                        "pred_parent": pred_parent_global,
                        "pred_parent_sym": names[pred_parent_global]
                        if pred_parent_global >= 0
                        else "ROOT",
                        "pred_edge": pred_edge,
                        "parent_conf": round(parent_conf, 3),
                        "pred_seq": pred_seq_global,
                    }
                )

            # Build mini-tree from subset predictions → LaTeX
            try:
                sub_nodes: list[Node] = []
                for i in range(n_sub):
                    p = diag["predictions"][i]
                    local_parent = ROOT_ID
                    if p["pred_parent"] >= 0 and p["pred_parent"] in subset_indices:
                        local_parent = list(subset_indices).index(p["pred_parent"])
                    sym = Symbol(id=i, name=p["symbol"], bbox=BBox(*bboxes[subset_indices[i]]))
                    sub_nodes.append(Node(sym, local_parent, p["pred_edge"], i))
                diag["latex"] = tree_to_latex(Tree(tuple(sub_nodes)))
            except Exception:  # noqa: BLE001
                diag["latex"] = ""

            subset_diags.append(diag)

        return {
            "latex": latex,
            "confidence": confidence,
            "tree": tree,
            "evidence": evidence,
            "subsets": subset_diags,
        }


# ── Concrete implementations ────────────────────────────────────────


class SubsetTreeParser(TreeParser):
    """Evidence → cost strategy → Edmonds'.

    Uses a pluggable cost strategy from tree_parser.costs to compute
    edge weights for Edmonds' algorithm.
    """

    def __init__(self, cost: str = "propagate", **kwargs) -> None:
        super().__init__(**kwargs)
        self.cost = cost

    @property
    def mode(self) -> str:
        return f"subset({self.cost})"

    def _evidence_to_tree(self, evidence, symbols):
        return build_tree_from_evidence(
            evidence,
            symbols,
            cost=self.cost,
        )


class GNNTreeParser(TreeParser):
    """Evidence → GNN refinement → Edmonds'.

    The GNN takes evidence features and produces refined parent/edge
    scores, anchored by the original evidence as a residual. Uses its
    own seq_scores for a sibling bonus in probability space.
    """

    def __init__(
        self,
        gnn_run: str = "default",
        anchor: bool = True,
        seq_bonus: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.anchor = anchor
        self.seq_bonus = seq_bonus

        from mathnote_ocr.tree_parser.gnn.model import EvidenceGNN

        gnn_ckpt = load_checkpoint(
            "tree_gnn", gnn_run, device=self.device, weights_dir=self.weights_dir
        )
        gnn_cfg = gnn_ckpt["config"]
        self.gnn_model = EvidenceGNN(**gnn_cfg).to(self.device)
        self.gnn_model.load_state_dict(gnn_ckpt["model_state_dict"])
        self.gnn_model.eval()

    @property
    def mode(self) -> str:
        parts = ["gnn"]
        if self.anchor:
            parts.append("+anchor")
        if self.seq_bonus:
            parts.append("+seq")
        return "".join(parts)

    def _evidence_to_tree(self, evidence, symbols):
        N = len(symbols)
        names = [s.name for s in symbols]
        bboxes = [s.bbox.to_list() for s in symbols]
        _, edge_features = evidence_to_features(evidence)

        sym_ids = torch.tensor(
            [self._symbol_id(n) for n in names],
            dtype=torch.long,
            device=self.device,
        )
        _, size_feats = compute_features_from_bbox_list(bboxes, N)
        size_feats = size_feats.to(self.device)
        edge_features = edge_features.to(self.device)
        pad_mask = torch.zeros(N, dtype=torch.bool, device=self.device)

        out = self.gnn_model(
            sym_ids.unsqueeze(0),
            size_feats.unsqueeze(0),
            edge_features.unsqueeze(0),
            pad_mask.unsqueeze(0),
        )

        from mathnote_ocr.tree_parser.costs import anchor_with_evidence, apply_seq_bonus

        parent_scores = out["parent_scores"][0]  # (N, N+1)
        edge_type_scores = out["edge_type_scores"][0]  # (N, N+1, E)

        if self.anchor:
            parent_scores = anchor_with_evidence(parent_scores, evidence, N)

        if self.seq_bonus and "seq_scores" in out:
            seq_scores = out["seq_scores"][0]
            evidence["gnn_seq_scores"] = seq_scores.cpu()
            parent_scores = apply_seq_bonus(parent_scores, seq_scores, N)

        return build_tree_from_scores(
            parent_scores,
            edge_type_scores,
            symbols,
        )
