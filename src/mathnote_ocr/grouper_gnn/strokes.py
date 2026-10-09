"""The stroke grouper at reading time: attention over all of an ink's strokes
(StrokeGNN with stroke vectors), for every pair of strokes the log-odds that
they make one symbol.

A reading's probability under the pairs is the product over stroke pairs of
p (same group) or 1 - p (not) — a constant times exp(sum over its groups of
the log-odds of their pairs). So each candidate group scores that sum, and
the grouper's exact search over candidate groups finds the most probable
readings and their shares (engine.grouper.group_and_classify with
GrouperParams.stroke_model).

Each stroke is the shape vector of the symbol classifier the model was
trained with (its checkpoint's "classifier"), the stroke drawn alone as in
training (pen 2, size over "source_size") — not necessarily the reader's
classifier, which still labels the groups.
"""

from __future__ import annotations

from pathlib import Path

import statistics

import numpy as np
import torch

from mathnote_ocr.engine.stroke import Stroke, StrokePoint
from mathnote_ocr.grouper_gnn.features import compute_edge_features, compute_features_v2, compute_node_features
from mathnote_ocr.grouper_gnn.model import StrokeGNN, StrokeGroupGNN

MAX_GROUP = 4
# Readings of group pictures kept (read_groups): a group drawn from the same points at the
# same scale is the same picture — read once. A region, its cells, the line around it and the
# next request after one more stroke share most of their groups. ~3.5 kB each.
CACHE_SIZE = 10_000


class StrokeGrouperModel:
    def __init__(self, net: StrokeGNN, vectors, source_size: float, run: str, labels: list[str] | None = None,
                 features: str = "v1", kind: str = "pairs"):
        self.net = net
        self.kind = kind                # "pairs" (StrokeGNN) or "groups" (StrokeGroupGNN: with group pictures)
        # temperatures (fitted on held-out inks, in the checkpoint): a reading's
        # score and the label head's logits divided by them — honest probabilities, same ranking
        self.t_grouping = 1.0
        self.t_labels = 1.0
        # the writing size it was trained at: a typical stroke this tall (px, pen 2 — the
        # training inks were scaled so); the ink is brought to it before reading (its
        # pictures and sizes are in pen widths)
        self.writing_height = 22.0
        # each symbol's usual size at that writing size (px, max of width and height; median
        # over the training inks, in the checkpoint): the line is measured
        # against its biggest symbol read with confidence (reference_scale)
        self.class_sizes: dict[str, float] = {}
        self.features = features        # grouper_gnn.features: "v1" (over the expression) or "v2" (writing scale)
        self.labels = labels or []      # the label head's classes, in order
        self.vectors = vectors          # SymbolClassifier: the stroke vectors' source
        self.source_size = source_size
        self.run = run
        self._read_cache: dict = {}     # group picture (its strokes' points) -> ClassificationResult

    @classmethod
    def load(cls, path: str, weights_dir: str | None = None) -> "StrokeGrouperModel":
        """A checkpoint path, or a run under <weights_dir>/grouper_gnn/."""
        from mathnote_ocr.classifier.inference import SymbolClassifier
        p = Path(path)
        if not p.suffix:
            p = Path(weights_dir or "weights") / "grouper_gnn" / path / "checkpoint.pth"
        ck = torch.load(p, map_location="cpu", weights_only=False)
        kind = ck.get("kind", "pairs")
        net = (StrokeGroupGNN if kind == "groups" else StrokeGNN)(num_classes=len(ck["label_vocab"]), **ck["config"])
        net.load_state_dict(ck["model_state_dict"])
        net.eval()
        clf_path = p.parents[2] / "classifier" / ck["classifier"] / "checkpoint.pth"
        vectors = SymbolClassifier(str(clf_path), device=torch.device("cpu"))
        labels = [n for n, _i in sorted(ck["label_vocab"].items(), key=lambda kv: kv[1])]
        model = cls(net, vectors, ck.get("source_size", 800), p.parent.name, labels, ck.get("features", "v1"), kind)
        cal = ck.get("calibration") or {}
        model.t_grouping, model.t_labels = float(cal.get("grouping", 1.0)), float(cal.get("labels", 1.0))
        model.writing_height = float(ck.get("writing_height", 22.0))
        model.class_sizes = dict(ck.get("class_sizes") or {})
        return model

    def read_groups(self, strokes: list[Stroke], groups) -> list:
        """Each group (positions) drawn whole, as in training (pen 2, the classifier's
        own drawing), read by the model's classifier: its ClassificationResults. A
        group drawn before from the same points (at the same scale: the points are
        the scaled ones) is not read again."""
        from mathnote_ocr.engine.grouper import _size_feat, ink_size_feat
        from mathnote_ocr.engine.renderer import render_strokes
        clf = self.vectors
        pen2 = [Stroke(id=s.id, points=s.points, bbox=s.bbox, width=2.0) for s in strokes]
        stroke_key = [tuple((p.x, p.y) for p in s.points) for s in strokes]
        keys = [tuple(stroke_key[i] for i in sorted(g)) for g in groups]
        out = [self._read_cache.get(key) for key in keys]
        todo = [k for k, r in enumerate(out) if r is None]
        if todo:
            imgs, sizes = [], []
            for k in todo:
                gs = [pen2[i] for i in sorted(groups[k])]
                imgs.append(render_strokes(gs, canvas_size=clf.canvas_size, source_size=self.source_size,
                                           round_joins=clf.round_joins, fill_pens=clf.fill_pens))
                sizes.append(ink_size_feat(gs) if clf.fill_pens else _size_feat(gs, self.source_size))
            read = clf.classify_batch(imgs, size_feats=sizes if clf.use_size_feat else None)
            if len(self._read_cache) + len(todo) > CACHE_SIZE:
                self._read_cache.clear()     # full: begin again (this call's readings are in hand)
            for k, r in zip(todo, read):
                out[k] = r
                self._read_cache[keys[k]] = r
        return out

    def _stroke_vectors(self, strokes: list[Stroke]) -> torch.Tensor:
        res = self.read_groups(strokes, [[i] for i in range(len(strokes))])
        return torch.from_numpy(np.stack([r.features for r in res])).float()

    def pair_logits(self, strokes: list[Stroke]) -> torch.Tensor:
        """(n, n) symmetric log-odds that strokes i and j make one symbol."""
        return self.outputs(strokes)[0]

    @torch.no_grad()
    def outputs(self, strokes: list[Stroke]) -> tuple[torch.Tensor, torch.Tensor]:
        """(pair log-odds (n, n), each stroke's label log-probabilities (n, labels) —
        the labels in self.labels' order)."""
        es, lp, _x = self.encode(strokes)
        return es, lp

    def default_scale(self, strokes: list[Stroke]) -> float:
        """The factor that brings the ink's typical stroke (the median height of those over
        0.3 of the tallest, as the training inks were scaled) to writing_height — only ever
        down (an ink of dots is not small writing to blow up); fewer than 3 strokes: 1."""
        if len(strokes) < 3:
            return 1.0
        hs = sorted(s.bbox.h for s in strokes)
        big = [h for h in hs if h > 0.3 * hs[-1]] or hs
        med = big[len(big) // 2]
        return min(self.writing_height / med, 1.0) if med > 0 else 1.0

    def reference_scale(self, symbols) -> float | None:
        """The factor from a reading: each symbol read with confidence over one half (one with a
        usual size: not a bar, a bracket, a root — their size is what they span — nor a dot)
        says what the writing's size is (its usual size against its size here); the median of
        what they say — one odd symbol does not decide — and never over 1: the ink is only ever
        shrunk to the training size, never blown up (small marks read alone are not small
        writing). None: no such symbol."""
        says = [self.class_sizes[s.name] / max(s.bbox.w, s.bbox.h) for s in symbols
                if s.confidence > 0.5 and self.class_sizes.get(s.name, 0.0) >= 4.0 and max(s.bbox.w, s.bbox.h) > 0]
        if not says:
            return None
        return min(max(statistics.median(says), 0.2), 1.0)

    def at_training_scale(self, strokes: list[Stroke], scale: float | None = None) -> list[Stroke]:
        """The ink scaled by *scale* (else default_scale)."""
        k = self.default_scale(strokes) if scale is None else scale
        if abs(k - 1.0) < 0.05:
            return strokes
        return [Stroke.from_points([StrokePoint(p.x * k, p.y * k, p.t) for p in s.points], id=s.id, width=s.width)
                for s in strokes]

    @torch.no_grad()
    def encode(self, strokes: list[Stroke], scale: float | None = None):
        """(pair log-odds (n, n), stroke label log-probabilities (n, labels), stroke tokens (n, d))."""
        strokes = self.at_training_scale(strokes, scale)
        n = len(strokes)
        st = [[{"x": p.x, "y": p.y} for p in s.points] for s in strokes]
        if self.features == "v2":
            geo, edge = compute_features_v2(st)         # pen 2, as trained (the ink at the app's scale)
        else:
            _r, geo = compute_node_features(st, render_size=8)
            edge = compute_edge_features(st)
        args = (self._stroke_vectors(strokes)[None], geo[None], edge[None],
                torch.zeros(1, n, dtype=torch.bool), torch.ones(1, n, n, dtype=torch.bool))
        if self.kind == "groups":
            es, nl, x = self.net.encode(*args)
        else:
            es, nl = self.net(*args)
            x = None
        return (es[0] + es[0].T) / 2, nl[0].log_softmax(-1), (x[0] if x is not None else None)

    @torch.no_grad()
    def group_terms(self, strokes: list[Stroke], tokens: torch.Tensor, groups,
                    scale: float | None = None) -> tuple[torch.Tensor, torch.Tensor, list]:
        """For the "groups" kind: each group's picture term (G,), its label
        log-probabilities (G, labels), and the classifier's reading of its picture."""
        groups = [sorted(g) for g in groups]
        res = self.read_groups(self.at_training_scale(strokes, scale), groups)
        n = len(strokes)
        member = torch.zeros(len(groups), n)
        for r, g in enumerate(groups):
            member[r, g] = 1.0 / len(g)
        vec = torch.from_numpy(np.stack([r.features for r in res])).float()
        conf = torch.tensor([r.confidence for r in res])
        dist = torch.tensor([r.prototype_distance / self.vectors.ood_threshold for r in res])
        size = torch.tensor([len(g) for g in groups])
        h, gl = self.net.groups(tokens, member, vec, conf, dist, size)
        return h, (gl / self.t_labels).log_softmax(-1), res

    @staticmethod
    def joined(scores: torch.Tensor, allowed: set[int]) -> list[frozenset[int]]:
        """The pairs' own grouping: pairs above even odds, best first, two groups
        merged only when every pair across them is, at most MAX_GROUP strokes —
        over the *allowed* positions. Its groups join the candidates, so the
        exact search can always reach it."""
        group = {i: i for i in allowed}
        members = {i: [i] for i in allowed}
        pairs = sorted(((float(scores[i, j]), i, j) for i in allowed for j in allowed
                        if i < j and scores[i, j] > 0), reverse=True)
        for _s, i, j in pairs:
            a, b = group[i], group[j]
            if a == b or len(members[a]) + len(members[b]) > MAX_GROUP:
                continue
            if all(scores[x, y] > 0 for x in members[a] for y in members[b]):
                for x in members[b]:
                    group[x] = a
                members[a] += members.pop(b)
        return [frozenset(m) for m in members.values()]

    @staticmethod
    def group_score(scores: torch.Tensor, group: frozenset[int]) -> float:
        idx = sorted(group)
        return sum(float(scores[a, b]) for k, a in enumerate(idx) for b in idx[k + 1:])
