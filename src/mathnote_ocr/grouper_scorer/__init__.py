"""Is this group of strokes one symbol? A small learned scorer for the grouper.

The grouper proposes candidate groups and the classifier labels each one; the
classifier's confidence says *which* symbol a group would be if it were one,
not *whether* it is one. This scorer answers the second question, from what
can be seen of the group (features()):

  conf, dist, ood   the classifier on the whole group: top probability,
                    prototype distance (over the reject distance), rejected
  s_min, s_mean     the classifier on each stroke alone: lowest, mean top
                    probability
  touch             how close its strokes come, ink to ink: the largest, over
                    its strokes, of the nearest other stroke (writing sizes)
  gap               the same between bounding boxes
  size, aspect      the group's extent (writing sizes), width over height

— not how many strokes it has (a symbol overdrawn or corrected is rare, and
must not be thrown away for that). Most of these are the classifier's own
outputs, on its own scale: a scorer belongs to the classifier it was trained
with (GroupScorer.classifier) and is refused with another. With grouper.scorer: model, a reading is
ranked by the product over its symbols of P(one symbol) × P(label).

    scorer = GroupScorer.load("v1", weights_dir)
    p = scorer.score([features(group, strokes, cache, scale, ood_threshold), ...])
"""

from __future__ import annotations

import math

import torch
from torch import nn

FEATURES = ["conf", "dist", "ood", "s_min", "s_mean", "touch", "gap", "size", "aspect"]


def writing_scale(strokes) -> float:
    """The writing's size: the median stroke diagonal."""
    diag = sorted(math.hypot(s.bbox.w, s.bbox.h) for s in strokes)
    return max(diag[len(diag) // 2], 1.0) if diag else 1.0


def _ink_distance(a, b, cap: int = 40) -> float:
    pa = a.points[:: max(1, len(a.points) // cap)]
    pb = b.points[:: max(1, len(b.points) // cap)]
    return min(math.hypot(p.x - q.x, p.y - q.y) for p in pa for q in pb)


def _box_gap(a, b) -> float:
    dx = max(0.0, max(a.bbox.x, b.bbox.x) - min(a.bbox.x + a.bbox.w, b.bbox.x + b.bbox.w))
    dy = max(0.0, max(a.bbox.y, b.bbox.y) - min(a.bbox.y + a.bbox.h, b.bbox.y + b.bbox.h))
    return math.hypot(dx, dy)


def features(group, strokes, cache, scale: float, ood_threshold: float) -> dict:
    """What the scorer sees of *group* (positions into *strokes*): FEATURES,
    plus the classifier's label for the record."""
    members = [strokes[i] for i in sorted(group)]
    r = cache[frozenset(s.id for s in members)]
    s_conf = [cache[frozenset([s.id])].confidence for s in members]
    touch = gap = 0.0
    if len(members) > 1:
        touch = max(min(_ink_distance(a, b) for b in members if b is not a) for a in members) / scale
        gap = max(min(_box_gap(a, b) for b in members if b is not a) for a in members) / scale
    xs = [s.bbox.x for s in members] + [s.bbox.x + s.bbox.w for s in members]
    ys = [s.bbox.y for s in members] + [s.bbox.y + s.bbox.h for s in members]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    shape = {}
    if getattr(r, "features", None) is not None:      # the shape: the group's and its strokes' (scorer "shape")
        import numpy as np
        g = np.asarray(r.features, dtype=np.float32)
        s = np.mean([np.asarray(cache[frozenset([m.id])].features, dtype=np.float32) for m in members], axis=0)
        cos = float(g @ s / (np.linalg.norm(g) * np.linalg.norm(s) + 1e-9))
        shape = {"g_emb": [round(float(v), 3) for v in g], "s_emb": [round(float(v), 3) for v in s],
                 "cos": round(cos, 4)}
    return {
        **shape,
        "conf": round(r.confidence, 4), "dist": round(r.prototype_distance / ood_threshold, 4),
        "ood": bool(r.is_ood), "s_min": round(min(s_conf), 4), "s_mean": round(sum(s_conf) / len(s_conf), 4),
        "touch": round(touch, 4), "gap": round(gap, 4), "size": round(math.hypot(w, h) / scale, 4),
        "aspect": round((w + 1e-3) / (h + 1e-3), 4), "label": r.symbol, "n": len(members),
    }


def _run_name(run: str) -> str:
    """A run's name (its folder), from a run name or a path to its checkpoint."""
    from pathlib import Path
    p = Path(run)
    return p.parent.name if p.suffix == ".pth" else p.name


class ScorerNet(nn.Module):
    """A small MLP: the features (standardised) -> logit of P(one symbol)."""

    def __init__(self, n_in: int = len(FEATURES), hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


SHAPE = 256          # the classifier's shape vector (its layer before the class scores)


class ShapeScorerNet(nn.Module):
    """FEATURES plus what the group looks like against its strokes: the classifier's shape
    vector of the group and the mean of its strokes' (each squeezed to *proj* numbers by a
    learned layer), and how alike the two are — a + is unlike either of its strokes, a 1 and
    a − merged by mistake look like a 1 and a − side by side."""

    def __init__(self, n_in: int = len(FEATURES), proj: int = 16, hidden: int = 48):
        super().__init__()
        self.g = nn.Linear(SHAPE, proj)
        self.s = nn.Linear(SHAPE, proj)
        self.net = nn.Sequential(nn.Linear(n_in + 2 * proj + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, g, s, cos):
        return self.net(torch.cat([x, self.g(g), self.s(s), cos.unsqueeze(-1)], -1)).squeeze(-1)


def shape_tensors(rows: list[dict]):
    """(group shape vectors, strokes' mean shape vectors, their cosine) for features() rows."""
    zero = [0.0] * SHAPE                          # a group classified without its vector: no shape
    g = torch.tensor([r.get("g_emb", zero) for r in rows], dtype=torch.float32)
    s = torch.tensor([r.get("s_emb", zero) for r in rows], dtype=torch.float32)
    c = torch.tensor([float(r.get("cos", 0.0)) for r in rows], dtype=torch.float32)
    return g / 10.0, s / 10.0, c


class GroupScorer:
    def __init__(self, net, mean, std, classifier: str | None = None, objective: str = "groups",
                 arch: str = "base"):
        self.net = net.eval()
        self.arch = arch                         # "base": FEATURES; "shape": and the shape vectors (ShapeScorerNet)
        self.classifier = classifier             # the classifier run it was trained with
        # "groups": trained group by group (P(one symbol)); "readings": trained so
        # the true reading wins over every other (the grouper ranks by exp of its score)
        self.objective = objective
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    @classmethod
    def load(cls, run: str, weights_dir=None) -> "GroupScorer":
        from mathnote_ocr.engine.checkpoint import load_checkpoint
        ck = load_checkpoint("grouper_scorer", run, device="cpu", weights_dir=weights_dir)
        arch = ck.get("arch", "base")
        net = (ShapeScorerNet(len(ck["features"])) if arch == "shape"
               else ScorerNet(len(ck["features"]), ck.get("hidden", 32)))
        net.load_state_dict(ck["model"])
        if list(ck["features"]) != FEATURES:
            raise ValueError(f"grouper_scorer/{run} was trained on other features: {ck['features']}")
        return cls(net, ck["mean"], ck["std"], ck.get("classifier"), ck.get("objective", "groups"), arch)

    def check(self, classifier_run: str) -> None:
        """Refuse a classifier other than the one the scorer learned from."""
        if self.classifier and _run_name(classifier_run) != self.classifier:
            raise ValueError(f"the group scorer was trained with classifier {self.classifier!r}, "
                             f"not {_run_name(classifier_run)!r}: retrain it for this classifier")

    @torch.no_grad()
    def logits(self, rows: list[dict]) -> list[float]:
        """The network's score for each features() row."""
        if not rows:
            return []
        x = torch.tensor([[float(r[f]) for f in FEATURES] for r in rows], dtype=torch.float32)
        if self.arch == "shape":
            return self.net((x - self.mean) / self.std, *shape_tensors(rows)).tolist()
        return self.net((x - self.mean) / self.std).tolist()

    def score(self, rows: list[dict]) -> list[float]:
        """P(one symbol) for each features() row (the logistic of its score)."""
        return [1 / (1 + math.exp(-max(min(v, 40.0), -40.0))) for v in self.logits(rows)]
