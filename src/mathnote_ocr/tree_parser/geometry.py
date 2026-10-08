"""The subset model's geometry, encoding v2: continuous, typographic, and in
units of the expression's own writing scale.

v1 (latex_utils.relations.compute_features_from_bbox_list) put three centre
measures into 8 fixed buckets over the subset's median height. On real trees
that lost what tells a barely raised exponent from a symbol on the line
(scripts/probe_geometry.py: with each symbol's size against its class's
usual, 60% fewer relation errors on the user's writing than the buckets with
the symbols' identities). v2:

  pair features  of j as seen from i, (S, S, PAIR): centre offsets, bottom
                 and top edge differences, the gaps between facing edges
                 (both ways), horizontal and vertical overlap — over the
                 writing scale, signed-log scaled — and the log height and
                 width ratios
  symbol features (S, SYMBOL): log height over the scale; log height over
                 the scale and its class's usual height (0, and a flag off,
                 for a class without one); log width over height; its centre
                 against the subset's median centre over the scale

The writing scale: the median height of the expression's ordinary symbols
(not flat — bars, dots — and not sized by what they span — brackets, sums,
roots), so a subset is measured with the whole expression's scale, not its
own few symbols'. Usual heights: per class, the median height over the
scale, measured on the training data (usual_heights) and kept in the
checkpoint.
"""

from __future__ import annotations

import math
import statistics

import torch

FLAT = {"-", "frac_bar", "dot", "cdot", ",", ".", "=", "ldots", "cdots", "prime", "times", "+", "pm", "mp"}
SPANNING = {"(", ")", "[", "]", "lbrace", "rbrace", "langle", "rangle", "|", "sum", "int", "prod", "oint",
            "bigcup", "bigcap", "sqrt", "frac_bar"}
PAIR = 10
SYMBOL = 5
MIN_USUAL = 10          # instances a class needs for a usual height


EXPRESSION = "expression"     # a collapsed sub-expression (latex_utils.collapse): not a symbol's size


def writing_scale(names: list[str], bboxes: list[list[float]]) -> float:
    """The median height of the expression's ordinary symbols (else of all)."""
    hs = [b[3] for n, b in zip(names, bboxes)
          if n not in FLAT and n not in SPANNING and n != EXPRESSION and b[3] > 0]
    if not hs:
        hs = [b[3] for b in bboxes if b[3] > 0]
    return statistics.median(hs) if hs else 1.0


def usual_heights(examples) -> dict[str, float]:
    """Per class, the median height over its expression's writing scale."""
    seen: dict[str, list[float]] = {}
    for ex in examples:
        names = [s["name"] for s in ex["symbols"]]
        bboxes = [s["bbox"] for s in ex["symbols"]]
        scale = writing_scale(names, bboxes)
        for n, b in zip(names, bboxes):
            if b[3] > 0 and n != EXPRESSION:
                seen.setdefault(n, []).append(b[3] / scale)
    return {n: round(statistics.median(v), 4) for n, v in sorted(seen.items()) if len(v) >= MIN_USUAL}


def _slog(t: torch.Tensor) -> torch.Tensor:
    """Signed log: fine near zero (barely raised or not), compressed far away."""
    return torch.sign(t) * torch.log1p(t.abs() * 4) / 4


def pair_features(bbox_list: list[list[float]], scale: float, S: int) -> torch.Tensor:
    """(S, S, PAIR): [i, j] — symbol j as seen from symbol i."""
    out = torch.zeros(S, S, PAIR)
    n = len(bbox_list)
    if n == 0:
        return out
    b = torch.tensor(bbox_list, dtype=torch.float32)
    x0, y0 = b[:, 0], b[:, 1]
    w, h = b[:, 2].clamp(min=1e-6), b[:, 3].clamp(min=1e-6)
    x1, y1 = x0 + w, y0 + h
    cx, cy = x0 + w / 2, y0 + h / 2
    d = lambda v: v.unsqueeze(0) - v.unsqueeze(1)                    # [i, j] = v_j - v_i
    rows = [d(cx), d(cy), d(y1), d(y0),
            x0.unsqueeze(0) - x1.unsqueeze(1),                       # j's left from i's right
            x0.unsqueeze(1) - x1.unsqueeze(0),                       # i's left from j's right
            torch.minimum(x1.unsqueeze(0), x1.unsqueeze(1)) - torch.maximum(x0.unsqueeze(0), x0.unsqueeze(1)),
            torch.minimum(y1.unsqueeze(0), y1.unsqueeze(1)) - torch.maximum(y0.unsqueeze(0), y0.unsqueeze(1))]
    feats = [_slog(r / max(scale, 1e-6)) for r in rows]
    feats += [torch.log(h.unsqueeze(0) / h.unsqueeze(1)), torch.log(w.unsqueeze(0) / w.unsqueeze(1))]
    out[:n, :n] = torch.stack(feats, dim=-1)
    out[torch.arange(n), torch.arange(n)] = 0.0
    return out


def symbol_features(names: list[str], bbox_list: list[list[float]], scale: float, usual: dict[str, float],
                    S: int) -> torch.Tensor:
    """(S, SYMBOL) per symbol of the subset."""
    out = torch.zeros(S, SYMBOL)
    if not bbox_list:
        return out
    centres = sorted(b[1] + b[3] / 2 for b in bbox_list)
    mid = centres[len(centres) // 2]
    for i, (n, b) in enumerate(zip(names, bbox_list)):
        h, w = max(b[3], 1e-6), max(b[2], 1e-6)
        rel = math.log(h / scale)
        known = n in usual
        out[i] = torch.tensor([rel, rel - math.log(usual[n]) if known else 0.0, 1.0 if known else 0.0,
                               math.log(w / h), float(_slog(torch.tensor((b[1] + h / 2 - mid) / scale)))])
    return out


def subset_inputs(encoding: str, names: list[str], bboxes: list[list[float]], subset: list[int], S: int,
                  scale: float | None = None, usual: dict[str, float] | None = None,
                  subset_bboxes: list[list[float]] | None = None):
    """The subset model's geometry inputs for one subset of an expression:
    (pair input, symbol input) — v1: the buckets (long); v2: the features
    above (float). *subset_bboxes*: the subset's boxes when they differ from
    the expression's (a training jitter); *scale*: the expression's writing
    scale (computed if not given)."""
    boxes = subset_bboxes if subset_bboxes is not None else [bboxes[g] for g in subset]
    if encoding == "v1":
        from mathnote_ocr.latex_utils.relations import compute_features_from_bbox_list
        return compute_features_from_bbox_list(boxes, S)
    scale = scale if scale is not None else writing_scale(names, bboxes)
    return (pair_features(boxes, scale, S),
            symbol_features([names[g] for g in subset], boxes, scale, usual or {}, S))
