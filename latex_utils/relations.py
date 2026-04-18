"""Spatial geometry computation between symbols.

Computes pairwise geometric features (vertical offset, horizontal offset,
size ratio) from bounding boxes, bucketed into discrete bins for use as
learned attention bias in the encoder. Also computes per-symbol size features.

Named binary relations (RIGHT, SUP, etc.) are kept for GUI visualization only.
"""

from __future__ import annotations

import math

import torch

from engine.stroke import BBox


# ── Pairwise geometry buckets (for encoder attention bias) ───────────

N_GEO_BUCKETS = 8

# 7 boundaries → 8 bins for each pairwise feature
_V_BOUNDS = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]   # vertical offset
_H_BOUNDS = [-1.0,  0.0,  0.5, 1.0, 2.0, 3.0, 5.0]    # horizontal offset
_S_BOUNDS = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]    # log2 size ratio


def _bucketize(value: float, bounds: list[float]) -> int:
    """Return bucket index for a value given sorted boundaries."""
    for i, b in enumerate(bounds):
        if value < b:
            return i
    return len(bounds)


def compute_geo_buckets(
    bboxes: list[BBox],
    max_n: int,
) -> torch.Tensor:
    """Compute bucketed pairwise geometry features.

    For each pair (i, j), computes 3 normalized geometric features and
    buckets each into N_GEO_BUCKETS bins:
      - v_off: (cy_j - cy_i) / median_h — vertical offset
      - h_off: (cx_j - cx_i) / median_h — horizontal offset (center-to-center)
      - size_r: log2(h_j / h_i) — log size ratio

    Args:
        bboxes: List of N bounding boxes (N <= max_n).
        max_n: Maximum number of symbols (for padding).

    Returns:
        Tensor of shape (3, max_n, max_n), dtype=long (bucket indices 0..7).
    """
    n = len(bboxes)
    buckets = torch.zeros(3, max_n, max_n, dtype=torch.long)

    if n == 0:
        return buckets

    heights = [b.h for b in bboxes]
    median_h = sorted(heights)[n // 2]
    if median_h < 0.001:
        median_h = 1.0

    for i in range(n):
        bi = bboxes[i]
        for j in range(n):
            if i == j:
                continue
            bj = bboxes[j]

            v_off = (bj.cy - bi.cy) / median_h
            h_off = (bj.cx - bi.cx) / median_h
            size_r = math.log2(bj.h / bi.h) if bi.h > 0 and bj.h > 0 else 0.0

            buckets[0, i, j] = _bucketize(v_off, _V_BOUNDS)
            buckets[1, i, j] = _bucketize(h_off, _H_BOUNDS)
            buckets[2, i, j] = _bucketize(size_r, _S_BOUNDS)

    return buckets


def compute_geo_buckets_from_bbox_list(
    bbox_list: list[list[float]],
    max_n: int,
) -> torch.Tensor:
    """Convenience: compute geo buckets from list of [x, y, w, h] lists."""
    bboxes = [BBox(x=b[0], y=b[1], w=b[2], h=b[3]) for b in bbox_list]
    return compute_geo_buckets(bboxes, max_n)


# ── Per-symbol size features ─────────────────────────────────────────

NUM_SIZE_BUCKETS = 4

# Height bucket boundaries (h / median_h)
_SH_BOUNDS = [0.5, 0.8, 1.2]  # → tiny | small | normal | large

# Y-offset bucket boundaries ((cy - median_cy) / median_h)
_SY_BOUNDS = [-0.3, 0.0, 0.3]  # → high-above | above | below | low-below


def compute_size_features(
    bboxes: list[BBox],
    max_n: int,
) -> torch.Tensor:
    """Compute per-symbol bucketed size features.

    Two bucketed features per symbol (4 buckets each):
      - height bucket: tiny(0) | small(1) | normal(2) | large(3)
      - y-offset bucket: high-above(0) | above(1) | below(2) | low-below(3)

    Returns:
        Tensor of shape (max_n, 2), dtype=long (bucket indices).
    """
    n = len(bboxes)
    feats = torch.zeros(max_n, 2, dtype=torch.long)

    if n == 0:
        return feats

    heights = [b.h for b in bboxes]
    centers_y = [b.cy for b in bboxes]

    median_h = sorted(heights)[n // 2]
    median_cy = sorted(centers_y)[n // 2]

    if median_h < 0.001:
        median_h = 1.0

    for i in range(n):
        rel_h = bboxes[i].h / median_h
        rel_y = (bboxes[i].cy - median_cy) / median_h
        feats[i, 0] = _bucketize(rel_h, _SH_BOUNDS)
        feats[i, 1] = _bucketize(rel_y, _SY_BOUNDS)

    return feats


def compute_features_from_bbox_list(
    bbox_list: list[list[float]],
    max_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute both geo buckets and size features from [x,y,w,h] lists.

    Used by training data pipeline. Vectorized for speed.

    Returns:
        (geo_buckets, size_feats) — (3, max_n, max_n), (max_n, 2)
    """
    n = len(bbox_list)
    geo = torch.zeros(3, max_n, max_n, dtype=torch.long)
    size = torch.zeros(max_n, 2, dtype=torch.long)

    if n == 0:
        return geo, size

    # Parse into tensor (n, 4): x, y, w, h
    bb = torch.tensor(bbox_list, dtype=torch.float32)
    h = bb[:, 3].clamp(min=0.001)
    cx = bb[:, 0] + bb[:, 2] * 0.5
    cy = bb[:, 1] + h * 0.5

    median_h = h.median().item()
    if median_h < 0.001:
        median_h = 1.0

    # ── Pairwise geo buckets (vectorized) ──
    v_off = (cy.unsqueeze(0) - cy.unsqueeze(1)) / median_h  # (n, n)
    h_off = (cx.unsqueeze(0) - cx.unsqueeze(1)) / median_h
    s_rat = (h.unsqueeze(0) / h.unsqueeze(1)).log2()        # (n, n)

    v_bounds = torch.tensor(_V_BOUNDS, dtype=torch.float32)
    h_bounds = torch.tensor(_H_BOUNDS, dtype=torch.float32)
    s_bounds = torch.tensor(_S_BOUNDS, dtype=torch.float32)

    geo[0, :n, :n] = torch.bucketize(v_off, v_bounds)
    geo[1, :n, :n] = torch.bucketize(h_off, h_bounds)
    geo[2, :n, :n] = torch.bucketize(s_rat, s_bounds)

    # ── Per-symbol size features (vectorized) ──
    median_cy = cy.median().item()
    rel_h = h / median_h
    rel_y = (cy - median_cy) / median_h

    sh_bounds = torch.tensor(_SH_BOUNDS, dtype=torch.float32)
    sy_bounds = torch.tensor(_SY_BOUNDS, dtype=torch.float32)

    size[:n, 0] = torch.bucketize(rel_h, sh_bounds)
    size[:n, 1] = torch.bucketize(rel_y, sy_bounds)

    return geo, size



# ── Named relations (GUI visualization only) ─────────────────────────

RIGHT = 0
SUP = 1
SUB = 2
ABOVE = 3
BELOW = 4
NUM_RELATIONS = 5

REL_NAMES = ["RIGHT", "SUP", "SUB", "ABOVE", "BELOW"]


def _h_overlap_ratio(a: BBox, b: BBox) -> float:
    """Horizontal overlap between two bboxes as fraction of the smaller width."""
    left = max(a.x, b.x)
    right = min(a.x2, b.x2)
    overlap = max(0.0, right - left)
    shorter = min(a.w, b.w)
    return overlap / shorter if shorter > 0 else 0.0


def _v_overlap_ratio(a: BBox, b: BBox) -> float:
    """Vertical overlap between two bboxes as fraction of the shorter height."""
    top = max(a.y, b.y)
    bot = min(a.y2, b.y2)
    overlap = max(0.0, bot - top)
    shorter = min(a.h, b.h)
    return overlap / shorter if shorter > 0 else 0.0


def compute_relations_from_bboxes(
    bboxes: list[BBox],
) -> list[list[list[int]]]:
    """Compute 5 binary adjacency matrices from bounding boxes.

    Used for GUI visualization only — NOT used by the model.
    """
    n = len(bboxes)
    matrices = [[[0] * n for _ in range(n)] for _ in range(NUM_RELATIONS)]

    for i in range(n):
        bi = bboxes[i]
        for j in range(n):
            if i == j:
                continue
            bj = bboxes[j]

            ref_h = max(bi.h, bj.h, 0.001)
            v_off = (bj.cy - bi.cy) / ref_h
            h_off = (bj.x - bi.x2) / ref_h
            v_overlap = _v_overlap_ratio(bi, bj)
            h_overlap = _h_overlap_ratio(bi, bj)

            if (abs(v_off) < 0.5
                    and h_off > -0.5 and h_off < 2.5
                    and v_overlap > 0.2):
                matrices[RIGHT][i][j] = 1

            if (v_off < -0.3
                    and h_off > -0.5 and h_off < 1.5):
                matrices[SUP][i][j] = 1

            if (v_off > 0.3
                    and h_off > -0.5 and h_off < 1.5):
                matrices[SUB][i][j] = 1

            if v_off < -0.3 and h_overlap > 0.3:
                matrices[ABOVE][i][j] = 1

            if v_off > 0.3 and h_overlap > 0.3:
                matrices[BELOW][i][j] = 1

    return matrices
