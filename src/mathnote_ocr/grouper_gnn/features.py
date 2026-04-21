"""Feature extraction for stroke GNN: mini-renders + geometric features."""

import math

import numpy as np
import torch
from PIL import Image, ImageDraw

_SUPERSAMPLE = 2  # 2x supersample for mini-renders


def render_stroke_mini(
    points: list[dict],
    size: int = 32,
    stroke_width: float = 2.0,
    padding_ratio: float = 0.15,
) -> np.ndarray:
    """Render a single stroke to a small grayscale image.

    Args:
        points: List of {x, y} dicts.
        size: Output image size (square).
        stroke_width: Original stroke width in canvas pixels.
        padding_ratio: Fraction of canvas for padding.

    Returns:
        (size, size) uint8 numpy array, 0=ink, 255=white.
    """
    hi = size * _SUPERSAMPLE

    if not points:
        return np.full((size, size), 255, dtype=np.uint8)

    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox_w = max(max_x - min_x, 1.0)
    bbox_h = max(max_y - min_y, 1.0)

    usable = hi * (1.0 - 2 * padding_ratio)
    scale = usable / max(bbox_w, bbox_h)

    offset_x = (hi - bbox_w * scale) / 2
    offset_y = (hi - bbox_h * scale) / 2
    width = max(1, round(stroke_width * scale))

    img = Image.new("L", (hi, hi), 255)
    draw = ImageDraw.Draw(img)

    pts = [
        ((p["x"] - min_x) * scale + offset_x, (p["y"] - min_y) * scale + offset_y) for p in points
    ]

    if len(pts) == 1:
        x, y = pts[0]
        r = max(width, 1)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=0)
    elif len(pts) > 1:
        draw.line(pts, fill=0, width=width)
        for x, y in [pts[0], pts[-1]]:
            r = width / 2
            draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def _stroke_arc_length(points: list[dict]) -> float:
    """Sum of point-to-point distances."""
    total = 0.0
    for i in range(1, len(points)):
        dx = points[i]["x"] - points[i - 1]["x"]
        dy = points[i]["y"] - points[i - 1]["y"]
        total += math.sqrt(dx * dx + dy * dy)
    return total


def _stroke_direction(points: list[dict]) -> float:
    """Angle from first to last point in radians, normalized to [-1, 1]."""
    if len(points) < 2:
        return 0.0
    dx = points[-1]["x"] - points[0]["x"]
    dy = points[-1]["y"] - points[0]["y"]
    return math.atan2(dy, dx) / math.pi  # [-1, 1]


def compute_node_features(
    strokes: list[list[dict]],
    stroke_width: float = 2.0,
    render_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-stroke features for a full expression.

    Args:
        strokes: List of strokes, each a list of {x, y} point dicts.
        stroke_width: Original stroke width.
        render_size: Size of mini-renders.

    Returns:
        renders: (N, 1, render_size, render_size) float32 tensor, normalized to [-1, 1]
        geo: (N, 8) float32 tensor of geometric features
    """
    N = len(strokes)
    if N == 0:
        return (
            torch.zeros(0, 1, render_size, render_size),
            torch.zeros(0, 8),
        )

    # Compute expression-level bounding box for normalization
    all_xs, all_ys = [], []
    for pts in strokes:
        for p in pts:
            all_xs.append(p["x"])
            all_ys.append(p["y"])
    expr_min_x, expr_max_x = min(all_xs), max(all_xs)
    expr_min_y, expr_max_y = min(all_ys), max(all_ys)
    expr_w = max(expr_max_x - expr_min_x, 1.0)
    expr_h = max(expr_max_y - expr_min_y, 1.0)

    renders = np.zeros((N, 1, render_size, render_size), dtype=np.float32)
    geo = np.zeros((N, 8), dtype=np.float32)

    # Collect arc lengths for normalization
    arc_lengths = []
    point_counts = []
    for pts in strokes:
        arc_lengths.append(_stroke_arc_length(pts))
        point_counts.append(len(pts))
    max_arc = max(arc_lengths) if arc_lengths else 1.0
    max_pts = max(point_counts) if point_counts else 1.0

    for i, pts in enumerate(strokes):
        # Mini-render
        img = render_stroke_mini(pts, size=render_size, stroke_width=stroke_width)
        renders[i, 0] = (img.astype(np.float32) / 127.5) - 1.0  # normalize to [-1, 1]

        if not pts:
            continue

        # Stroke bbox
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        s_min_x, s_max_x = min(xs), max(xs)
        s_min_y, s_max_y = min(ys), max(ys)
        s_w = s_max_x - s_min_x
        s_h = s_max_y - s_min_y

        # Geometric features (all normalized to ~[0, 1] or [-1, 1])
        geo[i, 0] = (s_min_x - expr_min_x) / expr_w  # x position
        geo[i, 1] = (s_min_y - expr_min_y) / expr_h  # y position
        geo[i, 2] = max(s_w, 1.0) / expr_w  # width
        geo[i, 3] = max(s_h, 1.0) / expr_h  # height
        geo[i, 4] = max(s_w, 1.0) / max(s_h, 1.0)  # aspect ratio (clamped)
        geo[i, 4] = min(geo[i, 4], 10.0) / 10.0  # normalize to [0, 1]
        geo[i, 5] = arc_lengths[i] / max_arc if max_arc > 0 else 0.0  # arc length
        geo[i, 6] = point_counts[i] / max_pts if max_pts > 0 else 0.0  # point count
        geo[i, 7] = _stroke_direction(pts)  # direction [-1, 1]

    return torch.from_numpy(renders), torch.from_numpy(geo)


def compute_edge_features(strokes: list[list[dict]]) -> torch.Tensor:
    """Compute pairwise edge features between all strokes.

    Args:
        strokes: List of strokes, each a list of {x, y} point dicts.

    Returns:
        (N, N, 6) float32 tensor of edge features.
    """
    N = len(strokes)
    if N == 0:
        return torch.zeros(0, 0, 6)

    # Compute per-stroke bbox info
    bboxes = []  # (min_x, min_y, max_x, max_y, cx, cy, diag)
    for pts in strokes:
        if not pts:
            bboxes.append((0, 0, 0, 0, 0, 0, 1.0))
            continue
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        w = max(max_x - min_x, 1.0)
        h = max(max_y - min_y, 1.0)
        diag = math.sqrt(w * w + h * h)
        bboxes.append((min_x, min_y, max_x, max_y, cx, cy, diag))

    # Expression-level scale for normalization
    all_xs = [b[0] for b in bboxes] + [b[2] for b in bboxes]
    all_ys = [b[1] for b in bboxes] + [b[3] for b in bboxes]
    expr_w = max(max(all_xs) - min(all_xs), 1.0)
    expr_h = max(max(all_ys) - min(all_ys), 1.0)
    expr_scale = max(expr_w, expr_h)

    feats = np.zeros((N, N, 6), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            bi = bboxes[i]
            bj = bboxes[j]

            # Gap distance between bboxes (0 if overlapping)
            gap_x = max(0, max(bi[0], bj[0]) - min(bi[2], bj[2]))
            gap_y = max(0, max(bi[1], bj[1]) - min(bi[3], bj[3]))
            gap_dist = math.sqrt(gap_x * gap_x + gap_y * gap_y)
            feats[i, j, 0] = gap_dist / expr_scale  # normalized gap

            # Center offsets (signed)
            feats[i, j, 1] = (bj[4] - bi[4]) / expr_scale  # horizontal
            feats[i, j, 2] = (bj[5] - bi[5]) / expr_scale  # vertical

            # Size ratio (min/max diagonals)
            d_min = min(bi[6], bj[6])
            d_max = max(bi[6], bj[6])
            feats[i, j, 3] = d_min / d_max  # [0, 1]

            # Overlap IoU
            ix1 = max(bi[0], bj[0])
            iy1 = max(bi[1], bj[1])
            ix2 = min(bi[2], bj[2])
            iy2 = min(bi[3], bj[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_i = (bi[2] - bi[0]) * (bi[3] - bi[1])
            area_j = (bj[2] - bj[0]) * (bj[3] - bj[1])
            union = area_i + area_j - inter
            feats[i, j, 4] = inter / max(union, 1e-6)

            # Stroke order distance (normalized)
            feats[i, j, 5] = abs(i - j) / max(N - 1, 1)

    return torch.from_numpy(feats)


def compute_adjacency_mask(
    strokes: list[list[dict]],
    k: int = 6,
) -> torch.Tensor:
    """Build a sparse adjacency mask: each stroke connects to its k nearest neighbours.

    Uses bbox gap distance (same as the grouper). Every stroke always
    connects to itself (diagonal=True) so self-attention is preserved.

    Args:
        strokes: List of strokes, each a list of {x, y} point dicts.
        k: Number of nearest neighbours per stroke.

    Returns:
        (N, N) bool tensor — True means the pair is connected.
    """
    N = len(strokes)
    if N == 0:
        return torch.zeros(0, 0, dtype=torch.bool)

    # Compute bboxes
    bboxes = []
    for pts in strokes:
        if not pts:
            bboxes.append((0.0, 0.0, 0.0, 0.0))
            continue
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        bboxes.append((min(xs), min(ys), max(xs), max(ys)))

    # Pairwise gap distances
    dists = np.full((N, N), float("inf"), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                dists[i, j] = 0.0
                continue
            bi, bj = bboxes[i], bboxes[j]
            gap_x = max(0, max(bi[0], bj[0]) - min(bi[2], bj[2]))
            gap_y = max(0, max(bi[1], bj[1]) - min(bi[3], bj[3]))
            dists[i, j] = math.sqrt(gap_x * gap_x + gap_y * gap_y)

    # k-nearest neighbours (symmetric)
    k_actual = min(k, N - 1)
    mask = np.eye(N, dtype=bool)  # self-connections
    for i in range(N):
        if k_actual > 0:
            neighbours = np.argpartition(dists[i], k_actual)[: k_actual + 1]
            mask[i, neighbours] = True
            mask[neighbours, i] = True  # symmetric

    return torch.from_numpy(mask)
