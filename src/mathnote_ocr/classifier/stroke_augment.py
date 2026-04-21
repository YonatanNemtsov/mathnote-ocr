"""Stroke-level augmentation for classifier training.

Augments raw (x,y) stroke points before rendering, producing more
natural variations than pixel-level transforms.
"""

import math
import random

from mathnote_ocr.engine.stroke import Stroke, StrokePoint


def augment_strokes(strokes: list[Stroke]) -> list[Stroke]:
    """Apply random stroke-level augmentations."""
    # Collect all points to compute center for affine transforms
    all_pts = [p for s in strokes for p in s.points]
    if not all_pts:
        return strokes

    cx = sum(p.x for p in all_pts) / len(all_pts)
    cy = sum(p.y for p in all_pts) / len(all_pts)

    # 1. Random affine: rotation + scale + shear
    strokes = _affine(strokes, cx, cy)

    # 2. Per-point jitter (hand tremor)
    strokes = _jitter(strokes)

    # 3. Per-stroke offset (multi-stroke alignment noise)
    if len(strokes) > 1:
        strokes = _stroke_offset(strokes)

    # Rebuild bboxes
    return [Stroke.from_points(s.points, id=s.id, width=s.width) for s in strokes]


def _affine(strokes: list[Stroke], cx: float, cy: float) -> list[Stroke]:
    """Random rotation, scale, and shear around center."""
    angle = random.gauss(0, 8)  # degrees, std=8
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    sx = random.uniform(0.8, 1.2)
    sy = random.uniform(0.8, 1.2)

    shear = random.gauss(0, 0.1)  # horizontal shear

    result = []
    for s in strokes:
        new_pts = []
        for p in s.points:
            # Center
            dx, dy = p.x - cx, p.y - cy
            # Shear
            dx = dx + shear * dy
            # Rotate
            rx = cos_a * dx - sin_a * dy
            ry = sin_a * dx + cos_a * dy
            # Scale
            rx *= sx
            ry *= sy
            new_pts.append(StrokePoint(cx + rx, cy + ry, p.t))
        result.append(Stroke(id=s.id, points=new_pts, bbox=s.bbox, width=s.width))
    return result


def _jitter(strokes: list[Stroke], scale: float = 0.015) -> list[Stroke]:
    """Add gaussian noise to each point. Scale is relative to symbol size."""
    all_pts = [p for s in strokes for p in s.points]
    xs = [p.x for p in all_pts]
    ys = [p.y for p in all_pts]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    sigma = scale * span

    result = []
    for s in strokes:
        new_pts = [
            StrokePoint(p.x + random.gauss(0, sigma), p.y + random.gauss(0, sigma), p.t)
            for p in s.points
        ]
        result.append(Stroke(id=s.id, points=new_pts, bbox=s.bbox, width=s.width))
    return result


def _stroke_offset(strokes: list[Stroke], scale: float = 0.02) -> list[Stroke]:
    """Shift each stroke independently by a small random amount."""
    all_pts = [p for s in strokes for p in s.points]
    xs = [p.x for p in all_pts]
    ys = [p.y for p in all_pts]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    sigma = scale * span

    result = []
    for s in strokes:
        ox = random.gauss(0, sigma)
        oy = random.gauss(0, sigma)
        new_pts = [StrokePoint(p.x + ox, p.y + oy, p.t) for p in s.points]
        result.append(Stroke(id=s.id, points=new_pts, bbox=s.bbox, width=s.width))
    return result
