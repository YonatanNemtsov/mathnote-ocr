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


def _affine(strokes: list[Stroke], cx: float, cy: float, shear_sigma: float = 0.1) -> list[Stroke]:
    """Random rotation, scale, and shear around center."""
    angle = random.gauss(0, 8)  # degrees, std=8
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    sx = random.uniform(0.8, 1.2)
    sy = random.uniform(0.8, 1.2)

    shear = random.gauss(0, shear_sigma)  # horizontal shear

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


# ── Strong augmentation: the basic one plus the variation real handwriting
#    shows beyond a global affine (classifier v13 onward) ───────────────────


def augment_strokes_strong(strokes: list[Stroke]) -> list[Stroke]:
    """The basic augmentation with a wider slant and its tremor smooth (points
    near each other move together), plus: points resampled at a
    random spacing (pen devices sample differently), a smooth local warp
    (loops fatter or thinner, stems bent), each stroke turned and scaled a
    little on its own, and stroke ends cut short or a stroke broken by a
    small gap. Each change is small enough to keep the symbol's identity:
    an o stays closed but for a sliver, a stroke loses at most a few percent."""
    all_pts = [p for s in strokes for p in s.points]
    if not all_pts:
        return strokes
    xs = [p.x for p in all_pts]
    ys = [p.y for p in all_pts]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)

    strokes = [_resample(s, span * random.uniform(0.005, 0.03)) for s in strokes]
    strokes = _affine(strokes, cx, cy, shear_sigma=0.15)
    strokes = _warp(strokes, cx, cy, span)
    if len(strokes) > 1:
        strokes = _stroke_affine(strokes)
        strokes = _stroke_offset(strokes)
    strokes = _smooth_jitter(strokes, span)
    strokes = [piece for s in strokes for piece in _cut(s)]
    return [Stroke.from_points(s.points, id=i, width=s.width) for i, s in enumerate(strokes)]


def _resample(s: Stroke, step: float) -> Stroke:
    """The stroke's path at one point every *step* (dots stay as they are)."""
    pts = s.points
    if len(pts) < 2:
        return s
    out = [pts[0]]
    carry = 0.0
    for a, b in zip(pts, pts[1:]):
        seg = math.hypot(b.x - a.x, b.y - a.y)
        d = step - carry
        while d <= seg:
            t = d / seg
            out.append(StrokePoint(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), a.t + t * (b.t - a.t)))
            d += step
        carry = (carry + seg) % step if seg > 0 else carry
    if out[-1] is not pts[-1]:
        out.append(pts[-1])
    return Stroke(id=s.id, points=out, bbox=s.bbox, width=s.width)


def _warp(strokes: list[Stroke], cx: float, cy: float, span: float) -> list[Stroke]:
    """A smooth random displacement field: a few low-frequency waves (under
    ~1.5 cycles across the symbol), each moving points by up to ~3% of its size."""
    waves = []
    for _ in range(2):
        for axis in (0, 1):
            fx, fy = random.uniform(-1.5, 1.5) / span, random.uniform(-1.5, 1.5) / span
            waves.append((axis, random.gauss(0, 0.02) * span, fx, fy, random.uniform(0, 2 * math.pi)))
    result = []
    for s in strokes:
        new_pts = []
        for p in s.points:
            dx = dy = 0.0
            for axis, amp, fx, fy, ph in waves:
                v = amp * math.sin(2 * math.pi * (fx * (p.x - cx) + fy * (p.y - cy)) + ph)
                if axis == 0:
                    dx += v
                else:
                    dy += v
            new_pts.append(StrokePoint(p.x + dx, p.y + dy, p.t))
        result.append(Stroke(id=s.id, points=new_pts, bbox=s.bbox, width=s.width))
    return result


def _smooth_jitter(strokes: list[Stroke], span: float, scale: float = 0.012, window: int = 5) -> list[Stroke]:
    """Hand tremor: gaussian noise averaged over *window* neighbouring points,
    so the path wobbles instead of fraying (the basic _jitter moves each point
    on its own, which frays densely sampled strokes)."""
    sigma = scale * span * math.sqrt(window)     # the average shrinks the noise by sqrt(window)
    half = window // 2
    result = []
    for s in strokes:
        n = len(s.points)
        nx = [random.gauss(0, sigma) for _ in range(n)]
        ny = [random.gauss(0, sigma) for _ in range(n)]
        new_pts = []
        for i, p in enumerate(s.points):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            k = hi - lo
            new_pts.append(StrokePoint(p.x + sum(nx[lo:hi]) / k, p.y + sum(ny[lo:hi]) / k, p.t))
        result.append(Stroke(id=s.id, points=new_pts, bbox=s.bbox, width=s.width))
    return result


def _stroke_affine(strokes: list[Stroke]) -> list[Stroke]:
    """Each stroke turned (std 4 degrees) and scaled (0.9-1.1) around its own centre."""
    result = []
    for s in strokes:
        if len(s.points) < 2:
            result.append(s)
            continue
        sx = sum(p.x for p in s.points) / len(s.points)
        sy = sum(p.y for p in s.points) / len(s.points)
        rad = math.radians(random.gauss(0, 4))
        c, si = math.cos(rad), math.sin(rad)
        k = random.uniform(0.9, 1.1)
        new_pts = [StrokePoint(sx + k * (c * (p.x - sx) - si * (p.y - sy)),
                               sy + k * (si * (p.x - sx) + c * (p.y - sy)), p.t) for p in s.points]
        result.append(Stroke(id=s.id, points=new_pts, bbox=s.bbox, width=s.width))
    return result


def _cut(s: Stroke) -> list[Stroke]:
    """Sometimes the stroke's ends cut short (up to 5% of its points each),
    rarely a long stroke broken by a gap of 1-3% of its points."""
    pts = s.points
    n = len(pts)
    if n < 8:
        return [s]
    if random.random() < 0.3:
        a = random.randint(0, max(0, int(0.05 * n)))
        b = n - random.randint(0, max(0, int(0.05 * n)))
        pts = pts[a:b]
        n = len(pts)
    if n >= 20 and random.random() < 0.1:
        g = max(1, int(random.uniform(0.01, 0.03) * n))
        at = random.randint(n // 4, 3 * n // 4)
        return [Stroke(id=s.id, points=pts[:at], bbox=s.bbox, width=s.width),
                Stroke(id=s.id, points=pts[at + g:], bbox=s.bbox, width=s.width)]
    return [Stroke(id=s.id, points=pts, bbox=s.bbox, width=s.width)]
