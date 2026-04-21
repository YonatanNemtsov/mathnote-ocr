"""Stroke and bounding box data structures.

Every Stroke has a required `id`. The id is a stable identifier within a
session; it never shifts when other strokes are added or removed.

The id must be supplied explicitly — there is no default. The public API
(MathOCR, Session) generates ids for callers that don't provide them.
Internal code must always provide an id.
"""

from dataclasses import dataclass, field

from mathnote_ocr.bbox import BBox


@dataclass
class StrokePoint:
    x: float
    y: float
    t: float = 0.0  # timestamp (optional)


@dataclass
class Stroke:
    id: int
    points: list[StrokePoint] = field(default_factory=list)
    bbox: BBox = field(default_factory=lambda: BBox(0, 0, 0, 0))
    width: float = 2.0  # pen width used to draw this stroke

    @staticmethod
    def from_dicts(points: list[dict], *, id: int, width: float = 2.0) -> "Stroke":
        """Create a Stroke from a list of {x, y, t?} dicts. `id` required."""
        stroke_points = [StrokePoint(p["x"], p["y"], p.get("t", 0.0)) for p in points]
        return Stroke.from_points(stroke_points, id=id, width=width)

    @staticmethod
    def from_points(
        points: list[StrokePoint],
        *,
        id: int,
        min_bbox_size: float = 6.0,
        width: float = 2.0,
    ) -> "Stroke":
        """Create a Stroke and compute its bounding box. `id` required.

        Tiny strokes (dots, taps) get their bbox expanded to min_bbox_size
        so they can participate in grouping with nearby strokes.
        """
        if not points:
            return Stroke(id=id, width=width)

        xs = [p.x for p in points]
        ys = [p.y for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        w = max_x - min_x
        h = max_y - min_y

        # Expand tiny bboxes to minimum size, centered on the stroke
        if w < min_bbox_size:
            cx = (min_x + max_x) / 2
            min_x = cx - min_bbox_size / 2
            w = min_bbox_size
        if h < min_bbox_size:
            cy = (min_y + max_y) / 2
            min_y = cy - min_bbox_size / 2
            h = min_bbox_size

        bbox = BBox(min_x, min_y, w, h)
        return Stroke(id=id, points=points, bbox=bbox, width=width)


def compute_bbox(strokes: list[Stroke]) -> BBox:
    """Compute the combined bounding box of multiple strokes."""
    if not strokes:
        return BBox(0, 0, 0, 0)
    min_x = min(s.bbox.x for s in strokes)
    min_y = min(s.bbox.y for s in strokes)
    max_x = max(s.bbox.x + s.bbox.w for s in strokes)
    max_y = max(s.bbox.y + s.bbox.h for s in strokes)
    return BBox(min_x, min_y, max_x - min_x, max_y - min_y)
