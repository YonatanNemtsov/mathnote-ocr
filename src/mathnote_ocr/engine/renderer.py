"""Render strokes to PIL images for classification."""

from PIL import Image, ImageDraw

from mathnote_ocr.engine.stroke import Stroke, compute_bbox

_SUPERSAMPLE_DEFAULT = 4


def render_strokes(
    strokes: list[Stroke],
    canvas_size: int = 128,
    padding_ratio: float = 0.15,
    source_size: float | None = None,
    round_joins: bool = False,
    fill_pens: float | None = None,
) -> Image.Image:
    """
    Render strokes to a grayscale image.

    Renders at 4x resolution then downsamples with LANCZOS for smooth
    anti-aliased output. Each stroke uses its own ``Stroke.width``
    (line width in source canvas pixels, scaled with the coordinates).

    Args:
        strokes: List of strokes to render.
        canvas_size: Output image size (square).
        padding_ratio: Fraction of canvas to leave as padding on each side.
        source_size: Max dimension of the original drawing canvas (e.g. 800 for
                     an 800x400 canvas). When provided, caps the scale factor so
                     small symbols stay small.
        round_joins: Round the corners where a stroke's segments meet, as a pen
                     tip does. Off: each corner leaves a notch, so thick densely
                     sampled strokes come out frayed. A classifier is read with
                     the rendering it was trained with (its checkpoint's round_joins).
        fill_pens: Draw at the ink's real size: a symbol fills the image only
                   once it is this many pen widths across (the pen: the strokes'
                   median ``Stroke.width``); a smaller one — a dot, a comma, a
                   prime — stays that small in the image, drawn with its real pen.
                   Replaces ``source_size`` and the small-canvas exception. None:
                   the older rendering (every symbol blown up to fill the image).

    Returns:
        Grayscale PIL Image of size (canvas_size, canvas_size).
    """
    if not strokes or all(len(s.points) == 0 for s in strokes):
        return Image.new("L", (canvas_size, canvas_size), 255)

    # Scale supersampling with canvas size — 4x for 128, 2x for small
    supersample = _SUPERSAMPLE_DEFAULT if canvas_size >= 64 else 2
    hi = canvas_size * supersample

    bbox = compute_bbox(strokes)
    bbox_w = max(bbox.w, 1.0)
    bbox_h = max(bbox.h, 1.0)

    # Less padding for small canvases — maximize usable area
    effective_padding = padding_ratio if canvas_size >= 64 else 0.05
    usable = hi * (1.0 - 2 * effective_padding)
    scale = usable / max(bbox_w, bbox_h)

    if fill_pens is not None:
        widths = sorted(s.width for s in strokes if s.points)
        pen = widths[len(widths) // 2]
        scale = min(scale, usable / (fill_pens * pen))
        source_size = None

    # source_size cap keeps small symbols small at large canvas sizes,
    # but skip it for small canvases where every pixel matters
    if source_size is not None and source_size > 0 and canvas_size >= 64:
        max_scale = usable / (source_size * 0.1)
        scale = min(scale, max_scale)

    offset_x = (hi - bbox_w * scale) / 2
    offset_y = (hi - bbox_h * scale) / 2

    img = Image.new("L", (hi, hi), 255)
    draw = ImageDraw.Draw(img)

    for stroke in strokes:
        pts = [
            (
                (p.x - bbox.x) * scale + offset_x,
                (p.y - bbox.y) * scale + offset_y,
            )
            for p in stroke.points
        ]

        width = max(1, round(stroke.width * scale))

        if len(pts) == 1:
            x, y = pts[0]
            r = width
            draw.ellipse([x - r, y - r, x + r, y + r], fill=0)
        elif len(pts) > 1:
            draw.line(pts, fill=0, width=width, joint="curve" if round_joins else None)
            for x, y in [pts[0], pts[-1]]:
                r = width / 2
                draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    return img.resize((canvas_size, canvas_size), Image.LANCZOS)
