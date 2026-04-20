"""Render strokes to PIL images for classification."""

from PIL import Image, ImageDraw

from mathnote_ocr.engine.stroke import Stroke, compute_bbox

_SUPERSAMPLE_DEFAULT = 4


def render_strokes(
    strokes: list[Stroke],
    canvas_size: int = 128,
    stroke_width: float = 2.0,
    padding_ratio: float = 0.15,
    source_size: float | None = None,
) -> Image.Image:
    """
    Render strokes to a grayscale image.

    Renders at 4x resolution then downsamples with LANCZOS for smooth
    anti-aliased output.

    Args:
        strokes: List of strokes to render.
        canvas_size: Output image size (square).
        stroke_width: Line width in source canvas pixels. Scaled proportionally
                      with the symbol coordinates.
        padding_ratio: Fraction of canvas to leave as padding on each side.
        source_size: Max dimension of the original drawing canvas (e.g. 800 for
                     an 800x400 canvas). When provided, caps the scale factor so
                     small symbols stay small.

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

    # source_size cap keeps small symbols small at large canvas sizes,
    # but skip it for small canvases where every pixel matters
    if source_size is not None and source_size > 0 and canvas_size >= 64:
        max_scale = usable / (source_size * 0.1)
        scale = min(scale, max_scale)

    offset_x = (hi - bbox_w * scale) / 2
    offset_y = (hi - bbox_h * scale) / 2

    width = max(1, round(stroke_width * scale))

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

        if len(pts) == 1:
            x, y = pts[0]
            r = width
            draw.ellipse([x - r, y - r, x + r, y + r], fill=0)
        elif len(pts) > 1:
            draw.line(pts, fill=0, width=width)
            for x, y in [pts[0], pts[-1]]:
                r = width / 2
                draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    return img.resize((canvas_size, canvas_size), Image.LANCZOS)
