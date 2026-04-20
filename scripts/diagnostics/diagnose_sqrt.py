"""Diagnose sqrt overlap: compare font bboxes vs augmented bboxes."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont

from mathnote_ocr.latex_utils.glyphs import _extract_glyphs
from mathnote_ocr.tree_parser.gen_data import latex_to_tree_labels
from mathnote_ocr.tree_parser.hw_bbox_augment import augment_bboxes

COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F0B27A", "#82E0AA", "#F1948A", "#AED6F1", "#D7BDE2",
]


def render_bboxes(draw, bboxes, names, ox, oy, cell, label, font, tree=None):
    """Render a set of bboxes in a cell."""
    draw.rectangle([ox, oy, ox + cell - 1, oy + cell - 1], outline="#555")
    draw.text((ox + 4, oy + 4), label, fill="white", font=font)

    for i, (bbox, name) in enumerate(zip(bboxes, names)):
        x, y, w, h = bbox
        px = ox + x * (cell - 20) + 10
        py = oy + 30 + y * (cell - 50) + 10
        pw = w * (cell - 20)
        ph = h * (cell - 50)
        color = COLORS[i % len(COLORS)]
        draw.rectangle([px, py, px + pw, py + ph], outline=color)
        draw.text((px + 1, py + 1), f"{i}:{name}", fill=color, font=font)

    # Draw parent arrows if tree provided
    if tree:
        for i, node in enumerate(tree):
            p = node["parent"]
            if p < 0:
                continue
            # Draw line from child center to parent center
            cx = ox + bboxes[i][0] * (cell - 20) + 10 + bboxes[i][2] * (cell - 20) / 2
            cy = oy + 30 + bboxes[i][1] * (cell - 50) + 10 + bboxes[i][3] * (cell - 50) / 2
            px_p = ox + bboxes[p][0] * (cell - 20) + 10 + bboxes[p][2] * (cell - 20) / 2
            py_p = oy + 30 + bboxes[p][1] * (cell - 50) + 10 + bboxes[p][3] * (cell - 50) / 2
            draw.line([(cx, cy), (px_p, py_p)], fill="#333", width=1)


def main():
    latex = r"3{\cdot}\sqrt{\frac{r}{n{\geq}229}}{\neq}\sqrt{p}"

    random.seed(42)

    # Extract font glyphs
    glyphs = _extract_glyphs(latex)
    if glyphs is None:
        print("Failed to parse LaTeX")
        return

    n = len(glyphs)
    tree_labels = latex_to_tree_labels(latex, n)
    if tree_labels is None:
        print("Failed to generate tree labels")
        return

    tree = [{"parent": p, "edge_type": e, "order": o} for p, e, o in tree_labels]
    symbols = [{"name": g["name"], "bbox": g["bbox"]} for g in glyphs]

    # Print font bboxes
    print(f"Expression: {latex}")
    print(f"\n{'idx':>3} {'name':>10} {'parent':>6} {'edge':>5} {'order':>5}  font_bbox")
    print("-" * 75)
    for i in range(n):
        p, e, o = tree_labels[i]
        b = glyphs[i]["bbox"]
        print(f"{i:3d} {glyphs[i]['name']:>10} {p:6d} {e:5d} {o:5d}  "
              f"x={b[0]:.4f} y={b[1]:.4f} w={b[2]:.4f} h={b[3]:.4f}  "
              f"right={b[0]+b[2]:.4f}")

    # Augment
    aug = augment_bboxes(symbols, tree)
    if aug is None:
        print("Augmentation failed")
        return

    print(f"\n{'idx':>3} {'name':>10}  augmented_bbox")
    print("-" * 65)
    for i in range(n):
        b = aug[i]["bbox"]
        print(f"{i:3d} {aug[i]['name']:>10}  "
              f"x={b[0]:.4f} y={b[1]:.4f} w={b[2]:.4f} h={b[3]:.4f}  "
              f"right={b[0]+b[2]:.4f}")

    # Find the sqrt symbols and their right extents vs neq left
    print("\n--- Key comparisons ---")
    # Find first sqrt (symbol 2 typically)
    sqrt_indices = [i for i in range(n) if glyphs[i]["name"] == "sqrt"]
    neq_indices = [i for i in range(n) if glyphs[i]["name"] == "neq"]

    for si in sqrt_indices:
        fb = glyphs[si]["bbox"]
        ab = aug[si]["bbox"]
        print(f"sqrt[{si}]: font right={fb[0]+fb[2]:.4f}, aug right={ab[0]+ab[2]:.4f}")

    for ni in neq_indices:
        fb = glyphs[ni]["bbox"]
        ab = aug[ni]["bbox"]
        print(f" neq[{ni}]: font left={fb[0]:.4f}, aug left={ab[0]:.4f}")

    # Render side by side
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 10)
    except Exception:
        font = ImageFont.load_default()

    cell = 600
    img = Image.new("RGB", (cell * 2 + 20, cell + 60), "black")
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((10, 5), latex, fill="#888", font=font)

    font_bboxes = [g["bbox"] for g in glyphs]
    names = [g["name"] for g in glyphs]
    aug_bboxes = [a["bbox"] for a in aug]

    render_bboxes(draw, font_bboxes, names, 10, 30, cell, "FONT", font, tree)
    render_bboxes(draw, aug_bboxes, names, cell + 20, 30, cell, "AUGMENTED", font, tree)

    out = "data/tmp/sqrt_diagnosis.png"
    img.save(out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
