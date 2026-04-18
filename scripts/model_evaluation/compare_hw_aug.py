"""Compare real handwritten bboxes vs augmented font bboxes side by side."""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image, ImageDraw, ImageFont

from latex_utils.glyphs import _extract_glyphs
from tree_parser.hw_bbox_augment import augment_bboxes
from tree_parser.gen_data import latex_to_tree_labels


COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F0B27A", "#82E0AA", "#F1948A", "#AED6F1", "#D7BDE2",
]


def _match_glyphs_to_hw(glyphs, tree_labels, hw_symbols, hw_tree):
    """Match font-order glyphs to HW-order symbols using tree isomorphism.

    Returns list of font bboxes in HW order, or None if matching fails.
    """
    n = len(hw_symbols)
    if len(glyphs) != n or len(tree_labels) != n:
        return None

    # Build children maps for both trees
    font_children = defaultdict(list)
    for i, (p, e, _o) in enumerate(tree_labels):
        font_children[p].append((i, e, glyphs[i]["name"]))

    hw_children = defaultdict(list)
    for i, t in enumerate(hw_tree):
        hw_children[t["parent"]].append((i, t["edge_type"], hw_symbols[i]["name"]))

    font_to_hw = {}

    def match(hw_parent, font_parent):
        fc = defaultdict(list)
        for ci, et, nm in font_children[font_parent]:
            fc[(et, nm)].append(ci)
        hc = defaultdict(list)
        for ci, et, nm in hw_children[hw_parent]:
            hc[(et, nm)].append(ci)
        for key in hc:
            h_kids = hc[key]
            f_kids = fc.get(key, [])
            if len(h_kids) != len(f_kids):
                return False
            # Sort by x to disambiguate same (et, name) siblings
            h_sorted = sorted(h_kids, key=lambda i: hw_symbols[i]["bbox"][0])
            f_sorted = sorted(f_kids, key=lambda i: glyphs[i]["bbox"][0])
            for hi, fi in zip(h_sorted, f_sorted):
                font_to_hw[fi] = hi
                if not match(hi, fi):
                    return False
        return True

    # Match roots
    f_roots = defaultdict(list)
    h_roots = defaultdict(list)
    for i, (p, _e, _o) in enumerate(tree_labels):
        if p == -1:
            f_roots[glyphs[i]["name"]].append(i)
    for i, t in enumerate(hw_tree):
        if t["parent"] == -1:
            h_roots[hw_symbols[i]["name"]].append(i)

    for nm in h_roots:
        hl = h_roots[nm]
        fl = f_roots.get(nm, [])
        if len(hl) != len(fl):
            return None
        hl_s = sorted(hl, key=lambda i: hw_symbols[i]["bbox"][0])
        fl_s = sorted(fl, key=lambda i: glyphs[i]["bbox"][0])
        for hi, fi in zip(hl_s, fl_s):
            font_to_hw[fi] = hi
            if not match(hi, fi):
                return None

    if len(font_to_hw) != n:
        return None

    # Build font bboxes in HW order
    hw_to_font = {v: k for k, v in font_to_hw.items()}
    return [glyphs[hw_to_font[i]]["bbox"] for i in range(n)]


def draw_bboxes(draw, bboxes, names, ox, oy, w, h):
    """Draw bboxes in a cell."""
    draw.rectangle([ox, oy, ox + w - 1, oy + h - 1], outline="#333")
    for i, (bbox, name) in enumerate(zip(bboxes, names)):
        x, y, bw, bh = bbox
        px = ox + x * (w - 4) + 2
        py = oy + y * (h - 4) + 2
        pw = bw * (w - 4)
        ph = bh * (h - 4)
        color = COLORS[i % len(COLORS)]
        draw.rectangle([px, py, px + pw, py + ph], outline=color)
        draw.text((px + 1, py + 1), name[:3], fill=color)


def main():
    random.seed(42)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 9)
    except Exception:
        font = ImageFont.load_default()

    # Load handwritten data (clean runs only)
    hw_data = []
    for run in ["run_001", "run_002", "run_003"]:
        path = Path(f"data/shared/tree_handwritten/{run}/train.jsonl")
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                try:
                    hw_data.append(json.loads(line))
                except Exception:
                    continue

    # For each HW expression, generate augmented version using HW tree
    pairs = []
    for item in hw_data:
        latex = item["latex"]
        hw_symbols = item["symbols"]
        hw_tree = item["tree"]
        n = len(hw_symbols)
        names = [s["name"] for s in hw_symbols]

        # Normalize HW bboxes to [0,1] range
        hw_bboxes = [s["bbox"] for s in hw_symbols]
        all_x = [b[0] for b in hw_bboxes]
        all_y = [b[1] for b in hw_bboxes]
        all_x2 = [b[0] + b[2] for b in hw_bboxes]
        all_y2 = [b[1] + b[3] for b in hw_bboxes]
        xmin, ymin = min(all_x), min(all_y)
        xmax, ymax = max(all_x2), max(all_y2)
        ref = max(xmax - xmin, ymax - ymin, 1e-6)
        hw_norm = [
            [(b[0] - xmin) / ref, (b[1] - ymin) / ref, b[2] / ref, b[3] / ref]
            for b in hw_bboxes
        ]

        # Get font glyphs and tree labels
        glyphs = _extract_glyphs(latex)
        if glyphs is None or len(glyphs) != n:
            continue

        tree_labels = latex_to_tree_labels(latex, n)
        if tree_labels is None:
            continue

        # Match font glyphs to HW order using tree isomorphism
        font_bboxes_hw_order = _match_glyphs_to_hw(
            glyphs, tree_labels, hw_symbols, hw_tree
        )
        if font_bboxes_hw_order is None:
            continue

        # Augment using HW tree directly — output is in HW order
        aug_symbols = [{"name": names[i], "bbox": font_bboxes_hw_order[i]}
                       for i in range(n)]
        aug = augment_bboxes(aug_symbols, hw_tree)
        if aug is None:
            continue

        aug_bboxes = [a["bbox"] for a in aug]
        aug_names = [a["name"] for a in aug]
        if aug_names != names:
            continue

        pairs.append({
            "latex": latex,
            "hw_bboxes": hw_norm,
            "hw_names": names,
            "aug_bboxes": aug_bboxes,
            "aug_names": aug_names,
        })

    print(f"Matched {len(pairs)} / {len(hw_data)} expressions")

    # Render grid: each row has HW on left, AUG on right
    cols = 8  # 4 pairs per row
    cell = 150
    label_h = 24
    n_pairs = min(len(pairs), 80)
    rows = (n_pairs + cols // 2 - 1) // (cols // 2)

    img = Image.new("RGB", (cols * cell, rows * (cell + label_h)), "black")
    draw = ImageDraw.Draw(img)

    for idx in range(n_pairs):
        p = pairs[idx]
        r = idx // (cols // 2)
        c = (idx % (cols // 2)) * 2

        ox_hw = c * cell
        ox_aug = (c + 1) * cell
        oy = r * (cell + label_h)

        # LaTeX label
        latex = p["latex"]
        max_chars = (cell * 2) // 6
        draw.text((ox_hw + 2, oy + 1), latex[:max_chars], fill="#888", font=font)
        if len(latex) > max_chars:
            draw.text((ox_hw + 2, oy + 12), latex[max_chars:max_chars*2], fill="#888", font=font)

        boy = oy + label_h
        # HW label
        draw.text((ox_hw + 2, boy + 1), "HW", fill="#4F4", font=font)
        draw_bboxes(draw, p["hw_bboxes"], p["hw_names"], ox_hw, boy, cell, cell)
        # AUG label
        draw.text((ox_aug + 2, boy + 1), "AUG", fill="#F84", font=font)
        draw_bboxes(draw, p["aug_bboxes"], p["aug_names"], ox_aug, boy, cell, cell)

    out = "data/tmp/compare_hw_aug.png"
    img.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
