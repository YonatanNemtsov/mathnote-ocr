"""Render augmented bbox examples in a grid with handwritten strokes."""

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tree_parser.hw_bbox_augment import augment_bboxes
from tree_parser.gen_data import _process_batch, _worker_init


def get_stroke_loader(hw_dir: Path):
    cache = {}

    def get_strokes(name):
        if name in cache:
            return cache[name]
        d = hw_dir / name
        if not d.exists():
            cache[name] = None
            return None
        jsons = list(d.glob("*.json"))
        if not jsons:
            cache[name] = None
            return None
        with open(random.choice(jsons)) as f:
            data = json.load(f)
        cache[name] = data["strokes"]
        return data["strokes"]

    return get_strokes


def render_grid(
    items: list[dict],
    output_path: str,
    cols: int = 10,
    cell: int = 200,
):
    rows = (len(items) + cols - 1) // cols
    latex_h = 38
    total_h = cell + latex_h

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 9)
    except Exception:
        font = ImageFont.load_default()

    hw_dir = Path(__file__).parent.parent.parent / "data" / "symbols_from_expr"
    get_strokes = get_stroke_loader(hw_dir)

    img = Image.new("RGB", (cols * cell, rows * total_h), "black")
    draw = ImageDraw.Draw(img)

    for idx, t in enumerate(items):
        symbols = t["symbols"]
        tree = t["tree"]
        aug = augment_bboxes(symbols, tree)
        if aug is None:
            continue

        r, c = idx // cols, idx % cols
        ox, oy = c * cell, r * total_h

        # LaTeX label (word-wrapped)
        latex = t.get("latex", "")
        max_chars = cell // 6  # ~6px per char at font size 9
        lines = [latex[i:i+max_chars] for i in range(0, len(latex), max_chars)]
        for li, line in enumerate(lines[:3]):  # max 3 lines
            draw.text((ox + 2, oy + 1 + li * 12), line, fill="#888", font=font)

        # Bboxes + strokes
        boy = oy + latex_h
        draw.rectangle([ox, boy, ox + cell - 1, boy + cell - 1], outline="#333")
        for s in aug:
            x, y, w, h = s["bbox"]
            px = ox + x * (cell - 4) + 2
            py = boy + y * (cell - 4) + 2
            pw = w * (cell - 4)
            ph = h * (cell - 4)
            draw.rectangle([px, py, px + pw, py + ph], outline="#444")
            strokes = get_strokes(s["name"])
            if strokes:
                all_xs = [p["x"] for st in strokes for p in st]
                all_ys = [p["y"] for st in strokes for p in st]
                if all_xs:
                    sxmin, sxmax = min(all_xs), max(all_xs)
                    symin, symax = min(all_ys), max(all_ys)
                    sw = sxmax - sxmin or 1
                    sh = symax - symin or 1
                    for st in strokes:
                        pts = []
                        for p in st:
                            sx = px + (p["x"] - sxmin) / sw * pw
                            sy = py + (p["y"] - symin) / sh * ph
                            pts.append((sx, sy))
                        if len(pts) > 1:
                            draw.line(pts, fill="white", width=1)

    img.save(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("-o", "--output", default="data/tmp/augmented_v2.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cols", type=int, default=10)
    args = parser.parse_args()

    _worker_init(args.seed, "v2")
    random.seed(args.seed)
    items = _process_batch((max(2000, args.n * 20), 20))
    random.shuffle(items)
    items = items[: args.n]
    print(f"Got {len(items)} examples")

    render_grid(items, args.output, cols=args.cols)
