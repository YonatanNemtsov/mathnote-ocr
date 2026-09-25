"""Render stroke-JSON crop directories to a self-contained HTML gallery.

Each cell shows the rendered crop, its filename, and stroke count; with
--classify, the classifier's top prediction is shown and colored red when
it disagrees with the directory label. v0 of the data-browsing tools — a
static generator; the interactive workbench replaces it later.

Usage:
    python3.10 scripts/data/render_gallery.py data/shared/symbols_from_expr/I_cap data/shared/symbols/I_cap
    python3.10 scripts/data/render_gallery.py --classify --limit 60 data/shared/symbols/i
    open tmp/gallery.html
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
from pathlib import Path

from mathnote_ocr.engine.renderer import render_strokes
from mathnote_ocr.engine.stroke import Stroke

CELL = """
<div class="cell">
  <img src="data:image/png;base64,{b64}">
  <div class="cap">{name} · {n_strokes}str{pred}</div>
</div>"""

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>crop gallery</title>
<style>
  body {{ font: 13px -apple-system, sans-serif; margin: 20px; background: #fafafa; }}
  h2 {{ margin: 24px 0 8px; }} h2 small {{ color: #888; font-weight: normal; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .cell {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 6px; text-align: center; }}
  .cell img {{ display: block; image-rendering: auto; }}
  .cap {{ color: #555; margin-top: 4px; font-size: 11px; }}
  .bad {{ color: #c0392b; font-weight: bold; }}
  .ok {{ color: #27ae60; }}
</style>
{sections}
"""


def load_crop(path: Path):
    d = json.loads(path.read_text())
    strokes = [
        Stroke.from_dicts(pts, id=i, width=2.0) for i, pts in enumerate(d["strokes"])
    ]
    pts = [(p.x, p.y) for s in strokes for p in s.points]
    if not pts:
        return None
    source_size = max(d.get("canvas_width", 800), d.get("canvas_height", 400))
    bw = max(x for x, _ in pts) - min(x for x, _ in pts)
    bh = max(y for _, y in pts) - min(y for _, y in pts)
    size_feat = math.hypot(bw, bh) / max(source_size, 1.0)
    return strokes, source_size, size_feat, len(d["strokes"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", help="crop directories (stroke JSONs)")
    ap.add_argument("--out", default="tmp/gallery.html")
    ap.add_argument("--size", type=int, default=72, help="cell image size px")
    ap.add_argument("--limit", type=int, default=200, help="max crops per dir")
    ap.add_argument("--classify", action="store_true",
                    help="show classifier prediction, red = disagrees with dir label")
    ap.add_argument("--classifier-run", default="v9_combined")
    ap.add_argument("--weights-dir", default="weights",
                    help="searched before the bundled package weights")
    args = ap.parse_args()

    classifier = None
    if args.classify:
        from mathnote_ocr.classifier.inference import SymbolClassifier

        classifier = SymbolClassifier(run=args.classifier_run, weights_dir=args.weights_dir)

    sections = []
    for d in args.dirs:
        d = Path(d)
        label = d.name
        paths = sorted(p for p in d.glob("*.json") if p.stem.isdigit())[: args.limit]
        n_total = len(sorted(p for p in d.glob("*.json") if p.stem.isdigit()))

        crops, cls_imgs, cls_sfs = [], [], []
        for p in paths:
            loaded = load_crop(p)
            if loaded is None:
                continue
            strokes, source_size, size_feat, n_strokes = loaded
            img = render_strokes(strokes, canvas_size=args.size, source_size=source_size)
            crops.append((p, img, n_strokes))
            if classifier is not None:
                cls_imgs.append(
                    render_strokes(strokes, canvas_size=classifier.canvas_size,
                                   source_size=source_size)
                )
                cls_sfs.append(size_feat)

        preds = (
            [r.alternatives[0] for r in classifier.classify_batch(cls_imgs, size_feats=cls_sfs)]
            if classifier is not None and cls_imgs
            else [None] * len(crops)
        )

        cells = []
        n_mismatch = 0
        for (p, img, n_strokes), pred in zip(crops, preds):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            pred_html = ""
            if pred is not None:
                name, conf = pred
                bad = name != label
                n_mismatch += bad
                pred_html = (
                    f'<br><span class="{"bad" if bad else "ok"}">{name} {conf:.2f}</span>'
                )
            cells.append(CELL.format(b64=b64, name=p.stem, n_strokes=n_strokes, pred=pred_html))

        head = f"{d} <small>({len(crops)} of {n_total} shown"
        if classifier is not None:
            head += f", {n_mismatch} prediction≠label"
        head += ")</small>"
        sections.append(f"<h2>{head}</h2>\n<div class='grid'>{''.join(cells)}</div>")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.format(sections="\n".join(sections)))
    print(f"wrote {out} ({sum(1 for _ in sections)} sections)")


if __name__ == "__main__":
    main()
