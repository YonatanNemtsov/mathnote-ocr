"""Audit a crop pool: classifier shape-read vs directory label, per class.

Surfaces classes where drawings may not match their labels (e.g. capitals
drawn as lowercase). High disagreement = review that class visually with
scripts/data/render_gallery.py before using the pool for shape training or
semantic-supervision tables.

Usage:
    python3.10 scripts/data/audit_faithfulness.py data/shared/symbols_from_expr
    python3.10 scripts/data/audit_faithfulness.py data/shared/symbols --report tmp/symbols_audit.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from mathnote_ocr.classifier.inference import SymbolClassifier
from mathnote_ocr.engine.renderer import render_strokes
from mathnote_ocr.engine.stroke import Stroke


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pool", help="pool directory with per-class crop subdirs")
    ap.add_argument("--classifier-run", default="v9_combined")
    ap.add_argument("--weights-dir", default="weights",
                    help="searched before the bundled package weights")
    ap.add_argument("--report", default="tmp/faithfulness_report.json")
    ap.add_argument("--min-disagree", type=float, default=0.0,
                    help="only print classes with disagreement above this rate")
    args = ap.parse_args()

    clf = SymbolClassifier(run=args.classifier_run, weights_dir=args.weights_dir)
    pool = Path(args.pool)
    report = {}

    for class_dir in sorted(d for d in pool.iterdir() if d.is_dir()):
        label = class_dir.name
        images, sfs, files = [], [], []
        for p in sorted(class_dir.glob("*.json")):
            d = json.loads(p.read_text())
            if "strokes" not in d:
                continue
            strokes = [Stroke.from_dicts(pts, id=i, width=2.0)
                       for i, pts in enumerate(d["strokes"])]
            pts = [(pt.x, pt.y) for s in strokes for pt in s.points]
            if not pts:
                continue
            src = max(d.get("canvas_width", 800), d.get("canvas_height", 400))
            bw = max(x for x, _ in pts) - min(x for x, _ in pts)
            bh = max(y for _, y in pts) - min(y for _, y in pts)
            images.append(render_strokes(strokes, canvas_size=clf.canvas_size, source_size=src))
            sfs.append(math.hypot(bw, bh) / max(src, 1.0))
            files.append(p.name)
        if not images:
            continue

        preds = [r.alternatives[0][0] for r in clf.classify_batch(images, size_feats=sfs)]
        disagree = [(f, p) for f, p in zip(files, preds) if p != label]
        report[label] = {
            "n": len(files),
            "disagree": len(disagree),
            "rate": len(disagree) / len(files),
            "wrong_preds": Counter(p for _, p in disagree).most_common(3),
            "files": [[f, p] for f, p in disagree],  # [filename, shape-read]
        }

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))

    rows = sorted(report.items(), key=lambda kv: -kv[1]["rate"])
    total = sum(r["n"] for r in report.values())
    total_dis = sum(r["disagree"] for r in report.values())
    print(f"{pool}: {total} crops, {total_dis} shape-read≠label "
          f"({total_dis / max(total, 1):.1%})   report → {out}\n")
    print(f"{'label':14s} {'n':>5s} {'≠label':>7s} {'rate':>6s}  top shape-reads")
    for label, r in rows:
        if r["rate"] < args.min_disagree or r["disagree"] == 0:
            continue
        wp = ", ".join(f"{n}×{c}" for n, c in r["wrong_preds"])
        print(f"{label:14s} {r['n']:5d} {r['disagree']:7d} {r['rate']:6.1%}  {wp}")


if __name__ == "__main__":
    main()
