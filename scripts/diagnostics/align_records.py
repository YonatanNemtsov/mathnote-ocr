"""Forced alignment on the web app's confirmed records.

Each record holds the strokes, the confirmed LaTeX and the confirmed
stroke -> symbol assignment. Hide the assignment, align the strokes to the
LaTeX (align.py), compare with the record.

Targets:
  latex  — symbols derived from the record's LaTeX (the real use case)
  names  — the record's own symbol names (isolates the search from parsing)

    python3.10 scripts/diagnostics/align_records.py
    python3.10 scripts/diagnostics/align_records.py --records ../math_ocr_web/data/feedback.jsonl --verbose
"""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from mathnote_ocr import MathOCR
from mathnote_ocr.align import UnknownSymbol, align, latex_to_labels, same_symbol

WEB = Path(__file__).resolve().parents[3] / "math_ocr_web"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def strokes_and_truth(rec: dict):
    """Flat strokes [(x, y), …] and the record's assignment: stroke index -> symbol name."""
    strokes, truth, groups = [], {}, []
    for sym in rec.get("symbols", []):
        ids = []
        for st in sym.get("strokes", []):
            if not st:
                continue
            truth[len(strokes)] = sym["name"]
            ids.append(len(strokes))
            strokes.append([(p[0], p[1]) for p in st])
        if ids:
            groups.append(frozenset(ids))
    return strokes, truth, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(WEB / "configs" / "mixed_v10_backtrack_gnn.yaml"))
    ap.add_argument("--records", nargs="+", default=[str(WEB / "data_prod" / "feedback.jsonl"),
                                                     str(WEB / "data" / "feedback.jsonl")])
    ap.add_argument("--target", choices=["latex", "names"], default="latex")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ocr = MathOCR(config=args.config)
    names = ocr.classifier.label_names
    for path in map(Path, args.records):
        recs = load(path)
        if not recs:
            continue
        n = exact = seg_exact = no_cover = skipped = 0
        ok_strokes = all_strokes = 0
        confusions, skips, failures, times = Counter(), Counter(), [], []
        for i, rec in enumerate(recs):
            strokes, truth, groups = strokes_and_truth(rec)
            if not strokes:
                continue
            if args.target == "latex":
                try:
                    labels = latex_to_labels(rec["latex"], names)
                except UnknownSymbol as e:
                    skipped += 1
                    skips[str(e).split(":")[0][:40]] += 1
                    continue
            else:
                labels = [s["name"] for s in rec["symbols"] if any(s.get("strokes"))]
            n += 1
            canvas = max(rec.get("canvas_width") or 800, rec.get("canvas_height") or 800)
            random.seed(0)
            t0 = time.perf_counter()
            a = align(ocr, strokes, labels, canvas_size=canvas)
            times.append(time.perf_counter() - t0)
            all_strokes += len(truth)
            if a is None:
                no_cover += 1
                failures.append((i, "no cover", rec["latex"]))
                continue
            pred = {p: s.label for s in a.symbols for p in s.stroke_ids}
            good = [same_symbol(pred.get(p, ""), name) for p, name in truth.items()]
            ok_strokes += sum(good)
            for p, name in truth.items():
                if not same_symbol(pred.get(p, ""), name):
                    confusions[(name, pred.get(p))] += 1
            segs_right = {frozenset(s.stroke_ids) for s in a.symbols} == set(groups)
            seg_exact += segs_right
            if all(good) and segs_right:
                exact += 1
            else:
                bad = sorted({truth[p] for p, g in zip(truth, good) if not g})
                failures.append((i, f"wrong: {bad}" if bad else "segmentation differs", rec["latex"]))

        print(f"\n{path.parent.name}/{path.name}: {len(recs)} records, {n} aligned, {skipped} skipped (LaTeX not in the vocabulary)")
        if not n:
            continue
        print(f"  expressions entirely right : {exact}/{n} = {100 * exact / n:.1f}%   (segmentation exact: {seg_exact})")
        print(f"  strokes labelled right     : {ok_strokes}/{all_strokes} = {100 * ok_strokes / max(all_strokes, 1):.1f}%")
        print(f"  no cover found             : {no_cover}")
        ts = sorted(times)
        print(f"  time: median {1000 * ts[len(ts) // 2]:.0f} ms, max {1000 * ts[-1]:.0f} ms")
        if skips:
            print(f"  skipped because: {dict(skips.most_common(6))}")
        if confusions:
            print(f"  confusions (record -> aligned): {confusions.most_common(8)}")
        if args.verbose:
            for f in failures:
                print("   ", f)


if __name__ == "__main__":
    main()
