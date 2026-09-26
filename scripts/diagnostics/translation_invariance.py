"""Is the reading independent of where the expression sits on the canvas?

Reads every handwritten example in place and shifted (e.g. far into
negative coordinates, as after panning the web canvas) and compares.
A panned-view bug came from clamping jittered boxes at 0 (evidence.py).

Usage:
    python3.10 scripts/diagnostics/translation_invariance.py [--shift -1500 -1500]
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model_evaluation"))
from eval_handwritten_e2e import load_examples, normalize_latex, strokes_from_example  # noqa: E402

from mathnote_ocr import MathOCR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../math_ocr_web/configs/mixed_v10_backtrack_gnn.yaml")
    ap.add_argument("--runs", nargs="+", default=["run_001", "run_002", "run_003"])
    ap.add_argument("--data-dir", default="data/shared/tree_handwritten")
    ap.add_argument("--shift", nargs=2, type=float, action="append", help="dx dy (repeatable)")
    args = ap.parse_args()
    shifts = args.shift or [(-1500.0, -1500.0), (800.0, 800.0)]

    ocr = MathOCR(config=args.config)
    n = 0
    right = {s: 0 for s in [(0.0, 0.0), *map(tuple, shifts)]}
    differ = {s: 0 for s in right}
    for run in args.runs:
        for ex in load_examples(Path(args.data_dir) / run / "train_strokes.jsonl"):
            strokes = strokes_from_example(ex)
            if not strokes:
                continue
            n += 1
            canvas = max(ex.get("canvas_width", 800), ex.get("canvas_height", 800))
            base = None
            for dx, dy in right:
                moved = [[(p[0] + dx, p[1] + dy, *p[2:]) for p in s] for s in strokes]
                random.seed(0)
                latex = normalize_latex(ocr.detect(moved, canvas_size=canvas).latex)
                base = latex if base is None else base
                right[(dx, dy)] += latex == normalize_latex(ex["latex"])
                differ[(dx, dy)] += latex != base
    print(f"{n} expressions")
    for s in right:
        print(f"  shift {s}: {right[s]}/{n} right ({100 * right[s] / n:.1f}%), reading differs from in-place: {differ[s]}")


if __name__ == "__main__":
    main()
