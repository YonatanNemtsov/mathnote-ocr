"""How often do two CROSSING strokes belong to different symbols?

For every pair of strokes in the ground truth, checks whether their
polylines intersect, and tallies same-symbol vs different-symbol crossings
(by GT class pair). If different-symbol crossings are rare, a grouping
that splits crossing strokes into different symbols is suspect.

Usage:
    python3.10 scripts/diagnostics/crossing_strokes.py
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model_evaluation"))
from eval_handwritten_e2e import load_examples  # noqa: E402


def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


def _seg_cross(p1, p2, q1, q2) -> bool:
    d1, d2 = _ccw(q1, q2, p1), _ccw(q1, q2, p2)
    d3, d4 = _ccw(p1, p2, q1), _ccw(p1, p2, q2)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


def strokes_cross(a, b) -> bool:
    ax = [p[0] for p in a]; ay = [p[1] for p in a]
    bx = [p[0] for p in b]; by = [p[1] for p in b]
    if max(ax) < min(bx) or max(bx) < min(ax) or max(ay) < min(by) or max(by) < min(ay):
        return False
    return any(
        _seg_cross(a[i], a[i + 1], b[j], b[j + 1])
        for i in range(len(a) - 1) for j in range(len(b) - 1)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["run_001", "run_002", "run_003"])
    ap.add_argument("--data-dir", default="data/shared/tree_handwritten")
    args = ap.parse_args()

    same = diff = pairs = 0
    same_cls, diff_cls = Counter(), Counter()
    for run in args.runs:
        for ex in load_examples(Path(args.data_dir) / run / "train_strokes.jsonl"):
            flat = [(sym["name"], si, [(p["x"], p["y"]) for p in s])
                    for si, sym in enumerate(ex["symbols"]) for s in sym["strokes"] if len(s) > 1]
            for i in range(len(flat)):
                for j in range(i + 1, len(flat)):
                    (na, sa, a), (nb, sb, b) = flat[i], flat[j]
                    pairs += 1
                    if not strokes_cross(a, b):
                        continue
                    if sa == sb:
                        same += 1
                        same_cls[na] += 1
                    else:
                        diff += 1
                        diff_cls[tuple(sorted((na, nb)))] += 1
    print(f"{pairs} stroke pairs; crossing: {same} same-symbol, {diff} different-symbol "
          f"({100 * diff / max(same + diff, 1):.1f}% of crossings)")
    print(f"\nsame-symbol crossings by class: {same_cls.most_common(15)}")
    print(f"\ndifferent-symbol crossings by pair: {diff_cls.most_common(25)}")


if __name__ == "__main__":
    main()
