"""How well does forced alignment recover the ground-truth segmentation?

For every handwritten example: hide the segmentation, align the strokes to
the expression (mathnote_ocr.align), compare with the ground truth.

Targets:
  latex  — symbols derived from the example's LaTeX (the real use case)
  gt     — the ground-truth symbol names (isolates the search from parsing)

Metrics: per-stroke label accuracy (sum/Sigma_up, prod/Pi_up count as
equal), expressions aligned entirely right, no-cover failures, budget
hits, time. For comparison, the unconstrained engine reads ~85% of strokes
right on the same data (eval_handwritten_e2e.py).

Usage:
    python3.10 scripts/diagnostics/eval_alignment.py
    python3.10 scripts/diagnostics/eval_alignment.py --target gt --verbose
"""

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model_evaluation"))
from eval_handwritten_e2e import load_examples, normalize_latex, strokes_from_example  # noqa: E402

from mathnote_ocr import MathOCR  # noqa: E402
from mathnote_ocr.align import UnknownSymbol, align, latex_to_labels, parse_aligned, same_symbol  # noqa: E402


def gt_groups(ex) -> list[tuple[str, frozenset[int]]]:
    out, flat = [], 0
    for sym in ex["symbols"]:
        ids = []
        for stroke in sym["strokes"]:
            if stroke:
                ids.append(flat)
            flat += 1
        if ids:
            out.append((sym["name"], frozenset(ids)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../math_ocr_web/configs/mixed_v10_backtrack_gnn.yaml")
    ap.add_argument("--runs", nargs="+", default=["run_001", "run_002", "run_003"])
    ap.add_argument("--data-dir", default="data/shared/tree_handwritten")
    ap.add_argument("--target", choices=["latex", "gt"], default="latex")
    ap.add_argument("--node-budget", type=int, default=200_000)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ocr = MathOCR(config=args.config)
    names = ocr.classifier.label_names
    n_ex = exact = no_cover = budget = unknown = 0
    strokes_total = strokes_ok = 0
    seg_exact = 0
    times, node_counts = [], []
    confusions = Counter()
    failures = []
    # structure check: does the tree parser rebuild the target LaTeX from the alignment?
    structure = Counter()   # (alignment right?, structure agrees?) -> count

    for run in args.runs:
        for i, ex in enumerate(load_examples(Path(args.data_dir) / run / "train_strokes.jsonl")):
            strokes = strokes_from_example(ex)
            if not strokes:
                continue
            n_ex += 1
            gt = gt_groups(ex)
            gt_label = {p: name for name, g in gt for p in g}
            strokes_total += len(gt_label)
            if args.target == "latex":
                try:
                    labels = latex_to_labels(ex["latex"], names)
                except UnknownSymbol as e:
                    unknown += 1
                    failures.append((run, i, f"unknown: {e}", ex["latex"]))
                    continue
            else:
                labels = [name for name, _g in gt]
            canvas = max(ex.get("canvas_width", 800), ex.get("canvas_height", 800))
            t0 = time.perf_counter()
            a = align(ocr, strokes, labels, canvas_size=canvas, node_budget=args.node_budget)
            times.append(time.perf_counter() - t0)
            if a is None:
                no_cover += 1
                failures.append((run, i, "no cover", ex["latex"]))
                continue
            node_counts.append(a.nodes)
            budget += not a.complete
            pred_label = {p: s.label for s in a.symbols for p in s.stroke_ids}
            ok = [same_symbol(pred_label.get(p, ""), name) for p, name in gt_label.items()]
            strokes_ok += sum(ok)
            for p, name in gt_label.items():
                if not same_symbol(pred_label.get(p, ""), name):
                    confusions[(name, pred_label.get(p))] += 1
            segs = {frozenset(s.stroke_ids) for s in a.symbols}
            seg_exact += segs == {g for _n, g in gt}
            random.seed(0)   # the tree parser's test-time jitter
            agrees = normalize_latex(parse_aligned(ocr, strokes, a)) == normalize_latex(ex["latex"])
            right = all(ok) and segs == {g for _n, g in gt}
            structure[(right, agrees)] += 1
            if right:
                exact += 1
            else:
                bad = sorted({gt_label[p] for p, good in zip(gt_label, ok) if not good})
                failures.append((run, i, f"wrong strokes in {bad}", ex["latex"]))

    print(f"\n{n_ex} expressions, target = {args.target}")
    print(f"  per-stroke label accuracy : {strokes_ok}/{strokes_total} = {100 * strokes_ok / strokes_total:.1f}%")
    print(f"  expressions entirely right: {exact}/{n_ex} = {100 * exact / n_ex:.1f}%  "
          f"(segmentation exact: {seg_exact})")
    print(f"  no cover found: {no_cover}   budget hit: {budget}   unparseable LaTeX: {unknown}")
    if times:
        ts = sorted(times)
        ns = sorted(node_counts) or [0]
        print(f"  time: median {1000 * ts[len(ts) // 2]:.0f} ms, max {1000 * ts[-1]:.0f} ms; "
              f"nodes: median {ns[len(ns) // 2]}, max {ns[-1]}")
    print(f"  top confusions (gt -> aligned): {confusions.most_common(12)}")
    agree = structure[(True, True)] + structure[(False, True)]
    print(f"  structure check (tree parser rebuilds the LaTeX): {agree}/{sum(structure.values())} agree")
    print(f"    right alignments: {structure[(True, True)]} agree, {structure[(True, False)]} disagree   "
          f"wrong alignments: {structure[(False, True)]} agree, {structure[(False, False)]} disagree")
    if args.verbose:
        for f in failures:
            print("   ", f)


if __name__ == "__main__":
    main()
