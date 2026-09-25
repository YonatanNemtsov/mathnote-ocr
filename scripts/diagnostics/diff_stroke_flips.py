"""Per-stroke A/B diff between two engine configs on handwritten runs.

For every stroke, compares GT symbol vs each engine's predicted symbol and
reports flips: strokes A got right but B got wrong (regressions) and vice
versa (wins), aggregated by GT class and by (GT -> wrong-pred) pair.

Usage:
    python3.10 scripts/diagnostics/diff_stroke_flips.py \
        --config-a <a.yaml> --config-b <b.yaml> [--runs run_001 ...]
    # or compare two code versions via eval_handwritten_e2e.py --dump files:
    python3.10 scripts/diagnostics/diff_stroke_flips.py \
        --dump-a before.jsonl --dump-b after.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model_evaluation"))
from eval_handwritten_e2e import (  # noqa: E402
    gt_stroke_labels,
    load_examples,
    normalize_latex,
    pred_stroke_labels,
    strokes_from_example,
)

from mathnote_ocr import MathOCR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-a", help="Baseline config (name or path)")
    ap.add_argument("--config-b", help="Candidate config (name or path)")
    ap.add_argument("--dump-a", help="Baseline predictions (eval_handwritten_e2e.py --dump)")
    ap.add_argument("--dump-b", help="Candidate predictions (eval_handwritten_e2e.py --dump)")
    ap.add_argument("--runs", nargs="+", default=["run_001", "run_002", "run_003"])
    ap.add_argument("--data-dir", default="data/shared/tree_handwritten")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    use_dumps = bool(args.dump_a and args.dump_b)
    if not use_dumps and not (args.config_a and args.config_b):
        ap.error("give --config-a/--config-b or --dump-a/--dump-b")
    if use_dumps:
        print(f"A = {args.dump_a}\nB = {args.dump_b}")
        dumps = {tag: {(r["run"], r["idx"]): r for r in map(json.loads, open(path))}
                 for tag, path in (("a", args.dump_a), ("b", args.dump_b))}
    else:
        print(f"A = {args.config_a}\nB = {args.config_b}")
        ocr_a = MathOCR(config=args.config_a)
        ocr_b = MathOCR(config=args.config_b)
    expr_flips = []                # (run, idx, gt, a_pred, b_pred, a_ok, b_ok)

    per_class = Counter()          # gt -> total strokes
    a_ok = Counter()               # gt -> strokes A correct
    b_ok = Counter()               # gt -> strokes B correct
    regress = Counter()            # (gt, b_pred) for A-ok B-wrong strokes
    win = Counter()                # (gt, a_pred) for B-ok A-wrong strokes

    for run in args.runs:
        path = Path(args.data_dir) / run / "train_strokes.jsonl"
        if not path.exists():
            print(f"[skip] {path}")
            continue
        examples = load_examples(path)
        if args.limit:
            examples = examples[: args.limit]
        print(f"=== {run}: {len(examples)} examples ===", flush=True)
        for i, ex in enumerate(examples):
            strokes = strokes_from_example(ex)
            if not strokes:
                continue
            gt = gt_stroke_labels(ex)
            canvas = max(ex.get("canvas_width", 800), ex.get("canvas_height", 800))
            preds = {}
            if use_dumps:
                recs = {tag: dumps[tag].get((run, i)) for tag in ("a", "b")}
                if not all(recs.values()):
                    continue
                for tag, r in recs.items():
                    preds[tag] = {int(k): v for k, v in r["strokes"].items()}
                ok = {tag: normalize_latex(r["pred"]) == normalize_latex(r["gt"]) for tag, r in recs.items()}
                if ok["a"] != ok["b"]:
                    expr_flips.append((run, i, ex["latex"], recs["a"]["pred"], recs["b"]["pred"], ok["a"], ok["b"]))
            else:
                for tag, ocr in (("a", ocr_a), ("b", ocr_b)):
                    try:
                        preds[tag] = pred_stroke_labels(ocr.detect(strokes, canvas_size=canvas))
                    except Exception:
                        preds[tag] = {}
            for sid, sym in gt.items():
                pa = preds["a"].get(sid)
                pb = preds["b"].get(sid)
                per_class[sym] += 1
                if pa == sym:
                    a_ok[sym] += 1
                if pb == sym:
                    b_ok[sym] += 1
                if pa == sym and pb != sym:
                    regress[(sym, pb)] += 1
                elif pb == sym and pa != sym:
                    win[(sym, pa)] += 1
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(examples)}", flush=True)

    n_reg, n_win = sum(regress.values()), sum(win.values())
    print(f"\nflipped strokes: {n_reg} A-ok→B-wrong, {n_win} B-ok→A-wrong "
          f"(net {n_win - n_reg:+d} for B)\n")

    print(f"{'gt class':12s} {'n':>5s} {'A ok':>6s} {'B ok':>6s} {'Δ':>5s}")
    rows = sorted(per_class, key=lambda s: (b_ok[s] - a_ok[s]))
    for sym in rows:
        d = b_ok[sym] - a_ok[sym]
        if d != 0:
            print(f"{sym:12s} {per_class[sym]:5d} {a_ok[sym]:6d} {b_ok[sym]:6d} {d:+5d}")

    print("\ntop regressions (gt -> B's wrong read):")
    for (sym, pb), c in regress.most_common(20):
        print(f"  {c:4d}  {sym} -> {pb}")
    print("\ntop wins (gt -> A's wrong read):")
    for (sym, pa), c in win.most_common(10):
        print(f"  {c:4d}  {sym} -> {pa}")

    if expr_flips:
        fixed = sum(1 for f in expr_flips if f[6])
        print(f"\nexpressions flipped (normalized match): {fixed} fixed by B, "
              f"{len(expr_flips) - fixed} broken by B")
        for run, i, gt, pa, pb, _oa, ob in expr_flips:
            print(f"  {'FIXED ' if ob else 'BROKEN'} {run}#{i}  gt: {gt}\n"
                  f"         A: {pa}\n         B: {pb}")


if __name__ == "__main__":
    main()
