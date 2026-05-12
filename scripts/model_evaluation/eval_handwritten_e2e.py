"""End-to-end accuracy on handwritten datasets.

Drives the full pipeline: strokes -> grouper -> classifier -> tree parser -> LaTeX.

Usage:
    python3.10 scripts/model_evaluation/eval_handwritten_e2e.py
    python3.10 scripts/model_evaluation/eval_handwritten_e2e.py --config default
    python3.10 scripts/model_evaluation/eval_handwritten_e2e.py --runs run_001 run_002 --verbose
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from mathnote_ocr import MathOCR


_MACRO_RE = re.compile(r"\\[A-Za-z]+")


def _strip_macro_braces(s: str) -> str:
    """Remove redundant single-macro brace wrappers: `{\nabla}` -> `\nabla`."""
    # Repeat until stable — nested cases like {{\nabla}}.
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\{(\\[A-Za-z]+)\}", r"\1", s)
    return s


def _strip_outer_braces(s: str) -> str:
    """If the whole string is wrapped in matched braces, peel them."""
    while len(s) >= 2 and s[0] == "{" and s[-1] == "}":
        depth = 0
        peel = True
        for i, c in enumerate(s):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    peel = False
                    break
        if peel:
            s = s[1:-1]
        else:
            break
    return s


def normalize_latex(s: str) -> str:
    """Tolerate harmless rendering differences."""
    s = s.replace(r"{\prod}", r"\Pi").replace(r"\prod", r"\Pi")
    s = s.replace(r"{\sum}", r"\Sigma").replace(r"\sum", r"\Sigma")
    # function name + single-char arg: \sin{x} == \sin x
    s = re.sub(r"(\\(?:sin|cos|tan|log|ln|lim))\{([^{}])\}", r"\1\2", s)
    s = re.sub(r"\{(\\(?:sin|cos|tan|log|ln|lim))\}\{([^{}])\}", r"{\1}\2", s)
    s = _strip_macro_braces(s)
    s = _strip_outer_braces(s)
    s = re.sub(r"\s+", "", s)
    return s


def load_examples(path: Path) -> list[dict]:
    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def strokes_from_example(ex: dict) -> list[list[tuple[float, float]]]:
    """Flatten symbol-grouped strokes into a flat stroke list of (x, y) tuples."""
    out = []
    for sym in ex["symbols"]:
        for stroke in sym["strokes"]:
            pts = [(p["x"], p["y"]) for p in stroke]
            if pts:
                out.append(pts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default", help="Config name or path")
    ap.add_argument(
        "--runs",
        nargs="+",
        default=["run_001", "run_002", "run_003"],
        help="Handwritten run directories to evaluate",
    )
    ap.add_argument(
        "--data-dir",
        default="data/shared/tree_handwritten",
        help="Base directory holding the run subfolders",
    )
    ap.add_argument("--verbose", action="store_true", help="Print every mismatch")
    ap.add_argument("--limit", type=int, default=0, help="Cap examples per run (0 = all)")
    args = ap.parse_args()

    print(f"Loading MathOCR(config={args.config!r}) ...")
    ocr = MathOCR(config=args.config)
    print("Loaded.\n")

    grand_total = 0
    grand_exact = 0
    grand_norm = 0
    grand_err = 0
    grand_time = 0.0
    per_run_summary = []

    for run in args.runs:
        path = Path(args.data_dir) / run / "train_strokes.jsonl"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        examples = load_examples(path)
        if args.limit:
            examples = examples[: args.limit]
        n = len(examples)
        print(f"=== {run}: {n} examples ===")

        exact = 0
        norm = 0
        errors: list[tuple[int, str, str]] = []
        empties = 0
        run_start = time.time()

        for i, ex in enumerate(examples):
            gt = ex["latex"]
            strokes = strokes_from_example(ex)
            if not strokes:
                empties += 1
                continue
            try:
                expr = ocr.detect(strokes, canvas_size=max(ex.get("canvas_width", 800), ex.get("canvas_height", 800)))
                pred = expr.latex
            except Exception as e:
                pred = f"<ERROR: {type(e).__name__}: {e}>"
            if pred == gt:
                exact += 1
                norm += 1
            elif normalize_latex(pred) == normalize_latex(gt):
                norm += 1
            else:
                errors.append((i, gt, pred))

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{n} ...")

        run_time = time.time() - run_start
        real_err = len(errors)
        denom = n - empties
        per_run_summary.append((run, denom, exact, norm, real_err, run_time))
        print(
            f"  exact: {exact}/{denom} = {exact / denom:.1%}    "
            f"normalized: {norm}/{denom} = {norm / denom:.1%}    "
            f"errors: {real_err}    time: {run_time:.1f}s"
        )

        if args.verbose and errors:
            print("\n  Mismatches:")
            for idx, gt_l, pred_l in errors:
                print(f"    [{idx}] GT:   {gt_l}")
                print(f"         Pred: {pred_l}")
        print()

        grand_total += denom
        grand_exact += exact
        grand_norm += norm
        grand_err += real_err
        grand_time += run_time

    print("─" * 64)
    print(f"{'Run':<14} {'N':>5} {'Exact':>12} {'Normalized':>14} {'Errs':>6} {'Time':>7}")
    print("─" * 64)
    for run, n, ex, nm, er, t in per_run_summary:
        print(f"{run:<14} {n:>5} {ex:>4}/{n:<3} {ex/n:>5.1%} "
              f"  {nm:>4}/{n:<3} {nm/n:>5.1%}   {er:>4} {t:>6.1f}s")
    if grand_total:
        print("─" * 64)
        print(
            f"{'TOTAL':<14} {grand_total:>5} "
            f"{grand_exact:>4}/{grand_total:<3} {grand_exact / grand_total:>5.1%} "
            f"  {grand_norm:>4}/{grand_total:<3} {grand_norm / grand_total:>5.1%}"
            f"   {grand_err:>4} {grand_time:>6.1f}s"
        )


if __name__ == "__main__":
    main()
