"""A/B variants of the exact-cover conflict rule (grouper._symbols_clash).

Each variant replaces the rule in-process, then runs the handwritten e2e
eval (with --dump, for diff_stroke_flips.py) and the GT-reachability
check. Motivation: symbols touching a fraction bar trip the rule because
its threshold scales with the AVERAGE diagonal, which a long bar inflates.

Result (2026-09-26, 285 exprs): min_diag_frac_bar won (+61/-2 strokes,
per-stroke 82.4 -> 85.1%) and is now the engine rule, so "base" below is
the adopted rule; the other variants are kept for re-running the A/B.

Usage:
    python3.10 scripts/diagnostics/conflict_rule_variants.py --out-dir tmp/conflict_variants
    python3.10 scripts/diagnostics/diff_stroke_flips.py \
        --dump-a tmp/conflict_variants/base.jsonl --dump-b tmp/conflict_variants/min_diag.jsonl
"""

import argparse
import runpy
import sys
from pathlib import Path

from mathnote_ocr.engine import grouper

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent / "model_evaluation" / "eval_handwritten_e2e.py"
REACH = HERE / "gt_reachability.py"

_base_clash = grouper._symbols_clash


def _centre_dist(a, b) -> float:
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


def _clash_scaled(scale):
    """Same rule, centre distance normalised by scale(diag_a, diag_b)."""

    def clash(name_a, bbox_a, name_b, bbox_b, threshold=0.32):
        if "sqrt" in (name_a, name_b):
            return False
        if not grouper._bboxes_overlap(bbox_a, bbox_b):
            return False
        s = scale(bbox_a.diagonal, bbox_b.diagonal)
        return s > 0 and _centre_dist(bbox_a, bbox_b) / s < threshold

    return clash


def _exempt_frac_bar(name_a, bbox_a, name_b, bbox_b):
    if "frac_bar" in (name_a, name_b):
        return False
    return _base_clash(name_a, bbox_a, name_b, bbox_b)


def _min_diag_for_frac_bar(name_a, bbox_a, name_b, bbox_b):
    if "frac_bar" in (name_a, name_b):
        return _clash_scaled(min)(name_a, bbox_a, name_b, bbox_b)
    return _base_clash(name_a, bbox_a, name_b, bbox_b)


VARIANTS = {
    "base": _base_clash,
    "exempt_frac_bar": _exempt_frac_bar,         # frac_bar never clashes (like sqrt)
    "min_diag": _clash_scaled(min),              # normalise by the SMALLER symbol, all pairs
    "min_diag_frac_bar": _min_diag_for_frac_bar,  # smaller-symbol normalisation only with frac_bar
}


def _run(script: Path, argv: list[str]) -> None:
    sys.argv = [str(script), *argv]
    runpy.run_path(str(script), run_name="__main__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../math_ocr_web/configs/mixed_v10_backtrack_gnn.yaml")
    ap.add_argument("--out-dir", default="tmp/conflict_variants")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name in args.variants:
        grouper._symbols_clash = VARIANTS[name]
        print(f"\n{'#' * 30} {name} {'#' * 30}", flush=True)
        _run(EVAL, ["--config", args.config, "--dump", str(out / f"{name}.jsonl")])
        _run(REACH, ["--config", args.config])
    grouper._symbols_clash = _base_clash


if __name__ == "__main__":
    main()
