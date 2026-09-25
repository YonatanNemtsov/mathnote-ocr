"""Can the grouper even produce the ground-truth partition?

For every example, captures the exact-cover input (the scored candidate
groups) and checks the ground-truth partition against it:

  missing   — a GT symbol's stroke set is not among the candidates
              (rejected by confidence / singleton gate, or never enumerated)
  conflict  — all GT groups are candidates, but two GT symbols trip
              _symbols_conflict, so exact cover can never pick them together

Examples where the GT partition is unreachable are errors no scoring
change can fix; the tallies say which grouper rule is responsible.

Usage:
    python3.10 scripts/diagnostics/gt_reachability.py
    python3.10 scripts/diagnostics/gt_reachability.py --verbose
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model_evaluation"))
from eval_handwritten_e2e import load_examples, strokes_from_example  # noqa: E402

from mathnote_ocr import MathOCR  # noqa: E402
from mathnote_ocr.engine import grouper  # noqa: E402

_captured: dict = {}
_orig_find = grouper._find_best_partitions


def _capturing_find(n, scored_groups, top_k, max_results=100, **kw):
    _captured.update(n=n, scored_groups=scored_groups)
    return _orig_find(n, scored_groups, top_k, max_results, **kw)


grouper._find_best_partitions = _capturing_find


def gt_symbol_groups(ex) -> list[tuple[str, frozenset[int]]]:
    """Ground-truth symbols as (name, flat stroke positions)."""
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
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ocr = MathOCR(config=args.config)
    total = reachable = 0
    blocked_by = Counter()        # "missing" / "conflict" / "both"
    missing_names = Counter()
    conflict_pairs = Counter()
    conflict_with_frac_bar = 0

    for run in args.runs:
        for i, ex in enumerate(load_examples(Path(args.data_dir) / run / "train_strokes.jsonl")):
            strokes = strokes_from_example(ex)
            if not strokes:
                continue
            total += 1
            _captured.clear()
            ocr.detect(strokes, canvas_size=max(ex.get("canvas_width", 800), ex.get("canvas_height", 800)))
            by_set = {g[0]: g[2] for g in _captured.get("scored_groups", [])}

            gt = gt_symbol_groups(ex)
            missing = [name for name, ss in gt if ss not in by_set]
            present = [(name, by_set[ss]) for name, ss in gt if ss in by_set]
            conflicts = []
            for a in range(len(present)):
                for b in range(a + 1, len(present)):
                    (na, sa), (nb, sb) = present[a], present[b]
                    if grouper._symbols_clash(sa.name, sa.bbox, sb.name, sb.bbox):
                        conflicts.append(tuple(sorted((na, nb))))

            if not missing and not conflicts:
                reachable += 1
                continue
            blocked_by["both" if missing and conflicts else "missing" if missing else "conflict"] += 1
            missing_names.update(missing)
            conflict_pairs.update(conflicts)
            if any("frac_bar" in p for p in conflicts):
                conflict_with_frac_bar += 1
            if args.verbose:
                print(f"{run}#{i:<3} missing={missing} conflicts={conflicts}  gt: {ex['latex'][:60]}")

    print(f"\n{total} expressions: GT partition reachable in {reachable} "
          f"({100 * reachable / total:.1f}%), unreachable in {total - reachable}")
    print(f"  blocked by: missing candidate only {blocked_by['missing']}, "
          f"conflict only {blocked_by['conflict']}, both {blocked_by['both']}")
    print(f"  expressions with a GT conflict involving frac_bar: {conflict_with_frac_bar}")
    print(f"\nmissing GT symbols: {missing_names.most_common(25)}")
    print(f"\nconflicting GT pairs: {conflict_pairs.most_common(25)}")


if __name__ == "__main__":
    main()
