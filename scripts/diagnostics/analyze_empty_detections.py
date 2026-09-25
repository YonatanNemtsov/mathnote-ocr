"""Why do detections come back empty, and would skipping strokes help?

An empty detection means the grouper's exact cover found no partition
covering every stroke. For each empty example this script records:

  cause (a) — strokes with NO surviving candidate group (all rejected)
  cause (b) — every stroke has candidates, but no conflict-free full cover

then prototypes a skip-enabled cover (fewest skipped strokes first, then
score) and checks the skipped strokes against ground truth: which GT
symbol they belong to, and how well the partial result matches the
remaining strokes.

Usage:
    python3.10 scripts/diagnostics/analyze_empty_detections.py
    python3.10 scripts/diagnostics/analyze_empty_detections.py \
        --config ../math_ocr_web/configs/mixed_v10_backtrack_gnn.yaml --max-skips 3
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model_evaluation"))
from eval_handwritten_e2e import (  # noqa: E402
    gt_stroke_labels,
    load_examples,
    strokes_from_example,
)

from mathnote_ocr import MathOCR  # noqa: E402
from mathnote_ocr.engine import grouper  # noqa: E402

# ── Capture the exact-cover inputs of the most recent detect ─────────

_captured: dict = {}
_orig_find = grouper._find_best_partitions


def _capturing_find(n, scored_groups, top_k, max_results=100, **kw):
    _captured.clear()
    _captured.update(n=n, scored_groups=scored_groups, kw=kw)
    return _orig_find(n, scored_groups, top_k, max_results, **kw)


grouper._find_best_partitions = _capturing_find


# ── Prototype: exact cover that may skip strokes ─────────────────────


def cover_with_skips(n, scored_groups, max_skips, initial_covered=frozenset(),
                     initial_symbols=None, max_results=100):
    """Best partitions using exactly the fewest skips that make a cover exist.

    Returns (n_skips, [(score, symbols, skipped_positions), ...]) or
    (None, []) if even max_skips skips don't suffice.
    """
    stroke_to_groups = {i: [] for i in range(n)}
    for gi, (indices, _score, _sym) in enumerate(scored_groups):
        for s in indices:
            stroke_to_groups[s].append(gi)

    for budget in range(1, max_skips + 1):
        results = []

        def search(uncovered, symbols, skipped, score):
            if len(results) >= max_results:
                return
            if not uncovered:
                results.append((score, list(symbols), list(skipped)))
                return
            best_s, best_valid = None, None
            for s in uncovered:
                valid = [g for g in stroke_to_groups[s] if scored_groups[g][0] <= uncovered]
                if best_valid is None or len(valid) < len(best_valid):
                    best_s, best_valid = s, valid
            best_valid.sort(key=lambda g: scored_groups[g][1], reverse=True)
            for gi in best_valid:
                indices, conf, sym = scored_groups[gi]
                if any(
                    sym.name != "sqrt" and e.name != "sqrt" and grouper._symbols_conflict(sym.bbox, e.bbox)
                    for e in symbols
                ):
                    continue
                symbols.append(sym)
                search(uncovered - indices, symbols, skipped, score * conf)
                symbols.pop()
            if len(skipped) < budget:
                skipped.append(best_s)
                search(uncovered - {best_s}, symbols, skipped, score)
                skipped.pop()

        seed = list(initial_symbols) if initial_symbols else []
        search(frozenset(range(n)) - initial_covered, seed, [], 1.0)
        results = [r for r in results if len(r[2]) == budget]
        if results:
            results.sort(key=lambda r: r[0] ** (1.0 / max(len(r[1]), 1)), reverse=True)
            return budget, results
    return None, []


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


# ── Main ─────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../math_ocr_web/configs/mixed_v10_backtrack_gnn.yaml")
    ap.add_argument("--runs", nargs="+", default=["run_001", "run_002", "run_003"])
    ap.add_argument("--data-dir", default="data/shared/tree_handwritten")
    ap.add_argument("--max-skips", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ocr = MathOCR(config=args.config)
    total = 0
    empties = []
    t_normal = []

    for run in args.runs:
        for i, ex in enumerate(load_examples(Path(args.data_dir) / run / "train_strokes.jsonl")):
            strokes = strokes_from_example(ex)
            if not strokes:
                continue
            total += 1
            t0 = time.perf_counter()
            expr = ocr.detect(strokes, canvas_size=max(ex.get("canvas_width", 800), ex.get("canvas_height", 800)))
            t_normal.append(time.perf_counter() - t0)
            if len(expr.symbols) == 0:
                empties.append((run, i, ex, strokes, dict(_captured)))

    print(f"\n{total} expressions, {len(empties)} empty ({100 * len(empties) / max(total, 1):.1f}%)")
    print(f"normal detect: median {sorted(t_normal)[len(t_normal) // 2] * 1000:.0f} ms\n")

    causes = Counter()
    skips_needed = Counter()
    skipped_gt = Counter()
    fallback_ms = []
    rest_correct = rest_total = 0
    gt_missing = Counter()
    missing_names = Counter()
    gt_conflict = 0
    conflict_pairs = Counter()

    for run, i, ex, strokes, cap in empties:
        n, scored = cap["n"], cap["scored_groups"]
        init_cov = cap["kw"].get("initial_covered", frozenset())
        covered_by_some = set().union(*(g[0] for g in scored)) if scored else set()
        dead = sorted(set(range(n)) - covered_by_some - set(init_cov))
        cause = "a" if dead else "b"
        causes[cause] += 1

        t0 = time.perf_counter()
        k, results = cover_with_skips(n, scored, args.max_skips, init_cov, cap["kw"].get("initial_symbols"))
        fallback_ms.append((time.perf_counter() - t0) * 1000)
        skips_needed[k] += 1

        gt = gt_stroke_labels(ex)
        line = f"{run}#{i:<3} ({cause}) n={n:<3} gt: {ex['latex'][:50]}"

        # Was the ground-truth partition available, and what blocked it?
        gt_groups = gt_symbol_groups(ex)
        by_set = {g[0]: g for g in scored}
        missing = [(name, sorted(ss)) for name, ss in gt_groups if ss not in by_set]
        present = [(name, ss) for name, ss in gt_groups if ss in by_set]
        conflicts = []
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                (na, sa), (nb, sb) = present[a], present[b]
                ba, bb = by_set[sa][2].bbox, by_set[sb][2].bbox
                if "sqrt" in (by_set[sa][2].name, by_set[sb][2].name):
                    continue
                if grouper._symbols_conflict(ba, bb):
                    conflicts.append((na, nb))
        if missing:
            gt_missing[len(missing) > 0] += 1
            for name, _ in missing:
                missing_names[name] += 1
            line += f"\n      GT groups not among candidates: {missing}"
        if conflicts:
            gt_conflict += 1
            for pair in conflicts:
                conflict_pairs[tuple(sorted(pair))] += 1
            line += f"\n      GT symbols that conflict: {conflicts}"
        if dead:
            line += f"\n      dead strokes: {[(s, gt.get(s)) for s in dead]}"
        if results:
            score, syms, skipped = results[0]
            for s in skipped:
                skipped_gt[gt.get(s)] += 1
            # per-stroke label accuracy on the strokes the fallback kept
            for sym in syms:
                for st in sym.strokes:
                    rest_total += 1
                    rest_correct += gt.get(st.id) == sym.name
            detected = sorted(grouper._postprocess(syms), key=lambda s: s.bbox.x)
            latex, *_ = ocr.tree_parser.parse_with_tree(detected, None)
            line += f"\n      skip {k}: {[(s, gt.get(s)) for s in skipped]} -> partial: {latex[:60]}"
        else:
            line += f"\n      no cover within {args.max_skips} skips"
        print(line)

    if not empties:
        return
    print("\n── Summary ──")
    print(f"cause (a) dead strokes: {causes['a']}   cause (b) conflicts only: {causes['b']}")
    print(f"skips needed: {dict(sorted(skips_needed.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)))}")
    print(f"GT label of skipped strokes: {skipped_gt.most_common()}")
    print(f"GT partition blocked by a missing candidate: {gt_missing[True]}  "
          f"by a GT-vs-GT bbox conflict: {gt_conflict}")
    print(f"  missing GT symbols: {missing_names.most_common()}")
    print(f"  conflicting GT pairs: {conflict_pairs.most_common()}")
    if rest_total:
        print(f"kept-stroke label accuracy: {rest_correct}/{rest_total} = {100 * rest_correct / rest_total:.1f}%")
    fb = sorted(fallback_ms)
    print(f"fallback search: median {fb[len(fb) // 2]:.1f} ms, max {fb[-1]:.1f} ms")


if __name__ == "__main__":
    main()
