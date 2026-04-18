"""Evaluate the grouper+classifier on handwritten expression data.

Runs the full grouper pipeline (stroke grouping + classification) on
each expression's raw strokes, then compares with ground truth symbols.

Reports:
  - Symbol count mismatches (grouper found more/fewer symbols)
  - Misclassified symbols (wrong name after grouping)
  - Missing/extra symbols
  - Common confusion pairs

Usage:
    cd math_ocr_v2
    python3.10 tools/eval_grouper.py data/shared/tree_handwritten/run_001/train_strokes.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.stroke import Stroke, BBox
from engine.grouper import group_and_classify
from classifier.inference import SymbolClassifier
from grouper_gnn.inference import GNNGrouper
from engine.grouper import GrouperParams

_SIMILAR_MAP = GrouperParams().similar_symbol_map


def _symbols_match(gt_name: str, pred_name: str) -> bool:
    """Check if gt and pred are equivalent (same or in same similar group)."""
    if gt_name == pred_name:
        return True
    group = _SIMILAR_MAP.get(gt_name)
    return group is not None and pred_name in group


def load_strokes_data(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def expr_strokes(item: dict) -> list[Stroke]:
    """Flatten all per-symbol strokes into a single stroke list."""
    all_strokes = []
    for sym in item["symbols"]:
        for stroke_points in sym["strokes"]:
            all_strokes.append(Stroke.from_dicts(stroke_points))
    return all_strokes


def match_symbols(gt_symbols, pred_symbols, canvas_w, canvas_h):
    """Match predicted symbols to GT by stroke index overlap (greedy).

    GT symbols know which strokes they own (sequential order from the
    strokes file). Predicted symbols have stroke_indices. We match by
    IoU of stroke index sets.
    """
    # Build GT stroke index sets: symbols are stored in order, each
    # symbol's strokes are consecutive in the flat stroke list.
    gt_stroke_sets = []
    stroke_offset = 0
    for gt in gt_symbols:
        n_strokes = len(gt["strokes"])
        gt_stroke_sets.append(set(range(stroke_offset, stroke_offset + n_strokes)))
        stroke_offset += n_strokes

    matched = []
    used_pred = set()

    for gi in range(len(gt_symbols)):
        gt_set = gt_stroke_sets[gi]
        best_pi = None
        best_iou = 0.0

        for pi, pred in enumerate(pred_symbols):
            if pi in used_pred:
                continue
            pred_set = set(pred.stroke_indices)
            inter = len(gt_set & pred_set)
            union = len(gt_set | pred_set)
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_pi = pi

        if best_pi is not None and best_iou > 0.3:
            matched.append((gi, best_pi))
            used_pred.add(best_pi)

    return matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["data/shared/tree_handwritten/run_001/train_strokes.jsonl"])
    ap.add_argument("--classifier-run", type=str, default="default")
    ap.add_argument("--gnn", action="store_true", help="Use GNN grouper instead of Algorithm X")
    ap.add_argument("--gnn-run", type=str, default=None, help="GNN checkpoint run name")
    args = ap.parse_args()
    paths = args.paths

    gnn_grouper = None
    if args.gnn:
        kwargs = {}
        if args.gnn_run:
            kwargs["gnn_run"] = args.gnn_run
        if args.classifier_run != "default":
            kwargs["classifier_run"] = args.classifier_run
        print(f"Loading GNN grouper (gnn_run={kwargs.get('gnn_run', 'default')})...")
        gnn_grouper = GNNGrouper(**kwargs)
        print(f"  GNN + classifier loaded")
    else:
        print(f"Loading classifier (run={args.classifier_run})...")
        classifier = SymbolClassifier(run=args.classifier_run)
        print(f"  canvas_size={classifier.canvas_size}, use_size_feat={classifier.use_size_feat}, "
              f"classes={len(classifier.label_names)}")

    confusion = Counter()   # (gt, pred) → count
    merge_counter = Counter()  # "a(1)+b(1)" → count
    split_counter = Counter()  # gt_name → count
    count_mismatch = 0
    total_symbols = 0
    correct_symbols = 0
    total_exprs = 0
    total_ood_rejected = 0
    proto_dists_correct = []  # prototype distances for correct predictions
    proto_dists_wrong = []    # prototype distances for wrong predictions
    proto_dists_by_class = {} # class → list of (dist, correct?)
    # Single vs multi-stroke analysis
    single_correct = []  # (dist, conf) for correct single-stroke
    single_wrong = []
    multi_correct = []   # (dist, conf) for correct multi-stroke
    multi_wrong = []

    for path in paths:
        print(f"\n{'='*60}")
        print(f"Evaluating: {path}")
        data = load_strokes_data(path)
        print(f"  {len(data)} expressions")

        for i, item in enumerate(data):
            gt_symbols = item["symbols"]
            canvas_w = item.get("canvas_width", 600)
            canvas_h = item.get("canvas_height", 300)
            stroke_w = item.get("stroke_width", 2)

            # Flatten strokes
            strokes = expr_strokes(item)
            if not strokes:
                continue

            # Run grouper
            if gnn_grouper:
                partitions = gnn_grouper.group_and_classify(
                    strokes,
                    stroke_width=stroke_w,
                    source_size=max(canvas_w, canvas_h),
                )
            else:
                partitions = group_and_classify(
                    strokes, classifier,
                    stroke_width=stroke_w,
                    source_size=max(canvas_w, canvas_h),
                    top_k=1,
                )
            pred_symbols = partitions[0] if partitions else []

            total_exprs += 1
            n_gt = len(gt_symbols)
            n_pred = len(pred_symbols)

            if n_gt != n_pred:
                count_mismatch += 1

            # Build GT stroke sets for cross-symbol analysis
            gt_stroke_sets = []
            stroke_offset = 0
            for gt in gt_symbols:
                n_s = len(gt["strokes"])
                gt_stroke_sets.append(set(range(stroke_offset, stroke_offset + n_s)))
                stroke_offset += n_s

            # Match by stroke overlap
            matched = match_symbols(gt_symbols, pred_symbols, canvas_w, canvas_h)

            # Analyze matches
            expr_errors = []
            for gi, pi in matched:
                gt_name = gt_symbols[gi]["name"]
                pred = pred_symbols[pi]
                pred_name = pred.symbol
                dist = pred.prototype_distance
                total_symbols += 1
                n_strokes = len(pred.stroke_indices)
                conf = pred.confidence
                is_correct = _symbols_match(gt_name, pred_name)
                if is_correct:
                    correct_symbols += 1
                    proto_dists_correct.append(dist)
                    (single_correct if n_strokes == 1 else multi_correct).append((dist, conf))
                else:
                    confusion[(gt_name, pred_name)] += 1
                    expr_errors.append((gt_name, pred_name))
                    proto_dists_wrong.append(dist)
                    (single_wrong if n_strokes == 1 else multi_wrong).append((dist, conf))
                # Per-class tracking
                if gt_name not in proto_dists_by_class:
                    proto_dists_by_class[gt_name] = []
                proto_dists_by_class[gt_name].append((dist, is_correct))

            # Cross-symbol errors: pred symbol contains strokes from multiple GT symbols
            cross_errors = []
            for pi, pred in enumerate(pred_symbols):
                pred_set = set(pred.stroke_indices)
                sources = []
                for gi, gt_set in enumerate(gt_stroke_sets):
                    overlap = pred_set & gt_set
                    if overlap:
                        sources.append((gt_symbols[gi]["name"], len(overlap)))
                if len(sources) > 1:
                    src_str = "+".join(f"{n}({c})" for n, c in sources)
                    cross_errors.append(f"MERGE {src_str} → {pred.symbol}")
                    merge_counter[src_str] += 1

            # Split errors: GT symbol's strokes spread across multiple preds
            split_errors = []
            for gi, gt_set in enumerate(gt_stroke_sets):
                targets = []
                for pi, pred in enumerate(pred_symbols):
                    overlap = gt_set & set(pred.stroke_indices)
                    if overlap:
                        targets.append((pred.symbol, len(overlap)))
                if len(targets) > 1:
                    tgt_str = "+".join(f"{n}({c})" for n, c in targets)
                    split_errors.append(f"SPLIT {gt_symbols[gi]['name']} → {tgt_str}")
                    split_counter[gt_symbols[gi]["name"]] += 1

            # Unmatched GT = missing
            matched_gt = {gi for gi, _ in matched}
            for gi in range(n_gt):
                if gi not in matched_gt:
                    total_symbols += 1
                    confusion[(gt_symbols[gi]["name"], "<missing>")] += 1
                    expr_errors.append((gt_symbols[gi]["name"], "<missing>"))

            # Unmatched pred = extra
            matched_pred = {pi for _, pi in matched}
            for pi in range(n_pred):
                if pi not in matched_pred:
                    confusion[("<extra>", pred_symbols[pi].symbol)] += 1

            has_issues = expr_errors or cross_errors or split_errors or n_gt != n_pred
            if has_issues:
                gt_names = [s["name"] for s in gt_symbols]
                pred_names = [s.symbol for s in pred_symbols]
                print(f"\n  [{i}] GT({n_gt}): {gt_names}")
                print(f"       PR({n_pred}): {pred_names}")
                if expr_errors:
                    for gt_n, pred_n in expr_errors:
                        print(f"       {gt_n} → {pred_n}")
                for ce in cross_errors:
                    print(f"       {ce}")
                for se in split_errors:
                    print(f"       {se}")

            if (i + 1) % 20 == 0:
                print(f"  ... {i+1}/{len(data)}")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Expressions: {total_exprs}")
    print(f"Count mismatches: {count_mismatch}/{total_exprs} "
          f"({count_mismatch/max(1,total_exprs)*100:.1f}%)")
    print(f"Symbol accuracy: {correct_symbols}/{total_symbols} "
          f"({correct_symbols/max(1,total_symbols)*100:.1f}%)")

    if confusion:
        print(f"\nTop confusions:")
        for (gt, pred), cnt in confusion.most_common(30):
            print(f"  {gt:>15s} → {pred:<15s}  {cnt}")

    if merge_counter:
        print(f"\nTop merges (strokes from multiple GT symbols grouped together):")
        for src, cnt in merge_counter.most_common(20):
            print(f"  {src}  {cnt}")

    if split_counter:
        print(f"\nTop splits (one GT symbol's strokes spread across preds):")
        for name, cnt in split_counter.most_common(20):
            print(f"  {name:>15s}  {cnt}")

    # Prototype distance analysis
    print(f"\n{'='*60}")
    print(f"PROTOTYPE DISTANCE ANALYSIS")
    print(f"{'='*60}")
    if proto_dists_correct:
        dc = sorted(proto_dists_correct)
        print(f"Correct predictions ({len(dc)}):")
        print(f"  mean={sum(dc)/len(dc):.1f}  median={dc[len(dc)//2]:.1f}  "
              f"p90={dc[int(len(dc)*0.9)]:.1f}  max={dc[-1]:.1f}")
    if proto_dists_wrong:
        dw = sorted(proto_dists_wrong)
        print(f"Wrong predictions ({len(dw)}):")
        print(f"  mean={sum(dw)/len(dw):.1f}  median={dw[len(dw)//2]:.1f}  "
              f"p90={dw[int(len(dw)*0.9)]:.1f}  max={dw[-1]:.1f}")

    # Single vs multi-stroke analysis
    print(f"\nSingle-stroke (1 stroke per group):")
    if single_correct:
        dc = sorted(d for d, _ in single_correct)
        cc = sorted(c for _, c in single_correct)
        print(f"  Correct ({len(dc)}): dist mean={sum(dc)/len(dc):.1f} median={dc[len(dc)//2]:.1f}  "
              f"conf mean={sum(cc)/len(cc):.3f}")
    if single_wrong:
        dw = sorted(d for d, _ in single_wrong)
        cw = sorted(c for _, c in single_wrong)
        print(f"  Wrong   ({len(dw)}): dist mean={sum(dw)/len(dw):.1f} median={dw[len(dw)//2]:.1f}  "
              f"conf mean={sum(cw)/len(cw):.3f}")
    print(f"\nMulti-stroke (2+ strokes per group):")
    if multi_correct:
        dc = sorted(d for d, _ in multi_correct)
        cc = sorted(c for _, c in multi_correct)
        print(f"  Correct ({len(dc)}): dist mean={sum(dc)/len(dc):.1f} median={dc[len(dc)//2]:.1f}  "
              f"conf mean={sum(cc)/len(cc):.3f}")
    if multi_wrong:
        dw = sorted(d for d, _ in multi_wrong)
        cw = sorted(c for _, c in multi_wrong)
        print(f"  Wrong   ({len(dw)}): dist mean={sum(dw)/len(dw):.1f} median={dw[len(dw)//2]:.1f}  "
              f"conf mean={sum(cw)/len(cw):.3f}")

    # Per-class: classes with worst accuracy and highest distances
    print(f"\nPer-class (worst accuracy, min 3 samples):")
    class_stats = []
    for cls, entries in proto_dists_by_class.items():
        if len(entries) < 3:
            continue
        n_correct = sum(1 for _, c in entries if c)
        acc = n_correct / len(entries)
        avg_dist = sum(d for d, _ in entries) / len(entries)
        class_stats.append((cls, acc, avg_dist, len(entries)))
    class_stats.sort(key=lambda x: x[1])
    for cls, acc, avg_dist, n in class_stats[:20]:
        print(f"  {cls:>15s}  acc={acc*100:5.1f}%  avg_dist={avg_dist:6.1f}  n={n}")


if __name__ == "__main__":
    main()
