"""Comprehensive stats comparing HW bboxes vs augmented font bboxes.

Computes per-relationship metrics (sibling gaps, sup/sub offsets, frac gaps,
aspect ratios, size ratios) for both HW and AUG, prints side-by-side stats.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mathnote_ocr.latex_utils.glyphs import _extract_glyphs
from mathnote_ocr.tree_parser.hw_bbox_augment import augment_bboxes
from mathnote_ocr.tree_parser.gen_data import latex_to_tree_labels
from mathnote_ocr.tree_parser.tree import NUM, DEN, SUP, SUB, SQRT_CONTENT, UPPER, LOWER
from scripts.model_evaluation.compare_hw_aug import _match_glyphs_to_hw


def _compute_stats(bboxes, names, tree):
    """Compute geometric stats from bboxes + tree.

    bboxes: list of [x, y, w, h] (top-left format, normalized)
    tree: list of {parent, edge_type, order}
    """
    n = len(bboxes)
    stats = defaultdict(list)

    # Build children map
    children_by_et = defaultdict(lambda: defaultdict(list))
    for i, t in enumerate(tree):
        if t["parent"] >= 0:
            children_by_et[t["parent"]][t["edge_type"]].append(i)

    # Build sibling groups
    sibling_groups = defaultdict(list)
    for i, t in enumerate(tree):
        sibling_groups[(t["parent"], t["edge_type"])].append(i)

    # Helper: subtree extent
    children_of = defaultdict(list)
    for i, t in enumerate(tree):
        if t["parent"] >= 0:
            children_of[t["parent"]].append(i)

    def subtree_top(idx):
        top = bboxes[idx][1]
        for c in children_of[idx]:
            top = min(top, subtree_top(c))
        return top

    def subtree_bot(idx):
        bot = bboxes[idx][1] + bboxes[idx][3]
        for c in children_of[idx]:
            bot = max(bot, subtree_bot(c))
        return bot

    def subtree_right(idx):
        r = bboxes[idx][0] + bboxes[idx][2]
        for c in children_of[idx]:
            r = max(r, subtree_right(c))
        return r

    def subtree_left(idx):
        l = bboxes[idx][0]
        for c in children_of[idx]:
            l = min(l, subtree_left(c))
        return l

    # 1. Aspect ratios (w/h) per symbol
    for i in range(n):
        w, h = bboxes[i][2], bboxes[i][3]
        if h > 1e-6:
            stats["aspect_ratio"].append(w / h)

    # 2. Height ratios (child_h / parent_h) for sup, sub
    for p_i in range(n):
        p_h = bboxes[p_i][3]
        if p_h < 1e-6:
            continue
        for et, label in [(SUP, "sup_h_ratio"), (SUB, "sub_h_ratio")]:
            for ci in children_by_et[p_i].get(et, []):
                stats[label].append(bboxes[ci][3] / p_h)

    # 3. Sup/sub vertical offset (dy / parent_h)
    for p_i in range(n):
        p_cy = bboxes[p_i][1] + bboxes[p_i][3] / 2
        p_h = bboxes[p_i][3]
        if p_h < 1e-6:
            continue
        for et, label in [(SUP, "sup_dy"), (SUB, "sub_dy")]:
            for ci in children_by_et[p_i].get(et, []):
                c_cy = bboxes[ci][1] + bboxes[ci][3] / 2
                stats[label].append((c_cy - p_cy) / p_h)

    # 4. Sup/sub horizontal offset (dx / parent_w) — gap between parent right and child left
    for p_i in range(n):
        p_right = bboxes[p_i][0] + bboxes[p_i][2]
        p_w = bboxes[p_i][2]
        if p_w < 1e-6:
            continue
        for et, label in [(SUP, "sup_dx"), (SUB, "sub_dx")]:
            for ci in children_by_et[p_i].get(et, []):
                c_left = bboxes[ci][0]
                stats[label].append((c_left - p_right) / p_w)

    # 5. Frac: num-to-bar gap, den-to-bar gap (normalized by avg child h)
    for p_i in range(n):
        if names[p_i] != "frac_bar":
            continue
        p_top = bboxes[p_i][1]
        p_bot = bboxes[p_i][1] + bboxes[p_i][3]
        num_kids = children_by_et[p_i].get(NUM, [])
        den_kids = children_by_et[p_i].get(DEN, [])
        if num_kids:
            num_bot = max(bboxes[ci][1] + bboxes[ci][3] for ci in num_kids)
            avg_h = mean(bboxes[ci][3] for ci in num_kids)
            if avg_h > 1e-6:
                stats["frac_num_gap"].append((p_top - num_bot) / avg_h)
        if den_kids:
            den_top = min(bboxes[ci][1] for ci in den_kids)
            avg_h = mean(bboxes[ci][3] for ci in den_kids)
            if avg_h > 1e-6:
                stats["frac_den_gap"].append((den_top - p_bot) / avg_h)

    # 6. Frac bar height ratio (bar_h / avg content h)
    for p_i in range(n):
        if names[p_i] != "frac_bar":
            continue
        all_kids = children_by_et[p_i].get(NUM, []) + children_by_et[p_i].get(DEN, [])
        if all_kids:
            avg_h = mean(bboxes[ci][3] for ci in all_kids)
            if avg_h > 1e-6:
                stats["frac_bar_h_ratio"].append(bboxes[p_i][3] / avg_h)

    # 7. Sibling sequential gaps (gap / avg_visual_h between adjacent siblings)
    for key, indices in sibling_groups.items():
        if len(indices) < 2:
            continue
        sorted_idx = sorted(indices, key=lambda i: bboxes[i][0])
        for j in range(1, len(sorted_idx)):
            prev, curr = sorted_idx[j - 1], sorted_idx[j]
            prev_right = subtree_right(prev)
            curr_left = subtree_left(curr)
            gap = curr_left - prev_right
            prev_vis_h = subtree_bot(prev) - subtree_top(prev)
            curr_vis_h = subtree_bot(curr) - subtree_top(curr)
            avg_h = (prev_vis_h + curr_vis_h) / 2
            if avg_h > 1e-6:
                stats["seq_gap"].append(gap / avg_h)

    # 8. Sibling cy alignment (std of cy within group / avg h)
    for key, indices in sibling_groups.items():
        if len(indices) < 2:
            continue
        cys = [bboxes[i][1] + bboxes[i][3] / 2 for i in indices]
        avg_h = mean(bboxes[i][3] for i in indices)
        if avg_h > 1e-6 and len(cys) > 1:
            stats["sibling_cy_std"].append(stdev(cys) / avg_h)

    # 9. Sqrt: content offset (hook_frac = hook_width / sqrt_width)
    for p_i in range(n):
        content = children_by_et[p_i].get(SQRT_CONTENT, [])
        if not content:
            continue
        sqrt_left = bboxes[p_i][0]
        sqrt_w = bboxes[p_i][2]
        content_left = min(subtree_left(ci) for ci in content)
        if sqrt_w > 1e-6:
            stats["sqrt_hook_frac"].append((content_left - sqrt_left) / sqrt_w)

    # 10. Sqrt height ratio (sqrt_h / content_h)
    for p_i in range(n):
        content = children_by_et[p_i].get(SQRT_CONTENT, [])
        if not content:
            continue
        content_top = min(subtree_top(ci) for ci in content)
        content_bot = max(subtree_bot(ci) for ci in content)
        content_h = content_bot - content_top
        if content_h > 1e-6:
            stats["sqrt_h_ratio"].append(bboxes[p_i][3] / content_h)

    # 11. Size stats: h relative to expression median h
    all_h = [bboxes[i][3] for i in range(n)]
    med_h = median(all_h) if all_h else 1e-6
    if med_h > 1e-6:
        for i in range(n):
            stats["size_h_rel_median"].append(bboxes[i][3] / med_h)
            stats["size_w_rel_median"].append(bboxes[i][2] / med_h)

    # 12. Size stats per edge type: child_h / parent_h
    for p_i in range(n):
        p_h = bboxes[p_i][3]
        if p_h < 1e-6:
            continue
        for ci in children_of[p_i]:
            et = tree[ci]["edge_type"]
            label = {NUM: "num", DEN: "den", SUP: "sup", SUB: "sub",
                     SQRT_CONTENT: "sqrt_content", UPPER: "upper", LOWER: "lower"}.get(et, "other")
            stats[f"child_h_ratio_{label}"].append(bboxes[ci][3] / p_h)

    # 13. Overall expression width/height ratio
    all_x1 = [bboxes[i][0] for i in range(n)]
    all_x2 = [bboxes[i][0] + bboxes[i][2] for i in range(n)]
    all_y1 = [bboxes[i][1] for i in range(n)]
    all_y2 = [bboxes[i][1] + bboxes[i][3] for i in range(n)]
    expr_w = max(all_x2) - min(all_x1)
    expr_h = max(all_y2) - min(all_y1)
    if expr_h > 1e-6:
        stats["expr_aspect"].append(expr_w / expr_h)

    # 14. Size variance within sibling groups
    for key, indices in sibling_groups.items():
        if len(indices) < 2:
            continue
        heights = [bboxes[i][3] for i in indices]
        med = median(heights)
        if med > 1e-6:
            stats["sibling_h_cv"].append(stdev(heights) / med)

    return stats


def fmt(vals):
    if not vals:
        return "n=0"
    m = median(vals)
    s = stdev(vals) if len(vals) > 1 else 0
    return f"med={m:+.3f}  std={s:.3f}  n={len(vals)}"


def main():
    random.seed(42)

    # Load all HW runs with correct trees
    hw_items = []
    for run in ["run_001", "run_002", "run_003"]:
        path = Path(f"data/shared/tree_handwritten/{run}/train.jsonl")
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                try:
                    hw_items.append(json.loads(line))
                except Exception:
                    continue

    print(f"Loaded {len(hw_items)} HW expressions")

    hw_all_stats = defaultdict(list)
    aug_all_stats = defaultdict(list)
    matched = 0

    for item in hw_items:
        latex = item["latex"]
        hw_symbols = item["symbols"]
        hw_tree = item["tree"]
        n = len(hw_symbols)

        # Normalize HW bboxes
        hw_bboxes_raw = [s["bbox"] for s in hw_symbols]
        all_x = [b[0] for b in hw_bboxes_raw]
        all_y = [b[1] for b in hw_bboxes_raw]
        all_x2 = [b[0] + b[2] for b in hw_bboxes_raw]
        all_y2 = [b[1] + b[3] for b in hw_bboxes_raw]
        xmin, ymin = min(all_x), min(all_y)
        xmax, ymax = max(all_x2), max(all_y2)
        ref = max(xmax - xmin, ymax - ymin, 1e-6)
        hw_bboxes = [
            [(b[0] - xmin) / ref, (b[1] - ymin) / ref, b[2] / ref, b[3] / ref]
            for b in hw_bboxes_raw
        ]
        names = [s["name"] for s in hw_symbols]

        # Generate augmented using tree isomorphism matching
        glyphs = _extract_glyphs(latex)
        if glyphs is None or len(glyphs) != n:
            continue

        tree_labels = latex_to_tree_labels(latex, n)
        if tree_labels is None:
            continue

        font_bboxes_hw_order = _match_glyphs_to_hw(
            glyphs, tree_labels, hw_symbols, hw_tree
        )
        if font_bboxes_hw_order is None:
            continue

        match_list = [{"name": names[i], "bbox": font_bboxes_hw_order[i]}
                      for i in range(n)]
        aug = augment_bboxes(match_list, hw_tree)
        if aug is None:
            continue

        aug_bboxes = [a["bbox"] for a in aug]
        aug_names = [a["name"] for a in aug]
        matched += 1

        # Compute stats
        hw_stats = _compute_stats(hw_bboxes, names, hw_tree)
        aug_stats = _compute_stats(aug_bboxes, aug_names, hw_tree)

        for k, v in hw_stats.items():
            hw_all_stats[k].extend(v)
        for k, v in aug_stats.items():
            aug_all_stats[k].extend(v)

    print(f"Matched {matched} / {len(hw_items)} expressions\n")

    # Print comparison
    all_keys = sorted(set(list(hw_all_stats.keys()) + list(aug_all_stats.keys())))
    for k in all_keys:
        print(f"{'─' * 60}")
        print(f"  {k}")
        print(f"    HW:  {fmt(hw_all_stats[k])}")
        print(f"    AUG: {fmt(aug_all_stats[k])}")


if __name__ == "__main__":
    main()
