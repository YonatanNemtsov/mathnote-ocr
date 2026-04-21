"""Compare augmented bbox distributions against handwritten data.

Usage:
    PYTHONPATH=. python3.10 scripts/diagnostics/compare_aug_hw.py
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mathnote_ocr.tree_parser.hw_bbox_augment import augment_bboxes
from mathnote_ocr.tree_parser.tree import DEN, NUM, SQRT_CONTENT, SUB, SUP

ET_NAMES = {-1: "ROOT", NUM: "NUM", DEN: "DEN", SUP: "SUP", SUB: "SUB", SQRT_CONTENT: "SQRT"}


def measure_stats(items, augment=False):
    sup_dy, sub_dy = [], []
    gap_by_et = defaultdict(list)
    h_ratio_by_et = defaultdict(list)
    num_off, den_off = [], []
    overlap_count, total_pairs = 0, 0

    for item in items:
        syms = item["symbols"]
        tree = item["tree"]
        n = len(syms)

        if augment:
            aug = augment_bboxes(syms, tree)
            if aug is None:
                continue
            syms = aug

        if not syms:
            continue
        heights = [s["bbox"][3] for s in syms]
        med_h = sorted(heights)[n // 2]
        if med_h < 1e-6:
            med_h = 1

        # SUP/SUB dy
        for i in range(n):
            et = tree[i]["edge_type"]
            p = tree[i]["parent"]
            if p < 0:
                continue
            p_h = syms[p]["bbox"][3]
            if p_h < 1e-6:
                continue
            p_cy = syms[p]["bbox"][1] + p_h / 2
            c_cy = syms[i]["bbox"][1] + syms[i]["bbox"][3] / 2
            dy = (c_cy - p_cy) / p_h
            if et == SUP:
                sup_dy.append(dy)
            elif et == SUB:
                sub_dy.append(dy)
            if et in (SUP, SUB, NUM, DEN, SQRT_CONTENT) and syms[p]["bbox"][3] > 1e-6:
                h_ratio_by_et[et].append(syms[i]["bbox"][3] / syms[p]["bbox"][3])

        # NUM/DEN centering
        for i in range(n):
            if syms[i]["name"] != "frac_bar":
                continue
            bw = syms[i]["bbox"][2]
            bcx = syms[i]["bbox"][0] + bw / 2
            if bw < 1e-6:
                continue
            nk = [j for j in range(n) if tree[j]["parent"] == i and tree[j]["edge_type"] == NUM]
            dk = [j for j in range(n) if tree[j]["parent"] == i and tree[j]["edge_type"] == DEN]
            if nk:
                xs = [syms[j]["bbox"][0] for j in nk]
                x2 = [syms[j]["bbox"][0] + syms[j]["bbox"][2] for j in nk]
                num_off.append(((min(xs) + max(x2)) / 2 - bcx) / bw)
            if dk:
                xs = [syms[j]["bbox"][0] for j in dk]
                x2 = [syms[j]["bbox"][0] + syms[j]["bbox"][2] for j in dk]
                den_off.append(((min(xs) + max(x2)) / 2 - bcx) / bw)

        # Sibling gaps
        groups = defaultdict(list)
        for i in range(n):
            groups[(tree[i]["parent"], tree[i]["edge_type"])].append(i)
        for (p, et), indices in groups.items():
            if len(indices) < 2:
                continue
            sorted_idx = sorted(indices, key=lambda i: syms[i]["bbox"][0])
            for j in range(1, len(sorted_idx)):
                prev_r = syms[sorted_idx[j - 1]]["bbox"][0] + syms[sorted_idx[j - 1]]["bbox"][2]
                curr_l = syms[sorted_idx[j]]["bbox"][0]
                gap_by_et[et].append((curr_l - prev_r) / med_h)

        # Overlaps (non parent-child, non frac_bar)
        for i in range(n):
            for j in range(i + 1, n):
                if tree[i]["parent"] == j or tree[j]["parent"] == i:
                    continue
                if syms[i]["name"] == "frac_bar" or syms[j]["name"] == "frac_bar":
                    continue
                total_pairs += 1
                ax, ay, aw, ah = syms[i]["bbox"]
                bx, by, bw, bh = syms[j]["bbox"]
                ox = max(0, min(ax + aw, bx + bw) - max(ax, bx))
                oy = max(0, min(ay + ah, by + bh) - max(ay, by))
                if min(aw * ah, bw * bh) > 1e-9 and (ox * oy) / min(aw * ah, bw * bh) > 0.3:
                    overlap_count += 1

    return {
        "sup_dy": sup_dy,
        "sub_dy": sub_dy,
        "gap_by_et": dict(gap_by_et),
        "h_ratio_by_et": dict(h_ratio_by_et),
        "num_off": num_off,
        "den_off": den_off,
        "overlap_count": overlap_count,
        "total_pairs": total_pairs,
    }


def main():
    random.seed(42)

    hw = []
    with open("data/shared/tree_handwritten/run_001/train.jsonl") as f:
        for line in f:
            hw.append(json.loads(line))

    font = []
    with open("data/shared/tree/mixed_dg/train.jsonl") as f:
        for line in f:
            font.append(json.loads(line))
            if len(font) >= 2000:
                break

    print("Measuring HW...")
    hw_s = measure_stats(hw)
    print("Measuring augmented...")
    aug_s = measure_stats(font, augment=True)

    # Print summary
    print("\n=== SUP/SUB dy ===")
    print(
        f"  SUP  HW: {np.mean(hw_s['sup_dy']):.3f}±{np.std(hw_s['sup_dy']):.3f}  Aug: {np.mean(aug_s['sup_dy']):.3f}±{np.std(aug_s['sup_dy']):.3f}  (extreme <-3: {sum(1 for d in aug_s['sup_dy'] if d < -3)})"
    )
    print(
        f"  SUB  HW: {np.mean(hw_s['sub_dy']):.3f}±{np.std(hw_s['sub_dy']):.3f}  Aug: {np.mean(aug_s['sub_dy']):.3f}±{np.std(aug_s['sub_dy']):.3f}"
    )

    print("\n=== Sibling gaps (gap/median_h) ===")
    for et in [-1, NUM, DEN, SUP, SUB]:
        hg = hw_s["gap_by_et"].get(et, [])
        ag = aug_s["gap_by_et"].get(et, [])
        neg = sum(1 for g in ag if g < -0.1)
        if hg or ag:
            print(
                f"  {ET_NAMES[et]:5s}  HW: {np.mean(hg):.3f}±{np.std(hg):.3f}  Aug: {np.mean(ag):.3f}±{np.std(ag):.3f}  neg: {neg}/{len(ag)}"
            )

    print("\n=== NUM/DEN centering ===")
    print(f"  NUM  HW: {np.mean(hw_s['num_off']):.4f}  Aug: {np.mean(aug_s['num_off']):.4f}")
    print(f"  DEN  HW: {np.mean(hw_s['den_off']):.4f}  Aug: {np.mean(aug_s['den_off']):.4f}")

    print("\n=== Height ratio (child/parent) ===")
    for et in [SUP, SUB, NUM, DEN]:
        hr = hw_s["h_ratio_by_et"].get(et, [])
        ar = aug_s["h_ratio_by_et"].get(et, [])
        if hr or ar:
            print(
                f"  {ET_NAMES[et]:5s}  HW: {np.mean(hr):.3f}±{np.std(hr):.3f}  Aug: {np.mean(ar):.3f}±{np.std(ar):.3f}"
            )

    print("\n=== Overlaps (>30% area, non parent-child) ===")
    print(f"  HW:  {hw_s['overlap_count']}/{hw_s['total_pairs']}")
    print(f"  Aug: {aug_s['overlap_count']}/{aug_s['total_pairs']}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].hist(
        hw_s["sup_dy"], bins=40, alpha=0.6, color="blue", density=True, label="HW", range=(-3, 1)
    )
    axes[0, 0].hist(
        aug_s["sup_dy"], bins=40, alpha=0.6, color="red", density=True, label="Aug", range=(-3, 1)
    )
    axes[0, 0].set_title("SUP dy/parent_h")
    axes[0, 0].legend()

    axes[0, 1].hist(
        hw_s["sub_dy"], bins=30, alpha=0.6, color="blue", density=True, label="HW", range=(-0.5, 2)
    )
    axes[0, 1].hist(
        aug_s["sub_dy"], bins=30, alpha=0.6, color="red", density=True, label="Aug", range=(-0.5, 2)
    )
    axes[0, 1].set_title("SUB dy/parent_h")
    axes[0, 1].legend()

    for idx, et in enumerate([NUM, DEN]):
        hg = hw_s["gap_by_et"].get(et, [])
        ag = aug_s["gap_by_et"].get(et, [])
        axes[0, 2].hist(
            hg,
            bins=30,
            alpha=0.4,
            color="blue",
            density=True,
            label=f"HW {ET_NAMES[et]}",
            range=(-0.5, 2),
        )
        axes[0, 2].hist(
            ag,
            bins=30,
            alpha=0.4,
            color="red" if et == NUM else "orange",
            density=True,
            label=f"Aug {ET_NAMES[et]}",
            range=(-0.5, 2),
        )
    axes[0, 2].set_title("NUM/DEN gaps")
    axes[0, 2].legend()

    axes[1, 0].hist(hw_s["num_off"], bins=30, alpha=0.6, color="blue", density=True, label="HW")
    axes[1, 0].hist(aug_s["num_off"], bins=30, alpha=0.6, color="red", density=True, label="Aug")
    axes[1, 0].axvline(0, color="black", linestyle="--")
    axes[1, 0].set_title("NUM centering offset")
    axes[1, 0].legend()

    axes[1, 1].hist(hw_s["den_off"], bins=30, alpha=0.6, color="blue", density=True, label="HW")
    axes[1, 1].hist(aug_s["den_off"], bins=30, alpha=0.6, color="red", density=True, label="Aug")
    axes[1, 1].axvline(0, color="black", linestyle="--")
    axes[1, 1].set_title("DEN centering offset")
    axes[1, 1].legend()

    rg = hw_s["gap_by_et"].get(-1, [])
    ag = aug_s["gap_by_et"].get(-1, [])
    axes[1, 2].hist(rg, bins=30, alpha=0.6, color="blue", density=True, label="HW", range=(-0.5, 3))
    axes[1, 2].hist(ag, bins=30, alpha=0.6, color="red", density=True, label="Aug", range=(-0.5, 3))
    axes[1, 2].set_title("ROOT sibling gaps")
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig("data/tmp/aug_hw_comparison.png", dpi=150)
    print("\nPlot saved to data/tmp/aug_hw_comparison.png")


if __name__ == "__main__":
    main()
