"""Render original vs collapsed expressions side by side."""

import json
import random
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mathnote_ocr.latex_utils.collapse import EXPR_NAME, random_collapse

DATA = Path(__file__).parent.parent.parent / "data/runs/tree_subset/mixed_v7b/train.jsonl"
N_EXAMPLES = 50


def draw_expr(ax, symbols, tree, title=""):
    """Draw symbols as labeled bboxes, colored by parent."""
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    colors = plt.cm.tab10.colors
    for i, (sym, t) in enumerate(zip(symbols, tree)):
        x, y, w, h = sym["bbox"]
        color = (0.9, 0.6, 0.6) if sym["name"] == EXPR_NAME else colors[t["parent"] % 10]
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1, edgecolor="black", facecolor=color, alpha=0.5
        )
        ax.add_patch(rect)
        label = f"{sym['name']}\no={t['order']}"
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=6)

    # Auto-scale
    if symbols:
        all_x = [s["bbox"][0] for s in symbols]
        all_y = [s["bbox"][1] for s in symbols]
        all_r = [s["bbox"][0] + s["bbox"][2] for s in symbols]
        all_b = [s["bbox"][1] + s["bbox"][3] for s in symbols]
        margin = 0.02
        ax.set_xlim(min(all_x) - margin, max(all_r) + margin)
        ax.set_ylim(max(all_b) + margin, min(all_y) - margin)


def main():
    examples = []
    with open(DATA) as f:
        for line in f:
            ex = json.loads(line.strip())
            if 6 <= len(ex["symbols"]) <= 20:
                examples.append(ex)
            if len(examples) >= 500:
                break

    random.shuffle(examples)

    fig, axes = plt.subplots(N_EXAMPLES, 2, figsize=(14, N_EXAMPLES * 2.5))
    shown = 0

    for ex in examples:
        if shown >= N_EXAMPLES:
            break

        symbols = ex["symbols"]
        tree = ex["tree"]

        new_syms, new_tree = random_collapse(symbols, tree, collapse_prob=0.04)
        has_expr = any(s["name"] == EXPR_NAME for s in new_syms)
        if not has_expr:
            continue

        latex = ex.get("latex", "")
        draw_expr(axes[shown, 0], symbols, tree, f"Original ({len(symbols)} syms): {latex[:50]}")
        draw_expr(axes[shown, 1], new_syms, new_tree, f"Collapsed ({len(new_syms)} syms)")
        shown += 1

    plt.tight_layout()
    out = Path(__file__).parent.parent.parent / "data/tmp/collapse_compare.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
