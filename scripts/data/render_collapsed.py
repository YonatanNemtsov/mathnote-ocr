"""Render examples with collapsed subtrees highlighted."""
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mathnote_ocr.latex_utils.collapse import random_collapse, EXPR_NAME

DATA = Path(__file__).parent.parent.parent / "data/shared/tree/mixed_v7b/train.jsonl"

# Load examples with enough symbols
examples = []
with open(DATA) as f:
    for line in f:
        item = json.loads(line)
        if len(item["symbols"]) >= 6:
            examples.append(item)
        if len(examples) >= 5000:
            break

random.seed(42)

# Find examples where collapsing actually happens
rendered = 0
fig, axes = plt.subplots(4, 2, figsize=(16, 20))

for item in examples:
    if rendered >= 4:
        break
    symbols = item["symbols"]
    tree = item["tree"]

    # Try collapsing with higher prob to get more hits for visualization
    new_syms, new_tree = random_collapse(symbols, tree, collapse_prob=0.5)

    # Check if any expression nodes were created
    has_expr = any(s["name"] == EXPR_NAME for s in new_syms)
    if not has_expr:
        continue

    # Plot original
    ax_orig = axes[rendered][0]
    ax_orig.set_title(f"Original ({len(symbols)} symbols)", fontsize=11)
    ax_orig.set_aspect("equal")
    ax_orig.invert_yaxis()

    for i, sym in enumerate(symbols):
        x, y, w, h = sym["bbox"]
        rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                  edgecolor="steelblue", facecolor="lightyellow", alpha=0.7)
        ax_orig.add_patch(rect)
        ax_orig.text(x + w / 2, y + h / 2, sym["name"], fontsize=7,
                     ha="center", va="center", color="black")

    all_x = [s["bbox"][0] for s in symbols] + [s["bbox"][0] + s["bbox"][2] for s in symbols]
    all_y = [s["bbox"][1] for s in symbols] + [s["bbox"][1] + s["bbox"][3] for s in symbols]
    pad = 0.05
    ax_orig.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax_orig.set_ylim(max(all_y) + pad, min(all_y) - pad)

    # Plot collapsed
    ax_col = axes[rendered][1]
    n_expr = sum(1 for s in new_syms if s["name"] == EXPR_NAME)
    ax_col.set_title(f"Collapsed ({len(new_syms)} symbols, {n_expr} expression nodes)", fontsize=11)
    ax_col.set_aspect("equal")
    ax_col.invert_yaxis()

    for i, sym in enumerate(new_syms):
        x, y, w, h = sym["bbox"]
        if sym["name"] == EXPR_NAME:
            rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                      edgecolor="red", facecolor="lightsalmon", alpha=0.4)
            ax_col.add_patch(rect)
            ax_col.text(x + w / 2, y + h / 2, "EXPR", fontsize=9,
                        ha="center", va="center", color="red", fontweight="bold")
        else:
            rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                      edgecolor="steelblue", facecolor="lightyellow", alpha=0.7)
            ax_col.add_patch(rect)
            ax_col.text(x + w / 2, y + h / 2, sym["name"], fontsize=7,
                        ha="center", va="center", color="black")

    all_x2 = [s["bbox"][0] for s in new_syms] + [s["bbox"][0] + s["bbox"][2] for s in new_syms]
    all_y2 = [s["bbox"][1] for s in new_syms] + [s["bbox"][1] + s["bbox"][3] for s in new_syms]
    ax_col.set_xlim(min(all_x2) - pad, max(all_x2) + pad)
    ax_col.set_ylim(max(all_y2) + pad, min(all_y2) - pad)

    rendered += 1

plt.tight_layout()
out = Path(__file__).parent.parent.parent / "data/tmp/collapsed_examples.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
