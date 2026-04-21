"""Compare tree parsing approaches on generated examples.

Usage:
    python3.10 scripts/compare_gnn_vs_iterative.py --subset-run dg_all_v2 --gnn-run v2 --n 1000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

import torch

from mathnote_ocr.data_gen import sample_all
from mathnote_ocr.latex_utils.glyphs import _extract_glyphs
from mathnote_ocr.tree_parser.gen_data import latex_to_tree_labels
from mathnote_ocr.tree_parser.gnn.model import EvidenceGNN
from mathnote_ocr.tree_parser.subset_model import SubsetTreeModel
from mathnote_ocr.tree_parser.tree import SymbolNode, build_tree, tree_to_latex
from scripts.diagnostics.visualize_predictions import (
    predict_tree_gnn,
    predict_tree_gnn_iterative,
    predict_tree_iterative,
)


def _get_assignments(roots):
    nodes = {}

    def walk(node):
        nodes[node.index] = (node.parent, node.edge_type)
        for children in node.children.values():
            for c in children:
                walk(c)

    for r in roots:
        walk(r)
    return nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-run", default="dg_all_v2")
    parser.add_argument("--gnn-run", default="v2")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--max-n", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device

    # Load models
    weights_dir = Path(__file__).parent.parent / "weights" / "tree_subset" / args.subset_run
    ckpt = torch.load(weights_dir / "checkpoint.pth", map_location=device, weights_only=False)
    symbol_vocab = ckpt["symbol_vocab"]
    cfg = ckpt["config"]
    subset_model = SubsetTreeModel(**cfg).to(device)
    subset_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    subset_model.eval()

    gnn_dir = Path(__file__).parent.parent / "weights" / "tree_gnn" / args.gnn_run
    gnn_ckpt = torch.load(gnn_dir / "checkpoint.pth", map_location=device, weights_only=False)
    gnn_cfg = gnn_ckpt["config"]
    gnn_model = EvidenceGNN(**gnn_cfg).to(device)
    gnn_model.load_state_dict(gnn_ckpt["model_state_dict"])
    gnn_model.eval()
    print(f"Subset: {args.subset_run}, GNN: {args.gnn_run}\n")

    # Approach names
    APPROACHES = ["iter", "gnn", "gnn+iter"]
    match_count = {a: 0 for a in APPROACHES}
    parent_correct = {a: 0 for a in APPROACHES}
    edge_correct = {a: 0 for a in APPROACHES}
    total_symbols = 0
    count = 0
    attempts = 0

    while count < args.n and attempts < args.n * 5:
        attempts += 1
        latex = sample_all()
        glyphs = _extract_glyphs(latex)
        if glyphs is None:
            continue
        n = len(glyphs)
        if n < 2 or n > args.max_n:
            continue
        tree_labels = latex_to_tree_labels(latex, n)
        if tree_labels is None:
            continue

        symbols = [{"name": g["name"], "bbox": g["bbox"]} for g in glyphs]
        gt_tree = [{"parent": p, "edge_type": e, "order": o} for p, e, o in tree_labels]

        gt_nodes = []
        for j, (s, t) in enumerate(zip(symbols, gt_tree)):
            gt_nodes.append(
                SymbolNode(
                    symbol=s["name"],
                    bbox=s["bbox"],
                    index=j,
                    parent=t["parent"],
                    edge_type=t["edge_type"],
                    order=t["order"],
                )
            )
        gt_latex = tree_to_latex(build_tree(gt_nodes))
        gt_assign = {j: (t["parent"], t["edge_type"]) for j, t in enumerate(gt_tree)}

        # Run all three
        results = {}
        try:
            r_iter, _ = predict_tree_iterative(subset_model, symbol_vocab, symbols, device)
            results["iter"] = (tree_to_latex(r_iter), _get_assignments(r_iter))
        except Exception:
            continue
        try:
            r_gnn, _ = predict_tree_gnn(gnn_model, subset_model, symbol_vocab, symbols, device)
            results["gnn"] = (tree_to_latex(r_gnn), _get_assignments(r_gnn))
        except Exception:
            continue
        try:
            r_gi, _ = predict_tree_gnn_iterative(
                gnn_model, subset_model, symbol_vocab, symbols, device
            )
            results["gnn+iter"] = (tree_to_latex(r_gi), _get_assignments(r_gi))
        except Exception:
            continue

        count += 1
        total_symbols += n

        for a in APPROACHES:
            pred_latex, assigns = results[a]
            if pred_latex == gt_latex:
                match_count[a] += 1
            for j in range(n):
                gt_p, gt_e = gt_assign[j]
                if j in assigns:
                    if assigns[j] == (gt_p, gt_e):
                        parent_correct[a] += 1
                        edge_correct[a] += 1
                    elif assigns[j][0] == gt_p:
                        parent_correct[a] += 1

        if count % 100 == 0:
            print(
                f"  {count}/{args.n}  "
                + "  ".join(
                    f"{a}={match_count[a]}/{count} ({match_count[a] / count:.1%})"
                    for a in APPROACHES
                )
            )

    print()
    print(f"{'=' * 70}")
    print(f"Results on {count} examples ({total_symbols} symbols):")
    print(f"{'=' * 70}")
    print()
    print(f"  {'Metric':<20} {'Iterative':>12} {'GNN':>12} {'GNN+Iter':>12}")
    print(f"  {'-' * 56}")
    print(
        f"  {'LaTeX match':<20} " + "".join(f"{match_count[a] / count:>11.1%} " for a in APPROACHES)
    )
    print(
        f"  {'Parent acc':<20} "
        + "".join(f"{parent_correct[a] / total_symbols:>11.1%} " for a in APPROACHES)
    )
    print(
        f"  {'Parent+Edge acc':<20} "
        + "".join(f"{edge_correct[a] / total_symbols:>11.1%} " for a in APPROACHES)
    )
    print()


if __name__ == "__main__":
    main()
