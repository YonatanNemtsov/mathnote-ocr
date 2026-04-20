"""Compare hard vs soft evidence aggregation on failures and val."""

import json
import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from mathnote_ocr.engine.checkpoint import load_checkpoint
from mathnote_ocr.tree_parser.subset_model import SubsetTreeModel
from mathnote_ocr.tree_parser.tree_latex import tree_to_latex
from mathnote_ocr.tree_parser.tree import ROOT
from scripts.diagnostics.visualize_predictions import (
    _spatial_subsets, _run_subsets, _build_trees,
)
from mathnote_ocr.tree_parser.evidence import (
    aggregate_evidence, aggregate_evidence_soft, propagate_seq,
)
from mathnote_ocr.tree_parser.tree_builder import build_tree_from_evidence


def aggregate_evidence_hard(n_symbols, partial_outputs):
    """Convert raw outputs to hard predictions, then use original aggregation."""
    partial_trees = []
    for subset_indices, out, n_sub in partial_outputs:
        S = out["parent_scores"].shape[0]
        preds = []
        for i in range(n_sub):
            pi = out["parent_scores"][i].argmax().item()
            if pi == S:
                pi = ROOT
            if pi == ROOT:
                et = out["edge_type_scores"][i, S].argmax().item()
                order = out["order_preds"][i, S].item()
            else:
                et = out["edge_type_scores"][i, pi].argmax().item()
                order = out["order_preds"][i, pi].item()
            si = out["seq_scores"][i].argmax().item()
            seq_prev = ROOT if si == S else si
            preds.append((pi, et, order, seq_prev))
        partial_trees.append((subset_indices, preds))
    return aggregate_evidence(n_symbols, partial_trees)


def load_examples(path, n_max=100):
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            n = len(ex["symbols"])
            if n < 2 or n > 20:
                continue
            examples.append(ex)
    if len(examples) > n_max:
        random.seed(42)
        examples = random.sample(examples, n_max)
    return examples


def eval_examples(model, symbol_vocab, examples, device, max_subset, use_soft):
    correct = 0
    for ex in examples:
        symbols = ex["symbols"]
        N = len(symbols)
        names = [s["name"] for s in symbols]
        bbox_lists = [s["bbox"] for s in symbols]

        gt_roots, _ = _build_trees(symbols, ex["tree"])
        gt_latex = tree_to_latex(gt_roots)

        subsets = _spatial_subsets(symbols, max_subset, 4.0)

        raw = _run_subsets(model, symbol_vocab, symbols, device, subsets, max_subset)
        if use_soft:
            evidence = aggregate_evidence_soft(N, raw)
        else:
            evidence = aggregate_evidence_hard(N, raw)

        propagate_seq(evidence)
        roots = build_tree_from_evidence(evidence, names, bbox_lists)
        pred_latex = tree_to_latex(roots)

        if pred_latex == gt_latex:
            correct += 1
    return correct, len(examples)


def main():
    device = torch.device("cpu")
    run = "dg_all"

    ckpt = load_checkpoint("tree_subset", run, device=device)
    cfg = ckpt["config"]
    symbol_vocab = ckpt["symbol_vocab"]
    max_subset = cfg.get("max_symbols", 8)

    model = SubsetTreeModel(
        num_symbols=cfg["num_symbols"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        d_arc=cfg["d_arc"],
        max_symbols=max_subset,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load failures
    fail_path = os.path.join(os.path.dirname(__file__), "..", "data", "tree", "failures", "train.jsonl")
    fail_examples = load_examples(fail_path, n_max=274) if os.path.exists(fail_path) else []

    # Load val (100 per version)
    data_root = os.path.join(os.path.dirname(__file__), "..", "data", "tree")
    val_examples = []
    for v in sorted(os.listdir(data_root)):
        vpath = os.path.join(data_root, v, "val.jsonl")
        if os.path.isfile(vpath):
            val_examples.extend(load_examples(vpath, n_max=100))

    print(f"Failures: {len(fail_examples)}, Val: {len(val_examples)}")
    print(f"\n{'Method':>10s}  {'Failures':>12s}  {'Val':>12s}")
    print("-" * 40)

    for name, soft in [("hard", False), ("soft", True)]:
        if fail_examples:
            fc, ft = eval_examples(model, symbol_vocab, fail_examples, device, max_subset, soft)
        else:
            fc, ft = 0, 0
        vc, vt = eval_examples(model, symbol_vocab, val_examples, device, max_subset, soft)
        print(f"{name:>10s}  {fc:>3d}/{ft:<3d} {fc/max(ft,1):>5.1%}  {vc:>4d}/{vt:<4d} {vc/max(vt,1):>5.1%}")


if __name__ == "__main__":
    main()
