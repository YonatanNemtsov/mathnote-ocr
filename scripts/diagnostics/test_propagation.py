#!/usr/bin/env python3
"""Compare SEQ propagation strategies on vote-only tree accuracy.

Runs the subset model on val examples, aggregates evidence, then builds
trees using different propagation methods. Reports per-symbol accuracy
for each method.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import copy
import torch
import random

from tree_parser.subset_model import SubsetTreeModel
from tree_parser.evidence import aggregate_evidence_soft
from tree_parser.subset_selection import make_spatial_subsets
from tree_parser.propagation import (
    propagate_none,
    propagate_seq,
    propagate_old_symmetric,
    propagate_bidir,
    normalize_scores,
)
from tree_parser.tree_builder import build_tree_from_evidence
from tree_parser.tree import ROOT
from latex_utils.relations import compute_features_from_bbox_list


# ── Evaluation ───────────────────────────────────────────────────────


def run_model_on_example(model, vocab, example, device="cpu"):
    """Run subset model on all spatial subsets of an example, return evidence."""
    symbols = example["symbols"]
    tree = example["tree"]
    N = len(symbols)

    bboxes = [s["bbox"] for s in symbols]
    names = [s["name"] for s in symbols]

    subsets = make_spatial_subsets(bboxes)

    unk_id = vocab.get("<unk>", 1)
    partial_outputs = []

    for subset in subsets:
        sub_S = len(subset)
        bbox_list = [bboxes[i] for i in subset]
        sym_ids = [vocab.get(names[i], unk_id) for i in subset]
        geo_buckets, size_feats = compute_features_from_bbox_list(bbox_list, sub_S)

        symbol_ids = torch.tensor(sym_ids, dtype=torch.long).unsqueeze(0).to(device)
        geo_b = geo_buckets.unsqueeze(0).to(device)
        sf = size_feats.unsqueeze(0).to(device)
        pad_mask = torch.zeros(1, sub_S, dtype=torch.bool).to(device)

        with torch.no_grad():
            out = model(symbol_ids, geo_b, pad_mask, sf)

        # Squeeze batch dim
        out_squeezed = {k: v[0].cpu() for k, v in out.items()}
        partial_outputs.append((subset, out_squeezed, sub_S))

    evidence = aggregate_evidence_soft(N, partial_outputs)
    return evidence, names, bboxes, tree


def evaluate_tree(predicted_nodes, ground_truth, N):
    """Compare predicted tree against ground truth. Returns (correct, total)."""
    def collect_all(roots):
        result = {}
        for n in roots:
            result[n.index] = (n.parent, n.edge_type)
            for _, children in n.children.items():
                result.update(collect_all(children))
        return result

    pred_map = collect_all(predicted_nodes)

    correct = 0
    total = 0
    for i in range(N):
        gt_parent = ground_truth[i]["parent"]
        gt_et = ground_truth[i]["edge_type"]
        pred_parent, pred_et = pred_map.get(i, (ROOT, -1))

        total += 1
        if pred_parent == gt_parent and pred_et == gt_et:
            correct += 1

    return correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="weights/tree_subset/dg_all_v2/checkpoint.pth")
    parser.add_argument("--val", default="data/tree_val_hard.jsonl")
    parser.add_argument("--n", type=int, default=200, help="Number of examples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=1.0, help="SEQ threshold for pooling")
    parser.add_argument("--alpha", type=float, default=0.5, help="Damping factor for pool_damp")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    vocab = ckpt["symbol_vocab"]
    model = SubsetTreeModel(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Checkpoint: {args.checkpoint} (epoch {ckpt['epoch']})")
    print(f"Val data: {args.val}")
    print(f"Examples: {args.n}")
    print()

    # Load examples
    examples = []
    with open(args.val) as f:
        for line in f:
            ex = json.loads(line)
            if 3 <= len(ex["symbols"]) <= 20:
                examples.append(ex)
    random.shuffle(examples)
    examples = examples[: args.n]
    print(f"Loaded {len(examples)} examples")

    strategies = {
        "none": propagate_none,
        "old symmetric i=3": lambda ev: propagate_old_symmetric(ev, n_iters=3),
        "fwd i=3": lambda ev: propagate_seq(ev, n_iters=3),
        "fwd i=3 + norm": lambda ev: (propagate_seq(ev, n_iters=3), normalize_scores(ev)),
        "bidir f3 b1": lambda ev: propagate_bidir(ev, n_fwd=3, n_bwd=1),
    }

    # Run model once per example, then test each strategy
    results = {name: {"correct": 0, "total": 0} for name in strategies}
    all_evidence = []

    print("Running model on examples...")
    for idx, ex in enumerate(examples):
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(examples)}")
        evidence, names, bboxes, tree = run_model_on_example(model, vocab, ex)
        all_evidence.append((evidence, names, bboxes, tree, len(ex["symbols"])))

    print("\nEvaluating strategies...")
    for name, propagate_fn in strategies.items():
        for evidence_orig, names, bboxes, tree, N in all_evidence:
            evidence = copy.deepcopy(evidence_orig)
            propagate_fn(evidence)
            roots = build_tree_from_evidence(evidence, names, bboxes)
            correct, total = evaluate_tree(roots, tree, N)
            results[name]["correct"] += correct
            results[name]["total"] += total

    print(f"\n{'Strategy':<25} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print("-" * 60)
    for name, r in results.items():
        acc = r["correct"] / r["total"] * 100 if r["total"] > 0 else 0
        print(f"{name:<25} {acc:>9.1f}% {r['correct']:>10} {r['total']:>10}")


if __name__ == "__main__":
    main()
