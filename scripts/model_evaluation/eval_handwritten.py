#!/usr/bin/env python3
"""Evaluate tree parser on handwritten expression data.

Usage:
    python3.10 tools/eval_handwritten.py
    python3.10 tools/eval_handwritten.py --subset-run dg_all_v4 --gnn-run v6
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import re

import torch

from mathnote_ocr.engine.checkpoint import load_checkpoint
from mathnote_ocr.latex_utils.relations import compute_features_from_bbox_list
from mathnote_ocr.tree_parser.subset_model import SubsetTreeModel
from mathnote_ocr.tree_parser.tree_latex import tree_to_latex
from mathnote_ocr.tree_parser.evidence import aggregate_evidence_soft, evidence_to_features
from mathnote_ocr.tree_parser.propagation import propagate_seq
from mathnote_ocr.tree_parser.subset_selection import make_spatial_subsets
from mathnote_ocr.tree_parser.tree_builder import build_tree_from_evidence, build_tree_from_scores, find_seq_conflicts
from mathnote_ocr.tree_parser.gnn.model import EvidenceGNN


def _normalize_latex(s: str) -> str:
    """Normalize LaTeX for comparison — handle equivalent forms."""
    s = s.replace(r"{\prod}", r"{\Pi}")
    s = s.replace(r"{\sum}", r"{\Sigma}")
    s = re.sub(r'(\\(?:sin|cos|tan|log|ln|lim))\{([^{}])\}', r'\1\2', s)
    s = re.sub(r'\{(\\(?:sin|cos|tan|log|ln|lim))\}\{([^{}])\}',
               r'{\1}\2', s)
    return s


def _run_subsets(subset_model, symbol_vocab, symbols, device, subsets, max_subset):
    partial_outputs = []
    for subset_indices in subsets:
        n_sub = len(subset_indices)
        sub_S = max_subset

        sub_ids = torch.zeros(sub_S, dtype=torch.long, device=device)
        for i, gi in enumerate(subset_indices):
            name = symbols[gi]["name"]
            sub_ids[i] = symbol_vocab.get(name, symbol_vocab.get("<unk>", 1))

        bbox_list = [symbols[gi]["bbox"] for gi in subset_indices]
        geo_buckets, size_feats = compute_features_from_bbox_list(bbox_list, sub_S)
        geo_buckets = geo_buckets.to(device)
        size_feats = size_feats.to(device)

        sub_pad = torch.ones(sub_S, dtype=torch.bool, device=device)
        sub_pad[:n_sub] = False

        out = subset_model.forward(
            sub_ids.unsqueeze(0), geo_buckets.unsqueeze(0),
            sub_pad.unsqueeze(0), size_feats.unsqueeze(0),
        )
        out_cpu = {k: v[0].cpu() for k, v in out.items()}
        partial_outputs.append((subset_indices, out_cpu, n_sub))
    return partial_outputs


@torch.no_grad()
def predict_iterative(subset_model, symbol_vocab, symbols, device,
                      max_subset=8, radius_mult=4.0, max_iters=3):
    N = len(symbols)
    bbox_lists = [s["bbox"] for s in symbols]
    names = [s["name"] for s in symbols]

    subsets = make_spatial_subsets(bbox_lists, max_subset, radius_mult)
    all_partial = _run_subsets(subset_model, symbol_vocab, symbols, device, subsets, max_subset)

    for _ in range(max_iters):
        evidence = aggregate_evidence_soft(N, all_partial)
        propagate_seq(evidence)
        roots = build_tree_from_evidence(evidence, names, bbox_lists)
        targets = find_seq_conflicts(
            evidence, roots, bbox_lists,
            seq_threshold=2.0, max_subset_size=min(max_subset, N),
        )
        if not targets:
            break
        new_partial = _run_subsets(subset_model, symbol_vocab, symbols, device, targets, max_subset)
        all_partial.extend(new_partial)

    evidence = aggregate_evidence_soft(N, all_partial)
    propagate_seq(evidence)
    roots = build_tree_from_evidence(evidence, names, bbox_lists)
    return roots, names


@torch.no_grad()
def predict_gnn(gnn_model, subset_model, symbol_vocab, symbols, device,
                max_subset=8, radius_mult=4.0):
    N = len(symbols)
    bbox_lists = [s["bbox"] for s in symbols]
    names = [s["name"] for s in symbols]

    subsets = make_spatial_subsets(bbox_lists, max_subset, radius_mult)
    all_partial = _run_subsets(subset_model, symbol_vocab, symbols, device, subsets, max_subset)
    evidence = aggregate_evidence_soft(N, all_partial)
    _, edge_features = evidence_to_features(evidence)

    unk_id = symbol_vocab.get("<unk>", 1)
    sym_ids = torch.tensor(
        [symbol_vocab.get(names[i], unk_id) for i in range(N)],
        dtype=torch.long, device=device,
    )
    _, size_feats = compute_features_from_bbox_list(bbox_lists, N)
    size_feats = size_feats.to(device)
    edge_features = edge_features.to(device)
    pad_mask = torch.zeros(N, dtype=torch.bool, device=device)

    out = gnn_model(
        sym_ids.unsqueeze(0),
        size_feats.unsqueeze(0),
        edge_features.unsqueeze(0),
        pad_mask.unsqueeze(0),
    )

    parent_scores = out["parent_scores"][0]
    edge_type_scores = out["edge_type_scores"][0]
    order_preds = torch.zeros(N, N + 1)

    roots = build_tree_from_scores(
        parent_scores, edge_type_scores, order_preds,
        names, bbox_lists,
    )
    return roots, names


def _eval_method(name, predict_fn, examples):
    """Run one method, return (exact, normalized, misses)."""
    exact = 0
    normalized = 0
    misses = []
    for i, ex in enumerate(examples):
        gt_latex = ex["latex"]
        roots, _ = predict_fn(ex["symbols"])
        pred_latex = tree_to_latex(roots)

        if pred_latex == gt_latex:
            exact += 1
            normalized += 1
        elif _normalize_latex(pred_latex) == _normalize_latex(gt_latex):
            normalized += 1
            misses.append((i, gt_latex, pred_latex, True))
        else:
            misses.append((i, gt_latex, pred_latex, False))

    return exact, normalized, misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="default", help="Handwritten data run name")
    ap.add_argument("--subset-run", default="dg_all_v4", help="Subset model run")
    ap.add_argument("--gnn-run", default="v6", help="GNN model run")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verbose", action="store_true", help="Show all mismatches")
    args = ap.parse_args()

    # Load data
    data_path = f"data/shared/tree_handwritten/{args.run}/train.jsonl"
    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))
    total = len(examples)
    print(f"Loaded {total} expressions from {data_path}")

    # Load subset model
    ckpt = load_checkpoint("tree_subset", args.subset_run, device=args.device)
    cfg = ckpt["config"]
    symbol_vocab = ckpt["symbol_vocab"]
    subset_model = SubsetTreeModel(**cfg)
    subset_model.load_state_dict(ckpt["model_state_dict"])
    subset_model.eval()
    subset_model.to(args.device)

    # Load GNN model
    gnn_ckpt = load_checkpoint("tree_gnn", args.gnn_run, device=args.device)
    gnn_cfg = gnn_ckpt["config"]
    gnn_model = EvidenceGNN(**gnn_cfg)
    gnn_model.load_state_dict(gnn_ckpt["model_state_dict"])
    gnn_model.eval()
    gnn_model.to(args.device)

    print(f"Models: subset={args.subset_run}, gnn={args.gnn_run}\n")

    # Evaluate both methods
    methods = {
        "Iterative": lambda syms: predict_iterative(
            subset_model, symbol_vocab, syms, args.device),
        "GNN": lambda syms: predict_gnn(
            gnn_model, subset_model, symbol_vocab, syms, args.device),
    }

    results = {}
    for name, predict_fn in methods.items():
        exact, normalized, misses = _eval_method(name, predict_fn, examples)
        results[name] = (exact, normalized, misses)

    # Summary table
    print(f"{'Method':<12} {'Exact':>12} {'Normalized':>14} {'Real Errors':>12}")
    print("─" * 52)
    for name, (exact, normalized, misses) in results.items():
        real_err = total - normalized
        print(f"{name:<12} {exact:>3}/{total} {exact/total:>5.1%}"
              f"   {normalized:>3}/{total} {normalized/total:>5.1%}"
              f"   {real_err:>5}")

    if args.verbose:
        for name, (exact, normalized, misses) in results.items():
            real = [m for m in misses if not m[3]]
            if real:
                print(f"\n{'─'*60}")
                print(f"{name} — structural mismatches ({len(real)}):")
                for idx, gt, pred, _ in real:
                    print(f"  [{idx}] GT:   {gt}")
                    print(f"       Pred: {pred}")
                    print()


if __name__ == "__main__":
    main()
