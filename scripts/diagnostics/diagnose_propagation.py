#!/usr/bin/env python3
"""Diagnose: find examples where old symmetric and new forward-only disagree.

Shows the actual predictions so we can see what's going wrong.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import copy
import json
import random

import torch

from mathnote_ocr.latex_utils.relations import compute_features_from_bbox_list
from mathnote_ocr.tree_parser.evidence import aggregate_evidence_soft
from mathnote_ocr.tree_parser.propagation import propagate_old_symmetric, propagate_seq
from mathnote_ocr.tree_parser.subset_model import SubsetTreeModel
from mathnote_ocr.tree_parser.subset_selection import make_spatial_subsets
from mathnote_ocr.tree_parser.tree import ROOT
from mathnote_ocr.tree_parser.tree_builder import build_tree_from_evidence

ET_NAMES = ["NUM", "DEN", "SUP", "SUB", "SQRT", "UPPER", "LOWER"]


def tree_to_tuples(roots, N):
    """Extract (parent, edge_type) per symbol from tree."""
    result = {}

    def walk(nodes):
        for n in nodes:
            result[n.index] = (n.parent, n.edge_type)
            for _, children in n.children.items():
                walk(children)

    walk(roots)
    return result


def run_model(model, vocab, example):
    symbols = example["symbols"]
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
        symbol_ids = torch.tensor(sym_ids, dtype=torch.long).unsqueeze(0)
        pad_mask = torch.zeros(1, sub_S, dtype=torch.bool)
        with torch.no_grad():
            out = model(symbol_ids, geo_buckets.unsqueeze(0), pad_mask, size_feats.unsqueeze(0))
        out_squeezed = {k: v[0].cpu() for k, v in out.items()}
        partial_outputs.append((subset, out_squeezed, sub_S))
    evidence = aggregate_evidence_soft(N, partial_outputs)
    return evidence, names, bboxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="weights/tree_subset/dg_all/checkpoint.pth")
    parser.add_argument("--val", default="data/tree_val_hard.jsonl")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    vocab = ckpt["symbol_vocab"]
    model = SubsetTreeModel(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Checkpoint: {args.checkpoint} (epoch {ckpt['epoch']})")

    examples = []
    with open(args.val) as f:
        for line in f:
            ex = json.loads(line)
            if 3 <= len(ex["symbols"]) <= 20:
                examples.append(ex)
    random.shuffle(examples)
    examples = examples[: args.n]

    old_better = 0
    fwd_better = 0
    same = 0
    both_perfect = 0

    regressions = []  # cases where fwd is worse than old

    for idx, ex in enumerate(examples):
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(examples)}")

        evidence, names, bboxes = run_model(model, vocab, ex)
        tree = ex["tree"]
        N = len(names)

        # Old symmetric
        ev_old = copy.deepcopy(evidence)
        propagate_old_symmetric(ev_old)
        roots_old = build_tree_from_evidence(ev_old, names, bboxes)
        pred_old = tree_to_tuples(roots_old, N)

        # Forward
        ev_fwd = copy.deepcopy(evidence)
        propagate_seq(ev_fwd)
        roots_fwd = build_tree_from_evidence(ev_fwd, names, bboxes)
        pred_fwd = tree_to_tuples(roots_fwd, N)

        # Compare
        old_correct = 0
        fwd_correct = 0
        for i in range(N):
            gt_p, gt_et = tree[i]["parent"], tree[i]["edge_type"]
            op, oet = pred_old.get(i, (ROOT, -1))
            fp, fet = pred_fwd.get(i, (ROOT, -1))
            if op == gt_p and oet == gt_et:
                old_correct += 1
            if fp == gt_p and fet == gt_et:
                fwd_correct += 1

        if old_correct == N and fwd_correct == N:
            both_perfect += 1
        elif old_correct > fwd_correct:
            old_better += 1
            regressions.append((ex, evidence, names, bboxes, pred_old, pred_fwd, tree))
        elif fwd_correct > old_correct:
            fwd_better += 1
        else:
            same += 1

    print(f"\n=== Results ({len(examples)} examples) ===")
    print(f"Both perfect:      {both_perfect}")
    print(f"Same (imperfect):  {same}")
    print(f"Forward better:    {fwd_better}")
    print(f"Old sym better:    {old_better}")

    # Show regressions in detail
    if regressions:
        print(f"\n=== Regressions (old symmetric was better, {len(regressions)} cases) ===")
        for ex, evidence, names, bboxes, pred_old, pred_fwd, tree_gt in regressions[:10]:
            latex = ex.get("latex", "?")
            N = len(names)
            print(f"\nLaTeX: {latex}")
            print(f"Symbols: {names}")
            for i in range(N):
                gt_p, gt_et = tree_gt[i]["parent"], tree_gt[i]["edge_type"]
                op, oet = pred_old.get(i, (ROOT, -1))
                fp, fet = pred_fwd.get(i, (ROOT, -1))
                gt_ok_old = "+" if (op == gt_p and oet == gt_et) else "-"
                gt_ok_fwd = "+" if (fp == gt_p and fet == gt_et) else "-"
                if gt_ok_old != gt_ok_fwd:
                    gt_et_name = ET_NAMES[gt_et] if 0 <= gt_et < len(ET_NAMES) else "ROOT"
                    oet_name = ET_NAMES[oet] if 0 <= oet < len(ET_NAMES) else "ROOT"
                    fet_name = ET_NAMES[fet] if 0 <= fet < len(ET_NAMES) else "ROOT"
                    gt_p_name = names[gt_p] if gt_p != ROOT and 0 <= gt_p < N else "ROOT"
                    op_name = names[op] if op != ROOT and 0 <= op < N else "ROOT"
                    fp_name = names[fp] if fp != ROOT and 0 <= fp < N else "ROOT"
                    print(
                        f"  [{i}] {names[i]}: "
                        f"GT=({gt_p_name},{gt_et_name}) "
                        f"old={gt_ok_old}({op_name},{oet_name}) "
                        f"fwd={gt_ok_fwd}({fp_name},{fet_name})"
                    )


if __name__ == "__main__":
    main()
