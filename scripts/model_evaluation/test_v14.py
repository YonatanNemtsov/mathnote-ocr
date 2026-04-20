#!/usr/bin/env python3
"""Test subset model failure rate on v14 generated expressions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import random
from mathnote_ocr.data_gen.latex_sampling import v14
from mathnote_ocr.latex_utils.glyphs import _extract_glyphs
from mathnote_ocr.tree_parser.gen_data import latex_to_tree_labels
from mathnote_ocr.tree_parser.subset_model import SubsetTreeModel
from mathnote_ocr.latex_utils.relations import compute_features_from_bbox_list


def test_checkpoint(ckpt_path: str, n_examples: int = 100, max_attempts: int = 500):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    vocab = ckpt["symbol_vocab"]
    unk_id = vocab.get("<unk>", 1)

    model = SubsetTreeModel(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Checkpoint: {ckpt_path}")
    print(f"  Epoch: {ckpt['epoch']}")
    print(f"  Best val loss: {ckpt['best_val_loss']}")
    if "metrics" in ckpt:
        m = ckpt["metrics"]
        print(f"  Parent acc: {m.get('parent_acc', '?'):.4f}")
        print(f"  Edge acc: {m.get('edge_acc', '?'):.4f}")
        print(f"  Seq acc: {m.get('seq_acc', '?'):.4f}")
    print()

    fails = 0
    total = 0
    attempts = 0

    while total < n_examples and attempts < max_attempts:
        attempts += 1
        latex = v14.sample()
        glyphs = _extract_glyphs(latex)
        if glyphs is None:
            continue
        n = len(glyphs)
        if n < 3 or n > 20:
            continue
        tree_labels = latex_to_tree_labels(latex, n)
        if tree_labels is None:
            continue
        total += 1

        subset = list(range(n))
        if n > 8:
            subset = sorted(random.sample(range(n), 8))
        sub_S = len(subset)

        bbox_list = [glyphs[i]["bbox"] for i in subset]
        sym_ids = [vocab.get(glyphs[i]["name"], unk_id) for i in subset]
        geo_buckets, size_feats = compute_features_from_bbox_list(bbox_list, sub_S)

        symbol_ids = torch.tensor(sym_ids, dtype=torch.long).unsqueeze(0)
        pad_mask = torch.zeros(1, sub_S, dtype=torch.bool)

        with torch.no_grad():
            out = model(
                symbol_ids,
                geo_buckets.unsqueeze(0),
                pad_mask,
                size_feats.unsqueeze(0),
            )

        ps = out["parent_scores"][0]
        ets = out["edge_type_scores"][0]
        ops = out["order_preds"][0]

        wrong = False
        for i in range(sub_S):
            gi = subset[i]
            tp, te, to_ = tree_labels[gi]
            tpl = sub_S if (tp == -1 or tp not in subset) else subset.index(tp)
            pp = ps[i].argmax().item()
            if pp != tpl:
                wrong = True
                break
            if tpl != sub_S:
                pe = ets[i, tpl].argmax().item()
                po = round(ops[i, tpl].item())
                if pe != te or po != to_:
                    wrong = True
                    break
        if wrong:
            fails += 1

    print(f"  Tested: {total}, Failed: {fails}, Rate: {fails/total*100:.1f}%")
    print()
    return fails, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of examples to test")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    base = Path(__file__).parent.parent / "weights" / "tree_subset"
    checkpoints = ["dg_all", "dg_all_v2"]

    for name in checkpoints:
        ckpt_path = base / name / "checkpoint.pth"
        if ckpt_path.exists():
            test_checkpoint(str(ckpt_path), n_examples=args.n)
        else:
            print(f"Checkpoint not found: {ckpt_path}\n")


if __name__ == "__main__":
    main()
