"""Diagnostic: evaluate subset model on e^{\frac{...}} patterns.

Checks whether the model correctly predicts:
  1. frac_bar has base as parent with SUP edge
  2. NUM/DEN children of the frac_bar
  3. Any `-` sign that's a SUP sibling of the frac_bar (e.g. e^{-\frac{...}})

For each matching example, samples multiple 8-symbol subsets that include
the base symbol and frac_bar, runs the model, and reports accuracy.

Usage:
    cd math_ocr_v2
    PYTHONPATH=. python3.10 scripts/diagnostics/eval_sup_frac.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np

from mathnote_ocr.latex_utils.relations import compute_features_from_bbox_list
from mathnote_ocr.tree_parser.subset_model import load_subset_model
from mathnote_ocr.tree_parser.tree import NUM, DEN, SUP, SUB, SQRT_CONTENT, UPPER, LOWER, MATCH
from mathnote_ocr.tree_parser.tree import NUM_EDGE_TYPES, ROOT, EDGE_NAMES


# ── Config ──────────────────────────────────────────────────────────

CHECKPOINT = "weights/tree_subset/mixed_v3/checkpoint.pth"
DATA_PATH = "data/tree_mixed_v5_train.jsonl"
MAX_SUBSET = 8
SUBSETS_PER_EXAMPLE = 5   # number of random subsets per example
SEED = 42


# ── Helpers ─────────────────────────────────────────────────────────


def find_sup_frac_examples(data_path: str) -> list[dict]:
    """Load examples containing ^{\\frac or ^{-\\frac patterns."""
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            latex = ex.get("latex", "")
            if r"^{\frac" in latex or r"^{-\frac" in latex:
                examples.append(ex)
    return examples


def find_sup_frac_relationships(ex: dict) -> list[dict]:
    """Find all (base, frac_bar) pairs where frac_bar is SUP child of base.

    Also finds:
      - NUM/DEN children of that frac_bar
      - Any `-` that is a SUP sibling of the frac_bar (same parent, same edge)

    Returns list of dicts describing each relationship group.
    """
    symbols = ex["symbols"]
    tree = ex["tree"]
    n = len(symbols)

    groups = []

    for fb_idx in range(n):
        if symbols[fb_idx]["name"] != "frac_bar":
            continue
        t = tree[fb_idx]
        if t["edge_type"] != SUP:
            continue
        base_idx = t["parent"]
        if base_idx == ROOT or base_idx < 0 or base_idx >= n:
            continue

        # This frac_bar is a SUP child of base_idx
        group = {
            "base_idx": base_idx,
            "base_name": symbols[base_idx]["name"],
            "frac_bar_idx": fb_idx,
            "frac_bar_parent": base_idx,
            "frac_bar_edge": SUP,
            "num_children": [],   # indices of NUM children of frac_bar
            "den_children": [],   # indices of DEN children of frac_bar
            "sup_siblings": [],   # indices of other SUP children of base (e.g. `-`)
        }

        # Find NUM/DEN children of frac_bar
        for ci in range(n):
            ct = tree[ci]
            if ct["parent"] == fb_idx:
                if ct["edge_type"] == NUM:
                    group["num_children"].append(ci)
                elif ct["edge_type"] == DEN:
                    group["den_children"].append(ci)

        # Find SUP siblings of frac_bar (same parent=base, edge=SUP, different index)
        for ci in range(n):
            if ci == fb_idx:
                continue
            ct = tree[ci]
            if ct["parent"] == base_idx and ct["edge_type"] == SUP:
                group["sup_siblings"].append(ci)

        groups.append(group)

    return groups


def sample_subsets(
    ex: dict,
    must_include: list[int],
    max_subset: int,
    n_subsets: int,
) -> list[list[int]]:
    """Sample random subsets of size max_subset that include must_include indices.

    Uses spatial locality: picks remaining slots from nearest neighbors.
    """
    symbols = ex["symbols"]
    n = len(symbols)
    must_set = set(must_include)
    n_must = len(must_set)

    if n_must > max_subset:
        return []  # can't fit all required symbols

    subsets = []

    # Compute centers for spatial sampling
    centers = []
    for s in symbols:
        b = s["bbox"]
        centers.append((b[0] + b[2] / 2, b[1] + b[3] / 2))

    # Centroid of must_include symbols
    mcx = sum(centers[i][0] for i in must_include) / len(must_include)
    mcy = sum(centers[i][1] for i in must_include) / len(must_include)

    # Available pool (not in must_include)
    pool = [i for i in range(n) if i not in must_set]
    # Sort pool by distance to centroid
    pool_dists = [(i, ((centers[i][0] - mcx) ** 2 + (centers[i][1] - mcy) ** 2) ** 0.5)
                  for i in pool]
    pool_dists.sort(key=lambda x: x[1])

    n_extra = max_subset - n_must

    for _ in range(n_subsets):
        if n_extra <= 0 or len(pool_dists) == 0:
            subset = sorted(must_include)
        else:
            # Take some nearby + some random
            n_near = min(n_extra, len(pool_dists))
            # 60% nearest, 40% random from rest
            n_near_take = max(1, int(0.6 * n_near))
            n_random_take = n_near - n_near_take

            near_candidates = [p[0] for p in pool_dists[:n_near_take + 3]]
            random.shuffle(near_candidates)
            chosen_near = near_candidates[:n_near_take]

            far_candidates = [p[0] for p in pool_dists[n_near_take:]]
            if n_random_take > 0 and far_candidates:
                chosen_far = random.sample(far_candidates,
                                           min(n_random_take, len(far_candidates)))
            else:
                chosen_far = []

            extra = chosen_near + chosen_far
            # If we still need more, fill from nearest
            if len(extra) < n_extra:
                remaining = [p[0] for p in pool_dists if p[0] not in set(extra)]
                extra += remaining[:n_extra - len(extra)]

            extra = extra[:n_extra]
            subset = sorted(list(must_set) + extra)

        subsets.append(subset)

    return subsets


def prepare_model_input(
    ex: dict,
    subset: list[int],
    symbol_vocab: dict[str, int],
    max_subset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[int, int]]:
    """Prepare model input tensors for a subset.

    Returns (symbol_ids, geo_buckets, pad_mask, size_feats, g2l_map)
    """
    symbols = ex["symbols"]
    S = max_subset
    k = len(subset)

    g2l = {g: l for l, g in enumerate(subset)}

    # Symbol IDs
    symbol_ids = torch.zeros(S, dtype=torch.long)
    for i, gi in enumerate(subset):
        name = symbols[gi]["name"]
        symbol_ids[i] = symbol_vocab.get(name, symbol_vocab.get("<unk>", 1))

    # Bounding boxes
    bbox_list = [symbols[gi]["bbox"] for gi in subset]
    geo_buckets, size_feats = compute_features_from_bbox_list(bbox_list, S)

    # Pad mask
    pad_mask = torch.ones(S, dtype=torch.bool)
    pad_mask[:k] = False

    return symbol_ids, geo_buckets, pad_mask, size_feats, g2l


# ── Main diagnostic ─────────────────────────────────────────────────


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cpu")

    # Load model
    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = load_subset_model(ckpt, device)
    model.eval()
    symbol_vocab = ckpt["symbol_vocab"]
    cfg = ckpt["config"]
    S = cfg["max_symbols"]
    print(f"  Model version: {cfg.get('model_version', 1)}, max_symbols={S}")

    # Load examples
    print(f"Loading data: {DATA_PATH}")
    examples = find_sup_frac_examples(DATA_PATH)
    print(f"  Found {len(examples)} examples with ^{{\\frac patterns")

    # Counters
    total_subsets = 0
    relationship_counts = defaultdict(int)  # type -> total
    relationship_correct = defaultdict(int)  # type -> correct

    # Parent accuracy by relationship type
    parent_correct = defaultdict(int)
    parent_total = defaultdict(int)

    # Edge type confusion matrix: [true_edge][predicted_edge] -> count
    edge_confusion = defaultdict(lambda: defaultdict(int))
    # Also track parent-conditioned edge accuracy (edge acc given correct parent)
    edge_correct_given_parent = defaultdict(int)
    edge_total_given_parent = defaultdict(int)

    # Detailed error log (first N)
    errors = []
    MAX_ERRORS = 30

    n_examples_processed = 0

    for ex_i, ex in enumerate(examples):
        groups = find_sup_frac_relationships(ex)
        if not groups:
            continue

        n_examples_processed += 1
        symbols = ex["symbols"]
        tree = ex["tree"]

        for group in groups:
            base_idx = group["base_idx"]
            fb_idx = group["frac_bar_idx"]

            # Decide which symbols must be in subset
            # Always include base + frac_bar
            must_include = [base_idx, fb_idx]

            # Add a few NUM/DEN children if available
            for ci in group["num_children"][:2]:
                if ci not in must_include:
                    must_include.append(ci)
            for ci in group["den_children"][:2]:
                if ci not in must_include:
                    must_include.append(ci)

            # Add SUP siblings (e.g. `-` sign)
            for ci in group["sup_siblings"][:1]:
                if ci not in must_include:
                    must_include.append(ci)

            # Cap must_include to max_subset
            must_include = must_include[:S]

            subsets = sample_subsets(ex, must_include, S, SUBSETS_PER_EXAMPLE)
            if not subsets:
                continue

            for subset in subsets:
                total_subsets += 1
                g2l = {g: l for l, g in enumerate(subset)}
                k = len(subset)

                symbol_ids, geo_buckets, pad_mask, size_feats, _ = prepare_model_input(
                    ex, subset, symbol_vocab, S,
                )

                with torch.no_grad():
                    out = model(
                        symbol_ids.unsqueeze(0),
                        geo_buckets.unsqueeze(0),
                        pad_mask.unsqueeze(0),
                        size_feats.unsqueeze(0),
                    )

                parent_scores = out["parent_scores"][0]        # (S, S+1)
                edge_type_scores = out["edge_type_scores"][0]  # (S, S+1, E)

                # ── Check frac_bar: parent should be base with SUP edge ──
                if fb_idx in g2l and base_idx in g2l:
                    fb_local = g2l[fb_idx]
                    base_local = g2l[base_idx]

                    pred_parent = parent_scores[fb_local].argmax().item()
                    true_parent = base_local

                    rel_type = "frac_bar->base (SUP)"
                    parent_total[rel_type] += 1
                    if pred_parent == true_parent:
                        parent_correct[rel_type] += 1

                        # Check edge type
                        pred_edge = edge_type_scores[fb_local, pred_parent].argmax().item()
                        true_edge = SUP
                        edge_confusion[true_edge][pred_edge] += 1
                        edge_total_given_parent[rel_type] += 1
                        if pred_edge == true_edge:
                            edge_correct_given_parent[rel_type] += 1
                        elif len(errors) < MAX_ERRORS:
                            errors.append({
                                "type": "edge_error",
                                "rel": rel_type,
                                "latex": ex["latex"],
                                "true_edge": EDGE_NAMES[true_edge],
                                "pred_edge": EDGE_NAMES[pred_edge],
                                "fb_name": symbols[fb_idx]["name"],
                                "base_name": symbols[base_idx]["name"],
                            })
                    else:
                        # Parent wrong — still record edge confusion
                        # What edge did it predict for the wrong parent?
                        pred_edge = edge_type_scores[fb_local, pred_parent].argmax().item()
                        edge_confusion[SUP][pred_edge] += 1  # true was SUP

                        if len(errors) < MAX_ERRORS:
                            # What did it think the parent was?
                            if pred_parent == S:
                                pred_parent_name = "ROOT"
                            else:
                                pred_parent_global = subset[pred_parent]
                                pred_parent_name = symbols[pred_parent_global]["name"]
                            errors.append({
                                "type": "parent_error",
                                "rel": rel_type,
                                "latex": ex["latex"],
                                "true_parent": symbols[base_idx]["name"],
                                "pred_parent": pred_parent_name,
                                "fb_name": symbols[fb_idx]["name"],
                            })

                # ── Check NUM children of frac_bar ──
                for ci in group["num_children"]:
                    if ci in g2l and fb_idx in g2l:
                        ci_local = g2l[ci]
                        fb_local = g2l[fb_idx]

                        pred_parent = parent_scores[ci_local].argmax().item()
                        true_parent = fb_local

                        rel_type = "child->frac_bar (NUM)"
                        parent_total[rel_type] += 1
                        if pred_parent == true_parent:
                            parent_correct[rel_type] += 1

                            pred_edge = edge_type_scores[ci_local, pred_parent].argmax().item()
                            edge_confusion[NUM][pred_edge] += 1
                            edge_total_given_parent[rel_type] += 1
                            if pred_edge == NUM:
                                edge_correct_given_parent[rel_type] += 1
                        else:
                            pred_edge = edge_type_scores[ci_local, pred_parent].argmax().item()
                            edge_confusion[NUM][pred_edge] += 1

                # ── Check DEN children of frac_bar ──
                for ci in group["den_children"]:
                    if ci in g2l and fb_idx in g2l:
                        ci_local = g2l[ci]
                        fb_local = g2l[fb_idx]

                        pred_parent = parent_scores[ci_local].argmax().item()
                        true_parent = fb_local

                        rel_type = "child->frac_bar (DEN)"
                        parent_total[rel_type] += 1
                        if pred_parent == true_parent:
                            parent_correct[rel_type] += 1

                            pred_edge = edge_type_scores[ci_local, pred_parent].argmax().item()
                            edge_confusion[DEN][pred_edge] += 1
                            edge_total_given_parent[rel_type] += 1
                            if pred_edge == DEN:
                                edge_correct_given_parent[rel_type] += 1
                        else:
                            pred_edge = edge_type_scores[ci_local, pred_parent].argmax().item()
                            edge_confusion[DEN][pred_edge] += 1

                # ── Check SUP siblings of frac_bar (e.g. `-` sign) ──
                for ci in group["sup_siblings"]:
                    if ci in g2l and base_idx in g2l:
                        ci_local = g2l[ci]
                        base_local = g2l[base_idx]

                        pred_parent = parent_scores[ci_local].argmax().item()
                        true_parent = base_local

                        sib_name = symbols[ci]["name"]
                        rel_type = f"sup_sibling({sib_name})->base (SUP)"
                        parent_total[rel_type] += 1
                        if pred_parent == true_parent:
                            parent_correct[rel_type] += 1

                            pred_edge = edge_type_scores[ci_local, pred_parent].argmax().item()
                            edge_confusion[SUP][pred_edge] += 1
                            edge_total_given_parent[rel_type] += 1
                            if pred_edge == SUP:
                                edge_correct_given_parent[rel_type] += 1
                        else:
                            pred_edge = edge_type_scores[ci_local, pred_parent].argmax().item()
                            edge_confusion[SUP][pred_edge] += 1

        if (ex_i + 1) % 500 == 0:
            print(f"  Processed {ex_i + 1}/{len(examples)} examples, "
                  f"{total_subsets} subsets so far...")

    # ── Report ──────────────────────────────────────────────────────

    print("\n" + "=" * 72)
    print("DIAGNOSTIC: Subset model on e^{\\frac{...}} patterns")
    print("=" * 72)
    print(f"Examples with ^{{\\frac:  {len(examples)}")
    print(f"Examples with groups:   {n_examples_processed}")
    print(f"Total subsets tested:   {total_subsets}")
    print()

    # Parent accuracy by relationship type
    print("── Parent Accuracy by Relationship ──")
    print(f"{'Relationship':<40s}  {'Correct':>8s}  {'Total':>8s}  {'Acc':>8s}")
    print("-" * 72)
    for rel_type in sorted(parent_total.keys()):
        correct = parent_correct[rel_type]
        total = parent_total[rel_type]
        acc = correct / total if total > 0 else 0
        print(f"{rel_type:<40s}  {correct:>8d}  {total:>8d}  {acc:>8.1%}")
    print()

    # Edge accuracy (given correct parent)
    print("── Edge Type Accuracy (given correct parent) ──")
    print(f"{'Relationship':<40s}  {'Correct':>8s}  {'Total':>8s}  {'Acc':>8s}")
    print("-" * 72)
    for rel_type in sorted(edge_total_given_parent.keys()):
        correct = edge_correct_given_parent[rel_type]
        total = edge_total_given_parent[rel_type]
        acc = correct / total if total > 0 else 0
        print(f"{rel_type:<40s}  {correct:>8d}  {total:>8d}  {acc:>8.1%}")
    print()

    # Edge confusion matrix
    print("── Edge Type Confusion Matrix ──")
    print("Rows = true edge, Cols = predicted edge")
    print("(includes all predictions, not just correct-parent ones)")
    print()

    # Only show edge types that appear
    active_true = sorted(edge_confusion.keys())
    active_pred = set()
    for t in active_true:
        for p in edge_confusion[t]:
            active_pred.add(p)
    active_pred = sorted(active_pred)

    # Header
    true_pred_label = "True \\ Pred"
    header = f"{true_pred_label:<12s}"
    for p in active_pred:
        header += f"  {EDGE_NAMES[p]:>7s}"
    header += f"  {'Total':>7s}"
    print(header)
    print("-" * len(header))

    for t in active_true:
        row = f"{EDGE_NAMES[t]:<12s}"
        row_total = sum(edge_confusion[t].values())
        for p in active_pred:
            count = edge_confusion[t].get(p, 0)
            if count > 0:
                pct = count / row_total * 100
                row += f"  {count:>4d}({pct:2.0f}%)"
            else:
                row += f"  {'':>7s}"
        row += f"  {row_total:>7d}"
        print(row)
    print()

    # Sample errors
    if errors:
        print(f"── Sample Errors (first {len(errors)}) ──")
        for i, err in enumerate(errors):
            print(f"\n  [{i+1}] {err['type']} — {err['rel']}")
            print(f"      LaTeX: {err['latex'][:80]}")
            if err["type"] == "parent_error":
                print(f"      True parent: {err['true_parent']}")
                print(f"      Pred parent: {err['pred_parent']}")
            elif err["type"] == "edge_error":
                print(f"      True edge: {err['true_edge']}")
                print(f"      Pred edge: {err['pred_edge']}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
