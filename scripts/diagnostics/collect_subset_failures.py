"""Run subset model on spatial SUBSETS of training examples and save failures.

For each training example, generates spatial subsets (one per symbol),
runs the subset model on each, and checks if predictions match ground truth.
Saves examples where at least one subset prediction is wrong.

This tests the subset model in its actual inference setting — partial context.

Usage:
    cd math_ocr_v2
    python3.10 tools/collect_subset_failures.py --run dg_all
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from mathnote_ocr.engine.checkpoint import load_checkpoint
from mathnote_ocr.latex_utils.relations import compute_features_from_bbox_list
from mathnote_ocr.tree_parser.evidence import _bbox_edge_dist
from mathnote_ocr.tree_parser.subset_model import SubsetTreeModel
from mathnote_ocr.tree_parser.tree import ROOT


def _spatial_subsets(symbols, max_subset, radius_mult=4.0):
    """Generate one subset per symbol using spatial radius (edge distance)."""
    N = len(symbols)
    bboxes = [s["bbox"] for s in symbols]
    heights = [b[3] for b in bboxes]
    median_h = sorted(heights)[len(heights) // 2]
    radius = radius_mult * median_h

    subsets = []
    for seed in range(N):
        dists = []
        for j in range(N):
            if j == seed:
                continue
            d = _bbox_edge_dist(bboxes[seed], bboxes[j])
            if d <= radius:
                dists.append((j, d))
        dists.sort(key=lambda x: x[1])

        neighbors = [d[0] for d in dists]
        if len(neighbors) < 2:
            all_dists = []
            for j in range(N):
                if j == seed:
                    continue
                all_dists.append((j, _bbox_edge_dist(bboxes[seed], bboxes[j])))
            all_dists.sort(key=lambda x: x[1])
            neighbors = [d[0] for d in all_dists[:2]]
        elif len(neighbors) > max_subset - 1:
            neighbors = neighbors[: max_subset - 1]

        subsets.append(sorted([seed] + neighbors))
    return subsets


def _check_subset(pred_parents, pred_edges, subset_indices, tree, S):
    """Check if subset predictions match ground truth.

    Returns (parent_correct, edge_correct, n_symbols) for this subset.
    parent_correct/edge_correct count symbols with correct predictions.
    """
    n_sub = len(subset_indices)
    g2l = {g: l for l, g in enumerate(subset_indices)}

    parent_ok = 0
    edge_ok = 0

    for i, gi in enumerate(subset_indices):
        t = tree[gi]
        global_parent = t["parent"]

        # Ground truth parent in local coords
        if global_parent == ROOT or global_parent not in g2l:
            gt_parent = S  # ROOT column
        else:
            gt_parent = g2l[global_parent]

        # Predicted parent
        pi = pred_parents[i]

        if pi == gt_parent:
            parent_ok += 1

        # Edge type (only check if parent is correct)
        if pi == gt_parent:
            gt_et = t["edge_type"]
            pred_et = pred_edges[i]
            if pred_et == gt_et:
                edge_ok += 1

    return parent_ok, edge_ok, n_sub


@torch.no_grad()
def run_all_subsets(
    model, symbol_vocab, examples, device, max_subset, batch_size=512, radius_mult=4.0
):
    """Run subset model on spatial subsets of all examples.

    Returns list of (example, n_subsets, n_subset_fails, per_symbol_fails)
    where per_symbol_fails counts total wrong parent predictions across subsets.
    """
    S = max_subset
    results = []

    for ex in examples:
        symbols = ex["symbols"]
        tree = ex["tree"]
        N = len(symbols)

        if N < 2:
            continue

        subsets = _spatial_subsets(symbols, max_subset, radius_mult)

        # Batch all subsets for this example
        n_subsets = len(subsets)
        all_ids = torch.zeros(n_subsets, S, dtype=torch.long, device=device)
        all_geo = []
        all_size = []
        all_pad = torch.ones(n_subsets, S, dtype=torch.bool, device=device)
        n_reals = []

        for b, subset_indices in enumerate(subsets):
            n_sub = len(subset_indices)
            n_reals.append(n_sub)

            for i, gi in enumerate(subset_indices):
                name = symbols[gi]["name"]
                all_ids[b, i] = symbol_vocab.get(name, symbol_vocab.get("<unk>", 1))

            bbox_list = [symbols[gi]["bbox"] for gi in subset_indices]
            geo_buckets, size_feats = compute_features_from_bbox_list(bbox_list, S)
            all_geo.append(geo_buckets)
            all_size.append(size_feats)
            all_pad[b, :n_sub] = False

        all_geo = torch.stack(all_geo).to(device)
        all_size = torch.stack(all_size).to(device)

        # Run in batches if needed
        total_parent_wrong = 0
        n_subset_fails = 0
        per_symbol_wrong = [0] * N  # track which symbols fail most

        for start in range(0, n_subsets, batch_size):
            end = min(start + batch_size, n_subsets)
            out = model(
                all_ids[start:end],
                all_geo[start:end],
                all_pad[start:end],
                all_size[start:end],
            )

            for b_local in range(end - start):
                b = start + b_local
                subset_indices = subsets[b]
                n_sub = n_reals[b]

                parent_scores = out["parent_scores"][b_local]
                edge_scores = out["edge_type_scores"][b_local]

                # Extract predictions
                pred_parents = []
                pred_edges = []
                for i in range(n_sub):
                    pi = parent_scores[i].argmax().item()
                    if pi == S:
                        pred_parents.append(S)
                        et = edge_scores[i, S].argmax().item()
                    else:
                        pred_parents.append(pi)
                        et = edge_scores[i, pi].argmax().item()
                    pred_edges.append(et)

                parent_ok, edge_ok, _ = _check_subset(
                    pred_parents,
                    pred_edges,
                    subset_indices,
                    tree,
                    S,
                )
                if parent_ok < n_sub:
                    n_subset_fails += 1
                    total_parent_wrong += n_sub - parent_ok
                    # Track which global symbols failed
                    g2l = {g: l for l, g in enumerate(subset_indices)}
                    for i, gi in enumerate(subset_indices):
                        t = tree[gi]
                        gp = t["parent"]
                        gt_p = S if (gp == ROOT or gp not in g2l) else g2l[gp]
                        if pred_parents[i] != gt_p:
                            per_symbol_wrong[gi] += 1

        results.append((ex, n_subsets, n_subset_fails, total_parent_wrong, per_symbol_wrong))

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="dg_all")
    parser.add_argument("--out", default="data/shared/tree/failures/subset_fails.jsonl")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--versions", nargs="*", default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Device: {device}")

    ckpt = load_checkpoint("tree_subset", args.run, device=device)
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

    # Find data
    data_root = Path(os.path.dirname(__file__)) / ".." / "data" / "tree"
    if args.versions:
        version_dirs = [data_root / v for v in args.versions]
    else:
        version_dirs = sorted(
            d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("v")
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_examples = 0
    total_with_fails = 0
    total_subsets = 0
    total_subset_fails = 0
    t0 = time.time()

    with open(out_path, "w") as fout:
        for vdir in version_dirs:
            train_path = vdir / "train.jsonl"
            if not train_path.exists():
                continue

            examples = []
            with open(train_path) as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    ex = json.loads(line)
                    if len(ex["symbols"]) >= 2:
                        examples.append(ex)

            results = run_all_subsets(
                model,
                symbol_vocab,
                examples,
                device,
                max_subset,
                args.batch_size,
                args.radius,
            )

            v_total = len(results)
            v_with_fails = 0
            v_subsets = 0
            v_sub_fails = 0

            for ex, n_sub, n_fail, n_wrong, per_sym in results:
                total_examples += 1
                v_subsets += n_sub
                total_subsets += n_sub
                v_sub_fails += n_fail
                total_subset_fails += n_fail

                if n_fail > 0:
                    total_with_fails += 1
                    v_with_fails += 1
                    # Save with failure info
                    record = dict(ex)
                    record["_subset_fails"] = n_fail
                    record["_total_subsets"] = n_sub
                    record["_per_symbol_wrong"] = per_sym
                    fout.write(json.dumps(record) + "\n")

            print(
                f"{vdir.name}: {v_with_fails}/{v_total} examples with fails, "
                f"{v_sub_fails}/{v_subsets} subsets wrong "
                f"({v_sub_fails / max(v_subsets, 1):.1%})"
            )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(
        f"Examples: {total_with_fails}/{total_examples} have at least one failing subset "
        f"({total_with_fails / max(total_examples, 1):.1%})"
    )
    print(
        f"Subsets: {total_subset_fails}/{total_subsets} wrong "
        f"({total_subset_fails / max(total_subsets, 1):.1%})"
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
