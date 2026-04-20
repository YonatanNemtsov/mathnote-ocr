"""Collect training examples the model fails on into a new dataset.

Batches subset model inference for speed.

Usage:
    cd math_ocr_v2
    PYTHONPATH=. python3.10 tools/collect_failures.py --run dg_all --out data/shared/tree/failures/train.jsonl
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
from mathnote_ocr.tree_parser.subset_model import SubsetTreeModel
from mathnote_ocr.tree_parser.tree import (
    SymbolNode, build_tree, tree_to_latex, ROOT,
)
from mathnote_ocr.tree_parser.evidence import aggregate_evidence_soft, propagate_seq
from mathnote_ocr.tree_parser.tree_builder import build_tree_from_evidence
from scripts.diagnostics.visualize_predictions import (
    _spatial_subsets, _build_trees,
)


@torch.no_grad()
def run_subsets_batched(model, symbol_vocab, symbols_list, subsets_list, device, max_subset, batch_size=128):
    """Run subset model on many (example, subset) pairs in large batches.

    Args:
        symbols_list: list of symbols per example
        subsets_list: list of list-of-subsets per example
        Returns: list of list-of-(subset_indices, preds) per example
    """
    # Flatten all subsets into one big list, track which example they belong to
    flat_jobs = []  # (example_idx, subset_indices, n_sub)
    for ex_idx, (symbols, subsets) in enumerate(zip(symbols_list, subsets_list)):
        for subset_indices in subsets:
            flat_jobs.append((ex_idx, symbols, subset_indices))

    # Process in batches
    all_results = [[] for _ in range(len(symbols_list))]

    for batch_start in range(0, len(flat_jobs), batch_size):
        batch_jobs = flat_jobs[batch_start:batch_start + batch_size]
        B = len(batch_jobs)
        S = max_subset

        all_ids = torch.zeros(B, S, dtype=torch.long, device=device)
        all_geo = torch.zeros(B, 3, S, S, dtype=torch.long, device=device)
        all_size = torch.zeros(B, S, 2, dtype=torch.long, device=device)
        all_pad = torch.ones(B, S, dtype=torch.bool, device=device)
        n_reals = []

        for b, (ex_idx, symbols, subset_indices) in enumerate(batch_jobs):
            n_sub = len(subset_indices)
            n_reals.append(n_sub)

            for i, gi in enumerate(subset_indices):
                name = symbols[gi]["name"]
                all_ids[b, i] = symbol_vocab.get(name, symbol_vocab.get("<unk>", 1))

            bbox_list = [symbols[gi]["bbox"] for gi in subset_indices]
            geo_buckets, size_feats = compute_features_from_bbox_list(bbox_list, S)
            all_geo[b] = geo_buckets
            all_size[b] = size_feats
            all_pad[b, :n_sub] = False

        out = model(all_ids, all_geo, all_pad, all_size)
        parent_scores = out["parent_scores"]      # (B, S, S+1)
        edge_scores = out["edge_type_scores"]      # (B, S, S+1, E)
        order_preds = out["order_preds"]           # (B, S, S+1)
        seq_scores = out["seq_scores"]             # (B, S, S+1)

        for b, (ex_idx, symbols, subset_indices) in enumerate(batch_jobs):
            n_sub = n_reals[b]
            out_cpu = {
                "parent_scores": parent_scores[b].cpu(),
                "edge_type_scores": edge_scores[b].cpu(),
                "order_preds": order_preds[b].cpu(),
                "seq_scores": seq_scores[b].cpu(),
            }
            all_results[ex_idx].append((subset_indices, out_cpu, n_sub))

    return all_results


def check_batch(model, symbol_vocab, examples, device, max_subset):
    """Check a batch of examples. Returns list of (example, gt_latex, pred_latex, match)."""
    symbols_list = [ex["symbols"] for ex in examples]
    subsets_list = [_spatial_subsets(syms, max_subset, 3.0) for syms in symbols_list]

    partial_results = run_subsets_batched(
        model, symbol_vocab, symbols_list, subsets_list, device, max_subset
    )

    results = []
    for ex, symbols, partial in zip(examples, symbols_list, partial_results):
        N = len(symbols)
        names = [s["name"] for s in symbols]
        bbox_lists = [s["bbox"] for s in symbols]

        gt_roots, _ = _build_trees(symbols, ex["tree"])
        gt_latex = tree_to_latex(gt_roots)

        evidence = aggregate_evidence_soft(N, partial)
        propagate_seq(evidence)
        roots = build_tree_from_evidence(evidence, names, bbox_lists)
        pred_latex = tree_to_latex(roots)

        results.append((ex, gt_latex, pred_latex, pred_latex == gt_latex))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="dg_all")
    parser.add_argument("--out", default="data/shared/tree/failures/train.jsonl")
    parser.add_argument("--max-symbols", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Examples to process at once")
    parser.add_argument("--versions", nargs="*", default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Device: {device}")

    # Load model
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

    # Find data dirs
    data_root = Path(os.path.dirname(__file__)) / ".." / "data" / "tree"
    if args.versions:
        version_dirs = [data_root / v for v in args.versions]
    else:
        version_dirs = sorted(data_root.iterdir())
        version_dirs = [d for d in version_dirs if d.is_dir() and d.name.startswith("v")]

    # Output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    failures = 0
    skipped = 0
    t0 = time.time()

    with open(out_path, "w") as fout:
        for vdir in version_dirs:
            train_path = vdir / "train.jsonl"
            if not train_path.exists():
                continue

            v_total = 0
            v_fail = 0

            # Load all examples for this version
            batch = []
            with open(train_path) as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    ex = json.loads(line)
                    n = len(ex["symbols"])
                    if n < 2 or n > args.max_symbols:
                        skipped += 1
                        continue
                    batch.append(ex)

                    if len(batch) >= args.batch_size:
                        results = check_batch(model, symbol_vocab, batch, device, max_subset)
                        for ex_r, gt, pred, match in results:
                            total += 1
                            v_total += 1
                            if not match:
                                fout.write(json.dumps(ex_r) + "\n")
                                failures += 1
                                v_fail += 1
                        batch = []

                        if total % 1000 == 0:
                            elapsed = time.time() - t0
                            rate = total / elapsed
                            print(f"  {total:>6d} processed | {failures} failures "
                                  f"({failures/total:.1%}) | {rate:.0f} ex/s")

            # Remaining
            if batch:
                results = check_batch(model, symbol_vocab, batch, device, max_subset)
                for ex_r, gt, pred, match in results:
                    total += 1
                    v_total += 1
                    if not match:
                        fout.write(json.dumps(ex_r) + "\n")
                        failures += 1
                        v_fail += 1

            print(f"{vdir.name}: {v_fail}/{v_total} failures "
                  f"({v_fail/max(v_total,1):.1%})")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Total: {total} processed, {skipped} skipped, "
          f"{failures} failures ({failures/max(total,1):.1%})")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
