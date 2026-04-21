"""Evaluate tree parser models on handwritten data."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from mathnote_ocr.engine.checkpoint import load_checkpoint
from mathnote_ocr.latex_utils.relations import compute_features_from_bbox_list
from mathnote_ocr.tree_parser.costs import anchor_with_evidence, apply_seq_bonus
from mathnote_ocr.tree_parser.evidence import aggregate_evidence_soft, evidence_to_features
from mathnote_ocr.tree_parser.gnn.model import EvidenceGNN
from mathnote_ocr.tree_parser.subset_model import SubsetTreeModel
from mathnote_ocr.tree_parser.subset_selection import make_spatial_subsets
from mathnote_ocr.tree_parser.tree_builder import (
    build_tree_from_evidence,
    build_tree_from_scores,
    find_seq_conflicts,
)
from mathnote_ocr.tree_parser.tree_latex import tree_to_latex

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(run):
    ckpt = load_checkpoint("tree_subset", run)
    model = SubsetTreeModel(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["symbol_vocab"]


def run_subsets(model, vocab, names, bboxes, subsets, max_subset=8):
    partials = []
    for subset_indices in subsets:
        k = len(subset_indices)
        S = max_subset
        sym_ids = torch.zeros(S, dtype=torch.long)
        for i, gi in enumerate(subset_indices):
            sym_ids[i] = vocab.get(names[gi], vocab.get("<unk>", 1))
        bbox_list = [bboxes[gi] for gi in subset_indices]
        geo_b, size_f = compute_features_from_bbox_list(bbox_list, S)
        pad = torch.ones(S, dtype=torch.bool)
        pad[:k] = False
        with torch.no_grad():
            out = model(
                sym_ids.unsqueeze(0).to(device),
                geo_b.unsqueeze(0).to(device),
                pad.unsqueeze(0).to(device),
                size_f.unsqueeze(0).to(device),
            )
        out_cpu = {k2: v[0].cpu() for k2, v in out.items()}
        partials.append((subset_indices, out_cpu, k))
    return partials


def predict(model, vocab, names, bboxes):
    N = len(names)
    if N == 0:
        return ""
    if N == 1:
        return names[0]
    subsets = make_spatial_subsets(bboxes, 8, 4.0)
    partials = run_subsets(model, vocab, names, bboxes, subsets)
    for _ in range(3):
        evidence = aggregate_evidence_soft(N, partials)
        roots = build_tree_from_evidence(evidence, names, bboxes, cost="propagate")
        targets = find_seq_conflicts(
            evidence, roots, bboxes, seq_threshold=2.0, max_subset_size=min(8, N)
        )
        if not targets:
            break
        partials.extend(run_subsets(model, vocab, names, bboxes, targets))
    evidence = aggregate_evidence_soft(N, partials)
    roots = build_tree_from_evidence(evidence, names, bboxes, cost="propagate")
    return tree_to_latex(roots)


def load_gnn(run):
    ckpt = load_checkpoint("tree_gnn", run)
    model = EvidenceGNN(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict_gnn(model, vocab, gnn_model, names, bboxes):
    N = len(names)
    if N == 0:
        return ""
    if N == 1:
        return names[0]
    subsets = make_spatial_subsets(bboxes, 8, 4.0)
    partials = run_subsets(model, vocab, names, bboxes, subsets)
    for _ in range(3):
        evidence = aggregate_evidence_soft(N, partials)
        roots = build_tree_from_evidence(evidence, names, bboxes, cost="propagate")
        targets = find_seq_conflicts(
            evidence, roots, bboxes, seq_threshold=2.0, max_subset_size=min(8, N)
        )
        if not targets:
            break
        partials.extend(run_subsets(model, vocab, names, bboxes, targets))
    evidence = aggregate_evidence_soft(N, partials)

    # Run GNN on evidence
    _, edge_features = evidence_to_features(evidence)
    sym_ids = torch.tensor(
        [vocab.get(n, vocab.get("<unk>", 1)) for n in names],
        dtype=torch.long,
        device=device,
    )
    _, size_feats = compute_features_from_bbox_list(bboxes, N)
    pad_mask = torch.zeros(N, dtype=torch.bool, device=device)
    with torch.no_grad():
        out = gnn_model(
            sym_ids.unsqueeze(0).to(device),
            size_feats.unsqueeze(0).to(device),
            edge_features.unsqueeze(0).to(device),
            pad_mask.unsqueeze(0).to(device),
        )
    parent_scores = out["parent_scores"][0]
    edge_type_scores = out["edge_type_scores"][0]
    order_preds = torch.zeros(N, N + 1)

    parent_scores = anchor_with_evidence(parent_scores, evidence, N)

    if "seq_scores" in out:
        seq_scores = out["seq_scores"][0]
        parent_scores = apply_seq_bonus(parent_scores, seq_scores, N)

    roots = build_tree_from_scores(parent_scores, edge_type_scores, order_preds, names, bboxes)
    return tree_to_latex(roots)


def main():
    # Parse --gnn flag
    args = sys.argv[1:]
    gnn_run = None
    if "--gnn" in args:
        idx = args.index("--gnn")
        gnn_run = args[idx + 1]
        args = args[:idx] + args[idx + 2 :]
    runs = args if args else ["dg_all_v2", "dg_aug"]

    print("Loading models...")
    models = {}
    for run in runs:
        models[run] = load_model(run)
        print(f"  {run}: loaded")

    gnn_model = None
    if gnn_run:
        gnn_model = load_gnn(gnn_run)
        print(f"  GNN {gnn_run}: loaded")

    hw_data = []
    with open("data/shared/tree_handwritten/default/train.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                hw_data.append(json.loads(line))
            except Exception:
                continue
    print(f"Loaded {len(hw_data)} HW examples\n")

    # Build label list: each run has iterative; if GNN, also add "+gnn" variants
    labels = list(runs)
    if gnn_model:
        labels += [f"{r}+gnn" for r in runs]

    counts = {l: 0 for l in labels}
    diffs = []

    for i, item in enumerate(hw_data):
        names = [s["name"] for s in item["symbols"]]
        bboxes = [s["bbox"] for s in item["symbols"]]
        gt = item["latex"]

        preds = {}
        for run in runs:
            model, vocab = models[run]
            preds[run] = predict(model, vocab, names, bboxes)
            if preds[run] == gt:
                counts[run] += 1
            if gnn_model:
                label = f"{run}+gnn"
                preds[label] = predict_gnn(model, vocab, gnn_model, names, bboxes)
                if preds[label] == gt:
                    counts[label] += 1

        # Log differences between first two runs
        if len(runs) >= 2:
            r0, r1 = runs[0], runs[1]
            m0 = preds[r0] == gt
            m1 = preds[r1] == gt
            if m0 != m1:
                tag = f"{r1}+" if m1 else f"{r0}+"
                diffs.append(f"  [{tag}] {gt}")
                if not m1:
                    diffs.append(f"    {r1}: {preds[r1]}")
                if not m0:
                    diffs.append(f"    {r0}: {preds[r0]}")

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(hw_data)}...")

    print()
    for d in diffs:
        print(d)

    n = len(hw_data)
    print()
    for label in labels:
        print(f"{label}: {counts[label]}/{n} = {counts[label] / n * 100:.1f}%")


if __name__ == "__main__":
    main()
