"""Dataset for GNN training.

Supports loading from:
  1. Pre-computed .pt files (from gen_data.py) — fast, no subset model needed
  2. JSONL files — computes evidence on-the-fly (slower)
  3. Generators — samples + computes evidence on-the-fly (slowest)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from mathnote_ocr.latex_utils.glyphs import _extract_glyphs
from mathnote_ocr.latex_utils.relations import compute_features_from_bbox_list
from mathnote_ocr.tree_parser.evidence import aggregate_evidence_soft, evidence_to_features
from mathnote_ocr.tree_parser.gen_data import latex_to_tree_labels
from mathnote_ocr.tree_parser.subset_model import SubsetTreeModel
from mathnote_ocr.tree_parser.subset_selection import make_spatial_subsets
from mathnote_ocr.tree_parser.tree import ROOT

# ── Evidence computation ─────────────────────────────────────────────


@torch.no_grad()
def _compute_evidence_for_example(
    symbols: list[dict],
    tree: list[dict],
    subset_model: SubsetTreeModel,
    symbol_vocab: dict[str, int],
    max_subset: int,
    device: torch.device,
    augment: bool = False,
) -> dict | None:
    """Run subset model → aggregate evidence → return cached tensors."""
    # Optionally apply expression collapsing augmentation
    if augment and len(symbols) > 3:
        from mathnote_ocr.latex_utils.collapse import random_collapse

        symbols, tree = random_collapse(symbols, tree)
        if len(symbols) < 2:
            return None

    N = len(symbols)
    bboxes = [s["bbox"] for s in symbols]
    names = [s["name"] for s in symbols]
    unk_id = symbol_vocab.get("<unk>", 1)

    subsets = make_spatial_subsets(bboxes, max_subset=max_subset)
    partial_outputs = []
    for subset_indices in subsets:
        n_sub = len(subset_indices)
        S = max_subset

        sub_ids = torch.zeros(S, dtype=torch.long, device=device)
        for i, gi in enumerate(subset_indices):
            sub_ids[i] = symbol_vocab.get(names[gi], unk_id)

        bbox_list = [bboxes[gi] for gi in subset_indices]
        geo, size_feats = compute_features_from_bbox_list(bbox_list, S)
        geo = geo.to(device)
        size_feats = size_feats.to(device)

        pad_mask = torch.ones(S, dtype=torch.bool, device=device)
        pad_mask[:n_sub] = False

        out = subset_model.forward(
            sub_ids.unsqueeze(0),
            geo.unsqueeze(0),
            pad_mask.unsqueeze(0),
            size_feats.unsqueeze(0),
        )
        out_cpu = {k: v[0].cpu() for k, v in out.items()}
        partial_outputs.append((subset_indices, out_cpu, n_sub))

    evidence = aggregate_evidence_soft(N, partial_outputs)
    _, edge_features = evidence_to_features(evidence)

    sym_ids = torch.tensor(
        [symbol_vocab.get(names[i], unk_id) for i in range(N)],
        dtype=torch.long,
    )
    _, full_size = compute_features_from_bbox_list(bboxes, N)

    gt_parent = torch.full((N,), -100, dtype=torch.long)
    gt_edge_type = torch.full((N,), -100, dtype=torch.long)
    gt_seq = torch.full((N,), N, dtype=torch.long)  # default: NONE (=N)
    for i in range(N):
        t = tree[i]
        p = t["parent"]
        gt_parent[i] = N if p == ROOT else p
        et = t["edge_type"]
        gt_edge_type[i] = et if et >= 0 else -100

    # Compute gt_seq: previous sibling = same parent+edge_type, order-1
    # Group symbols by (parent, edge_type) → sorted by order
    from collections import defaultdict

    sibling_groups = defaultdict(list)
    for i in range(N):
        t = tree[i]
        key = (t["parent"], t["edge_type"])
        sibling_groups[key].append((t["order"], i))
    for key, group in sibling_groups.items():
        group.sort()  # sort by order
        for rank, (order, idx) in enumerate(group):
            if rank > 0:
                gt_seq[idx] = group[rank - 1][1]  # prev sibling's index
            # rank == 0 → gt_seq stays N (NONE)

    return {
        "symbol_ids": sym_ids,
        "size_feats": full_size,
        "edge_features": edge_features,
        "gt_parent": gt_parent,
        "gt_edge_type": gt_edge_type,
        "gt_seq": gt_seq,
        "N": N,
    }


# ── Dataset ──────────────────────────────────────────────────────────


class GNNDataset(Dataset):
    """GNN training dataset with variable-N examples.

    Use collate_fn() to pad within each batch.
    """

    def __init__(self, examples: list[dict]) -> None:
        self.examples = examples

    @classmethod
    def from_file(cls, pt_path: Path) -> GNNDataset:
        """Load pre-computed examples from a .pt file (from gen_data.py)."""
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        examples = data["examples"]
        print(f"  Loaded {len(examples)} examples from {pt_path}")
        return cls(examples)

    @classmethod
    def from_jsonl(
        cls,
        jsonl_path: Path,
        subset_model: SubsetTreeModel,
        symbol_vocab: dict[str, int],
        max_subset: int,
        device: torch.device,
        max_n: int = 30,
        max_examples: int | None = None,
    ) -> GNNDataset:
        """Load expressions from a JSONL file."""
        raw = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                n = len(ex["symbols"])
                if n >= 2 and n <= max_n:
                    raw.append(ex)
                    if max_examples and len(raw) >= max_examples:
                        break

        print(f"  Computing evidence for {len(raw)} examples from {jsonl_path}...")
        examples = _process_raw(raw, subset_model, symbol_vocab, max_subset, device)
        print(f"  Cached {len(examples)} examples")
        return cls(examples)

    @classmethod
    def from_generators(
        cls,
        per_version: int,
        subset_model: SubsetTreeModel,
        symbol_vocab: dict[str, int],
        max_subset: int,
        device: torch.device,
        max_n: int = 30,
        max_attempts_mult: int = 5,
    ) -> GNNDataset:
        """Sample expressions from each of the 14 data_gen versions."""
        from mathnote_ocr.data_gen.latex_sampling import (
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            v8,
            v9,
            v10,
            v11,
            v12,
            v13,
            v14,
            v15,
        )

        versions = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15]
        raw = []

        for ver in versions:
            name = ver.__name__.split(".")[-1]
            count = 0
            attempts = 0
            while count < per_version and attempts < per_version * max_attempts_mult:
                attempts += 1
                latex = ver.sample()
                glyphs = _extract_glyphs(latex)
                if glyphs is None:
                    continue
                n = len(glyphs)
                if n < 2 or n > max_n:
                    continue
                tree_labels = latex_to_tree_labels(latex, n)
                if tree_labels is None:
                    continue

                raw.append(
                    {
                        "symbols": [{"name": g["name"], "bbox": g["bbox"]} for g in glyphs],
                        "tree": [
                            {"parent": p, "edge_type": e, "order": o} for p, e, o in tree_labels
                        ],
                    }
                )
                count += 1

            print(f"    {name}: {count}/{per_version} ({attempts} attempts)")

        print(f"  Computing evidence for {len(raw)} examples...")
        examples = _process_raw(raw, subset_model, symbol_vocab, max_subset, device)
        print(f"  Cached {len(examples)} examples")
        return cls(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def _process_raw(
    raw: list[dict],
    subset_model: SubsetTreeModel,
    symbol_vocab: dict[str, int],
    max_subset: int,
    device: torch.device,
) -> list[dict]:
    """Convert raw examples to cached evidence tensors."""
    subset_model.eval()
    examples = []
    for idx, ex in enumerate(raw):
        result = _compute_evidence_for_example(
            ex["symbols"],
            ex["tree"],
            subset_model,
            symbol_vocab,
            max_subset,
            device,
        )
        if result is not None:
            examples.append(result)
        if (idx + 1) % 500 == 0:
            print(f"    {idx + 1}/{len(raw)}")
    return examples


# ── Collate ──────────────────────────────────────────────────────────


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad variable-N examples to max_N within the batch."""
    max_N = max(ex["N"] for ex in batch)
    B = len(batch)
    d_edge = batch[0]["edge_features"].shape[-1]

    symbol_ids = torch.zeros(B, max_N, dtype=torch.long)
    # Match dtype from data (long for bucketed, float for continuous)
    size_dtype = batch[0]["size_feats"].dtype
    size_feats = torch.zeros(B, max_N, 2, dtype=size_dtype)
    edge_features = torch.zeros(B, max_N, max_N + 1, d_edge)
    pad_mask = torch.ones(B, max_N, dtype=torch.bool)
    gt_parent = torch.full((B, max_N), -100, dtype=torch.long)
    gt_edge_type = torch.full((B, max_N), -100, dtype=torch.long)
    gt_seq = torch.full((B, max_N), -100, dtype=torch.long)

    for i, ex in enumerate(batch):
        N = ex["N"]
        symbol_ids[i, :N] = ex["symbol_ids"]
        size_feats[i, :N] = ex["size_feats"]
        # Node-node block: (N, N) → top-left
        edge_features[i, :N, :N] = ex["edge_features"][:, :N]
        # ROOT column: column N → column max_N
        edge_features[i, :N, max_N] = ex["edge_features"][:, N]
        pad_mask[i, :N] = False
        # Remap gt_parent: ROOT was N (per-example) → max_N (per-batch)
        gp = ex["gt_parent"].clone()
        gp[gp == N] = max_N
        gt_parent[i, :N] = gp
        gt_edge_type[i, :N] = ex["gt_edge_type"]
        # Remap gt_seq: NONE was N (per-example) → max_N (per-batch)
        if "gt_seq" in ex:
            gs = ex["gt_seq"].clone()
            gs[gs == N] = max_N
            gt_seq[i, :N] = gs

    return {
        "symbol_ids": symbol_ids,
        "size_feats": size_feats,
        "edge_features": edge_features,
        "pad_mask": pad_mask,
        "gt_parent": gt_parent,
        "gt_edge_type": gt_edge_type,
        "gt_seq": gt_seq,
    }
