"""Dataset for tree subset training.

Loads full expressions from JSONL, yields spatially-local subsets with
remapped tree labels for training the SubsetTreeModel.
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path

log = logging.getLogger(__name__)

import torch
from torch.utils.data import Dataset

from mathnote_ocr.latex_utils.relations import compute_features_from_bbox_list
from mathnote_ocr.tree_parser.tree import ROOT


# ── Bbox jitter (online augmentation) ────────────────────────────────


def _augment_bboxes(bboxes: list[list[float]]) -> list[list[float]]:
    """Heavy online jitter for raw font data: ±15% pos, ±35% size."""
    if not bboxes:
        return bboxes

    theta = random.uniform(-0.05, 0.05)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx = sum(x + w / 2 for x, y, w, h in bboxes) / len(bboxes)
    cy = sum(y + h / 2 for x, y, w, h in bboxes) / len(bboxes)
    sx = random.uniform(0.8, 1.2)
    sy = random.uniform(0.8, 1.2)

    result = []
    for x, y, w, h in bboxes:
        gx, gy = x + w / 2 - cx, y + h / 2 - cy
        rx = gx * cos_t - gy * sin_t + cx
        ry = gx * sin_t + gy * cos_t + cy
        rx = (rx - cx) * sx + cx
        ry = (ry - cy) * sy + cy
        w *= sx
        h *= sy
        dx = random.uniform(-0.15, 0.15) * h
        dy = random.uniform(-0.15, 0.15) * h
        sw = random.uniform(0.65, 1.35)
        sh = random.uniform(0.65, 1.35)
        result.append([
            max(0.0, rx - w * sw / 2 + dx),
            max(0.0, ry - h * sh / 2 + dy),
            max(0.001, w * sw),
            max(0.001, h * sh),
        ])
    return result


def _augment_bboxes_gentle(bboxes: list[list[float]]) -> list[list[float]]:
    """Light online jitter for pre-augmented data: ±5% pos, ±12% size."""
    if not bboxes:
        return bboxes

    theta = random.uniform(-0.05, 0.05)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx = sum(x + w / 2 for x, y, w, h in bboxes) / len(bboxes)
    cy = sum(y + h / 2 for x, y, w, h in bboxes) / len(bboxes)
    sx = random.uniform(0.88, 1.12)
    sy = random.uniform(0.88, 1.12)

    result = []
    for x, y, w, h in bboxes:
        gx, gy = x + w / 2 - cx, y + h / 2 - cy
        rx = gx * cos_t - gy * sin_t + cx
        ry = gx * sin_t + gy * cos_t + cy
        rx = (rx - cx) * sx + cx
        ry = (ry - cy) * sy + cy
        w *= sx
        h *= sy
        dx = random.uniform(-0.05, 0.05) * h
        dy = random.uniform(-0.05, 0.05) * h
        sw = random.uniform(0.88, 1.12)
        sh = random.uniform(0.88, 1.12)
        result.append([
            max(0.0, rx - w * sw / 2 + dx),
            max(0.0, ry - h * sh / 2 + dy),
            max(0.001, w * sw),
            max(0.001, h * sh),
        ])
    return result


from mathnote_ocr.latex_utils.collapse import EXPR_NAME, random_collapse


# ── Symbol vocab ─────────────────────────────────────────────────────


def build_symbol_vocab(jsonl_path: Path) -> dict[str, int]:
    """Scan training data and build symbol name → ID mapping."""
    names: set[str] = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            for sym in ex["symbols"]:
                names.add(sym["name"])

    names.add(EXPR_NAME)  # ensure 'expression' is in vocab
    vocab = {"<pad>": 0, "<unk>": 1}
    for i, name in enumerate(sorted(names)):
        vocab[name] = i + 2
    return vocab


# ── Dataset ──────────────────────────────────────────────────────────


class TreeSubsetDataset(Dataset):
    """Loads full expressions, yields spatially-local subsets for training.

    Each symbol in each expression serves as the center seed exactly once
    per epoch, guaranteeing full coverage.
    """

    def __init__(
        self,
        jsonl_path: Path,
        symbol_vocab: dict[str, int],
        max_subset: int = 8,
        min_subset: int = 3,
        subsets_per_example: int = 3,  # unused, kept for CLI compat
        max_examples: int | None = None,
        augment: bool = False,
    ) -> None:
        self.examples: list[dict] = []
        self.symbol_vocab = symbol_vocab
        self.max_subset = max_subset
        self.min_subset = min_subset
        self.augment = augment

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                if len(ex["symbols"]) >= min_subset:
                    self.examples.append(ex)
                    if max_examples and len(self.examples) >= max_examples:
                        break

        # Build index: one entry per (example, seed_symbol)
        self.index: list[tuple[int, int]] = []
        for i, ex in enumerate(self.examples):
            for s in range(len(ex["symbols"])):
                self.index.append((i, s))

        log.info("Loaded %d examples from %s (%d subsets)",
                 len(self.examples), jsonl_path, len(self.index))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex_idx, seed = self.index[idx]
        ex = self.examples[ex_idx]
        symbols = ex["symbols"]
        tree = ex["tree"]
        n = len(symbols)
        S = self.max_subset

        # Optionally collapse subtrees into "expression" symbols
        if self.augment and n > self.min_subset + 1:
            symbols, tree = random_collapse(symbols, tree)
            n = len(symbols)
            # Seed may have been removed — clamp to valid range
            if seed >= n:
                seed = random.randint(0, n - 1)

        if n <= self.min_subset:
            # After collapsing, expression might be too small — use original
            symbols = ex["symbols"]
            tree = ex["tree"]
            n = len(symbols)
            seed = self.index[idx][1]

        # Compute neighbors on the fly (cheap for n < 30)
        bboxes_all = [s["bbox"] for s in symbols]
        centers = [(b[0] + b[2] / 2, b[1] + b[3] / 2) for b in bboxes_all]
        heights = [b[3] for b in bboxes_all]
        med_h = sorted(heights)[len(heights) // 2]

        cx, cy = centers[seed]
        dists = []
        for j in range(n):
            if j == seed:
                continue
            dx = centers[j][0] - cx
            dy = centers[j][1] - cy
            dists.append((j, (dx * dx + dy * dy) ** 0.5))
        dists.sort(key=lambda x: x[1])
        neighbors = [d[0] for d in dists]
        neighbor_dists = [d[1] for d in dists]

        # Build subset: radius-based neighbor selection
        radius_mult = random.uniform(2.0, 4.0)
        radius = radius_mult * med_h

        # Count neighbors within radius
        n_in_radius = 0
        for d in neighbor_dists:
            if d <= radius:
                n_in_radius += 1
            else:
                break

        # Clamp to [min_subset-1, max_subset-1] neighbors
        n_take = max(self.min_subset - 1, min(n_in_radius, self.max_subset - 1, n - 1))
        k = n_take + 1  # include seed
        subset = sorted([seed] + neighbors[:n_take])

        # Build global→local index map
        g2l = {g: l for l, g in enumerate(subset)}

        # Symbol IDs
        symbol_ids = torch.zeros(S, dtype=torch.long)
        for i, gi in enumerate(subset):
            name = symbols[gi]["name"]
            symbol_ids[i] = self.symbol_vocab.get(name, self.symbol_vocab.get("<unk>", 1))

        # Bounding boxes for subset (with optional augmentation)
        bbox_list = [symbols[gi]["bbox"] for gi in subset]
        if self.augment:
            name_list = [symbols[gi]["name"] for gi in subset]
            bbox_list = _augment_bboxes_gentle(bbox_list)
        geo_feats, size_feats = compute_features_from_bbox_list(bbox_list, S)

        # Pad mask
        pad_mask = torch.ones(S, dtype=torch.bool)
        pad_mask[:k] = False

        # Remap tree labels
        parent_targets = torch.full((S,), -100, dtype=torch.long)  # -100 = ignore
        edge_targets = torch.full((S,), -100, dtype=torch.long)
        order_targets = torch.zeros(S, dtype=torch.float32)
        seq_targets = torch.full((S,), -100, dtype=torch.long)

        for i, gi in enumerate(subset):
            t = tree[gi]
            global_parent = t["parent"]

            if global_parent == ROOT or global_parent not in g2l:
                # Parent is ROOT or outside subset → ROOT (last column in model output)
                parent_targets[i] = S
            else:
                parent_targets[i] = g2l[global_parent]

            et = t["edge_type"]
            edge_targets[i] = et if et >= 0 else -100  # mask ROOT children
            order_targets[i] = t["order"]

            # SEQ target: previous sibling (same parent + edge_type, order - 1)
            if t["order"] == 0:
                seq_targets[i] = S  # first child → "no previous sibling"
            else:
                # Find previous sibling in subset
                for j, gj in enumerate(subset):
                    tj = tree[gj]
                    if (tj["parent"] == t["parent"]
                            and tj["edge_type"] == t["edge_type"]
                            and tj["order"] == t["order"] - 1):
                        seq_targets[i] = j
                        break
                # else: -100 (prev sibling not in subset, ignore)

        return {
            "symbol_ids": symbol_ids,
            "geo_feats": geo_feats,
            "size_feats": size_feats,
            "pad_mask": pad_mask,
            "parent_targets": parent_targets,
            "edge_targets": edge_targets,
            "order_targets": order_targets,
            "seq_targets": seq_targets,
            "n_real": k,
        }
