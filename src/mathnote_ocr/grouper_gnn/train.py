#!/usr/bin/env python3
"""Train the stroke GNN grouper."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from mathnote_ocr import config
from mathnote_ocr.grouper_gnn.model import StrokeGNN
from mathnote_ocr.grouper_gnn.features import (
    compute_node_features,
    compute_edge_features,
    compute_adjacency_mask,
)
from mathnote_ocr.engine.checkpoint import save_checkpoint, _checkpoint_path


# ── Data loading ─────────────────────────────────────────────────────


def _load_train_strokes(path: Path) -> list[dict]:
    """Load expressions from train_strokes.jsonl.

    Each entry has symbols with strokes and labels already grouped.
    Returns list of {strokes: [[{x,y},...], ...], group_ids: [int,...], labels: [str,...]}
    """
    entries = []
    with open(path) as f:
        for line in f:
            expr = json.loads(line)
            strokes = []
            group_ids = []
            labels = []
            for gid, sym in enumerate(expr["symbols"]):
                for stroke_pts in sym["strokes"]:
                    strokes.append(stroke_pts)
                    group_ids.append(gid)
                    labels.append(sym["name"])
            if len(strokes) >= 2:  # need at least 2 strokes
                entries.append({
                    "strokes": strokes,
                    "group_ids": group_ids,
                    "labels": labels,
                })
    return entries


def _load_expr_mapping(mapping_path: Path) -> list[dict]:
    """Load expressions from expr_mapping.json.

    Reconstructs full expressions by loading symbol JSONs and grouping
    by expr_idx.
    """
    with open(mapping_path) as f:
        mapping = json.load(f)

    base_dir = config.ROOT_DIR  # paths in mapping are relative to math_ocr_v2/
    by_expr = defaultdict(list)
    for entry in mapping:
        by_expr[entry["expr_idx"]].append(entry)

    entries = []
    for expr_idx in sorted(by_expr.keys()):
        syms = sorted(by_expr[expr_idx], key=lambda e: e["sym_idx"])
        strokes = []
        group_ids = []
        labels = []

        for gid, sym in enumerate(syms):
            json_path = base_dir / sym["json"]
            if not json_path.exists():
                continue
            with open(json_path) as f:
                sdata = json.load(f)
            for stroke_pts in sdata["strokes"]:
                strokes.append(stroke_pts)
                group_ids.append(gid)
                labels.append(sym["name"])

        if len(strokes) >= 2:
            entries.append({
                "strokes": strokes,
                "group_ids": group_ids,
                "labels": labels,
            })

    return entries


def build_label_vocab(entries: list[dict]) -> dict[str, int]:
    """Build label → index mapping from all entries."""
    names = sorted(set(l for e in entries for l in e["labels"]))
    return {name: idx for idx, name in enumerate(names)}


# ── Dataset ──────────────────────────────────────────────────────────


class StrokeGraphDataset(Dataset):
    """Dataset of expression stroke graphs."""

    def __init__(
        self,
        entries: list[dict],
        label_vocab: dict[str, int],
        stroke_width: float = 2.0,
        augment: bool = False,
    ):
        self.entries = entries
        self.label_vocab = label_vocab
        self.stroke_width = stroke_width
        self.augment = augment

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        strokes = list(entry["strokes"])
        group_ids = list(entry["group_ids"])
        labels = list(entry["labels"])

        # Optional augmentation
        if self.augment:
            # Shuffle symbol order (reorder whole symbol groups)
            strokes, group_ids, labels = _shuffle_symbol_order(
                strokes, group_ids, labels)
            # Spatial augmentation
            strokes = _augment_strokes(strokes)

        # Compute features
        renders, geo = compute_node_features(
            strokes, stroke_width=self.stroke_width,
        )
        edge_feats = compute_edge_features(strokes)
        adj_mask = compute_adjacency_mask(strokes)

        N = len(strokes)

        # Edge targets: 1 if same group, 0 otherwise
        edge_target = torch.zeros(N, N)
        for i in range(N):
            for j in range(N):
                if i != j and group_ids[i] == group_ids[j]:
                    edge_target[i, j] = 1.0

        # Node targets: class index
        node_target = torch.tensor(
            [self.label_vocab.get(l, 0) for l in labels],
            dtype=torch.long,
        )

        return renders, geo, edge_feats, adj_mask, edge_target, node_target


def _shuffle_symbol_order(
    strokes: list[list[dict]],
    group_ids: list[int],
    labels: list[str],
) -> tuple[list, list, list]:
    """Gently perturb symbol order by swapping 1 pair of adjacent groups.

    50% chance of doing anything. When active, picks one random adjacent
    pair of symbol groups and swaps them. This preserves most of the
    original stroke order while teaching the model not to over-rely on it.
    """
    # Group strokes by symbol
    groups: dict[int, list[int]] = {}
    for i, gid in enumerate(group_ids):
        groups.setdefault(gid, []).append(i)

    group_keys = list(groups.keys())
    if len(group_keys) < 2 or random.random() > 0.5:
        return strokes, group_ids, labels

    # Swap one random adjacent pair
    swap_idx = random.randrange(len(group_keys) - 1)
    group_keys[swap_idx], group_keys[swap_idx + 1] = \
        group_keys[swap_idx + 1], group_keys[swap_idx]

    # Rebuild in new order
    new_strokes, new_group_ids, new_labels = [], [], []
    for new_gid, old_gid in enumerate(group_keys):
        for stroke_idx in groups[old_gid]:
            new_strokes.append(strokes[stroke_idx])
            new_group_ids.append(new_gid)
            new_labels.append(labels[stroke_idx])

    return new_strokes, new_group_ids, new_labels


def _augment_strokes(strokes: list[list[dict]]) -> list[list[dict]]:
    """Apply light spatial augmentation: random scale + translation."""
    all_x = [p["x"] for pts in strokes for p in pts]
    all_y = [p["y"] for pts in strokes for p in pts]
    if not all_x:
        return strokes

    cx = sum(all_x) / len(all_x)
    cy = sum(all_y) / len(all_y)

    # Random scale
    sx = random.uniform(0.85, 1.15)
    sy = random.uniform(0.85, 1.15)

    # Random translation
    tx = random.uniform(-10, 10)
    ty = random.uniform(-10, 10)

    result = []
    for pts in strokes:
        new_pts = []
        for p in pts:
            new_pts.append({
                "x": (p["x"] - cx) * sx + cx + tx,
                "y": (p["y"] - cy) * sy + cy + ty,
            })
        result.append(new_pts)
    return result


def collate_fn(batch):
    """Pad variable-length stroke graphs to batch max N."""
    max_n = max(r.shape[0] for r, _, _, _, _, _ in batch)
    B = len(batch)

    render_size = batch[0][0].shape[-1]  # infer from data
    renders_batch = torch.zeros(B, max_n, 1, render_size, render_size)
    geo_batch = torch.zeros(B, max_n, 8)
    edge_feats_batch = torch.zeros(B, max_n, max_n, 6)
    adj_mask_batch = torch.zeros(B, max_n, max_n, dtype=torch.bool)
    edge_target_batch = torch.zeros(B, max_n, max_n)
    node_target_batch = torch.full((B, max_n), -100, dtype=torch.long)  # -100 = ignore
    pad_mask = torch.ones(B, max_n, dtype=torch.bool)  # True = pad

    for i, (renders, geo, edge_feats, adj_mask, edge_target, node_target) in enumerate(batch):
        n = renders.shape[0]
        renders_batch[i, :n] = renders
        geo_batch[i, :n] = geo
        edge_feats_batch[i, :n, :n] = edge_feats
        adj_mask_batch[i, :n, :n] = adj_mask
        edge_target_batch[i, :n, :n] = edge_target
        node_target_batch[i, :n] = node_target
        pad_mask[i, :n] = False

    return renders_batch, geo_batch, edge_feats_batch, adj_mask_batch, edge_target_batch, node_target_batch, pad_mask


# ── Training ─────────────────────────────────────────────────────────


class _Tee:
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file
    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()
    def flush(self):
        self._stream.flush()
        self._log.flush()


def train(
    model: StrokeGNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    pos_weight: float = 4.0,
):
    """Train the GNN model."""
    node_criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_edge_loss_sum = 0.0
        train_node_loss_sum = 0.0
        train_edge_correct = 0
        train_edge_total = 0
        train_node_correct = 0
        train_node_total = 0
        n_batches = 0

        for renders, geo, edge_feats, adj_mask, edge_target, node_target, pad_mask in train_loader:
            renders = renders.to(device)
            geo = geo.to(device)
            edge_feats = edge_feats.to(device)
            adj_mask = adj_mask.to(device)
            edge_target = edge_target.to(device)
            node_target = node_target.to(device)
            pad_mask = pad_mask.to(device)

            optimizer.zero_grad()

            edge_scores, node_logits = model(renders, geo, edge_feats, pad_mask, adj_mask)

            # Edge loss: BCE on valid (non-padded) pairs
            B, N = pad_mask.shape
            valid_mask = ~pad_mask  # (B, N)
            # Valid pairs: both strokes are real
            pair_valid = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)  # (B, N, N)
            # Exclude diagonal
            diag = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
            pair_valid = pair_valid & ~diag

            if pair_valid.any():
                edge_pred = edge_scores[pair_valid]
                edge_tgt = edge_target[pair_valid]
                # Weighted BCE: positive edges are rarer
                weight = torch.where(edge_tgt == 1, pos_weight, 1.0)
                edge_loss = nn.functional.binary_cross_entropy_with_logits(
                    edge_pred, edge_tgt, weight=weight,
                )
            else:
                edge_loss = torch.tensor(0.0, device=device)

            # Node loss: CE on valid strokes
            valid_nodes = ~pad_mask  # (B, N)
            if valid_nodes.any():
                node_loss = node_criterion(
                    node_logits[valid_nodes],
                    node_target[valid_nodes],
                )
            else:
                node_loss = torch.tensor(0.0, device=device)

            loss = edge_loss + 0.5 * node_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_edge_loss_sum += edge_loss.item()
            train_node_loss_sum += node_loss.item()
            n_batches += 1

            # Accuracy
            if pair_valid.any():
                with torch.no_grad():
                    pred_binary = (edge_scores[pair_valid] > 0).long()
                    tgt_binary = edge_target[pair_valid].long()
                    train_edge_correct += (pred_binary == tgt_binary).sum().item()
                    train_edge_total += pred_binary.numel()

            if valid_nodes.any():
                with torch.no_grad():
                    pred_cls = node_logits[valid_nodes].argmax(dim=-1)
                    real_nodes = node_target[valid_nodes] != -100
                    if real_nodes.any():
                        train_node_correct += (pred_cls[real_nodes] == node_target[valid_nodes][real_nodes]).sum().item()
                        train_node_total += real_nodes.sum().item()

        scheduler.step()

        # Validate
        model.eval()
        val_edge_loss_sum = 0.0
        val_node_loss_sum = 0.0
        val_edge_correct = 0
        val_edge_total = 0
        val_node_correct = 0
        val_node_total = 0
        val_batches = 0

        with torch.no_grad():
            for renders, geo, edge_feats, adj_mask, edge_target, node_target, pad_mask in val_loader:
                renders = renders.to(device)
                geo = geo.to(device)
                edge_feats = edge_feats.to(device)
                adj_mask = adj_mask.to(device)
                edge_target = edge_target.to(device)
                node_target = node_target.to(device)
                pad_mask = pad_mask.to(device)

                edge_scores, node_logits = model(renders, geo, edge_feats, pad_mask, adj_mask)

                B, N = pad_mask.shape
                valid_mask = ~pad_mask
                pair_valid = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
                diag = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
                pair_valid = pair_valid & ~diag

                if pair_valid.any():
                    edge_pred = edge_scores[pair_valid]
                    edge_tgt = edge_target[pair_valid]
                    weight = torch.where(edge_tgt == 1, pos_weight, 1.0)
                    edge_loss = nn.functional.binary_cross_entropy_with_logits(
                        edge_pred, edge_tgt, weight=weight,
                    )
                    val_edge_loss_sum += edge_loss.item()

                    pred_binary = (edge_pred > 0).long()
                    tgt_binary = edge_tgt.long()
                    val_edge_correct += (pred_binary == tgt_binary).sum().item()
                    val_edge_total += pred_binary.numel()

                valid_nodes = ~pad_mask
                if valid_nodes.any():
                    node_loss = node_criterion(
                        node_logits[valid_nodes],
                        node_target[valid_nodes],
                    )
                    val_node_loss_sum += node_loss.item()

                    pred_cls = node_logits[valid_nodes].argmax(dim=-1)
                    real_nodes = node_target[valid_nodes] != -100
                    if real_nodes.any():
                        val_node_correct += (pred_cls[real_nodes] == node_target[valid_nodes][real_nodes]).sum().item()
                        val_node_total += real_nodes.sum().item()

                val_batches += 1

        # Metrics
        train_edge_acc = 100.0 * train_edge_correct / max(train_edge_total, 1)
        train_node_acc = 100.0 * train_node_correct / max(train_node_total, 1)
        val_edge_acc = 100.0 * val_edge_correct / max(val_edge_total, 1)
        val_node_acc = 100.0 * val_node_correct / max(val_node_total, 1)
        val_loss = (val_edge_loss_sum + 0.5 * val_node_loss_sum) / max(val_batches, 1)

        print(
            f"Epoch {epoch + 1}/{epochs}  "
            f"train_edge={train_edge_loss_sum / max(n_batches, 1):.4f} "
            f"train_node={train_node_loss_sum / max(n_batches, 1):.4f} "
            f"edge_acc={train_edge_acc:.1f}% "
            f"node_acc={train_node_acc:.1f}% | "
            f"val_edge={val_edge_loss_sum / max(val_batches, 1):.4f} "
            f"val_node={val_node_loss_sum / max(val_batches, 1):.4f} "
            f"edge_acc={val_edge_acc:.1f}% "
            f"node_acc={val_node_acc:.1f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  -> Best (val_loss={val_loss:.4f}, edge={val_edge_acc:.1f}%, node={val_node_acc:.1f}%)")

    return best_val_loss, best_state


# ── Main ─────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="v1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=4.0,
                     help="Weight for positive (same_symbol) edges in BCE loss")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Logging
    run_dir = _checkpoint_path("grouper_gnn", args.run).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    log_file = open(log_path, "a")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    print(f"Device: {device}")
    print(f"Args: {vars(args)}\n")

    # Load data from both sources
    data_dir = config.DATA_DIR

    entries = []
    # Load all train_strokes.jsonl from tree_handwritten/*/
    hw_dir = data_dir / "tree_handwritten"
    if hw_dir.exists():
        for run_subdir in sorted(hw_dir.iterdir()):
            ts_path = run_subdir / "train_strokes.jsonl"
            if ts_path.exists():
                ts_entries = _load_train_strokes(ts_path)
                print(f"Loaded {len(ts_entries)} expressions from {ts_path.relative_to(data_dir)}")
                entries.extend(ts_entries)

    expr_mapping_path = data_dir / "symbols_from_expr" / "expr_mapping.json"
    if expr_mapping_path.exists():
        em_entries = _load_expr_mapping(expr_mapping_path)
        print(f"Loaded {len(em_entries)} expressions from expr_mapping.json")
        entries.extend(em_entries)

    print(f"Total: {len(entries)} expressions")
    total_strokes = sum(len(e["strokes"]) for e in entries)
    print(f"Total strokes: {total_strokes}")

    # Build label vocab
    label_vocab = build_label_vocab(entries)
    label_names = sorted(label_vocab.keys(), key=lambda k: label_vocab[k])
    print(f"Classes: {len(label_names)}")

    # Train/val split
    random.shuffle(entries)
    split = int(len(entries) * 0.85)
    train_entries = entries[:split]
    val_entries = entries[split:]
    print(f"Train: {len(train_entries)}, Val: {len(val_entries)}\n")

    # Datasets
    train_dataset = StrokeGraphDataset(
        train_entries, label_vocab, augment=True,
    )
    val_dataset = StrokeGraphDataset(
        val_entries, label_vocab, augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Model
    model = StrokeGNN(num_classes=len(label_names), d_edge=6).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}\n")

    # Train
    best_val_loss, best_state = train(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, pos_weight=args.pos_weight,
    )

    # Save
    filepath = save_checkpoint("grouper_gnn", args.run, {
        "model_state_dict": best_state,
        "label_names": label_names,
        "label_vocab": label_vocab,
        "config": {
            "num_classes": len(label_names),
            "render_size": 32,
            "d_render": 32,
            "d_geo": 16,
            "n_geo_feats": 8,
            "d_edge": 6,
            "n_heads": 4,
            "n_layers": 3,
        },
    })
    print(f"\nSaved to {filepath}")

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == "__main__":
    main()
