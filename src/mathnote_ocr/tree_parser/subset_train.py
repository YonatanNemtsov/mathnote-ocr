"""Train the SubsetTreeModel for tree parsing.

Each training expression provides many subset training examples:
a random subset of 3-8 symbols is sampled, parent indices are remapped
to subset-local indices, and the model learns to predict the tree
structure from just that subset.

Usage:
    python3.10 tree_parser/subset_train.py --run mixed_v9 --train data/runs/tree_subset/mixed_v7b/train.jsonl --val data/runs/tree_subset/mixed_v7b/val.jsonl --epochs 200
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

from mathnote_ocr.engine.checkpoint import save_checkpoint, _checkpoint_path
from mathnote_ocr.tree_parser.subset_model import SubsetTreeModel
from mathnote_ocr.tree_parser.subset_dataset import TreeSubsetDataset, build_symbol_vocab
from mathnote_ocr.tree_parser.subset_loss import compute_loss

log = logging.getLogger(__name__)


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    run: str,
    train_path: str | Path,
    val_path: str | Path,
    device: torch.device | None = None,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    dropout: float = 0.1,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 256,
    d_arc: int = 64,
    max_subset: int = 8,
    subsets_per_example: int = 3,
    max_examples: int | None = None,
    epoch_size: int | None = None,
    resume: bool = False,
    reset_val: bool = False,
) -> Path:
    """Train a SubsetTreeModel.

    Args:
        run: Run name (checkpoint saved to weights/tree_subset/{run}/).
        train_path: Path to training JSONL file.
        val_path: Path to validation JSONL file.
        device: Torch device (auto-detected if None).
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        dropout: Dropout rate.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        d_ff: Feed-forward dimension.
        d_arc: Biaffine scorer dimension.
        max_subset: Maximum symbols per subset.
        subsets_per_example: Unused, kept for CLI compat.
        max_examples: Limit training examples (None=all).
        epoch_size: Subsample this many subsets per epoch (None=all).
        resume: Resume from existing checkpoint.
        reset_val: Reset best val loss when resuming.

    Returns:
        Path to the checkpoint directory.
    """
    if device is None:
        device = _default_device()
    train_path = Path(train_path)
    val_path = Path(val_path)

    ckpt_kind = "tree_subset"
    run_dir = _checkpoint_path(ckpt_kind, run).parent
    run_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging to file (root logger so all modules are captured)
    log_path = run_dir / "train.log"
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    log.info("Run:    %s", run)
    log.info("Device: %s", device)
    log.info("Train:  %s", train_path)
    log.info("Val:    %s", val_path)

    # Build vocab (or load from checkpoint when resuming)
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    start_epoch = 1
    resumed_ckpt = None

    if resume:
        ckpt_path = run_dir / "checkpoint.pth"
        if ckpt_path.exists():
            resumed_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if resumed_ckpt is not None and "symbol_vocab" in resumed_ckpt:
        symbol_vocab = resumed_ckpt["symbol_vocab"]
        log.info("Using checkpoint vocab: %d symbols", len(symbol_vocab))
    else:
        log.info("Building symbol vocab...")
        symbol_vocab = build_symbol_vocab(train_path)
        log.info("  %d symbols", len(symbol_vocab))

    # Datasets
    log.info("Loading datasets...")
    train_ds = TreeSubsetDataset(
        train_path, symbol_vocab,
        max_subset=max_subset,
        subsets_per_example=subsets_per_example,
        max_examples=max_examples,
        augment=True,
    )
    val_ds = TreeSubsetDataset(
        val_path, symbol_vocab,
        max_subset=max_subset,
        subsets_per_example=1,
        max_examples=max_examples // 5 if max_examples else None,
    )

    if epoch_size and epoch_size < len(train_ds):
        train_sampler = RandomSampler(train_ds, replacement=False,
                                       num_samples=epoch_size)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler,
            num_workers=2, pin_memory=True,
        )
        log.info("Subsampling %d/%d subsets per epoch", epoch_size, len(train_ds))
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True,
        )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Model — use checkpoint config when resuming
    if resumed_ckpt is not None and "config" in resumed_ckpt:
        cfg = resumed_ckpt["config"]
        model = SubsetTreeModel(**cfg).to(device)
    else:
        model = SubsetTreeModel(
            num_symbols=len(symbol_vocab),
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            d_arc=d_arc,
            max_symbols=max_subset,
            dropout=dropout,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %s", f"{total_params:,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    amp_ctx = lambda: torch.autocast("cuda", dtype=torch.float16, enabled=use_amp)

    # Resume weights
    if resumed_ckpt is not None:
        model.load_state_dict(resumed_ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in resumed_ckpt:
            optimizer.load_state_dict(resumed_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resumed_ckpt:
            scheduler.load_state_dict(resumed_ckpt["scheduler_state_dict"])
        if "epoch" in resumed_ckpt:
            start_epoch = resumed_ckpt["epoch"] + 1
        if "best_val_loss" in resumed_ckpt and not reset_val:
            best_val_loss = resumed_ckpt["best_val_loss"]
        log.info("Resumed from epoch %d (best_val_loss=%.4f)", start_epoch - 1, best_val_loss)

    # Save run config
    run_config = {
        "train": str(train_path),
        "val": str(val_path),
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "d_arc": d_arc,
        "max_subset": max_subset,
        "subsets_per_example": subsets_per_example,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "dropout": dropout,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    for epoch in range(start_epoch, start_epoch + epochs):
        _epoch_t0 = time.time()
        # ── Train ──
        model.train()
        epoch_metrics = {k: 0.0 for k in [
            "parent_loss", "edge_loss", "order_loss", "seq_loss",
            "parent_acc", "edge_acc", "seq_acc", "order_mae",
        ]}
        n_batches = 0

        for batch in train_loader:
            with amp_ctx():
                out = model(
                    batch["symbol_ids"].to(device),
                    batch["geo_feats"].to(device),
                    batch["pad_mask"].to(device),
                    batch["size_feats"].to(device),
                )

                loss, metrics = compute_loss(
                    out,
                    batch["parent_targets"].to(device),
                    batch["edge_targets"].to(device),
                    batch["order_targets"].to(device),
                    batch["pad_mask"].to(device),
                    seq_targets=batch["seq_targets"].to(device),
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            for k, v in metrics.items():
                epoch_metrics[k] += v
            n_batches += 1

        for k in epoch_metrics:
            epoch_metrics[k] /= max(n_batches, 1)

        # ── Validate ──
        model.eval()
        val_metrics = {k: 0.0 for k in epoch_metrics}
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                with amp_ctx():
                    out = model(
                        batch["symbol_ids"].to(device),
                        batch["geo_feats"].to(device),
                        batch["pad_mask"].to(device),
                        batch["size_feats"].to(device),
                    )

                    _, metrics = compute_loss(
                        out,
                        batch["parent_targets"].to(device),
                        batch["edge_targets"].to(device),
                        batch["order_targets"].to(device),
                        batch["pad_mask"].to(device),
                        seq_targets=batch["seq_targets"].to(device),
                    )

                for k, v in metrics.items():
                    val_metrics[k] += v
                val_batches += 1

        for k in val_metrics:
            val_metrics[k] /= max(val_batches, 1)

        val_loss = val_metrics["parent_loss"] + val_metrics["edge_loss"]
        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]["lr"]

        log.info(
            "Epoch %3d/%d  t_par=%.1f%%  t_edge=%.1f%%  t_seq=%.1f%%  "
            "v_par=%.1f%%  v_edge=%.1f%%  v_seq=%.1f%%  v_ord=%.2f  "
            "lr=%.2e  (%ds)",
            epoch, start_epoch + epochs - 1,
            100 * epoch_metrics["parent_acc"],
            100 * epoch_metrics["edge_acc"],
            100 * epoch_metrics["seq_acc"],
            100 * val_metrics["parent_acc"],
            100 * val_metrics["edge_acc"],
            100 * val_metrics["seq_acc"],
            val_metrics["order_mae"],
            cur_lr,
            int(time.time() - _epoch_t0),
        )

        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                "model_state_dict": {k: v.clone() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "symbol_vocab": symbol_vocab,
                "config": {
                    "num_symbols": len(symbol_vocab),
                    "d_model": d_model,
                    "n_heads": n_heads,
                    "n_layers": n_layers,
                    "d_ff": d_ff,
                    "d_arc": d_arc,
                    "max_symbols": max_subset,
                    "dropout": dropout,
                },
                "metrics": val_metrics,
            }
            save_checkpoint(ckpt_kind, run, best_state)
            log.info("  -> Best (val_loss=%.4f) -> saved", val_loss)

    if best_state is not None:
        run_config["best_epoch"] = best_epoch
        run_config["best_val_loss"] = round(best_val_loss, 6)
        run_config["best_parent_acc"] = round(best_state["metrics"]["parent_acc"], 4)
        run_config["best_edge_acc"] = round(best_state["metrics"]["edge_acc"], 4)
        run_config["best_seq_acc"] = round(best_state["metrics"]["seq_acc"], 4)
        run_config["finished_at"] = datetime.now(timezone.utc).isoformat()
        (run_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")

        log.info("Done. Best val_loss: %.4f (epoch %d)", best_val_loss, best_epoch)
        log.info("  parent_acc: %.1f%%", 100 * best_state["metrics"]["parent_acc"])
        log.info("  edge_acc:   %.1f%%", 100 * best_state["metrics"]["edge_acc"])
        log.info("Checkpoint: %s", run_dir / "checkpoint.pth")
    else:
        log.info("Done. No improvement found.")

    log.info("Log: %s", log_path)
    log.removeHandler(fh)
    fh.close()
    return run_dir


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = Path(__file__).parent.parent / "data"
    parser = argparse.ArgumentParser(description="Train SubsetTreeModel")

    # Data
    parser.add_argument("--train", default=str(data_dir / "tree_train.jsonl"))
    parser.add_argument("--val", default=str(data_dir / "tree_val.jsonl"))
    parser.add_argument("--run", default="default")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset-val", action="store_true",
                        help="Reset best val loss when resuming (use with new data)")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Model
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--d-arc", type=int, default=64)

    # Subset
    parser.add_argument("--max-subset", type=int, default=8)
    parser.add_argument("--subsets-per-example", type=int, default=3)
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit training examples (None=all)")
    parser.add_argument("--epoch-size", type=int, default=None,
                        help="Subsample this many subsets per epoch (None=all)")

    args = parser.parse_args()

    train(
        run=args.run,
        train_path=args.train,
        val_path=args.val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        d_arc=args.d_arc,
        max_subset=args.max_subset,
        subsets_per_example=args.subsets_per_example,
        max_examples=args.max_examples,
        epoch_size=args.epoch_size,
        resume=args.resume,
        reset_val=args.reset_val,
    )
