#!/usr/bin/env python3
"""Train the symbol classifier with prototype computation.

Follows the package training-script standard (subset_train, gnn/train):
run dir with train.log + config.json, checkpoint saved on every val
improvement (with prototypes, so any saved checkpoint is usable for
inference), true resume with optimizer/scheduler/epoch state.

Usage:
    python3.10 -m mathnote_ocr.classifier.train --run <run_name> \
        --data data/runs/classifier/<run_name>/pool \
        --canvas-size 32 --use-size-feat
    # continue an existing run for 20 more epochs:
    python3.10 -m mathnote_ocr.classifier.train --run <run_name> \
        --data data/runs/classifier/<run_name>/pool --resume --epochs 20
"""

import argparse
import json
import logging
import math
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from mathnote_ocr import config
from mathnote_ocr.classifier.model import SymbolCNNWithPrototypes
from mathnote_ocr.classifier.stroke_augment import augment_strokes
from mathnote_ocr.engine.checkpoint import _checkpoint_path, save_checkpoint
from mathnote_ocr.engine.renderer import render_strokes
from mathnote_ocr.engine.stroke import Stroke

log = logging.getLogger(__name__)


def _default_device() -> torch.device:
    # No MPS: slower than CPU at this model size (launch overhead)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SymbolDataset(Dataset):
    """Symbol dataset that re-renders from stroke JSON with random width.

    For each sample, loads the stroke JSON, picks a random stroke width,
    and renders a fresh image. Falls back to the pre-rendered PNG if no
    JSON is available.
    """

    def __init__(
        self,
        image_paths,
        labels,
        transform=None,
        width_range=None,
        stroke_augment=False,
        canvas_size=128,
        use_size_feat=False,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.width_range = width_range  # (min, max) stroke width or None
        self.stroke_augment = stroke_augment
        self.canvas_size = canvas_size
        self.use_size_feat = use_size_feat

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img_path = self.image_paths[idx]

        image = None
        size_feat = 0.5  # default: medium size
        if self.width_range is not None:
            json_path = img_path.with_suffix(".json")
            if json_path.exists():
                image, size_feat = self._render_from_json(json_path)

        if image is None:
            image = Image.open(img_path).convert("L")
            if self.canvas_size != 128:
                image = image.resize((self.canvas_size, self.canvas_size), Image.LANCZOS)

        if self.transform:
            image = self.transform(image)
        if self.use_size_feat:
            return image, label, size_feat
        return image, label

    def _render_from_json(self, json_path: Path) -> tuple[Image.Image, float]:
        with open(json_path) as f:
            data = json.load(f)
        w_min, w_max = self.width_range
        width = random.uniform(w_min, w_max)
        strokes = [
            Stroke.from_dicts(pts, id=i, width=width)
            for i, pts in enumerate(data["strokes"])
        ]
        if self.stroke_augment:
            strokes = augment_strokes(strokes)
        source_size = max(
            data.get("canvas_width", 800),
            data.get("canvas_height", 400),
        )

        # Compute relative size: symbol bbox diagonal / source diagonal
        all_x = [p.x for s in strokes for p in s.points]
        all_y = [p.y for s in strokes for p in s.points]
        if all_x:
            bw = max(all_x) - min(all_x)
            bh = max(all_y) - min(all_y)
            sym_diag = math.sqrt(bw * bw + bh * bh)
            size_feat = sym_diag / max(source_size, 1.0)
        else:
            size_feat = 0.5

        # Size variation: randomly inflate source_size so the symbol
        # renders smaller within the canvas (1x to ~0.4x)
        if self.stroke_augment:
            source_size *= random.uniform(1.0, 2.5)
        img = render_strokes(
            strokes,
            canvas_size=self.canvas_size,
            source_size=source_size,
        )
        return img, size_feat


def load_data(data_dirs: list[Path] | Path):
    """Load all symbol images and create label mapping.

    Accepts a single directory or a list of directories. When multiple
    directories are given, classes are merged by name (e.g. '+' from
    data/shared/symbols/ and '+' from data/shared/symbols_from_expr/ are combined).
    """
    if isinstance(data_dirs, Path):
        data_dirs = [data_dirs]

    class_images: dict[str, list[Path]] = {}
    for data_dir in data_dirs:
        if not data_dir.exists():
            log.info("  Skipping %s (does not exist)", data_dir)
            continue
        symbol_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        for symbol_dir in symbol_dirs:
            name = symbol_dir.name
            jsons = list(symbol_dir.glob("*.json"))
            if jsons:
                class_images.setdefault(name, []).extend(jsons)

    if not class_images:
        raise ValueError(f"No symbol data found in {data_dirs}")

    label_names = sorted(class_images.keys())
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}

    log.info("Found %d classes from %d source(s)", len(label_names), len(data_dirs))

    all_images = []
    all_labels = []

    for name in label_names:
        images = class_images[name]
        label = label_to_idx[name]
        log.info("  %s: %d images", name, len(images))
        for img_path in images:
            all_images.append(img_path)
            all_labels.append(label)

    log.info("Total: %d images", len(all_images))
    return all_images, all_labels, label_names


def split_data(images, labels, train_ratio=0.8):
    """Shuffle and split into train/val."""
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)

    split_idx = int(len(images) * train_ratio)
    log.info("Train: %d, Val: %d", split_idx, len(images) - split_idx)

    return (
        list(images[:split_idx]),
        list(labels[:split_idx]),
        list(images[split_idx:]),
        list(labels[split_idx:]),
    )


def train(
    run: str,
    data_dirs: list[str | Path],
    weights_dir: str | Path = "weights",
    device: torch.device | None = None,
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 0.001,
    canvas_size: int = 128,
    use_size_feat: bool = False,
    seed: int | None = None,
    threads: int = 4,
    resume: bool = False,
    reset_val: bool = False,
) -> Path:
    """Train the classifier. Returns the checkpoint path.

    With ``resume``, weights/optimizer/scheduler/epoch continue from the
    existing checkpoint and ``epochs`` means additional epochs; canvas
    size and size-feat flag are taken from the checkpoint. If the data's
    class set differs from the checkpoint's, falls back to a warm start
    (shape-compatible weights only, fresh optimizer).
    """
    seed = config.SEED if seed is None else seed
    random.seed(seed)
    torch.manual_seed(seed)
    device = device or _default_device()
    if device.type == "cpu":
        # All-core parallelism is pure overhead at this model size
        torch.set_num_threads(threads)

    ckpt_path = _checkpoint_path("classifier", run, weights_dir=weights_dir)
    run_dir = ckpt_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "train.log", mode="a")
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    )
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    resumed = None
    if resume:
        if ckpt_path.exists():
            resumed = torch.load(ckpt_path, map_location=device, weights_only=False)
        else:
            log.info("--resume: no checkpoint at %s, starting fresh", ckpt_path)

    log.info("Run:    %s", run)
    log.info("Device: %s (threads=%d)", device, threads)
    log.info("Data:   %s", ", ".join(str(d) for d in data_dirs))

    images, labels, label_names = load_data([Path(d) for d in data_dirs])

    warm_start = False
    if resumed is not None:
        if resumed["label_names"] == label_names:
            # True resume: model shape and flags come from the checkpoint
            canvas_size = resumed.get("canvas_size", canvas_size)
            use_size_feat = resumed.get("use_size_feat", use_size_feat)
        else:
            warm_start = True
            log.info(
                "Class set changed (%d ckpt vs %d data) — warm start, fresh optimizer",
                len(resumed["label_names"]), len(label_names),
            )

    train_images, train_labels, val_images, val_labels = split_data(images, labels)

    # Transforms — geometric augmentation is done at stroke level,
    # keep only pixel-level effects here
    train_transform = transforms.Compose(
        [
            transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), value=1.0),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    width_range = (1.0, 4.0)
    log.info("Stroke width range: %s", width_range)

    train_dataset = SymbolDataset(
        train_images,
        train_labels,
        train_transform,
        width_range=width_range,
        stroke_augment=True,
        canvas_size=canvas_size,
        use_size_feat=use_size_feat,
    )
    val_dataset = SymbolDataset(
        val_images,
        val_labels,
        val_transform,
        width_range=(2.0, 2.0),  # fixed width for consistent val renders
        canvas_size=canvas_size,
        use_size_feat=use_size_feat,
    )

    # Balanced sampling: weight each sample inversely by class frequency
    class_counts = Counter(train_labels)
    sample_weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SymbolCNNWithPrototypes(
        num_classes=len(label_names),
        canvas_size=canvas_size,
        use_size_feat=use_size_feat,
    ).to(device)
    log.info("Model parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_epoch = None
    start_epoch = 1
    if resumed is not None:
        if warm_start:
            state = resumed["model_state_dict"]
            model_state = model.state_dict()
            compatible = {
                k: v for k, v in state.items()
                if k in model_state and v.shape == model_state[k].shape
            }
            skipped = set(state.keys()) - set(compatible.keys())
            model.load_state_dict(compatible, strict=False)
            if skipped:
                log.info("  Skipped %d keys (shape mismatch): %s", len(skipped), skipped)
        else:
            model.load_state_dict(resumed["model_state_dict"])
            if "optimizer_state_dict" in resumed:
                optimizer.load_state_dict(resumed["optimizer_state_dict"])
            if "scheduler_state_dict" in resumed:
                scheduler.load_state_dict(resumed["scheduler_state_dict"])
            start_epoch = resumed.get("epoch", 0) + 1
            if not reset_val:
                best_val_loss = resumed.get("best_val_loss", float("inf"))
            log.info(
                "Resumed from epoch %d (best val_loss %.4f)", start_epoch - 1, best_val_loss
            )

    run_config = {
        "data": [str(d) for d in data_dirs],
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "canvas_size": canvas_size,
        "use_size_feat": use_size_feat,
        "seed": seed,
        "device": str(device),
        "resumed_from_epoch": start_epoch - 1 if resumed is not None else None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    def _unpack(batch):
        if use_size_feat:
            imgs, lbls, sf = batch
            return imgs, lbls, sf.float()
        imgs, lbls = batch
        return imgs, lbls, None

    for epoch in range(start_epoch, start_epoch + epochs):
        t0 = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch in train_loader:
            images_b, labels_b, size_feats = _unpack(batch)
            images_b, labels_b = images_b.to(device), labels_b.to(device)
            if size_feats is not None:
                size_feats = size_feats.to(device)
            optimizer.zero_grad()
            logits, _ = model(images_b, size_feats)
            loss = criterion(logits, labels_b)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels_b.size(0)
            train_correct += predicted.eq(labels_b).sum().item()

        train_acc = 100.0 * train_correct / train_total

        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                images_b, labels_b, size_feats = _unpack(batch)
                images_b, labels_b = images_b.to(device), labels_b.to(device)
                if size_feats is not None:
                    size_feats = size_feats.to(device)
                logits, _ = model(images_b, size_feats)
                val_loss_sum += criterion(logits, labels_b).item()
                _, predicted = logits.max(1)
                val_total += labels_b.size(0)
                val_correct += predicted.eq(labels_b).sum().item()

        val_acc = 100.0 * val_correct / val_total
        val_loss = val_loss_sum / len(val_loader)
        scheduler.step(val_loss)

        log.info(
            "Epoch %d/%d: train_loss=%.4f train_acc=%.1f%% val_loss=%.4f val_acc=%.1f%% (%ds)",
            epoch, start_epoch + epochs - 1,
            train_loss / len(train_loader), train_acc, val_loss, val_acc,
            int(time.time() - t0),
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Prototypes recomputed each save so the checkpoint is always
            # inference-ready, even if the run is interrupted later.
            model.compute_prototypes(train_loader, device)
            save_checkpoint(
                "classifier",
                run,
                weights_dir=weights_dir,
                state_dict={
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "label_names": label_names,
                    "prototypes": model.prototypes,
                    "canvas_size": canvas_size,
                    "use_size_feat": use_size_feat,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "val_acc": val_acc,
                },
            )
            log.info("  -> Best (val_loss=%.4f, val_acc=%.1f%%) -> saved", val_loss, val_acc)

    run_config["best_epoch"] = best_epoch
    run_config["best_val_loss"] = round(best_val_loss, 6)
    run_config["finished_at"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    log.info("Best val_loss %.4f (epoch %s) -> %s", best_val_loss, best_epoch, ckpt_path)
    return ckpt_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run", type=str, default="default", help="Run name (saves to weights/classifier/<name>/)"
    )
    ap.add_argument("--epochs", type=int, default=15, help="Epochs (additional when resuming)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument(
        "--data",
        type=str,
        nargs="+",
        default=["data/shared/symbols"],
        help="Data directories (default: ./data/shared/symbols). Can specify multiple.",
    )
    ap.add_argument(
        "--weights-dir",
        type=str,
        default="weights",
        help="Directory to save weights (default: ./weights)",
    )
    ap.add_argument("--canvas-size", type=int, default=128, help="Image size (default 128)")
    ap.add_argument(
        "--use-size-feat", action="store_true", help="Pass relative symbol size to model"
    )
    ap.add_argument("--resume", action="store_true", help="Continue from existing checkpoint")
    ap.add_argument("--reset-val", action="store_true", help="Reset best val loss when resuming")
    ap.add_argument("--seed", type=int, default=None, help="Default: config.SEED")
    ap.add_argument("--device", default=None, help="cpu | cuda | mps (default: auto)")
    ap.add_argument("--threads", type=int, default=4, help="CPU torch threads (heat vs speed)")
    args = ap.parse_args()

    train(
        run=args.run,
        data_dirs=args.data,
        weights_dir=args.weights_dir,
        device=torch.device(args.device) if args.device else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        canvas_size=args.canvas_size,
        use_size_feat=args.use_size_feat,
        seed=args.seed,
        threads=args.threads,
        resume=args.resume,
        reset_val=args.reset_val,
    )


if __name__ == "__main__":
    main()
