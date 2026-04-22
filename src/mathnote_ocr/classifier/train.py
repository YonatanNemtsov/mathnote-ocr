#!/usr/bin/env python3
"""Train the symbol classifier with prototype computation."""

import sys
from pathlib import Path

# Allow running from math_ocr_v2/ or math_ocr_v2/training/
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from mathnote_ocr import config
from mathnote_ocr.classifier.model import SymbolCNNWithPrototypes
from mathnote_ocr.classifier.stroke_augment import augment_strokes
from mathnote_ocr.engine.checkpoint import _checkpoint_path, load_checkpoint, save_checkpoint
from mathnote_ocr.engine.renderer import render_strokes
from mathnote_ocr.engine.stroke import Stroke


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

    # Collect all (class_name, png_path) pairs across directories
    class_images: dict[str, list[Path]] = {}
    for data_dir in data_dirs:
        if not data_dir.exists():
            print(f"  Skipping {data_dir} (does not exist)")
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

    print(f"Found {len(label_names)} classes from {len(data_dirs)} source(s)")

    all_images = []
    all_labels = []

    for name in label_names:
        images = class_images[name]
        label = label_to_idx[name]
        print(f"  {name}: {len(images)} images")
        for img_path in images:
            all_images.append(img_path)
            all_labels.append(label)

    print(f"Total: {len(all_images)} images")
    return all_images, all_labels, label_names


def split_data(images, labels, train_ratio=0.8):
    """Shuffle and split into train/val."""
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)

    split_idx = int(len(images) * train_ratio)
    print(f"Train: {split_idx}, Val: {len(images) - split_idx}")

    return (
        list(images[:split_idx]),
        list(labels[:split_idx]),
        list(images[split_idx:]),
        list(labels[split_idx:]),
    )


def train(
    model, train_loader, val_loader, device, label_names, epochs=15, lr=0.001, use_size_feat=False
):
    """Train the model, return best state dict and metrics."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = None

    def _unpack(batch):
        if use_size_feat:
            imgs, lbls, sf = batch
            return imgs, lbls, sf.float()
        else:
            imgs, lbls = batch
            return imgs, lbls, None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch in train_loader:
            images, labels, size_feats = _unpack(batch)
            images, labels = images.to(device), labels.to(device)
            if size_feats is not None:
                size_feats = size_feats.to(device)
            optimizer.zero_grad()
            logits, _ = model(images, size_feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / train_total

        # Validate
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                images, labels, size_feats = _unpack(batch)
                images, labels = images.to(device), labels.to(device)
                if size_feats is not None:
                    size_feats = size_feats.to(device)
                logits, _ = model(images, size_feats)
                val_loss_sum += criterion(logits, labels).item()
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        val_loss = val_loss_sum / len(val_loader)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss / len(train_loader):.4f} "
            f"train_acc={train_acc:.1f}% "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.1f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  -> Best so far (val_loss: {val_loss:.4f}, val_acc: {val_acc:.1f}%)")

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    return best_val_loss, best_val_acc, best_state


class _Tee:
    """Write to both a file and the original stream."""

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


def main():
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Set up logging
    run_dir = _checkpoint_path("classifier", args.run, weights_dir=args.weights_dir).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    log_file = open(log_path, "a")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    print(f"Device: {device}\n")

    # Load and split data
    data_dirs = [Path(d) for d in args.data]
    images, labels, label_names = load_data(data_dirs)
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

    # Stroke width range: 1.0-4.0 (default is 2.0)
    width_range = (1.0, 4.0)
    print(f"Stroke width range: {width_range}")

    use_size = args.use_size_feat
    train_dataset = SymbolDataset(
        train_images,
        train_labels,
        train_transform,
        width_range=width_range,
        stroke_augment=True,
        canvas_size=args.canvas_size,
        use_size_feat=use_size,
    )
    val_dataset = SymbolDataset(
        val_images,
        val_labels,
        val_transform,
        width_range=(2.0, 2.0),  # fixed width for consistent val renders
        canvas_size=args.canvas_size,
        use_size_feat=use_size,
    )

    # Balanced sampling: weight each sample inversely by class frequency
    from collections import Counter

    class_counts = Counter(train_labels)
    sample_weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = SymbolCNNWithPrototypes(
        num_classes=len(label_names),
        canvas_size=args.canvas_size,
        use_size_feat=use_size,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}\n")

    # Resume from checkpoint if requested
    if args.resume:
        try:
            ckpt = load_checkpoint(
                "classifier", args.run, device=device, weights_dir=args.weights_dir
            )
            state = ckpt["model_state_dict"]
            # Filter out keys with shape mismatches (e.g. different num_classes)
            model_state = model.state_dict()
            compatible = {
                k: v
                for k, v in state.items()
                if k in model_state and v.shape == model_state[k].shape
            }
            skipped = set(state.keys()) - set(compatible.keys())
            model.load_state_dict(compatible, strict=False)
            if skipped:
                print(f"  Skipped {len(skipped)} keys (shape mismatch): {skipped}")
            print("Resumed from existing checkpoint\n")
        except FileNotFoundError:
            print("No checkpoint found to resume from.\n")

    # Train
    best_val_loss, best_val_acc, best_state = train(
        model,
        train_loader,
        val_loader,
        device,
        label_names,
        epochs=args.epochs,
        lr=args.lr,
        use_size_feat=use_size,
    )

    # Compute prototypes on training data using best weights
    print("\nComputing prototypes...")
    model.load_state_dict(best_state)
    model.compute_prototypes(train_loader, device)

    filepath = save_checkpoint(
        "classifier",
        args.run,
        weights_dir=args.weights_dir,
        state_dict={
            "model_state_dict": model.state_dict(),
            "label_names": label_names,
            "prototypes": model.prototypes,
            "canvas_size": args.canvas_size,
            "use_size_feat": use_size,
        },
    )
    print(f"\nModel with prototypes saved to: {filepath}")
    print(f"Log: {log_path}")
    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run", type=str, default="default", help="Run name (saves to weights/classifier/<name>/)"
    )
    ap.add_argument("--epochs", type=int, default=15)
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
    ap.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    args = ap.parse_args()
    main()
