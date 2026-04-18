#!/usr/bin/env python3
"""Evaluate the symbol classifier: confusion matrix, per-class accuracy."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

import config
from classifier.model import SymbolCNNWithPrototypes
from classifier.train import SymbolDataset, load_data, split_data
from engine.checkpoint import load_checkpoint


def evaluate(model, dataloader, device, label_names):
    """Run evaluation and return per-class stats."""
    model.eval()
    num_classes = len(label_names)

    confusion = np.zeros((num_classes, num_classes), dtype=int)
    all_distances = {i: [] for i in range(num_classes)}

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits, features = model(images)
            _, predicted = logits.max(1)

            for pred, true, feat in zip(predicted, labels, features):
                confusion[true.item()][pred.item()] += 1

                # Prototype distance
                dist = torch.norm(feat - model.prototypes[pred.item()]).item()
                all_distances[true.item()].append(dist)

    return confusion, all_distances


def print_confusion_matrix(confusion, label_names):
    """Print a formatted confusion matrix."""
    num_classes = len(label_names)

    # Header
    max_label = max(len(n) for n in label_names)
    header = " " * (max_label + 2) + "  ".join(f"{n:>4}" for n in label_names)
    print(f"\nConfusion Matrix (rows=true, cols=predicted):\n")
    print(header)
    print(" " * (max_label + 2) + "-" * (6 * num_classes))

    for i, name in enumerate(label_names):
        row = "  ".join(f"{confusion[i][j]:4d}" for j in range(num_classes))
        correct = confusion[i][i]
        total = confusion[i].sum()
        acc = 100.0 * correct / total if total > 0 else 0
        print(f"{name:>{max_label}} | {row}  ({acc:.0f}%)")


def print_per_class_stats(confusion, all_distances, label_names):
    """Print per-class accuracy and prototype distances."""
    print(f"\nPer-class statistics:\n")
    print(f"{'Class':<10} {'Acc':>6} {'Samples':>8} {'Avg Dist':>9} {'Max Dist':>9}")
    print("-" * 50)

    total_correct = 0
    total_samples = 0

    for i, name in enumerate(label_names):
        correct = confusion[i][i]
        total = confusion[i].sum()
        acc = 100.0 * correct / total if total > 0 else 0

        dists = all_distances[i]
        avg_dist = np.mean(dists) if dists else 0
        max_dist = np.max(dists) if dists else 0

        print(f"{name:<10} {acc:>5.1f}% {total:>8} {avg_dist:>9.2f} {max_dist:>9.2f}")

        total_correct += correct
        total_samples += total

    overall_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0
    print("-" * 50)
    print(f"{'Overall':<10} {overall_acc:>5.1f}% {total_samples:>8}")


def main():
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    try:
        checkpoint = load_checkpoint("classifier", args.run, device=device)
    except FileNotFoundError:
        print(f"No model found at weights/classifier/{args.run}/. Run classifier/train.py first.")
        return
    label_names = checkpoint["label_names"]

    model = SymbolCNNWithPrototypes(num_classes=len(label_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.prototypes_computed = True

    # Load validation data (same split as training)
    images, labels, _ = load_data()
    _, _, val_images, val_labels = split_data(images, labels)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    val_dataset = SymbolDataset(val_images, val_labels, transform)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    print(f"Evaluating on {len(val_dataset)} validation samples...\n")

    confusion, distances = evaluate(model, val_loader, device, label_names)
    print_confusion_matrix(confusion, label_names)
    print_per_class_stats(confusion, distances, label_names)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="default", help="Classifier run name")
    args = ap.parse_args()
    main()
