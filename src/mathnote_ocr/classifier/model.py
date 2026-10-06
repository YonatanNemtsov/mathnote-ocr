"""
Symbol CNN with prototype-based OOD detection.

Single source of truth for the model architecture.
Based on: "Deep Nearest Neighbor Anomaly Detection" (Bergman & Hoshen, 2020)
"""

import numpy as np
import torch
import torch.nn as nn


class SymbolCNNWithPrototypes(nn.Module):
    """
    3-layer CNN that outputs both class logits and a 256-dim feature vector.
    Optionally accepts a relative size scalar (symbol_diag / canvas_diag)
    concatenated before the FC layers so the model knows how big the
    original symbol was even when the render fills the canvas.

    After training, per-class prototypes (mean feature vectors) are computed.
    At inference, distance to the predicted class's prototype is used for OOD detection.
    """

    def __init__(self, num_classes: int, canvas_size: int = 128, use_size_feat: bool = False):
        super().__init__()
        self.use_size_feat = use_size_feat

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # After 3 pooling layers: canvas_size / 8
        flat_size = 128 * (canvas_size // 8) ** 2
        fc1_in = flat_size + 1 if use_size_feat else flat_size
        self.fc1 = nn.Linear(fc1_in, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

        # Per-class prototypes, computed after training
        self.register_buffer("prototypes", torch.zeros(num_classes, 256))
        self.prototypes_computed = False

    def forward(
        self, x: torch.Tensor, size_feat: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        if self.use_size_feat and size_feat is not None:
            x = torch.cat([x, size_feat.unsqueeze(-1)], dim=-1)

        features = self.relu(self.fc1(x))

        logits = self.fc2(self.dropout(features))
        return logits, features

    def compute_prototypes(self, dataloader, device: torch.device):
        """Compute per-class prototypes as the mean feature vector of each class."""
        self.eval()
        class_features: dict[int, list] = {i: [] for i in range(len(self.prototypes))}

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    images, labels, size_feats = batch
                    size_feats = size_feats.to(torch.float32).to(device)
                else:
                    images, labels = batch
                    size_feats = None
                images = images.to(device)
                _, features = self.forward(images, size_feats)
                for feat, label in zip(features, labels):
                    class_features[label.item()].append(feat.cpu().numpy())

        for class_idx in range(len(self.prototypes)):
            if class_features[class_idx]:
                class_mean = np.mean(class_features[class_idx], axis=0)
                self.prototypes[class_idx] = torch.from_numpy(class_mean)

        self.prototypes_computed = True


class _Block(nn.Module):
    """Residual block: two 3x3 conv + batch norm, a 1x1 projection when the
    shape changes."""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.skip = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))
                     if stride != 1 or cin != cout else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + self.skip(x))


class SymbolResNetWithPrototypes(SymbolCNNWithPrototypes):
    """A small ResNet in place of the 3-layer CNN — the same interface:
    class logits and a 256-dim feature vector (so the prototypes and their
    out-of-vocabulary check work unchanged), and the optional relative size.

    A stem conv, four stages of two residual blocks (32, 64, 128, 256
    channels; each stage after the first halves the resolution), global
    average pooling: any input size works. About 2.8M parameters.
    """

    def __init__(self, num_classes: int, canvas_size: int = 48, use_size_feat: bool = False):
        nn.Module.__init__(self)
        self.use_size_feat = use_size_feat
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        widths = [32, 64, 128, 256]
        layers, cin = [], 32
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            layers += [_Block(cin, w, stride), _Block(w, w, 1)]
            cin = w
        self.stages = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        fc1_in = 256 + 1 if use_size_feat else 256
        self.fc1 = nn.Linear(fc1_in, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.register_buffer("prototypes", torch.zeros(num_classes, 256))
        self.prototypes_computed = False

    def forward(self, x: torch.Tensor, size_feat: torch.Tensor | None = None):
        x = self.pool(self.stages(self.stem(x))).flatten(1)
        if self.use_size_feat and size_feat is not None:
            x = torch.cat([x, size_feat.unsqueeze(-1)], dim=-1)
        features = self.relu(self.fc1(x))
        return self.fc2(self.dropout(features)), features


ARCHITECTURES = {"cnn": SymbolCNNWithPrototypes, "resnet": SymbolResNetWithPrototypes}


def build_model(arch: str, num_classes: int, canvas_size: int, use_size_feat: bool) -> SymbolCNNWithPrototypes:
    """The classifier of architecture *arch* ("cnn": the 3-layer CNN, the
    default and every checkpoint without an "arch"; "resnet")."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown classifier architecture {arch!r}: {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[arch](num_classes=num_classes, canvas_size=canvas_size, use_size_feat=use_size_feat)
