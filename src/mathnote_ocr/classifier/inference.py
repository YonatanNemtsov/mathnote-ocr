"""Symbol classification with prototype-based OOD detection."""

from dataclasses import dataclass

import torch
from PIL import Image
from torchvision import transforms

from mathnote_ocr.classifier.model import SymbolCNNWithPrototypes, build_model
from mathnote_ocr.engine.checkpoint import load_checkpoint


@dataclass
class ClassificationResult:
    symbol: str | None  # None if rejected as OOD
    confidence: float
    prototype_distance: float
    is_ood: bool
    alternatives: list[tuple[str, float]] = None  # [(symbol, confidence), ...]
    probs: object = None  # full distribution over label_names (np.ndarray), batch path only
    features: object = None  # the classifier's view of the shape: its layer before the class scores
                             # (np.ndarray, 256), batch path only — the space its prototypes live in


_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


class SymbolClassifier:
    """Loads a trained model and classifies grayscale symbol images."""

    def __init__(
        self,
        run: str = "v4",
        device: torch.device | None = None,
        ood_threshold: float = 15.0,
        per_class_thresholds: dict[str, float] | None = None,
        weights_dir: str | None = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = load_checkpoint("classifier", run, device=self.device, weights_dir=weights_dir)
        self.label_names: list[str] = checkpoint["label_names"]
        self.canvas_size: int = checkpoint.get("canvas_size", 128)
        self.use_size_feat: bool = checkpoint.get("use_size_feat", False)
        # the rendering it was trained with (renderer.render_strokes); older checkpoints: off
        self.round_joins: bool = checkpoint.get("round_joins", False)
        self.fill_pens: float | None = checkpoint.get("fill_pens")      # renderer.render_strokes; None: the older rendering

        self.arch: str = checkpoint.get("arch", "cnn")
        self.model = build_model(self.arch, len(self.label_names), self.canvas_size, self.use_size_feat)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.prototypes_computed = True
        self.model.to(self.device)
        self.model.eval()

        # Several prototypes per class — one per writing style (scripts/style_prototypes.py):
        # (classes, styles, features) and which of each class's slots are used
        sp = checkpoint.get("style_prototypes")
        self.style_prototypes = sp.to(self.device) if sp is not None else None
        sm = checkpoint.get("style_mask")
        self.style_mask = sm.to(self.device) if sm is not None else None

        self.ood_threshold = ood_threshold
        self.per_class_thresholds = per_class_thresholds or {}

    def _distances(self, features: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        """Each feature vector's distance to its predicted class: to the
        nearest of the class's style prototypes when the checkpoint has
        them, else to its one prototype."""
        if self.style_prototypes is None:
            return torch.norm(features - self.model.prototypes[preds], dim=1)
        protos = self.style_prototypes[preds]                       # (B, K, D)
        d = torch.norm(features.unsqueeze(1) - protos, dim=2)       # (B, K)
        d = d.masked_fill(~self.style_mask[preds], float("inf"))
        return d.min(dim=1).values

    def classify(
        self,
        image: Image.Image,
        ood_threshold: float | None = None,
        size_feat: float | None = None,
    ) -> ClassificationResult:
        """
        Classify a grayscale symbol image.

        Args:
            image: PIL Image (grayscale, canvas_size × canvas_size).
            ood_threshold: Override default OOD threshold.
            size_feat: Relative symbol size (bbox_diag / source_size). Required
                       if model was trained with use_size_feat=True.

        Returns:
            ClassificationResult with symbol, confidence, and OOD info.
        """
        tensor = _transform(image).unsqueeze(0).to(self.device)
        sf = None
        if self.use_size_feat:
            sf = torch.tensor(
                [size_feat if size_feat is not None else 0.5],
                dtype=torch.float32,
                device=self.device,
            )

        with torch.no_grad():
            logits, features = self.model(tensor, sf)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(1)

            predicted_class = pred.item()
            predicted_symbol = self.label_names[predicted_class]
            confidence = conf.item()

            # Distance to predicted class prototype
            distance = self._distances(features[:1], torch.tensor([predicted_class], device=features.device)).item()

            # Top-N alternatives
            top_n = min(5, probs.shape[1])
            top_confs, top_indices = probs[0].topk(top_n)
            alternatives = [
                (self.label_names[top_indices[j].item()], top_confs[j].item()) for j in range(top_n)
            ]

        # Determine threshold
        if ood_threshold is not None:
            threshold = ood_threshold
        else:
            threshold = self.per_class_thresholds.get(predicted_symbol, self.ood_threshold)

        is_ood = distance > threshold

        return ClassificationResult(
            symbol=None if is_ood else predicted_symbol,
            confidence=confidence,
            prototype_distance=distance,
            is_ood=is_ood,
            alternatives=alternatives,
        )

    def classify_batch(
        self,
        images: list[Image.Image],
        size_feats: list[float] | None = None,
    ) -> list[ClassificationResult]:
        """Classify a batch of images in a single forward pass."""
        tensors = torch.stack([_transform(img) for img in images]).to(self.device)
        sf = None
        if self.use_size_feat:
            if size_feats is not None:
                sf = torch.tensor(size_feats, dtype=torch.float32, device=self.device)
            else:
                sf = torch.full((len(images),), 0.5, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits, features = self.model(tensors, sf)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(1)

            # Batch distance computation
            distances = self._distances(features, preds)  # (B,)

        # Top-N alternatives per sample
        top_n = min(5, probs.shape[1])
        top_confs, top_indices = probs.topk(top_n, dim=1)
        probs_np = probs.cpu().numpy()
        features_np = features.cpu().numpy()

        results = []
        for i in range(len(images)):
            predicted_class = preds[i].item()
            predicted_symbol = self.label_names[predicted_class]
            confidence = confs[i].item()
            distance = distances[i].item()

            threshold = self.per_class_thresholds.get(predicted_symbol, self.ood_threshold)
            is_ood = distance > threshold

            alternatives = [
                (self.label_names[top_indices[i, j].item()], top_confs[i, j].item())
                for j in range(top_n)
            ]

            results.append(
                ClassificationResult(
                    symbol=None if is_ood else predicted_symbol,
                    confidence=confidence,
                    prototype_distance=distance,
                    is_ood=is_ood,
                    alternatives=alternatives,
                    probs=probs_np[i],
                    features=features_np[i],
                )
            )
        return results

    def classify_topn(
        self,
        image: Image.Image,
        n: int = 5,
        ood_threshold: float | None = None,
        size_feat: float | None = None,
    ) -> tuple[ClassificationResult, list[dict]]:
        """
        Classify and return top-N predictions.

        Returns:
            (result, all_predictions) where all_predictions is a sorted list of
            {"symbol": str, "confidence": float} dicts.
        """
        tensor = _transform(image).unsqueeze(0).to(self.device)
        sf = None
        if self.use_size_feat:
            sf = torch.tensor(
                [size_feat if size_feat is not None else 0.5],
                dtype=torch.float32,
                device=self.device,
            )

        with torch.no_grad():
            logits, features = self.model(tensor, sf)
            probs = torch.softmax(logits, dim=1)[0]

            top_conf, top_pred = probs.max(0)
            predicted_class = top_pred.item()
            predicted_symbol = self.label_names[predicted_class]
            confidence = top_conf.item()

            distance = self._distances(features[:1], torch.tensor([predicted_class], device=features.device)).item()

        if ood_threshold is not None:
            threshold = ood_threshold
        else:
            threshold = self.per_class_thresholds.get(predicted_symbol, self.ood_threshold)

        is_ood = distance > threshold

        result = ClassificationResult(
            symbol=None if is_ood else predicted_symbol,
            confidence=confidence,
            prototype_distance=distance,
            is_ood=is_ood,
        )

        all_predictions = sorted(
            [
                {"symbol": self.label_names[i], "confidence": float(probs[i])}
                for i in range(len(self.label_names))
            ],
            key=lambda x: x["confidence"],
            reverse=True,
        )[:n]

        return result, all_predictions
