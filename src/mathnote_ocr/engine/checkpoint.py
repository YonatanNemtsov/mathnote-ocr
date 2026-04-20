"""Simple checkpoint save/load.

Layout: weights/{model}/{run}/checkpoint.pth
"""

from pathlib import Path

import torch

from mathnote_ocr import config


def _checkpoint_path(model: str, run: str) -> Path:
    return config.WEIGHTS_DIR / model / run / "checkpoint.pth"


def save_checkpoint(model: str, run: str, state_dict: dict) -> Path:
    """Save checkpoint to weights/{model}/{run}/checkpoint.pth."""
    path = _checkpoint_path(model, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)
    return path


def load_checkpoint(model: str, run: str, device: torch.device | str = "cpu") -> dict:
    """Load checkpoint from weights/{model}/{run}/checkpoint.pth."""
    path = _checkpoint_path(model, run)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)
