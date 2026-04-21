"""Checkpoint save/load.

Layout: {weights_dir}/{model}/{run}/checkpoint.pth

By default, weights_dir is the packaged weights directory. Pass an explicit
weights_dir to use a different location (e.g. user-trained weights).
"""

from pathlib import Path

import torch

from mathnote_ocr import config


def _checkpoint_path(model: str, run: str, weights_dir: Path | str | None = None) -> Path:
    base = Path(weights_dir) if weights_dir is not None else config.WEIGHTS_DIR
    return base / model / run / "checkpoint.pth"


def save_checkpoint(
    model: str,
    run: str,
    state_dict: dict,
    weights_dir: Path | str | None = None,
) -> Path:
    path = _checkpoint_path(model, run, weights_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)
    return path


def load_checkpoint(
    model: str,
    run: str,
    device: torch.device | str = "cpu",
    weights_dir: Path | str | None = None,
) -> dict:
    path = _checkpoint_path(model, run, weights_dir)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)
