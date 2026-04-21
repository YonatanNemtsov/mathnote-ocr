"""Checkpoint save/load.

Two different resolution strategies for save vs. load.

Save layout: {weights_dir}/{model}/{run}/checkpoint.pth
    — weights_dir defaults to the packaged weights dir.

Load lookup order (in order):
    1. If `run` is a path (contains / or ends in .pth), load that file directly.
    2. If weights_dir is given, try {weights_dir}/{model}/{run}/checkpoint.pth.
    3. Fall back to bundled {packaged_weights}/{model}/{run}/checkpoint.pth.

This lets users:
- Call MathOCR() and get everything from the bundle
- Train a custom run and point weights_dir at it (bundled still serves as fallback)
- Point a run name at an arbitrary .pth file
"""

from pathlib import Path

import torch

from mathnote_ocr import config


def _save_path(model: str, run: str, weights_dir: Path | str | None = None) -> Path:
    """Resolve the path to save a checkpoint to."""
    base = Path(weights_dir) if weights_dir is not None else config.WEIGHTS_DIR
    return base / model / run / "checkpoint.pth"


def _resolve_checkpoint(
    model: str, run: str, weights_dir: Path | str | None = None
) -> Path:
    """Resolve a checkpoint for loading. Raises FileNotFoundError with all tried paths."""
    # 1. Explicit path (user pointed at a file)
    if run.endswith(".pth") or "/" in run or "\\" in run:
        p = Path(run)
        if p.exists():
            return p
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    # 2. Custom weights_dir
    tried: list[Path] = []
    if weights_dir is not None:
        custom = Path(weights_dir) / model / run / "checkpoint.pth"
        if custom.exists():
            return custom
        tried.append(custom)

    # 3. Bundled fallback
    bundled = config.WEIGHTS_DIR / model / run / "checkpoint.pth"
    if bundled.exists():
        return bundled
    tried.append(bundled)

    raise FileNotFoundError(
        f"Checkpoint not found for {model}/{run}. Tried:\n  "
        + "\n  ".join(str(p) for p in tried)
    )


# Backwards-compat alias — used by training scripts that want the save path
_checkpoint_path = _save_path


def save_checkpoint(
    model: str,
    run: str,
    state_dict: dict,
    weights_dir: Path | str | None = None,
) -> Path:
    path = _save_path(model, run, weights_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)
    return path


def load_checkpoint(
    model: str,
    run: str,
    device: torch.device | str = "cpu",
    weights_dir: Path | str | None = None,
) -> dict:
    path = _resolve_checkpoint(model, run, weights_dir)
    return torch.load(path, map_location=device, weights_only=False)
