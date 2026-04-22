"""Generate GNN training data — samples expressions and computes evidence.

Saves pre-computed evidence tensors to data/runs/gnn/{subset_run}/{name}.pt
so training can load them instantly without needing the subset model.

Usage:
    python3.10 tree_parser/gnn/gen_data.py --subset-run mixed_v8 --name train_7k --per-version 500
    python3.10 tree_parser/gnn/gen_data.py --subset-run mixed_v8 --name val_1400 --per-version 100
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import torch

from mathnote_ocr.engine.checkpoint import load_checkpoint
from mathnote_ocr.latex_utils.glyphs import _extract_glyphs
from mathnote_ocr.tree_parser.gen_data import latex_to_tree_labels
from mathnote_ocr.tree_parser.gnn.dataset import _compute_evidence_for_example
from mathnote_ocr.tree_parser.subset_model import load_subset_model

log = logging.getLogger(__name__)

# Repo-root data/runs/gnn/. Parents: gnn → tree_parser → mathnote_ocr → src → repo-root.
DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "runs" / "gnn"


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def generate(
    subset_run: str,
    name: str,
    device: torch.device | None = None,
    *,
    jsonl: str | Path | None = None,
    per_version: int = 500,
    max_n: int = 30,
    max_examples: int | None = None,
    augment: bool = False,
) -> Path:
    """Generate GNN evidence data.

    Args:
        subset_run: Name of the subset model checkpoint to use.
        name: Output filename (without .pt extension).
        device: Torch device (auto-detected if None).
        augment: If True, apply expression collapsing augmentation.
        jsonl: Path to JSONL file to load examples from.
               If None, samples from latex generators.
        per_version: Number of examples per generator version (when sampling).
        max_n: Maximum number of symbols per expression.
        max_examples: Limit examples when loading from JSONL.

    Returns:
        Path to the saved .pt file.
    """
    if device is None:
        device = _default_device()
    run_dir = DATA_DIR / subset_run
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"{name}.pt"
    log_path = run_dir / f"{name}.log"

    # Add file handler for this run (line-buffered so log is always up to date)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    fh.stream = open(log_path, "w", buffering=1)
    log.addHandler(fh)

    log.info("Output:     %s", out_path)
    log.info("Subset run: %s", subset_run)
    log.info("Device:     %s", device)
    log.info("Max N:      %d", max_n)

    # Load subset model
    log.info("Loading subset model...")
    t0 = time.time()
    ckpt = load_checkpoint("tree_subset", subset_run, device=device)
    cfg = ckpt["config"]
    symbol_vocab = ckpt["symbol_vocab"]
    max_subset = cfg["max_symbols"]

    subset_model = load_subset_model(ckpt, device=device)
    log.info(
        "  Loaded in %.1fs — max_subset=%d, vocab=%d symbols",
        time.time() - t0,
        max_subset,
        len(symbol_vocab),
    )

    # ── Collect raw examples ─────────────────────────────────────────
    raw = []

    if jsonl:
        jsonl_path = Path(jsonl)
        log.info("Loading examples from %s...", jsonl_path)
        t0 = time.time()
        all_examples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                n = len(ex["symbols"])
                if 2 <= n <= max_n:
                    all_examples.append(ex)
        if max_examples and len(all_examples) > max_examples:
            random.shuffle(all_examples)
            all_examples = all_examples[:max_examples]
        raw = all_examples
        log.info("  %d examples in %.1fs", len(raw), time.time() - t0)
    else:
        log.info("Sampling from generators...")
        from mathnote_ocr.data_gen.latex_sampling import (
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            v8,
            v9,
            v10,
            v11,
            v12,
            v13,
            v14,
            v15,
            v16,
        )
        from mathnote_ocr.data_gen.latex_sampling_v2 import generator as gen

        versions = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, gen]

        for ver in versions:
            ver_name = ver.__name__.split(".")[-1]
            count = 0
            attempts = 0
            while count < per_version and attempts < per_version * 5:
                attempts += 1
                latex = ver.sample()
                glyphs = _extract_glyphs(latex)
                if glyphs is None:
                    continue
                n = len(glyphs)
                if n < 2 or n > max_n:
                    continue
                tree_labels = latex_to_tree_labels(latex, n)
                if tree_labels is None:
                    continue

                raw.append(
                    {
                        "symbols": [{"name": g["name"], "bbox": g["bbox"]} for g in glyphs],
                        "tree": [
                            {"parent": p, "edge_type": e, "order": o} for p, e, o in tree_labels
                        ],
                    }
                )
                count += 1
            log.info("  %s: %d/%d (%d attempts)", ver_name, count, per_version, attempts)

    # ── Compute evidence ─────────────────────────────────────────────
    total = len(raw)
    log.info("Computing evidence for %d examples...", total)
    t0 = time.time()
    examples = []
    failed = 0
    log_interval = max(1, min(500, total // 20))  # every 500 or ~20 updates

    for idx, ex in enumerate(raw):
        result = _compute_evidence_for_example(
            ex["symbols"],
            ex["tree"],
            subset_model,
            symbol_vocab,
            max_subset,
            device,
            augment=augment,
        )
        if result is not None:
            examples.append(result)
        else:
            failed += 1

        done = idx + 1
        if done % log_interval == 0 or done == total:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (total - done) / rate if rate > 0 else 0
            log.info(
                "  %6d/%d  (%5.1f%%)  %.0f ex/s  elapsed %s  eta %s",
                done,
                total,
                100 * done / total,
                rate,
                _fmt_time(elapsed),
                _fmt_time(eta),
            )

    elapsed = time.time() - t0
    log.info("Evidence done: %d ok, %d failed in %s", len(examples), failed, _fmt_time(elapsed))

    # ── Save ─────────────────────────────────────────────────────────
    log.info("Saving to %s...", out_path)
    torch.save(
        {
            "examples": examples,
            "symbol_vocab": symbol_vocab,
            "subset_run": subset_run,
            "per_version": per_version,
            "max_n": max_n,
        },
        out_path,
    )
    log.info("  %d examples, %.1f MB", len(examples), out_path.stat().st_size / 1024 / 1024)
    log.info("Log: %s", log_path)
    log.removeHandler(fh)
    fh.close()
    return out_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Generate GNN training data")
    parser.add_argument("--subset-run", default="dg_all_v2")
    parser.add_argument("--name", required=True, help="Output name (e.g. train_7k)")
    parser.add_argument("--per-version", type=int, default=500)
    parser.add_argument("--max-n", type=int, default=30)
    parser.add_argument(
        "--jsonl", type=str, default=None, help="Load from JSONL file instead of sampling"
    )
    parser.add_argument(
        "--max-examples", type=int, default=None, help="Limit examples when using --jsonl"
    )
    args = parser.parse_args()

    generate(
        subset_run=args.subset_run,
        name=args.name,
        jsonl=args.jsonl,
        per_version=args.per_version,
        max_n=args.max_n,
        max_examples=args.max_examples,
    )
