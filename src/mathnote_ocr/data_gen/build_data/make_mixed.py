"""Create mixed dataset: 50% original font + 50% augmented bboxes.

Concatenates data/shared/tree/v*/train.jsonl with data/shared/tree_augmented/v*/train.jsonl
into data/tree_aug_50_pct/train.jsonl (and same for val).
"""

import random
from pathlib import Path

SRC_ORIG = Path("data/shared/tree")
SRC_AUG = Path("data/shared/tree_augmented")
DST = Path("data/tree_aug_50_pct")
RUN = None


def concat(split: str):
    DST.mkdir(parents=True, exist_ok=True)
    lines = []

    # Original data
    for vdir in sorted(SRC_ORIG.iterdir()):
        if not vdir.is_dir():
            continue
        if RUN and vdir.name != RUN:
            continue
        if not RUN and not vdir.name.startswith("v"):
            continue
        f = vdir / f"{split}.jsonl"
        if f.exists():
            lines.extend(f.read_text().splitlines())

    n_orig = len(lines)

    # Augmented data
    for vdir in sorted(SRC_AUG.iterdir()):
        if not vdir.is_dir():
            continue
        if RUN and vdir.name != RUN:
            continue
        if not RUN and not vdir.name.startswith("v"):
            continue
        f = vdir / f"{split}.jsonl"
        if f.exists():
            lines.extend(f.read_text().splitlines())

    n_aug = len(lines) - n_orig

    # Shuffle so original and augmented are interleaved
    random.seed(42)
    random.shuffle(lines)

    out = DST / f"{split}.jsonl"
    out.write_text("\n".join(lines) + "\n")
    print(f"{split}: {n_orig} orig + {n_aug} aug = {len(lines)} total → {out}")


if __name__ == "__main__":
    import sys
    RUN = sys.argv[1] if len(sys.argv) > 1 else None
    concat("train")
    concat("val")
