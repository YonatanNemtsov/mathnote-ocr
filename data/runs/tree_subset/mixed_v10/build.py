#!/usr/bin/env python3.10
"""mixed_v10: build training data with improved generators.

Changes from mixed_v9:
- Generator v1-v16 cleanup: spaces between all tokens, no f-strings,
  structured content, _pick_base for sup/sub bases
- gen3: tree-first generator using composable templates
- 50/50 mix of dg_all (v1-v16) and gen3
- clean_latex round-trip normalization
- \\left(\\right) for matched parens

Final dataset:
  Train: ~87k (42k raw + 42k augmented + 3k handwritten [285 x ~10])
  Val:   ~8k  (4k raw + 4k augmented)
"""

import json
import os
import random
import sys
from pathlib import Path

# Ensure project root is on path
# data/runs/tree_subset/mixed_v10/build.py -> math_ocr_v2/
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT)

DIR = Path("data/runs/tree_subset/mixed_v10")
HW_DIR = Path("data/shared/tree_handwritten")

TRAIN_N = 42000
VAL_N = 4000
MAX_SYMBOLS = 30
SEED = 42
SAMPLER = "dg_all_gen3"
HW_REPEAT_TARGET = 3000


def step1_generate():
    """Generate raw LaTeX data."""
    if (DIR / "raw_train.jsonl").exists():
        print("Step 1: Raw data exists, skipping.")
        return

    print(f"Step 1: Generating {TRAIN_N} train + {VAL_N} val (sampler={SAMPLER}, max_symbols={MAX_SYMBOLS})...")
    from tree_parser.gen_data import generate_dataset

    DIR.mkdir(parents=True, exist_ok=True)
    generate_dataset(TRAIN_N, DIR / "raw_train.jsonl",
                     seed=SEED, max_symbols=MAX_SYMBOLS, sampler_name=SAMPLER)
    generate_dataset(VAL_N, DIR / "raw_val.jsonl",
                     seed=SEED + 1000, max_symbols=MAX_SYMBOLS, sampler_name=SAMPLER)


def step2_augment():
    """Augment with handwritten-like bboxes."""
    if (DIR / "aug_train.jsonl").exists():
        print("Step 2: Augmented data exists, skipping.")
        return

    print("Step 2: Augmenting bboxes...")
    from tree_parser.hw_bbox_augment import augment_bboxes

    for src_name, dst_name in [("raw_train.jsonl", "aug_train.jsonl"),
                                ("raw_val.jsonl", "aug_val.jsonl")]:
        total = ok = 0
        with open(DIR / src_name) as fin, open(DIR / dst_name, "w") as fout:
            for line in fin:
                total += 1
                item = json.loads(line)
                aug = augment_bboxes(item["symbols"], item["tree"])
                if aug is None:
                    fout.write(line)
                else:
                    item["symbols"] = aug
                    fout.write(json.dumps(item) + "\n")
                    ok += 1
                if total % 10000 == 0:
                    print(f"  {total}...")
        print(f"  {src_name}: {ok}/{total} augmented")


def step3_mix():
    """Mix raw + augmented + handwritten."""
    print("Step 3: Mixing...")
    random.seed(SEED)

    # Train: raw + augmented
    train = []
    for f in [DIR / "raw_train.jsonl", DIR / "aug_train.jsonl"]:
        with open(f) as fh:
            train.extend(fh.readlines())

    # Add handwritten (repeated to ~3000)
    hw_lines = []
    for run_dir in sorted(HW_DIR.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "broken_default":
            continue
        f = run_dir / "train.jsonl"
        if f.exists():
            with open(f) as fh:
                lines = fh.readlines()
                hw_lines.extend(lines)
                print(f"  {run_dir.name}: {len(lines)} handwritten")

    repeats = HW_REPEAT_TARGET // len(hw_lines) + 1
    hw_repeated = (hw_lines * repeats)[:HW_REPEAT_TARGET]
    random.shuffle(hw_repeated)
    train.extend(hw_repeated)
    print(f"  Handwritten: {len(hw_lines)} x ~{repeats} -> {len(hw_repeated)}")

    random.shuffle(train)
    with open(DIR / "train.jsonl", "w") as f:
        f.writelines(train)
    print(f"  Train: {len(train)}")

    # Val (no handwritten)
    val = []
    for f in [DIR / "raw_val.jsonl", DIR / "aug_val.jsonl"]:
        with open(f) as fh:
            val.extend(fh.readlines())
    random.shuffle(val)
    with open(DIR / "val.jsonl", "w") as f:
        f.writelines(val)
    print(f"  Val: {len(val)}")


if __name__ == "__main__":
    step1_generate()
    step2_augment()
    step3_mix()

    print()
    print("Done.")
    print()
    print("Train with:")
    print(f"  PYTHONPATH=. python3.10 tree_parser/subset_train.py --run mixed_v10 \\")
    print(f"    --train {DIR}/train.jsonl \\")
    print(f"    --val {DIR}/val.jsonl \\")
    print(f"    --epochs 200")
