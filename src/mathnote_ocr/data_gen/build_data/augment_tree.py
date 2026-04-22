"""Augment tree data with handwritten-like bboxes.

Reads data/shared/tree/{run}/train.jsonl and val.jsonl,
applies hw_bbox_augment.augment_bboxes to each example,
saves to data/shared/tree_augmented/{tag}/train.jsonl and val.jsonl.

Supports --sup-dy / --sub-dy to control sup/sub vertical offsets,
allowing multiple augmentation variants (tight/mid/wide).
"""

import json
import time
from pathlib import Path

from mathnote_ocr.tree_parser.hw_bbox_augment import augment_bboxes

SRC = Path("data/shared/tree")
DST = Path("data/shared/tree_augmented")


def process_file(src_path: Path, dst_path: Path,
                 sup_dy: float, sup_dy_std: float,
                 sub_dy: float, sub_dy_std: float) -> tuple[int, int]:
    """Augment one jsonl file. Returns (total, success)."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    ok = 0
    with open(src_path) as fin, open(dst_path, "w") as fout:
        for line in fin:
            total += 1
            try:
                item = json.loads(line)
            except Exception:
                continue
            symbols = item["symbols"]
            tree = item["tree"]
            aug = augment_bboxes(symbols, tree,
                                 sup_dy=sup_dy, sup_dy_std=sup_dy_std,
                                 sub_dy=sub_dy, sub_dy_std=sub_dy_std)
            if aug is None:
                fout.write(line)
            else:
                item["symbols"] = aug
                fout.write(json.dumps(item) + "\n")
                ok += 1
    return total, ok


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True,
                        help="Source run to augment (e.g. mixed_dg)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Output tag (default: same as run)")
    parser.add_argument("--sup-dy", type=float, default=-0.910)
    parser.add_argument("--sup-dy-std", type=float, default=0.300)
    parser.add_argument("--sub-dy", type=float, default=0.665)
    parser.add_argument("--sub-dy-std", type=float, default=0.250)
    args = parser.parse_args()

    src_dir = SRC / args.run
    if not src_dir.exists():
        print(f"Run directory not found: {src_dir}")
        return

    tag = args.tag or args.run
    print(f"Augmenting {args.run} -> {tag}")
    print(f"  SUP dy={args.sup_dy} std={args.sup_dy_std}")
    print(f"  SUB dy={args.sub_dy} std={args.sub_dy_std}")

    t0 = time.time()
    for split in ("train", "val"):
        src = src_dir / f"{split}.jsonl"
        if not src.exists():
            continue
        dst = DST / tag / f"{split}.jsonl"
        total, ok = process_file(src, dst,
                                 sup_dy=args.sup_dy, sup_dy_std=args.sup_dy_std,
                                 sub_dy=args.sub_dy, sub_dy_std=args.sub_dy_std)
        elapsed = time.time() - t0
        print(f"  {split}: {ok}/{total} augmented ({elapsed:.1f}s)")

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"Output: {DST / tag}")


if __name__ == "__main__":
    main()
