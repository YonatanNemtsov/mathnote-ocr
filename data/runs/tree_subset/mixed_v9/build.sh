#!/usr/bin/env bash
# mixed_v9: same data as mixed_v8, new collapse augmentation strategy
#
# Changes from mixed_v8:
# - collapse_subtrees refactored to latex_utils/collapse.py (deterministic, takes explicit indices)
# - random_collapse: per-symbol 4% prob (was 7% per-group), geometric run lengths
# - Supports single-symbol-with-children collapses (min_total=2 on descendants, not min_len=2 on siblings)
# - Multiple collapse runs per sibling group possible
# - Fixed sibling order renumbering after collapse (was a no-op before)
#
# Data: reuse mixed_v8 train/val (symlinked)

set -euo pipefail
cd "$(dirname "$0")"

ln -sf ../mixed_v8/train.jsonl train.jsonl
ln -sf ../mixed_v8/val.jsonl val.jsonl

echo "Symlinked train/val from mixed_v8"
echo "Train: PYTHONPATH=. python3.10 tree_parser/subset_train.py --run mixed_v9 --train data/runs/tree_subset/mixed_v9/train.jsonl --val data/runs/tree_subset/mixed_v9/val.jsonl --epochs 200"
