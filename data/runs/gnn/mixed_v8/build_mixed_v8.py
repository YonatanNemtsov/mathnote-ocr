"""Generate GNN evidence data using the mixed_v8 subset model.

Source data: mixed_v7b train/val (same data mixed_v8 was trained on).

Usage:
    cd math_ocr_v2
    PYTHONPATH=. python3.10 data/runs/gnn/mixed_v8/build_mixed_v8.py
"""

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

from mathnote_ocr.tree_parser.gnn.gen_data import generate, DATA_DIR

SUBSET_RUN = "mixed_v8"
SOURCE = DATA_DIR.parent / "tree_subset" / "mixed_v7b"

generate(SUBSET_RUN, "train", jsonl=SOURCE / "train.jsonl", max_examples=7000)
generate(SUBSET_RUN, "val", jsonl=SOURCE / "val.jsonl", max_examples=1400)
