"""Generate GNN evidence data using the mixed_v9 subset model.

Source data: mixed_v7b train/val.

Usage:
    cd math_ocr_v2
    PYTHONPATH=. python3.10 data/runs/gnn/mixed_v9/build_mixed_v9.py
"""

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

from tree_parser.gnn.gen_data import generate, DATA_DIR

SUBSET_RUN = "mixed_v9"
SOURCE = DATA_DIR.parent / "tree_subset" / "mixed_v7b"

generate(SUBSET_RUN, "train", jsonl=SOURCE / "train.jsonl", max_examples=7000, augment=True)
generate(SUBSET_RUN, "val", jsonl=SOURCE / "val.jsonl", max_examples=1400)
