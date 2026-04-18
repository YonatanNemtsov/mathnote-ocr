"""Generate GNN evidence data using the mixed_v10 subset model.

Source data: mixed_v10 train/val.

Usage:
    cd math_ocr_v2
    PYTHONPATH=. python3.10 data/runs/gnn/mixed_v10/build_mixed_v10.py
"""

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

from tree_parser.gnn.gen_data import generate, DATA_DIR

SUBSET_RUN = "mixed_v10"
SOURCE = DATA_DIR.parent / "tree_subset" / "mixed_v10"

generate(SUBSET_RUN, "train", jsonl=SOURCE / "train.jsonl", max_examples=21000, augment=True)
generate(SUBSET_RUN, "val", jsonl=SOURCE / "val.jsonl", max_examples=4000)
