"""GNN evidence refiner for tree parsing.

Pipeline:
    subsets → evidence → EvidenceGNN → Edmonds → tree
"""

from mathnote_ocr.tree_parser.gnn.model import EvidenceBiasLayer, EvidenceGNN

__all__ = ["EvidenceGNN", "EvidenceBiasLayer"]
