"""GNN evidence refiner for tree parsing.

Pipeline:
    subsets → evidence → EvidenceGNN → Edmonds → tree
"""

from tree_parser.gnn.model import EvidenceGNN, EvidenceBiasLayer

__all__ = ["EvidenceGNN", "EvidenceBiasLayer"]
