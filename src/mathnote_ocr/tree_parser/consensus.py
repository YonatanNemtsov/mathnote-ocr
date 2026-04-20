"""Subtree consensus detection from evidence.

Before running Edmonds' algorithm, identifies high-confidence local
structures that are consistent across subsets and boosts their scores.

The key insight: if "x is SUP child of 2" appears in 15/20 subsets
with high confidence, that edge is almost certainly correct regardless
of what parent the "2" itself has. By detecting and boosting these
consensus edges, we make the tree builder more robust.
"""

from __future__ import annotations

import torch

from mathnote_ocr.tree_parser.tree import NUM_EDGE_TYPES


def find_consensus_edges(
    evidence: dict[str, torch.Tensor],
    n_symbols: int,
    agreement_threshold: float = 0.7,
    min_votes: float = 2.0,
) -> list[dict]:
    """Find edges with strong consensus across subsets.

    Args:
        evidence: Aggregated evidence from subset model.
        n_symbols: Number of symbols.
        agreement_threshold: Minimum ratio of best (parent, edge) votes
            to total votes for a symbol. Range [0, 1].
        min_votes: Minimum total vote weight to consider (filters out
            symbols seen in very few subsets).

    Returns:
        List of consensus edges:
        [{"child": int, "parent": int, "edge_type": int,
          "agreement": float, "votes": float}, ...]
    """
    N = n_symbols
    parent_votes = evidence["parent_votes"]  # (N, N+1, E)

    consensus = []
    for i in range(N):
        # Total positive votes for this symbol across all (parent, edge) combos
        votes = parent_votes[i]  # (N+1, E)
        positive_votes = votes.clamp(min=0)
        total = positive_votes.sum().item()

        if total < min_votes:
            continue

        # Best (parent, edge_type) combo
        flat_idx = positive_votes.argmax().item()
        best_parent = flat_idx // NUM_EDGE_TYPES
        best_edge = flat_idx % NUM_EDGE_TYPES
        best_votes = positive_votes[best_parent, best_edge].item()

        agreement = best_votes / total

        if agreement >= agreement_threshold:
            # Map ROOT column
            parent = best_parent if best_parent < N else -1
            consensus.append({
                "child": i,
                "parent": parent,
                "edge_type": best_edge if parent >= 0 else -1,
                "agreement": round(agreement, 3),
                "votes": round(best_votes, 2),
            })

    return consensus


def boost_consensus_edges(
    parent_scores: torch.Tensor,
    evidence: dict[str, torch.Tensor],
    n_symbols: int,
    agreement_threshold: float = 0.7,
    min_votes: float = 2.0,
    boost: float = 5.0,
) -> torch.Tensor:
    """Boost parent_scores for high-consensus edges.

    Modifies scores in-place and returns them.

    Args:
        parent_scores: (N, N+1) cost matrix for Edmonds'.
        evidence: Aggregated evidence.
        n_symbols: Number of symbols.
        agreement_threshold: Min agreement ratio to boost.
        min_votes: Min total votes to consider.
        boost: Additive boost to consensus edges.

    Returns:
        Modified parent_scores tensor.
    """
    consensus = find_consensus_edges(
        evidence, n_symbols, agreement_threshold, min_votes,
    )

    N = n_symbols
    for edge in consensus:
        child = edge["child"]
        parent_col = edge["parent"] if edge["parent"] >= 0 else N
        parent_scores[child, parent_col] += boost

    return parent_scores
