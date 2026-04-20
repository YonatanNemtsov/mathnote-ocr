"""Edge cost strategies for tree building.

Each strategy takes aggregated evidence and returns an (N, N+1) score
matrix suitable for Edmonds' algorithm. The score at [i, j] represents
how strongly symbol i wants j as its parent.

Strategies:
    raw         — parent_votes summed across edge types, no seq signal
    seq_blend   — blend own parent distribution with siblings' in prob space
    propagate   — forward-only SEQ propagation on parent_votes, then sum

All strategies also enrich evidence["parent_votes"] in-place when they
use SEQ information, so downstream consumers (edge type selection,
scoring, conflict detection) benefit too.
"""

from __future__ import annotations

import torch


def _base_scores(evidence: dict[str, torch.Tensor], N: int) -> torch.Tensor:
    """Sum parent_votes across edge types, mask self-loops."""
    scores = evidence["parent_votes"][:N, :].sum(dim=-1).clone()  # (N, N+1)
    for i in range(N):
        scores[i, i] = float("-inf")
    return scores


# ── Strategies ──────────────────────────────────────────────────────


def cost_raw(
    evidence: dict[str, torch.Tensor],
    N: int,
) -> torch.Tensor:
    """Raw parent votes summed across edge types. No seq signal."""
    return _base_scores(evidence, N)


def cost_seq_blend(
    evidence: dict[str, torch.Tensor],
    N: int,
    alpha: float = 0.2,
) -> torch.Tensor:
    """Blend own parent distribution with siblings' in probability space.

    alpha controls the mix: (1-alpha) * own + alpha * siblings'.
    Operates entirely in probability space — scale-invariant.

    Does NOT modify evidence in-place (pure cost function).
    """
    scores = _base_scores(evidence, N)
    seq_votes = evidence.get("seq_votes")
    if seq_votes is None or N <= 1:
        return scores

    seq_sym = seq_votes[:N, :N]                         # (N, N)
    sibling = seq_sym + seq_sym.t()                     # symmetric
    sib_total = sibling.sum(dim=1, keepdim=True).clamp(min=1e-6)
    sib_probs = sibling / sib_total                     # (N, N)
    parent_probs = torch.softmax(scores, dim=-1)        # (N, N+1)
    sib_parent_probs = sib_probs @ parent_probs         # (N, N+1)
    blended = (1 - alpha) * parent_probs + alpha * sib_parent_probs
    return torch.log(blended.clamp(min=1e-10))


def cost_propagate(
    evidence: dict[str, torch.Tensor],
    N: int,
    n_iters: int = 3,
) -> torch.Tensor:
    """Forward-only SEQ propagation on parent_votes, then sum.

    Modifies evidence["parent_votes"] in-place so downstream consumers
    (edge type selection, scoring, conflict detection) also benefit.

    Each iteration: parent_votes += T @ parent_votes, where T[i,j] is
    the normalized probability that j is the previous sibling of i.
    """
    parent_votes = evidence["parent_votes"]     # (N, N+1, E)
    seq_votes = evidence.get("seq_votes")       # (N, N+1) or None

    if seq_votes is not None and N > 1:
        seq_sym = seq_votes[:N, :N]
        total_seq = seq_votes[:N].sum(dim=1, keepdim=True).clamp(min=1e-6)
        T = seq_sym / total_seq                 # (N, N) row-normalized

        for _ in range(n_iters):
            propagated = torch.einsum("ik,kje->ije", T, parent_votes[:N])
            parent_votes[:N] += propagated

        evidence["parent_votes"] = parent_votes

    return _base_scores(evidence, N)


def cost_propagate_normalized(
    evidence: dict[str, torch.Tensor],
    N: int,
    n_iters: int = 3,
) -> torch.Tensor:
    """Propagate, then normalize by pair cooccurrence count.

    Symbols that rarely co-occur in subsets get few votes, unfairly
    losing to ROOT which always accumulates. Dividing by cooccurrence
    count gives a "votes per opportunity" measure, making distant
    parent-child relationships competitive with ROOT.
    """
    # Propagate first (enriches evidence)
    parent_votes = evidence["parent_votes"]     # (N, N+1, E)
    seq_votes = evidence.get("seq_votes")

    if seq_votes is not None and N > 1:
        seq_sym = seq_votes[:N, :N]
        total_seq = seq_votes[:N].sum(dim=1, keepdim=True).clamp(min=1e-6)
        T = seq_sym / total_seq

        for _ in range(n_iters):
            propagated = torch.einsum("ik,kje->ije", T, parent_votes[:N])
            parent_votes[:N] += propagated

        evidence["parent_votes"] = parent_votes

    # Sum across edge types
    scores = parent_votes[:N, :].sum(dim=-1).clone()  # (N, N+1)

    # Normalize non-ROOT columns by cooccurrence
    cooc = evidence["pair_cooccurrence"]  # (N, N)
    for i in range(N):
        scores[i, i] = float("-inf")
        for j in range(N):
            if j == i:
                continue
            c = cooc[i, j].item()
            if c > 0:
                scores[i, j] = scores[i, j] / c

    # ROOT column: normalize by total subsets seen
    # (each subset contributes one ROOT vote opportunity)
    for i in range(N):
        total_cooc = cooc[i].sum().item()
        n_subsets = total_cooc / max(N - 1, 1)  # approximate
        if n_subsets > 0:
            scores[i, N] = scores[i, N] / n_subsets

    return scores


def cost_propagate_and_blend(
    evidence: dict[str, torch.Tensor],
    N: int,
    n_iters: int = 3,
    alpha: float = 0.2,
) -> torch.Tensor:
    """Propagate first (enriches evidence), then blend in probability space.

    Combines both: propagation enriches parent_votes for downstream,
    then the probability-space blend provides a cleaner cost for Edmonds'.
    """
    # Propagate (modifies evidence in-place)
    cost_propagate(evidence, N, n_iters=n_iters)
    # Blend on the enriched scores
    return cost_seq_blend(evidence, N, alpha=alpha)


# ── Score post-processing (used by GNN path) ──────────────────────


def anchor_with_evidence(
    parent_scores: torch.Tensor,
    evidence: dict[str, torch.Tensor],
    N: int,
) -> torch.Tensor:
    """Add evidence parent_votes as residual to model-produced scores."""
    ev_parent = evidence["parent_votes"][:N].sum(dim=-1).to(parent_scores.device)
    for i in range(N):
        ev_parent[i, i] = float("-inf")
    return parent_scores + ev_parent


def apply_seq_bonus(
    parent_scores: torch.Tensor,
    seq_scores: torch.Tensor,
    N: int,
) -> torch.Tensor:
    """Sibling bonus in probability space using model seq_scores."""
    seq_probs = torch.softmax(seq_scores[:N], dim=-1)[:, :N]
    parent_probs = torch.softmax(parent_scores[:N], dim=-1)
    return parent_scores + seq_probs @ parent_probs


# ── Registry ────────────────────────────────────────────────────────

COST_STRATEGIES = {
    "raw": cost_raw,
    "seq_blend": cost_seq_blend,
    "propagate": cost_propagate,
    "propagate_normalized": cost_propagate_normalized,
    "propagate_and_blend": cost_propagate_and_blend,
}
