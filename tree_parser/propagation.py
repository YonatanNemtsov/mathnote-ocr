"""SEQ propagation strategies for spreading parent evidence along sibling chains.

All strategies operate on an evidence dict with at least:
  - parent_votes: (N, N+1, E) float
  - seq_votes: (N, N+1) float (optional)

They modify parent_votes in-place.

Production:
  propagate_seq — forward-only additive propagation (default)

Experimental:
  propagate_none — no-op baseline
  propagate_old_symmetric — bidirectional (seq + seq.t())
  propagate_forward — alias for propagate_seq
  propagate_bidir — forward pass then backward pass
  propagate_forward_decayed — forward with geometric decay
  propagate_decayed — symmetric with geometric decay
  propagate_pool — pool sibling components via union-find
  propagate_pool_damped — pool + damping factor
  normalize_scores — post-propagation normalization
"""

from __future__ import annotations

from collections import defaultdict

import torch


# ── Production ───────────────────────────────────────────────────────


def propagate_seq(evidence: dict[str, torch.Tensor], n_iters: int = 3) -> None:
    """Propagate parent votes forward along SEQ (sibling) chains.

    seq_votes[i, j] means "j is the previous sibling of i", so votes
    flow from j -> i (forward along the chain: a -> b -> c -> d).

    Only propagates forward -- a symbol with strong parent evidence
    pushes that evidence to its *next* siblings, but weak/wrong
    evidence from later siblings never flows backward.

    Modifies parent_votes in-place.
    """
    parent_votes = evidence["parent_votes"]     # (N, N+1, E)
    seq_votes = evidence.get("seq_votes")       # (N, N+1) or None
    if seq_votes is None:
        return

    N = parent_votes.shape[0]

    # T[i, j] = confidence that j is prev sibling of i
    # -> j's parent votes flow to i (forward direction only)
    seq_sym = seq_votes[:, :N]                  # (N, N)
    total_seq = seq_votes.sum(dim=1, keepdim=True).clamp(min=1e-6)
    T = seq_sym / total_seq                     # (N, N) row-normalized, directed

    for _ in range(n_iters):
        propagated = torch.einsum("ik,kje->ije", T, parent_votes)
        parent_votes += propagated

    evidence["parent_votes"] = parent_votes


# ── Experimental strategies ──────────────────────────────────────────


def propagate_none(evidence: dict[str, torch.Tensor]) -> None:
    """No propagation (baseline)."""
    pass


propagate_forward = propagate_seq  # alias


def propagate_old_symmetric(
    evidence: dict[str, torch.Tensor], n_iters: int = 3,
) -> None:
    """Old symmetric propagation (before the forward-only change).

    Uses seq_sym + seq_sym.t() so votes flow in both directions.
    """
    parent_votes = evidence["parent_votes"]
    seq_votes = evidence.get("seq_votes")
    if seq_votes is None:
        return

    N = parent_votes.shape[0]
    seq_sym = seq_votes[:, :N]
    sibling = seq_sym + seq_sym.t()  # symmetric
    total_seq = seq_votes.sum(dim=1, keepdim=True).clamp(min=1e-6)
    S = sibling / total_seq

    for _ in range(n_iters):
        propagated = torch.einsum("ik,kje->ije", S, parent_votes)
        parent_votes += propagated

    evidence["parent_votes"] = parent_votes


def propagate_bidir(
    evidence: dict[str, torch.Tensor], n_fwd: int = 3, n_bwd: int = 1,
) -> None:
    """Forward pass then backward pass -- separate, no mixing.

    Forward: seq_votes[i,j] = j is prev of i -> j's votes flow to i
    Backward: seq_votes[i,j].t() = i is prev of j -> j's votes flow to i
    """
    parent_votes = evidence["parent_votes"]
    seq_votes = evidence.get("seq_votes")
    if seq_votes is None:
        return

    N = parent_votes.shape[0]
    seq_sym = seq_votes[:, :N]
    total = seq_votes.sum(dim=1, keepdim=True).clamp(min=1e-6)

    # Forward: T_fwd[i,j] = j is prev of i -> pull j's votes to i
    T_fwd = seq_sym / total

    # Backward: T_bwd[i,j] = j is next of i -> pull j's votes to i
    seq_bwd = seq_sym.t()
    total_bwd = seq_bwd.sum(dim=1, keepdim=True).clamp(min=1e-6)
    T_bwd = seq_bwd / total_bwd

    for _ in range(n_fwd):
        propagated = torch.einsum("ik,kje->ije", T_fwd, parent_votes)
        parent_votes += propagated

    for _ in range(n_bwd):
        propagated = torch.einsum("ik,kje->ije", T_bwd, parent_votes)
        parent_votes += propagated

    evidence["parent_votes"] = parent_votes


def propagate_forward_decayed(
    evidence: dict[str, torch.Tensor], alpha: float = 0.5, n_iters: int = 10,
) -> None:
    """Forward-only with geometric decay.

    Each iteration propagates only what was received LAST round, scaled
    by alpha. Hop k contributes alpha^k * original_votes.
    """
    parent_votes = evidence["parent_votes"]
    seq_votes = evidence.get("seq_votes")
    if seq_votes is None:
        return

    N = parent_votes.shape[0]
    seq_sym = seq_votes[:, :N]
    total = seq_votes.sum(dim=1, keepdim=True).clamp(min=1e-6)
    T = seq_sym / total

    received = parent_votes.clone()
    for _ in range(n_iters):
        received = alpha * torch.einsum("ik,kje->ije", T, received)
        parent_votes += received

    evidence["parent_votes"] = parent_votes


def propagate_decayed(
    evidence: dict[str, torch.Tensor], alpha: float = 0.5, n_iters: int = 10,
) -> None:
    """Symmetric propagation with geometric decay.

    Uses seq_sym + seq_sym.t() but each hop decays by alpha^k.
    No blowup since alpha^k -> 0.
    """
    parent_votes = evidence["parent_votes"]
    seq_votes = evidence.get("seq_votes")
    if seq_votes is None:
        return

    N = parent_votes.shape[0]
    seq_sym = seq_votes[:, :N]
    sibling = seq_sym + seq_sym.t()
    total = sibling.sum(dim=1, keepdim=True).clamp(min=1e-6)
    T = sibling / total

    received = parent_votes.clone()
    for _ in range(n_iters):
        received = alpha * torch.einsum("ik,kje->ije", T, received)
        parent_votes += received

    evidence["parent_votes"] = parent_votes


def propagate_pool(
    evidence: dict[str, torch.Tensor], threshold: float = 1.0,
) -> None:
    """Find sibling components from SEQ votes, pool parent votes across each.

    Uses union-find on symmetric sibling affinity to find connected
    components. All members of a component share the summed votes.
    """
    parent_votes = evidence["parent_votes"]
    seq_votes = evidence.get("seq_votes")
    if seq_votes is None:
        return

    N = parent_votes.shape[0]
    seq_sym = seq_votes[:, :N]
    sibling = seq_sym + seq_sym.t()

    # Union-find
    parent_uf = list(range(N))

    def find(x):
        while parent_uf[x] != x:
            parent_uf[x] = parent_uf[parent_uf[x]]
            x = parent_uf[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent_uf[a] = b

    for i in range(N):
        for j in range(i + 1, N):
            if sibling[i, j].item() >= threshold:
                union(i, j)

    components = defaultdict(list)
    for i in range(N):
        components[find(i)].append(i)

    for root, members in components.items():
        if len(members) <= 1:
            continue
        pooled = parent_votes[members].sum(dim=0)
        for m in members:
            parent_votes[m] = pooled

    evidence["parent_votes"] = parent_votes


def propagate_pool_damped(
    evidence: dict[str, torch.Tensor],
    threshold: float = 1.0,
    alpha: float = 0.5,
) -> None:
    """Pool + damp: add alpha * (pooled - original) to each member.

    Gentler than full pooling -- each symbol keeps (1-alpha) of its own
    votes and blends in alpha of the component's pooled votes.
    """
    parent_votes = evidence["parent_votes"]
    seq_votes = evidence.get("seq_votes")
    if seq_votes is None:
        return

    N = parent_votes.shape[0]
    seq_sym = seq_votes[:, :N]
    sibling = seq_sym + seq_sym.t()

    parent_uf = list(range(N))

    def find(x):
        while parent_uf[x] != x:
            parent_uf[x] = parent_uf[parent_uf[x]]
            x = parent_uf[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent_uf[a] = b

    for i in range(N):
        for j in range(i + 1, N):
            if sibling[i, j].item() >= threshold:
                union(i, j)

    components = defaultdict(list)
    for i in range(N):
        components[find(i)].append(i)

    original = parent_votes.clone()
    for root, members in components.items():
        if len(members) <= 1:
            continue
        pooled = original[members].sum(dim=0)
        for m in members:
            others_votes = pooled - original[m]
            parent_votes[m] = original[m] + alpha * others_votes

    evidence["parent_votes"] = parent_votes


def normalize_scores(evidence: dict[str, torch.Tensor]) -> None:
    """Normalize parent_votes per symbol to remove inflation bias.

    After propagation, some symbols have much higher total vote mass
    than others. This normalizes each symbol's votes so total mass = 1,
    preserving the relative distribution but removing scale differences.
    """
    parent_votes = evidence["parent_votes"]  # (N, N+1, E)
    total_per_symbol = parent_votes.sum(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    evidence["parent_votes"] = parent_votes / total_per_symbol
