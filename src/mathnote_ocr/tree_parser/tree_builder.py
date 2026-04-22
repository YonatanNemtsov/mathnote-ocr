"""Build trees using Edmonds' algorithm (maximum spanning arborescence).

Edmonds' algorithm guarantees a valid tree: every symbol has exactly
one parent, no cycles. Works with any parent scores — from subset
model evidence, GNN output, or raw scores.
"""

from __future__ import annotations

import torch

from mathnote_ocr.bbox import BBox
from mathnote_ocr.tree_parser.tree_ops import reorder_siblings
from mathnote_ocr.tree_parser.tree_v2 import ROOT_ID, Edge, Node, Symbol, Tree

# ── Edmonds' algorithm ───────────────────────────────────────────────


def _edmonds(weights: list[list[float]], root: int) -> list[int]:
    """Find maximum spanning arborescence rooted at ``root``.

    Returns parents[i] = parent of node i. parents[root] = -1.
    """
    n = len(weights)
    INF = float("inf")

    best_in = [0] * n
    best_weight = [-INF] * n

    for i in range(n):
        if i == root:
            continue
        for j in range(n):
            if j == i:
                continue
            if weights[i][j] > best_weight[i]:
                best_weight[i] = weights[i][j]
                best_in[i] = j

    visited = [-1] * n
    cycle_id = [-1] * n
    n_cycles = 0

    for i in range(n):
        if i == root:
            continue
        u = i
        while u != root and visited[u] == -1:
            visited[u] = i
            u = best_in[u]
        if u != root and visited[u] == i:
            cid = n_cycles
            n_cycles += 1
            v = u
            while True:
                cycle_id[v] = cid
                v = best_in[v]
                if v == u:
                    break

    if n_cycles == 0:
        parents = list(best_in)
        parents[root] = -1
        return parents

    node_map = [0] * n
    next_id = 0
    cycle_repr = {}
    for i in range(n):
        if cycle_id[i] >= 0:
            if cycle_id[i] not in cycle_repr:
                cycle_repr[cycle_id[i]] = next_id
                next_id += 1
            node_map[i] = cycle_repr[cycle_id[i]]
        else:
            node_map[i] = next_id
            next_id += 1

    n_contracted = next_id
    new_weights = [[-INF] * n_contracted for _ in range(n_contracted)]
    edge_map: dict[tuple[int, int], tuple[int, int]] = {}

    for i in range(n):
        if i == root:
            continue
        ni = node_map[i]
        for j in range(n):
            if j == i:
                continue
            nj = node_map[j]
            if ni == nj:
                continue
            w = weights[i][j]
            if cycle_id[i] >= 0:
                w = w - best_weight[i]
            if w > new_weights[ni][nj]:
                new_weights[ni][nj] = w
                edge_map[(ni, nj)] = (i, j)

    new_root = node_map[root]
    contracted_parents = _edmonds(new_weights, new_root)

    parents = list(best_in)
    parents[root] = -1
    for ni in range(n_contracted):
        if ni == new_root:
            continue
        nj = contracted_parents[ni]
        if nj == -1:
            continue
        if (ni, nj) in edge_map:
            i, j = edge_map[(ni, nj)]
            parents[i] = j

    return parents


# ── Shared helpers ───────────────────────────────────────────────────


def _mask_k_nearest(scores: torch.Tensor, symbols: list[Symbol], k: int) -> torch.Tensor:
    """Mask parent scores to only allow k nearest spatial neighbors + ROOT."""
    N = len(symbols)
    for i in range(N):
        dists = sorted(
            ((j, symbols[i].bbox.center_distance(symbols[j].bbox)) for j in range(N) if j != i),
            key=lambda x: x[1],
        )
        allowed = {d[0] for d in dists[:k]}
        for j in range(N):
            if j != i and j not in allowed:
                scores[i, j] = float("-inf")
    return scores


def _scores_to_tree(
    parent_scores: torch.Tensor,
    edge_scores: torch.Tensor,
    symbols: list[Symbol],
) -> Tree:
    """Run Edmonds' on scores and build Tree. Common to all builders."""
    N = len(symbols)
    root_idx = N
    n_total = N + 1

    weights = [[float("-inf")] * n_total for _ in range(n_total)]
    for i in range(N):
        for j in range(N + 1):
            if j != i:
                weights[i][j] = parent_scores[i, j].item()

    parents = _edmonds(weights, root_idx)

    nodes = []
    for i in range(N):
        p = parents[i]
        if p == root_idx or p == -1:
            nodes.append(Node(symbols[i], ROOT_ID, Edge.ROOT, 0))
        else:
            et = int(edge_scores[i, p].argmax().item())
            nodes.append(Node(symbols[i], p, et, 0))

    return reorder_siblings(Tree(tuple(nodes)))


# ── Tree builders ────────────────────────────────────────────────────


def build_tree_from_scores(
    parent_scores: torch.Tensor,
    edge_type_scores: torch.Tensor,
    symbols: list[Symbol],
    k_neighbors: int | None = None,
) -> Tree:
    """Build tree using Edmonds' algorithm on parent scores."""
    scores = parent_scores[: len(symbols), :].detach().cpu()
    if k_neighbors is not None and k_neighbors < len(symbols) - 1:
        scores = _mask_k_nearest(scores, symbols, k_neighbors)
    return _scores_to_tree(scores, edge_type_scores, symbols)


def build_tree_from_evidence(
    evidence: dict[str, torch.Tensor],
    symbols: list[Symbol],
    cost: str = "propagate",
    k_neighbors: int | None = None,
    use_consensus: bool = True,
    consensus_threshold: float = 0.7,
    consensus_boost: float = 5.0,
) -> Tree:
    """Build tree using Edmonds' algorithm on aggregated evidence."""
    from mathnote_ocr.tree_parser.consensus import boost_consensus_edges
    from mathnote_ocr.tree_parser.costs import COST_STRATEGIES

    N = len(symbols)
    parent_scores = COST_STRATEGIES[cost](evidence, N)

    if use_consensus:
        parent_scores = boost_consensus_edges(
            parent_scores,
            evidence,
            N,
            agreement_threshold=consensus_threshold,
            boost=consensus_boost,
        )

    if k_neighbors is not None and k_neighbors < N - 1:
        parent_scores = _mask_k_nearest(parent_scores, symbols, k_neighbors)

    return _scores_to_tree(parent_scores, evidence["parent_votes"], symbols)


def find_seq_conflicts(
    evidence: dict[str, torch.Tensor],
    tree: Tree,
    seq_threshold: float = 2.0,
    max_subset_size: int = 6,
) -> list[list[int]]:
    """Find SEQ conflicts and return targeted subsets to resolve them."""
    seq_votes = evidence.get("seq_votes")
    if seq_votes is None:
        return []

    N = seq_votes.shape[0]
    seq = seq_votes[:N, :N]

    targeted: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    for i in range(N):
        if i not in tree.nodes:
            continue
        ni = tree[i]
        for j in range(i + 1, N):
            if j not in tree.nodes:
                continue
            nj = tree[j]

            if seq[i, j].item() + seq[j, i].item() < seq_threshold:
                continue
            if ni.parent_id == nj.parent_id and ni.edge_type == nj.edge_type:
                continue

            core: set[int] = {i, j}
            if ni.parent_id != ROOT_ID and ni.parent_id >= 0:
                core.add(ni.parent_id)
            if nj.parent_id != ROOT_ID and nj.parent_id >= 0:
                core.add(nj.parent_id)

            mid = BBox.union_all([tree[k].symbol.bbox for k in core])
            others = sorted(
                (k for k in range(N) if k not in core and k in tree.nodes),
                key=lambda k: tree[k].symbol.bbox.center_distance(mid),
            )
            subset = sorted(core | set(others[: max_subset_size - len(core)]))

            key = tuple(subset)
            if key not in seen:
                seen.add(key)
                targeted.append(list(subset))

    return targeted
