"""Aggregate partial tree predictions into an evidence graph.

Takes many partial trees (each from a different symbol subset) and
builds a dense evidence tensor summarizing all parent-child votes.
This evidence graph is then fed to the GNN for refinement.
"""

from __future__ import annotations

import torch

from tree_parser.tree import NUM_EDGE_TYPES, ROOT

# Re-exports for backward compatibility — these now live in dedicated modules
from tree_parser.propagation import propagate_seq  # noqa: F401
from tree_parser.subset_selection import (  # noqa: F401
    _bbox_centers,
    _bbox_edge_dist,
    _bbox_dist,
    sample_subsets_spatial,
    enumerate_subsets_exhaustive,
    sample_subsets_with_coverage,
    make_spatial_subsets,
)


# ── Evidence aggregation ─────────────────────────────────────────────


@torch.no_grad()
def aggregate_evidence(
    n_symbols: int,
    partial_trees: list[tuple[list[int], list[tuple]]],
) -> dict[str, torch.Tensor]:
    """Aggregate partial tree predictions into evidence tensors.

    Args:
        n_symbols: Total number of symbols.
        partial_trees: List of (subset_indices, predictions) where
            predictions is a list of (parent_idx, edge_type, order)
            or (parent_idx, edge_type, order, seq_prev) per symbol
            in the subset. parent_idx is relative to the subset
            (-1 for root).

    Returns dict with:
        parent_votes: (N, N+1, E) float — vote counts for each
            (child, parent, edge_type) triple. Last parent index = root.
        order_sum: (N, N+1) float — sum of predicted orders per
            (child, parent) pair.
        order_count: (N, N+1) float — count for averaging order.
        pair_cooccurrence: (N, N) float — how many subsets each pair
            appeared in together.
        seq_votes: (N, N+1) float — votes for previous sibling.
            seq_votes[i, j] = times j was predicted as prev sibling of i.
            seq_votes[i, N] = times i was predicted as having no prev sibling.
    """
    N = n_symbols
    E = NUM_EDGE_TYPES

    parent_votes = torch.zeros(N, N + 1, E)
    order_sum = torch.zeros(N, N + 1)
    order_count = torch.zeros(N, N + 1)
    pair_cooccurrence = torch.zeros(N, N)
    seq_votes = torch.zeros(N, N + 1)

    for subset_indices, predictions in partial_trees:
        # Map subset-local indices to global indices
        for local_i, pred in enumerate(predictions):
            local_parent = pred[0]
            edge_type = pred[1]
            order = pred[2]
            seq_prev = pred[3] if len(pred) > 3 else None

            global_child = subset_indices[local_i]

            if local_parent == ROOT:
                global_parent = N  # root column
            else:
                global_parent = subset_indices[local_parent]

            order_sum[global_child, global_parent] += order
            order_count[global_child, global_parent] += 1.0

            if edge_type >= 0:
                et = min(edge_type, E - 1)
                parent_votes[global_child, global_parent, et] += 1.0

            # SEQ votes
            if seq_prev is not None:
                if seq_prev == ROOT:
                    seq_votes[global_child, N] += 1.0
                else:
                    global_seq = subset_indices[seq_prev]
                    seq_votes[global_child, global_seq] += 1.0

        # Track co-occurrence
        for i in range(len(subset_indices)):
            for j in range(len(subset_indices)):
                if i != j:
                    pair_cooccurrence[subset_indices[i], subset_indices[j]] += 1.0

    return {
        "parent_votes": parent_votes,
        "order_sum": order_sum,
        "order_count": order_count,
        "pair_cooccurrence": pair_cooccurrence,
        "seq_votes": seq_votes,
    }


@torch.no_grad()
def aggregate_evidence_soft(
    n_symbols: int,
    partial_outputs: list[tuple[list[int], dict[str, torch.Tensor], int]],
) -> dict[str, torch.Tensor]:
    """Aggregate using confidence-weighted hard votes.

    Same as aggregate_evidence (argmax -> vote for the winner), but each
    vote is weighted by the softmax confidence of the prediction.
    A 95%-confident prediction adds 0.95 to the winner; a 30% guess
    adds only 0.30. Wrong candidates still get exactly 0.

    Args:
        n_symbols: Total number of symbols.
        partial_outputs: List of (subset_indices, model_output, n_real).
            model_output has parent_scores, edge_type_scores,
            order_preds, seq_scores — batch dim already squeezed.

    Returns same dict format as aggregate_evidence.
    """
    import torch.nn.functional as F

    N = n_symbols
    E = NUM_EDGE_TYPES

    parent_votes = torch.zeros(N, N + 1, E)
    order_sum = torch.zeros(N, N + 1)
    order_count = torch.zeros(N, N + 1)
    pair_cooccurrence = torch.zeros(N, N)
    seq_votes = torch.zeros(N, N + 1)

    for subset_indices, out, n_sub in partial_outputs:
        S = out["parent_scores"].shape[0]  # padded subset size

        parent_probs = F.softmax(out["parent_scores"][:n_sub], dim=-1)   # (n_sub, S+1)
        seq_probs = F.softmax(out["seq_scores"][:n_sub], dim=-1)         # (n_sub, S+1)
        order_preds = out["order_preds"][:n_sub]                         # (n_sub, S+1)

        for i in range(n_sub):
            gi = subset_indices[i]

            # Parent: argmax, weighted by confidence
            pi = out["parent_scores"][i].argmax().item()
            confidence = parent_probs[i, pi].item()

            if pi == S:
                global_parent = N  # root
            elif pi < n_sub:
                global_parent = subset_indices[pi]
            else:
                continue

            # Edge type: argmax for the chosen parent
            et = out["edge_type_scores"][i, pi].argmax().item()
            et = min(et, E - 1)

            order = order_preds[i, pi].item()

            order_sum[gi, global_parent] += confidence * order
            order_count[gi, global_parent] += confidence

            if pi != S:  # non-root: add edge type vote
                parent_votes[gi, global_parent, et] += confidence
            else:
                # ROOT: positive vote for ROOT column
                parent_votes[gi, N, 0] += confidence
                # Also negative evidence: "nobody in this subset is my parent"
                for j in range(n_sub):
                    if j != i:
                        gj = subset_indices[j]
                        parent_votes[gi, gj, 0] -= confidence / (n_sub - 1)

            # SEQ: argmax, weighted by confidence
            si = out["seq_scores"][i].argmax().item()
            seq_conf = seq_probs[i, si].item()
            if si == S:
                seq_votes[gi, N] += seq_conf
            elif si < n_sub:
                seq_votes[gi, subset_indices[si]] += seq_conf

        # Co-occurrence
        for i in range(len(subset_indices)):
            for j in range(len(subset_indices)):
                if i != j:
                    pair_cooccurrence[subset_indices[i], subset_indices[j]] += 1.0

    return {
        "parent_votes": parent_votes,
        "order_sum": order_sum,
        "order_count": order_count,
        "pair_cooccurrence": pair_cooccurrence,
        "seq_votes": seq_votes,
    }


# ── Test-time augmentation ────────────────────────────────────────────


def jitter_bboxes(
    bboxes: list[list[float]],
    dx_scale: float = 0.05,
    dy_scale: float = 0.15,
    size_scale: float = 0.05,
) -> list[list[float]]:
    """Add gaussian jitter to bounding boxes for TTA."""
    import random
    result = []
    for x, y, w, h in bboxes:
        ref = max(h, 5)
        dx = random.gauss(0, dx_scale * ref)
        dy = random.gauss(0, dy_scale * ref)
        sw = random.gauss(1, size_scale) if size_scale > 0 else 1.0
        sh = random.gauss(1, size_scale) if size_scale > 0 else 1.0
        new_w = max(0.001, w * sw)
        new_h = max(0.001, h * sh)
        cx = x + w / 2 + dx
        cy = y + h / 2 + dy
        result.append([max(0, cx - new_w / 2), max(0, cy - new_h / 2), new_w, new_h])
    return result


def get_evidence_tta(
    names: list[str],
    bboxes: list[list[float]],
    run_subsets_fn,
    make_subsets_fn,
    *,
    tta_runs: int = 1,
    tta_dx: float = 0.05,
    tta_dy: float = 0.15,
    tta_size: float = 0.05,
):
    """Run subset model with optional TTA, return aggregated evidence.

    Run 0 uses original bboxes. Runs 1..n add gaussian jitter.
    """
    all_partial = []
    for i in range(tta_runs):
        bb = bboxes if i == 0 else jitter_bboxes(bboxes, tta_dx, tta_dy, tta_size)
        subsets = make_subsets_fn(bb)
        partial = run_subsets_fn(names, bb, subsets)
        all_partial.extend(partial)
    return aggregate_evidence_soft(len(names), all_partial)


# ── Evidence to features ─────────────────────────────────────────────


def evidence_to_features(
    evidence: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw evidence into feature tensors for the GNN.

    Returns:
        node_features: (N, F_node) — per-symbol features (currently
            just a placeholder; the GNN will add symbol embeddings).
        edge_features: (N, N+1, F_edge) — per-pair features encoding
            the aggregated evidence.
    """
    parent_votes = evidence["parent_votes"]     # (N, N+1, E)
    order_sum = evidence["order_sum"]           # (N, N+1)
    order_count = evidence["order_count"]       # (N, N+1)
    pair_cooccurrence = evidence["pair_cooccurrence"]  # (N, N)
    seq_votes = evidence.get("seq_votes")       # (N, N+1) or None

    N = parent_votes.shape[0]

    # Normalize votes by co-occurrence (what fraction of times they
    # appeared together did the model predict this relationship?)
    # Add root co-occurrence column (total subsets this child appeared in)
    root_cooccurrence = pair_cooccurrence.sum(dim=1, keepdim=True) / max(
        1, N - 1
    )  # approximate
    full_cooccurrence = torch.cat(
        [pair_cooccurrence, root_cooccurrence], dim=1,
    )  # (N, N+1)

    # Vote fractions per edge type
    total_votes = parent_votes.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    vote_fractions = parent_votes / total_votes  # (N, N+1, E)

    # Total vote count (normalized by co-occurrence)
    vote_density = total_votes.squeeze(-1) / full_cooccurrence.clamp(min=1e-6)
    vote_density = vote_density.unsqueeze(-1)  # (N, N+1, 1)

    # Average order
    avg_order = order_sum / order_count.clamp(min=1e-6)
    avg_order = avg_order.unsqueeze(-1)  # (N, N+1, 1)

    # SEQ density: seq_votes normalized by co-occurrence
    parts = [vote_fractions, vote_density, avg_order]
    if seq_votes is not None:
        seq_density = seq_votes / full_cooccurrence.clamp(min=1e-6)
        parts.append(seq_density.unsqueeze(-1))  # (N, N+1, 1)

    # Concatenate: E vote fractions + 1 vote density + 1 avg order [+ 1 seq density]
    edge_features = torch.cat(parts, dim=-1)

    # Node features: how many times this symbol appeared in subsets
    appearances = pair_cooccurrence.sum(dim=1) / max(1, N - 1)
    node_features = appearances.unsqueeze(-1)  # (N, 1)

    return node_features, edge_features
