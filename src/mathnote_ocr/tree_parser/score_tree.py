"""Tree confidence scoring strategies.

Each scorer takes (evidence, tree, N) and returns a float in [0, 1].
tree is a tree_v2.Tree.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from mathnote_ocr.tree_parser.tree_v2 import Tree, Edge, ROOT_ID


def _parent_decisiveness(evidence: dict, tree: Tree, N: int) -> float:
    """How decisively each symbol's parent was chosen by the subset votes."""
    parent_votes = evidence["parent_votes"].sum(dim=-1)  # (N, N+1)
    pos_votes = parent_votes.clamp(min=0)
    total_votes = pos_votes.sum(dim=-1).clamp(min=1e-6)
    max_votes = pos_votes.max(dim=-1).values
    return (max_votes / total_votes).clamp(max=1.0).mean().item()


def _subset_agreement(evidence: dict, tree: Tree, N: int) -> float:
    """How consistently subsets agree on the chosen parent."""
    parent_votes = evidence["parent_votes"]            # (N, N+1, E)
    pair_cooccurrence = evidence["pair_cooccurrence"]   # (N, N)

    total_per_parent = parent_votes.sum(dim=-1)  # (N, N+1)
    root_cooc = pair_cooccurrence.sum(dim=1) / max(1, N - 1)

    agreement_sum = 0.0
    for i in range(N):
        if i not in tree.nodes:
            continue
        p = tree[i].parent_id
        p_col = p if p != ROOT_ID else N

        chosen_votes = total_per_parent[i, p_col].clamp(min=0).item()
        cooc = root_cooc[i].item() if p == ROOT_ID else pair_cooccurrence[i, p].item()

        if cooc > 0:
            agreement_sum += min(1.0, chosen_votes / cooc)
        else:
            agreement_sum += 0.5

    return agreement_sum / max(N, 1)


def _edge_type_decisiveness(evidence: dict, tree: Tree, N: int) -> float:
    """How decisively each symbol's edge type was chosen."""
    parent_votes = evidence["parent_votes"]  # (N, N+1, E)

    scores = []
    for i in range(N):
        if i not in tree.nodes:
            continue
        p = tree[i].parent_id
        if p == ROOT_ID:
            continue
        edge_votes = parent_votes[i, p].clamp(min=0)
        total = edge_votes.sum().item()
        if total > 0:
            scores.append(edge_votes.max().item() / total)

    return sum(scores) / max(len(scores), 1) if scores else 1.0


def _parent_margin(evidence: dict, tree: Tree, N: int) -> float:
    """Margin between top-1 and top-2 parent vote counts."""
    parent_votes = evidence["parent_votes"].sum(dim=-1)  # (N, N+1)
    pos_votes = parent_votes.clamp(min=0)

    margin_sum = 0.0
    for i in range(N):
        row = pos_votes[i]
        total = row.sum().item()
        if total < 1e-6:
            continue
        sorted_vals = row.sort(descending=True).values
        top1 = sorted_vals[0].item()
        top2 = sorted_vals[1].item() if row.shape[0] > 1 else 0.0
        margin_sum += (top1 - top2) / total

    return margin_sum / max(N, 1)


def _spatial_consistency(evidence: dict, tree: Tree, N: int) -> float:
    """How well the tree's edge types match bounding box positions."""
    EXPECT_ABOVE = {Edge.NUM, Edge.SUP, Edge.UPPER}
    EXPECT_BELOW = {Edge.DEN, Edge.SUB, Edge.LOWER}

    scores = []
    for i in range(N):
        if i not in tree.nodes:
            continue
        node = tree[i]
        p = node.parent_id
        et = node.edge_type
        if p == ROOT_ID or et < 0:
            continue
        if et not in EXPECT_ABOVE and et not in EXPECT_BELOW:
            continue
        if p not in tree.nodes:
            continue

        cb = node.symbol.bbox
        pb = tree[p].symbol.bbox

        child_cy = cb.cy
        parent_cy = pb.cy
        parent_h = max(pb.h, 1e-6)
        dy = (child_cy - parent_cy) / parent_h

        if et in EXPECT_ABOVE:
            score = 1.0 / (1.0 + max(0.0, dy) * 3.0)
        else:
            score = 1.0 / (1.0 + max(0.0, -dy) * 3.0)

        scores.append(score)

    return sum(scores) / max(len(scores), 1) if scores else 1.0


def _x_position_consistency(evidence: dict, tree: Tree, N: int) -> float:
    """Check that children's x-positions stay within parent's span."""
    def cx(sid: int) -> float:
        return tree[sid].symbol.bbox.cx

    # Build sibling groups: (parent, edge_type) → [ids sorted by x]
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for sid, node in tree.nodes.items():
        groups[(node.parent_id, node.edge_type)].append(sid)
    for key in groups:
        groups[key].sort(key=cx)

    def _next_sibling_cx(p: int) -> float | None:
        if p not in tree.nodes:
            return None
        p_node = tree[p]
        sibs = groups.get((p_node.parent_id, p_node.edge_type), [])
        for k, s in enumerate(sibs):
            if s == p and k + 1 < len(sibs):
                return cx(sibs[k + 1])
        if p_node.parent_id != ROOT_ID:
            return _next_sibling_cx(p_node.parent_id)
        return None

    scores = []
    for sid, node in tree.nodes.items():
        if node.parent_id == ROOT_ID:
            continue
        bound_cx = _next_sibling_cx(node.parent_id)
        if bound_cx is None:
            continue
        child_cx = cx(sid)
        parent_cx = cx(node.parent_id)
        span = abs(bound_cx - parent_cx)
        if span < 1e-6:
            continue
        overshoot = max(0.0, child_cx - bound_cx) / span
        scores.append(1.0 / (1.0 + overshoot * 2.0))

    return sum(scores) / max(len(scores), 1) if scores else 1.0


def _seq_agreement(evidence: dict, tree: Tree, N: int) -> float:
    """How well SEQ predictions agree with the final tree structure."""
    gnn_seq = evidence.get("gnn_seq_scores")
    if gnn_seq is not None and N > 1:
        probs = torch.softmax(gnn_seq[:N], dim=-1)
        seq = probs[:, :N]
    else:
        seq_votes = evidence.get("seq_votes")
        if seq_votes is None or N <= 1:
            return 1.0
        seq = seq_votes[:N, :N]

    agree_sum = 0.0
    total_sum = 0.0
    for i in range(N):
        if i not in tree.nodes:
            continue
        ni = tree[i]
        for j in range(N):
            if i == j or j not in tree.nodes:
                continue
            w = seq[i, j].item()
            if w < 0.5:
                continue
            total_sum += w
            nj = tree[j]
            if ni.parent_id == nj.parent_id and ni.edge_type == nj.edge_type:
                agree_sum += w

    return agree_sum / total_sum if total_sum > 0 else 1.0


def _weakest_subtree(evidence: dict, tree: Tree, N: int) -> float:
    """Minimum subtree structural confidence."""
    parent_votes = evidence["parent_votes"]  # (N, N+1, E)
    total_per_parent = parent_votes.sum(dim=-1)
    pos_parent = total_per_parent.clamp(min=0)
    parent_total = pos_parent.sum(dim=-1).clamp(min=1e-6)
    parent_max = pos_parent.max(dim=-1).values
    parent_dec = (parent_max / parent_total).clamp(max=1.0)

    node_score = [0.0] * N
    for i in range(N):
        if i not in tree.nodes:
            continue
        pd = parent_dec[i].item()
        p = tree[i].parent_id
        if p != ROOT_ID:
            edge_votes = parent_votes[i, p].clamp(min=0)
            etotal = edge_votes.sum().item()
            ed = edge_votes.max().item() / etotal if etotal > 0 else 1.0
            node_score[i] = pd * ed
        else:
            node_score[i] = pd

    min_score = 1.0

    def _check(sid: int):
        nonlocal min_score
        indices = tree.walk(sid)
        if len(indices) >= 2:
            product = 1.0
            for i in indices:
                product *= node_score[i]
            geo = product ** (1.0 / len(indices))
            min_score = min(min_score, geo)
        for cid, _, _ in tree.children_of(sid):
            _check(cid)

    for rid in tree.root_ids():
        _check(rid)

    return min_score


# ── Strategies ───────────────────────────────────────────────────────


def _geo_mean(*vals: float) -> float:
    product = 1.0
    for v in vals:
        product *= v
    return product ** (1.0 / len(vals))


def parent_decisiveness(evidence: dict, tree: Tree, N: int) -> float:
    return _parent_decisiveness(evidence, tree, N)

def parent_seq(evidence: dict, tree: Tree, N: int) -> float:
    return _geo_mean(
        _parent_decisiveness(evidence, tree, N),
        _seq_agreement(evidence, tree, N))

def subset_agreement(evidence: dict, tree: Tree, N: int) -> float:
    return _subset_agreement(evidence, tree, N)

def parent_seq_agree(evidence: dict, tree: Tree, N: int) -> float:
    return _geo_mean(
        _parent_decisiveness(evidence, tree, N),
        _seq_agreement(evidence, tree, N),
        _subset_agreement(evidence, tree, N))

def full(evidence: dict, tree: Tree, N: int) -> float:
    return _geo_mean(
        _parent_decisiveness(evidence, tree, N),
        _seq_agreement(evidence, tree, N),
        _subset_agreement(evidence, tree, N),
        _parent_margin(evidence, tree, N))

def full_spatial(evidence: dict, tree: Tree, N: int) -> float:
    return _geo_mean(
        _parent_decisiveness(evidence, tree, N),
        _seq_agreement(evidence, tree, N),
        _subset_agreement(evidence, tree, N),
        _parent_margin(evidence, tree, N),
        _spatial_consistency(evidence, tree, N),
        _x_position_consistency(evidence, tree, N),
        _weakest_subtree(evidence, tree, N))

def parent_seq_xpos(evidence: dict, tree: Tree, N: int) -> float:
    return _geo_mean(
        _parent_decisiveness(evidence, tree, N),
        _seq_agreement(evidence, tree, N),
        _x_position_consistency(evidence, tree, N))


_STRATEGIES = {
    "parent_decisiveness": parent_decisiveness,
    "parent_seq": parent_seq,
    "subset_agreement": subset_agreement,
    "parent_seq_agree": parent_seq_agree,
    "parent_seq_xpos": parent_seq_xpos,
    "full": full,
    "full_spatial": full_spatial,
}


def score_tree(method: str, evidence: dict, tree: Tree, N: int) -> float:
    fn = _STRATEGIES.get(method)
    if fn is None:
        raise ValueError(f"Unknown scoring method {method!r}. Available: {list(_STRATEGIES.keys())}")
    return fn(evidence, tree, N)
