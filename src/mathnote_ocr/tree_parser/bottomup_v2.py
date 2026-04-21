"""Bottom-up tree builder with beam search."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch

from mathnote_ocr.tree_parser.evidence import get_evidence_tta
from mathnote_ocr.tree_parser.tree_ops import reorder_siblings
from mathnote_ocr.tree_parser.tree_v2 import ROOT_ID, Symbol, SymbolId, Tree

log = logging.getLogger(__name__)


def resolve_frac_bars(
    symbols: list[Symbol],
    run_subsets_fn: Callable,
    make_subsets_fn: Callable,
    high_threshold: float = 0.7,
    low_threshold: float = 0.3,
) -> list[list[Symbol]]:
    """For each '-' symbol, decide: minus, frac_bar, or ambiguous.

    Returns a list of symbol variants to try (for beam search):
        - If all '-' are unambiguous: returns [symbols] (one variant)
        - If some are ambiguous: returns multiple variants (branch)

    Runs the model once with all '-' as 'frac_bar' and checks confidence.
    """
    from mathnote_ocr.tree_parser.evidence import aggregate_evidence_soft

    minus_indices = [i for i, s in enumerate(symbols) if s.name == "-"]
    if not minus_indices:
        return [symbols]

    # Run with all '-' as 'frac_bar'
    bboxes = [s.bbox.to_list() for s in symbols]
    names_frac = [s.name for s in symbols]
    for mi in minus_indices:
        names_frac[mi] = "frac_bar"

    subsets = make_subsets_fn(bboxes)
    partial = run_subsets_fn(names_frac, bboxes, subsets)
    ev = aggregate_evidence_soft(len(symbols), partial)
    pv = ev["parent_votes"]
    N = len(symbols)

    # Classify each minus
    certain_frac: list[int] = []  # definitely frac_bar
    certain_minus: list[int] = []  # definitely minus
    ambiguous: list[int] = []  # could be either

    for mi in minus_indices:
        num_score = 0.0
        den_score = 0.0
        for j in range(N):
            if j == mi:
                continue
            votes = pv[j, mi].clamp(min=0)
            total = pv[j].clamp(min=0).sum().item()
            if total < 1e-6:
                continue
            frac = votes.sum().item() / total
            if frac < low_threshold:
                continue
            edge = votes.argmax().item()
            if edge == 0:
                num_score = max(num_score, frac)
            elif edge == 1:
                den_score = max(den_score, frac)

        has_both = min(num_score, den_score)
        if has_both >= high_threshold:
            certain_frac.append(mi)
        elif has_both >= low_threshold:
            ambiguous.append(mi)
        else:
            certain_minus.append(mi)

    # Build variants
    base = list(symbols)
    for mi in certain_frac:
        base[mi] = Symbol(base[mi].id, "frac_bar", base[mi].bbox)

    if not ambiguous:
        return [base]

    # Fork on ambiguous: 2^len(ambiguous) variants
    variants = [list(base)]
    for mi in ambiguous:
        new_variants = []
        for v in variants:
            # Keep as minus
            new_variants.append(list(v))
            # Promote to frac_bar
            promoted = list(v)
            promoted[mi] = Symbol(v[mi].id, "frac_bar", v[mi].bbox)
            new_variants.append(promoted)
        variants = new_variants

    return variants


def find_leaves(
    evidence: dict[str, torch.Tensor], n_symbols: int, threshold: float = 0.3
) -> list[SymbolId]:
    """Find symbols that are most certainly leaves (nothing points to them as parent).

    For each symbol, checks how much evidence there is that ANY other symbol
    has it as parent. If nobody claims it as parent with confidence above
    threshold, it's a leaf.

    Returns leaf ids sorted by certainty (most certain first).
    """
    parent_votes = evidence["parent_votes"]  # (N, N+1, E)
    N = n_symbols

    # For each symbol, compute how much total vote it receives as a parent
    parent_evidence = torch.zeros(N)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # How much does j vote for i as parent?
            votes_for_i = parent_votes[j, i].clamp(min=0).sum().item()
            total_votes = parent_votes[j].clamp(min=0).sum().item()
            if total_votes > 0:
                parent_evidence[i] += votes_for_i / total_votes

    # Leaves: symbols with low parent_evidence (nobody claims them as parent)
    leaves = []
    for i in range(N):
        if parent_evidence[i] < threshold:
            leaves.append(i)

    # Sort by certainty: lowest parent_evidence first (most certain leaves)
    leaves.sort(key=lambda i: parent_evidence[i].item())
    return leaves


@dataclass
class Attachment:
    """A candidate attachment for a leaf symbol."""

    leaf: int  # leaf position
    parent: int  # parent position (-1 = ROOT)
    edge_type: int  # edge type to parent
    parent_score: float  # confidence in this parent
    seq_prev: int  # SEQ previous sibling (-1 = none)
    seq_score: float  # confidence in this SEQ link
    conflict: bool  # True if SEQ and parent disagree


def find_attachments(
    leaves: list[int],
    evidence: dict[str, torch.Tensor],
    n_symbols: int,
) -> list[Attachment]:
    """For each leaf, find best parent and best SEQ. Flag conflicts.

    A conflict: SEQ says leaf's previous sibling is X, but leaf's
    best parent differs from X's best parent.
    """
    parent_votes = evidence["parent_votes"]  # (N, N+1, E)
    seq_votes = evidence.get("seq_votes")  # (N, N+1) or None
    N = n_symbols
    E = parent_votes.shape[-1]

    # Precompute best parent for every symbol (used for conflict detection)
    best_parent_of: dict[int, int] = {}
    for i in range(N):
        votes = parent_votes[i].clamp(min=0)
        total = votes.sum().item()
        if total > 1e-6:
            flat = votes.argmax().item()
            best_parent_of[i] = flat // E

    attachments = []
    for leaf in leaves:
        # Best parent
        votes = parent_votes[leaf].clamp(min=0)
        total = votes.sum().item()
        if total < 1e-6:
            continue

        flat = votes.argmax().item()
        parent = flat // E
        edge = flat % E
        parent_score = votes[parent, edge].item() / total

        if parent >= N:
            parent = -1  # ROOT

        # Best SEQ prev
        seq_prev = -1
        seq_score = 0.0
        if seq_votes is not None:
            sv = seq_votes[leaf, : N + 1]
            seq_total = sv.sum().item()
            if seq_total > 1e-6:
                best_seq = sv.argmax().item()
                seq_score = sv[best_seq].item() / seq_total
                seq_prev = best_seq if best_seq < N else -1

        # Conflict: leaf's parent differs from seq_prev's parent
        conflict = False
        if seq_prev >= 0 and parent >= 0:
            seq_prev_parent = best_parent_of.get(seq_prev, -1)
            if seq_prev_parent != parent:
                conflict = True

        attachments.append(
            Attachment(
                leaf=leaf,
                parent=parent,
                edge_type=edge,
                parent_score=parent_score,
                seq_prev=seq_prev,
                seq_score=seq_score,
                conflict=conflict,
            )
        )

    # Sort: non-conflicting first, then by parent confidence
    attachments.sort(key=lambda a: (a.conflict, -a.parent_score))
    return attachments


@dataclass
class ChainAssignment:
    """A candidate assignment for a sibling chain."""

    chain: list[int]  # symbol positions in the chain (ordered by SEQ)
    parent: int  # parent position (-1 = ROOT)
    edge_type: int  # edge type to parent
    confidence: float  # how confident we are
    alternatives: list[tuple[int, int, float]]  # [(parent, edge_type, score), ...] other candidates


def group_into_chains(
    attachments: list[Attachment],
    evidence: dict[str, torch.Tensor],
    n_symbols: int,
    confidence_threshold: float = 0.8,
    root_discount: float = 0.2,
) -> tuple[list[ChainAssignment], list[ChainAssignment]]:
    """Group leaf attachments into SEQ chains and find parent for each chain.

    Returns (certain, uncertain):
        certain: chains with high confidence, no conflicts
        uncertain: chains with conflicts or low confidence (branch points)
    """
    # Build SEQ chains from attachments: follow seq_prev links
    leaf_set = {a.leaf for a in attachments}
    att_by_leaf = {a.leaf: a for a in attachments}

    # Find chain heads: leaves whose seq_prev is not a leaf (or is -1)
    heads = []
    for a in attachments:
        if a.seq_prev == -1 or a.seq_prev not in leaf_set:
            heads.append(a.leaf)

    # Build next_of: for each leaf, who comes after it in the chain?
    next_of: dict[int, int] = {}
    for a in attachments:
        if a.seq_prev >= 0 and a.seq_prev in leaf_set:
            next_of[a.seq_prev] = a.leaf

    # Follow chains from heads
    chains: list[list[int]] = []
    used: set[int] = set()
    for head in heads:
        if head in used:
            continue
        chain = [head]
        used.add(head)
        cur = head
        while cur in next_of:
            nxt = next_of[cur]
            if nxt in used:
                break
            chain.append(nxt)
            used.add(nxt)
            cur = nxt
        chains.append(chain)

    # Singletons: leaves not in any chain
    for a in attachments:
        if a.leaf not in used:
            chains.append([a.leaf])

    # For each chain, find parent candidates
    parent_votes = evidence["parent_votes"]
    N = n_symbols
    E = parent_votes.shape[-1]

    certain: list[ChainAssignment] = []
    uncertain: list[ChainAssignment] = []

    for chain in chains:
        # Aggregate parent votes across chain members, excluding chain members as candidates
        chain_set = set(chain)
        combined = torch.zeros(N + 1, E)
        for i in chain:
            for j in range(N + 1):
                if j < N and j in chain_set:
                    continue
                combined[j] += parent_votes[i, j].clamp(min=0)

        # Discount ROOT votes
        combined[N] *= root_discount

        # Rank candidates
        flat_scores = combined.view(-1)
        top_k = min(3, flat_scores.shape[0])
        top_vals, top_idxs = flat_scores.topk(top_k)
        total = flat_scores.clamp(min=0).sum().item()

        candidates = []
        for val, idx in zip(top_vals, top_idxs):
            p = idx.item() // E
            e = idx.item() % E
            score = val.item() / total if total > 0 else 0
            parent_id = p if p < N else -1
            candidates.append((parent_id, e, score))

        best_parent, best_edge, best_score = candidates[0]
        alternatives = candidates[1:]

        # Check for conflicts within the chain
        has_conflict = any(att_by_leaf[i].conflict for i in chain if i in att_by_leaf)

        if best_score >= confidence_threshold and not has_conflict:
            certain.append(ChainAssignment(chain, best_parent, best_edge, best_score, alternatives))
        else:
            uncertain.append(
                ChainAssignment(chain, best_parent, best_edge, best_score, alternatives)
            )

    return certain, uncertain


def apply_assignments(
    tree: Tree,
    assignments: list[ChainAssignment],
    symbols: list[Symbol],
    run_subsets_fn: Callable | None = None,
    make_subsets_fn: Callable | None = None,
) -> tuple[Tree, float]:
    """Apply chain assignments to a tree. Each chain member becomes
    a child of the assigned parent.

    Returns (tree, verification_score). Score is the average agreement
    across all verified assignments (0-1). If no verification functions
    provided, score is 1.0.
    """
    total_score = 0.0
    n_verified = 0
    for assignment in assignments:
        if assignment.parent >= 0:
            parent_id = symbols[assignment.parent].id
            edge_type = assignment.edge_type
        else:
            parent_id = ROOT_ID
            edge_type = -1
        for order, pos in enumerate(assignment.chain):
            tree = tree.move_node(symbols[pos].id, parent_id, edge_type, order)

        # Verify this assignment with the model
        if run_subsets_fn is not None and make_subsets_fn is not None and assignment.parent >= 0:
            conf = verify_subtree(
                assignment.parent,
                assignment.chain,
                [assignment.edge_type] * len(assignment.chain),
                symbols,
                run_subsets_fn,
                make_subsets_fn,
            )
            total_score += conf
            n_verified += 1

    return tree, total_score, n_verified


def _find_chain_breaks(
    chain: list[int],
    evidence: dict[str, torch.Tensor],
    n_symbols: int,
) -> list[int]:
    """Find positions where a chain should potentially be broken.

    A break point is where adjacent members have different best parents.
    Returns indices into the chain (0-based) where a break could happen
    (between chain[i] and chain[i+1]).
    """
    parent_votes = evidence["parent_votes"]
    E = parent_votes.shape[-1]

    best_parent = {}
    for pos in chain:
        votes = parent_votes[pos].clamp(min=0)
        total = votes.sum().item()
        if total > 1e-6:
            best_parent[pos] = votes.argmax().item() // E

    breaks = []
    for i in range(len(chain) - 1):
        p1 = best_parent.get(chain[i], -1)
        p2 = best_parent.get(chain[i + 1], -1)
        if p1 != p2 and p1 < n_symbols and p2 < n_symbols:
            breaks.append(i + 1)  # break BEFORE chain[i+1]

    return breaks


def fork_on_uncertain(
    tree: Tree,
    uncertain: list[ChainAssignment],
    symbols: list[Symbol],
    evidence: dict[str, torch.Tensor],
    n_symbols: int,
    run_subsets_fn: Callable | None = None,
    make_subsets_fn: Callable | None = None,
) -> list[tuple[Tree, float]]:
    """For each uncertain assignment, fork: try different parents
    and chain breaks.

    Returns list of (tree, verification_score) pairs.
    """
    if not uncertain:
        return [(tree, 1.0)]

    candidates: list[tuple[Tree, float]] = [(tree, 0.0)]
    for assignment in uncertain:
        new_candidates: list[tuple[Tree, float]] = []

        # Option 1: keep chain intact, try different parents
        parent_options = [(assignment.parent, assignment.edge_type)]
        parent_options.extend((p, e) for p, e, _ in assignment.alternatives)

        # Option 2: break chain at conflict points, assign each piece independently
        breaks = _find_chain_breaks(assignment.chain, evidence, n_symbols)

        for candidate, prev_score in candidates:
            # Try intact chain with each parent
            for parent, edge_type in parent_options:
                if parent >= 0:
                    parent_id = symbols[parent].id
                else:
                    parent_id = ROOT_ID
                    edge_type = -1
                forked = candidate
                for order, pos in enumerate(assignment.chain):
                    forked = forked.move_node(symbols[pos].id, parent_id, edge_type, order)

                # Verify
                fork_score = 0.0
                if run_subsets_fn is not None and make_subsets_fn is not None and parent >= 0:
                    fork_score = verify_subtree(
                        parent,
                        assignment.chain,
                        [edge_type] * len(assignment.chain),
                        symbols,
                        run_subsets_fn,
                        make_subsets_fn,
                    )
                new_candidates.append((forked, prev_score + fork_score))

            # Try breaking the chain at each break point
            for brk in breaks:
                left = assignment.chain[:brk]
                right = assignment.chain[brk:]

                forked = candidate
                break_score = 0.0
                for piece in [left, right]:
                    piece_set = set(piece)
                    combined = torch.zeros(n_symbols + 1, evidence["parent_votes"].shape[-1])
                    for pos in piece:
                        for j in range(n_symbols + 1):
                            if j < n_symbols and j in piece_set:
                                continue
                            combined[j] += evidence["parent_votes"][pos, j].clamp(min=0)

                    E = combined.shape[-1]
                    best_flat = combined.argmax().item()
                    p = best_flat // E
                    e = best_flat % E
                    parent_id = symbols[p].id if p < n_symbols else ROOT_ID

                    for order, pos in enumerate(piece):
                        forked = forked.move_node(symbols[pos].id, parent_id, e, order)

                    # Verify this piece
                    if run_subsets_fn is not None and make_subsets_fn is not None and p < n_symbols:
                        break_score += verify_subtree(
                            p,
                            piece,
                            [e] * len(piece),
                            symbols,
                            run_subsets_fn,
                            make_subsets_fn,
                        )

                new_candidates.append((forked, prev_score + break_score))

        candidates = new_candidates

    return candidates


def verify_subtree(
    parent_pos: int,
    child_positions: list[int],
    child_edges: list[int],
    symbols: list[Symbol],
    run_subsets_fn: Callable,
    make_subsets_fn: Callable,
) -> float:
    """Verify a parent+children assignment by running the model on just that group.

    Returns a continuous confidence score (0-1): the average probability the model
    assigns to the proposed parent+edge for each child.
    """
    from mathnote_ocr.tree_parser.evidence import aggregate_evidence_soft

    group_positions = [parent_pos] + child_positions
    group_syms = [symbols[p] for p in group_positions]
    names = [s.name for s in group_syms]
    bboxes = [s.bbox.to_list() for s in group_syms]

    subsets = make_subsets_fn(bboxes)
    partial = run_subsets_fn(names, bboxes, subsets)
    evidence = aggregate_evidence_soft(len(group_syms), partial)

    parent_votes = evidence["parent_votes"]
    N = len(group_syms)
    E = parent_votes.shape[-1]

    # Parent is at local position 0. For each child, how much does the model
    # agree with this parent+edge assignment?
    total_conf = 0.0
    for local_child_idx in range(1, N):
        votes = parent_votes[local_child_idx].clamp(min=0)
        total = votes.sum().item()
        if total < 1e-6:
            continue
        expected_edge = child_edges[local_child_idx - 1]
        # Confidence = vote for (parent=0, edge=expected_edge) / total votes
        total_conf += votes[0, expected_edge].item() / total

    n_children = N - 1
    return total_conf / n_children if n_children > 0 else 1.0


def score_tree(tree: Tree, evidence: dict[str, torch.Tensor] | None = None) -> float:
    """Score a tree by spatial + structural consistency. Higher = better.

    Components:
        1. Vertical consistency: edge types match child/parent y-positions
        2. Parent proximity: children are spatially near their parent
        3. SEQ agreement: model's sibling predictions match tree structure
    """
    scores = []

    # 1. Vertical consistency
    EXPECT_ABOVE = {0, 2, 5}  # NUM, SUP, UPPER
    EXPECT_BELOW = {1, 3, 6}  # DEN, SUB, LOWER

    vert_ok = 0
    vert_total = 0
    for sid, node in tree.nodes.items():
        if sid == tree.root or node.parent_id == tree.root or node.edge_type < 0:
            continue
        parent = tree.nodes.get(node.parent_id)
        if parent is None:
            continue
        dy = node.symbol.bbox.cy - parent.symbol.bbox.cy
        vert_total += 1
        if node.edge_type in EXPECT_ABOVE and dy <= 0:
            vert_ok += 1
        elif node.edge_type in EXPECT_BELOW and dy >= 0:
            vert_ok += 1
        elif node.edge_type not in EXPECT_ABOVE and node.edge_type not in EXPECT_BELOW:
            vert_ok += 1
    if vert_total > 0:
        scores.append(vert_ok / vert_total)

    # 2. Parent proximity: penalize children far from parent
    # Use ratio of child-parent distance to expression width
    all_bboxes = [n.symbol.bbox for sid, n in tree.nodes.items() if sid != tree.root]
    if all_bboxes:
        expr_width = max(b.x2 for b in all_bboxes) - min(b.x for b in all_bboxes)
        if expr_width > 1e-6:
            proximity_scores = []
            for sid, node in tree.nodes.items():
                if sid == tree.root or node.parent_id == tree.root:
                    continue
                parent = tree.nodes.get(node.parent_id)
                if parent is None:
                    continue
                dist = node.symbol.bbox.center_distance(parent.symbol.bbox)
                # Closer = better. Score decays with distance relative to expression width.
                proximity_scores.append(1.0 / (1.0 + (dist / expr_width) * 3.0))
            if proximity_scores:
                scores.append(sum(proximity_scores) / len(proximity_scores))

    # 3. SEQ agreement: if model predicts siblings, they should share parent+edge
    if evidence is not None:
        seq_votes = evidence.get("seq_votes")
        if seq_votes is not None:
            N = seq_votes.shape[0]
            agree = 0.0
            total_w = 0.0
            for i in range(N):
                if i not in tree.nodes:
                    continue
                ni = tree[i]
                for j in range(N):
                    if i == j or j not in tree.nodes:
                        continue
                    w = seq_votes[i, j].item()
                    if w < 0.5:
                        continue
                    total_w += w
                    nj = tree[j]
                    if ni.parent_id == nj.parent_id and ni.edge_type == nj.edge_type:
                        agree += w
            if total_w > 0:
                scores.append(agree / total_w)

    if not scores:
        return 1.0
    # Geometric mean
    product = 1.0
    for s in scores:
        product *= max(s, 0.01)
    return product ** (1.0 / len(scores))


def find_ready(
    evidence: dict[str, torch.Tensor],
    n_symbols: int,
    resolved: set[SymbolId],
    threshold: float = 0.3,
) -> list[SymbolId]:
    """Find symbols that are ready to be assigned — all their children are resolved.

    A symbol is ready if every symbol that claims it as parent (above threshold)
    is already in the resolved set.
    """
    parent_votes = evidence["parent_votes"]
    N = n_symbols

    ready = []
    for i in range(N):
        sid = i  # position = id for original symbols
        if sid in resolved:
            continue

        # Find unresolved children of this symbol
        has_unresolved_child = False
        for j in range(N):
            if j == i or j in resolved:
                continue
            votes_for_i = parent_votes[j, i].clamp(min=0).sum().item()
            total = parent_votes[j].clamp(min=0).sum().item()
            if total > 0 and votes_for_i / total > threshold:
                # j claims i as parent, and j is not resolved
                has_unresolved_child = True
                break

        if not has_unresolved_child:
            ready.append(sid)

    return ready


def build(
    symbols: list[Symbol],
    run_subsets_fn: Callable,
    make_subsets_fn: Callable,
    *,
    root_discount: float = 0.2,
    max_iterations: int = 8,
    beam_width: int = 10,
    confidence_threshold: float = 0.5,
    tta_runs: int = 1,
    tta_dx: float = 0.05,
    tta_dy: float = 0.15,
    tta_size: float = 0.05,
    gnn_model=None,
    symbol_vocab: dict | None = None,
    device=None,
) -> Tree:
    """Build expression tree bottom-up with beam search."""
    from mathnote_ocr.tree_parser.tree_v2 import Edge, Node

    N = len(symbols)
    if N == 0:
        return Tree(())
    if N == 1:
        return Tree((Node(symbols[0], ROOT_ID, Edge.ROOT, 0),))

    # Resolve frac bars — may produce multiple variants to try
    symbol_variants = resolve_frac_bars(symbols, run_subsets_fn, make_subsets_fn)

    # Start beam with all variants
    # Entry: (tree, resolved, symbols, evidence, score_sum, score_count)
    beam: list[tuple[Tree, set[SymbolId], list[Symbol], dict, float, int]] = []
    for variant in symbol_variants:
        names = [s.name for s in variant]
        bboxes = [s.bbox.to_list() for s in variant]
        evidence = get_evidence_tta(
            names,
            bboxes,
            run_subsets_fn,
            make_subsets_fn,
            tta_runs=tta_runs,
            tta_dx=tta_dx,
            tta_dy=tta_dy,
            tta_size=tta_size,
        )
        initial_tree = Tree(tuple(Node(s, ROOT_ID, Edge.ROOT, i) for i, s in enumerate(variant)))
        beam.append((initial_tree, set(), variant, evidence, 0.0, 0))

    for iteration in range(max_iterations):
        next_beam: list[tuple[Tree, set[SymbolId], list[Symbol], dict, float, int]] = []

        for tree, resolved, syms, evidence, sc_sum, sc_cnt in beam:
            ready = find_ready(evidence, N, resolved)
            if not ready:
                next_beam.append((tree, resolved, syms, evidence, sc_sum, sc_cnt))
                continue

            att = find_attachments(ready, evidence, N)
            certain, uncertain = group_into_chains(
                att, evidence, N, confidence_threshold, root_discount
            )

            tree, c_score, c_count = apply_assignments(
                tree,
                certain,
                syms,
                run_subsets_fn,
                make_subsets_fn,
            )
            newly_resolved = set()
            for c in certain:
                newly_resolved.update(c.chain)

            forked = fork_on_uncertain(
                tree,
                uncertain,
                syms,
                evidence,
                N,
                run_subsets_fn,
                make_subsets_fn,
            )
            for f, fork_score in forked:
                fork_resolved = resolved | newly_resolved
                for u in uncertain:
                    fork_resolved.update(u.chain)
                # fork_score is raw confidence sum, count = number of uncertain chains verified
                fork_count = sum(1 for u in uncertain if u.parent >= 0)
                next_beam.append(
                    (
                        f,
                        fork_resolved,
                        syms,
                        evidence,
                        sc_sum + c_score + fork_score,
                        sc_cnt + c_count + fork_count,
                    )
                )

        # Deduplicate and keep top beam_width by average score
        seen: set[int] = set()
        pruned = []
        for entry in next_beam:
            h = hash(entry[0])
            if h not in seen:
                seen.add(h)
                pruned.append(entry)
        pruned.sort(key=lambda e: -(e[4] / e[5] if e[5] > 0 else 0.0))
        beam = pruned[:beam_width]

        if not beam:
            break
        if all(len(entry[1]) >= N for entry in beam):
            break

        best = beam[0]
        avg = best[4] / best[5] if best[5] > 0 else 0.0
        log.info(
            "  iter %d: %d candidates, best_score=%.3f, resolved=%d",
            iteration,
            len(beam),
            avg,
            max(len(entry[1]) for entry in beam),
        )

    # Final scoring: verify every subtree in each completed candidate
    if beam and len(beam) > 1:
        best_tree = max(
            beam, key=lambda b: _verify_score_subset(b[0], b[2], run_subsets_fn, make_subsets_fn)
        )[0]
    elif beam:
        best_tree = beam[0][0]
    else:
        best_tree = Tree(tuple(Node(s, ROOT_ID, Edge.ROOT, i) for i, s in enumerate(symbols)))
    return reorder_siblings(best_tree)


def _verify_score(
    tree: Tree,
    symbols: list[Symbol],
    run_subsets_fn,
    make_subsets_fn,
    gnn_model=None,
    symbol_vocab=None,
    device=None,
) -> float:
    """Score a tree by model verification.

    If gnn_model is provided, uses GNN for full-expression verification.
    Otherwise falls back to per-group subset model verification.
    """
    if gnn_model is not None:
        return _verify_score_gnn(
            tree, symbols, gnn_model, symbol_vocab, device, run_subsets_fn, make_subsets_fn
        )
    return _verify_score_subset(tree, symbols, run_subsets_fn, make_subsets_fn)


def _verify_score_gnn(
    tree: Tree,
    symbols: list[Symbol],
    gnn_model,
    symbol_vocab,
    device,
    run_subsets_fn,
    make_subsets_fn,
) -> float:
    """Score a tree using GNN — full expression, no size limit."""
    from mathnote_ocr.latex_utils.relations import compute_features_from_bbox_list
    from mathnote_ocr.tree_parser.evidence import aggregate_evidence_soft, evidence_to_features

    N = len(symbols)
    names = [s.name for s in symbols]
    bboxes = [s.bbox.to_list() for s in symbols]

    # Get base evidence
    subsets = make_subsets_fn(bboxes)
    partial = run_subsets_fn(names, bboxes, subsets)
    evidence = aggregate_evidence_soft(N, partial)

    # Run GNN
    _, edge_features = evidence_to_features(evidence)

    sym_ids = torch.tensor(
        [symbol_vocab.get(n, symbol_vocab.get("<unk>", 1)) for n in names],
        dtype=torch.long,
        device=device,
    )
    _, size_feats = compute_features_from_bbox_list(bboxes, N)
    size_feats = size_feats.to(device)
    edge_features = edge_features.to(device)
    pad_mask = torch.zeros(N, dtype=torch.bool, device=device)

    with torch.no_grad():
        out = gnn_model(
            sym_ids.unsqueeze(0),
            size_feats.unsqueeze(0),
            edge_features.unsqueeze(0),
            pad_mask.unsqueeze(0),
        )

    parent_scores = out["parent_scores"][0]  # (N, N+1)
    edge_type_scores = out["edge_type_scores"][0]  # (N, N+1, E)

    # Score: for each node in the tree, does the GNN agree?
    agree = 0
    total = 0
    for sid, node in tree.nodes.items():
        if sid == tree.root or sid >= N:
            continue
        total += 1
        gnn_parent = parent_scores[sid].argmax().item()
        if node.parent_id == tree.root:
            if gnn_parent == N:  # GNN also says ROOT
                agree += 1
        elif node.parent_id >= 0 and node.parent_id < N:
            if gnn_parent == node.parent_id:
                gnn_edge = edge_type_scores[sid, gnn_parent].argmax().item()
                if gnn_edge == node.edge_type:
                    agree += 1

    return agree / total if total > 0 else 1.0


def _verify_score_subset(
    tree: Tree, symbols: list[Symbol], run_subsets_fn, make_subsets_fn
) -> float:
    """Score a tree by verifying each parent+children group with the subset model."""
    parents_with_children: dict[int, list[tuple[int, int]]] = {}
    for sid, node in tree.nodes.items():
        if sid == tree.root or node.parent_id == tree.root:
            continue
        parents_with_children.setdefault(node.parent_id, []).append((sid, node.edge_type))

    if not parents_with_children:
        return 1.0

    total_score = 0.0
    total = 0
    for parent_id, children in parents_with_children.items():
        if parent_id not in tree.nodes:
            continue
        child_positions = [c for c, _ in children]
        child_edges = [e for _, e in children]

        parent_pos = next((i for i, s in enumerate(symbols) if s.id == parent_id), None)
        child_pos_list = [
            next((i for i, s in enumerate(symbols) if s.id == cid), None) for cid in child_positions
        ]

        if parent_pos is None or any(p is None for p in child_pos_list):
            continue

        conf = verify_subtree(
            parent_pos, child_pos_list, child_edges, symbols, run_subsets_fn, make_subsets_fn
        )
        total_score += conf
        total += 1

    return total_score / total if total > 0 else 1.0


def build_with_collapse(
    symbols: list[Symbol],
    run_subsets_fn: Callable,
    make_subsets_fn: Callable,
    *,
    root_discount: float = 0.2,
    max_iterations: int = 8,
    beam_width: int = 10,
    confidence_threshold: float = 0.5,
    tta_runs: int = 1,
    tta_dx: float = 0.05,
    tta_dy: float = 0.15,
    tta_size: float = 0.05,
    gnn_model=None,
    symbol_vocab: dict | None = None,
    device=None,
) -> Tree:
    """Build tree bottom-up with beam search + collapsing.

    After each iteration, verified subtrees are collapsed into EXPR nodes.
    Evidence is re-run on the smaller expression for better parent predictions.
    """
    from mathnote_ocr.bbox import BBox
    from mathnote_ocr.tree_parser.evidence import aggregate_evidence_soft
    from mathnote_ocr.tree_parser.tree_ops import expand
    from mathnote_ocr.tree_parser.tree_v2 import Edge, Node

    N = len(symbols)
    if N == 0:
        return Tree(())
    if N == 1:
        return Tree((Node(symbols[0], ROOT_ID, Edge.ROOT, 0),))

    # Resolve frac bars
    symbol_variants = resolve_frac_bars(symbols, run_subsets_fn, make_subsets_fn)

    # State per beam entry: (tree, resolved, current_syms, evidence, subtrees_dict)
    BeamEntry = tuple[Tree, set[SymbolId], list[Symbol], dict, dict[SymbolId, Tree]]

    beam: list[BeamEntry] = []
    for variant in symbol_variants:
        names = [s.name for s in variant]
        bboxes = [s.bbox.to_list() for s in variant]
        evidence = get_evidence_tta(
            names,
            bboxes,
            run_subsets_fn,
            make_subsets_fn,
            tta_runs=tta_runs,
            tta_dx=tta_dx,
            tta_dy=tta_dy,
            tta_size=tta_size,
        )
        tree = Tree(tuple(Node(s, ROOT_ID, Edge.ROOT, i) for i, s in enumerate(variant)))
        beam.append((tree, set(), variant, evidence, {}))

    next_expr_id = max(s.id for s in symbols) + 1000

    for iteration in range(max_iterations):
        next_beam: list[BeamEntry] = []

        for tree, resolved, syms, evidence, stored_subtrees in beam:
            cur_N = len(syms)
            ready = find_ready(evidence, cur_N, resolved)
            if not ready:
                # No ready symbols — force-assign all unresolved as leaves
                unresolved = [i for i in range(cur_N) if i not in resolved]
                if unresolved:
                    ready = unresolved
                else:
                    next_beam.append((tree, resolved, syms, evidence, stored_subtrees))
                    continue

            att = find_attachments(ready, evidence, cur_N)
            certain, uncertain = group_into_chains(
                att, evidence, cur_N, confidence_threshold, root_discount
            )

            tree, _ = apply_assignments(tree, certain, syms, run_subsets_fn, make_subsets_fn)
            newly_resolved = set()
            for c in certain:
                newly_resolved.update(c.chain)

            forked = fork_on_uncertain(
                tree, uncertain, syms, evidence, cur_N, run_subsets_fn, make_subsets_fn
            )
            for f, _ in forked:
                fork_resolved = resolved | newly_resolved
                for u in uncertain:
                    fork_resolved.update(u.chain)
                next_beam.append((f, fork_resolved, syms, evidence, dict(stored_subtrees)))

        # Deduplicate and limit
        seen: set[int] = set()
        pruned: list[BeamEntry] = []
        for entry in next_beam:
            h = hash(entry[0])
            if h not in seen:
                seen.add(h)
                pruned.append(entry)
        beam = pruned[:beam_width]

        if not beam:
            break

        # Collapse verified subtrees in each beam entry
        collapsed_beam: list[BeamEntry] = []
        for tree, resolved, syms, evidence, stored_subtrees in beam:
            cur_N = len(syms)
            # Find parents whose children are all resolved
            parent_votes = evidence["parent_votes"]
            collapsible = []
            for p_pos in range(cur_N):
                if p_pos in resolved:
                    continue
                children_pos = []
                has_unresolved = False
                for j in range(cur_N):
                    if j == p_pos:
                        continue
                    v = parent_votes[j, p_pos].clamp(min=0).sum().item()
                    t = parent_votes[j].clamp(min=0).sum().item()
                    if t > 0 and v / t > 0.3:
                        if j in resolved:
                            children_pos.append(j)
                        else:
                            has_unresolved = True
                            break
                if not has_unresolved and children_pos:
                    # Verify before collapsing
                    child_edges = [tree[syms[cp].id].edge_type for cp in children_pos]
                    conf = verify_subtree(
                        p_pos, children_pos, child_edges, syms, run_subsets_fn, make_subsets_fn
                    )
                    if conf > 0.5:
                        collapsible.append((p_pos, children_pos))

            if not collapsible:
                collapsed_beam.append((tree, resolved, syms, evidence, stored_subtrees))
                continue

            # Collapse each verified group
            new_stored = dict(stored_subtrees)
            collapse_positions: set[int] = set()
            new_expr_syms: list[Symbol] = []

            for p_pos, c_positions in collapsible:
                group_positions = {p_pos} | set(c_positions)
                if group_positions & collapse_positions:
                    continue  # overlap
                collapse_positions |= group_positions

                # Store subtree
                subtree_nodes = []
                for pos in group_positions:
                    node = tree[syms[pos].id]
                    if pos == p_pos:
                        subtree_nodes.append(Node(node.symbol, ROOT_ID, Edge.ROOT, 0))
                    else:
                        subtree_nodes.append(
                            Node(node.symbol, node.parent_id, node.edge_type, node.order)
                        )
                new_stored[next_expr_id] = Tree(tuple(subtree_nodes))

                # Create EXPR symbol
                expr_bbox = BBox.union_all([syms[pos].bbox for pos in group_positions])
                new_expr_syms.append(Symbol(next_expr_id, "expr", expr_bbox))

                log.info(
                    "    collapse iter %d: {%s} → expr_%d",
                    iteration,
                    ",".join(syms[p].name for p in group_positions),
                    next_expr_id,
                )
                next_expr_id += 1

            # Rebuild syms and evidence
            new_syms = [
                s for i, s in enumerate(syms) if i not in collapse_positions
            ] + new_expr_syms
            new_names = [s.name for s in new_syms]
            new_bboxes = [s.bbox.to_list() for s in new_syms]
            new_evidence = aggregate_evidence_soft(
                len(new_syms),
                run_subsets_fn(new_names, new_bboxes, make_subsets_fn(new_bboxes)),
            )
            new_tree = Tree(tuple(Node(s, ROOT_ID, Edge.ROOT, i) for i, s in enumerate(new_syms)))
            collapsed_beam.append((new_tree, set(), new_syms, new_evidence, new_stored))

        # Check if collapsing made progress
        prev_sizes = [len(entry[2]) for entry in beam]
        beam = collapsed_beam
        new_sizes = [len(entry[2]) for entry in beam]

        if all(len(entry[2]) <= 1 for entry in beam):
            break

        if new_sizes == prev_sizes[: len(new_sizes)]:
            break  # no collapsing happened — stop

        log.info(
            "  iter %d: %d candidates, %d syms",
            iteration,
            len(beam),
            min(len(entry[2]) for entry in beam),
        )

    # Score and pick best
    if beam and len(beam) > 1:
        best_entry = max(
            beam,
            key=lambda b: _verify_score(
                b[0], b[2], run_subsets_fn, make_subsets_fn, gnn_model, symbol_vocab, device
            ),
        )
    elif beam:
        best_entry = beam[0]
    else:
        return reorder_siblings(
            Tree(tuple(Node(s, ROOT_ID, Edge.ROOT, i) for i, s in enumerate(symbols)))
        )

    tree, _, _, _, stored_subtrees = best_entry

    # Expand all EXPR nodes
    changed = True
    while changed:
        changed = False
        for sid in list(tree.nodes):
            if sid in stored_subtrees and sid in tree.nodes:
                sub = stored_subtrees[sid]
                # Recursively expand subtrees within subtrees
                for sub_sid in list(sub.nodes):
                    if sub_sid in stored_subtrees and sub_sid in sub.nodes:
                        sub = expand(sub, sub_sid, stored_subtrees[sub_sid])
                tree = expand(tree, sid, sub)
                changed = True

    return reorder_siblings(tree)
