"""Expression collapsing for tree structures.

Collapses specified symbols (and their descendants) into single "expression"
nodes with union bounding boxes. Used during training (augmentation) and
at inference time for hierarchical decomposition.
"""

from __future__ import annotations

import random

EXPR_NAME = "expression"


def collapse_subtrees(
    symbols: list[dict],
    tree: list[dict],
    collapse_indices: set[int],
    return_mapping: bool = False,
) -> tuple[list[dict], list[dict]] | tuple[list[dict], list[dict], dict]:
    """Collapse specified symbols and all their descendants into EXPR nodes.

    Symbols in `collapse_indices` that share the same (parent, edge_type)
    are merged into a single EXPR node. The EXPR gets the union bbox and
    the order of the first collapsed member. Remaining sibling orders are
    renumbered to stay contiguous.

    Args:
        symbols: list of {"name": str, "bbox": [x, y, w, h]}
        tree: list of {"parent": int, "edge_type": int, "order": int}
        collapse_indices: set of symbol indices to collapse
        return_mapping: if True, returns a third element with index mappings:
            {
                "old2new": dict[int, int],  # old index -> new index (kept symbols only)
                "new2old": dict[int, int],  # new index -> old index (kept symbols only)
                "expr_nodes": list[{"new_idx": int, "old_indices": list[int],
                                    "collapsed_members": list[int]}],
            }
    """
    if not collapse_indices:
        if return_mapping:
            identity = {i: i for i in range(len(symbols))}
            return symbols, tree, {"old2new": identity, "new2old": identity, "expr_nodes": []}
        return symbols, tree

    n = len(symbols)

    # Build children map
    children_map: dict[int, list[int]] = {}
    for i in range(n):
        p = tree[i]["parent"]
        if p >= 0:
            children_map.setdefault(p, []).append(i)

    def _get_all(node: int) -> set[int]:
        """Get node + all descendants."""
        result = {node}
        stack = children_map.get(node, [])[:]
        while stack:
            c = stack.pop()
            result.add(c)
            stack.extend(children_map.get(c, []))
        return result

    # Group collapse targets by (parent, edge_type)
    groups: dict[tuple[int, int], list[int]] = {}
    for i in collapse_indices:
        p = tree[i]["parent"]
        et = tree[i]["edge_type"]
        groups.setdefault((p, et), []).append(i)

    # Sort each group by order
    for key in groups:
        groups[key].sort(key=lambda i: tree[i]["order"])

    # Collect all nodes to remove and EXPR nodes to insert
    remove = set()
    # (parent, edge_type, all_nodes_for_bbox, order_of_expr, kept_siblings)
    insert: list[tuple[int, int, list[int], int, list[int]]] = []

    for (p, et), members_to_collapse in groups.items():
        all_nodes = set()
        for m in members_to_collapse:
            all_nodes |= _get_all(m)
        # Skip if parent is being removed by another collapse
        if p >= 0 and p in remove:
            continue
        if all_nodes & remove:
            continue
        remove |= all_nodes

        # Find kept siblings in the same (parent, edge_type) group
        kept_siblings = [
            i
            for i in range(n)
            if tree[i]["parent"] == p and tree[i]["edge_type"] == et and i not in all_nodes
        ]

        expr_order = tree[members_to_collapse[0]]["order"]
        insert.append((p, et, list(all_nodes), expr_order, kept_siblings))

    if not remove:
        return symbols, tree

    # Build new symbols/tree
    keep = [i for i in range(n) if i not in remove]
    old2new = {old: new for new, old in enumerate(keep)}

    new_symbols = []
    new_tree = []
    for old_i in keep:
        new_symbols.append(symbols[old_i])
        t = tree[old_i]
        old_parent = t["parent"]
        new_parent = old2new.get(old_parent, -1) if old_parent >= 0 else old_parent
        new_tree.append(
            {
                "parent": new_parent,
                "edge_type": t["edge_type"],
                "order": t["order"],
            }
        )

    # Add expression nodes
    for p, et, all_nodes, expr_order, kept_siblings in insert:
        xs = [symbols[j]["bbox"][0] for j in all_nodes]
        ys = [symbols[j]["bbox"][1] for j in all_nodes]
        rs = [symbols[j]["bbox"][0] + symbols[j]["bbox"][2] for j in all_nodes]
        bs = [symbols[j]["bbox"][1] + symbols[j]["bbox"][3] for j in all_nodes]
        x, y = min(xs), min(ys)

        expr_idx = len(new_symbols)
        new_symbols.append({"name": EXPR_NAME, "bbox": [x, y, max(rs) - x, max(bs) - y]})
        new_tree.append(
            {
                "parent": old2new.get(p, -1) if p >= 0 else p,
                "edge_type": et,
                "order": expr_order,
            }
        )

        # Renumber sibling orders to be contiguous (0, 1, 2, ...)
        sibs = []
        for sib in kept_siblings:
            if sib in old2new:
                sibs.append((tree[sib]["order"], old2new[sib]))
        sibs.append((expr_order, expr_idx))
        sibs.sort()
        for rank, (_, idx) in enumerate(sibs):
            new_tree[idx]["order"] = rank

    if not return_mapping:
        return new_symbols, new_tree

    # Build mapping
    new2old = {new: old for old, new in old2new.items()}
    expr_info = []
    for p, et, all_nodes, expr_order, kept_siblings in insert:
        # Find which members were directly collapsed (not descendants)
        members = [i for i in all_nodes if i in collapse_indices]
        members.sort(key=lambda i: tree[i]["order"])
        expr_idx = len(new2old) + len(expr_info)
        # Find actual expr_idx in new_symbols
        for ni in range(len(new_symbols)):
            if ni not in new2old.values() and new_symbols[ni]["name"] == EXPR_NAME:
                if ni not in [e["new_idx"] for e in expr_info]:
                    expr_idx = ni
                    break
        expr_info.append(
            {
                "new_idx": expr_idx,
                "old_indices": sorted(all_nodes),
                "collapsed_members": members,
            }
        )

    mapping = {
        "old2new": old2new,
        "new2old": new2old,
        "expr_nodes": expr_info,
    }
    return new_symbols, new_tree, mapping


def random_collapse(
    symbols: list[dict],
    tree: list[dict],
    collapse_prob: float = 0.04,
    min_total: int = 2,
) -> tuple[list[dict], list[dict]]:
    """Training augmentation: randomly select sibling runs to collapse.

    For each (parent, edge_type) sibling group, may pick multiple
    non-overlapping contiguous runs. Each candidate run rolls against
    `collapse_prob` and must have total symbols (selected + descendants)
    >= `min_total`.
    """
    n = len(symbols)

    # Build children map for descendant counting
    children_map: dict[int, list[int]] = {}
    for i in range(n):
        p = tree[i]["parent"]
        if p >= 0:
            children_map.setdefault(p, []).append(i)

    def _count_descendants(node: int) -> int:
        count = 1
        stack = children_map.get(node, [])[:]
        while stack:
            c = stack.pop()
            count += 1
            stack.extend(children_map.get(c, []))
        return count

    # Group children by (parent, edge_type)
    groups: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        p = tree[i]["parent"]
        et = tree[i]["edge_type"]
        groups.setdefault((p, et), []).append(i)

    for key in groups:
        groups[key].sort(key=lambda i: tree[i]["order"])

    # For each group, walk through siblings and randomly start runs
    collapse_indices: set[int] = set()
    for members in groups.values():
        i = 0
        while i < len(members):
            if random.random() >= collapse_prob:
                i += 1
                continue
            # Start a run from position i, extend with geometric falloff
            max_run = len(members) - i
            run_len = 1
            while run_len < max_run and random.random() < 0.5:
                run_len += 1
            run = members[i : i + run_len]
            total = sum(_count_descendants(m) for m in run)
            if total >= min_total:
                collapse_indices.update(run)
            i += run_len

    return collapse_subtrees(symbols, tree, collapse_indices)
