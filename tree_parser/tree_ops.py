"""Tree operations — higher-level functions on tree_v2.Tree."""

from __future__ import annotations

from bbox import BBox
from tree_parser.tree_v2 import Symbol, Node, Tree, Edge, ROOT_ID, SymbolId, EdgeType, SiblingOrder


def subtree(tree: Tree, sym_id: SymbolId) -> Tree:
    """Extract subtree rooted at sym_id."""
    ids = set(tree.walk(sym_id))
    return Tree(tuple(
        Node(n.symbol, tree.root, Edge.ROOT, 0) if n.symbol.id == sym_id else n
        for n in tree._nodes if n.symbol.id in ids
    ), tree.root)


def graft(tree: Tree, other: Tree, parent_id: SymbolId, edge_type: EdgeType, order: SiblingOrder = 0) -> Tree:
    """Graft another tree under parent_id."""
    root_ids = set(other.root_ids())
    return Tree(tree._nodes + tuple(
        Node(n.symbol, parent_id, edge_type, n.order) if n.symbol.id in root_ids else n
        for n in other._nodes if n.symbol.id != other.root
    ), tree.root)


def collapse(tree: Tree, sym_ids: set[SymbolId], expr_id: SymbolId) -> tuple[Tree, Tree]:
    """Collapse symbols into one EXPR node. Returns (collapsed_tree, extracted_subtree)."""
    # Root of the group: parent is outside the group
    group_roots = [sid for sid in sym_ids if tree.nodes[sid].parent_id not in sym_ids]
    group_root_set = set(group_roots)

    # Extract subtree preserving internal structure
    extracted = Tree(tuple(
        Node(n.symbol, tree.root, Edge.ROOT, n.order) if n.symbol.id in group_root_set else n
        for n in tree._nodes if n.symbol.id in sym_ids
    ), tree.root)

    # EXPR inherits first group root's position
    ref = tree.nodes[group_roots[0]] if group_roots else tree.nodes[next(iter(sym_ids))]
    expr_bbox = BBox.union_all([tree.nodes[sid].symbol.bbox for sid in sym_ids])
    expr_node = Node(Symbol(expr_id, "expr", expr_bbox), ref.parent_id, ref.edge_type, ref.order)

    # Remove collapsed symbols, add EXPR
    collapsed = Tree(
        tuple(n for n in tree._nodes if n.symbol.id not in sym_ids) + (expr_node,),
        tree.root,
    )

    return collapsed, extracted


def expand(tree: Tree, expr_id: SymbolId, extracted: Tree) -> Tree:
    """Expand an EXPR node back into its full subtree."""
    expr_node = tree.nodes[expr_id]
    root_ids = set(extracted.root_ids())

    return Tree(
        tuple(n for n in tree._nodes if n.symbol.id != expr_id) + tuple(
            Node(n.symbol, expr_node.parent_id, expr_node.edge_type, n.order)
            if n.symbol.id in root_ids else n
            for n in extracted._nodes if n.symbol.id != extracted.root
        ),
        tree.root,
    )


def reorder_siblings(tree: Tree) -> Tree:
    """Return tree with siblings ordered by bbox x-position."""
    groups: dict[tuple[SymbolId, EdgeType], list[SymbolId]] = {}
    for node in tree._nodes:
        sid = node.symbol.id
        if sid == tree.root:
            continue
        groups.setdefault((node.parent_id, node.edge_type), []).append(sid)

    new_order: dict[SymbolId, SiblingOrder] = {}
    for sids in groups.values():
        for rank, sid in enumerate(sorted(sids, key=lambda s: tree.nodes[s].symbol.bbox.cx)):
            new_order[sid] = rank

    return Tree(tuple(
        Node(n.symbol, n.parent_id, n.edge_type, new_order.get(n.symbol.id, n.order))
        if n.symbol.id in new_order else n
        for n in tree._nodes
    ), tree.root)


def fix_dot_cdot(tree: Tree) -> Tree:
    """Fix dot/cdot disambiguation by vertical position relative to siblings."""
    renames: dict[SymbolId, str] = {}

    for parent_id, kids in tree.children.items():
        child_ids = [cid for cid, _, _ in kids]
        ref = [s for s in child_ids if tree.nodes[s].symbol.name not in ("dot", "cdot", "prime")]
        if not ref:
            continue
        ref_cys = sorted(tree.nodes[r].symbol.bbox.cy for r in ref)
        ref_hs = sorted(tree.nodes[r].symbol.bbox.h for r in ref)
        median_cy = ref_cys[len(ref_cys) // 2]
        median_h = ref_hs[len(ref_hs) // 2]
        if median_h < 1e-6:
            continue
        for s in child_ids:
            if tree.nodes[s].symbol.name not in ("dot", "cdot"):
                continue
            dy = (tree.nodes[s].symbol.bbox.cy - median_cy) / median_h
            renames[s] = "dot" if dy > 0.25 else "cdot"

    if not renames:
        return tree
    result = tree
    for sid, name in renames.items():
        result = result.rename_node(sid, name)
    return result
