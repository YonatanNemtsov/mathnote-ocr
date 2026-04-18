"""Tree constants and legacy helpers.

Edge constants are re-exported from tree_v2 for convenience.
SymbolNode and build_tree are kept for diagnostic scripts only —
production code should use tree_v2 directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from tree_parser.tree_v2 import Edge, ROOT_ID as ROOT

NUM = Edge.NUM
DEN = Edge.DEN
SUP = Edge.SUP
SUB = Edge.SUB
SQRT_CONTENT = Edge.SQRT
UPPER = Edge.UPPER
LOWER = Edge.LOWER
MATCH = Edge.MATCH

NUM_EDGE_TYPES = 8
EDGE_NAMES = tuple(e.name.lower() for e in Edge if e >= 0)


# ── Legacy helpers (used by diagnostic scripts) ────────────────────


@dataclass
class SymbolNode:
    """Mutable node for diagnostic scripts. Do not use in production."""
    symbol: str
    bbox: list[float]
    index: int
    parent: int = ROOT
    edge_type: int = -1
    order: int = 0
    children: dict = None

    def __post_init__(self):
        if self.children is None:
            self.children = {}


def build_tree(nodes: list[SymbolNode]) -> list[SymbolNode]:
    """Build linked SymbolNode tree from parent pointers. Returns roots."""
    by_idx = {n.index: n for n in nodes}
    for n in nodes:
        n.children = {}
    roots = []
    for n in nodes:
        if n.parent == ROOT:
            roots.append(n)
        else:
            parent = by_idx.get(n.parent)
            if parent is not None:
                parent.children.setdefault(n.edge_type, []).append(n)
    for n in nodes:
        for et in n.children:
            n.children[et].sort(key=lambda c: c.order)
    roots.sort(key=lambda n: n.order)
    return roots


def tree_to_latex(roots: list[SymbolNode]) -> str:
    """Convert SymbolNode tree to LaTeX via tree_v2."""
    from tree_parser.tree_v2 import Symbol, Node, Tree, ROOT_ID
    from tree_parser.tree_latex import tree_to_latex as _v2_to_latex
    from bbox import BBox

    nodes: list[Node] = []

    def _walk(sn: SymbolNode):
        sym = Symbol(id=sn.index, name=sn.symbol, bbox=BBox(*sn.bbox))
        parent_id = ROOT_ID if sn.parent == ROOT else sn.parent
        nodes.append(Node(sym, parent_id, sn.edge_type, sn.order))
        for children in sn.children.values():
            for child in children:
                _walk(child)

    for r in roots:
        _walk(r)

    return _v2_to_latex(Tree(tuple(nodes)))
