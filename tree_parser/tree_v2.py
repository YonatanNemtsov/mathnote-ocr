"""Math expression tree — persistent, immutable.

Symbol: id, name, bbox.
Node: symbol + parent pointer. Frozen.
Tree: tuple of nodes. Mutations return new trees sharing unchanged nodes.

The tree always has a virtual root node (ROOT_ID) whose children are
the top-level expression symbols. Every node's parent is a real node
in the tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from typing import TypeAlias

from bbox import BBox


class Edge(IntEnum):
    ROOT = -1
    NUM = 0
    DEN = 1
    SUP = 2
    SUB = 3
    SQRT = 4
    UPPER = 5
    LOWER = 6
    MATCH = 7


ROOT_ID = -1

SymbolId: TypeAlias = int
EdgeType: TypeAlias = int
SiblingOrder: TypeAlias = int

ChildrenIndex: TypeAlias = dict[SymbolId, tuple[tuple[SymbolId, EdgeType, SiblingOrder], ...]]


@dataclass(frozen=True)
class Symbol:
    id: SymbolId
    name: str
    bbox: BBox


@dataclass(frozen=True)
class Node:
    symbol: Symbol
    parent_id: SymbolId = ROOT_ID
    edge_type: EdgeType = Edge.ROOT
    order: SiblingOrder = 0


ROOT_SYMBOL = Symbol(ROOT_ID, "ROOT", BBox(0, 0, 0, 0))
ROOT_NODE = Node(ROOT_SYMBOL)


class Tree:
    """Persistent immutable tree. Nodes store parent pointers.

    Mutations return new trees. Unchanged nodes are shared.
    The root node (ROOT_ID) is always present.
    """

    def __init__(self, nodes: tuple[Node, ...], root: SymbolId = ROOT_ID):
        self._nodes = nodes
        self.root = root

    @cached_property
    def nodes(self) -> dict[SymbolId, Node]:
        d = {n.symbol.id: n for n in self._nodes}
        if self.root not in d:
            d[self.root] = ROOT_NODE
        return d

    @cached_property
    def children(self) -> ChildrenIndex:
        groups: dict[SymbolId, list[tuple[SymbolId, EdgeType, SiblingOrder]]] = {}
        for node in self._nodes:
            sid = node.symbol.id
            if sid == self.root:
                continue
            groups.setdefault(node.parent_id, []).append((sid, node.edge_type, node.order))
        return {pid: tuple(sorted(kids, key=lambda x: x[2])) for pid, kids in groups.items()}

    # ── Query ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.nodes) - 1  # exclude root

    def __bool__(self) -> bool:
        return len(self.nodes) > 1

    def __getitem__(self, sym_id: SymbolId) -> Node:
        return self.nodes[sym_id]

    def __contains__(self, sym_id: SymbolId) -> bool:
        return sym_id in self.nodes

    def root_ids(self) -> tuple[SymbolId, ...]:
        return tuple(sym_id for sym_id, _, _ in self.children.get(self.root, ()))

    def children_of(self, sym_id: SymbolId) -> tuple[tuple[SymbolId, EdgeType, SiblingOrder], ...]:
        """Children of a node: ((child_id, edge_type, order), ...) sorted by order."""
        return self.children.get(sym_id, ())

    def children_by_edge(self, sym_id: SymbolId, edge: EdgeType) -> tuple[SymbolId, ...]:
        """Child ids of a specific edge type, ordered."""
        return tuple(cid for cid, et, _ in self.children_of(sym_id) if et == edge)

    def is_leaf(self, sym_id: SymbolId) -> bool:
        return sym_id not in self.children

    def is_root(self, sym_id: SymbolId) -> bool:
        return self.nodes[sym_id].parent_id == self.root

    # ── Mutations (return new Tree) ──────────────────────────────────

    def add_node(self, node: Node) -> Tree:
        return Tree(self._nodes + (node,), self.root)

    def remove_node(self, sym_id: SymbolId) -> Tree:
        to_remove = self._descendants(sym_id) | {sym_id}
        return Tree(tuple(n for n in self._nodes if n.symbol.id not in to_remove), self.root)

    def move_node(self, sym_id: SymbolId, new_parent_id: SymbolId, edge_type: EdgeType, order: SiblingOrder = 0) -> Tree:
        return Tree(tuple(
            Node(n.symbol, new_parent_id, edge_type, order) if n.symbol.id == sym_id else n
            for n in self._nodes
        ), self.root)

    def rename_node(self, sym_id: SymbolId, new_name: str) -> Tree:
        return Tree(tuple(
            Node(Symbol(n.symbol.id, new_name, n.symbol.bbox), n.parent_id, n.edge_type, n.order)
            if n.symbol.id == sym_id else n
            for n in self._nodes
        ), self.root)

    # ── Traversal ────────────────────────────────────────────────────

    def walk(self, sym_id: SymbolId) -> tuple[SymbolId, ...]:
        """All ids in subtree (depth-first)."""
        return (sym_id,) + sum(
            (self.walk(cid) for cid, _, _ in self.children_of(sym_id)), ()
        )

    def path(self, sym_id: SymbolId) -> tuple[tuple[SymbolId, EdgeType, SiblingOrder], ...]:
        """Path from root to symbol: ((id, edge_type, order), ...). Root itself returns ()."""
        if sym_id == self.root:
            return ()
        node = self.nodes[sym_id]
        entry = (sym_id, node.edge_type, node.order)
        if node.parent_id == self.root:
            return (entry,)
        return self.path(node.parent_id) + (entry,)

    # ── Comparison ───────────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tree):
            return NotImplemented
        if self.root != other.root or len(self) != len(other):
            return False
        for sym_id, node in self.nodes.items():
            if sym_id == self.root:
                continue
            if other.nodes.get(sym_id) != node:
                return False
        return True

    def __hash__(self) -> int:
        return hash(frozenset(
            (n.symbol.id, n.symbol.name, n.parent_id, n.edge_type, n.order)
            for n in self._nodes if n.symbol.id != self.root
        ))

    def __repr__(self) -> str:
        return f"Tree({len(self)} symbols)"

    # ── Internal ─────────────────────────────────────────────────────

    def _descendants(self, sym_id: SymbolId) -> set[SymbolId]:
        result: set[SymbolId] = set()
        for cid, _, _ in self.children_of(sym_id):
            result.add(cid)
            result |= self._descendants(cid)
        return result


def tree_from_arrays(
    names: list[str],
    bboxes: list[list[float]],
    parent: list[int],
    edge_type: list[int],
    order: list[int],
) -> Tree:
    """Build Tree from flat arrays (names, bboxes, parent, edge_type, order)."""
    nodes = tuple(
        Node(
            Symbol(id=i, name=names[i], bbox=BBox(*bboxes[i])),
            ROOT_ID if parent[i] == -1 else parent[i],
            edge_type[i],
            order[i],
        )
        for i in range(len(names))
    )
    return Tree(nodes)
