"""Math expression tree — persistent, immutable.

Symbol: id, name, bbox.
Node: symbol + parent pointer. Frozen.
Tree: tuple of nodes. Mutations return new trees sharing unchanged nodes.

The tree always has a virtual root node (ROOT_ID) whose children are
the top-level expression symbols. Every node's parent is a real node
in the tree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from typing import TYPE_CHECKING, TypeAlias

from mathnote_ocr.bbox import BBox

if TYPE_CHECKING:
    from mathnote_ocr.engine.stroke import Stroke


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
    stroke_ids: tuple[int, ...] = ()


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
        """Append *node* to the tree. Caller must ensure its id is unique
        and its parent_id already exists in the tree."""
        return Tree(self._nodes + (node,), self.root)

    def remove_node(self, sym_id: SymbolId) -> Tree:
        """Remove *sym_id* **and all of its descendants**. Siblings and
        ancestors are untouched."""
        to_remove = self._descendants(sym_id) | {sym_id}
        return Tree(tuple(n for n in self._nodes if n.symbol.id not in to_remove), self.root)

    def move_node(
        self,
        sym_id: SymbolId,
        new_parent_id: SymbolId,
        edge_type: EdgeType,
        order: SiblingOrder = 0,
    ) -> Tree:
        """Re-parent *sym_id* under *new_parent_id* with the given edge
        and sibling order. The node's own descendants are carried along
        unchanged. Caller must ensure no cycle is introduced."""
        return Tree(
            tuple(
                Node(n.symbol, new_parent_id, edge_type, order) if n.symbol.id == sym_id else n
                for n in self._nodes
            ),
            self.root,
        )

    def rename_node(self, sym_id: SymbolId, new_name: str) -> Tree:
        """Replace the *name* of the symbol at *sym_id*. Bbox, id,
        stroke_ids, and tree structure are preserved."""
        return Tree(
            tuple(
                Node(
                    Symbol(n.symbol.id, new_name, n.symbol.bbox, n.symbol.stroke_ids),
                    n.parent_id,
                    n.edge_type,
                    n.order,
                )
                if n.symbol.id == sym_id
                else n
                for n in self._nodes
            ),
            self.root,
        )

    # ── Factories ────────────────────────────────────────────────────

    @classmethod
    def pin_spec(
        cls,
        strokes: dict[int, Stroke],
        symbols: Sequence[tuple[str, Sequence[int]]],
        edges: Sequence[tuple[int, int, EdgeType] | tuple[int, int, EdgeType, int]] = (),
    ) -> Tree:
        """Build a Tree describing a pin (constraint subtree).

        A pin captures a set of (label, stroke_ids) and the internal edges
        between them (when their parent is also in the pin). The OCR
        pipeline enforces:
          1. Each pinned symbol's strokes group with the given label.
          2. Each internal edge in the pin is preserved in the output tree.
          3. The pinned symbols form a **connected subtree** — if internal
             edges leave the pin as a forest (e.g. multi-top-level
             selection), a synthetic ``expr`` node is inserted as their
             common parent.

        Args:
            strokes: Mapping from stroke id to Stroke. Used to compute
                each symbol's bbox by union over its strokes.
            symbols: List of ``(label, stroke_ids)`` tuples. Each entry's
                position in this list becomes its local id in the tree
                (and serves as its index in ``edges``).
            edges: Internal tree edges as ``(parent, child, edge)`` or
                ``(parent, child, edge, order)`` tuples. Indices reference
                positions in ``symbols``. May leave the pin as a forest
                (multiple local roots).

        Returns:
            A Tree with ``len(symbols)`` non-root nodes, ids 0..N-1.

        Raises:
            ValueError: empty symbols, missing/duplicate stroke refs,
                empty label, edge index out of range, multiple parents,
                cycles, or use of Edge.ROOT in user edges.
        """
        if not symbols:
            raise ValueError("pin must have at least one symbol")

        seen_strokes: set[int] = set()
        for i, (label, sids) in enumerate(symbols):
            if not label:
                raise ValueError(f"symbol {i}: label must be non-empty")
            for sid in sids:
                if sid not in strokes:
                    raise ValueError(f"symbol {i}: stroke {sid} not in strokes")
                if sid in seen_strokes:
                    raise ValueError(f"stroke {sid} appears in multiple symbols")
                seen_strokes.add(sid)

        norm_edges: list[tuple[int, int, EdgeType, int]] = []
        children_seen: set[int] = set()
        N = len(symbols)
        for e in edges:
            if len(e) == 3:
                p, c, et = e
                order = 0
            elif len(e) == 4:
                p, c, et, order = e
            else:
                raise ValueError(f"edge {e}: expected 3 or 4 elements")
            if not (0 <= p < N and 0 <= c < N):
                raise ValueError(f"edge {e}: index out of range [0, {N})")
            if p == c:
                raise ValueError(f"edge {e}: self-loop")
            if et == Edge.ROOT:
                raise ValueError(f"edge {e}: Edge.ROOT is reserved for the pin root")
            if c in children_seen:
                raise ValueError(f"symbol {c} has multiple parents")
            children_seen.add(c)
            norm_edges.append((p, c, et, order))

        # Cycle / connectivity check — pin may be a forest, but each component
        # must be acyclic and reachable from a local root.
        roots = [i for i in range(N) if i not in children_seen]
        if not roots:
            raise ValueError("pin has no root — likely a cycle in edges")
        adj: dict[int, list[int]] = {}
        for p, c, _, _ in norm_edges:
            adj.setdefault(p, []).append(c)
        visited: set[int] = set()
        for r in roots:
            stack = [r]
            while stack:
                cur = stack.pop()
                if cur in visited:
                    raise ValueError(f"cycle through symbol {cur}")
                visited.add(cur)
                stack.extend(adj.get(cur, []))
        if len(visited) != N:
            raise ValueError(f"symbols {set(range(N)) - visited} not reachable from any root")

        edge_info = {c: (p, et, o) for p, c, et, o in norm_edges}
        root_set = set(roots)
        nodes: list[Node] = []
        for i, (label, sids) in enumerate(symbols):
            sid_t = tuple(sids)
            bbox = BBox.union_all([strokes[sid].bbox for sid in sid_t])
            sym = Symbol(id=i, name=label, bbox=bbox, stroke_ids=sid_t)
            if i in root_set:
                nodes.append(Node(sym, ROOT_ID, Edge.ROOT, 0))
            else:
                p, et, order = edge_info[i]
                nodes.append(Node(sym, p, et, order))

        return cls(tuple(nodes))

    # ── Traversal ────────────────────────────────────────────────────

    def walk(self, sym_id: SymbolId) -> tuple[SymbolId, ...]:
        """All ids in the subtree rooted at *sym_id* (depth-first, self first)."""
        result: list[SymbolId] = [sym_id]
        for cid, _, _ in self.children_of(sym_id):
            result.extend(self.walk(cid))
        return tuple(result)

    def path(self, sym_id: SymbolId) -> tuple[tuple[SymbolId, EdgeType, SiblingOrder], ...]:
        """Path from root to symbol: ((id, edge_type, order), ...). Root itself returns ()."""
        if sym_id == self.root:
            return ()
        node = self.nodes[sym_id]
        entry = (sym_id, node.edge_type, node.order)
        if node.parent_id == self.root:
            return (entry,)
        return self.path(node.parent_id) + (entry,)

    def to_latex(self) -> str:
        """Render this tree to a LaTeX string."""
        from mathnote_ocr.tree_parser.tree_latex import tree_to_latex
        return tree_to_latex(self)

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
        return hash(
            frozenset(
                (n.symbol.id, n.symbol.name, n.parent_id, n.edge_type, n.order)
                for n in self._nodes
                if n.symbol.id != self.root
            )
        )

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
