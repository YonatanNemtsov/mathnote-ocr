"""PinnedTree: a constraint subtree the OCR pipeline preserves.

A pin captures a set of labelled symbols (each owning one or more
strokes) and the internal edges between them. The pipeline enforces:
  1. Each pinned symbol's strokes group with the given label.
  2. Each internal edge in the pin is preserved in the output tree.
  3. The pinned symbols form a connected subtree. When internal edges
     leave the pin as a forest (e.g. multi-top-level selection), a
     synthetic ``expr`` node is inserted as their common parent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mathnote_ocr.bbox import BBox
from mathnote_ocr.tree_parser.tree_v2 import (
    ROOT_ID,
    Edge,
    EdgeType,
    Node,
    Symbol,
    Tree,
)

if TYPE_CHECKING:
    from mathnote_ocr.engine.stroke import Stroke


@dataclass
class PinSymbol:
    """One symbol inside a PinnedTree: a label and the strokes it owns."""

    label: str
    strokes: list[Stroke]


@dataclass
class PinEdge:
    """An internal edge in a PinnedTree, between two PinSymbols by index."""

    parent: int
    child: int
    edge: EdgeType
    order: int = 0


class PinnedTree(Tree):
    """A constraint subtree for the OCR pipeline.

    Structurally a Tree; the type tag distinguishes it from a parsed
    output tree. Construct via :meth:`PinnedTree.build`.
    """

    @classmethod
    def build(
        cls,
        symbols: Sequence[PinSymbol],
        edges: Sequence[PinEdge] = (),
    ) -> PinnedTree:
        """Construct a pin from labelled symbols and internal edges.

        Indices in ``edges`` reference positions in ``symbols``. Edges
        may leave the pin as a forest (multiple local roots).

        Raises:
            ValueError: empty symbols, empty label, duplicate strokes,
                edge index out of range, multiple parents, cycles, or
                use of Edge.ROOT in user edges.
        """
        if not symbols:
            raise ValueError("pin must have at least one symbol")

        seen_strokes: set[int] = set()
        for i, ps in enumerate(symbols):
            if not ps.label:
                raise ValueError(f"symbol {i}: label must be non-empty")
            if not ps.strokes:
                raise ValueError(f"symbol {i}: must own at least one stroke")
            for st in ps.strokes:
                if st.id in seen_strokes:
                    raise ValueError(f"stroke {st.id} appears in multiple symbols")
                seen_strokes.add(st.id)

        N = len(symbols)
        children_seen: set[int] = set()
        norm_edges: list[PinEdge] = []
        for e in edges:
            if not (0 <= e.parent < N and 0 <= e.child < N):
                raise ValueError(f"edge {e}: index out of range [0, {N})")
            if e.parent == e.child:
                raise ValueError(f"edge {e}: self-loop")
            if e.edge == Edge.ROOT:
                raise ValueError(f"edge {e}: Edge.ROOT is reserved for the pin root")
            if e.child in children_seen:
                raise ValueError(f"symbol {e.child} has multiple parents")
            children_seen.add(e.child)
            norm_edges.append(e)

        roots = [i for i in range(N) if i not in children_seen]
        if not roots:
            raise ValueError("pin has no root — likely a cycle in edges")
        adj: dict[int, list[int]] = {}
        for e in norm_edges:
            adj.setdefault(e.parent, []).append(e.child)
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

        edge_info = {e.child: (e.parent, e.edge, e.order) for e in norm_edges}
        root_set = set(roots)
        nodes: list[Node] = []
        for i, ps in enumerate(symbols):
            stroke_ids = tuple(s.id for s in ps.strokes)
            bbox = BBox.union_all([s.bbox for s in ps.strokes])
            sym = Symbol(id=i, name=ps.label, bbox=bbox, stroke_ids=stroke_ids)
            if i in root_set:
                nodes.append(Node(sym, ROOT_ID, Edge.ROOT, 0))
            else:
                p, et, order = edge_info[i]
                nodes.append(Node(sym, p, et, order))

        return cls(tuple(nodes))