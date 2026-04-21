"""Expression — result of detection.

Three joined concerns:
  - strokes (input)    — list, indexed by stroke id
  - symbols (detection) — dict keyed by symbol id
  - tree (structure)   — tree_v2.Tree, keyed by symbol id

Symbol.id and tree node id are the same integer.

Expression is immutable by convention: rename() returns a new Expression.
Structural sharing is used throughout (tree_v2 shares unchanged nodes; the
new symbols dict is shallow-copied, unchanged Symbol objects are reused).

An "empty" Expression is returned when nothing was detected. Use
``if expr:`` or ``len(expr)`` to check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Iterator

from mathnote_ocr.bbox import BBox
from mathnote_ocr.engine.stroke import Stroke
from mathnote_ocr.tree_parser.tree_latex import tree_to_latex
from mathnote_ocr.tree_parser.tree_v2 import ROOT_ID

if TYPE_CHECKING:
    from mathnote_ocr.tree_parser.tree_v2 import Tree


@dataclass(frozen=True)
class Symbol:
    """A detected symbol. Frozen — construct new ones, don't mutate."""

    id: int
    name: str
    bbox: BBox
    strokes: list[Stroke]
    confidence: float = 1.0
    alternatives: list[tuple[str, float]] = field(default_factory=list)


class Expression:
    """A recognized math expression."""

    strokes: list[Stroke]
    symbols: dict[int, Symbol]
    tree: Tree | None
    confidence: float
    alternatives: list[Expression]

    def __init__(
        self,
        strokes: list[Stroke],
        symbols: dict[int, Symbol],
        tree: Tree | None,
        confidence: float = 0.0,
        alternatives: list[Expression] | None = None,
    ) -> None:
        self.strokes = strokes
        self.symbols = symbols
        self.tree = tree
        self.confidence = confidence
        self.alternatives = alternatives or []

    # ── Derived ──────────────────────────────────────────────────────

    @cached_property
    def latex(self) -> str:
        return tree_to_latex(self.tree) if self.tree else ""

    # ── Query ────────────────────────────────────────────────────────

    def __bool__(self) -> bool:
        return len(self.symbols) > 0

    def __len__(self) -> int:
        return len(self.symbols)

    def __iter__(self) -> Iterator[Symbol]:
        return iter(self.symbols.values())

    def __repr__(self) -> str:
        return f"Expression(latex={self.latex!r}, n_symbols={len(self.symbols)})"

    # ── Mutations (return new Expression) ────────────────────────────

    def rename(self, sym_id: int, new_name: str) -> Expression:
        """Return a new Expression with one symbol renamed.

        Does NOT re-run the pipeline. Safe only for structurally compatible
        renames (x → y, 0 → o). For structural changes (- → frac_bar),
        re-run with ``ocr.detect(strokes, hints={...})``.
        """
        old = self.symbols[sym_id]
        new_sym = Symbol(
            old.id, new_name, old.bbox, old.strokes, old.confidence, old.alternatives
        )
        new_symbols = {**self.symbols, sym_id: new_sym}
        new_tree = self.tree.rename_node(sym_id, new_name) if self.tree else None
        return Expression(self.strokes, new_symbols, new_tree, self.confidence)

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        stroke_idx = {id(s): i for i, s in enumerate(self.strokes)}
        tree_rows = []
        if self.tree is not None:
            for sid, node in self.tree.nodes.items():
                if sid == ROOT_ID:
                    continue
                tree_rows.append(
                    {
                        "id": sid,
                        "parent": node.parent_id,
                        "edge_type": int(node.edge_type),
                        "order": node.order,
                    }
                )
        return {
            "latex": self.latex,
            "confidence": round(self.confidence, 4),
            "symbols": [
                {
                    "id": s.id,
                    "name": s.name,
                    "bbox": {"x": s.bbox.x, "y": s.bbox.y, "w": s.bbox.w, "h": s.bbox.h},
                    "stroke_ids": [stroke_idx[id(st)] for st in s.strokes],
                    "confidence": round(s.confidence, 4),
                    "alternatives": [
                        {"name": n, "confidence": round(c, 4)} for n, c in s.alternatives
                    ],
                }
                for s in self.symbols.values()
            ],
            "tree": tree_rows,
        }


def empty_expression() -> Expression:
    """Construct an empty Expression (nothing detected)."""
    return Expression(strokes=[], symbols={}, tree=None, confidence=0.0)
