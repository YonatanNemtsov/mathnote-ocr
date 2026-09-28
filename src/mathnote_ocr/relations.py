"""Relations: which relations a reading may have — for an app that says so.

    ocr = MathOCR(config, relations=Relations(exclude={"sub"}))   # no subscripts

Like an excluded symbol, an excluded relation gets no votes in the tree
parser: it reads the most probable tree among the allowed ones (a hard
constraint inside the search — no re-reading).

Repair (mathnote_ocr.grammar) uses the same for one link: forbid(child
strokes, parent strokes, relation) — "this minus is not u's superscript" —
so the symbol is placed elsewhere and the rest is read as before.

Relation names: "sup", "sub", "num", "den", "sqrt", "upper", "lower".
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from mathnote_ocr.tree_parser.tree_v2 import Edge

EDGES = {"sup": Edge.SUP, "sub": Edge.SUB, "num": Edge.NUM, "den": Edge.DEN,
         "sqrt": Edge.SQRT, "upper": Edge.UPPER, "lower": Edge.LOWER}

Link = tuple[frozenset[int], frozenset[int], int]     # (child strokes, parent strokes, edge)


@dataclass(frozen=True)
class Relations:
    exclude: frozenset[str] = frozenset()
    forbid: tuple[Link, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "exclude", frozenset(self.exclude))
        unknown = self.exclude - EDGES.keys()
        if unknown:
            raise ValueError(f"unknown relations {sorted(unknown)}; known: {sorted(EDGES)}")

    def __bool__(self) -> bool:
        return bool(self.exclude or self.forbid)

    def forbidding(self, links: Iterable[Link]) -> Relations:
        """These relations, and the given links forbidden too."""
        return Relations(self.exclude, self.forbid + tuple(links))

    def restrict(self, evidence: dict, symbols: Sequence) -> dict:
        """The parser's evidence with the excluded relations' and forbidden
        links' votes removed. *symbols*: the parser's symbols, in evidence
        order (each with stroke_ids)."""
        if not self:
            return evidence
        votes = evidence["parent_votes"].clone()
        for name in self.exclude:
            votes[:, :, int(EDGES[name])] = 0
        if self.forbid:
            pos = {frozenset(s.stroke_ids): i for i, s in enumerate(symbols)}
            for child, parent, edge in self.forbid:
                i, j = pos.get(child), pos.get(parent)
                if i is not None and j is not None:
                    votes[i, j, int(edge)] = 0
        return {**evidence, "parent_votes": votes}
