"""Grammar: what a well-formed reading is — for an app that says so.

The engine provides the mechanism; the app provides the rules, the rewrites
and which repairs to try. A Grammar is given to the engine like its vocabulary:

    ocr = MathOCR(config, grammar=Grammar(rule_a, rule_b, rewrites=[...], moves=my_moves))
    ocr.detect(strokes)                        # a reading is repaired
    ocr.detect(strokes, grammar=Grammar())     # or another grammar, for one call

Rules. A rule is a function (names, where) -> positions it objects to: it
sees one line of the reading — the siblings of one parent under one
relation, in reading order, named as shown (a fraction bar with nothing
above or below it is shown as a minus) — and which line that is (lines()):
"main", "sup", "sub", "num", "den", "sqrt", "upper", "lower", or "cell" (a
matrix cell). A matrix atom in a line is named "grid". A rule that takes a
third argument also gets the line's boxes, (x0, y0, x1, y1) per symbol —
for what the ink shows about the line (two neighbours stacked are no line).
A fraction's part with nothing in it is a line too (empty — a fraction bar
with nothing on either side is a minus, not a fraction), and position -1
objects to the line as a whole: to the symbol it hangs off (a fraction's
bar).

Rewrites. Rewrite(parts, into) names two symbols that may be one:
it applies where the grouper weighed the two symbols' strokes as one symbol
(they're close enough to be one) and the classifier's first reading of them
together is a target — the classifier judges the shape — with at least
the rewrite's min_confidence (default 0: any first reading). Like any
repair it is kept only if it fixes something: the reading breaks fewer
rules with the two joined.

Moves. For a symbol a rule flags, the app's moves(name, where) lists the
repairs to try, in order (default: relabel only). Two are built in:
  "relabel"    read as one of its own alternatives (most probable first)
  "elsewhere"  placed elsewhere: its link to its parent forbidden
               (mathnote_ocr.relations), its reading kept
and any other move is the app's own: a function (expr, symbol id, keep) ->
Proposal | None — the pins (and forbidden links) to re-read with. The first
move that leaves fewer violations is kept.

Repair is local: every other symbol keeps its reading (pinned, as shown)
while one place is re-read, so fixes in a long expression don't disturb each
other; a move must leave everything else as it was — labels and structure —
or it is refused, unless it says it reshapes (a Proposal with reshapes=True;
a rewrite always may: two symbols joined, what they split apart moves back).
The kept fixes are read together. The user's pins (their corrections) are
never changed.
"""

from __future__ import annotations

import inspect
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from mathnote_ocr.pin import PinnedTree, PinSymbol
from mathnote_ocr.tree_parser.tree_v2 import ROOT_ID, Edge

Rule = Callable[..., set[int]]      # (names, where) or (names, where, boxes) -> positions

WHERE = {Edge.ROOT: "main", Edge.SUP: "sup", Edge.SUB: "sub", Edge.NUM: "num", Edge.DEN: "den",
         Edge.SQRT: "sqrt", Edge.UPPER: "upper", Edge.LOWER: "lower", Edge.MATCH: "match"}


@dataclass(frozen=True)
class Rewrite:
    """Two symbols (either order) that may be one of *into* — when the
    classifier reads them together as it with at least *min_confidence*."""
    parts: tuple[str, str]
    into: tuple[str, ...]
    min_confidence: float = 0.0


def lines(expr) -> list[tuple[str, list[int]]]:
    """The reading's lines: (where, symbol ids in reading order) — the siblings
    under each parent and relation, a fraction's empty part, and each matrix
    cell (by x)."""
    return [(where, ids) for where, ids, _owner in _owned_lines(expr)]


def _owned_lines(expr) -> list[tuple[str, list[int], int | None]]:
    """lines(), with the symbol each line hangs off (None: the main line, a cell)."""
    if expr.tree is None:
        return []
    t = expr.tree
    groups: dict[tuple, list] = defaultdict(list)
    for sid, node in t.nodes.items():
        if sid == ROOT_ID:
            continue
        groups[(node.parent_id, int(node.edge_type))].append((node.order, sid))
    out = [(WHERE.get(Edge(e), "other"), [sid for _o, sid in sorted(v)], None if p == ROOT_ID else p)
           for (p, e), v in groups.items()]
    # a fraction's part with nothing in it (a bar with neither is a minus: no fraction)
    for sid, node in t.nodes.items():
        if sid == ROOT_ID or node.symbol.name != "frac_bar":
            continue
        num, den = t.children_by_edge(sid, Edge.NUM), t.children_by_edge(sid, Edge.DEN)
        if bool(num) != bool(den):
            out.append(("den" if num else "num", [], sid))
    for g in (expr.grids or {}).values():
        for row in g.cells:
            for cell in row:
                ids = [i for i in cell if i in expr.symbols]
                out.append(("cell", sorted(ids, key=lambda i: expr.symbols[i].bbox.x), None))
    return out


def _box(expr, sid) -> tuple[float, float, float, float]:
    """A line member's box: a symbol's, a matrix atom's, or a pinned group's
    (the tree's expr node, around the pinned symbols)."""
    if sid in expr.symbols:
        b = expr.symbols[sid].bbox
    elif sid in (expr.grids or {}):
        b = expr.grids[sid].bbox
    else:
        b = expr.tree.nodes[sid].symbol.bbox
    return (b.x, b.y, b.x + b.w, b.y + b.h)


def _takes_boxes(rule) -> bool:
    try:
        return len(inspect.signature(rule).parameters) >= 3
    except (TypeError, ValueError):
        return False


def _name(expr, sid) -> str:
    """A symbol's name as rules see it: as shown — the tree's name (a minus
    the parser made a fraction bar is one), and a fraction bar with neither
    a numerator nor a denominator is shown, and seen, as a minus."""
    if sid not in expr.symbols:
        return "grid"                          # a matrix atom: an operand
    t = expr.tree
    name = t.nodes[sid].symbol.name if t is not None and sid in t.nodes else expr.symbols[sid].name
    if name == "frac_bar" and not (t.children_by_edge(sid, Edge.NUM) or t.children_by_edge(sid, Edge.DEN)):
        return "-"
    return name


BUILT_IN_MOVES = ("relabel", "elsewhere")


@dataclass(frozen=True)
class Proposal:
    """An app's repair for a flagged symbol: re-read with these *pins* (and
    *forbid* links, mathnote_ocr.relations), re-deciding the strokes *taken*
    (the flagged symbol's are always included). Unless *reshapes*, everything
    outside them must read as before."""
    pins: tuple
    taken: frozenset = frozenset()
    forbid: tuple = ()
    reshapes: bool = False


def _relabel_only(name: str, where: str) -> Sequence:
    return ("relabel",)


class Grammar:
    def __init__(self, *rules: Rule, rewrites: Sequence[Rewrite] = (),
                 moves: Callable[[str, str], Sequence] = _relabel_only):
        self.rules = rules
        self.rewrites = tuple(rewrites)
        self.moves = moves

    def violations(self, expr) -> set[int]:
        """The symbol ids some rule flags (in some line of the reading)."""
        bad = set()
        for where, ids, owner in _owned_lines(expr):
            names = [_name(expr, i) for i in ids]
            boxes = None
            for rule in self.rules:
                if _takes_boxes(rule):
                    if boxes is None:
                        boxes = [_box(expr, i) for i in ids]
                    flagged = rule(names, where, boxes)
                else:
                    flagged = rule(names, where)
                for k in flagged:
                    sid = owner if k == -1 else ids[k]
                    if sid is not None and sid in expr.symbols:
                        bad.add(sid)
        return bad


def _strokes(sym) -> frozenset[int]:
    return frozenset(st.id for st in sym.strokes)


def _frozen(expr, free: frozenset[int]) -> list[PinnedTree]:
    """Every symbol's reading, pinned as shown (a lone fraction bar as the minus
    it is shown as — pinned as a fraction bar it would invite a fraction) —
    but those with strokes in *free*."""
    return [PinnedTree.build([PinSymbol(_name(expr, sid), list(s.strokes))]) for sid, s in expr.symbols.items()
            if not (_strokes(s) & free)]


def _rest(expr, region: frozenset[int]):
    """The reading outside *region* (stroke ids): its symbols (label, strokes)
    and the relations between them."""
    # names as shown: the parser's promotions (a minus as a fraction bar) count
    syms = {sid: (_name(expr, sid), _strokes(s)) for sid, s in expr.symbols.items() if not (_strokes(s) & region)}
    rel = {(syms[n.parent_id], syms[sid], int(n.edge_type)) for sid, n in expr.tree.nodes.items()
           if sid in syms and n.parent_id in syms}
    return set(syms.values()), rel


def _joins(expr, grammar: Grammar, keep: frozenset[int], classified) -> list[tuple[str, list]]:
    """The rewrites that apply: [(symbol, strokes)], no symbol in two."""
    found = []
    # named as rules see them: a lone fraction bar is a minus
    syms = [(_name(expr, sid), s) for sid, s in expr.symbols.items() if not (_strokes(s) & keep)]
    for i, (na, a) in enumerate(syms):
        for nb, b in syms[i + 1:]:
            for rw in grammar.rewrites:
                if sorted((na, nb)) != sorted(rw.parts):
                    continue
                r = classified(_strokes(a) | _strokes(b))
                if r is not None and r.symbol in rw.into and r.confidence >= rw.min_confidence:
                    found.append((r.confidence, r.symbol, a, b))
    out, used = [], set()
    for _c, name, a, b in sorted(found, key=lambda f: -f[0]):
        if id(a) in used or id(b) in used:
            continue
        used |= {id(a), id(b)}
        out.append((name, list(a.strokes) + list(b.strokes)))
    return out


def repair(expr, read: Callable[[list[PinnedTree]], object], grammar: Grammar,
           keep: frozenset[int] = frozenset(), classified=None, per_symbol: int = 3):
    """*expr*, repaired locally (see the module doc); *expr* itself when
    nothing applies or helps.

    *read(pins, forbid=links)* reads the same ink again with the extra pins,
    and the given links forbidden (mathnote_ocr.relations) — through the
    caller's whole pipeline. Symbols with strokes in *keep* (the user's pins,
    marked regions) are never changed, nor pinned again. *classified(stroke
    ids)* is the grouper's reading of those strokes as one symbol (under the
    vocabulary), or None if it never weighed them as one.
    """
    # Rewrites: split symbols joined — each kept only if it fixes something
    # (the reading breaks fewer rules with it), like any other repair
    if grammar.rewrites and classified is not None:
        kept: list[tuple[str, list]] = []
        for join in _joins(expr, grammar, keep, classified):
            trial = kept + [join]
            free = keep | frozenset(st.id for _n, strokes in trial for st in strokes)
            alt = read(_frozen(expr, free) + [PinnedTree.build([PinSymbol(n, s)]) for n, s in trial])
            if alt is not None and alt.tree is not None \
                    and len(grammar.violations(alt)) < len(grammar.violations(expr)):
                kept, expr = trial, alt

    # Moves: each flagged symbol on its own, the rest as read
    bad = grammar.violations(expr)
    if not bad:
        return expr
    n_bad = len(bad)
    fixes = []
    for sid in sorted(bad):
        sym = expr.symbols.get(sid)
        if sym is None or _strokes(sym) & keep:
            continue
        node = expr.tree.nodes[sid]
        frozen = _frozen(expr, keep | _strokes(sym))
        rest = _rest(expr, _strokes(sym))

        def fewer(alt):
            return alt is not None and alt.tree is not None and len(grammar.violations(alt)) < n_bad

        def good(alt):
            return fewer(alt) and _rest(alt, _strokes(sym)) == rest

        def relabel():
            for name, _p in [(n, p) for n, p in (sym.alternatives or []) if n != sym.name][:per_symbol]:
                pin = PinnedTree.build([PinSymbol(name, list(sym.strokes))])
                alt = read(frozen + [pin])
                if good(alt):
                    return [pin], [], _strokes(sym), alt
            return None

        def elsewhere():
            parent = expr.symbols.get(node.parent_id)
            if parent is None or node.parent_id == ROOT_ID:
                return None
            link = (_strokes(sym), _strokes(parent), int(node.edge_type))
            pin = PinnedTree.build([PinSymbol(_name(expr, sid), list(sym.strokes))])
            alt = read(frozen + [pin], forbid=[link])
            return ([pin], [link], _strokes(sym), alt) if good(alt) else None

        def proposed(move):
            p = move(expr, sid, keep)
            if p is None:
                return None
            taken = frozenset(p.taken) | _strokes(sym)
            alt = read(_frozen(expr, keep | taken) + list(p.pins), forbid=list(p.forbid))
            ok = fewer(alt) and (p.reshapes or _rest(alt, taken) == _rest(expr, taken))
            return (list(p.pins), list(p.forbid), taken, alt) if ok else None

        built_in = {"relabel": relabel, "elsewhere": elsewhere}
        for move in grammar.moves(_name(expr, sid), WHERE.get(Edge(node.edge_type), "other")):
            if not callable(move) and move not in built_in:
                raise ValueError(f"unknown move {move!r}: built in are {BUILT_IN_MOVES}, else pass a function")
            try:
                found = proposed(move) if callable(move) else built_in[move]()
            except ValueError:      # a move that can't be pinned here (strokes it can't claim): no fix
                found = None
            if found is not None:
                fixes.append(found)
                break
    if not fixes:
        return expr
    if len(fixes) == 1:
        return fixes[0][3]
    # the fixes together — unless two claim the same strokes, or together they reshape the rest
    taken = [t for _p, _f, t, _a in fixes]
    fixed = frozenset().union(*taken)
    if sum(len(t) for t in taken) == len(fixed):
        alt = read(_frozen(expr, keep | fixed) + [p for ps, _f, _t, _a in fixes for p in ps],
                   forbid=[link for _p, fs, _t, _a in fixes for link in fs])
        if alt is not None and alt.tree is not None and len(grammar.violations(alt)) < n_bad \
                and _rest(alt, fixed) == _rest(expr, fixed):
            return alt
    return min((a for _p, _f, _t, a in fixes), key=lambda a: len(grammar.violations(a)))    # the best one alone
