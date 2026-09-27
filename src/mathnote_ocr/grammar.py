"""Grammar: what a well-formed reading is — for an app that says so.

The engine provides the mechanism; the app provides the rules. A rule is
any function from a line's symbol names to the positions it objects to;
a Grammar is the app's list of rules, given to the engine like its
vocabulary:

    ocr = MathOCR(config, grammar=Grammar(rule_a, rule_b))
    ocr.detect(strokes)                        # a broken reading is repaired
    ocr.detect(strokes, grammar=Grammar())     # or another grammar, for one call

A reading that breaks a rule is repaired inside detect — the most probable
grammatical reading is kept (a hard constraint), found by re-reading with
one flagged symbol changed to one of its own alternatives. Pins (the user's
corrections) are never changed.

Rules see the reading's lines: the siblings of one parent under one
relation (the main line, a numerator, a superscript …) in reading order,
and each matrix cell (lines()). A matrix atom in a line is named "grid".
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

from mathnote_ocr.pin import PinnedTree, PinSymbol
from mathnote_ocr.tree_parser.tree_v2 import ROOT_ID

Rule = Callable[[Sequence[str]], set[int]]


def lines(expr) -> list[list[int]]:
    """The reading's lines, as symbol ids in reading order: the siblings under
    each parent and relation, and each matrix cell (by x)."""
    if expr.tree is None:
        return []
    groups: dict[tuple, list] = defaultdict(list)
    for sid, node in expr.tree.nodes.items():
        if sid == ROOT_ID:
            continue
        groups[(node.parent_id, int(node.edge_type))].append((node.order, sid))
    out = [[sid for _o, sid in sorted(v)] for v in groups.values()]
    for g in (expr.grids or {}).values():
        for row in g.cells:
            for cell in row:
                ids = [i for i in cell if i in expr.symbols]
                out.append(sorted(ids, key=lambda i: expr.symbols[i].bbox.x))
    return out


def _name(expr, sid) -> str:
    return expr.symbols[sid].name if sid in expr.symbols else "grid"    # a matrix atom: an operand


class Grammar:
    def __init__(self, *rules: Rule):
        self.rules = rules

    def violations(self, expr) -> set[int]:
        """The symbol ids some rule flags (in some line of the reading)."""
        bad = set()
        for ids in lines(expr):
            names = [_name(expr, i) for i in ids]
            for rule in self.rules:
                bad |= {ids[k] for k in rule(names) if ids[k] in expr.symbols}
        return bad


def repair(expr, read: Callable[[list[PinnedTree]], object], grammar: Grammar,
           keep: frozenset[int] = frozenset(), max_tries: int = 8, per_symbol: int = 3):
    """*expr* if it is grammatical; else the most probable grammatical re-reading
    with one flagged symbol changed to one of its alternatives (tried in order
    of the alternative's probability, at most *max_tries*); else *expr*.

    *read(extra_pins)* reads the same ink again with the extra pins — through
    the caller's whole pipeline. Symbols with strokes in *keep* are never changed.
    """
    bad = grammar.violations(expr)
    if not bad:
        return expr
    options = []
    for sid in bad:
        sym = expr.symbols.get(sid)
        if sym is None or any(st.id in keep for st in sym.strokes):
            continue
        alts = [(p, n) for n, p in (sym.alternatives or []) if n != sym.name][:per_symbol]
        options += [(p, sid, n) for p, n in alts]
    options.sort(key=lambda o: -o[0])
    for _p, sid, name in options[:max_tries]:
        sym = expr.symbols[sid]
        alt = read([PinnedTree.build([PinSymbol(name, list(sym.strokes))])])
        if alt is not None and alt.tree is not None and not grammar.violations(alt):
            return alt
    return expr
