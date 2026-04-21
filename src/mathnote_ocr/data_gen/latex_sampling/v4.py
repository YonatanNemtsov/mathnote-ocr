"""V4: Wide and flat — many siblings, long sequences, juxtaposition.

V3 was deep. V4 is the opposite: shallow but wide. Many terms at top level
and inside structures. Long numerators/denominators. Multiple children
under one parent.
"""

import random

from .symbols import (
    ARITH_OPS,
    BIGOPS,
    DIGITS,
    FUNCS,
    GREEK,
    GREEK_UPPER,
    MISC_SYMBOLS,
    RELOPS,
    UPPER,
    VARS,
    _pick_base,
    _shared_expr,
    _shared_sub_content,
    _shared_term,
)


def _atom():
    if random.random() < 0.05:
        return random.choice(MISC_SYMBOLS)
    pool = random.choices(
        [VARS, UPPER, DIGITS, GREEK, GREEK_UPPER, ARITH_OPS],
        weights=[10, 3, 8, 4, 1, 2],
    )[0]
    return random.choice(pool)


# ── Wide recursive builder ────────────────────────────────────────────


def _expr(d=0):
    return _shared_expr(d, _atom, _pick_base, _struct)


def _term(d):
    return _shared_term(d, _atom, _pick_base, _struct)


def _slot(d):
    return _expr(d + 1)


def _struct(d):
    kind = random.choices(
        ["frac", "sup", "sub", "subsup", "sqrt", "func", "bigop", "parens", "abs"],
        weights=[18, 14, 10, 8, 8, 7, 10, 8, 5],
    )[0]

    if kind == "frac":
        return "\\frac{" + _slot(d) + "}{" + _slot(d) + "}"
    if kind == "sup":
        return _pick_base() + " " + "^{" + _slot(d) + "}"
    if kind == "sub":
        return _pick_base() + "_{" + _shared_sub_content(_pick_base) + "}"
    if kind == "subsup":
        return (
            _pick_base()
            + "_{"
            + _shared_sub_content(_pick_base)
            + "}"
            + " "
            + "^{"
            + _slot(d)
            + "}"
        )
    if kind == "sqrt":
        return "\\sqrt{" + _slot(d) + "}"
    if kind == "func":
        fn = random.choice(FUNCS)
        return fn + " " + _slot(d)
    if kind == "bigop":
        op = random.choice(BIGOPS)
        v = random.random()
        if v < 0.2:
            return op + " " + _slot(d)
        if v < 0.5:
            return op + " " + "_{" + _slot(d) + "}" + " " + _slot(d)
        return op + " " + "_{" + _slot(d) + "}" + " " + "^{" + _slot(d) + "}" + " " + _slot(d)
    if kind == "parens":
        return "( " + _slot(d) + " )"
    if kind == "abs":
        return "| " + _slot(d) + " |"


def sample():
    expr = _expr(0)
    if random.random() < 0.25:
        expr = expr + " " + random.choice(RELOPS) + " " + _expr(0)
    return expr
