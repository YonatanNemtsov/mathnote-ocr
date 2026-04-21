"""V3: Structure-dense, shorter expressions — higher nesting ratio.

V1/V2 gaps: many short/flat expressions, not enough deep nesting.
Approach: higher structure probability, fewer top-level terms, deeper depth limit.
Every expression guaranteed to have at least one structure.
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


# ── High-density recursive builder ────────────────────────────────────


def _expr(d=0):
    return _shared_expr(d, _atom, _pick_base, _struct)


def _term(d):
    return _shared_term(d, _atom, _pick_base, _struct)


def _slot(d):
    return _expr(d + 1)


def _struct(d):
    kind = random.choices(
        ["frac", "sup", "sub", "subsup", "sqrt", "func", "bigop", "parens", "binom", "abs"],
        weights=[20, 15, 10, 8, 10, 8, 12, 6, 4, 5],
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
        if random.random() < 0.2:
            return fn + " " + "^{" + random.choice(DIGITS) + "}" + " " + _slot(d)
        return fn + " " + _slot(d)
    if kind == "bigop":
        op = random.choice(BIGOPS)
        v = random.random()
        if v < 0.15:
            return op + " " + _slot(d)
        if v < 0.35:
            return op + " " + "_{" + _slot(d) + "}" + " " + _slot(d)
        if v < 0.8:
            return op + " " + "_{" + _slot(d) + "}" + " " + "^{" + _slot(d) + "}" + " " + _slot(d)
        return op + " " + "^{" + _slot(d) + "}" + " " + _slot(d)
    if kind == "parens":
        return "( " + _slot(d) + " )"
    if kind == "binom":
        return "\\binom{" + _slot(d) + "}{" + _slot(d) + "}"
    if kind == "abs":
        return "| " + _slot(d) + " |"


def sample():
    expr = _expr(0)
    if random.random() < 0.2:
        expr = expr + " " + random.choice(RELOPS) + " " + _expr(0)
    return expr
