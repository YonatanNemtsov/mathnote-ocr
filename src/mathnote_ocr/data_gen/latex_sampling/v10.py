"""V10: Kitchen sink — randomly picks a generation *strategy* per sample.

Each sample randomly selects one of several strategies:
- deep (like v6)
- wide (like v4)
- calculus (like v7)
- discrete (like v8)
- balanced (like v1)

This ensures maximum variety — every sample could look completely different.
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
    SET_OPS,
    UPPER,
    VARS,
    _pick_base,
    _shared_expr,
    _shared_term,
)


def _atom():
    if random.random() < 0.05:
        return random.choice(MISC_SYMBOLS)
    pool = random.choices(
        [VARS, UPPER, DIGITS, GREEK, GREEK_UPPER, ARITH_OPS],
        weights=[10, 3, 7, 4, 1, 2],
    )[0]
    return random.choice(pool)


def _var():
    return random.choice(VARS)


# ── Strategy names (used to dispatch _struct) ────────────────────────

STRATEGY_NAMES = ["deep", "wide", "balanced", "calculus", "discrete"]

_strat = "balanced"  # set per-sample in sample()


def _expr(d=0):
    return _shared_expr(d, _atom, _pick_base, _struct)


def _term(d):
    return _shared_term(d, _atom, _pick_base, _struct)


def _slot(d):
    return _expr(d + 1)


def _struct(d):
    if _strat == "calculus":
        return _struct_calculus(d)
    if _strat == "discrete":
        return _struct_discrete(d)
    return _struct_standard(d)


def _struct_standard(d):
    kind = random.choices(
        ["frac", "sup", "sub", "subsup", "sqrt", "func", "bigop", "parens", "binom", "abs"],
        weights=[18, 14, 10, 7, 10, 7, 10, 7, 4, 5],
    )[0]
    return _build(kind, d)


def _struct_calculus(d):
    kind = random.choices(
        [
            "frac",
            "sup",
            "sub",
            "subsup",
            "sqrt",
            "func",
            "integral",
            "sum_prod",
            "deriv",
            "partial",
            "lim",
            "parens",
            "nabla",
            "pm",
        ],
        weights=[12, 10, 8, 5, 6, 6, 10, 8, 6, 5, 5, 5, 3, 3],
    )[0]
    return _build(kind, d)


def _struct_discrete(d):
    kind = random.choices(
        [
            "frac",
            "sup",
            "sub",
            "subsup",
            "sqrt",
            "func",
            "parens",
            "binom",
            "abs",
            "set_op",
            "element",
            "subset_rel",
            "arrow",
            "inequality",
            "factorial",
            "ldots",
            "bigop",
        ],
        weights=[10, 10, 8, 5, 5, 5, 5, 8, 4, 6, 5, 4, 3, 5, 4, 3, 6],
    )[0]
    return _build(kind, d)


def _build(kind, d):
    s = lambda: _slot(d)

    if kind == "frac":
        return "\\frac{" + s() + "}{" + s() + "}"
    if kind == "sup":
        return _pick_base() + "^{" + s() + "}"
    if kind == "sub":
        return _pick_base() + "_{" + s() + "}"
    if kind == "subsup":
        return _pick_base() + "_{" + s() + "}^{" + s() + "}"
    if kind == "sqrt":
        return "\\sqrt{" + s() + "}"
    if kind == "func":
        fn = random.choice(FUNCS)
        if random.random() < 0.2:
            return fn + "^{" + random.choice(DIGITS) + "} " + s()
        return fn + " " + s()
    if kind == "bigop":
        op = random.choice(BIGOPS)
        v = random.random()
        if v < 0.2:
            return op + " " + s()
        if v < 0.5:
            return op + "_{" + s() + "}" + s()
        return op + "_{" + s() + "}^{" + s() + "}" + s()
    if kind == "parens":
        return "(" + s() + ")"
    if kind == "binom":
        return "\\binom{" + s() + "}{" + s() + "}"
    if kind == "abs":
        return "| " + s() + " |"

    # ── Calculus ──
    if kind == "integral":
        x = _var()
        v = random.random()
        if v < 0.3:
            return "\\int " + s() + "d " + x
        if v < 0.7:
            return "\\int_{" + s() + "}^{" + s() + "}" + s() + "d " + x
        return "\\int_{" + _atom() + "}^{\\infty}" + s() + "d " + x
    if kind == "sum_prod":
        op = random.choice(["\\sum", "\\prod"])
        v = random.random()
        if v < 0.3:
            return op + " " + s()
        if v < 0.6:
            return op + "_{" + _var() + "=" + _atom() + "}" + s()
        hi = "\\infty" if random.random() < 0.3 else _atom()
        return op + "_{" + _var() + "=" + _atom() + "}^{" + hi + "}" + s()
    if kind == "deriv":
        x, y = _var(), _var()
        if random.random() < 0.6:
            return "\\frac{" + "d " + y + "}{" + "d " + x + "}"
        return "\\frac{" + "d^{2} " + y + "}{" + "d " + x + "^{2}}"
    if kind == "partial":
        x, y = _var(), _var()
        if random.random() < 0.5:
            return "\\frac{\\partial " + y + "}{\\partial " + x + "}"
        z = _var()
        return "\\frac{\\partial ^{2}" + y + "}{\\partial " + x + "\\partial " + z + "}"
    if kind == "lim":
        x = _var()
        target = "\\infty" if random.random() < 0.4 else _atom()
        return "\\lim_{" + x + "\\rightarrow " + target + "}" + s()
    if kind == "nabla":
        return ("\\nabla " + _var()) if random.random() < 0.6 else ("\\nabla ^{2}" + _var())
    if kind == "pm":
        return _atom() + "\\pm " + s()

    # ── Discrete ──
    if kind == "set_op":
        return random.choice(UPPER) + " " + random.choice(SET_OPS) + " " + random.choice(UPPER)
    if kind == "element":
        return _atom() + "\\in " + random.choice(UPPER)
    if kind == "subset_rel":
        return random.choice(UPPER) + "\\subset " + random.choice(UPPER)
    if kind == "arrow":
        if random.random() < 0.4:
            return _var() + ":" + random.choice(UPPER) + "\\rightarrow " + random.choice(UPPER)
        return _atom() + "\\rightarrow " + _atom()
    if kind == "inequality":
        rel = random.choice(["\\leq", "\\geq", "\\neq"])
        return _atom() + " " + rel + " " + _atom()
    if kind == "factorial":
        return _atom() + " !" if random.random() < 0.5 else "(" + s() + ") !"
    if kind == "ldots":
        op = random.choice(["+", "-", ","])
        return _atom() + op + _atom() + op + "\\ldots " + op + _atom()

    return _atom()


def sample():
    global _strat
    _strat = random.choice(STRATEGY_NAMES)
    expr = _expr()
    if random.random() < 0.2:
        expr = expr + " " + random.choice(RELOPS) + " " + _expr()
    return expr
