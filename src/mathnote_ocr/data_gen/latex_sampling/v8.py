"""V8: Discrete math and relations — sets, logic, arrows, binomials.

Covers: cup, cap, in, subset, rightarrow, leq/geq/neq, binom,
factorial, ldots, upper greek, upper variables as sets.
"""

import random

from .symbols import (
    DIGITS,
    GREEK,
    GREEK_UPPER,
    MAX_DEPTH,
    MISC_SYMBOLS,
    RELOPS,
    SET_OPS,
    UPPER,
    VARS,
    _pick_base,
    _shared_expr,
    _shared_sub_content,
    _shared_term,
)

ARITH_OPS = ["+", "-"]  # v8-specific: discrete math focus


def _atom():
    if random.random() < 0.05:
        return random.choice(MISC_SYMBOLS)
    pool = random.choices(
        [VARS, UPPER, DIGITS, GREEK, GREEK_UPPER, ARITH_OPS],
        weights=[10, 5, 6, 4, 2, 2],
    )[0]
    return random.choice(pool)


def _var():
    return random.choice(VARS)


def _expr(d=0):
    return _shared_expr(d, _atom, _pick_base, _struct)


def _term(d):
    return _shared_term(d, _atom, _pick_base, _struct)


def _slot(d):
    return _expr(d + 1)


def _struct(d):
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
        weights=[10, 10, 8, 5, 5, 5, 5, 8, 4, 6, 5, 4, 4, 6, 4, 3, 6],
    )[0]

    if kind == "frac":
        return f"\\frac{{{_slot(d)}}}{{{_slot(d)}}}"
    if kind == "sup":
        return f"{_pick_base()}^{{{_slot(d)}}}"
    if kind == "sub":
        return _pick_base() + "_{" + _shared_sub_content(_pick_base) + "}"
    if kind == "subsup":
        return _pick_base() + "_{" + _shared_sub_content(_pick_base) + "}^{" + _slot(d) + "}"
    if kind == "sqrt":
        return f"\\sqrt{{{_slot(d)}}}"
    if kind == "func":
        fn = random.choice(["\\sin", "\\cos", "\\tan", "\\log", "\\ln"])
        return fn + " " + _slot(d)
    if kind == "parens":
        return f"({_slot(d)})"
    if kind == "abs":
        return f"|{_slot(d)}|"

    # ── Discrete / set structures ──
    if kind == "binom":
        return f"\\binom{{{_slot(d)}}}{{{_slot(d)}}}"
    if kind == "set_op":
        op = random.choice(SET_OPS)
        a, b = random.choice(UPPER), random.choice(UPPER)
        if random.random() < 0.3 and d < MAX_DEPTH:
            # chain: A cup B cap C
            c = random.choice(UPPER)
            return a + " " + op + " " + b + " " + random.choice(SET_OPS) + " " + c
        return a + " " + op + " " + b
    if kind == "element":
        return _atom() + "\\in " + random.choice(UPPER)
    if kind == "subset_rel":
        return random.choice(UPPER) + "\\subset " + random.choice(UPPER)
    if kind == "arrow":
        if random.random() < 0.4:
            return _var() + ":" + random.choice(UPPER) + "\\rightarrow " + random.choice(UPPER)
        return _atom() + "\\rightarrow " + _atom()
    if kind == "inequality":
        rel = random.choice(RELOPS)
        if random.random() < 0.3:
            # chain: a <= b <= c
            return _atom() + " " + rel + " " + _atom() + " " + rel + " " + _atom()
        return _atom() + " " + rel + " " + _atom()
    if kind == "factorial":
        if random.random() < 0.5:
            return f"{_atom()}!"
        return f"({_slot(d)})!"
    if kind == "ldots":
        op = random.choice(ARITH_OPS + [","])
        return _atom() + op + _atom() + op + "\\ldots " + op + _atom()
    if kind == "bigop":
        op = random.choice(["\\sum", "\\prod"])
        v = random.random()
        if v < 0.3:
            return op + " " + _slot(d)
        if v < 0.7:
            return op + "_{" + _var() + "=" + _atom() + "}" + _slot(d)
        return op + "_{" + _var() + "=" + _atom() + "}^{" + _atom() + "}" + _slot(d)


def sample():
    expr = _expr(0)
    if random.random() < 0.15:
        expr = f"{expr}={_expr(0)}"
    return expr
