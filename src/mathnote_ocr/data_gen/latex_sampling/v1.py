"""V1: Balanced recursive generator — all symbols, controlled depth.

Approach: recursive expr builder with proper separation of concerns:
- Arithmetic ops (+,-,*,.) between terms
- Relational ops (=,<=,>=) only at top level
- Misc symbols (nabla, partial, infty) only in appropriate contexts
- Structure probability decreases with depth
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
    """Pick from variable, digit, or greek pools (5% misc symbols)."""
    if random.random() < 0.05:
        return random.choice(MISC_SYMBOLS)
    pool = random.choices(
        [VARS, UPPER, DIGITS, GREEK, GREEK_UPPER, ARITH_OPS],
        weights=[10, 3, 8, 4, 1, 2],
    )[0]
    return random.choice(pool)


# ── Recursive builder ─────────────────────────────────────────────────


def _expr(d=0):
    return _shared_expr(d, _atom, _pick_base, _struct)


def _term(d):
    return _shared_term(d, _atom, _pick_base, _struct)


def _slot(d):
    return _expr(d + 1)


def _struct(d):
    """Pick a structure. All types represented but weighted by commonality."""
    kind = random.choices(
        ["frac", "sup", "sub", "subsup", "sqrt", "func", "bigop", "parens", "binom", "abs"],
        weights=[20, 15, 10, 8, 10, 8, 12, 8, 4, 5],
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
        fn = random.choice(FUNCS)
        # sometimes func^n (like sin^2)
        if random.random() < 0.2:
            return fn + "^{" + random.choice(DIGITS) + "} " + _slot(d)
        return fn + " " + _slot(d)
    if kind == "bigop":
        op = random.choice(BIGOPS)
        v = random.random()
        if v < 0.25:
            return op + " " + _slot(d)
        if v < 0.5:
            return op + "_{" + _slot(d) + "} " + _slot(d)
        if v < 0.85:
            return op + "_{" + _slot(d) + "}^{" + _slot(d) + "} " + _slot(d)
        return op + "^{" + _slot(d) + "} " + _slot(d)
    if kind == "parens":
        return f"({_slot(d)})"
    if kind == "binom":
        return f"\\binom{{{_slot(d)}}}{{{_slot(d)}}}"
    if kind == "abs":
        return f"|{_slot(d)}|"


def sample():
    """Sample an expression, optionally wrapped in a relation."""
    expr = _expr(0)
    # 20% chance: make it an equation/inequality
    if random.random() < 0.2:
        expr = expr + " " + random.choice(RELOPS) + " " + _expr(0)
    return expr
