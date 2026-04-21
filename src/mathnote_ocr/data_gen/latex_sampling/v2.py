"""V2: Context-aware recursive generator — misc symbols in proper contexts.

V1 gaps: no partial, nabla, infty, pm, ldots, cup/cap, in, subset, rightarrow.
These symbols need specific contexts to make sense.

Approach: same recursive core as v1, but add special "contextual" structure
types that naturally use these symbols: limits, set expressions, arrows,
derivative fracs, series with ldots, pm expressions.
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


# ── Recursive builder ─────────────────────────────────────────────────


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
            "bigop",
            "parens",
            "binom",
            "abs",
            # new contextual types
            "deriv",
            "partial_deriv",
            "lim",
            "set_expr",
            "arrow",
            "pm_expr",
            "ldots_seq",
            "nabla_expr",
            "infty_expr",
        ],
        weights=[
            15,
            12,
            8,
            6,
            8,
            7,
            10,
            6,
            3,
            4,
            # contextual weights
            4,
            3,
            4,
            3,
            2,
            3,
            2,
            2,
            2,
        ],
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
            return fn + "^{" + random.choice(DIGITS) + "} " + _slot(d)
        return fn + " " + _slot(d)
    if kind == "bigop":
        op = random.choice(BIGOPS)
        v = random.random()
        if v < 0.2:
            return op + " " + _slot(d)
        if v < 0.4:
            return op + "_{" + _slot(d) + "}" + _slot(d)
        if v < 0.8:
            return op + "_{" + _slot(d) + "}^{" + _slot(d) + "}" + _slot(d)
        return op + "^{" + _slot(d) + "}" + _slot(d)
    if kind == "parens":
        return "( " + _slot(d) + " )"
    if kind == "binom":
        return "\\binom{" + _slot(d) + "}{" + _slot(d) + "}"
    if kind == "abs":
        return "| " + _slot(d) + " |"

    # ── Contextual structures ──
    if kind == "deriv":
        x = random.choice(VARS)
        y = random.choice(VARS)
        if random.random() < 0.5:
            return "\\frac{" + "d" + " " + y + "}{" + "d" + " " + x + "}"
        return "\\frac{" + "d" + " " + "^{2}" + " " + y + "}{" + "d" + " " + x + " " + "^{2}" + "}"

    if kind == "partial_deriv":
        x = random.choice(VARS)
        y = random.choice(VARS)
        if random.random() < 0.5:
            return "\\frac{\\partial " + y + "}{\\partial " + x + "}"
        z = random.choice(VARS)
        return "\\frac{\\partial^{2}" + y + "}{\\partial " + x + "\\partial " + z + "}"

    if kind == "lim":
        x = random.choice(VARS)
        target = "\\infty" if random.random() < 0.4 else _atom()
        return "\\lim_{" + x + " \\rightarrow " + target + "}" + _slot(d)

    if kind == "set_expr":
        op = random.choice(SET_OPS)
        if random.random() < 0.5:
            return random.choice(UPPER) + " " + op + " " + random.choice(UPPER)
        return random.choice(VARS) + " \\in " + random.choice(UPPER)

    if kind == "arrow":
        return _atom() + " \\rightarrow " + _atom()

    if kind == "pm_expr":
        return _atom() + " \\pm " + _slot(d)

    if kind == "ldots_seq":
        a = _atom()
        return (
            a
            + random.choice(ARITH_OPS)
            + _atom()
            + random.choice(ARITH_OPS)
            + "\\ldots "
            + random.choice(ARITH_OPS)
            + _atom()
        )

    if kind == "nabla_expr":
        v = random.choice(VARS)
        if random.random() < 0.5:
            return "\\nabla " + v
        return "\\nabla^{2}" + v

    if kind == "infty_expr":
        # infty as limit in a sum/int
        op = random.choice(["\\sum", "\\int"])
        return op + "_{" + _slot(d) + "}^{\\infty}" + _slot(d)


def sample():
    expr = _expr(0)
    if random.random() < 0.2:
        expr = expr + " " + random.choice(RELOPS) + " " + _expr(0)
    # small chance of set relation at top level
    if random.random() < 0.05:
        expr = random.choice(VARS) + " \\in " + random.choice(UPPER)
    if random.random() < 0.05:
        expr = random.choice(UPPER) + " \\subset " + random.choice(UPPER)
    return expr
