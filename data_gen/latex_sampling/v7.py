"""V7: Academic math — derivatives, integrals with dx, limits, series.

Focuses on patterns that appear in calculus/analysis textbooks.
Uses partial, nabla, infty, rightarrow, pm in proper contexts.
"""

import random

from .symbols import (_pick_base, _shared_expr, _shared_term, _shared_sub_content,
                       VARS, UPPER, DIGITS, GREEK, GREEK_UPPER, MISC_SYMBOLS,
                       FUNCS)

ARITH_OPS = ["+", "-"]  # v7-specific: calculus focus


def _atom():
    if random.random() < 0.05:
        return random.choice(MISC_SYMBOLS)
    pool = random.choices(
        [VARS, DIGITS, GREEK, ARITH_OPS],
        weights=[10, 5, 3, 2],
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
        ["frac", "sup", "sub", "subsup", "sqrt", "func",
         "integral", "sum_prod", "deriv", "partial", "lim",
         "parens", "nabla", "pm", "infty_bound", "prime"],
        weights=[12, 10, 8, 6, 6, 6,
                 10, 8, 6, 5, 5,
                 5, 3, 3, 4, 6],
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
        if random.random() < 0.2:
            return f"{fn}^{{{random.choice(DIGITS)}}}{{{_slot(d)}}}"
        return f"{fn}{{{_slot(d)}}}"
    if kind == "parens":
        return f"({_slot(d)})"

    # ── Calculus structures ──
    if kind == "integral":
        x = _var()
        v = random.random()
        if v < 0.3:
            return "\\int " + _slot(d) + "d " + x
        if v < 0.7:
            return "\\int_{" + _slot(d) + "}^{" + _slot(d) + "}" + _slot(d) + "d " + x
        return "\\int_{" + _atom() + "}^{\\infty}" + _slot(d) + "d " + x

    if kind == "sum_prod":
        op = random.choice(["\\sum", "\\prod"])
        v = random.random()
        if v < 0.2:
            return op + " " + _slot(d)
        if v < 0.5:
            return op + "_{" + _var() + "=" + _atom() + "}" + _slot(d)
        if v < 0.85:
            hi = "\\infty" if random.random() < 0.3 else _atom()
            return op + "_{" + _var() + "=" + _atom() + "}^{" + hi + "}" + _slot(d)
        return op + "^{" + _slot(d) + "}" + _slot(d)

    if kind == "deriv":
        x, y = _var(), _var()
        if random.random() < 0.6:
            return "\\frac{d" + y + "}{d" + x + "}"
        return "\\frac{d^{2}" + y + "}{d" + x + "^{2}}"

    if kind == "partial":
        x, y = _var(), _var()
        if random.random() < 0.5:
            return "\\frac{\\partial " + y + "}{\\partial " + x + "}"
        z = _var()
        return "\\frac{\\partial^{2}" + y + "}{\\partial " + x + " \\partial " + z + "}"

    if kind == "lim":
        x = _var()
        target = "\\infty" if random.random() < 0.4 else _atom()
        return "\\lim_{" + x + " \\rightarrow " + target + "}" + _slot(d)

    if kind == "nabla":
        v = _var()
        if random.random() < 0.6:
            return "\\nabla " + v
        return "\\nabla^{2}" + v

    if kind == "prime":
        base = _var()
        n_primes = random.choices([1, 2, 3], weights=[6, 2, 1])[0]
        primes = "\\prime" * n_primes
        return f"{base}^{{{primes}}}"

    if kind == "pm":
        return _atom() + "\\pm " + _slot(d)

    if kind == "infty_bound":
        op = random.choice(["\\sum", "\\int"])
        return op + "_{" + _slot(d) + "}^{\\infty}" + _slot(d)


def sample():
    expr = _expr(0)
    if random.random() < 0.2:
        expr = f"{expr}={_expr(0)}"
    return expr
