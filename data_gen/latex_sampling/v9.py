"""V9: Juxtaposition-heavy — terms next to each other without operators.

Real math often has implicit multiplication: 2x, abc, xy^2, f(x)g(x).
This version biases toward juxtaposition over explicit operators, and
uses structures as bases for other structures (frac^2, sqrt_i, etc).
"""

import random

from .symbols import (_pick_base, _shared_expr, _shared_term, _shared_sub_content,
                       VARS, UPPER, DIGITS, GREEK, GREEK_UPPER, MISC_SYMBOLS,
                       BIGOPS, FUNCS)

ARITH_OPS = ["+", "-"]  # v9-specific: juxtaposition focus


def _atom():
    if random.random() < 0.05:
        return random.choice(MISC_SYMBOLS)
    pool = random.choices(
        [VARS, UPPER, DIGITS, GREEK, GREEK_UPPER, ARITH_OPS],
        weights=[12, 3, 6, 4, 1, 2],
    )[0]
    return random.choice(pool)

def _expr(d=0):
    return _shared_expr(d, _atom, _pick_base, _struct)

def _term(d):
    return _shared_term(d, _atom, _pick_base, _struct)

def _slot(d):
    return _expr(d + 1)

def _struct(d):
    kind = random.choices(
        ["frac", "sup", "sub", "subsup", "sqrt", "func", "bigop",
         "parens", "binom", "abs",
         "struct_sup", "struct_sub", "func_chain"],
        weights=[14, 12, 8, 6, 8, 7, 8,
                 6, 3, 4,
                 6, 5, 4],
    )[0]

    if kind == "frac":
        return "\\frac{" + _slot(d) + "}{" + _slot(d) + "}"
    if kind == "sup":
        return _pick_base() + "^{" + _slot(d) + "}"
    if kind == "sub":
        return _pick_base() + "_{" + _shared_sub_content(_pick_base) + "}"
    if kind == "subsup":
        return _pick_base() + "_{" + _shared_sub_content(_pick_base) + "}^{" + _slot(d) + "}"
    if kind == "sqrt":
        return "\\sqrt{" + _slot(d) + "}"
    if kind == "func":
        fn = random.choice(FUNCS)
        if random.random() < 0.2:
            return fn + "^{" + random.choice(DIGITS) + "}" + " " + _slot(d)
        return fn + " " + _slot(d)
    if kind == "bigop":
        op = random.choice(BIGOPS)
        v = random.random()
        if v < 0.2:
            return "{" + op + "}" + " " + _slot(d)
        if v < 0.5:
            return "{" + op + "_{" + _slot(d) + "}" + "}" + " " + _slot(d)
        return "{" + op + "_{" + _slot(d) + "}^{" + _slot(d) + "}" + "}" + " " + _slot(d)
    if kind == "parens":
        return "( " + _slot(d) + " )"
    if kind == "binom":
        return "\\binom{" + _slot(d) + "}{" + _slot(d) + "}"
    if kind == "abs":
        return "| " + _slot(d) + " |"

    # ── Structure as base ──
    if kind == "struct_sup":
        # structure^{exp}: (a+b)^2, \frac{a}{b}^n, \sqrt{x}^2
        base_kind = random.choice(["parens", "sqrt", "frac"])
        if base_kind == "parens":
            base = "( " + _slot(d) + " )"
        elif base_kind == "sqrt":
            base = "\\sqrt{" + _slot(d) + "}"
        else:
            base = "\\frac{" + _slot(d) + "}{" + _slot(d) + "}"
        return base + "^{" + _slot(d) + "}"

    if kind == "struct_sub":
        base_kind = random.choice(["parens", "func"])
        if base_kind == "parens":
            base = "( " + _slot(d) + " )"
        else:
            base = random.choice(FUNCS) + " " + _slot(d)
        return base + "_{" + _slot(d) + "}"

    if kind == "func_chain":
        # f(g(x)) style
        fn1 = random.choice(FUNCS)
        fn2 = random.choice(FUNCS)
        return fn1 + " " + fn2 + " " + _slot(d)


def sample():
    expr = _expr(0)
    if random.random() < 0.15:
        expr = expr + " " + "=" + " " + _expr(0)
    return expr
