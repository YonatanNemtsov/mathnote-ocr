"""V5: Frac-heavy and index-heavy — the two hardest structure types.

V1-V4 spread weight across all structures. V5 deliberately overweights
fractions (nested, complex num/den) and subscript/superscript patterns
(multi-index, chains, tensor-like). Also adds factorial and commas.
"""

import random

from .symbols import (_pick_base, _shared_expr, _shared_term, _shared_sub_content,
                       VARS, UPPER, DIGITS, GREEK, GREEK_UPPER, MISC_SYMBOLS,
                       BIGOPS, FUNCS)

ARITH_OPS = ["+", "-"]  # v5-specific: fewer ops for frac/index focus


def _atom():
    if random.random() < 0.05:
        return random.choice(MISC_SYMBOLS)
    pool = random.choices(
        [VARS, UPPER, DIGITS, GREEK, GREEK_UPPER, ARITH_OPS],
        weights=[12, 2, 6, 5, 1, 2],
    )[0]
    return random.choice(pool)

def _expr(d=0):
    return _shared_expr(d, _atom, _pick_base, _struct)

def _term(d):
    return _shared_term(d, _atom, _pick_base, _struct)

def _slot(d):
    return _expr(d + 1)

def _struct(d):
    # Heavily biased toward frac and sub/sup
    kind = random.choices(
        ["frac", "sup", "sub", "subsup", "sqrt", "func", "bigop",
         "parens", "factorial", "comma_list"],
        weights=[30, 18, 14, 10, 6, 5, 8, 4, 3, 2],
    )[0]

    if kind == "frac":
        return "\\frac{" + _slot(d) + "}{" + _slot(d) + "}"
    if kind == "sup":
        base = _pick_base()
        # sometimes nested sup: x^{y^z}
        if d < 2 and random.random() < 0.15:
            inner = _pick_base() + "^{" + _slot(d + 1) + "}"
            return base + "^{" + inner + "}"
        return base + "^{" + _slot(d) + "}"
    if kind == "sub":
        base = _pick_base()
        # sometimes multi-char subscript
        if random.random() < 0.3:
            n = random.randint(2, 4)
            idx = " ".join(_atom() for _ in range(n))
            return base + "_{" + idx + "}"
        return base + "_{" + _slot(d) + "}"
    if kind == "subsup":
        return _pick_base() + "_{" + _shared_sub_content(_pick_base) + "}^{" + _slot(d) + "}"
    if kind == "sqrt":
        return "\\sqrt{" + _slot(d) + "}"
    if kind == "func":
        fn = random.choice(FUNCS)
        return fn + " " + _slot(d)
    if kind == "bigop":
        op = random.choice(BIGOPS)
        v = random.random()
        if v < 0.15:
            return "{" + op + "}" + " " + _slot(d)
        if v < 0.4:
            return "{" + op + "_{" + _slot(d) + "}" + "}" + " " + _slot(d)
        return "{" + op + "_{" + _slot(d) + "}^{" + _slot(d) + "}" + "}" + " " + _slot(d)
    if kind == "parens":
        return "( " + _slot(d) + " )"
    if kind == "factorial":
        return _atom() + " " + "!"
    if kind == "comma_list":
        n = random.randint(2, 4)
        items = [_atom() for _ in range(n)]
        return "(" + ",".join(items) + ")"


def sample():
    expr = _expr(0)
    if random.random() < 0.15:
        expr = expr + " " + "=" + " " + _expr(0)
    return expr
