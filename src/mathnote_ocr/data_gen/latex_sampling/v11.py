"""Hard cases: expressions that confuse the subset model.

Focused on structural ambiguities where small subsets see symbols
whose edge types are easy to misclassify:

1. int + fraction: NUM/DEN confused with UPPER/LOWER limits
2. bigop + adjacent fraction: limits vs fraction children
3. Nested fractions: which bar is parent?
4. sqrt containing fractions
5. sup/sub near fraction bars
6. Multiple adjacent structures
"""

import random

from .symbols import (_pick_base, _shared_short_content, VARS, DIGITS, GREEK, MISC_SYMBOLS, ARITH_OPS)


def _atom():
    if random.random() < 0.05:
        return random.choice(MISC_SYMBOLS)
    pool = random.choices([VARS, DIGITS, GREEK, ARITH_OPS], weights=[10, 6, 3, 2])[0]
    return random.choice(pool)


def _short_expr(dot_chance=0.0):
    """Short structured content."""
    result = _shared_short_content(1, _atom, _pick_base)
    if dot_chance > 0 and random.random() < dot_chance:
        result += " ."
    return result


def _frac():
    return "\\frac{" + _short_expr() + "}{" + _short_expr() + "}"


# ── Pattern generators ───────────────────────────────────────────────


def _int_frac():
    """∫ followed by a fraction — NUM/DEN vs UPPER/LOWER confusion."""
    v = random.random()
    body = _frac()
    dx = "d " + random.choice(VARS[:5])

    if v < 0.3:
        # bare integral + fraction
        return "\\int " + body + " " + dx
    elif v < 0.6:
        # integral with limits + fraction
        lo = _short_expr()
        hi = _short_expr()
        return "\\int_{" + lo + "}^{" + hi + "}" + body + " " + dx
    elif v < 0.8:
        # integral + fraction + extra terms
        return "\\int " + body + "+" + _short_expr()
    else:
        # integral with limits + nested fraction
        lo = _atom()
        hi = _atom()
        inner_frac = "\\frac{" + _atom() + "}{" + _atom() + "}"
        return "\\int_{" + lo + "}^{" + hi + "}\\frac{" + _short_expr() + "}{" + inner_frac + "}"


def _bigop_frac():
    """Big operator (sum/prod) adjacent to fraction."""
    op = random.choice(["\\sum", "\\prod"])
    v = random.random()

    if v < 0.4:
        # bigop with limits + fraction
        lo = _atom() + "=" + _atom()
        hi = _atom()
        return op + "_{" + lo + "}^{" + hi + "}" + _frac()
    elif v < 0.7:
        # bigop + fraction as the summed term
        lo = _atom()
        return op + "_{" + lo + "}" + _frac()
    else:
        # bigop with fraction IN the limits
        return op + "_{" + _atom() + "}^{" + _atom() + "}" + _short_expr()


def _nested_frac():
    """Nested fractions — which bar is parent?"""
    v = random.random()

    if v < 0.3:
        # fraction in numerator
        inner = _frac()
        return "\\frac{" + inner + "}{" + _short_expr() + "}"
    elif v < 0.6:
        # fraction in denominator
        inner = _frac()
        return "\\frac{" + _short_expr() + "}{" + inner + "}"
    elif v < 0.8:
        # fraction in both
        return "\\frac{" + _frac() + "}{" + _frac() + "}"
    else:
        # triple nesting
        inner = "\\frac{" + _atom() + "}{" + _atom() + "}"
        mid = "\\frac{" + inner + "}{" + _short_expr() + "}"
        return "\\frac{" + mid + "}{" + _short_expr() + "}"


def _sqrt_frac():
    """sqrt containing or adjacent to fraction."""
    v = random.random()

    if v < 0.4:
        # sqrt of a fraction
        return "\\sqrt{" + _frac() + "}"
    elif v < 0.7:
        # fraction containing sqrt
        return "\\frac{" + "\\sqrt{" + _short_expr() + "}" + "}{" + _short_expr() + "}"
    else:
        # sqrt next to fraction
        return "\\sqrt{" + _short_expr() + "}" + "+" + _frac()


def _sup_near_frac():
    """Superscript/subscript adjacent to fraction bars."""
    v = random.random()

    if v < 0.25:
        # atom^exp followed by fraction
        return _pick_base() + "^{" + _short_expr() + "}" + _frac()
    elif v < 0.5:
        # fraction followed by superscript (fraction^exp is rare but confusing)
        return _frac() + "+" + _pick_base() + "^{" + _short_expr() + "}"
    elif v < 0.75:
        # sub+sup on same base, followed by fraction
        base = _atom()
        return base + "_{" + _atom() + "}^{" + _atom() + "}" + _frac()
    else:
        # fraction with superscripted terms in num/den
        num = _pick_base() + "^{" + _atom() + "}"
        den = _pick_base() + "_{" + _atom() + "}"
        return "\\frac{" + num + "+" + _atom() + "}{" + den + "}"


def _multi_structure():
    """Multiple structures adjacent — forces model to learn boundaries."""
    structs = [_frac, _sqrt_frac, _sup_near_frac]
    v = random.random()

    if v < 0.4:
        # Two fractions side by side
        return _frac() + "+" + _frac()
    elif v < 0.6:
        # Fraction + bigop
        op = random.choice(["\\sum", "\\int"])
        return _frac() + "+" + op + "_{" + _atom() + "}" + _short_expr()
    elif v < 0.8:
        # Three terms with mixed structure
        return _pick_base() + "^{" + _atom() + "}+" + _frac() + "-" + _atom()
    else:
        # Long expression with fraction in middle
        return _short_expr() + "+" + _frac() + "+" + _short_expr()


def _dotted_expr():
    """1-3 atoms with dots mixed in (as an atom-level token)."""
    n = random.randint(1, 3)
    parts = []
    for i in range(n):
        if random.random() < 0.3:
            parts.append(".")
        parts.append(_atom())
        if i < n - 1:
            parts.append(random.choice(ARITH_OPS + ["."]))
    if random.random() < 0.3:
        parts.append(".")
    return " ".join(parts)


def _dot_exprs():
    """Expressions with periods (dots) at various tree depths."""
    v = random.random()
    if v < 0.2:
        # dot inside fraction numerator or denominator
        num = _dotted_expr()
        den = _short_expr() if random.random() < 0.5 else _dotted_expr()
        return "\\frac{" + num + "}{" + den + "}"
    elif v < 0.35:
        # dot inside sqrt
        return "\\sqrt{" + _dotted_expr() + "}"
    elif v < 0.5:
        # dot in superscript
        base = _atom()
        return base + "^{" + _dotted_expr() + "}"
    elif v < 0.65:
        # dot inside fraction + dot at top level
        return _atom() + " . " + "\\frac{" + _dotted_expr() + "}{" + _short_expr() + "}"
    elif v < 0.8:
        # nested: fraction with dotted content + superscript
        frac = "\\frac{" + _dotted_expr() + "}{" + _dotted_expr() + "}"
        return frac + "+" + _pick_base() + "^{" + _dotted_expr() + "}"
    else:
        # dot near bigop limits
        op = random.choice(["\\sum", "\\int"])
        lo = _dotted_expr()
        return op + "_{" + lo + "}" + _dotted_expr()


def _int_complex():
    """Complex integral expressions."""
    v = random.random()
    lo = _atom()
    hi = _atom()

    if v < 0.3:
        # Double integral
        body = _frac()
        dx = "d " + random.choice(VARS[:3])
        dy = "d " + random.choice(VARS[:3])
        return "\\int \\int " + body + " " + dx + dy
    elif v < 0.6:
        # Integral of sqrt over fraction
        return "\\int_{" + lo + "}^{" + hi + "}\\sqrt{" + _frac() + "}" + _atom()
    else:
        # Integral with fraction limits
        lo_frac = "\\frac{" + _atom() + "}{" + _atom() + "}"
        return "\\int_{" + lo_frac + "}^{" + hi + "}" + _short_expr()


# ── Main sampler ─────────────────────────────────────────────────────

_GENERATORS = [
    (_int_frac, 20),
    (_bigop_frac, 15),
    (_nested_frac, 20),
    (_sqrt_frac, 10),
    (_sup_near_frac, 15),
    (_multi_structure, 10),
    (_int_complex, 10),
    (_dot_exprs, 15),
]

_funcs = [g for g, _ in _GENERATORS]
_weights = [w for _, w in _GENERATORS]


def sample():
    gen = random.choices(_funcs, weights=_weights)[0]
    return gen()
