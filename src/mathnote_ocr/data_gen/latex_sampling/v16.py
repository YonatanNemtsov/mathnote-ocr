"""V16: Superscript-juxtaposition patterns — var^{exp} followed by vars.

Targets implicit multiplication after superscripts:
- x^{2}y, ax^{n}b, x^{2}y^{3}z
- Polynomials: ax^{2}+bx+c
- Inside fracs, sqrt, etc.
"""

import random

from .symbols import _pick_base, MISC_SYMBOLS, RELOPS

VARS = list("xyzabcnmkpqr")
DIGITS = list("0123456789")
COMMON_VARS = list("xyz")
GREEK = [
    "\\alpha", "\\beta", "\\gamma", "\\theta", "\\pi", "\\sigma",
]
ARITH_OPS = ["+", "-"]


def _var():
    return random.choice(COMMON_VARS)


def _any_var():
    return random.choice(VARS)


def _atom():
    if random.random() < 0.05:
        return random.choice(MISC_SYMBOLS)
    if random.random() < 0.08:
        return random.choice(ARITH_OPS)
    if random.random() < 0.7:
        return _any_var()
    if random.random() < 0.5:
        return random.choice(DIGITS)
    return random.choice(GREEK)


def _small_exp():
    """Small exponent: digit, var, or short expr."""
    v = random.random()
    if v < 0.5:
        return _atom()
    if v < 0.8:
        return _atom()
    return _atom() + " " + random.choice(ARITH_OPS) + " " + _atom()


def _sup_var():
    """var^{exp} — a variable with a superscript."""
    return _any_var() + "^{" + _small_exp() + "}"


def _coeff_var():
    """Optional coefficient before a variable: a, 2, or nothing."""
    v = random.random()
    if v < 0.3:
        return random.choice(DIGITS[:5])  # small digit coefficient
    if v < 0.5:
        return _any_var()
    return ""


# ── Patterns ──────────────────────────────────────────────────────


def _sup_then_var():
    """x^{2}y — superscripted var followed by plain var."""
    return _sup_var() + " " + _any_var()


def _sup_then_vars():
    """x^{2}yz — superscripted var followed by 2-3 juxtaposed vars."""
    n = random.randint(2, 3)
    tail = " ".join(_any_var() for _ in range(n))
    return _sup_var() + " " + tail


def _coeff_sup_var():
    """ax^{2}y — coeff + sup var + trailing var."""
    c = _coeff_var()
    return c + " " + _sup_var() + " " + _any_var()


def _consecutive_sups():
    """x^{2}y^{3} or x^{n}y^{m}z — consecutive sup vars, possibly with trailing."""
    parts = [_sup_var(), _sup_var()]
    if random.random() < 0.4:
        parts.append(_any_var())
    return " ".join(parts)


def _sup_term():
    """A term like ax^{n} or just x^{n} or just a."""
    v = random.random()
    if v < 0.6:
        coeff = _coeff_var()
        base = _any_var() + "^{" + _atom() + "}"
        return (coeff + " " + base).strip()
    if v < 0.8:
        return _coeff_var() + " " + _any_var()
    return _atom()


def _polynomial():
    """Polynomial-like: mix of sup terms with ops and juxtaposition.
    e.g. y^{2} x^{7} + 3 a^{n} - b
    """
    n = random.randint(2, 5)
    parts = [_sup_term()]
    for _ in range(n - 1):
        if random.random() < 0.6:
            parts.append(random.choice(ARITH_OPS))
        parts.append(_sup_term())
    return " ".join(parts)


def _sup_juxt_plus():
    """x^{2}y+z or a^{n}b-c+d — sup+juxt mixed with ops."""
    head = _sup_var() + " " + _any_var()
    n = random.randint(1, 3)
    parts = [head]
    for _ in range(n):
        parts.append(random.choice(ARITH_OPS))
        if random.random() < 0.3:
            parts.append(_sup_var() + " " + _any_var())
        else:
            parts.append(_atom())
    return " ".join(parts)


def _frac_sup_juxt():
    """frac{x^{2}y}{z+w} — sup+juxt in numerator."""
    num = _sup_var() + " " + _any_var()
    if random.random() < 0.5:
        num += " " + random.choice(ARITH_OPS) + " " + _atom()
    return "\\frac{" + num + "}{" + _polynomial() + "}"


def _frac_poly():
    """frac{polynomial}{polynomial} — polynomial in frac."""
    return "\\frac{" + _polynomial() + "}{" + _polynomial() + "}"


def _sqrt_sup_juxt():
    """sqrt{x^{2}y+z} — sup+juxt inside sqrt."""
    content = _sup_var() + " " + _any_var() + " " + random.choice(ARITH_OPS) + " " + _atom()
    return "\\sqrt{" + content + "}"


def _equation_sup_juxt():
    """x^{2}y=az or x^{n}b+c=d — equation with sup+juxt."""
    lhs = _sup_var() + " " + _any_var()
    if random.random() < 0.5:
        lhs += " " + random.choice(ARITH_OPS) + " " + _atom()
    rhs = _atom()
    if random.random() < 0.5:
        rhs = _atom() + " " + random.choice(ARITH_OPS) + " " + _atom()
    return lhs + " " + random.choice(RELOPS) + " " + rhs


def _equation_poly():
    """polynomial = polynomial."""
    lhs = _polynomial()
    rhs = random.choice(["0", _atom(), _polynomial()])
    return lhs + " = " + rhs


def _multi_sup_juxt():
    """a^{2}b^{3}c+x^{n}y — multiple sup+juxt groups."""
    g1 = _sup_var() + " " + _any_var()
    g2 = _sup_var() + " " + _any_var()
    op = random.choice(ARITH_OPS)
    return g1 + " " + op + " " + g2


# ── Main sampler ──────────────────────────────────────────────────

_GENERATORS = [
    (_sup_then_var, 15),
    (_sup_then_vars, 8),
    (_coeff_sup_var, 10),
    (_consecutive_sups, 10),
    (_polynomial, 20),
    (_sup_juxt_plus, 12),
    (_frac_sup_juxt, 10),
    (_frac_poly, 10),
    (_sqrt_sup_juxt, 5),
    (_equation_sup_juxt, 10),
    (_equation_poly, 10),
    (_multi_sup_juxt, 8),
]

_funcs = [g for g, _ in _GENERATORS]
_weights = [w for _, w in _GENERATORS]


def sample():
    gen = random.choices(_funcs, weights=_weights)[0]
    return gen()
