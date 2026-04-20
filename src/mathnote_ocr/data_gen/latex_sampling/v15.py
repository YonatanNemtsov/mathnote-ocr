"""V15: Complex numerators/denominators with integrals, cdot, and inline fractions.

Target pattern: expressions like
    \frac{{\int}\frac{1+x}{2-3}{\cdot}\frac{11}{x}}{x}

Key features:
- Bare \int (no bounds) or bounded \int as a sibling among fractions
- \cdot and \times joining fractions/terms within num/den
- Multiple fractions chained by operators in a single num or den
- Integrals multiplied by fractions
"""

import random

from .symbols import _pick_base, MISC_SYMBOLS, ARITH_OPS

VARS = list("xyzabcnmkpqr")
DIGITS = list("0123456789")
COMMON_VARS = list("xyz")
GREEK = [
    "\\alpha", "\\beta", "\\gamma", "\\theta", "\\pi", "\\sigma",
]
MUL_OPS = ["\\cdot", "\\times"]
ADD_OPS = ["+", "-"]


def _var():
    return random.choice(COMMON_VARS)


def _any_var():
    return random.choice(VARS)


def _atom():
    if random.random() < 0.05:
        return random.choice(MISC_SYMBOLS)
    if random.random() < 0.08:
        return random.choice(ARITH_OPS)
    if random.random() < 0.5:
        return _any_var()
    if random.random() < 0.6:
        return random.choice(DIGITS)
    return random.choice(GREEK)


def _small_frac():
    """Frac with structured content."""
    from .symbols import _shared_short_content, _pick_base
    num = _shared_short_content(1, _atom, _pick_base)
    den = _shared_short_content(1, _atom, _pick_base)
    return "\\frac{" + num + "}{" + den + "}"


def _mul_op():
    return random.choice(MUL_OPS)


def _int_bare():
    """\int with no bounds."""
    return "\\int"


def _int_bounded():
    """\int_{a}^{b}."""
    lo = random.choice(["0", "1", _atom()])
    hi = random.choice(["1", _atom(), "\\infty"])
    return "\\int_{" + lo + "}^{" + hi + "}}"


def _int_any():
    """Either bare or bounded integral."""
    if random.random() < 0.5:
        return _int_bare()
    return _int_bounded()


# ── Core patterns: int/frac/cdot chains in num or den ─────────────


def _int_cdot_fracs_over_simple():
    r"""frac{ \int \frac{...}{...} \cdot \frac{...}{...} }{simple}

    The flagship pattern: integral times fraction(s) in numerator.
    """
    parts = [_int_any(), _small_frac()]
    # Add 1-2 more fracs joined by cdot/times
    for _ in range(random.randint(1, 2)):
        parts.append(_mul_op())
        parts.append(_small_frac())
    num = " ".join(parts)
    den = _atom() if random.random() < 0.5 else (_atom() + " " + random.choice(ADD_OPS) + " " + _atom())
    return "\\frac{" + num + "}{" + den + "}"


def _int_cdot_frac_plus_terms():
    r"""frac{ \int \frac{a}{b} \cdot x + c }{den}

    Integral times fraction plus extra terms in numerator.
    """
    core = _int_any() + " " + _small_frac() + " " + _mul_op() + " " + _atom()
    extras = []
    for _ in range(random.randint(1, 3)):
        extras.append(random.choice(ADD_OPS))
        extras.append(_atom())
    num = core + " " + " ".join(extras)
    den = _atom() + " " + random.choice(ADD_OPS) + " " + _atom()
    return "\\frac{" + num + "}{" + den + "}"


def _fracs_chained_by_cdot():
    r"""frac{ \frac{a}{b} \cdot \frac{c}{d} \cdot \frac{e}{f} }{g}

    Multiple fractions joined by cdot in numerator, no integral.
    """
    n_fracs = random.randint(2, 3)
    parts = [_small_frac()]
    for _ in range(n_fracs - 1):
        parts.append(_mul_op())
        parts.append(_small_frac())
    num = " ".join(parts)
    den = _atom() if random.random() < 0.6 else (_atom() + " " + random.choice(ADD_OPS) + " " + _atom())
    return "\\frac{" + num + "}{" + den + "}"


def _int_frac_over_int_frac():
    r"""frac{ \int \frac{a}{b} }{ \int \frac{c}{d} }

    Both numerator and denominator have integral + fraction.
    """
    num = _int_any() + " " + _small_frac()
    den = _int_any() + " " + _small_frac()
    if random.random() < 0.4:
        num += " " + _mul_op() + " " + _atom()
    if random.random() < 0.4:
        den += " " + _mul_op() + " " + _atom()
    return "\\frac{" + num + "}{" + den + "}"


def _cdot_fracs_in_den():
    r"""frac{simple}{ \frac{a}{b} \cdot \frac{c}{d} }

    Chain of fracs in denominator instead of numerator.
    """
    num = _atom() if random.random() < 0.4 else (_atom() + " " + random.choice(ADD_OPS) + " " + _atom())
    n_fracs = random.randint(2, 3)
    parts = [_small_frac()]
    for _ in range(n_fracs - 1):
        parts.append(_mul_op())
        parts.append(_small_frac())
    den = " ".join(parts)
    return "\\frac{" + num + "}{" + den + "}"


def _int_frac_cdot_frac_dx():
    r"""\int \frac{a}{b} \cdot \frac{c}{d} dx

    Free-standing integral (not inside a frac) with chained fracs.
    """
    head = _int_any()
    parts = [_small_frac()]
    for _ in range(random.randint(1, 2)):
        parts.append(_mul_op())
        parts.append(_small_frac())
    dx = "d " + _var()
    return head + " " + " ".join(parts) + " " + dx


def _frac_cdot_frac_plus_frac():
    r"""\frac{a}{b} \cdot \frac{c}{d} + \frac{e}{f}

    Free-standing chain of fracs with mixed operators (no outer frac).
    """
    n_terms = random.randint(2, 4)
    parts = [_small_frac()]
    for _ in range(n_terms - 1):
        if random.random() < 0.5:
            parts.append(_mul_op())
        else:
            parts.append(random.choice(ADD_OPS))
        parts.append(_small_frac())
    return " ".join(parts)


def _int_over_simple_cdot_frac():
    r"""frac{\int x}{y} \cdot \frac{a}{b}

    Fraction containing integral, then cdot another fraction.
    """
    int_content = _atom() if random.random() < 0.5 else (_atom() + " " + random.choice(ADD_OPS) + " " + _atom())
    frac1 = "\\frac{" + _int_any() + " " + int_content + "}{" + _atom() + "}"
    frac2 = _small_frac()
    return frac1 + " " + _mul_op() + " " + frac2


def _sum_cdot_fracs():
    r"""frac{ \sum \frac{a}{b} \cdot \frac{c}{d} }{e}

    Like int pattern but with \sum or \prod.
    """
    op = random.choice(["\\sum", "\\prod"])
    v = random.random()
    if v < 0.4:
        head = op
    elif v < 0.7:
        head = op + "_{" + _atom() + " = " + _atom() + "}^{" + _atom() + "}"
    else:
        head = op + "_{" + _atom() + "}^{" + _atom() + "}"
    parts = [_small_frac()]
    for _ in range(random.randint(1, 2)):
        parts.append(_mul_op())
        parts.append(_small_frac())
    num = head + " " + " ".join(parts)
    den = _atom() if random.random() < 0.5 else (_atom() + " " + random.choice(ADD_OPS) + " " + _atom())
    return "\\frac{" + num + "}{" + den + "}"


def _nested_frac_cdot():
    r"""frac{ \frac{a \cdot b}{c} }{d}

    cdot inside a nested fraction's numerator.
    """
    inner_num_parts = [_atom()]
    for _ in range(random.randint(1, 2)):
        inner_num_parts.append(_mul_op())
        inner_num_parts.append(_atom())
    inner_num = " ".join(inner_num_parts)
    inner = "\\frac{" + inner_num + "}{" + _atom() + "}"
    v = random.random()
    if v < 0.4:
        num = inner
    elif v < 0.7:
        num = inner + " " + _mul_op() + " " + _small_frac()
    else:
        num = inner + " " + random.choice(ADD_OPS) + " " + _atom()
    den = (_atom() + " " + random.choice(ADD_OPS) + " " + _atom()) if random.random() < 0.5 else _atom()
    return "\\frac{" + num + "}{" + den + "}"


def _int_frac_cdot_atom_over_frac():
    r"""frac{ \int \frac{a}{b} \cdot x }{ \frac{c}{d} }

    Numerator has int+frac+cdot+atom, denominator is a fraction.
    """
    num = _int_any() + " " + _small_frac() + " " + _mul_op() + " " + _atom()
    den = _small_frac()
    return "\\frac{" + num + "}{" + den + "}"


# ── Main sampler ────────────────────────────────────────────────────

_GENERATORS = [
    # Flagship: int cdot fracs in numerator
    (_int_cdot_fracs_over_simple, 15),
    (_int_cdot_frac_plus_terms, 12),
    # Chained fracs by cdot
    (_fracs_chained_by_cdot, 12),
    (_cdot_fracs_in_den, 10),
    (_frac_cdot_frac_plus_frac, 8),
    # Int in both num and den
    (_int_frac_over_int_frac, 10),
    # Free-standing int with chained fracs
    (_int_frac_cdot_frac_dx, 10),
    # Sum/prod variants
    (_sum_cdot_fracs, 8),
    # Nested cdot inside fracs
    (_nested_frac_cdot, 10),
    # Mixed
    (_int_over_simple_cdot_frac, 8),
    (_int_frac_cdot_atom_over_frac, 8),
]

_funcs = [g for g, _ in _GENERATORS]
_weights = [w for _, w in _GENERATORS]


def sample():
    gen = random.choices(_funcs, weights=_weights)[0]
    return gen()
