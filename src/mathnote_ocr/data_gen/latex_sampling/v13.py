"""V13: Long numerators/denominators, nested fractions, long sup/sub.

Focused on expressions with many children per parent — stresses the
subset model's ability to predict correct parent when only seeing a
partial window of siblings.
"""

import random

from .symbols import ARITH_OPS, DIGITS, GREEK, MISC_SYMBOLS, VARS, _pick_base, _shared_short_content


def _atom():
    if random.random() < 0.05:
        return random.choice(MISC_SYMBOLS)
    pool = random.choices([VARS, DIGITS, GREEK, ARITH_OPS], weights=[10, 6, 3, 2])[0]
    return random.choice(pool)


def _op():
    return random.choice(ARITH_OPS)


def _short_expr():
    """Short structured content."""
    return _shared_short_content(1, _atom, _pick_base)


def _long_expr(lo=4, hi=8):
    """lo-hi terms, mix of ops, juxtaposition, and structures."""
    n = random.randint(lo, hi)
    parts = [_long_expr_term()]
    for _ in range(n - 1):
        # Sometimes op, sometimes juxtaposition (no op)
        if random.random() < 0.6:
            parts.append(_op())
        parts.append(_long_expr_term())
    return " ".join(parts)


def _long_expr_term():
    """Single term — varied structures."""
    v = random.random()
    if v < 0.3:
        return _atom()
    if v < 0.5:
        return _pick_base() + "^{" + _short_expr() + "}"
    if v < 0.6:
        return _pick_base() + "_{" + _short_expr() + "}"
    if v < 0.7:
        return _frac()
    if v < 0.8:
        return "\\left( " + _short_expr() + " \\right)"
    if v < 0.9:
        op = random.choice(["\\sum", "\\int", "\\prod"])
        return op + "_{" + _atom() + "}^{" + _atom() + "} " + _atom()
    return "\\sqrt{" + _short_expr() + "}"


def _frac(num=None, den=None):
    num = num or _short_expr()
    den = den or _short_expr()
    return "\\frac{" + num + "}{" + den + "}"


# ── Long numerator/denominator ──────────────────────────────────────


def _long_num():
    """Fraction with long numerator (4-8 atoms)."""
    return _frac(num=_long_expr(4, 8))


def _long_den():
    """Fraction with long denominator (4-8 atoms)."""
    return _frac(den=_long_expr(4, 8))


def _long_both():
    """Fraction with long numerator AND denominator."""
    return _frac(num=_long_expr(3, 6), den=_long_expr(3, 6))


def _long_frac_with_context():
    """Long fraction plus extra terms around it."""
    v = random.random()
    if v < 0.3:
        return _short_expr() + " + " + _frac(num=_long_expr(4, 7))
    elif v < 0.6:
        return _frac(den=_long_expr(4, 7)) + " + " + _short_expr()
    else:
        return _atom() + " + " + _frac(num=_long_expr(3, 5), den=_long_expr(3, 5)) + " + " + _atom()


# ── Nested fractions ────────────────────────────────────────────────


def _nested_frac_long_num():
    """Nested fraction in long numerator: \\frac{a+b+\\frac{c}{d}+e}{f+g}."""
    n = random.randint(2, 5)
    parts = [_atom() for _ in range(n)]
    inner = _frac()
    pos = random.randint(0, len(parts))
    parts.insert(pos, inner)
    num = " ".join(parts)
    return _frac(num=num)


def _nested_frac_long_den():
    """Nested fraction in long denominator."""
    n = random.randint(2, 5)
    parts = [_atom() for _ in range(n)]
    inner = _frac()
    pos = random.randint(0, len(parts))
    parts.insert(pos, inner)
    den = " ".join(parts)
    return _frac(den=den)


def _nested_frac_both():
    """Nested fractions in both num and den with extra terms."""
    v = random.random()
    if v < 0.3:
        return _frac(num=_frac() + " + " + _short_expr(), den=_frac() + " + " + _short_expr())
    elif v < 0.6:
        return _frac(num=_long_expr(2, 4) + " + " + _frac(), den=_frac())
    else:
        return _frac(num=_frac(), den=_long_expr(2, 4) + " + " + _frac())


def _triple_nested():
    """Triple nested fraction."""
    v = random.random()
    inner = _frac()
    mid = _frac(num=inner)
    if v < 0.3:
        return _frac(num=mid, den=_short_expr())
    elif v < 0.6:
        return _frac(num=_short_expr(), den=mid)
    else:
        return _frac(num=_atom() + " + " + mid, den=_long_expr(2, 4))


def _nested_frac_with_long_children():
    """Nested fraction where inner fraction also has long num/den."""
    inner = _frac(num=_long_expr(2, 4), den=_short_expr())
    v = random.random()
    if v < 0.5:
        return _frac(num=inner + " + " + _short_expr(), den=_long_expr(2, 4))
    else:
        return _frac(num=_long_expr(2, 4), den=_short_expr() + " + " + inner)


# ── Long superscripts ──────────────────────────────────────────────


def _long_sup():
    """Long superscript: x^{a+b+c+d+e}."""
    base = _pick_base()
    n = random.randint(3, 6)
    sup = " ".join(_atom() for _ in range(n))
    return base + "^{" + sup + "}"


def _sup_with_frac():
    """Superscript containing a fraction: x^{a+\\frac{b}{c}+d}."""
    base = _pick_base()
    parts = [_atom() for _ in range(random.randint(1, 3))]
    parts.insert(random.randint(0, len(parts)), _frac())
    sup = " ".join(parts)
    return base + "^{" + sup + "}"


def _sup_with_nested_frac():
    """Superscript containing nested fraction."""
    base = _pick_base()
    inner = _frac()
    sup = _atom() + " + " + _frac(num=inner)
    return base + "^{" + sup + "}"


def _long_sup_on_frac():
    """Fraction with long superscript on result: (\\frac{a}{b})^{long}."""
    v = random.random()
    if v < 0.5:
        # sup on atom next to fraction
        n = random.randint(3, 5)
        sup = " ".join(_atom() for _ in range(n))
        return _pick_base() + "^{" + sup + "} + " + _frac()
    else:
        # fraction then atom with long sup
        n = random.randint(3, 5)
        sup = " ".join(_atom() for _ in range(n))
        return _frac() + " + " + _pick_base() + "^{" + sup + "}"


# ── Long subscripts ─────────────────────────────────────────────────


def _long_sub():
    """Long subscript: x_{i+j+k+l}."""
    base = _pick_base()
    n = random.randint(3, 5)
    sub = " ".join(_atom() for _ in range(n))
    return base + "_{" + sub + "}"


def _long_sub_and_sup():
    """Both long subscript and superscript."""
    base = _pick_base()
    n_sub = random.randint(2, 4)
    n_sup = random.randint(2, 4)
    sub = " ".join(_atom() for _ in range(n_sub))
    sup = " ".join(_atom() for _ in range(n_sup))
    return base + "_{" + sub + "}^{" + sup + "}"


def _sub_with_frac():
    """Subscript containing fraction."""
    base = _pick_base()
    parts = [_atom() for _ in range(random.randint(1, 2))]
    parts.insert(random.randint(0, len(parts)), _frac())
    sub = " ".join(parts)
    return base + "_{" + sub + "}"


def _long_sub_in_frac():
    """Long subscript inside fraction num or den."""
    v = random.random()
    base = _pick_base()
    n = random.randint(2, 4)
    sub = " ".join(_atom() for _ in range(n))
    term = base + "_{" + sub + "}"
    if v < 0.5:
        return _frac(num=term + " + " + _short_expr())
    else:
        return _frac(den=_short_expr() + " + " + term)


# ── Combined hard patterns ──────────────────────────────────────────


def _frac_with_sup_in_num():
    """Fraction with superscripted terms in long numerator."""
    n = random.randint(2, 4)
    parts = []
    for _ in range(n):
        if random.random() < 0.4:
            parts.append(_pick_base() + "^{" + _short_expr() + "}")
        else:
            parts.append(_atom())
    return _frac(num=" ".join(parts), den=_long_expr(2, 4))


def _frac_with_sub_in_den():
    """Fraction with subscripted terms in long denominator."""
    n = random.randint(2, 4)
    parts = []
    for _ in range(n):
        if random.random() < 0.4:
            parts.append(_pick_base() + "_{" + _short_expr() + "}")
        else:
            parts.append(_atom())
    return _frac(num=_short_expr(), den=" ".join(parts))


def _nested_with_sup_sub():
    """Nested fraction with sup/sub at various levels."""
    v = random.random()
    if v < 0.3:
        inner = _frac(num=_pick_base() + "^{" + _atom() + "}", den=_short_expr())
        return _frac(num=inner + " + " + _atom(), den=_long_expr(2, 4))
    elif v < 0.6:
        inner = _frac()
        return _pick_base() + "^{" + inner + "} + " + _frac(den=_long_expr(2, 4))
    else:
        return _frac(
            num=_pick_base() + "^{" + _short_expr() + "} + " + _frac(),
            den=_pick_base() + "_{" + _short_expr() + "} + " + _short_expr(),
        )


# ── Main sampler ────────────────────────────────────────────────────

_GENERATORS = [
    # Long num/den (core focus)
    (_long_num, 12),
    (_long_den, 12),
    (_long_both, 10),
    (_long_frac_with_context, 8),
    # Nested fractions with long content
    (_nested_frac_long_num, 12),
    (_nested_frac_long_den, 12),
    (_nested_frac_both, 10),
    (_triple_nested, 8),
    (_nested_frac_with_long_children, 8),
    # Long sup/sub
    (_long_sup, 10),
    (_sup_with_frac, 8),
    (_sup_with_nested_frac, 5),
    (_long_sup_on_frac, 8),
    (_long_sub, 8),
    (_long_sub_and_sup, 8),
    (_sub_with_frac, 5),
    (_long_sub_in_frac, 8),
    # Combined
    (_frac_with_sup_in_num, 8),
    (_frac_with_sub_in_den, 8),
    (_nested_with_sup_sub, 8),
]

_funcs = [g for g, _ in _GENERATORS]
_weights = [w for _, w in _GENERATORS]


def sample():
    gen = random.choices(_funcs, weights=_weights)[0]
    return gen()
