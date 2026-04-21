"""V14: Hard realistic patterns targeting model weaknesses.

Key patterns that cause failures:
- Nested fracs with long denominators and duplicate variables
- Integrals of nested fracs
- Absolute value in superscripts with nested powers
- Deep nesting: frac inside frac inside sup
- Many symbols sharing same parent (long sibling chains)
"""

import random

from .symbols import ARITH_OPS, MISC_SYMBOLS, _pick_base, _shared_short_content

VARS = list("xyzabcnmkpqr")
DIGITS = list("0123456789")
COMMON_VARS = list("xyz")
GREEK = [
    "\\alpha",
    "\\beta",
    "\\gamma",
    "\\theta",
    "\\pi",
    "\\sigma",
]


def _var():
    return random.choice(COMMON_VARS)


def _any_var():
    return random.choice(VARS)


def _atom():
    if random.random() < 0.05:
        return random.choice(MISC_SYMBOLS)
    if random.random() < 0.08:
        return random.choice(ARITH_OPS)
    if random.random() < 0.6:
        return _any_var()
    if random.random() < 0.5:
        return random.choice(DIGITS)
    return random.choice(GREEK)


def _expr(lo=2, hi=4):
    """Structured terms with mixed ops and juxtaposition."""
    n = random.randint(lo, hi)
    parts = [_shared_short_content(1, _atom, _pick_base)]
    for _ in range(n - 1):
        v = random.random()
        if v < 0.4:
            parts.append("+")
        elif v < 0.6:
            parts.append("-")
        parts.append(_shared_short_content(1, _atom, _pick_base))
    return " ".join(parts)


def _frac(num=None, den=None):
    num = num or _expr(1, 2)
    den = den or _expr(1, 2)
    return "\\frac{" + num + "}{" + den + "}"


def _small_frac():
    return "\\frac{" + _atom() + "}{" + _atom() + "}"


def _abs(content):
    return "| " + content + " |"


# ── Nested frac + long den + dup vars ───────────────────────────────


def _nested_frac_dup():
    """frac{1+frac{1}{y}}{x+y+z} with shared variable."""
    shared = _var()
    inner_den = shared if random.random() < 0.6 else shared + " " + _atom()
    inner = "\\frac{" + _atom() + "}{" + inner_den + "}"
    den_parts = [shared]
    for _ in range(random.randint(2, 5)):
        den_parts.append(_atom())
    random.shuffle(den_parts)
    den = "+".join(den_parts)
    v = random.random()
    if v < 0.4:
        num = _atom() + " + " + inner
    elif v < 0.7:
        num = inner + " + " + _atom()
    else:
        num = _atom() + " + " + inner + " + " + _atom()
    return "\\frac{" + num + "}{" + den + "}"


def _nested_frac_deep():
    """frac{frac{a}{b}+c}{frac{d}{e}+f+g} — nested on both sides."""
    num = _small_frac() + " + " + _expr(1, 3)
    den = _small_frac() + " + " + _expr(2, 4)
    return "\\frac{" + num + "}{" + den + "}"


def _triple_nested():
    """frac{a+frac{b}{frac{c}{d}}}{e+f+g} — triple nesting."""
    innermost = _small_frac()
    mid = "\\frac{" + _atom() + "}{" + innermost + "}"
    num = _atom() + " + " + mid
    den = _expr(3, 5)
    return "\\frac{" + num + "}{" + den + "}"


def _frac_with_sup_and_nested():
    """frac{x^{2}+frac{a}{b}}{y^{3}+z} — sup + nested frac in num."""
    sup1 = _expr(1, 2)
    inner = _small_frac()
    num = _var() + "^{" + sup1 + "} + " + inner + " + " + _atom()
    sup2 = _atom()
    den = _var() + "^{" + sup2 + "} + " + _expr(1, 3)
    return "\\frac{" + num + "}{" + den + "}"


# ── Integral of complex fractions ───────────────────────────────────


def _int_nested_frac_dup():
    """int_{0}^{1} frac{a+frac{1}{x}}{x+y+z} with dup var."""
    shared = _var()
    lo = random.choice(["0", "1", _atom()])
    hi = random.choice(["1", "\\infty", _atom()])
    inner = "\\frac{" + _atom() + "}{" + shared + "}"
    num = _atom() + " + " + inner
    den_parts = [shared] + [_atom() for _ in range(random.randint(2, 4))]
    random.shuffle(den_parts)
    den = "+".join(den_parts)
    dx = "d " + _var()
    return "\\int_{" + lo + "}^{" + hi + "}} " + "\\frac{" + num + "}{" + den + "} " + dx


def _int_long_sub_frac():
    """int_{long sub} frac{...}{...} — long lower bound + fraction."""
    lo = _expr(2, 5)
    v = random.random()
    if v < 0.4:
        # with upper bound
        hi = _atom()
        head = "\\int_{" + lo + "}^{" + hi + "}}"
    else:
        # sub only
        head = "\\int_{" + lo + "}}"
    num = random.choice([_atom(), _expr(1, 2)])
    den = _expr(3, 6)
    dx = "d " + _var()
    return head + " " + "\\frac{" + num + "}{" + den + "} " + dx


def _sum_long_sub_frac():
    """sum_{long sub} frac{...}{...} — sum/prod with long lower bound + fraction."""
    op = random.choice(["\\sum", "\\prod"])
    v = random.random()
    if v < 0.5:
        lo = _expr(2, 4)
    else:
        lo = _var() + " = " + _expr(1, 3)
    hi = _atom()
    num = random.choice([_atom(), _expr(1, 2)])
    den = _expr(3, 6)
    return "{" + op + "_{" + lo + "}^{" + hi + "}} " + "\\frac{" + num + "}{" + den + "}"


def _int_long_sub_nested_frac():
    """int_{long} frac{a+frac{b}{c}}{d+e+f} — long sub + nested frac."""
    lo = _expr(2, 4)
    v = random.random()
    if v < 0.4:
        hi = _atom()
        head = "\\int_{" + lo + "}^{" + hi + "}}"
    else:
        head = "\\int_{" + lo + "}}"
    inner = _small_frac()
    num = _atom() + " + " + inner
    den = _expr(3, 5)
    dx = "d " + _var()
    return head + " " + "\\frac{" + num + "}{" + den + "} " + dx


def _int_of_nested_frac():
    """int frac{a+frac{b}{c}+d}{e+f+g+h} — integral of nested frac."""
    op = random.choice(["\\int", "\\sum"])
    lo = random.choice(["0", _atom(), _expr(1, 2)])
    hi = random.choice([_atom(), "\\infty"])
    inner = _small_frac()
    num = (
        (_expr(1, 2) + " + " + inner + " + " + _expr(0, 1))
        if random.random() < 0.5
        else (inner + " + " + _expr(1, 3))
    )
    den = _expr(3, 6)
    dx = " d " + _var() if "int" in op else ""
    return "{" + op + "_{" + lo + "}^{" + hi + "}} " + "\\frac{" + num + "}{" + den + "}" + dx


def _int_frac_sup_in_num():
    """int frac{x^{2}+y}{z+w+v} — integral of frac with sup in num."""
    op = random.choice(["\\int", "\\sum"])
    lo = random.choice(["0", _atom()])
    hi = _atom()
    base = _var()
    sup = _expr(1, 2)
    num = base + "^{" + sup + "} + " + _expr(1, 3)
    den = _expr(3, 5)
    dx = " d " + _var() if "int" in op else ""
    return "{" + op + "_{" + lo + "}^{" + hi + "}} " + "\\frac{" + num + "}{" + den + "}" + dx


# ── Absolute value patterns ─────────────────────────────────────────


def _sup_abs_nested():
    """e^{|x^{2}-5+x-5|} — abs with squared term inside sup."""
    base = random.choice(["e", _var()])
    sq_var = _var()
    sq_pow = _atom()
    terms = [sq_var + "^{" + sq_pow + "}"]
    for _ in range(random.randint(2, 4)):
        terms.append(random.choice(["+", "-"]))
        terms.append(_atom())
    return base + "^{" + _abs("".join(terms)) + "}"


def _frac_sup_abs():
    """frac{e^{|x^{2}-5+x|}}{4+y} — frac with abs+sup in num."""
    base = random.choice(["e", _var()])
    sq_var = _var()
    terms = [sq_var + "^{" + random.choice(["2", "3"]) + "}"]
    for _ in range(random.randint(2, 4)):
        terms.append(random.choice(["+", "-"]))
        terms.append(_atom())
    num = base + "^{" + _abs("".join(terms)) + "}"
    den = _expr(1, 3)
    return "\\frac{" + num + "}{" + den + "}"


def _abs_in_frac_num():
    """frac{|a+b+c|+d}{e+f} — abs expression in numerator."""
    content = _expr(2, 4)
    extra = _expr(1, 2)
    num = (
        (_abs(content) + " + " + extra)
        if random.random() < 0.5
        else (extra + " + " + _abs(content))
    )
    den = _expr(2, 4)
    return "\\frac{" + num + "}{" + den + "}"


def _sup_abs_plus_frac():
    """x^{|a+b|}+frac{c}{d+e+f} — abs sup next to fraction."""
    base = _var()
    content = _expr(2, 4)
    sup_part = base + "^{" + _abs(content) + "}"
    frac_part = _frac(den=_expr(2, 4))
    op = random.choice(["+", "-"])
    if random.random() < 0.5:
        return sup_part + " " + op + " " + frac_part
    return frac_part + " " + op + " " + sup_part


# ── Deep mixed nesting ──────────────────────────────────────────────


def _frac_in_sup_in_frac():
    """frac{x^{frac{a}{b}}+c}{d+e+f} — frac inside sup inside frac."""
    inner_frac = _small_frac()
    base = _var()
    num = base + "^{" + inner_frac + "} + " + _expr(1, 2)
    den = _expr(3, 5)
    return "\\frac{" + num + "}{" + den + "}"


def _sqrt_nested_frac():
    """sqrt{frac{a+b}{c}+d} or frac{sqrt{x+y}}{z+w+v}."""
    v = random.random()
    if v < 0.5:
        content = _small_frac() + " + " + _expr(1, 2)
        return "\\sqrt{" + content + "}"
    else:
        num = "\\sqrt{" + _expr(2, 3) + "}"
        den = _expr(3, 5)
        return "\\frac{" + num + "}{" + den + "}"


def _multi_level():
    """Expressions with 3+ nesting levels."""
    v = random.random()
    if v < 0.3:
        # sup inside frac inside int
        sup = _var() + "^{" + _expr(1, 2) + "}"
        num = sup + " + " + _small_frac()
        den = _expr(2, 4)
        return (
            "\\int_{"
            + _atom()
            + "}^{"
            + _atom()
            + "}} "
            + "\\frac{"
            + num
            + "}{"
            + den
            + "} d "
            + _var()
        )
    elif v < 0.6:
        # frac inside sqrt inside frac
        inner = _small_frac()
        mid = "\\sqrt{" + inner + " + " + _atom() + "}"
        den = _expr(3, 5)
        return "\\frac{" + mid + "}{" + den + "}"
    else:
        # abs inside sup inside frac
        abs_content = _var() + "^{2} + " + _atom()
        sup = _var() + "^{" + _abs(abs_content) + "}"
        return "\\frac{" + sup + "}{" + _expr(2, 4) + "}"


def _long_sibling_chain():
    """frac{a+b+c+d+e+f}{g+h} — many siblings under one parent."""
    n = random.randint(5, 8)
    parts = []
    for _ in range(n):
        if random.random() < 0.2:
            parts.append(_var() + "^{" + _atom() + "}")
        else:
            parts.append(_atom())
    num = "+".join(parts)
    den = _expr(2, 3)
    return "\\frac{" + num + "}{" + den + "}"


# ── Main sampler ────────────────────────────────────────────────────

_GENERATORS = [
    # Nested frac + dup vars (hardest)
    (_nested_frac_dup, 15),
    (_nested_frac_deep, 12),
    (_triple_nested, 12),
    (_frac_with_sup_and_nested, 10),
    # Integral of complex fracs
    (_int_nested_frac_dup, 12),
    (_int_long_sub_frac, 15),
    (_sum_long_sub_frac, 10),
    (_int_long_sub_nested_frac, 12),
    (_int_of_nested_frac, 10),
    (_int_frac_sup_in_num, 8),
    # Absolute value
    (_sup_abs_nested, 12),
    (_frac_sup_abs, 12),
    (_abs_in_frac_num, 8),
    (_sup_abs_plus_frac, 8),
    # Deep mixed
    (_frac_in_sup_in_frac, 10),
    (_sqrt_nested_frac, 6),
    (_multi_level, 12),
    (_long_sibling_chain, 8),
]

_funcs = [g for g, _ in _GENERATORS]
_weights = [w for _, w in _GENERATORS]


def sample():
    gen = random.choices(_funcs, weights=_weights)[0]
    return gen()
