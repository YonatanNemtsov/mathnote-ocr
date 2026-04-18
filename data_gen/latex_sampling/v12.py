"""V12: Patterns missing from v1-v11.

Long superscripts/subscripts, equations, function application,
deeply nested mixed structures, trig with complex args, limits.
"""

import random

from .symbols import (_pick_base, _shared_short_content, VARS, DIGITS, GREEK, MISC_SYMBOLS, ARITH_OPS)


def _atom():
    if random.random() < 0.05:
        return random.choice(MISC_SYMBOLS)
    pool = random.choices([VARS, DIGITS, GREEK, ARITH_OPS], weights=[10, 6, 3, 2])[0]
    return random.choice(pool)


def _short_expr():
    """Short structured content."""
    return _shared_short_content(1, _atom, _pick_base)


def _long_expr():
    """3-6 structured terms."""
    n = random.randint(3, 6)
    parts = [_shared_short_content(1, _atom, _pick_base)]
    for _ in range(n - 1):
        if random.random() < 0.5:
            parts.append(random.choice(ARITH_OPS))
        parts.append(_shared_short_content(1, _atom, _pick_base))
    return " ".join(parts)


def _frac():
    return "\\frac{" + _short_expr() + "}{" + _short_expr() + "}"


# ── Pattern generators ───────────────────────────────────────────────


def _long_sum_sup():
    """Long sum/difference in exponent: e^{a+b+c+d}."""
    base = _atom()
    n = random.randint(3, 5)
    sup = "+".join(_atom() for _ in range(n))
    return base + "^{" + sup + "}"


def _frac_in_sup():
    """Fraction in exponent: x^{\\frac{a}{b}}."""
    base = _atom()
    return base + "^{" + _frac() + "}"


def _int_in_sup():
    """Integral in exponent: e^{\\int f dx}."""
    base = _atom()
    dx = "d " + random.choice(VARS[:5])
    body = _short_expr()
    v = random.random()
    if v < 0.5:
        return base + "^{\\int " + body + " " + dx + "}"
    else:
        lo = _atom()
        hi = _atom()
        return base + "^{\\int_{" + lo + "}^{" + hi + "} " + body + " " + dx + "}"


def _sum_in_sup():
    """Summation in exponent: e^{\\sum a_i}."""
    base = _atom()
    op = random.choice(["\\sum", "\\prod"])
    v = random.random()
    if v < 0.5:
        return base + "^{" + op + " " + _short_expr() + "}"
    else:
        lo = _atom() + "=" + _atom()
        hi = _atom()
        return base + "^{" + op + "_{" + lo + "}^{" + hi + "} " + _short_expr() + "}"


def _sqrt_in_sup():
    """Sqrt in exponent: x^{\\sqrt{a+b}}."""
    base = _atom()
    return base + "^{\\sqrt{" + _short_expr() + "}}"


def _nested_sup():
    """Nested superscripts: x^{y^{z}}, or long nested."""
    v = random.random()
    if v < 0.4:
        return _pick_base() + "^{" + _pick_base() + "^{" + _atom() + "}}"
    elif v < 0.7:
        inner = _pick_base() + "^{" + _short_expr() + "}"
        return _pick_base() + "^{" + inner + "+" + _atom() + "}"
    else:
        return _pick_base() + "^{" + _pick_base() + "^{" + _pick_base() + "^{" + _atom() + "}}}"


def _long_sup_with_context():
    """Long superscript alongside other structures."""
    v = random.random()
    if v < 0.3:
        # long sup + fraction
        n = random.randint(2, 4)
        sup = random.choice(ARITH_OPS).join(_atom() for _ in range(n))
        return _pick_base() + "^{" + sup + "}+" + _frac()
    elif v < 0.5:
        # fraction + long sup
        n = random.randint(2, 4)
        sup = random.choice(ARITH_OPS).join(_atom() for _ in range(n))
        return _frac() + "+" + _pick_base() + "^{" + sup + "}"
    elif v < 0.7:
        # parens base with long sup
        base = "(" + _short_expr() + ")"
        sup = _long_expr()
        return base + "^{" + sup + "}"
    else:
        # long sub + long sup
        base = _atom()
        sub = _short_expr()
        sup = _short_expr()
        return base + "_{" + sub + "}^{" + sup + "}"


def _long_sub():
    """Long subscripts: x_{i+j+k}, a_{n-1}."""
    v = random.random()
    if v < 0.4:
        n = random.randint(2, 4)
        sub = random.choice(ARITH_OPS).join(_atom() for _ in range(n))
        return _pick_base() + "_{" + sub + "}"
    elif v < 0.7:
        # long sub + sup
        sub = _short_expr()
        sup = _short_expr()
        return _pick_base() + "_{" + sub + "}^{" + sup + "}"
    else:
        # long sub inside fraction
        sub = _short_expr()
        return "\\frac{" + _pick_base() + "_{" + sub + "}}{" + _short_expr() + "}"


def _equations():
    """Equations: a = b+c, f(x) = frac, chained a = b = c."""
    v = random.random()
    if v < 0.3:
        return _short_expr() + " = " + _short_expr()
    elif v < 0.5:
        return _atom() + " (" + _short_expr() + ") = " + _frac()
    elif v < 0.7:
        # chained equality
        return _atom() + " = " + _short_expr() + " = " + _short_expr()
    elif v < 0.85:
        # equation with structures on both sides
        lhs = random.choice([_frac, lambda: "\\sqrt{" + _short_expr() + "}"])()
        return lhs + " = " + _short_expr()
    else:
        # inequality
        rel = random.choice(["=", "\\leq", "\\geq", "\\neq"])
        return _short_expr() + " " + rel + " " + _short_expr()


def _func_application():
    """Function calls: f(x), g(x,y), sin(x^2+1), f(g(x))."""
    v = random.random()
    fn = random.choice(VARS[:6])
    if v < 0.25:
        return fn + " (" + _short_expr() + ")"
    elif v < 0.45:
        # two args
        return fn + " (" + _short_expr() + " , " + _short_expr() + ")"
    elif v < 0.65:
        # trig with complex arg
        trig = random.choice(["\\sin", "\\cos", "\\tan"])
        return trig + " (" + _short_expr() + ")"
    elif v < 0.8:
        # composed: f(g(x))
        g = random.choice(VARS[:6])
        return fn + " (" + g + " (" + _atom() + "))"
    else:
        # function = expression
        return fn + " (" + _atom() + ") = " + _short_expr()


def _deep_mixed():
    """Deeply nested mixed structures: frac with sqrt and sup inside."""
    v = random.random()
    if v < 0.25:
        # sqrt of sum with sup: sqrt{x^2 + y^2}
        return "\\sqrt{" + _pick_base() + "^{2}+" + _pick_base() + "^{2}}"
    elif v < 0.45:
        # frac with sqrt in num, sup in den
        return "\\frac{" + "\\sqrt{" + _short_expr() + "}" + "}{" + _pick_base() + "^{" + _short_expr() + "}}"
    elif v < 0.65:
        # bigop of a fraction with sup
        op = random.choice(["\\sum", "\\int"])
        lo = _atom()
        hi = _atom()
        return op + "_{" + lo + "}^{" + hi + "} \\frac{" + _pick_base() + "^{" + _atom() + "}}{" + _short_expr() + "}"
    elif v < 0.8:
        # fraction inside sqrt inside fraction
        inner = "\\sqrt{" + _frac() + "}"
        return "\\frac{" + inner + "}{" + _short_expr() + "}"
    else:
        # sqrt{frac} + sup + frac
        return "\\sqrt{" + _frac() + "}+" + _pick_base() + "^{" + _short_expr() + "}+" + _frac()


def _limits():
    """Limit expressions: lim_{x->inf}, lim_{n->0}."""
    v = random.random()
    var = random.choice(VARS[:5])
    if v < 0.4:
        target = random.choice(["\\infty", "0", "1", _atom()])
        return "\\lim_{" + var + " \\rightarrow " + target + "} " + _short_expr()
    elif v < 0.7:
        target = random.choice(["\\infty", "0"])
        return "\\lim_{" + var + " \\rightarrow " + target + "} " + _frac()
    else:
        target = _atom()
        return "\\lim_{" + var + " \\rightarrow " + target + "} " + _pick_base() + "^{" + _short_expr() + "}"


def _long_top_level():
    """Long top-level expressions: a+b-c+d*e=f."""
    n = random.randint(4, 7)
    parts = [_atom()]
    for _ in range(n - 1):
        parts.append(random.choice(ARITH_OPS + ["="]))
        # sometimes insert a structure instead of atom
        if random.random() < 0.2:
            parts.append(random.choice([_frac, lambda: _pick_base() + "^{" + _atom() + "}"])())
        else:
            parts.append(_atom())
    return " ".join(parts)


def _nested_frac():
    """Nested fractions: frac with frac in numerator or denominator.

    Weighted heavily toward hard cases: frac-in-both and triple nesting.
    """
    v = random.random()
    inner = _frac()
    if v < 0.15:
        # frac in numerator: \frac{a+\frac{b}{c}}{d+e}
        return "\\frac{" + _short_expr() + "+" + inner + "}{" + _short_expr() + "}"
    elif v < 0.25:
        # frac in denominator: \frac{a}{b+\frac{c}{d}}
        return "\\frac{" + _short_expr() + "}{" + _short_expr() + "+" + inner + "}"
    elif v < 0.55:
        # frac in both: \frac{\frac{a}{b}}{\frac{c}{d}} — HARD
        v2 = random.random()
        if v2 < 0.5:
            return "\\frac{" + _frac() + "}{" + _frac() + "}"
        else:
            # frac-in-both with extra terms
            return "\\frac{" + _short_expr() + "+" + _frac() + "}{" + _frac() + "+" + _short_expr() + "}"
    elif v < 0.7:
        # nested frac alone in numerator: \frac{\frac{a}{b}}{c+d}
        return "\\frac{" + inner + "}{" + _short_expr() + "}"
    elif v < 0.85:
        # double nested: \frac{1+\frac{1}{\frac{a}{b}}}{c} — HARD
        return "\\frac{" + _atom() + "+" + "\\frac{" + _atom() + "}{" + _frac() + "}}{" + _short_expr() + "}"
    else:
        # nested frac in denominator alone: \frac{a+b}{\frac{c}{d}}
        return "\\frac{" + _short_expr() + "}{" + _frac() + "}"


def _bigop_with_frac():
    """Big operator adjacent to fraction: \int \frac{...}{...}.

    Weighted heavily toward hard cases: bigop+limits+nested frac.
    """
    op = random.choice(["\\sum", "\\int", "\\prod"])
    v = random.random()
    if v < 0.1:
        # bare bigop + frac
        return op + " " + _frac()
    elif v < 0.25:
        # bigop with limits + simple frac
        lo = _short_expr()
        hi = _atom()
        return op + "_{" + lo + "}^{" + hi + "} " + _frac()
    elif v < 0.55:
        # bigop with limits + nested frac — HARD
        lo = _short_expr()
        hi = _atom()
        inner = _frac()
        num = _short_expr() + "+" + inner
        return op + "_{" + lo + "}^{" + hi + "} \\frac{" + num + "}{" + _short_expr() + "}"
    elif v < 0.7:
        # bigop with sub only + frac
        lo = _short_expr()
        return op + "_{" + lo + "} " + _frac()
    elif v < 0.85:
        # bigop + frac + more terms
        return op + " " + _frac() + "+" + _short_expr()
    else:
        # bigop with limits + frac-in-both — HARDEST
        lo = _short_expr()
        hi = _atom()
        return op + "_{" + lo + "}^{" + hi + "} \\frac{" + _frac() + "}{" + _frac() + "}"


# ── Main sampler ─────────────────────────────────────────────────────

_GENERATORS = [
    (_long_sum_sup, 10),
    (_frac_in_sup, 8),
    (_int_in_sup, 8),
    (_sum_in_sup, 8),
    (_sqrt_in_sup, 5),
    (_nested_sup, 8),
    (_long_sup_with_context, 12),
    (_long_sub, 8),
    (_equations, 12),
    (_func_application, 10),
    (_deep_mixed, 10),
    (_limits, 8),
    (_long_top_level, 8),
    (_nested_frac, 15),
    (_bigop_with_frac, 12),
]

_funcs = [g for g, _ in _GENERATORS]
_weights = [w for _, w in _GENERATORS]


def sample():
    gen = random.choices(_funcs, weights=_weights)[0]
    return gen()
