"""Compositional expression templates for data generation.

All templates are classes with a classmethod `sample()`.
Composition functions (OneOf, Frac, Sup, ...) return new classes.

    AnyAtom = OneOf(Variable, Digit, Greek)
    SimpleFrac = Frac(AnyAtom, Digit)
    SimpleFrac.sample()  # → "\\frac{x}{3}"

    SumExpr = Seq(BigOpLowHigh(Lit("\\sum"),
                               lo=Seq(Lit("i"), Lit("="), Lit("0")),
                               hi=Lit("n")),
                  Sup(Lit("x"), Lit("i")))
    SumExpr.sample()  # → "\\sum_{i=0}^{n}x^{i}"
"""

import random

# ── Base ──────────────────────────────────────────────────────────────


class Template:
    @classmethod
    def sample(cls) -> str:
        raise NotImplementedError


class Empty(Template):
    """Produces nothing. Use as default for optional slots."""

    @classmethod
    def sample(cls) -> str:
        return ""


# ── Atoms ─────────────────────────────────────────────────────────────


class Atom(Template):
    choices: list[str] = []

    @classmethod
    def sample(cls) -> str:
        return random.choice(cls.choices)


def Lit(text: str):
    class C(Atom):
        choices = [text]

    return C


# ── Pools ─────────────────────────────────────────────────────────────


class Variable(Atom):
    choices = list("abcdefghijklmnopqrstuvwxyz")


class UpperVar(Atom):
    choices = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class Digit(Atom):
    choices = list("0123456789")


class Greek(Atom):
    choices = [
        "{\\alpha}",
        "{\\beta}",
        "{\\gamma}",
        "{\\delta}",
        "{\\epsilon}",
        "{\\theta}",
        "{\\lambda}",
        "{\\mu}",
        "{\\pi}",
        "{\\sigma}",
        "{\\phi}",
        "{\\psi}",
        "{\\omega}",
    ]


class GreekUpper(Atom):
    choices = [
        "{\\Gamma}",
        "{\\Delta}",
        "{\\Sigma}",
        "{\\Pi}",
        "{\\Phi}",
        "{\\Psi}",
        "{\\Omega}",
    ]


class Op(Atom):
    choices = ["+", "-", "{\\times}", "{\\cdot}", "{\\pm}", "{\\div}"]


class RelOp(Atom):
    choices = [
        "=",
        "<",
        ">",
        "{\\leq}",
        "{\\geq}",
        "{\\neq}",
        "{\\cup}",
        "{\\cap}",
        "{\\in}",
        "{\\subset}",
        "{\\rightarrow}",
        "{\\leftarrow}",
    ]


class Quant(Atom):
    choices = ["{\\forall}", "{\\exists}"]


class BigOp(Atom):
    choices = ["\\sum", "\\int", "\\prod"]


class Misc(Atom):
    choices = ["{\\infty}", "{\\partial}", "{\\nabla}", "{\\ldots}"]


class Bracket(Atom):
    choices = ["[", "]", "{\\lbrace}", "{\\rbrace}"]


class Punct(Atom):
    choices = ["/", ";"]


# ── Combinators ───────────────────────────────────────────────────────


def OneOf(*templates, weights=None):
    class C(Template):
        @classmethod
        def sample(cls):
            return random.choices(templates, weights)[0].sample()

    return C


def Seq(*parts):
    class C(Template):
        @classmethod
        def sample(cls):
            return "".join(t.sample() for t in parts)

    return C


# ── Composites ────────────────────────────────────────────────────────


def Sup(base, exp):
    """base^{exp}"""

    class C(Template):
        @classmethod
        def sample(cls):
            return f"{base.sample()}^{{{exp.sample()}}}"

    return C


def Sub(base, sub):
    """base_{sub}"""

    class C(Template):
        @classmethod
        def sample(cls):
            return f"{base.sample()}_{{{sub.sample()}}}"

    return C


def SupSub(base, sub, exp):
    """base_{sub}^{exp}"""

    class C(Template):
        @classmethod
        def sample(cls):
            return f"{base.sample()}_{{{sub.sample()}}}^{{{exp.sample()}}}"

    return C


def Frac(num, den):
    """\\frac{num}{den}"""

    class C(Template):
        @classmethod
        def sample(cls):
            return f"\\frac{{{num.sample()}}}{{{den.sample()}}}"

    return C


def Sqrt(inner):
    """\\sqrt{inner}"""

    class C(Template):
        @classmethod
        def sample(cls):
            return f"\\sqrt{{{inner.sample()}}}"

    return C


def Parens(inner):
    """(inner)"""

    class C(Template):
        @classmethod
        def sample(cls):
            return f"({inner.sample()})"

    return C


def BigOpBare(op=BigOp):
    """{\\sum} (bare, no bounds)"""

    class C(Template):
        @classmethod
        def sample(cls):
            return "{" + op.sample() + "}"

    return C


def BigOpLow(op=BigOp, lo=Empty):
    """{\\sum_{lo}}"""

    class C(Template):
        @classmethod
        def sample(cls):
            return "{" + op.sample() + "_{" + lo.sample() + "}}"

    return C


def BigOpHigh(op=BigOp, hi=Empty):
    """{\\sum^{hi}}"""

    class C(Template):
        @classmethod
        def sample(cls):
            return "{" + op.sample() + "^{" + hi.sample() + "}}"

    return C


def BigOpLowHigh(op=BigOp, lo=Empty, hi=Empty):
    """{\\sum_{lo}^{hi}}"""

    class C(Template):
        @classmethod
        def sample(cls):
            return "{" + op.sample() + "_{" + lo.sample() + "}^{" + hi.sample() + "}}"

    return C


def Binom(top, bottom):
    """\\binom{top}{bottom}"""

    class C(Template):
        @classmethod
        def sample(cls):
            return f"\\binom{{{top.sample()}}}{{{bottom.sample()}}}"

    return C
