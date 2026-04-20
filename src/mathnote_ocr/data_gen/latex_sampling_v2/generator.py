"""Generative expression sampler.

One flat pool of ALL symbols. Randomly picks structures and fills
their slots recursively. Depth and budget control complexity.

Usage:
    from mathnote_ocr.data_gen.latex_sampling_v2.generator import sample
    latex = sample()
"""

import random

from .templates import (
    Template, Variable, UpperVar, Digit, Greek, GreekUpper, Misc, Op, BigOp,
    RelOp,
    OneOf, Seq, Frac as TFrac, Sqrt as TSqrt, Sup as TSup, Sub as TSub,
    BigOpBare as TBigOpBare, BigOpLow as TBigOpLow, BigOpHigh as TBigOpHigh, BigOpLowHigh as TBigOpLowHigh,
)


# ── Symbol pool ──────────────────────────────────────────────────────

LOWER = list("abcdefghijklmnopqrstuvwxyz")
UPPER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGIT = list("0123456789")
GREEK = [
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon",
    r"\theta", r"\lambda", r"\mu", r"\pi", r"\sigma",
    r"\phi", r"\psi", r"\omega",
]
GREEK_UPPER = [r"\Gamma", r"\Delta", r"\Sigma", r"\Pi",
               r"\Phi", r"\Psi", r"\Omega"]
OPERATORS = ["+", "-", r"\times", r"\cdot", r"\pm", r"\div"]
RELATIONS = ["=", r"\leq", r"\geq", r"\neq", "<", ">"]
SET_LOGIC = [r"\cup", r"\cap", r"\in", r"\subset", r"\forall", r"\exists"]
CALCULUS = [r"\infty", r"\partial", r"\nabla"]
ARROWS = [r"\rightarrow", r"\leftarrow"]
PUNCTUATION = [".", ",", ";", ":", "!", "/", "|"]
BRACKETS = ["[", "]", r"\lbrace", r"\rbrace"]
FUNCTIONS = [r"\sin", r"\cos", r"\tan", r"\log", r"\ln", r"\exp",
             r"\max", r"\min"]
OTHER = [r"\ldots"]

SYMBOLS = (LOWER + UPPER + DIGIT + GREEK + GREEK_UPPER +
           OPERATORS + RELATIONS + SET_LOGIC + CALCULUS +
           ARROWS + PUNCTUATION + BRACKETS + FUNCTIONS + OTHER)

BIG_OPS = [r"\sum", r"\int", r"\prod"]

_NEEDS_BRACES = {s for s in SYMBOLS + BIG_OPS if s.startswith("\\")}


def _brace(s):
    return f"{{{s}}}" if s in _NEEDS_BRACES else s


def _pick():
    return random.choice(SYMBOLS)


# Symbols that can be a base for sup/sub (exclude operators, relations, etc.)
BASE_SYMBOLS = (LOWER + UPPER + DIGIT + GREEK + GREEK_UPPER + CALCULUS)

# ── Sub/sup content templates ────────────────────────────────────────
# Weights proportional to pool size so each individual symbol is ~equally likely
_Atom1 = OneOf(Variable, UpperVar, Digit, Greek, GreekUpper, Misc,
               weights=[26, 26, 10, 13, 7, 4])
_Atom2 = Seq(_Atom1, _Atom1)
_Atom3 = Seq(_Atom1, _Atom1, _Atom1)

SUB_CONTENT = OneOf(_Atom1, _Atom2, _Atom3, weights=[10, 2, 1])

# ── Big op limit content (everything except big ops) ─────────────────
_LimitSup = TSup(_Atom1, SUB_CONTENT)
_SimpleFrac = TFrac(_Atom1, _Atom1)
_SimpleSqrt = TSqrt(_Atom1)
_LimitPiece = OneOf(_Atom1, Op, _LimitSup, _SimpleFrac, _SimpleSqrt, RelOp,
                    weights=[12, 3, 3, 2, 1, 2])
_Limit1 = _LimitPiece
_Limit2 = Seq(_LimitPiece, _LimitPiece)
_Limit3 = Seq(_LimitPiece, _LimitPiece, _LimitPiece)
LIMIT_CONTENT = OneOf(_Limit1, _Limit2, _Limit3, weights=[6, 2, 1])

# Big ops with limits (for use inside sups/fracs)
_BigOpLow = TBigOpLow(BigOp, lo=LIMIT_CONTENT)
_BigOpHigh = TBigOpHigh(BigOp, hi=LIMIT_CONTENT)
_BigOpLowHigh = TBigOpLowHigh(BigOp, lo=LIMIT_CONTENT, hi=LIMIT_CONTENT)
_BigOpWithLimits = OneOf(_BigOpLow, _BigOpHigh, _BigOpLowHigh, weights=[1, 1, 2])

_SupFrac = TFrac(LIMIT_CONTENT, LIMIT_CONTENT)
_NegSupFrac = Seq(Op, _SupFrac)  # e.g. -\frac{x}{y}
_BigOpBare = TBigOpBare(BigOp)
_SupAtom1 = OneOf(Variable, UpperVar, Digit, Greek, GreekUpper, Misc, _BigOpBare, _BigOpWithLimits, _SupFrac, _NegSupFrac,
                  weights=[26, 26, 10, 13, 7, 4, 2, 2, 2, 2])
_SupAtom2 = Seq(_SupAtom1, _SupAtom1)
_SupAtom3 = Seq(_SupAtom1, _SupAtom1, _SupAtom1)
SUP_CONTENT = OneOf(_SupAtom1, _SupAtom2, _SupAtom3, weights=[6, 2, 1])

# ── Frac content templates ─────────────────────────────────────────
# Sup piece for use inside fracs
_SupAtom = TSup(_Atom1, SUP_CONTENT)

# Inner frac content (no further nesting)
_FracPieceInner = OneOf(_Atom1, Op, _SupAtom, weights=[4, 1, 1])
_FracInner1 = _FracPieceInner
_FracInner2 = Seq(_FracPieceInner, _FracPieceInner)
_FracInner3 = Seq(_FracPieceInner, _FracPieceInner, _FracPieceInner)
FRAC_CONTENT_INNER = OneOf(_FracInner1, _FracInner2, _FracInner3,
                           weights=[4, 4, 2])

# Nested frac uses inner content (max 1 level deep)
_NestedFrac = TFrac(FRAC_CONTENT_INNER, FRAC_CONTENT_INNER)

# Outer frac content (can contain one nested frac)
_FracPiece = OneOf(_Atom1, Op, _NestedFrac, _SupAtom, RelOp,
                   weights=[6, 1, 1, 2, 1])
_Frac1 = _FracPiece
_Frac2 = Seq(_FracPiece, _FracPiece)
_Frac3 = Seq(_FracPiece, _FracPiece, _FracPiece)
_Frac4 = Seq(_FracPiece, _FracPiece, _FracPiece, _FracPiece)
_Frac5 = Seq(_FracPiece, _FracPiece, _FracPiece, _FracPiece, _FracPiece)
_Frac6 = Seq(_FracPiece, _FracPiece, _FracPiece, _FracPiece, _FracPiece, _FracPiece)
_Frac7 = Seq(_FracPiece, _FracPiece, _FracPiece, _FracPiece, _FracPiece, _FracPiece, _FracPiece,)
FRAC_CONTENT = OneOf(_Frac1, _Frac2, _Frac3, _Frac4, _Frac5, _Frac6, _Frac7,
                     weights=[7, 4, 4, 3, 2,1,1])


def _pick_base():
    return random.choice(BASE_SYMBOLS)


def _pick_big_op():
    return random.choice(BIG_OPS)


# ── Structures ───────────────────────────────────────────────────────
# Each returns (latex, n_symbols).
# Slots are filled by calling random_expr() directly.


def gen_atom(budget, depth, symbols=None):
    return _brace(random.choice(symbols or SYMBOLS)), 1


def gen_sup(budget, depth):
    base = _brace(_pick_base())
    exp = SUP_CONTENT.sample()
    return f"{base}^{{{exp}}}", 2


def gen_sub(budget, depth):
    base = _brace(_pick_base())
    sub = SUB_CONTENT.sample()
    return f"{base}_{{{sub}}}", 2


def gen_sup_sub(budget, depth):
    base = _brace(_pick_base())
    sub = SUB_CONTENT.sample()
    exp = SUP_CONTENT.sample()
    return f"{base}_{{{sub}}}^{{{exp}}}", 3


def gen_frac(budget, depth):
    num = FRAC_CONTENT.sample()
    den = FRAC_CONTENT.sample()
    return f"\\frac{{{num}}}{{{den}}}", 3


def gen_sqrt(budget, depth):
    inner, n = random_expr(min(budget - 1, 4), depth + 1, exclude={gen_sqrt})
    return f"\\sqrt{{{inner}}}", 1 + n


def gen_parens(budget, depth):
    inner, n = random_expr(min(budget - 2, 4), depth + 1, exclude={gen_parens})
    return f"({inner})", 2 + n


def gen_big_op(budget, depth):
    op = _pick_big_op()
    return f"{{{op}}}", 1


def gen_big_op_low(budget, depth):
    op = _pick_big_op()
    lo = LIMIT_CONTENT.sample()
    return f"{{{op}_{{{lo}}}}}", 2


def gen_big_op_high(budget, depth):
    op = _pick_big_op()
    hi = LIMIT_CONTENT.sample()
    return f"{{{op}^{{{hi}}}}}", 2


def gen_big_op_low_high(budget, depth):
    op = _pick_big_op()
    lo = LIMIT_CONTENT.sample()
    hi = LIMIT_CONTENT.sample()
    return f"{{{op}_{{{lo}}}^{{{hi}}}}}", 3


def gen_lim(budget, depth):
    lo = LIMIT_CONTENT.sample()
    return r"{\lim_{" + lo + "}}", 2


def gen_binom(budget, depth):
    top_b = max(2, budget - 1)
    top, tn = random_expr(top_b, depth + 1, exclude={gen_binom})
    bot, bn = random_expr(max(2, budget - 1 - tn), depth + 1, exclude={gen_binom})
    return f"\\binom{{{top}}}{{{bot}}}", tn + bn


def gen_prime(budget, depth):
    base = _brace(random.choice(LOWER + UPPER + GREEK + GREEK_UPPER + [")"]))
    n_primes = random.choices([1, 2, 3], weights=[6, 2, 1])[0]
    primes = "{\\prime}" * n_primes
    return f"{base}^{{{primes}}}", 1 + n_primes



# (generator_fn, min_budget, weight, max_depth)
STRUCTURES = [
    (gen_atom,            1, 11, 4),
    (gen_sup,             2,  2, 1),
    (gen_sub,             2,  2, 1),
    (gen_sup_sub,         3,  1, 1),
    (gen_frac,            3,  4, 2),
    (gen_sqrt,            2,  1, 2),
    (gen_big_op,          1,  1, 2),
    (gen_big_op_low,      2,  1, 2),
    (gen_big_op_high,     2,  1, 2),
    (gen_big_op_low_high, 3,  1, 2),
    (gen_parens,          3,  1, 2),
    (gen_lim,             2,  1, 2),
    (gen_binom,           3,  2, 1),
    (gen_prime,            2,  1, 2),
]


# ── Generator ────────────────────────────────────────────────────────


def random_term(budget, depth, exclude=None, symbols=None):
    """Pick a random structure and generate one term."""
    if budget <= 0:
        return "", 0

    opts = [(fn, w) for fn, min_b, w, max_d in STRUCTURES
            if budget >= min_b and depth <= max_d
            and (exclude is None or fn not in exclude)]
    if not opts:
        return _brace(random.choice(symbols or SYMBOLS)), 1

    fns, weights = zip(*opts)
    fn = random.choices(fns, weights, k=1)[0]
    if fn is gen_atom:
        return fn(budget, depth, symbols=symbols)
    return fn(budget, depth)


def random_expr(budget, depth, exclude=None, symbols=None):
    """Generate 1+ juxtaposed terms within the budget."""
    if budget <= 0:
        return _brace(random.choice(symbols or SYMBOLS)), 1

    max_terms = max(1, 7 - depth * 2)
    parts = []
    used = 0

    while used < budget and len(parts) < max_terms:
        term, n = random_term(budget - used, depth, exclude, symbols)
        if n == 0:
            break
        parts.append(term)
        used += n
        if random.random() < 0.2:
            break

    if not parts:
        return _brace(_pick()), 1
    if len(parts) == 1:
        return parts[0], used
    return "".join(_brace(p) for p in parts), used


def sample(max_symbols=None):
    """Generate a random LaTeX expression."""
    if max_symbols is None:
        sizes = list(range(2, 41))
        weights = [max(1, 14 - abs(s - 15)) for s in sizes]
        max_symbols = random.choices(sizes, weights, k=1)[0]
    latex, _ = random_expr(max_symbols, depth=0)
    return latex


if __name__ == "__main__":
    random.seed(42)
    for _ in range(30):
        print(sample())
