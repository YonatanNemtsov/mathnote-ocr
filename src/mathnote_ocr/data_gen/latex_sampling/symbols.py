"""Shared symbol pools for all data generation versions.

Defines the complete symbol vocabulary matching the classifier's
capabilities. Each version imports from here to ensure full coverage.
"""

import random

# ── Core pools (used by all versions) ────────────────────────────────

VARS = list("abcdefghijklmnopqrstuvwxyz")
UPPER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = list("0123456789")
GREEK = [
    "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon",
    "\\theta", "\\lambda", "\\mu", "\\pi", "\\sigma",
    "\\phi", "\\psi", "\\omega",
]
GREEK_UPPER = [
    "\\Gamma", "\\Delta", "\\Sigma", "\\Pi",
    "\\Phi", "\\Psi", "\\Omega",
]

# ── Operator pools ───────────────────────────────────────────────────

ARITH_OPS = ["+", "-", "\\times", "\\cdot", "\\pm", "\\div"]
RELOPS = ["=", "\\leq", "\\geq", "\\neq", "<", ">"]
SET_OPS = ["\\cup", "\\cap"]
LOGIC = ["\\forall", "\\exists"]
ARROWS = ["\\rightarrow", "\\leftarrow"]
BIGOPS = ["\\sum", "\\int", "\\prod"]
FUNCS = ["\\sin", "\\cos", "\\tan", "\\log", "\\ln",
         "\\exp", "\\max", "\\min"]

# ── Extra symbols for coverage ───────────────────────────────────────

CALCULUS = ["\\infty", "\\partial", "\\nabla"]
PUNCT = [",", ";", ":", "!", "/", "|"]
BRACKETS = ["\\left[", "\\right]", "\\lbrace", "\\rbrace"]
OTHER = ["\\ldots"]

# Flat pool of all "rare" symbols that are underrepresented in v1-v16
MISC_SYMBOLS = (
    ["\\pm", "\\div"]               # extended ops
    + ["<", ">"]                     # extended relations
    + LOGIC + ARROWS                 # logic + arrows
    + PUNCT + BRACKETS               # punctuation + brackets
    + CALCULUS + OTHER               # calculus + misc
)


# Symbols that can be a base for ^{} _{} (no operators, no punctuation)
BASE_POOL = VARS + UPPER + DIGITS + GREEK + GREEK_UPPER


def _pick_base():
    """Pick a symbol suitable as base for ^{} or _{}."""
    return random.choice(BASE_POOL)


def _pick_misc():
    """Pick a random symbol from the underrepresented pool."""
    return random.choice(MISC_SYMBOLS)


# ── Shared expression builder ────────────────────────────────────────
# Used by v1-v10+ to replace the old flat _expr/_term pattern.

MAX_DEPTH = 3


def _shared_short_content(d, atom_fn, base_fn):
    """Short content for sup/sub/frac: varied small structures."""
    v = random.random()
    if v < 0.35:
        return base_fn()
    if v < 0.45:
        # frac with possible sup in num: \frac{x^{2}}{y}
        num = base_fn()
        if random.random() < 0.4:
            num = base_fn() + "^{" + base_fn() + "}"
        return "\\frac{" + num + "}{" + base_fn() + "}"
    if v < 0.55:
        # negative frac: -\frac{x}{y}
        num = base_fn()
        if random.random() < 0.3:
            num = base_fn() + "^{" + base_fn() + "}"
        return "- \\frac{" + num + "}{" + base_fn() + "}"
    if v < 0.65:
        # atom with sup: x^{n}
        return base_fn() + "^{" + base_fn() + "}"
    if v < 0.75:
        # two atoms with op: a + b
        return base_fn() + " " + random.choice(["+", "-"]) + " " + base_fn()
    if v < 0.85:
        # juxtaposition: a b
        return base_fn() + " " + base_fn()
    # negative atom: -x
    return "- " + base_fn()


def _shared_sub_content(base_fn):
    """Content for subscripts: flat atoms only, no sups."""
    n = random.randint(1, 3)
    return " ".join(base_fn() for _ in range(n))


def _shared_term(d, atom_fn, base_fn, struct_fn):
    """Always try to produce something structured, not just a flat atom."""
    if d >= MAX_DEPTH:
        return base_fn()

    v = random.random()
    if v < 0.35 and d <= 1:
        return struct_fn(d)
    if v < 0.45:
        return base_fn() + "^{" + _shared_short_content(d + 1, atom_fn, base_fn) + "}"
    if v < 0.52 and d == 0:
        # Sub content: just atoms and juxtaposition, no sups
        n = random.randint(1, 3)
        sub = " ".join(base_fn() for _ in range(n))
        return base_fn() + "_{" + sub + "}"
    if v < 0.62:
        return " ".join(base_fn() for _ in range(random.randint(2, 3)))
    if v < 0.75:
        return "\\frac{" + _shared_short_content(d + 1, atom_fn, base_fn) + "}{" + _shared_short_content(d + 1, atom_fn, base_fn) + "}"
    return base_fn()


def _shared_expr(d, atom_fn, base_fn, struct_fn):
    """Build expression: structured terms."""
    if d >= 2:
        return _shared_term(d, atom_fn, base_fn, struct_fn)

    if d == 0:
        # Top level: 1-3 terms, ops or juxtaposition
        max_terms = 3
        n = random.choices(range(1, max_terms + 1),
                           weights=[3, 2, 1])[0]
        parts = [_shared_term(d, atom_fn, base_fn, struct_fn)]
        for _ in range(n - 1):
            if random.random() < 0.5:
                parts.append(random.choice(ARITH_OPS))
            parts.append(_shared_term(d, atom_fn, base_fn, struct_fn))
        return " ".join(parts)

    # d == 1: inside a structure (frac, sup, sqrt, abs, etc.)
    # 1-2 structured terms, juxtaposition only (no op chains)
    if random.random() < 0.6:
        return _shared_term(d, atom_fn, base_fn, struct_fn)
    return _shared_term(d, atom_fn, base_fn, struct_fn) + " " + _shared_term(d, atom_fn, base_fn, struct_fn)


def clean_latex(latex: str) -> str:
    """Normalize LaTeX: parse into tree, render back with clean conventions.

    Handles brace stripping, spacing, \\left/\\right for parens.
    """
    from mathnote_ocr.tree_parser.tree_latex import latex_to_tree, tree_to_latex
    try:
        tree = latex_to_tree(latex)
        return tree_to_latex(tree)
    except Exception:
        return latex


