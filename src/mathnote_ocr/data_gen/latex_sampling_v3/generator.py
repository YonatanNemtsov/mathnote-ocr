"""Tree-first expression generator — port of latex_sampling_v2.

Builds expression trees directly using tree_v2 Node/Tree, then renders
to LaTeX via tree_latex. Same structures, weights, and content
distributions as v2, but valid by construction.

Usage:
    from mathnote_ocr.data_gen.latex_sampling_v3.generator import sample
    latex = sample()
"""

from __future__ import annotations

import random

from mathnote_ocr.bbox import BBox
from mathnote_ocr.tree_parser.tree_latex import tree_to_latex
from mathnote_ocr.tree_parser.tree_v2 import ROOT_ID, Edge, Node, Symbol, Tree

# ── Symbol pools ────────────────────────────────────────────────────

LOWER = list("abcdefghijklmnopqrstuvwxyz")
UPPER = [f"{c}_cap" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
DIGITS = list("0123456789")
GREEK = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "theta",
    "lambda",
    "mu",
    "pi",
    "sigma",
    "phi",
    "psi",
    "omega",
]
GREEK_UPPER = ["Gamma_cap", "Delta_cap", "Sigma_up", "Pi_up", "Phi_up", "Psi_up", "Omega_up"]

OPERATORS = ["+", "-", "times", "cdot", "pm", "div"]
RELATIONS = [
    "=",
    "leq",
    "geq",
    "neq",
    "<",
    ">",
    "cup",
    "cap",
    "in",
    "subset",
    "rightarrow",
    "leftarrow",
]
CALCULUS = ["infty", "partial", "nabla"]
MISC = CALCULUS + ["ldots"]
BIG_OPS = ["int", "sum", "prod"]
BASE_SYMBOLS = LOWER + UPPER + DIGITS + GREEK + GREEK_UPPER + CALCULUS
ALL_SYMBOLS = (
    LOWER
    + UPPER
    + DIGITS
    + GREEK
    + GREEK_UPPER
    + OPERATORS
    + RELATIONS
    + CALCULUS
    + [
        "rightarrow",
        "leftarrow",
        "ldots",
        "dot",
        ",",
        ";",
        "colon",
        "!",
        "slash",
        "|",
        "[",
        "]",
        "lbrace",
        "rbrace",
    ]
)

# ── Node ID counter ─────────────────────────────────────────────────

_counter = [0]


def _reset():
    _counter[0] = 0


def _node(name: str, parent_id: int, edge: int, order: int = 0) -> Node:
    sid = _counter[0]
    _counter[0] += 1
    return Node(Symbol(sid, name, BBox(0, 0, 0, 0)), parent_id, edge, order)


# ── Template system ─────────────────────────────────────────────────
# Each template has sample(parent_id, edge, order) -> list[Node]


class T:
    """Base template."""

    def sample(self, parent_id: int, edge: int, order: int = 0) -> list[Node]:
        raise NotImplementedError


class Pool(T):
    """Pick one symbol from a pool."""

    def __init__(self, pool: list[str]):
        self.pool = pool

    def sample(self, parent_id, edge, order=0):
        return [_node(random.choice(self.pool), parent_id, edge, order)]


class OneOf(T):
    """Weighted choice between templates."""

    def __init__(self, *templates: T, weights: list[int | float]):
        self.templates = templates
        self.weights = weights

    def sample(self, parent_id, edge, order=0):
        t = random.choices(self.templates, self.weights)[0]
        return t.sample(parent_id, edge, order)


class Seq(T):
    """Sequence of templates as siblings."""

    def __init__(self, *templates: T):
        self.templates = templates

    def sample(self, parent_id, edge, order=0):
        nodes = []
        for i, t in enumerate(self.templates):
            nodes.extend(t.sample(parent_id, edge, order + i))
        return nodes


class Repeat(T):
    """Repeat a template n times (n chosen by weighted random)."""

    def __init__(self, template: T, counts: list[int], weights: list[int | float]):
        self.template = template
        self.counts = counts
        self.weights = weights

    def sample(self, parent_id, edge, order=0):
        n = random.choices(self.counts, self.weights)[0]
        return Seq(*[self.template] * n).sample(parent_id, edge, order)


class Sup(T):
    """base^{exp}"""

    def __init__(self, base: T, exp: T):
        self.base = base
        self.exp = exp

    def sample(self, parent_id, edge, order=0):
        base_nodes = self.base.sample(parent_id, edge, order)
        base_id = base_nodes[0].symbol.id
        exp_nodes = self.exp.sample(base_id, Edge.SUP)
        return base_nodes + exp_nodes


class Sub(T):
    """base_{sub}"""

    def __init__(self, base: T, sub: T):
        self.base = base
        self.sub = sub

    def sample(self, parent_id, edge, order=0):
        base_nodes = self.base.sample(parent_id, edge, order)
        base_id = base_nodes[0].symbol.id
        sub_nodes = self.sub.sample(base_id, Edge.SUB)
        return base_nodes + sub_nodes


class SupSub(T):
    """base_{sub}^{exp}"""

    def __init__(self, base: T, sub: T, exp: T):
        self.base = base
        self.sub = sub
        self.exp = exp

    def sample(self, parent_id, edge, order=0):
        base_nodes = self.base.sample(parent_id, edge, order)
        base_id = base_nodes[0].symbol.id
        sub_nodes = self.sub.sample(base_id, Edge.SUB)
        exp_nodes = self.exp.sample(base_id, Edge.SUP)
        return base_nodes + sub_nodes + exp_nodes


class Frac(T):
    """\\frac{num}{den}"""

    def __init__(self, num: T, den: T):
        self.num = num
        self.den = den

    def sample(self, parent_id, edge, order=0):
        bar = _node("frac_bar", parent_id, edge, order)
        num_nodes = self.num.sample(bar.symbol.id, Edge.NUM)
        den_nodes = self.den.sample(bar.symbol.id, Edge.DEN)
        return [bar] + num_nodes + den_nodes


class Sqrt(T):
    """\\sqrt{content}"""

    def __init__(self, content: T):
        self.content = content

    def sample(self, parent_id, edge, order=0):
        sq = _node("sqrt", parent_id, edge, order)
        inner = self.content.sample(sq.symbol.id, Edge.SQRT)
        return [sq] + inner


class BigOpBare(T):
    """\\int (no limits)"""

    def __init__(self, op: T | None = None):
        self.op = op or Pool(BIG_OPS)

    def sample(self, parent_id, edge, order=0):
        return self.op.sample(parent_id, edge, order)


class BigOpLow(T):
    """\\int_{lo}"""

    def __init__(self, op: T, lo: T):
        self.op = op
        self.lo = lo

    def sample(self, parent_id, edge, order=0):
        op_nodes = self.op.sample(parent_id, edge, order)
        op_id = op_nodes[0].symbol.id
        lo_nodes = self.lo.sample(op_id, Edge.LOWER)
        return op_nodes + lo_nodes


class BigOpHigh(T):
    """\\int^{hi}"""

    def __init__(self, op: T, hi: T):
        self.op = op
        self.hi = hi

    def sample(self, parent_id, edge, order=0):
        op_nodes = self.op.sample(parent_id, edge, order)
        op_id = op_nodes[0].symbol.id
        hi_nodes = self.hi.sample(op_id, Edge.UPPER)
        return op_nodes + hi_nodes


class BigOpLowHigh(T):
    """\\int_{lo}^{hi}"""

    def __init__(self, op: T, lo: T, hi: T):
        self.op = op
        self.lo = lo
        self.hi = hi

    def sample(self, parent_id, edge, order=0):
        op_nodes = self.op.sample(parent_id, edge, order)
        op_id = op_nodes[0].symbol.id
        lo_nodes = self.lo.sample(op_id, Edge.LOWER)
        hi_nodes = self.hi.sample(op_id, Edge.UPPER)
        return op_nodes + lo_nodes + hi_nodes


class Lim(T):
    """\\lim_{lo} — expands to l, i, m with LOWER on m."""

    def __init__(self, lo: T):
        self.lo = lo

    def sample(self, parent_id, edge, order=0):
        l = _node("l", parent_id, edge, order)
        i = _node("i", parent_id, edge, order + 1)
        m = _node("m", parent_id, edge, order + 2)
        lo_nodes = self.lo.sample(m.symbol.id, Edge.LOWER)
        return [l, i, m] + lo_nodes


class Func(T):
    """\\sin etc. — expands to individual letters, sup/sub on last."""

    def __init__(self, letters: tuple[str, ...], arg: T | None = None, sup: T | None = None):
        self.letters = letters
        self.arg = arg
        self.sup = sup

    def sample(self, parent_id, edge, order=0):
        nodes = []
        for i, ch in enumerate(self.letters):
            nodes.append(_node(ch, parent_id, edge, order + i))
        last_id = nodes[-1].symbol.id
        if self.sup:
            nodes.extend(self.sup.sample(last_id, Edge.SUP))
        extra_order = order + len(self.letters)
        if self.arg:
            nodes.extend(self.arg.sample(parent_id, edge, extra_order))
        return nodes


class Prime(T):
    """base^{\\prime ...}"""

    def __init__(self, base: T, counts: list[int] = [1, 2, 3], weights: list[int] = [6, 2, 1]):
        self.base = base
        self.counts = counts
        self.weights = weights

    def sample(self, parent_id, edge, order=0):
        base_nodes = self.base.sample(parent_id, edge, order)
        base_id = base_nodes[0].symbol.id
        n = random.choices(self.counts, self.weights)[0]
        primes = [_node("prime", base_id, Edge.SUP, i) for i in range(n)]
        return base_nodes + primes


class Parens(T):
    """( content ) as siblings, with optional sup/sub on close paren."""

    def __init__(self, content: T, sup: T | None = None, sub: T | None = None):
        self.content = content
        self.sup = sup
        self.sub = sub

    def sample(self, parent_id, edge, order=0):
        open_p = _node("(", parent_id, edge, order)
        inner = self.content.sample(parent_id, edge, order + 1)
        sibling_count = sum(1 for nd in inner if nd.parent_id == parent_id)
        close_p = _node(")", parent_id, edge, order + 1 + sibling_count)
        nodes = [open_p] + inner + [close_p]
        if self.sup:
            nodes.extend(self.sup.sample(close_p.symbol.id, Edge.SUP))
        if self.sub:
            nodes.extend(self.sub.sample(close_p.symbol.id, Edge.SUB))
        return nodes


class Binom(T):
    """\\binom{top}{bot}"""

    def __init__(self, top: T, bot: T):
        self.top = top
        self.bot = bot

    def sample(self, parent_id, edge, order=0):
        open_p = _node("(", parent_id, edge, order)
        top_nodes = self.top.sample(open_p.symbol.id, Edge.NUM)
        bot_nodes = self.bot.sample(open_p.symbol.id, Edge.DEN)
        close_p = _node(")", open_p.symbol.id, Edge.MATCH, 0)
        return [open_p] + top_nodes + bot_nodes + [close_p]


# ── Content definitions (mirroring v2 exactly) ──────────────────────

_Base = Pool(BASE_SYMBOLS)
_Op = Pool(OPERATORS)
_RelOp = Pool(RELATIONS)
_BigOp = Pool(BIG_OPS)

Atom1 = OneOf(
    Pool(LOWER),
    Pool(UPPER),
    Pool(DIGITS),
    Pool(GREEK),
    Pool(GREEK_UPPER),
    Pool(MISC),
    weights=[26, 26, 10, 13, 7, 4],
)

SubContent = Repeat(Atom1, counts=[1, 2, 3], weights=[10, 2, 1])

# Limit content pieces
_LimitSup = Sup(_Base, SubContent)
_SimpleFrac = Frac(Atom1, Atom1)
_SimpleSqrt = Sqrt(Atom1)
_SimpleParensContent = OneOf(
    Repeat(Atom1, counts=[1, 2, 3], weights=[4, 4, 2]),
    Frac(Atom1, Atom1),
    Seq(Atom1, _Op, Atom1),
    weights=[3, 1, 2],
)
_SimpleParens = Parens(_SimpleParensContent)
LimitPiece = OneOf(
    Atom1,
    _Op,
    _LimitSup,
    _SimpleFrac,
    _SimpleSqrt,
    _RelOp,
    _SimpleParens,
    weights=[12, 3, 3, 2, 1, 2, 1],
)
LimitContent = Repeat(LimitPiece, counts=[1, 2, 3], weights=[6, 2, 1])

# Big ops with limits (for use inside sups/fracs)
_BigOpWithLimits = OneOf(
    BigOpLow(_BigOp, lo=LimitContent),
    BigOpHigh(_BigOp, hi=LimitContent),
    BigOpLowHigh(_BigOp, lo=LimitContent, hi=LimitContent),
    weights=[1, 1, 2],
)

# Sup content
_SupFrac = Frac(LimitContent, LimitContent)
_NegSupFrac = Seq(_Op, _SupFrac)

SupAtom1 = OneOf(
    Pool(LOWER),
    Pool(UPPER),
    Pool(DIGITS),
    Pool(GREEK),
    Pool(GREEK_UPPER),
    Pool(MISC),
    BigOpBare(_BigOp),
    _BigOpWithLimits,
    _SupFrac,
    _NegSupFrac,
    _SimpleParens,
    weights=[26, 26, 10, 13, 7, 4, 2, 2, 2, 2, 1],
)
SupContent = Repeat(SupAtom1, counts=[1, 2, 3], weights=[6, 2, 1])

# Frac content (inner — no nesting)
_SupAtom = Sup(_Base, SupContent)
FracPieceInner = OneOf(Atom1, _Op, _SupAtom, weights=[4, 1, 1])
FracContentInner = Repeat(FracPieceInner, counts=[1, 2, 3], weights=[4, 4, 2])

# Frac content (outer — can nest one level)
_NestedFrac = Frac(FracContentInner, FracContentInner)
_ParensOuter = Parens(FracContentInner)  # placeholder, updated below
FracPiece = OneOf(
    Atom1, _Op, _NestedFrac, _SupAtom, _RelOp, _ParensOuter, weights=[6, 1, 1, 2, 1, 1]
)
FracContent = Repeat(FracPiece, counts=[1, 2, 3, 4, 5, 6, 7], weights=[7, 4, 4, 3, 2, 1, 1])

# Now update parens to use full FracContent (with nested fracs)
_ParensOuter.content = FracContent


# ── Structure generators for random_expr ────────────────────────────
# Each takes (budget, depth, parent_id, edge, order) → (nodes, n_counted)


def _gen_atom(budget, depth, parent_id, edge, order):
    return Pool(ALL_SYMBOLS).sample(parent_id, edge, order), 1


def _gen_sup(budget, depth, parent_id, edge, order):
    nodes = Sup(_Base, SupContent).sample(parent_id, edge, order)
    return nodes, 2


def _gen_sub(budget, depth, parent_id, edge, order):
    nodes = Sub(_Base, SubContent).sample(parent_id, edge, order)
    return nodes, 2


def _gen_sup_sub(budget, depth, parent_id, edge, order):
    nodes = SupSub(_Base, SubContent, SupContent).sample(parent_id, edge, order)
    return nodes, 3


def _gen_frac(budget, depth, parent_id, edge, order):
    nodes = Frac(FracContent, FracContent).sample(parent_id, edge, order)
    return nodes, 3


def _gen_sqrt(budget, depth, parent_id, edge, order):
    sq = _node("sqrt", parent_id, edge, order)
    inner, n = random_expr(min(budget - 1, 4), depth + 1, parent_id=sq.symbol.id, edge=Edge.SQRT)
    return [sq] + inner, 1 + n


def _gen_parens(budget, depth, parent_id, edge, order):
    open_p = _node("(", parent_id, edge, order)
    inner, n = random_expr(
        min(budget - 2, 8), depth + 1, parent_id=parent_id, edge=edge, order_start=order + 1
    )
    sibling_count = sum(1 for nd in inner if nd.parent_id == parent_id)
    close_p = _node(")", parent_id, edge, order + 1 + sibling_count)
    nodes = [open_p] + inner + [close_p]
    # Sometimes add sup/sub on close paren
    r = random.random()
    if r < 0.05:
        n_primes = random.choices([1, 2, 3], weights=[6, 2, 1])[0]
        for i in range(n_primes):
            nodes.append(_node("prime", close_p.symbol.id, Edge.SUP, i))
    elif r < 0.4:
        exp = SupContent.sample(close_p.symbol.id, Edge.SUP)
        nodes.extend(exp)
    return nodes, 2 + n


def _gen_big_op(budget, depth, parent_id, edge, order):
    return BigOpBare(_BigOp).sample(parent_id, edge, order), 1


def _gen_big_op_low(budget, depth, parent_id, edge, order):
    return BigOpLow(_BigOp, LimitContent).sample(parent_id, edge, order), 2


def _gen_big_op_high(budget, depth, parent_id, edge, order):
    return BigOpHigh(_BigOp, LimitContent).sample(parent_id, edge, order), 2


def _gen_big_op_low_high(budget, depth, parent_id, edge, order):
    return BigOpLowHigh(_BigOp, LimitContent, LimitContent).sample(parent_id, edge, order), 3


def _gen_lim(budget, depth, parent_id, edge, order):
    return Lim(LimitContent).sample(parent_id, edge, order), 2


def _gen_binom(budget, depth, parent_id, edge, order):
    open_p = _node("(", parent_id, edge, order)
    top, tn = random_expr(
        max(2, budget - 1), depth + 1, parent_id=open_p.symbol.id, edge=Edge.NUM, exclude={"binom"}
    )
    bot, bn = random_expr(
        max(2, budget - 1 - tn),
        depth + 1,
        parent_id=open_p.symbol.id,
        edge=Edge.DEN,
        exclude={"binom"},
    )
    close_p = _node(")", open_p.symbol.id, Edge.MATCH, 0)
    return [open_p] + top + bot + [close_p], tn + bn


def _gen_prime(budget, depth, parent_id, edge, order):
    pool = LOWER + UPPER + GREEK + GREEK_UPPER
    nodes = Prime(Pool(pool)).sample(parent_id, edge, order)
    return nodes, 2


# (name, fn, min_budget, weight, max_depth)
STRUCTURES = [
    ("atom", _gen_atom, 1, 11, 4),
    ("sup", _gen_sup, 2, 2, 1),
    ("sub", _gen_sub, 2, 2, 1),
    ("sup_sub", _gen_sup_sub, 3, 1, 1),
    ("frac", _gen_frac, 3, 4, 2),
    ("sqrt", _gen_sqrt, 2, 1, 2),
    ("big_op", _gen_big_op, 1, 1, 2),
    ("big_op_low", _gen_big_op_low, 2, 1, 2),
    ("big_op_high", _gen_big_op_high, 2, 1, 2),
    ("big_op_low_high", _gen_big_op_low_high, 3, 1, 2),
    ("parens", _gen_parens, 3, 1, 2),
    ("lim", _gen_lim, 2, 1, 2),
    ("binom", _gen_binom, 3, 2, 1),
    ("prime", _gen_prime, 2, 1, 2),
]


# ── Main generator ────────────────────���─────────────────────────────


def random_term(budget, depth, parent_id, edge, order, exclude=None):
    """Pick a random structure and generate one term. Returns (nodes, n_counted)."""
    if budget <= 0:
        return _gen_atom(budget, depth, parent_id, edge, order)

    opts = [
        (name, fn, w)
        for name, fn, min_b, w, max_d in STRUCTURES
        if budget >= min_b and depth <= max_d and (exclude is None or name not in exclude)
    ]
    if not opts:
        return _gen_atom(budget, depth, parent_id, edge, order)

    names, fns, weights = zip(*opts)
    fn = random.choices(fns, weights, k=1)[0]
    return fn(budget, depth, parent_id, edge, order)


def random_expr(budget, depth, parent_id=ROOT_ID, edge=Edge.ROOT, order_start=0, exclude=None):
    """Generate 1+ juxtaposed terms within the budget. Returns (nodes, n_counted)."""
    if budget <= 0:
        return _gen_atom(0, depth, parent_id, edge, order_start)

    max_terms = max(1, 7 - depth * 2)
    all_nodes: list[Node] = []
    used = 0
    order = order_start

    while used < budget and (order - order_start) < max_terms:
        nodes, n = random_term(budget - used, depth, parent_id, edge, order, exclude)
        if n == 0:
            break
        all_nodes.extend(nodes)
        used += n
        root_added = sum(1 for nd in nodes if nd.parent_id == parent_id)
        order += max(1, root_added)
        if random.random() < 0.2:
            break

    if not all_nodes:
        return _gen_atom(0, depth, parent_id, edge, order_start)

    return all_nodes, used


def sample(max_symbols=None) -> str:
    """Generate a random LaTeX expression."""
    _reset()

    if max_symbols is None:
        sizes = list(range(2, 41))
        weights = [max(1, 14 - abs(s - 15)) for s in sizes]
        max_symbols = random.choices(sizes, weights, k=1)[0]

    nodes, _ = random_expr(max_symbols, depth=0)
    tree = Tree(tuple(nodes))
    return tree_to_latex(tree)


if __name__ == "__main__":
    random.seed(42)
    for _ in range(30):
        print(sample())
