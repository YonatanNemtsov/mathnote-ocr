"""Forced alignment: match strokes to the symbols of a known LaTeX.

Recognition has to guess what was written; alignment is told. Given the
strokes and the expression's LaTeX, it finds which strokes form which
symbol: an exact cover of the strokes by candidate groups whose labels use
up exactly the symbols the LaTeX contains, maximising the classifier's
log-probability. Used to label collected data — every confirmed expression
becomes symbol-level training data, even where the engine read it wrong.
"""

from __future__ import annotations

import heapq
import math
import time
from collections import Counter
from dataclasses import dataclass, field

from mathnote_ocr.bbox import BBox
from mathnote_ocr.latex_utils.expr_aug import LNode, parse_latex
from mathnote_ocr.latex_utils.glyphs import SYMBOL_TO_LATEX

# Spacing and layout commands that draw nothing
_INVISIBLE = {"\\,", "\\;", "\\:", "\\!", "\\quad", "\\qquad", "\\ ", "\\EXPR"}
# Brace commands and their class names
_BRACES = {"\\{": "lbrace", "\\}": "rbrace", "\\lbrace": "lbrace", "\\rbrace": "rbrace"}
# Commands written with another class's glyph
_ALIASES = {"\\to": "\\rightarrow", "\\gets": "\\leftarrow", "\\lvert": "|", "\\rvert": "|"}
# Accents: a mark written above the base symbol, labelled by meaning. The
# parser reads \hat{x} as "\hat" followed by its argument, so the mark
# is one more symbol and the argument is walked as usual.
ACCENTS = {"\\hat": "accent_hat", "\\bar": "accent_bar", "\\vec": "accent_vec",
           "\\dot": "accent_dot", "\\tilde": "accent_tilde"}
# Function names, handwritten letter by letter. Only these are spelled out:
# any other unknown command must fail, not silently become letters.
_FUNCTIONS = {
    "\\sin", "\\cos", "\\tan", "\\cot", "\\sec", "\\csc", "\\sinh", "\\cosh", "\\tanh",
    "\\arcsin", "\\arccos", "\\arctan", "\\log", "\\ln", "\\exp", "\\lim", "\\max", "\\min",
    "\\sup", "\\inf", "\\det", "\\arg", "\\gcd", "\\deg", "\\dim", "\\ker",
}


# Matrix / cases environments, for their symbols: the delimiters and the
# cells ("&" and "\\\\" are layout, not ink)
_ENVIRONMENTS = {
    "pmatrix": (r"\left(", r"\right)"), "bmatrix": (r"\left[", r"\right]"),
    "vmatrix": ("|", "|"), "Bmatrix": (r"\{", r"\}"), "matrix": ("", ""), "cases": (r"\{", ""),
}


def _flatten_environments(latex: str) -> str:
    for env, (left, right) in _ENVIRONMENTS.items():
        latex = latex.replace(f"\\begin{{{env}}}", f" {left} ").replace(f"\\end{{{env}}}", f" {right} ")
    return latex.replace("\\\\", " ").replace("&", " ")


class UnknownSymbol(ValueError):
    """The LaTeX contains something the classifier has no class for."""


def _latex_to_class(label_names: list[str]) -> dict[str, str]:
    """How each class is written in LaTeX, inverted: latex -> class name.

    frac_bar is excluded — it is written '-' like minus, and fractions are
    recognised structurally (\\frac) instead.
    """
    inverse = {}
    for name in label_names:
        if name == "frac_bar":
            continue
        inverse[SYMBOL_TO_LATEX.get(name, name)] = name
    inverse["'"] = inverse.get("\\prime", "prime")
    return inverse


def latex_to_labels(latex: str, label_names: list[str]) -> list[str]:
    """The symbols (class names) a handwritten rendering of *latex* contains.

    Raises UnknownSymbol for LaTeX the parser can't read or symbols outside
    the classifier's vocabulary.
    """
    tree = parse_latex(_flatten_environments(latex))
    if tree is None:
        raise UnknownSymbol(f"could not parse {latex!r}")
    to_class = _latex_to_class(label_names)
    out: list[str] = []
    blackboard = False      # \mathbb seen: the next letter is ℝ, ℕ, …

    def walk(node: LNode) -> None:
        nonlocal blackboard
        if node.kind == "char":
            if node.text.isspace():
                return
            if blackboard:
                blackboard = False
                name = f"bb_{node.text}" if f"bb_{node.text}" in label_names else None
                if name is None:
                    raise UnknownSymbol(f"no class for \\mathbb{{{node.text}}}")
                out.append(name)
                return
            name = to_class.get(node.text)
            if name is None:
                raise UnknownSymbol(f"no class for {node.text!r}")
            out.append(name)
        elif node.kind == "command":
            cmd = _ALIASES.get(node.text, node.text)
            if cmd in _INVISIBLE:
                return
            if cmd == "\\mathbb":
                blackboard = True
                return
            if cmd in ACCENTS:
                if ACCENTS[cmd] not in label_names:
                    raise UnknownSymbol(f"no class for {cmd!r}")
                out.append(ACCENTS[cmd])
                return
            if cmd in _BRACES:
                out.append(_BRACES[cmd])
                return
            name = to_class.get(cmd)
            if name is not None:
                out.append(name)
                return
            # Function names (\sin, \lim, ...) are written letter by letter
            if cmd in _FUNCTIONS and all(ch in to_class for ch in cmd[1:]):
                out.extend(to_class[ch] for ch in cmd[1:])
                return
            raise UnknownSymbol(f"no class for {cmd!r}")
        elif node.kind == "frac":
            out.append("frac_bar")
            for child in node.children:
                walk(child)
        elif node.kind == "sqrt":
            # \sqrt[n]{x}: the parser keeps the index's "]" inside the root
            # as a symbol (not supported yet)
            inner = [c for child in node.children for c in ([child] + list(child.children))]
            if any(c.kind == "char" and c.text == "]" for c in inner):
                raise UnknownSymbol("\\sqrt with an index")
            out.append("sqrt")
            for child in node.children:
                walk(child)
        elif node.kind == "binom":
            out.extend([to_class["("], to_class[")"]])
            for child in node.children:
                walk(child)
        else:   # seq, sup, sub, func
            for child in node.children:
                walk(child)

    walk(tree)
    return out


# Classes drawn identically, told apart only by role: a big-operator \sum is
# often labelled Sigma_up in the data (same for \prod / Pi_up). Alignment
# scores a target label by the probability of its whole group.
EQUIVALENT = [frozenset({"sum", "Sigma_up"}), frozenset({"prod", "Pi_up"})]
_EQUIV = {name: group for group in EQUIVALENT for name in group}


# Minus and fraction bar are the same stroke; only what surrounds it tells
# them apart. Alignment scores both by their pooled probability, then gives
# the fraction-bar labels to the bars that have ink above and below them.
BARS = frozenset({"-", "frac_bar"})


def same_symbol(a: str, b: str) -> bool:
    return a == b or (a in _EQUIV and b in _EQUIV[a])


def labels_match(a: list[str], b: list[str]) -> bool:
    """Same multiset of symbols, up to EQUIVALENT classes."""
    canon = lambda xs: Counter(min(_EQUIV.get(x, {x})) for x in xs)
    return canon(a) == canon(b)


# ── Alignment search ─────────────────────────────────────────────────


def _ldots_candidates(strokes, groups, ocr, cache, source_size) -> list[frozenset[int]]:
    """Extra candidates for \ldots: three dots in a row.

    The grouper rejects them (tiny strokes spanning a wide group), so offer
    every run of three consecutive strokes the classifier itself reads as
    a dot, and let it score the rendered triple like any other group.
    """
    from mathnote_ocr.engine.grouper import classify_groups

    ids = [s.id for s in strokes]
    dots = sorted(
        (p for p in range(len(strokes))
         if (r := cache.get(frozenset([ids[p]]))) and r.alternatives
         and r.alternatives[0][0] in ("dot", "cdot")),
        key=lambda p: strokes[p].bbox.x,
    )
    existing = set(groups)
    extra = [frozenset(dots[i:i + 3]) for i in range(len(dots) - 2)]
    extra = [g for g in extra if g not in existing]
    classify_groups(strokes, extra, ocr.classifier, cache=cache, source_size=source_size)
    return extra


COMPACT_MAX_STROKES = 4      # strokes in one compact candidate
COMPACT_BOX = 1.6            # its box: at most this many typical stroke heights wide and tall
COMPACT_NEAR = 0.8           # strokes this close (typical heights, box to box) are neighbours
COMPACT_LIMIT = 4000         # candidates per ink at most (a dense ink stops adding)


def _compact_candidates(strokes, groups, ocr, cache, source_size, box_size: float = COMPACT_BOX) -> list[frozenset[int]]:
    """Extra candidates for symbols whose strokes lie apart: every set of 2
    to COMPACT_MAX_STROKES neighbouring strokes whose box together is about
    one symbol's size — a pi's bar and two legs, a colon's two dots, an i's
    dot over its stem, the two bars of =. The grouper weighs strokes as one
    symbol only when they are close; these symbols' parts often are not, so
    the right group was no choice at all: a colon's second dot or an i's dot
    went to a neighbour. Only candidates: the classifier scores each like
    any other group, and the expression's symbols decide.

    Sizes in the ink's own unit: the median height of its strokes that are
    not tiny (a dot says nothing about the writing's size)."""
    from mathnote_ocr.engine.grouper import classify_groups

    n = len(strokes)
    if n < 2:
        return []
    box = [s.bbox for s in strokes]
    diag = [(b.w ** 2 + b.h ** 2) ** 0.5 for b in box]
    typical = sorted(diag)[n // 2] or 1.0
    heights = sorted(box[p].h for p in range(n) if diag[p] >= 0.4 * typical) or [typical]
    unit = heights[len(heights) // 2] or typical

    def gap(a, b):
        dx = max(0.0, max(box[a].x, box[b].x) - min(box[a].x2, box[b].x2))
        dy = max(0.0, max(box[a].y, box[b].y) - min(box[a].y2, box[b].y2))
        return (dx * dx + dy * dy) ** 0.5

    near = {p: [q for q in range(n) if q != p and gap(p, q) <= COMPACT_NEAR * unit] for p in range(n)}
    limit = box_size * unit
    found: set[frozenset[int]] = set()
    frontier = [frozenset([p]) for p in range(n)]
    for _size in range(2, COMPACT_MAX_STROKES + 1):
        grown = set()
        for g in frontier:
            for p in g:
                for q in near[p]:
                    if q in g:
                        continue
                    h = g | {q}
                    if h in grown or h in found:
                        continue
                    x0 = min(box[i].x for i in h); x1 = max(box[i].x2 for i in h)
                    y0 = min(box[i].y for i in h); y1 = max(box[i].y2 for i in h)
                    if x1 - x0 <= limit and y1 - y0 <= limit:
                        grown.add(h)
        found |= grown
        frontier = list(grown)
        if len(found) > COMPACT_LIMIT or not frontier:
            break
    existing = set(groups)
    extra = [g for g in found if g not in existing][:COMPACT_LIMIT]
    classify_groups(strokes, extra, ocr.classifier, cache=cache, source_size=source_size)
    return extra


LOG_FLOOR = math.log(1e-12)


@dataclass
class AlignedSymbol:
    label: str
    stroke_ids: list[int]
    logp: float


@dataclass
class Alignment:
    symbols: list[AlignedSymbol]
    logp: float                 # total log-probability of the chosen labels
    complete: bool              # True: optimal among candidates; False: node budget hit
    nodes: int
    seconds: float
    candidates: int = 0
    notes: list[str] = field(default_factory=list)
    order_inversions: int = 0   # pairs written in an order contradicting the LaTeX (with order_weight)
    alternatives: list = field(default_factory=list)   # [(logp, symbols)], the next best covers (alternatives=)

    @property
    def weakest(self) -> float:
        """The lowest per-symbol log-probability — the confidence signal.

        On the 285 handwritten expressions, 2 of the 3 wrong alignments had
        the 1st and 3rd lowest value: review alignments weakest-first.
        (The tree-parser structure check is NOT a usable signal: it
        rejects ~30% of correct alignments.)
        """
        return min((s.logp for s in self.symbols), default=0.0)


def align(
    ocr,
    strokes,
    labels: list[str],
    *,
    canvas_size: int | None = None,
    node_budget: int = 200_000,
    order_weight: float = 0.0,
    order_by: str = "time",
    k_best: int = 50,
    alternatives: int = 0,
    compact: bool = False,
    compact_box: float | dict | None = None,
) -> Alignment | None:
    """Assign strokes to the symbols in *labels* (e.g. from latex_to_labels).

    Exact cover of the strokes by the grouper's candidate groups — all of
    them, unfiltered — where the groups' labels use up *labels* exactly,
    maximising the sum of log P(label | group). Returns None when no such
    cover exists among the candidates.

    With *order_weight* > 0, the *k_best* best covers are re-ranked by
    log P minus order_weight x the pairs of symbols written in an order
    contradicting *labels*' order (order_inversions): the labels are a
    sequence, not a bag — on a stranger's handwriting, where the classifier
    is unsure, the best bag shuffles labels between similar groups.
    *order_by*: "time" (a symbol's first stroke) or "x" (its left edge).
    Fraction bars take no part (written before or after their numerator).

    *compact*: also the groups of neighbouring strokes that fit one
    symbol's box (_compact_candidates: pi, colon, i, =) — off by default.
    *compact_box*: that box, in the ink's typical stroke heights — a number,
    or {label: size} (the largest of the expression's symbols is used: a
    product sign is far bigger than a colon); default COMPACT_BOX.

    *alternatives* > 0: the next best covers too, best first
    (Alignment.alternatives) — for a caller with its own check (the
    expression's tree against the ink) to take the best one that passes.
    """
    from mathnote_ocr.api import _autocanvas, _normalize_strokes
    from mathnote_ocr.engine.grouper import GrouperCache, generate_candidates

    t0 = time.perf_counter()
    stroke_objs = _normalize_strokes(strokes)
    if not stroke_objs or not labels:
        return None
    n = len(stroke_objs)
    cs = canvas_size if canvas_size is not None else _autocanvas(stroke_objs, ocr._default_canvas_size)
    cache = GrouperCache()
    groups, _ = generate_candidates(stroke_objs, ocr.classifier, params=ocr.grouper_params,
                                    cache=cache, source_size=cs)
    ids = [s.id for s in stroke_objs]
    if "ldots" in labels:
        groups = groups + _ldots_candidates(stroke_objs, groups, ocr, cache, cs)
    if compact:
        # the box: the largest symbol's in the expression (compact_box per label, measured), else COMPACT_BOX
        box_size = (max([compact_box.get(lab, COMPACT_BOX) for lab in labels] + [COMPACT_BOX])
                    if isinstance(compact_box, dict) else (compact_box or COMPACT_BOX))
        groups = groups + _compact_candidates(stroke_objs, groups, ocr, cache, cs, box_size)
    index = {name: i for i, name in enumerate(ocr.classifier.label_names)}

    # log P(label | group) for the labels we need, equivalent classes pooled.
    # Minus and fraction bar are searched as one label, "-" (their
    # probabilities pooled), and told apart afterwards by _assign_bars.
    fractions = labels.count("frac_bar")
    target = Counter("-" if lab in BARS else lab for lab in labels)
    need = sorted(target)
    pooled = lambda lab: BARS if lab in BARS else _EQUIV.get(lab, {lab})
    cols = {lab: [index[x] for x in pooled(lab) if x in index] for lab in need}
    scored = []   # (group positions, {label: logp})
    for g in groups:
        result = cache[frozenset(ids[p] for p in g)]
        probs = result.probs
        if probs is None:
            continue
        lp = {}
        for lab in need:
            p = float(sum(probs[c] for c in cols[lab]))
            lp[lab] = math.log(p) if p > 0 else LOG_FLOOR
        scored.append((g, lp))

    by_stroke: dict[int, list[int]] = {i: [] for i in range(n)}
    for gi, (g, _) in enumerate(scored):
        for p in g:
            by_stroke[p].append(gi)
    max_size = max((len(g) for g, _ in scored), default=1)
    # Per label, groups by descending score — for the optimistic bound
    ranked = {lab: sorted(range(len(scored)), key=lambda gi, lab=lab: -scored[gi][1][lab]) for lab in need}

    k = max(max(1, k_best) if order_weight > 0 else 1, alternatives + 1)
    found: list[tuple[float, int, list]] = []          # min-heap of the k best covers (logp, tiebreak, choice)

    def kth() -> float:
        return found[0][0] if len(found) >= k else -math.inf

    nodes = 0
    remaining = dict(target)
    chosen: list[tuple[int, str]] = []
    # Best score seen per state (strokes left, symbols left): a state reached
    # again with no better score can't lead anywhere better
    seen: dict[tuple, list[float]] = {}

    def bound(uncovered: frozenset[int]) -> float:
        """Optimistic: each remaining symbol gets the best group still available
        (disjointness ignored)."""
        total = 0.0
        for lab, k in remaining.items():
            if not k:
                continue
            for gi in ranked[lab]:
                if scored[gi][0] <= uncovered:
                    total += k * scored[gi][1][lab]
                    break
            else:
                return -math.inf
        return total

    def search(uncovered: frozenset[int], score: float) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > node_budget:
            return
        left = sum(remaining.values())
        if not uncovered:
            if left == 0 and score > kth():
                entry = (score, nodes, list(chosen))
                if len(found) < k:
                    heapq.heappush(found, entry)
                else:
                    heapq.heapreplace(found, entry)
            return
        # Each symbol takes 1..max_size strokes
        if left == 0 or len(uncovered) < left or len(uncovered) > left * max_size:
            return
        state = (uncovered, tuple(sorted((lab, c) for lab, c in remaining.items() if c)))
        # the k best ways into a state: a worse one can't complete any better
        tops = seen.setdefault(state, [])
        if len(tops) >= k and score <= tops[0]:
            return
        if len(tops) >= k:
            heapq.heapreplace(tops, score)
        else:
            heapq.heappush(tops, score)
        if score + bound(uncovered) <= kth():
            return
        # Most constrained stroke first
        pick_opts = None
        for s_ in uncovered:
            opts = [(scored[gi][1][lab], gi, lab)
                    for gi in by_stroke[s_] if scored[gi][0] <= uncovered
                    for lab in need if remaining[lab]]
            if pick_opts is None or len(opts) < len(pick_opts):
                pick_opts = opts
                if not opts:
                    return
        pick_opts.sort(reverse=True)
        for value, gi, lab in pick_opts:
            remaining[lab] -= 1
            chosen.append((gi, lab))
            search(uncovered - scored[gi][0], score + value)
            chosen.pop()
            remaining[lab] += 1
            if nodes > node_budget:
                return

    search(frozenset(range(n)), 0.0)
    elapsed = time.perf_counter() - t0
    if not found:
        return None
    seq = ["-" if lab in BARS else lab for lab in labels]

    def key(gi: int) -> float:
        g = scored[gi][0]
        return min(g) if order_by == "time" else min(stroke_objs[p].bbox.x for p in g)

    def ranked_cover(entry):
        logp, _t, choice = entry
        inv = order_inversions(seq, [(lab, key(gi)) for gi, lab in choice])
        return logp - order_weight * inv, logp, inv, choice

    ranked = sorted((ranked_cover(e) for e in found), key=lambda r: -r[0])

    def symbols_of(choice) -> list[AlignedSymbol]:
        syms = [AlignedSymbol(label=lab, stroke_ids=sorted(ids[p] for p in scored[gi][0]), logp=scored[gi][1][lab])
                for gi, lab in choice]
        boxes = [BBox.union_all([stroke_objs[p].bbox for p in scored[gi][0]]) for gi, _ in choice]
        _assign_bars(syms, boxes, fractions)
        return syms

    _value, logp, inversions, choice = ranked[0]
    return Alignment(symbols=symbols_of(choice), logp=logp, complete=nodes <= node_budget, nodes=nodes,
                     seconds=elapsed, candidates=len(scored), order_inversions=inversions,
                     alternatives=[(lp, symbols_of(ch)) for _v, lp, _i, ch in ranked[1:alternatives + 1]])


def order_inversions(sequence: list[str], placed: list[tuple[str, float]]) -> int:
    """Pairs of symbols placed (label, position: first stroke or left edge)
    in an order contradicting *sequence* (the LaTeX's labels in order).
    Equal labels are matched in order (the first one written is the first
    in the LaTeX); bars ("-") take no part."""
    slots: dict[str, list[int]] = {}
    for i, lab in enumerate(sequence):
        if lab != "-":
            slots.setdefault(lab, []).append(i)
    by_label: dict[str, list[float]] = {}
    for lab, pos in placed:
        if lab != "-" and lab not in BARS:
            by_label.setdefault(lab, []).append(pos)
    order = []                                  # (position, place in the LaTeX)
    for lab, positions in by_label.items():
        for pos, i in zip(sorted(positions), slots.get(lab, [])):
            order.append((pos, i))
    order.sort()
    idx = [i for _p, i in order]
    return sum(1 for a in range(len(idx)) for b in range(a + 1, len(idx)) if idx[a] > idx[b])


def _assign_bars(symbols: list[AlignedSymbol], boxes: list[BBox], fractions: int) -> None:
    """Relabel the horizontal bars: the *fractions* bars that have a
    numerator and a denominator become frac_bar, the rest minus.

    A bar's numerator is the symbols centred over it with no other bar in
    between (in \\frac{x}{2 - y} the x is over the minus too, but the
    fraction bar separates them); likewise its denominator. Bars with both
    come first, wider first among equals.
    """
    bars = [i for i, s in enumerate(symbols) if s.label in BARS]

    def separated(t: int, b: int) -> bool:
        lo, hi = sorted((boxes[t].cy, boxes[b].cy))
        return any(c != b and boxes[c].x <= boxes[t].cx <= boxes[c].x2 and lo < boxes[c].cy < hi
                   for c in bars)

    def side(b: int, above: bool) -> bool:
        return any(t not in bars and boxes[b].x <= boxes[t].cx <= boxes[b].x2
                   and (boxes[t].cy < boxes[b].cy if above else boxes[t].cy > boxes[b].cy)
                   and not separated(t, b)
                   for t in range(len(symbols)))

    ranked = sorted(bars, key=lambda b: (side(b, True) and side(b, False), boxes[b].w), reverse=True)
    for rank, b in enumerate(ranked):
        symbols[b].label = "frac_bar" if rank < fractions else "-"


def parse_aligned(ocr, strokes, alignment: Alignment) -> str:
    """The LaTeX the tree parser builds from the aligned symbols.

    A structure check: if it reproduces the target LaTeX, the alignment is
    consistent end to end (labels and layout).
    """
    from mathnote_ocr.api import _normalize_strokes
    from mathnote_ocr.engine.stroke import compute_bbox
    from mathnote_ocr.expression import DetectedSymbol

    by_id = {s.id: s for s in _normalize_strokes(strokes)}
    detected = []
    for sym in alignment.symbols:
        group = [by_id[i] for i in sym.stroke_ids]
        detected.append(DetectedSymbol(name=sym.label, bbox=compute_bbox(group), strokes=group,
                                       confidence=math.exp(sym.logp)))
    detected.sort(key=lambda d: d.bbox.x)
    latex, _conf, _tree, _ev = ocr.tree_parser.parse_with_tree(detected, None)
    return latex
