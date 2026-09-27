"""Forced alignment: match strokes to the symbols of a known LaTeX.

Recognition has to guess what was written; alignment is told. Given the
strokes and the expression's LaTeX, it finds which strokes form which
symbol: an exact cover of the strokes by candidate groups whose labels use
up exactly the symbols the LaTeX contains, maximising the classifier's
log-probability. Used to label collected data — every confirmed expression
becomes symbol-level training data, even where the engine read it wrong.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field

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
) -> Alignment | None:
    """Assign strokes to the symbols in *labels* (e.g. from latex_to_labels).

    Exact cover of the strokes by the grouper's candidate groups — all of
    them, unfiltered — where the groups' labels use up *labels* exactly,
    maximising the sum of log P(label | group). Returns None when no such
    cover exists among the candidates.
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
    index = {name: i for i, name in enumerate(ocr.classifier.label_names)}

    # log P(label | group) for the labels we need, equivalent classes pooled
    target = Counter(labels)
    need = sorted(target)
    cols = {lab: [index[x] for x in _EQUIV.get(lab, {lab}) if x in index] for lab in need}
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

    best: dict = {"logp": -math.inf, "choice": None}
    nodes = 0
    remaining = dict(target)
    chosen: list[tuple[int, str]] = []
    # Best score seen per state (strokes left, symbols left): a state reached
    # again with no better score can't lead anywhere better
    seen: dict[tuple, float] = {}

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
            if left == 0 and score > best["logp"]:
                best["logp"], best["choice"] = score, list(chosen)
            return
        # Each symbol takes 1..max_size strokes
        if left == 0 or len(uncovered) < left or len(uncovered) > left * max_size:
            return
        state = (uncovered, tuple(sorted((lab, k) for lab, k in remaining.items() if k)))
        if seen.get(state, -math.inf) >= score:
            return
        seen[state] = score
        if score + bound(uncovered) <= best["logp"]:
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
    if best["choice"] is None:
        return None
    symbols = [
        AlignedSymbol(label=lab, stroke_ids=sorted(ids[p] for p in scored[gi][0]), logp=scored[gi][1][lab])
        for gi, lab in best["choice"]
    ]
    return Alignment(symbols=symbols, logp=best["logp"], complete=nodes <= node_budget,
                     nodes=nodes, seconds=elapsed, candidates=len(scored))


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
