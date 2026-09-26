"""Synthetic matrices and cases with ground-truth cells.

Renders a random \\begin{pmatrix}/bmatrix/vmatrix/cases with ziamath
(latex_utils.glyphs), recovers which cell every glyph belongs to, then
makes the layout messy like handwriting: per-symbol size from measured
handwriting statistics, drifting and tilted rows, jittered cells. Used to
evaluate (and later train) the grid splitter.

Cell recovery: the grid is rendered WITHOUT delimiters (ziamath draws tall
delimiters from several glyph pieces, which the extractor can't name), as
\begin{matrix} (or a left-aligned array for cases); glyphs come in reading
order, cell by cell, except fraction bars, which come last (in reading
order among themselves). Each cell's glyph and bar counts come from
align.latex_to_labels, so glyphs are dealt out cell by cell. The delimiter
boxes are then added around the content — boxes are all the splitter sees.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from mathnote_ocr.align import latex_to_labels
from mathnote_ocr.bbox import BBox
from mathnote_ocr.latex_utils.glyphs import _extract_glyphs
from mathnote_ocr.tree_parser.hw_bbox_augment import _HW_SYMBOL_STATS

ENVIRONMENTS = {"pmatrix": ("(", ")"), "bmatrix": ("[", "]"), "vmatrix": ("|", "|"), "cases": ("lbrace", None)}


@dataclass
class GridSymbol:
    name: str
    bbox: BBox
    cell: tuple[int, int] | None     # (row, col); None for a delimiter


@dataclass
class GridSample:
    kind: str                        # pmatrix / bmatrix / vmatrix / cases
    latex: str
    cells: list[list[str]]           # cell LaTeX, rows x cols
    symbols: list[GridSymbol]

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.cells), len(self.cells[0])


# ── Random content ───────────────────────────────────────────────────

_LETTERS = "abcdxyzmnpqt"
_GREEK = [r"\alpha", r"\beta", r"\lambda", r"\theta", r"\pi", r"\sigma"]


def _atom(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.45:
        return str(rng.randint(0, 9)) if rng.random() < 0.8 else str(rng.randint(10, 99))
    if r < 0.85:
        return rng.choice(_LETTERS)
    return rng.choice(_GREEK)


def _cell(rng: random.Random) -> str:
    """Mostly simple entries, sometimes structure (the hard cases)."""
    r = rng.random()
    a, b = _atom(rng), _atom(rng)
    if r < 0.45:
        return a
    if r < 0.60:
        return f"-{a}"
    if r < 0.70:
        return f"{a}^{{{rng.randint(2, 3)}}}"
    if r < 0.78:
        return f"{a}_{{{rng.choice('ijk12')}}}"
    if r < 0.88:
        return f"{a} {rng.choice('+-')} {b}"
    if r < 0.95:
        return rf"\frac{{{a}}}{{{b}}}"
    return rf"\sqrt{{{a}}}"


def _condition(rng: random.Random) -> str:
    v = rng.choice("xnt")
    return rng.choice([f"{v} > 0", f"{v} < 0", rf"{v} \leq {rng.randint(0, 9)}", rf"{v} \geq 1", f"{v} = 0"])


def random_latex(rng: random.Random, kind: str | None = None) -> tuple[str, str, list[list[str]]]:
    kind = kind or rng.choice(["pmatrix", "bmatrix", "vmatrix", "cases"])
    if kind == "cases":
        rows = rng.randint(2, 4)
        cells = [[_cell(rng), _condition(rng)] for _ in range(rows)]
    else:
        rows, cols = rng.randint(2, 4), rng.randint(2, 4)
        cells = [[_cell(rng) for _ in range(cols)] for _ in range(rows)]
    body = r" \\ ".join(" & ".join(row) for row in cells)
    return kind, rf"\begin{{{kind}}} {body} \end{{{kind}}}", cells


# ── Rendering with ground-truth cells ────────────────────────────────


def render(kind: str, latex: str, cells: list[list[str]], label_names: list[str]) -> GridSample | None:
    """Glyph boxes (font layout) with their cell; None if counts don't line up."""
    body = r" \\ ".join(" & ".join(row) for row in cells)
    bare = (rf"\begin{{array}}{{ll}} {body} \end{{array}}" if kind == "cases"
            else rf"\begin{{matrix}} {body} \end{{matrix}}")
    glyphs = _extract_glyphs(bare)
    if not glyphs:
        return None
    chars = [g for g in glyphs if not g.get("is_frac_bar")]
    bars = [g for g in glyphs if g.get("is_frac_bar")]

    per_cell = []   # ((r, c), n_chars, n_bars)
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            labels = latex_to_labels(cell, label_names)
            n_bars = labels.count("frac_bar")
            per_cell.append(((r, c), len(labels) - n_bars, n_bars))

    if len(chars) != sum(n for _, n, _ in per_cell) or len(bars) != sum(b for _, _, b in per_cell):
        return None

    out: list[GridSymbol] = []
    k = 0
    for cell_rc, n, _ in per_cell:
        for g in chars[k:k + n]:
            out.append(GridSymbol(g["name"], _bbox(g), cell_rc))
        k += n
    b = 0
    for cell_rc, _, n_bars in per_cell:
        for g in bars[b:b + n_bars]:
            out.append(GridSymbol("frac_bar", _bbox(g), cell_rc))
        b += n_bars
    return GridSample(kind=kind, latex=latex, cells=cells, symbols=_with_delimiters(out, kind))


# Delimiter width as a fraction of its height (thin | vs curved ( and {)
_DELIM_WIDTH = {"(": 0.10, ")": 0.10, "[": 0.07, "]": 0.07, "|": 0.02, "lbrace": 0.12}


def _with_delimiters(content: list[GridSymbol], kind: str, rng: random.Random | None = None,
                     amount: float = 1.0) -> list[GridSymbol]:
    """Add the environment's delimiters around *content*: tall boxes spanning
    it, a small gap away (with handwriting-like jitter when rng is given)."""
    def j(sd: float) -> float:
        return rng.gauss(0, sd * amount) if rng else 0.0
    x0 = min(g.bbox.x for g in content)
    x1 = max(g.bbox.x + g.bbox.w for g in content)
    y0 = min(g.bbox.y for g in content)
    y1 = max(g.bbox.y + g.bbox.h for g in content)
    h = y1 - y0
    left, right = ENVIRONMENTS[kind]
    out = list(content)
    for name, side in ((left, "L"), (right, "R")):
        if not name:
            continue
        top = y0 - h * (0.05 + j(0.04))
        bottom = y1 + h * (0.05 + j(0.04))
        w = h * _DELIM_WIDTH[name] * max(0.4, 1 + j(0.25))
        gap = h * max(0.01, 0.06 + j(0.03))
        x = x0 - gap - w if side == "L" else x1 + gap
        box = GridSymbol(name, BBox(x, top, w, bottom - top), None)
        out.insert(0, box) if side == "L" else out.append(box)
    return out


def _bbox(g: dict) -> BBox:
    x, y, w, h = g["bbox"]
    return BBox(x, y, w, h)


# ── Handwriting-like messiness ───────────────────────────────────────


def messy(sample: GridSample, rng: random.Random, scale: float = 400.0, amount: float = 1.0) -> GridSample:
    """Scale to pixels and perturb like handwriting.

    Per-symbol size from measured handwriting stats (_HW_SYMBOL_STATS),
    per-row vertical drift, a slight tilt (rows slope as you write), and
    per-cell horizontal jitter. `amount` scales all perturbations.
    """
    rows = sample.shape[0]
    content = [s for s in sample.symbols if s.cell is not None]
    row_dy = [rng.gauss(0, 0.035 * amount) for _ in range(rows)]
    tilt = rng.gauss(0, 0.04 * amount)                     # dy per unit x
    cell_dx: dict = {}
    out = []
    for s in content:
        b = s.bbox
        h_scale, h_std, w_scale, w_std = _HW_SYMBOL_STATS.get(s.name, (1.0, 0.15, 1.0, 0.15))
        if s.name == "frac_bar":
            h_scale, w_scale = 1.0, 1.0                    # structural: keep its extent
        hs = max(0.5, rng.gauss(h_scale, h_std * amount))
        ws = max(0.5, rng.gauss(w_scale, w_std * amount))
        cx, cy = b.x + b.w / 2, b.y + b.h / 2
        w, h = b.w * ws, b.h * hs
        cx += cell_dx.setdefault(s.cell, rng.gauss(0, 0.03 * amount))
        cy += row_dy[s.cell[0]]
        cy += tilt * cx
        cx += rng.gauss(0, 0.006 * amount)
        cy += rng.gauss(0, 0.006 * amount)
        out.append(GridSymbol(s.name, BBox((cx - w / 2) * scale, (cy - h / 2) * scale, w * scale, h * scale), s.cell))
    # Delimiters are drawn around wherever the content ended up
    return GridSample(kind=sample.kind, latex=sample.latex, cells=sample.cells,
                      symbols=_with_delimiters(out, sample.kind, rng, amount))


def generate(n: int, label_names: list[str], seed: int = 0, amount: float = 1.0, kind: str | None = None) -> list[GridSample]:
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        k, latex, cells = random_latex(rng, kind)
        sample = render(k, latex, cells, label_names)
        if sample is not None:
            out.append(messy(sample, rng, amount=amount))
    return out
