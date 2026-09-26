"""Split an explicitly marked matrix / cases region into a grid of cells.

The user marks a region as a matrix or cases; this module doesn't detect
*whether* something is a grid, only how it most reasonably splits. Real
grids are ambiguous (is `x  -1` one cell or two?), so it proposes candidate
grids and picks the most reasonable one — the same generate-and-score
pattern as the grouper and tree parser.

  1. Delimiters: the tallest bracket-class symbol on each side, taller
     than the content's typical symbol (it spans rows).
  2. Units: a fraction bar with its numerator and denominator, a radical
     with what's under it — never split across cells.
  3. Candidates: R rows (cut at the R-1 largest vertical gaps between unit
     centres) x C columns (each row cut at its C-1 largest gaps, plus
     variants that move a cut across an operator). Rectangular by
     construction; cases have exactly 2 columns.
  4. Score, lexicographically: fewest ill-formed cells (math grammar: no
     cell ends with an operator, starts with a binary one, or has two
     operators in a row), then the cleanest gap separation (smallest gap
     used as a cut vs largest gap left inside a cell, rows and columns).

The tree parser's confidence can't judge cells: it is 1.0 even for "x +".
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from statistics import median

# Binary operators and relations: need operands on both sides
BINARY = {"+", "=", "<", ">", "leq", "geq", "neq", "times", "cdot", "pm", "div"}
# May also stand first, as a sign (unary)
UNARY = {"-", "+", "pm"}
OPERATORS = BINARY | {"-"}
LEFT = {"(", "[", "|", "lbrace"}
RIGHT = {")", "]", "|", "rbrace"}
ENV_BY_DELIMS = {("(", ")"): "pmatrix", ("[", "]"): "bmatrix", ("|", "|"): "vmatrix", (None, None): "matrix"}
MAX_ROWS = 8
MAX_COLS = 8
MAX_VARIANTS = 16


@dataclass
class Grid:
    env: str                         # pmatrix / bmatrix / vmatrix / matrix / cases
    left: int | None                 # symbol index of the left delimiter
    right: int | None
    cells: list[list[list[int]]]     # rows x cols x symbol indices
    ill_formed: int = 0              # cells failing the grammar check
    separation: float = 0.0          # log gap-separation score (higher = cleaner)
    alternatives: list["Grid"] = field(default_factory=list, repr=False)
    n_candidates: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.cells), max((len(r) for r in self.cells), default=0)


def _box(s):
    b = s.bbox
    return b.x, b.y, b.x + b.w, b.y + b.h


# ── 1. Delimiters ────────────────────────────────────────────────────


def _find_delimiters(symbols, kind: str) -> tuple[int | None, int | None]:
    heights = [s.bbox.h for s in symbols]
    typical = median(heights) if heights else 0
    xs = [s.bbox.x + s.bbox.w / 2 for s in symbols]
    mid = median(xs) if xs else 0

    def tallest(names, side):
        best = None
        for i, s in enumerate(symbols):
            cx = s.bbox.x + s.bbox.w / 2
            on_side = cx < mid if side == "L" else cx > mid
            # A delimiter spans rows: clearly taller than a typical symbol
            if s.name in names and on_side and s.bbox.h > 1.5 * typical:
                if best is None or s.bbox.h > symbols[best].bbox.h:
                    best = i
        return best

    left = tallest(LEFT, "L")
    right = None if kind == "cases" else tallest(RIGHT, "R")
    return left, right


# ── 2. Units ─────────────────────────────────────────────────────────


def _units(symbols, idx: list[int]) -> list[list[int]]:
    """Union symbols that must share a cell (fractions, radicals)."""
    parent = {i: i for i in idx}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(a)] = find(b)

    typical = median(symbols[i].bbox.h for i in idx) if idx else 0
    for i in idx:
        s = symbols[i]
        x0, y0, x1, y1 = _box(s)
        if s.name == "frac_bar":
            # numerator/denominator: within the bar's width, close above/below
            reach = 0.8 * typical
            for k in idx:
                if k == i:
                    continue
                a0, b0, a1, b1 = _box(symbols[k])
                cx = (a0 + a1) / 2
                if not (x0 <= cx <= x1):
                    continue
                if (b1 <= y0 + 1e-9 and y0 - b1 <= reach) or (b0 >= y1 - 1e-9 and b0 - y1 <= reach):
                    union(k, i)
        elif s.name == "sqrt":
            for k in idx:
                if k == i:
                    continue
                a0, b0, a1, b1 = _box(symbols[k])
                if x0 <= (a0 + a1) / 2 <= x1 and y0 <= (b0 + b1) / 2 <= y1:
                    union(k, i)
    groups: dict[int, list[int]] = {}
    for i in idx:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _unit_box(symbols, unit):
    boxes = [_box(symbols[i]) for i in unit]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


# ── 3. Candidates ────────────────────────────────────────────────────


def _row_candidates(boxes) -> list[list[list[int]]]:
    """Row structures: for each R, cut at the R-1 largest gaps between
    consecutive unit centres (top to bottom)."""
    order = sorted(range(len(boxes)), key=lambda u: (boxes[u][1] + boxes[u][3]) / 2)
    cy = [(boxes[u][1] + boxes[u][3]) / 2 for u in order]
    gaps = sorted(range(len(order) - 1), key=lambda k: cy[k + 1] - cy[k], reverse=True)
    out = []
    for r in range(1, min(MAX_ROWS, len(order)) + 1):
        cuts = sorted(gaps[: r - 1])
        rows, start = [], 0
        for k in cuts:
            rows.append(order[start:k + 1])
            start = k + 1
        rows.append(order[start:])
        out.append(rows)
    return out


def _row_splits(row: list[int], boxes, n_cols: int, names) -> list[list[list[int]]]:
    """Ways to cut one row into n_cols cells: at its n_cols-1 widest gaps,
    plus variants moving a cut across an adjacent single operator."""
    units = sorted(row, key=lambda u: boxes[u][0])
    if len(units) < n_cols:
        return []
    width = [boxes[b][0] - boxes[a][2] for a, b in zip(units, units[1:])]
    base = sorted(sorted(range(len(width)), key=lambda k: width[k], reverse=True)[: n_cols - 1])
    options = []
    for k in base:
        alts = {k}
        # cut k sits between units[k] and units[k+1]; an operator on either
        # side may belong to the other cell
        if names(units[k + 1]) in OPERATORS and k + 1 < len(units) - 1:
            alts.add(k + 1)
        if names(units[k]) in OPERATORS and k > 0:
            alts.add(k - 1)
        options.append(sorted(alts))
    splits = []
    for combo in itertools.islice(itertools.product(*options), MAX_VARIANTS):
        if len(set(combo)) != len(combo):
            continue
        cells, start = [], 0
        for k in sorted(combo):
            cells.append(units[start:k + 1])
            start = k + 1
        cells.append(units[start:])
        splits.append(cells)
    return splits


# ── 4. Scoring ───────────────────────────────────────────────────────


def _well_formed(seq: list[str]) -> bool:
    """Math grammar on a cell's symbols in reading order (frac bars excluded)."""
    seq = [n for n in seq if n != "frac_bar"]
    if not seq:
        return False
    if seq[-1] in OPERATORS:
        return False
    if seq[0] in BINARY and seq[0] not in UNARY:
        return False
    for a, b in zip(seq, seq[1:]):
        if a in OPERATORS and b in OPERATORS and b not in UNARY:
            return False
    return True


def _separation(rows: list[list[list[int]]], boxes) -> float:
    """Natural-breaks score, the same way in both directions.

    Consecutive units are separated by gaps; a candidate cuts some of them
    (between cells / rows) and keeps the rest (inside a cell / row). Score
    = log(narrowest cut / widest kept gap), summed over x and y. No cuts in
    a direction scores 0, so a split only wins when it is a clean break
    (ratio > 1). Kept gaps are floored at a small fraction of the typical
    unit height, so single-symbol cells don't score infinitely well.
    """
    heights = [b[3] - b[1] for b in boxes]
    floor = 0.1 * median(heights) if heights else 1e-6
    # x: within each row, left to right
    cut_x, keep_x = math.inf, 0.0
    for row in rows:
        seq = sorted(((u, c) for c, cell in enumerate(row) for u in cell), key=lambda t: boxes[t[0]][0])
        for (a, ca), (b, cb) in zip(seq, seq[1:]):
            gap = boxes[b][0] - boxes[a][2]
            if ca != cb:
                cut_x = min(cut_x, gap)
            else:
                keep_x = max(keep_x, gap)
    # y: unit centres top to bottom
    centre = lambda u: (boxes[u][1] + boxes[u][3]) / 2
    seq = sorted(((u, r) for r, row in enumerate(rows) for cell in row for u in cell), key=lambda t: centre(t[0]))
    cut_y, keep_y = math.inf, 0.0
    for (a, ra), (b, rb) in zip(seq, seq[1:]):
        gap = centre(b) - centre(a)
        if ra != rb:
            cut_y = min(cut_y, gap)
        else:
            keep_y = max(keep_y, gap)
    score = 0.0
    for cut, keep in ((cut_x, keep_x), (cut_y, keep_y)):
        if cut == math.inf:
            continue
        score += math.log(max(cut, 1e-6) / max(keep, floor))
    return score


# ── Entry point ──────────────────────────────────────────────────────


def split_grid(symbols, kind: str = "auto", n_alternatives: int | None = 5) -> Grid:
    """*kind*: "matrix" (delimiters decide p/b/vmatrix), "cases", or "auto"
    (a lone left delimiter means cases, anything else a matrix).

    The runners-up are kept in .alternatives (all of them if n_alternatives
    is None).
    """
    if kind == "auto":
        # A left delimiter with no right partner can only be cases — whatever
        # the classifier called it (a tall hand-drawn { is often read as |)
        left, right = _find_delimiters(symbols, "matrix")
        kind = "cases" if left is not None and right is None else "matrix"
    left, right = _find_delimiters(symbols, kind)
    content = [i for i in range(len(symbols)) if i not in (left, right)]
    units = _units(symbols, content)
    boxes = [_unit_box(symbols, u) for u in units]

    def unit_name(u):
        return symbols[units[u][0]].name if len(units[u]) == 1 else None

    def seq(cell):
        idx = sorted((i for u in cell for i in units[u]), key=lambda i: symbols[i].bbox.x)
        return [symbols[i].name for i in idx]

    if kind == "cases":
        env = "cases"
    else:
        names = (symbols[left].name if left is not None else None,
                 symbols[right].name if right is not None else None)
        env = ENV_BY_DELIMS.get(names, "matrix")

    candidates, seen = [], set()
    for rows in _row_candidates(boxes) if boxes else [[]]:
        rows = [r for r in rows if r]
        max_c = min(len(r) for r in rows) if rows else 1
        col_counts = [2] if kind == "cases" else range(1, min(MAX_COLS, max_c) + 1)
        for n_cols in col_counts:
            per_row = [_row_splits(r, boxes, n_cols, unit_name) for r in rows]
            if any(not p for p in per_row):
                continue
            for choice in itertools.islice(itertools.product(*per_row), MAX_VARIANTS):
                key = tuple(tuple(tuple(sorted(c)) for c in row) for row in choice)
                if key in seen:
                    continue
                seen.add(key)
                grid_rows = [list(row) for row in choice]
                bad = sum(not _well_formed(seq(c)) for row in grid_rows for c in row)
                candidates.append((bad, _separation(grid_rows, boxes), grid_rows))

    candidates.sort(key=lambda c: (c[0], -c[1]))

    def to_grid(cand) -> Grid:
        bad, sep, grid_rows = cand
        cells = [[sorted(i for u in cell for i in units[u]) for cell in row] for row in grid_rows]
        return Grid(env=env, left=left, right=right, cells=cells, ill_formed=bad, separation=sep)

    if not candidates:
        return Grid(env=env, left=left, right=right, cells=[[sorted(content)]] if content else [])
    best = to_grid(candidates[0])
    rest = candidates[1:] if n_alternatives is None else candidates[1:1 + n_alternatives]
    best.alternatives = [to_grid(c) for c in rest]
    best.n_candidates = len(candidates)
    return best
