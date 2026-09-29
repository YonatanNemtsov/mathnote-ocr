"""Split an explicitly marked matrix / cases region into a grid of cells.

The user marks a region as a matrix or cases; this module doesn't detect
*whether* something is a grid, only how it most reasonably splits. Real
grids are ambiguous (is `x  -1` one cell or two?), so it proposes candidate
grids and picks the most reasonable one — the same generate-and-score
pattern as the grouper and tree parser.

  1. Delimiters: the outermost bracket-class symbol on each side, spanning
     the content's height.
  2. Units: a fraction bar with its numerator and denominator, a radical
     with what's under it — never split across cells.
  3. Candidates: R rows (cut at the R-1 largest vertical gaps between unit
     centres) x C columns (each row cut at its C-1 largest gaps, plus
     variants that move a cut across an operator). Rectangular by
     construction; cases have 1 or 2 columns.
     Commas written between entries ("(1, 2, 3)", rows "a, b") are explicit
     separators: a candidate cuts each row at its commas and leaves them
     out of the cells, and is preferred over cuts guessed from gaps.
  4. Score, lexicographically: fewest ill-formed cells (math grammar: no
     cell ends with an operator, starts with a binary one, or has two
     operators in a row; and a cell is one line — no two of its entries
     one above the other), then — cases only — two columns (value &
     condition) over one, then the cleanest gap separation (smallest gap
     used as a cut vs largest gap left inside a cell, rows and columns).
     Gaps alone can't tell a cases' columns apart: the gap before the
     condition is often no wider than those around its < or =.
  5. Rows and columns line up (matrices): a row is a line — units
     overlapping vertically — and a column boundary a vertical line through
     every row that crosses no unit, cut or not by how likely the gaps it
     crosses are column breaks. A row with nothing on one side of a
     boundary (a matrix being written, a row still short) says nothing
     about it, and its missing entries are empty cells. This grid is the
     answer unless it leaves more entries ill-formed than the best
     candidate (or commas separate the entries).

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
FLAT = {"-", "frac_bar", "=", ",", ".", "dot", "cdot"}
LEFT = {"(", "[", "|", "lbrace"}
RIGHT = {")", "]", "|", "rbrace"}
ENV_BY_DELIMS = {("(", ")"): "pmatrix", ("[", "]"): "bmatrix", ("|", "|"): "vmatrix", (None, None): "matrix"}
MAX_ROWS = 8
MAX_COLS = 8
MAX_VARIANTS = 16
# A delimiter covers at least this fraction of the content's height
DELIM_COVER = 0.7
# A gap between two neighbouring units of a row, in typical unit heights, is
# a column break with P = 1 / (1 + exp(-GAP_SLOPE (gap - GAP_MID))) — fit on
# ~400 states of matrices written by hand (inside an entry: median 0.32, 95%
# below 0.54; between entries: median 1.31, 95% above 0.68)
GAP_MID = 0.5
GAP_SLOPE = 16.0
# Columns are apart: a vertical line passes between two neighbouring columns
# through every row (479 of 481 boundaries in those matrices; the others miss
# by 0.02 unit heights — touching is allowed up to this much)
COLUMN_TOUCH = 0.05


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
    """The delimiters enclose the content: on each side, the outermost
    bracket-class symbol (left of / right of every other symbol's centre)
    spanning most of the content's height.

    Judged against the content itself, not a typical symbol height: with
    few symbols the delimiters skew any median (a column vector is two
    parens and two letters).
    """
    brackets = LEFT | RIGHT
    content = [s for s in symbols if s.name not in brackets] or list(symbols)
    top = min((s.bbox.y for s in content), default=0)
    bottom = max((s.bbox.y + s.bbox.h for s in content), default=0)

    def find(names, side):
        best = None
        for i, s in enumerate(symbols):
            if s.name not in names:
                continue
            cx = s.bbox.x + s.bbox.w / 2
            others = [o.bbox.x + o.bbox.w / 2 for j, o in enumerate(symbols) if j != i]
            if not others or (cx >= min(others) if side == "L" else cx <= max(others)):
                continue
            covered = min(bottom, s.bbox.y + s.bbox.h) - max(top, s.bbox.y)
            if covered < DELIM_COVER * (bottom - top):
                continue
            if best is None or s.bbox.h > symbols[best].bbox.h:
                best = i
        return best

    left = find(LEFT, "L")
    right = None if kind == "cases" else find(RIGHT, "R")
    return left, right


def _spans_rows(symbols, i: int) -> bool:
    """Clearly taller than the content's typical symbol."""
    brackets = LEFT | RIGHT
    heights = [s.bbox.h for j, s in enumerate(symbols) if j != i and s.name not in brackets]
    return bool(heights) and symbols[i].bbox.h > 1.5 * median(heights)


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

    # the writing's height: flat symbols (bars) say nothing about it — with two
    # bars among four symbols a median of all put the denominator out of reach
    tall = [symbols[i].bbox.h for i in idx if symbols[i].name not in FLAT]
    typical = median(tall or [symbols[i].bbox.h for i in idx]) if idx else 0
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
        # Cells sit side by side: a cut where units overlap isn't a column break
        if len(set(combo)) != len(combo) or any(width[k] <= 0 for k in combo):
            continue
        cells, start = [], 0
        for k in sorted(combo):
            cells.append(units[start:k + 1])
            start = k + 1
        cells.append(units[start:])
        splits.append(cells)
    return splits


def _comma_candidates(units, boxes, unit_name, seq, kind) -> list[tuple]:
    """Grids cut at written commas: rows from the other units' vertical
    gaps (each comma joins the row it sits in), each row cut at its commas,
    the commas left out of the cells. Only rectangular grids with no empty
    cell, and only if some row has a comma."""
    commas = [u for u in range(len(units)) if unit_name(u) == ","]
    rest = [u for u in range(len(units)) if u not in commas]
    if not commas or not rest:
        return []
    out = []
    for rows in _row_candidates([boxes[u] for u in rest]):
        rows = [[rest[i] for i in r] for r in rows if r]
        spans = [(min(boxes[u][1] for u in r), max(boxes[u][3] for u in r)) for r in rows]
        with_commas = [list(r) for r in rows]
        for c in commas:
            cy = (boxes[c][1] + boxes[c][3]) / 2
            k = min(range(len(rows)), key=lambda i: 0 if spans[i][0] <= cy <= spans[i][1]
                    else min(abs(cy - spans[i][0]), abs(cy - spans[i][1])))
            with_commas[k].append(c)
        grid_rows, ok = [], True
        for r in with_commas:
            cells, cell = [], []
            for u in sorted(r, key=lambda u: boxes[u][0]):
                if u in commas:
                    cells.append(cell)
                    cell = []
                else:
                    cell.append(u)
            cells.append(cell)
            if any(not c for c in cells):
                ok = False
                break
            grid_rows.append(cells)
        if not ok or len({len(r) for r in grid_rows}) != 1 or len(grid_rows[0]) < 2:
            continue
        bad = _bad(grid_rows, boxes, seq)
        out.append((bad, False, _separation(grid_rows, boxes), grid_rows))
    return out


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


def _stacked(cell: list[int], boxes) -> bool:
    """A cell is one line of writing: two of its units one above the other
    (apart vertically, overlapping horizontally over most of the narrower
    one's width) are entries of different rows. A fraction is one unit, so
    its parts never count."""
    for a, b in itertools.combinations(cell, 2):
        (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = boxes[a], boxes[b]
        apart = by0 >= ay1 or ay0 >= by1
        over = min(ax1, bx1) - max(ax0, bx0)
        if apart and over > 0.5 * min(ax1 - ax0, bx1 - bx0):
            return True
    return False


def _bad(rows: list[list[list[int]]], boxes, seq) -> int:
    """Cells that are no single, well-formed entry."""
    return sum(not _well_formed(seq(c)) or _stacked(c, boxes) for row in rows for c in row if c)


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


# ── 5. Rows and columns that line up ─────────────────────────────────


def _log_cut(gap: float) -> tuple[float, float]:
    """(log P(column break), log P(inside an entry)) for a gap in unit heights."""
    z = GAP_SLOPE * (gap - GAP_MID)
    softplus = lambda t: t + math.log1p(math.exp(-t)) if t > 0 else math.log1p(math.exp(t))
    return -softplus(-z), -softplus(z)


def _line_rows(boxes) -> list[list[int]]:
    """The units in rows, top to bottom: a row is a line — units overlapping
    vertically, directly or through others (in the handwritten matrices, a
    unit always overlaps another of its row, by 0.41 of its height or more,
    and never one of the next row)."""
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in itertools.combinations(range(n), 2):
        if min(boxes[a][3], boxes[b][3]) > max(boxes[a][1], boxes[b][1]):
            parent[find(a)] = find(b)
    rows: dict[int, list[int]] = {}
    for u in range(n):
        rows.setdefault(find(u), []).append(u)
    return sorted(rows.values(), key=lambda r: min(boxes[u][1] for u in r))


def _aligned_columns(rows: list[list[int]], boxes, h: float, seq) -> list[list[list[int]]]:
    """*rows*' units in columns found across all rows: the boundaries are the
    vertical lines through every row crossing no unit; each is a column
    break if the gaps it crosses — in the rows with a unit on both sides and
    no other boundary in that gap (a wide gap says there is a break, not
    where) — are more likely breaks than not; a boundary no row speaks for
    is judged by its own width. If that leaves an entry ill-formed, the
    least sure decision is flipped, then the next. Cells a row has nothing
    in are empty."""
    spans = sorted((boxes[u][0], boxes[u][2]) for r in rows for u in r)
    lines, reach = [], spans[0][1]
    for x0, x1 in spans[1:]:
        if x0 > reach - COLUMN_TOUCH * h:
            lines.append((reach + x0) / 2)
        reach = max(reach, x1)
    centre = lambda u: (boxes[u][0] + boxes[u][2]) / 2
    margin = []
    for x in lines:
        cut = keep = 0.0
        evidence = False
        for r in rows:
            left = [u for u in r if centre(u) < x]
            right = [u for u in r if centre(u) > x]
            if not (left and right):
                continue
            x0, x1 = max(boxes[u][2] for u in left), min(boxes[u][0] for u in right)
            if sum(x0 - COLUMN_TOUCH * h <= y <= x1 + COLUMN_TOUCH * h for y in lines) > 1:
                continue                          # this gap holds another boundary too
            c, k = _log_cut((x1 - x0) / h)
            cut, keep, evidence = cut + c, keep + k, True
        if not evidence:
            below = max(x1 for x0, x1 in spans if (x0 + x1) / 2 < x)
            above = min(x0 for x0, x1 in spans if (x0 + x1) / 2 > x)
            cut, keep = _log_cut((above - below) / h)
        margin.append(cut - keep)

    def grid_for(chosen):
        bounds = sorted(x for x, m in zip(lines, chosen) if m)
        return [[[u for u in r if sum(centre(u) > b for b in bounds) == c] for c in range(len(bounds) + 1)]
                for r in rows]

    chosen = [m > 0 for m in margin]
    grid = grid_for(chosen)
    for k in sorted(range(len(lines)), key=lambda k: abs(margin[k])):
        if not _bad(grid, boxes, seq):
            break
        flipped = chosen[:k] + [not chosen[k]] + chosen[k + 1:]
        alt = grid_for(flipped)
        if _bad(alt, boxes, seq) < _bad(grid, boxes, seq):
            chosen, grid = flipped, alt
    return grid


# ── Entry point ──────────────────────────────────────────────────────


def split_grid(symbols, kind: str = "auto", n_alternatives: int | None = 5) -> Grid:
    """*kind*: "matrix" (delimiters decide p/b/vmatrix), "cases", or "auto"
    (a lone left delimiter means cases, anything else a matrix).

    The runners-up are kept in .alternatives (all of them if n_alternatives
    is None).
    """
    if kind == "auto":
        # A left delimiter with no right partner is cases — whatever the
        # classifier called it (a tall hand-drawn { is often read as |) —
        # if it is a brace or spans rows; a bracket of row height with no
        # partner is just content (|x| at the start of a row)
        left, right = _find_delimiters(symbols, "matrix")
        lone = left is not None and right is None
        kind = "cases" if lone and (symbols[left].name == "lbrace" or _spans_rows(symbols, left)) else "matrix"
    left, right = _find_delimiters(symbols, kind)
    if kind != "cases" and (left is None) != (right is None):
        left = right = None   # a matrix's delimiters come in pairs
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
        col_counts = range(1, min(2 if kind == "cases" else MAX_COLS, max_c) + 1)
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
                bad = _bad(grid_rows, boxes, seq)
                one_col = kind == "cases" and n_cols == 1
                candidates.append((bad, one_col, _separation(grid_rows, boxes), grid_rows))

    # Commas between entries: explicit column breaks (preferred when as well-formed)
    explicit = _comma_candidates(units, boxes, unit_name, seq, kind) if kind != "cases" else []
    candidates = [(bad, 0, one_col, sep, rows) for bad, one_col, sep, rows in explicit] + \
                 [(bad, 1, one_col, sep, rows) for bad, one_col, sep, rows in candidates]
    candidates.sort(key=lambda c: (c[0], c[1], c[2], -c[3]))

    def to_grid(cand) -> Grid:
        bad, _guessed, _one_col, sep, grid_rows = cand
        cells = [[sorted(i for u in cell for i in units[u]) for cell in row] for row in grid_rows]
        return Grid(env=env, left=left, right=right, cells=cells, ill_formed=bad, separation=sep)

    if not candidates:
        return Grid(env=env, left=left, right=right, cells=[[sorted(content)]] if content else [])
    # 5. A matrix's rows and columns as they line up — unless that leaves
    # more entries ill-formed (or the entries are separated by commas)
    top = candidates[0]
    if kind != "cases" and top[1] == 1:
        rows_units = _line_rows(boxes)
        if len(rows_units) >= 2:
            aligned = _aligned_columns(rows_units, boxes, median(b[3] - b[1] for b in boxes), seq)
            bad = _bad(aligned, boxes, seq)
            if bad <= top[0]:
                candidates.insert(0, (bad, 1, False, _separation(aligned, boxes), aligned))
    best = to_grid(candidates[0])
    rest = candidates[1:] if n_alternatives is None else candidates[1:1 + n_alternatives]
    best.alternatives = [to_grid(c) for c in rest]
    best.n_candidates = len(candidates)
    return best
