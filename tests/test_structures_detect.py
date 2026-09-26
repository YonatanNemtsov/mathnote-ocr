"""Detection with marked grid regions (Structure("grid", ...)).

Handwritten glyphs (math_ocr_web's fixtures) composed into a matrix with
tall hand-drawn delimiters.
"""

import json
import random
from pathlib import Path

import pytest

from mathnote_ocr import GridBlock, MathOCR, Structure

FONT = Path(__file__).resolve().parent.parent.parent / "math_ocr_web" / "tests" / "fixtures" / "hand_written_font"
pytestmark = pytest.mark.skipif(not FONT.exists(), reason="glyph fixtures not available")


def glyph(name, x, y, h, w_scale=1.0):
    strokes = json.loads(next((FONT / name).glob("*.json")).read_text())["strokes"]
    xs = [p["x"] for s in strokes for p in s]
    ys = [p["y"] for s in strokes for p in s]
    k = h / max(max(ys) - min(ys), 1)
    return [[((p["x"] - min(xs)) * k * w_scale + x, (p["y"] - min(ys)) * k + y) for p in s] for s in strokes]


def matrix(x0, y0, entries, left="(", right=")", h=36, col=70, row=64):
    rows, cols = len(entries), len(entries[0])
    out = glyph(left, x0, y0 - 8, rows * row, 0.45)
    for r, er in enumerate(entries):
        for c, e in enumerate(er):
            cx = x0 + 40 + c * col
            for n in e:
                out += glyph(n, cx, y0 + r * row, h)
                cx += h * 0.8
    out += glyph(right, x0 + 40 + cols * col - 10, y0 - 8, rows * row, 0.45)
    return out


@pytest.fixture(scope="module")
def ocr():
    return MathOCR()


ENTRIES = [[["1"], ["2"]], [["3"], ["4"]]]
MATRIX = r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}"


def test_marked_matrix_alone(ocr):
    strokes = matrix(200, 100, ENTRIES)
    random.seed(0)
    e = ocr.detect(strokes, structures=[Structure("grid", tuple(range(len(strokes))))])
    assert e.latex == MATRIX
    (g,) = e.grids.values()
    assert isinstance(g, GridBlock) and g.env == "pmatrix" and g.shape == (2, 2)
    # every cell's symbols are real, correctable symbols of the expression
    assert all(sid in e.symbols for row in g.cells for cell in row for sid in cell)
    assert e.alternatives, "ranked alternative splits for cycling"
    assert e.to_dict()["grids"][0]["latex"] == MATRIX


def test_marked_matrix_inside_expression(ocr):
    prefix = glyph("A_cap", 40, 120, 40) + glyph("=", 110, 135, 14)
    strokes = prefix + matrix(200, 100, ENTRIES)
    random.seed(0)
    e = ocr.detect(strokes, structures=[Structure("grid", tuple(range(len(prefix), len(strokes))))])
    assert e.latex == "A = " + MATRIX
    assert not e.unexplained_stroke_ids


def test_unknown_structure_stroke_raises(ocr):
    strokes = matrix(200, 100, ENTRIES)
    with pytest.raises(ValueError):
        ocr.detect(strokes, structures=[Structure("grid", (999,))])


def cases(x0, y0, rows, h=36, row=64):
    out = glyph("lbrace", x0, y0 - 8, len(rows) * row, 0.5)
    for r, (value, cond) in enumerate(rows):
        cx = x0 + 45
        for n in value:
            out += glyph(n, cx, y0 + r * row, h)
            cx += h * 0.8
        cx = x0 + 190
        for n in cond:
            small = n in "<>"
            out += glyph(n, cx, y0 + r * row + (8 if small else 0), h * (0.7 if small else 1))
            cx += h * 0.9
    return out


def test_marked_cases(ocr):
    """A lone left delimiter means cases — even when the tall { reads as |."""
    strokes = cases(100, 80, [(["x"], ["x", ">", "0"]), (["0"], ["x", "<", "0"])])
    random.seed(0)
    e = ocr.detect(strokes, structures=[Structure("grid", tuple(range(len(strokes))))])
    assert e.latex == r"\begin{cases} x & x > 0 \\ 0 & x < 0 \end{cases}"
