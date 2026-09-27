"""structures.grid (matrix/cases splitting) and structures.synth — sanity checks.

Work in progress: the splitter is evaluated on synthetic grids by
scripts/diagnostics/eval_grid.py; these tests pin the basics.
"""

import random

import pytest

from mathnote_ocr.bbox import BBox
from mathnote_ocr.structures.grid import _well_formed, split_grid
from mathnote_ocr.structures.synth import GridSymbol, generate, random_latex, render


@pytest.mark.parametrize("seq, ok", [
    (["x"], True),
    (["y", "-", "2"], True),
    (["-", "8"], True),                  # unary minus
    (["x", "=", "-", "1"], True),        # operator then unary minus
    (["x", ">", "0"], True),
    (["sigma", "+"], False),             # ends with an operator
    (["=", "2"], False),                 # starts with a relation
    (["+"], False),
    (["a", "+", "=", "b"], False),       # two operators in a row
    (["frac_bar", "a", "b"], True),      # bars are structural, ignored
    ([], False),
])
def test_well_formed(seq, ok):
    assert _well_formed(seq) is ok


def sym(name, x, y, w=20, h=30, cell=None):
    return GridSymbol(name, BBox(x, y, w, h), cell)


def test_clean_2x2_with_brackets():
    syms = [
        sym("(", 0, 0, 10, 120),
        sym("1", 40, 10, cell=(0, 0)), sym("2", 120, 10, cell=(0, 1)),
        sym("3", 40, 80, cell=(1, 0)), sym("4", 120, 80, cell=(1, 1)),
        sym(")", 170, 0, 10, 120),
    ]
    g = split_grid(syms, kind="matrix")
    assert g.env == "pmatrix" and g.shape == (2, 2)
    got = {i: (r, c) for r, row in enumerate(g.cells) for c, cell in enumerate(row) for i in cell}
    assert all(got[i] == s.cell for i, s in enumerate(syms) if s.cell)


def test_binary_minus_stays_in_its_cell():
    # [ y - 2 , 5 ] on one row, plus a second row so it's a grid
    syms = [
        sym("[", 0, 0, 8, 120),
        sym("y", 30, 10, cell=(0, 0)), sym("-", 58, 22, 14, 4, cell=(0, 0)), sym("2", 80, 10, cell=(0, 0)),
        sym("5", 160, 10, cell=(0, 1)),
        sym("1", 55, 80, cell=(1, 0)), sym("0", 160, 80, cell=(1, 1)),
        sym("]", 200, 0, 8, 120),
    ]
    g = split_grid(syms, kind="matrix")
    assert g.env == "bmatrix" and g.shape == (2, 2)
    assert sorted(g.cells[0][0]) == [1, 2, 3]


def test_cases_has_two_columns():
    syms = [
        sym("lbrace", 0, 0, 12, 120),
        sym("x", 30, 10, cell=(0, 0)), sym("x", 120, 10, cell=(0, 1)), sym(">", 148, 14, cell=(0, 1)), sym("0", 176, 10, cell=(0, 1)),
        sym("0", 30, 80, cell=(1, 0)), sym("x", 120, 80, cell=(1, 1)), sym("<", 148, 84, cell=(1, 1)), sym("0", 176, 80, cell=(1, 1)),
    ]
    g = split_grid(syms, kind="cases")
    assert g.env == "cases" and g.shape == (2, 2)


@pytest.fixture(scope="module")
def names():
    from mathnote_ocr import MathOCR
    return MathOCR().classifier.label_names


def test_synth_recovers_every_cell(names):
    rng = random.Random(0)
    for _ in range(50):
        kind, latex, cells = random_latex(rng)
        sample = render(kind, latex, cells, names)
        assert sample is not None, latex
        got = {s.cell for s in sample.symbols if s.cell is not None}
        assert got == {(r, c) for r in range(len(cells)) for c in range(len(cells[0]))}


def test_generate_is_deterministic(names):
    a = generate(5, names, seed=7)
    b = generate(5, names, seed=7)
    assert [s.latex for s in a] == [s.latex for s in b]
    assert [x.bbox for x in a[0].symbols] == [x.bbox for x in b[0].symbols]


def test_commas_separate_the_entries_of_a_row():
    syms = [
        sym("(", 0, 0, 10, 40),
        sym("1", 20, 5), sym(",", 44, 28, 6, 10), sym("2", 60, 5), sym(",", 84, 28, 6, 10), sym("x", 100, 5),
        sym(")", 130, 0, 10, 40),
    ]
    g = split_grid(syms, kind="matrix")
    assert g.env == "pmatrix" and g.shape == (1, 3)
    assert [[[syms[i].name for i in c] for c in row] for row in g.cells] == [[["1"], ["2"], ["x"]]]


def test_commas_in_rows_of_a_matrix_and_entries_with_operators():
    syms = [
        sym("[", 0, 0, 10, 110),
        sym("x", 20, 5), sym("+", 44, 8), sym("1", 66, 5), sym(",", 90, 28, 6, 10), sym("2", 110, 5),
        sym("3", 20, 70), sym(",", 90, 93, 6, 10), sym("4", 110, 70),
        sym("]", 140, 0, 10, 110),
    ]
    g = split_grid(syms, kind="matrix")
    assert g.env == "bmatrix" and g.shape == (2, 2)
    assert [[[syms[i].name for i in c] for c in row] for row in g.cells] == [[["x", "+", "1"], ["2"]], [["3"], ["4"]]]
