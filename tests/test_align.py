"""Forced alignment: LaTeX -> symbols, and strokes -> symbols given the LaTeX."""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from mathnote_ocr.align import (
    UnknownSymbol,
    align,
    labels_match,
    latex_to_labels,
    AlignedSymbol,
    _assign_bars,
    same_symbol,
)
from mathnote_ocr.bbox import BBox

ROOT = Path(__file__).resolve().parent.parent
FONT = ROOT.parent / "math_ocr_web" / "tests" / "fixtures" / "hand_written_font"


@pytest.fixture(scope="module")
def ocr():
    from mathnote_ocr import MathOCR
    return MathOCR()


@pytest.fixture(scope="module")
def names(ocr):
    return ocr.classifier.label_names


# ── LaTeX -> symbols ─────────────────────────────────────────────────


@pytest.mark.parametrize("latex, expected", [
    (r"\frac{x^{2}}{3}", ["frac_bar", "x", "2", "3"]),
    (r"\sqrt{a}+b", ["sqrt", "a", "+", "b"]),
    (r"\sin x", ["s", "i", "n", "x"]),
    (r"X_{i}", ["X_cap", "i"]),
    (r"\{a\}", ["lbrace", "a", "rbrace"]),
    (r"\left( x \right)", ["(", "x", ")"]),
    (r"f'(x)", ["f", "prime", "(", "x", ")"]),
    (r"\sum_{n=1}^{\infty} \frac{1}{n}", ["sum", "n", "=", "1", "infty", "frac_bar", "1", "n"]),
    (r"a \, b", ["a", "b"]),
])
def test_latex_to_labels(names, latex, expected):
    assert Counter(latex_to_labels(latex, names)) == Counter(expected)


def test_unknown_symbol_raises(names):
    with pytest.raises(UnknownSymbol):
        latex_to_labels(r"\aleph", names)


# The study vocabulary: trained classes + additions without a model yet
ADDITIONS = ["zeta", "varepsilon", "Theta_cap", "subseteq", "notin", "mapsto", "vdots", "bb_R", "bb_Q",
             "accent_hat", "accent_bar", "accent_vec", "accent_dot", "accent_tilde"]


@pytest.mark.parametrize("latex, expected", [
    (r"\hat{\theta}", ["accent_hat", "theta"]),
    (r"\bar{x} = 1", ["accent_bar", "x", "=", "1"]),
    (r"\vec{v} \cdot \vec{w}", ["accent_vec", "v", "cdot", "accent_vec", "w"]),
    (r"\dot{x} + \tilde{f}", ["accent_dot", "x", "+", "accent_tilde", "f"]),
    (r"\mathbb{R}^n", ["bb_R", "n"]),
    (r"x \notin \mathbb{Q}", ["x", "notin", "bb_Q"]),
    (r"\zeta(2)", ["zeta", "(", "2", ")"]),
    (r"f: A \to B", ["f", "colon", "A_cap", "rightarrow", "B_cap"]),
    (r"\sqrt{x} \in [0, 1]", ["sqrt", "x", "in", "[", "0", ",", "1", "]"]),
])
def test_study_vocabulary(names, latex, expected):
    assert Counter(latex_to_labels(latex, list(names) + ADDITIONS)) == Counter(expected)


def test_additions_need_their_class(names):
    for latex in (r"\hat{x}", r"\mathbb{R}", r"\zeta"):
        with pytest.raises(UnknownSymbol):
            latex_to_labels(latex, names)          # the trained vocabulary alone


def test_root_index_is_refused_not_mislabelled(names):
    with pytest.raises(UnknownSymbol):
        latex_to_labels(r"\sqrt[3]{x}", names)


def test_equivalent_classes():
    assert same_symbol("sum", "Sigma_up") and same_symbol("Pi_up", "prod")
    assert not same_symbol("sum", "prod")
    assert labels_match(["sum", "x"], ["Sigma_up", "x"])


def test_bars_told_apart_by_what_surrounds_them():
    # \frac{p}{\frac{a - b}{c}} + \frac{1}{2}: p is centred over the minus
    # of a - b and c under it, but fraction bars separate them — the minus
    # is no fraction bar, even though it is wider than the 1/2 bar
    layout = [
        ("p", BBox(15, 0, 10, 20)), ("-", BBox(0, 25, 40, 2)),            # outer fraction bar
        ("a", BBox(2, 30, 10, 15)), ("-", BBox(14, 37, 12, 2)), ("b", BBox(28, 30, 10, 15)),
        ("-", BBox(2, 48, 36, 2)),                                        # inner fraction bar
        ("c", BBox(15, 52, 10, 15)),
        ("+", BBox(45, 20, 10, 10)),
        ("1", BBox(60, 12, 6, 10)), ("-", BBox(59, 25, 8, 2)), ("2", BBox(60, 29, 6, 10)),
    ]
    symbols = [AlignedSymbol(label=lab, stroke_ids=[i], logp=0.0) for i, (lab, _) in enumerate(layout)]
    _assign_bars(symbols, [b for _, b in layout], fractions=3)
    fracs = {i for i, s in enumerate(symbols) if s.label == "frac_bar"}
    assert fracs == {1, 5, 9}


# ── Alignment ────────────────────────────────────────────────────────


def compose(names_, x0=60.0, y0=60.0, height=40.0):
    """Handwritten glyphs side by side (math_ocr_web's glyph fixtures)."""
    out = []
    for n in names_:
        strokes = json.loads(next((FONT / n).glob("*.json")).read_text())["strokes"]
        xs = [p["x"] for s in strokes for p in s]
        ys = [p["y"] for s in strokes for p in s]
        k = height / max(max(ys) - min(ys), 1)
        g = [[((p["x"] - min(xs)) * k, (p["y"] - min(ys)) * k) for p in s] for s in strokes]
        w = max(x for s in g for x, _ in s)
        out.append([[(x + x0, y + y0) for x, y in s] for s in g])
        x0 += w + height * 0.35
    return out


def flat(glyphs):
    return [s for g in glyphs for s in g]


def expected_groups(glyphs):
    groups, i = [], 0
    for g in glyphs:
        groups.append(frozenset(range(i, i + len(g))))
        i += len(g)
    return groups


needs_font = pytest.mark.skipif(not FONT.exists(), reason="glyph fixtures not available")


@needs_font
def test_align_x_plus_1(ocr, names):
    glyphs = compose(["x", "+", "1"])
    a = align(ocr, flat(glyphs), latex_to_labels("x+1", names))
    assert a is not None and a.complete
    got = {(s.label, frozenset(s.stroke_ids)) for s in a.symbols}
    assert got == set(zip(["x", "+", "1"], expected_groups(glyphs)))


@needs_font
def test_align_x_d_not_slash_W(ocr, names):
    """Unconstrained, x next to d could read '/ W'; the LaTeX rules it out."""
    glyphs = compose(["x", "d"])
    a = align(ocr, flat(glyphs), latex_to_labels("x d", names))
    assert {s.label for s in a.symbols} == {"x", "d"}
    assert {frozenset(s.stroke_ids) for s in a.symbols} == set(expected_groups(glyphs))


@needs_font
def test_no_cover_returns_none(ocr):
    # three glyphs' strokes can't be covered by a single symbol
    assert align(ocr, flat(compose(["x", "+", "1"])), ["x"]) is None


def test_ldots_on_real_handwriting(ocr, names):
    """run_001#54 'X^{7}\\ldots': the three dots are too spread for the grouper."""
    path = ROOT / "data" / "shared" / "tree_handwritten" / "run_001" / "train_strokes.jsonl"
    ex = json.loads(path.read_text().splitlines()[54])
    strokes = [[(p["x"], p["y"]) for p in s] for sym in ex["symbols"] for s in sym["strokes"] if s]
    gt_dots = {7, 8, 9}
    a = align(ocr, strokes, latex_to_labels(ex["latex"], names),
              canvas_size=max(ex["canvas_width"], ex["canvas_height"]))
    assert a is not None
    ldots = [s for s in a.symbols if s.label == "ldots"]
    assert len(ldots) == 1 and set(ldots[0].stroke_ids) == gt_dots
