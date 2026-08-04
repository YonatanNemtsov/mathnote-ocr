"""Regression tests for the '=' merge fix.

Covers the two layers: _merge_equal accepting frac_bar-labeled bars (with
the emptiness guard protecting real fractions), and the scale-relative
merge-distance floor.
"""

import pytest

from mathnote_ocr.bbox import BBox
from mathnote_ocr.engine.grouper import (
    _effective_min_merge_distance,
    _merge_equal,
    _merge_pm,
)
from mathnote_ocr.engine.stroke import Stroke, StrokePoint
from mathnote_ocr.expression import DetectedSymbol


def bar(name: str, x: float, y: float, w: float, h: float = 4.0) -> DetectedSymbol:
    return DetectedSymbol(name=name, bbox=BBox(x, y, w, h), strokes=[], confidence=0.9)


def names(symbols) -> list[str]:
    return sorted(s.name for s in symbols)


# ── _merge_equal ─────────────────────────────────────────────────────


def test_merge_minus_minus():
    out = _merge_equal([bar("-", 0, 0, 60), bar("-", 2, 25, 58)])
    assert names(out) == ["="]


def test_merge_minus_fracbar():
    # The fix: one bar of a drawn '=' frequently classifies as frac_bar
    out = _merge_equal([bar("-", 0, 0, 60), bar("frac_bar", 2, 25, 58)])
    assert names(out) == ["="]


def test_merge_fracbar_fracbar():
    out = _merge_equal([bar("frac_bar", 0, 0, 60), bar("frac_bar", 2, 25, 58)])
    assert names(out) == ["="]


def test_no_merge_with_content_between():
    # A fraction bar above another bar with the numerator between them —
    # must NOT collapse into '='.
    content = DetectedSymbol(
        name="x", bbox=BBox(30, 15, 20, 30), strokes=[], confidence=0.9
    )
    syms = [bar("frac_bar", 0, 0, 80), content, bar("frac_bar", 2, 60, 76)]
    out = _merge_equal(syms)
    assert "=" not in names(out)
    assert len(out) == 3


def test_no_merge_when_not_stacked():
    # vertical distance exceeds bar width → not an '='
    out = _merge_equal([bar("-", 0, 0, 40), bar("-", 0, 80, 40)])
    assert "=" not in names(out)


def test_no_merge_dissimilar_width():
    out = _merge_equal([bar("-", 0, 0, 20), bar("frac_bar", 0, 25, 80)])
    assert "=" not in names(out)


# ── scale-relative merge floor ───────────────────────────────────────


def _stroke(w: float, h: float, id: int) -> Stroke:
    return Stroke.from_points([StrokePoint(0, 0, 0), StrokePoint(w, h, 0)], id=id)


def test_floor_small_scale_unchanged():
    small = [_stroke(20, 20, i) for i in range(3)]  # diag ~28 → scaled 9.9 < 14
    assert _effective_min_merge_distance(small, 14.0, 0.35) == 14.0


def test_floor_scales_with_writing_size():
    big = [_stroke(80, 80, i) for i in range(3)]  # diag ~113 → scaled ~39.6
    expected = 0.35 * big[0].bbox.diagonal
    assert _effective_min_merge_distance(big, 14.0, 0.35) == pytest.approx(expected)


def test_floor_disabled_and_empty():
    big = [_stroke(80, 80, 0)]
    assert _effective_min_merge_distance(big, 14.0, 0.0) == 14.0
    assert _effective_min_merge_distance([], 14.0, 0.35) == 14.0


# ── end to end ───────────────────────────────────────────────────────


def _hline(y: float, x0: float = 100.0, x1: float = 180.0, n: int = 9):
    return [(x0 + (x1 - x0) * i / (n - 1), y) for i in range(n)]


def test_e2e_equals_with_wide_gap():
    """Two bars 24px apart at large scale → '=' (the math_steps failure)."""
    from mathnote_ocr import MathOCR

    ocr = MathOCR()
    expr = ocr.detect([_hline(100), _hline(124)])
    assert expr.latex.strip() == "="


# ── _merge_pm ────────────────────────────────────────────────────────


def plus(x: float, y: float, w: float = 40.0) -> DetectedSymbol:
    return DetectedSymbol(name="+", bbox=BBox(x, y, w, w), strokes=[], confidence=0.9)


def test_merge_pm_minus():
    out = _merge_pm([plus(0, 0), bar("-", 2, 50, 38)])
    assert names(out) == ["pm"]


def test_merge_pm_fracbar():
    # The fix: the bar of a drawn 'pm' misclassified as frac_bar
    out = _merge_pm([plus(0, 0), bar("frac_bar", 2, 50, 38)])
    assert names(out) == ["pm"]


def test_no_merge_pm_with_content_between():
    content = DetectedSymbol(name="x", bbox=BBox(10, 45, 20, 25), strokes=[], confidence=0.9)
    out = _merge_pm([plus(0, 0), content, bar("frac_bar", 2, 75, 38)])
    assert "pm" not in names(out)


def test_no_merge_pm_bar_above():
    out = _merge_pm([bar("-", 2, 0, 38), plus(0, 10)])
    assert "pm" not in names(out)
