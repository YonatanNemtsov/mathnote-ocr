"""Regression tests for the two causes of empty detections.

1. Singleton-gate / conflict deadlock: the gate rejected merges like '+'
   in favour of reading '-' and '|' separately — but those two conflict
   (crossing, same centre), so exact cover had no valid partition. The
   gate now only applies when the singletons could coexist.
2. Fraction-bar conflicts: a small symbol touching a long bar's middle
   clashed with it, because closeness was scaled by the average diagonal.
   With a frac_bar it is now scaled by the smaller symbol.
"""

import json
from pathlib import Path

from mathnote_ocr.classifier.inference import ClassificationResult
from mathnote_ocr.engine.grouper import _singletons_can_coexist
from mathnote_ocr.engine.stroke import Stroke, StrokePoint

DATA = Path(__file__).resolve().parent.parent / "data" / "shared" / "tree_handwritten"


def line(sid: int, x0: float, y0: float, x1: float, y1: float, n: int = 9) -> Stroke:
    pts = [StrokePoint(x0 + (x1 - x0) * i / (n - 1), y0 + (y1 - y0) * i / (n - 1)) for i in range(n)]
    return Stroke.from_points(pts, id=sid)


class FakeCache:
    """Singleton classifications keyed like GrouperCache."""

    def __init__(self, labels: dict[int, str]):
        self._data = {
            frozenset([sid]): ClassificationResult(symbol=name, confidence=0.9, prototype_distance=1.0, is_ood=False)
            for sid, name in labels.items()
        }

    def get(self, key, default=None):
        return self._data.get(key, default)


def test_crossing_strokes_cannot_coexist():
    # '+' drawn as '-' crossing '|' at the same centre
    strokes = [line(0, 100, 120, 140, 120), line(1, 120, 100, 120, 140)]
    cache = FakeCache({0: "-", 1: "1"})
    assert not _singletons_can_coexist(frozenset([0, 1]), strokes, [0, 1], cache)


def test_side_by_side_strokes_can_coexist():
    # two separate glyphs next to each other: the gate should still apply
    strokes = [line(0, 100, 100, 100, 140), line(1, 140, 100, 140, 140)]
    cache = FakeCache({0: "1", 1: "1"})
    assert _singletons_can_coexist(frozenset([0, 1]), strokes, [0, 1], cache)


def test_sqrt_singleton_exempt_from_conflict():
    # the cover never lets sqrt conflict (it encloses its radicand)
    strokes = [line(0, 100, 120, 140, 120), line(1, 120, 100, 120, 140)]
    cache = FakeCache({0: "sqrt", 1: "1"})
    assert _singletons_can_coexist(frozenset([0, 1]), strokes, [0, 1], cache)


def _example(run: str, idx: int) -> dict:
    with (DATA / run / "train_strokes.jsonl").open() as f:
        return json.loads(f.readlines()[idx])


def test_e2e_two_stroke_capital_not_empty():
    """run_003#7 'A_{i}' came back empty: A_cap lost to its two strokes."""
    from mathnote_ocr import MathOCR

    ex = _example("run_003", 7)
    strokes = [[(p["x"], p["y"]) for p in s] for sym in ex["symbols"] for s in sym["strokes"] if s]
    expr = MathOCR().detect(strokes, canvas_size=max(ex["canvas_width"], ex["canvas_height"]))
    assert len(expr.symbols) > 0
    assert "A_cap" in [s.name for s in expr.symbols.values()]


# ── Conflict rule with a fraction bar ────────────────────────────────


def test_small_symbol_touching_frac_bar_middle_does_not_clash():
    # '8' sitting on the middle of a long bar: the bar's length used to
    # inflate the threshold until the pair clashed
    from mathnote_ocr.bbox import BBox
    from mathnote_ocr.engine.grouper import _symbols_clash

    bar = BBox(100, 150, 200, 4)
    eight = BBox(190, 125, 18, 27)  # bottom edge overlaps the bar
    assert _symbols_clash("frac_bar", bar, "8", eight) is False


def test_symbol_centred_on_frac_bar_still_clashes():
    from mathnote_ocr.bbox import BBox
    from mathnote_ocr.engine.grouper import _symbols_clash

    bar = BBox(100, 150, 60, 4)
    vertical = BBox(128, 132, 4, 40)  # '|' crossing the bar's centre
    assert _symbols_clash("frac_bar", bar, "1", vertical) is True


def test_e2e_symbol_on_frac_bar_not_empty():
    """run_001#15: 'Y' touching a fraction bar made the whole expression empty."""
    from mathnote_ocr import MathOCR

    ex = _example("run_001", 15)
    strokes = [[(p["x"], p["y"]) for p in s] for sym in ex["symbols"] for s in sym["strokes"] if s]
    expr = MathOCR().detect(strokes, canvas_size=max(ex["canvas_width"], ex["canvas_height"]))
    assert len(expr.symbols) > 0
