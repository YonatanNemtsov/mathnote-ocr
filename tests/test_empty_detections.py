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


# ── Skip fallback: partial result + unexplained strokes ──────────────


def _sym(name: str, x: float, y: float, sids: list[int], w: float = 20.0, h: float = 30.0):
    from mathnote_ocr.bbox import BBox
    from mathnote_ocr.expression import DetectedSymbol

    strokes = [Stroke(id=i) for i in sids]
    return DetectedSymbol(name=name, bbox=BBox(x, y, w, h), strokes=strokes, confidence=0.9)


def _covered(partition) -> set[int]:
    return {st.id for sym in partition for st in sym.strokes}


def test_full_cover_uses_no_skips():
    from mathnote_ocr.engine.grouper import _find_best_partitions

    groups = [(frozenset([0]), 0.9, _sym("a", 0, 0, [0])), (frozenset([1]), 0.9, _sym("b", 40, 0, [1]))]
    (score, partition), = _find_best_partitions(2, groups, top_k=1)
    assert _covered(partition) == {0, 1}


def test_clashing_strokes_fall_back_to_one_skip():
    # strokes 0 and 1 only have candidates that clash (same place);
    # stroke 2 is fine. No full cover exists -> best cover skips one.
    from mathnote_ocr.engine.grouper import _find_best_partitions

    groups = [
        (frozenset([0]), 0.9, _sym("a", 0, 0, [0])),
        (frozenset([1]), 0.8, _sym("b", 1, 1, [1])),
        (frozenset([2]), 0.9, _sym("c", 60, 0, [2])),
    ]
    results = _find_best_partitions(3, groups, top_k=3)
    assert results
    for _score, partition in results:
        assert len(_covered(partition)) == 2
        assert 2 in _covered(partition)


def test_stroke_without_candidates_is_skipped():
    from mathnote_ocr.engine.grouper import _find_best_partitions

    groups = [(frozenset([0]), 0.9, _sym("a", 0, 0, [0])), (frozenset([2]), 0.9, _sym("c", 60, 0, [2]))]
    (score, partition), = _find_best_partitions(3, groups, top_k=1)
    assert _covered(partition) == {0, 2}


def test_api_reports_unexplained_strokes(monkeypatch):
    """The partial result keeps the explained symbols and names the rest."""
    from mathnote_ocr import MathOCR, api

    ocr = MathOCR()
    strokes = [[(0, 0), (20, 30)], [(300, 0), (320, 30)], [(600, 0), (620, 30)]]  # far apart: 3 symbols
    real = api.group_and_classify

    def drop_middle(stroke_objs, *args, **kwargs):
        partitions = real(stroke_objs, *args, **kwargs)
        return [[s for s in p if all(st.id != 1 for st in s.strokes)] for p in partitions]

    monkeypatch.setattr(api, "group_and_classify", drop_middle)
    expr = ocr.detect(strokes)
    assert expr.unexplained_stroke_ids == [1]
    assert all(st.id != 1 for s in expr for st in s.strokes)
    assert expr.to_dict()["unexplained_stroke_ids"] == [1]
