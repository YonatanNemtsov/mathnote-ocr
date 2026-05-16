"""API plumbing tests for pins.

Pins are constraints passed per detect call. ``MathOCR.detect`` and
``Session.detect`` both accept ``pins=[...]``; the session itself does
not store pins.
"""

import json
from pathlib import Path

import pytest

from mathnote_ocr import MathOCR, PinnedTree, PinSymbol

REPO_ROOT = Path(__file__).resolve().parents[1]
HANDWRITTEN = REPO_ROOT / "data" / "shared" / "tree_handwritten" / "run_001" / "train_strokes.jsonl"


@pytest.fixture(scope="module")
def ocr() -> MathOCR:
    if not HANDWRITTEN.exists():
        pytest.skip(f"Handwritten test data not found at {HANDWRITTEN}")
    return MathOCR()


@pytest.fixture(scope="module")
def sample_strokes() -> list[list[tuple[float, float]]]:
    if not HANDWRITTEN.exists():
        pytest.skip(f"Handwritten test data not found at {HANDWRITTEN}")
    with open(HANDWRITTEN) as f:
        sample = json.loads(f.readline())
    return [
        [(p["x"], p["y"]) for p in stroke]
        for sym in sample["symbols"]
        for stroke in sym["strokes"]
    ]


# ── MathOCR.detect(strokes, pins=...) ───────────────────────────────────


def test_detect_accepts_empty_pins_list(ocr, sample_strokes):
    expr = ocr.detect(sample_strokes, pins=[])
    assert bool(expr)


def test_detect_accepts_none_pins(ocr, sample_strokes):
    expr = ocr.detect(sample_strokes, pins=None)
    assert bool(expr)


def test_detect_accepts_valid_pin(ocr, sample_strokes):
    session = ocr.session()
    sids = [session.add_stroke(s) for s in sample_strokes]
    first = session._strokes[sids[0]]
    pin = PinnedTree.build(symbols=[PinSymbol("x", [first])])
    expr = ocr.detect(list(session._strokes.values()), pins=[pin])
    assert bool(expr)


def test_detect_rejects_pin_with_unknown_stroke(ocr, sample_strokes):
    from mathnote_ocr.bbox import BBox
    from mathnote_ocr.engine.stroke import Stroke

    foreign = Stroke(id=999, bbox=BBox(0, 0, 5, 5))
    out_of_range_pin = PinnedTree.build(symbols=[PinSymbol("x", [foreign])])
    with pytest.raises(ValueError, match="references stroke id 999"):
        ocr.detect(sample_strokes, pins=[out_of_range_pin])


def test_overlapping_pins_at_detect_raises(ocr, sample_strokes):
    session = ocr.session()
    sid = session.add_stroke(sample_strokes[0])
    p1 = PinnedTree.build(symbols=[PinSymbol("x", [session._strokes[sid]])])
    p2 = PinnedTree.build(symbols=[PinSymbol("y", [session._strokes[sid]])])
    with pytest.raises(ValueError, match=f"stroke {sid} is claimed by both"):
        ocr.detect(list(session._strokes.values()), pins=[p1, p2])


# ── Session.detect with pins ────────────────────────────────────────────


def test_session_detect_accepts_pins(ocr, sample_strokes):
    session = ocr.session()
    for s in sample_strokes:
        session.add_stroke(s)
    sid_first = list(session._strokes.keys())[0]
    pin = PinnedTree.build(symbols=[PinSymbol("x", [session._strokes[sid_first]])])
    expr = session.detect(pins=[pin])
    assert bool(expr)


def test_session_detect_rejects_pin_with_unknown_stroke(ocr, sample_strokes):
    from mathnote_ocr.bbox import BBox
    from mathnote_ocr.engine.stroke import Stroke

    session = ocr.session()
    session.add_stroke(sample_strokes[0])
    foreign = Stroke(id=999, bbox=BBox(0, 0, 5, 5))
    pin = PinnedTree.build(symbols=[PinSymbol("x", [foreign])])
    with pytest.raises(ValueError, match="references stroke id 999"):
        session.detect(pins=[pin])


# ── Session.detect on subset of strokes ────────────────────────────────


def test_session_detect_subset(ocr, sample_strokes):
    """Subset detect runs on only the given stroke ids."""
    session = ocr.session()
    sids = [session.add_stroke(s) for s in sample_strokes]
    subset = sids[:3]
    expr = session.detect(stroke_ids=subset)
    detected_stroke_ids = {s.id for sym in expr for s in sym.strokes}
    assert detected_stroke_ids.issubset(set(subset))


def test_session_detect_subset_unknown_id_raises(ocr, sample_strokes):
    session = ocr.session()
    session.add_stroke(sample_strokes[0])
    with pytest.raises(ValueError, match="unknown stroke id"):
        session.detect(stroke_ids=[999])


# ── Subtree pin ────────────────────────────────────────────────────────


def test_subtree_pin_passes_validation(ocr, sample_strokes):
    session = ocr.session()
    sid_a = session.add_stroke(sample_strokes[0])
    sid_b = session.add_stroke(sample_strokes[1])
    pin = PinnedTree.build(
        symbols=[
            PinSymbol("x", [session._strokes[sid_a]]),
            PinSymbol("2", [session._strokes[sid_b]]),
        ],
    )
    expr = ocr.detect(list(session._strokes.values()), pins=[pin])
    assert bool(expr)
