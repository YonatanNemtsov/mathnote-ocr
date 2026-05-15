"""API plumbing tests for pins on MathOCR.detect and Session.

These verify the parameter is accepted, validation fires on bad references,
and Session pin lifecycle works. Enforcement (grouper / classifier / builder
honoring pins) is tested separately once it's implemented.
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


# ── one-shot: detect(strokes, pins=...) ─────────────────────────────────


def test_detect_accepts_empty_pins_list(ocr, sample_strokes):
    expr = ocr.detect(sample_strokes, pins=[])
    assert bool(expr)


def test_detect_accepts_none_pins(ocr, sample_strokes):
    expr = ocr.detect(sample_strokes, pins=None)
    assert bool(expr)


def test_detect_accepts_valid_pin(ocr, sample_strokes):
    """Pin referencing real stroke ids is accepted (pre-enforcement: doesn't
    necessarily change the output, just shouldn't raise)."""
    session = ocr.session()
    sids = [session.add_stroke(s) for s in sample_strokes]
    first = session._strokes[sids[0]]
    pin = PinnedTree.build(symbols=[PinSymbol("x", [first])])
    expr = ocr.detect(list(session._strokes.values()), pins=[pin])
    assert bool(expr)


def test_detect_rejects_pin_with_unknown_stroke(ocr, sample_strokes):
    """Pin that references a stroke not in the input must raise."""
    from mathnote_ocr.bbox import BBox
    from mathnote_ocr.engine.stroke import Stroke

    foreign = Stroke(id=999, bbox=BBox(0, 0, 5, 5))
    out_of_range_pin = PinnedTree.build(symbols=[PinSymbol("x", [foreign])])
    with pytest.raises(ValueError, match="references stroke id 999"):
        ocr.detect(sample_strokes, pins=[out_of_range_pin])


# ── Session lifecycle ──────────────────────────────────────────────────


def test_session_pins_default_empty(ocr):
    session = ocr.session()
    assert session.pins == ()


def test_session_pin_adds_to_list(ocr, sample_strokes):
    session = ocr.session()
    sid = session.add_stroke(sample_strokes[0])
    pin = PinnedTree.build(symbols=[PinSymbol("x", [session._strokes[sid]])])
    session.add_pin(pin)
    assert len(session.pins) == 1
    assert session.pins[0] == pin


def test_session_pin_validates_strokes(ocr):
    session = ocr.session()
    from mathnote_ocr.bbox import BBox
    from mathnote_ocr.engine.stroke import Stroke

    foreign = Stroke(id=0, bbox=BBox(0, 0, 5, 5))
    pin = PinnedTree.build(symbols=[PinSymbol("x", [foreign])])
    with pytest.raises(ValueError, match="references stroke id 0"):
        session.add_pin(pin)


def test_session_unpin(ocr, sample_strokes):
    session = ocr.session()
    sid = session.add_stroke(sample_strokes[0])
    pin = PinnedTree.build(symbols=[PinSymbol("x", [session._strokes[sid]])])
    session.add_pin(pin)
    session.remove_pin(pin)
    assert session.pins == ()


def test_session_unpin_unknown_raises(ocr, sample_strokes):
    session = ocr.session()
    sid = session.add_stroke(sample_strokes[0])
    pin = PinnedTree.build(symbols=[PinSymbol("x", [session._strokes[sid]])])
    with pytest.raises(ValueError, match="not found"):
        session.remove_pin(pin)


def test_session_clear_pins(ocr, sample_strokes):
    session = ocr.session()
    sid = session.add_stroke(sample_strokes[0])
    pin = PinnedTree.build(symbols=[PinSymbol("x", [session._strokes[sid]])])
    session.add_pin(pin)
    session.clear_pins()
    assert session.pins == ()


def test_remove_stroke_drops_referencing_pin(ocr, sample_strokes):
    """If a stroke is removed, any pin referencing it should be dropped."""
    session = ocr.session()
    sid = session.add_stroke(sample_strokes[0])
    pin = PinnedTree.build(symbols=[PinSymbol("x", [session._strokes[sid]])])
    session.add_pin(pin)
    session.remove_stroke(sid)
    assert session.pins == ()


def test_remove_stroke_keeps_unrelated_pins(ocr, sample_strokes):
    session = ocr.session()
    sid_a = session.add_stroke(sample_strokes[0])
    sid_b = session.add_stroke(sample_strokes[1])
    pin_a = PinnedTree.build(symbols=[PinSymbol("x", [session._strokes[sid_a]])])
    pin_b = PinnedTree.build(symbols=[PinSymbol("y", [session._strokes[sid_b]])])
    session.add_pin(pin_a)
    session.add_pin(pin_b)
    session.remove_stroke(sid_a)
    assert session.pins == (pin_b,)


def test_session_clear_also_drops_pins(ocr, sample_strokes):
    session = ocr.session()
    sid = session.add_stroke(sample_strokes[0])
    pin = PinnedTree.build(symbols=[PinSymbol("x", [session._strokes[sid]])])
    session.add_pin(pin)
    session.clear()
    assert session.pins == ()


def test_session_detect_with_pins_runs(ocr, sample_strokes):
    """Detect with pins should not crash; output shape unchanged for now."""
    session = ocr.session()
    for s in sample_strokes:
        session.add_stroke(s)
    sid_first = list(session._strokes.keys())[0]
    pin = PinnedTree.build(symbols=[PinSymbol("x", [session._strokes[sid_first]])])
    session.add_pin(pin)
    expr = session.detect()
    assert bool(expr)


# ── Edge sub-tree pin ──────────────────────────────────────────────────


def test_subtree_pin_passes_validation(ocr, sample_strokes):
    """Build a 2-symbol pin and feed to detect — should not raise."""
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


# ── Cross-pin disjointness ──────────────────────────────────────────────


def test_overlapping_pins_at_detect_raises(ocr, sample_strokes):
    """Two pins claiming the same stroke must raise at detect time."""
    session = ocr.session()
    sid = session.add_stroke(sample_strokes[0])
    p1 = PinnedTree.build(symbols=[PinSymbol("x", [session._strokes[sid]])])
    p2 = PinnedTree.build(symbols=[PinSymbol("y", [session._strokes[sid]])])
    with pytest.raises(ValueError, match=f"stroke {sid} is claimed by both"):
        ocr.detect(list(session._strokes.values()), pins=[p1, p2])


def test_session_pin_rejects_overlap_with_existing(ocr, sample_strokes):
    """Session.pin() should reject a pin that overlaps with an already-active pin."""
    session = ocr.session()
    sid = session.add_stroke(sample_strokes[0])
    p1 = PinnedTree.build(symbols=[PinSymbol("x", [session._strokes[sid]])])
    p2 = PinnedTree.build(symbols=[PinSymbol("y", [session._strokes[sid]])])
    session.add_pin(p1)
    with pytest.raises(ValueError, match=f"stroke {sid} is claimed by both"):
        session.add_pin(p2)
    assert session.pins == (p1,)


def test_session_pin_allows_disjoint_pins(ocr, sample_strokes):
    """Two pins on disjoint strokes should both be added."""
    session = ocr.session()
    sid_a = session.add_stroke(sample_strokes[0])
    sid_b = session.add_stroke(sample_strokes[1])
    p1 = PinnedTree.build(symbols=[PinSymbol("x", [session._strokes[sid_a]])])
    p2 = PinnedTree.build(symbols=[PinSymbol("y", [session._strokes[sid_b]])])
    session.add_pin(p1)
    session.add_pin(p2)
    assert session.pins == (p1, p2)
