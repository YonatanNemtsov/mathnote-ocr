"""API plumbing tests for pins on MathOCR.detect and Session.

These verify the parameter is accepted, validation fires on bad references,
and Session pin lifecycle works. Enforcement (grouper / classifier / builder
honoring pins) is tested separately once it's implemented.
"""

import json
from pathlib import Path

import pytest

from mathnote_ocr import MathOCR, Tree

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
    pin = Tree.pin_spec(
        strokes={sid: session._strokes[sid] for sid in sids[:1]},
        symbols=[("x", [sids[0]])],
    )
    # Use one-shot detect with the actual stroke list from session
    expr = ocr.detect(list(session._strokes.values()), pins=[pin])
    assert bool(expr)


def test_detect_rejects_pin_with_unknown_stroke(ocr, sample_strokes):
    """Pin that references a stroke not in the input must raise."""
    # Build a session, get a real stroke, then craft a pin referencing
    # an id that we know isn't in the one-shot input.
    session = ocr.session()
    real_id = session.add_stroke(sample_strokes[0])
    pin = Tree.pin_spec(
        strokes={real_id: session._strokes[real_id]},
        symbols=[("x", [real_id])],
    )
    # Now call detect with a *different* stroke set that doesn't include real_id
    # by passing the raw points list (auto-allocated ids start at 0).
    # The pin references real_id (which is also 0 in this case if first); use a
    # pin that explicitly references an out-of-range id by making one ourselves:
    out_of_range_pin = Tree.pin_spec(
        strokes={999: session._strokes[real_id]},  # use the bbox, label as id 999
        symbols=[("x", [999])],
    )
    with pytest.raises(ValueError, match="references stroke id 999"):
        ocr.detect(sample_strokes, pins=[out_of_range_pin])


# ── Session lifecycle ──────────────────────────────────────────────────


def test_session_pins_default_empty(ocr):
    session = ocr.session()
    assert session.pins == ()


def test_session_pin_adds_to_list(ocr, sample_strokes):
    session = ocr.session()
    sid = session.add_stroke(sample_strokes[0])
    pin = Tree.pin_spec(
        strokes={sid: session._strokes[sid]},
        symbols=[("x", [sid])],
    )
    session.add_pin(pin)
    assert len(session.pins) == 1
    assert session.pins[0] == pin


def test_session_pin_validates_strokes(ocr):
    session = ocr.session()
    # No strokes added, so pin referencing stroke 0 should fail
    fake_stroke = type("S", (), {"bbox": type("B", (), {"x": 0, "y": 0, "x2": 5, "y2": 5})()})()
    # Easier: make a pin via pin_spec with a session-foreign stroke
    from mathnote_ocr.bbox import BBox
    from mathnote_ocr.engine.stroke import Stroke

    foreign = Stroke(id=0, bbox=BBox(0, 0, 5, 5))
    pin = Tree.pin_spec(strokes={0: foreign}, symbols=[("x", [0])])
    with pytest.raises(ValueError, match="references stroke id 0"):
        session.add_pin(pin)


def test_session_unpin(ocr, sample_strokes):
    session = ocr.session()
    sid = session.add_stroke(sample_strokes[0])
    pin = Tree.pin_spec(
        strokes={sid: session._strokes[sid]},
        symbols=[("x", [sid])],
    )
    session.add_pin(pin)
    session.remove_pin(pin)
    assert session.pins == ()


def test_session_unpin_unknown_raises(ocr, sample_strokes):
    session = ocr.session()
    sid = session.add_stroke(sample_strokes[0])
    pin = Tree.pin_spec(
        strokes={sid: session._strokes[sid]},
        symbols=[("x", [sid])],
    )
    with pytest.raises(ValueError, match="not found"):
        session.remove_pin(pin)


def test_session_clear_pins(ocr, sample_strokes):
    session = ocr.session()
    sid = session.add_stroke(sample_strokes[0])
    pin = Tree.pin_spec(
        strokes={sid: session._strokes[sid]},
        symbols=[("x", [sid])],
    )
    session.add_pin(pin)
    session.clear_pins()
    assert session.pins == ()


def test_remove_stroke_drops_referencing_pin(ocr, sample_strokes):
    """If a stroke is removed, any pin referencing it should be dropped."""
    session = ocr.session()
    sid = session.add_stroke(sample_strokes[0])
    pin = Tree.pin_spec(
        strokes={sid: session._strokes[sid]},
        symbols=[("x", [sid])],
    )
    session.add_pin(pin)
    session.remove_stroke(sid)
    assert session.pins == ()


def test_remove_stroke_keeps_unrelated_pins(ocr, sample_strokes):
    session = ocr.session()
    sid_a = session.add_stroke(sample_strokes[0])
    sid_b = session.add_stroke(sample_strokes[1])
    pin_a = Tree.pin_spec(
        strokes={sid_a: session._strokes[sid_a]},
        symbols=[("x", [sid_a])],
    )
    pin_b = Tree.pin_spec(
        strokes={sid_b: session._strokes[sid_b]},
        symbols=[("y", [sid_b])],
    )
    session.add_pin(pin_a)
    session.add_pin(pin_b)
    session.remove_stroke(sid_a)
    assert session.pins == (pin_b,)


def test_session_clear_also_drops_pins(ocr, sample_strokes):
    session = ocr.session()
    sid = session.add_stroke(sample_strokes[0])
    pin = Tree.pin_spec(
        strokes={sid: session._strokes[sid]},
        symbols=[("x", [sid])],
    )
    session.add_pin(pin)
    session.clear()
    assert session.pins == ()


def test_session_detect_with_pins_runs(ocr, sample_strokes):
    """Detect with pins should not crash; output shape unchanged for now."""
    session = ocr.session()
    for s in sample_strokes:
        session.add_stroke(s)
    sid_first = list(session._strokes.keys())[0]
    pin = Tree.pin_spec(
        strokes={sid_first: session._strokes[sid_first]},
        symbols=[("x", [sid_first])],
    )
    session.add_pin(pin)
    expr = session.detect()
    assert bool(expr)


# ── Edge sub-tree pin ──────────────────────────────────────────────────


def test_subtree_pin_passes_validation(ocr, sample_strokes):
    """Build a 2-symbol pin and feed to detect — should not raise."""
    session = ocr.session()
    sid_a = session.add_stroke(sample_strokes[0])
    sid_b = session.add_stroke(sample_strokes[1])
    pin = Tree.pin_spec(
        strokes={sid_a: session._strokes[sid_a], sid_b: session._strokes[sid_b]},
        symbols=[("x", [sid_a]), ("2", [sid_b])],
    )
    expr = ocr.detect(list(session._strokes.values()), pins=[pin])
    assert bool(expr)


# ── Cross-pin disjointness ──────────────────────────────────────────────


def test_overlapping_pins_at_detect_raises(ocr, sample_strokes):
    """Two pins claiming the same stroke must raise at detect time."""
    session = ocr.session()
    sid = session.add_stroke(sample_strokes[0])
    p1 = Tree.pin_spec(
        strokes={sid: session._strokes[sid]},
        symbols=[("x", [sid])],
    )
    p2 = Tree.pin_spec(
        strokes={sid: session._strokes[sid]},
        symbols=[("y", [sid])],
    )
    with pytest.raises(ValueError, match=f"stroke {sid} is claimed by both"):
        ocr.detect(list(session._strokes.values()), pins=[p1, p2])


def test_session_pin_rejects_overlap_with_existing(ocr, sample_strokes):
    """Session.pin() should reject a pin that overlaps with an already-active pin."""
    session = ocr.session()
    sid = session.add_stroke(sample_strokes[0])
    p1 = Tree.pin_spec(
        strokes={sid: session._strokes[sid]},
        symbols=[("x", [sid])],
    )
    p2 = Tree.pin_spec(
        strokes={sid: session._strokes[sid]},
        symbols=[("y", [sid])],
    )
    session.add_pin(p1)
    with pytest.raises(ValueError, match=f"stroke {sid} is claimed by both"):
        session.add_pin(p2)
    # And the overlapping pin should NOT be added
    assert session.pins == (p1,)


def test_session_pin_allows_disjoint_pins(ocr, sample_strokes):
    """Two pins on disjoint strokes should both be added."""
    session = ocr.session()
    sid_a = session.add_stroke(sample_strokes[0])
    sid_b = session.add_stroke(sample_strokes[1])
    p1 = Tree.pin_spec(
        strokes={sid_a: session._strokes[sid_a]},
        symbols=[("x", [sid_a])],
    )
    p2 = Tree.pin_spec(
        strokes={sid_b: session._strokes[sid_b]},
        symbols=[("y", [sid_b])],
    )
    session.add_pin(p1)
    session.add_pin(p2)
    assert session.pins == (p1, p2)
