"""End-to-end tests that verify pins actually constrain detection output."""

import json
from pathlib import Path

import pytest

from mathnote_ocr import Edge, MathOCR, Tree

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


# ── Single-symbol pin enforcement ───────────────────────────────────────


def test_single_stroke_pin_forces_label(ocr, sample_strokes):
    """Pinning one stroke with an unusual label should produce an expression
    containing a symbol with that exact label, claiming exactly that stroke."""
    session = ocr.session()
    sids = [session.add_stroke(s) for s in sample_strokes]

    # Pin the first stroke as "alpha" (regardless of what it actually is)
    target_sid = sids[0]
    pin = Tree.pin_spec(
        strokes={target_sid: session._strokes[target_sid]},
        symbols=[("alpha", [target_sid])],
    )

    expr = ocr.detect(list(session._strokes.values()), pins=[pin])

    # Find the symbol that owns target_sid
    owner = None
    for sym in expr:
        if any(s.id == target_sid for s in sym.strokes):
            owner = sym
            break
    assert owner is not None, "no symbol claims the pinned stroke"
    assert owner.name == "alpha"
    # The pinned symbol should have exactly the pinned strokes (no extras
    # absorbed by grouping).
    assert {s.id for s in owner.strokes} == {target_sid}
    # Forced confidence is 1.0
    assert owner.confidence == 1.0


def test_multistroke_pin_forces_grouping(ocr, sample_strokes):
    """Pinning two strokes together as a single symbol should produce one
    symbol claiming both — even if the grouper would normally split them."""
    session = ocr.session()
    sids = [session.add_stroke(s) for s in sample_strokes]

    # Pin two arbitrary strokes as a single multi-stroke symbol "="
    a, b = sids[0], sids[1]
    pin = Tree.pin_spec(
        strokes={
            a: session._strokes[a],
            b: session._strokes[b],
        },
        symbols=[("=", [a, b])],
    )

    expr = ocr.detect(list(session._strokes.values()), pins=[pin])

    # There should be exactly one symbol claiming both a and b
    matching = [s for s in expr if {st.id for st in s.strokes} == {a, b}]
    assert len(matching) == 1, f"expected one merged symbol, got {len(matching)}"
    assert matching[0].name == "="


def test_pin_does_not_affect_other_strokes(ocr, sample_strokes):
    """Strokes outside the pin should still be grouped/classified normally."""
    session = ocr.session()
    sids = [session.add_stroke(s) for s in sample_strokes]

    # Run baseline (no pins) to see how the unpinned strokes resolve
    baseline = ocr.detect(list(session._strokes.values()))
    target_sid = sids[0]

    pin = Tree.pin_spec(
        strokes={target_sid: session._strokes[target_sid]},
        symbols=[("alpha", [target_sid])],
    )
    expr = ocr.detect(list(session._strokes.values()), pins=[pin])

    # Strokes other than target_sid should belong to the same symbol *labels*
    # in both runs (they may have different ids since symbol order shifts).
    def labels_for_strokes_excluding(e, excluded_sid):
        return sorted(
            s.name for s in e if not any(st.id == excluded_sid for st in s.strokes)
        )

    assert labels_for_strokes_excluding(baseline, target_sid) == \
        labels_for_strokes_excluding(expr, target_sid)


# ── Subtree pin (structure not yet enforced — just verify no crash) ─────


def test_subtree_pin_forces_internal_edge(ocr, sample_strokes):
    """A 2-symbol pin with internal edge SUP should produce a tree where
    the pinned child is a SUP child of the pinned parent — overriding
    whatever the model would have predicted."""
    session = ocr.session()
    sids = [session.add_stroke(s) for s in sample_strokes]

    a, b = sids[0], sids[1]
    pin = Tree.pin_spec(
        strokes={a: session._strokes[a], b: session._strokes[b]},
        symbols=[("x", [a]), ("2", [b])],
        edges=[(0, 1, Edge.SUP)],
    )
    expr = ocr.detect(list(session._strokes.values()), pins=[pin])

    # Both pinned symbols should appear (strokes are unique per symbol)
    by_stroke_set = {frozenset(s.id for s in sym.strokes): sym for sym in expr}
    parent_sym = by_stroke_set[frozenset([a])]
    child_sym = by_stroke_set[frozenset([b])]
    assert parent_sym.name == "x"
    assert child_sym.name == "2"

    # Find their tree symbol ids and verify the edge
    assert expr.tree is not None
    parent_id = next(
        sid for sid, ds in expr.symbols.items() if ds is parent_sym
    )
    child_id = next(
        sid for sid, ds in expr.symbols.items() if ds is child_sym
    )
    child_node = expr.tree[child_id]
    assert child_node.parent_id == parent_id, (
        f"pin says '2' is SUP child of 'x', "
        f"but tree has parent={child_node.parent_id} (expected {parent_id})"
    )
    assert child_node.edge_type == Edge.SUP, (
        f"pin says edge=SUP, but tree has edge={child_node.edge_type}"
    )


def test_subtree_pin_with_sub_edge(ocr, sample_strokes):
    """Same as above but with SUB edge — verifies edge type is honored."""
    session = ocr.session()
    sids = [session.add_stroke(s) for s in sample_strokes]

    a, b = sids[0], sids[1]
    pin = Tree.pin_spec(
        strokes={a: session._strokes[a], b: session._strokes[b]},
        symbols=[("x", [a]), ("i", [b])],
        edges=[(0, 1, Edge.SUB)],
    )
    expr = ocr.detect(list(session._strokes.values()), pins=[pin])

    by_stroke_set = {frozenset(s.id for s in sym.strokes): sym for sym in expr}
    parent_sym = by_stroke_set[frozenset([a])]
    child_sym = by_stroke_set[frozenset([b])]
    parent_id = next(sid for sid, ds in expr.symbols.items() if ds is parent_sym)
    child_id = next(sid for sid, ds in expr.symbols.items() if ds is child_sym)

    assert expr.tree[child_id].parent_id == parent_id
    assert expr.tree[child_id].edge_type == Edge.SUB
