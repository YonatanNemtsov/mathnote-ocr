"""Tests for Tree.pin_spec factory."""

import pytest

from mathnote_ocr.bbox import BBox
from mathnote_ocr.engine.stroke import Stroke, StrokePoint
from mathnote_ocr.tree_parser.tree_v2 import ROOT_ID, Edge, Tree


def _stroke(id: int, x: float, y: float) -> Stroke:
    return Stroke(
        id=id,
        points=[StrokePoint(x, y), StrokePoint(x + 4, y + 4)],
        bbox=BBox(x, y, 5, 5),
    )


# ── Single-symbol pins ──────────────────────────────────────────────────


def test_single_symbol_pin():
    s = _stroke(3, 0, 0)
    t = Tree.pin_spec(strokes={3: s}, symbols=[("x", [3])])
    assert len(t) == 1
    assert t[0].symbol.name == "x"
    assert t[0].symbol.stroke_ids == (3,)
    assert t[0].parent_id == ROOT_ID
    assert t[0].edge_type == Edge.ROOT


def test_multistroke_single_symbol_pin():
    """A symbol composed of multiple strokes (e.g. '=' with two bars)."""
    strokes = {1: _stroke(1, 0, 0), 2: _stroke(2, 0, 10)}
    t = Tree.pin_spec(strokes=strokes, symbols=[("=", [1, 2])])
    assert len(t) == 1
    assert t[0].symbol.stroke_ids == (1, 2)
    # Bbox should union both strokes
    bb = t[0].symbol.bbox
    assert bb.x == 0
    assert bb.y == 0
    assert bb.h == 15  # spans y=0..15


# ── Multi-symbol subtree pins ───────────────────────────────────────────


def test_subtree_pin_x_squared():
    """x with superscript 2."""
    strokes = {3: _stroke(3, 0, 0), 4: _stroke(4, 10, 0)}
    t = Tree.pin_spec(
        strokes=strokes,
        symbols=[("x", [3]), ("2", [4])],
        edges=[(0, 1, Edge.SUP)],
    )
    assert len(t) == 2
    # Symbol 0 (x) is root
    assert t[0].parent_id == ROOT_ID
    # Symbol 1 (2) is SUP of symbol 0
    assert t[1].parent_id == 0
    assert t[1].edge_type == Edge.SUP
    assert t[1].order == 0


def test_subtree_pin_with_explicit_order():
    strokes = {1: _stroke(1, 0, 0), 2: _stroke(2, 10, 0), 3: _stroke(3, 20, 0)}
    t = Tree.pin_spec(
        strokes=strokes,
        symbols=[("x", [1]), ("2", [2]), ("3", [3])],
        edges=[(0, 1, Edge.SUP, 0), (0, 2, Edge.SUP, 1)],
    )
    sup_kids = t.children_by_edge(0, Edge.SUP)
    assert sup_kids == (1, 2)


def test_pin_bbox_derived_correctly():
    s1 = _stroke(1, 0, 0)
    s2 = _stroke(2, 100, 0)
    t = Tree.pin_spec(strokes={1: s1, 2: s2}, symbols=[("=", [1, 2])])
    bb = t[0].symbol.bbox
    # Should span from (0,0) to (105, 5)
    assert bb.x == 0
    assert bb.x2 == 105


# ── Validation: stroke refs ─────────────────────────────────────────────


def test_missing_stroke_raises():
    with pytest.raises(ValueError, match="stroke 99 not in strokes"):
        Tree.pin_spec(strokes={1: _stroke(1, 0, 0)}, symbols=[("x", [99])])


def test_empty_label_raises():
    with pytest.raises(ValueError, match="label must be non-empty"):
        Tree.pin_spec(strokes={1: _stroke(1, 0, 0)}, symbols=[("", [1])])


def test_duplicate_stroke_within_pin_raises():
    s = {1: _stroke(1, 0, 0), 2: _stroke(2, 10, 0)}
    with pytest.raises(ValueError, match="stroke 1 appears in multiple symbols"):
        Tree.pin_spec(strokes=s, symbols=[("x", [1, 2]), ("y", [1])])


# ── Validation: edges ───────────────────────────────────────────────────


def test_edge_index_out_of_range_raises():
    s = {1: _stroke(1, 0, 0)}
    with pytest.raises(ValueError, match="index out of range"):
        Tree.pin_spec(strokes=s, symbols=[("x", [1])], edges=[(0, 5, Edge.SUP)])


def test_self_loop_raises():
    s = {1: _stroke(1, 0, 0), 2: _stroke(2, 10, 0)}
    with pytest.raises(ValueError, match="self-loop"):
        Tree.pin_spec(
            strokes=s,
            symbols=[("x", [1]), ("y", [2])],
            edges=[(0, 0, Edge.SUP)],
        )


def test_root_edge_in_user_edges_raises():
    s = {1: _stroke(1, 0, 0), 2: _stroke(2, 10, 0)}
    with pytest.raises(ValueError, match="Edge.ROOT is reserved"):
        Tree.pin_spec(
            strokes=s,
            symbols=[("x", [1]), ("y", [2])],
            edges=[(0, 1, Edge.ROOT)],
        )


def test_multiple_parents_raises():
    s = {1: _stroke(1, 0, 0), 2: _stroke(2, 10, 0), 3: _stroke(3, 20, 0)}
    with pytest.raises(ValueError, match="multiple parents"):
        Tree.pin_spec(
            strokes=s,
            symbols=[("x", [1]), ("y", [2]), ("z", [3])],
            edges=[(0, 1, Edge.SUP), (2, 1, Edge.SUB)],
        )


# ── Validation: structure ───────────────────────────────────────────────


def test_empty_symbols_raises():
    with pytest.raises(ValueError, match="at least one symbol"):
        Tree.pin_spec(strokes={}, symbols=[])


def test_multiple_roots_raises():
    s = {1: _stroke(1, 0, 0), 2: _stroke(2, 10, 0)}
    with pytest.raises(ValueError, match="exactly one root"):
        Tree.pin_spec(strokes=s, symbols=[("x", [1]), ("y", [2])], edges=[])


def test_disconnected_node_raises():
    s = {1: _stroke(1, 0, 0), 2: _stroke(2, 10, 0), 3: _stroke(3, 20, 0)}
    # 0 is the only root, but 2 has no edge to it
    # Actually with my validation: root candidates are {0, 2}, raises "multiple roots".
    # To get disconnected without multi-root we'd need a cycle.  Test a cycle:
    with pytest.raises(ValueError, match="exactly one root|cycle|not reachable"):
        Tree.pin_spec(
            strokes=s,
            symbols=[("x", [1]), ("y", [2]), ("z", [3])],
            edges=[(1, 2, Edge.SUP), (2, 1, Edge.SUB)],  # cycle 1→2→1; 0 isolated
        )
