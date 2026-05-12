"""Tests for Tree.pin_spec factory.

A pin captures (label, stroke_ids) per symbol plus optional internal edges
between pinned symbols. The OCR pipeline preserves those edges and injects
an `expr` parent for connectedness when the pin is a forest.
"""

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
    bb = t[0].symbol.bbox
    assert bb.x == 0
    assert bb.y == 0
    assert bb.h == 15  # spans y=0..15


# ── Multi-symbol pins ──────────────────────────────────────────────────


def test_multi_symbol_pin_no_edges():
    """Pin with no internal edges = forest of local roots."""
    strokes = {3: _stroke(3, 0, 0), 4: _stroke(4, 10, 0)}
    t = Tree.pin_spec(strokes=strokes, symbols=[("x", [3]), ("2", [4])])
    assert len(t) == 2
    assert t[0].parent_id == ROOT_ID
    assert t[1].parent_id == ROOT_ID


def test_multi_symbol_pin_with_internal_edge():
    """Pin with internal edge: child has parent in pin, edge type preserved."""
    strokes = {3: _stroke(3, 0, 0), 4: _stroke(4, 10, 0)}
    t = Tree.pin_spec(
        strokes=strokes,
        symbols=[("x", [3]), ("2", [4])],
        edges=[(0, 1, Edge.SUP)],
    )
    assert t[0].parent_id == ROOT_ID  # local root
    assert t[1].parent_id == 0
    assert t[1].edge_type == Edge.SUP


def test_pin_self_loop_raises():
    s = {1: _stroke(1, 0, 0), 2: _stroke(2, 10, 0)}
    with pytest.raises(ValueError, match="self-loop"):
        Tree.pin_spec(strokes=s, symbols=[("x", [1]), ("y", [2])], edges=[(0, 0, Edge.SUP)])


def test_pin_root_edge_raises():
    s = {1: _stroke(1, 0, 0), 2: _stroke(2, 10, 0)}
    with pytest.raises(ValueError, match="Edge.ROOT is reserved"):
        Tree.pin_spec(strokes=s, symbols=[("x", [1]), ("y", [2])], edges=[(0, 1, Edge.ROOT)])


def test_pin_multi_parent_raises():
    s = {1: _stroke(1, 0, 0), 2: _stroke(2, 10, 0), 3: _stroke(3, 20, 0)}
    with pytest.raises(ValueError, match="multiple parents"):
        Tree.pin_spec(
            strokes=s,
            symbols=[("x", [1]), ("y", [2]), ("z", [3])],
            edges=[(0, 1, Edge.SUP), (2, 1, Edge.SUB)],
        )


def test_pin_cycle_raises():
    s = {1: _stroke(1, 0, 0), 2: _stroke(2, 10, 0)}
    with pytest.raises(ValueError, match="cycle|no root"):
        Tree.pin_spec(
            strokes=s,
            symbols=[("x", [1]), ("y", [2])],
            edges=[(0, 1, Edge.SUP), (1, 0, Edge.SUB)],
        )


def test_pin_bbox_derived_correctly():
    s1 = _stroke(1, 0, 0)
    s2 = _stroke(2, 100, 0)
    t = Tree.pin_spec(strokes={1: s1, 2: s2}, symbols=[("=", [1, 2])])
    bb = t[0].symbol.bbox
    # Should span from (0,0) to (105, 5)
    assert bb.x == 0
    assert bb.x2 == 105


# ── Validation ──────────────────────────────────────────────────────────


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


def test_empty_symbols_raises():
    with pytest.raises(ValueError, match="at least one symbol"):
        Tree.pin_spec(strokes={}, symbols=[])
