"""Tests for PinnedTree.build factory.

A pin captures PinSymbols (label + strokes) plus optional internal edges
between pinned symbols. The OCR pipeline preserves those edges and
injects an `expr` parent for connectedness when the pin is a forest.
"""

import pytest

from mathnote_ocr.bbox import BBox
from mathnote_ocr.engine.stroke import Stroke, StrokePoint
from mathnote_ocr.pin import PinEdge, PinnedTree, PinSymbol
from mathnote_ocr.tree_parser.tree_v2 import ROOT_ID, Edge


def _stroke(id: int, x: float, y: float) -> Stroke:
    return Stroke(
        id=id,
        points=[StrokePoint(x, y), StrokePoint(x + 4, y + 4)],
        bbox=BBox(x, y, 5, 5),
    )


# ── Single-symbol pins ──────────────────────────────────────────────────


def test_single_symbol_pin():
    s = _stroke(3, 0, 0)
    t = PinnedTree.build(symbols=[PinSymbol("x", [s])])
    assert len(t) == 1
    assert t[0].symbol.name == "x"
    assert t[0].symbol.stroke_ids == (3,)
    assert t[0].parent_id == ROOT_ID
    assert t[0].edge_type == Edge.ROOT


def test_multistroke_single_symbol_pin():
    """A symbol composed of multiple strokes (e.g. '=' with two bars)."""
    s1 = _stroke(1, 0, 0)
    s2 = _stroke(2, 0, 10)
    t = PinnedTree.build(symbols=[PinSymbol("=", [s1, s2])])
    assert len(t) == 1
    assert t[0].symbol.stroke_ids == (1, 2)
    bb = t[0].symbol.bbox
    assert bb.x == 0
    assert bb.y == 0
    assert bb.h == 15  # spans y=0..15


# ── Multi-symbol pins ──────────────────────────────────────────────────


def test_multi_symbol_pin_no_edges():
    """Pin with no internal edges = forest of local roots."""
    s3 = _stroke(3, 0, 0)
    s4 = _stroke(4, 10, 0)
    t = PinnedTree.build(symbols=[PinSymbol("x", [s3]), PinSymbol("2", [s4])])
    assert len(t) == 2
    assert t[0].parent_id == ROOT_ID
    assert t[1].parent_id == ROOT_ID


def test_multi_symbol_pin_with_internal_edge():
    """Pin with internal edge: child has parent in pin, edge type preserved."""
    s3 = _stroke(3, 0, 0)
    s4 = _stroke(4, 10, 0)
    t = PinnedTree.build(
        symbols=[PinSymbol("x", [s3]), PinSymbol("2", [s4])],
        edges=[PinEdge(0, 1, Edge.SUP)],
    )
    assert t[0].parent_id == ROOT_ID  # local root
    assert t[1].parent_id == 0
    assert t[1].edge_type == Edge.SUP


def test_pin_self_loop_raises():
    s1, s2 = _stroke(1, 0, 0), _stroke(2, 10, 0)
    with pytest.raises(ValueError, match="self-loop"):
        PinnedTree.build(
            symbols=[PinSymbol("x", [s1]), PinSymbol("y", [s2])],
            edges=[PinEdge(0, 0, Edge.SUP)],
        )


def test_pin_root_edge_raises():
    s1, s2 = _stroke(1, 0, 0), _stroke(2, 10, 0)
    with pytest.raises(ValueError, match="Edge.ROOT is reserved"):
        PinnedTree.build(
            symbols=[PinSymbol("x", [s1]), PinSymbol("y", [s2])],
            edges=[PinEdge(0, 1, Edge.ROOT)],
        )


def test_pin_multi_parent_raises():
    s1, s2, s3 = _stroke(1, 0, 0), _stroke(2, 10, 0), _stroke(3, 20, 0)
    with pytest.raises(ValueError, match="multiple parents"):
        PinnedTree.build(
            symbols=[PinSymbol("x", [s1]), PinSymbol("y", [s2]), PinSymbol("z", [s3])],
            edges=[PinEdge(0, 1, Edge.SUP), PinEdge(2, 1, Edge.SUB)],
        )


def test_pin_cycle_raises():
    s1, s2 = _stroke(1, 0, 0), _stroke(2, 10, 0)
    with pytest.raises(ValueError, match="cycle|no root"):
        PinnedTree.build(
            symbols=[PinSymbol("x", [s1]), PinSymbol("y", [s2])],
            edges=[PinEdge(0, 1, Edge.SUP), PinEdge(1, 0, Edge.SUB)],
        )


def test_pin_bbox_derived_correctly():
    s1 = _stroke(1, 0, 0)
    s2 = _stroke(2, 100, 0)
    t = PinnedTree.build(symbols=[PinSymbol("=", [s1, s2])])
    bb = t[0].symbol.bbox
    assert bb.x == 0
    assert bb.x2 == 105


# ── Validation ──────────────────────────────────────────────────────────


def test_empty_label_raises():
    s1 = _stroke(1, 0, 0)
    with pytest.raises(ValueError, match="label must be non-empty"):
        PinnedTree.build(symbols=[PinSymbol("", [s1])])


def test_no_strokes_raises():
    with pytest.raises(ValueError, match="must own at least one stroke"):
        PinnedTree.build(symbols=[PinSymbol("x", [])])


def test_duplicate_stroke_within_pin_raises():
    s1, s2 = _stroke(1, 0, 0), _stroke(2, 10, 0)
    with pytest.raises(ValueError, match="stroke 1 appears in multiple symbols"):
        PinnedTree.build(symbols=[PinSymbol("x", [s1, s2]), PinSymbol("y", [s1])])


def test_empty_symbols_raises():
    with pytest.raises(ValueError, match="at least one symbol"):
        PinnedTree.build(symbols=[])
