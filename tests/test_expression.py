"""Contract tests for Expression — public-facing return type of detect()."""

import pytest

from mathnote_ocr.bbox import BBox
from mathnote_ocr.engine.stroke import Stroke, StrokePoint
from mathnote_ocr.expression import DetectedSymbol, Expression, empty_expression
from mathnote_ocr.tree_parser.tree_v2 import ROOT_ID, Edge, Node, Symbol, Tree


def _stroke(id: int) -> Stroke:
    return Stroke(
        id=id,
        points=[StrokePoint(id * 10, 0), StrokePoint(id * 10 + 5, 5)],
        bbox=BBox(id * 10, 0, 5, 5),
    )


def _symbol(id: int, name: str = "x", strokes: list[Stroke] | None = None) -> DetectedSymbol:
    s = strokes or [_stroke(id)]
    return DetectedSymbol(name=name, bbox=BBox(id * 10, 0, 5, 5), strokes=s, confidence=0.9)


def _xy_expr() -> Expression:
    """Two-symbol expression: x with superscript y."""
    s0, s1 = _stroke(0), _stroke(1)
    syms = {0: _symbol(0, "x", [s0]), 1: _symbol(1, "y", [s1])}
    tree = Tree(
        (
            Node(Symbol(0, "x", BBox(0, 0, 5, 5)), ROOT_ID, Edge.ROOT, 0),
            Node(Symbol(1, "y", BBox(10, 0, 5, 5)), 0, Edge.SUP, 0),
        )
    )
    return Expression(strokes=[s0, s1], symbols=syms, tree=tree, confidence=0.85)


# ── empty_expression ────────────────────────────────────────────────────


def test_empty_expression_is_falsy():
    e = empty_expression()
    assert not e
    assert len(e) == 0
    assert e.latex == ""
    assert e.confidence == 0.0
    assert e.alternatives == []


def test_empty_expression_iter():
    e = empty_expression()
    assert list(e) == []


# ── basic queries ───────────────────────────────────────────────────────


def test_truthy_when_symbols_present():
    e = _xy_expr()
    assert bool(e)
    assert len(e) == 2


def test_iter_yields_symbols():
    e = _xy_expr()
    names = sorted(s.name for s in e)
    assert names == ["x", "y"]


def test_repr_includes_latex():
    e = _xy_expr()
    r = repr(e)
    assert "Expression" in r
    assert "n_symbols=2" in r


# ── alternatives ────────────────────────────────────────────────────────


def test_alternatives_default_empty():
    e = _xy_expr()
    assert e.alternatives == []


def test_alternatives_passed_through():
    alt = empty_expression()
    e = Expression(strokes=[], symbols={}, tree=None, alternatives=[alt])
    assert e.alternatives == [alt]


# ── latex ───────────────────────────────────────────────────────────────


def test_latex_empty_when_no_tree():
    e = Expression(strokes=[], symbols={}, tree=None)
    assert e.latex == ""


def test_latex_renders_xy():
    e = _xy_expr()
    # x with y as superscript
    assert e.latex == "x^{y}"


# ── rename ──────────────────────────────────────────────────────────────


def test_rename_returns_new_expression():
    e1 = _xy_expr()
    e2 = e1.rename(0, "z")
    assert e1.symbols[0].name == "x"  # original unchanged
    assert e2.symbols[0].name == "z"
    assert e2.symbols[1].name == "y"  # other symbol untouched


def test_rename_updates_tree():
    e1 = _xy_expr()
    e2 = e1.rename(0, "z")
    assert e2.tree[0].symbol.name == "z"


def test_rename_preserves_strokes():
    e1 = _xy_expr()
    e2 = e1.rename(0, "z")
    assert e2.symbols[0].strokes == e1.symbols[0].strokes


# ── to_dict ─────────────────────────────────────────────────────────────


def test_to_dict_shape():
    e = _xy_expr()
    d = e.to_dict()
    assert "latex" in d
    assert "confidence" in d
    assert "symbols" in d
    assert "tree" in d
    assert isinstance(d["symbols"], list)
    assert len(d["symbols"]) == 2


def test_to_dict_symbol_fields():
    e = _xy_expr()
    d = e.to_dict()
    s = d["symbols"][0]
    assert "id" in s
    assert "name" in s
    assert "bbox" in s
    assert "stroke_ids" in s
    assert "confidence" in s


def test_to_dict_stroke_ids_match():
    e = _xy_expr()
    d = e.to_dict()
    by_id = {s["id"]: s for s in d["symbols"]}
    assert by_id[0]["stroke_ids"] == [0]
    assert by_id[1]["stroke_ids"] == [1]


def test_to_dict_tree_excludes_root():
    e = _xy_expr()
    d = e.to_dict()
    assert ROOT_ID not in d["tree"]


def test_to_dict_empty():
    e = empty_expression()
    d = e.to_dict()
    assert d["symbols"] == []
    assert d["tree"] == {}
    assert d["latex"] == ""


# ── DetectedSymbol ──────────────────────────────────────────────────────


def test_detected_symbol_is_frozen():
    s = _symbol(0)
    with pytest.raises(Exception):
        s.name = "y"  # type: ignore[misc]


def test_detected_symbol_defaults():
    s = DetectedSymbol(name="x", bbox=BBox(0, 0, 5, 5), strokes=[])
    assert s.confidence == 1.0
    assert s.prototype_distance == 0.0
    assert s.alternatives == []
