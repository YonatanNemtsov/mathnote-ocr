"""End-to-end detection tests.

Loads real handwritten samples and runs them through MathOCR().detect().
Asserts structural invariants — not exact LaTeX equality (the model isn't
deterministic enough across configs to make that a useful contract).

These tests load the model, so they're slower (~5–10s per session). The
fixture is module-scoped so the model loads once per test module.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mathnote_ocr import MathOCR
from mathnote_ocr.tree_parser.tree_v2 import ROOT_ID

REPO_ROOT = Path(__file__).resolve().parents[1]
HANDWRITTEN = REPO_ROOT / "data" / "shared" / "tree_handwritten" / "run_001" / "train_strokes.jsonl"


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ocr() -> MathOCR:
    """One MathOCR instance per test module — model load is expensive."""
    if not HANDWRITTEN.exists():
        pytest.skip(f"Handwritten test data not found at {HANDWRITTEN}")
    return MathOCR()


@pytest.fixture(scope="module")
def sample_strokes() -> list[list[tuple[float, float]]]:
    """A handwritten sample, flattened to a list of stroke point lists."""
    if not HANDWRITTEN.exists():
        pytest.skip(f"Handwritten test data not found at {HANDWRITTEN}")
    with open(HANDWRITTEN) as f:
        sample = json.loads(f.readline())
    return [
        [(p["x"], p["y"]) for p in stroke]
        for sym in sample["symbols"]
        for stroke in sym["strokes"]
    ]


# ── Smoke ───────────────────────────────────────────────────────────────


def test_detect_returns_nonempty_expression(ocr, sample_strokes):
    expr = ocr.detect(sample_strokes)
    assert bool(expr)
    assert len(expr) > 0
    assert expr.latex
    assert 0.0 <= expr.confidence <= 1.0


def test_detect_empty_input_returns_empty_expression(ocr):
    expr = ocr.detect([])
    assert not bool(expr)
    assert len(expr) == 0
    assert expr.latex == ""


# ── Structural invariants ───────────────────────────────────────────────


def test_strokes_preserved_in_expression(ocr, sample_strokes):
    expr = ocr.detect(sample_strokes)
    assert len(expr.strokes) == len(sample_strokes)


def test_stroke_ids_unique(ocr, sample_strokes):
    expr = ocr.detect(sample_strokes)
    ids = [s.id for s in expr.strokes]
    assert len(ids) == len(set(ids))


def test_every_symbol_strokes_are_in_expr_strokes(ocr, sample_strokes):
    """No symbol should reference a stroke that isn't in expr.strokes."""
    expr = ocr.detect(sample_strokes)
    expr_stroke_ids = {s.id for s in expr.strokes}
    for sym in expr:
        for stroke in sym.strokes:
            assert stroke.id in expr_stroke_ids


def test_strokes_partition_across_symbols(ocr, sample_strokes):
    """Each input stroke ends up in exactly one detected symbol (no double-claim)."""
    expr = ocr.detect(sample_strokes)
    seen: set[int] = set()
    for sym in expr:
        for stroke in sym.strokes:
            assert stroke.id not in seen, f"stroke {stroke.id} in multiple symbols"
            seen.add(stroke.id)


def test_tree_node_ids_match_symbol_keys(ocr, sample_strokes):
    """Every non-root tree node corresponds to a symbol id, and vice versa."""
    expr = ocr.detect(sample_strokes)
    if expr.tree is None:
        pytest.skip("no tree returned")
    tree_ids = {sid for sid in expr.tree.nodes if sid != ROOT_ID}
    symbol_ids = set(expr.symbols.keys())
    assert tree_ids == symbol_ids


def test_tree_parents_are_valid(ocr, sample_strokes):
    """Every tree node's parent must be ROOT or another node in the tree."""
    expr = ocr.detect(sample_strokes)
    if expr.tree is None:
        pytest.skip("no tree returned")
    valid_parents = set(expr.tree.nodes.keys())
    for sid, node in expr.tree.nodes.items():
        if sid == ROOT_ID:
            continue
        assert node.parent_id == ROOT_ID or node.parent_id in valid_parents


def test_symbol_bbox_within_stroke_bounds(ocr, sample_strokes):
    """Each symbol's bbox should encompass at least one of its stroke points
    (sanity check that bbox derivation is coherent)."""
    expr = ocr.detect(sample_strokes)
    for sym in expr:
        if not sym.strokes:
            continue
        bb = sym.bbox
        # Find at least one point inside the symbol's bbox
        any_inside = False
        for stroke in sym.strokes:
            for p in stroke.points:
                if bb.x <= p.x <= bb.x2 and bb.y <= p.y <= bb.y2:
                    any_inside = True
                    break
            if any_inside:
                break
        assert any_inside, f"symbol {sym.name} bbox {bb} contains no stroke points"


# ── to_dict serialization ───────────────────────────────────────────────


def test_to_dict_round_trips_basic_shape(ocr, sample_strokes):
    expr = ocr.detect(sample_strokes)
    d = expr.to_dict()
    assert d["latex"] == expr.latex
    assert len(d["symbols"]) == len(expr)
    # Every stroke_id in to_dict belongs to a real stroke
    expr_stroke_ids = {s.id for s in expr.strokes}
    for s in d["symbols"]:
        for sid in s["stroke_ids"]:
            assert sid in expr_stroke_ids


# ── Determinism ─────────────────────────────────────────────────────────


def test_detect_is_deterministic(ocr, sample_strokes):
    """Same input → same LaTeX. Catches accidental nondeterminism in the pipeline."""
    e1 = ocr.detect(sample_strokes)
    e2 = ocr.detect(sample_strokes)
    assert e1.latex == e2.latex
    assert len(e1) == len(e2)


# ── Session ─────────────────────────────────────────────────────────────


def test_session_incremental_matches_oneshot(ocr, sample_strokes):
    """Adding strokes one-by-one to a session and detecting at the end
    should produce the same result as a single detect() call."""
    one_shot = ocr.detect(sample_strokes)

    session = ocr.session()
    for stroke in sample_strokes:
        session.add_stroke(stroke)
    incremental = session.detect()

    assert one_shot.latex == incremental.latex
    assert len(one_shot) == len(incremental)


def test_session_remove_stroke(ocr, sample_strokes):
    session = ocr.session()
    ids = [session.add_stroke(s) for s in sample_strokes]
    expr_full = session.detect()
    session.remove_stroke(ids[-1])
    expr_partial = session.detect()
    # Removing a stroke should reduce stroke count by exactly one
    assert len(expr_partial.strokes) == len(expr_full.strokes) - 1
