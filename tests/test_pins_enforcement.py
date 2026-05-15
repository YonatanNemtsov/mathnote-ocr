"""End-to-end tests that verify pins actually constrain detection output.

A pin captures PinSymbols (label + strokes). The pipeline enforces:
  1. Each pinned stroke set is grouped with the forced label.
  2. The pinned symbols form a connected subtree in the output tree —
     if not naturally, an `expr` node is inserted as their common parent.
"""

import json
from pathlib import Path

import pytest

from mathnote_ocr import MathOCR, PinnedTree, PinSymbol
from mathnote_ocr.tree_parser.tree_v2 import ROOT_ID

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


def _is_connected_subtree(tree, sym_ids: set[int]) -> bool:
    """True if the pinned set has a common ancestor (real symbol, not ROOT)
    such that paths from each pinned symbol up to that ancestor pass only
    through pinned symbols + the ancestor itself.

    For an `expr` node injected as the common parent of multiple free
    roots, this picks up expr as the LCA and the pinned set forms a
    connected subtree rooted at expr.
    """
    if len(sym_ids) <= 1:
        return True

    sym_list = list(sym_ids)
    paths: dict[int, list[int]] = {}
    for sid in sym_list:
        anc: list[int] = []
        cur = sid
        while cur != tree.root:
            anc.append(cur)
            cur = tree.nodes[cur].parent_id
        paths[sid] = anc

    common = set(paths[sym_list[0]])
    for sid in sym_list[1:]:
        common &= set(paths[sid])
    if not common:
        return False

    lca = next(s for s in paths[sym_list[0]] if s in common)

    augmented: set[int] = set()
    for sid in sym_list:
        cur = sid
        while cur != lca:
            augmented.add(cur)
            cur = tree.nodes[cur].parent_id
        augmented.add(lca)

    for sid in augmented:
        if sid == lca:
            continue
        if tree.nodes[sid].parent_id not in augmented:
            return False
    return True


# ── Single-symbol pin enforcement ───────────────────────────────────────


def test_single_stroke_pin_forces_label(ocr, sample_strokes):
    """Pinning one stroke with a label should produce a symbol with that label."""
    session = ocr.session()
    sids = [session.add_stroke(s) for s in sample_strokes]
    target_sid = sids[0]
    pin = PinnedTree.build(symbols=[PinSymbol("alpha", [session._strokes[target_sid]])])
    expr = ocr.detect(list(session._strokes.values()), pins=[pin])

    owner = next((sym for sym in expr if any(s.id == target_sid for s in sym.strokes)), None)
    assert owner is not None
    assert owner.name == "alpha"
    assert {s.id for s in owner.strokes} == {target_sid}
    assert owner.confidence == 1.0


def test_multistroke_pin_forces_grouping(ocr, sample_strokes):
    """Pinning two strokes as a single symbol forces grouping."""
    session = ocr.session()
    sids = [session.add_stroke(s) for s in sample_strokes]
    a, b = sids[0], sids[1]
    pin = PinnedTree.build(
        symbols=[PinSymbol("=", [session._strokes[a], session._strokes[b]])],
    )
    expr = ocr.detect(list(session._strokes.values()), pins=[pin])

    matching = [s for s in expr if {st.id for st in s.strokes} == {a, b}]
    assert len(matching) == 1
    assert matching[0].name == "="


def test_pin_does_not_affect_other_strokes(ocr, sample_strokes):
    """Strokes outside the pin should still be grouped/classified normally."""
    session = ocr.session()
    sids = [session.add_stroke(s) for s in sample_strokes]

    baseline = ocr.detect(list(session._strokes.values()))
    target_sid = sids[0]
    pin = PinnedTree.build(symbols=[PinSymbol("alpha", [session._strokes[target_sid]])])
    expr = ocr.detect(list(session._strokes.values()), pins=[pin])

    def labels_excluding(e, excluded_sid):
        return sorted(
            s.name for s in e if not any(st.id == excluded_sid for st in s.strokes)
        )

    assert labels_excluding(baseline, target_sid) == labels_excluding(expr, target_sid)


# ── Multi-symbol pin: connectedness enforcement ─────────────────────────


def test_multi_symbol_pin_forms_connected_subtree(ocr, sample_strokes):
    """Two pinned symbols must end up in a connected subtree — possibly via
    a synthetic `expr` parent node if the model wouldn't naturally connect them."""
    session = ocr.session()
    sids = [session.add_stroke(s) for s in sample_strokes]
    a, b = sids[0], sids[1]
    pin = PinnedTree.build(
        symbols=[
            PinSymbol("x", [session._strokes[a]]),
            PinSymbol("2", [session._strokes[b]]),
        ],
    )
    expr = ocr.detect(list(session._strokes.values()), pins=[pin])

    by_stroke_set = {frozenset(s.id for s in sym.strokes): sym for sym in expr}
    a_sym = by_stroke_set[frozenset([a])]
    b_sym = by_stroke_set[frozenset([b])]
    a_id = next(sid for sid, ds in expr.symbols.items() if ds is a_sym)
    b_id = next(sid for sid, ds in expr.symbols.items() if ds is b_sym)

    assert _is_connected_subtree(expr.tree, {a_id, b_id}), (
        f"pinned symbols not connected: a_id={a_id} parent={expr.tree.nodes[a_id].parent_id}, "
        f"b_id={b_id} parent={expr.tree.nodes[b_id].parent_id}"
    )


def test_many_symbol_pin_inserts_expr_node(ocr, sample_strokes):
    """Pinning many top-level symbols (which wouldn't naturally connect) should
    insert an `expr` parent node so the pinned set becomes a connected subtree."""
    session = ocr.session()
    sids = [session.add_stroke(s) for s in sample_strokes]
    pin = PinnedTree.build(
        symbols=[
            PinSymbol("a", [session._strokes[sids[0]]]),
            PinSymbol("b", [session._strokes[sids[1]]]),
            PinSymbol("c", [session._strokes[sids[2]]]),
        ],
    )
    expr = ocr.detect(list(session._strokes.values()), pins=[pin])

    by_stroke_set = {frozenset(s.id for s in sym.strokes): sym for sym in expr}
    pinned_tree_ids = set()
    for sid in sids[:3]:
        ds = by_stroke_set[frozenset([sid])]
        pinned_tree_ids.add(next(tid for tid, dsm in expr.symbols.items() if dsm is ds))

    assert _is_connected_subtree(expr.tree, pinned_tree_ids), (
        "pinned 3-symbol set must form a connected subtree"
    )

    parents = {expr.tree.nodes[tid].parent_id for tid in pinned_tree_ids}
    parents.discard(ROOT_ID)
    for p in parents:
        if p not in pinned_tree_ids and p != ROOT_ID:
            assert p in expr.tree.nodes
