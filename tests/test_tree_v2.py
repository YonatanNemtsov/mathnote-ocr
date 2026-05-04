"""Structural tests for tree_v2 — pure data structure, no pipeline."""

from mathnote_ocr.bbox import BBox
from mathnote_ocr.tree_parser.tree_v2 import (
    ROOT_ID,
    Edge,
    Node,
    Symbol,
    Tree,
    tree_from_arrays,
)


def _sym(id: int, name: str = "x") -> Symbol:
    return Symbol(id=id, name=name, bbox=BBox(id * 10, 0, 10, 10))


def _node(id: int, parent: int = ROOT_ID, edge: int = Edge.ROOT, order: int = 0) -> Node:
    return Node(symbol=_sym(id), parent_id=parent, edge_type=edge, order=order)


def _xy_tree() -> Tree:
    """Tiny tree: ROOT -> x(0); x has SUP=y(1)."""
    return Tree(
        (
            _node(0, ROOT_ID, Edge.ROOT, 0),
            _node(1, 0, Edge.SUP, 0),
        )
    )


# ── Construction & basic queries ────────────────────────────────────────


def test_empty_tree():
    t = Tree(())
    assert len(t) == 0
    assert not t
    assert t.root_ids() == ()


def test_len_excludes_root():
    t = _xy_tree()
    assert len(t) == 2  # x and y, not the virtual root
    assert bool(t)


def test_contains_and_getitem():
    t = _xy_tree()
    assert 0 in t
    assert 1 in t
    assert 99 not in t
    assert t[0].symbol.name == "x"


def test_root_ids_returns_top_level():
    t = _xy_tree()
    assert t.root_ids() == (0,)


def test_root_ids_multiple():
    t = Tree((_node(0), _node(1, ROOT_ID, Edge.ROOT, 1)))
    assert t.root_ids() == (0, 1)


# ── Children ────────────────────────────────────────────────────────────


def test_children_of_returns_sorted_by_order():
    t = Tree(
        (
            _node(0),
            _node(2, 0, Edge.SUB, 1),
            _node(1, 0, Edge.SUP, 0),
        )
    )
    kids = t.children_of(0)
    assert [c[0] for c in kids] == [1, 2]


def test_children_by_edge_filters():
    t = Tree(
        (
            _node(0),
            _node(1, 0, Edge.SUP, 0),
            _node(2, 0, Edge.SUB, 0),
            _node(3, 0, Edge.SUP, 1),
        )
    )
    assert t.children_by_edge(0, Edge.SUP) == (1, 3)
    assert t.children_by_edge(0, Edge.SUB) == (2,)
    assert t.children_by_edge(0, Edge.NUM) == ()


def test_is_leaf_and_is_root():
    t = _xy_tree()
    assert t.is_root(0)
    assert not t.is_root(1)
    assert not t.is_leaf(0)
    assert t.is_leaf(1)


# ── Mutations ───────────────────────────────────────────────────────────


def test_add_node_returns_new_tree():
    t1 = _xy_tree()
    t2 = t1.add_node(_node(2, 0, Edge.SUB, 0))
    assert len(t1) == 2
    assert len(t2) == 3
    assert 2 not in t1
    assert 2 in t2


def test_remove_node_removes_descendants():
    # ROOT -> 0 -> 1 -> 2; remove 1, expect 2 also gone
    t = Tree(
        (
            _node(0),
            _node(1, 0, Edge.SUP, 0),
            _node(2, 1, Edge.SUP, 0),
        )
    )
    t2 = t.remove_node(1)
    assert 1 not in t2
    assert 2 not in t2
    assert 0 in t2


def test_remove_leaves_siblings():
    t = Tree(
        (
            _node(0),
            _node(1, 0, Edge.SUP, 0),
            _node(2, 0, Edge.SUB, 0),
        )
    )
    t2 = t.remove_node(1)
    assert 2 in t2
    assert 1 not in t2


def test_move_node():
    t = _xy_tree()
    # move y under itself's parent with new edge
    t2 = t.move_node(1, 0, Edge.SUB, 0)
    assert t2[1].edge_type == Edge.SUB
    assert t2[1].parent_id == 0


def test_move_node_carries_descendants():
    # ROOT -> 0; 0 has SUP=1; 1 has SUP=2.  Move 1 to ROOT.  2 stays under 1.
    t = Tree(
        (
            _node(0),
            _node(1, 0, Edge.SUP, 0),
            _node(2, 1, Edge.SUP, 0),
        )
    )
    t2 = t.move_node(1, ROOT_ID, Edge.ROOT, 1)
    assert t2[1].parent_id == ROOT_ID
    assert t2[2].parent_id == 1  # descendant unchanged


def test_rename_node_preserves_structure():
    t = _xy_tree()
    t2 = t.rename_node(0, "y")
    assert t2[0].symbol.name == "y"
    assert t2[0].symbol.bbox == t[0].symbol.bbox  # bbox preserved
    assert t2.children_of(0) == t.children_of(0)  # structure preserved


# ── Traversal ───────────────────────────────────────────────────────────


def test_walk_depth_first_self_first():
    # 0 -> 1, 2 ; 1 -> 3
    t = Tree(
        (
            _node(0),
            _node(1, 0, Edge.SUP, 0),
            _node(3, 1, Edge.SUP, 0),
            _node(2, 0, Edge.SUB, 1),
        )
    )
    assert t.walk(0) == (0, 1, 3, 2)


def test_path_from_root():
    t = Tree(
        (
            _node(0),
            _node(1, 0, Edge.SUP, 0),
            _node(2, 1, Edge.SUB, 0),
        )
    )
    p = t.path(2)
    assert [step[0] for step in p] == [0, 1, 2]
    assert p[1][1] == Edge.SUP
    assert p[2][1] == Edge.SUB


# ── Equality & hash ─────────────────────────────────────────────────────


def test_eq_ignores_construction_order():
    t1 = Tree((_node(0), _node(1, 0, Edge.SUP, 0)))
    t2 = Tree((_node(1, 0, Edge.SUP, 0), _node(0)))
    assert t1 == t2


def test_eq_distinguishes_structure():
    t1 = Tree((_node(0), _node(1, 0, Edge.SUP, 0)))
    t2 = Tree((_node(0), _node(1, 0, Edge.SUB, 0)))
    assert t1 != t2


def test_hash_matches_eq():
    t1 = _xy_tree()
    t2 = Tree((_node(1, 0, Edge.SUP, 0), _node(0)))
    assert hash(t1) == hash(t2)


# ── tree_from_arrays ────────────────────────────────────────────────────


def test_tree_from_arrays():
    t = tree_from_arrays(
        names=["x", "2"],
        bboxes=[[0, 0, 10, 10], [10, 0, 10, 10]],
        parent=[-1, 0],
        edge_type=[Edge.ROOT, Edge.SUP],
        order=[0, 0],
    )
    assert len(t) == 2
    assert t[0].symbol.name == "x"
    assert t[1].symbol.name == "2"
    assert t[1].parent_id == 0
    assert t[1].edge_type == Edge.SUP
