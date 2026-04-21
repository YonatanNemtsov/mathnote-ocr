#!/usr/bin/env python3
"""Generate training data for the tree parser.

Reuses the existing expression samplers and glyph extraction from
parser/gen_data.py, then converts each (LaTeX, glyphs) pair into
dependency tree labels: for each glyph, (parent_idx, edge_type, order).

The conversion walks the LNode tree (from expr_aug.py) in sync with
the glyph list to assign parent pointers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import math
import multiprocessing as mp
import os
import random

from mathnote_ocr.latex_utils.expr_aug import (
    _FUNC_GLYPH_COUNTS,
    LNode,
    _n_char_glyphs,
    _n_frac_bars,
    parse_latex,
)
from mathnote_ocr.latex_utils.glyphs import _extract_glyphs
from mathnote_ocr.latex_utils.sampling import _set_sampler, sample_expression
from mathnote_ocr.tree_parser.tree import (
    DEN,
    LOWER,
    MATCH,
    NUM,
    ROOT,
    SQRT_CONTENT,
    SUB,
    SUP,
    UPPER,
)

# Big operator commands — these render with display-style limits (above/below)
# when unbraced. ziamath renders their glyph order differently: sup before sub.
_BIGOP_CMDS = {"\\sum", "\\int", "\\prod", "\\lim"}


def _is_bigop(node: LNode) -> bool:
    """Check if a node is an unbraced big operator command."""
    return node.kind == "command" and node.text in _BIGOP_CMDS


# ── LNode tree → dependency labels ───────────────────────────────────


def _head_glyph(
    node: LNode,
    char_c: list[int],
    bar_c: list[int],
    n_chars: int,
) -> int:
    """Find the 'head' glyph index for an LNode subtree without advancing counters.

    The head is the glyph that represents this subtree as a whole:
    - char/command: the glyph itself (first glyph for multi-char commands)
    - frac: the fraction bar
    - sqrt: the sqrt radical glyph
    - sup/sub: head of the base (child[0])
    - seq: head of the first child
    - func: first glyph of function name
    - binom: the ( glyph
    """
    if node.kind == "char":
        return char_c[0]

    if node.kind == "command":
        return char_c[0]

    if node.kind == "frac":
        # Bar index = n_chars + current bar counter
        # But we need to count past the char glyphs in children first
        child_chars = sum(_n_char_glyphs(c) for c in node.children)
        child_bars = sum(_n_frac_bars(c) for c in node.children) - 1  # exclude this bar
        # Actually the bar is appended AFTER all children's bars
        # Let's count: DFS order for frac is num chars, den chars, num bars, den bars, THIS bar
        # Wait — the glyph order is: all chars (DFS), then all bars (DFS)
        # So frac bar index = n_chars + bar_c[0] + child bars
        # But we need to know how many bars are in the children
        total_child_bars = sum(_n_frac_bars(c) for c in node.children)
        return n_chars + bar_c[0] + total_child_bars

    if node.kind == "binom":
        # binom head is the ( glyph
        return char_c[0]

    if node.kind == "sqrt":
        # sqrt radical is the first char glyph in this subtree
        return char_c[0]

    if node.kind == "sup" or node.kind == "sub":
        # Head = head of base (child[0])
        if node.children:
            return _head_glyph(node.children[0], char_c, bar_c, n_chars)
        return char_c[0]

    if node.kind == "func":
        if node.children:
            return _head_glyph(node.children[0], char_c, bar_c, n_chars)
        return char_c[0]

    if node.kind == "seq":
        if node.children:
            return _head_glyph(node.children[0], char_c, bar_c, n_chars)
        return char_c[0]

    return char_c[0]


def _tail_glyph(
    node: LNode,
    char_c: list[int],
    bar_c: list[int],
    n_chars: int,
) -> int:
    """Find the 'tail' (last) glyph index for an LNode subtree.

    Used for sup/sub parent assignment — the superscript of \\log should
    attach to 'g' (last glyph), not 'l' (first glyph).
    """
    if node.kind == "char":
        return char_c[0]

    if node.kind == "command":
        n = _FUNC_GLYPH_COUNTS.get(node.text, 1)
        return char_c[0] + n - 1

    if node.kind == "frac":
        # Tail is the bar (same as head)
        total_child_bars = sum(_n_frac_bars(c) for c in node.children)
        return n_chars + bar_c[0] + total_child_bars

    if node.kind == "binom":
        # Tail is the ) glyph — after ( + all children
        return char_c[0] + 1 + sum(_n_char_glyphs(c) for c in node.children)

    if node.kind == "sqrt":
        return char_c[0]

    if node.kind == "sup" or node.kind == "sub":
        if node.children:
            return _tail_glyph(node.children[0], char_c, bar_c, n_chars)
        return char_c[0]

    if node.kind in ("func", "seq"):
        if node.children:
            # Advance counters past all children except the last
            c = list(char_c)
            b = list(bar_c)
            for child in node.children[:-1]:
                c[0] += _n_char_glyphs(child)
                b[0] += _n_frac_bars(child)
            return _tail_glyph(node.children[-1], c, b, n_chars)
        return char_c[0]

    return char_c[0]


def _assign_labels(
    node: LNode,
    parent_idx: int,
    edge_type: int,
    order: int,
    char_c: list[int],
    bar_c: list[int],
    n_chars: int,
    labels: dict[int, tuple[int, int, int]],
) -> None:
    """Recursively assign (parent, edge_type, order) to each glyph.

    Walks the LNode tree in the same DFS order as glyph extraction.

    Args:
        node: Current LNode.
        parent_idx: Glyph index of the parent (-1 for root).
        edge_type: Edge type to parent.
        order: Sibling order among same (parent, edge_type) group.
        char_c: [next_char_glyph_index], mutated.
        bar_c: [next_bar_glyph_index], mutated (offset from n_chars).
        n_chars: Total char glyphs in the expression (bars start at n_chars).
        labels: Output dict: glyph_index → (parent, edge_type, order).
    """
    if node.kind == "char":
        idx = char_c[0]
        char_c[0] += 1
        labels[idx] = (parent_idx, edge_type, order)
        return

    if node.kind == "command":
        n = _FUNC_GLYPH_COUNTS.get(node.text, 1)
        first_idx = char_c[0]
        # First glyph gets the parent assignment
        labels[first_idx] = (parent_idx, edge_type, order)
        char_c[0] += 1
        # Additional glyphs of multi-char commands are siblings
        for k in range(1, n):
            idx = char_c[0]
            char_c[0] += 1
            labels[idx] = (parent_idx, edge_type, order)
        return

    if node.kind == "seq":
        for i, child in enumerate(node.children):
            _assign_labels(child, parent_idx, edge_type, order + i, char_c, bar_c, n_chars, labels)
        return

    if node.kind == "frac":
        # Fraction bar index: after all children's chars and bars
        num_node, den_node = node.children[0], node.children[1]

        # Save counters to compute bar index
        saved_bar = bar_c[0]
        num_child_bars = _n_frac_bars(num_node)
        den_child_bars = _n_frac_bars(den_node)
        bar_idx = n_chars + saved_bar + num_child_bars + den_child_bars

        # Assign bar to the incoming parent
        labels[bar_idx] = (parent_idx, edge_type, order)

        # Process numerator children → parent is bar, edge=NUM
        _assign_labels(num_node, bar_idx, NUM, 0, char_c, bar_c, n_chars, labels)

        # Process denominator children → parent is bar, edge=DEN
        _assign_labels(den_node, bar_idx, DEN, 0, char_c, bar_c, n_chars, labels)

        # Advance bar counter for this frac's bar
        bar_c[0] += 1
        return

    if node.kind == "binom":
        # binom: ( is head, top=NUM, bot=DEN, )=MATCH. No invisible bar.
        # Glyph order: ( glyph, top glyphs, bot glyphs, ) glyph
        top_node, bot_node = node.children[0], node.children[1]

        # ( glyph — head of binom
        lparen_idx = char_c[0]
        char_c[0] += 1
        labels[lparen_idx] = (parent_idx, edge_type, order)

        # top children → parent is (, edge = NUM
        _assign_labels(top_node, lparen_idx, NUM, 0, char_c, bar_c, n_chars, labels)

        # bot children → parent is (, edge = DEN
        _assign_labels(bot_node, lparen_idx, DEN, 0, char_c, bar_c, n_chars, labels)

        # ) glyph → parent is (, edge = MATCH
        rparen_idx = char_c[0]
        char_c[0] += 1
        labels[rparen_idx] = (lparen_idx, MATCH, 0)

        return

    if node.kind == "sqrt":
        # sqrt radical glyph comes first
        sqrt_idx = char_c[0]
        char_c[0] += 1

        # Assign sqrt to incoming parent
        labels[sqrt_idx] = (parent_idx, edge_type, order)

        # Content children → parent is sqrt, edge=SQRT_CONTENT
        if node.children:
            _assign_labels(
                node.children[0], sqrt_idx, SQRT_CONTENT, 0, char_c, bar_c, n_chars, labels
            )
        return

    if node.kind == "sup":
        base, exp = node.children[0], node.children[1]

        # Special case: sup(sub(innerbase, sub_content), sup_content)
        # Both sub and sup on same base.
        # ziamath renders: base, sub, sup (brace-wrapped big ops keep this order).
        if base.kind == "sub" and len(base.children) == 2:
            innerbase = base.children[0]
            sub_content = base.children[1]

            innerbase_tail = _tail_glyph(innerbase, list(char_c), list(bar_c), n_chars)

            # Use LOWER/UPPER for big ops, SUB/SUP for everything else
            if _is_bigop(innerbase):
                lo_et, hi_et = LOWER, UPPER
            else:
                lo_et, hi_et = SUB, SUP

            _assign_labels(innerbase, parent_idx, edge_type, order, char_c, bar_c, n_chars, labels)
            # ziamath renders \sum/\prod limits in different order depending
            # on context: standalone (top-level) = UPPER first, nested = LOWER first.
            # parent_idx == ROOT means the big op is at top level.
            if _is_bigop(innerbase) and parent_idx == ROOT:
                _assign_labels(exp, innerbase_tail, hi_et, 0, char_c, bar_c, n_chars, labels)
                _assign_labels(
                    sub_content, innerbase_tail, lo_et, 0, char_c, bar_c, n_chars, labels
                )
            else:
                _assign_labels(
                    sub_content, innerbase_tail, lo_et, 0, char_c, bar_c, n_chars, labels
                )
                _assign_labels(exp, innerbase_tail, hi_et, 0, char_c, bar_c, n_chars, labels)
            return

        # Normal sup: base then exponent
        base_tail = _tail_glyph(base, list(char_c), list(bar_c), n_chars)

        # Big op with upper limit only
        sup_et = UPPER if _is_bigop(base) else SUP
        _assign_labels(base, parent_idx, edge_type, order, char_c, bar_c, n_chars, labels)
        _assign_labels(exp, base_tail, sup_et, 0, char_c, bar_c, n_chars, labels)
        return

    if node.kind == "sub":
        base, sub = node.children[0], node.children[1]

        # Special case: sub(sup(innerbase, sup_content), sub_content)
        # ziamath renders: base, sub, sup (brace-wrapped big ops keep this order).
        if base.kind == "sup" and len(base.children) == 2:
            innerbase = base.children[0]
            sup_content = base.children[1]

            innerbase_tail = _tail_glyph(innerbase, list(char_c), list(bar_c), n_chars)

            if _is_bigop(innerbase):
                lo_et, hi_et = LOWER, UPPER
            else:
                lo_et, hi_et = SUB, SUP

            _assign_labels(innerbase, parent_idx, edge_type, order, char_c, bar_c, n_chars, labels)
            _assign_labels(sub, innerbase_tail, lo_et, 0, char_c, bar_c, n_chars, labels)
            _assign_labels(sup_content, innerbase_tail, hi_et, 0, char_c, bar_c, n_chars, labels)
            return

        base_tail = _tail_glyph(base, list(char_c), list(bar_c), n_chars)

        # Big op with lower limit only
        sub_et = LOWER if _is_bigop(base) else SUB
        _assign_labels(base, parent_idx, edge_type, order, char_c, bar_c, n_chars, labels)
        _assign_labels(sub, base_tail, sub_et, 0, char_c, bar_c, n_chars, labels)
        return

    if node.kind == "func":
        # func has [func_name_node, arg_node]
        # Both inherit the same parent — function is just symbols in sequence
        _assign_labels(
            node.children[0], parent_idx, edge_type, order, char_c, bar_c, n_chars, labels
        )
        if len(node.children) > 1:
            _assign_labels(
                node.children[1], parent_idx, edge_type, order + 1, char_c, bar_c, n_chars, labels
            )
        return

    # Fallback: treat unknown kinds like seq
    for i, child in enumerate(node.children):
        _assign_labels(child, parent_idx, edge_type, i, char_c, bar_c, n_chars, labels)


def latex_to_tree_labels(
    latex: str,
    n_glyphs: int,
) -> list[tuple[int, int, int]] | None:
    """Convert LaTeX string to dependency tree labels.

    Returns list of (parent_idx, edge_type, order) per glyph,
    or None if parsing fails.
    """
    tree = parse_latex(latex)
    if tree is None:
        return None

    n_chars = _n_char_glyphs(tree)
    n_bars = _n_frac_bars(tree)

    if n_chars + n_bars != n_glyphs:
        return None

    labels: dict[int, tuple[int, int, int]] = {}
    char_c = [0]
    bar_c = [0]

    _assign_labels(tree, ROOT, -1, 0, char_c, bar_c, n_chars, labels)

    if len(labels) != n_glyphs:
        return None

    return [labels[i] for i in range(n_glyphs)]


# ── Data generation ──────────────────────────────────────────────────


def _worker_init(seed: int, sampler_name: str):
    random.seed(seed + os.getpid())
    _set_sampler(sampler_name)


def _fix_upper_lower(glyphs, labels):
    """Fix UPPER/LOWER edge types by checking spatial positions.

    ziamath renders big op limits in different glyph order depending on
    context (standalone \\sum vs nested inside ^{} or \\frac{}{}), so
    _assign_labels can assign UPPER to a glyph that's spatially below
    its parent (or vice versa). This fixes it by swapping UPPER<->LOWER
    when the spatial position contradicts the edge type.
    """
    labels = list(labels)
    for i, (parent, edge, order) in enumerate(labels):
        if edge not in (UPPER, LOWER):
            continue
        if parent < 0 or parent >= len(glyphs):
            continue
        child_cy = glyphs[i]["bbox"][1] + glyphs[i]["bbox"][3] / 2
        parent_cy = glyphs[parent]["bbox"][1] + glyphs[parent]["bbox"][3] / 2
        above = child_cy < parent_cy
        if edge == UPPER and not above:
            labels[i] = (parent, LOWER, order)
        elif edge == LOWER and above:
            labels[i] = (parent, UPPER, order)
    return labels


def _process_batch(args: tuple) -> list[dict]:
    """Generate a batch of (expression, tree labels) pairs."""
    batch_size, max_symbols = args
    results = []

    for _ in range(batch_size):
        latex = sample_expression()

        glyphs = _extract_glyphs(latex)
        if glyphs is None:
            continue

        n_glyphs = len(glyphs)
        if n_glyphs < 2 or n_glyphs > max_symbols:
            continue

        tree_labels = latex_to_tree_labels(latex, n_glyphs)
        if tree_labels is None:
            continue

        # Fix UPPER/LOWER misalignment: ziamath renders \sum/\prod limits
        # in different order depending on context (standalone vs nested).
        # _assign_labels walks the parse tree sequentially which may not
        # match the glyph order. Fix by checking spatial positions.
        tree_labels = _fix_upper_lower(glyphs, tree_labels)

        results.append(
            {
                "latex": latex,
                "symbols": [{"name": g["name"], "bbox": g["bbox"]} for g in glyphs],
                "tree": [{"parent": p, "edge_type": e, "order": o} for p, e, o in tree_labels],
            }
        )

    return results


BATCH_SIZE = 500


def generate_dataset(
    n: int,
    output_path: Path,
    seed: int = 42,
    max_symbols: int = 20,
    sampler_name: str = "v2",
) -> None:
    """Generate n valid tree-labeled examples."""

    num_workers = max(1, os.cpu_count() - 1)

    total = 0
    attempts = 0

    with (
        open(output_path, "w") as f,
        mp.Pool(num_workers, initializer=_worker_init, initargs=(seed, sampler_name)) as pool,
    ):
        while total < n:
            remaining = n - total
            n_batches = max(num_workers, math.ceil(remaining * 3 / BATCH_SIZE))
            tasks = [(BATCH_SIZE, max_symbols)] * n_batches
            attempts += n_batches * BATCH_SIZE

            for results in pool.imap_unordered(_process_batch, tasks):
                for item in results:
                    if total >= n:
                        break
                    f.write(json.dumps(item) + "\n")
                    total += 1

                if total >= n:
                    break

                if total % 5000 < len(results):
                    print(f"  Progress: {total}/{n}")

    print(
        f"\nGenerated {total} examples ({attempts} attempts, "
        f"{attempts / max(total, 1):.1f}x ratio) -> {output_path}"
    )


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate tree parser training data",
    )
    parser.add_argument("--train-n", type=int, default=50000)
    parser.add_argument("--val-n", type=int, default=5000)
    parser.add_argument("--max-symbols", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--templates", type=str, default="dg_all")
    parser.add_argument(
        "--run", type=str, default="v1", help="Output subdirectory under data/shared/tree/"
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data" / "tree" / args.run
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating training data (templates={args.templates}, run={args.run})...")
    generate_dataset(
        args.train_n,
        data_dir / "train.jsonl",
        seed=args.seed,
        max_symbols=args.max_symbols,
        sampler_name=args.templates,
    )

    print("\nGenerating validation data...")
    generate_dataset(
        args.val_n,
        data_dir / "val.jsonl",
        seed=args.seed + 1000,
        max_symbols=args.max_symbols,
        sampler_name=args.templates,
    )
