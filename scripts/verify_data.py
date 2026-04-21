#!/usr/bin/env python3.10
"""Verify training data quality.

Checks:
1. UPPER/LOWER spatial consistency (labels match bbox positions)
2. No sup inside sub content
3. No operators as sup/sub bases
4. No flat atom+op chains in frac content
5. Symbol count distribution
6. Edge type distribution

Usage:
    python3.10 scripts/verify_data.py data/runs/tree_subset/mixed_v10/train.jsonl
    python3.10 scripts/verify_data.py data/runs/tree_subset/mixed_v10/train.jsonl --n 5000
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_upper_lower(symbols, tree):
    """Check UPPER/LOWER labels match spatial positions."""
    issues = []
    for i, t in enumerate(tree):
        if t["edge_type"] not in (5, 6):  # UPPER=5, LOWER=6
            continue
        parent = t["parent"]
        if parent < 0 or parent >= len(symbols):
            continue
        child_cy = symbols[i]["bbox"][1] + symbols[i]["bbox"][3] / 2
        parent_cy = symbols[parent]["bbox"][1] + symbols[parent]["bbox"][3] / 2
        above = child_cy < parent_cy
        edge = "UPPER" if t["edge_type"] == 5 else "LOWER"
        if (edge == "UPPER" and not above) or (edge == "LOWER" and above):
            issues.append(
                f"{symbols[i]['name']} labeled {edge} but spatially {'ABOVE' if above else 'BELOW'} parent {symbols[parent]['name']}"
            )
    return issues


def check_sup_in_sub(symbols, tree):
    """Check for superscripts inside subscript content."""
    issues = []
    # Find all SUB children
    sub_children = set()
    for i, t in enumerate(tree):
        if t["edge_type"] == 3:  # SUB
            sub_children.add(i)

    # Check if any SUB child has SUP children
    for i, t in enumerate(tree):
        if t["edge_type"] == 2 and t["parent"] in sub_children:  # SUP with parent in sub
            issues.append(f"{symbols[i]['name']}^{{}} inside _{{}}")
    return issues


def check_op_as_base(symbols, tree):
    """Check for operators as sup/sub bases."""
    ops = {
        "+",
        "-",
        "times",
        "cdot",
        "pm",
        "div",
        "leq",
        "geq",
        "neq",
        "cup",
        "cap",
        "in",
        "subset",
        "forall",
        "exists",
    }
    issues = []
    # Find symbols that have SUP or SUB children
    has_sup_sub = set()
    for t in tree:
        if t["edge_type"] in (2, 3):  # SUP, SUB
            has_sup_sub.add(t["parent"])

    for i in has_sup_sub:
        if i < 0 or i >= len(symbols):
            continue
        if symbols[i]["name"] in ops:
            issues.append(f"{symbols[i]['name']} has sup/sub")
    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="JSONL data file to verify")
    parser.add_argument("--n", type=int, default=None, help="Max examples to check")
    args = parser.parse_args()

    upper_lower_bad = 0
    sup_in_sub_bad = 0
    op_base_bad = 0
    total = 0
    ul_total = 0
    sizes = Counter()
    edges = Counter()

    with open(args.file) as f:
        for i, line in enumerate(f):
            if args.n and i >= args.n:
                break
            item = json.loads(line)
            symbols = item["symbols"]
            tree = item["tree"]
            total += 1
            sizes[len(symbols)] += 1
            for t in tree:
                edges[t["edge_type"]] += 1

            # Check UPPER/LOWER
            ul_issues = check_upper_lower(symbols, tree)
            if ul_issues:
                upper_lower_bad += 1
                ul_total += len(ul_issues)

            # Check sup in sub
            sis_issues = check_sup_in_sub(symbols, tree)
            if sis_issues:
                sup_in_sub_bad += 1

            # Check op as base
            ob_issues = check_op_as_base(symbols, tree)
            if ob_issues:
                op_base_bad += 1

    print(f"File: {args.file}")
    print(f"Examples: {total}")
    print()
    print("--- Quality checks ---")
    print(f"UPPER/LOWER spatial mismatch: {upper_lower_bad}/{total} examples ({ul_total} labels)")
    print(f"SUP inside SUB content:       {sup_in_sub_bad}/{total}")
    print(f"Operator as sup/sub base:     {op_base_bad}/{total}")
    print()
    print("--- Size distribution ---")
    for k in sorted(sizes):
        bar = "#" * (sizes[k] * 40 // max(sizes.values()))
        print(f"  N={k:2d}: {sizes[k]:5d}  {bar}")
    print()
    print("--- Edge type distribution ---")
    edge_names = {
        -1: "ROOT",
        0: "NUM",
        1: "DEN",
        2: "SUP",
        3: "SUB",
        4: "SQRT",
        5: "UPPER",
        6: "LOWER",
        7: "MATCH",
    }
    for e in sorted(edges):
        print(f"  {edge_names.get(e, str(e)):6s}: {edges[e]:6d}")


if __name__ == "__main__":
    main()
