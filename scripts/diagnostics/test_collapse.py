"""Quick test for random_collapse order renumbering.

Runs collapse many times on each expression, checks that orders are
always contiguous and seq_prev targets are always findable.
"""

import sys
from collections import defaultdict

sys.path.insert(0, ".")

from mathnote_ocr.latex_utils.collapse import EXPR_NAME, random_collapse


def make_expr(names, tree_dicts):
    symbols = [{"name": n, "bbox": [i * 10, 0, 8, 8]} for i, n in enumerate(names)]
    return symbols, tree_dicts


def check_contiguous_orders(tree):
    groups = defaultdict(list)
    for i, t in enumerate(tree):
        groups[(t["parent"], t["edge_type"])].append(t["order"])
    for key, orders in groups.items():
        orders_sorted = sorted(orders)
        expected = list(range(len(orders_sorted)))
        if orders_sorted != expected:
            return False, f"group {key} has orders {orders_sorted}, expected {expected}"
    return True, ""


def check_seq_targets(symbols, tree):
    groups = defaultdict(dict)
    for i, t in enumerate(tree):
        groups[(t["parent"], t["edge_type"])][t["order"]] = i

    for key, order_map in groups.items():
        for order, idx in order_map.items():
            if order == 0:
                continue
            if order - 1 not in order_map:
                return False, (
                    f"{symbols[idx]['name']} (order={order}) "
                    f"can't find prev sibling (order {order - 1}), "
                    f"available: {sorted(order_map.keys())}"
                )
    return True, ""


def run_stress(name, names, tree, n_iters=1000):
    """Run collapse n_iters times, check every result."""
    print(f"Test: {name}  ({n_iters} iterations)")
    symbols, tree = make_expr(names, tree)
    n_collapsed = 0
    for _ in range(n_iters):
        s2, t2 = random_collapse(list(symbols), [dict(t) for t in tree], collapse_prob=1.0)
        has_expr = any(s["name"] == EXPR_NAME for s in s2)
        if not has_expr:
            continue
        n_collapsed += 1

        ok1, msg1 = check_contiguous_orders(t2)
        if not ok1:
            orig = [s["name"] for s in symbols]
            result = [s["name"] for s in s2]
            removed = set(orig) - set(result)
            print(f"  FAIL (contiguous): {msg1}")
            print(f"    Original: {orig}")
            print(f"    Result:   {result}  orders={[t['order'] for t in t2]}")
            return False

        ok2, msg2 = check_seq_targets(s2, t2)
        if not ok2:
            print(f"  FAIL (seq_prev): {msg2}")
            print(f"    Result: {[s['name'] for s in s2]}  orders={[t['order'] for t in t2]}")
            return False

    print(f"  PASS  ({n_collapsed}/{n_iters} had collapses)")
    return True


def main():
    all_ok = True

    # e^{abcd}: 4 siblings under superscript
    all_ok &= run_stress(
        "e^{abcd}",
        ["e", "a", "b", "c", "d"],
        [
            {"parent": -1, "edge_type": 0, "order": 0},
            {"parent": 0, "edge_type": 1, "order": 0},
            {"parent": 0, "edge_type": 1, "order": 1},
            {"parent": 0, "edge_type": 1, "order": 2},
            {"parent": 0, "edge_type": 1, "order": 3},
        ],
    )

    # frac{a+b}{c}: siblings in numerator
    all_ok &= run_stress(
        "frac{a+b}{c}",
        ["frac", "a", "+", "b", "c"],
        [
            {"parent": -1, "edge_type": 0, "order": 0},
            {"parent": 0, "edge_type": 1, "order": 0},
            {"parent": 0, "edge_type": 1, "order": 1},
            {"parent": 0, "edge_type": 1, "order": 2},
            {"parent": 0, "edge_type": 2, "order": 0},
        ],
    )

    # 5 root siblings
    all_ok &= run_stress(
        "a+b+c+d+e (5 root sibs)",
        ["a", "+", "b", "+", "c"],
        [{"parent": -1, "edge_type": 0, "order": i} for i in range(5)],
    )

    # 3 root siblings (can fully collapse)
    all_ok &= run_stress(
        "abc (3 root sibs)",
        ["a", "b", "c"],
        [{"parent": -1, "edge_type": 0, "order": i} for i in range(3)],
    )

    # Nested: x + frac{a}{b} with 2 root siblings
    all_ok &= run_stress(
        "x + frac{a}{b}",
        ["x", "+", "frac", "a", "b"],
        [
            {"parent": -1, "edge_type": 0, "order": 0},
            {"parent": -1, "edge_type": 0, "order": 1},
            {"parent": -1, "edge_type": 0, "order": 2},
            {"parent": 2, "edge_type": 1, "order": 0},
            {"parent": 2, "edge_type": 2, "order": 0},
        ],
    )

    print(f"\n{'ALL PASSED' if all_ok else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
