"""The grouper's exact search (search: exact) finds exactly the readings a
brute-force enumeration ranks best, its group probabilities are
probabilities (each stroke's groups sum to 1 over the readings), and its
best reading is never worse than the first-found search's."""

import itertools
import random

from mathnote_ocr.bbox import BBox
from mathnote_ocr.engine.grouper import _find_best_partitions, _find_partitions_exact
from mathnote_ocr.expression import DetectedSymbol


def make_groups(rng: random.Random, n: int):
    """Singletons plus random small groups of neighbouring strokes, each far
    from the others (no clashes), with random confidences."""
    groups = set(frozenset([i]) for i in range(n))
    for _ in range(n * 2):
        start = rng.randrange(n)
        size = rng.randint(2, 3)
        groups.add(frozenset(range(start, min(n, start + size))))
    out = []
    for k, g in enumerate(sorted(groups, key=sorted)):
        sym = DetectedSymbol(name=f"s{k}", bbox=BBox(100.0 * k, 0.0, 10.0, 10.0), strokes=[])
        out.append((g, round(rng.uniform(0.05, 0.99), 3), sym))
    return out


def brute_force(n, groups):
    covers = []

    def rec(left, chosen, p):
        if not left:
            covers.append((p, chosen))
            return
        s = min(left)
        for g, c, sym in groups:
            if s in g and g <= left:
                rec(left - g, chosen + [sym], p * c)
    rec(frozenset(range(n)), [], 1.0)
    return sorted(covers, key=lambda t: -(t[0] ** (1.0 / len(t[1]))))


def test_matches_brute_force_and_probabilities_add_up():
    rng = random.Random(0)
    for _ in range(40):
        n = rng.randint(3, 9)
        groups = make_groups(rng, n)
        want = brute_force(n, groups)
        got = _find_partitions_exact(n, groups, top_k=5, keep=100)
        gm = lambda p, syms: p ** (1.0 / len(syms))
        assert [round(gm(p, s), 9) for p, s in got] == [round(gm(p, s), 9) for p, s in want[:5]]
        # each stroke: the probabilities of the groups holding it sum to 1
        z = sum(p for p, _s in want)
        for stroke in range(n):
            share = sum(p for p, syms in want for sym in syms
                        if stroke in next(g for g, _c, s in groups if s.name == sym.name)) / z
            assert abs(share - 1.0) < 1e-9
        # and the probability each returned symbol carries is its share of all readings
        for _p, syms in got:
            for sym in syms:
                g = next(g for g, _c, s in groups if s.name == sym.name)
                exact = sum(p for p, ss in want if any(x.name == sym.name for x in ss)) / z
                assert abs(sym.grouping_prob - round(exact, 4)) < 1e-4


def test_never_worse_than_first_found():
    rng = random.Random(1)
    for _ in range(40):
        n = rng.randint(6, 14)
        groups = make_groups(rng, n)
        old = _find_best_partitions(n, groups, top_k=1)
        new = _find_partitions_exact(n, groups, top_k=1)
        gm = lambda r: r[0][0] ** (1.0 / len(r[0][1]))
        assert gm(new) >= gm(old) - 1e-12
