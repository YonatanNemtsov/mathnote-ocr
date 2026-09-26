"""How well does structures.grid split messy matrices / cases into cells?

Synthetic samples (structures.synth: ziamath layout + handwriting-like
messiness, ground-truth cell per symbol) at several messiness levels.

Metrics per level:
  exact      every symbol in its cell and the right environment
  shape      right number of rows and columns
  env        right environment (pmatrix / bmatrix / vmatrix / cases)
  symbols    content symbols placed in the right cell
  top3       the right grid is the pick or one of the next two (what the
             app's alternatives cycling would reach in <= 2 taps)
  oracle     the right grid is among the candidates at all (generator
             coverage — separates "never proposed" from "scored wrong")

Usage:
    python3.10 scripts/diagnostics/eval_grid.py
    python3.10 scripts/diagnostics/eval_grid.py --n 500 --amounts 0.5 1 1.5 2 --failures 5
"""

import argparse
from collections import Counter

from mathnote_ocr import MathOCR
from mathnote_ocr.structures.grid import split_grid
from mathnote_ocr.structures.synth import generate


def cell_map(grid) -> dict:
    return {i: (r, c) for r, row in enumerate(grid.cells) for c, cell in enumerate(row) for i in cell}


def is_right(sample, grid) -> bool:
    predicted = cell_map(grid)
    return all(predicted.get(i) == s.cell for i, s in enumerate(sample.symbols) if s.cell is not None)


def score(sample, grid) -> dict:
    predicted = cell_map(grid)
    content = [(i, s) for i, s in enumerate(sample.symbols) if s.cell is not None]
    right_cells = sum(predicted.get(i) == s.cell for i, s in content)
    env_ok = grid.env == sample.kind
    shape_ok = grid.shape == sample.shape
    return {
        "exact": env_ok and right_cells == len(content),
        "shape": shape_ok,
        "env": env_ok,
        "symbols": (right_cells, len(content)),
        "why": None if env_ok and right_cells == len(content) else (
            "env" if not env_ok else
            "rows" if grid.shape[0] != sample.shape[0] else
            "cols" if grid.shape[1] != sample.shape[1] else "cell membership"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="samples per messiness level")
    ap.add_argument("--amounts", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--failures", type=int, default=0, help="print this many failures per level")
    args = ap.parse_args()

    names = MathOCR().classifier.label_names
    print(f"{'messiness':>9} {'exact':>7} {'top3':>7} {'oracle':>7} {'shape':>7} {'env':>7} {'symbols':>8} {'cands':>6}   failure kinds")
    for amount in args.amounts:
        samples = generate(args.n, names, seed=args.seed, amount=amount)
        n = len(samples)
        tot = Counter()
        why = Counter()
        sym_ok = sym_all = 0
        shown = 0
        n_cands = []
        for s in samples:
            g = split_grid(s.symbols, kind="cases" if s.kind == "cases" else "matrix", n_alternatives=None)
            r = score(s, g)
            ranked = [g] + g.alternatives
            tot["top3"] += any(is_right(s, x) for x in ranked[:3])
            tot["oracle"] += any(is_right(s, x) for x in ranked)
            n_cands.append(g.n_candidates)
            tot["exact"] += r["exact"]
            tot["shape"] += r["shape"]
            tot["env"] += r["env"]
            sym_ok += r["symbols"][0]
            sym_all += r["symbols"][1]
            if r["why"]:
                why[(s.kind, r["why"])] += 1
                if shown < args.failures:
                    shown += 1
                    print(f"    {s.kind} {s.shape} -> {g.env} {g.shape} ({r['why']}): {s.latex[:80]}")
        pct = lambda k: f"{100 * tot[k] / n:6.1f}%"
        print(f"{amount:>9} {pct('exact')} {pct('top3')} {pct('oracle')} {pct('shape')} {pct('env')} "
              f"{100 * sym_ok / sym_all:7.1f}% {sorted(n_cands)[len(n_cands) // 2]:>6}   {dict(why.most_common(5))}")


if __name__ == "__main__":
    main()
