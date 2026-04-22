#!/usr/bin/env python3
"""Utilities for manipulating JSONL datasets (tree_train.jsonl, etc.).

Each line is a JSON object with: {latex, symbols, tree}

Commands:
    info      — print stats (count, symbol count distribution, etc.)
    subsample — randomly pick n examples
    concat    — merge multiple files into one
    split     — split into two files by count or ratio
    filter    — filter by symbol count, symbol names, etc.
    shuffle   — shuffle lines
    head/tail — first/last n examples
    dedupe    — remove duplicate latex strings

Usage:
    python3.10 tools/data_utils.py info      data/tree_train.jsonl
    python3.10 tools/data_utils.py subsample data/tree_train.jsonl 5000 -o data/tree_train_5k.jsonl
    python3.10 tools/data_utils.py concat    data/tree_train.jsonl data/tree_train_hard.jsonl -o data/combined.jsonl
    python3.10 tools/data_utils.py split     data/tree_train.jsonl 12000 -o data/split_a.jsonl data/split_b.jsonl
    python3.10 tools/data_utils.py filter    data/tree_train.jsonl --min-n 5 --max-n 15 -o data/filtered.jsonl
    python3.10 tools/data_utils.py filter    data/tree_train.jsonl --has-symbol int -o data/with_int.jsonl
    python3.10 tools/data_utils.py shuffle   data/tree_train.jsonl -o data/shuffled.jsonl
    python3.10 tools/data_utils.py head      data/tree_train.jsonl 100 -o data/first_100.jsonl
    python3.10 tools/data_utils.py dedupe    data/tree_train.jsonl -o data/deduped.jsonl
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} examples to {path}")


def cmd_info(args):
    rows = _read_jsonl(args.src)
    sizes = [len(r["symbols"]) for r in rows]
    print(f"File: {args.src}")
    print(f"  Examples: {len(rows)}")
    if not rows:
        return
    print(f"  N range:  {min(sizes)}-{max(sizes)}")
    print(f"  N mean:   {sum(sizes)/len(sizes):.1f}")

    # Symbol frequency
    sym_counts = Counter()
    for r in rows:
        for s in r["symbols"]:
            sym_counts[s["name"]] += 1
    print(f"  Unique symbols: {len(sym_counts)}")
    print(f"  Top 15: {', '.join(f'{s}({c})' for s, c in sym_counts.most_common(15))}")

    # Edge type distribution
    edge_counts = Counter()
    for r in rows:
        for t in r["tree"]:
            edge_counts[t["edge_type"]] += 1
    print(f"  Edge types: {dict(sorted(edge_counts.items()))}")

    # Size distribution
    buckets = Counter()
    for n in sizes:
        b = (n // 5) * 5
        buckets[f"{b:>2}-{b+4:<2}"] += 1
    print(f"  Size distribution:")
    for label in sorted(buckets, key=lambda x: int(x.strip().split("-")[0])):
        count = buckets[label]
        bar = "#" * (count * 50 // len(rows))
        print(f"    N={label}: {count:>6}  {bar}")


def cmd_subsample(args):
    rows = _read_jsonl(args.src)
    n = min(args.n, len(rows))
    sampled = random.sample(rows, n)
    _write_jsonl(sampled, args.out[0])


def cmd_concat(args):
    all_rows = []
    for src in args.sources:
        rows = _read_jsonl(src)
        print(f"  {src}: {len(rows)} examples")
        all_rows.extend(rows)
    random.shuffle(all_rows)
    _write_jsonl(all_rows, args.out[0])


def cmd_split(args):
    rows = _read_jsonl(args.src)
    random.shuffle(rows)
    n = args.n
    if n >= len(rows):
        print(f"Error: split size {n} >= dataset size {len(rows)}")
        sys.exit(1)
    if len(args.out) < 2:
        print("Error: split requires -o file_a file_b")
        sys.exit(1)
    _write_jsonl(rows[:n], args.out[0])
    _write_jsonl(rows[n:], args.out[1])


def cmd_filter(args):
    rows = _read_jsonl(args.src)
    filtered = rows

    if args.min_n is not None:
        filtered = [r for r in filtered if len(r["symbols"]) >= args.min_n]
    if args.max_n is not None:
        filtered = [r for r in filtered if len(r["symbols"]) <= args.max_n]
    if args.has_symbol:
        filtered = [
            r for r in filtered
            if any(s["name"] == args.has_symbol for s in r["symbols"])
        ]
    if args.has_latex:
        filtered = [r for r in filtered if args.has_latex in r.get("latex", "")]

    print(f"  {len(rows)} -> {len(filtered)} examples")
    _write_jsonl(filtered, args.out[0])


def cmd_shuffle(args):
    rows = _read_jsonl(args.src)
    random.shuffle(rows)
    _write_jsonl(rows, args.out[0])


def cmd_head(args):
    rows = _read_jsonl(args.src)
    _write_jsonl(rows[:args.n], args.out[0])


def cmd_tail(args):
    rows = _read_jsonl(args.src)
    _write_jsonl(rows[-args.n:], args.out[0])


def cmd_dedupe(args):
    rows = _read_jsonl(args.src)
    seen = set()
    deduped = []
    for r in rows:
        key = r.get("latex", json.dumps(r["symbols"]))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    print(f"  {len(rows)} -> {len(deduped)} ({len(rows) - len(deduped)} duplicates)")
    _write_jsonl(deduped, args.out[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSONL dataset utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info")
    p.add_argument("src")

    p = sub.add_parser("subsample")
    p.add_argument("src")
    p.add_argument("n", type=int)
    p.add_argument("-o", "--out", nargs="+", required=True)

    p = sub.add_parser("concat")
    p.add_argument("sources", nargs="+")
    p.add_argument("-o", "--out", nargs="+", required=True)

    p = sub.add_parser("split")
    p.add_argument("src")
    p.add_argument("n", type=int)
    p.add_argument("-o", "--out", nargs="+", required=True)

    p = sub.add_parser("filter")
    p.add_argument("src")
    p.add_argument("-o", "--out", nargs="+", required=True)
    p.add_argument("--min-n", type=int)
    p.add_argument("--max-n", type=int)
    p.add_argument("--has-symbol", type=str)
    p.add_argument("--has-latex", type=str)

    p = sub.add_parser("shuffle")
    p.add_argument("src")
    p.add_argument("-o", "--out", nargs="+", required=True)

    p = sub.add_parser("head")
    p.add_argument("src")
    p.add_argument("n", type=int)
    p.add_argument("-o", "--out", nargs="+", required=True)

    p = sub.add_parser("tail")
    p.add_argument("src")
    p.add_argument("n", type=int)
    p.add_argument("-o", "--out", nargs="+", required=True)

    p = sub.add_parser("dedupe")
    p.add_argument("src")
    p.add_argument("-o", "--out", nargs="+", required=True)

    args = parser.parse_args()
    {
        "info": cmd_info,
        "subsample": cmd_subsample,
        "concat": cmd_concat,
        "split": cmd_split,
        "filter": cmd_filter,
        "shuffle": cmd_shuffle,
        "head": cmd_head,
        "tail": cmd_tail,
        "dedupe": cmd_dedupe,
    }[args.cmd](args)
