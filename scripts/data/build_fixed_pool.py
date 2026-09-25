"""Build symbols_from_expr_fixed: crops re-binned by their internal 'label' field.

The internal field is the collection-time record — authoritative over both
directory placement and after-the-fact review. A relabel journal can be
applied on top with --journal, but is OFF by default.

The source pool is not touched. Copies are renumbered per class, their
internal 'label' is set to the final class, and 'src' records provenance.
Regenerable: this script fully determines the output.

Usage:
    python3.10 scripts/data/build_fixed_pool.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="data/shared/symbols_from_expr")
    ap.add_argument("--out", default="data/shared/symbols_from_expr_fixed")
    ap.add_argument("--journal", default=None,
                    help="optional relabel journal; by default ONLY internal labels are used "
                    "(the internal field is the collection-time record)")
    args = ap.parse_args()

    pool, out = Path(args.pool), Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists — remove it first (regenerable)")

    journal: dict[tuple[str, str], dict] = {}
    if args.journal:
        jpath = Path(args.journal)
        if jpath.exists():
            for line in jpath.read_text().splitlines():
                if line.strip():
                    v = json.loads(line)
                    journal[(v["label"], v["file"])] = v

    seq: Counter[str] = Counter()
    moved: Counter[tuple[str, str]] = Counter()
    n_total = n_human = n_garbage = 0

    for class_dir in sorted(p for p in pool.iterdir() if p.is_dir()):
        for p in sorted(class_dir.glob("*.json")):
            if not p.stem.isdigit():
                continue
            n_total += 1
            data = json.loads(p.read_text())
            dir_label = class_dir.name

            v = journal.get((dir_label, p.name))
            if v is not None:
                n_human += 1
                if v["action"] == "garbage":
                    n_garbage += 1
                    continue
                final = v["new_label"] if v["action"] == "relabel" else dir_label
            else:
                final = data.get("label") or dir_label

            if final != dir_label:
                moved[(dir_label, final)] += 1

            seq[final] += 1
            stem = f"{seq[final]:04d}"
            dst = out / final
            dst.mkdir(parents=True, exist_ok=True)
            data["label"] = final
            data["src"] = str(p.relative_to(pool.parent))
            (dst / f"{stem}.json").write_text(json.dumps(data))
            png = p.with_suffix(".png")
            if png.exists():
                shutil.copy(png, dst / f"{stem}.png")

    (out / "README.md").write_text(
        "Generated pool — do not hand-edit. Built by scripts/data/build_fixed_pool.py\n"
        f"from {pool}, binned by each crop's internal 'label' field (the\n"
        "collection-time record). Regenerate: delete this directory and re-run.\n"
    )

    print(f"{n_total} crops -> {out}  ({n_human} human verdicts applied, "
          f"{n_garbage} garbage excluded)")
    print("\nre-binned (old dir -> new dir):")
    for (old, new), c in moved.most_common():
        print(f"  {old:10s} -> {new:10s} x{c}")


if __name__ == "__main__":
    main()
