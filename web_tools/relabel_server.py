"""WebSocket server for eyes-on crop review and relabeling.

Serves crops from a pool directory (per-class stroke-JSON subdirs) to
relabel.html, which renders them and sends back verdicts. Verdicts are
APPENDED to a patch journal — pool files are never modified (data/shared
is immutable and expr_mapping references paths). Re-reviewing a crop
appends a new line; the last verdict wins on replay. Consumers apply the
journal at load time.

Usage:
    python3.10 web_tools/relabel_server.py --pool data/shared/symbols_from_expr \
        --report tmp/faithfulness_report.json
    # then open web_tools/relabel.html in a browser
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import websockets

PORT = 8772


class Store:
    def __init__(self, pool: Path, report: Path | None, journal: Path) -> None:
        self.pool = pool
        self.journal = journal
        # (label, file) -> suggested shape label, from the audit report
        self.suggestions: dict[tuple[str, str], str] = {}
        if report and report.exists():
            for label, r in json.loads(report.read_text()).items():
                for entry in r.get("files", []):
                    if isinstance(entry, list) and len(entry) == 2:
                        self.suggestions[(label, entry[0])] = entry[1]

        # (label, file) -> last verdict dict
        self.verdicts: dict[tuple[str, str], dict] = {}
        if journal.exists():
            for line in journal.read_text().splitlines():
                if line.strip():
                    v = json.loads(line)
                    self.verdicts[(v["label"], v["file"])] = v

    def class_files(self, label: str) -> list[Path]:
        return sorted(
            p for p in (self.pool / label).glob("*.json") if p.stem.isdigit()
        )

    def classes(self) -> list[dict]:
        out = []
        for d in sorted(p for p in self.pool.iterdir() if p.is_dir()):
            files = self.class_files(d.name)
            flagged = sum(1 for p in files if (d.name, p.name) in self.suggestions)
            reviewed = sum(1 for p in files if (d.name, p.name) in self.verdicts)
            out.append(
                {"label": d.name, "n": len(files), "flagged": flagged, "reviewed": reviewed}
            )
        return out

    def crops(self, label: str, only_flagged: bool) -> list[dict]:
        out = []
        for p in self.class_files(label):
            key = (label, p.name)
            if only_flagged and key not in self.suggestions:
                continue
            data = json.loads(p.read_text())
            out.append(
                {
                    "file": p.name,
                    "strokes": data["strokes"],
                    "suggestion": self.suggestions.get(key),
                    "verdict": self.verdicts.get(key),
                }
            )
        return out

    def record(self, items: list[dict]) -> int:
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with self.journal.open("a") as f:
            for it in items:
                v = {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "pool": str(self.pool),
                    "label": it["label"],
                    "file": it["file"],
                    "action": it["action"],  # relabel | keep | garbage
                    "new_label": it.get("new_label"),
                }
                f.write(json.dumps(v) + "\n")
                self.verdicts[(v["label"], v["file"])] = v
                n += 1
        return n


async def handler(ws, store: Store) -> None:
    async for raw in ws:
        msg = json.loads(raw)
        t = msg.get("type")
        if t == "init":
            await ws.send(json.dumps(
                {"type": "init", "pool": str(store.pool), "classes": store.classes()}
            ))
        elif t == "get_class":
            await ws.send(json.dumps(
                {
                    "type": "crops",
                    "label": msg["label"],
                    "crops": store.crops(msg["label"], msg.get("only_flagged", True)),
                }
            ))
        elif t == "verdicts":
            n = store.record(msg["items"])
            await ws.send(json.dumps(
                {"type": "ack", "n": n, "classes": store.classes()}
            ))


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="data/shared/symbols_from_expr")
    ap.add_argument("--report", default="tmp/faithfulness_report.json",
                    help="audit report for suggestions (see audit_faithfulness.py)")
    ap.add_argument("--journal", default=None,
                    help="patch journal path (default: data/shared/patches/<pool>_relabels.jsonl)")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    pool = Path(args.pool)
    journal = Path(args.journal) if args.journal else (
        Path("data/shared/patches") / f"{pool.name}_relabels.jsonl"
    )
    store = Store(pool, Path(args.report) if args.report else None, journal)
    print(f"pool: {pool}  ({len(store.suggestions)} flagged, "
          f"{len(store.verdicts)} already reviewed)")
    print(f"journal: {journal}")
    print(f"ws://localhost:{args.port} — open web_tools/relabel.html")

    async with websockets.serve(lambda ws: handler(ws, store), "localhost", args.port):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
