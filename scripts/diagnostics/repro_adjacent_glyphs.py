"""Compose adjacent handwritten glyphs (math_ocr_web's hand_written_font)
and show how the engine groups them, e.g. 'x' next to 'd'.

Usage:
    python3.10 scripts/diagnostics/repro_adjacent_glyphs.py x d
    python3.10 scripts/diagnostics/repro_adjacent_glyphs.py x d --gaps 0 4 8 --old-gate
"""
import argparse
import json
from pathlib import Path

from mathnote_ocr import MathOCR
from mathnote_ocr.engine import grouper

FONT = Path(__file__).resolve().parents[3] / "math_ocr_web" / "static" / "hand_written_font"


def glyph(name: str, height: float = 40.0) -> list[list[tuple[float, float]]]:
    strokes = json.loads(next((FONT / name).glob("*.json")).read_text())["strokes"]
    xs = [p["x"] for s in strokes for p in s]
    ys = [p["y"] for s in strokes for p in s]
    k = height / (max(ys) - min(ys))
    return [[((p["x"] - min(xs)) * k, (p["y"] - min(ys)) * k) for p in s] for s in strokes]


def compose(names: list[str], gap: float) -> list[list[tuple[float, float]]]:
    out, x0 = [], 100.0
    for n in names:
        g = glyph(n)
        w = max(x for s in g for x, _ in s)
        out += [[(x + x0, y + 100.0) for x, y in s] for s in g]
        x0 += w + gap
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glyphs", nargs="+")
    ap.add_argument("--gaps", nargs="+", type=float, default=[0, 3, 6, 10, 15])
    ap.add_argument("--config", default="../math_ocr_web/configs/mixed_v10_backtrack_gnn.yaml")
    ap.add_argument("--old-gate", action="store_true", help="gate always applies (pre-2026-09-26)")
    args = ap.parse_args()
    if args.old_gate:
        grouper._singletons_can_coexist = lambda *a, **k: True
    ocr = MathOCR(config=args.config)
    for gap in args.gaps:
        strokes = compose(args.glyphs, gap)
        e = ocr.detect(strokes)
        groups = [(s.name, sorted(st.id for st in s.strokes)) for s in e]
        print(f"gap {gap:>4}: {e.latex!r:28} {groups}  unexplained={e.unexplained_stroke_ids}")


if __name__ == "__main__":
    main()
