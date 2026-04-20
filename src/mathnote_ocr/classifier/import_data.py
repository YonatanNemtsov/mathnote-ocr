"""Import symbols from MathWriting-2024 dataset into our training format.

Parses InkML files from mathwriting-2024/symbols/, converts strokes to our
JSON format, renders 128x128 PNGs, and saves into data/shared/symbols/{class_name}/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
import xml.etree.ElementTree as ET

from mathnote_ocr.engine.renderer import render_strokes
from mathnote_ocr.engine.stroke import Stroke, StrokePoint
from mathnote_ocr import config

# ── Label mapping: mathwriting label → our internal class name ────────

LABEL_MAP = {
    # Basic characters
    **{c: c for c in "abcdefghijklmnopqrstuvwxyz"},
    **{c: f"{c}_cap" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
    **{c: c for c in "0123456789"},
    "+": "+", "-": "-", "=": "=",
    "(": "(", ")": ")",
    "[": "[", "]": "]",
    "!": "!", ",": ",", ";": ";", ":": ":", "|": "|",
    ".": "dot", "*": "asterisk", "/": "slash",
    "<": "lt", ">": "gt", "?": "question",

    # Greek lowercase
    "\\alpha": "alpha", "\\beta": "beta", "\\gamma": "gamma",
    "\\delta": "delta", "\\epsilon": "epsilon", "\\zeta": "zeta",
    "\\eta": "eta", "\\theta": "theta", "\\iota": "iota",
    "\\kappa": "kappa", "\\lambda": "lambda", "\\mu": "mu",
    "\\nu": "nu", "\\xi": "xi", "\\pi": "pi",
    "\\rho": "rho", "\\sigma": "sigma", "\\tau": "tau",
    "\\upsilon": "upsilon", "\\phi": "phi", "\\chi": "chi",
    "\\psi": "psi", "\\omega": "omega",
    "\\varphi": "varphi", "\\vartheta": "vartheta",
    "\\varpi": "varpi", "\\varsigma": "varsigma",

    # Greek uppercase
    "\\Gamma": "Gamma_cap", "\\Delta": "Delta_cap", "\\Lambda": "Lambda_up",
    "\\Sigma": "Sigma_up", "\\Pi": "Pi_up", "\\Phi": "Phi_up",
    "\\Psi": "Psi_up", "\\Omega": "Omega_up", "\\Theta": "Theta_up",
    "\\Upsilon": "Upsilon_up", "\\Xi": "Xi_up",

    # Big operators
    "\\sum": "sum", "\\prod": "prod", "\\int": "int",
    "\\oint": "oint", "\\iint": "iint",

    # Binary operators
    "\\cdot": "dot", "\\times": "times", "\\pm": "pm", "\\mp": "mp",
    "\\div": "div", "\\circ": "circ", "\\bullet": "bullet",
    "\\oplus": "oplus", "\\otimes": "otimes", "\\odot": "odot",
    "\\ominus": "ominus", "\\dagger": "dagger",

    # Relations
    "\\le": "leq", "\\ge": "geq", "\\ne": "neq",
    "\\approx": "approx", "\\equiv": "equiv", "\\sim": "sim",
    "\\simeq": "simeq", "\\cong": "cong", "\\propto": "propto",
    "\\ll": "ll", "\\gg": "gg",
    "\\subseteq": "subseteq", "\\subsetneq": "subsetneq",
    "\\supset": "supset", "\\supseteq": "supseteq",
    "\\triangleq": "triangleq",

    # Set / logic
    "\\in": "in", "\\notin": "notin", "\\ni": "ni",
    "\\subset": "subset", "\\cup": "cup", "\\cap": "cap",
    "\\emptyset": "emptyset", "\\forall": "forall", "\\exists": "exists",
    "\\wedge": "wedge", "\\vee": "vee", "\\neg": "neg",
    "\\vdash": "vdash", "\\Vdash": "Vdash", "\\models": "models",
    "\\top": "top", "\\perp": "perp",

    # Big set operators
    "\\bigcap": "bigcap", "\\bigcup": "bigcup",
    "\\bigoplus": "bigoplus", "\\bigvee": "bigvee", "\\bigwedge": "bigwedge",

    # Calculus / special
    "\\partial": "partial", "\\nabla": "nabla", "\\infty": "infty",
    "\\sqrt": "sqrt", "\\prime": "prime", "\\angle": "angle",
    "\\aleph": "aleph", "\\hbar": "hbar",

    # Arrows
    "\\rightarrow": "rightarrow", "\\leftarrow": "leftarrow",
    "\\Rightarrow": "Rightarrow", "\\Leftrightarrow": "Leftrightarrow",
    "\\leftrightarrow": "leftrightarrow", "\\longrightarrow": "longrightarrow",
    "\\hookrightarrow": "hookrightarrow", "\\mapsto": "mapsto",
    "\\iff": "iff", "\\rightleftharpoons": "rightleftharpoons",

    # Delimiters
    "\\langle": "langle", "\\rangle": "rangle",
    "\\lceil": "lceil", "\\rceil": "rceil",
    "\\lfloor": "lfloor", "\\rfloor": "rfloor",
    "\\{": "lbrace", "\\}": "rbrace", "\\|": "Vert",

    # Accents (standalone drawn strokes)
    "\\hat": "hat", "\\tilde": "tilde",
    "\\vec": "vec", "\\overline": "overline",

    # Structural
    "\\frac": "frac_bar",

    # Misc
    "\\vdots": "vdots", "\\backslash": "backslash",
    "\\#": "hash", "\\%": "percent",

    # Blackboard bold
    "\\mathbb{R}": "bbR", "\\mathbb{C}": "bbC", "\\mathbb{N}": "bbN",
    "\\mathbb{Z}": "bbZ", "\\mathbb{Q}": "bbQ", "\\mathbb{P}": "bbP",
    "\\mathbb{E}": "bbE", "\\mathbb{F}": "bbF", "\\mathbb{I}": "bbI",
    "\\mathbb{S}": "bbS", "\\mathbb{W}": "bbW", "\\mathbb{T}": "bbT",
    "\\mathbb{D}": "bbD", "\\mathbb{K}": "bbK", "\\mathbb{A}": "bbA",
    "\\mathbb{L}": "bbL", "\\mathbb{X}": "bbX",
}

# Skip: \dot (accent — visually identical to "dot"), \underline (just a line),
# \& (only 3 samples)
SKIP_LABELS = {"\\dot", "\\underline", "\\&"}

# Minimum samples required to import a class
MIN_SAMPLES = 3


# ── InkML parsing ────────────────────────────────────────────────────

NS = {"ink": "http://www.w3.org/2003/InkML"}


def parse_inkml(path: Path) -> tuple[str | None, list[list[dict]]]:
    """Parse an InkML file. Returns (label, strokes) where strokes is
    [[{x, y, t}, ...], ...]."""
    tree = ET.parse(path)
    root = tree.getroot()

    label = None
    for ann in root.findall("ink:annotation", NS):
        if ann.get("type") == "label":
            label = ann.text
            break

    strokes = []
    for trace in root.findall("ink:trace", NS):
        points = []
        for pt_str in trace.text.strip().split(","):
            parts = pt_str.strip().split()
            if len(parts) >= 2:
                points.append({
                    "x": float(parts[0]),
                    "y": float(parts[1]),
                    "t": float(parts[2]) if len(parts) >= 3 else 0.0,
                })
        if points:
            strokes.append(points)

    return label, strokes


def strokes_to_engine(raw_strokes: list[list[dict]]) -> list[Stroke]:
    """Convert raw stroke dicts to engine Stroke objects."""
    return [Stroke.from_dicts(pts) for pts in raw_strokes]


# ── Main ─────────────────────────────────────────────────────────────

def main():
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser(description="Import MathWriting-2024 symbols")
    ap.add_argument("--source", default="data/mathwriting-2024/symbols",
                    help="MathWriting source dir (default: ./data/mathwriting-2024/symbols)")
    ap.add_argument("--output-dir", default="data/shared/symbols",
                    help="Output dir (default: ./data/shared/symbols)")
    ap.add_argument("--dry-run", action="store_true", help="Just show stats, don't write")
    args = ap.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output_dir)

    symbols_dir = source_dir
    if not symbols_dir.exists():
        print(f"Not found: {symbols_dir}")
        return

    # First pass: count labels
    inkml_files = sorted(symbols_dir.glob("*.inkml"))
    print(f"Found {len(inkml_files)} InkML files")

    label_counts = Counter()
    for path in inkml_files:
        label, _ = parse_inkml(path)
        if label:
            label_counts[label] += 1

    # Map labels and count
    mapped = Counter()
    skipped_labels = set()
    for label, count in label_counts.items():
        if label in SKIP_LABELS:
            continue
        name = LABEL_MAP.get(label)
        if name is None:
            skipped_labels.add(label)
            continue
        mapped[name] += count

    if skipped_labels:
        print(f"\nSkipped (no mapping): {sorted(skipped_labels)}")

    # Check existing counts
    existing = {}
    for d in output_dir.iterdir():
        if d.is_dir():
            existing[d.name] = len(list(d.glob("*.png")))

    new_classes = sorted(set(mapped.keys()) - set(existing.keys()))
    augmented_classes = sorted(set(mapped.keys()) & set(existing.keys()))

    print(f"\nNew classes to create: {len(new_classes)}")
    for name in new_classes:
        print(f"  {name:20s} +{mapped[name]} samples")

    print(f"\nExisting classes to augment: {len(augmented_classes)}")
    for name in augmented_classes:
        print(f"  {name:20s} {existing[name]:3d} existing + {mapped[name]} new")

    print(f"\nTotal: {sum(mapped.values())} samples across {len(mapped)} classes")

    if args.dry_run:
        return

    # Second pass: import
    imported = Counter()
    # Track next file ID per class
    next_id = {}
    for name in mapped:
        class_dir = output_dir / name
        class_dir.mkdir(parents=True, exist_ok=True)
        existing_files = sorted(class_dir.glob("*.png"))
        if existing_files:
            last_num = int(existing_files[-1].stem)
            next_id[name] = last_num + 1
        else:
            next_id[name] = 1

    for path in inkml_files:
        label, raw_strokes = parse_inkml(path)
        if not label or label in SKIP_LABELS:
            continue

        name = LABEL_MAP.get(label)
        if name is None:
            continue

        if not raw_strokes:
            continue

        # Convert and render
        engine_strokes = strokes_to_engine(raw_strokes)
        img = render_strokes(engine_strokes, canvas_size=128, stroke_width=2.0)

        # Save
        file_id = next_id[name]
        next_id[name] = file_id + 1
        class_dir = output_dir / name

        png_path = class_dir / f"{file_id:04d}.png"
        json_path = class_dir / f"{file_id:04d}.json"

        img.save(png_path)
        json_path.write_text(json.dumps({
            "strokes": raw_strokes,
            "label": name,
            "source": "mathwriting-2024",
            "source_file": path.name,
        }) + "\n")

        imported[name] += 1

    print(f"\nImported {sum(imported.values())} samples across {len(imported)} classes")
    for name in sorted(imported):
        print(f"  {name:20s} {imported[name]:3d}")


if __name__ == "__main__":
    main()
