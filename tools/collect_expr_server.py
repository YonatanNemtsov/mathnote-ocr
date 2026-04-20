#!/usr/bin/env python3
"""WebSocket server for symbol-by-symbol expression data collection.

Generates LaTeX expressions, walks the user through drawing each symbol,
and saves real handwritten bboxes paired with ground truth tree labels.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import json
import random
import re

import websockets

from mathnote_ocr.engine.stroke import Stroke, compute_bbox
from mathnote_ocr.engine.renderer import render_strokes
from mathnote_ocr.latex_utils.glyphs import _extract_glyphs
from mathnote_ocr.latex_utils.sampling import _set_sampler, sample_expression, sampler_list
from mathnote_ocr.latex_utils.expr_aug import parse_latex, _n_frac_bars, _FUNC_GLYPH_COUNTS
from mathnote_ocr.tree_parser.gen_data import latex_to_tree_labels
from mathnote_ocr import config


# ── Template expansion ────────────────────────────────────────────────

from mathnote_ocr.data_gen.latex_sampling_v2.templates import (
    Variable, UpperVar, Digit, Greek, GreekUpper, Op, RelOp,
    BigOp, Misc, Quant, Bracket, Punct,
)

_TEMPLATE_ATOMS = {
    "var": Variable,
    "Var": UpperVar,
    "digit": Digit,
    "greek": Greek,
    "Greek": GreekUpper,
    "op": Op,
    "rel": RelOp,
    "bigop": BigOp,
    "misc": Misc,
    "quant": Quant,
    "bracket": Bracket,
    "punct": Punct,
}

_TEMPLATE_RE = re.compile(r"\[([a-zA-Z]+)\]")


def expand_template(template: str) -> str:
    """Replace [placeholder] tokens with random values from template atoms."""
    def _replace(m):
        name = m.group(1)
        atom = _TEMPLATE_ATOMS.get(name)
        if atom is None:
            return m.group(0)  # unknown placeholder, leave as-is
        val = atom.sample()
        # Wrap LaTeX commands in braces so they don't merge with adjacent text
        if val.startswith("\\"):
            val = "{" + val + "}"
        return val
    return _TEMPLATE_RE.sub(_replace, template)


# ── LaTeX highlighting ────────────────────────────────────────────────

# Commands that produce a visible glyph (matched by _extract_glyphs)
_GLYPH_COMMANDS = {
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\theta",
    r"\lambda", r"\mu", r"\pi", r"\sigma", r"\phi", r"\psi", r"\omega",
    r"\Gamma", r"\Delta", r"\Sigma", r"\Pi", r"\Phi", r"\Psi", r"\Omega",
    r"\times", r"\cdot", r"\pm", r"\div",
    r"\leq", r"\geq", r"\neq",
    r"\in", r"\subset", r"\cup", r"\cap", r"\forall", r"\exists",
    r"\partial", r"\nabla", r"\infty",
    r"\rightarrow", r"\leftarrow", r"\ldots", r"\cdots",
    r"\lbrace", r"\rbrace",
    r"\sin", r"\cos", r"\tan", r"\log", r"\ln", r"\lim",
    r"\int", r"\sum", r"\prod",
    r"\sqrt",
}

# Structural commands — no own glyph
_STRUCTURAL_COMMANDS = {r"\frac", r"\binom"}

# Multi-glyph commands (rendered as multiple <use> elements)
_MULTI_GLYPH = {
    r"\sin": 3, r"\cos": 3, r"\tan": 3,
    r"\log": 3, r"\ln": 2, r"\lim": 3,
}


def _find_brace_end(latex: str, pos: int) -> int:
    """Find the end of a {}-delimited group starting at pos."""
    if pos >= len(latex) or latex[pos] != '{':
        return pos
    depth = 1
    k = pos + 1
    while k < len(latex) and depth > 0:
        if latex[k] == '{':
            depth += 1
        elif latex[k] == '}':
            depth -= 1
        k += 1
    return k


def _highlight_latex(latex: str, glyph_index: int, is_bar: list,
                     bar_string_indices: list,
                     binom_paren_str_idx: list) -> str:
    """Return LaTeX with the glyph_index-th symbol highlighted.

    Uses is_bar list to determine whether to highlight a char glyph,
    a \\frac/\\binom bar, or a binom paren (highlight whole \\binom).
    bar_string_indices maps bar positions to their \\frac/\\binom string-order index.
    binom_paren_str_idx maps binom paren positions to their \\binom string-order index.
    """
    # Binom paren — highlight whole \binom{...}{...}
    if binom_paren_str_idx[glyph_index] >= 0:
        binom_index = binom_paren_str_idx[glyph_index]
        count = 0
        i = 0
        while i < len(latex):
            if latex[i:].startswith(r"\binom"):
                if count == binom_index:
                    j = i + 6  # len(r"\binom")
                    end_top = _find_brace_end(latex, j)
                    end_bot = _find_brace_end(latex, end_top)
                    token = latex[i:end_bot]
                    return (
                        latex[:i]
                        + r"\colorbox{#fde68a}{$"
                        + token
                        + "$}"
                        + latex[end_bot:]
                    )
                count += 1
                i += 6
            else:
                i += 1
        return latex  # fallback

    if is_bar[glyph_index]:
        bar_index = bar_string_indices[glyph_index]
        frac_count = 0
        i = 0
        while i < len(latex):
            if latex[i:].startswith(r"\frac") or latex[i:].startswith(r"\binom"):
                if frac_count == bar_index:
                    cmd_len = 5 if latex[i] == '\\' and latex[i+1:i+5] == 'frac' else 6
                    j = i + cmd_len
                    end_num = _find_brace_end(latex, j)
                    end_den = _find_brace_end(latex, end_num)
                    token = latex[i:end_den]
                    return (
                        latex[:i]
                        + r"\colorbox{#fde68a}{$"
                        + token
                        + "$}"
                        + latex[end_den:]
                    )
                frac_count += 1
                cmd_len = 5 if latex[i:].startswith(r"\frac") else 6
                i += cmd_len
            else:
                i += 1
        return latex  # fallback

    # Count which char this is (0-based among non-bar, non-binom-paren symbols)
    char_index = sum(1 for i in range(glyph_index)
                     if not is_bar[i] and binom_paren_str_idx[i] < 0)
    count = 0
    i = 0
    while i < len(latex):
        ch = latex[i]

        # Skip whitespace
        if ch == ' ':
            i += 1
            continue

        # Skip structural characters
        if ch in '{}^_':
            i += 1
            continue

        # LaTeX command
        if ch == '\\':
            j = i + 1
            if j < len(latex) and latex[j].isalpha():
                while j < len(latex) and latex[j].isalpha():
                    j += 1
                cmd = latex[i:j]

                if cmd in _STRUCTURAL_COMMANDS:
                    i = j
                    continue

                if cmd in _GLYPH_COMMANDS:
                    n_glyphs = _MULTI_GLYPH.get(cmd, 1)
                    if count <= char_index < count + n_glyphs:
                        # \sqrt needs its {arg} included
                        if cmd == r"\sqrt":
                            end = _find_brace_end(latex, j)
                            token = latex[i:end]
                        else:
                            token = latex[i:j]
                            end = j
                        return (
                            latex[:i]
                            + r"\colorbox{#fde68a}{$"
                            + token
                            + "$}"
                            + latex[end:]
                        )
                    count += n_glyphs
                    i = j
                    continue

            # Unknown single-char command (e.g. \, \;) — skip
            i = j if j <= len(latex) else i + 1
            continue

        # Regular character — visible glyph
        if count == char_index:
            return (
                latex[:i]
                + r"\colorbox{#fde68a}{$"
                + ch
                + "$}"
                + latex[i + 1:]
            )
        count += 1
        i += 1

    return latex  # fallback


# ── Expression generation ────────────────────────────────────────────

def _emit_drawing_order(node, char_c, bar_c, frac_str_c, binom_str_c,
                         n_chars, order, frac_str_idx, binom_paren_idx):
    """Walk LNode tree emitting glyph indices in natural drawing order.

    Mirrors _assign_labels() from tree_parser/gen_data.py but emits
    glyph indices instead of assigning labels.  For fracs, the order is
    numerator glyphs → bar → denominator glyphs.

    Tracks frac_str_idx (bar → frac/binom string-order index) and
    binom_paren_idx (binom paren → binom-only string-order index).
    """
    _r = dict(char_c=char_c, bar_c=bar_c, frac_str_c=frac_str_c,
              binom_str_c=binom_str_c, n_chars=n_chars, order=order,
              frac_str_idx=frac_str_idx, binom_paren_idx=binom_paren_idx)

    if node.kind == "char":
        order.append(char_c[0])
        frac_str_idx.append(-1)
        binom_paren_idx.append(-1)
        char_c[0] += 1
        return

    if node.kind == "command":
        n = _FUNC_GLYPH_COUNTS.get(node.text, 1)
        for _ in range(n):
            order.append(char_c[0])
            frac_str_idx.append(-1)
            binom_paren_idx.append(-1)
            char_c[0] += 1
        return

    if node.kind == "seq":
        for child in node.children:
            _emit_drawing_order(child, **_r)
        return

    if node.kind == "frac":
        num_node, den_node = node.children[0], node.children[1]
        my_str_idx = frac_str_c[0]
        frac_str_c[0] += 1
        saved_bar = bar_c[0]
        num_child_bars = _n_frac_bars(num_node)
        den_child_bars = _n_frac_bars(den_node)
        bar_idx = n_chars + saved_bar + num_child_bars + den_child_bars
        _emit_drawing_order(num_node, **_r)
        order.append(bar_idx)
        frac_str_idx.append(my_str_idx)
        binom_paren_idx.append(-1)
        _emit_drawing_order(den_node, **_r)
        bar_c[0] += 1
        return

    if node.kind == "binom":
        top_node, bot_node = node.children[0], node.children[1]
        my_binom_idx = binom_str_c[0]
        binom_str_c[0] += 1
        # ( glyph
        order.append(char_c[0])
        frac_str_idx.append(-1)
        binom_paren_idx.append(my_binom_idx)
        char_c[0] += 1
        # top, bot (no bar)
        _emit_drawing_order(top_node, **_r)
        _emit_drawing_order(bot_node, **_r)
        # ) glyph
        order.append(char_c[0])
        frac_str_idx.append(-1)
        binom_paren_idx.append(my_binom_idx)
        char_c[0] += 1
        return

    if node.kind == "sqrt":
        order.append(char_c[0])
        frac_str_idx.append(-1)
        binom_paren_idx.append(-1)
        char_c[0] += 1
        if node.children:
            _emit_drawing_order(node.children[0], **_r)
        return

    if node.kind == "sup":
        base, exp = node.children[0], node.children[1]
        if base.kind == "sub" and len(base.children) == 2:
            innerbase = base.children[0]
            sub_content = base.children[1]
            _emit_drawing_order(innerbase, **_r)
            _emit_drawing_order(sub_content, **_r)
            _emit_drawing_order(exp, **_r)
            return
        _emit_drawing_order(base, **_r)
        _emit_drawing_order(exp, **_r)
        return

    if node.kind == "sub":
        base, sub_content = node.children[0], node.children[1]
        if base.kind == "sup" and len(base.children) == 2:
            innerbase = base.children[0]
            sup_content = base.children[1]
            _emit_drawing_order(innerbase, **_r)
            _emit_drawing_order(sub_content, **_r)
            _emit_drawing_order(sup_content, **_r)
            return
        _emit_drawing_order(base, **_r)
        _emit_drawing_order(sub_content, **_r)
        return

    if node.kind == "func":
        for child in node.children:
            _emit_drawing_order(child, **_r)
        return

    # Fallback
    for child in node.children:
        _emit_drawing_order(child, **_r)


def _reorder_for_collection(symbols, tree_labels, n_chars, latex):
    """Reorder symbols for natural drawing order using the LNode parse tree.

    Returns (new_symbols, new_tree_labels, is_bar, frac_str_idx, binom_paren_idx).
    """
    n = len(symbols)
    if n == n_chars and r"\binom" not in latex:
        return symbols, tree_labels, [False] * n, [-1] * n, [-1] * n

    tree = parse_latex(latex)
    if tree is None:
        is_bar = [i >= n_chars for i in range(n)]
        return symbols, tree_labels, is_bar, [-1] * n, [-1] * n

    new_order = []
    frac_str_idx = []
    binom_paren_idx = []
    _emit_drawing_order(tree, [0], [0], [0], [0], n_chars,
                        new_order, frac_str_idx, binom_paren_idx)

    # Safety: append any missed indices
    visited = set(new_order)
    for i in range(n):
        if i not in visited:
            new_order.append(i)
            frac_str_idx.append(-1)
            binom_paren_idx.append(-1)

    # Remap
    old_to_new = {old: new for new, old in enumerate(new_order)}
    new_symbols = [symbols[old] for old in new_order]
    new_tree = []
    for old_idx in new_order:
        parent, edge, order = tree_labels[old_idx]
        new_parent = old_to_new[parent] if parent >= 0 else parent
        new_tree.append((new_parent, edge, order))

    is_bar = [old >= n_chars for old in new_order]
    return new_symbols, new_tree, is_bar, frac_str_idx, binom_paren_idx


def _generate_from_latex(latex: str):
    """Process a concrete LaTeX string into collection-ready data.

    Returns (latex, symbols, tree, is_bar, bar_str_idx, binom_paren_idx) or None.
    """
    glyphs = _extract_glyphs(latex)
    if glyphs is None:
        return None
    n = len(glyphs)
    tree_labels = latex_to_tree_labels(latex, n)
    if tree_labels is None:
        return None
    symbols = [g["name"] for g in glyphs]
    n_chars = sum(1 for g in glyphs if not g.get("is_frac_bar"))

    symbols, tree_labels, is_bar, bar_str_idx, binom_paren_idx = \
        _reorder_for_collection(symbols, tree_labels, n_chars, latex)

    tree = [
        {"parent": p, "edge_type": e, "order": o}
        for p, e, o in tree_labels
    ]
    return latex, symbols, tree, is_bar, bar_str_idx, binom_paren_idx


def _generate_expression(max_symbols: int = 30, min_symbols: int = 3):
    """Generate a valid expression with symbols and tree labels.

    Uses the active sampler (set via _set_sampler).
    Returns (latex, symbols_list, tree_labels, is_bar) or None.
    """
    for _ in range(200):
        latex = sample_expression()
        result = _generate_from_latex(latex)
        if result is None:
            continue
        n = len(result[1])  # symbols list
        if n < min_symbols or n > max_symbols:
            continue
        return result
    return None


# ── Symbol data saving ───────────────────────────────────────────────

def _get_next_id(label_dir: Path, run: str) -> str:
    """Find the next numeric ID for a given run prefix in label_dir."""
    prefix = f"{run}_"
    max_num = 0
    for f in label_dir.glob(f"{prefix}*.png"):
        match = re.match(rf"{re.escape(prefix)}(\d+)", f.stem)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return f"{max_num + 1:04d}"


def _save_symbol_image(
    name: str, raw_strokes: list, canvas_width: int, canvas_height: int,
    stroke_width: float, symbols_dir: Path, run: str,
):
    """Render and save individual symbol PNG + stroke JSON."""
    strokes = [Stroke.from_dicts(pts) for pts in raw_strokes]
    source_size = max(canvas_width, canvas_height)
    image = render_strokes(strokes, stroke_width=stroke_width, source_size=source_size)

    label_dir = symbols_dir / name
    label_dir.mkdir(parents=True, exist_ok=True)
    file_id = _get_next_id(label_dir, run)

    png_path = label_dir / f"{run}_{file_id}.png"
    image.save(png_path)

    json_path = label_dir / f"{run}_{file_id}.json"
    stroke_data = {
        "strokes": raw_strokes,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "stroke_width": stroke_width,
        "label": name,
    }
    json_path.write_text(json.dumps(stroke_data))
    return png_path, json_path


# ── Session state ────────────────────────────────────────────────────

class ExpressionSession:
    def __init__(self, output_dir: Path, run: str):
        self.output_dir = output_dir
        self.run = run
        self.symbols_dir = output_dir.parent / "symbols"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.symbols_dir.mkdir(parents=True, exist_ok=True)

        # Template mode (None = use sampler)
        self.template = None

        # Current expression state
        self.latex = None
        self.expected_symbols = []
        self.tree_labels = []
        self.current_index = 0
        self.collected_bboxes = []  # list of [x, y, w, h] normalized
        self.collected_strokes = []  # buffered strokes per symbol (saved on completion)

        # Stats
        self.total_saved = self._count_saved()

    def _count_saved(self):
        out = self.output_dir / "train.jsonl"
        if not out.exists():
            return 0
        return sum(1 for _ in open(out))

    def new_expression(self):
        if self.template:
            latex = expand_template(self.template)
            result = _generate_from_latex(latex)
        else:
            result = _generate_expression()
        if result is None:
            return None
        self.latex, self.expected_symbols, self.tree_labels, self.is_bar, \
            self.bar_str_idx, self.binom_paren_idx = result
        self.current_index = 0
        self.collected_bboxes = []
        self.collected_strokes = []
        return {
            "type": "expression",
            "latex": self.latex,
            "highlighted_latex": _highlight_latex(
                self.latex, 0, self.is_bar, self.bar_str_idx,
                self.binom_paren_idx),
            "symbols": self.expected_symbols,
            "total": len(self.expected_symbols),
            "is_bar": self.is_bar,
            "current_index": 0,
            "count": self.total_saved,
        }

    def save_symbol(self, raw_strokes, canvas_width, canvas_height, stroke_width):
        """Save one symbol's strokes, compute bbox, advance to next."""
        if self.current_index >= len(self.expected_symbols):
            return {"type": "error", "message": "All symbols already collected"}

        name = self.expected_symbols[self.current_index]

        # Compute bbox from strokes (normalized to canvas)
        strokes = [Stroke.from_dicts(pts) for pts in raw_strokes]
        bbox = compute_bbox(strokes)
        ref = max(canvas_width, canvas_height)
        norm_bbox = [
            round(bbox.x / ref, 6),
            round(bbox.y / ref, 6),
            round(bbox.w / ref, 6),
            round(bbox.h / ref, 6),
        ]
        self.collected_bboxes.append(norm_bbox)
        self.collected_strokes.append({
            "name": name,
            "raw_strokes": raw_strokes,
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
            "stroke_width": stroke_width,
        })

        self.current_index += 1

        # Check if expression is complete
        if self.current_index >= len(self.expected_symbols):
            return self._save_expression()

        return {
            "type": "symbol_saved",
            "index": self.current_index - 1,
            "current_index": self.current_index,
            "remaining": len(self.expected_symbols) - self.current_index,
            "highlighted_latex": _highlight_latex(
                self.latex, self.current_index, self.is_bar, self.bar_str_idx,
                self.binom_paren_idx,
            ),
        }

    def undo_symbol(self):
        """Go back one symbol."""
        if self.current_index <= 0:
            return {"type": "error", "message": "Nothing to undo"}
        self.current_index -= 1
        self.collected_bboxes.pop()
        self.collected_strokes.pop()
        return {
            "type": "symbol_undone",
            "current_index": self.current_index,
            "remaining": len(self.expected_symbols) - self.current_index,
            "highlighted_latex": _highlight_latex(
                self.latex, self.current_index, self.is_bar, self.bar_str_idx,
                self.binom_paren_idx,
            ),
        }

    def _save_expression(self):
        """Save the complete expression to JSONL + individual symbol images."""
        # Save individual symbol images now that expression is confirmed
        symbol_paths = []
        for s in self.collected_strokes:
            png_path, json_path = _save_symbol_image(
                s["name"], s["raw_strokes"], s["canvas_width"],
                s["canvas_height"], s["stroke_width"], self.symbols_dir,
                self.run,
            )
            # Store paths relative to tree_handwritten dir
            symbol_paths.append((
                str(png_path.relative_to(self.output_dir.parent)),
                str(json_path.relative_to(self.output_dir.parent)),
            ))

        # Expression labels (lightweight — used by tree parser training/eval)
        sample = {
            "latex": self.latex,
            "symbols": [
                {"name": name, "bbox": bbox, "png": png, "json": js}
                for name, bbox, (png, js) in zip(
                    self.expected_symbols, self.collected_bboxes, symbol_paths)
            ],
            "tree": self.tree_labels,
        }
        out_path = self.output_dir / "train.jsonl"
        with open(out_path, "a") as f:
            f.write(json.dumps(sample) + "\n")

        # Expression strokes (heavy — used for hard negative generation)
        stroke_sample = {
            "latex": self.latex,
            "canvas_width": self.collected_strokes[0]["canvas_width"],
            "canvas_height": self.collected_strokes[0]["canvas_height"],
            "stroke_width": self.collected_strokes[0]["stroke_width"],
            "symbols": [
                {
                    "name": name,
                    "bbox": bbox,
                    "strokes": s["raw_strokes"],
                }
                for name, bbox, s in zip(
                    self.expected_symbols,
                    self.collected_bboxes,
                    self.collected_strokes,
                )
            ],
        }
        stroke_path = self.output_dir / "train_strokes.jsonl"
        with open(stroke_path, "a") as f:
            f.write(json.dumps(stroke_sample) + "\n")

        self.total_saved += 1
        print(f"  Saved expression #{self.total_saved}: {self.latex}")

        return {
            "type": "expression_saved",
            "count": self.total_saved,
            "latex": self.latex,
        }


# ── WebSocket handler ────────────────────────────────────────────────

async def handler(websocket, state: dict):
    addr = websocket.remote_address
    print(f"[connect] {addr}")

    # Send init info on connect
    await websocket.send(json.dumps({
        "type": "init",
        "samplers": sampler_list(),
        "current_sampler": state["sampler"],
        "runs": _list_runs(),
        "current_run": state["run"],
        "count": state["session"].total_saved,
    }))

    async for message in websocket:
        try:
            msg = json.loads(message)
            msg_type = msg.get("type")
            session = state["session"]

            if msg_type == "set_run":
                run = msg["run"]
                if run == "__new__":
                    run = _next_run_name()
                state["run"] = run
                state["session"] = ExpressionSession(BASE_DIR / run, run)
                session = state["session"]
                print(f"  Switched to run: {run}")
                await websocket.send(json.dumps({
                    "type": "run_changed",
                    "run": run,
                    "runs": _list_runs(),
                    "count": session.total_saved,
                }))

            elif msg_type == "set_sampler":
                name = msg["sampler"]
                _set_sampler(name)
                state["sampler"] = name
                print(f"  Switched sampler: {name}")
                await websocket.send(json.dumps({
                    "type": "sampler_changed",
                    "sampler": name,
                }))

            elif msg_type == "set_template":
                template = msg.get("template", "").strip()
                session.template = template or None
                mode = f"template: {template}" if template else "sampler"
                print(f"  Mode: {mode}")
                await websocket.send(json.dumps({
                    "type": "template_changed",
                    "template": template,
                }))

            elif msg_type == "get_expression":
                result = session.new_expression()
                if result is None:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Failed to generate expression",
                    }))
                else:
                    await websocket.send(json.dumps(result))

            elif msg_type == "save_symbol":
                raw_strokes = msg.get("strokes", [])
                if not raw_strokes:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "No strokes",
                    }))
                    continue

                result = session.save_symbol(
                    raw_strokes,
                    msg.get("canvas_width", 800),
                    msg.get("canvas_height", 400),
                    msg.get("stroke_width", config.RENDER_STROKE_WIDTH),
                )
                await websocket.send(json.dumps(result))

                # Auto-send next expression if this one is complete
                if result["type"] == "expression_saved":
                    next_expr = session.new_expression()
                    if next_expr:
                        await websocket.send(json.dumps(next_expr))

            elif msg_type == "undo_symbol":
                result = session.undo_symbol()
                await websocket.send(json.dumps(result))

            elif msg_type == "skip":
                result = session.new_expression()
                if result:
                    await websocket.send(json.dumps(result))

        except Exception as e:
            import traceback
            traceback.print_exc()
            await websocket.send(json.dumps({
                "type": "error",
                "message": str(e),
            }))

    print(f"[disconnect] {addr}")


BASE_DIR = Path("data/shared/tree_handwritten")


def _list_runs() -> list[str]:
    """List existing run directories (run_NNN pattern only)."""
    if not BASE_DIR.exists():
        return []
    return sorted(
        d.name for d in BASE_DIR.iterdir()
        if d.is_dir() and re.match(r"run_\d+", d.name)
    )


def _next_run_name() -> str:
    """Auto-increment: run_001, run_002, ..."""
    existing = sorted(BASE_DIR.glob("run_*"))
    max_num = 0
    for d in existing:
        match = re.match(r"run_(\d+)", d.name)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return f"run_{max_num + 1:03d}"


async def main(args):
    random.seed(args.seed)
    _set_sampler(args.sampler)

    runs = _list_runs()
    if args.run:
        run = args.run
    elif runs:
        run = runs[-1]
    else:
        run = _next_run_name()
    state = {
        "run": run,
        "sampler": args.sampler,
        "session": ExpressionSession(BASE_DIR / run, run),
    }

    print("Expression Collection Server")
    print(f"  Run: {run}")
    print(f"  Sampler: {args.sampler}")
    print(f"  Output: {BASE_DIR / run}/train.jsonl")
    print(f"  Symbols: {BASE_DIR}/symbols/")
    print(f"  Existing: {state['session'].total_saved} expressions")
    print(f"  WebSocket: ws://localhost:{args.port}")
    print(f"\nOpen tools/collect_expr.html in your browser.\n")

    async with websockets.serve(
        lambda ws: handler(ws, state),
        "localhost", args.port,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sampler", default="dg_all")
    args = ap.parse_args()
    asyncio.run(main(args))
