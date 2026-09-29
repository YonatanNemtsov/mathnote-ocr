"""Grammar: the engine's mechanism — an app's rules, repaired inside detect
(mathnote_ocr.grammar). The rules here are throwaway ones; apps write their own.

    python3.10 -m pytest tests/test_grammar.py -q
"""

import json
import random
from pathlib import Path

import pytest

from mathnote_ocr import Grammar, MathOCR, Rewrite, Vocabulary
from mathnote_ocr.grammar import lines

WEB = Path(__file__).resolve().parents[2] / "math_ocr_web"
CONFIG = WEB / "configs" / "mixed_v10_backtrack_gnn.yaml"
# handwritten glyphs, as test_structures_detect.py uses (not part of this repo: skipped without them)
FONT = WEB / "tests" / "fixtures" / "hand_written_font"


def no_bars(names, where):
    """A throwaway rule: no | anywhere."""
    return {i for i, n in enumerate(names) if n == "|"}


@pytest.fixture(scope="module")
def ocr():
    if not CONFIG.exists() or not FONT.exists():
        pytest.skip("math_ocr_web's config / the handwritten glyphs not beside this repo")
    # an app's vocabulary (no prime; a slash reads as a comma): which alternative is most probable depends on it
    return MathOCR(config=str(CONFIG), vocabulary=Vocabulary(exclude={"prime"}, aliases={"slash": ","}),
                   grammar=Grammar(no_bars))


def glyph(name, height):
    """A handwritten glyph scaled to *height* (a flat one, -, to 0.6 x height wide)."""
    strokes = json.loads(next((FONT / name).glob("*.json")).read_text())["strokes"]
    xs = [p["x"] for s in strokes for p in s]
    ys = [p["y"] for s in strokes for p in s]
    k = (0.6 * height / max(max(xs) - min(xs), 1)) if name == "-" else height / max(max(ys) - min(ys), 1)
    return [[((p["x"] - min(xs)) * k, (p["y"] - min(ys)) * k) for p in s] for s in strokes]


LOW = {",": 0.3, ".": 0.15, "dot": 0.15}      # glyphs that sit on the baseline, their height x the line's


def ink(tokens, x0=40, y0=60, height=40):
    """A handwritten line: glyphs left to right, commas small and on the baseline."""
    out = []
    for n in tokens:
        h = height * LOW.get(n, 0.7 if n == "+" else 1.0)
        g = glyph(n, h)
        w = max(x for s in g for x, _ in s)
        dy = y0 + (height - h if n in LOW else (height - max(y for s in g for _, y in s)) / 2)
        out += [[(x + x0, y + dy) for x, y in s] for s in g]
        x0 += w + height * 0.35
    return out


def test_lines_are_the_readings_siblings_in_order(ocr):
    random.seed(0)
    e = ocr.detect(ink(["(", "1", ",", "2", ")"]), grammar=Grammar())
    assert [(w, [e.symbols[i].name for i in ln]) for w, ln in lines(e)] == [("main", ["(", "1", ",", "2", ")"])]


def test_a_broken_reading_is_repaired_with_a_symbols_own_alternative(ocr):
    random.seed(0)
    assert ocr.detect(ink(["(", "1", "|", "2", ")"])).latex == r"\left( 1 , 2 \right)"


def test_no_grammar_for_a_call_and_a_good_reading_untouched(ocr):
    random.seed(0)
    assert ocr.detect(ink(["(", "1", "|", "2", ")"]), grammar=Grammar()).latex == r"\left( 1 | 2 \right)"
    random.seed(0)
    assert ocr.detect(ink(["(", "1", ",", "2", ")"])).latex == r"\left( 1 , 2 \right)"


def test_the_users_pins_are_never_changed(ocr):
    from mathnote_ocr.api import _normalize_strokes
    from mathnote_ocr.pin import PinnedTree, PinSymbol
    strokes = _normalize_strokes(ink(["(", "1", "|", "2", ")"]))
    bar = max(strokes[1:4], key=lambda s: s.bbox.h)
    random.seed(0)
    assert "|" in ocr.detect(strokes, pins=[PinnedTree.build([PinSymbol("|", [bar])])]).latex


def test_a_rewrite_joins_two_symbols_the_classifier_reads_as_one(ocr):
    """= written as two strokes the grouper split: - - joined, when the
    classifier reads the two strokes together as =."""
    eq = glyph("=", 40)
    assert len(eq) == 2
    strokes = ink(["x"]) + [[(x + 110, y + 60) for x, y in s] for s in eq] + [[(x + 60, y) for x, y in s] for s in ink(["2"])]
    g = Grammar(rewrites=[Rewrite(("-", "-"), ("=",))])
    random.seed(0)
    got = ocr.detect(strokes, grammar=g).latex
    assert "- -" not in got, got
