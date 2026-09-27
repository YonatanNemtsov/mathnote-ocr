"""Grammar: the engine's mechanism — an app's rules, repaired inside detect
(mathnote_ocr.grammar). The rules here are throwaway ones; apps write their own.

    python3.10 -m pytest tests/test_grammar.py -q
"""

import random
import sys
from pathlib import Path

import pytest

from mathnote_ocr import Grammar, MathOCR, Vocabulary
from mathnote_ocr.grammar import lines

WEB = Path(__file__).resolve().parents[2] / "math_ocr_web"
CONFIG = WEB / "configs" / "mixed_v10_backtrack_gnn.yaml"
GLYPHS = Path(__file__).resolve().parents[2] / "matrix_app" / "tests"


def no_bars(names):
    """A throwaway rule: no | anywhere."""
    return {i for i, n in enumerate(names) if n == "|"}


@pytest.fixture(scope="module")
def ocr():
    if not CONFIG.exists() or not (GLYPHS / "glyphs.py").exists():
        pytest.skip("math_ocr_web's config / the handwritten glyphs not beside this repo")
    sys.path.insert(0, str(GLYPHS))
    # the matrix app's vocabulary: which alternative is most probable depends on it
    return MathOCR(config=str(CONFIG), vocabulary=Vocabulary(exclude={"prime"}, aliases={"slash": ","}),
                   grammar=Grammar(no_bars))


def ink(tokens):
    from glyphs import line
    return line(tokens, 40, 60)


def test_lines_are_the_readings_siblings_in_order(ocr):
    random.seed(0)
    e = ocr.detect(ink(["(", "1", ",", "2", ")"]), grammar=Grammar())
    assert [[e.symbols[i].name for i in ln] for ln in lines(e)] == [["(", "1", ",", "2", ")"]]


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
