"""Vocabulary: what an app lets the engine read (mathnote_ocr.vocabulary).

    python3.10 -m pytest tests/test_vocabulary.py -q
"""

import json
import random
from pathlib import Path

import numpy as np
import pytest

from mathnote_ocr import MathOCR, Vocabulary
from mathnote_ocr.classifier.inference import ClassificationResult
from mathnote_ocr.pin import PinnedTree, PinSymbol

LABELS = ["slash", ",", "1", "x", "X_cap", "times"]
WEB = Path(__file__).resolve().parents[2] / "math_ocr_web"
CONFIG = WEB / "configs" / "mixed_v10_backtrack_gnn.yaml"


def result(probs, top_n=3):
    p = np.array(probs, dtype=float)
    order = np.argsort(-p)[:top_n]
    return ClassificationResult(symbol=LABELS[order[0]], confidence=float(p[order[0]]), prototype_distance=1.0,
                                is_ood=False, alternatives=[(LABELS[i], float(p[i])) for i in order], probs=p)


def test_excluded_symbol_is_never_the_answer_and_the_rest_is_rescaled():
    r = Vocabulary(exclude={"slash"}).apply(result([0.7, 0.2, 0.1, 0, 0, 0]), LABELS)
    assert r.symbol == "," and r.confidence == pytest.approx(2 / 3)
    assert [n for n, _ in r.alternatives] == [",", "1"]
    assert r.prototype_distance == 1.0 and not r.is_ood


def test_alias_pools_into_its_target():
    r = Vocabulary(aliases={"X_cap": "x", "times": "x"}).apply(result([0, 0, 0.3, 0.25, 0.25, 0.2]), LABELS)
    assert r.symbol == "x" and r.confidence == pytest.approx(0.7)
    assert all(n not in ("X_cap", "times") for n, _ in r.alternatives)


def test_default_changes_nothing_and_without_full_probs_uses_alternatives():
    r0 = result([0.7, 0.2, 0.1, 0, 0, 0])
    assert Vocabulary().apply(r0, LABELS) is r0
    partial = ClassificationResult("slash", 0.7, 1.0, False, [("slash", 0.7), (",", 0.2), ("1", 0.1)], None)
    r = Vocabulary(exclude={"slash"}).apply(partial, LABELS)
    assert r.symbol == "," and r.confidence == pytest.approx(2 / 3)


def test_inconsistent_vocabularies_are_refused():
    with pytest.raises(ValueError):
        Vocabulary(exclude={"x"}, aliases={"X_cap": "x"})
    with pytest.raises(ValueError):
        Vocabulary(aliases={"a": "b", "b": "c"})
    with pytest.raises(ValueError):
        Vocabulary(exclude={"no_such_symbol"}).check(LABELS)


# ── With the engine, on real handwriting ─────────────────────────────


@pytest.fixture(scope="module")
def ocr():
    if not CONFIG.exists():
        pytest.skip("math_ocr_web's production config not beside this repo")
    return MathOCR(config=str(CONFIG))


def _records():
    rows = []
    for f in (WEB / "data_prod" / "feedback.jsonl", WEB / "data" / "feedback.jsonl"):
        if f.exists():
            rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return rows


def _ink(r):
    flat = sorted(((st[0][2] if len(st[0]) > 2 else 0, [(p[0], p[1]) for p in st])
                   for s in r["symbols"] for st in s["strokes"] if st), key=lambda x: x[0])
    return [p for _t, p in flat]


def test_default_vocabulary_reads_exactly_as_before(ocr):
    for r in _records()[:25]:
        ink = _ink(r)
        random.seed(0)                      # the tree parser's test-time jitter
        a = ocr.detect(ink)
        random.seed(0)
        b = ocr.detect(ink, vocabulary=Vocabulary())
        assert a.latex == b.latex


def test_an_excluded_symbol_never_appears_but_a_pin_may_say_it(ocr):
    vocab = Vocabulary(exclude={"slash"})
    slashy = [r for r in _records() if any(s["name"] == "slash" for s in r["symbols"])
              or "slash" in [s.name for s in ocr.detect(_ink(r)).symbols.values()]]
    if not slashy:
        pytest.skip("no record read with a slash")
    for r in slashy[:6]:
        ink = _ink(r)
        e = ocr.detect(ink, vocabulary=vocab)
        names = [s.name for s in e.symbols.values()]
        assert "slash" not in names
        assert all(n != "slash" for s in e.symbols.values() for n, _ in s.alternatives)
    # the user's word wins: a stroke pinned as a slash is one
    ink = _ink(slashy[0])
    from mathnote_ocr.api import _normalize_strokes
    strokes = _normalize_strokes(ink)
    pin = PinnedTree.build([PinSymbol("slash", [strokes[0]])])
    e = ocr.detect(strokes, pins=[pin], vocabulary=vocab)
    assert any(s.name == "slash" for s in e.symbols.values())


def test_instance_vocabulary_and_symbols(ocr):
    o = MathOCR(config=str(CONFIG), vocabulary=Vocabulary(exclude={"slash"}, aliases={"X_cap": "x"}))
    assert "slash" not in o.symbols and "X_cap" not in o.symbols and "x" in o.symbols
    with pytest.raises(ValueError):
        MathOCR(config=str(CONFIG), vocabulary=Vocabulary(exclude={"no_such_symbol"}))
