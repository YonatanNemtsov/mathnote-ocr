"""Seeded readings: the parser's sampling repeats within seeded(seed)."""

from mathnote_ocr.seeding import seeded
from mathnote_ocr.tree_parser.evidence import jitter_bboxes
from mathnote_ocr.tree_parser.subset_selection import sample_subsets_spatial

BOXES = [[0, 0, 10, 20], [15, 0, 10, 20], [30, 5, 10, 10], [45, 0, 10, 20], [60, 0, 10, 20]]


def _sample():
    return jitter_bboxes(BOXES), sample_subsets_spatial(len(BOXES), BOXES, n_subsets=8, min_size=2, max_size=4)


def test_the_same_seed_samples_the_same():
    with seeded(0):
        a = _sample()
    with seeded(0):
        b = _sample()
    with seeded(1):
        c = _sample()
    assert a == b and a != c


def test_unseeded_is_the_global_random():
    import random
    random.seed(5)
    a = _sample()
    random.seed(5)
    with seeded(None):
        b = _sample()
    assert a == b
