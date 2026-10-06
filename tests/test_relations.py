"""Relations: excluded relations, forbidden links and an app's geometric
test all remove the parser's votes for those links."""

import torch

from mathnote_ocr.bbox import BBox
from mathnote_ocr.relations import EDGES, Relations
from mathnote_ocr.tree_parser.tree_v2 import Symbol

E = 7


def _evidence(n):
    return {"parent_votes": torch.ones(n, n + 1, E)}


def _symbols():
    # a bar, a symbol over it, a symbol beside it
    return [Symbol(0, "frac_bar", BBox(100, 100, 60, 4), (0,)),
            Symbol(1, "1", BBox(120, 60, 10, 30), (1,)),
            Symbol(2, "(", BBox(40, 70, 15, 60), (2,))]


def test_allow_removes_the_links_it_rejects():
    def over_bar(child, parent, relation):
        if relation not in ("num", "den"):
            return True
        (_n, c), (_p, b) = child, parent
        return min(c[2], b[2]) - max(c[0], b[0]) > 0

    votes = Relations(allow=over_bar).restrict(_evidence(3), _symbols())["parent_votes"]
    num, den, sup = int(EDGES["num"]), int(EDGES["den"]), int(EDGES["sup"])
    assert votes[1, 0, num] == 1                    # the 1 over the bar: its numerator
    assert votes[2, 0, num] == 0 and votes[2, 0, den] == 0     # the ( beside it: neither part
    assert votes[2, 0, sup] == 1                    # other relations untouched
    assert votes[:, 3, :].eq(1).all()               # the root column untouched


def test_allow_is_kept_when_links_are_forbidden():
    r = Relations(allow=lambda c, p, rel: False).forbidding([(frozenset({1}), frozenset({0}), int(EDGES["sup"]))])
    assert r.allow is not None
    votes = r.restrict(_evidence(3), _symbols())["parent_votes"]
    links = votes[:, :3, :].clone()
    links[range(3), range(3)] = 0                   # a symbol is never its own parent
    assert links.sum() == 0 and votes[:, 3, :].eq(1).all()
