"""Which label is each symbol? A small learned layer of context over the
classifier.

The classifier labels a group of strokes from its picture alone, at a
fixed size: it can't see that this "x" is twice the height of the writing
around it (an X), that this dot sits on the line (a period, not a centred
dot), or that this "1" stands right after a letter, small and raised (a
prime). Its top few labels almost always hold the right one; the choice among them is
what goes wrong. This scorer makes that choice again, for each symbol of a
reading, from:

  the classifier   each candidate's log probability and rank (the scorer
                   starts from the classifier's ranking and learns changes)
  the candidate    which label it is (a learned embedding)
  geometry         the symbol's size against the reading's writing scale
                   (the median height of its ordinary symbols), its width
                   and shape, where it sits against the line (centre, bottom
                   and top against the ordinary symbols' medians), and its
                   height and place against its two nearest neighbours
  neighbours       the nearest symbol on its left and on its right: their
                   labels (the classifier's), and where they are
                   — or, the attention model (LabelScorerAttn): every symbol
                   of the reading, each a token (its geometry, its place in
                   the reading, the classifier's label and how sure it is),
                   two attention layers over them, then the same scoring of
                   each symbol's candidates
                   — or, jointly (LabelScorerJoint): the same, in passes; in
                   each, a symbol sees the others through their labels as the
                   pass before chose them (as probabilities over their
                   candidates — the classifier's in the first), so the
                   symbols' choices inform each other

A fraction bar and a minus are left as they are (the parser decides between
them, from what is above and below), and so are symbols the user pinned.
Like the group scorer, it belongs to the classifier it was trained with
(LabelScorer.classifier) and is refused with another.

    scorer = LabelScorer.load("v1", weights_dir)
    symbols = scorer.rerank(symbols)        # DetectedSymbols, relabelled
"""

from __future__ import annotations

import math
import statistics
from dataclasses import replace

import torch
from torch import nn

FLAT = {"-", "frac_bar", "dot", "cdot", ",", ".", "=", "ldots", "cdots", "prime", "times", "+", "pm", "mp"}
SPANNING = {"(", ")", "[", "]", "lbrace", "rbrace", "langle", "rangle", "|", "sum", "int", "prod", "oint", "sqrt"}
BARS = {"-", "frac_bar"}
GEOMETRY = 9          # geometry features per symbol
NEIGHBOUR = 3         # per neighbour: dx, dy (writing scales), log height ratio
NONE = "<none>"


def context(boxes: list[tuple[float, float, float, float]], names: list[str]) -> list[dict]:
    """Each symbol's context: {"geo": [GEOMETRY floats], "left"/"right": (label, [NEIGHBOUR floats])}.
    *boxes*: (x0, y0, x1, y1); *names*: each symbol's label as read (the classifier's top one)."""
    ordinary = [b for b, n in zip(boxes, names) if n not in FLAT and n not in SPANNING and b[3] > b[1]]
    ref = ordinary or [b for b in boxes if b[3] > b[1]] or boxes
    s = max(statistics.median(b[3] - b[1] for b in ref), 1e-6)
    mid = statistics.median((b[1] + b[3]) / 2 for b in ref)
    bottom = statistics.median(b[3] for b in ref)
    top = statistics.median(b[1] for b in ref)
    centre = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    height = [max(b[3] - b[1], 1e-3 * s) for b in boxes]
    mid_x = statistics.median(c[0] for c in centre)
    out = []
    for i, b in enumerate(boxes):
        h, w = height[i], max(b[2] - b[0], 1e-3 * s)
        cx, cy = centre[i]
        nbrs = sorted((abs(centre[j][0] - cx), j) for j in range(len(boxes)) if j != i)[:2]
        if nbrs:
            nh = statistics.mean(height[j] for _d, j in nbrs)
            ncy = statistics.mean(centre[j][1] for _d, j in nbrs)
            local = [math.log(h / nh), (cy - ncy) / s, 0.0]
        else:
            local = [0.0, 0.0, 1.0]
        geo = [math.log(h / s), math.log(w / s), math.log(w / h), (cy - mid) / s, (b[3] - bottom) / s,
               (b[1] - top) / s] + local

        def side(sign):
            cand = [(math.hypot(centre[j][0] - cx, centre[j][1] - cy), j) for j in range(len(boxes))
                    if j != i and sign * (centre[j][0] - cx) > 0]
            if not cand:
                return NONE, [0.0, 0.0, 0.0]
            j = min(cand)[1]
            return names[j], [(centre[j][0] - cx) / s, (centre[j][1] - cy) / s, math.log(height[j] / h)]
        out.append({"geo": geo, "left": side(-1), "right": side(1),
                    "pos": [(cx - mid_x) / s, (cy - mid) / s]})
    return out


class LabelScorerNet(nn.Module):
    def __init__(self, n_labels: int, k: int, emb: int = 16, hidden: int = 96):
        super().__init__()
        self.k = k
        self.emb = nn.Embedding(n_labels, emb)
        self.net = nn.Sequential(nn.Linear(2 + emb + GEOMETRY + 2 * (emb + NEIGHBOUR), hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, logp, rank, cand, geo, left, left_f, right, right_f):
        """(B, K) log probabilities, ranks, candidate label ids; (B, GEOMETRY); neighbours'
        label ids (B,) and features (B, NEIGHBOUR) — the candidates' scores (B, K)."""
        per_symbol = torch.cat([geo, self.emb(left), left_f, self.emb(right), right_f], -1)
        x = torch.cat([logp.unsqueeze(-1), rank.unsqueeze(-1), self.emb(cand),
                       per_symbol.unsqueeze(1).expand(-1, self.k, -1)], -1)
        return self.net(x).squeeze(-1) + logp          # the classifier's ranking, changed where context says so


def encode(items: list[dict], vocab: dict[str, int], k: int):
    """Tensors for symbols [{cands: [(label, p)], geo, left, right}] (unknown labels: <none>)."""
    idx = lambda n: vocab.get(n, vocab[NONE])
    logp, rank, cand, geo, left, left_f, right, right_f, mask = ([] for _ in range(9))
    for it in items:
        cs = list(it["cands"])[:k]
        names = [n for n, _p in cs] + [NONE] * (k - len(cs))
        probs = [p for _n, p in cs] + [1e-9] * (k - len(cs))
        logp.append([math.log(max(p, 1e-9)) for p in probs])
        rank.append([i / k for i in range(k)])
        cand.append([idx(n) for n in names])
        mask.append([n != NONE for n in names])
        geo.append(it["geo"])
        left.append(idx(it["left"][0]))
        left_f.append(it["left"][1])
        right.append(idx(it["right"][0]))
        right_f.append(it["right"][1])
    f = lambda v: torch.tensor(v, dtype=torch.float32)
    l = lambda v: torch.tensor(v, dtype=torch.long)
    return f(logp), f(rank), l(cand), f(geo), l(left), f(left_f), l(right), f(right_f), torch.tensor(mask)


class LabelScorerAttn(nn.Module):
    """Every symbol of a reading a token; attention over them; each symbol's candidates scored."""

    def __init__(self, n_labels: int, k: int, emb: int = 16, d: int = 64, layers: int = 2, heads: int = 4):
        super().__init__()
        self.k = k
        self.emb = nn.Embedding(n_labels, emb)
        self.inp = nn.Linear(GEOMETRY + emb + 1 + 2, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.Linear(d + emb + 2, 96), nn.GELU(), nn.Linear(96, 1))

    def forward(self, tok_label, tok_logp, geo, pos, pad, cand, logp, rank):
        """(B, N) the classifier's top label and its log probability per symbol, (B, N, GEOMETRY),
        (B, N, 2) place in the reading, (B, N) padding; (B, N, K) candidates — scores (B, N, K)."""
        x = self.inp(torch.cat([geo, self.emb(tok_label), tok_logp.unsqueeze(-1), pos], -1))
        h = self.enc(x, src_key_padding_mask=pad)
        z = torch.cat([h.unsqueeze(2).expand(-1, -1, self.k, -1), self.emb(cand),
                       logp.unsqueeze(-1), rank.unsqueeze(-1)], -1)
        return self.head(z).squeeze(-1) + logp


class LabelScorerJoint(nn.Module):
    """Attention over the reading, in passes: each symbol a token of its geometry, its place and
    its candidates' embeddings weighted by their current probabilities (the classifier's, then
    the previous pass's) — the labels chosen jointly, by refinement."""

    def __init__(self, n_labels: int, k: int, emb: int = 16, d: int = 64, layers: int = 2, heads: int = 4,
                 passes: int = 3):
        super().__init__()
        self.k = k
        self.passes = passes
        self.emb = nn.Embedding(n_labels, emb)
        self.inp = nn.Linear(GEOMETRY + emb + 2 + 2, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.Linear(d + emb + 2, 96), nn.GELU(), nn.Linear(96, 1))

    def forward(self, tok_label, tok_logp, geo, pos, pad, cand, logp, rank, mask=None, all_passes=False):
        """Scores (B, N, K) after the last pass (with *all_passes*, a list: one per pass)."""
        if mask is None:
            mask = logp > math.log(1e-8)
        ce = self.emb(cand)                                          # (B, N, K, emb)
        p = torch.softmax(logp.masked_fill(~mask, -1e9), -1)          # the classifier's, renormalised
        out = []
        for _ in range(self.passes):
            soft = (p.unsqueeze(-1) * ce).sum(2)                      # the label as believed now
            top = p.max(-1).values
            ent = -(p * p.clamp(min=1e-9).log()).sum(-1)
            x = self.inp(torch.cat([geo, soft, top.unsqueeze(-1), ent.unsqueeze(-1), pos], -1))
            h = self.enc(x, src_key_padding_mask=pad)
            z = torch.cat([h.unsqueeze(2).expand(-1, -1, self.k, -1), ce, logp.unsqueeze(-1), rank.unsqueeze(-1)], -1)
            s = (self.head(z).squeeze(-1) + logp).masked_fill(~mask, -1e9)
            out.append(s)
            p = torch.softmax(s, -1)
        return out if all_passes else out[-1]


def encode_readings(readings: list[list[dict]], vocab: dict[str, int], k: int):
    """Padded tensors for readings (each a list of symbols [{cands, geo, pos}]): the attention model's
    inputs, then the candidates' mask (B, N, K)."""
    idx = lambda n: vocab.get(n, vocab[NONE])
    n = max(len(r) for r in readings)
    B = len(readings)
    tok_label = torch.zeros(B, n, dtype=torch.long)
    tok_logp = torch.zeros(B, n)
    geo = torch.zeros(B, n, GEOMETRY)
    pos = torch.zeros(B, n, 2)
    pad = torch.ones(B, n, dtype=torch.bool)
    cand = torch.zeros(B, n, k, dtype=torch.long)
    logp = torch.full((B, n, k), math.log(1e-9))
    rank = torch.tensor([i / k for i in range(k)]).expand(B, n, k).clone()
    mask = torch.zeros(B, n, k, dtype=torch.bool)
    for b, r in enumerate(readings):
        for i, it in enumerate(r):
            cs = list(it["cands"])[:k]
            pad[b, i] = False
            tok_label[b, i] = idx(cs[0][0]) if cs else 0
            tok_logp[b, i] = math.log(max(cs[0][1], 1e-9)) if cs else math.log(1e-9)
            geo[b, i] = torch.tensor(it["geo"])
            pos[b, i] = torch.tensor(it["pos"])
            for j, (name, p) in enumerate(cs):
                cand[b, i, j] = idx(name)
                logp[b, i, j] = math.log(max(p, 1e-9))
                mask[b, i, j] = True
    return (tok_label, tok_logp, geo, pos, pad, cand, logp, rank), mask


class LabelScorer:
    def __init__(self, net, vocab: dict[str, int], k: int, classifier: str | None = None, arch: str = "neighbours"):
        self.net = net.eval()
        self.arch = arch                         # "neighbours" (or "geometry": the same net, no neighbours) | "attention"
        self.vocab = vocab
        self.k = k
        self.classifier = classifier             # the classifier run it was trained with

    @classmethod
    def load(cls, run: str, weights_dir=None) -> "LabelScorer":
        from pathlib import Path
        p = Path(run)
        if not p.exists():
            base = Path(weights_dir) if weights_dir else Path(__file__).resolve().parents[1] / "weights"
            p = base / "label_scorer" / run / "checkpoint.pth"
        ck = torch.load(p, map_location="cpu", weights_only=False)
        arch = ck.get("arch", "neighbours")
        if arch == "joint":
            net = LabelScorerJoint(len(ck["vocab"]), ck["k"], ck.get("emb", 16), ck.get("d", 64),
                                   passes=ck.get("passes", 3))
        elif arch == "attention":
            net = LabelScorerAttn(len(ck["vocab"]), ck["k"], ck.get("emb", 16), ck.get("d", 64))
        else:
            net = LabelScorerNet(len(ck["vocab"]), ck["k"], ck.get("emb", 16), ck.get("hidden", 96))
        net.load_state_dict(ck["model"])
        return cls(net, ck["vocab"], ck["k"], ck.get("classifier"), arch)

    def check(self, classifier_run: str | None) -> None:
        """Refuse a classifier other than the one this scorer was trained with."""
        from pathlib import Path
        name = lambda r: Path(str(r)).parent.name if str(r).endswith(".pth") else str(r)
        if self.classifier and classifier_run and name(classifier_run) != name(self.classifier):
            raise ValueError(f"label scorer trained with classifier {self.classifier}, not {classifier_run}")

    @torch.no_grad()
    def rerank(self, symbols: list, keep=lambda s: False, name_map: dict | None = None) -> list:
        """The DetectedSymbols, each labelled again (name, confidence, alternatives in the new
        order) — except bars (the parser's call), symbols with no other label (pinned:
        no alternatives) and those *keep* says to leave. *name_map*: the reader's names
        for labels (X_cap -> x ...), applied to the chosen label as the grouper applies it.
        The symbol's confidence keeps its grouping part: scaled by the new label's
        probability over the classifier's top one's."""
        if not symbols:
            return symbols
        boxes = [(s.bbox.x, s.bbox.y, s.bbox.x + s.bbox.w, s.bbox.y + s.bbox.h) for s in symbols]
        names = [s.name for s in symbols]
        ctx = context(boxes, names)
        items = []
        for s, c in zip(symbols, ctx):
            alts = [(a[0], float(a[1])) for a in (s.alternatives or []) if a[0]]
            if not any(a[0] == s.name for a in alts):
                alts = [(s.name, float(s.confidence))] + alts
            if self.arch == "geometry":
                c = {**c, "left": (NONE, [0.0] * NEIGHBOUR), "right": (NONE, [0.0] * NEIGHBOUR)}
            items.append({"cands": alts[:self.k], **c})
        if self.arch in ("attention", "joint"):
            inputs, mask = encode_readings([items], self.vocab, self.k)
            scores = self.net(*inputs).masked_fill(~mask, -1e9)[0]
        else:
            *inputs, mask = encode(items, self.vocab, self.k)
            scores = self.net(*inputs).masked_fill(~mask, -1e9)
        probs = torch.softmax(scores, -1)
        out = []
        for s, it, p in zip(symbols, items, probs):
            if s.name in BARS or keep(s) or len(it["cands"]) < 2:
                out.append(s)
                continue
            order = sorted(range(len(it["cands"])), key=lambda i: -float(p[i]))
            best = it["cands"][order[0]][0]
            alts = [(it["cands"][i][0], round(float(p[i]), 6)) for i in order]
            top_p = max(it["cands"][0][1], 1e-9)
            conf = min(1.0, s.confidence * float(p[order[0]]) / top_p)
            out.append(replace(s, name=(name_map or {}).get(best, best), confidence=conf, alternatives=alts))
        return out
