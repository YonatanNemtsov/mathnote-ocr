"""What an app lets the engine read.

An app that never expects some symbols (a matrix app: no slash, which a
hurried comma looks like) or doesn't tell some apart says so once:

    ocr = MathOCR(config, vocabulary=Vocabulary(exclude={"slash"}, aliases={"X_cap": "x"}))
    ocr.detect(strokes)                          # read with it
    ocr.detect(strokes, vocabulary=Vocabulary()) # or another, for one call

It shapes the engine's guesses — what a group of strokes is read as, and
the alternatives offered with it:

  exclude   never guessed: its probability is removed and the rest
            rescaled ("given it isn't a slash, what is it?")
  aliases   read as the target: its probability is added to the target's

Two things it leaves alone:
  - single strokes as building blocks: the grouper's stroke patterns still
    see the raw class of each stroke (the "\\" of an x is slash-shaped
    whether or not slash can be an answer);
  - pins, the user's corrections: taken as they are, any symbol, excluded
    or not — a vocabulary is about guessing, not about what a user may say.

The default Vocabulary() changes nothing.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence

import numpy as np


class Vocabulary:
    def __init__(self, exclude: Iterable[str] = (), aliases: Mapping[str, str] | None = None):
        self.exclude = frozenset(exclude)
        self.aliases = dict(aliases or {})
        for src, tgt in self.aliases.items():
            if tgt in self.exclude:
                raise ValueError(f"alias {src!r} -> {tgt!r}: the target is excluded")
            if tgt in self.aliases:
                raise ValueError(f"alias {src!r} -> {tgt!r}: the target is itself an alias")
            if src in self.exclude:
                raise ValueError(f"{src!r} is both excluded and an alias")
        self._index: dict[tuple, tuple] = {}

    def __bool__(self) -> bool:
        """False for the default vocabulary (it changes nothing)."""
        return bool(self.exclude or self.aliases)

    def __repr__(self) -> str:
        return f"Vocabulary(exclude={sorted(self.exclude)}, aliases={self.aliases})"

    def check(self, label_names: Sequence[str]) -> None:
        """Every name must be a class the classifier knows."""
        known = set(label_names)
        unknown = sorted((self.exclude | set(self.aliases) | set(self.aliases.values())) - known)
        if unknown:
            raise ValueError(f"not symbols the classifier knows: {unknown}")

    def symbols(self, label_names: Sequence[str]) -> list[str]:
        """What the engine can read a symbol as, under this vocabulary."""
        return [n for n in label_names if n not in self.exclude and n not in self.aliases]

    def _indices(self, label_names: Sequence[str]):
        key = tuple(label_names)
        if key not in self._index:
            pos = {n: i for i, n in enumerate(label_names)}
            self._index[key] = (
                [pos[n] for n in self.exclude if n in pos],
                [(pos[s], pos[t]) for s, t in self.aliases.items() if s in pos and t in pos],
            )
        return self._index[key]

    def apply(self, result, label_names: Sequence[str]):
        """The classification as this vocabulary reads it: top symbol,
        confidence and alternatives from the adjusted distribution (the
        full one when the classifier gave it, else its top alternatives).
        Out-of-distribution results are left as they are."""
        if not self or result.symbol is None:
            return result
        if result.probs is not None:
            p = np.array(result.probs, dtype=np.float64)
            excluded, aliased = self._indices(label_names)
            for s, t in aliased:
                p[t] += p[s]
                p[s] = 0.0
            p[excluded] = 0.0
            names = list(label_names)
        else:
            pooled: dict[str, float] = {}
            for name, conf in result.alternatives or [(result.symbol, result.confidence)]:
                if name in self.exclude:
                    continue
                name = self.aliases.get(name, name)
                pooled[name] = pooled.get(name, 0.0) + conf
            names = list(pooled)
            p = np.array([pooled[n] for n in names], dtype=np.float64)
        total = p.sum()
        if total <= 0:
            return dataclasses.replace(result, confidence=0.0, alternatives=[], probs=None)
        p = p / total
        n_alt = max(len(result.alternatives or []), 1)
        order = np.argsort(-p)[:n_alt]
        alternatives = [(names[i], float(p[i])) for i in order if p[i] > 0]
        return dataclasses.replace(
            result,
            symbol=alternatives[0][0],
            confidence=alternatives[0][1],
            alternatives=alternatives,
            probs=p if result.probs is not None else None,
        )
