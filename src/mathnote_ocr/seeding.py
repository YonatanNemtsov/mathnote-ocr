"""Seeded readings: the same ink and seed always read the same.

The tree parser samples (the symbol subsets it scores, the jitter of its
test-time augmentation). Within ``seeded(seed)`` those samples come from a
generator of that seed, private to the calling thread (a server reads in
several at once); outside it, from Python's global ``random`` as before (in
training, data generation).

    with seeded(0):
        ...                      # sampling code calls rng()
"""

from __future__ import annotations

import contextvars
import random
from contextlib import contextmanager

_current: contextvars.ContextVar[random.Random | None] = contextvars.ContextVar("mathnote_ocr_rng", default=None)


def rng():
    """The generator to sample from: the seeded one inside ``seeded``, else the global ``random``."""
    r = _current.get()
    return random if r is None else r


@contextmanager
def seeded(seed: int | None):
    """Samples within come from a generator of *seed* (None: unseeded, the global one)."""
    if seed is None:
        yield
        return
    token = _current.set(random.Random(seed))
    try:
        yield
    finally:
        _current.reset(token)
