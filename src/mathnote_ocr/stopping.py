"""Stopping a reading that is no longer wanted: the caller runs its readings on this thread
with a flag (``with stop_when(event): ...``) and sets it from elsewhere when the reading is
no use any more (a newer one is asked for); the engine checks it at a few points of its
work — each batch of pictures, every few hundred search steps, each grouping pass and
parse — and stops with ReadingStopped once it is set. No flag: never stopped."""

from __future__ import annotations

import threading
from contextlib import contextmanager

_local = threading.local()


class ReadingStopped(Exception):
    """The reading's flag (stopping.stop_when) was set: it was stopped."""


@contextmanager
def stop_when(flag: threading.Event | None):
    """Readings on this thread inside the block stop once *flag* is set."""
    before = getattr(_local, "flag", None)
    _local.flag = flag
    try:
        yield
    finally:
        _local.flag = before


def check() -> None:
    """Raise ReadingStopped if this thread's flag is set."""
    flag = getattr(_local, "flag", None)
    if flag is not None and flag.is_set():
        raise ReadingStopped
