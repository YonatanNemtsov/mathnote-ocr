"""Explicitly marked 2D structures (matrices, cases).

The user marks a region; detection then splits it into a grid of cells
(structures.grid) and treats it as one atom in the surrounding expression.
structures.synth generates synthetic grids with ground-truth cells.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Structure:
    """A marked region of strokes.

    kind: "grid" (matrix or cases, told apart by the delimiters),
          "matrix", or "cases".
    """

    kind: str
    stroke_ids: tuple[int, ...]
