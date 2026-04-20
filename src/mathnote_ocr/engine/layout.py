"""Geometry graph for detected symbols.

Computes pairwise spatial measurements between all symbol pairs.
No classification — just raw normalized geometry. Downstream consumers
(rules-based parser or learned model) interpret the measurements.
"""

from __future__ import annotations

from dataclasses import dataclass

from mathnote_ocr.engine.stroke import BBox
from mathnote_ocr.engine.grouper import DetectedSymbol


@dataclass
class LayoutSymbol:
    """Symbol node in the geometry graph."""

    symbol: str
    bbox: BBox
    confidence: float
    stroke_indices: list[int]


@dataclass
class SpatialEdge:
    """Raw geometric measurements between two symbols.

    All distances normalized by ref_h = max(source.h, target.h, 1.0).

    Attributes:
        source: Index into the symbols list.
        target: Index into the symbols list.
        v_offset: (target.cy - source.cy) / ref_h.
                  Negative = target is above source.
        h_offset: (target.x - source.x2) / ref_h.
                  Positive = gap to the right, negative = overlapping horizontally.
        size_ratio: target.h / source.h.
        overlap_v: Fraction of the shorter symbol's height that overlaps
                   vertically with the taller one. 0 = no overlap, 1 = fully contained.
    """

    source: int
    target: int
    v_offset: float
    h_offset: float
    size_ratio: float
    overlap_v: float


@dataclass
class ExpressionLayout:
    """Geometry graph. Nodes are symbols, edges carry spatial measurements."""

    symbols: list[LayoutSymbol]
    edges: list[SpatialEdge]


def _vertical_overlap(a: BBox, b: BBox) -> float:
    """Fraction of the shorter symbol's height that overlaps vertically."""
    top = max(a.y, b.y)
    bot = min(a.y2, b.y2)
    overlap = max(0.0, bot - top)
    shorter = min(a.h, b.h)
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def _compute_edge(source: BBox, target: BBox) -> SpatialEdge:
    """Compute raw geometric measurements between two bboxes."""
    ref_h = max(source.h, target.h, 1.0)

    v_offset = (target.cy - source.cy) / ref_h
    h_offset = (target.x - source.x2) / ref_h
    size_ratio = target.h / ref_h if source.h > 0 else 1.0
    overlap_v = _vertical_overlap(source, target)

    # source/target indices filled in by caller
    return SpatialEdge(
        source=-1,
        target=-1,
        v_offset=v_offset,
        h_offset=h_offset,
        size_ratio=size_ratio,
        overlap_v=overlap_v,
    )


def analyze_layout(symbols: list[DetectedSymbol]) -> ExpressionLayout:
    """Build geometry graph from detected symbols.

    Connects every symbol pair (both directions) and records
    raw geometric measurements on each edge. O(n^2) edges.
    """
    nodes = [
        LayoutSymbol(
            symbol=s.symbol,
            bbox=s.bbox,
            confidence=s.confidence,
            stroke_indices=s.stroke_indices,
        )
        for s in symbols
    ]

    edges: list[SpatialEdge] = []
    n = len(nodes)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            edge = _compute_edge(nodes[i].bbox, nodes[j].bbox)
            edge.source = i
            edge.target = j
            edges.append(edge)

    return ExpressionLayout(symbols=nodes, edges=edges)
