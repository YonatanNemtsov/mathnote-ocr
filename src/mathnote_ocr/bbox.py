"""Bounding box — shared geometric primitive."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class BBox:
    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def diagonal(self) -> float:
        return sqrt(self.w**2 + self.h**2)

    def union(self, other: BBox) -> BBox:
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        x2 = max(self.x2, other.x2)
        y2 = max(self.y2, other.y2)
        return BBox(x, y, x2 - x, y2 - y)

    def intersection(self, other: BBox) -> BBox | None:
        x = max(self.x, other.x)
        y = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        if x2 <= x or y2 <= y:
            return None
        return BBox(x, y, x2 - x, y2 - y)

    def iou(self, other: BBox) -> float:
        inter = self.intersection(other)
        if inter is None:
            return 0.0
        inter_area = inter.area
        union_area = self.area + other.area - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def distance(self, other: BBox) -> float:
        """Edge-to-edge distance (0 if overlapping)."""
        dx = max(0, max(self.x - other.x2, other.x - self.x2))
        dy = max(0, max(self.y - other.y2, other.y - self.y2))
        return sqrt(dx**2 + dy**2)

    def center_distance(self, other: BBox) -> float:
        """Center-to-center distance."""
        return sqrt((self.cx - other.cx) ** 2 + (self.cy - other.cy) ** 2)

    def contains(self, other: BBox) -> bool:
        return (
            self.x <= other.x and self.y <= other.y and self.x2 >= other.x2 and self.y2 >= other.y2
        )

    def pad(self, amount: float) -> BBox:
        return BBox(self.x - amount, self.y - amount, self.w + 2 * amount, self.h + 2 * amount)

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.w, self.h]

    @staticmethod
    def from_points(xs: list[float], ys: list[float]) -> BBox:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return BBox(min_x, min_y, max_x - min_x, max_y - min_y)

    @staticmethod
    def union_all(boxes: list[BBox]) -> BBox:
        x = min(b.x for b in boxes)
        y = min(b.y for b in boxes)
        x2 = max(b.x2 for b in boxes)
        y2 = max(b.y2 for b in boxes)
        return BBox(x, y, x2 - x, y2 - y)
