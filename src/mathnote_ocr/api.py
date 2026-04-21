"""Public Python API for mathnote_ocr: strokes → Expression.

Main entry points:
    ocr = MathOCR()                  # bundled defaults
    expr = ocr.detect(strokes)       # list[list[(x, y)]] → Expression

Expression is immutable; corrections return new Expression.
"""

from __future__ import annotations

from typing import Sequence

from mathnote_ocr.classifier.inference import SymbolClassifier
from mathnote_ocr.engine.grouper import (
    GrouperCache,
    GrouperParams,
    group_and_classify,
)
from mathnote_ocr.engine.stroke import Stroke, StrokePoint
from mathnote_ocr.expression import Expression, Symbol, empty_expression
from mathnote_ocr.pipeline_config import get, load_config
from mathnote_ocr.tree_parser.inference import SubsetTreeParser

# Input types accepted by detect()
PointInput = tuple[float, float] | tuple[float, float, float] | dict
StrokeInput = Sequence[PointInput]
StrokesInput = Sequence[StrokeInput]


class MathOCR:
    """Stroke-based math OCR engine. Stateless — safe to share."""

    def __init__(
        self,
        config: str | None = "default",
        *,
        classifier_run: str | None = None,
        subset_run: str | None = None,
        gnn_run: str | None = None,
        scoring: str | None = None,
        weights_dir: str | None = None,
        canvas_size: int = 800,
    ) -> None:
        self._default_canvas_size = canvas_size
        cfg = load_config(config)

        _cls_run = classifier_run or get(cfg, "classifier.run", "v9_combined")
        _subset_run = subset_run or get(cfg, "tree_parser.subset_run", "mixed_v8")
        _gnn_run = gnn_run or get(cfg, "tree_parser.gnn_run")
        _scoring = scoring or get(cfg, "tree_parser.scoring", "full_spatial")

        self.classifier = SymbolClassifier(
            run=_cls_run,
            ood_threshold=get(cfg, "classifier.ood_threshold", 15.0),
            per_class_thresholds=get(cfg, "classifier.per_class_thresholds", {}),
            weights_dir=weights_dir,
        )

        self.grouper_params = GrouperParams(
            max_strokes_per_symbol=get(cfg, "grouper.max_strokes_per_symbol", 4),
            size_multiplier=get(cfg, "grouper.size_multiplier", 0.1),
            min_merge_distance=get(cfg, "grouper.min_merge_distance", 14.0),
            max_group_diameter_ratio=get(cfg, "grouper.max_group_diameter_ratio", 2.2),
            conflict_threshold=get(cfg, "grouper.conflict_threshold", 0.32),
            min_confidence=get(cfg, "classifier.min_confidence", 0.15),
            ood_threshold=get(cfg, "classifier.ood_threshold", 15.0),
            stroke_width=get(cfg, "grouper.stroke_width", 2.0),
        )
        self._top_k_default = get(cfg, "grouper.top_k", 1)

        tp_kwargs = dict(
            subset_run=_subset_run,
            scoring=_scoring,
            tree_strategy=get(cfg, "tree_parser.tree_strategy", "edmonds"),
            tta_runs=get(cfg, "tree_parser.tta_runs", 1),
            tta_dx=get(cfg, "tree_parser.tta_dx", 0.05),
            tta_dy=get(cfg, "tree_parser.tta_dy", 0.05),
            tta_size=get(cfg, "tree_parser.tta_size", 0.05),
            root_discount=get(cfg, "tree_parser.root_discount", 0.2),
            weights_dir=weights_dir,
        )
        if _gnn_run:
            from mathnote_ocr.tree_parser.inference import GNNTreeParser

            self.tree_parser = GNNTreeParser(gnn_run=_gnn_run, **tp_kwargs)
        else:
            self.tree_parser = SubsetTreeParser(**tp_kwargs)

    # ── Session factory ──────────────────────────────────────────────

    def session(
        self,
        *,
        canvas_size: int | None = None,
        stroke_width: float | None = None,
    ) -> Session:
        """Create a stateful session for incremental detection."""
        return Session(self, canvas_size=canvas_size, stroke_width=stroke_width)

    # ── Detection ────────────────────────────────────────────────────

    def detect(
        self,
        strokes: StrokesInput,
        *,
        canvas_size: int | None = None,
        stroke_width: float | None = None,
        hints: dict[int, str] | None = None,
        top_k: int = 1,
        cache: GrouperCache | None = None,
    ) -> Expression:
        """Detect a math expression from strokes.

        Args:
            strokes: List of strokes. Each stroke is a list of (x, y) or
                (x, y, t) tuples, or {"x", "y", "t"?} dicts.
            canvas_size: Source canvas max dimension. Auto-computed from
                stroke extents when absent.
            stroke_width: Rendering stroke width. Defaults to config.
            hints: Map of stroke_id → forced symbol name. The classifier
                output is overridden when all strokes of a detected symbol
                share a hint label.
            top_k: How many candidate partitions to consider. Extras are
                placed on ``expr.alternatives``.
            cache: Optional GrouperCache for reuse across detections.

        Returns:
            An Expression. Empty Expression (``len(expr) == 0``) when
            nothing was detected.
        """
        stroke_objs = _normalize_strokes(strokes)
        if not stroke_objs:
            return empty_expression()

        cs = canvas_size if canvas_size is not None else _autocanvas(stroke_objs, self._default_canvas_size)
        sw = stroke_width if stroke_width is not None else self.grouper_params.stroke_width
        k = max(1, top_k)

        partitions = group_and_classify(
            stroke_objs,
            self.classifier,
            stroke_width=sw,
            source_size=cs,
            top_k=k,
            cache=cache,
            params=self.grouper_params,
        )
        if not partitions:
            return Expression(strokes=stroke_objs, symbols={}, tree=None, confidence=0.0)

        results: list[Expression] = []
        for partition in partitions:
            detected = sorted(partition, key=lambda s: s.bbox.x)
            if hints:
                _apply_hints(detected, hints)
            latex, parse_conf, tree, _ev = self.tree_parser.parse_with_tree(detected)
            symbols = _symbols_from_detected(detected, stroke_objs)
            sym_conf = _geomean_confidence(detected)
            expr = Expression(
                strokes=stroke_objs,
                symbols=symbols,
                tree=tree,
                confidence=round(sym_conf * parse_conf, 4),
            )
            results.append(expr)

        results.sort(key=lambda e: e.confidence, reverse=True)
        best = results[0]
        best.alternatives = results[1:] if k > 1 else []
        return best


# ── Helpers ──────────────────────────────────────────────────────────


def _normalize_strokes(strokes) -> list[Stroke]:
    """Convert point tuples to Stroke objects. Pass-through if already Stroke."""
    out: list[Stroke] = []
    for raw in strokes:
        if isinstance(raw, Stroke):
            out.append(raw)
        elif raw:
            out.append(Stroke.from_points([StrokePoint(*p) for p in raw]))
    return out


def _autocanvas(strokes: list[Stroke], fallback: int) -> int:
    """Infer canvas size from the max extent of stroke points."""
    coords = (c for s in strokes for p in s.points for c in (p.x, p.y))
    return int(max(coords, default=fallback))


def _apply_hints(detected, hints: dict[int, str]) -> None:
    """Override symbol names when all strokes of a symbol share a hint."""
    for ds in detected:
        labels = {hints[i] for i in ds.stroke_indices if i in hints}
        if len(labels) == 1:
            ds.symbol = labels.pop()


def _symbols_from_detected(detected, strokes: list[Stroke]) -> dict[int, Symbol]:
    """Convert pipeline DetectedSymbols to our Symbol dict."""
    return {
        i: Symbol(
            id=i,
            name=ds.symbol,
            bbox=ds.bbox,
            strokes=[strokes[idx] for idx in ds.stroke_indices],
            confidence=ds.confidence,
            alternatives=list(ds.alternatives or []),
        )
        for i, ds in enumerate(detected)
    }


def _geomean_confidence(detected) -> float:
    if not detected:
        return 0.0
    conf = 1.0
    for s in detected:
        conf *= s.confidence
    return conf ** (1.0 / len(detected))


# ── Session ──────────────────────────────────────────────────────────


class Session:
    """Stateful stroke buffer + grouper cache. Produces Expressions on demand.

    For interactive drawing UIs. Maintains a list of strokes and a
    GrouperCache so repeated detect() calls after adding strokes are fast.
    """

    def __init__(
        self,
        ocr: MathOCR,
        *,
        canvas_size: int | None = None,
        stroke_width: float | None = None,
    ) -> None:
        self._ocr = ocr
        self._strokes: list[Stroke] = []
        self._cache = GrouperCache()
        self.canvas_size = canvas_size
        self.stroke_width = stroke_width

    @property
    def strokes(self) -> list[Stroke]:
        """Snapshot of the current strokes."""
        return list(self._strokes)

    def __len__(self) -> int:
        return len(self._strokes)

    def add_stroke(self, points: StrokeInput, *, width: float = 2.0) -> int:
        """Append a stroke. Returns the new stroke's id."""
        self._strokes.append(
            Stroke.from_points([StrokePoint(*p) for p in points], width=width)
        )
        return len(self._strokes) - 1

    def remove_stroke(self, stroke_id: int) -> None:
        """Drop a stroke by id. Invalidates cache."""
        del self._strokes[stroke_id]
        self._cache = GrouperCache()

    def clear(self) -> None:
        """Reset strokes and cache."""
        self._strokes.clear()
        self._cache = GrouperCache()

    def detect(
        self,
        *,
        hints: dict[int, str] | None = None,
        top_k: int = 1,
    ) -> Expression:
        """Run detection on the current strokes. Uses the cache."""
        return self._ocr.detect(
            self._strokes,
            canvas_size=self.canvas_size,
            stroke_width=self.stroke_width,
            hints=hints,
            top_k=top_k,
            cache=self._cache,
        )
