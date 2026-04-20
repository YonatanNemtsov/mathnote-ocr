"""Public Python API for math_ocr_v2: strokes → LaTeX."""

from __future__ import annotations

from mathnote_ocr.engine.stroke import Stroke
from mathnote_ocr.engine.grouper import group_and_classify, GrouperCache, GrouperParams
from mathnote_ocr.classifier.inference import SymbolClassifier
from mathnote_ocr.tree_parser.inference import SubsetTreeParser
from mathnote_ocr.pipeline_config import load_config, get


class MathOCR:
    """Stroke-based math OCR engine.

    Usage::

        ocr = MathOCR()                    # uses configs/default.yaml
        ocr = MathOCR(config="mixed_v3")   # uses configs/mixed_v3.yaml
        results = ocr.parse([
            [{"x": 10, "y": 20}, {"x": 15, "y": 25}],
            [{"x": 30, "y": 20}, {"x": 35, "y": 25}],
        ])
        latex = results[0]["latex"]
    """

    def __init__(
        self,
        config: str | None = "default",
        *,
        classifier_run: str | None = None,
        tree_run: str | None = None,
        gnn_run: str | None = None,
        scoring: str | None = None,
    ) -> None:
        cfg = load_config(config)

        # Resolve: explicit kwarg > yaml config > hardcoded default
        _cls_run = classifier_run or get(cfg, "classifier.run", "v9_combined")
        _tree_run = tree_run or get(cfg, "tree_parser.subset_run", "mixed_v8")
        _gnn_run = gnn_run or get(cfg, "tree_parser.gnn_run")
        _scoring = scoring or get(cfg, "tree_parser.scoring", "full_spatial")

        self.classifier = SymbolClassifier(
            run=_cls_run,
            ood_threshold=get(cfg, "classifier.ood_threshold", 15.0),
            per_class_thresholds=get(cfg, "classifier.per_class_thresholds", {}),
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

        tp_kwargs = dict(
            subset_run=_tree_run,
            scoring=_scoring,
            tree_strategy=get(cfg, "tree_parser.tree_strategy", "edmonds"),
            tta_runs=get(cfg, "tree_parser.tta_runs", 1),
            tta_dx=get(cfg, "tree_parser.tta_dx", 0.05),
            tta_dy=get(cfg, "tree_parser.tta_dy", 0.05),
            tta_size=get(cfg, "tree_parser.tta_size", 0.05),
            root_discount=get(cfg, "tree_parser.root_discount", 0.2),
        )
        if _gnn_run:
            from mathnote_ocr.tree_parser.inference import GNNTreeParser
            self.tree_parser = GNNTreeParser(gnn_run=_gnn_run, **tp_kwargs)
        else:
            self.tree_parser = SubsetTreeParser(**tp_kwargs)
        self._cache = GrouperCache()

    def parse(
        self,
        strokes: list[list[dict]],
        top_k: int = 5,
        stroke_width: float | None = None,
        canvas_size: int = 800,
    ) -> list[dict]:
        """Parse handwritten strokes into LaTeX.

        Args:
            strokes: List of strokes. Each stroke is a list of
                ``{"x": float, "y": float}`` point dicts.
            top_k: Maximum number of candidate partitions to return.
            stroke_width: Stroke width used during rendering.
                If None, uses the value from pipeline config.
            canvas_size: Source canvas size (max of width/height).

        Returns:
            List of result dicts sorted by confidence (best first).
            Each dict has keys ``latex``, ``confidence``, and ``symbols``.
            Returns ``[]`` if no symbols are detected.
        """
        if not strokes:
            return []

        sw = stroke_width if stroke_width is not None else self.grouper_params.stroke_width

        stroke_objs = [Stroke.from_dicts(pts) for pts in strokes]
        self._cache.update(len(stroke_objs))

        partitions = group_and_classify(
            stroke_objs,
            self.classifier,
            stroke_width=sw,
            source_size=canvas_size,
            top_k=top_k,
            cache=self._cache,
            params=self.grouper_params,
        )

        if not partitions:
            return []

        results = []
        for partition in partitions:
            symbols = sorted(partition, key=lambda s: s.bbox.x)
            latex, parse_conf, _tree, _ev = self.tree_parser.parse_with_tree(symbols)

            sym_conf = 1.0
            for s in symbols:
                sym_conf *= s.confidence
            sym_conf = sym_conf ** (1.0 / max(len(symbols), 1))

            results.append({
                "latex": latex,
                "confidence": round(sym_conf * parse_conf, 4),
                "symbols": [
                    {
                        "symbol": s.symbol,
                        "confidence": s.confidence,
                        "stroke_indices": s.stroke_indices,
                        "bbox": {"x": s.bbox.x, "y": s.bbox.y,
                                 "w": s.bbox.w, "h": s.bbox.h},
                    }
                    for s in symbols
                ],
            })

        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results

    def clear(self) -> None:
        """Reset the internal cache. Call when the user clears the canvas."""
        self._cache = GrouperCache()
