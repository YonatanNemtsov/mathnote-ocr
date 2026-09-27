"""Public Python API for mathnote_ocr: strokes → Expression.

Main entry points:
    ocr = MathOCR()                  # bundled defaults
    expr = ocr.detect(strokes)       # list[list[(x, y)]] → Expression

Expression is immutable; corrections return new Expression.
"""

from __future__ import annotations

from collections.abc import Sequence

from mathnote_ocr.classifier.inference import SymbolClassifier
from mathnote_ocr.engine.grouper import (
    GrouperCache,
    GrouperParams,
    group_and_classify,
)
from mathnote_ocr.engine.stroke import Stroke, StrokePoint, compute_bbox
from mathnote_ocr.expression import DetectedSymbol, Expression, GridBlock, empty_expression
from mathnote_ocr.pin import PinnedTree
from mathnote_ocr.pipeline_config import get, load_config
from mathnote_ocr.structures import Structure
from mathnote_ocr.grammar import Grammar, repair
from mathnote_ocr.vocabulary import Vocabulary
from mathnote_ocr.structures.grid import split_grid
from mathnote_ocr.tree_parser.inference import SubsetTreeParser
from mathnote_ocr.tree_parser.tree_v2 import ROOT_ID

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
        vocabulary: Vocabulary | None = None,
        grammar: Grammar | None = None,
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

        self.grouper_params = GrouperParams.from_config(cfg)
        # What symbols are read as (mathnote_ocr.vocabulary); the default changes nothing
        self.vocabulary = vocabulary or Vocabulary()
        # What a well-formed reading is (mathnote_ocr.grammar); None: any reading
        self.grammar = grammar
        self.vocabulary.check(self.classifier.label_names)
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

    @property
    def symbols(self) -> list[str]:
        """What a symbol can be read as, under this instance's vocabulary."""
        return self.vocabulary.symbols(self.classifier.label_names)

    def session(self, *, canvas_size: int | None = None) -> Session:
        """Create a stateful session for incremental detection."""
        return Session(self, canvas_size=canvas_size)

    # ── Detection ────────────────────────────────────────────────────

    def detect(
        self,
        strokes: StrokesInput,
        *,
        canvas_size: int | None = None,
        top_k: int = 1,
        pins: Sequence[PinnedTree] | None = None,
        structures: Sequence[Structure] | None = None,
        vocabulary: Vocabulary | None = None,
        grammar: Grammar | None = None,
    ) -> Expression:
        """Detect a math expression from strokes.

        Args:
            strokes: List of strokes. Each stroke is a list of (x, y) or
                (x, y, t) tuples, or {"x", "y", "t"?} dicts, or Stroke
                objects. Rendering uses each ``Stroke.width``.
            canvas_size: Source canvas max dimension. Auto-computed from
                stroke extents when absent.
            top_k: How many candidate partitions to consider. Extras are
                placed on ``expr.alternatives``.
            pins: Optional list of constraint pins. Build each with
                ``PinnedTree.build(...)``. Pin stroke ids must reference
                strokes in the input.
            structures: Optional user-marked regions, e.g.
                ``Structure("grid", stroke_ids)`` for a matrix / cases:
                split into cells and treated as one atom in the expression
                (see Expression.grids).
            vocabulary: What symbols may be read as, for this call (default:
                this instance's; see mathnote_ocr.vocabulary).
            grammar: What a well-formed reading is, for this call (default:
                this instance's; see mathnote_ocr.grammar). A reading that
                breaks it is re-read with a flagged symbol's alternatives.

        Returns:
            An Expression. Empty Expression (``len(expr) == 0``) when
            nothing was detected.
        """
        return self._detect_with_cache(
            strokes,
            GrouperCache(),
            canvas_size=canvas_size,
            top_k=top_k,
            pins=pins,
            structures=structures,
            vocabulary=vocabulary,
            grammar=grammar,
        )

    def _detect_with_cache(
        self,
        strokes: StrokesInput,
        cache: GrouperCache,
        *,
        canvas_size: int | None = None,
        top_k: int = 1,
        pins: Sequence[PinnedTree] | None = None,
        structures: Sequence[Structure] | None = None,
        vocabulary: Vocabulary | None = None,
        grammar: Grammar | None = None,
    ) -> Expression:
        """Detection with an explicit cache (used by Session to reuse
        classification results across calls), then — with a grammar — the
        repair of a reading that breaks it: re-read with a flagged symbol's
        alternatives; the pins given (the user's corrections) are never
        changed. Not part of the public API."""
        stroke_objs = _normalize_strokes(strokes)

        def read(extra_pins=()):
            return self._read(stroke_objs, cache, canvas_size=canvas_size, top_k=top_k,
                              pins=(list(pins or []) + list(extra_pins)) or None,
                              structures=structures, vocabulary=vocabulary)

        expr = read()
        g = self.grammar if grammar is None else grammar
        if g is None or not expr or expr.tree is None:
            return expr
        keep = frozenset(i for p in (pins or []) for i in _pin_stroke_ids(p))
        return repair(expr, read, g, keep=keep)

    def _read(
        self,
        strokes: StrokesInput,
        cache: GrouperCache,
        *,
        canvas_size: int | None = None,
        top_k: int = 1,
        pins: Sequence[PinnedTree] | None = None,
        structures: Sequence[Structure] | None = None,
        vocabulary: Vocabulary | None = None,
    ) -> Expression:
        """One reading (no grammar). Not part of the public API."""
        stroke_objs = _normalize_strokes(strokes)
        if not stroke_objs:
            return empty_expression()

        if pins:
            _validate_pin_strokes(pins, {s.id for s in stroke_objs})

        cs = canvas_size if canvas_size is not None else _autocanvas(stroke_objs, self._default_canvas_size)
        k = max(1, top_k)
        vocab = self.vocabulary if vocabulary is None else vocabulary
        vocab.check(self.classifier.label_names)
        if structures:
            return self._detect_with_structures(stroke_objs, cache, cs, pins, structures, vocab)

        partitions = group_and_classify(
            stroke_objs,
            self.classifier,
            params=self.grouper_params,
            cache=cache,
            source_size=cs,
            top_k=k,
            pins=list(pins) if pins else None,
            vocabulary=vocab,
        )
        if not partitions:
            return Expression(
                strokes=stroke_objs, symbols={}, tree=None, confidence=0.0,
                unexplained_stroke_ids=[s.id for s in stroke_objs],
            )

        results: list[Expression] = []
        pin_list = list(pins) if pins else None
        for partition in partitions:
            detected = sorted(partition, key=lambda s: s.bbox.x)
            _latex, parse_conf, tree, _ev = self.tree_parser.parse_with_tree(detected, pin_list)
            symbols = {i: s for i, s in enumerate(detected)}
            sym_conf = _geomean_confidence(detected)
            covered = {st.id for s in detected for st in s.strokes}
            results.append(
                Expression(
                    strokes=stroke_objs,
                    symbols=symbols,
                    tree=tree,
                    confidence=round(sym_conf * parse_conf, 4),
                    unexplained_stroke_ids=[s.id for s in stroke_objs if s.id not in covered],
                )
            )

        results.sort(key=lambda e: e.confidence, reverse=True)
        best = results[0]
        return Expression(
            strokes=best.strokes,
            symbols=best.symbols,
            tree=best.tree,
            confidence=best.confidence,
            alternatives=results[1:] if k > 1 else [],
            unexplained_stroke_ids=best.unexplained_stroke_ids,
        )


    def _detect_with_structures(
        self,
        stroke_objs: list[Stroke],
        cache: GrouperCache,
        cs: float,
        pins: Sequence[PinnedTree] | None,
        structures: Sequence[Structure],
        vocabulary: Vocabulary | None = None,
    ) -> Expression:
        """Detection with marked regions (grids).

        Each region's strokes are grouped and classified on their own (no
        symbol straddles its border), split into cells, and every cell is
        parsed. The rest is parsed with each region as one "expression"
        atom — the parser's token for a collapsed sub-expression — renamed
        "grid" in the result and rendered from its GridBlock. Alternative
        splits of the first region become Expression.alternatives.
        """
        by_id = {s.id: s for s in stroke_objs}
        claimed: set[int] = set()
        regions = []
        for st in structures:
            unknown = [i for i in st.stroke_ids if i not in by_id]
            if unknown:
                raise ValueError(f"structure references unknown stroke ids {unknown}")
            ids = [i for i in st.stroke_ids if i not in claimed]
            if ids:
                claimed.update(ids)
                regions.append((st, ids))
        pin_list = list(pins) if pins else []

        def pins_within(idset: set[int]):
            inside = [p for p in pin_list if _pin_stroke_ids(p) <= idset]
            return inside or None

        def best_partition(strokes_: list[Stroke]) -> list[DetectedSymbol]:
            if not strokes_:
                return []
            parts = group_and_classify(
                strokes_, self.classifier, params=self.grouper_params, cache=cache,
                source_size=cs, top_k=1, pins=pins_within({s.id for s in strokes_}),
                vocabulary=vocabulary,
            )
            return list(parts[0]) if parts else []

        # Each region: its symbols and its ranked grid splits
        region_data = []
        for st, ids in regions:
            syms = best_partition([by_id[i] for i in ids])
            kind = st.kind if st.kind in ("matrix", "cases") else "auto"
            g = split_grid(syms, kind=kind, n_alternatives=2)
            region_data.append((ids, syms, [g] + g.alternatives))

        outer = [s for s in stroke_objs if s.id not in claimed]
        outer_syms = best_partition(outer)
        outer_pins = pins_within({s.id for s in outer})
        cell_cache: dict = {}

        def cell_parse(r: int, syms: list[DetectedSymbol], cell: list[int]) -> tuple:
            """(LaTeX, tree) of one cell, parsed on its own."""
            key = (r, tuple(sorted(cell)))
            if key not in cell_cache:
                cs_ = sorted((syms[j] for j in cell), key=lambda s: s.bbox.x)
                if cs_:
                    latex, _conf, cell_tree, _ev = self.tree_parser.parse_with_tree(cs_, None)
                    cell_cache[key] = (latex, cell_tree)
                else:
                    cell_cache[key] = ("", None)
            return cell_cache[key]

        def build(variant: int) -> Expression:
            atoms = [
                DetectedSymbol(name="expression", bbox=compute_bbox([by_id[i] for i in ids]),
                               strokes=[by_id[i] for i in ids], confidence=1.0)
                for ids, _syms, _grids in region_data
            ]
            parser_input = sorted(outer_syms + atoms, key=lambda s: s.bbox.x)
            _latex, parse_conf, tree, _ev = self.tree_parser.parse_with_tree(parser_input, outer_pins)
            atom_pos = {id(a): i for i, a in enumerate(parser_input) if any(a is b for b in atoms)}
            symbols = {i: s for i, s in enumerate(parser_input) if id(s) not in atom_pos}
            next_id = len(parser_input)
            grids: dict[int, GridBlock] = {}
            for r, (atom, (ids, syms, splits)) in enumerate(zip(atoms, region_data)):
                aid = atom_pos[id(atom)]
                tree = tree.rename_node(aid, "grid")
                local = {}
                for j, sym in enumerate(syms):
                    symbols[next_id] = sym
                    local[j] = next_id
                    next_id += 1
                g = splits[min(variant, len(splits) - 1)] if r == 0 else splits[0]
                parsed = [[cell_parse(r, syms, cell) for cell in row] for row in g.cells]
                grids[aid] = GridBlock(
                    env=g.env,
                    cells=tuple(tuple(tuple(local[j] for j in cell) for cell in row) for row in g.cells),
                    cell_latex=tuple(tuple(latex for latex, _t in row) for row in parsed),
                    bbox=atom.bbox,
                    cell_trees=tuple(tuple(t for _l, t in row) for row in parsed),
                )
            covered = {st.id for s in symbols.values() for st in s.strokes}
            conf = _geomean_confidence(list(symbols.values())) * parse_conf
            return Expression(
                strokes=stroke_objs, symbols=symbols, tree=tree, confidence=round(conf, 4),
                unexplained_stroke_ids=[s.id for s in stroke_objs if s.id not in covered],
                grids=grids,
            )

        best = build(0)
        n_variants = len(region_data[0][2]) if region_data else 1
        best.alternatives = [build(v) for v in range(1, n_variants)]
        return best


# ── Helpers ──────────────────────────────────────────────────────────


def _pin_stroke_ids(pin) -> set[int]:
    return {
        sid for node_id, node in pin.nodes.items() if node_id != ROOT_ID
        for sid in node.symbol.stroke_ids
    }



def _normalize_strokes(strokes) -> list[Stroke]:
    """Convert point tuples or dicts to Stroke objects; assign auto-incremented ids.

    Each stroke is a list of (x, y) / (x, y, t) tuples or {"x", "y", "t"?}
    dicts. Pass-through if item is already a Stroke (keeps its id).
    """
    out: list[Stroke] = []
    next_id = 0
    for raw in strokes:
        if isinstance(raw, Stroke):
            out.append(raw)
            next_id = max(next_id, raw.id + 1)
        elif raw:
            out.append(Stroke.from_points([_to_point(p) for p in raw], id=next_id))
            next_id += 1
    return out


def _to_point(p) -> StrokePoint:
    """Convert a point in tuple or dict form to a StrokePoint."""
    if isinstance(p, dict):
        return StrokePoint(p["x"], p["y"], p.get("t", 0.0))
    return StrokePoint(*p)


def _validate_pin_strokes(pins: Sequence[PinnedTree], available_stroke_ids: set[int]) -> None:
    """Every pin's stroke ids must reference an available stroke. Stroke
    sets across pins must be disjoint (no stroke can belong to two pins)."""
    claimed: dict[int, int] = {}  # stroke_id -> pin_index that claimed it
    for i, pin in enumerate(pins):
        for sid_node, node in pin.nodes.items():
            if sid_node == ROOT_ID:
                continue
            for sid in node.symbol.stroke_ids:
                if sid not in available_stroke_ids:
                    raise ValueError(
                        f"pin {i} references stroke id {sid} which is not in the input"
                    )
                if sid in claimed:
                    raise ValueError(
                        f"stroke {sid} is claimed by both pin {claimed[sid]} and pin {i}"
                    )
                claimed[sid] = i


def _autocanvas(strokes: list[Stroke], fallback: int) -> int:
    """Infer canvas size from the max extent of stroke points."""
    coords = (c for s in strokes for p in s.points for c in (p.x, p.y))
    return int(max(coords, default=fallback))


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

    For interactive drawing UIs. Maintains strokes and a GrouperCache so
    repeated detect() calls reuse classification work. Pins are *not*
    stored on the session; they are constraints passed per detect call
    (see :meth:`detect`).
    """

    def __init__(
        self,
        ocr: MathOCR,
        *,
        canvas_size: int | None = None,
    ) -> None:
        self._ocr = ocr
        self._strokes: dict[int, Stroke] = {}
        self._cache = GrouperCache()
        self.canvas_size = canvas_size

    @property
    def strokes(self) -> list[Stroke]:
        """Current strokes in insertion order."""
        return list(self._strokes.values())

    def __len__(self) -> int:
        return len(self._strokes)

    def __contains__(self, stroke_id: int) -> bool:
        return stroke_id in self._strokes

    def __getitem__(self, stroke_id: int) -> Stroke:
        return self._strokes[stroke_id]

    def _allocate_id(self) -> int:
        """Lowest unused id, one past the current max."""
        return max(self._strokes, default=-1) + 1

    def add_stroke(
        self,
        points: StrokeInput,
        *,
        id: int | None = None,
        width: float = 2.0,
    ) -> int:
        """Append a stroke. If `id` is None, Session assigns a new one.
        If `id` is provided, it must not already exist. Returns the id.
        """
        if id is None:
            id = self._allocate_id()
        elif id in self._strokes:
            raise ValueError(f"Stroke id {id} already exists")
        self._strokes[id] = Stroke.from_points(
            [_to_point(p) for p in points], id=id, width=width
        )
        return id

    def remove_stroke(self, stroke_id: int) -> None:
        """Drop a stroke by id. Other strokes keep their ids. Invalidates
        cache entries that referenced this stroke."""
        if stroke_id not in self._strokes:
            raise KeyError(f"Stroke id {stroke_id} not found")
        del self._strokes[stroke_id]
        self._cache.invalidate_stroke(stroke_id)

    def move_stroke(self, stroke_id: int, points: StrokeInput) -> None:
        """Replace a stroke's points (keeping its id). Invalidates cache entries
        for this stroke; other strokes stay cached."""
        if stroke_id not in self._strokes:
            raise KeyError(f"Stroke id {stroke_id} not found")
        old = self._strokes[stroke_id]
        self._strokes[stroke_id] = Stroke.from_points(
            [_to_point(p) for p in points], id=stroke_id, width=old.width
        )
        self._cache.invalidate_stroke(stroke_id)

    def clear(self) -> None:
        """Reset strokes and cache."""
        self._strokes.clear()
        self._cache = GrouperCache()

    # ── Detection ────────────────────────────────────────────────────

    def detect(
        self,
        *,
        stroke_ids: Sequence[int] | None = None,
        pins: Sequence[PinnedTree] | None = None,
        top_k: int = 1,
        structures: Sequence[Structure] | None = None,
        vocabulary: Vocabulary | None = None,
        grammar: Grammar | None = None,
    ) -> Expression:
        """Run detection on session strokes.

        Args:
            stroke_ids: Subset of session stroke ids to detect on. If None,
                runs on all session strokes. Unknown ids raise ValueError.
            pins: Optional list of constraint pins for this call. Each pin's
                stroke ids must be in the detection subset.
            top_k: How many candidate partitions to consider.
            structures: Marked regions (e.g. Structure("grid", ids)); ids
                must be in the detection subset.
            vocabulary: What symbols may be read as, for this call (default:
                the MathOCR's).
            grammar: What a well-formed reading is, for this call (default:
                the MathOCR's).
        """
        if stroke_ids is None:
            strokes = list(self._strokes.values())
        else:
            strokes = []
            for sid in stroke_ids:
                if sid not in self._strokes:
                    raise ValueError(f"unknown stroke id {sid}")
                strokes.append(self._strokes[sid])

        return self._ocr._detect_with_cache(
            strokes,
            self._cache,
            canvas_size=self.canvas_size,
            top_k=top_k,
            pins=pins,
            structures=structures,
            vocabulary=vocabulary,
            grammar=grammar,
        )
