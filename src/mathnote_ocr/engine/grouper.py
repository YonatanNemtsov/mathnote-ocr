"""Stroke grouping and symbol detection.

Enumerates candidate stroke groups (size 1–4), classifies each with the
CNN classifier, then finds the best non-overlapping partition via
exact cover search.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from mathnote_ocr.classifier.inference import ClassificationResult, SymbolClassifier
from mathnote_ocr.engine.renderer import render_strokes
from mathnote_ocr.engine.stroke import BBox, Stroke, compute_bbox

# ── Default config values (can be overridden via GrouperParams) ──────

_DEFAULT_SIMILAR_SYMBOLS = [
    {"x", "X_cap", "times"},
    {"1", "l", "|"},
    {"0", "O_cap", "o"},
    {"c", "C_cap"},
    {"s", "S_cap"},
    {"v", "V_cap"},
    {"w", "W_cap"},
    {"z", "Z_cap"},
    {"u", "U_cap"},
    {"k", "K_cap"},
    {"-", "frac_bar"},
    {"dot", "cdot"},
    {"slash", "div"},
    {"p", "P_cap"},
    {"Sigma_up", "sum"},
    {"Pi_up", "prod"},
]


def _build_similar_map(groups: list[set[str]]) -> dict[str, set[str]]:
    result = {}
    for group in groups:
        for sym in group:
            result[sym] = group
    return result


@dataclass
class GrouperParams:
    """Runtime parameters for the grouper."""

    max_strokes_per_symbol: int = 4
    size_multiplier: float = 0.1
    min_merge_distance: float = 14.0
    max_group_diameter_ratio: float = 2.2
    conflict_threshold: float = 0.32
    min_confidence: float = 0.15
    ood_threshold: float = 15.0
    similar_symbols: list[set[str]] | None = None

    def __post_init__(self):
        groups = self.similar_symbols or _DEFAULT_SIMILAR_SYMBOLS
        self.similar_symbol_map: dict[str, set[str]] = _build_similar_map(groups)

    @classmethod
    def from_config(cls, cfg: dict) -> "GrouperParams":
        """Build from a pipeline config dict (typically YAML-loaded)."""
        from mathnote_ocr.pipeline_config import get

        return cls(
            max_strokes_per_symbol=get(cfg, "grouper.max_strokes_per_symbol", 4),
            size_multiplier=get(cfg, "grouper.size_multiplier", 0.1),
            min_merge_distance=get(cfg, "grouper.min_merge_distance", 14.0),
            max_group_diameter_ratio=get(cfg, "grouper.max_group_diameter_ratio", 2.2),
            conflict_threshold=get(cfg, "grouper.conflict_threshold", 0.32),
            min_confidence=get(cfg, "classifier.min_confidence", 0.15),
            ood_threshold=get(cfg, "classifier.ood_threshold", 15.0),
        )


# Default confusable symbols to most common variant
_DEFAULTS = {
    "Sigma_up": "sum",
    "Pi_up": "prod",
    "X_cap": "x",
    "times": "x",
    "Z_cap": "z",
    "o": "0",
    "O_cap": "0",
    "C_cap": "c",
    "S_cap": "s",
}

# ── Stroke decomposition heuristic ──────────────────────────────────
# Maps sorted tuple of individual stroke classes → set of plausible
# merged symbols. Used to boost multi-stroke groups when the group
# classifier's top-1 agrees with the heuristic.

_STROKE_NORMALIZE = {
    "1": "|",
    "l": "|",
    "frac_bar": "-",
    "O_cap": "o",
    "0": "o",
    "cdot": "dot",
}

_STROKE_PATTERNS: dict[tuple[str, ...], set[str]] = {
    # 2-stroke
    ("-", "|"): {"+", "T_cap", "t"},
    ("-", "-"): {"="},
    ("|", "dot"): {"i", "j", "!"},
    ("-", "o"): {"theta"},
    ("(", "slash"): {"times", "x", "X_cap"},
    ("slash", "times"): {"x", "times"},
    ("o", "|"): {"phi"},
    ("<", "|"): {"k"},
    ("(", "|"): {"k"},
    ("-", "int"): {"f"},
    ("-", "7"): {"7"},
    ("cup", "|"): {"Psi_up", "psi"},
    ("dot", "j"): {"i", "j"},
    ("cup", "dot"): {"j"},
    ("c", "dot"): {"i"},
    ("!", "slash"): {"x"},
    ("-", "A_cap"): {"A_cap"},
    ("-", "V_cap"): {"V_cap"},
    ("-", "W_cap"): {"W_cap"},
    ("-", "U_cap"): {"U_cap"},
    ("-", "Z_cap"): {"z", "Z_cap"},
    ("-", "L_cap"): {"t"},
    ("(", "-"): {"t", "J_cap"},
    (")", "-"): {"J_cap"},
    ("-", "["): {"E_cap"},
    ("-", "]"): {"exists"},
    ("-", "j"): {"J_cap"},
    ("-", "c"): {"in"},
    ("-", "<"): {"leq"},
    ("-", "sqrt"): {"sqrt"},
    ("-", "w"): {"W_cap"},
    ("dot", "dot"): {"colon"},
    ("-", "Delta_cap"): {"theta"},
    # 3-stroke
    ("-", "|", "|"): {"H_cap", "Pi_up"},
    ("-", "-", "slash"): {"neq"},
    ("-", "dot", "dot"): {"div"},
    ("-", "-", "|"): {"I_cap", "pm"},
    ("(", "-", "|"): {"Pi_up", "pi"},
    ("-", "-", "<"): {"z"},
    ("-", "|", "L_cap"): {"pi"},
    ("dot", "dot", "dot"): {"ldots"},
    # 4-stroke
    ("-", "|", "|", "prime"): {"Pi_up"},
    ("-", "-", "o", "|"): {"Phi_up"},
}

_HEURISTIC_BOOST = 0.85


def _check_stroke_pattern(
    group: frozenset[int],
    cache: dict,
    strokes: list[Stroke],
) -> set[str] | None:
    """Check if individual stroke classifications match a known pattern.

    `group` holds list positions; cache is keyed by stable stroke ids,
    so we translate via `strokes[p].id`.
    Returns set of plausible merged symbols, or None if no match.
    """
    stroke_syms = []
    for si in sorted(group):
        singleton = frozenset([strokes[si].id])
        result = cache.get(singleton)
        if result is None or result.symbol is None:
            return None
        sym = result.symbol
        stroke_syms.append(_STROKE_NORMALIZE.get(sym, sym))
    pattern = tuple(sorted(stroke_syms))
    return _STROKE_PATTERNS.get(pattern)


def _group_confidence(result: ClassificationResult, similar_map: dict[str, set[str]]) -> float:
    """Sum probabilities across similar/confusable symbols for partition scoring.

    E.g. if classifier gives x=0.4, X_cap=0.3, times=0.2, the group
    confidence is 0.9 — meaning the classifier is very sure this is
    *one of* the confusable variants, even if uncertain which one.
    """
    if result.alternatives is None or result.symbol is None:
        return result.confidence
    group = similar_map.get(result.symbol)
    if group is None:
        return result.confidence
    return sum(conf for sym, conf in result.alternatives if sym in group)


def _singleton_geo_mean(
    group: frozenset[int],
    stroke_ids: list[int],
    cache: "GrouperCache",
    params: "GrouperParams",
) -> float | None:
    """Geometric mean of per-stroke singleton confidences in *group*.

    Returns ``None`` when none of the strokes' singletons have been classified
    yet — callers should treat that as "no gate available".
    """
    confs = []
    for si in group:
        sr = cache.get(frozenset([stroke_ids[si]]))
        if sr and sr.confidence is not None:
            confs.append(_group_confidence(sr, similar_map=params.similar_symbol_map))
    if not confs:
        return None
    product = 1.0
    for c in confs:
        product *= c
    return product ** (1.0 / len(confs))


class GrouperCache:
    """Classification cache keyed by stroke-id sets.

    Keys are ``frozenset[int]`` of stable stroke ids; values are the
    ``ClassificationResult`` for that group. Invalidation is per-stroke:
    when a stroke is removed or moved, entries referencing it are dropped;
    entries that don't reference it remain valid.

    Usable like a read/write dict (``key in cache``, ``cache[key]``,
    ``cache.get(key)``) plus the two invalidation helpers below.
    """

    def __init__(self) -> None:
        self._data: dict[frozenset[int], ClassificationResult] = {}

    def __contains__(self, key: frozenset[int]) -> bool:
        return key in self._data

    def __getitem__(self, key: frozenset[int]) -> ClassificationResult:
        return self._data[key]

    def __setitem__(self, key: frozenset[int], value: ClassificationResult) -> None:
        self._data[key] = value

    def get(
        self,
        key: frozenset[int],
        default: ClassificationResult | None = None,
    ) -> ClassificationResult | None:
        return self._data.get(key, default)

    def invalidate_stroke(self, stroke_id: int) -> None:
        """Drop cache entries that reference a specific stroke id."""
        bad = [k for k in self._data if stroke_id in k]
        for k in bad:
            del self._data[k]

    def clear(self) -> None:
        """Full reset (e.g. canvas clear)."""
        self._data.clear()


@dataclass
class DetectedSymbol:
    stroke_indices: list[int]
    bbox: BBox
    symbol: str
    confidence: float
    prototype_distance: float
    alternatives: list[tuple[str, float]] = None  # [(symbol, confidence), ...]


# ── Geometry ─────────────────────────────────────────────────────────


def _bbox_gap(s1: Stroke, s2: Stroke) -> float:
    """Minimum distance between two stroke bounding boxes.

    Returns 0 if the bboxes overlap.  This is more discriminative than
    Hausdorff distance for distinguishing same-symbol strokes (overlapping
    bboxes) from adjacent-symbol strokes (clear gap).
    """
    b1, b2 = s1.bbox, s2.bbox
    dx = max(0.0, b1.x - b2.x2, b2.x - b1.x2)
    dy = max(0.0, b1.y - b2.y2, b2.y - b1.y2)
    return (dx * dx + dy * dy) ** 0.5


def _compute_distance_matrix(strokes: list[Stroke]) -> list[list[float]]:
    """Pairwise bbox gap distances between all strokes."""
    n = len(strokes)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _bbox_gap(strokes[i], strokes[j])
            dist[i][j] = d
            dist[j][i] = d
    return dist


def _stroke_diagonal(stroke: Stroke) -> float:
    return stroke.bbox.diagonal


def _max_merge_distance(
    s1: Stroke,
    s2: Stroke,
    size_mult: float,
    min_merge_distance: float = 14.0,
) -> float:
    """Max bbox gap for two strokes to be considered neighbours."""
    bigger = max(_stroke_diagonal(s1), _stroke_diagonal(s2))
    return max(size_mult * bigger, min_merge_distance)


def _compute_neighbors(
    strokes: list[Stroke],
    distances: list[list[float]],
    size_mult: float,
    min_merge_distance: float = 14.0,
) -> dict[int, set[int]]:
    """Build adjacency based on size-relative proximity."""
    n = len(strokes)
    neighbors: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if distances[i][j] <= _max_merge_distance(
                strokes[i],
                strokes[j],
                size_mult,
                min_merge_distance=min_merge_distance,
            ):
                neighbors[i].add(j)
                neighbors[j].add(i)
    return neighbors


def _enumerate_candidate_groups(
    strokes: list[Stroke],
    distances: list[list[float]],
    neighbors: dict[int, set[int]],
    max_strokes: int,
    size_mult: float,
    cache: dict | None = None,
    min_merge_distance: float = 14.0,
    max_group_diameter_ratio: float = 2.2,
) -> list[frozenset[int]]:
    """Enumerate all valid stroke groups of size 1 to max_strokes.

    Default: pairwise distance (all strokes must be within merge distance
    of each other). For groups that fail pairwise but pass chaining
    (each stroke has >= 1 neighbour), we check if individual stroke
    classifications match a known pattern — if so, the group is allowed.
    """
    n = len(strokes)
    groups: list[frozenset[int]] = [frozenset([i]) for i in range(n)]

    if max_strokes < 2:
        return groups

    # Pre-classify singletons for pattern checking (lazy, only if needed)
    _singleton_cache: dict[int, str | None] = {}

    def _get_singleton_class(idx: int) -> str | None:
        if idx in _singleton_cache:
            return _singleton_cache[idx]
        singleton = frozenset([strokes[idx].id])
        result = cache.get(singleton) if cache else None
        if result is None:
            _singleton_cache[idx] = None
            return None
        sym = result.symbol
        _singleton_cache[idx] = _STROKE_NORMALIZE.get(sym, sym) if sym else None
        return _singleton_cache[idx]

    def _matches_pattern(indices: list[int]) -> bool:
        """Check if individual stroke classes match a known pattern."""
        syms = []
        for i in indices:
            s = _get_singleton_class(i)
            if s is None:
                return False
            syms.append(s)
        pattern = tuple(sorted(syms))
        return pattern in _STROKE_PATTERNS

    # Neighbor-based groups: allows non-consecutive strokes, e.g. going
    # back to add a stroke to an earlier symbol.
    candidates: set[frozenset[int]] = set()
    for i in range(n):
        for j in neighbors.get(i, set()):
            candidates.add(frozenset([i, j]))
    for _ in range(max_strokes - 2):
        grown: set[frozenset[int]] = set()
        for group in candidates:
            for member in group:
                for nb in neighbors.get(member, set()):
                    if nb not in group:
                        grown.add(group | {nb})
        candidates |= grown

    for indices in sorted(candidates, key=lambda s: (len(s), min(s))):
        indices = sorted(indices)

        # 1. Connectivity: each stroke must have at least one neighbour
        ok = True
        for i in indices:
            has_neighbor = False
            for j in indices:
                if i != j and j in neighbors[i]:
                    has_neighbor = True
                    break
            if not has_neighbor:
                ok = False
                break
        if not ok:
            continue

        # 2. Pairwise distance check
        pairwise_ok = True
        for ai in range(len(indices)):
            for bi in range(ai + 1, len(indices)):
                if distances[indices[ai]][indices[bi]] > _max_merge_distance(
                    strokes[indices[ai]],
                    strokes[indices[bi]],
                    size_mult,
                    min_merge_distance=min_merge_distance,
                ):
                    pairwise_ok = False
                    break
            if not pairwise_ok:
                break

        if not pairwise_ok:
            # Chaining fallback: allow if stroke pattern matches
            if not _matches_pattern(indices):
                continue

        # 3. Compactness check
        group_strokes = [strokes[i] for i in indices]
        group_bbox = compute_bbox(group_strokes)
        max_stroke_diag = max(_stroke_diagonal(strokes[i]) for i in indices)
        if max_stroke_diag > 0 and group_bbox.diagonal > max_group_diameter_ratio * max_stroke_diag:
            continue

        groups.append(frozenset(indices))

    return groups


def _find_best_partitions(
    n: int,
    scored_groups: list[tuple[frozenset[int], float, DetectedSymbol]],
    top_k: int,
    max_results: int = 100,
) -> list[tuple[float, list[DetectedSymbol]]]:
    """Find top-k non-overlapping covers of all n strokes.

    Algorithm X style: pick the most constrained uncovered stroke
    (fewest valid groups), try each group, recurse.
    """
    # stroke → list of scored_group indices
    stroke_to_groups: dict[int, list[int]] = {i: [] for i in range(n)}
    for gi, (indices, _score, _sym) in enumerate(scored_groups):
        for s in indices:
            stroke_to_groups[s].append(gi)

    results: list[tuple[float, list[DetectedSymbol]]] = []

    def search(
        uncovered: frozenset[int],
        symbols: list[DetectedSymbol],
        score: float,
    ) -> None:
        if len(results) >= max_results:
            return
        if not uncovered:
            results.append((score, list(symbols)))
            return

        # Pick stroke with fewest valid groups (most constrained first)
        best_valid: list[int] = []
        best_count = len(scored_groups) + 1
        for s in uncovered:
            valid = [g for g in stroke_to_groups[s] if scored_groups[g][0] <= uncovered]
            if len(valid) < best_count:
                best_count = len(valid)
                best_valid = valid

        if not best_valid:
            return  # dead end

        # Try groups in descending confidence order
        best_valid.sort(key=lambda g: scored_groups[g][1], reverse=True)

        for gi in best_valid:
            indices, conf, sym = scored_groups[gi]

            # Early conflict check against already-chosen symbols
            # Skip for sqrt — its bbox naturally encloses the radicand
            conflict = False
            for existing in symbols:
                if sym.symbol == "sqrt" or existing.symbol == "sqrt":
                    continue
                if _symbols_conflict(sym.bbox, existing.bbox):
                    conflict = True
                    break
            if conflict:
                continue

            symbols.append(sym)
            search(uncovered - indices, symbols, score * conf)
            symbols.pop()

    search(frozenset(range(n)), [], 1.0)
    results.sort(
        key=lambda x: x[0] ** (1.0 / max(len(x[1]), 1)),
        reverse=True,
    )
    return results[:top_k]


# ── Spatial validation ───────────────────────────────────────────────


def _bboxes_overlap(bbox1: BBox, bbox2: BBox) -> bool:
    """True if two bboxes have any overlap."""
    return bbox1.x < bbox2.x2 and bbox2.x < bbox1.x2 and bbox1.y < bbox2.y2 and bbox2.y < bbox1.y2


def _symbols_conflict(
    bbox1: BBox,
    bbox2: BBox,
    threshold: float = 0.32,
) -> bool:
    """True if two symbol bboxes overlap AND centres are too close."""
    if not _bboxes_overlap(bbox1, bbox2):
        return False
    dist = ((bbox1.cx - bbox2.cx) ** 2 + (bbox1.cy - bbox2.cy) ** 2) ** 0.5
    avg_diag = (bbox1.diagonal + bbox2.diagonal) / 2
    if avg_diag == 0:
        return False
    return (dist / avg_diag) < threshold


def _size_feat(group_strokes: list[Stroke], source_size: float) -> float:
    """Symbol-relative diagonal, used as an extra classifier feature.

    Returns 0.5 when strokes have no points (degenerate input).
    """
    all_x = [p.x for s in group_strokes for p in s.points]
    all_y = [p.y for s in group_strokes for p in s.points]
    if not all_x:
        return 0.5
    bw = max(all_x) - min(all_x)
    bh = max(all_y) - min(all_y)
    sym_diag = math.sqrt(bw * bw + bh * bh)
    return sym_diag / max(source_size, 1.0)


# ── Top-level ────────────────────────────────────────────────────────


def group_and_classify(
    strokes: list[Stroke],
    classifier: SymbolClassifier,
    *,
    params: GrouperParams,
    cache: GrouperCache,
    source_size: float,
    top_k: int = 1,
    debug: bool = False,
) -> list[list[DetectedSymbol]]:
    """Detect symbols in a set of strokes.

    Returns up to *top_k* valid partitions, each a list of
    ``DetectedSymbol``, sorted by descending total confidence.
    Returns ``[[]]`` (one empty partition) if nothing is found.

    The *cache* is reused across calls on overlapping stroke sets —
    callers that want one-shot detection should construct a fresh
    ``GrouperCache()`` at the call site.
    """
    if not strokes:
        return [[]]

    n = len(strokes)

    # The grouper runs on positions (array lookups), but the cache keys on
    # stable stroke ids. Translate position-sets → id-sets at the boundary.
    stroke_ids: list[int] = [s.id for s in strokes]

    def _pos_to_ids(pos_set: frozenset[int]) -> frozenset[int]:
        return frozenset(stroke_ids[p] for p in pos_set)

    def _classify_uncached(groups: list[frozenset[int]]) -> int:
        """Classify every group in *groups* that isn't already in the cache,
        store results, and return how many were classified."""
        uncached = [g for g in groups if _pos_to_ids(g) not in cache]
        if not uncached:
            return 0
        use_size = classifier.use_size_feat
        images = []
        size_feats = []
        for group in uncached:
            group_strokes = [strokes[i] for i in group]
            images.append(
                render_strokes(
                    group_strokes,
                    canvas_size=classifier.canvas_size,
                    source_size=source_size,
                )
            )
            size_feats.append(
                _size_feat(group_strokes, source_size) if use_size else 0.5
            )
        sf = size_feats if use_size else None
        results = classifier.classify_batch(images, size_feats=sf)
        for group, result in zip(uncached, results):
            cache[_pos_to_ids(group)] = result
        return len(uncached)

    # 1. Distances + neighbours
    t0 = time.perf_counter()
    distances = _compute_distance_matrix(strokes)
    neighbors = _compute_neighbors(
        strokes,
        distances,
        params.size_multiplier,
        min_merge_distance=params.min_merge_distance,
    )
    t_geo = time.perf_counter() - t0

    # 2. Classify singletons first (needed for pattern matching in enumeration)
    t0 = time.perf_counter()
    _classify_uncached([frozenset([i]) for i in range(n)])
    t_singletons = time.perf_counter() - t0

    # 3. Enumerate candidate groups (singletons + multi-stroke)
    #    Pattern matching uses singleton classifications from step 2.
    t0 = time.perf_counter()
    candidate_groups = _enumerate_candidate_groups(
        strokes,
        distances,
        neighbors,
        params.max_strokes_per_symbol,
        params.size_multiplier,
        cache=cache,
        min_merge_distance=params.min_merge_distance,
        max_group_diameter_ratio=params.max_group_diameter_ratio,
    )
    t_enum = time.perf_counter() - t0

    # 4. Classify remaining uncached multi-stroke groups
    t0 = time.perf_counter()
    n_new = _classify_uncached(candidate_groups)
    t_classify = time.perf_counter() - t0

    # 4. Filter valid groups
    t0 = time.perf_counter()
    scored_groups: list[tuple[frozenset[int], float, DetectedSymbol]] = []
    rejected_ood = 0
    rejected_conf = 0

    for group in candidate_groups:
        result = cache[_pos_to_ids(group)]

        if result.is_ood:
            rejected_ood += 1
            if debug:
                print(
                    f"  group {set(group)} → REJECT OOD: '{result.symbol}' dist={result.prototype_distance:.1f}"
                )
            continue
        if result.confidence < params.min_confidence:
            rejected_conf += 1
            if debug:
                print(
                    f"  group {set(group)} → REJECT LOW CONF: '{result.symbol}' conf={result.confidence:.3f}"
                )
            continue

        # Sum probabilities across confusion group (e.g. x+X_cap+times)
        group_conf = _group_confidence(result, similar_map=params.similar_symbol_map)

        # Weight by prototype quality
        proto_quality = 1.0 / (1.0 + (result.prototype_distance / params.ood_threshold) ** 2)
        effective_conf = group_conf * proto_quality

        # Multi-stroke groups: either get a pattern boost, or must beat the
        # geometric mean of their singleton confidences (prevents merging two
        # confident singletons like s+i into 'n').
        # Multi-stroke merges either match a known stroke-decomposition
        # pattern (boost), or must beat the geometric mean of the singleton
        # confidences — otherwise two confident singletons would merge into
        # an unlikely combined symbol (e.g. s+i → 'n').
        if len(group) >= 2:
            pattern = _check_stroke_pattern(group, cache, strokes)
            if pattern and result.symbol in pattern:
                effective_conf = max(effective_conf, _HEURISTIC_BOOST)
                if debug:
                    print(
                        f"  group {set(group)} → PATTERN BOOST '{result.symbol}' to {effective_conf:.3f}"
                    )
            else:
                geo_mean = _singleton_geo_mean(group, stroke_ids, cache, params)
                if geo_mean is not None and effective_conf < geo_mean:
                    if debug:
                        print(
                            f"  group {set(group)} → REJECT SINGLETON GATE: "
                            f"'{result.symbol}' conf={effective_conf:.3f} < geo_mean={geo_mean:.3f}"
                        )
                    continue

        group_strokes = [strokes[i] for i in group]
        bbox = compute_bbox(group_strokes)
        sym_name = _DEFAULTS.get(result.symbol, result.symbol)

        sym = DetectedSymbol(
            stroke_indices=list(group),
            bbox=bbox,
            symbol=sym_name,
            confidence=effective_conf,
            prototype_distance=result.prototype_distance,
            alternatives=result.alternatives,
        )
        scored_groups.append((group, effective_conf, sym))

    if debug:
        print(
            f"[grouper] {len(scored_groups)} valid groups  "
            f"rejected: ood={rejected_ood} low_conf={rejected_conf}"
        )

    # 5. Find best non-overlapping partitions (exact cover)
    t0 = time.perf_counter()
    partitions = _find_best_partitions(n, scored_groups, top_k)
    t_cover = time.perf_counter() - t0

    print(
        f"  [grouper] {n}s {len(candidate_groups)}g {n_new}new: "
        f"geo={t_geo * 1000:.0f}ms singles={t_singletons * 1000:.0f}ms "
        f"enum={t_enum * 1000:.0f}ms classify={t_classify * 1000:.0f}ms "
        f"cover={t_cover * 1000:.0f}ms"
    )

    if debug:
        print(f"[grouper] {len(partitions)} valid partitions")
        if partitions:
            best = partitions[0]
            syms = ", ".join(f"'{s.symbol}'({s.confidence:.2f})" for s in best[1])
            print(f"[grouper] best: [{syms}] score={best[0]:.2f}")

    if not partitions:
        return [[]]

    return [_postprocess(syms) for _, syms in partitions]


# ── Symbol post-processing (composite symbol merging) ────────────────


def _merge_bbox(a: BBox, b: BBox) -> BBox:
    """Combine two bounding boxes."""
    mx = min(a.x, b.x)
    my = min(a.y, b.y)
    return BBox(x=mx, y=my, w=max(a.x2, b.x2) - mx, h=max(a.y2, b.y2) - my)


def _merged_symbol(
    parts: list[DetectedSymbol],
    name: str,
) -> DetectedSymbol:
    """Create a merged symbol from constituent parts."""
    bbox = parts[0].bbox
    for p in parts[1:]:
        bbox = _merge_bbox(bbox, p.bbox)
    indices = []
    for p in parts:
        indices.extend(p.stroke_indices)
    return DetectedSymbol(
        stroke_indices=indices,
        bbox=bbox,
        symbol=name,
        confidence=min(p.confidence for p in parts),
        prototype_distance=max(p.prototype_distance for p in parts),
    )


def _horizontally_aligned(a: DetectedSymbol, b: DetectedSymbol, tol: float = 0.3) -> bool:
    """Centers are horizontally close relative to the wider symbol."""
    w = max(a.bbox.w, b.bbox.w)
    return abs(a.bbox.cx - b.bbox.cx) <= tol * w


def _vertically_stacked(a: DetectedSymbol, b: DetectedSymbol, gap_tol: float = 1.0) -> bool:
    """a is above/below b, vertical gap small relative to width."""
    w = max(a.bbox.w, b.bbox.w)
    return abs(a.bbox.cy - b.bbox.cy) <= gap_tol * w


def _similar_width(a: DetectedSymbol, b: DetectedSymbol, tol: float = 0.5) -> bool:
    return min(a.bbox.w, b.bbox.w) >= tol * max(a.bbox.w, b.bbox.w)


def _overlapping(a: DetectedSymbol, b: DetectedSymbol) -> bool:
    """Bboxes overlap significantly (for slash-through patterns)."""
    return _bboxes_overlap(a.bbox, b.bbox)


def _by_symbol(symbols: list[DetectedSymbol], name: str) -> list[tuple[int, DetectedSymbol]]:
    return [(i, s) for i, s in enumerate(symbols) if s.symbol == name]


def _postprocess(symbols: list[DetectedSymbol]) -> list[DetectedSymbol]:
    """Apply all composite symbol merge rules."""
    for rule in _MERGE_RULES:
        symbols = rule(symbols)
    return symbols


def _merge_equal(symbols: list[DetectedSymbol]) -> list[DetectedSymbol]:
    """Two stacked '-' → '='."""
    minuses = _by_symbol(symbols, "-")
    if len(minuses) < 2:
        return symbols
    used: set[int] = set()
    merged: list[DetectedSymbol] = []
    for ai in range(len(minuses)):
        if minuses[ai][0] in used:
            continue
        for bi in range(ai + 1, len(minuses)):
            if minuses[bi][0] in used:
                continue
            a, b = minuses[ai][1], minuses[bi][1]
            if _similar_width(a, b) and _horizontally_aligned(a, b) and _vertically_stacked(a, b):
                merged.append(_merged_symbol([a, b], "="))
                used.add(minuses[ai][0])
                used.add(minuses[bi][0])
                break
    if not used:
        return symbols
    return [s for i, s in enumerate(symbols) if i not in used] + merged


def _merge_leq(symbols: list[DetectedSymbol]) -> list[DetectedSymbol]:
    """'<' with '-' below → 'leq'."""
    lts = _by_symbol(symbols, "lt")
    minuses = _by_symbol(symbols, "-")
    if not lts or not minuses:
        return symbols
    used: set[int] = set()
    merged: list[DetectedSymbol] = []
    for li, lt in lts:
        for mi, m in minuses:
            if mi in used:
                continue
            if (
                _horizontally_aligned(lt, m)
                and m.bbox.cy > lt.bbox.cy
                and _vertically_stacked(lt, m, gap_tol=1.5)
            ):
                merged.append(_merged_symbol([lt, m], "leq"))
                used.add(li)
                used.add(mi)
                break
    if not used:
        return symbols
    return [s for i, s in enumerate(symbols) if i not in used] + merged


def _merge_geq(symbols: list[DetectedSymbol]) -> list[DetectedSymbol]:
    """'>' with '-' below → 'geq'."""
    gts = _by_symbol(symbols, "gt")
    minuses = _by_symbol(symbols, "-")
    if not gts or not minuses:
        return symbols
    used: set[int] = set()
    merged: list[DetectedSymbol] = []
    for gi, gt in gts:
        for mi, m in minuses:
            if mi in used:
                continue
            if (
                _horizontally_aligned(gt, m)
                and m.bbox.cy > gt.bbox.cy
                and _vertically_stacked(gt, m, gap_tol=1.5)
            ):
                merged.append(_merged_symbol([gt, m], "geq"))
                used.add(gi)
                used.add(mi)
                break
    if not used:
        return symbols
    return [s for i, s in enumerate(symbols) if i not in used] + merged


def _merge_neq(symbols: list[DetectedSymbol]) -> list[DetectedSymbol]:
    """'=' with '/' overlapping → 'neq'."""
    eqs = _by_symbol(symbols, "=")
    slashes = _by_symbol(symbols, "slash")
    if not eqs or not slashes:
        return symbols
    used: set[int] = set()
    merged: list[DetectedSymbol] = []
    for ei, eq in eqs:
        for si, sl in slashes:
            if si in used:
                continue
            if _overlapping(eq, sl):
                merged.append(_merged_symbol([eq, sl], "neq"))
                used.add(ei)
                used.add(si)
                break
    if not used:
        return symbols
    return [s for i, s in enumerate(symbols) if i not in used] + merged


def _merge_pm(symbols: list[DetectedSymbol]) -> list[DetectedSymbol]:
    """'+' with '-' below → 'pm'."""
    pluses = _by_symbol(symbols, "+")
    minuses = _by_symbol(symbols, "-")
    if not pluses or not minuses:
        return symbols
    used: set[int] = set()
    merged: list[DetectedSymbol] = []
    for pi, plus in pluses:
        for mi, m in minuses:
            if mi in used:
                continue
            if (
                _similar_width(plus, m)
                and _horizontally_aligned(plus, m)
                and m.bbox.cy > plus.bbox.cy
                and _vertically_stacked(plus, m, gap_tol=0.8)
            ):
                merged.append(_merged_symbol([plus, m], "pm"))
                used.add(pi)
                used.add(mi)
                break
    if not used:
        return symbols
    return [s for i, s in enumerate(symbols) if i not in used] + merged


def _merge_div(symbols: list[DetectedSymbol]) -> list[DetectedSymbol]:
    """'dot' above + '-' + 'dot' below → 'div'."""
    dots = _by_symbol(symbols, "dot")
    minuses = _by_symbol(symbols, "-")
    if len(dots) < 2 or not minuses:
        return symbols
    used: set[int] = set()
    merged: list[DetectedSymbol] = []
    for mi, m in minuses:
        if mi in used:
            continue
        above = None
        below = None
        for di, d in dots:
            if di in used:
                continue
            if not _horizontally_aligned(m, d, tol=0.5):
                continue
            if d.bbox.cy < m.bbox.cy and above is None:
                above = (di, d)
            elif d.bbox.cy > m.bbox.cy and below is None:
                below = (di, d)
        if above and below:
            merged.append(_merged_symbol([above[1], m, below[1]], "div"))
            used.add(mi)
            used.add(above[0])
            used.add(below[0])
    if not used:
        return symbols
    return [s for i, s in enumerate(symbols) if i not in used] + merged


# Order matters: merge '=' first so 'neq' can find it
_MERGE_RULES = [
    _merge_equal,
    _merge_neq,
    _merge_leq,
    _merge_geq,
    _merge_pm,
    _merge_div,
]
