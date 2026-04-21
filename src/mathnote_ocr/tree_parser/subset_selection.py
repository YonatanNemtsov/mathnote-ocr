"""Subset selection and spatial geometry utilities.

Functions for sampling subsets of symbols based on spatial proximity,
used by the subset model for both training and inference.

Geometry helpers:
  _bbox_centers, _bbox_edge_dist, _bbox_dist

Subset sampling (training):
  sample_subsets_spatial — random spatially-local subsets
  enumerate_subsets_exhaustive — all combinations of given sizes
  sample_subsets_with_coverage — spatial + guaranteed pair coverage

Inference:
  make_spatial_subsets — one deterministic subset per symbol (spatial radius)
  make_xaxis_subsets — x-sorted neighborhoods of varying sizes
"""

from __future__ import annotations

import math
import random

# ── Geometry helpers ─────────────────────────────────────────────────


def _bbox_centers(bboxes: list[list[float]]) -> list[tuple[float, float]]:
    """Compute center (cx, cy) for each [x, y, w, h] bbox."""
    return [(b[0] + b[2] / 2, b[1] + b[3] / 2) for b in bboxes]


def _bbox_edge_dist(b1: list[float], b2: list[float]) -> float:
    """Min distance between edges of two [x, y, w, h] bboxes.

    Returns 0 if bboxes overlap. Much better than center-to-center
    for wide symbols like fraction bars -- a bar that horizontally
    contains another symbol has near-zero edge distance to it.
    """
    # Horizontal gap (0 if overlapping)
    dx = max(0, max(b1[0], b2[0]) - min(b1[0] + b1[2], b2[0] + b2[2]))
    # Vertical gap (0 if overlapping)
    dy = max(0, max(b1[1], b2[1]) - min(b1[1] + b1[3], b2[1] + b2[3]))
    return math.hypot(dx, dy)


def _bbox_dist(c1: tuple[float, float], c2: tuple[float, float]) -> float:
    """Euclidean distance between two bbox centers."""
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


# ── Subset sampling (training) ──────────────────────────────────────


def sample_subsets_spatial(
    n_symbols: int,
    bboxes: list[list[float]],
    n_subsets: int = 50,
    min_size: int = 3,
    max_size: int = 6,
    seed_idx: int | None = None,
) -> list[list[int]]:
    """Sample spatially local subsets by seeding a symbol
    and growing outward to nearest neighbors.

    If seed_idx is given, use that symbol as the center seed for all
    subsets. Otherwise pick a random seed each time.
    Deduplicates results.
    """
    seen: set[tuple[int, ...]] = set()
    subsets: list[list[int]] = []
    all_indices = list(range(n_symbols))

    if n_symbols <= max_size:
        key = tuple(all_indices)
        seen.add(key)
        subsets.append(all_indices[:])

    for _ in range(n_subsets):
        k = random.randint(min(min_size, n_symbols), min(max_size, n_symbols))

        seed = seed_idx if seed_idx is not None else random.randrange(n_symbols)

        # Sort all other symbols by edge-to-edge bbox distance to seed
        dists = [
            (i, _bbox_edge_dist(bboxes[seed], bboxes[i])) for i in range(n_symbols) if i != seed
        ]
        dists.sort(key=lambda x: x[1])

        # Take k-1 nearest neighbors
        subset = sorted([seed] + [d[0] for d in dists[: k - 1]])
        key = tuple(subset)
        if key not in seen:
            seen.add(key)
            subsets.append(subset)

    return subsets


def enumerate_subsets_exhaustive(
    n_symbols: int,
    min_size: int = 2,
    max_size: int = 3,
) -> list[list[int]]:
    """Enumerate ALL subsets of each size from min_size to max_size.

    For size=2: C(N,2) pairs.
    For size=3: C(N,3) triples.
    etc.

    This guarantees every possible parent-child relationship is
    directly observed at least once.
    """
    from itertools import combinations

    subsets: list[list[int]] = []

    for size in range(min_size, max_size + 1):
        if n_symbols >= size:
            for combo in combinations(range(n_symbols), size):
                subsets.append(list(combo))

    return subsets


def sample_subsets_with_coverage(
    n_symbols: int,
    bboxes: list[list[float]],
    n_subsets: int = 50,
    min_size: int = 3,
    max_size: int = 6,
) -> list[list[int]]:
    """Sample spatially local subsets ensuring every pair appears at least once."""
    subsets = sample_subsets_spatial(n_symbols, bboxes, n_subsets, min_size, max_size)

    centers = _bbox_centers(bboxes)

    # Check pair coverage
    covered: set[tuple[int, int]] = set()
    for subset in subsets:
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                covered.add((subset[i], subset[j]))

    # Add targeted subsets for uncovered pairs
    for i in range(n_symbols):
        for j in range(i + 1, n_symbols):
            if (i, j) not in covered:
                others = [k for k in range(n_symbols) if k != i and k != j]
                mid = ((centers[i][0] + centers[j][0]) / 2, (centers[i][1] + centers[j][1]) / 2)
                others.sort(key=lambda k: _bbox_dist(centers[k], mid))
                n_extra = min(max_size - 2, len(others))
                extra = others[:n_extra]
                subset = sorted([i, j] + extra)
                subsets.append(subset)
                for a in range(len(subset)):
                    for b in range(a + 1, len(subset)):
                        covered.add((subset[a], subset[b]))

    return subsets


# ── Inference subset generation ──────────────────────────────────────


def make_neighborhood_subsets(
    bboxes: list[list[float]],
    k_neighbors: int = 8,
    min_size: int = 2,
    max_size: int = 6,
) -> list[list[int]]:
    """Enumerate all subsets within each symbol's k-nearest neighborhood.

    For each symbol, finds k nearest neighbors (by edge distance), forming
    a local neighborhood. Then enumerates all combinations of size
    min_size..max_size within that neighborhood. Results are deduplicated.

    This gives dense, uniform coverage of all local relationships without
    redundancy.
    """
    from itertools import combinations

    N = len(bboxes)
    if N <= max_size:
        return [list(range(N))]

    # Build neighborhoods
    neighborhoods: list[list[int]] = []
    for i in range(N):
        dists = []
        for j in range(N):
            if j == i:
                continue
            dists.append((j, _bbox_edge_dist(bboxes[i], bboxes[j])))
        dists.sort(key=lambda x: x[1])
        hood = [i] + [d[0] for d in dists[:k_neighbors]]
        neighborhoods.append(hood)

    # Enumerate subsets within each neighborhood, dedup globally
    seen: set[tuple[int, ...]] = set()
    subsets: list[list[int]] = []

    for hood in neighborhoods:
        for size in range(min_size, min(max_size, len(hood)) + 1):
            for combo in combinations(hood, size):
                key = tuple(sorted(combo))
                if key not in seen:
                    seen.add(key)
                    subsets.append(list(key))

    return subsets


def make_xaxis_subsets(
    bboxes: list[list[float]],
    min_size: int = 2,
    max_size: int = 8,
    **_kwargs,
) -> list[list[int]]:
    """X-axis neighborhood subsets of varying sizes.

    Sorts symbols by x-center, then for each consecutive run of symbols
    in x-order, also includes any symbol whose x-range overlaps with
    the run's x-span. This ensures wide symbols like fraction bars are
    included when their span covers the window.

    Args:
        bboxes: List of [x, y, w, h] bounding boxes.
        min_size: Minimum subset size (number of consecutive symbols).
        max_size: Maximum subset size.
    """
    N = len(bboxes)
    if N <= 2:
        return [list(range(N))]

    # Sort by x-center, tiebreak by y-center
    order = sorted(
        range(N),
        key=lambda i: (
            bboxes[i][0] + bboxes[i][2] / 2,
            bboxes[i][1] + bboxes[i][3] / 2,
        ),
    )

    # Precompute x-ranges [x_left, x_right] for each symbol
    x_ranges = [(bboxes[i][0], bboxes[i][0] + bboxes[i][2]) for i in range(N)]

    seen: set[tuple[int, ...]] = set()
    subsets: list[list[int]] = []

    for size in range(min_size, min(max_size, N) + 1):
        for start in range(N - size + 1):
            core = order[start : start + size]

            # Compute the x-span of this run
            x_lo = min(x_ranges[i][0] for i in core)
            x_hi = max(x_ranges[i][1] for i in core)

            # Include any symbol whose x-range overlaps with this span
            subset = set(core)
            for i in range(N):
                if i in subset:
                    continue
                sx, sx2 = x_ranges[i]
                if sx < x_hi and sx2 > x_lo:  # overlap
                    subset.add(i)

            subset = sorted(subset)
            if len(subset) > max_size:
                continue
            key = tuple(subset)
            if key not in seen:
                seen.add(key)
                subsets.append(subset)

    return subsets


def make_spatial_subsets(
    bboxes: list[list[float]],
    max_subset: int = 8,
    radius_mult: float | list[float] = 4.0,
) -> list[list[int]]:
    """Deterministic subsets per symbol, seeded by spatial proximity.

    For each symbol, finds neighbors within radius_mult * median_height,
    then clamps to [3, max_subset] neighbors.

    If radius_mult is a list (e.g. [2.5, 5.0]), generates one subset per
    symbol per radius — tight subsets for local context, wide subsets for
    distant relationships. Deduplicates across all radii.
    """
    N = len(bboxes)
    heights = [b[3] for b in bboxes]
    median_h = sorted(heights)[len(heights) // 2]

    if isinstance(radius_mult, (int, float)):
        radii = [radius_mult]
    else:
        radii = list(radius_mult)

    # Precompute distances once
    all_dists_by_seed = []
    for seed in range(N):
        dists = []
        for j in range(N):
            if j == seed:
                continue
            dists.append((j, _bbox_edge_dist(bboxes[seed], bboxes[j])))
        dists.sort(key=lambda x: x[1])
        all_dists_by_seed.append(dists)

    subsets = []
    for radius_m in radii:
        radius = radius_m * median_h
        for seed in range(N):
            dists = all_dists_by_seed[seed]
            neighbors = [d[0] for d in dists if d[1] <= radius]

            if len(neighbors) < 2:
                neighbors = [d[0] for d in dists[:2]]
            elif len(neighbors) > max_subset - 1:
                neighbors = neighbors[: max_subset - 1]

            subsets.append(sorted([seed] + neighbors))

    # Deduplicate (preserve order of first occurrence)
    seen: set[tuple[int, ...]] = set()
    unique = []
    for s in subsets:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique
