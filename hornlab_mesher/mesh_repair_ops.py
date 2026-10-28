"""Shared array operations for generated and imported mesh repair."""

from __future__ import annotations

import numpy as np

WELD_TOLERANCE_MM = 5.0e-3  # 5 micrometres; closes near-duplicate patch nodes
DEGENERATE_MIN_QUALITY = 1.0e-4  # drops needle slivers that make dense solves singular


def _remove_degenerate_triangles(
    points: np.ndarray,
    triangles: np.ndarray,
    tags: np.ndarray,
    *,
    eps: float = 1e-18,
    min_quality: float = 0.0,
    preserve_empty: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop zero-area triangles and, optionally, needle slivers.

    ``min_quality`` is a scale-invariant shape threshold: a triangle whose area
    falls below ``min_quality * longest_edge**2`` is removed. Fine OCC meshes
    carry micrometre-wide needles bridging near-duplicate patch-boundary nodes
    whose quadrature-degenerate rows make the dense metal-bem solve singular
    (LAPACK info > 0).
    """
    if preserve_empty and len(triangles) == 0:
        return triangles, tags, 0
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    area2 = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    keep = area2 > eps
    if min_quality > 0.0:
        longest_sq = np.maximum(
            np.maximum(
                np.sum((p1 - p0) ** 2, axis=1),
                np.sum((p2 - p1) ** 2, axis=1),
            ),
            np.sum((p0 - p2) ** 2, axis=1),
        )
        keep &= (0.5 * area2) > (min_quality * longest_sq)
    return triangles[keep], tags[keep], int(np.count_nonzero(~keep))


_WELD_FORWARD_NEIGHBOR_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) > (0, 0, 0)
)


def _weld_near_duplicate_vertices(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    tol_mm: float = WELD_TOLERANCE_MM,
    merge_when_unordered: bool = False,
) -> np.ndarray:
    """Remap triangle indices so vertices closer than ``tol_mm`` coincide.

    Uses a spatial hash with cells of the weld tolerance; clusters merge to
    the lowest vertex index. Orphaned duplicate vertices become unused and
    are dropped by the final compaction. Returns the remapped triangles.

    STEP repair historically requires ``distance <= tolerance``; the
    generated-mesh path rejects only ``distance > tolerance``. The option
    preserves that distinction when NaN makes the comparison unordered.
    """

    if len(points) == 0 or len(triangles) == 0:
        return triangles
    cells = np.floor(points / tol_mm).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(map(tuple, cells)):
        buckets.setdefault(key, []).append(index)

    parent = np.arange(len(points))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    tol_sq = tol_mm * tol_mm

    def merge_if_close(i: int, j: int) -> None:
        delta = points[j] - points[i]
        distance_sq = float(delta @ delta)
        if not (distance_sq <= tol_sq or (merge_when_unordered and not distance_sq > tol_sq)):
            return
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for key, indices in buckets.items():
        for position, i in enumerate(indices):
            for j in indices[position + 1 :]:
                merge_if_close(i, j)
        # Visit each pair of neighboring cells once. The old all-neighbor
        # traversal visited every pair from both cells and discarded half of
        # the candidates by vertex index.
        for dx, dy, dz in _WELD_FORWARD_NEIGHBOR_OFFSETS:
            others = buckets.get((key[0] + dx, key[1] + dy, key[2] + dz))
            if others is None:
                continue
            for i in indices:
                for j in others:
                    merge_if_close(i, j)

    roots = np.fromiter(
        (find(i) for i in range(len(points))), dtype=np.int64, count=len(points)
    )
    if np.array_equal(roots, np.arange(len(points))):
        return triangles
    return roots[triangles]


def _compact_unused_vertices(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    preserve_empty: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop unreferenced vertices and renumber triangles to the survivors."""
    if preserve_empty and len(triangles) == 0:
        return points, triangles
    used = np.unique(triangles)
    if len(used) == len(points):
        return points, triangles
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return points[used], remap[triangles]
