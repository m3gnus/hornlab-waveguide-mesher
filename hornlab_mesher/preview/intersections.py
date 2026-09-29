"""Intersection check for a folded preview wall and the acoustic skin."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _triangles_intersect(
    a: NDArray[np.float64], b: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Separating-axis test for paired triangles, including coplanar overlap."""
    ae = np.roll(a, -1, axis=1) - a
    be = np.roll(b, -1, axis=1) - b
    an = np.cross(ae[:, 0], ae[:, 1])
    bn = np.cross(be[:, 0], be[:, 1])
    axes = [an, bn]
    axes.extend(np.cross(ae[:, i], be[:, j]) for i in range(3) for j in range(3))
    # Face normals and edge cross-products vanish as separating axes for
    # coplanar triangles. Edge normals within the common plane cover that case.
    axes.extend(np.cross(ae[:, i], an) for i in range(3))
    axes.extend(np.cross(be[:, i], bn) for i in range(3))
    separated = np.zeros(len(a), dtype=bool)
    for axis in axes:
        length = np.linalg.norm(axis, axis=1)
        usable = length > 1.0e-14
        pa = np.einsum("nij,nj->ni", a, axis)
        pb = np.einsum("nij,nj->ni", b, axis)
        # The tolerance is in length units after projection; it scales with
        # the axis and avoids false gaps from roundoff at touching facets.
        tolerance = 1.0e-10 * length
        separated |= usable & ((pa.max(axis=1) < pb.min(axis=1) - tolerance)
                               | (pb.max(axis=1) < pa.min(axis=1) - tolerance))
    return ~separated


def folded_wall_crosses_inner(
    outer_positions: NDArray[np.float64], outer_indices: NDArray[np.uint32],
    folded: NDArray[np.bool_],
    inner_positions: NDArray[np.float64], inner_indices: NDArray[np.uint32],
) -> bool:
    """Check every outer facet that can reach the inner skin.

    A fold can cross the skin away from its reversed facets and their one-ring.
    Whole-surface and batch AABBs prune those regions before paired, vectorized
    axis tests. Bounded batches keep the pair matrix and narrow phase small.
    """
    outer_triangles = np.asarray(outer_indices).reshape(-1, 3)
    inner_triangles = np.asarray(inner_indices).reshape(-1, 3)
    if not np.any(folded) or not len(inner_triangles):
        return False
    outer = np.asarray(outer_positions)[outer_triangles]
    inner = np.asarray(inner_positions)[inner_triangles]
    inner_lo, inner_hi = inner.min(axis=1), inner.max(axis=1)
    outer_lo, outer_hi = outer.min(axis=1), outer.max(axis=1)
    tolerance = 1.0e-10
    reachable = (np.all(outer_lo <= inner_hi.max(axis=0) + tolerance, axis=1)
                 & np.all(inner_lo.min(axis=0) <= outer_hi + tolerance, axis=1))
    outer = outer[reachable]
    for offset in range(0, len(outer), 64):
        batch = outer[offset:offset + 64]
        lo, hi = batch.min(axis=1), batch.max(axis=1)
        nearby = (np.all(inner_lo <= hi.max(axis=0) + tolerance, axis=1)
                  & np.all(lo.min(axis=0) <= inner_hi + tolerance, axis=1))
        if not np.any(nearby):
            continue
        candidates = inner[nearby]
        oi, ii = np.nonzero(
            np.all(lo[:, None] <= inner_hi[nearby][None] + tolerance, axis=2)
            & np.all(inner_lo[nearby][None] <= hi[:, None] + tolerance, axis=2)
        )
        for start in range(0, len(oi), 4096):
            pairs = slice(start, start + 4096)
            if np.any(_triangles_intersect(batch[oi[pairs]], candidates[ii[pairs]])):
                return True
    return False
