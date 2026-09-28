"""Vectorised triangle edge table shared by the post-processing passes.

The orientation repair, the orientation report, the closed-shell contract and
the infinite-baffle contract each need to know which triangles share which edge
and in which direction each one traverses it. They used to rebuild that with a
pure-Python dict per pass. One ``np.unique`` builds the same table, with the
edges numbered in order of first appearance, which is the iteration order the
dict passes exposed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EdgeTable:
    """Per-use and per-edge views of a triangle list.

    A "use" is one traversal of an edge by one triangle; use ``3 * t + k`` is
    edge ``k`` of triangle ``t``, going from corner ``k`` to corner ``k + 1``.
    Unique edges are numbered in order of first use.
    """

    tri: NDArray[np.int64]  # triangle index of each use
    direction: NDArray[np.int8]  # +1 if traversed low -> high vertex id, else -1
    edge: NDArray[np.int64]  # unique-edge id of each use
    lo: NDArray[np.int64]  # per unique edge
    hi: NDArray[np.int64]  # per unique edge
    count: NDArray[np.int64]  # uses per unique edge
    first_use: NDArray[np.int64]  # index of the first use of each unique edge

    @property
    def n_edges(self) -> int:
        return int(len(self.count))

    def uses_of(self, edge_ids: NDArray[np.int64]) -> NDArray[np.int64]:
        """Use indices of the given unique edges, grouped by edge, in use order."""

        order = np.argsort(self.edge, kind="stable")
        sorted_edges = self.edge[order]
        wanted = np.zeros(self.n_edges, dtype=bool)
        wanted[np.asarray(edge_ids, dtype=np.int64)] = True
        return order[wanted[sorted_edges]]


def build_edge_table(
    triangles: NDArray[np.int64], *, drop_degenerate: bool = False
) -> EdgeTable:
    """Number the edges of ``triangles``.

    ``drop_degenerate`` skips uses whose two ends coincide (a collapsed edge),
    as the orientation passes always did; the closed-shell contract counted them.
    """

    tris = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    start = tris.reshape(-1)
    end = np.roll(tris, -1, axis=1).reshape(-1)
    tri = np.repeat(np.arange(len(tris), dtype=np.int64), 3)
    if drop_degenerate:
        keep = start != end
        start, end, tri = start[keep], end[keep], tri[keep]
    lo = np.minimum(start, end)
    hi = np.maximum(start, end)
    direction = np.where(start < end, 1, -1).astype(np.int8)
    if len(lo) == 0:
        empty = np.empty(0, dtype=np.int64)
        return EdgeTable(tri, direction, empty, empty, empty, empty, empty)
    span = int(hi.max()) + 1
    key = lo * span + hi
    unique_key, first_use, inverse, count = np.unique(
        key, return_index=True, return_inverse=True, return_counts=True
    )
    # np.unique numbers edges by sorted key; renumber by first appearance.
    by_appearance = np.argsort(first_use, kind="stable")
    rank = np.empty(len(unique_key), dtype=np.int64)
    rank[by_appearance] = np.arange(len(unique_key), dtype=np.int64)
    return EdgeTable(
        tri=tri,
        direction=direction,
        edge=rank[inverse.reshape(-1)],
        lo=(unique_key // span)[by_appearance],
        hi=(unique_key % span)[by_appearance],
        count=count[by_appearance].astype(np.int64),
        first_use=first_use[by_appearance].astype(np.int64),
    )
