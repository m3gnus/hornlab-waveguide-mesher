from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from .tags import PhysicalGroup


class MeshOrientationError(ValueError):
    """Raised when mesh orientation validation fails."""


@dataclass(frozen=True)
class MeshOrientationReport:
    n_triangles: int
    n_edges: int
    boundary_edges: int
    nonmanifold_edges: int
    inconsistent_edges: int
    signed_volume: float
    source_normal_projection: float

    @property
    def watertight(self) -> bool:
        return self.boundary_edges == 0 and self.nonmanifold_edges == 0

    @property
    def edge_consistent(self) -> bool:
        return self.nonmanifold_edges == 0 and self.inconsistent_edges == 0


def _source_axis_index_and_sign(source_axis: str) -> tuple[int, float, str]:
    axis = str(source_axis or "z").strip().lower()
    sign = -1.0 if axis.startswith("-") else 1.0
    axis = axis[1:] if axis[:1] in {"+", "-"} else axis
    if axis not in {"x", "y", "z"}:
        axis = "z"
    return {"x": 0, "y": 1, "z": 2}[axis], sign, ("-" if sign < 0.0 else "") + axis


def remove_degenerate_triangles(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    tags: NDArray[np.int32],
    *,
    eps: float = 1e-18,
    min_quality: float = 0.0,
) -> tuple[NDArray[np.int64], NDArray[np.int32], int]:
    """Drop zero-area triangles and, optionally, needle slivers.

    ``min_quality`` is a scale-invariant shape threshold: triangles whose
    area falls below ``min_quality * longest_edge**2`` are removed. Gmsh can
    emit needle triangles bridging near-duplicate OCC patch-boundary nodes
    (observed: micrometre-wide needles on fine grids) whose
    quadrature-degenerate rows make dense BEM solves singular.
    """

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


def validate_orientation(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    tags: NDArray[np.int32],
    *,
    source_axis: str = "z",
    require_watertight: bool = False,
    require_edge_consistency: bool = False,
    require_positive_volume: bool = True,
    require_source_normal: bool = True,
    eps: float = 1e-12,
) -> MeshOrientationReport:
    """Validate triangle winding without mutating the mesh.

    The report always includes watertightness, edge consistency, signed volume,
    and primary-source normal diagnostics. Callers choose which diagnostics are
    hard failures so open-but-valid canonical surfaces can still be checked
    without receiving post-hoc winding repairs.
    """

    if len(triangles) == 0:
        raise MeshOrientationError("mesh contains no triangles")
    if len(tags) != len(triangles):
        raise MeshOrientationError("triangle and physical-tag counts differ")

    edge_dirs: dict[tuple[int, int], list[int]] = defaultdict(list)
    for tri in np.asarray(triangles, dtype=np.int64):
        for start, end in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a = int(start)
            b = int(end)
            if a == b:
                continue
            if a < b:
                edge_dirs[(a, b)].append(1)
            else:
                edge_dirs[(b, a)].append(-1)

    boundary_edges = 0
    nonmanifold_edges = 0
    inconsistent_edges = 0
    for dirs in edge_dirs.values():
        if len(dirs) == 1:
            boundary_edges += 1
        elif len(dirs) != 2:
            nonmanifold_edges += 1
        elif dirs[0] == dirs[1]:
            inconsistent_edges += 1

    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    signed_volume = float(np.sum(p0 * np.cross(p1, p2)) / 6.0)

    axis_idx, axis_sign, axis_label = _source_axis_index_and_sign(source_axis)
    source_mask = tags == int(PhysicalGroup.PRIMARY_SOURCE)
    source_projection = 0.0
    if np.any(source_mask):
        s0 = points[triangles[source_mask, 0]]
        s1 = points[triangles[source_mask, 1]]
        s2 = points[triangles[source_mask, 2]]
        source_projection = float(
            axis_sign * np.sum(np.cross(s1 - s0, s2 - s0)[:, axis_idx])
        )

    report = MeshOrientationReport(
        n_triangles=int(len(triangles)),
        n_edges=int(len(edge_dirs)),
        boundary_edges=int(boundary_edges),
        nonmanifold_edges=int(nonmanifold_edges),
        inconsistent_edges=int(inconsistent_edges),
        signed_volume=signed_volume,
        source_normal_projection=source_projection,
    )

    failures: list[str] = []
    if require_watertight and not report.watertight:
        failures.append(
            f"mesh is not watertight: {report.boundary_edges} boundary edges, "
            f"{report.nonmanifold_edges} nonmanifold edges"
        )
    elif report.nonmanifold_edges:
        failures.append(f"mesh has {report.nonmanifold_edges} nonmanifold edges")
    if require_edge_consistency and not report.edge_consistent:
        failures.append(f"mesh has {report.inconsistent_edges} inconsistent shared edges")
    if require_positive_volume and report.watertight and report.signed_volume < -eps:
        failures.append(f"mesh signed volume is negative ({report.signed_volume:.6g})")
    if require_source_normal and np.any(source_mask) and report.source_normal_projection < -eps:
        failures.append(
            "primary source normals point opposite "
            f"{axis_label}-axis ({report.source_normal_projection:.6g})"
        )
    if failures:
        raise MeshOrientationError("; ".join(failures))
    return report


def repair_orientation(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    tags: NDArray[np.int32],
    *,
    source_axis: str = "z",
) -> tuple[NDArray[np.int64], dict[str, int]]:
    """Repair triangle winding for generated meshes or opt-in imports."""

    repaired = triangles.copy()
    stats = {
        "flipped_consistency": 0,
        "flipped_global": 0,
        "flipped_primary_source": 0,
    }
    if len(repaired) == 0:
        return repaired, stats

    edge_to_triangles: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for tri_idx, tri in enumerate(repaired):
        for start, end in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a = int(start)
            b = int(end)
            if a == b:
                continue
            if a < b:
                edge_to_triangles[(a, b)].append((tri_idx, 1))
            else:
                edge_to_triangles[(b, a)].append((tri_idx, -1))

    neighbours: list[list[tuple[int, bool]]] = [[] for _ in range(len(repaired))]
    for uses in edge_to_triangles.values():
        if len(uses) != 2:
            continue
        (ta, da), (tb, db) = uses
        # Adjacent triangles are edge-consistent when they traverse their
        # shared edge in opposite directions. If their current directions
        # match, exactly one side of the pair must be flipped.
        must_differ = da == db
        neighbours[ta].append((tb, must_differ))
        neighbours[tb].append((ta, must_differ))

    flip = np.zeros(len(repaired), dtype=bool)
    seen = np.zeros(len(repaired), dtype=bool)
    for seed in range(len(repaired)):
        if seen[seed]:
            continue
        seen[seed] = True
        stack = [seed]
        while stack:
            tri_idx = stack.pop()
            for other, must_differ in neighbours[tri_idx]:
                required = bool(flip[tri_idx]) ^ bool(must_differ)
                if seen[other]:
                    continue
                flip[other] = required
                seen[other] = True
                stack.append(other)

    if np.any(flip):
        repaired[flip] = repaired[flip][:, [0, 2, 1]]
        stats["flipped_consistency"] = int(np.count_nonzero(flip))

    p0 = points[repaired[:, 0]]
    p1 = points[repaired[:, 1]]
    p2 = points[repaired[:, 2]]
    signed = float(np.sum(p0 * np.cross(p1, p2)))
    if signed < 0.0:
        repaired[:, [1, 2]] = repaired[:, [2, 1]]
        stats["flipped_global"] = int(len(repaired))

    axis_idx, axis_sign, _axis_label = _source_axis_index_and_sign(source_axis)
    mask = tags == int(PhysicalGroup.PRIMARY_SOURCE)
    if np.any(mask):
        p0 = points[repaired[mask, 0]]
        p1 = points[repaired[mask, 1]]
        p2 = points[repaired[mask, 2]]
        projection = float(axis_sign * np.sum(np.cross(p1 - p0, p2 - p0)[:, axis_idx]))
        if projection < 0.0:
            # Flip the source's whole connected component(s), not just the
            # tagged triangles: flipping only the source of an edge-connected
            # mesh manufactures the inconsistent shared edges the validator
            # then rejects. A detached source cap (its own component) reduces
            # to the old behavior; a source welded to walls flips with them,
            # and if that contradicts the global volume the validator reports
            # a genuinely defective geometry instead of a self-inflicted one.
            component = np.zeros(len(repaired), dtype=bool)
            stack = list(np.where(mask)[0])
            component[stack] = True
            while stack:
                tri_idx = stack.pop()
                for other, _must_differ in neighbours[tri_idx]:
                    if not component[other]:
                        component[other] = True
                        stack.append(other)
            idx = np.where(component)[0]
            repaired[idx] = repaired[idx][:, [0, 2, 1]]
            stats["flipped_primary_source"] = int(len(idx))

    return repaired, stats
