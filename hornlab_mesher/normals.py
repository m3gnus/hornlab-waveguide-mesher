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
    open_shell_bore_alignment: float | None = None

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


def open_shell_wall_orientation_references(
    points: NDArray[np.float64],
    *,
    closed: bool,
    max_phi_samples: int = 8,
    max_axial_samples: int = 8,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sample desired bare-wall normals from a point-grid parameterisation.

    Point-grid rows advance in positive azimuth and columns advance from throat
    to mouth. Their ``axial x azimuth`` cross product points out of the wall
    material and into the acoustic bore. Sparse cell-centre samples preserve
    that semantic side through rollbacks, morphs, rotations, and reduced
    domains without inferring a frame from an asymmetric source cap.
    """

    grid = np.asarray(points, dtype=np.float64)
    if grid.ndim != 3 or grid.shape[2] != 3:
        raise ValueError("open-shell wall grid must have shape (n_phi, n_length, 3)")
    n_phi, n_length, _ = grid.shape
    phi_count = n_phi if closed else n_phi - 1
    axial_count = n_length - 1
    if phi_count < 1 or axial_count < 1:
        raise ValueError("open-shell wall grid needs an angular and an axial span")

    def sampled_indices(count: int, limit: int) -> NDArray[np.int64]:
        sample_count = min(count, max(1, int(limit)))
        return np.unique(
            np.linspace(0, count - 1, sample_count, dtype=np.int64)
        )

    reference_points: list[NDArray[np.float64]] = []
    reference_normals: list[NDArray[np.float64]] = []
    for i in sampled_indices(phi_count, max_phi_samples):
        ni = (int(i) + 1) % n_phi
        for j in sampled_indices(axial_count, max_axial_samples):
            p00 = grid[int(i), int(j)]
            p10 = grid[ni, int(j)]
            p01 = grid[int(i), int(j) + 1]
            p11 = grid[ni, int(j) + 1]
            angular = 0.5 * ((p10 - p00) + (p11 - p01))
            axial = 0.5 * ((p01 - p00) + (p11 - p10))
            normal = np.cross(axial, angular)
            length = float(np.linalg.norm(normal))
            if length <= 1.0e-12:
                continue
            reference_points.append(0.25 * (p00 + p10 + p01 + p11))
            reference_normals.append(normal / length)

    if not reference_points:
        raise ValueError("open-shell wall grid produced no valid orientation samples")
    return (
        np.asarray(reference_points, dtype=np.float64),
        np.asarray(reference_normals, dtype=np.float64),
    )


def open_shell_bore_alignment(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    tags: NDArray[np.int32],
    *,
    band_fraction: float = 0.2,
    source_tags: set[int] | None = None,
    wall_tags: set[int] | None = None,
) -> float | None:
    """Return the near-throat wall area fraction whose normal faces the bore.

    Signed volume cannot express this contract. It is not translation
    invariant on an open surface, and even about a fixed material point it
    reports the *wrong* sign for correctly wound rollback profiles, whose wall
    curls back past the mouth plane (measured: R-OSSE and ICW rollbacks both
    come out positive while every wall normal faces the bore). Symmetry-reduced
    bare shells flip it too, depending on which point the sum is taken about.

    What is well defined without a closed interior is the throat collar. There
    the wall is a monotonically flaring tube around the source cap, so an
    inward normal is exactly a normal with a negative radial component about
    the cap's own axis. Both references come from the mesh itself -- the cap's
    area-weighted centroid and its net area vector -- so the measure survives
    translation, rotation, vertical offsets and reduced domains. The band is
    kept near the throat because a rollback deliberately reverses the
    bore-facing normal's radial component further out.

    Returns ``None`` when the mesh carries no source, no wall, a cap whose area
    vectors cancel, or no nondegenerate axial wall collar -- cases this measure
    cannot judge.

    ``source_tags`` and ``wall_tags`` name which physical tags play those two
    roles. They default to this package's own canonical groups, so a caller
    that omits them gets exactly the behaviour this function has always had.
    STEP import cannot use the canonical groups: its tags are allocated per
    imported source by the caller, so ``PRIMARY_SOURCE`` names nothing there
    and a multi-source import carries tags 3 and up that the canonical mask
    would drop silently.
    """

    resolved_sources = (
        {int(PhysicalGroup.PRIMARY_SOURCE)} if source_tags is None else {int(t) for t in source_tags}
    )
    resolved_walls = (
        {int(PhysicalGroup.RIGID_WALL)} if wall_tags is None else {int(t) for t in wall_tags}
    )
    if not resolved_sources or not resolved_walls:
        return None
    wall_mask = np.isin(tags, tuple(resolved_walls))
    source_mask = np.isin(tags, tuple(resolved_sources))
    if not np.any(wall_mask) or not np.any(source_mask):
        return None

    # More than one declared cap in the same component makes the references
    # below meaningless, so decline rather than average them. `cap_centroid` is
    # the area-weighted centroid of every cap at once: with a small throat cap
    # and a large mouth cap it lands nowhere near either axis, and the radial
    # test is then measured about a line outside the body. Measured on a flared
    # half-horn with caps 2 and 3 declared and a 400 mm mouth, an INVERTED mesh
    # read 0.867 -- a confident verdict, in the wrong direction, which is worse
    # than no verdict. One cap is the case this collar was derived for.
    if len(np.unique(tags[source_mask])) > 1:
        return None

    s0 = points[triangles[source_mask, 0]]
    s1 = points[triangles[source_mask, 1]]
    s2 = points[triangles[source_mask, 2]]
    source_area_vectors = np.cross(s1 - s0, s2 - s0)
    source_areas = 0.5 * np.linalg.norm(source_area_vectors, axis=1)
    if not np.any(source_areas > 0.0):
        return None
    cap_centroid = np.average(
        (s0 + s1 + s2) / 3.0, weights=source_areas, axis=0
    )
    axis = source_area_vectors.sum(axis=0)
    axis_length = float(np.linalg.norm(axis))
    if axis_length <= 1.0e-12:
        return None
    axis = axis / axis_length

    w0 = points[triangles[wall_mask, 0]]
    w1 = points[triangles[wall_mask, 1]]
    w2 = points[triangles[wall_mask, 2]]
    wall_area_vectors = np.cross(w1 - w0, w2 - w0)
    wall_areas = 0.5 * np.linalg.norm(wall_area_vectors, axis=1)
    offsets = (w0 + w1 + w2) / 3.0 - cap_centroid
    axial = offsets @ axis
    radial = offsets - np.outer(axial, axis)
    radial_lengths = np.linalg.norm(radial, axis=1)

    span = float(axial.max() - axial.min())
    if span <= 0.0:
        return None
    band = axial <= axial.min() + float(band_fraction) * span
    band &= (radial_lengths > 1.0e-12) & (wall_areas > 0.0)
    if not np.any(band):
        return None

    inward = -radial[band] / radial_lengths[band, None]
    facing_bore = np.sum(wall_area_vectors[band] * inward, axis=1) > 0.0
    banded_areas = wall_areas[band]
    return float(np.sum(banded_areas[facing_bore]) / np.sum(banded_areas))


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
    require_open_shell_bore_normal: bool = False,
    open_shell_bore_tolerance: float = 0.9,
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
        open_shell_bore_alignment=open_shell_bore_alignment(
            points, triangles, tags
        ),
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
    if require_open_shell_bore_normal:
        alignment = report.open_shell_bore_alignment
        if alignment is None:
            failures.append(
                "bare open shell has no measurable throat collar to orient "
                "(missing or degenerate primary source or rigid wall)"
            )
        elif alignment < float(open_shell_bore_tolerance):
            failures.append(
                "bare open-shell wall normals do not face the bore: only "
                f"{alignment:.3f} of near-throat wall area points inward "
                f"(need >= {float(open_shell_bore_tolerance):.3f})"
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
    open_shell_wall_points_mm: NDArray[np.float64] | None = None,
    open_shell_wall_normals: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.int64], dict[str, int]]:
    """Repair triangle winding for generated meshes or opt-in imports."""

    repaired = triangles.copy()
    stats = {
        "flipped_consistency": 0,
        "flipped_global": 0,
        "flipped_primary_source": 0,
        "flipped_open_shell_wall": 0,
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
    components: list[NDArray[np.int64]] = []
    for seed in range(len(repaired)):
        if seen[seed]:
            continue
        seen[seed] = True
        stack = [seed]
        component: list[int] = []
        while stack:
            tri_idx = stack.pop()
            component.append(tri_idx)
            for other, must_differ in neighbours[tri_idx]:
                required = bool(flip[tri_idx]) ^ bool(must_differ)
                if seen[other]:
                    continue
                flip[other] = required
                seen[other] = True
                stack.append(other)
        components.append(np.asarray(component, dtype=np.int64))

    if np.any(flip):
        repaired[flip] = repaired[flip][:, [0, 2, 1]]
        stats["flipped_consistency"] = int(np.count_nonzero(flip))

    open_shell_requested = (
        open_shell_wall_points_mm is not None
        or open_shell_wall_normals is not None
    )
    if open_shell_requested:
        if open_shell_wall_points_mm is None or open_shell_wall_normals is None:
            raise ValueError(
                "open-shell wall orientation requires both reference points and normals"
            )
        reference_points = np.asarray(
            open_shell_wall_points_mm, dtype=np.float64
        )
        reference_normals = np.asarray(
            open_shell_wall_normals, dtype=np.float64
        )
        if (
            reference_points.ndim != 2
            or reference_points.shape[1] != 3
            or reference_normals.shape != reference_points.shape
            or len(reference_points) == 0
            or not np.all(np.isfinite(reference_points))
            or not np.all(np.isfinite(reference_normals))
        ):
            raise ValueError(
                "open-shell wall orientation references must be finite (N, 3) arrays"
            )
        reference_lengths = np.linalg.norm(reference_normals, axis=1)
        if np.any(reference_lengths <= 1.0e-12):
            raise ValueError("open-shell wall orientation contains a zero normal")
        reference_normals = reference_normals / reference_lengths[:, None]

        component_ids = np.empty(len(repaired), dtype=np.int64)
        for component_id, component in enumerate(components):
            component_ids[component] = component_id

        wall_mask = tags == int(PhysicalGroup.RIGID_WALL)
        wall_indices = np.flatnonzero(wall_mask)
        wall_votes: dict[int, list[float]] = defaultdict(list)
        if len(wall_indices):
            w0 = points[repaired[wall_indices, 0]]
            w1 = points[repaired[wall_indices, 1]]
            w2 = points[repaired[wall_indices, 2]]
            wall_centroids = (w0 + w1 + w2) / 3.0
            wall_area_vectors = np.cross(w1 - w0, w2 - w0)
            wall_lengths = np.linalg.norm(wall_area_vectors, axis=1)
            valid_wall = wall_lengths > 1.0e-18
            wall_unit_normals = np.zeros_like(wall_area_vectors)
            wall_unit_normals[valid_wall] = (
                wall_area_vectors[valid_wall] / wall_lengths[valid_wall, None]
            )
            # Each builder sample votes only on its globally nearest wall face.
            # This assigns samples to the correct disconnected OCC component
            # without guessing from triangle order or a reduced-domain centroid.
            for reference_point, reference_normal in zip(
                reference_points, reference_normals, strict=True
            ):
                distances_sq = np.sum(
                    (wall_centroids - reference_point) ** 2, axis=1
                )
                local_index = int(np.argmin(distances_sq))
                if not valid_wall[local_index]:
                    continue
                triangle_index = int(wall_indices[local_index])
                component_id = int(component_ids[triangle_index])
                wall_votes[component_id].append(
                    float(wall_unit_normals[local_index] @ reference_normal)
                )

        axis_idx, axis_sign, _axis_label = _source_axis_index_and_sign(source_axis)
        source_mask = tags == int(PhysicalGroup.PRIMARY_SOURCE)
        for component_id, component in enumerate(components):
            component_source = component[source_mask[component]]
            component_wall = component[wall_mask[component]]
            source_vote: float | None = None
            wall_vote: float | None = None
            if len(component_source):
                s0 = points[repaired[component_source, 0]]
                s1 = points[repaired[component_source, 1]]
                s2 = points[repaired[component_source, 2]]
                source_vote = float(
                    axis_sign
                    * np.sum(np.cross(s1 - s0, s2 - s0)[:, axis_idx])
                )
            if len(component_wall):
                votes = wall_votes.get(component_id, [])
                if not votes:
                    raise MeshOrientationError(
                        "open-shell rigid-wall component has no parameterisation "
                        "orientation reference"
                    )
                wall_vote = float(np.median(np.asarray(votes, dtype=np.float64)))
                if abs(wall_vote) <= 1.0e-6:
                    raise MeshOrientationError(
                        "open-shell rigid-wall orientation reference is tangent "
                        "to its nearest mesh face"
                    )

            source_flip = source_vote is not None and source_vote < 0.0
            wall_flip = wall_vote is not None and wall_vote < 0.0
            if (
                source_vote is not None
                and wall_vote is not None
                and source_flip != wall_flip
            ):
                raise MeshOrientationError(
                    "open-shell source and rigid-wall parameterisation require "
                    "opposite component flips"
                )
            should_flip = source_flip if source_vote is not None else wall_flip
            if should_flip:
                repaired[component] = repaired[component][:, [0, 2, 1]]
                if source_vote is not None:
                    stats["flipped_primary_source"] += int(len(component))
                else:
                    stats["flipped_open_shell_wall"] += int(len(component))
        return repaired, stats

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
