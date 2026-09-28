"""Read-only checks on a tagged surface mesh: topology, symmetry rims, frequency.

Coordinates are in the caller's "step units"; functions that apply a
millimetre quantity take ``unit_scale_to_m`` and convert through it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from .step_mapping import RIGID_TAG, StepFaceGroup
from .step_prepare import millimetres_to_step_units

SPEED_OF_SOUND_M_S = 343.0
FREQUENCY_ELEMENTS_PER_WAVELENGTH = 6.0


def _triangle_area2(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    return np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def _edge_direction_stats(triangles: np.ndarray) -> dict[str, object]:
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

    return {
        "n_edges": int(len(edge_dirs)),
        "boundary_edges": int(boundary_edges),
        "free_edges": int(boundary_edges),
        "nonmanifold_edges": int(nonmanifold_edges),
        "inconsistent_edges": int(inconsistent_edges),
    }


def _signed_volume(points: np.ndarray, triangles: np.ndarray) -> float:
    if len(triangles) == 0:
        return 0.0
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    return float(np.sum(p0 * np.cross(p1, p2)) / 6.0)


#: Fraction of a component's own ``area ** 1.5`` below which its signed volume
#: is rounding noise rather than an orientation. The sum is taken over
#: coordinates, so its absolute error scales with the component's size and
#: triangle count, and a near-degenerate component -- a sliver panel, a shell
#: whose two cut planes almost coincide -- encloses so little that the sign is
#: whatever the accumulation happened to leave. Exactly ``0.0`` is therefore not
#: the only unresolved case, and flipping on a noise sign would silently invert
#: normals that the solver then trusts. A genuinely reduced closed shell sits
#: many orders of magnitude above this: a half-box scores ~0.09.
SIGNED_VOLUME_NOISE_REL = 1.0e-9


def _signed_volume_noise_floor(points: np.ndarray, triangles: np.ndarray) -> float:
    """Magnitude below which ``_signed_volume`` cannot be read as a sign."""

    if len(triangles) == 0:
        return 0.0
    area = 0.5 * float(np.sum(_triangle_area2(points, triangles)))
    return SIGNED_VOLUME_NOISE_REL * area**1.5


def _edge_on_expected_plane(
    points: np.ndarray,
    edge: tuple[int, int],
    planes: Iterable[str],
    tol: float,
) -> bool:
    for plane in planes:
        axis = {"x0": 0, "y0": 1, "z0": 2}[plane]
        if all(abs(float(points[vertex, axis])) <= tol for vertex in edge):
            return True
    return False

def detect_symmetry_planes(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    tolerance: float,
    min_edges_per_plane: int = 3,
) -> tuple[tuple[str, ...], dict[str, object]]:
    """Detect symmetry cut planes from free edges lying on x=0/y=0/z=0.

    Only free edges lying exclusively on a single coordinate plane count
    toward that plane. A cut rim in the x=0 wall crossing height z=0
    contributes edges that sit on both planes at once; counting those toward
    z0 would misread an internal level as a cut plane. A true cut outline
    always has edges away from the other coordinate planes, so the exclusive
    count stays robust. ``min_edges_per_plane`` additionally keeps an
    isolated leak vertex near the origin from masquerading as a plane. A
    candidate must also have the whole mesh on one side of it: stray free
    edges along an internal origin plane cannot turn a full-span model into a
    native symmetry-reduced solve.
    """
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    edges = np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    if len(edges):
        _unique, first, counts = np.unique(
            edges, axis=0, return_index=True, return_counts=True
        )
        free_rows = first[counts == 1]
    else:
        free_rows = np.zeros(0, dtype=np.int64)
    free_edges = edges[free_rows]
    free_edge_triangles = free_rows // 3

    on_plane = np.stack(
        [np.all(np.abs(points[free_edges, axis]) <= tolerance, axis=1) for axis in range(3)],
        axis=1,
    ) if len(free_edges) else np.zeros((0, 3), dtype=bool)
    planes_per_edge = on_plane.sum(axis=1)
    plane_counts = {
        plane: int(np.count_nonzero(on_plane[:, axis] & (planes_per_edge == 1)))
        for axis, plane in enumerate(("x0", "y0", "z0"))
    }
    shared_plane_edges = int(np.count_nonzero(planes_per_edge > 1))

    # How squarely the surface meets each plane along its rim: the median
    # |normal . plane normal| of the triangles owning the plane's free edges.
    # A mirror-symmetric smooth surface crosses its mirror plane at a right
    # angle (0); a horn mouth that merely ends on the plane meets it at its
    # flare angle. Reported only: it is evidence for a caller deciding whether
    # an undeclared open rim is a cut, never a verdict of this detector.
    rim_normal_component: dict[str, float | None] = {}
    for axis, plane in enumerate(("x0", "y0", "z0")):
        rows = free_edge_triangles[on_plane[:, axis] & (planes_per_edge == 1)] if len(free_edges) else []
        if len(rows) == 0:
            rim_normal_component[plane] = None
            continue
        owners = triangles[rows]
        normals = np.cross(
            points[owners[:, 1]] - points[owners[:, 0]],
            points[owners[:, 2]] - points[owners[:, 0]],
        )
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 0.0
        rim_normal_component[plane] = (
            float(np.median(np.abs(normals[valid, axis]) / lengths[valid]))
            if np.any(valid)
            else None
        )

    plane_vertex_side_counts: dict[str, dict[str, int]] = {}
    one_sided: dict[str, bool] = {}
    for axis, plane in enumerate(("x0", "y0", "z0")):
        coordinates = points[:, axis]
        negative = int(np.count_nonzero(coordinates < -tolerance))
        positive = int(np.count_nonzero(coordinates > tolerance))
        plane_vertex_side_counts[plane] = {
            "negative": negative,
            "on_plane": int(len(coordinates) - negative - positive),
            "positive": positive,
        }
        one_sided[plane] = not (negative and positive)

    detected = tuple(
        plane for plane in ("x0", "y0", "z0")
        if plane_counts[plane] >= min_edges_per_plane and one_sided[plane]
    )
    rejected_spanning_planes = [
        plane for plane in ("x0", "y0", "z0")
        if plane_counts[plane] >= min_edges_per_plane and not one_sided[plane]
    ]
    detection = {
        "mode": "auto",
        "free_edges": int(len(free_edges)),
        "plane_free_edge_counts": {k: int(v) for k, v in plane_counts.items()},
        "plane_vertex_side_counts": plane_vertex_side_counts,
        "rejected_spanning_planes": rejected_spanning_planes,
        "shared_plane_free_edges": int(shared_plane_edges),
        "rim_normal_component": rim_normal_component,
        "min_edges_per_plane": int(min_edges_per_plane),
        "tolerance": float(tolerance),
        "detected_planes": list(detected),
    }
    return detected, detection


def _triangle_edge_lengths(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=np.int64)
    if len(triangles) == 0:
        return np.empty(0, dtype=np.float64)
    edges = triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    edges.sort(axis=1)
    _unique_edges, first_indices = np.unique(edges, axis=0, return_index=True)
    unique_edges = edges[np.sort(first_indices)]
    return np.linalg.norm(
        points[unique_edges[:, 0]] - points[unique_edges[:, 1]],
        axis=1,
    )


def _edge_frequency_stats(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    unit_scale_to_m: float,
    elements_per_wavelength: float,
    speed_of_sound_m_s: float,
) -> dict[str, object]:
    lengths = _triangle_edge_lengths(points, triangles)
    max_edge_step_units = float(np.max(lengths)) if len(lengths) else 0.0
    p95_edge_step_units = float(np.percentile(lengths, 95.0)) if len(lengths) else 0.0
    max_edge_m = max_edge_step_units * unit_scale_to_m
    max_valid_frequency_hz = (
        speed_of_sound_m_s / (elements_per_wavelength * max_edge_m)
        if max_edge_m > 0.0
        else 0.0
    )
    return {
        "max_edge_step_units": max_edge_step_units,
        "max_edge_m": float(max_edge_m),
        "p95_edge_step_units": p95_edge_step_units,
        "p95_edge_m": float(p95_edge_step_units * unit_scale_to_m),
        "max_valid_frequency_hz": float(max_valid_frequency_hz),
    }


def _source_wall_stats(
    points: np.ndarray,
    triangles: np.ndarray,
    tags: np.ndarray,
    spec: StepFaceGroup,
    *,
    transition_mm: float,
    unit_scale_to_m: float,
    elements_per_wavelength: float,
    speed_of_sound_m_s: float,
) -> dict[str, object] | None:
    """Edge statistics for rigid triangles near a source patch.

    The wave launched by a source travels along the surrounding rigid
    surfaces, so the usable solve band of that source is limited by the
    rigid mesh it traverses, not only by the source patch itself. Rigid
    triangles whose centroid lies within the source refinement transition
    distance are taken as the local wall region.
    """
    source_mask = tags == spec.tag
    rigid_mask = tags == RIGID_TAG
    if not np.any(source_mask) or not np.any(rigid_mask):
        return None
    patch_vertices = points[np.unique(triangles[source_mask])]
    rigid_triangles = triangles[rigid_mask]
    centroids = points[rigid_triangles].mean(axis=1)
    min_distance = np.full(len(centroids), np.inf)
    for start in range(0, len(patch_vertices), 512):
        chunk = patch_vertices[start:start + 512]
        distances = np.linalg.norm(
            centroids[:, None, :] - chunk[None, :, :],
            axis=2,
        ).min(axis=1)
        min_distance = np.minimum(min_distance, distances)
    transition = millimetres_to_step_units(transition_mm, unit_scale_to_m)
    near_triangles = rigid_triangles[min_distance <= transition]
    if len(near_triangles) == 0:
        return None
    stats = _edge_frequency_stats(
        points,
        near_triangles,
        unit_scale_to_m=unit_scale_to_m,
        elements_per_wavelength=elements_per_wavelength,
        speed_of_sound_m_s=speed_of_sound_m_s,
    )
    return {
        "wall_triangle_count": int(len(near_triangles)),
        "wall_distance_mm": float(transition_mm),
        "wall_max_edge_step_units": float(stats["max_edge_step_units"]),
        "wall_max_edge_m": float(stats["max_edge_m"]),
        "wall_p95_edge_step_units": float(stats["p95_edge_step_units"]),
        "wall_p95_edge_m": float(stats["p95_edge_m"]),
        "wall_max_valid_frequency_hz": float(stats["max_valid_frequency_hz"]),
    }


def mesh_frequency_validation(
    points: np.ndarray,
    triangles: np.ndarray,
    tags: np.ndarray,
    source_specs: list[StepFaceGroup],
    *,
    unit_scale_to_m: float,
    requested_max_frequency_hz: float | None,
    transition_mm: float = 200.0,
    elements_per_wavelength: float = FREQUENCY_ELEMENTS_PER_WAVELENGTH,
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
) -> dict[str, object]:
    """Report conservative global and explicit-group frequency limits."""
    global_stats = _edge_frequency_stats(
        points,
        triangles,
        unit_scale_to_m=unit_scale_to_m,
        elements_per_wavelength=elements_per_wavelength,
        speed_of_sound_m_s=speed_of_sound_m_s,
    )
    edge_limit_m = (
        speed_of_sound_m_s / (elements_per_wavelength * requested_max_frequency_hz)
        if requested_max_frequency_hz is not None
        else None
    )
    warnings: list[str] = []
    global_status = "unknown"
    if requested_max_frequency_hz is not None:
        global_status = "valid"
        if requested_max_frequency_hz > float(global_stats["max_valid_frequency_hz"]):
            global_status = "invalid"
            warnings.append(
                "requested max frequency exceeds conservative global mesh limit "
                f"({requested_max_frequency_hz:.6g} Hz > "
                f"{float(global_stats['max_valid_frequency_hz']):.6g} Hz); "
                "global coarse regions are reported but only active source patches hard-fail"
            )

    per_source: dict[str, dict[str, object]] = {}
    invalid_sources: list[str] = []
    for spec in source_specs:
        mask = tags == spec.tag
        source_triangles = triangles[mask]
        stats = _edge_frequency_stats(
            points,
            source_triangles,
            unit_scale_to_m=unit_scale_to_m,
            elements_per_wavelength=elements_per_wavelength,
            speed_of_sound_m_s=speed_of_sound_m_s,
        )
        wall_stats = _source_wall_stats(
            points,
            triangles,
            tags,
            spec,
            transition_mm=transition_mm,
            unit_scale_to_m=unit_scale_to_m,
            elements_per_wavelength=elements_per_wavelength,
            speed_of_sound_m_s=speed_of_sound_m_s,
        )
        patch_limit = float(stats["max_valid_frequency_hz"])
        effective_limit = patch_limit
        if wall_stats is not None:
            wall_limit = float(wall_stats["wall_max_valid_frequency_hz"])
            if wall_limit > 0.0:
                effective_limit = (
                    min(patch_limit, wall_limit) if patch_limit > 0.0 else wall_limit
                )
        source_status = "unknown"
        if requested_max_frequency_hz is not None:
            source_status = "valid"
            if requested_max_frequency_hz > effective_limit:
                source_status = "invalid"
                invalid_sources.append(spec.name)
                if effective_limit < patch_limit:
                    warnings.append(
                        f"{spec.name} rigid walls within the transition distance are "
                        f"underresolved for {requested_max_frequency_hz:.6g} Hz "
                        f"(wall valid {effective_limit:.6g} Hz, patch valid "
                        f"{patch_limit:.6g} Hz)"
                    )
                else:
                    warnings.append(
                        f"{spec.name} source patch is underresolved for "
                        f"{requested_max_frequency_hz:.6g} Hz "
                        f"(max valid {patch_limit:.6g} Hz)"
                    )
        per_source[spec.name] = {
            "name": spec.name,
            "tag": int(spec.tag),
            "requested_resolution_mm": float(spec.resolution_mm),
            "triangle_count": int(len(source_triangles)),
            "status": source_status,
            "effective_max_valid_frequency_hz": float(effective_limit),
            **stats,
            **(wall_stats or {}),
        }

    status = "unknown"
    if requested_max_frequency_hz is not None:
        status = "invalid" if invalid_sources else "valid"

    return {
        "status": status,
        "scope": "global_warn_source_hard",
        "frequency_policy": "global_warn_source_hard",
        "global_status": global_status,
        "global_max_edge_step_units": float(global_stats["max_edge_step_units"]),
        "global_max_edge_m": float(global_stats["max_edge_m"]),
        "global_p95_edge_step_units": float(global_stats["p95_edge_step_units"]),
        "global_p95_edge_m": float(global_stats["p95_edge_m"]),
        "elements_per_wavelength": float(elements_per_wavelength),
        "speed_of_sound_m_s": float(speed_of_sound_m_s),
        "edge_limit_step_units": (
            None if edge_limit_m is None else float(edge_limit_m / unit_scale_to_m)
        ),
        "edge_limit_m": None if edge_limit_m is None else float(edge_limit_m),
        "max_valid_frequency_hz": float(global_stats["max_valid_frequency_hz"]),
        "global_max_valid_frequency_hz": float(global_stats["max_valid_frequency_hz"]),
        "requested_max_frequency_hz": (
            None if requested_max_frequency_hz is None else float(requested_max_frequency_hz)
        ),
        "invalid_sources": invalid_sources,
        "per_source": per_source,
        "warnings": warnings,
    }

def _topology_stats(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    symmetry_planes: tuple[str, ...],
    tolerance: float,
) -> dict:
    edge_count: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = tuple(sorted((int(a), int(b))))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    free_edges = [edge for edge, count in edge_count.items() if count == 1]
    nonmanifold_edges = [edge for edge, count in edge_count.items() if count > 2]
    edge_direction_stats = _edge_direction_stats(triangles)
    unexpected = []
    samples = []
    for edge in free_edges:
        midpoint = 0.5 * (points[edge[0]] + points[edge[1]])
        if not _edge_on_expected_plane(
            points,
            edge,
            symmetry_planes,
            tolerance,
        ):
            unexpected.append(edge)
            if len(samples) < 20:
                samples.append([float(v) for v in midpoint])

    return {
        "triangles": int(len(triangles)),
        "vertices": int(len(points)),
        "free_edges": int(len(free_edges)),
        "boundary_edges": int(len(free_edges)),
        "nonmanifold_edges": int(len(nonmanifold_edges)),
        "inconsistent_edges": int(edge_direction_stats["inconsistent_edges"]),
        "expected_symmetry_planes": list(symmetry_planes),
        "unexpected_free_edges": int(len(unexpected)),
        "unexpected_free_edge_midpoint_samples": samples,
    }
