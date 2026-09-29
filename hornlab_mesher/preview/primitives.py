"""Surface-building blocks shared by the horn, source-cap and enclosure roles.

Grid indexing, smooth parametric grids, flat strips and caps, ring zippers,
and the arc-interval sizing rule.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .contract import (
    PreviewSurfaceV1,
    _orient_indices_to_normals,
    _orientation_metadata,
)
from .fidelity import (
    analytic_grid_curvature,
    analytic_grid_normals,
    estimate_grid_fidelity,
    resample_grid_vectors,
    resample_parametric_grid,
)


_MAX_ARC_INTERVALS = 1024


def _surface_grid(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert canonical ``(phi,t,xyz)`` to surface ``(t,phi,xyz)``."""

    return np.ascontiguousarray(np.transpose(points, (1, 0, 2)), dtype=np.float64)


def _grid_indices(n_t: int, n_phi: int, *, closed_phi: bool) -> NDArray[np.uint32]:
    """Two triangles per quad, in the same order the scalar loop emitted."""

    phi_intervals = n_phi if closed_phi else n_phi - 1
    if n_t < 2 or phi_intervals < 1:
        return np.empty(0, dtype=np.uint32)
    row0 = (np.arange(n_t - 1, dtype=np.uint32) * n_phi)[:, None]
    row1 = row0 + n_phi
    ip = np.arange(phi_intervals, dtype=np.uint32)[None, :]
    ip1 = (ip + 1) % n_phi
    triangles = np.empty((n_t - 1, phi_intervals, 6), dtype=np.uint32)
    triangles[:, :, 0] = row0 + ip
    triangles[:, :, 1] = row0 + ip1
    triangles[:, :, 2] = row1 + ip1
    triangles[:, :, 3] = row0 + ip
    triangles[:, :, 4] = row1 + ip1
    triangles[:, :, 5] = row1 + ip
    return triangles.reshape(-1)


def _smooth_grid_surface(
    role: str,
    points: NDArray[np.float64],
    reference: NDArray[np.float64],
    *,
    closed_phi: bool,
    point_t: NDArray[np.float64] | None = None,
    point_phi: NDArray[np.float64] | None = None,
    reference_t: NDArray[np.float64] | None = None,
    reference_phi: NDArray[np.float64] | None = None,
    orientation_hint: NDArray[np.float64] | None = None,
    include_curvature: bool = True,
) -> tuple[PreviewSurfaceV1, dict[str, float]]:
    ref_normals = analytic_grid_normals(
        reference,
        closed_phi=closed_phi,
        t_coordinates=reference_t,
        phi_coordinates=reference_phi,
    )
    resampling = any(
        value is not None
        for value in (point_t, point_phi, reference_t, reference_phi)
    )
    if resampling:
        normals = resample_parametric_grid(
            ref_normals,
            points.shape[:2],
            source_t=reference_t,
            source_phi=reference_phi,
            target_t=point_t,
            target_phi=point_phi,
            normalise=True,
            closed_phi=closed_phi,
        )
    else:
        normals = resample_grid_vectors(
            ref_normals, points.shape[:2], closed_phi=closed_phi
        )
    curvature_mean = curvature_principal = None
    if include_curvature:
        ref_mean, ref_principal = analytic_grid_curvature(
            reference,
            closed_phi=closed_phi,
            t_coordinates=reference_t,
            phi_coordinates=reference_phi,
        )
        if resampling:
            curvature = resample_parametric_grid(
                np.stack((ref_mean, ref_principal), axis=2),
                points.shape[:2],
                source_t=reference_t,
                source_phi=reference_phi,
                target_t=point_t,
                target_phi=point_phi,
                closed_phi=closed_phi,
            )
        else:
            curvature = resample_parametric_grid(
                np.stack((ref_mean, ref_principal), axis=2),
                points.shape[:2],
                closed_phi=closed_phi,
            )
        curvature_mean = curvature[:, :, 0]
        curvature_principal = curvature[:, :, 1]
    if orientation_hint is not None:
        hint = np.broadcast_to(np.asarray(orientation_hint, dtype=np.float64), normals.shape)
        if float(np.median(np.sum(normals * hint, axis=2))) < 0.0:
            normals = -normals
            if curvature_mean is not None:
                curvature_mean = -curvature_mean
                assert curvature_principal is not None
                curvature_principal = -curvature_principal
    positions = points.reshape(-1, 3)
    flat_normals = normals.reshape(-1, 3)
    oriented = _orient_indices_to_normals(
        role,
        positions,
        _grid_indices(*points.shape[:2], closed_phi=closed_phi),
        flat_normals,
    )
    surface = PreviewSurfaceV1(
        role=role,
        positions=positions,
        indices=oriented.indices,
        normals=flat_normals,
        shading="smooth",
        normal_method="analytic-parametric",
        closed_phi=closed_phi,
        curvature_mean=(
            None if curvature_mean is None else curvature_mean.reshape(-1)
        ),
        curvature_principal=(
            None
            if curvature_principal is None
            else curvature_principal.reshape(-1)
        ),
        metadata=_orientation_metadata(oriented),
    )
    fidelity = estimate_grid_fidelity(
        points,
        reference,
        normals,
        closed_phi=closed_phi,
        coarse_t=point_t,
        coarse_phi=point_phi,
        reference_t=reference_t,
        reference_phi=reference_phi,
    )
    return surface, fidelity


def _flat_strip(
    role: str,
    inner: NDArray[np.float64],
    outer: NDArray[np.float64],
    normal: tuple[float, float, float],
    *,
    closed_phi: bool,
    include_curvature: bool = True,
) -> PreviewSurfaceV1:
    points = np.stack((inner, outer), axis=0)
    normals = np.broadcast_to(np.asarray(normal, dtype=np.float64), points.shape).copy()
    positions = points.reshape(-1, 3)
    flat_normals = normals.reshape(-1, 3)
    oriented = _orient_indices_to_normals(
        role,
        positions,
        _grid_indices(2, points.shape[1], closed_phi=closed_phi),
        flat_normals,
    )
    return PreviewSurfaceV1(
        role=role,
        positions=positions,
        indices=oriented.indices,
        normals=flat_normals,
        shading="flat",
        normal_method="exact-planar",
        closed_phi=closed_phi,
        curvature_mean=(
            np.zeros(len(positions), dtype=np.float64) if include_curvature else None
        ),
        curvature_principal=(
            np.zeros(len(positions), dtype=np.float64) if include_curvature else None
        ),
        metadata=_orientation_metadata(oriented),
    )


def _flat_triangle(
    role: str,
    points: NDArray[np.float64],
    normal: tuple[float, float, float],
    *,
    include_curvature: bool = True,
) -> PreviewSurfaceV1:
    positions = np.asarray(points, dtype=np.float64).reshape(3, 3)
    flat_normals = np.broadcast_to(
        np.asarray(normal, dtype=np.float64), positions.shape
    ).copy()
    oriented = _orient_indices_to_normals(
        role,
        positions,
        np.asarray((0, 1, 2), dtype=np.uint32),
        flat_normals,
    )
    return PreviewSurfaceV1(
        role=role,
        positions=positions,
        indices=oriented.indices,
        normals=flat_normals,
        shading="flat",
        normal_method="exact-planar",
        closed_phi=False,
        curvature_mean=(
            np.zeros(len(positions), dtype=np.float64) if include_curvature else None
        ),
        curvature_principal=(
            np.zeros(len(positions), dtype=np.float64) if include_curvature else None
        ),
        metadata=_orientation_metadata(oriented),
    )


def _simplify_planar_ring(
    ring: NDArray[np.float64], tolerance: float
) -> NDArray[np.float64]:
    """Drop ring vertices whose removal moves the boundary less than ``tolerance``.

    A cap fans every boundary chord against a center a couple of hundred mm
    away, so a floored plan corner (a 2 um chamfer chord, a fillet's 0.1 mm
    inner arc walked in thirty samples) turns into fan slivers tens of
    thousands to one. The cap has no ring-correspondence obligation, so it may
    simplify its own boundary; the band it abuts stays within ``tolerance`` of
    the simplified polygon.
    """

    points = [np.asarray(p, dtype=np.float64) for p in ring]
    changed = True
    while changed and len(points) > 3:
        changed = False
        for index in range(len(points)):
            previous = points[index - 1]
            candidate = points[index]
            following = points[(index + 1) % len(points)]
            edge = following - previous
            edge_len = float(np.linalg.norm(edge))
            if edge_len <= 1.0e-12:
                deviation = float(np.linalg.norm(candidate - previous))
            else:
                fraction = float(
                    np.clip(np.dot(candidate - previous, edge) / edge_len**2, 0.0, 1.0)
                )
                deviation = float(
                    np.linalg.norm(candidate - (previous + fraction * edge))
                )
            if deviation < tolerance:
                points.pop(index)
                changed = True
                break
    return np.asarray(points, dtype=np.float64)


def _flat_cap(
    role: str,
    ring: NDArray[np.float64],
    normal: tuple[float, float, float],
    *,
    closed_phi: bool,
    include_curvature: bool = True,
    simplify_tolerance: float | None = None,
) -> PreviewSurfaceV1:
    ring = np.asarray(ring, dtype=np.float64)
    if simplify_tolerance is not None and closed_phi:
        ring = _simplify_planar_ring(ring, simplify_tolerance)
    center = np.mean(ring, axis=0)
    if closed_phi:
        # A corner-refined morph lattice can contain locally out-of-order phi
        # samples even though its boundary is star-shaped. The cap has no row
        # correspondence to preserve, so angular ordering avoids manufacturing
        # inverted fan faces from that sampling artifact.
        angles = np.arctan2(ring[:, 1] - center[1], ring[:, 0] - center[0])
        ring = ring[np.argsort(angles, kind="stable")]
    positions = np.vstack((ring, center))
    center_index = len(ring)
    triangles: list[int] = []
    limit = len(ring) if closed_phi else len(ring) - 1
    for ip in range(limit):
        ip1 = (ip + 1) % len(ring)
        if normal[2] >= 0.0:
            triangles.extend((center_index, ip, ip1))
        else:
            triangles.extend((center_index, ip1, ip))
    normals = np.broadcast_to(np.asarray(normal, dtype=np.float64), positions.shape).copy()
    oriented = _orient_indices_to_normals(
        role,
        positions,
        np.asarray(triangles, dtype=np.uint32),
        normals,
    )
    return PreviewSurfaceV1(
        role=role,
        positions=positions,
        indices=oriented.indices,
        normals=normals,
        shading="flat",
        normal_method="exact-planar",
        closed_phi=closed_phi,
        curvature_mean=(
            np.zeros(len(positions), dtype=np.float64) if include_curvature else None
        ),
        curvature_principal=(
            np.zeros(len(positions), dtype=np.float64) if include_curvature else None
        ),
        metadata=_orientation_metadata(oriented),
    )


def _mouth_exit_direction(inner: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-phi direction the inner profile travels as it leaves the mouth.

    The rim is the end face of the wall, so this is the direction it faces. It
    is +z only while the mouth still opens forward: a rolled-back termination
    (R-OSSE at high ``tmax``, any strong roundover) exits backwards, and a fixed
    +z hint would invert the whole rim rather than orient it.
    """

    if inner.shape[1] < 2:
        return np.tile(np.asarray((0.0, 0.0, 1.0), dtype=np.float64), (len(inner), 1))
    direction = np.asarray(inner[:, -1, :] - inner[:, -2, :], dtype=np.float64)
    lengths = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = direction / np.where(lengths > 0.0, lengths, 1.0)
    direction[lengths[:, 0] <= 1.0e-12] = (0.0, 0.0, 1.0)
    return direction


def _smooth_mouth_rim(
    inner: NDArray[np.float64],
    outer: NDArray[np.float64],
    *,
    closed_phi: bool,
    exit_direction: NDArray[np.float64],
    include_curvature: bool,
) -> PreviewSurfaceV1:
    grid = np.stack((inner, outer), axis=0)
    surface, _fidelity = _smooth_grid_surface(
        "mouth_rim",
        grid,
        grid,
        closed_phi=closed_phi,
        orientation_hint=np.asarray(exit_direction, dtype=np.float64)[None, :, :],
        include_curvature=include_curvature,
    )
    return surface


def _ring_angle_table(
    ring: NDArray[np.float64], center: NDArray[np.float64], base: float | None = None
) -> tuple[NDArray[np.float64], int]:
    theta = np.arctan2(ring[:, 1] - center[1], ring[:, 0] - center[0])
    if base is None:
        start = 0
        first = float(theta[0])
    else:
        delta = np.abs(np.angle(np.exp(1j * (theta - base))))
        start = int(np.argmin(delta))
        first = base + float(np.angle(np.exp(1j * (theta[start] - base))))
    unwrapped = [first]
    previous = first
    for k in range(1, len(ring) + 1):
        value = float(theta[(start + k) % len(ring)])
        step = (value - previous) % math.tau
        if step > math.tau - 1.0e-9:
            step = 0.0
        unwrapped.append(unwrapped[-1] + step)
        previous = unwrapped[-1]
    return np.asarray(unwrapped, dtype=np.float64), start


def _zipper(
    ring_a: NDArray[np.float64], ring_b: NDArray[np.float64], offset_b: int
) -> list[int]:
    center = np.mean(ring_a[:, :2], axis=0)
    angles_a, start_a = _ring_angle_table(ring_a, center)
    angles_b, start_b = _ring_angle_table(ring_b, center, float(angles_a[0]))
    n_a, n_b = len(ring_a), len(ring_b)

    def index_a(k: int) -> int:
        return (start_a + k) % n_a

    def index_b(k: int) -> int:
        return offset_b + (start_b + k) % n_b

    triangles: list[int] = []
    i = j = 0
    while i < n_a or j < n_b:
        advance_a = j >= n_b or (i < n_a and angles_a[i + 1] <= angles_b[j + 1])
        if advance_a:
            triangles.extend((index_b(j), index_a(i), index_a(i + 1)))
            i += 1
        else:
            triangles.extend((index_b(j), index_a(i), index_b(j + 1)))
            j += 1
    return triangles


def _combine_surfaces(role: str, parts: list[PreviewSurfaceV1]) -> PreviewSurfaceV1:
    positions: list[NDArray[np.float64]] = []
    normals: list[NDArray[np.float64]] = []
    indices: list[NDArray[np.uint32]] = []
    curvature_mean: list[NDArray[np.float64]] = []
    curvature_principal: list[NDArray[np.float64]] = []
    offset = 0
    for part in parts:
        positions.append(part.positions)
        normals.append(part.normals)
        indices.append(part.indices + np.uint32(offset))
        if part.curvature_mean is not None and part.curvature_principal is not None:
            curvature_mean.append(part.curvature_mean)
            curvature_principal.append(part.curvature_principal)
        offset += len(part.positions)
    return PreviewSurfaceV1(
        role=role,
        positions=np.vstack(positions),
        indices=np.concatenate(indices),
        normals=np.vstack(normals),
        shading=parts[0].shading,
        normal_method=parts[0].normal_method,
        closed_phi=all(part.closed_phi for part in parts),
        curvature_mean=(
            np.concatenate(curvature_mean) if len(curvature_mean) == len(parts) else None
        ),
        curvature_principal=(
            np.concatenate(curvature_principal)
            if len(curvature_principal) == len(parts)
            else None
        ),
    )


def _even_indices(size: int, count: int, *, closed: bool) -> list[int]:
    if closed:
        count = min(size, max(3, int(count)))
        return sorted({int(index * size // count) for index in range(count)})
    count = min(size, max(2, int(count)))
    return sorted(
        {int(round(index * (size - 1) / (count - 1))) for index in range(count)}
    )


def _grid_surface_from_selection(
    role: str,
    points: NDArray[np.float64],
    normals: NDArray[np.float64],
    t_indices: NDArray[np.int64],
    phi_indices: NDArray[np.int64],
    *,
    closed_phi: bool,
    normal_sign: float = 1.0,
    curvature_mean: NDArray[np.float64] | None = None,
    curvature_principal: NDArray[np.float64] | None = None,
    wind_folds_individually: bool = False,
) -> PreviewSurfaceV1:
    selected_points = points[np.ix_(t_indices, phi_indices)]
    selected_normals = normal_sign * normals[np.ix_(t_indices, phi_indices)]
    positions = selected_points.reshape(-1, 3)
    flat_normals = selected_normals.reshape(-1, 3)
    selected_mean = (
        None
        if curvature_mean is None
        else normal_sign * curvature_mean[np.ix_(t_indices, phi_indices)]
    )
    selected_principal = (
        None
        if curvature_principal is None
        else normal_sign * curvature_principal[np.ix_(t_indices, phi_indices)]
    )
    oriented = _orient_indices_to_normals(
        role,
        positions,
        _grid_indices(*selected_points.shape[:2], closed_phi=closed_phi),
        flat_normals,
        wind_folds_individually=wind_folds_individually,
    )
    return PreviewSurfaceV1(
        role=role,
        positions=positions,
        indices=oriented.indices,
        normals=flat_normals,
        shading="smooth",
        normal_method="analytic-parametric",
        closed_phi=closed_phi,
        curvature_mean=(None if selected_mean is None else selected_mean.reshape(-1)),
        curvature_principal=(
            None if selected_principal is None else selected_principal.reshape(-1)
        ),
        metadata=_orientation_metadata(oriented),
    )


def _intervals_for_arc(
    radius: float, span_deg: float, chord: float, normal_deg: float, floor: int
) -> int:
    by_normal = int(math.ceil(span_deg / normal_deg)) + 2
    if radius <= chord:
        by_chord = 1
    else:
        half_angle = math.acos(np.clip(1.0 - chord / radius, -1.0, 1.0))
        by_chord = int(math.ceil(math.radians(span_deg) / max(2.0 * half_angle, 1.0e-12)))
    return min(_MAX_ARC_INTERVALS, max(int(floor), by_normal, by_chord))
