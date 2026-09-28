"""Enclosure preview surfaces: plan rings, baffle, edge treatment, sides, rear.

The rounded-rectangle fillet has a closed form (:func:`_fillet_pieces`); the
chamfer is a ruled band of planes (:func:`_faceted_band`); ellipse and
superellipse plans keep a sampled roundover grid (:func:`_roundover_piece`).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..builders.enclosure import sample_enclosure_plan
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
from .primitives import (
    _MAX_ARC_INTERVALS,
    _combine_surfaces,
    _flat_cap,
    _flat_strip,
    _flat_triangle,
    _intervals_for_arc,
    _smooth_grid_surface,
    _zipper,
)


# Plan corner radius floors. ``sample_rounded_rect`` treats anything at or below
# 1e-3 mm as a sharp box and emits a different vertex count, so both floors stay
# above it; see ``_plan_ring``.
_PLAN_CORNER_FLOOR_MM = 0.1
_FACETED_CORNER_FLOOR_MM = 2.0e-3
# A ruled-band column whose ring chord is below this is a floored-corner
# artifact, not geometry: collapse it to the corner triangle the solver builds.
_DEGENERATE_COLUMN_MM = 1.0e-2


def _ccw_ring(points: NDArray[np.float64]) -> NDArray[np.float64]:
    ring = np.asarray(points, dtype=np.float64)
    x = ring[:, 0]
    y = ring[:, 1]
    area2 = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return ring if area2 >= 0.0 else ring[::-1].copy()


def _faceted_edge(enclosure: Mapping[str, Any]) -> bool:
    """True when the plan is polygonal at every inset, so its edge is planes.

    ``sample_rounded_rect`` walks ``edge_type == 2`` corners along the straight
    chord between the arc endpoints, so a chamfered rounded rectangle is a
    polygon at every ``radial_t`` and the band it sweeps is a fan of planes.
    Ellipse and superellipse plans stay curved and keep the smooth path.
    """

    return int(enclosure["plan_type"]) == 1 and int(enclosure["edge_type"]) == 2


def _plan_ring(
    enclosure: Mapping[str, Any],
    z: float,
    radial_t: float,
    *,
    corner_intervals: int = 4,
) -> NDArray[np.float64]:
    bounds = enclosure["bounds"]
    edge = float(enclosure["edge_mm"])
    d = edge * (1.0 - radial_t)
    # The inset ring of an edge treatment has a genuinely sharp plan corner, but
    # ``sample_rounded_rect`` switches to its four-corner path below 1e-3 mm and
    # would then emit a different vertex count than the outer ring, so the
    # radius is floored rather than zeroed. A chamfer's band is ruled column by
    # column between the two rings, and 0.1 mm of floor there put the emitted
    # corner 0.05 mm off the plane the solver builds -- right at the finest LOD's
    # whole chord budget. Faceted plans keep the smallest floor that stays on the
    # rounded path.
    radius = max(
        _FACETED_CORNER_FLOOR_MM if _faceted_edge(enclosure) else _PLAN_CORNER_FLOOR_MM,
        edge * radial_t,
    )
    plan_type = int(enclosure["plan_type"])
    if plan_type in {2, 3}:
        count = max(4, 4 * int(corner_intervals))
        cx = 0.5 * (float(bounds["bx0"]) + float(bounds["bx1"]))
        cy = 0.5 * (float(bounds["by0"]) + float(bounds["by1"]))
        a = 0.5 * (float(bounds["bx1"]) - float(bounds["bx0"])) - d
        b = 0.5 * (float(bounds["by1"]) - float(bounds["by0"])) - d
        n = 2.0 if plan_type == 2 else float(enclosure.get("plan_n", 2.0))
        theta = np.arange(count, dtype=np.float64) * math.tau / count
        cosine = np.cos(theta)
        sine = np.sin(theta)
        radial = (
            np.abs(cosine / a) ** n + np.abs(sine / b) ** n
        ) ** (-1.0 / n)
        return np.column_stack(
            (
                cx + radial * cosine,
                cy + radial * sine,
                np.full(count, float(z), dtype=np.float64),
            )
        )
    return _ccw_ring(
        sample_enclosure_plan(
            bx0=float(bounds["bx0"]) + d,
            bx1=float(bounds["bx1"]) - d,
            by0=float(bounds["by0"]) + d,
            by1=float(bounds["by1"]) - d,
            corner_radius=radius,
            edge_type=int(enclosure["edge_type"]),
            z=float(z),
            plan_type=plan_type,
            plan_n=float(enclosure.get("plan_n", 2.0)),
            n_per_edge=1,
            n_per_corner=max(1, int(corner_intervals)),
        )
    )


def _plan_fidelity(
    enclosure: Mapping[str, Any], corner_intervals: int
) -> dict[str, float]:
    """Measure an emitted ellipse/superellipse plan against dense true samples."""

    plan_type = int(enclosure["plan_type"])
    if plan_type == 1:
        if _faceted_edge(enclosure):
            # Straight chords, sampled on the chord: the emitted polygon IS the
            # plan at every subdivision. Modelling it as a 90 degree arc bought
            # dozens of collinear samples per corner and reported a chord error
            # the geometry never had.
            return {
                "max_chord_error_mm": 0.0,
                "max_normal_step_deg": 0.0,
                "reference_density_multiplier": 1,
            }
        radius = float(enclosure.get("edge_mm", 0.1))
        return {
            "max_chord_error_mm": max(
                np.finfo(np.float64).eps,
                radius * (1.0 - math.cos(math.pi / (4.0 * corner_intervals))),
            ),
            "max_normal_step_deg": 90.0 / corner_intervals,
            "reference_density_multiplier": 4,
        }

    bounds = enclosure["bounds"]
    cx = 0.5 * (float(bounds["bx0"]) + float(bounds["bx1"]))
    cy = 0.5 * (float(bounds["by0"]) + float(bounds["by1"]))
    half_width = 0.5 * (float(bounds["bx1"]) - float(bounds["bx0"]))
    half_height = 0.5 * (float(bounds["by1"]) - float(bounds["by0"]))
    n = 2.0 if plan_type == 2 else float(enclosure.get("plan_n", 2.0))
    chord_error = 0.0
    normal_step = 0.0
    # Roundovers traverse the inset family. Sampling several true members makes
    # the published plan bound apply to the emitted transition, not just its
    # largest outer ring.
    for radial_t in np.linspace(0.0, 1.0, 5):
        ring = _plan_ring(
            enclosure, 0.0, float(radial_t), corner_intervals=corner_intervals
        )
        count = len(ring)
        dense_count = 8 * count
        dense = _plan_ring(
            enclosure, 0.0, float(radial_t), corner_intervals=2 * count
        )
        segment = np.arange(dense_count, dtype=np.int64) // 8
        weight = (np.arange(dense_count, dtype=np.float64) % 8) / 8.0
        chord = ring[segment] * (1.0 - weight[:, None]) + ring[
            (segment + 1) % count
        ] * weight[:, None]
        chord_error = max(
            chord_error, float(np.max(np.linalg.norm(dense - chord, axis=1)))
        )

        inset = float(enclosure["edge_mm"]) * (1.0 - float(radial_t))
        a = half_width - inset
        b = half_height - inset
        ux = (ring[:, 0] - cx) / a
        uy = (ring[:, 1] - cy) / b
        gradients = np.column_stack(
            (
                np.sign(ux) * np.abs(ux) ** (n - 1.0) / a,
                np.sign(uy) * np.abs(uy) ** (n - 1.0) / b,
            )
        )
        gradients /= np.linalg.norm(gradients, axis=1, keepdims=True)
        dots = np.sum(gradients * np.roll(gradients, -1, axis=0), axis=1)
        normal_step = max(
            normal_step,
            float(np.degrees(np.arccos(np.clip(np.min(dots), -1.0, 1.0)))),
        )
    return {
        "max_chord_error_mm": max(chord_error, np.finfo(np.float64).eps),
        "max_normal_step_deg": normal_step,
        "reference_density_multiplier": 8,
    }


def _adaptive_plan_intervals(
    enclosure: Mapping[str, Any],
    chord_target: float,
    normal_target: float,
    floor: int,
) -> tuple[int, bool]:
    if int(enclosure["plan_type"]) == 1:
        if _faceted_edge(enclosure):
            return max(1, int(floor)), False
        intervals = _intervals_for_arc(
            float(enclosure.get("edge_mm", 0.1)),
            90.0,
            chord_target,
            normal_target,
            floor,
        )
        return intervals, intervals >= _MAX_ARC_INTERVALS

    low = max(1, int(floor))
    measured = _plan_fidelity(enclosure, low)
    if (
        measured["max_chord_error_mm"] <= chord_target
        and measured["max_normal_step_deg"] <= normal_target
    ):
        return low, False
    high = low
    while high < _MAX_ARC_INTERVALS:
        high = min(_MAX_ARC_INTERVALS, high * 2)
        measured = _plan_fidelity(enclosure, high)
        if (
            measured["max_chord_error_mm"] <= chord_target
            and measured["max_normal_step_deg"] <= normal_target
        ):
            break
    else:
        return _MAX_ARC_INTERVALS, True
    if high == _MAX_ARC_INTERVALS and (
        measured["max_chord_error_mm"] > chord_target
        or measured["max_normal_step_deg"] > normal_target
    ):
        return high, True
    left = low + 1
    right = high
    while left < right:
        middle = (left + right) // 2
        measured = _plan_fidelity(enclosure, middle)
        if (
            measured["max_chord_error_mm"] <= chord_target
            and measured["max_normal_step_deg"] <= normal_target
        ):
            right = middle
        else:
            left = middle + 1
    return left, False


def _ray_cast(
    plan: NDArray[np.float64], center: NDArray[np.float64], direction: NDArray[np.float64]
) -> NDArray[np.float64]:
    best: tuple[float, NDArray[np.float64]] | None = None
    for j, a in enumerate(plan):
        b = plan[(j + 1) % len(plan)]
        segment = b[:2] - a[:2]
        denominator = direction[0] * segment[1] - direction[1] * segment[0]
        if abs(denominator) <= 1.0e-12:
            continue
        offset = a[:2] - center
        ray_t = (offset[0] * segment[1] - offset[1] * segment[0]) / denominator
        seg_t = (offset[0] * direction[1] - offset[1] * direction[0]) / denominator
        if ray_t >= -1.0e-9 and -1.0e-9 <= seg_t <= 1.0 + 1.0e-9:
            hit = a + np.clip(seg_t, 0.0, 1.0) * (b - a)
            if best is None or ray_t < best[0]:
                best = (ray_t, hit)
    if best is None:
        angles = np.arctan2(plan[:, 1] - center[1], plan[:, 0] - center[0])
        target = math.atan2(direction[1], direction[0])
        delta = np.abs(np.angle(np.exp(1j * (angles - target))))
        return plan[int(np.argmin(delta))]
    return best[1]


def _ray_aligned_ring(
    plan: NDArray[np.float64], reference: NDArray[np.float64]
) -> NDArray[np.float64]:
    center = np.mean(reference[:, :2], axis=0)
    result = np.empty_like(reference)
    for index, point in enumerate(reference):
        direction = point[:2] - center
        direction /= max(float(np.linalg.norm(direction)), 1.0e-14)
        result[index] = _ray_cast(plan, center, direction)
    return result


def _plan_corner_angles(
    plan: NDArray[np.float64], center: NDArray[np.float64]
) -> list[float]:
    """Angles (about ``center``) at which the plan breaks tangency.

    A ray-aligned ring only preserves the plan where a ray happens to sample
    it: between two rays that straddle a convex corner, the aligned polyline
    chord-cuts the corner. These are the directions that must become columns
    of their own. A run of tangent breaks packed inside a floored corner
    radius (the 2 um chamfer chord, a fillet's 0.1 mm inner arc) collapses to
    its centroid -- one column, not thirty needles.
    """

    edges = np.roll(plan[:, :2], -1, axis=0) - plan[:, :2]
    lengths = np.linalg.norm(edges, axis=1)
    directions = edges / np.where(lengths > 1.0e-12, lengths, 1.0)[:, None]
    dots = np.clip(
        np.sum(directions * np.roll(directions, 1, axis=0), axis=1), -1.0, 1.0
    )
    turn_deg = np.degrees(np.arccos(dots))
    flagged = np.flatnonzero((turn_deg > 1.0) & (lengths > 0.0))
    if len(flagged) == 0:
        return []
    # Group cyclically-consecutive flagged vertices.
    groups: list[list[int]] = [[int(flagged[0])]]
    for index in flagged[1:]:
        if int(index) == groups[-1][-1] + 1:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    if len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == len(plan) - 1:
        groups[0] = groups.pop() + groups[0]
    angles: list[float] = []
    for group in groups:
        points = plan[group, :2]
        extent = float(
            np.max(np.linalg.norm(points - points.mean(axis=0), axis=1))
        )
        if extent <= 0.5:
            probes = [points.mean(axis=0)]
        else:
            probes = [points[k] for k in range(len(points))]
        for probe in probes:
            angles.append(
                math.atan2(float(probe[1] - center[1]), float(probe[0] - center[0]))
            )
    return angles


def _insert_corner_columns(
    mouth: NDArray[np.float64],
    aligned: NDArray[np.float64],
    plan: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Add a column at every plan corner the mouth stations miss.

    The baffle strip and the aligned ring stay column-matched; the inserted
    mouth points interpolate the (smooth) mouth curve, and the inserted
    aligned points are exact ray hits, which land on the plan's corner. A
    corner that already coincides with a mouth station (the symmetric default
    box puts all eight at power-of-two angles) inserts nothing.
    """

    center = np.mean(mouth[:, :2], axis=0)
    corner_angles = _plan_corner_angles(plan, center)
    if not corner_angles:
        return mouth, aligned
    mouth_theta = np.arctan2(mouth[:, 1] - center[1], mouth[:, 0] - center[0])
    n = len(mouth)
    aligned = np.array(aligned, dtype=np.float64)
    insertions: list[tuple[int, float, NDArray[np.float64], NDArray[np.float64]]] = []
    taken: list[float] = []
    for theta in corner_angles:
        if any(abs(float(np.angle(np.exp(1j * (theta - t))))) < 1.0e-3 for t in taken):
            continue
        offsets = np.abs(np.angle(np.exp(1j * (mouth_theta - theta))))
        nearest = int(np.argmin(offsets))
        if float(offsets[nearest]) < 1.0e-3:
            # A corner that (nearly) coincides with a mouth station gets no
            # column of its own -- that would rule a hair-width baffle quad
            # against the station. Move the station's outer point onto the
            # corner instead: the mouth-side weld is untouched, and the
            # residual chord-cut at fine LOD (0.1 mm for a station 1 mrad off
            # the corner) goes with it.
            direction = np.asarray(
                (math.cos(theta), math.sin(theta)), dtype=np.float64
            )
            aligned[nearest] = _ray_cast(plan, center, direction)
            taken.append(theta)
            continue
        slot = None
        for i in range(n):
            span = float(np.angle(np.exp(1j * (mouth_theta[(i + 1) % n] - mouth_theta[i]))))
            local = float(np.angle(np.exp(1j * (theta - mouth_theta[i]))))
            if abs(span) < 1.0e-12:
                continue
            fraction = local / span
            if 0.0 < fraction < 1.0 and abs(local) <= abs(span):
                slot = (i, fraction)
                break
        if slot is None:
            continue
        i, fraction = slot
        mouth_point = mouth[i] + fraction * (mouth[(i + 1) % n] - mouth[i])
        direction = np.asarray((math.cos(theta), math.sin(theta)), dtype=np.float64)
        aligned_point = _ray_cast(plan, center, direction)
        taken.append(theta)
        insertions.append((i, fraction, mouth_point, aligned_point))
    if not insertions:
        return mouth, aligned
    insertions.sort(key=lambda item: (item[0], item[1]))
    mouth_out = list(map(np.asarray, mouth))
    aligned_out = list(map(np.asarray, aligned))
    for i, _fraction, mouth_point, aligned_point in reversed(insertions):
        mouth_out.insert(i + 1, mouth_point)
        aligned_out.insert(i + 1, aligned_point)
    return (
        np.asarray(mouth_out, dtype=np.float64),
        np.asarray(aligned_out, dtype=np.float64),
    )


def _roundover_piece(
    role: str,
    first_ring: NDArray[np.float64],
    output_grid: NDArray[np.float64],
    reference_grid: NDArray[np.float64],
    center_xy: NDArray[np.float64],
    *,
    include_curvature: bool,
) -> tuple[PreviewSurfaceV1, dict[str, float]]:
    # Ring zero is ray-aligned to the horn; subsequent rings retain the canonical
    # enclosure plan. The first stitch is therefore the unequal-ring zipper used
    # by the browser tessellator, while all following rings are modulo-closed.
    native_normals = analytic_grid_normals(reference_grid, closed_phi=True)
    output_normals = resample_grid_vectors(
        native_normals, output_grid.shape[:2], closed_phi=True
    )
    first_ref_normals = resample_grid_vectors(
        native_normals[:2], (2, len(first_ring)), closed_phi=True
    )[0]
    output_mean = output_principal = first_ref_mean = first_ref_principal = None
    if include_curvature:
        native_mean, native_principal = analytic_grid_curvature(
            reference_grid, closed_phi=True
        )
        output_curvature = resample_parametric_grid(
            np.stack((native_mean, native_principal), axis=2),
            output_grid.shape[:2],
            closed_phi=True,
        )
        first_curvature = resample_parametric_grid(
            np.stack((native_mean[:2], native_principal[:2]), axis=2),
            (2, len(first_ring)),
            closed_phi=True,
        )[0]
        output_mean = output_curvature[:, :, 0]
        output_principal = output_curvature[:, :, 1]
        first_ref_mean = first_curvature[:, 0]
        first_ref_principal = first_curvature[:, 1]
    native_hint = output_grid.copy()
    native_hint[:, :, 0] -= center_xy[0]
    native_hint[:, :, 1] -= center_xy[1]
    native_hint[:, :, 2] = 0.0
    if float(np.median(np.sum(output_normals * native_hint, axis=2))) < 0.0:
        output_normals = -output_normals
        first_ref_normals = -first_ref_normals
        if output_mean is not None:
            output_mean = -output_mean
            output_principal = -output_principal
            first_ref_mean = -first_ref_mean
            first_ref_principal = -first_ref_principal
    positions = [first_ring, *list(output_grid[1:])]
    normals = [first_ref_normals, *list(output_normals[1:])]
    curvature_mean = (
        None
        if first_ref_mean is None or output_mean is None
        else np.concatenate((first_ref_mean, output_mean[1:].reshape(-1)))
    )
    curvature_principal = (
        None
        if first_ref_principal is None or output_principal is None
        else np.concatenate(
            (first_ref_principal, output_principal[1:].reshape(-1))
        )
    )
    offsets = [0]
    for ring in positions[:-1]:
        offsets.append(offsets[-1] + len(ring))
    triangles = _zipper(positions[0], positions[1], offsets[1])
    for level in range(1, len(positions) - 1):
        n_phi = len(positions[level])
        for ip in range(n_phi):
            ip1 = (ip + 1) % n_phi
            a0, a1 = offsets[level] + ip, offsets[level] + ip1
            b0, b1 = offsets[level + 1] + ip, offsets[level + 1] + ip1
            triangles.extend((a0, a1, b1, a0, b1, b0))
    position_array = np.vstack(positions)
    normal_array = np.vstack(normals)
    oriented = _orient_indices_to_normals(
        role,
        position_array,
        np.asarray(triangles, dtype=np.uint32),
        normal_array,
    )
    surface = PreviewSurfaceV1(
        role=role,
        positions=position_array,
        indices=oriented.indices,
        normals=normal_array,
        shading="smooth",
        normal_method="analytic-parametric",
        closed_phi=True,
        curvature_mean=curvature_mean,
        curvature_principal=curvature_principal,
        metadata=_orientation_metadata(oriented),
    )
    fidelity = estimate_grid_fidelity(
        output_grid, reference_grid, output_normals, closed_phi=True
    )
    return surface, fidelity


def _hard_side_surface(
    front: NDArray[np.float64],
    back: NDArray[np.float64],
    *,
    include_curvature: bool,
) -> PreviewSurfaceV1:
    """Build one planar quad per plan edge, duplicating every hard seam."""

    parts: list[PreviewSurfaceV1] = []
    for index in range(len(front)):
        next_index = (index + 1) % len(front)
        edge = front[next_index, :2] - front[index, :2]
        outward = np.asarray((edge[1], -edge[0], 0.0), dtype=np.float64)
        outward /= np.linalg.norm(outward)
        parts.append(
            _flat_strip(
                "enclosure.side",
                front[[index, next_index]],
                back[[index, next_index]],
                tuple(float(value) for value in outward),
                closed_phi=False,
                include_curvature=include_curvature,
            )
        )
    return _combine_surfaces("enclosure.side", parts)


def _faceted_band(
    role: str,
    near: NDArray[np.float64],
    far: NDArray[np.float64],
    center: NDArray[np.float64],
    *,
    include_curvature: bool,
) -> PreviewSurfaceV1:
    """Build one planar quad per plan segment across a ruled band.

    A chamfer is not a curved surface being approximated -- it is a finite set
    of planes meeting at real tangent breaks. Ruling it in the plan sampler's
    own parameterisation keeps every column inside a single plane, so each quad
    carries that plane's exact normal and the band hard-shades correctly however
    unevenly the plan itself is sampled. ``center`` is any interior point; it
    only picks the outward sign.
    """

    parts: list[PreviewSurfaceV1] = []
    for index in range(len(near)):
        next_index = (index + 1) % len(near)
        near_edge = near[next_index] - near[index]
        far_edge = far[next_index] - far[index]
        near_len = float(np.linalg.norm(near_edge))
        far_len = float(np.linalg.norm(far_edge))
        # A corner column carries the plan sampler's floored corner radius: a
        # micron-scale chord on one ring against the real corner facet on the
        # other. The solver builds a single corner *triangle* there
        # (``build_sector``'s chamfer is two side parallelograms plus one
        # corner triangle per quadrant), and emitting the trapezoid instead
        # split it into that triangle plus a ~10000:1 sliver.
        if near_len < _DEGENERATE_COLUMN_MM and far_len >= _DEGENERATE_COLUMN_MM:
            apex = 0.5 * (near[index] + near[next_index])
            triangle = np.stack((apex, far[index], far[next_index]))
            normal = np.cross(far_edge, apex - far[index])
        elif far_len < _DEGENERATE_COLUMN_MM and near_len >= _DEGENERATE_COLUMN_MM:
            apex = 0.5 * (far[index] + far[next_index])
            triangle = np.stack((near[index], near[next_index], apex))
            normal = np.cross(near_edge, apex - near[index])
        else:
            triangle = None
            normal = np.cross(near_edge, far[index] - near[index])
        length = float(np.linalg.norm(normal))
        if length <= 1.0e-12:
            # A collapsed column (coincident plan samples, or a ruling parallel
            # to the plan edge) spans no area and cannot orient itself.
            continue
        normal = normal / length
        centroid = 0.25 * (near[index] + near[next_index] + far[index] + far[next_index])
        if float(np.dot(normal, centroid - center)) < 0.0:
            normal = -normal
        if triangle is not None:
            parts.append(
                _flat_triangle(
                    role,
                    triangle,
                    tuple(float(value) for value in normal),
                    include_curvature=include_curvature,
                )
            )
        else:
            parts.append(
                _flat_strip(
                    role,
                    near[[index, next_index]],
                    far[[index, next_index]],
                    tuple(float(value) for value in normal),
                    closed_phi=False,
                    include_curvature=include_curvature,
                )
            )
    if not parts:
        raise ValueError(f"{role}: every faceted column collapsed")
    return _combine_surfaces(role, parts)


def _analytic_fillet(enclosure: Mapping[str, Any]) -> bool:
    """True when the fillet is a rounded box edge with a closed-form model.

    A rounded-rectangle fillet sweeps its plan by ``inset = edge*(1-sin(theta))``
    and ``radius = edge*sin(theta)`` while dropping ``depth*(1-cos(theta))``, so
    every corner arc keeps the *same* centre, ``edge`` in from the box corner.
    That makes each corner exactly one octant of the spheroid with semi-axes
    ``(edge, edge, depth)`` about that centre and each side exactly one quarter
    of an elliptic cylinder -- see :func:`_fillet_pieces`. Ellipse and
    superellipse plans have no such fixed centres and keep the sampled path.
    """

    return int(enclosure["plan_type"]) == 1 and int(enclosure["edge_type"]) == 1


def _fillet_corner_frames(
    enclosure: Mapping[str, Any]
) -> list[tuple[float, float, float]]:
    """The four corner-arc centres and the phi each octant starts at."""

    bounds = enclosure["bounds"]
    edge = float(enclosure["edge_mm"])
    x1 = float(bounds["bx1"]) - edge
    x0 = float(bounds["bx0"]) + edge
    y1 = float(bounds["by1"]) - edge
    y0 = float(bounds["by0"]) + edge
    return [
        (x1, y1, 0.0),
        (x0, y1, 0.5 * math.pi),
        (x0, y0, math.pi),
        (x1, y0, 1.5 * math.pi),
    ]


def _fillet_inset_rectangle(
    enclosure: Mapping[str, Any], z: float
) -> NDArray[np.float64]:
    """The sharp rectangle a rounded-box fillet is tangent to.

    ``_plan_ring`` floors its corner radius at 0.1 mm because
    ``sample_rounded_rect`` changes vertex count below 1e-3 mm, but the fillet
    genuinely runs out to a sharp corner there -- that is where the solver
    starts its roundover, and where :func:`_fillet_pieces` puts each octant's
    pole. The baffle has to end on the same curve, or the 0.041 mm between the
    floored arc and the corner belongs to neither surface.
    """

    return np.asarray(
        [(x, y, float(z)) for x, y, _phi in _fillet_corner_frames(enclosure)],
        dtype=np.float64,
    )


def _fillet_phi_intervals(
    theta: float, edge: float, depth: float, chord: float, normal_deg: float, cap: int
) -> int:
    """How many phi intervals a spheroid octant's row at ``theta`` really needs.

    The octant's unit normal is proportional to ``(sin(theta) cos(phi)/edge,
    sin(theta) sin(phi)/edge, cos(theta)/depth)``: turning ``phi`` rotates only
    the tangential part, so the normal step closes on zero towards the pole
    exactly as the arc radius ``edge*sin(theta)`` does. Spending the equator's
    sample count on every row -- what one fixed ``corner_intervals`` does -- is
    what put thirty-two samples on a 0.1 mm arc and then fanned them onto the
    next ring. Zero intervals means the row is the pole itself.
    """

    radius = edge * math.sin(theta)
    if radius <= 0.0:
        return 0
    by_chord = 1
    if radius > chord:
        half = math.acos(float(np.clip(1.0 - chord / radius, -1.0, 1.0)))
        by_chord = int(math.ceil(0.5 * math.pi / max(2.0 * half, 1.0e-12)))
    tangential = math.sin(theta) / edge
    axial = math.cos(theta) / depth
    by_normal = 1
    scale = tangential * tangential + axial * axial
    if tangential > 0.0:
        cosine = 1.0 - (1.0 - math.cos(math.radians(normal_deg))) * scale / (
            tangential * tangential
        )
        if cosine > -1.0:
            step = math.acos(float(np.clip(cosine, -1.0, 1.0)))
            by_normal = int(math.ceil(0.5 * math.pi / max(step, 1.0e-12)))
    return max(1, min(int(cap), max(by_chord, by_normal)))


def _fillet_patch(
    role: str,
    rows: list[NDArray[np.float64]],
    normals: list[NDArray[np.float64]],
    params: list[NDArray[np.float64]],
    curvature: list[NDArray[np.float64]] | None,
) -> PreviewSurfaceV1:
    """Stitch rows that may hold different sample counts into one patch.

    Rows carry a shared parameter, so the stitch is a two-pointer merge on it:
    a row that collapses to a single sample (the octant's pole) becomes a fan
    rather than a column of needles, and a row that gains samples over its
    neighbour picks them up one triangle at a time.
    """

    offsets = [0]
    for row in rows[:-1]:
        offsets.append(offsets[-1] + len(row))
    triangles: list[int] = []
    for level in range(len(rows) - 1):
        lower, upper = params[level], params[level + 1]
        base_l, base_u = offsets[level], offsets[level + 1]
        i = j = 0
        while i < len(lower) - 1 or j < len(upper) - 1:
            take_lower = j >= len(upper) - 1 or (
                i < len(lower) - 1 and lower[i + 1] <= upper[j + 1]
            )
            if take_lower:
                triangles.extend((base_l + i, base_l + i + 1, base_u + j))
                i += 1
            else:
                triangles.extend((base_l + i, base_u + j + 1, base_u + j))
                j += 1
    positions = np.vstack(rows)
    normal_array = np.vstack(normals)
    oriented = _orient_indices_to_normals(
        role, positions, np.asarray(triangles, dtype=np.uint32), normal_array
    )
    return PreviewSurfaceV1(
        role=role,
        positions=positions,
        indices=oriented.indices,
        normals=normal_array,
        shading="smooth",
        normal_method="analytic-parametric",
        closed_phi=False,
        curvature_mean=(None if curvature is None else np.concatenate(curvature[0])),
        curvature_principal=(
            None if curvature is None else np.concatenate(curvature[1])
        ),
        metadata=_orientation_metadata(oriented),
    )


def _fillet_pieces(
    enclosure: Mapping[str, Any],
    *,
    z_ref: float,
    axial_sign: float,
    rows: int,
    corner_cap: int,
    chord_target: float,
    normal_target: float,
    include_curvature: bool,
) -> tuple[list[PreviewSurfaceV1], dict[str, float]]:
    """Emit a rounded-box fillet as four cylinder strips and four octants.

    ``axial_sign`` is +1 for the front band (which drops away from ``z_ref``
    towards the sides) and -1 for the back one. Positions and normals are the
    closed-form surface, not differences of a sampled grid, so the returned
    fidelity is a measurement of the emitted triangles against that surface
    rather than against a resampling of themselves.
    """

    edge = float(enclosure["edge_mm"])
    depth = float(enclosure["edge_depth"])
    frames = _fillet_corner_frames(enclosure)
    theta = np.linspace(0.0, 0.5 * math.pi, rows + 1)
    # A patch bows away from its chords in both parameters at once and both
    # bows point the same way, so the two budgets add rather than compete. The
    # row count is fixed by the caller and is normal-step bound at every LOD,
    # which leaves the whole chord budget to be split; giving phi half of it
    # keeps the measured total inside what was asked for.
    phi_chord_target = 0.5 * chord_target
    intervals = [
        _fillet_phi_intervals(
            float(value), edge, depth, phi_chord_target, normal_target, corner_cap
        )
        for value in theta
    ]
    intervals[-1] = int(corner_cap)
    for index in range(1, len(intervals)):
        intervals[index] = max(intervals[index], intervals[index - 1])

    def surface_point(cx: float, cy: float, t: float, phi: float) -> NDArray[np.float64]:
        radius = edge * math.sin(t)
        return np.asarray(
            (
                cx + radius * math.cos(phi),
                cy + radius * math.sin(phi),
                z_ref - axial_sign * depth * (1.0 - math.cos(t)),
            ),
            dtype=np.float64,
        )

    def surface_normal(t: float, phi: float) -> NDArray[np.float64]:
        vector = np.asarray(
            (
                math.sin(t) * math.cos(phi) / edge,
                math.sin(t) * math.sin(phi) / edge,
                axial_sign * math.cos(t) / depth,
            ),
            dtype=np.float64,
        )
        return vector / np.linalg.norm(vector)

    def curvatures(t: float, *, ruled: bool) -> tuple[float, float]:
        # Meridian and parallel curvature of the spheroid of revolution with
        # profile ``(edge sin t, depth cos t)``; a side strip is the same
        # meridian swept along a straight ruling, so its parallel curvature is
        # zero. Positive is convex towards the outward normal above.
        root = math.hypot(edge * math.cos(t), depth * math.sin(t))
        meridian = edge * depth / max(root**3, 1.0e-30)
        parallel = 0.0 if ruled else depth / max(edge * root, 1.0e-30)
        mean = 0.5 * (meridian + parallel)
        principal = meridian if abs(meridian) >= abs(parallel) else parallel
        return mean, principal

    pieces: list[PreviewSurfaceV1] = []
    for index, (cx, cy, phi0) in enumerate(frames):
        corner_rows: list[NDArray[np.float64]] = []
        corner_normals: list[NDArray[np.float64]] = []
        corner_params: list[NDArray[np.float64]] = []
        corner_mean: list[NDArray[np.float64]] = []
        corner_principal: list[NDArray[np.float64]] = []
        for level, t in enumerate(theta):
            count = intervals[level]
            fractions = (
                np.zeros(1, dtype=np.float64)
                if count == 0
                else np.linspace(0.0, 1.0, count + 1)
            )
            phis = phi0 + fractions * 0.5 * math.pi
            corner_rows.append(
                np.asarray(
                    [surface_point(cx, cy, float(t), float(p)) for p in phis],
                    dtype=np.float64,
                )
            )
            corner_normals.append(
                np.asarray(
                    [surface_normal(float(t), float(p)) for p in phis],
                    dtype=np.float64,
                )
            )
            corner_params.append(fractions)
            mean, principal = curvatures(float(t), ruled=False)
            corner_mean.append(np.full(len(phis), mean, dtype=np.float64))
            corner_principal.append(np.full(len(phis), principal, dtype=np.float64))
        pieces.append(
            _fillet_patch(
                "enclosure.roundover",
                corner_rows,
                corner_normals,
                corner_params,
                (corner_mean, corner_principal) if include_curvature else None,
            )
        )

        # The side that leaves this corner: a quarter of an elliptic cylinder
        # ruled between this octant's end tangent line and the next octant's
        # start tangent line. Two columns describe it exactly.
        nx, ny, next_phi0 = frames[(index + 1) % len(frames)]
        side_rows: list[NDArray[np.float64]] = []
        side_normals: list[NDArray[np.float64]] = []
        side_params: list[NDArray[np.float64]] = []
        side_mean: list[NDArray[np.float64]] = []
        side_principal: list[NDArray[np.float64]] = []
        for t in theta:
            end_phi = phi0 + 0.5 * math.pi
            side_rows.append(
                np.stack(
                    (
                        surface_point(cx, cy, float(t), end_phi),
                        surface_point(nx, ny, float(t), next_phi0),
                    )
                )
            )
            normal = surface_normal(float(t), end_phi)
            side_normals.append(np.stack((normal, normal)))
            side_params.append(np.asarray((0.0, 1.0), dtype=np.float64))
            mean, principal = curvatures(float(t), ruled=True)
            side_mean.append(np.full(2, mean, dtype=np.float64))
            side_principal.append(np.full(2, principal, dtype=np.float64))
        pieces.append(
            _fillet_patch(
                "enclosure.roundover",
                side_rows,
                side_normals,
                side_params,
                (side_mean, side_principal) if include_curvature else None,
            )
        )

    # Measure the emitted band against the closed-form surface it was built
    # from. All four octants are congruent and the sides share their meridian,
    # so one meridian and one octant bound the whole band; a chord is deepest
    # at its parameter midpoint. This replaces comparing the grid against a
    # reference built at the same interval count, which measured nothing.
    def angle_between(
        first: NDArray[np.float64], second: NDArray[np.float64]
    ) -> float:
        return math.degrees(
            math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0)))
        )

    origin = frames[0]
    meridian_chord = 0.0
    normal_step = 0.0
    for level in range(rows):
        lower, upper = float(theta[level]), float(theta[level + 1])
        emitted = 0.5 * (
            surface_point(origin[0], origin[1], lower, origin[2])
            + surface_point(origin[0], origin[1], upper, origin[2])
        )
        meridian_chord = max(
            meridian_chord,
            float(
                np.linalg.norm(
                    surface_point(
                        origin[0], origin[1], 0.5 * (lower + upper), origin[2]
                    )
                    - emitted
                )
            ),
        )
        normal_step = max(
            normal_step,
            angle_between(surface_normal(lower, 0.0), surface_normal(upper, 0.0)),
        )
    arc_chord = 0.0
    for level, count in enumerate(intervals):
        if count <= 0:
            continue
        step = 0.5 * math.pi / count
        arc_chord = max(
            arc_chord,
            edge * math.sin(float(theta[level])) * (1.0 - math.cos(0.5 * step)),
        )
        normal_step = max(
            normal_step,
            angle_between(
                surface_normal(float(theta[level]), 0.0),
                surface_normal(float(theta[level]), step),
            ),
        )
    fidelity = {
        "max_chord_error_mm": max(
            meridian_chord + arc_chord, float(np.finfo(np.float64).eps)
        ),
        "max_normal_step_deg": normal_step,
        "reference_density_multiplier": 4,
    }
    return pieces, fidelity


def _enclosure_surfaces(
    enclosure: Mapping[str, Any],
    mouth: NDArray[np.float64],
    roundover_intervals: int,
    plan_corner_intervals: int,
    *,
    chord_target: float,
    normal_target: float,
    include_rear: bool,
    include_curvature: bool,
) -> tuple[list[PreviewSurfaceV1], dict[str, dict[str, float]]]:
    bounds = enclosure["bounds"]
    depth = float(enclosure["edge_depth"])
    rounded_edge = int(enclosure["edge_type"]) == 1
    z_front = float(bounds["z_front"])
    z_back = float(bounds["z_back"])
    center_xy = np.asarray((float(bounds["cx"]), float(bounds["cy"])), dtype=np.float64)

    analytic_fillet = depth > 0.0 and _analytic_fillet(enclosure)
    front_native = (
        _fillet_inset_rectangle(enclosure, z_front)
        if analytic_fillet
        else _plan_ring(enclosure, z_front, 0.0, corner_intervals=plan_corner_intervals)
    )
    front_aligned = _ray_aligned_ring(front_native, mouth)
    # The aligned ring only preserves the plan where the mouth stations sample
    # it. On an asymmetric box the plan corners fall between stations, the
    # aligned polyline chord-cuts them, and the flat baffle plane between that
    # chord and the corner is covered by neither the baffle nor the edge band:
    # a real hole (chamfer) or an off-surface ruling (fillet). Give every
    # missed corner its own column in both rings.
    mouth, front_aligned = _insert_corner_columns(mouth, front_aligned, front_native)
    # One annulus, not two. An enclosure horn has no wall end face to name --
    # ``config_builder`` forces wall thickness to zero and ``HornEnclosure``
    # forbids outer points, so the baffle runs unbroken from the mouth to the
    # edge treatment. Splitting it at the midpoint used to emit half of it as
    # ``mouth_rim``, which the renderer paints in the horn material and outlines
    # in edge mode: a hard colour seam and a drawn feature line partway across a
    # surface that is flat and continuous.
    surfaces = [
        _flat_strip(
            "enclosure.front",
            mouth,
            front_aligned,
            (0.0, 0.0, 1.0),
            closed_phi=True,
            include_curvature=include_curvature,
        ),
    ]
    fidelity: dict[str, dict[str, float]] = {}

    if depth > 0.0:
        def front_grid(intervals: int) -> NDArray[np.float64]:
            rings = []
            for fraction in np.linspace(0.0, 1.0, intervals + 1):
                if rounded_edge:
                    theta = float(fraction) * math.pi / 2.0
                    axial_t = 1.0 - math.cos(theta)
                    radial_t = math.sin(theta)
                else:
                    axial_t = radial_t = float(fraction)
                rings.append(
                    _plan_ring(
                        enclosure,
                        z_front - axial_t * depth,
                        radial_t,
                        corner_intervals=plan_corner_intervals,
                    )
                )
            return np.asarray(rings, dtype=np.float64)

        def back_grid(intervals: int) -> NDArray[np.float64]:
            rings = []
            for fraction in np.linspace(0.0, 1.0, intervals + 1):
                if rounded_edge:
                    theta = float(fraction) * math.pi / 2.0
                    axial_t = math.sin(theta)
                    radial_t = math.cos(theta)
                else:
                    axial_t = float(fraction)
                    radial_t = 1.0 - float(fraction)
                rings.append(
                    _plan_ring(
                        enclosure,
                        z_back + (1.0 - axial_t) * depth,
                        radial_t,
                        corner_intervals=plan_corner_intervals,
                    )
                )
            return np.asarray(rings, dtype=np.float64)

        edge_intervals = roundover_intervals if rounded_edge else 1
        front_out = front_grid(edge_intervals)
        back_out = back_grid(edge_intervals)
        if _faceted_edge(enclosure):
            # A rounded-rectangle chamfer is a ruled band between two
            # piecewise-linear plans, so it is exactly a fan of planes. Rule it
            # in the plan's own parameterisation -- the ray-aligned front ring
            # was welded to the baffle but forced an unequal-ring zipper that
            # fanned dozens of mouth stations onto one plan sample, and the
            # resulting slivers were then smooth-shaded across tangent breaks
            # that are real. The inner boundary still lands on the baffle's
            # outer edge, now as coincident points along it rather than shared
            # ones.
            center_xyz = np.asarray(
                (center_xy[0], center_xy[1], 0.5 * (z_front + z_back)), dtype=np.float64
            )
            surfaces.append(
                _combine_surfaces(
                    "enclosure.roundover",
                    [
                        _faceted_band(
                            "enclosure.roundover",
                            front_out[0],
                            front_out[-1],
                            center_xyz,
                            include_curvature=include_curvature,
                        ),
                        _faceted_band(
                            "enclosure.roundover",
                            back_out[-1],
                            back_out[0],
                            center_xyz,
                            include_curvature=include_curvature,
                        ),
                    ],
                )
            )
            # Planes, exactly represented. The 45 degree steps between adjacent
            # facets are the geometry itself, not a sampling error, and no
            # subdivision reduces them.
            fidelity["enclosure.roundover"] = {
                "max_chord_error_mm": 0.0,
                "max_normal_step_deg": 0.0,
                "reference_density_multiplier": 1,
            }
        elif analytic_fillet:
            # A rounded-box fillet has a closed form, so it does not need the
            # ray-aligned first ring and the unequal-ring zipper that welded it
            # to the baffle. That stitch fanned the mouth's evenly spread
            # stations onto a ring carrying one sample per straight side, which
            # is where the band's 7000:1 needles came from; the plan sampler
            # then spent a full corner_intervals on every ring, including the
            # 0.1 mm floored one. The band's inner boundary still lands on the
            # baffle's outer edge -- as coincident points along the same inset
            # rectangle rather than shared ones, exactly as the chamfer does.
            front_pieces, front_fidelity = _fillet_pieces(
                enclosure,
                z_ref=z_front,
                axial_sign=1.0,
                rows=edge_intervals,
                corner_cap=plan_corner_intervals,
                chord_target=chord_target,
                normal_target=normal_target,
                include_curvature=include_curvature,
            )
            back_pieces, back_fidelity = _fillet_pieces(
                enclosure,
                z_ref=z_back,
                axial_sign=-1.0,
                rows=edge_intervals,
                corner_cap=plan_corner_intervals,
                chord_target=chord_target,
                normal_target=normal_target,
                include_curvature=include_curvature,
            )
            surfaces.append(
                _combine_surfaces(
                    "enclosure.roundover", [*front_pieces, *back_pieces]
                )
            )
            fidelity["enclosure.roundover"] = {
                key: max(front_fidelity[key], back_fidelity[key])
                for key in ("max_chord_error_mm", "max_normal_step_deg")
            } | {"reference_density_multiplier": 4}
        else:
            # An ellipse or superellipse plan has no fixed corner centres, so
            # its roundover stays a sampled grid. Its reference must be denser
            # than the emitted grid or the comparison measures nothing.
            front_surface, front_fidelity = _roundover_piece(
                "enclosure.roundover",
                front_aligned,
                front_out,
                front_grid(2 * edge_intervals),
                center_xy,
                include_curvature=include_curvature,
            )
            back_surface, back_fidelity = _roundover_piece(
                "enclosure.roundover",
                back_out[0],
                back_out,
                back_grid(2 * edge_intervals),
                center_xy,
                include_curvature=include_curvature,
            )
            surfaces.append(
                _combine_surfaces("enclosure.roundover", [front_surface, back_surface])
            )
            fidelity["enclosure.roundover"] = {
                key: max(front_fidelity[key], back_fidelity[key])
                for key in ("max_chord_error_mm", "max_normal_step_deg")
            } | {"reference_density_multiplier": 4}
        side_front = front_out[-1]
        side_back = back_out[0]
    else:
        side_front = front_native
        side_back = _plan_ring(
            enclosure, z_back, 0.0, corner_intervals=plan_corner_intervals
        )

    side_grid = np.stack((side_front, side_back), axis=0)
    # Four axial reference intervals measure the exact canonical extrusion.
    side_ref = np.stack(
        [
            side_front + (side_back - side_front) * fraction
            for fraction in np.linspace(0.0, 1.0, 5)
        ],
        axis=0,
    )
    if int(enclosure["edge_type"]) == 2:
        side_surface = _hard_side_surface(
            side_front, side_back, include_curvature=include_curvature
        )
        side_fidelity = _plan_fidelity(enclosure, plan_corner_intervals)
    else:
        side_surface, side_fidelity = _smooth_grid_surface(
            "enclosure.side",
            side_grid,
            side_ref,
            closed_phi=True,
            orientation_hint=np.stack(
                (
                    side_grid[:, :, 0] - center_xy[0],
                    side_grid[:, :, 1] - center_xy[1],
                    np.zeros(side_grid.shape[:2], dtype=np.float64),
                ),
                axis=2,
            ),
            include_curvature=include_curvature,
        )
    surfaces.append(side_surface)
    fidelity["enclosure.side"] = side_fidelity
    if include_rear:
        if analytic_fillet:
            # Same sharp tangent rectangle the back band's poles sit on.
            rear_ring = _fillet_inset_rectangle(enclosure, z_back)
        elif depth > 0.0:
            rear_ring = back_out[-1]
        else:
            rear_ring = side_back
        surfaces.append(
            _flat_cap(
                "enclosure.rear",
                rear_ring,
                (0.0, 0.0, -1.0),
                closed_phi=True,
                include_curvature=include_curvature,
                # The rear ring carries the sampler's floored corners (a 2 um
                # chamfer chord; a fillet's 0.1 mm arc in dozens of samples).
                # Fanned against a center hundreds of mm away they become
                # 50000:1 slivers. The real rear face corner is sharp.
                simplify_tolerance=0.15,
            )
        )
    return surfaces, fidelity
