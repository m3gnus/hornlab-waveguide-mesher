"""Public ``hornlab.preview/1`` geometry API."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..builders.enclosure import sample_enclosure_plan
from ..builders.point_grid_sources import _source_cap_height, _source_cap_radius
from ..freeform import build_freeform_geometry
from ..geometry import PointGridHornGeometry
from ..profiles import eval_param
from ..viewport import build_viewport_geometry_from_config
from .fidelity import (
    analytic_grid_normals,
    estimate_grid_fidelity,
    resample_grid_vectors,
    resample_parametric_grid,
)


_API_VERSION = "hornlab.preview/1"
_LOD_PRESETS = {
    "coarse": (64, 12, 6, 8),
    "fine": (96, 48, 12, 16),
    "inspection": (160, 80, 16, 24),
}


@dataclass(frozen=True)
class PreviewOptionsV1:
    lod: str = "fine"
    include_outer: bool = True
    include_enclosure: bool = True
    include_source_cap: bool = True
    include_rear_cap: bool = True
    # Accepted for wire compatibility. Stage 1 measures but does not target them.
    max_chord_error_mm: float | None = None
    max_normal_step_deg: float | None = None


@dataclass(frozen=True)
class PreviewSurfaceV1:
    role: str
    positions: NDArray[np.float64]
    indices: NDArray[np.uint32]
    normals: NDArray[np.float64]
    shading: str
    normal_method: str
    closed_phi: bool

    def __post_init__(self) -> None:
        positions = np.ascontiguousarray(self.positions, dtype=np.float64)
        normals = np.ascontiguousarray(self.normals, dtype=np.float64)
        indices = np.ascontiguousarray(self.indices, dtype=np.uint32).reshape(-1)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"{self.role}: positions must have shape (N, 3)")
        if normals.shape != positions.shape:
            raise ValueError(f"{self.role}: normals must be row-aligned with positions")
        if indices.size % 3:
            raise ValueError(f"{self.role}: indices must contain triangles")
        if indices.size and int(indices.max()) >= len(positions):
            raise ValueError(f"{self.role}: index exceeds vertex count")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(normals)):
            raise ValueError(f"{self.role}: positions and normals must be finite")
        if not np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1.0e-3):
            raise ValueError(f"{self.role}: normals must be unit length")
        if self.shading not in {"smooth", "flat"}:
            raise ValueError(f"{self.role}: unsupported shading {self.shading!r}")
        if self.normal_method not in {"analytic-parametric", "exact-planar"}:
            raise ValueError(
                f"{self.role}: unsupported normal method {self.normal_method!r}"
            )
        positions.setflags(write=False)
        normals.setflags(write=False)
        indices.setflags(write=False)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "indices", indices)


@dataclass(frozen=True)
class PreviewGeometryV1:
    surfaces: list[PreviewSurfaceV1]
    metadata: dict[str, Any] = field(default_factory=dict)


def _lod_config(config: Mapping[str, Any], angular: int, axial: int) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    mesh = result.get("mesh")
    if not isinstance(mesh, Mapping):
        mesh = {}
    else:
        mesh = dict(mesh)
    mesh["angular_segments"] = int(angular)
    mesh["length_segments"] = int(axial)
    # Canonical names win over any imported/camel-case aliases in config_builder.
    mesh.pop("angularSegments", None)
    mesh.pop("lengthSegments", None)
    result["mesh"] = mesh
    return result


def _grid(raw: Any, n_phi: int, n_length: int) -> NDArray[np.float64]:
    return np.asarray(raw, dtype=np.float64).reshape(n_phi, n_length + 1, 3)


def _surface_grid(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert canonical ``(phi,t,xyz)`` to surface ``(t,phi,xyz)``."""

    return np.ascontiguousarray(np.transpose(points, (1, 0, 2)), dtype=np.float64)


def _grid_indices(n_t: int, n_phi: int, *, closed_phi: bool) -> NDArray[np.uint32]:
    triangles: list[int] = []
    phi_intervals = n_phi if closed_phi else n_phi - 1
    for jt in range(n_t - 1):
        row0 = jt * n_phi
        row1 = (jt + 1) * n_phi
        for ip in range(phi_intervals):
            ip1 = (ip + 1) % n_phi
            triangles.extend((row0 + ip, row0 + ip1, row1 + ip1))
            triangles.extend((row0 + ip, row1 + ip1, row1 + ip))
    return np.asarray(triangles, dtype=np.uint32)


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
) -> tuple[PreviewSurfaceV1, dict[str, float]]:
    ref_normals = analytic_grid_normals(
        reference,
        closed_phi=closed_phi,
        t_coordinates=reference_t,
        phi_coordinates=reference_phi,
    )
    if closed_phi and any(
        value is not None
        for value in (point_t, point_phi, reference_t, reference_phi)
    ):
        normals = resample_parametric_grid(
            ref_normals,
            points.shape[:2],
            source_t=reference_t,
            source_phi=reference_phi,
            target_t=point_t,
            target_phi=point_phi,
            normalise=True,
        )
    else:
        normals = resample_grid_vectors(
            ref_normals, points.shape[:2], closed_phi=closed_phi
        )
    if orientation_hint is not None:
        hint = np.broadcast_to(np.asarray(orientation_hint, dtype=np.float64), normals.shape)
        if float(np.median(np.sum(normals * hint, axis=2))) < 0.0:
            normals = -normals
    surface = PreviewSurfaceV1(
        role=role,
        positions=points.reshape(-1, 3),
        indices=_grid_indices(*points.shape[:2], closed_phi=closed_phi),
        normals=normals.reshape(-1, 3),
        shading="smooth",
        normal_method="analytic-parametric",
        closed_phi=closed_phi,
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
) -> PreviewSurfaceV1:
    points = np.stack((inner, outer), axis=0)
    normals = np.broadcast_to(np.asarray(normal, dtype=np.float64), points.shape).copy()
    return PreviewSurfaceV1(
        role=role,
        positions=points.reshape(-1, 3),
        indices=_grid_indices(2, points.shape[1], closed_phi=closed_phi),
        normals=normals.reshape(-1, 3),
        shading="flat",
        normal_method="exact-planar",
        closed_phi=closed_phi,
    )


def _flat_cap(
    role: str,
    ring: NDArray[np.float64],
    normal: tuple[float, float, float],
    *,
    closed_phi: bool,
) -> PreviewSurfaceV1:
    center = np.mean(ring, axis=0)
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
    return PreviewSurfaceV1(
        role=role,
        positions=positions,
        indices=np.asarray(triangles, dtype=np.uint32),
        normals=normals,
        shading="flat",
        normal_method="exact-planar",
        closed_phi=closed_phi,
    )


def _source_geometry(
    params: Mapping[str, Any], formula: str, inner: NDArray[np.float64]
) -> PointGridHornGeometry:
    auto_angle = float(eval_param(params.get("a0"), 0.0, 15.5))
    if formula == "FREEFORM":
        auto_angle = float(
            build_freeform_geometry(params).report()["tangentAnglesDeg"]["H"]["throat"]
        )
    return PointGridHornGeometry(
        inner_points=inner,
        source_shape=int(round(float(eval_param(params.get("sourceShape"), 0.0, 1)))),
        source_radius_mm=float(eval_param(params.get("sourceRadius"), 0.0, -1.0)),
        source_curv=int(round(float(eval_param(params.get("sourceCurv"), 0.0, 0)))),
        source_auto_angle_deg=auto_angle,
    )


def _source_cap(
    inner: NDArray[np.float64],
    params: Mapping[str, Any],
    formula: str,
    radial_intervals: int,
    *,
    closed_phi: bool,
) -> tuple[PreviewSurfaceV1, dict[str, float] | None, dict[str, float]]:
    ring = np.asarray(inner[:, 0, :], dtype=np.float64)
    center = np.mean(ring, axis=0)
    radial = ring[:, :2] - center[:2]
    radii = np.linalg.norm(radial, axis=1)
    throat_radius = float(np.mean(radii[radii > 1.0e-12]))
    geometry = _source_geometry(params, formula, inner)
    cap_height = _source_cap_height(throat_radius, geometry)
    radius = _source_cap_radius(throat_radius, geometry)
    details = {
        "source_cap_height_mm": float(cap_height),
        "source_cap_radius_mm": float(radius),
    }
    if int(geometry.source_shape) == 0 or cap_height <= 1.0e-12 or not math.isfinite(radius):
        return _flat_cap("source_cap", ring, (0.0, 0.0, 1.0), closed_phi=closed_phi), None, details

    radius = max(float(radius), throat_radius * 1.001)
    sign = -1.0 if int(geometry.source_curv) == -1 else 1.0
    sphere_center = center.copy()
    sphere_center[2] += sign * (cap_height - radius)
    rim_angle = math.asin(np.clip(throat_radius / radius, -1.0, 1.0))
    directions = radial / radii[:, None]

    positions: list[NDArray[np.float64]] = []
    normals: list[NDArray[np.float64]] = []
    pole = sphere_center.copy()
    pole[2] += sign * radius
    positions.append(pole)
    normals.append(np.asarray((0.0, 0.0, sign), dtype=np.float64))
    for level in range(1, radial_intervals + 1):
        theta = rim_angle * level / radial_intervals
        rho = radius * math.sin(theta)
        z = sphere_center[2] + sign * radius * math.cos(theta)
        for direction in directions:
            point = np.asarray(
                (center[0] + rho * direction[0], center[1] + rho * direction[1], z),
                dtype=np.float64,
            )
            positions.append(point)
            normals.append(sign * (point - sphere_center) / radius)

    triangles: list[int] = []
    n_phi = len(ring)
    limit = n_phi if closed_phi else n_phi - 1
    first_ring = 1
    for ip in range(limit):
        ip1 = (ip + 1) % n_phi
        triangles.extend((0, first_ring + ip, first_ring + ip1))
    for level in range(1, radial_intervals):
        row0 = 1 + (level - 1) * n_phi
        row1 = 1 + level * n_phi
        for ip in range(limit):
            ip1 = (ip + 1) % n_phi
            triangles.extend((row0 + ip, row1 + ip, row1 + ip1))
            triangles.extend((row0 + ip, row1 + ip1, row0 + ip1))

    normal_array = np.asarray(normals, dtype=np.float64)
    surface = PreviewSurfaceV1(
        role="source_cap",
        positions=np.asarray(positions, dtype=np.float64),
        indices=np.asarray(triangles, dtype=np.uint32),
        normals=normal_array,
        shading="smooth",
        normal_method="analytic-parametric",
        closed_phi=closed_phi,
    )
    angular_step = 2.0 * math.pi / n_phi if closed_phi else math.pi / max(n_phi - 1, 1)
    polar_step = rim_angle / radial_intervals
    max_angle = max(polar_step, math.sin(rim_angle) * angular_step)
    fidelity = {
        "max_chord_error_mm": max(
            radius * (1.0 - math.cos(polar_step / 2.0)),
            radius * (1.0 - math.cos(max_angle / 2.0)),
        ),
        "max_normal_step_deg": math.degrees(max_angle),
        "reference_density_multiplier": 4,
    }
    return surface, fidelity, details


def _ccw_ring(points: NDArray[np.float64]) -> NDArray[np.float64]:
    ring = np.asarray(points, dtype=np.float64)
    x = ring[:, 0]
    y = ring[:, 1]
    area2 = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return ring if area2 >= 0.0 else ring[::-1].copy()


def _plan_ring(enclosure: Mapping[str, Any], z: float, radial_t: float) -> NDArray[np.float64]:
    bounds = enclosure["bounds"]
    edge = float(enclosure["edge_mm"])
    d = edge * (1.0 - radial_t)
    radius = max(0.1, edge * radial_t)
    return _ccw_ring(
        sample_enclosure_plan(
            bx0=float(bounds["bx0"]) + d,
            bx1=float(bounds["bx1"]) - d,
            by0=float(bounds["by0"]) + d,
            by1=float(bounds["by1"]) - d,
            corner_radius=radius,
            edge_type=int(enclosure["edge_type"]),
            z=float(z),
            plan_type=int(enclosure["plan_type"]),
            plan_n=float(enclosure.get("plan_n", 2.0)),
        )
    )


def _ray_aligned_ring(
    plan: NDArray[np.float64], reference: NDArray[np.float64]
) -> NDArray[np.float64]:
    center = np.mean(reference[:, :2], axis=0)
    result = np.empty_like(reference)
    for index, point in enumerate(reference):
        direction = point[:2] - center
        direction /= max(float(np.linalg.norm(direction)), 1.0e-14)
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
            result[index] = plan[int(np.argmin(delta))]
        else:
            result[index] = best[1]
    return result


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


def _roundover_piece(
    role: str,
    first_ring: NDArray[np.float64],
    output_grid: NDArray[np.float64],
    reference_grid: NDArray[np.float64],
    center_xy: NDArray[np.float64],
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
    native_hint = output_grid.copy()
    native_hint[:, :, 0] -= center_xy[0]
    native_hint[:, :, 1] -= center_xy[1]
    native_hint[:, :, 2] = 0.0
    if float(np.median(np.sum(output_normals * native_hint, axis=2))) < 0.0:
        output_normals = -output_normals
        first_ref_normals = -first_ref_normals
    positions = [first_ring, *list(output_grid[1:])]
    normals = [first_ref_normals, *list(output_normals[1:])]
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
    surface = PreviewSurfaceV1(
        role=role,
        positions=np.vstack(positions),
        indices=np.asarray(triangles, dtype=np.uint32),
        normals=np.vstack(normals),
        shading="smooth",
        normal_method="analytic-parametric",
        closed_phi=True,
    )
    fidelity = estimate_grid_fidelity(
        output_grid, reference_grid, output_normals, closed_phi=True
    )
    return surface, fidelity


def _combine_surfaces(role: str, parts: list[PreviewSurfaceV1]) -> PreviewSurfaceV1:
    positions: list[NDArray[np.float64]] = []
    normals: list[NDArray[np.float64]] = []
    indices: list[NDArray[np.uint32]] = []
    offset = 0
    for part in parts:
        positions.append(part.positions)
        normals.append(part.normals)
        indices.append(part.indices + np.uint32(offset))
        offset += len(part.positions)
    return PreviewSurfaceV1(
        role=role,
        positions=np.vstack(positions),
        indices=np.concatenate(indices),
        normals=np.vstack(normals),
        shading=parts[0].shading,
        normal_method=parts[0].normal_method,
        closed_phi=all(part.closed_phi for part in parts),
    )


def _enclosure_surfaces(
    enclosure: Mapping[str, Any],
    mouth: NDArray[np.float64],
    roundover_intervals: int,
    *,
    include_rear: bool,
) -> tuple[list[PreviewSurfaceV1], dict[str, dict[str, float]]]:
    bounds = enclosure["bounds"]
    depth = float(enclosure["edge_depth"])
    rounded_edge = int(enclosure["edge_type"]) == 1
    z_front = float(bounds["z_front"])
    z_back = float(bounds["z_back"])
    center_xy = np.asarray((float(bounds["cx"]), float(bounds["cy"])), dtype=np.float64)

    front_native = _plan_ring(enclosure, z_front, 0.0)
    front_aligned = _ray_aligned_ring(front_native, mouth)
    middle = 0.5 * (mouth + front_aligned)
    surfaces = [
        _flat_strip("mouth_rim", mouth, middle, (0.0, 0.0, 1.0), closed_phi=True),
        _flat_strip(
            "enclosure.front", middle, front_aligned, (0.0, 0.0, 1.0), closed_phi=True
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
                rings.append(_plan_ring(enclosure, z_front - axial_t * depth, radial_t))
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
                    _plan_ring(enclosure, z_back + (1.0 - axial_t) * depth, radial_t)
                )
            return np.asarray(rings, dtype=np.float64)

        edge_intervals = roundover_intervals if rounded_edge else 1
        front_out = front_grid(edge_intervals)
        front_ref = front_grid(edge_intervals * 4)
        front_surface, front_fidelity = _roundover_piece(
            "enclosure.roundover", front_aligned, front_out, front_ref, center_xy
        )

        back_out = back_grid(edge_intervals)
        back_ref = back_grid(edge_intervals * 4)
        back_surface, back_fidelity = _roundover_piece(
            "enclosure.roundover", back_out[0], back_out, back_ref, center_xy
        )
        surfaces.append(_combine_surfaces("enclosure.roundover", [front_surface, back_surface]))
        fidelity["enclosure.roundover"] = {
            key: max(front_fidelity[key], back_fidelity[key])
            for key in ("max_chord_error_mm", "max_normal_step_deg")
        } | {"reference_density_multiplier": 4}
        side_front = front_out[-1]
        side_back = back_out[0]
    else:
        side_front = front_native
        side_back = _plan_ring(enclosure, z_back, 0.0)

    side_grid = np.stack((side_front, side_back), axis=0)
    # Four axial reference intervals measure the exact canonical extrusion.
    side_ref = np.stack(
        [
            side_front + (side_back - side_front) * fraction
            for fraction in np.linspace(0.0, 1.0, 5)
        ],
        axis=0,
    )
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
    )
    surfaces.append(side_surface)
    fidelity["enclosure.side"] = side_fidelity
    if include_rear:
        rear_ring = back_out[-1] if depth > 0.0 else side_back
        surfaces.append(_flat_cap("enclosure.rear", rear_ring, (0.0, 0.0, -1.0), closed_phi=True))
    return surfaces, fidelity


def build_preview_geometry(
    config: Mapping[str, Any], options: PreviewOptionsV1 = PreviewOptionsV1()
) -> PreviewGeometryV1:
    """Build complete, versioned render geometry from a mesher config.

    Smooth normals use ``normalize(dP/dphi x dP/dt)`` from a four-times denser
    canonical analytic sampling.  The sign convention is the parameterisation
    convention: phi follows the canonical ring order and t runs throat-to-mouth
    (or front-to-rear for enclosure sweeps). Flat faces carry exact constant
    planar normals. Every role owns its vertices, so hard boundaries never
    share a row with incompatible normals.
    """

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if not isinstance(options, PreviewOptionsV1):
        raise TypeError("options must be PreviewOptionsV1")
    lod = str(options.lod).strip().lower()
    if lod not in _LOD_PRESETS:
        raise ValueError("lod must be 'coarse', 'fine', or 'inspection'")
    angular, axial, roundover_intervals, cap_intervals = _LOD_PRESETS[lod]
    start = time.perf_counter()
    warnings: list[str] = []
    if options.max_chord_error_mm is not None:
        warnings.append("max_chord_error_mm is reserved for stage 2 and was ignored")
    if options.max_normal_step_deg is not None:
        warnings.append("max_normal_step_deg is reserved for stage 2 and was ignored")

    canonical_start = time.perf_counter()
    output = build_viewport_geometry_from_config(_lod_config(config, angular, axial))
    reference = build_viewport_geometry_from_config(
        _lod_config(config, angular * 4, axial * 4)
    )
    canonical_ms = (time.perf_counter() - canonical_start) * 1000.0

    grid_data = output["grid"]
    ref_data = reference["grid"]
    n_phi = int(grid_data["grid_n_phi"])
    n_length = int(grid_data["grid_n_length"])
    ref_phi = int(ref_data["grid_n_phi"])
    ref_length = int(ref_data["grid_n_length"])
    closed_phi = bool(grid_data.get("full_circle", True))
    inner = _grid(grid_data["inner_points"], n_phi, n_length)
    ref_inner = _grid(ref_data["inner_points"], ref_phi, ref_length)
    point_t = np.asarray(grid_data.get("slice_map"), dtype=np.float64)
    reference_t = np.asarray(ref_data.get("slice_map"), dtype=np.float64)
    point_phi = (
        np.asarray(grid_data["phi_grid"], dtype=np.float64).T
        if grid_data.get("phi_grid") is not None
        else None
    )
    reference_phi = (
        np.asarray(ref_data["phi_grid"], dtype=np.float64).T
        if ref_data.get("phi_grid") is not None
        else None
    )

    assembly_start = time.perf_counter()
    surfaces: list[PreviewSurfaceV1] = []
    fidelity: dict[str, dict[str, float]] = {}
    inner_surface, inner_fidelity = _smooth_grid_surface(
        "horn.inner",
        _surface_grid(inner),
        _surface_grid(ref_inner),
        closed_phi=closed_phi,
        point_t=point_t,
        point_phi=point_phi,
        reference_t=reference_t,
        reference_phi=reference_phi,
    )
    surfaces.append(inner_surface)
    fidelity["horn.inner"] = inner_fidelity

    outer = ref_outer = None
    if grid_data.get("outer_points") is not None:
        outer = _grid(grid_data["outer_points"], n_phi, n_length)
        ref_outer = _grid(ref_data["outer_points"], ref_phi, ref_length)
        if options.include_outer:
            outer_surface, outer_fidelity = _smooth_grid_surface(
                "horn.outer",
                _surface_grid(outer),
                _surface_grid(ref_outer),
                closed_phi=closed_phi,
                point_t=point_t,
                point_phi=point_phi,
                reference_t=reference_t,
                reference_phi=reference_phi,
            )
            surfaces.append(outer_surface)
            fidelity["horn.outer"] = outer_fidelity
            rim_surface, rim_fidelity = _smooth_grid_surface(
                "mouth_rim",
                np.stack((inner[:, -1, :], outer[:, -1, :]), axis=0),
                np.stack((ref_inner[:, -1, :], ref_outer[:, -1, :]), axis=0),
                closed_phi=closed_phi,
                point_t=np.asarray((0.0, 1.0), dtype=np.float64),
                point_phi=(
                    np.repeat(point_phi[-1:, :], 2, axis=0)
                    if point_phi is not None
                    else None
                ),
                reference_t=np.asarray((0.0, 1.0), dtype=np.float64),
                reference_phi=(
                    np.repeat(reference_phi[-1:, :], 2, axis=0)
                    if reference_phi is not None
                    else None
                ),
                orientation_hint=np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
            )
            surfaces.append(rim_surface)
            fidelity["mouth_rim"] = rim_fidelity

    if options.include_source_cap:
        cap_surface, cap_fidelity, source_details = _source_cap(
            inner,
            output["params"],
            output["formula"],
            cap_intervals,
            closed_phi=closed_phi,
        )
        surfaces.append(cap_surface)
        if cap_fidelity is not None:
            fidelity["source_cap"] = cap_fidelity
    else:
        source_details = {}

    if output.get("enclosure") is not None and options.include_enclosure:
        enclosure_payload = dict(output["enclosure"])
        enclosure_config = config.get("enclosure")
        if isinstance(enclosure_config, Mapping):
            plan_n = enclosure_config.get(
                "plan_n", enclosure_config.get("planN", enclosure_config.get("encPlanN"))
            )
            if plan_n is not None:
                enclosure_payload["plan_n"] = float(plan_n)
        enclosure_surfaces, enclosure_fidelity = _enclosure_surfaces(
            enclosure_payload,
            inner[:, -1, :],
            roundover_intervals,
            include_rear=options.include_rear_cap,
        )
        surfaces.extend(enclosure_surfaces)
        fidelity.update(enclosure_fidelity)
    elif outer is not None and options.include_rear_cap:
        surfaces.append(
            _flat_cap(
                "wall.rear_cap", outer[:, 0, :], (0.0, 0.0, -1.0), closed_phi=closed_phi
            )
        )

    assembly_ms = (time.perf_counter() - assembly_start) * 1000.0
    total_ms = (time.perf_counter() - start) * 1000.0
    metadata: dict[str, Any] = {
        "api_version": _API_VERSION,
        "units": "mm",
        "coordinate_frame": "mesher-xyz",
        "formula": output["formula"],
        "mode": output["mode"],
        "lod": lod,
        "actual_segment_counts": {
            "horn_phi": n_phi,
            "horn_axial": n_length,
            "enclosure_roundover_quarter": (
                roundover_intervals
                if output.get("enclosure") is not None
                and float(output["enclosure"].get("edge_depth", 0.0)) > 0.0
                and options.include_enclosure
                and int(output["enclosure"].get("edge_type", 1)) == 1
                else 1
                if output.get("enclosure") is not None
                and float(output["enclosure"].get("edge_depth", 0.0)) > 0.0
                and options.include_enclosure
                else 0
            ),
            "source_cap_radial": (
                cap_intervals
                if options.include_source_cap
                and float(source_details.get("source_cap_height_mm", 0.0)) > 0.0
                else (1 if options.include_source_cap else 0)
            ),
        },
        "timings_ms": {
            "canonical_sampling": canonical_ms,
            "surface_assembly_and_fidelity": assembly_ms,
            "total": total_ms,
        },
        "fidelity": fidelity,
        "warnings": warnings,
        "normal_convention": "normalize(dP/dphi x dP/dt); exact constants on planar faces",
        **source_details,
    }
    return PreviewGeometryV1(surfaces=surfaces, metadata=metadata)


__all__ = [
    "PreviewGeometryV1",
    "PreviewOptionsV1",
    "PreviewSurfaceV1",
    "build_preview_geometry",
]
