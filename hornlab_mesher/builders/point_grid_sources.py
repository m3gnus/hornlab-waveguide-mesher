from __future__ import annotations

import math

import numpy as np

from ..geometry import PointGridHornGeometry
from .point_grid_surfaces import (
    _GeoSurfaceBuilder,
    _SharedSurfaceBuilder,
    _phi_segments,
    _source_cap_phi_groups,
    _spline_span_phi_groups,
)

SOURCE_SHAPE_FLAT_DISC = 0
SOURCE_SHAPE_ROUNDED_CAP = 1
SUPPORTED_SOURCE_SHAPES = {SOURCE_SHAPE_FLAT_DISC, SOURCE_SHAPE_ROUNDED_CAP}


def _source_shape(geometry: PointGridHornGeometry) -> int:
    return int(geometry.source_shape)


def _validate_source_shape(geometry: PointGridHornGeometry) -> int:
    shape = _source_shape(geometry)
    if shape not in SUPPORTED_SOURCE_SHAPES:
        raise NotImplementedError(
            f"PointGridHornGeometry.source_shape={shape} is not supported "
            "by hornlab-mesher (supported: 0=flat disc, 1=rounded cap)."
        )
    return shape


def _geo_radial_source_curves(
    builder: _GeoSurfaceBuilder,
    inner_points: np.ndarray,
    geometry: PointGridHornGeometry,
    *,
    pole_tag: int,
    center: np.ndarray,
    throat_radius: float,
    cap_height: float,
    mesh_size: float,
) -> dict[int, int]:
    n_phi = inner_points.shape[0]
    if cap_height <= 1.0e-12:
        return {
            i: builder.line_tags(builder.point("inner", i, 0), pole_tag)
            for i in range(n_phi)
        }

    sign = -1.0 if int(geometry.source_curv) == -1 else 1.0
    radius = max(_source_cap_radius(throat_radius, geometry), throat_radius * 1.001)
    sqrt_base = math.sqrt(max(0.0, radius * radius - throat_radius * throat_radius))
    ring = inner_points[:, 0, :]
    radial = ring - center
    radial[:, 2] = 0.0
    radii = np.linalg.norm(radial[:, :2], axis=1)
    curves: dict[int, int] = {}
    for i in range(n_phi):
        control_tags = [builder.point("inner", i, 0)]
        unit = radial[i] / max(radii[i], 1.0e-12)
        for frac in (2.0 / 3.0, 1.0 / 3.0):
            rj = throat_radius * frac
            hj = math.sqrt(max(0.0, radius * radius - rj * rj)) - sqrt_base
            p = np.array(center, dtype=np.float64)
            p[:2] += unit[:2] * rj
            p[2] += sign * hj
            control_tags.append(builder.add_point(p, mesh_size=mesh_size))
        control_tags.append(pole_tag)
        curves[i] = int(builder.geo.addSpline(control_tags))
    return curves


def _add_geo_source_cap_surfaces(
    builder: _GeoSurfaceBuilder,
    inner_points: np.ndarray,
    geometry: PointGridHornGeometry,
    *,
    mesh_size: float,
) -> list[tuple[int, int]]:
    shape = _validate_source_shape(geometry)
    n_phi = inner_points.shape[0]
    ring = inner_points[:, 0, :]
    center = np.mean(ring, axis=0)
    center[2] = float(np.mean(ring[:, 2]))
    throat_radius = _throat_radius(inner_points, closed=geometry.closed)
    cap_height = _source_cap_height(throat_radius, geometry) if shape == SOURCE_SHAPE_ROUNDED_CAP else 0.0

    if geometry.closed:
        pole = np.array(center, dtype=np.float64)
        sign = -1.0 if int(geometry.source_curv) == -1 else 1.0
        pole[2] += sign * cap_height
        pole_tag = builder.add_point(pole, mesh_size=mesh_size)
        radial_curves = _geo_radial_source_curves(
            builder,
            inner_points,
            geometry,
            pole_tag=pole_tag,
            center=center,
            throat_radius=throat_radius,
            cap_height=cap_height,
            mesh_size=mesh_size,
        )
        cap: list[tuple[int, int]] = []
        for indices in _source_cap_phi_groups(n_phi, closed=True):
            start = indices[0]
            end = indices[-1]
            phi_curve = builder.spline([("inner", i, 0) for i in indices])
            cap.append(builder.surface([phi_curve, radial_curves[end], -radial_curves[start]]))
        return cap

    center[0] = 0.0
    center[1] = 0.0

    # A rounded cap with zero height (flat auto source) shares the flat-disc
    # construction; the arc-based path below cannot represent it.
    if shape == SOURCE_SHAPE_FLAT_DISC or cap_height <= 1.0e-12:
        center_tag = builder.add_point(center, mesh_size=mesh_size)
        radial_lines = {
            i: builder.line_tags(center_tag, builder.point("inner", i, 0))
            for i in range(n_phi)
        }
        boundary = [
            radial_lines[0],
            *[
                builder.spline([("inner", i, 0) for i in indices])
                for indices in _spline_span_phi_groups(n_phi, closed=False)
            ],
            -radial_lines[n_phi - 1],
        ]
        return [builder.surface(boundary)]

    if throat_radius <= 1.0e-9:
        return []
    radius = max(_source_cap_radius(throat_radius, geometry), throat_radius * 1.001)
    cap_height = _source_cap_height(throat_radius, geometry)
    sign = -1.0 if int(geometry.source_curv) == -1 else 1.0

    pole = np.array(center, dtype=np.float64)
    pole[2] += sign * cap_height
    sphere_center = np.array(center, dtype=np.float64)
    sphere_center[2] += sign * (cap_height - radius)

    pole_tag = builder.add_point(pole, mesh_size=mesh_size)
    sphere_center_tag = builder.add_point(sphere_center, mesh_size=mesh_size)
    arc_start = builder.circle_arc(
        pole_tag,
        sphere_center_tag,
        builder.point("inner", 0, 0),
    )
    arc_end = builder.circle_arc(
        pole_tag,
        sphere_center_tag,
        builder.point("inner", n_phi - 1, 0),
    )
    boundary = [
        arc_start,
        *[
            builder.spline([("inner", i, 0) for i in indices])
            for indices in _spline_span_phi_groups(n_phi, closed=False)
        ],
        -arc_end,
    ]
    return [builder.surface(boundary, sphere_center_tag=sphere_center_tag)]


def _add_occ_source_cap_surfaces(
    builder: _SharedSurfaceBuilder,
    inner_points: np.ndarray,
    geometry: PointGridHornGeometry,
) -> list[tuple[int, int]]:
    shape = _validate_source_shape(geometry)
    n_phi = inner_points.shape[0]
    ring = inner_points[:, 0, :]
    center = np.mean(ring, axis=0)
    center[2] = float(np.mean(ring[:, 2]))
    if not geometry.closed:
        center[0] = 0.0
        center[1] = 0.0
    throat_radius = _throat_radius(inner_points, closed=geometry.closed)
    cap_height = _source_cap_height(throat_radius, geometry) if shape == SOURCE_SHAPE_ROUNDED_CAP else 0.0
    sign = -1.0 if int(geometry.source_curv) == -1 else 1.0
    pole = np.array(center, dtype=np.float64)
    pole[2] += sign * cap_height
    pole_tag = builder.add_point(pole)

    radial_lines: dict[int, int] = {}
    for i in range(n_phi):
        if cap_height <= 1.0e-12:
            radial_lines[i] = builder.line_tags(builder.point("inner", i, 0), pole_tag)
            continue
        control_tags = [builder.point("inner", i, 0)]
        radial = ring[i] - center
        radial[2] = 0.0
        radial_len = float(np.linalg.norm(radial[:2]))
        unit = radial / max(radial_len, 1.0e-12)
        radius = max(_source_cap_radius(throat_radius, geometry), throat_radius * 1.001)
        sqrt_base = math.sqrt(max(0.0, radius * radius - throat_radius * throat_radius))
        for frac in (2.0 / 3.0, 1.0 / 3.0):
            rj = throat_radius * frac
            hj = math.sqrt(max(0.0, radius * radius - rj * rj)) - sqrt_base
            p = np.array(center, dtype=np.float64)
            p[:2] += unit[:2] * rj
            p[2] += sign * hj
            control_tags.append(builder.add_point(p))
        control_tags.append(pole_tag)
        radial_lines[i] = builder.bspline_tags(control_tags)

    cap: list[tuple[int, int]] = []
    spans = _source_cap_phi_groups(n_phi, closed=geometry.closed)
    if geometry.closed:
        for indices in spans:
            start = indices[0]
            end = indices[-1]
            phi_curve = builder.bspline_tags([builder.point("inner", i, 0) for i in indices])
            cap.append(builder.surface([phi_curve, radial_lines[end], -radial_lines[start]]))
        return cap

    boundary: list[int] = []
    for indices in spans:
        boundary.append(builder.bspline_tags([builder.point("inner", i, 0) for i in indices]))
    if boundary:
        cap.append(builder.surface([*boundary, radial_lines[n_phi - 1], -radial_lines[0]]))
    return cap


def _source_cap_radius(
    throat_radius: float,
    geometry: PointGridHornGeometry,
) -> float:
    if geometry.source_radius_mm and geometry.source_radius_mm > 0.0:
        return float(geometry.source_radius_mm)
    if geometry.source_auto_angle_deg is not None:
        angle = math.radians(float(geometry.source_auto_angle_deg))
        if abs(math.sin(angle)) > 1.0e-9:
            return abs(float(throat_radius) / math.sin(angle))
        # ATH matches the auto cap to the throat opening angle; a zero
        # angle means an infinite wavefront radius, i.e. a flat source.
        return math.inf
    return 3.75 * float(throat_radius)


def _source_cap_height(throat_radius: float, geometry: PointGridHornGeometry) -> float:
    if _source_shape(geometry) != SOURCE_SHAPE_ROUNDED_CAP or throat_radius <= 1.0e-9:
        return 0.0
    radius = _source_cap_radius(throat_radius, geometry)
    if not math.isfinite(radius):
        return 0.0
    radius = max(radius, throat_radius * 1.001)
    return radius - math.sqrt(max(0.0, radius * radius - throat_radius * throat_radius))


def _throat_radius(inner_points: np.ndarray, *, closed: bool) -> float:
    ring = inner_points[:, 0, :]
    center = np.mean(ring, axis=0)
    if not closed:
        center[0] = 0.0
        center[1] = 0.0
    radial = ring[:, :2] - center[:2]
    radii = np.linalg.norm(radial, axis=1)
    return float(np.mean(radii[radii > 1.0e-9])) if np.any(radii > 1.0e-9) else 0.0


def _add_source_surfaces(
    builder: _SharedSurfaceBuilder,
    inner_points: np.ndarray,
    geometry: PointGridHornGeometry,
) -> list[tuple[int, int]]:
    shape = _validate_source_shape(geometry)
    n_phi = inner_points.shape[0]
    closed = bool(geometry.closed)
    ring = inner_points[:, 0, :]
    center = np.mean(ring, axis=0)
    if not closed:
        center[0] = 0.0
        center[1] = 0.0
    center[2] = float(np.mean(ring[:, 2]))

    radial = ring - center
    radial[:, 2] = 0.0
    radii = np.linalg.norm(radial[:, :2], axis=1)
    throat_radius = float(np.mean(radii[radii > 1.0e-9])) if np.any(radii > 1.0e-9) else 0.0
    if throat_radius <= 1.0e-9:
        return []

    cap_height = 0.0
    if shape == SOURCE_SHAPE_ROUNDED_CAP:
        cap_height = _source_cap_height(throat_radius, geometry)
    sign = -1.0 if int(geometry.source_curv) == -1 else 1.0

    pole = np.array(center, dtype=np.float64)
    pole[2] += sign * cap_height
    pole_tag = builder.add_point(pole)

    radial_lines: dict[int, int] = {}
    for i in range(n_phi):
        if cap_height <= 1.0e-12:
            radial_lines[i] = builder.line_tags(builder.point("inner", i, 0), pole_tag)
            continue
        control_tags = [builder.point("inner", i, 0)]
        unit = radial[i] / max(radii[i], 1.0e-12)
        radius = max(_source_cap_radius(throat_radius, geometry), throat_radius * 1.001)
        sqrt_base = math.sqrt(max(0.0, radius * radius - throat_radius * throat_radius))
        for frac in (2.0 / 3.0, 1.0 / 3.0):
            rj = throat_radius * frac
            hj = math.sqrt(max(0.0, radius * radius - rj * rj)) - sqrt_base
            p = np.array(center, dtype=np.float64)
            p[:2] += unit[:2] * rj
            p[2] += sign * hj
            control_tags.append(builder.add_point(p))
        control_tags.append(pole_tag)
        radial_lines[i] = builder.bspline_tags(control_tags)

    surfaces: list[tuple[int, int]] = []
    cap_boundary: list[int] = []
    for i in _phi_segments(n_phi, closed=closed):
        ni = (i + 1) % n_phi
        cap_boundary.append(builder.line(("inner", i, 0), ("inner", ni, 0)))
        if closed:
            surfaces.append(
                builder.surface(
                    [
                        cap_boundary[-1],
                        radial_lines[ni],
                        -radial_lines[i],
                    ]
                )
            )
    if not closed and cap_boundary:
        surfaces.append(builder.surface([*cap_boundary, radial_lines[n_phi - 1], -radial_lines[0]]))
    return surfaces
