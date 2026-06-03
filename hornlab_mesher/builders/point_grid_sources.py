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

def _add_geo_source_cap_surfaces(
    builder: _GeoSurfaceBuilder,
    inner_points: np.ndarray,
    geometry: PointGridHornGeometry,
    *,
    mesh_size: float,
) -> list[tuple[int, int]]:
    if int(geometry.source_shape) != 1:
        return []

    n_phi = inner_points.shape[0]
    ring = inner_points[:, 0, :]
    center = np.mean(ring, axis=0)
    center[2] = float(np.mean(ring[:, 2]))
    if geometry.closed:
        center_tag = builder.add_point(center, mesh_size=mesh_size)
        radial_lines = {
            i: builder.line_tags(center_tag, builder.point("inner", i, 0))
            for i in range(n_phi)
        }
        cap: list[tuple[int, int]] = []
        for indices in _source_cap_phi_groups(n_phi, closed=True):
            start = indices[0]
            end = indices[-1]
            phi_curve = builder.spline([("inner", i, 0) for i in indices])
            cap.append(builder.surface([radial_lines[start], phi_curve, -radial_lines[end]]))
        return cap

    center[0] = 0.0
    center[1] = 0.0

    throat_radius = _throat_radius(inner_points, closed=False)
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
    if int(geometry.source_shape) != 1:
        return []

    n_phi = inner_points.shape[0]
    ring = inner_points[:, 0, :]
    center = np.mean(ring, axis=0)
    center[2] = float(np.mean(ring[:, 2]))
    if not geometry.closed:
        center[0] = 0.0
        center[1] = 0.0
    center_tag = builder.add_point(center)
    radial_lines = {
        i: builder.line_tags(center_tag, builder.point("inner", i, 0))
        for i in range(n_phi)
    }

    cap: list[tuple[int, int]] = []
    spans = _source_cap_phi_groups(n_phi, closed=geometry.closed)
    if geometry.closed:
        for indices in spans:
            start = indices[0]
            end = indices[-1]
            phi_curve = builder.bspline_tags([builder.point("inner", i, 0) for i in indices])
            cap.append(builder.surface([radial_lines[start], phi_curve, -radial_lines[end]]))
        return cap

    boundary: list[int] = []
    for indices in spans:
        boundary.append(builder.bspline_tags([builder.point("inner", i, 0) for i in indices]))
    if boundary:
        cap.append(builder.surface([radial_lines[0], *boundary, -radial_lines[n_phi - 1]]))
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
    return 3.75 * float(throat_radius)


def _source_cap_height(throat_radius: float, geometry: PointGridHornGeometry) -> float:
    if int(geometry.source_shape) != 1 or throat_radius <= 1.0e-9:
        return 0.0
    radius = max(_source_cap_radius(throat_radius, geometry), throat_radius * 1.001)
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
    if int(geometry.source_shape) == 1:
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
