from __future__ import annotations

import math

import numpy as np

from ..geometry import BuiltGeometry, HornInterface, PointGridHornGeometry
from ..tags import PhysicalGroup
from ._occ import (
    build_surface_from_points,
    make_planar_fill_from_boundary,
    make_planar_fill_from_ring,
    make_planar_sector_fill_from_ring,
    require_gmsh,
)
from .enclosure import (
    build_enclosure_box,
)


def _safe_surface_from_curves(curves: list[int]) -> tuple[int, int]:
    gmsh = require_gmsh()
    loop = int(gmsh.model.occ.addCurveLoop([int(c) for c in curves]))
    try:
        surf = int(gmsh.model.occ.addPlaneSurface([loop]))
    except Exception:
        surf = int(gmsh.model.occ.addSurfaceFilling(loop))
    return (2, surf)


class _SharedSurfaceBuilder:
    """Small OCC helper that keeps adjacent faceted surfaces on shared curves."""

    def __init__(self) -> None:
        self.gmsh = require_gmsh()
        self.points: dict[tuple[str, int, int], int] = {}
        self.line_cache: dict[tuple[int, int], int] = {}
        self.spline_cache: dict[tuple[int, ...], int] = {}

    def add_point(self, xyz: np.ndarray | tuple[float, float, float]) -> int:
        x, y, z = xyz
        return int(self.gmsh.model.occ.addPoint(float(x), float(y), float(z)))

    def add_grid(self, name: str, points: np.ndarray) -> None:
        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                self.points[(name, i, j)] = self.add_point(points[i, j])

    def point(self, name: str, i: int, j: int) -> int:
        return self.points[(name, int(i), int(j))]

    def line_tags(self, a: int, b: int) -> int:
        if a == b:
            raise ValueError("cannot build a line with identical endpoints")
        key = (int(a), int(b))
        if key in self.line_cache:
            return self.line_cache[key]
        rev = (int(b), int(a))
        if rev in self.line_cache:
            return -self.line_cache[rev]
        tag = int(self.gmsh.model.occ.addLine(int(a), int(b)))
        self.line_cache[key] = tag
        return tag

    def line(self, a: tuple[str, int, int], b: tuple[str, int, int]) -> int:
        return self.line_tags(self.point(*a), self.point(*b))

    def bspline_tags(self, point_tags: list[int]) -> int:
        key = tuple(int(p) for p in point_tags)
        if key in self.spline_cache:
            return self.spline_cache[key]
        rev = tuple(reversed(key))
        if rev in self.spline_cache:
            return -self.spline_cache[rev]
        tag = int(self.gmsh.model.occ.addBSpline(list(key)))
        self.spline_cache[key] = tag
        return tag

    def surface(self, curves: list[int]) -> tuple[int, int]:
        return _safe_surface_from_curves(curves)

    def quad(
        self,
        a: tuple[str, int, int],
        b: tuple[str, int, int],
        c: tuple[str, int, int],
        d: tuple[str, int, int],
    ) -> tuple[int, int]:
        return self.surface([
            self.line(a, b),
            self.line(b, c),
            self.line(c, d),
            self.line(d, a),
        ])


class _GeoSurfaceBuilder:
    """Gmsh built-in-kernel helper for spline-grouped surfaces."""

    def __init__(self) -> None:
        self.gmsh = require_gmsh()
        self.geo = self.gmsh.model.geo
        self.points: dict[tuple[str, int, int], int] = {}
        self.line_cache: dict[tuple[int, int], int] = {}
        self.spline_cache: dict[tuple[int, ...], int] = {}

    def add_point(self, xyz: np.ndarray | tuple[float, float, float], mesh_size: float = 0.0) -> int:
        x, y, z = xyz
        return int(self.geo.addPoint(float(x), float(y), float(z), float(mesh_size)))

    def add_grid(
        self,
        name: str,
        points: np.ndarray,
        mesh_size: float | np.ndarray = 0.0,
    ) -> None:
        sizes = np.asarray(mesh_size, dtype=np.float64)
        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                size = float(sizes[i, j] if sizes.shape == points.shape[:2] else sizes)
                self.points[(name, i, j)] = self.add_point(points[i, j], mesh_size=size)

    def point(self, name: str, i: int, j: int) -> int:
        return self.points[(name, int(i), int(j))]

    def line_tags(self, a: int, b: int) -> int:
        if a == b:
            raise ValueError("cannot build a line with identical endpoints")
        key = (int(a), int(b))
        if key in self.line_cache:
            return self.line_cache[key]
        rev = (int(b), int(a))
        if rev in self.line_cache:
            return -self.line_cache[rev]
        tag = int(self.geo.addLine(int(a), int(b)))
        self.line_cache[key] = tag
        return tag

    def line(self, a: tuple[str, int, int], b: tuple[str, int, int]) -> int:
        return self.line_tags(self.point(*a), self.point(*b))

    def spline(self, points: list[tuple[str, int, int]]) -> int:
        key = tuple(self.point(*p) for p in points)
        if key in self.spline_cache:
            return self.spline_cache[key]
        rev = tuple(reversed(key))
        if rev in self.spline_cache:
            return -self.spline_cache[rev]
        tag = int(self.geo.addSpline(list(key)))
        self.spline_cache[key] = tag
        return tag

    def circle_arc(self, start: int, center: int, end: int) -> int:
        return int(self.geo.addCircleArc(int(start), int(center), int(end)))

    def surface(self, curves: list[int], *, sphere_center_tag: int = -1) -> tuple[int, int]:
        loop = int(self.geo.addCurveLoop([int(c) for c in curves]))
        surf = int(self.geo.addSurfaceFilling([loop], sphereCenterTag=int(sphere_center_tag)))
        return (2, surf)


def _phi_segments(n_phi: int, *, closed: bool) -> range:
    return range(n_phi if closed else n_phi - 1)


def _restored_outer_throat_points(
    inner_points: np.ndarray,
    outer_points: np.ndarray,
    *,
    wall_thickness_mm: float,
) -> np.ndarray:
    """Undo the legacy point-grid throat clamp before adding the rear return.

    WG's legacy payload path flattened the outer throat ring to
    ``inner_z - wallThickness`` as a shortcut for the rear plate. ATH builds
    the outer shell at the throat and then adds a normal/backward rear return.
    """

    out = np.array(outer_points, dtype=np.float64, copy=True)
    expected = inner_points[:, 0, 2] - float(wall_thickness_mm)
    if np.allclose(out[:, 0, 2], expected, rtol=0.0, atol=1.0e-6):
        out[:, 0, 2] = inner_points[:, 0, 2]
    return out


def _add_grid_wall_surfaces(
    builder: _SharedSurfaceBuilder,
    name: str,
    *,
    n_phi: int,
    n_len: int,
    closed: bool,
    reverse: bool = False,
) -> list[tuple[int, int]]:
    surfaces: list[tuple[int, int]] = []
    for i in _phi_segments(n_phi, closed=closed):
        ni = (i + 1) % n_phi
        for j in range(n_len - 1):
            if reverse:
                surfaces.append(
                    builder.quad(
                        (name, i, j),
                        (name, i, j + 1),
                        (name, ni, j + 1),
                        (name, ni, j),
                    )
                )
            else:
                surfaces.append(
                    builder.quad(
                        (name, i, j),
                        (name, ni, j),
                        (name, ni, j + 1),
                        (name, i, j + 1),
                    )
                )
    return surfaces


def _spline_span_phi_groups(n_phi: int, *, closed: bool) -> list[list[int]]:
    """Group angular samples into stable spline spans.

    Closed rings are split on symmetry/cardinal boundaries. If the angular
    grid can also represent exact octants, wall surfaces use those shorter
    spans to reduce spline span size while preserving symmetry.
    """
    if closed:
        if n_phi < 8:
            return [[*range(n_phi), 0]]
        span_count = 8 if n_phi % 8 == 0 else 4
        step = max(1, n_phi // span_count)
        spans: list[list[int]] = []
        for span in range(span_count):
            start = span * step
            stop = (span + 1) * step
            if span == span_count - 1:
                indices = list(range(start, n_phi)) + [0]
            else:
                indices = list(range(start, stop + 1))
            spans.append(indices)
        return spans
    if n_phi < 3:
        return [list(range(n_phi))]
    mid = n_phi // 2
    return [list(range(0, mid + 1)), list(range(mid, n_phi))]


def _source_cap_phi_groups(n_phi: int, *, closed: bool) -> list[list[int]]:
    """Group source-cap boundary splines by symmetry sectors."""
    if not closed:
        return _spline_span_phi_groups(n_phi, closed=False)
    if n_phi < 4:
        return [[*range(n_phi), 0]]
    span_count = 4
    step = max(1, n_phi // span_count)
    spans: list[list[int]] = []
    for span in range(span_count):
        start = span * step
        stop = (span + 1) * step
        if span == span_count - 1:
            indices = list(range(start, n_phi)) + [0]
        else:
            indices = list(range(start, stop + 1))
        spans.append(indices)
    return spans


def _snap_open_symmetry_grid(points: np.ndarray, *, closed: bool) -> np.ndarray:
    out = np.array(points, dtype=np.float64, copy=True)
    if not closed and out.shape[0] >= 2:
        out[0, :, 1] = 0.0
        out[-1, :, 0] = 0.0
    return out


def _outer_wall_axial_ring_indices(inner_points: np.ndarray) -> list[int]:
    """Select axial rings used by the outer return wall.

    The throat ring is always present. Intermediate rings are retained until
    the horn reaches the mouth-side maximum-z plane, avoiding a degenerate
    outer-wall strip at the mouth rim.
    """
    z_by_ring = np.mean(inner_points[:, :, 2], axis=0)
    max_z = float(np.max(z_by_ring))
    tol = max(1.0e-3, 1.0e-8 * max(1.0, abs(max_z)))
    return [
        j
        for j in range(1, inner_points.shape[1])
        if float(z_by_ring[j]) < max_z - tol
    ]


def _add_mouth_rim_surfaces(
    builder: _SharedSurfaceBuilder,
    *,
    n_phi: int,
    n_len: int,
    closed: bool,
) -> list[tuple[int, int]]:
    j = n_len - 1
    surfaces: list[tuple[int, int]] = []
    for i in _phi_segments(n_phi, closed=closed):
        ni = (i + 1) % n_phi
        surfaces.append(
            builder.quad(
                ("inner", i, j),
                ("inner", ni, j),
                ("outer", ni, j),
                ("outer", i, j),
            )
        )
    return surfaces


def _rear_rim_points(
    outer_points: np.ndarray,
    *,
    rear_z: float,
) -> np.ndarray:
    n_phi = outer_points.shape[0]
    out = np.empty((n_phi, 3), dtype=np.float64)
    for i in range(n_phi):
        p0 = outer_points[i, 0]
        if outer_points.shape[1] < 2:
            out[i] = (p0[0], p0[1], rear_z)
            continue
        p1 = outer_points[i, 1]
        dz = float(p1[2] - p0[2])
        if abs(dz) <= 1.0e-9:
            out[i] = (p0[0], p0[1], rear_z)
            continue
        t = (rear_z - float(p0[2])) / dz
        out[i] = p0 + (p1 - p0) * t
        out[i, 2] = rear_z
    return out


def _add_rear_return_and_cap(
    builder: _SharedSurfaceBuilder,
    rear_points: np.ndarray,
    *,
    n_phi: int,
    closed: bool,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    builder.add_grid("rear", rear_points[:, np.newaxis, :])
    transition: list[tuple[int, int]] = []
    cap: list[tuple[int, int]] = []
    center_xy = (
        (float(np.mean(rear_points[:, 0])), float(np.mean(rear_points[:, 1])))
        if closed
        else (0.0, 0.0)
    )
    center_tag = builder.add_point(
        (
            center_xy[0],
            center_xy[1],
            float(np.mean(rear_points[:, 2])),
        )
    )
    radial_lines: dict[int, int] = {}
    for i in range(n_phi):
        radial_lines[i] = builder.line_tags(builder.point("rear", i, 0), center_tag)

    cap_boundary: list[int] = []
    for i in _phi_segments(n_phi, closed=closed):
        ni = (i + 1) % n_phi
        transition.append(
            builder.quad(
                ("outer", i, 0),
                ("outer", ni, 0),
                ("rear", ni, 0),
                ("rear", i, 0),
            )
        )
        cap_boundary.append(builder.line(("rear", i, 0), ("rear", ni, 0)))
        if closed:
            cap.append(
                builder.surface(
                    [
                        cap_boundary[-1],
                        radial_lines[ni],
                        -radial_lines[i],
                    ]
                )
            )
    if not closed and cap_boundary:
        cap.append(builder.surface([*cap_boundary, radial_lines[n_phi - 1], -radial_lines[0]]))
    return transition, cap


def _add_geo_spline_span_wall_surfaces(
    builder: _GeoSurfaceBuilder,
    name: str,
    *,
    n_phi: int,
    n_len: int,
    closed: bool,
    reverse: bool = False,
) -> list[tuple[int, int]]:
    surfaces: list[tuple[int, int]] = []
    for indices in _spline_span_phi_groups(n_phi, closed=closed):
        start = indices[0]
        end = indices[-1]
        prev_phi = builder.spline([(name, i, 0) for i in indices])
        for j in range(n_len - 1):
            next_phi = builder.spline([(name, i, j + 1) for i in indices])
            left = builder.line((name, start, j), (name, start, j + 1))
            right = builder.line((name, end, j), (name, end, j + 1))
            curves = (
                [prev_phi, right, -next_phi, -left]
                if reverse
                else [next_phi, -right, -prev_phi, left]
            )
            surfaces.append(builder.surface(curves))
            prev_phi = next_phi
    return surfaces


def _add_occ_spline_span_wall_surfaces(
    builder: _SharedSurfaceBuilder,
    name: str,
    *,
    n_phi: int,
    n_len: int,
    closed: bool,
    reverse: bool = False,
) -> list[tuple[int, int]]:
    surfaces: list[tuple[int, int]] = []
    for indices in _spline_span_phi_groups(n_phi, closed=closed):
        start = indices[0]
        end = indices[-1]
        prev_phi = builder.bspline_tags([builder.point(name, i, 0) for i in indices])
        for j in range(n_len - 1):
            next_phi = builder.bspline_tags([builder.point(name, i, j + 1) for i in indices])
            left = builder.line((name, start, j), (name, start, j + 1))
            right = builder.line((name, end, j), (name, end, j + 1))
            curves = (
                [prev_phi, right, -next_phi, -left]
                if reverse
                else [next_phi, -right, -prev_phi, left]
            )
            surfaces.append(builder.surface(curves))
            prev_phi = next_phi
    return surfaces


def _add_geo_spline_span_mouth_rim_surfaces(
    builder: _GeoSurfaceBuilder,
    *,
    n_phi: int,
    inner_len: int,
    outer_len: int,
    closed: bool,
) -> list[tuple[int, int]]:
    surfaces: list[tuple[int, int]] = []
    ji = inner_len - 1
    jo = outer_len - 1
    for indices in _spline_span_phi_groups(n_phi, closed=closed):
        start = indices[0]
        end = indices[-1]
        inner_phi = builder.spline([("inner", i, ji) for i in indices])
        outer_phi = builder.spline([("outer", i, jo) for i in indices])
        left = builder.line(("inner", start, ji), ("outer", start, jo))
        right = builder.line(("inner", end, ji), ("outer", end, jo))
        surfaces.append(builder.surface([outer_phi, -right, -inner_phi, left]))
    return surfaces


def _add_geo_spline_span_rear_cap(
    builder: _GeoSurfaceBuilder,
    rear_points: np.ndarray,
    *,
    n_phi: int,
    closed: bool,
    mesh_size: float,
) -> list[tuple[int, int]]:
    center_xy = (
        (float(np.mean(rear_points[:, 0])), float(np.mean(rear_points[:, 1])))
        if closed
        else (0.0, 0.0)
    )
    center_tag = builder.add_point(
        (
            center_xy[0],
            center_xy[1],
            float(np.mean(rear_points[:, 2])),
        ),
        mesh_size=mesh_size,
    )
    radial_lines = {
        i: builder.line_tags(center_tag, builder.point("outer", i, 0))
        for i in range(n_phi)
    }
    cap: list[tuple[int, int]] = []
    for indices in _spline_span_phi_groups(n_phi, closed=closed):
        start = indices[0]
        end = indices[-1]
        phi_curve = builder.spline([("outer", i, 0) for i in indices])
        cap.append(builder.surface([radial_lines[start], phi_curve, -radial_lines[end]]))
    return cap


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


def _build_freestanding_point_grid(geometry: PointGridHornGeometry) -> BuiltGeometry:
    inner_points = _validated_grid(geometry.inner_points, name="inner_points")
    if geometry.outer_points is None:
        raise ValueError("freestanding point-grid build requires outer_points")
    outer_points = _validated_grid(geometry.outer_points, name="outer_points")
    if geometry.wg_topology:
        return _build_wg_freestanding_point_grid(
            geometry,
            inner_points,
            outer_points,
        )
    outer_points = _restored_outer_throat_points(
        inner_points,
        outer_points,
        wall_thickness_mm=float(geometry.wall_thickness_mm),
    )

    n_phi, n_len, _ = inner_points.shape
    builder = _SharedSurfaceBuilder()
    builder.add_grid("inner", inner_points)
    builder.add_grid("outer", outer_points)

    wall = _add_grid_wall_surfaces(
        builder,
        "inner",
        n_phi=n_phi,
        n_len=n_len,
        closed=geometry.closed,
    )
    outer_wall = _add_grid_wall_surfaces(
        builder,
        "outer",
        n_phi=n_phi,
        n_len=n_len,
        closed=geometry.closed,
        reverse=True,
    )
    mouth_dimtags = _add_mouth_rim_surfaces(
        builder,
        n_phi=n_phi,
        n_len=n_len,
        closed=geometry.closed,
    )
    rear_extra = 0.5 * _source_cap_height(
        _throat_radius(inner_points, closed=geometry.closed),
        geometry,
    )
    rear_z = float(
        np.mean(inner_points[:, 0, 2])
        - float(geometry.wall_thickness_mm)
        - rear_extra
    )
    rear_points = _rear_rim_points(outer_points, rear_z=rear_z)
    rear_transition, rear_cap = _add_rear_return_and_cap(
        builder,
        rear_points,
        n_phi=n_phi,
        closed=geometry.closed,
    )
    throat = _add_source_surfaces(builder, inner_points, geometry)

    wall_tags = [tag for _, tag in wall]
    outer_tags = [tag for _, tag in outer_wall]
    mouth_tags = [tag for _, tag in mouth_dimtags]
    rear_transition_tags = [tag for _, tag in rear_transition]
    rear_tags = [tag for _, tag in rear_cap]
    throat_tags = [tag for _, tag in throat]
    rigid_wall_tags = [
        *wall_tags,
        *outer_tags,
        *mouth_tags,
        *rear_transition_tags,
        *rear_tags,
    ]

    z0 = float(np.mean(inner_points[:, 0, 2]))
    z1 = float(np.mean(inner_points[:, -1, 2]))
    return BuiltGeometry(
        surface_groups={
            int(PhysicalGroup.RIGID_WALL): rigid_wall_tags,
            int(PhysicalGroup.PRIMARY_SOURCE): throat_tags,
        },
        axial_bounds_mm=(z0, z1),
        source_axis="z",
        mesh_surface_groups={
            "inner": wall_tags,
            "throat_disc": throat_tags,
            "outer": outer_tags,
            "mouth": mouth_tags,
            "rear_transition": rear_transition_tags,
            "rear": rear_tags,
            "rear_cap": rear_tags,
        },
        symmetry_snap_axes=() if geometry.closed else ("x", "y"),
        symmetry_snap_tol_mm=1.0,
    )


def _build_wg_freestanding_point_grid(
    geometry: PointGridHornGeometry,
    inner_points: np.ndarray,
    outer_points: np.ndarray,
) -> BuiltGeometry:
    inner_points = _snap_open_symmetry_grid(inner_points, closed=geometry.closed)
    outer_points = _snap_open_symmetry_grid(outer_points, closed=geometry.closed)

    n_phi, inner_len, _ = inner_points.shape
    outer_indices = _outer_wall_axial_ring_indices(inner_points)
    outer_topology = np.empty((n_phi, len(outer_indices) + 1, 3), dtype=np.float64)
    outer_topology[:, 0, :] = outer_points[:, 0, :]
    for out_j, src_j in enumerate(outer_indices, start=1):
        outer_topology[:, out_j, :] = outer_points[:, src_j, :]
    rear_points = outer_topology[:, 0, :]

    builder = _GeoSurfaceBuilder()
    inner_mesh_sizes = np.full(inner_points.shape[:2], 8.0, dtype=np.float64)
    inner_mesh_sizes[:, 0] = 5.0
    builder.add_grid("inner", inner_points, mesh_size=inner_mesh_sizes)
    builder.add_grid("outer", outer_topology, mesh_size=25.0)

    wall = _add_geo_spline_span_wall_surfaces(
        builder,
        "inner",
        n_phi=n_phi,
        n_len=inner_len,
        closed=geometry.closed,
    )
    outer_wall = _add_geo_spline_span_wall_surfaces(
        builder,
        "outer",
        n_phi=n_phi,
        n_len=outer_topology.shape[1],
        closed=geometry.closed,
        reverse=True,
    )
    mouth_dimtags = _add_geo_spline_span_mouth_rim_surfaces(
        builder,
        n_phi=n_phi,
        inner_len=inner_len,
        outer_len=outer_topology.shape[1],
        closed=geometry.closed,
    )
    rear_cap = _add_geo_spline_span_rear_cap(
        builder,
        rear_points,
        n_phi=n_phi,
        closed=geometry.closed,
        mesh_size=25.0,
    )
    throat = _add_geo_source_cap_surfaces(
        builder,
        inner_points,
        geometry,
        mesh_size=5.0,
    )
    require_gmsh().model.geo.synchronize()

    wall_tags = [tag for _, tag in wall]
    outer_tags = [tag for _, tag in outer_wall]
    mouth_tags = [tag for _, tag in mouth_dimtags]
    rear_tags = [tag for _, tag in rear_cap]
    throat_tags = [tag for _, tag in throat]
    rigid_wall_tags = [*wall_tags, *outer_tags, *mouth_tags, *rear_tags]

    z0 = float(np.mean(inner_points[:, 0, 2]))
    z1 = float(np.mean(inner_points[:, -1, 2]))
    return BuiltGeometry(
        surface_groups={
            int(PhysicalGroup.RIGID_WALL): rigid_wall_tags,
            int(PhysicalGroup.PRIMARY_SOURCE): throat_tags,
        },
        axial_bounds_mm=(z0, z1),
        source_axis="z",
        mesh_surface_groups={
            "inner": wall_tags,
            "throat_disc": throat_tags,
            "outer": outer_tags,
            "mouth": mouth_tags,
            "rear": rear_tags,
            "rear_cap": rear_tags,
        },
        symmetry_snap_axes=() if geometry.closed else ("x", "y"),
        symmetry_snap_tol_mm=1.0,
        mesh_algorithm=2,
    )


def _validated_grid(points: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{name} must be shaped (n_phi, n_length + 1, 3)")
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(f"{name} needs at least 2 phi samples and 2 axial rings")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _interface_phi_groups(n_phi: int, *, closed: bool) -> list[list[int]]:
    if not closed:
        return [list(range(n_phi))]
    if n_phi < 4:
        return [[*range(n_phi), 0]]
    step = max(1, n_phi // 4)
    groups: list[list[int]] = []
    for group in range(4):
        start = group * step
        stop = (group + 1) * step
        if group == 3:
            groups.append(list(range(start, n_phi)) + [0])
        else:
            groups.append(list(range(start, stop + 1)))
    return groups


def _split_interface_group(indices: list[int]) -> list[list[int]]:
    if len(indices) < 3:
        return [indices]
    mid = len(indices) // 2
    return [indices[: mid + 1], indices[mid:]]


def _normalise_interface_specs(geometry: PointGridHornGeometry, n_length: int) -> tuple[HornInterface, ...]:
    if geometry.interfaces:
        return tuple(
            HornInterface(slice_index=int(spec.slice_index), offset_mm=float(spec.offset_mm))
            for spec in geometry.interfaces
            if float(spec.offset_mm) > 0.0 and 0 <= int(spec.slice_index) < n_length
        )
    if geometry.interface_offset_mm <= 0.0:
        return ()
    return (HornInterface(slice_index=n_length - 1, offset_mm=float(geometry.interface_offset_mm)),)


def _add_offset_interface_surfaces(
    inner_points: np.ndarray,
    *,
    slice_index: int,
    closed: bool,
    offset_mm: float,
) -> list[tuple[int, int]]:
    if offset_mm <= 0.0:
        return []

    gmsh = require_gmsh()
    base = np.asarray(inner_points[:, int(slice_index), :], dtype=np.float64)
    offset = np.array(base, dtype=np.float64, copy=True)
    offset[:, 2] += float(offset_mm)
    center = np.asarray(
        (
            float(np.mean(base[:, 0])) if closed else 0.0,
            float(np.mean(base[:, 1])) if closed else 0.0,
            float(np.mean(offset[:, 2])),
        ),
        dtype=np.float64,
    )

    base_tags = [
        int(gmsh.model.occ.addPoint(float(p[0]), float(p[1]), float(p[2])))
        for p in base
    ]
    offset_tags = [
        int(gmsh.model.occ.addPoint(float(p[0]), float(p[1]), float(p[2])))
        for p in offset
    ]
    center_tag = int(gmsh.model.occ.addPoint(float(center[0]), float(center[1]), float(center[2])))
    radial_lines = {
        i: int(gmsh.model.occ.addLine(center_tag, offset_tags[i]))
        for i in range(len(offset_tags))
    }

    def spline(tags: list[int]) -> int:
        return int(gmsh.model.occ.addBSpline([int(tag) for tag in tags]))

    def line(a: int, b: int) -> int:
        return int(gmsh.model.occ.addLine(int(a), int(b)))

    def surface(curves: list[int], *, plane: bool = False) -> tuple[int, int]:
        try:
            loop = int(gmsh.model.occ.addCurveLoop([int(c) for c in curves], reorient=True))
        except TypeError:
            loop = int(gmsh.model.occ.addCurveLoop([int(c) for c in curves]))
        if plane:
            try:
                return (2, int(gmsh.model.occ.addPlaneSurface([loop])))
            except Exception:
                pass
        return (2, int(gmsh.model.occ.addSurfaceFilling(loop)))

    surfaces: list[tuple[int, int]] = []
    for group in _interface_phi_groups(len(base_tags), closed=closed):
        offset_curves: list[int] = []
        for span in _split_interface_group(group):
            base_curve = spline([base_tags[i] for i in span])
            offset_curve = spline([offset_tags[i] for i in span])
            left = line(base_tags[span[0]], offset_tags[span[0]])
            right = line(base_tags[span[-1]], offset_tags[span[-1]])
            surfaces.append(surface([base_curve, right, -offset_curve, -left]))
            offset_curves.append(offset_curve)
        surfaces.append(surface([radial_lines[group[0]], *offset_curves, -radial_lines[group[-1]]], plane=True))
    return surfaces


def build_point_grid(geometry: PointGridHornGeometry) -> BuiltGeometry:
    inner_points = _validated_grid(geometry.inner_points, name="inner_points")

    if geometry.outer_points is not None and geometry.enclosure is None:
        return _build_freestanding_point_grid(geometry)

    if geometry.enclosure is not None:
        inner_points = _snap_open_symmetry_grid(inner_points, closed=geometry.closed)
        span_builder = _SharedSurfaceBuilder()
        span_builder.add_grid("inner", inner_points)
        n_phi, n_len, _ = inner_points.shape
        wall = _add_occ_spline_span_wall_surfaces(
            span_builder,
            "inner",
            n_phi=n_phi,
            n_len=n_len,
            closed=geometry.closed,
        )
        throat = _add_occ_source_cap_surfaces(span_builder, inner_points, geometry)
        require_gmsh().model.occ.synchronize()
    else:
        wall = build_surface_from_points(
            inner_points,
            closed=geometry.closed,
            preserve_grid=geometry.preserve_grid,
        )
        require_gmsh().model.occ.synchronize()

        if geometry.closed:
            throat = make_planar_fill_from_boundary(
                wall,
                source_axis="z",
                use_min=True,
                closed=True,
            )
            if not throat:
                throat = make_planar_fill_from_ring(inner_points[:, 0, :])
        else:
            throat = make_planar_sector_fill_from_ring(
                inner_points[:, 0, :],
                source_axis="z",
            )
            if not throat:
                throat = make_planar_fill_from_boundary(
                    wall,
                    source_axis="z",
                    use_min=True,
                    closed=False,
                )

    wall_tags = [tag for _, tag in wall]
    throat_tags = [tag for _, tag in throat]

    mesh_surface_groups: dict[str, list[int]] = {
        "inner": list(wall_tags),
        "throat_disc": list(throat_tags),
    }
    rigid_wall_tags: list[int] = list(wall_tags)
    enclosure_tags: list[int] = []
    interface_tags: list[int] = []
    enclosure_bounds: dict[str, float] | None = None

    if geometry.enclosure is not None:
        interface_dimtags: list[tuple[int, int]] = []
        for interface in _normalise_interface_specs(geometry, inner_points.shape[1]):
            interface_dimtags.extend(
                _add_offset_interface_surfaces(
                    inner_points,
                    slice_index=int(interface.slice_index),
                    closed=geometry.closed,
                    offset_mm=float(interface.offset_mm),
                )
            )
        interface_tags = [tag for _, tag in interface_dimtags]
        if interface_tags:
            mesh_surface_groups["interface"] = list(interface_tags)
        enc_data = build_enclosure_box(
            inner_dimtags=wall,
            inner_points=inner_points,
            enclosure=geometry.enclosure,
            closed=geometry.closed,
        )
        enclosure_tags = [tag for _, tag in enc_data["dimtags"]]
        # All enclosure surfaces join the "enclosure" group so density.py's
        # z-interpolated front/back side-wall formula applies. The roundover
        # surfaces additionally appear in the front/back edge groups so the
        # panel-bilinear formula clamps them via the Min field.
        mesh_surface_groups["enclosure"] = list(enclosure_tags)
        mesh_surface_groups["enclosure_edges_front"] = list(enc_data["front_edges"])
        mesh_surface_groups["enclosure_edges_back"] = list(enc_data["back_edges"])
        enclosure_bounds = dict(enc_data["bounds"])

    z0 = float(np.mean(inner_points[:, 0, 2]))
    z1 = float(np.mean(inner_points[:, -1, 2]))
    surface_groups = {
        int(PhysicalGroup.RIGID_WALL): rigid_wall_tags,
        int(PhysicalGroup.PRIMARY_SOURCE): throat_tags,
    }
    if enclosure_tags:
        surface_groups[int(PhysicalGroup.ENCLOSURE_WALL)] = enclosure_tags
    if interface_tags:
        surface_groups[int(PhysicalGroup.INTERFACE)] = interface_tags
    return BuiltGeometry(
        surface_groups=surface_groups,
        axial_bounds_mm=(z0, z1),
        source_axis="z",
        mesh_surface_groups=mesh_surface_groups,
        enclosure_bounds=enclosure_bounds,
    )
