from __future__ import annotations

import math

import numpy as np

from ..geometry import BuiltGeometry, PointGridHornGeometry
from ..tags import PhysicalGroup
from ._occ import (
    build_surface_from_points,
    make_planar_fill_from_boundary,
    make_planar_fill_from_ring,
    make_planar_sector_fill_from_ring,
    require_gmsh,
)
from .enclosure import (
    _add_curve_loop_from_curves,
    _boundary_curves_at_z_extreme,
    _make_wire,
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
    """Gmsh built-in-kernel helper for ATH-compatible spline surfaces."""

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


def _ath_phi_spans(n_phi: int, *, closed: bool) -> list[tuple[int, int]]:
    if closed or n_phi < 3:
        return [(0, n_phi - 1)]
    mid = n_phi // 2
    return [(0, mid), (mid, n_phi - 1)]


def _snap_open_symmetry_grid(points: np.ndarray, *, closed: bool) -> np.ndarray:
    out = np.array(points, dtype=np.float64, copy=True)
    if not closed and out.shape[0] >= 2:
        out[0, :, 1] = 0.0
        out[-1, :, 0] = 0.0
    return out


def _ath_outer_axial_indices(inner_points: np.ndarray) -> list[int]:
    z_by_ring = np.mean(inner_points[:, :, 2], axis=0)
    max_z = float(np.max(z_by_ring))
    tol = max(1.0e-6, 1.0e-9 * max(1.0, abs(max_z)))
    return [
        j
        for j in range(1, inner_points.shape[1])
        if float(z_by_ring[j]) < max_z - tol
    ]


def _ath_rear_extra(throat_radius: float, geometry: PointGridHornGeometry) -> float:
    extra = 0.5 * _source_cap_height(throat_radius, geometry)
    if geometry.source_auto_angle_deg is None:
        return extra
    cos_angle = math.cos(math.radians(float(geometry.source_auto_angle_deg)))
    if abs(cos_angle) <= 1.0e-9:
        return extra
    return extra / abs(cos_angle)


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


def _add_geo_ath_span_wall_surfaces(
    builder: _GeoSurfaceBuilder,
    name: str,
    *,
    n_phi: int,
    n_len: int,
    closed: bool,
    reverse: bool = False,
) -> list[tuple[int, int]]:
    surfaces: list[tuple[int, int]] = []
    for start, end in _ath_phi_spans(n_phi, closed=closed):
        prev_phi = builder.spline([(name, i, 0) for i in range(start, end + 1)])
        for j in range(n_len - 1):
            next_phi = builder.spline([(name, i, j + 1) for i in range(start, end + 1)])
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


def _add_geo_ath_mouth_rim_surfaces(
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
    for start, end in _ath_phi_spans(n_phi, closed=closed):
        inner_phi = builder.spline([("inner", i, ji) for i in range(start, end + 1)])
        outer_phi = builder.spline([("outer", i, jo) for i in range(start, end + 1)])
        left = builder.line(("inner", start, ji), ("outer", start, jo))
        right = builder.line(("inner", end, ji), ("outer", end, jo))
        surfaces.append(builder.surface([outer_phi, -right, -inner_phi, left]))
    return surfaces


def _add_geo_ath_rear_cap(
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
    for start, end in _ath_phi_spans(n_phi, closed=closed):
        phi_curve = builder.spline([("outer", i, 0) for i in range(start, end + 1)])
        cap.append(builder.surface([radial_lines[start], phi_curve, -radial_lines[end]]))
    return cap


def _add_geo_ath_source_surface(
    builder: _GeoSurfaceBuilder,
    inner_points: np.ndarray,
    geometry: PointGridHornGeometry,
    *,
    mesh_size: float,
) -> list[tuple[int, int]]:
    if geometry.closed or int(geometry.source_shape) != 1:
        return []

    n_phi = inner_points.shape[0]
    ring = inner_points[:, 0, :]
    center = np.mean(ring, axis=0)
    center[0] = 0.0
    center[1] = 0.0
    center[2] = float(np.mean(ring[:, 2]))

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
            builder.spline([("inner", i, 0) for i in range(start, end + 1)])
            for start, end in _ath_phi_spans(n_phi, closed=False)
        ],
        -arc_end,
    ]
    return [builder.surface(boundary, sphere_center_tag=sphere_center_tag)]


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
    if geometry.ath_parity_topology:
        return _build_ath_parity_freestanding_point_grid(
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


def _build_ath_parity_freestanding_point_grid(
    geometry: PointGridHornGeometry,
    inner_points: np.ndarray,
    outer_points: np.ndarray,
) -> BuiltGeometry:
    inner_points = _snap_open_symmetry_grid(inner_points, closed=geometry.closed)
    outer_points = _snap_open_symmetry_grid(outer_points, closed=geometry.closed)

    n_phi, inner_len, _ = inner_points.shape
    throat_radius = _throat_radius(inner_points, closed=geometry.closed)
    rear_z = float(
        np.mean(inner_points[:, 0, 2])
        - float(geometry.wall_thickness_mm)
        - _ath_rear_extra(throat_radius, geometry)
    )
    rear_points = _rear_rim_points(outer_points, rear_z=rear_z)
    rear_points = _snap_open_symmetry_grid(rear_points[:, np.newaxis, :], closed=geometry.closed)[:, 0, :]

    outer_indices = _ath_outer_axial_indices(inner_points)
    outer_topology = np.empty((n_phi, len(outer_indices) + 1, 3), dtype=np.float64)
    outer_topology[:, 0, :] = rear_points
    for out_j, src_j in enumerate(outer_indices, start=1):
        outer_topology[:, out_j, :] = outer_points[:, src_j, :]

    builder = _GeoSurfaceBuilder()
    inner_mesh_sizes = np.full(inner_points.shape[:2], 8.0, dtype=np.float64)
    inner_mesh_sizes[:, 0] = 5.0
    builder.add_grid("inner", inner_points, mesh_size=inner_mesh_sizes)
    builder.add_grid("outer", outer_topology, mesh_size=25.0)

    wall = _add_geo_ath_span_wall_surfaces(
        builder,
        "inner",
        n_phi=n_phi,
        n_len=inner_len,
        closed=geometry.closed,
    )
    outer_wall = _add_geo_ath_span_wall_surfaces(
        builder,
        "outer",
        n_phi=n_phi,
        n_len=outer_topology.shape[1],
        closed=geometry.closed,
        reverse=True,
    )
    mouth_dimtags = _add_geo_ath_mouth_rim_surfaces(
        builder,
        n_phi=n_phi,
        inner_len=inner_len,
        outer_len=outer_topology.shape[1],
        closed=geometry.closed,
    )
    rear_cap = _add_geo_ath_rear_cap(
        builder,
        rear_points,
        n_phi=n_phi,
        closed=geometry.closed,
        mesh_size=25.0,
    )
    throat = _add_geo_ath_source_surface(
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


def _build_mouth_rim_from_boundaries(
    inner_dimtags: list[tuple[int, int]],
    outer_dimtags: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Annular ruled mouth rim using actual inner+outer wall boundary curves.

    Mirrors WG ``_build_mouth_rim_from_boundaries`` / ``_build_annular_surface_from_boundaries``.
    Requires a prior ``occ.synchronize()``.
    """

    gmsh = require_gmsh()
    inner_curves = _boundary_curves_at_z_extreme(inner_dimtags, want_min_z=False)
    outer_curves = _boundary_curves_at_z_extreme(outer_dimtags, want_min_z=False)
    if not inner_curves or not outer_curves:
        return []
    try:
        iw = _add_curve_loop_from_curves(inner_curves)
        ow = _add_curve_loop_from_curves(outer_curves)
    except Exception:
        return []
    return list(
        gmsh.model.occ.addThruSections(
            [int(iw), int(ow)], makeSolid=False, makeRuled=True
        )
    )


def _build_mouth_rim_from_control_points(
    inner_points: np.ndarray, outer_points: np.ndarray, *, closed: bool = True
) -> list[tuple[int, int]]:
    """Fallback mouth rim: build inner+outer mouth wires from control points."""

    gmsh = require_gmsh()
    j_mouth = inner_points.shape[1] - 1
    w_inner, _, _ = _make_wire(inner_points[:, j_mouth, :], closed=closed)
    w_outer, _, _ = _make_wire(outer_points[:, j_mouth, :], closed=closed)
    return list(
        gmsh.model.occ.addThruSections(
            [int(w_inner), int(w_outer)], makeSolid=False, makeRuled=True
        )
    )


def _build_rear_disc_assembly(
    outer_points: np.ndarray,
    *,
    closed: bool = True,
    outer_dimtags: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Flat rear disc that closes the freestanding wall shell at the outer throat ring.

    Preferred path: reuse the outer-wall throat boundary curves so the disc
    shares topology with the outer wall. Fallback: build the wire from the
    `outer_points[:, 0, :]` control ring.
    """

    gmsh = require_gmsh()

    if outer_dimtags:
        throat_curves = _boundary_curves_at_z_extreme(outer_dimtags, want_min_z=True)
        if throat_curves:
            try:
                loop = _add_curve_loop_from_curves(throat_curves)
                try:
                    disc_fill = int(gmsh.model.occ.addPlaneSurface([int(loop)]))
                except Exception:
                    disc_fill = int(gmsh.model.occ.addSurfaceFilling(int(loop)))
                return [(2, disc_fill)]
            except Exception:
                pass

    throat_ring = outer_points[:, 0, :]
    if not closed:
        return make_planar_sector_fill_from_ring(throat_ring, source_axis="z")

    wire, curves, _ = _make_wire(throat_ring, closed=closed)
    loop = _add_curve_loop_from_curves(curves)
    try:
        disc_fill = int(gmsh.model.occ.addPlaneSurface([int(loop)]))
    except Exception:
        disc_fill = int(gmsh.model.occ.addSurfaceFilling(int(loop)))
    return [(2, disc_fill)]


def _validated_grid(points: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{name} must be shaped (n_phi, n_length + 1, 3)")
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(f"{name} needs at least 2 phi samples and 2 axial rings")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def build_point_grid(geometry: PointGridHornGeometry) -> BuiltGeometry:
    inner_points = _validated_grid(geometry.inner_points, name="inner_points")

    if geometry.outer_points is not None and geometry.enclosure is None:
        return _build_freestanding_point_grid(geometry)

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
    enclosure_bounds: dict[str, float] | None = None

    if geometry.outer_points is not None and geometry.enclosure is None:
        outer_points = _validated_grid(geometry.outer_points, name="outer_points")

        outer_wall = build_surface_from_points(
            outer_points,
            closed=geometry.closed,
            preserve_grid=geometry.preserve_grid,
        )
        require_gmsh().model.occ.synchronize()

        # Mouth rim: preferred path reuses inner+outer wall boundary curves.
        mouth_dimtags = _build_mouth_rim_from_boundaries(wall, outer_wall)
        if not mouth_dimtags:
            mouth_dimtags = _build_mouth_rim_from_control_points(
                inner_points, outer_points, closed=geometry.closed
            )

        # Rear disc: preferred path reuses outer-wall throat boundary curves.
        rear_dimtags = _build_rear_disc_assembly(
            outer_points,
            closed=geometry.closed,
            outer_dimtags=outer_wall,
        )
        require_gmsh().model.occ.synchronize()

        outer_tags = [tag for _, tag in outer_wall]
        mouth_tags = [tag for dim, tag in mouth_dimtags if int(dim) == 2]
        rear_tags = [tag for dim, tag in rear_dimtags if int(dim) == 2]

        # Throat disc stays attached only to the inner wall — the freestanding
        # throat cavity is intentionally hollow so the source patch never
        # connects directly to the shell or rear closure.
        rigid_wall_tags.extend(outer_tags)
        rigid_wall_tags.extend(mouth_tags)
        rigid_wall_tags.extend(rear_tags)

        mesh_surface_groups["outer"] = list(outer_tags)
        mesh_surface_groups["mouth"] = list(mouth_tags)
        mesh_surface_groups["rear"] = list(rear_tags)
        mesh_surface_groups["rear_cap"] = list(rear_tags)

    if geometry.enclosure is not None:
        enc_data = build_enclosure_box(
            inner_dimtags=wall,
            inner_points=inner_points,
            enclosure=geometry.enclosure,
            closed=geometry.closed,
        )
        enc_tags = [tag for _, tag in enc_data["dimtags"]]
        rigid_wall_tags.extend(enc_tags)
        # All enclosure surfaces join the "enclosure" group so density.py's
        # z-interpolated front/back side-wall formula applies. The roundover
        # surfaces additionally appear in the front/back edge groups so the
        # panel-bilinear formula clamps them via the Min field.
        mesh_surface_groups["enclosure"] = list(enc_tags)
        mesh_surface_groups["enclosure_edges_front"] = list(enc_data["front_edges"])
        mesh_surface_groups["enclosure_edges_back"] = list(enc_data["back_edges"])
        enclosure_bounds = dict(enc_data["bounds"])

    z0 = float(np.mean(inner_points[:, 0, 2]))
    z1 = float(np.mean(inner_points[:, -1, 2]))
    return BuiltGeometry(
        surface_groups={
            int(PhysicalGroup.RIGID_WALL): rigid_wall_tags,
            int(PhysicalGroup.PRIMARY_SOURCE): throat_tags,
        },
        axial_bounds_mm=(z0, z1),
        source_axis="z",
        mesh_surface_groups=mesh_surface_groups,
        enclosure_bounds=enclosure_bounds,
    )
