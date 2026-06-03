from __future__ import annotations

import numpy as np

from ._occ import require_gmsh

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


def _validated_grid(points: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{name} must be shaped (n_phi, n_length + 1, 3)")
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(f"{name} needs at least 2 phi samples and 2 axial rings")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr
