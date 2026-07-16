from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from ..tags import PhysicalGroup


def require_gmsh():
    import gmsh

    return gmsh


def superellipse_ring(
    *,
    z: float,
    radius: float,
    exponent: float,
    aspect_ratio: float,
    n_phi: int,
) -> NDArray[np.float64]:
    if n_phi < 8:
        raise ValueError("n_phi must be at least 8")
    if radius <= 0.0 or not math.isfinite(radius):
        raise ValueError("radius must be finite and > 0")
    p = max(float(exponent), 0.25)
    aspect = max(float(aspect_ratio), 1e-6)
    a = radius * aspect
    b = radius

    pts = np.empty((n_phi, 3), dtype=np.float64)
    for i in range(n_phi):
        theta = 2.0 * math.pi * i / n_phi
        c = math.cos(theta)
        s = math.sin(theta)
        x = a * math.copysign(abs(c) ** (2.0 / p), c)
        y = b * math.copysign(abs(s) ** (2.0 / p), s)
        pts[i] = (x, y, z)
    return pts


def rounded_rect_ring(
    *,
    z: float,
    half_width: float,
    half_height: float,
    exponent: float,
    n_phi: int,
) -> NDArray[np.float64]:
    if half_width <= 0 or half_height <= 0:
        raise ValueError("rectangular horn half-width and half-height must be > 0")
    p = max(float(exponent), 2.0)
    pts = np.empty((n_phi, 3), dtype=np.float64)
    for i in range(n_phi):
        theta = 2.0 * math.pi * i / n_phi
        c = math.cos(theta)
        s = math.sin(theta)
        x = half_width * math.copysign(abs(c) ** (2.0 / p), c)
        y = half_height * math.copysign(abs(s) ** (2.0 / p), s)
        pts[i] = (x, y, z)
    return pts


def build_bspline_surface_from_rings(
    points: NDArray[np.float64],
) -> list[tuple[int, int]]:
    gmsh = require_gmsh()
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("point grid must be shaped (n_phi, n_length, 3)")
    n_phi, n_len, _ = points.shape
    if n_phi < 4 or n_len < 2:
        raise ValueError("point grid needs at least 4 phi samples and 2 axial rings")

    degree_u = min(3, n_phi)
    degree_v = min(3, max(1, n_len - 1))
    pt_tags: list[int] = []
    for j in range(n_len):
        for i in list(range(n_phi)) + [0]:
            x, y, z = points[i, j]
            pt_tags.append(gmsh.model.occ.addPoint(float(x), float(y), float(z)))
    surf = gmsh.model.occ.addBSplineSurface(
        pt_tags,
        n_phi + 1,
        degreeU=degree_u,
        degreeV=degree_v,
    )
    return [(2, int(surf))]


def build_faceted_surface_from_points(
    points: NDArray[np.float64],
    *,
    closed: bool = True,
) -> list[tuple[int, int]]:
    """Build a ruled surface that interpolates every sampled grid point."""

    gmsh = require_gmsh()
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("point grid must be shaped (n_phi, n_length, 3)")
    n_phi, n_len, _ = points.shape
    if n_phi < 2 or n_len < 2:
        raise ValueError("point grid needs at least 2 phi samples and 2 axial rings")

    point_tags: dict[tuple[int, int], int] = {}
    for i in range(n_phi):
        for j in range(n_len):
            x, y, z = points[i, j]
            point_tags[(i, j)] = int(
                gmsh.model.occ.addPoint(float(x), float(y), float(z))
            )

    line_cache: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}

    def line(a: tuple[int, int], b: tuple[int, int]) -> int:
        if (a, b) in line_cache:
            return line_cache[(a, b)]
        if (b, a) in line_cache:
            return -line_cache[(b, a)]
        tag = int(gmsh.model.occ.addLine(point_tags[a], point_tags[b]))
        line_cache[(a, b)] = tag
        return tag

    surfaces: list[tuple[int, int]] = []
    phi_count = n_phi if closed else n_phi - 1
    for i in range(phi_count):
        i_next = (i + 1) % n_phi
        for j in range(n_len - 1):
            curves = [
                line((i, j), (i_next, j)),
                line((i_next, j), (i_next, j + 1)),
                line((i_next, j + 1), (i, j + 1)),
                line((i, j + 1), (i, j)),
            ]
            loop = gmsh.model.occ.addCurveLoop(curves)
            try:
                surf = gmsh.model.occ.addPlaneSurface([loop])
            except Exception:
                surf = gmsh.model.occ.addSurfaceFilling(loop)
            surfaces.append((2, int(surf)))
    return surfaces


def build_surface_from_points(
    points: NDArray[np.float64],
    *,
    closed: bool = True,
    preserve_grid: bool = False,
) -> list[tuple[int, int]]:
    """Build the WG-compatible OCC horn surface from a point grid."""

    gmsh = require_gmsh()
    if preserve_grid:
        return build_faceted_surface_from_points(points, closed=closed)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("point grid must be shaped (n_phi, n_length, 3)")
    n_phi, n_len, _ = points.shape
    if n_phi < 2 or n_len < 2:
        raise ValueError("point grid needs at least 2 phi samples and 2 axial rings")

    degree_v = min(3, max(1, n_len - 1))

    def make_patch(column_indices: list[int]) -> int:
        n_u = len(column_indices)
        degree_u = min(3, max(1, n_u - 1))
        point_tags: list[int] = []
        for j in range(n_len):
            for i in column_indices:
                x, y, z = points[i, j]
                point_tags.append(gmsh.model.occ.addPoint(float(x), float(y), float(z)))
        return int(
            gmsh.model.occ.addBSplineSurface(
                point_tags,
                n_u,
                degreeU=degree_u,
                degreeV=degree_v,
            )
        )

    if closed:
        return [(2, make_patch(list(range(n_phi)) + [0]))]
    return [(2, make_patch(list(range(n_phi))))]


def make_ring_wire(points: NDArray[np.float64]) -> tuple[int, list[int]]:
    gmsh = require_gmsh()
    pt_tags = [
        gmsh.model.occ.addPoint(float(x), float(y), float(z)) for x, y, z in points
    ]
    pt_tags.append(pt_tags[0])
    curve = gmsh.model.occ.addBSpline(pt_tags)
    loop = gmsh.model.occ.addCurveLoop([int(curve)])
    return int(loop), [int(curve)]


def make_planar_fill_from_ring(points: NDArray[np.float64]) -> list[tuple[int, int]]:
    gmsh = require_gmsh()
    loop, _ = make_ring_wire(points)
    try:
        surf = gmsh.model.occ.addPlaneSurface([loop])
    except Exception:
        surf = gmsh.model.occ.addSurfaceFilling(loop)
    return [(2, int(surf))]


def make_planar_sector_fill_from_ring(
    points: NDArray[np.float64],
    *,
    source_axis: str = "z",
) -> list[tuple[int, int]]:
    """Fill an open symmetry-sector ring as one Gmsh-meshed planar surface."""

    gmsh = require_gmsh()
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
        return []
    axis_idx = {"x": 0, "y": 1, "z": 2}.get(source_axis, 2)
    center = np.zeros(3, dtype=np.float64)
    center[axis_idx] = float(np.mean(arr[:, axis_idx]))
    center_tag = int(
        gmsh.model.occ.addPoint(
            float(center[0]),
            float(center[1]),
            float(center[2]),
        )
    )
    point_tags = [
        int(gmsh.model.occ.addPoint(float(x), float(y), float(z))) for x, y, z in arr
    ]
    ring_curve = int(gmsh.model.occ.addBSpline(point_tags))
    end_to_center = int(gmsh.model.occ.addLine(point_tags[-1], center_tag))
    center_to_start = int(gmsh.model.occ.addLine(center_tag, point_tags[0]))
    loop = gmsh.model.occ.addCurveLoop([ring_curve, end_to_center, center_to_start])
    try:
        surf = gmsh.model.occ.addPlaneSurface([loop])
    except Exception:
        surf = gmsh.model.occ.addSurfaceFilling(loop)
    return [(2, int(surf))]


def extreme_boundary_loop_curves(
    dimtags: list[tuple[int, int]],
    *,
    source_axis: str = "z",
    use_min: bool = True,
) -> list[int]:
    """Boundary curves of ``dimtags`` lying entirely on the extreme axis plane.

    Requires a prior ``occ.synchronize()``.
    """

    gmsh = require_gmsh()
    boundary = gmsh.model.getBoundary(dimtags, oriented=False, combined=False)
    curve_tags: list[int] = []
    seen: set[int] = set()
    for dim, tag in boundary:
        if int(dim) != 1:
            continue
        curve_tag = int(tag)
        if curve_tag in seen:
            continue
        seen.add(curve_tag)
        curve_tags.append(curve_tag)
    if not curve_tags:
        return []

    axis_idx = {"x": 0, "y": 1, "z": 2}.get(source_axis, 2)
    bounds: dict[int, tuple[float, float]] = {}
    lo_all = float("inf")
    hi_all = float("-inf")
    for curve_tag in curve_tags:
        box = gmsh.model.getBoundingBox(1, curve_tag)
        lo = float(min(box[axis_idx], box[axis_idx + 3]))
        hi = float(max(box[axis_idx], box[axis_idx + 3]))
        bounds[curve_tag] = (lo, hi)
        lo_all = min(lo_all, lo)
        hi_all = max(hi_all, hi)
    if not math.isfinite(lo_all):
        return []

    target = lo_all if use_min else hi_all
    eps = max(1e-6, abs(hi_all - lo_all) * 1e-3)
    return [
        curve_tag
        for curve_tag, (lo, hi) in bounds.items()
        if abs(lo - target) <= eps and abs(hi - target) <= eps
    ]


def boundary_loop_curves_at_axis_value(
    dimtags: list[tuple[int, int]],
    *,
    source_axis: str = "z",
    axis_value: float,
    tolerance: float = 1.0e-6,
) -> list[int]:
    """Boundary curves of ``dimtags`` lying entirely on an axis-aligned plane."""

    gmsh = require_gmsh()
    boundary = gmsh.model.getBoundary(dimtags, oriented=False, combined=False)
    curve_tags: list[int] = []
    seen: set[int] = set()
    for dim, tag in boundary:
        if int(dim) != 1:
            continue
        curve_tag = int(tag)
        if curve_tag in seen:
            continue
        seen.add(curve_tag)
        curve_tags.append(curve_tag)
    if not curve_tags:
        return []

    axis_idx = {"x": 0, "y": 1, "z": 2}.get(source_axis, 2)
    target = float(axis_value)
    eps = max(float(tolerance), abs(target) * 1.0e-9)
    loop_curves: list[int] = []
    for curve_tag in curve_tags:
        box = gmsh.model.getBoundingBox(1, curve_tag)
        lo = float(min(box[axis_idx], box[axis_idx + 3]))
        hi = float(max(box[axis_idx], box[axis_idx + 3]))
        if abs(lo - target) <= eps and abs(hi - target) <= eps:
            loop_curves.append(curve_tag)
    return loop_curves


def _make_planar_fill_from_loop_curves(
    loop_curves: list[int],
    *,
    closed: bool = True,
    source_axis: str = "z",
) -> list[tuple[int, int]]:
    gmsh = require_gmsh()
    if not loop_curves:
        return []

    if closed:
        loop = gmsh.model.occ.addCurveLoop(loop_curves)
    else:
        try:
            boundary_points = gmsh.model.getBoundary(
                [(1, tag) for tag in loop_curves],
                oriented=False,
                combined=True,
            )
            point_tags = [
                int(abs(tag)) for dim, tag in boundary_points if int(dim) == 0
            ]
            if len(point_tags) >= 2:
                axis_idx = {"x": 0, "y": 1, "z": 2}.get(source_axis, 2)
                endpoint = np.asarray(
                    [gmsh.model.getValue(0, point_tags[0], [])],
                    dtype=np.float64,
                ).reshape(3)
                center = np.zeros(3, dtype=np.float64)
                center[axis_idx] = float(endpoint[axis_idx])
                center_tag = int(
                    gmsh.model.occ.addPoint(
                        float(center[0]), float(center[1]), float(center[2])
                    )
                )
                to_center = int(gmsh.model.occ.addLine(point_tags[0], center_tag))
                from_center = int(gmsh.model.occ.addLine(center_tag, point_tags[1]))
                loop = gmsh.model.occ.addCurveLoop(
                    loop_curves + [to_center, from_center]
                )
            else:
                loop = gmsh.model.occ.addCurveLoop(loop_curves)
        except Exception:
            loop = gmsh.model.occ.addCurveLoop(loop_curves)

    try:
        surf = gmsh.model.occ.addPlaneSurface([loop])
    except Exception:
        surf = gmsh.model.occ.addSurfaceFilling(loop)
    return [(2, int(surf))]


def make_planar_fill_from_boundary(
    dimtags: list[tuple[int, int]],
    *,
    source_axis: str = "z",
    use_min: bool = True,
    closed: bool = True,
) -> list[tuple[int, int]]:
    """Fill an extreme boundary loop using the existing OCC boundary curves."""

    loop_curves = extreme_boundary_loop_curves(
        dimtags, source_axis=source_axis, use_min=use_min
    )
    return _make_planar_fill_from_loop_curves(
        loop_curves, closed=closed, source_axis=source_axis
    )


def make_planar_fill_from_boundary_at_axis_value(
    dimtags: list[tuple[int, int]],
    *,
    source_axis: str = "z",
    axis_value: float,
    closed: bool = True,
    tolerance: float = 1.0e-6,
) -> list[tuple[int, int]]:
    """Fill a boundary loop using existing curves on a specific axis plane."""

    loop_curves = boundary_loop_curves_at_axis_value(
        dimtags,
        source_axis=source_axis,
        axis_value=axis_value,
        tolerance=tolerance,
    )
    return _make_planar_fill_from_loop_curves(
        loop_curves, closed=closed, source_axis=source_axis
    )


def add_physical_groups(surface_groups: dict[int, list[int]]) -> None:
    gmsh = require_gmsh()
    from ..tags import PHYSICAL_NAMES

    for tag, surfaces in sorted(surface_groups.items()):
        clean = sorted({int(s) for s in surfaces if int(s) > 0})
        if not clean:
            continue
        gmsh.model.addPhysicalGroup(2, clean, tag=int(tag))
        gmsh.model.setPhysicalName(
            2, int(tag), PHYSICAL_NAMES.get(int(tag), f"SD1D{1000 + int(tag) - 1}")
        )


def collect_wall_surfaces(excluding: Iterable[int] = ()) -> list[int]:
    gmsh = require_gmsh()
    skip = {int(v) for v in excluding}
    return [
        int(tag)
        for dim, tag in gmsh.model.getEntities(2)
        if int(dim) == 2 and int(tag) not in skip
    ]


def validate_source_tag(tag: int) -> None:
    if int(tag) == int(PhysicalGroup.RIGID_WALL):
        raise ValueError("driver/source tag 1 is reserved for rigid walls")
    if int(tag) < int(PhysicalGroup.PRIMARY_SOURCE):
        raise ValueError("driver/source tags must be >= 2")
