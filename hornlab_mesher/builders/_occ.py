from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


SURFACE_FIT_APPROXIMATE = "approximate"
SURFACE_FIT_INTERPOLATE = "interpolate"
SURFACE_FIT_MODES = (SURFACE_FIT_APPROXIMATE, SURFACE_FIT_INTERPOLATE)


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
    surface_fit: str = SURFACE_FIT_APPROXIMATE,
) -> list[tuple[int, int]]:
    """Build the WG-compatible OCC horn surface from a point grid."""

    require_gmsh()
    if preserve_grid:
        return build_faceted_surface_from_points(points, closed=closed)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("point grid must be shaped (n_phi, n_length, 3)")
    n_phi, n_len, _ = points.shape
    if n_phi < 2 or n_len < 2:
        raise ValueError("point grid needs at least 2 phi samples and 2 axial rings")

    degree_v = min(3, max(1, n_len - 1))

    def make_patch(column_indices: list[int]) -> int:
        return add_bspline_patch(
            points,
            column_indices,
            degree_v=degree_v,
            surface_fit=surface_fit,
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
    wall_grid: NDArray[np.float64] | None = None,
    wall_columns: list[int] | None = None,
    surface_fit: str = SURFACE_FIT_APPROXIMATE,
) -> list[tuple[int, int]]:
    """Fill an open symmetry-sector ring as one Gmsh-meshed planar surface.

    The ring curve must be the wall's own throat edge, because the seam welds
    on coincident mesh nodes rather than on shared topology. Pass the wall's
    grid and column partition so an interpolating wall gets an interpolating
    rim; without them the rim is the plain pole B-spline the approximating wall
    has.
    """

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
    if (
        surface_fit == SURFACE_FIT_INTERPOLATE
        and wall_grid is not None
        and wall_columns is not None
    ):
        poles, (knots, multiplicities), degree = throat_boundary_curve(
            wall_grid, list(wall_columns)
        )
        pole_tags = [point_tags[0]]
        pole_tags.extend(
            int(gmsh.model.occ.addPoint(float(x), float(y), float(z)))
            for x, y, z in poles[1:-1]
        )
        pole_tags.append(point_tags[-1])
        ring_curve = int(
            gmsh.model.occ.addBSpline(
                pole_tags,
                degree=int(degree),
                knots=[float(k) for k in knots],
                multiplicities=[int(m) for m in multiplicities],
            )
        )
    else:
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

    # Only curves that are themselves flat in ``source_axis`` can bound a
    # planar cap, so the extreme is taken over those rather than over every
    # boundary curve. Letting a wall-running curve set the target makes the
    # search fail closed (returning no loop, hence no cap) whenever such a
    # curve overshoots the cap plane by any amount at all.
    #
    # The ``else`` below is the pre-change selection, and it is a no-op: with no
    # flat curve, any curve within ``eps`` of the whole-boundary extreme at both
    # ends would have a span of at most ``eps`` and so would have been flat. It
    # therefore always yields an empty loop -- the same fail-closed answer as
    # before -- which is why this restriction moves no existing mesh.
    eps = max(1e-6, abs(hi_all - lo_all) * 1e-3)
    flat = {tag: lo for tag, (lo, hi) in bounds.items() if abs(hi - lo) <= eps}
    if flat:
        target = min(flat.values()) if use_min else max(flat.values())
    else:
        target = lo_all if use_min else hi_all
    return [
        curve_tag
        for curve_tag, (lo, hi) in bounds.items()
        if abs(lo - target) <= eps and abs(hi - target) <= eps
    ]


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


# ---------------------------------------------------------------------------
# B-spline surface fitting
# ---------------------------------------------------------------------------
#
# ``occ.addBSplineSurface`` treats the tags it is handed as *control points*
# (poles), not as points the surface passes through. Feeding it the sampled
# profile grid therefore meshes a surface that hangs systematically *inside*
# the sampled one: for a cubic pole fit the offset is about R*dtheta^2/6, which
# on a stock OSSE measures ~0.12 mm rms and ~0.25 mm peak, biased inward and
# growing toward the mouth. Refining the mesh converges onto that biased
# surface rather than onto the analytic one, so the error is a floor no
# element-size or curvature setting can reach.
#
# ``SURFACE_FIT_INTERPOLATE`` instead solves for the poles whose surface
# *interpolates* the sampled grid. The separable tensor-product solve below is
# exact to machine precision, costs no extra control points, and so leaves the
# meshed triangle count unchanged.


def _averaged_chord_parameters(grid: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Chord-length parameters along ``axis``, averaged over the other direction.

    A tensor-product surface carries one knot vector per direction, so every
    row has to share a single parameterisation. Averaging the per-row chord
    lengths is the standard construction. Degenerate grids (coincident rings,
    an apex ring collapsed to a point) fall back to a uniform parameterisation,
    which always yields a solvable collocation system.
    """

    count = int(grid.shape[axis])
    uniform = (
        np.arange(count, dtype=np.float64) / float(count - 1)
        if count > 1
        else np.zeros(1, dtype=np.float64)
    )
    if count < 2:
        return uniform

    chords = np.linalg.norm(np.diff(grid, axis=axis), axis=-1)
    mean_chords = chords.mean(axis=1 - axis)
    total = float(mean_chords.sum())
    if not math.isfinite(total) or total <= 0.0:
        return uniform

    params = np.concatenate(([0.0], np.cumsum(mean_chords))) / total
    if not np.all(np.diff(params) > 0.0):
        return uniform
    return params.astype(np.float64, copy=False)


def _knots_and_multiplicities(knots: NDArray[np.float64]) -> tuple[list[float], list[int]]:
    """Split a full B-spline knot vector into gmsh's (distinct, multiplicity) pair."""

    values, counts = np.unique(np.round(np.asarray(knots, dtype=np.float64), 12), return_counts=True)
    return [float(v) for v in values], [int(c) for c in counts]


def interpolating_surface_poles(
    grid: NDArray[np.float64],
    *,
    degree_u: int,
    degree_v: int,
    v_params: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], tuple[list[float], list[int]], tuple[list[float], list[int]]]:
    """Poles and knots of the B-spline surface that interpolates ``grid``.

    ``grid`` is ordered ``(n_v, n_u, 3)`` to match the order the patch builders
    emit point tags in (u fastest). The returned poles carry the same shape, so
    the caller's tag emission order is unchanged.

    ``v_params`` must be supplied whenever a wall is split into several patches
    that meet along v-running seams. A clamped interpolating spline reproduces
    its end data exactly, so neighbouring patches already share the poles of
    their common boundary curve — but they only trace the *same* curve if they
    also share its knot vector. Deriving v from each patch's own columns gives
    each patch a slightly different parameterisation and tears the shell open
    along every seam.
    """

    from scipy.interpolate import make_interp_spline

    u_params = _averaged_chord_parameters(grid, axis=1)
    if v_params is None:
        v_params = _averaged_chord_parameters(grid, axis=0)
    v_params = np.asarray(v_params, dtype=np.float64)

    # Interpolate along u first; scipy rolls the interpolated axis to the front
    # of ``.c``, so this yields (n_u, n_v, 3) and the second pass restores
    # (n_v, n_u, 3).
    along_u = make_interp_spline(u_params, grid, k=degree_u, axis=1)
    along_v = make_interp_spline(v_params, np.asarray(along_u.c), k=degree_v, axis=1)

    poles = np.asarray(along_v.c, dtype=np.float64)
    return poles, _knots_and_multiplicities(along_u.t), _knots_and_multiplicities(along_v.t)


def add_bspline_patch(
    points: NDArray[np.float64],
    column_indices: list[int],
    *,
    degree_v: int,
    surface_fit: str = SURFACE_FIT_APPROXIMATE,
    v_params: NDArray[np.float64] | None = None,
) -> int:
    """Emit one OCC B-spline patch spanning ``column_indices`` of a phi-major grid.

    Point tags are emitted v-major with u fastest, which is the order
    ``addBSplineSurface`` expects for ``numPointsU = len(column_indices)``.
    """

    gmsh = require_gmsh()
    n_u = len(column_indices)
    degree_u = min(3, max(1, n_u - 1))
    grid = np.ascontiguousarray(np.asarray(points)[column_indices, :, :].transpose(1, 0, 2))

    knots: dict[str, list[float] | list[int]] = {}
    if surface_fit == SURFACE_FIT_INTERPOLATE:
        grid, (knots_u, mults_u), (knots_v, mults_v) = interpolating_surface_poles(
            grid, degree_u=degree_u, degree_v=degree_v, v_params=v_params
        )
        knots = {
            "knotsU": knots_u,
            "multiplicitiesU": mults_u,
            "knotsV": knots_v,
            "multiplicitiesV": mults_v,
        }

    point_tags = [
        int(gmsh.model.occ.addPoint(float(x), float(y), float(z)))
        for x, y, z in grid.reshape(-1, 3)
    ]
    return int(
        gmsh.model.occ.addBSplineSurface(
            point_tags,
            n_u,
            degreeU=degree_u,
            degreeV=degree_v,
            **knots,
        )
    )


def throat_boundary_curve(
    points: NDArray[np.float64],
    column_indices: list[int],
) -> tuple[NDArray[np.float64], tuple[list[float], list[int]], int]:
    """Poles and knots of a wall patch's throat (v = 0) boundary curve.

    Gmsh welds the source cap to the wall by *coincident mesh nodes*, not by
    shared topology, so the cap's rim has to be the same curve the wall's
    throat edge is -- not merely another curve through the same ring points.
    Under the approximating fit those two happen to coincide (both are pole
    fits of the same points, at the same degree), which is why re-authoring the
    rim worked at all; under the interpolating fit they differ by the fit's own
    bias and every reduced-domain shell tore open at the source.

    A clamped tensor-product surface's v = 0 isocurve carries its first pole row
    over the surface's own u knot vector, and a clamped interpolating spline
    reproduces its end data, so that row is exactly the u-interpolation of the
    throat ring -- independent of the v parameterisation, and therefore of how
    the caller chose to share v across patches.
    """

    from scipy.interpolate import make_interp_spline

    grid = np.ascontiguousarray(
        np.asarray(points, dtype=np.float64)[column_indices, :, :].transpose(1, 0, 2)
    )
    degree_u = min(3, max(1, len(column_indices) - 1))
    u_params = _averaged_chord_parameters(grid, axis=1)
    curve = make_interp_spline(u_params, grid[0], k=degree_u)
    return (
        np.asarray(curve.c, dtype=np.float64),
        _knots_and_multiplicities(curve.t),
        degree_u,
    )


def grid_v_parameters(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Shared v-parameterisation for every patch cut from one phi-major grid."""

    grid = np.ascontiguousarray(np.asarray(points).transpose(1, 0, 2))
    return _averaged_chord_parameters(grid, axis=0)
