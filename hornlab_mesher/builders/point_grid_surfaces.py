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
        self.grids: dict[str, np.ndarray] = {}
        self.points: dict[tuple[str, int, int], int] = {}
        self.line_cache: dict[tuple[int, int], int] = {}
        self.spline_cache: dict[tuple[int, ...], int] = {}

    def add_point(self, xyz: np.ndarray | tuple[float, float, float]) -> int:
        x, y, z = xyz
        return int(self.gmsh.model.occ.addPoint(float(x), float(y), float(z)))

    def add_grid(self, name: str, points: np.ndarray) -> None:
        self.grids[name] = np.asarray(points, dtype=np.float64).copy()

    def point(self, name: str, i: int, j: int) -> int:
        key = (name, int(i), int(j))
        tag = self.points.get(key)
        if tag is not None:
            return tag
        grid = self.grids[name]
        tag = self.add_point(grid[int(i), int(j)])
        self.points[key] = tag
        return tag

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

    def circle_arc(self, start: int, center: int, end: int) -> int:
        return int(self.gmsh.model.occ.addCircleArc(int(start), int(center), int(end), center=True))

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


def _snap_open_symmetry_grid(
    points: np.ndarray,
    *,
    closed: bool,
    symmetry_planes: tuple[str, ...] = ("x", "y"),
) -> np.ndarray:
    """Snap the two open rim rays exactly onto the model's symmetry plane(s).

    The boundary rays of an open (non-full-circle) grid must lie precisely on
    the cut plane(s) so the BEM image solve sees a clean reflection edge:

    * quarter model (planes ``("x", "y")``): the first ray runs along +x on the
      xz / y=0 plane and the last along +y on the yz / x=0 plane, so each rim
      snaps to a *different* plane;
    * half model about xz (planes ``("y",)``, quadrants 12): both rim rays lie
      on y=0;
    * half model about yz (planes ``("x",)``, quadrants 14): both rim rays lie
      on x=0.
    """
    out = np.array(points, dtype=np.float64, copy=True)
    if closed or out.shape[0] < 2:
        return out
    axis_index = {"x": 0, "y": 1, "z": 2}
    if len(symmetry_planes) >= 2:
        # Quarter: rim rays land on two different planes (first on xz / y=0,
        # last on yz / x=0).
        out[0, :, 1] = 0.0
        out[-1, :, 0] = 0.0
    elif len(symmetry_planes) == 1:
        # Half: both rim rays share the single cut plane.
        idx = axis_index.get(symmetry_planes[0])
        if idx is not None:
            out[0, :, idx] = 0.0
            out[-1, :, idx] = 0.0
    return out



def _add_mouth_rim_surfaces(
    builder: _SharedSurfaceBuilder,
    *,
    n_phi: int,
    n_len: int,
    closed: bool,
    outer_len: int | None = None,
) -> list[tuple[int, int]]:
    ji = n_len - 1
    jo = (outer_len if outer_len is not None else n_len) - 1
    surfaces: list[tuple[int, int]] = []
    for i in _phi_segments(n_phi, closed=closed):
        ni = (i + 1) % n_phi
        surfaces.append(
            builder.quad(
                ("inner", i, ji),
                ("inner", ni, ji),
                ("outer", ni, jo),
                ("outer", i, jo),
            )
        )
    return surfaces


def _rear_rim_points(
    outer_points: np.ndarray,
    *,
    rear_z: float,
) -> np.ndarray:
    """Rear rim ring: the outer throat ring projected straight back to ``rear_z``.

    Straight axial projection (constant x/y) keeps the rear cover inside the
    reduced-domain quadrant — cut-plane rays stay exactly on their plane and the
    seam boundary is a straight axial line gmsh cannot chord into the symmetry
    plane. It also matches ATH's rear cover, which stays at the outer throat
    radius (an earlier version extrapolated backwards along the flaring wall
    rays, which both flared ~10 mm past ATH's rear cover and dipped below the
    y=0 cut plane near the seam).
    """
    out = outer_points[:, 0, :].copy()
    out[:, 2] = rear_z
    return out


def _add_rear_cap(
    builder: _SharedSurfaceBuilder,
    rear_points: np.ndarray,
    *,
    grid_name: str,
    n_phi: int,
    closed: bool,
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
        )
    )
    radial_lines = {
        i: builder.line_tags(builder.point(grid_name, i, 0), center_tag)
        for i in range(n_phi)
    }

    cap: list[tuple[int, int]] = []
    cap_boundary: list[int] = []
    for i in _phi_segments(n_phi, closed=closed):
        ni = (i + 1) % n_phi
        cap_boundary.append(builder.line((grid_name, i, 0), (grid_name, ni, 0)))
        if closed:
            cap.append(builder.surface([cap_boundary[-1], radial_lines[ni], -radial_lines[i]]))
    if not closed and cap_boundary:
        cap.append(builder.surface([*cap_boundary, radial_lines[n_phi - 1], -radial_lines[0]]))
    return cap


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


def _bspline_patch_phi_groups(
    n_phi: int, *, closed: bool, n_sectors: int = 1
) -> list[list[int]]:
    if not closed:
        # A quarter grid is one quadrant -> a single patch. A half grid spans
        # two quadrants that meet on an interior symmetry axis; split the rim
        # into ``n_sectors`` patches there (the rim is sampled symmetrically, so
        # the axis crossings land on evenly spaced indices). Consecutive spans
        # share their boundary column so the patches weld into a watertight seam,
        # and each patch contributes one mouth curve the enclosure can attach a
        # sector to.
        n_sectors = max(1, int(n_sectors))
        if n_sectors <= 1 or n_phi < n_sectors + 1:
            return [list(range(n_phi))]
        edges = [round(s * (n_phi - 1) / n_sectors) for s in range(n_sectors + 1)]
        return [list(range(edges[s], edges[s + 1] + 1)) for s in range(n_sectors)]
    if n_phi < 4:
        return [list(range(n_phi)) + [0]]
    span_count = 4 if n_phi % 4 == 0 else 1
    if span_count == 1:
        return [list(range(n_phi)) + [0]]
    step = n_phi // span_count
    spans: list[list[int]] = []
    for span in range(span_count):
        start = span * step
        stop = (span + 1) * step
        if span == span_count - 1:
            spans.append(list(range(start, n_phi)) + [0])
        else:
            spans.append(list(range(start, stop + 1)))
    return spans


def _add_occ_bspline_patch_wall_surfaces(
    points: np.ndarray,
    *,
    closed: bool,
    phi_groups: list[list[int]] | None = None,
) -> list[tuple[int, int]]:
    """Build enclosure-mode horn walls as large OCC BSpline patches.

    ``phi_groups`` overrides the default angular patch partition (used to split
    an open half-model wall into one patch per quadrant so the rear enclosure
    can attach a sector to each).
    """

    gmsh = require_gmsh()
    arr = _validated_grid(points, name="inner_points")
    n_phi, n_len, _ = arr.shape
    surfaces: list[tuple[int, int]] = []
    degree_v = min(3, max(1, n_len - 1))
    groups = phi_groups if phi_groups is not None else _bspline_patch_phi_groups(n_phi, closed=closed)
    for indices in groups:
        n_u = len(indices)
        degree_u = min(3, max(1, n_u - 1))
        point_tags: list[int] = []
        for j in range(n_len):
            for i in indices:
                x, y, z = arr[i, j]
                point_tags.append(
                    int(gmsh.model.occ.addPoint(float(x), float(y), float(z)))
                )
        surf = int(
            gmsh.model.occ.addBSplineSurface(
                point_tags,
                n_u,
                degreeU=degree_u,
                degreeV=degree_v,
            )
        )
        surfaces.append((2, surf))
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
