"""FREEFORM wall patches, angular fitting, mitres and boundary measurements."""
from __future__ import annotations

from .errors import control_refusal

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares, brentq
from scipy.interpolate import BSpline

from .freeform import FreeformGeometry, freeform_corner_z, offset_needs_envelope, validate_outer_offset_grid
from .geometry import BuiltGeometry


@dataclass(frozen=True)
class CornerWall:
    profile: FreeformGeometry
    scale: float = 1.0

    def cuts(self, points: np.ndarray) -> tuple[float, ...]:
        origin = float(self.profile._profile_h.anchors[0, 0])
        return tuple(float(points[0, 0, 2]) + (z - origin) * self.scale
                     for z in freeform_corner_z(self.profile, interior=True))

    def radius(self, phi: Any, z: float, points: np.ndarray) -> np.ndarray:
        t = (z - points[0, 0, 2]) / (self.profile.length_mm * self.scale)
        return self.scale * self.profile.cross_section_radius(phi, float(np.clip(t, 0, 1)))



def corner_geometry_class(base: type, recipe: CornerWall | None = None) -> type:
    """Keep the immutable build recipe outside the inherited dataclass fields.

    A class property survives dataclasses.replace without entering asdict (the
    profile contains deliberately immutable mapping proxies).
    """
    def __post_init__(self):
        if recipe is not None and freeform_corner_z(recipe.profile):
            _refuse_ignored_controls(self, recipe)
        base.__post_init__(self)

    name = f"Corner{base.__name__}" if recipe is None or freeform_corner_z(recipe.profile) else base.__name__
    return type(name, (base,), {"freeform_wall": property(lambda self: recipe),
                                                    "__post_init__": __post_init__})


def _refuse_ignored_controls(geometry, recipe: CornerWall) -> None:
    """Refuse explicit controls the split-patch corner builder would ignore.

    Runs before the base resolves ``surface_fit``: ``auto`` and ``approximate``
    describe what the corner builder does, ``interpolate`` does not.
    """
    ignored = []
    if geometry.surface_fit == "interpolate":
        ignored.append("mesh.surface_fit='interpolate'")
    if geometry.topology_mode == "legacy":
        ignored.append("mesh.topology='legacy'")
    if geometry.preserve_grid:
        ignored.append("mesh.preserve_grid")
    if getattr(geometry, "outer_clearance_points_mm", None) is not None:
        ignored.append("outer_clearance_points_mm")
    if ignored:
        corners = ", ".join(f"{z:g}" for z in freeform_corner_z(recipe.profile))
        raise ValueError(
            f"FREEFORM corner z={corners} mm: corner walls always build approximating "
            f"split patches and do not support {', '.join(ignored)}; "
            "remove the setting or make the corners smooth")


@dataclass
class _BuiltCornerGeometry(BuiltGeometry):
    freeform_wall_points_mm: np.ndarray | None = None
    freeform_wall_recipe: CornerWall | None = None
    freeform_strip_widths_mm: dict[str, float] | None = None
    freeform_wall_segments: dict[str, list[tuple[int, int]]] | None = None
    freeform_wall_groups: list[list[int]] | None = None


def cut_indices(points: np.ndarray, cuts: tuple[float, ...]) -> list[int]:
    """Resolve semantic z identities only after the current grid is finalized."""
    z = points[0, :, 2]
    indices = [0]
    for cut in cuts:
        j = int(np.argmin(np.abs(z - cut)))
        if abs(z[j] - cut) > 1e-7 or not 0 < j < len(z) - 1:
            raise ValueError(f"FREEFORM corner z={cut:g} mm is missing from the axial grid")
        indices.append(j)
    return [*indices, len(z) - 1]


def mitred_outer_grid(inner: np.ndarray, wall: float, *, full_circle: bool,
                      corner_z: tuple[float, ...]) -> np.ndarray:
    """The outer grid of :func:`mitred_outer_offset`."""
    return mitred_outer_offset(inner, wall, full_circle=full_circle, corner_z=corner_z)[0]


def mitred_outer_offset(inner: np.ndarray, wall: float, *, full_circle: bool,
                        corner_z: tuple[float, ...], force_envelope: bool = False,
                        allowance: float = 0.0) -> tuple[np.ndarray, bool]:
    """Offset each smooth span, intersecting its two offset tangent planes.

    At a circular meridian this is exactly the intersection of offset lines.
    The vector formula also handles unequal H/V: dot(delta, n_left/right)=wall.
    The shared ring is authored once, so both outer patches have identical poles.
    Returns ``(outer, enveloped)``; a mitre that folds is replaced by the shared
    exterior envelope where one can be built, and refused otherwise.
    """
    from .profile_sampling import _outer_offset_shell

    ends = cut_indices(inner, corner_z)
    outer = np.empty_like(inner)
    sides: dict[int, list[np.ndarray]] = {}
    for a, b in zip(ends[:-1], ends[1:]):
        span = inner[:, a:b + 1]
        offset = _outer_offset_shell(span, wall, full_circle=full_circle,
                                     t_coordinates=span[0, :, 2],
                                     phi_coordinates=np.unwrap(np.arctan2(span[..., 1], span[..., 0]), axis=0).T)
        outer[:, a:b + 1] = offset
        for j, local in ((a, 0), (b, -1)):
            if j in ends[1:-1]:
                sides.setdefault(j, []).append((offset[:, local] - span[:, local]) / wall)
    for j, normals in sides.items():
        left, right = normals
        denominator = 1 + np.einsum('ij,ij->i', left, right)
        cut = float(inner[0, j, 2])
        if np.any(denominator < 0.08):
            raise ValueError(f"FREEFORM corner z={cut:g} mm: unsafe offset mitre (opposing normals)")
        delta = wall * (left + right) / denominator[:, None]
        outer[:, j] = inner[:, j] + delta
    # A concave mitre consumes a finite part of each offset span. Samples
    # inside that swallowtail are not exterior boundary points. Trim them,
    # then resample the retained polyline to preserve semantic cut indices.
    for a, b in zip(ends[:-1], ends[1:]):
        if b - a < 2:
            continue
        start_tangent = inner[:, a + 1] - inner[:, a]
        end_tangent = inner[:, b] - inner[:, b - 1]
        rows = outer[:, a + 1:b]
        forward = np.einsum('ijk,ik->ij', rows - outer[:, a, None], start_tangent)
        backward = np.einsum('ijk,ik->ij', outer[:, b, None] - rows, end_tangent)
        keep = np.r_[True, np.all((forward > 0) & (backward > 0), axis=0), True]
        if np.all(keep):
            continue
        for i in range(len(outer)):
            retained = outer[i, a:b + 1][keep]
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(retained, axis=0), axis=1))]
            stations = np.linspace(0., arc[-1], b - a + 1)
            outer[i, a:b + 1] = np.column_stack([
                np.interp(stations, arc, retained[:, axis]) for axis in range(3)])
    from .offset_envelope import exterior_outer_offset
    candidate, enveloped = exterior_outer_offset(
        inner, outer, wall, full_circle=full_circle, force=force_envelope,
        allowance=allowance)
    if not enveloped:
        try:
            validate_outer_offset_grid(inner, outer, full_circle=full_circle)
        except ValueError as exc:
            label = ', '.join(f'{z:g}' for z in corner_z) or 'endpoints'
            raise control_refusal(f"FREEFORM corner z={label} mm: unsafe offset mitre: {exc}", "Wall thickness", "Reduce Wall thickness or soften the sharp profile corner.", message=f"The outer wall folds at the profile corner at {label} mm.") from exc
    return candidate, enveloped


def _refine(points: np.ndarray, recipe: CornerWall, intervals: set[int] | None = None) -> np.ndarray:
    """Bisect axial ring intervals: all of them, or only the given ones."""
    z = points[0, :, 2]
    selected = np.arange(len(z) - 1) if intervals is None else np.asarray(sorted(intervals), dtype=int)
    refined_z = np.sort(np.r_[z, (z[selected + 1] + z[selected]) / 2])
    phi = np.unwrap(np.arctan2(points[..., 1], points[..., 0]), axis=0)
    angles = np.array([np.interp(refined_z, z, row) for row in phi])
    out = np.empty((len(points), len(refined_z), 3))
    for j, zj in enumerate(refined_z):
        r = recipe.radius(angles[:, j], float(zj), points)
        out[:, j, 0] = r * np.cos(angles[:, j])
        out[:, j, 1] = r * np.sin(angles[:, j])
        out[:, j, 2] = zj
    # Preserve existing rings bit-for-bit, including their angular coordinates.
    out[:, np.searchsorted(refined_z, z)] = points
    return out


def _refine_angular(points: np.ndarray, recipe: CornerWall, closed: bool,
                    intervals: set[int]) -> np.ndarray:
    """Bisect selected angular intervals, preserving the original moving rows."""
    angles = np.unwrap(np.arctan2(points[..., 1], points[..., 0]), axis=0)
    if closed:
        angles = np.r_[angles, angles[:1] + math.tau]
    locations = np.sort(np.r_[np.arange(len(angles)), np.asarray(sorted(intervals)) + .5])
    phi = np.array([np.interp(locations, np.arange(len(angles)), row) for row in angles.T]).T
    refined = np.empty((*phi.shape, 3))
    for j, z in enumerate(points[0, :, 2]):
        r = recipe.radius(phi[:, j], float(z), points)
        refined[:, j] = np.c_[r * np.cos(phi[:, j]), r * np.sin(phi[:, j]), np.full(len(phi), z)]
    if closed:
        refined = refined[:-1]
    refined[np.searchsorted(locations, np.arange(len(points)))] = points
    return refined


def _angular_fit_error(points, recipe, groups):
    """Test each angular interpolant before creating any disposable OCC faces.

    Probe every data interval, including sharp morph vertices and moving
    rounded-rectangle tangencies. Mirror failing interval identities across
    sectors so cardinal boundaries stay at the same semantic grid indices.
    """
    from .builders._occ import _patch_boundary_spline
    fitting_groups = [columns.copy() for columns in groups]
    poles = _angular_poles(points, fitting_groups, recipe, refine=False)
    sections, tangents, parameters = [], [], []
    worst = 0.
    data_parameters = []
    for columns, fitted in zip(groups, fitting_groups):
        # Both angular fits place one unique knot at each data sample.
        values = np.asarray(poles.angular_knots[tuple(fitted)][0], dtype=float)
        quarters = np.interp(np.arange(4 * (len(values) - 1) + 1) / 4, np.arange(len(values)), values)
        u = np.unique(np.r_[np.linspace(0, 1, 129), quarters])
        curve = _patch_boundary_spline(poles, poles[fitted], fitted, -1, 'approximate')
        sections.append(curve(u * curve.t[-1]).transpose(1, 0, 2))
        tangents.append(curve.derivative()(u * curve.t[-1]).transpose(1, 0, 2))
        parameters.append(u)
        data_parameters.append(values)
    bad_local = set()
    for j, z in enumerate(points[0, :, 2]):
        xyz = np.concatenate([rows[j] for rows in sections])
        expected = recipe.radius(np.arctan2(xyz[:, 1], xyz[:, 0]), float(z), points)
        delta = abs(np.linalg.norm(xyz[:, :2], axis=1) - expected)
        tangent = np.concatenate([rows[j] for rows in tangents])
        polar_speed = xyz[:, 0] * tangent[:, 1] - xyz[:, 1] * tangent[:, 0]
        backwards = polar_speed < -1e-12 * np.maximum(1., np.sum(xyz[:, :2] ** 2, axis=1))
        worst = max(worst, float(np.max(delta)))
        offset = 0
        for columns, u, values in zip(groups, parameters, data_parameters):
            selected = slice(offset, offset + len(u))
            bad = u[(delta[selected] > WALL_ANGULAR_TOLERANCE_MM) | backwards[selected]]
            indices = np.clip(np.searchsorted(values, bad, side='right') - 1, 0, len(columns) - 2)
            bad_local.update(map(int, indices))
            bad_local.update(len(columns) - 2 - int(index) for index in indices)
            offset += len(u)
    # A C2 cubic interpolant couples neighboring intervals. Refine its support
    # around a failing interval too; repeatedly splitting only the peak can
    # leave the same bulge driven by an unchanged, much longer neighbor.
    # An authored Hermite piece depends on its two samples, and on their
    # neighbours only through the angular speed estimate; every round
    # re-checks all intervals, so it refines just the failing ones.
    interval_count = len(groups[0]) - 1
    reach = 0 if poles.hermite else 2
    bad_local = {neighbor for index in bad_local
                 for neighbor in range(max(0, index - reach), min(interval_count, index + reach + 1))}
    intervals = {columns[index] for columns in groups for index in bad_local}
    return worst, intervals


def _revolution_grid(points):
    """A fixed, uniform azimuth net on circular rings can use the cubic fit."""
    from .builders.point_grid_freestanding import _grid_is_axisymmetric
    if not _grid_is_axisymmetric(points):
        return False
    phi = np.unwrap(np.arctan2(points[..., 1], points[..., 0]), axis=0)
    spacing = np.diff(phi[:, 0])
    return bool(np.max(abs(phi - phi[:, :1])) < 1e-10
                and np.ptp(spacing) < 1e-10)


def _fit_angular_grid(points, recipe, groups, *, budget=None):
    """One angular refinement owner for walls, bounds and flush previews."""
    from .builders.point_grid_surfaces import _bspline_patch_phi_groups
    if budget is None:
        budget = MAX_WALL_FIT_REFINEMENTS
    closed = groups[-1][-1] == groups[0][0]
    sectors = len(groups)
    for count in range(budget + 1):
        error, intervals = _angular_fit_error(points, recipe, groups)
        if not intervals:
            return points, groups, count
        if _revolution_grid(points):
            # Keep its uniform angular basis while refining the source hull.
            intervals = {index for columns in groups for index in columns[:-1]}
        if count == budget or len(points) + len(intervals) > MAX_WALL_PROFILES:
            cause = (f"sampled angular error {error:g} mm exceeds {WALL_ANGULAR_TOLERANCE_MM:g} mm"
                     if error > WALL_ANGULAR_TOLERANCE_MM else "angular fit folds back")
            raise ValueError(f"FREEFORM corner wall {cause} at geometry refinement cap")
        points = _refine_angular(points, recipe, closed, intervals)
        groups = _bspline_patch_phi_groups(len(points), closed=closed, n_sectors=sectors)


def _axial_ring_parameters(points, groups):
    """Averaged chord-length ring parameters, shared by every sector.

    Parameters proportional to z cannot follow a meridian whose tangent turns
    radial (a mouth angle near 90 degrees): r(z) there is not polynomial and
    the interpolant converges only with the square root of the ring spacing.
    Arc-length parameters keep both z and r smooth along the meridian.
    """
    from .builders._occ import grid_v_parameters
    return grid_v_parameters(points, closed=groups[-1][-1] == groups[0][0])


def wall_probe_error(recipe, phi, z, radius, points):
    """Wall error, read radially except where the wall turns nearly radial.

    The budget's fixed-z radial probe reads sqrt(1 + slope^2) times the
    distance to the authored meridian. Up to RADIAL_PROBE_MAX_WALL_ANGLE_DEG
    that is the established measure. A wall approaching 90 degrees (a flared
    mouth) makes the probe run along the wall: a micron of normal error reads
    as a large radial one, and no ring spacing can meet the budget. Beyond that
    angle the radial error is scaled to the axial probe at the same limit
    angle (radial error times tan(limit) over the slope, to first order). The
    reading is therefore never below the true distance and at most
    1 / cos(limit) times it.
    """
    low, high = float(points[0, 0, 2]), float(points[0, -1, 2])
    step = 1e-3
    below, above = max(low, z - step), min(high, z + step)
    slope = (recipe.radius(phi, above, points) - recipe.radius(phi, below, points)) / (above - below)
    limit = math.tan(math.radians(RADIAL_PROBE_MAX_WALL_ANGLE_DEG))
    return abs(radius - recipe.radius(phi, z, points)) / np.maximum(1., abs(slope) / limit)


def _wall_fit_error(points, recipe, patches, groups):
    """Sample wall probe error on the actual patches, mesh-independently.

    Group sectors at each axial station so the analytic H/V evaluation is
    shared. Smooth walls also return the ring intervals holding failing stations, so
    refinement adds rings only where the interpolant needs them.
    """
    from .builders._occ import require_gmsh
    gmsh = require_gmsh()
    worst = 0.
    smooth = not freeform_corner_z(recipe.profile)
    ring_v = _axial_ring_parameters(points, groups) if smooth else None
    failing = set()
    spans = sorted({span for span, _, _ in patches})
    for span in spans:
        rows = []
        stations = np.linspace(0., 1., 17)
        if smooth:
            stations = np.unique(np.r_[np.linspace(0., 1., 65), ring_v,
                                       (ring_v[1:] + ring_v[:-1]) / 2])
        for _, _, face in (patch for patch in patches if patch[0] == span):
            lo, hi = map(np.asarray, gmsh.model.getParametrizationBounds(*face))
            count = len(stations)
            u, v = np.meshgrid(np.linspace(lo[0], hi[0], 129), lo[1] + stations * (hi[1] - lo[1]))
            rows.append(np.asarray(gmsh.model.getValue(*face, np.c_[u.ravel(), v.ravel()].ravel())).reshape(count, 129, 3))
        xyz = np.concatenate(rows, axis=1)
        for j, row in enumerate(xyz):
            error = float(np.max(wall_probe_error(
                recipe, np.arctan2(row[:, 1], row[:, 0]), float(row[0, 2]),
                np.linalg.norm(row[:, :2], axis=1), points)))
            worst = max(worst, error)
            if smooth and error > WALL_FIT_TOLERANCE_MM:
                failing.add(int(np.clip(np.searchsorted(ring_v, stations[j], side='right') - 1,
                                        0, len(ring_v) - 2)))
    # The axial interpolant is a cubic, coupling each interval to its neighbors.
    intervals = {neighbor for index in failing
                 for neighbor in range(max(0, index - 1), min(points.shape[1] - 1, index + 2))}
    return worst, intervals


def _end_derivative(points, j: int, sign: int, recipe: CornerWall,
                    columns: list[int], reference: np.ndarray) -> np.ndarray:
    """Radial derivative poles on the same angular Hermite basis as the wall.

    Compose the authored radial slope with the built boundary's azimuth and
    angular speed. This avoids a dense least-squares solve at every crease.
    """
    from .builders._occ import _angular_hermite_poles
    values, multiplicities = points.angular_knots[tuple(columns)]
    degree = multiplicities[0] - 1
    curve = BSpline(np.repeat(values, multiplicities), points[columns, j], degree)
    knots = np.repeat(values, multiplicities)
    u = (np.asarray(values) if points.hermite else
         np.array([np.mean(knots[i + 1:i + degree + 1]) for i in range(len(columns))]))
    boundary = curve(u)
    tangent = curve.derivative()(u)
    radius = np.linalg.norm(boundary[:, :2], axis=1)
    phi = np.arctan2(boundary[:, 1], boundary[:, 0])
    angular_speed = (boundary[:, 0] * tangent[:, 1] - boundary[:, 1] * tangent[:, 0]) / radius ** 2
    z = float(reference[0, j, 2])
    h = min(1e-4, abs(float(reference[0, j + sign, 2]) - z) / 1000)

    def derivative(angles):
        slope = (recipe.radius(angles, z + sign * 2 * h, reference)
                 - recipe.radius(angles, z + sign * h, reference)) / (sign * h)
        return np.c_[np.cos(angles) * slope, np.sin(angles) * slope, np.ones(len(angles))]

    data = derivative(phi)
    if not points.hermite:
        # Greville collocation uses the cubic's existing knots. Its banded
        # system avoids a dense solve even on a refined circular boundary.
        from scipy.linalg import solve_banded
        matrix = BSpline.design_matrix(u, knots, degree).toarray()
        row, column = np.nonzero(matrix)
        band = np.zeros((2 * degree + 1, len(columns)))
        band[degree + row - column, column] = matrix[row, column]
        return solve_banded((degree, degree), band, data)
    delta = 1e-5
    angular = (derivative(phi + delta) - derivative(phi - delta)) / (2 * delta)
    poles, _ = _angular_hermite_poles(data[None], (angular * angular_speed[:, None])[None], u)
    return poles[0]


def _azimuth_parameters(grid):
    """Angular sample parameters proportional to the mean azimuth of each row.

    Every Hermite net of one wall (inner, offset shell, rear return) uses the
    parameters of the wall's own rows. Shared parameters keep the ruled mouth
    and rear strips joining corresponding rows; a uniform index parameter on a
    locally refined net would make the shell's speed vary a thousandfold and
    skew those rulings into slivers.
    """
    mean = np.unwrap(np.arctan2(grid[..., 1], grid[..., 0]), axis=1).mean(axis=0)
    return (mean - mean[0]) / (mean[-1] - mean[0])


def _authored_hermite_poles(grid, recipe: CornerWall, reference: np.ndarray):
    """Angular Hermite poles carrying the authored cross-section's own tangents.

    Estimated (PCHIP) tangents are only low-order accurate, so the angular net
    had to be refined several times to meet its budget, and every OCC
    operation downstream paid for the larger control net. The authored radius
    supplies exact tangents. Parameters follow the mean azimuth, so each
    piece's tangent scale matches its own span on a locally refined net. Where
    the one-sided tangents disagree (a sharp cross-section vertex) the sample
    gets zero speed, which keeps the vertex sharp, as the shape-preserving
    fit does.
    """
    from .builders._occ import _angular_hermite_poles
    phi = np.unwrap(np.arctan2(grid[..., 1], grid[..., 0]), axis=1)
    parameters = _azimuth_parameters(grid)
    speed = np.gradient(phi, parameters, axis=1, edge_order=2 if len(parameters) > 2 else 1)
    chord = np.diff(phi, axis=1) / np.diff(parameters)
    # The second-order end formula can turn back on a strongly graded net.
    speed[:, 0] = np.where(speed[:, 0] > 0, speed[:, 0], chord[:, 0])
    speed[:, -1] = np.where(speed[:, -1] > 0, speed[:, -1], chord[:, -1])
    step = 1e-6
    derivative = np.zeros_like(grid)
    for j, angles in enumerate(phi):
        # One outline evaluation per ring for all three stencil azimuths.
        values = np.r_[angles - step, angles, angles + step]
        r = recipe.radius(values, float(grid[j, 0, 2]), reference)
        lower, centre, upper = np.split(np.c_[r * np.cos(values), r * np.sin(values)], 3)
        before = (centre - lower) / step
        after = (upper - centre) / step
        tangent = (before + after) / 2
        vertex = np.linalg.norm(after - before, axis=1) > 1e-3 * np.linalg.norm(tangent, axis=1)
        tangent[vertex] = 0.
        derivative[j, :, :2] = tangent * speed[j, :, None]
    return _angular_hermite_poles(grid, derivative, parameters)


def _angular_poles(points: np.ndarray, groups: list[list[int]], recipe: CornerWall | None,
                   *, refine: bool = True, hermite: bool | None = None,
                   wall: np.ndarray | None = None) -> Any:
    """Interpolate angular samples and retain their fitted knot vectors.

    Rewrite the caller's sector indices to address the returned control net;
    boundary consumers retain its explicit knots alongside its poles. Separate
    copies of shared endpoints remain identical, ready for the existing sewing.
    Axial rows remain approximating; interpolation is only angular.
    """
    from .builders._occ import angular_interpolating_surface_poles, _AngularPoleGrid

    if recipe is not None and refine:
        # Bounds and flush previews consume the same accepted fit as the wall.
        points, accepted_groups, _ = _fit_angular_grid(points, recipe, groups)
        groups[:] = accepted_groups
    result = []
    hermite = not _revolution_grid(points) if hermite is None else hermite
    angular_knots = {}
    offset = 0
    for sector, columns in enumerate(groups):
        grid = points[columns].transpose(1, 0, 2)
        if recipe is not None and hermite:
            poles, knots = _authored_hermite_poles(grid, recipe, points)
        else:
            # A shell without an authored outline keeps estimated tangents,
            # on the angular parameters of the wall rows it was offset from.
            parameters = (_azimuth_parameters((points if wall is None else wall)[columns].transpose(1, 0, 2))
                          if hermite else None)
            poles, knots = angular_interpolating_surface_poles(
                grid, degree_u=min(3, len(columns) - 1), shape_preserving=hermite, circular=not hermite,
                parameters=parameters)
        poles = poles.transpose(1, 0, 2)
        if recipe is not None and hermite:
            radius = np.linalg.norm(points[columns, 0, :2], axis=1)
            if np.ptp(radius) < 1e-9:
                # The circular source must stay inside its authored disk.
                # Interpolation can overshoot it by microns on a moving or
                # locally refined angle net. An inscribed pole hull preserves
                # the source guard; the same radial budget measures its bias.
                pole_radius = np.linalg.norm(poles[:, 0, :2], axis=1)
                poles[:, 0, :2] *= np.minimum(1., radius[0] / pole_radius)[:, None]
        poles[0], poles[-1] = points[columns[0]], points[columns[-1]]
        result.append(poles)
        groups[sector] = list(range(offset, offset + len(poles)))
        angular_knots[tuple(groups[sector])] = knots
        offset += len(poles)
    return _AngularPoleGrid(np.concatenate(result), angular_knots, hermite)


def _patches(points: np.ndarray, cuts: tuple[float, ...], groups: list[list[int]],
             recipe: CornerWall | None = None, identities: list[int] | None = None,
             hermite: bool | None = None, interpolate_axial: bool = False,
             wall: np.ndarray | None = None):
    from .builders._occ import add_bspline_patch, _AngularPoleGrid, axial_interpolating_surface_poles

    ends = cut_indices(points, cuts) if identities is None else identities
    fitted_groups = [columns.copy() for columns in groups]
    fitting_points = _angular_poles(points, fitted_groups, recipe, refine=False, hermite=hermite, wall=wall)
    patches = []
    for span, (a, b) in enumerate(zip(ends[:-1], ends[1:])):
        poles = np.asarray(fitting_points)[:, a:b + 1].copy()
        axial_knots = None
        if interpolate_axial or (recipe is not None and not freeform_corner_z(recipe.profile)):
            # Smooth profiles have no semantic axial creases. Interpolate
            # their accepted rings using the existing axial interpolation
            # pass, rather than shrinking an approximating pole hull.
            # All sectors share this axial basis, just as their seams do.
            v = _axial_ring_parameters(points[:, a:b + 1], groups)
            axial, axial_knots = axial_interpolating_surface_poles(poles, v, min(3, b - a))
            poles = axial.transpose(1, 0, 2)
        for sector, columns in enumerate(fitted_groups):
            local = poles.copy()
            if recipe is not None and b - a >= 3:
                if a > 0:
                    local[columns, 1] = local[columns, 0] + (poles[0, 1, 2] - poles[0, 0, 2]) * _end_derivative(fitting_points, a, 1, recipe, columns, points)
                if b < points.shape[1] - 1:
                    local[columns, -2] = local[columns, -1] - (poles[0, -1, 2] - poles[0, -2, 2]) * _end_derivative(fitting_points, b, -1, recipe, columns, points)
            net = _AngularPoleGrid(local, fitting_points.angular_knots, fitting_points.hermite, axial_knots)
            tag = add_bspline_patch(net, columns, degree_v=min(3, b - a))
            patches.append((span, sector, (2, tag)))
    return patches


def _sew_roles(roles: dict[str, list[tuple[int, int]]]) -> dict[str, list[tuple[int, int]]]:
    """Sew coincident boundaries without intersecting whole B-spline faces.

    Sewing retains the source surfaces, replacing their wires and sometimes
    their UV trim bounds. Gmsh does not expose OCC's Modified() ancestry;
    recover the one-to-one relation at unchanged UV coordinates, and refuse
    ambiguous or incomplete ancestry.
    """
    from .builders._occ import require_gmsh

    gmsh = require_gmsh()
    faces = list(dict.fromkeys(face for values in roles.values() for face in values))
    if len(faces) < 2:
        return roles
    gmsh.model.occ.synchronize()

    def parameters(face):
        lo, hi = map(np.asarray, gmsh.model.getParametrizationBounds(*face))
        return np.array([lo + fraction * (hi - lo) for fraction in (.23, .5, .77)])

    def signature(face, uv):
        return tuple(np.round(gmsh.model.getValue(*face, uv.ravel()), 7))

    source_uv = {face: parameters(face) for face in faces}
    source = {face: signature(face, uv) for face, uv in source_uv.items()}
    source_keys = set(source.values())
    source_bounds = {face: tuple(map(np.asarray, gmsh.model.getParametrizationBounds(*face)))
                     for face in faces}
    source_types = {face: gmsh.model.getType(*face) for face in faces}
    before = set(gmsh.model.occ.getEntities(2))
    gmsh.model.occ.addSurfaceLoop([tag for _, tag in faces], sewing=True)
    gmsh.model.occ.synchronize()
    children = set(gmsh.model.occ.getEntities(2)) - before
    by_signature = {}
    for face in children:
        uv = parameters(face)
        key = signature(face, uv)
        by_signature.setdefault(key, []).append(face)
        if key not in source_keys:
            # OCC sewing can shrink a roundover's UV trim bounds without
            # changing its supporting surface. Compare at the source UVs,
            # rather than moving the probes with the new bounds. Require
            # mutual interior overlap to distinguish disjoint trims on the
            # same supporting surface; keep the original signature precision.
            lo, hi = map(np.asarray, gmsh.model.getParametrizationBounds(*face))
            for parent, parent_uv in source_uv.items():
                parent_lo, parent_hi = source_bounds[parent]
                if (gmsh.model.getType(*face) == source_types[parent]
                        and np.all((parent_uv >= lo) & (parent_uv <= hi))
                        and np.all((uv >= parent_lo) & (uv <= parent_hi))
                        and gmsh.model.isInside(*face, parent_uv.ravel(), parametric=True) == len(parent_uv)
                        and gmsh.model.isInside(*parent, uv.ravel(), parametric=True) == len(uv)
                        and signature(face, parent_uv) == source[parent]
                        and signature(parent, uv) == key):
                    by_signature.setdefault(source[parent], []).append(face)
    mapping = {}
    for face, key in source.items():
        candidates = by_signature.get(key, [])
        if not candidates:
            # Faces whose boundaries already have shared topology survive.
            mapping[face] = face
        elif len(candidates) == 1:
            mapping[face] = candidates[0]
        else:
            raise ValueError("FREEFORM corner sewing has ambiguous face ancestry")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("FREEFORM corner sewing has ambiguous face ancestry")
    if set(mapping.values()) - set(faces) != children:
        raise ValueError("FREEFORM corner sewing lost face ancestry")
    obsolete = [face for face in faces if face not in mapping.values()]
    gmsh.model.occ.remove(obsolete, recursive=False)
    gmsh.model.occ.synchronize()
    return {role: list(dict.fromkeys(mapping[face] for face in values))
            for role, values in roles.items()}


def _crease_edges(left, right, wall_tags):
    """Select a crease by adjacent span ancestry, never tolerance-inflated boxes."""
    from .builders._occ import require_gmsh
    gmsh = require_gmsh()
    boundaries = [{int(tag) for dim, tag in gmsh.model.getBoundary(
        faces, oriented=False, combined=False) if dim == 1} for faces in (left, right)]
    return [{'edge': tag, 'wallFaces': len(set(map(int, gmsh.model.getAdjacencies(1, tag)[0])) & wall_tags)}
            for tag in sorted(boundaries[0] & boundaries[1])]


def occ_radius(tags: list[int], phi: float, z: float, expected: float,
               *, boxes: dict[int, tuple] | None = None) -> float:
    """Fixed-z radial half-plane intersection, not nearest-point deviation."""
    from .builders._occ import require_gmsh

    gmsh = require_gmsh()
    target = [expected * math.cos(phi), expected * math.sin(phi), z]
    if boxes is None:
        boxes = {tag: gmsh.model.getBoundingBox(2, tag) for tag in tags}
    # Try the box containing the expected point first. Keep every candidate
    # and stable tie order; this avoids projecting onto opposite sectors.
    def distance(tag):
        box = boxes[tag]
        return sum(max(box[i] - target[i], target[i] - box[i + 3], 0.) ** 2 for i in range(3))
    for tag in sorted(tags, key=distance):
        box = boxes[tag]
        if not box[2] - 1e-6 <= z <= box[5] + 1e-6:
            continue
        lo, hi = gmsh.model.getParametrizationBounds(2, tag)
        for v in (lo[1], hi[1]):
            if abs(gmsh.model.getValue(2, tag, [lo[0], v])[2] - z) > 1e-8:
                continue
            def boundary_phi(u):
                xyz = gmsh.model.getValue(2, tag, [u, v])
                return xyz[0] * math.sin(phi) - xyz[1] * math.cos(phi)
            stations = np.linspace(lo[0], hi[0], 17)
            for left, right in zip(stations[:-1], stations[1:]):
                if boundary_phi(left) * boundary_phi(right) > 0:
                    continue
                u = brentq(boundary_phi, left, right, xtol=1e-13)
                xyz = np.asarray(gmsh.model.getValue(2, tag, [u, v]))
                if abs(xyz[2] - z) < 1e-8 and np.dot(xyz[:2], [math.cos(phi), math.sin(phi)]) > 0:
                    return float(np.linalg.norm(xyz[:2]))
        fixed_v = None
        for u in (lo[0], hi[0]):
            def boundary_z(v):
                return gmsh.model.getValue(2, tag, [u, v])[2] - z
            if boundary_z(lo[1]) * boundary_z(hi[1]) <= 0:
                v = brentq(boundary_z, lo[1], hi[1], xtol=1e-13)
                if fixed_v is None:
                    fixed_v = v
                xyz = np.asarray(gmsh.model.getValue(2, tag, [u, v]))
                radial = xyz[0] * math.cos(phi) + xyz[1] * math.sin(phi)
                if abs(xyz[0] * math.sin(phi) - xyz[1] * math.cos(phi)) < 1e-8 and radial > 0:
                    return float(np.linalg.norm(xyz[:2]))
        if fixed_v is not None:
            # Corner patches have z independent of u. Solve the meridian on
            # that isocurve, verifying both residuals with the same tolerance
            # as the general solve below before accepting the intersection.
            def meridian(u):
                xyz = gmsh.model.getValue(2, tag, [u, fixed_v])
                return xyz[0] * math.sin(phi) - xyz[1] * math.cos(phi)
            if meridian(lo[0]) * meridian(hi[0]) <= 0:
                u = brentq(meridian, lo[0], hi[0], xtol=1e-13)
                xyz = np.asarray(gmsh.model.getValue(2, tag, [u, fixed_v]))
                residual = [xyz[0] * math.sin(phi) - xyz[1] * math.cos(phi), xyz[2] - z]
                if np.linalg.norm(residual) < 2e-6 and np.dot(xyz[:2], [math.cos(phi), math.sin(phi)]) > 0:
                    return float(np.linalg.norm(xyz[:2]))
        _, uv = gmsh.model.getClosestPoint(2, tag, target)
        def residual(parameters):
            xyz = gmsh.model.getValue(2, tag, parameters)
            return [xyz[0] * math.sin(phi) - xyz[1] * math.cos(phi), xyz[2] - z]
        sol = least_squares(residual, np.clip(uv, lo, hi), bounds=(lo, hi),
                            xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=50)
        xyz = np.asarray(gmsh.model.getValue(2, tag, sol.x))
        if np.linalg.norm(sol.fun) < 2e-6 and np.dot(xyz[:2], [math.cos(phi), math.sin(phi)]) > 0:
            return float(np.linalg.norm(xyz[:2]))
    raise ValueError(f"FREEFORM OCC wall has no radial intersection at z={z:g} mm, phi={math.degrees(phi):g} deg")


def slope_pair(fn, z: float, h: float) -> list[float]:
    return [math.degrees(math.atan((fn(z - h) - fn(z - 2 * h)) / h)),
            math.degrees(math.atan((fn(z + 2 * h) - fn(z + h)) / h))]


def _occ_probe(points, recipe, wall, segments=None, groups=None):
    """Immutable-pass index: semantic span/sector ancestry and cached boxes.

    Never carry this cache across sewing, removal or another OCC mutation.
    Boundary probes retain both incident spans/sectors in original order.
    """
    from .builders._occ import require_gmsh
    gmsh = require_gmsh()
    tags = [tag for _, tag in wall]
    boxes = {tag: gmsh.model.getBoundingBox(2, tag) for tag in tags}
    cuts = recipe.cuts(points)
    angular_bounds = []
    for columns in groups or []:
        angles = np.unwrap(np.arctan2(points[columns, 0, 1], points[columns, 0, 0]))
        angular_bounds.append((angles[0], angles[-1]))

    def probe(phi, z, expected):
        adjacent = tags
        if segments is not None and groups is not None:
            spans = range(int(np.searchsorted(cuts, z - 1e-9)),
                          int(np.searchsorted(cuts, z + 1e-9)) + 1)
            sectors = [i for i, (lo, hi) in enumerate(angular_bounds)
                       if (phi - lo) % math.tau <= hi - lo + 1e-9
                       or abs(math.remainder(phi - lo, math.tau)) < 1e-9]
            adjacent = [tag for span in spans for sector in sectors
                        for _, tag in segments[f'{span}:{sector}']]
        return occ_radius(adjacent, phi, z, expected, boxes=boxes)
    return probe


def measure_occ(points: np.ndarray, recipe: CornerWall, wall: list[tuple[int, int]],
                *, segments: dict[str, list[tuple[int, int]]] | None = None,
                groups: list[list[int]] | None = None) -> list[dict[str, Any]]:
    probe = _occ_probe(points, recipe, wall, segments, groups)
    cuts = recipe.cuts(points)
    records = []
    bounds = [float(points[0, 0, 2]), *cuts, float(points[0, -1, 2])]
    for i, cut in enumerate(cuts):
        h = min(0.001, (cut - bounds[i]) / 1000, (bounds[i + 2] - cut) / 1000)
        azimuths = []
        for deg in range(0, 360, 45):
            phi = math.radians(deg)
            # Reduced domains measure the equivalent reflected meridian.
            x, y = math.cos(phi), math.sin(phi)
            if np.min(points[..., 0]) >= -1e-6:
                x = abs(x)
            if np.min(points[..., 1]) >= -1e-6:
                y = abs(y)
            phi = math.atan2(y, x)
            expected = lambda z: float(recipe.radius(phi, z, points))
            fn = lambda z: probe(phi, z, expected(z))
            intended = slope_pair(expected, cut, h)
            built = slope_pair(fn, cut, h)
            azimuths.append({'phiDeg': deg, 'intendedSlopesDeg': intended, 'occSlopesDeg': built,
                             'occJumpDeg': built[1] - built[0],
                             'occJumpErrorDeg': abs((built[1] - built[0]) - (intended[1] - intended[0]))})
        # Incidence is attached from semantic span ancestry after sewing.
        edges = []
        records.append({'zMm': cut, 'azimuths': azimuths, 'sharedEdgeIncidence': edges})
    return records


def _boundary_near(surface: tuple[int, int], target: np.ndarray) -> int:
    from .builders._occ import require_gmsh
    gmsh = require_gmsh()
    box = np.r_[target.min(axis=0), target.max(axis=0)]
    curves = [tag for dim, tag in gmsh.model.getBoundary([surface], oriented=False) if dim == 1]
    return int(min(curves, key=lambda tag: np.linalg.norm(np.asarray(gmsh.model.getBoundingBox(1, tag)) - box)))


def _ruled(a: int, b: int) -> list[tuple[int, int]]:
    from .builders._occ import require_gmsh
    occ = require_gmsh().model.occ
    return [(int(dim), int(tag)) for dim, tag in occ.addThruSections(
        [occ.addWire([a]), occ.addWire([b])], makeSolid=False, makeRuled=True) if dim == 2]


def _cap(geometry, points, first):
    from .builders._occ import require_gmsh, make_planar_fill_from_boundary, extreme_boundary_loop_curves
    from .builders.point_grid_sources import (_source_cap_height, _throat_radius,
                                               _source_cap_radius, _occ_rounded_cap_on_wall_rim)
    gmsh = require_gmsh()
    radius = _throat_radius(points, closed=geometry.closed)
    height = _source_cap_height(radius, geometry) if geometry.source_shape == 1 else 0
    center = np.array([0., 0., float(points[0, 0, 2])])
    if height <= 1e-12:
        return make_planar_fill_from_boundary(first, closed=geometry.closed)
    if geometry.closed:
        return _occ_rounded_cap_on_wall_rim(first, geometry, center=center,
                                           throat_radius=radius, cap_height=height)
    curves = extreme_boundary_loop_curves(first)
    endpoints = []
    for curve in curves:
        endpoints.extend(abs(int(tag)) for dim, tag in gmsh.model.getBoundary([(1, curve)], oriented=False) if dim == 0)
    from collections import Counter
    ends = [tag for tag, count in Counter(endpoints).items() if count == 1]
    if len(ends) != 2:
        raise ValueError('FREEFORM corner source cap has no unique open rim')
    sign = -1 if geometry.source_curv == -1 else 1
    pole = gmsh.model.occ.addPoint(0, 0, center[2] + sign * height)
    sphere = gmsh.model.occ.addPoint(0, 0, center[2] + sign * (height - _source_cap_radius(radius, geometry)))
    arcs = [gmsh.model.occ.addCircleArc(end, sphere, pole) for end in ends]
    loop = gmsh.model.occ.addCurveLoop([*curves, arcs[1], -arcs[0]])
    return [(2, int(gmsh.model.occ.addSurfaceFilling(loop)))]


MAX_CORNER_REFINEMENTS = 4
MAX_CORNER_RINGS = 1025
WALL_ANGULAR_TOLERANCE_MM = 0.005
# Leave axial headroom above the angular floor, and margin below the separate
# 0.01 mm dense-probe acceptance bound, without over-refining source rims.
WALL_FIT_TOLERANCE_MM = 0.0075
# Steeper walls read the radial error at this angle's axial probe instead.
RADIAL_PROBE_MAX_WALL_ANGLE_DEG = 75.
MAX_WALL_FIT_REFINEMENTS = 16
MAX_WALL_PROFILES = 2048


def build_corner_wall(geometry):
    from .builders._occ import require_gmsh, make_planar_fill_from_boundary
    from .builders.point_grid_surfaces import _bspline_patch_phi_groups, _open_sector_count, _snap_open_symmetry_grid
    from .builders.point_grid_dispatch import _shift_coupled_baffle_grid
    from .builders.enclosure import build_enclosure_box
    from .builders.point_grid_interfaces import _normalise_interface_specs, _add_offset_interface_surfaces
    from .geometry import (PointGridBuildMode, MESH_ALGORITHM_MESHADAPT,
                           MESH_ALGORITHM_FRONTAL_DELAUNAY, MESH_ALGORITHM_DELAUNAY)
    from .normals import open_shell_wall_orientation_references
    from .tags import PhysicalGroup

    gmsh = require_gmsh()
    recipe = geometry.freeform_wall
    smooth = not freeform_corner_z(recipe.profile)
    points = np.asarray(geometry.inner_points).copy()
    mode = geometry.build_mode
    interface_stations = [(float(points[0, int(interface.slice_index), 2]), float(interface.offset_mm))
                          for interface in _normalise_interface_specs(geometry, points.shape[1])]
    if mode is PointGridBuildMode.INFINITE_BAFFLE:
        points = _shift_coupled_baffle_grid(points)
    points = _snap_open_symmetry_grid(points, closed=geometry.closed,
                                      symmetry_planes=geometry.symmetry_planes)
    slope_refinements = fit_refinements = angular_refinements = 0
    for attempt in range(MAX_CORNER_REFINEMENTS + MAX_WALL_FIT_REFINEMENTS + 1):
        groups = _bspline_patch_phi_groups(len(points), closed=geometry.closed,
                                          n_sectors=_open_sector_count(closed=geometry.closed,
                                                                       symmetry_planes=geometry.symmetry_planes))
        # Each accepted ring set gets the full angular budget: rings added by
        # axial refinement can cross a moving cross-section corner between
        # existing rows. MAX_WALL_PROFILES still bounds the net.
        points, groups, refined = _fit_angular_grid(points, recipe, groups)
        angular_refinements += refined
        before = set(gmsh.model.occ.getEntities())
        patches = _patches(points, recipe.cuts(points), groups, recipe)
        wall = [face for _, _, face in patches]
        gmsh.model.occ.synchronize()
        measurements = measure_occ(points, recipe, wall,
                                   segments={f'{span}:{sector}': [face] for span, sector, face in patches},
                                   groups=groups)
        worst = max(((row['occJumpErrorDeg'], record['zMm']) for record in measurements
                     for row in record['azimuths']), default=(0, 0))
        intervals = None
        if worst[0] > 0.5:
            if slope_refinements == MAX_CORNER_REFINEMENTS or 2 * points.shape[1] - 1 > MAX_CORNER_RINGS:
                raise ValueError(f"FREEFORM corner z={worst[1]:g} mm: OCC slope jump error {worst[0]:.6g} deg exceeds 0.5 deg at refinement cap")
            slope_refinements += 1
        else:
            fit_error, failing = _wall_fit_error(points, recipe, patches, groups)
            if fit_error <= WALL_FIT_TOLERANCE_MM:
                break
            # Smooth walls bisect only their failing intervals; corner spans
            # keep the uniform bisection their approximating patches assume.
            intervals = failing if smooth else None
            added = len(intervals) if smooth else points.shape[1] - 1
            if (fit_refinements == MAX_WALL_FIT_REFINEMENTS
                    or points.shape[1] + added > MAX_CORNER_RINGS):
                raise ValueError(f"FREEFORM corner wall sampled wall error {fit_error:g} mm "
                                 f"exceeds {WALL_FIT_TOLERANCE_MM:g} mm at geometry refinement cap")
            fit_refinements += 1
        owned = set(gmsh.model.occ.getEntities()) - before
        gmsh.model.occ.remove(sorted(owned, reverse=True), recursive=True)
        points = _refine(points, recipe, intervals)
    # Sewing happens before caps/enclosure see the first/last boundaries.
    segment_roles = {f'{span}:{sector}': [face] for span, sector, face in patches}
    segment_roles = _sew_roles(segment_roles)
    last_span = len(cut_indices(points, recipe.cuts(points))) - 2
    first = [face for key, faces in segment_roles.items() if key.startswith('0:') for face in faces]
    last = [face for key, faces in segment_roles.items() if key.startswith(f'{last_span}:') for face in faces]
    wall = [face for faces in segment_roles.values() for face in faces]
    roles = {'inner': wall, 'throat_disc': _cap(geometry, points, first)}
    enclosure_bounds = None
    strip_widths = clearance = None
    outer_envelope = None
    if mode is PointGridBuildMode.INFINITE_BAFFLE:
        roles['mouth_aperture'] = make_planar_fill_from_boundary(last, use_min=False, closed=geometry.closed)
    elif mode is PointGridBuildMode.ENCLOSURE:
        data = build_enclosure_box(inner_dimtags=last, inner_points=points,
                                    enclosure=geometry.enclosure, closed=geometry.closed,
                                    symmetry_planes=geometry.symmetry_planes,
                                    surface_fit="approximate", boundary_recipe=recipe)
        roles['enclosure'] = data['dimtags']
        roles['enclosure_edges_front'] = [(2, int(tag)) for tag in data['front_edges']]
        roles['enclosure_edges_back'] = [(2, int(tag)) for tag in data['back_edges']]
        enclosure_bounds = dict(data['bounds'])
    elif mode is PointGridBuildMode.FREESTANDING:
        from .builders.point_grid_freestanding import _wall_clearance_metadata
        from .rear_compatibility import geometry_rear_ring
        cuts = recipe.cuts(points)
        from .offset_envelope import envelope_allowance_mm
        # The refined fitting net may fold where the accepted sampling net did
        # not (or the reverse): its own envelope decision is what is published.
        outer, outer_envelope = mitred_outer_offset(
            points, geometry.wall_thickness_mm, full_circle=geometry.closed, corner_z=cuts,
            force_envelope=offset_needs_envelope(recipe.profile, geometry.wall_thickness_mm, recipe.scale),
            allowance=envelope_allowance_mm(geometry.wall_thickness_mm))
        outer = _snap_open_symmetry_grid(outer, closed=geometry.closed, symmetry_planes=geometry.symmetry_planes)
        ends = cut_indices(points, cuts)
        outer_hermite = not _revolution_grid(outer)
        outer_patches = _patches(outer, (), groups, identities=ends, hermite=outer_hermite, wall=points,
                                 interpolate_axial=not freeform_corner_z(recipe.profile))
        # Clearance is a discretisation-scale sizing probe, independent of
        # geometric fit refinement. Keep the smooth builder's accepted net;
        # corner shells instead need the new mitred net at their creases.
        sizing_outer = geometry.outer_points if smooth and geometry.outer_points is not None else outer
        clearance = _wall_clearance_metadata(geometry, sizing_outer)
        # The rear return is its own linear patch and cannot round a profile crease.
        rear = geometry_rear_ring(geometry, points, outer)
        rear = _snap_open_symmetry_grid(rear[:, None, :], closed=geometry.closed, symmetry_planes=geometry.symmetry_planes)[:, 0]
        extension = np.stack((rear, outer[:, 0]), axis=1)
        rear_patches = _patches(extension, (), groups, hermite=outer_hermite, wall=points)
        # Both ruled strips span only about one wall thickness.
        strip_widths = {'mouth': float(np.min(np.linalg.norm(outer[:, -1] - points[:, -1], axis=-1))),
                        'rear': float(np.min(np.linalg.norm(extension[:, 1] - extension[:, 0], axis=-1)))}
        outer_roles = {f'{span}:{sector}': [face] for span, sector, face in outer_patches}
        outer_roles.update({f'rear:{sector}': [face] for _, sector, face in rear_patches})
        outer_roles = _sew_roles(outer_roles)
        roles['outer'] = [face for key, faces in outer_roles.items() if not key.startswith('rear:') for face in faces]
        roles['rear'] = [face for key, faces in outer_roles.items() if key.startswith('rear:') for face in faces]
        gmsh.model.occ.synchronize()
        roles['rear_cap'] = make_planar_fill_from_boundary(roles['rear'], closed=geometry.closed)
        roles['mouth'] = []
        for sector, indices in enumerate(groups):
            inner_surface = segment_roles[f'{last_span}:{sector}'][0]
            outer_surface = outer_roles[f'{last_span}:{sector}'][0]
            roles['mouth'].extend(_ruled(_boundary_near(inner_surface, points[indices, -1]),
                                         _boundary_near(outer_surface, outer[indices, -1])))
    if mode is PointGridBuildMode.FREESTANDING:
        roles.update({f'_outer_segment:{key}': faces for key, faces in outer_roles.items()
                      if not key.startswith('rear:')})
    roles.update({f'_inner_segment:{key}': faces for key, faces in segment_roles.items()})
    roles = _sew_roles(roles)
    mapped_inner = {key.removeprefix('_inner_segment:'): roles.pop(key)
                    for key in list(roles) if key.startswith('_inner_segment:')}
    mapped_outer = {key.removeprefix('_outer_segment:'): roles.pop(key)
                    for key in list(roles) if key.startswith('_outer_segment:')}
    if geometry.enclosure is not None:
        roles['interface'] = [face for z, offset in interface_stations
                              for face in _add_offset_interface_surfaces(points,
                                  slice_index=int(np.argmin(abs(points[0, :, 2] - z))), closed=geometry.closed,
                                  offset_mm=offset)]
    gmsh.model.occ.synchronize()
    records = measure_occ(points, recipe, roles['inner'], segments=mapped_inner, groups=groups)
    for span, record in enumerate(records):
        edges = [edge for sector in range(len(groups))
                 for edge in _crease_edges(mapped_inner[f'{span}:{sector}'],
                     mapped_inner[f'{span + 1}:{sector}'], {tag for _, tag in roles['inner']})]
        record['sharedEdgeIncidence'] = edges
        if len(edges) != len(groups) or any(edge['wallFaces'] != 2 for edge in edges):
            raise ValueError(f"FREEFORM corner z={record['zMm']:g} mm: crease is not shared by exactly two wall faces")
    if mapped_outer:
        outer_tags = {tag for _, tag in roles['outer']}
        for span, record in enumerate(records):
            edges = []
            for sector in range(len(groups)):
                neighbors = [mapped_outer[f'{side}:{sector}'] for side in (span, span + 1)]
                boundaries = [{int(tag) for dim, tag in gmsh.model.getBoundary(faces, oriented=False, combined=False)
                               if dim == 1} for faces in neighbors]
                shared = boundaries[0] & boundaries[1]
                if len(shared) != 1:
                    raise ValueError(f"FREEFORM corner z={record['zMm']:g} mm: outer crease is not one shared edge")
                tag = shared.pop()
                up, _ = gmsh.model.getAdjacencies(1, tag)
                incidence = len(set(map(int, up)) & outer_tags)
                if incidence != 2:
                    raise ValueError(f"FREEFORM corner z={record['zMm']:g} mm: outer crease needs two incident wall faces")
                edges.append({'edge': tag, 'wallFaces': incidence})
            record['outerSharedEdgeIncidence'] = edges
    tags = {role: [int(tag) for _, tag in faces] for role, faces in roles.items()}
    rigid = [tag for role in ('inner', 'outer', 'rear', 'rear_cap', 'mouth') for tag in tags.get(role, [])]
    physical = {int(PhysicalGroup.RIGID_WALL): rigid,
                int(PhysicalGroup.PRIMARY_SOURCE): tags['throat_disc']}
    for role, group in (('enclosure', PhysicalGroup.ENCLOSURE_WALL), ('interface', PhysicalGroup.INTERFACE),
                        ('mouth_aperture', PhysicalGroup.MOUTH_APERTURE)):
        if tags.get(role):
            physical[int(group)] = tags[role]
    reference_points, reference_normals = (None, None)
    if mode is PointGridBuildMode.BARE:
        reference_points, reference_normals = open_shell_wall_orientation_references(points, closed=geometry.closed)
    metadata = {'cornerWall': {'corners': records, 'refinements': attempt, 'axialRings': points.shape[1],
                               'fitToleranceMm': WALL_FIT_TOLERANCE_MM,
                               'sampledFitErrorMm': fit_error, 'fitRefinements': fit_refinements,
                               'angularRefinements': angular_refinements}}
    if outer_envelope is not None:
        # Authoritative for the fitted net, which the build certifies.
        metadata['outerEnvelope'] = bool(outer_envelope)
    if clearance is not None:
        metadata['outerWallClearance'] = clearance
        if smooth:
            from .builders.point_grid_freestanding import _mouth_clearance_metadata
            metadata['mouthRimClearance'] = _mouth_clearance_metadata(geometry, geometry.inner_points)
    if mode is PointGridBuildMode.INFINITE_BAFFLE:
        metadata.update(apertureTag=int(PhysicalGroup.MOUTH_APERTURE), apertureZMm=0.,
                        cavityDepthMm=float(points[0, -1, 2] - points[0, 0, 2]))
    return _BuiltCornerGeometry(surface_groups=physical, axial_bounds_mm=(float(points[0, 0, 2]), float(points[0, -1, 2])),
                         source_axis='z', mesh_surface_groups=tags, enclosure_bounds=enclosure_bounds,
                         symmetry_snap_axes=() if geometry.closed else tuple(geometry.symmetry_planes),
                         open_shell_wall_points_mm=reference_points, open_shell_wall_normals=reference_normals,
                         mesh_algorithm=({PointGridBuildMode.FREESTANDING: MESH_ALGORITHM_FRONTAL_DELAUNAY,
                                          PointGridBuildMode.ENCLOSURE: MESH_ALGORITHM_DELAUNAY}.get(
                                              mode, MESH_ALGORITHM_MESHADAPT)
                                         if smooth else MESH_ALGORITHM_MESHADAPT), metadata=metadata,
                         freeform_wall_points_mm=points, freeform_wall_recipe=recipe,
                         freeform_strip_widths_mm=strip_widths,
                         freeform_wall_segments=mapped_inner, freeform_wall_groups=groups)


MAX_MERIDIAN_NODES = 4097


def corner_size_fields(built, density):
    """User density everywhere; bounded local grading only at curved creases."""
    if not freeform_corner_z(built.freeform_wall_recipe.profile):
        return [], []
    from .builders._occ import require_gmsh
    gmsh = require_gmsh()
    points, recipe = built.freeform_wall_points_mm, built.freeform_wall_recipe
    records = built.metadata['cornerWall']['corners']
    tags = built.mesh_surface_groups['inner']
    probe = _occ_probe(points, recipe, [(2, tag) for tag in tags],
                       built.freeform_wall_segments, built.freeform_wall_groups)
    fields, sizes = [], []
    bounds = (points[0, 0, 2], *recipe.cuts(points), points[0, -1, 2])
    for record in records:
        cut = record['zMm']
        t = (cut - bounds[0]) / (bounds[-1] - bounds[0])
        local_size = (1 - t) * density.throat_res_mm + t * density.mouth_res_mm
        distance = min(abs(cut - z) for z in bounds if z != cut)
        chord_limit = local_size
        for phi in (0., math.pi / 2):
            expected = lambda z: float(recipe.radius(phi, z, points))
            fn = lambda z: probe(phi, z, expected(z))
            intended = slope_pair(expected, cut, min(.001, distance / 1000))
            step = min(local_size, distance / 4)
            for _ in range(12):
                angles = [math.degrees(math.atan((fn(cut) - fn(cut-step)) / step)),
                          math.degrees(math.atan((fn(cut+step) - fn(cut)) / step))]
                if abs((angles[1] - angles[0]) - (intended[1] - intended[0])) <= .45:
                    break
                step /= 2
            else:
                raise ValueError(f"FREEFORM corner z={cut:g} mm: mesh meridian chord refinement cap exceeded; corners may be too close to each other or to the throat/mouth")
            # Do not tighten a straight meridian merely because the next cut
            # limits the probe distance. Only a measured failed probe refines.
            if step < min(local_size, distance / 4):
                chord_limit = min(chord_limit, step)
        count = math.ceil(min(local_size, distance / 4) / chord_limit) + 1
        if count > MAX_MERIDIAN_NODES:
            raise ValueError(f"FREEFORM corner z={cut:g} mm: mesh meridian needs {count} nodes, exceeding chord refinement cap; corners may be too close to each other or to the throat/mouth")
        if chord_limit >= local_size:
            continue
        # A short band covers both H/V curves and their adjacent faces.
        # Gmsh can overshoot its target; retain half the measured safe chord.
        size = chord_limit / 2
        band = min(local_size, distance / 4)
        distance_field = gmsh.model.mesh.field.add('Distance')
        gmsh.model.mesh.field.setNumbers(distance_field, 'CurvesList',
            [edge['edge'] for edge in record['sharedEdgeIncidence']])
        gmsh.model.mesh.field.setNumber(distance_field, 'Sampling', 200)
        field = gmsh.model.mesh.field.add('Threshold')
        for option, value in {'InField': distance_field, 'SizeMin': size,
                              'SizeMax': local_size, 'DistMin': size,
                              'DistMax': band, 'StopAtDistMax': 1}.items():
            gmsh.model.mesh.field.setNumber(field, option, value)
        restrict = gmsh.model.mesh.field.add('Restrict')
        gmsh.model.mesh.field.setNumber(restrict, 'InField', field)
        gmsh.model.mesh.field.setNumbers(restrict, 'SurfacesList', tags)
        curves = {tag for dim, tag in gmsh.model.getBoundary([(2, tag) for tag in tags],
                  combined=False, oriented=False) if dim == 1}
        gmsh.model.mesh.field.setNumbers(restrict, 'CurvesList', list(curves))
        fields.append(restrict)
        sizes.append(size)
    # A freestanding mouth rim or rear return is one wall thickness wide. Sized
    # by mouth/rear resolution alone, its triangles become slivers; an element
    # no longer than twice the strip width keeps them near atan(1/2). The size
    # grades back over neighbouring faces so they do not sliver instead.
    for role, width in (built.freeform_strip_widths_mm or {}).items():
        local_size = density.mouth_res_mm if role == 'mouth' else density.rear_res_mm
        size = 2 * width
        if size >= local_size:
            continue
        distance_field = gmsh.model.mesh.field.add('Distance')
        gmsh.model.mesh.field.setNumbers(distance_field, 'SurfacesList', built.mesh_surface_groups[role])
        gmsh.model.mesh.field.setNumber(distance_field, 'Sampling', 100)
        field = gmsh.model.mesh.field.add('Threshold')
        for option, value in {'InField': distance_field, 'SizeMin': size,
                              'SizeMax': local_size, 'DistMin': size,
                              'DistMax': local_size, 'StopAtDistMax': 1}.items():
            gmsh.model.mesh.field.setNumber(field, option, value)
        fields.append(field)
        sizes.append(size)
    return fields, sizes


def mesh_sections(xyz: np.ndarray, triangles: np.ndarray, phi: float) -> np.ndarray:
    """Intersect actual triangles with a radial half-plane (S1a helper)."""
    p = xyz[triangles]
    plane = p[:, :, 0] * math.sin(phi) - p[:, :, 1] * math.cos(phi)
    radial = p[:, :, 0] * math.cos(phi) + p[:, :, 1] * math.sin(phi)
    selected = np.flatnonzero((plane.min(axis=1) <= 1e-8) & (plane.max(axis=1) >= -1e-8)
                              & (radial.max(axis=1) > 0))
    segments = []
    for i in selected:
        hits = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            da, db = plane[i, a], plane[i, b]
            if abs(da) < 1e-8:
                hits.append(p[i, a])
            if da * db < -1e-18:
                hits.append(p[i, a] + da / (da - db) * (p[i, b] - p[i, a]))
        if len(hits) < 2:
            continue
        hits = np.unique(np.round(hits, 10), axis=0)
        if len(hits) < 2:
            continue
        r = hits[:, :2] @ [math.cos(phi), math.sin(phi)]
        if np.max(r) <= 0:
            continue
        order = np.argsort(hits[:, 2])
        a, b = order[0], order[-1]
        if hits[b, 2] - hits[a, 2] > 1e-9:
            segments.append((hits[a, 2], hits[b, 2], r[a], r[b]))
    return np.asarray(segments)


def mesh_radius(segments: np.ndarray, z: float) -> float:
    valid = segments[(segments[:, 0] <= z + 1e-7) & (segments[:, 1] >= z - 1e-7)]
    if not len(valid):
        raise ValueError(f'FREEFORM mesh has no radial intersection at z={z:g} mm')
    row = valid[0]
    return float(row[2] + (row[3] - row[2]) * (z - row[0]) / (row[1] - row[0]))


def wall_triangles(tags: list[int]) -> tuple[np.ndarray, np.ndarray]:
    from .builders._occ import require_gmsh
    gmsh = require_gmsh()
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    lookup = {int(tag): i for i, tag in enumerate(node_tags)}
    triangles = []
    for tag in tags:
        types, _, nodes = gmsh.model.mesh.getElements(2, tag)
        for kind, data in zip(types, nodes):
            if kind != 2:
                raise ValueError('FREEFORM corner diagnostics require linear triangles')
            triangles.extend(np.asarray(data).reshape(-1, 3).tolist())
    return np.asarray(coordinates).reshape(-1, 3), np.asarray([[lookup[int(tag)] for tag in row] for row in triangles])


def measure_corner_mesh(geometry, built) -> None:
    """Complete the FREEFORM report with actual meshed H/V intersections."""
    recipe = geometry.freeform_wall
    points = built.freeform_wall_points_mm
    records = built.metadata['cornerWall']['corners']
    report = dict(geometry.freeform_report or {})
    report['cornerWall'] = built.metadata['cornerWall']
    built.metadata['freeformReport'] = report
    if not records:
        return
    xyz, triangles = wall_triangles(built.mesh_surface_groups['inner'])
    probe = _occ_probe(points, recipe, [(2, tag) for tag in built.mesh_surface_groups['inner']],
                       built.freeform_wall_segments, built.freeform_wall_groups)
    z_values = np.unique(np.r_[np.linspace(points[0, 0, 2], points[0, -1, 2], 241), recipe.cuts(points)])
    for plane, phi in (('H', 0.), ('V', math.pi / 2)):
        sections = mesh_sections(xyz, triangles, phi)
        expected = lambda z: float(recipe.radius(phi, z, points))
        mesh_fn = lambda z: mesh_radius(sections, z)
        occ_fn = lambda z: probe(phi, z, expected(z))
        mesh_deviation = max(abs(mesh_fn(float(z)) - expected(float(z))) for z in z_values)
        occ_deviation = max(abs(occ_fn(float(z)) - expected(float(z))) for z in z_values)
        for record in records:
            cut = record['zMm']
            distance = min(abs(cut - z) for z in (points[0, 0, 2], *recipe.cuts(points), points[0, -1, 2]) if z != cut)
            h = min(0.001, distance / 1000)
            slopes = slope_pair(mesh_fn, cut, h)
            intended = slope_pair(expected, cut, h)
            record.setdefault('mesh', {})[plane] = {
                'slopesDeg': slopes, 'jumpDeg': slopes[1] - slopes[0],
                'jumpErrorDeg': abs((slopes[1] - slopes[0]) - (intended[1] - intended[0])),
                'maxRadialDeviationMm': mesh_deviation}
            record.setdefault('occMaxRadialDeviationMm', {})[plane] = occ_deviation
