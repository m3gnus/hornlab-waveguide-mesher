"""One mirror-symmetry check for every input kind.

Whether a model is mirror-symmetric about a coordinate plane is decided here,
once, for every caller: imported OCC surface models (through the adapter in
:mod:`hornlab_mesher.step_prepare`), sampled parametric point grids, and the
analytic enclosure box. Each input supplies a :class:`MirrorTarget` -- the
exact distance oracle for its own representation -- and the one evaluator,
:func:`evaluate_mirror_symmetry`, applies the same rule to all of them: every
sample, mirrored through the plane, lands on the model within tolerance, on a
surface of the same caller role.

Once a model is cut and meshed, :func:`verify_reduced_mesh` is the one check
of the result, for every input kind: open on each cut plane, on its positive
side, no hole elsewhere, wound like the model it mirrors.

Geometry only. Roles are opaque caller strings and are never interpreted; the
caller decides what a verdict means for its solve. Importing this module needs
numpy only: it never imports gmsh (or scipy).
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Literal, Protocol

import numpy as np


SYMMETRY_AXIS_FOR_PLANE = {"x0": 0, "y0": 1, "z0": 2}
#: The mesher's snap-axis spelling of the same planes (``"x"`` is x = 0).
PLANE_FOR_SNAP_AXIS = {"x": "x0", "y": "y0", "z": "z0"}
#: Each mirror plane by the coordinate plane it lies in, in words.
_PLANE_WORDS = {"x0": "yz", "y0": "xz", "z0": "xy"}

DEFAULT_SYMMETRY_SNAP_BAND_MM = 1.0e-4
#: The band, in millimetres, within which a reduced mesh's vertex lies on a
#: cut plane: cut-plane vertices are snapped exactly onto the plane with this
#: same band, so the reduced-mesh check only needs to absorb float noise.
SYMMETRY_SNAP_TOLERANCE_MM = DEFAULT_SYMMETRY_SNAP_BAND_MM
#: A model must reach this far (relative to its diagonal) across a plane on
#: both sides before there is a half to remove.
SPAN_TOLERANCE_REL = 1.0e-3
DEFAULT_MAX_FAILURE_SAMPLES = 5

ToleranceClass = Literal["exact", "fitted"]


@dataclass(frozen=True)
class ToleranceSpec:
    """A mirror tolerance relative to the model's diagonal, with a floor."""

    name: str
    relative: float
    floor_mm: float

    def resolve(self, diagonal: float) -> float:
        if self.floor_mm > 0.0:
            return max(self.floor_mm, self.relative * max(float(diagonal), self.floor_mm))
        return self.relative * float(diagonal)


#: The two tolerance classes (a project decision: keep two). ``exact`` is
#: for geometry built from formulas (sampled design grids); ``fitted`` is for
#: surfaces fitted or tessellated in CAD (imported OCC models).
TOLERANCES: dict[str, ToleranceSpec] = {
    "exact": ToleranceSpec("exact", relative=2.0e-4, floor_mm=1.0e-7),
    "fitted": ToleranceSpec("fitted", relative=5.0e-4, floor_mm=0.0),
}


@dataclass(frozen=True)
class PlaneSymmetryVerdict:
    """Per-plane result of the mirror test.

    Distances are in the units of the samples (STEP units for an imported
    model, millimetres for a design grid); ``to_dict`` keeps the imported
    model's established key names.
    """

    plane: str
    accepted: bool
    reason: str
    tolerance: float
    axis_min: float
    axis_max: float
    points_tested: int
    points_off_model: int
    max_residual: float
    worst_residual_surface: int | str | None
    worst_off_model_distance: float
    worst_surface: int | str | None
    role_mismatches: int
    failure_samples: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "plane": self.plane,
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "tolerance_step_units": float(self.tolerance),
            "axis_min_step_units": float(self.axis_min),
            "axis_max_step_units": float(self.axis_max),
            "points_tested": int(self.points_tested),
            "points_off_model": int(self.points_off_model),
            "max_residual_step_units": (
                None if self.max_residual < 0.0 else float(self.max_residual)
            ),
            "worst_residual_surface": self.worst_residual_surface,
            "worst_off_model_distance_step_units": (
                None
                if self.worst_off_model_distance < 0.0
                else float(self.worst_off_model_distance)
            ),
            "worst_surface": self.worst_surface,
            "role_mismatches": int(self.role_mismatches),
            "failure_samples": list(self.failure_samples),
        }


@dataclass(frozen=True)
class MirrorMatches:
    """Where one surface's mirrored samples land on the model.

    ``matched_distance`` is the distance to the nearest surface within
    tolerance, or -1.0 when none is; ``nearest_distance`` the distance to the
    nearest surface at all (-1.0: unknown). ``hits`` lists, per sample, every
    surface within tolerance; None means the only surface ever tested is the
    sample's own, so a hit can never land on another role. ``context`` is
    the target's own per-sample state, handed back to
    :meth:`MirrorTarget.role_seam_allows`.
    """

    matched_distance: np.ndarray
    nearest_distance: np.ndarray
    hits: Sequence[Sequence[Hashable]] | None = None
    context: Sequence[Any] | None = None


class MirrorTarget(Protocol):
    """The exact distance oracle for one geometry representation."""

    def match(self, surface: Hashable, mirrored: np.ndarray) -> MirrorMatches:
        """Where the mirrored samples of ``surface`` land on the model."""
        ...

    def role_seam_allows(
        self, surface: Hashable, point: np.ndarray, mirrored: np.ndarray,
        hits: Sequence[Hashable], context: Any,
    ) -> bool:
        """Whether a role mismatch is only a sample on a role seam."""
        ...


def _label(surface: Hashable) -> Hashable:
    """A surface id as it appears in a failure sample (OCC tags as plain ints)."""
    if isinstance(surface, (int, np.integer)) and not isinstance(surface, bool):
        return int(surface)
    return surface


def evaluate_mirror_symmetry(
    plane: str,
    *,
    samples: Mapping[Hashable, np.ndarray],
    roles: Mapping[Hashable, str],
    target: MirrorTarget,
    tolerance: float,
    span_tolerance: float,
    require_straddle: bool = True,
    max_failure_samples: int = DEFAULT_MAX_FAILURE_SAMPLES,
) -> PlaneSymmetryVerdict:
    """Require every sample to mirror onto the model, onto its own role.

    ``samples`` maps each surface id to its sample points ``(N, 3)``;
    ``roles`` gives each surface its caller role. With ``require_straddle``
    a model that does not reach past ``span_tolerance`` on both sides of the
    plane is rejected: there is no half to remove.
    """
    axis = SYMMETRY_AXIS_FOR_PLANE[plane]
    candidates = sorted(samples)
    missing_roles = set(candidates) - set(roles)
    if missing_roles:
        raise ValueError(f"missing roles for surfaces: {sorted(missing_roles)}")
    all_axis_values = (
        np.concatenate([samples[surface][:, axis] for surface in candidates])
        if candidates
        else np.zeros(1)
    )
    axis_min = float(np.min(all_axis_values))
    axis_max = float(np.max(all_axis_values))
    points_tested = int(sum(len(samples[surface]) for surface in candidates))
    if require_straddle and not (axis_min < -span_tolerance and axis_max > span_tolerance):
        return PlaneSymmetryVerdict(
            plane, False,
            "model does not straddle the plane; there is no half to remove",
            tolerance, axis_min, axis_max, points_tested, 0, -1.0, None, -1.0,
            None, 0, (),
        )

    max_residual = -1.0
    max_residual_surface: Hashable | None = None
    points_off_model = 0
    role_mismatches = 0
    worst_off_model = -1.0
    worst_surface: Hashable | None = None
    failures: list[dict[str, object]] = []
    for surface in candidates:
        role = roles[surface]
        points = samples[surface]
        mirrored_points = points.copy()
        mirrored_points[:, axis] *= -1
        matches = target.match(surface, mirrored_points)
        matched = np.asarray(matches.matched_distance, dtype=float)
        nearest = np.asarray(matches.nearest_distance, dtype=float)
        # A sample hits when any surface lies within tolerance: the hit list
        # when the target gives one (its distance may even be NaN), else a
        # matched distance.
        hit = (
            matched >= 0.0
            if matches.hits is None
            else np.fromiter((bool(hits) for hits in matches.hits), dtype=bool, count=len(points))
        )
        residuals = matched[hit]
        residuals = residuals[~np.isnan(residuals)]
        if residuals.size:
            best = float(np.max(residuals))
            if best > max_residual:
                max_residual = best
                max_residual_surface = _label(surface)
        # Only misses can fail when every hit is the sample's own surface;
        # otherwise every hit's roles are checked, in sample order.
        inspected = (
            np.flatnonzero(~hit) if matches.hits is None else range(len(points))
        )
        for index in inspected:
            point = points[index]
            mirrored = mirrored_points[index]
            if not hit[index]:
                distance = float(nearest[index])
                points_off_model += 1
                if distance > worst_off_model:
                    worst_off_model = distance
                    worst_surface = _label(surface)
                if len(failures) < max_failure_samples:
                    failures.append(
                        {
                            "kind": "off_model",
                            "surface": _label(surface),
                            "role": role,
                            "point_step_units": [float(value) for value in point],
                            "mirrored_point_step_units": [
                                float(value) for value in mirrored
                            ],
                            "nearest_distance_step_units": (
                                None if distance < 0.0 else distance
                            ),
                        }
                    )
                continue
            hits = matches.hits[index]
            if role in {roles[hit_surface] for hit_surface in hits}:
                continue
            context = matches.context[index] if matches.context is not None else None
            if target.role_seam_allows(surface, point, mirrored, hits, context):
                continue
            role_mismatches += 1
            if len(failures) < max_failure_samples:
                failures.append(
                    {
                        "kind": "role_mismatch",
                        "surface": _label(surface),
                        "role": role,
                        "point_step_units": [float(value) for value in point],
                        "mirrored_point_step_units": [
                            float(value) for value in mirrored
                        ],
                        "mirrored_surfaces": [_label(hit_surface) for hit_surface in hits],
                        "mirrored_roles": sorted({roles[hit_surface] for hit_surface in hits}),
                    }
                )
    accepted = points_off_model == 0 and role_mismatches == 0
    if accepted:
        reason = "geometry and caller roles mirror within tolerance"
    elif points_off_model:
        reason = (
            f"{points_off_model} of {points_tested} sampled points do not mirror "
            "onto the model"
        )
    else:
        reason = (
            f"{role_mismatches} sampled points mirror onto a face of a different role"
        )
    return PlaneSymmetryVerdict(
        plane, accepted, reason, tolerance, axis_min, axis_max, points_tested,
        points_off_model, max_residual, max_residual_surface, worst_off_model,
        worst_surface, role_mismatches, tuple(failures),
    )


# -- sampled point grids (designs) -------------------------------------------


def _point_segment_distances(points: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest segment of one closed ring."""

    starts = ring
    deltas = np.roll(ring, -1, axis=0) - starts
    lengths_sq = np.einsum("ij,ij->i", deltas, deltas)
    offsets = points[:, None, :] - starts[None, :, :]
    fractions = np.divide(
        np.einsum("pij,ij->pi", offsets, deltas),
        lengths_sq[None, :],
        out=np.zeros((len(points), len(ring)), dtype=np.float64),
        where=lengths_sq[None, :] > 1.0e-24,
    )
    fractions = np.clip(fractions, 0.0, 1.0)
    nearest = starts[None, :, :] + fractions[:, :, None] * deltas[None, :, :]
    return np.min(np.linalg.norm(points[:, None, :] - nearest, axis=2), axis=1)


class RingMirrorTarget:
    """Closed azimuthal rings of a sampled grid, shaped ``(n_phi, n_axial, 3)``.

    A mirrored sample is measured against the polyline of its own axial ring
    (point to nearest segment), so a grid whose azimuth samples do not
    themselves mirror is still judged by the surface they describe.
    """

    def __init__(self, grids: Mapping[Hashable, np.ndarray], tolerance: float) -> None:
        self.grids = grids
        self.tolerance = float(tolerance)

    def match(self, surface: Hashable, mirrored: np.ndarray) -> MirrorMatches:
        grid = self.grids[surface]
        reflected = mirrored.reshape(grid.shape)
        distances = np.empty(grid.shape[:2], dtype=np.float64)
        for axial_index in range(grid.shape[1]):
            distances[:, axial_index] = _point_segment_distances(
                np.ascontiguousarray(reflected[:, axial_index, :]), grid[:, axial_index, :]
            )
        nearest = distances.reshape(-1)
        matched = np.where(nearest <= self.tolerance, nearest, -1.0)
        return MirrorMatches(matched, nearest)

    def role_seam_allows(self, surface, point, mirrored, hits, context) -> bool:
        return False


class VertexMirrorTarget:
    """Every vertex of a sampled grid: a mirrored sample must land on a vertex.

    Stricter than :class:`RingMirrorTarget` and blind to which ring the vertex
    is on; fast (a KD-tree, O(N log N)) at any grid size. Right where the grid's
    azimuths are built to mirror onto each other, as the mesher's own grids are.
    """

    def __init__(self, grids: Mapping[Hashable, np.ndarray], tolerance: float) -> None:
        self.grids = grids
        self.tolerance = float(tolerance)
        self._trees: dict[Hashable, Any] = {}

    def match(self, surface: Hashable, mirrored: np.ndarray) -> MirrorMatches:
        tree = self._trees.get(surface)
        if tree is None:
            from scipy.spatial import cKDTree  # lazy: the module stays numpy-only

            tree = self._trees[surface] = cKDTree(self.grids[surface].reshape(-1, 3))
        nearest = np.asarray(tree.query(mirrored)[0], dtype=np.float64)
        matched = np.where(nearest <= self.tolerance, nearest, -1.0)
        return MirrorMatches(matched, nearest)

    def role_seam_allows(self, surface, point, mirrored, hits, context) -> bool:
        return False


_GRID_TARGETS = {"ring": RingMirrorTarget, "vertex": VertexMirrorTarget}


def evaluate_point_grid_symmetry(
    grids: Mapping[str, np.ndarray],
    *,
    planes: Sequence[str] = ("x0", "y0"),
    tolerance_class: ToleranceClass = "exact",
    tolerance_mm: float | None = None,
    match: Literal["ring", "vertex"] = "ring",
) -> dict[str, PlaneSymmetryVerdict]:
    """Mirror-test sampled surface grids, each ``(n_phi, n_axial, 3)``, per plane.

    Each grid is its own role, tested against its own surface only: with
    ``match="ring"`` a mirrored sample is measured to its own axial ring's
    polyline, with ``match="vertex"`` to the nearest grid vertex. The
    tolerance is the class's share of the grids' joint bounding-box diagonal
    unless ``tolerance_mm`` gives it. A grid samples a whole model, so no
    straddle is required.
    """
    if not grids:
        raise ValueError("no grid to mirror-test")
    checked: dict[str, np.ndarray] = {}
    for name, raw in grids.items():
        grid = np.asarray(raw, dtype=np.float64)
        if grid.ndim != 3 or grid.shape[2] != 3 or grid.shape[0] < 1 or grid.shape[1] < 1:
            raise ValueError(f"grid {name!r} must be shaped (n_phi, n_axial, 3)")
        if not np.isfinite(grid).all():
            raise ValueError(f"grid {name!r} contains non-finite points")
        checked[name] = grid
    if tolerance_mm is None:
        all_points = np.concatenate([grid.reshape(-1, 3) for grid in checked.values()])
        diagonal = float(np.linalg.norm(np.ptp(all_points, axis=0)))
        tolerance = TOLERANCES[tolerance_class].resolve(diagonal)
    else:
        tolerance = float(tolerance_mm)
        if not (math.isfinite(tolerance) and tolerance >= 0.0):
            raise ValueError("tolerance_mm must be finite and non-negative")
    target = _GRID_TARGETS[match](checked, tolerance)
    samples = {name: grid.reshape(-1, 3) for name, grid in checked.items()}
    roles = {name: name for name in checked}
    return {
        plane: evaluate_mirror_symmetry(
            plane, samples=samples, roles=roles, target=target,
            tolerance=tolerance, span_tolerance=0.0, require_straddle=False,
        )
        for plane in planes
    }


# -- the analytic enclosure box ----------------------------------------------


def _pair_reason(
    labels: tuple[str, str],
    values: tuple[float | None, float | None],
    tolerance_mm: float,
) -> str | None:
    first, second = values
    if (first is None or second is None
            or not math.isfinite(first) or not math.isfinite(second)):
        return f"enclosure {labels[0]}/{labels[1]} spacing is not a finite scalar"
    if abs(first - second) > tolerance_mm:
        return f"enclosure {labels[0]}={first:g} mm differs from {labels[1]}={second:g} mm"
    return None


def _distinct_numbers(first: float, second: float) -> tuple[str, str]:
    """Two different numbers, short, but never printed alike."""
    short = (f"{first:g}", f"{second:g}")
    return short if short[0] != short[1] else (repr(first), repr(second))


def evaluate_enclosure_symmetry(
    *,
    resolutions: Mapping[str, Sequence[float]] | None = None,
    dim_mm: Sequence[float] | None = None,
    spacing_lr: tuple[float | None, float | None] | None = None,
    spacing_tb: tuple[float | None, float | None] | None = None,
    spacing_tolerance_mm: float = 0.0,
) -> dict[str, list[str]]:
    """Why the enclosure box is not mirror-symmetric, per plane (``x0``, ``y0``).

    Only what the caller passes is checked, in this order:

    - ``resolutions``: per-quadrant mesh sizes (q1..q4, as the mesher expands
      them), by name; equal across each plane to 1e-12.
    - ``dim_mm``: ATH ``Mesh.Enclosure.Dim`` (+X, +Y, -X, -Y), which replaces
      the margins; the builder refuses any difference on a cut axis, so it is
      compared exactly.
    - ``spacing_lr`` / ``spacing_tb``: the left/right and top/bottom margins,
      within ``spacing_tolerance_mm``; None means not a finite scalar.

    An empty list accepts the plane.
    """
    reasons: dict[str, list[str]] = {"x0": [], "y0": []}
    for name, values in (resolutions or {}).items():
        for plane, pairs in (("y0", ((0, 3), (1, 2))), ("x0", ((0, 1), (3, 2)))):
            if any(
                not math.isclose(values[a], values[b], rel_tol=1e-12, abs_tol=1e-12)
                for a, b in pairs
            ):
                reasons[plane].append(
                    f"enclosure {name} quadrant sizes differ across the {_PLANE_WORDS[plane]} plane"
                )
    if dim_mm is not None:
        px, py, nx, ny = (float(value) for value in dim_mm)
        for plane, axis, plus, minus in (("x0", "X", px, nx), ("y0", "Y", py, ny)):
            if not (math.isfinite(plus) and math.isfinite(minus)):
                reasons[plane].append(f"enclosure Dim +{axis}/-{axis} is not a finite scalar")
            elif plus != minus:
                shown = _distinct_numbers(plus, minus)
                reasons[plane].append(
                    f"enclosure Dim +{axis}={shown[0]} mm differs from -{axis}={shown[1]} mm")
    for plane, labels, values in (
        ("x0", ("space_l", "space_r"), spacing_lr),
        ("y0", ("space_t", "space_b"), spacing_tb),
    ):
        if values is not None:
            reason = _pair_reason(labels, values, spacing_tolerance_mm)
            if reason is not None:
                reasons[plane].append(reason)
    return reasons


# -- the reduced-mesh check ----------------------------------------------------

#: Fraction of a component's own ``area ** 1.5`` below which its signed volume
#: is rounding noise rather than an orientation. The sum is taken over
#: coordinates, so its absolute error scales with the component's size and
#: triangle count, and a near-degenerate component -- a sliver panel, a shell
#: whose two cut planes almost coincide -- encloses so little that the sign is
#: whatever the accumulation happened to leave. Exactly ``0.0`` is therefore not
#: the only unresolved case, and flipping on a noise sign would silently invert
#: normals that the solver then trusts. A genuinely reduced closed shell sits
#: many orders of magnitude above this: a half-box scores ~0.09.
SIGNED_VOLUME_NOISE_REL = 1.0e-9
#: A triangle lying in a cut plane with at most this area (mm^2) is a sliver
#: that snapping a near-tangent facet onto the plane flattened, not a cap: the
#: parametric builder drops such slivers (``mesher._remove_symmetry_plane_slivers``)
#: and refuses anything larger, and the reduced-mesh check draws the same line.
SYMMETRY_SLIVER_MAX_AREA_MM2 = 0.25

_COORDINATE_PLANES = ("x0", "y0", "z0")


def _free_edges(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The edges used by exactly one triangle, and the row of that triangle."""

    edges = np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    if not len(edges):
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
    _unique, first, counts = np.unique(edges, axis=0, return_index=True, return_counts=True)
    rows = first[counts == 1]
    return edges[rows], rows // 3


def _detect_symmetry_planes(
    points: np.ndarray,
    triangles: np.ndarray,
    free_edges: np.ndarray,
    free_edge_triangles: np.ndarray,
    *,
    tolerance: float,
    min_edges_per_plane: int,
    side_points: np.ndarray | None = None,
) -> tuple[tuple[str, ...], dict[str, object]]:
    """``side_points`` (default: every point) are the vertices whose sides of
    each plane are counted."""
    on_plane = np.stack(
        [np.all(np.abs(points[free_edges, axis]) <= tolerance, axis=1) for axis in range(3)],
        axis=1,
    ) if len(free_edges) else np.zeros((0, 3), dtype=bool)
    planes_per_edge = on_plane.sum(axis=1)
    plane_counts = {
        plane: int(np.count_nonzero(on_plane[:, axis] & (planes_per_edge == 1)))
        for axis, plane in enumerate(_COORDINATE_PLANES)
    }
    shared_plane_edges = int(np.count_nonzero(planes_per_edge > 1))

    # How squarely the surface meets each plane along its rim: the median
    # |normal . plane normal| of the triangles owning the plane's free edges.
    # A mirror-symmetric smooth surface crosses its mirror plane at a right
    # angle (0); a horn mouth that merely ends on the plane meets it at its
    # flare angle. Reported only: it is evidence for a caller deciding whether
    # an undeclared open rim is a cut, never a verdict of this detector.
    rim_normal_component: dict[str, float | None] = {}
    for axis, plane in enumerate(_COORDINATE_PLANES):
        rows = free_edge_triangles[on_plane[:, axis] & (planes_per_edge == 1)] if len(free_edges) else []
        if len(rows) == 0:
            rim_normal_component[plane] = None
            continue
        owners = triangles[rows]
        normals = np.cross(
            points[owners[:, 1]] - points[owners[:, 0]],
            points[owners[:, 2]] - points[owners[:, 0]],
        )
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 0.0
        rim_normal_component[plane] = (
            float(np.median(np.abs(normals[valid, axis]) / lengths[valid]))
            if np.any(valid)
            else None
        )

    plane_vertex_side_counts: dict[str, dict[str, int]] = {}
    one_sided: dict[str, bool] = {}
    counted = points if side_points is None else side_points
    for axis, plane in enumerate(_COORDINATE_PLANES):
        coordinates = counted[:, axis]
        negative = int(np.count_nonzero(coordinates < -tolerance))
        positive = int(np.count_nonzero(coordinates > tolerance))
        plane_vertex_side_counts[plane] = {
            "negative": negative,
            "on_plane": int(len(coordinates) - negative - positive),
            "positive": positive,
        }
        one_sided[plane] = not (negative and positive)

    detected = tuple(
        plane for plane in _COORDINATE_PLANES
        if plane_counts[plane] >= min_edges_per_plane and one_sided[plane]
    )
    rejected_spanning_planes = [
        plane for plane in _COORDINATE_PLANES
        if plane_counts[plane] >= min_edges_per_plane and not one_sided[plane]
    ]
    detection = {
        "mode": "auto",
        "free_edges": int(len(free_edges)),
        "plane_free_edge_counts": {k: int(v) for k, v in plane_counts.items()},
        "plane_vertex_side_counts": plane_vertex_side_counts,
        "rejected_spanning_planes": rejected_spanning_planes,
        "shared_plane_free_edges": int(shared_plane_edges),
        "rim_normal_component": rim_normal_component,
        "min_edges_per_plane": int(min_edges_per_plane),
        "tolerance": float(tolerance),
        "detected_planes": list(detected),
    }
    return detected, detection


def detect_symmetry_planes(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    tolerance: float,
    min_edges_per_plane: int = 3,
) -> tuple[tuple[str, ...], dict[str, object]]:
    """Detect symmetry cut planes from free edges lying on x=0/y=0/z=0.

    Only free edges lying exclusively on a single coordinate plane count
    toward that plane. A cut rim in the x=0 wall crossing height z=0
    contributes edges that sit on both planes at once; counting those toward
    z0 would misread an internal level as a cut plane. A true cut outline
    always has edges away from the other coordinate planes, so the exclusive
    count stays robust. ``min_edges_per_plane`` additionally keeps an
    isolated leak vertex near the origin from masquerading as a plane. A
    candidate must also have the whole mesh on one side of it: stray free
    edges along an internal origin plane cannot turn a full-span model into a
    native symmetry-reduced solve.
    """
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    free_edges, owners = _free_edges(triangles)
    return _detect_symmetry_planes(
        points, triangles, free_edges, owners,
        tolerance=tolerance, min_edges_per_plane=min_edges_per_plane,
    )


def edge_connected_components(triangles: np.ndarray) -> np.ndarray:
    """Label each triangle with the component it reaches through shared edges."""

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    count = len(triangles)
    edges = np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    owners = np.repeat(np.arange(count), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owners = edges[order], owners[order]
    shared = np.all(edges[1:] == edges[:-1], axis=1)
    graph = coo_matrix(
        (np.ones(int(shared.sum())), (owners[:-1][shared], owners[1:][shared])),
        shape=(count, count),
    )
    _components, labels = connected_components(graph, directed=False)
    return labels


def reduced_mesh_orientation(
    points: np.ndarray,
    triangles: np.ndarray,
    cut_planes: Iterable[str],
    *,
    expected_volume_sign: Literal[1, -1],
) -> dict[str, Any]:
    """Check that a reduced domain is wound like the full model it mirrors.

    In a reduced domain whose every free edge lies on a cut plane through the
    origin, each component is closed by its own mirror images, and the
    cut-plane caps that would close it add nothing to the origin-based volume
    sum -- ``r . n`` vanishes on a plane through the origin. A component's own
    signed volume is therefore exactly ``1 / 2**k`` of the full body's, sign
    included. A sign against ``expected_volume_sign`` means it is wound
    against the model it mirrors: every normal the solver reads is reversed,
    and nothing else notices, because an inverted mesh is perfectly
    consistent.

    Counted per edge-connected component, so one inverted body cannot hide
    behind a larger correct one. A full domain (no cut plane) is not judged.
    """

    if expected_volume_sign not in (1, -1):
        raise ValueError("expected_volume_sign must be 1 or -1")
    planes = tuple(str(plane) for plane in cut_planes)
    record: dict[str, Any] = {
        "cut_planes": list(planes),
        "checked": False,
        "component_count": 0,
        "inverted_component_count": 0,
        "inverted_triangle_count": 0,
        "unresolved_component_count": 0,
    }
    tris = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if not planes or len(tris) == 0:
        return record
    points = np.asarray(points, dtype=float)
    labels = edge_connected_components(tris)
    p0, p1, p2 = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    volume_terms = np.einsum("ij,ij->i", p0, np.cross(p1, p2)) / 6.0
    areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    count = int(labels.max()) + 1
    volumes = np.bincount(labels, weights=volume_terms, minlength=count)
    floors = SIGNED_VOLUME_NOISE_REL * np.bincount(labels, weights=areas, minlength=count) ** 1.5
    unresolved = np.abs(volumes) <= floors
    inverted = ~unresolved & (expected_volume_sign * volumes < 0.0)
    record.update(
        {
            "checked": True,
            "component_count": count,
            "inverted_component_count": int(np.count_nonzero(inverted)),
            "inverted_triangle_count": int(
                np.bincount(labels, minlength=count)[inverted].sum()
            ),
            "unresolved_component_count": int(np.count_nonzero(unresolved)),
            "signed_volume_mm3": float(volumes.sum()),
        }
    )
    return record


#: Failure codes of :func:`verify_reduced_mesh`. The first four implicate the
#: planes they name; the last two are global: from a leak or a reversed
#: winding alone it cannot be told which cut caused it.
FAILURE_STRADDLING = "straddling-cut-plane"
FAILURE_WRONG_SIDE = "negative-side-spill"
FAILURE_MISSING_RIM = "missing-cut-rim"
FAILURE_CAPPED = "capped-cut-plane"
FAILURE_LEAK = "off-plane-open-edges"
FAILURE_INVERTED = "inverted-orientation"
GLOBAL_FAILURES = frozenset({FAILURE_LEAK, FAILURE_INVERTED})


def _plane_list(planes: Iterable[str]) -> str:
    return ", ".join(planes)


@dataclass(frozen=True)
class ReducedMeshVerification:
    """What one check of a reduced (or full) mesh found.

    ``failed_planes`` are the cut planes a failure names directly: one the
    mesh is not open on (no rim, or triangles lying in it), or one the mesh
    spills across or lies on the wrong side of. A leak away from every cut
    plane, or a reversed winding, names no plane (``GLOBAL_FAILURES``).
    Distances are in millimetres.
    """

    cut_planes: tuple[str, ...]
    closed: bool
    expected_volume_sign: int
    tolerance_mm: float
    free_edge_count: int
    detected_planes: tuple[str, ...]
    straddling_planes: tuple[str, ...]
    wrong_side_planes: tuple[str, ...]
    missing_rim_planes: tuple[str, ...]
    cap_face_counts: Mapping[str, int]
    off_plane_open_edge_count: int
    off_plane_open_edge_samples: tuple[tuple[float, float, float], ...]
    orientation: Mapping[str, Any]
    failure_codes: tuple[str, ...]
    failed_planes: tuple[str, ...]
    reason: str
    detection: Mapping[str, Any]

    @property
    def ok(self) -> bool:
        return not self.failure_codes

    @property
    def capped_planes(self) -> tuple[str, ...]:
        """Cut planes the mesh is closed on: no open rim, or a face lying in it."""

        return tuple(
            plane for plane in self.cut_planes
            if plane in self.missing_rim_planes or self.cap_face_counts.get(plane)
        )

    @property
    def negative_side_planes(self) -> tuple[str, ...]:
        """Cut planes with mesh on their negative side (spanning or wholly there)."""

        return tuple(
            plane for plane in self.cut_planes
            if plane in self.straddling_planes or plane in self.wrong_side_planes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cut_planes": list(self.cut_planes),
            "verified": self.ok,
            "reason": self.reason,
            "closed": self.closed,
            "expected_volume_sign": int(self.expected_volume_sign),
            "tolerance_mm": float(self.tolerance_mm),
            "free_edge_count": int(self.free_edge_count),
            "off_plane_free_edge_count": int(self.off_plane_open_edge_count),
            "off_plane_free_edge_samples": [list(sample) for sample in self.off_plane_open_edge_samples],
            "detected_planes": list(self.detected_planes),
            "capped_planes": list(self.capped_planes),
            "missing_rim_planes": list(self.missing_rim_planes),
            "cap_face_counts": {plane: int(count) for plane, count in self.cap_face_counts.items()},
            "straddling_planes": list(self.straddling_planes),
            "wrong_side_planes": list(self.wrong_side_planes),
            "failure_codes": list(self.failure_codes),
            "failed_planes": list(self.failed_planes),
            "reduced_orientation": dict(self.orientation),
            "detection": dict(self.detection),
        }


def verify_reduced_mesh(
    points: np.ndarray,
    triangles: np.ndarray,
    cut_planes: Iterable[str],
    *,
    closed: bool,
    expected_volume_sign: Literal[1, -1],
    snap_tolerance_mm: float = SYMMETRY_SNAP_TOLERANCE_MM,
    exempt_triangles: np.ndarray | None = None,
) -> ReducedMeshVerification:
    """Re-read a cut from the meshed boundary: the one reduced-mesh check.

    ``points`` are in millimetres. The mesh is judged as written, by a detector
    that knows nothing about how it was cut. A reduced domain is sound when,
    for every cut plane, the mesh keeps to the plane's positive side and is
    open on the plane (a rim of free edges in it, and no triangle lying in it);
    and, for a ``closed`` model, when no free edge lies anywhere else (a hole
    the mirrored solve would radiate through) and every component is wound
    with ``expected_volume_sign`` (the full model's). Each of these produces a
    wrong answer rather than an error, so none may pass silently.

    ``closed=False`` is for a model that owns an open rim (a bare horn's
    mouth): its off-plane free edges are reported, never failed, and it has
    no volume to orient. The winding is judged only on a boundary that passed,
    so a leak is reported as the leak it is rather than as a volume read off
    an open surface. A full domain (no cut plane) is verified by construction;
    the planes the detector finds open are reported for the caller to judge.

    ``exempt_triangles`` (a boolean mask over ``triangles``) are left out of
    the check entirely: an internal subdomain interface (ATH
    ``Mesh.SubdomainSlices``, tag ``INTERFACE``) has a free rim by
    construction, and the closed-shell contract the mesher enforces exempts it
    the same way (``mesher._CONTRACT_EXEMPT_TAGS``). A triangle in a cut plane
    no larger than ``SYMMETRY_SLIVER_MAX_AREA_MM2`` is a snap sliver, not a
    cap.
    """

    if expected_volume_sign not in (1, -1):
        raise ValueError("expected_volume_sign must be 1 or -1")
    tolerance = float(snap_tolerance_mm)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("snap_tolerance_mm must be finite and non-negative")
    planes = tuple(dict.fromkeys(str(plane) for plane in cut_planes))
    unknown = [plane for plane in planes if plane not in SYMMETRY_AXIS_FOR_PLANE]
    if unknown:
        raise ValueError(f"unknown symmetry planes: {unknown}")
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if exempt_triangles is not None:
        exempt = np.asarray(exempt_triangles, dtype=bool).reshape(-1)
        if exempt.shape != (len(triangles),):
            raise ValueError("exempt_triangles must be one flag per triangle")
        triangles = triangles[~exempt]
    free_edges, owners = _free_edges(triangles)
    # Sides are read off the vertices the triangles use: a stray unreferenced
    # vertex is not mesh on the wrong side.
    used = np.unique(triangles) if len(triangles) else np.zeros(0, dtype=np.int64)
    detected, detection = _detect_symmetry_planes(
        points, triangles, free_edges, owners, tolerance=tolerance, min_edges_per_plane=3,
        side_points=points[used],
    )

    def result(**fields: Any) -> ReducedMeshVerification:
        base: dict[str, Any] = dict(
            cut_planes=planes, closed=bool(closed),
            expected_volume_sign=int(expected_volume_sign), tolerance_mm=tolerance,
            free_edge_count=int(len(free_edges)), detected_planes=tuple(detected),
            straddling_planes=(), wrong_side_planes=(), missing_rim_planes=(),
            cap_face_counts={}, off_plane_open_edge_count=0,
            off_plane_open_edge_samples=(),
            orientation=reduced_mesh_orientation(
                points, triangles, (), expected_volume_sign=expected_volume_sign
            ),
            failure_codes=(), failed_planes=(), reason="", detection=detection,
        )
        base.update(fields)
        return ReducedMeshVerification(**base)

    if not planes:
        return result(reason="full domain: no cut plane to verify")

    sides = detection["plane_vertex_side_counts"]
    assert isinstance(sides, dict)
    missing = [plane for plane in planes if plane not in detected]
    # Three different things leave a plane without its open rim, and they
    # have different remedies, so they are kept apart.
    straddling = tuple(
        plane for plane in missing
        if sides[plane]["negative"] > 0 and sides[plane]["positive"] > 0
    )
    missing_rim = tuple(plane for plane in missing if plane not in straddling)
    # The positive side of every mirror plane is the one kept, so a domain on
    # the negative side is a mirror image of the model: accepted, it would
    # solve the wrong half.
    wrong_side = tuple(
        plane for plane in planes
        if sides[plane]["positive"] == 0 and sides[plane]["negative"] > 0
    )
    # A triangle lying in a cut plane is a wall on the mirror: the mirrored
    # solve sees it coincide with its own image.
    cap_faces: dict[str, int] = {}
    if len(triangles):
        corners = points[triangles]
        areas = np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
        )
        for plane in planes:
            axis = SYMMETRY_AXIS_FOR_PLANE[plane]
            in_plane = np.all(np.abs(corners[:, :, axis]) <= tolerance, axis=1) & (
                0.5 * areas > SYMMETRY_SLIVER_MAX_AREA_MM2
            )
            count = int(np.count_nonzero(in_plane))
            if count:
                cap_faces[plane] = count
    on_cut = np.zeros(len(free_edges), dtype=bool)
    for plane in planes:
        axis = SYMMETRY_AXIS_FOR_PLANE[plane]
        on_cut |= np.all(np.abs(points[free_edges, axis]) <= tolerance, axis=1)
    off_plane = free_edges[~on_cut]
    samples = tuple(
        tuple(float(value) for value in 0.5 * (points[a] + points[b]))
        for a, b in off_plane[:DEFAULT_MAX_FAILURE_SAMPLES]
    )

    codes: list[str] = []
    reasons: list[str] = []
    failed: set[str] = set()
    if straddling:
        codes.append(FAILURE_STRADDLING)
        failed.update(straddling)
        reasons.append(
            f"the boundary spans both sides of {_plane_list(straddling)}; that is not a "
            "reduced domain about that plane"
        )
    if wrong_side:
        codes.append(FAILURE_WRONG_SIDE)
        failed.update(wrong_side)
        reasons.append(
            f"the boundary lies on the negative side of {_plane_list(wrong_side)}; the "
            "positive half is the one kept, so mirror the model first"
        )
    if missing_rim:
        codes.append(FAILURE_MISSING_RIM)
        failed.update(missing_rim)
        reasons.append(
            f"the reduced boundary is closed on {_plane_list(missing_rim)}; the cut plane "
            "is capped, not open"
        )
    if cap_faces:
        codes.append(FAILURE_CAPPED)
        failed.update(cap_faces)
        reasons.append(
            f"{sum(cap_faces.values())} triangles lie in the cut plane "
            f"{_plane_list(cap_faces)}; the cut plane is capped, not open"
        )
    if closed and len(off_plane):
        codes.append(FAILURE_LEAK)
        reasons.append(
            f"{len(off_plane)} free edges lie off the cut planes; the reduced domain leaks"
        )
    orientation = reduced_mesh_orientation(
        points, triangles, planes if closed else (), expected_volume_sign=expected_volume_sign
    )
    if closed and not codes and orientation["inverted_component_count"]:
        codes.append(FAILURE_INVERTED)
        reasons.append(
            f"{orientation['inverted_component_count']} of {orientation['component_count']} "
            "reduced mesh component(s) came back wound against the model they mirror, "
            "which reverses every normal the solver reads"
        )
    return result(
        straddling_planes=straddling,
        wrong_side_planes=wrong_side,
        missing_rim_planes=missing_rim,
        cap_face_counts={plane: cap_faces[plane] for plane in planes if plane in cap_faces},
        off_plane_open_edge_count=int(len(off_plane)),
        off_plane_open_edge_samples=samples,
        orientation=orientation,
        failure_codes=tuple(codes),
        failed_planes=tuple(plane for plane in planes if plane in failed),
        reason=reasons[0] if reasons else "every cut plane is open in the mesh",
    )


# -- plane snapping and naming -----------------------------------------------


def snap_symmetry_plane_vertices(
    points: np.ndarray,
    *,
    symmetry_planes: Iterable[str],
    tolerance: float,
) -> None:
    """Snap vertices inside a tolerance band exactly onto symmetry planes.

    Planes are named ``x0``/``y0``/``z0`` or, in the mesher's snap-axis
    spelling, ``x``/``y``/``z``.
    """
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    for plane in symmetry_planes:
        axis = SYMMETRY_AXIS_FOR_PLANE[PLANE_FOR_SNAP_AXIS.get(plane, plane)]
        mask = np.abs(points[:, axis]) <= tolerance
        points[mask, axis] = 0.0


_NATIVE_PLANE_FOR_SNAP_AXES = {("x", "y"): "yz+xz", ("y",): "xz", ("x",): "yz"}


def native_symmetry_plane_for_quadrants(quadrants: Any) -> str | None:
    """The metal solver's symmetry-plane flag for a ``Mesh.Quadrants`` value.

    Quadrant 1 mirrors about both the yz and xz planes, "12" about xz and
    "14" about yz; the full model ("1234") has none. The quadrant reading is
    the mesher's one ATH-compatible parser.
    """
    from .profile_common import _symmetry_planes_for_quadrants

    return _NATIVE_PLANE_FOR_SNAP_AXES.get(_symmetry_planes_for_quadrants(quadrants))


__all__ = [
    "DEFAULT_MAX_FAILURE_SAMPLES",
    "FAILURE_CAPPED",
    "FAILURE_INVERTED",
    "FAILURE_LEAK",
    "FAILURE_MISSING_RIM",
    "FAILURE_STRADDLING",
    "FAILURE_WRONG_SIDE",
    "GLOBAL_FAILURES",
    "ReducedMeshVerification",
    "SIGNED_VOLUME_NOISE_REL",
    "SYMMETRY_SLIVER_MAX_AREA_MM2",
    "SYMMETRY_SNAP_TOLERANCE_MM",
    "DEFAULT_SYMMETRY_SNAP_BAND_MM",
    "MirrorMatches",
    "MirrorTarget",
    "PlaneSymmetryVerdict",
    "PLANE_FOR_SNAP_AXIS",
    "RingMirrorTarget",
    "SPAN_TOLERANCE_REL",
    "SYMMETRY_AXIS_FOR_PLANE",
    "TOLERANCES",
    "ToleranceSpec",
    "VertexMirrorTarget",
    "detect_symmetry_planes",
    "edge_connected_components",
    "evaluate_enclosure_symmetry",
    "evaluate_mirror_symmetry",
    "evaluate_point_grid_symmetry",
    "native_symmetry_plane_for_quadrants",
    "reduced_mesh_orientation",
    "snap_symmetry_plane_vertices",
    "verify_reduced_mesh",
]
