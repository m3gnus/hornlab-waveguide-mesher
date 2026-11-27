"""One mirror-symmetry check for every input kind.

Whether a model is mirror-symmetric about a coordinate plane is decided here,
once, for every caller: imported OCC surface models (through the adapter in
:mod:`hornlab_mesher.step_prepare`), sampled parametric point grids, and the
analytic enclosure box. Each input supplies a :class:`MirrorTarget` -- the
exact distance oracle for its own representation -- and the one evaluator,
:func:`evaluate_mirror_symmetry`, applies the same rule to all of them: every
sample, mirrored through the plane, lands on the model within tolerance, on a
surface of the same caller role.

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
    "evaluate_enclosure_symmetry",
    "evaluate_mirror_symmetry",
    "evaluate_point_grid_symmetry",
    "native_symmetry_plane_for_quadrants",
    "snap_symmetry_plane_vertices",
]
