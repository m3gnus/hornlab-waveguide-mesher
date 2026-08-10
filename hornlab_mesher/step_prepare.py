"""Caller-neutral OCC preparation helpers for imported STEP surface models.

The caller owns every group name and role string.  This module only reasons
about OCC surface tags, mirrored geometry, and exact boolean parentage.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import gmsh
import numpy as np


SYMMETRY_AXIS_FOR_PLANE = {"x0": 0, "y0": 1, "z0": 2}
DEFAULT_AUTO_CUT_TOLERANCE_REL = 5.0e-4
DEFAULT_AUTO_CUT_GRID = 7
DEFAULT_SYMMETRY_SNAP_BAND_MM = 1.0e-4
AUTO_CUT_SPAN_REL = 1.0e-3
AUTO_CUT_TRIM_NUDGE_REL = 1.0e-3
AUTO_CUT_BOX_PAD_REL = 0.1
AUTO_CUT_MAX_FAILURE_SAMPLES = 5


@dataclass(frozen=True)
class OccSurfaceSelector:
    """An explicit selection of OCC surface entity tags."""

    surface_tags: tuple[int, ...]

    def __init__(self, surface_tags: Iterable[int]) -> None:
        object.__setattr__(self, "surface_tags", tuple(int(tag) for tag in surface_tags))


@dataclass(frozen=True)
class OccSurfaceRole:
    """An opaque role name supplied and interpreted only by the caller."""

    name: str


@dataclass(frozen=True)
class OccSurfaceGroup:
    """A caller-named surface selection carrying an opaque caller role."""

    name: str
    selector: OccSurfaceSelector
    role: OccSurfaceRole


@dataclass(frozen=True)
class PlaneSymmetryVerdict:
    """Per-plane result of the trimmed-face geometry mirror test."""

    plane: str
    accepted: bool
    reason: str
    tolerance: float
    axis_min: float
    axis_max: float
    points_tested: int
    points_off_model: int
    max_residual: float
    worst_residual_surface: int | None
    worst_off_model_distance: float
    worst_surface: int | None
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
class OccAutoCutResult:
    """Geometry, group, and diagnostic result of an OCC auto-cut."""

    planes: tuple[str, ...]
    groups: tuple[OccSurfaceGroup, ...]
    parent_to_children: dict[int, list[int]]
    report: dict[str, object]

    def group(self, name: str) -> OccSurfaceGroup:
        matches = [group for group in self.groups if group.name == name]
        if len(matches) != 1:
            raise KeyError(f"expected exactly one surface group named {name!r}")
        return matches[0]


def millimetres_to_step_units(value_mm: float, unit_scale_to_m: float) -> float:
    """Convert a physical millimetre tolerance to imported STEP units."""
    if unit_scale_to_m <= 0.0:
        raise ValueError("unit_scale_to_m must be positive")
    return float(value_mm) * 1.0e-3 / float(unit_scale_to_m)


def snap_symmetry_plane_vertices(
    points: np.ndarray,
    *,
    symmetry_planes: tuple[str, ...],
    tolerance: float,
) -> None:
    """Snap vertices inside a tolerance band exactly onto symmetry planes."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    for plane in symmetry_planes:
        axis = SYMMETRY_AXIS_FOR_PLANE[plane]
        mask = np.abs(points[:, axis]) <= tolerance
        points[mask, axis] = 0.0


def remap_surface_tags(
    surfaces: Iterable[int], parent_to_children: dict[int, list[int]]
) -> list[int]:
    """Follow surface tags through an OCC boolean parent-to-children map."""
    remapped: list[int] = []
    seen: set[int] = set()
    for surface in surfaces:
        for child in parent_to_children.get(int(surface), []):
            if child not in seen:
                seen.add(child)
                remapped.append(child)
    return remapped


def _param_inside(surface: int, u: float, v: float) -> bool:
    try:
        return int(gmsh.model.isInside(2, surface, [u, v], parametric=True)) > 0
    except Exception:
        return True


def sample_occ_surface_points(surface: int, *, grid: int) -> np.ndarray:
    """Point-sample the actual trimmed OCC face, not its untrimmed extension."""
    if grid < 2:
        raise ValueError("grid must be at least 2")
    try:
        bounds_min, bounds_max = gmsh.model.getParametrizationBounds(2, surface)
        umin, vmin = float(bounds_min[0]), float(bounds_min[1])
        umax, vmax = float(bounds_max[0]), float(bounds_max[1])
    except Exception:
        umin = umax = vmin = vmax = 0.0
    if umax > umin and vmax > vmin:
        for factor in (1, 4):
            n = max(2, grid * factor)
            kept: list[float] = []
            for i in range(n):
                u = umin + (i + 0.5) * (umax - umin) / n
                for j in range(n):
                    v = vmin + (j + 0.5) * (vmax - vmin) / n
                    if _param_inside(surface, u, v):
                        kept.extend((u, v))
            if kept:
                coords = gmsh.model.getValue(2, surface, kept)
                return np.asarray(coords, dtype=float).reshape(-1, 3)
    return _sample_occ_surface_boundary_points(surface, grid=grid)


def _sample_occ_surface_boundary_points(surface: int, *, grid: int) -> np.ndarray:
    points: list[float] = []
    try:
        boundary = gmsh.model.getBoundary(
            [(2, surface)], combined=False, oriented=False, recursive=False
        )
    except Exception:
        boundary = []
    for dim, curve in boundary:
        if dim != 1:
            continue
        try:
            bounds_min, bounds_max = gmsh.model.getParametrizationBounds(1, abs(curve))
            tmin, tmax = float(bounds_min[0]), float(bounds_max[0])
            if not tmax > tmin:
                continue
            params = [tmin + (i + 0.5) * (tmax - tmin) / grid for i in range(grid)]
            points.extend(gmsh.model.getValue(1, abs(curve), params))
        except Exception:
            continue
    if not points:
        points = [float(value) for value in gmsh.model.occ.getCenterOfMass(2, surface)]
    return np.asarray(points, dtype=float).reshape(-1, 3)


def _parametric_point_on_face(surface: int, parametric: Iterable[float]) -> bool:
    par = [float(value) for value in parametric]
    if len(par) < 2:
        return False
    u, v = par[0], par[1]
    if _param_inside(surface, u, v):
        return True
    try:
        bounds_min, bounds_max = gmsh.model.getParametrizationBounds(2, surface)
    except Exception:
        return False
    umin, vmin = float(bounds_min[0]), float(bounds_min[1])
    umax, vmax = float(bounds_max[0]), float(bounds_max[1])
    du = AUTO_CUT_TRIM_NUDGE_REL * (umax - umin)
    dv = AUTO_CUT_TRIM_NUDGE_REL * (vmax - vmin)
    umid, vmid = 0.5 * (umin + umax), 0.5 * (vmin + vmax)
    u2 = min(max(u + (du if umid >= u else -du), umin), umax)
    v2 = min(max(v + (dv if vmid >= v else -dv), vmin), vmax)
    return _param_inside(surface, u2, v2)


def _closest_on_model(
    point: np.ndarray,
    *,
    candidates: list[int],
    bboxes: dict[int, tuple[float, ...]],
    tolerance: float,
) -> tuple[float, float, list[int]]:
    matched_distance = -1.0
    nearest = -1.0
    matched: list[int] = []
    px, py, pz = (float(value) for value in point)
    for surface in candidates:
        bbox = bboxes.get(surface)
        if bbox is None:
            continue
        gap = float(
            np.linalg.norm(
                [
                    max(bbox[0] - px, px - bbox[3], 0.0),
                    max(bbox[1] - py, py - bbox[4], 0.0),
                    max(bbox[2] - pz, pz - bbox[5], 0.0),
                ]
            )
        )
        if gap > tolerance:
            if nearest < 0.0 or gap < nearest:
                nearest = gap
            continue
        try:
            closest, parametric = gmsh.model.getClosestPoint(2, surface, [px, py, pz])
        except Exception:
            continue
        distance = float(np.linalg.norm(np.asarray(closest) - np.asarray([px, py, pz])))
        if nearest < 0.0 or distance < nearest:
            nearest = distance
        if distance > tolerance or not _parametric_point_on_face(surface, parametric):
            continue
        matched.append(surface)
        if matched_distance < 0.0 or distance < matched_distance:
            matched_distance = distance
    return matched_distance, nearest, matched


def evaluate_occ_plane_symmetry(
    plane: str,
    *,
    samples: dict[int, np.ndarray],
    roles: dict[int, str],
    bboxes: dict[int, tuple[float, ...]],
    tolerance: float,
    span_tolerance: float,
) -> PlaneSymmetryVerdict:
    """Require both trimmed geometry and caller-supplied roles to mirror."""
    axis = SYMMETRY_AXIS_FOR_PLANE[plane]
    candidates = sorted(samples)
    missing_roles = set(candidates) - set(roles)
    if missing_roles:
        raise ValueError(f"missing roles for OCC surfaces: {sorted(missing_roles)}")
    all_axis_values = (
        np.concatenate([samples[surface][:, axis] for surface in candidates])
        if candidates
        else np.zeros(1)
    )
    axis_min = float(np.min(all_axis_values))
    axis_max = float(np.max(all_axis_values))
    points_tested = int(sum(len(samples[surface]) for surface in candidates))
    if not (axis_min < -span_tolerance and axis_max > span_tolerance):
        return PlaneSymmetryVerdict(
            plane, False,
            "model does not straddle the plane; there is no half to remove",
            tolerance, axis_min, axis_max, points_tested, 0, -1.0, None, -1.0,
            None, 0, (),
        )

    max_residual = -1.0
    max_residual_surface: int | None = None
    points_off_model = 0
    role_mismatches = 0
    worst_off_model = -1.0
    worst_surface: int | None = None
    failures: list[dict[str, object]] = []
    for surface in candidates:
        role = roles[surface]
        for point in samples[surface]:
            mirrored = point.copy()
            mirrored[axis] = -mirrored[axis]
            matched_distance, nearest, hits = _closest_on_model(
                mirrored, candidates=candidates, bboxes=bboxes, tolerance=tolerance
            )
            if not hits:
                points_off_model += 1
                if nearest > worst_off_model:
                    worst_off_model = nearest
                    worst_surface = surface
                if len(failures) < AUTO_CUT_MAX_FAILURE_SAMPLES:
                    failures.append(
                        {
                            "kind": "off_model",
                            "surface": int(surface),
                            "role": role,
                            "point_step_units": [float(value) for value in point],
                            "mirrored_point_step_units": [
                                float(value) for value in mirrored
                            ],
                            "nearest_distance_step_units": (
                                None if nearest < 0.0 else float(nearest)
                            ),
                        }
                    )
                continue
            if matched_distance > max_residual:
                max_residual = matched_distance
                max_residual_surface = int(surface)
            if role not in {roles[hit] for hit in hits}:
                role_mismatches += 1
                if len(failures) < AUTO_CUT_MAX_FAILURE_SAMPLES:
                    failures.append(
                        {
                            "kind": "role_mismatch",
                            "surface": int(surface),
                            "role": role,
                            "point_step_units": [float(value) for value in point],
                            "mirrored_point_step_units": [
                                float(value) for value in mirrored
                            ],
                            "mirrored_surfaces": [int(hit) for hit in hits],
                            "mirrored_roles": sorted({roles[hit] for hit in hits}),
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


def _cut_occ_geometry_to_positive_side(
    planes: tuple[str, ...],
    *,
    bbox: tuple[float, float, float, float, float, float],
) -> tuple[dict[int, list[int]], dict[str, object]]:
    volumes = gmsh.model.getEntities(3)
    if volumes:
        gmsh.model.occ.remove(volumes, recursive=False)
        gmsh.model.occ.synchronize()
    surfaces = [tag for _dim, tag in sorted(gmsh.model.getEntities(2))]
    span = max(bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])
    pad = AUTO_CUT_BOX_PAD_REL * span + 1.0
    lo = [bbox[0] - pad, bbox[1] - pad, bbox[2] - pad]
    hi = [bbox[3] + pad, bbox[4] + pad, bbox[5] + pad]
    for plane in planes:
        lo[SYMMETRY_AXIS_FOR_PLANE[plane]] = 0.0
    box = gmsh.model.occ.addBox(
        lo[0], lo[1], lo[2], hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
    )
    gmsh.model.occ.synchronize()
    _out, out_map = gmsh.model.occ.intersect(
        [(2, surface) for surface in surfaces],
        [(3, box)],
        removeObject=True,
        removeTool=True,
    )
    gmsh.model.occ.synchronize()
    mapping = {
        surface: [tag for dim, tag in (out_map[index] if index < len(out_map) else []) if dim == 2]
        for index, surface in enumerate(surfaces)
    }
    remaining = [tag for _dim, tag in sorted(gmsh.model.getEntities(2))]
    stats = {
        "planes": list(planes),
        "half_space_box_step_units": [float(value) for value in lo + hi],
        "surfaces_before": len(surfaces),
        "surfaces_after": len(remaining),
        "surfaces_dropped": sum(not children for children in mapping.values()),
        "surfaces_split": sum(len(children) > 1 for children in mapping.values()),
        "residual_volumes": len(gmsh.model.getEntities(3)),
    }
    return mapping, stats


def auto_cut_occ_geometry(
    groups: Iterable[OccSurfaceGroup],
    *,
    grid: int = DEFAULT_AUTO_CUT_GRID,
    tolerance_rel: float = DEFAULT_AUTO_CUT_TOLERANCE_REL,
) -> OccAutoCutResult:
    """Mirror-test explicit caller groups and open-cut all accepted planes.

    Group names and roles are never interpreted. Every selected surface must
    occur exactly once so role symmetry cannot silently use a default role.
    """
    groups = tuple(groups)
    if not groups:
        raise ValueError("at least one OCC surface group is required")
    roles: dict[int, str] = {}
    for group in groups:
        for surface in group.selector.surface_tags:
            if surface in roles:
                raise ValueError(f"surface {surface} occurs in more than one group")
            roles[surface] = group.role.name
    surfaces = list(roles)
    samples = {
        surface: sample_occ_surface_points(surface, grid=grid) for surface in surfaces
    }
    samples = {surface: points for surface, points in samples.items() if len(points)}
    bboxes = {
        surface: tuple(float(value) for value in gmsh.model.getBoundingBox(2, surface))
        for surface in samples
    }
    if not samples:
        report = {
            "mode": "auto-cut",
            "cut_planes": [],
            "note": "no sampleable selected surfaces; auto reduction skipped",
        }
        return OccAutoCutResult((), groups, {}, report)
    stacked = np.concatenate(list(samples.values()))
    sample_bbox = tuple(
        float(value)
        for value in (*np.min(stacked, axis=0), *np.max(stacked, axis=0))
    )
    diagonal = float(np.linalg.norm(np.asarray(sample_bbox[3:]) - sample_bbox[:3]))
    tolerance = tolerance_rel * diagonal
    span_tolerance = AUTO_CUT_SPAN_REL * diagonal
    verdicts = [
        evaluate_occ_plane_symmetry(
            plane,
            samples=samples,
            roles=roles,
            bboxes=bboxes,
            tolerance=tolerance,
            span_tolerance=span_tolerance,
        )
        for plane in SYMMETRY_AXIS_FOR_PLANE
    ]
    planes = tuple(verdict.plane for verdict in verdicts if verdict.accepted)
    report: dict[str, object] = {
        "mode": "auto-cut",
        "tolerance_rel": float(tolerance_rel),
        "tolerance_step_units": float(tolerance),
        "model_diagonal_step_units": diagonal,
        "sample_grid": int(grid),
        "sampled_surfaces": len(samples),
        "sample_points": int(len(stacked)),
        "planes": {verdict.plane: verdict.to_dict() for verdict in verdicts},
        "cut_planes": list(planes),
    }
    if not planes:
        report["cut"] = None
        return OccAutoCutResult((), groups, {}, report)
    model_bbox = tuple(float(value) for value in gmsh.model.getBoundingBox(-1, -1))
    parentage, cut_stats = _cut_occ_geometry_to_positive_side(planes, bbox=model_bbox)
    remapped_groups = tuple(
        OccSurfaceGroup(
            group.name,
            OccSurfaceSelector(remap_surface_tags(group.selector.surface_tags, parentage)),
            group.role,
        )
        for group in groups
    )
    report["cut"] = cut_stats
    return OccAutoCutResult(planes, remapped_groups, parentage, report)
