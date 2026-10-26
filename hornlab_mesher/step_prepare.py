"""Caller-neutral OCC preparation helpers for imported STEP surface models.

The caller owns every group name and role string.  This module only reasons
about OCC surface tags, mirrored geometry, and exact boolean parentage.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


class _LazyGmsh:
    """Import gmsh on first use, never at module import.

    Every gmsh import in this package is lazy by contract: importing
    hornlab_mesher must stay numpy-only. Eagerly importing gmsh here broke two
    things at once when this module joined the package's eager exports — grid
    resolution stopped being importable without gmsh, and gmsh's C++ runtime
    installs its own signal handlers on load, which on Linux rearms SIGPIPE's
    default action and turned a closed-socket write in a WG v2 test into a
    process kill (pytest exit 141 on ubuntu CI). The first attribute access
    imports the real module and replaces this shim in the module globals, so
    the indirection costs one lookup, once.
    """

    def __getattr__(self, name: str):
        import gmsh as _gmsh_module

        globals()["gmsh"] = _gmsh_module
        return getattr(_gmsh_module, name)


gmsh = _LazyGmsh()


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
    # Scale factor first: for millimetre coordinates it is exactly 1.0, so a
    # millimetre value comes back bit-for-bit unchanged.
    return float(value_mm) * (1.0e-3 / float(unit_scale_to_m))


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
        # Trim membership is evidence for accepting a symmetry cut. If OCC
        # cannot answer, treating uncertainty as "inside" lets a failed sample
        # help prove symmetry and can remove real geometry. Abstain instead.
        return False


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
    nearest_contributions: dict[int, float] | None = None,
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
            if nearest_contributions is not None:
                nearest_contributions[surface] = gap
            if nearest < 0.0 or gap < nearest:
                nearest = gap
            continue
        try:
            closest, parametric = gmsh.model.getClosestPoint(2, surface, [px, py, pz])
        except Exception:
            continue
        distance = float(np.linalg.norm(np.asarray(closest) - np.asarray([px, py, pz])))
        if nearest_contributions is not None:
            nearest_contributions[surface] = distance
        if nearest < 0.0 or distance < nearest:
            nearest = distance
        if distance > tolerance or not _parametric_point_on_face(surface, parametric):
            continue
        matched.append(surface)
        if matched_distance < 0.0 or distance < matched_distance:
            matched_distance = distance
    return matched_distance, nearest, matched


class _OccMirrorIndex:
    """Conservative broad phase for the existing ordered face scan.

    All boxes within tolerance must be visited, even after a zero-distance
    hit: coincident faces can carry different caller roles. Farther boxes
    contribute only their gap to ``nearest`` in the original scan. They
    cannot change it once a local candidate gives a distance <= tolerance;
    otherwise we recover the omitted gap diagnostics in original order.
    """

    def __init__(self, candidates, bboxes, tolerance):
        self.candidates = candidates
        self.bboxes = bboxes
        self.tolerance = tolerance
        self.tree = None
        self.tags = [tag for tag in candidates if tag in bboxes]
        if not self.tags or not np.isfinite(tolerance) or tolerance < 0:
            return
        boxes = np.asarray([bboxes[tag] for tag in self.tags], dtype=float)
        if not np.all(np.isfinite(boxes)) or np.any(boxes[:, :3] > boxes[:, 3:]):
            return
        centers = boxes[:, :3] * 0.5 + boxes[:, 3:] * 0.5
        half_width = float(np.max(np.maximum(
            centers - boxes[:, :3], boxes[:, 3:] - centers
        )))
        self.scale = max(float(np.max(np.abs(boxes))), 1.0)
        self.boxes = boxes
        self.radius = half_width + tolerance
        if not np.isfinite(self.radius):
            return
        # Lazy by package contract. Other facet indices use triangle overlap
        # predicates and geometric margins; neither applies to this scan.
        from scipy.spatial import cKDTree

        self.tree = cKDTree(centers)

    def candidates_for_points(self, points):
        if self.tree is None or not np.all(np.isfinite(points)):
            return [self.candidates] * len(points)
        # Outward padding is only a broad-phase roundoff guard. The original
        # scalar norm below still decides membership at the unchanged tolerance.
        scale = np.maximum(np.max(np.abs(points), axis=1), self.scale)
        guard = 16 * np.finfo(float).eps * (scale + self.radius)
        radii = np.nextafter(self.radius + guard, np.inf)
        try:
            rows = self.tree.query_ball_point(points, radii, p=np.inf, eps=0)
        except ValueError:
            # Finite extreme coordinates can overflow tree distance bounds.
            # The scalar scan already defines the diagnostic for those inputs.
            return [self.candidates] * len(points)
        nearby = []
        for point, row, margin in zip(points, rows, guard):
            indices = np.asarray(sorted(row), dtype=int)
            boxes = self.boxes[indices]
            # A large face can inflate the tree radius. Reject its broad-phase
            # false positives in bulk, conservatively on each coordinate. The
            # scalar Euclidean norm remains the final membership predicate.
            keep = np.all(np.maximum(boxes[:, :3] - point, point - boxes[:, 3:])
                          <= self.tolerance + margin, axis=1)
            nearby.append([self.tags[index] for index in indices[keep]])
        return nearby

    def closest(self, point, candidates):
        contributions: dict[int, float] = {}
        result = _closest_on_model(
            point, candidates=candidates, bboxes=self.bboxes, tolerance=self.tolerance,
            nearest_contributions=contributions,
        )
        if candidates is self.candidates or 0 <= result[1] <= self.tolerance:
            return result
        # Never retry a projection or trim query: an OCC exception must remain
        # a failed sample, even if a second call would succeed. Omitted boxes
        # are beyond tolerance, so the existing scan reads only their gaps.
        # Reduce all contributions in original order, including NaN behavior.
        local = set(candidates)
        nearest = -1.0
        for surface in self.candidates:
            if surface in local:
                if surface not in contributions:
                    continue
                distance = contributions[surface]
            else:
                distance = _closest_on_model(
                    point, candidates=[surface], bboxes=self.bboxes,
                    tolerance=self.tolerance,
                )[1]
                if distance < 0.0:
                    continue
            if nearest < 0.0 or distance < nearest:
                nearest = distance
        return result[0], nearest, result[2]


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
    index = _OccMirrorIndex(candidates, bboxes, tolerance)
    for surface in candidates:
        role = roles[surface]
        mirrored_points = samples[surface].copy()
        mirrored_points[:, axis] *= -1
        nearby = index.candidates_for_points(mirrored_points)
        for point, mirrored, local in zip(samples[surface], mirrored_points, nearby):
            matched_distance, nearest, hits = index.closest(mirrored, local)
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


def _claim_evidence(parent: int, child: int, *, tolerance: float, grid: int) -> str:
    """Whether a cut child lies on a copy of the parent it was reported from.

    ``"off"`` is proof that it does not: a sample of the child is further than
    ``tolerance`` from the parent's carrier surface, or lies on the carrier
    outside the parent's trim and further than ``tolerance`` from its trim
    boundary. A real piece of the parent can do neither. ``"on"`` means every
    sample lies on the trimmed parent; ``"unknown"`` means the geometry could
    not decide.
    """
    points = sample_occ_surface_points(child, grid=grid)
    if not len(points):
        return "unknown"
    undecided = False
    for point in points:
        try:
            closest, parametric = gmsh.model.getClosestPoint(2, parent, list(point))
        except Exception:
            undecided = True
            continue
        if float(np.linalg.norm(np.asarray(closest) - point)) > tolerance:
            return "off"
        try:
            inside = int(gmsh.model.isInside(2, parent, list(parametric[:2]), parametric=True)) > 0
        except Exception:
            undecided = True
            continue
        if inside:
            continue
        edge = _distance_to_boundary(parent, point)
        if edge is not None and edge > tolerance:
            return "off"
        undecided = True
    return "unknown" if undecided else "on"


def _distance_to_boundary(surface: int, point: np.ndarray) -> float | None:
    try:
        curves = [
            abs(tag)
            for dim, tag in gmsh.model.getBoundary(
                [(2, surface)], combined=False, oriented=False, recursive=False
            )
            if dim == 1
        ]
        distances = [
            float(np.linalg.norm(np.asarray(gmsh.model.getClosestPoint(1, curve, list(point))[0]) - point))
            for curve in curves
        ]
    except Exception:
        return None
    return min(distances) if distances else None


def _verify_cut_parentage(
    mapping: dict[int, list[int]],
    copies: dict[int, int],
    remaining: list[int],
    *,
    tolerance: float,
    grid: int,
) -> tuple[dict[int, list[int]], set[tuple[int, int]], int]:
    """Keep only the out_map claims the geometry does not disprove.

    Gmsh (4.15) frees the tags of removed objects before numbering the new
    pieces of a split surface, so a new piece can take a removed surface's
    tag, and out_map then reports the removed surface as kept under it. Tag
    numbers and bounding boxes cannot tell that from real shared ancestry --
    overlapping or coincident inputs share children -- so each claim is
    checked against a copy of its parent taken before the cut. A claim is
    dropped only on proof; one the geometry cannot decide is kept and
    returned, so the caller can refuse it where it would matter. A surviving
    surface left with no parent is refused.
    """
    cleaned: dict[int, list[int]] = {}
    undecided: set[tuple[int, int]] = set()
    disproved = 0
    for parent, children in mapping.items():
        kept: list[int] = []
        for child in children:
            verdict = _claim_evidence(copies[parent], child, tolerance=tolerance, grid=grid)
            if verdict == "off":
                disproved += 1
                continue
            if verdict == "unknown":
                undecided.add((parent, child))
            kept.append(child)
        cleaned[parent] = kept
    claimed = {child for children in cleaned.values() for child in children}
    unclaimed = sorted(set(remaining) - claimed)
    if unclaimed:
        raise RuntimeError(f"OCC cut left surfaces {unclaimed} with no parent")
    return cleaned, undecided, disproved


def _remove_temporary_geometry(copies: Iterable[int], box: int | None) -> None:
    """Remove cut faces and the optional clipping box if they still exist."""
    present = {(dim, tag) for dim, tag in gmsh.model.occ.getEntities()}
    doomed = [(2, tag) for tag in copies if (2, tag) in present]
    if box is not None and (3, box) in present:
        doomed.append((3, box))
    if doomed:
        faces = [tag for dim, tag in doomed if dim == 2]
        # OCC's recursive removal of a list visits each face separately. A
        # temporary shell makes the same faces one removal operation, while
        # OCC still preserves boundaries used by surviving entities. Some
        # face combinations cannot form a shell; retain the original removal
        # for those combinations.
        container = None
        if len(faces) > 1:
            try:
                shell = gmsh.model.occ.addSurfaceLoop(faces, sewing=False)
                container = gmsh.model.occ.addVolume([shell])
            except Exception:
                # An OCC call can fail after creating an entity. Release any
                # new volume before falling back to the original cleanup.
                created = [item for item in gmsh.model.occ.getEntities(3)
                           if item not in present]
                if created:
                    gmsh.model.occ.remove(created, recursive=False)
                container = None
        if container is None:
            gmsh.model.occ.remove(doomed, recursive=True)
        else:
            try:
                gmsh.model.occ.remove(
                    [(3, container), *[item for item in doomed if item[0] == 3]],
                    recursive=True,
                )
                # A shell can be created even when OCC will not recursively
                # remove every face in it. Finish with the original operation
                # for any entities that remain.
                present = set(gmsh.model.occ.getEntities())
                if (3, container) in present:
                    gmsh.model.occ.remove([(3, container)], recursive=False)
                    present = set(gmsh.model.occ.getEntities())
                remaining = [item for item in doomed if item in present]
                if remaining:
                    gmsh.model.occ.remove(remaining, recursive=True)
            except Exception:
                # Do not leave the auxiliary volume holding copies if removal
                # fails. Preserve the original exception after best effort.
                try:
                    gmsh.model.occ.remove([(3, container)], recursive=False)
                    gmsh.model.occ.remove(doomed, recursive=True)
                except Exception:
                    pass
                raise
    gmsh.model.occ.synchronize()


def _uncovered_cut_parents(
    mapping: dict[int, list[int]],
    copies: dict[int, int],
    planes: tuple[str, ...],
    *,
    tolerance: float,
    grid: int,
) -> list[int]:
    """Check coverage in the other direction: retained parent onto children.

    Validating child ancestry alone cannot detect a face silently discarded
    by OCC. Use trimmed interior samples, clear of the clipping planes, so
    discarded faces and partially missing parents both trigger recovery.
    """
    axes = [SYMMETRY_AXIS_FOR_PLANE[plane] for plane in planes]
    missing = []
    for parent, copy in copies.items():
        points = sample_occ_surface_points(copy, grid=grid)
        points = points[np.all(points[:, axes] > tolerance, axis=1)]
        children = mapping[parent]
        bboxes = {
            child: tuple(gmsh.model.getBoundingBox(2, child)) for child in children
        }
        if any(
            not _closest_on_model(
                point, candidates=children, bboxes=bboxes, tolerance=tolerance
            )[2]
            for point in points
        ):
            missing.append(parent)
    return missing


def _align_cut_boundary_vertices(
    surfaces: list[int], planes: tuple[str, ...], *, tolerance: float, noise: float
) -> int:
    """Align tolerance-matched endpoints with their planar section carriers.

    OCC can retain an old vertex slightly outside the half-space while the
    new section curve lies exactly on the plane. Correct that CAD
    point before meshing; merely widening a mesh snap band would conceal it.
    Check interior positions and tangents at floating-point precision. OCC
    stores even straight sections of converted cylinders as B-splines, so
    the entity type alone cannot establish whether a section is planar.
    """
    aligned = set()
    for _dim, curve in gmsh.model.getBoundary(
        [(2, tag) for tag in surfaces], combined=True, oriented=False
    ):
        vertices = gmsh.model.getBoundary([(1, curve)], oriented=False)
        if not vertices:
            continue
        lo, hi = gmsh.model.getParametrizationBounds(1, curve)
        # At the parameter endpoints Gmsh can return the tolerance-matched
        # topological vertices instead of evaluating the curve's carrier.
        params = np.linspace(lo[0], hi[0], 11)[1:-1].tolist()
        carrier = np.asarray(gmsh.model.getValue(1, curve, params)).reshape(-1, 3)
        tangent = np.asarray(gmsh.model.getDerivative(1, curve, params)).reshape(-1, 3)
        for plane in planes:
            axis = SYMMETRY_AXIS_FOR_PLANE[plane]
            if (np.max(np.abs(carrier[:, axis])) > noise
                    or np.max(np.abs(tangent[:, axis])) * abs(hi[0] - lo[0]) > noise):
                continue
            for _dim, vertex in vertices:
                point = list(gmsh.model.getValue(0, vertex, []))
                if noise < abs(point[axis]) <= tolerance:
                    point[axis] = 0.0
                    gmsh.model.setCoordinates(vertex, *point)
                    aligned.add(vertex)
    return len(aligned)


@dataclass(frozen=True)
class _OccTriangleModel:
    """Validated triangle faces and their shared straight edges."""

    vertices: dict[int, np.ndarray]
    face_edges: dict[int, tuple[int, ...]]
    edge_vertices: dict[int, tuple[int, int]]
    edge_faces: dict[int, tuple[int, ...]]
    face_vertices: dict[int, tuple[int, int, int]]


def _triangle_model(surfaces: list[int]) -> _OccTriangleModel:
    """Prove the opt-in model consists of nonintersecting straight facets.

    A face must be one planar triangle with three straight topological edges.
    Intersections other than shared vertices and edges make it unsafe to keep
    or discard an input face independently of the boolean. In that case the
    caller must use the general OCC path instead.
    """
    face_edges: dict[int, tuple[int, ...]] = {}
    edge_vertices: dict[int, tuple[int, int]] = {}
    edge_faces: dict[int, list[int]] = {}
    face_vertices: dict[int, tuple[int, int, int]] = {}
    vertices: dict[int, np.ndarray] = {}
    for surface in surfaces:
        if gmsh.model.getType(2, surface) != "Plane":
            raise ValueError("triangle surface model contains a nonplanar face")
        boundary = gmsh.model.getBoundary(
            [(2, surface)], combined=False, oriented=False, recursive=False
        )
        edges = tuple(abs(tag) for dim, tag in boundary if dim == 1)
        if len(edges) != 3 or len(set(edges)) != 3:
            raise ValueError("triangle surface model contains a nontriangle face")
        face_edges[surface] = edges
        degree: dict[int, int] = {}
        for edge in edges:
            edge_faces.setdefault(edge, []).append(surface)
            if edge not in edge_vertices:
                if gmsh.model.getType(1, edge) != "Line":
                    raise ValueError("triangle surface model contains a curved edge")
                ends = gmsh.model.getBoundary([(1, edge)], oriented=False)
                tags = tuple(abs(tag) for dim, tag in ends if dim == 0)
                if len(tags) != 2 or tags[0] == tags[1]:
                    raise ValueError("triangle surface model has an invalid edge")
                edge_vertices[edge] = tags
                for tag in tags:
                    if tag not in vertices:
                        vertices[tag] = np.asarray(gmsh.model.getValue(0, tag, []), dtype=float)
            for vertex in edge_vertices[edge]:
                degree[vertex] = degree.get(vertex, 0) + 1
        if len(degree) != 3 or any(value != 2 for value in degree.values()):
            raise ValueError("triangle surface model has an invalid face boundary")
        face_vertices[surface] = tuple(sorted(degree))
    if any(len(faces) > 2 for faces in edge_faces.values()):
        raise ValueError("triangle surface model has a nonmanifold edge")
    if not vertices or any(not np.all(np.isfinite(point)) for point in vertices.values()):
        raise ValueError("triangle surface model has invalid vertices")
    facets = np.asarray([[vertices[v] for v in face_vertices[s]] for s in surfaces])
    lengths = np.linalg.norm(facets - np.roll(facets, 1, axis=1), axis=2)
    areas = np.linalg.norm(np.cross(facets[:, 1] - facets[:, 0],
                                    facets[:, 2] - facets[:, 0]), axis=1)
    if (not np.all(np.isfinite(lengths)) or not np.all(np.isfinite(areas))
            or np.any(lengths <= 0) or np.any(areas <= 0)):
        raise ValueError("triangle surface model has a degenerate face")
    origin = facets.min(axis=(0, 1))
    scale = float(np.max(facets.max(axis=(0, 1)) - origin))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("triangle surface model has invalid extent")
    # Reuse the existing conservative facet intersection and contact checks.
    # Normalize so their numerical guards apply independently of model units.
    from .shell_facets import BoundaryIndex, _contact_only

    normalized = (facets - origin) / scale
    for first, second in BoundaryIndex(normalized).hits(normalized):
        if first < second and not _contact_only(normalized[first], normalized[second]):
            raise ValueError("triangle surface model contains intersecting faces")
    return _OccTriangleModel(vertices, face_edges, edge_vertices,
                             {edge: tuple(faces) for edge, faces in edge_faces.items()},
                             face_vertices)


def _cut_occ_geometry_to_positive_side(
    planes: tuple[str, ...],
    *,
    bbox: tuple[float, float, float, float, float, float],
    tolerance: float,
    grid: int,
    triangle_model: _OccTriangleModel | None = None,
) -> tuple[dict[int, list[int]], set[tuple[int, int]], dict[str, object]]:
    volumes = gmsh.model.getEntities(3)
    if volumes:
        gmsh.model.occ.remove(volumes, recursive=False)
        gmsh.model.occ.synchronize()
    surfaces = [tag for _dim, tag in sorted(gmsh.model.getEntities(2))]
    inside: set[int] = set()
    outside: set[int] = set()
    cut_parents = surfaces
    if triangle_model is not None:
        axes = [SYMMETRY_AXIS_FOR_PLANE[plane] for plane in planes]
        boxes = {tag: tuple(gmsh.model.getBoundingBox(2, tag)) for tag in surfaces}
        if any(not np.all(np.isfinite(box)) for box in boxes.values()):
            raise ValueError("triangle surface model has invalid face bounds")
        outside = {tag for tag in surfaces if any(boxes[tag][axis + 3] < 0 for axis in axes)}
        inside = {tag for tag in surfaces if all(boxes[tag][axis] > 0 for axis in axes)}
        cut_parents = [tag for tag in surfaces if tag not in inside and tag not in outside]
    span = max(bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])
    pad = AUTO_CUT_BOX_PAD_REL * span + 1.0
    lo = [bbox[0] - pad, bbox[1] - pad, bbox[2] - pad]
    hi = [bbox[3] + pad, bbox[4] + pad, bbox[5] + pad]
    for plane in planes:
        lo[SYMMETRY_AXIS_FOR_PLANE[plane]] = 0.0
    # The cut deletes the originals; their parentage claims are checked
    # against these copies. The copies and the clipping box are temporary,
    # and must not outlive this call on any path -- a leftover copy would be
    # meshed as part of the model.
    copies: dict[int, int] = {}
    retry_copies: list[int] = []
    box: int | None = None
    try:
        if triangle_model is None:
            for surface in surfaces:
                copies[surface] = gmsh.model.occ.copy([(2, surface)])[0][1]
        elif cut_parents:
            before = set(gmsh.model.occ.getEntities(2))
            try:
                copied = gmsh.model.occ.copy([(2, surface) for surface in cut_parents])
                if len(copied) != len(cut_parents) or any(dim != 2 for dim, _ in copied):
                    raise RuntimeError("OCC returned incomplete triangle face copies")
                copy_tags = [tag for _, tag in copied]
                if len(set(copy_tags)) != len(cut_parents):
                    raise RuntimeError("OCC returned duplicate triangle face copies")
                created = set(gmsh.model.occ.getEntities(2)) - before
                if created != {(2, tag) for tag in copy_tags}:
                    raise RuntimeError("OCC returned unexpected triangle face copies")
                copies = dict(zip(cut_parents, copy_tags))
            except Exception:
                leaked = [tag for dim, tag in gmsh.model.occ.getEntities(2)
                          if (dim, tag) not in before]
                _remove_temporary_geometry(leaked, None)
                raise
        gmsh.model.occ.synchronize()
        cut_surfaces = cut_parents
        repaired: list[int] = []
        mapping: dict[int, list[int]] = {}
        undecided: set[tuple[int, int]] = set()
        disproved = 0
        remaining: list[int] = []
        for attempt in range(2) if cut_surfaces else ():
            box = gmsh.model.occ.addBox(
                lo[0], lo[1], lo[2], hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
            )
            gmsh.model.occ.synchronize()
            _out, out_map = gmsh.model.occ.intersect(
                [(2, surface) for surface in cut_surfaces],
                [(3, box)],
                removeObject=True,
                removeTool=True,
            )
            box = None
            retry_copies = []  # consumed; their tags can now name real children
            gmsh.model.occ.synchronize()
            mapping = {
                surface: [tag for dim, tag in (out_map[index] if index < len(out_map) else []) if dim == 2]
                for index, surface in enumerate(cut_parents)
            }
            copy_tags = set(copies.values())
            remaining = [
                tag for _dim, tag in sorted(gmsh.model.getEntities(2))
                if tag not in copy_tags and tag not in inside and tag not in outside
            ]
            mapping, undecided, disproved = _verify_cut_parentage(
                mapping, copies, remaining, tolerance=tolerance, grid=grid
            )
            uncovered = _uncovered_cut_parents(
                mapping, copies, planes, tolerance=tolerance, grid=grid
            )
            if not uncovered:
                break
            if attempt:
                raise RuntimeError(f"OCC cut lost retained portions of surfaces {uncovered}")
            # OCC can lose a periodic analytic face when its seam meets the
            # clipping plane. Converting that carrier to rational B-splines
            # avoids the periodic boolean path without approximating the
            # surface. Retry the whole boundary together to share cut edges;
            # adding a repaired face alone would leave a disconnected mesh.
            repaired = uncovered
            if triangle_model is None:
                gmsh.model.occ.remove([(2, tag) for tag in remaining], recursive=True)
            else:
                _remove_temporary_geometry(remaining, None)
            retry_copies = [
                tag for _dim, tag in gmsh.model.occ.copy(
                    [(2, copies[parent]) for parent in cut_parents]
                )
            ]
            gmsh.model.occ.convertToNURBS([
                (2, retry_copies[index])
                for index, parent in enumerate(cut_parents) if parent in uncovered
            ])
            cut_surfaces = retry_copies

        if triangle_model is not None:
            expected = inside | set(remaining)
            _remove_temporary_geometry([*sorted(outside), *copies.values()], None)
            actual = {tag for _dim, tag in gmsh.model.getEntities(2)}
            if actual != expected:
                raise RuntimeError("OCC could not remove the discarded triangle faces")
            remaining = sorted(actual)
            mapping = {surface: ([surface] if surface in inside else mapping.get(surface, []))
                       for surface in surfaces}
    finally:
        _remove_temporary_geometry([*copies.values(), *retry_copies], box)
    stats = {
        "planes": list(planes),
        "half_space_box_step_units": [float(value) for value in lo + hi],
        "surfaces_before": len(surfaces),
        "surfaces_after": len(remaining),
        "surfaces_dropped": sum(not children for children in mapping.values()),
        "surfaces_split": sum(len(children) > 1 for children in mapping.values()),
        "residual_volumes": len(gmsh.model.getEntities(3)),
        "parentage_claims_disproved": disproved,
        "parentage_claims_undecided": len(undecided),
    }
    if repaired:
        stats["coverage_repaired_parents"] = repaired
        stats["cut_boundary_vertices_aligned"] = _align_cut_boundary_vertices(
            remaining, planes, tolerance=tolerance,
            noise=64 * np.finfo(float).eps * max(span, 1.0),
        )
    return mapping, undecided, stats


def auto_cut_occ_geometry(
    groups: Iterable[OccSurfaceGroup],
    *,
    grid: int = DEFAULT_AUTO_CUT_GRID,
    tolerance_rel: float = DEFAULT_AUTO_CUT_TOLERANCE_REL,
    planes: Iterable[str] | None = None,
    triangle_surface_mesh: bool = False,
) -> OccAutoCutResult:
    """Mirror-test explicit caller groups and open-cut all accepted planes.

    Group names and roles are never interpreted. Every selected surface must
    occur exactly once so role symmetry cannot silently use a default role.

    ``planes`` restricts the candidate set. It exists because cutting a plane
    the caller's solver cannot mirror is worse than not cutting at all: the
    geometry is physically halved here and refused several stages later, with
    nothing in between able to tell that the domain no longer matches the
    model. A caller that supports only x0 and y0 says so.

    ``triangle_surface_mesh`` identifies a model built as one OCC face per
    input triangle. Its straight, nonintersecting facets are checked before
    the cut. The general CAD path is used unless the caller opts in.
    """
    groups = tuple(groups)
    if not groups:
        raise ValueError("at least one OCC surface group is required")
    candidate_planes = (
        tuple(SYMMETRY_AXIS_FOR_PLANE)
        if planes is None
        else tuple(dict.fromkeys(str(plane) for plane in planes))
    )
    unknown = [plane for plane in candidate_planes if plane not in SYMMETRY_AXIS_FOR_PLANE]
    if unknown:
        raise ValueError(f"unknown symmetry planes: {unknown}")
    roles: dict[int, str] = {}
    for group in groups:
        for surface in group.selector.surface_tags:
            if surface in roles:
                raise ValueError(f"surface {surface} occurs in more than one group")
            roles[surface] = group.role.name
    surfaces = list(roles)
    # The cut below intersects EVERY surface in the model, so a surface no
    # group names would be halved without ever having been mirror-tested, and
    # would then be missing from every returned group.
    unselected = sorted(
        {int(tag) for _dim, tag in gmsh.model.getEntities(2)}.difference(roles)
    )
    if unselected:
        raise ValueError(
            "auto-cut needs every model surface in a group (the cut halves them all); "
            f"surfaces in no group: {unselected[:12]}"
            + (" and more" if len(unselected) > 12 else "")
        )
    triangle_model = _triangle_model(surfaces) if triangle_surface_mesh else None
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
        for plane in candidate_planes
    ]
    accepted_planes = tuple(verdict.plane for verdict in verdicts if verdict.accepted)
    report: dict[str, object] = {
        "mode": "auto-cut",
        "candidate_planes": list(candidate_planes),
        "tolerance_rel": float(tolerance_rel),
        "tolerance_step_units": float(tolerance),
        "model_diagonal_step_units": diagonal,
        "sample_grid": int(grid),
        "sampled_surfaces": len(samples),
        "sample_points": int(len(stacked)),
        "planes": {verdict.plane: verdict.to_dict() for verdict in verdicts},
        "cut_planes": list(accepted_planes),
    }
    if not accepted_planes:
        report["cut"] = None
        return OccAutoCutResult((), groups, {}, report)
    model_bbox = tuple(float(value) for value in gmsh.model.getBoundingBox(-1, -1))
    cut_options = dict(bbox=model_bbox, tolerance=tolerance, grid=grid)
    if triangle_model is not None:
        cut_options["triangle_model"] = triangle_model
    parentage, undecided, cut_stats = _cut_occ_geometry_to_positive_side(
        accepted_planes, **cut_options
    )
    # A claim the geometry could not decide is harmless inside one group, but
    # it must not hand a surface to a second group -- not even one of the same
    # role, since two sources share a role and are still two sources.
    owners = {
        int(surface): group.name
        for group in groups
        for surface in group.selector.surface_tags
    }
    claimants: dict[int, set[str]] = {}
    for parent, children in parentage.items():
        for child in children:
            claimants.setdefault(child, set()).add(owners.get(parent, ""))
    for parent, child in sorted(undecided):
        if len(claimants[child]) > 1:
            raise RuntimeError(
                f"OCC cut reports surface {child} as a piece of surface {parent} "
                f"(group {owners.get(parent, '')!r}), which the geometry cannot confirm, "
                f"while other groups also claim it: {sorted(claimants[child])}"
            )
    remapped_groups = tuple(
        OccSurfaceGroup(
            group.name,
            OccSurfaceSelector(remap_surface_tags(group.selector.surface_tags, parentage)),
            group.role,
        )
        for group in groups
    )
    report["cut"] = cut_stats
    return OccAutoCutResult(accepted_planes, remapped_groups, parentage, report)
