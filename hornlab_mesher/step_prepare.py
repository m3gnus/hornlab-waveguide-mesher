"""Caller-neutral OCC preparation helpers for imported STEP surface models.

The caller owns every group name and role string.  This module only reasons
about OCC surface tags, mirrored geometry, and exact boolean parentage.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from . import occ_face_index

# The mirror check itself lives in .symmetry; these names stay importable
# from here for existing callers.
from .symmetry import (
    DEFAULT_MAX_FAILURE_SAMPLES,
    DEFAULT_SYMMETRY_SNAP_BAND_MM as DEFAULT_SYMMETRY_SNAP_BAND_MM,
    SPAN_TOLERANCE_REL,
    SYMMETRY_AXIS_FOR_PLANE,
    TOLERANCES,
    MirrorMatches,
    feature_mismatch_reason,
    PlaneSymmetryVerdict,
    evaluate_mirror_symmetry,
    snap_symmetry_plane_vertices as snap_symmetry_plane_vertices,
)


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


DEFAULT_AUTO_CUT_TOLERANCE_REL = TOLERANCES["fitted"].relative
DEFAULT_AUTO_CUT_GRID = 7
AUTO_CUT_SPAN_REL = SPAN_TOLERANCE_REL
AUTO_CUT_TRIM_NUDGE_REL = 1.0e-3
AUTO_CUT_BOX_PAD_REL = 0.1
AUTO_CUT_MAX_FAILURE_SAMPLES = DEFAULT_MAX_FAILURE_SAMPLES
AUTO_CUT_ROLE_BOUNDARY_EDGE_FRACTION = 0.005


class TriangleCutRefused(RuntimeError):
    """The triangle-surface cut cannot be done safely; the model stays uncut."""


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


def _params_inside(
    surface: int, uv: list[float], trim: occ_face_index.FaceTrim | None = None
) -> list[bool]:
    """``_param_inside`` for each (u, v) pair, asking OCC once first.

    ``gmsh.model.isInside`` answers how many of the points handed to it lie
    inside the trim, not which (see ``tessellation.parametric_trim_mask``).
    One call over the whole grid settles an untrimmed face (all inside) and a
    grid that misses the face (none); any other answer, or an exception, falls
    back to exactly the per-point ``_param_inside`` calls this replaces. OCC
    classifies point by point inside that call too, so a mixed grid is not
    split further: halving it re-classifies every point at each level, which
    measured twice as slow as asking per point.
    """

    if trim is not None:
        # The face's trim polygons answer the points they are certain of
        # (``occ_face_index.params_inside``), cross-checked against the same
        # one OCC count; every other point is asked exactly as below.
        return occ_face_index.params_inside(surface, uv, trim, _param_inside)
    count = len(uv) // 2
    try:
        inside = int(gmsh.model.isInside(2, surface, uv, parametric=True))
    except Exception:
        inside = -1
    if inside == count:
        return [True] * count
    if inside == 0:
        return [False] * count
    return [_param_inside(surface, uv[2 * index], uv[2 * index + 1]) for index in range(count)]


def sample_occ_surface_points(
    surface: int, *, grid: int, trims: dict[int, Any] | None = None
) -> np.ndarray:
    """Point-sample the actual trimmed OCC face, not its untrimmed extension.

    ``trims`` (surface -> ``occ_face_index.FaceTrim``, filled on demand) lets
    the face's own trim polygons answer the trim queries; the samples are the
    same either way.
    """
    return _sample_occ_surface(surface, grid=grid, trims=trims)[0]


def _sample_occ_surface(
    surface: int, *, grid: int, trims: dict[int, Any] | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """``sample_occ_surface_points`` and the samples' (u, v), when they have one."""
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
            uv: list[float] = []
            for i in range(n):
                u = umin + (i + 0.5) * (umax - umin) / n
                for j in range(n):
                    v = vmin + (j + 0.5) * (vmax - vmin) / n
                    uv.extend((u, v))
            # The same grid, in the same order, as one point at a time: the
            # block classification only saves OCC round trips.
            if trims is None:
                inside = _params_inside(surface, uv)
            else:
                trim = trims.get(surface)
                if trim is None:
                    trim = trims[surface] = occ_face_index.FaceTrim.build(surface)
                inside = _params_inside(surface, uv, trim)
            kept = [
                value
                for index, flag in enumerate(inside) if flag
                for value in (uv[2 * index], uv[2 * index + 1])
            ]
            if kept:
                coords = gmsh.model.getValue(2, surface, kept)
                return (np.asarray(coords, dtype=float).reshape(-1, 3),
                        np.asarray(kept, dtype=float).reshape(-1, 2))
    return _sample_occ_surface_boundary_points(surface, grid=grid), None


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
    preferred: frozenset[int] | set[int] | None = None,
    exit_at: float = -1.0,
) -> tuple[float, float, list[int]]:
    """Distances from ``point`` to the candidate faces: ``(matched, nearest, hits)``.

    ``matched`` is the least distance among faces within ``tolerance`` whose
    trim holds the projection (``hits``, in candidate order), ``nearest`` the
    least distance or box gap of any face; ``-1.0`` when there is none.

    ``preferred`` faces (the sample's own role) are projected first, and once
    one of them is a hit no farther than ``exit_at`` the rest are skipped. That
    answer is then exactly as good as the full scan for a mirror test: the
    sample is a hit on a face of its own role either way, and its matched
    distance (at most ``exit_at``, the largest residual already recorded)
    cannot change the test's largest residual. ``exit_at < 0`` scans every
    face. Results are reduced in candidate order, so NaN handling is the
    original scan's.
    """
    px, py, pz = (float(value) for value in point)
    order = candidates
    if preferred:
        order = [surface for surface in candidates if surface in preferred] + [
            surface for surface in candidates if surface not in preferred
        ]
    records: dict[int, tuple[float, bool]] = {}
    for surface in order:
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
            records[surface] = (gap, False)
            continue
        try:
            closest, parametric = gmsh.model.getClosestPoint(2, surface, [px, py, pz])
        except Exception:
            continue
        distance = float(np.linalg.norm(np.asarray(closest) - np.asarray([px, py, pz])))
        hit = not (distance > tolerance or not _parametric_point_on_face(surface, parametric))
        records[surface] = (distance, hit)
        if hit and preferred and surface in preferred and 0.0 <= exit_at and distance <= exit_at:
            break
    matched_distance = -1.0
    nearest = -1.0
    matched: list[int] = []
    for surface in candidates:
        record = records.get(surface)
        if record is None:
            continue
        distance, hit = record
        if nearest_contributions is not None:
            nearest_contributions[surface] = distance
        if nearest < 0.0 or distance < nearest:
            nearest = distance
        if not hit:
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

    def closest(self, point, candidates, *, preferred=None, exit_at=-1.0):
        contributions: dict[int, float] = {}
        result = _closest_on_model(
            point, candidates=candidates, bboxes=self.bboxes, tolerance=self.tolerance,
            nearest_contributions=contributions, preferred=preferred, exit_at=exit_at,
        )
        if candidates is self.candidates or 0 <= result[1] <= self.tolerance:
            return result
        # Never retry a projection or trim query: an OCC exception must remain
        # a failed sample, even if a second call would succeed. Omitted boxes
        # are beyond tolerance, so the existing scan reads only their gaps.
        # Reduce all contributions in original order, including NaN behavior.
        fast = self._nearest_with_omitted_gaps(point, candidates, contributions)
        if fast is not None:
            return result[0], fast, result[2]
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


    def _nearest_with_omitted_gaps(self, point, candidates, contributions):
        """The scan's ``nearest`` over the local contributions and every
        omitted box's gap, without a Python call per omitted box.

        The gaps are screened in bulk; only the boxes whose screened gap is
        within a relative 1e-9 of the least are measured with the scan's own
        formula, so the least is the scan's value bit for bit. Anything the
        bulk screen cannot vouch for -- a NaN, a gap within tolerance (which
        the scan would project) -- returns ``None`` and the scan runs as it
        always has.
        """

        local = set(candidates)
        values = [contributions[surface] for surface in candidates if surface in contributions]
        if any(not np.isfinite(value) for value in values):
            return None
        omitted = [surface for surface in self.candidates
                   if surface not in local and surface in self.bboxes]
        nearest = min(values) if values else -1.0
        if not omitted:
            return nearest
        boxes = np.asarray([self.bboxes[surface] for surface in omitted], dtype=float)
        p = np.asarray(point, dtype=float)
        deltas = np.maximum(np.maximum(boxes[:, :3] - p, p - boxes[:, 3:]), 0.0)
        gaps = np.sqrt(np.einsum("ij,ij->i", deltas, deltas))
        if not np.all(np.isfinite(gaps)) or np.any(gaps <= self.tolerance * (1.0 + 1.0e-9)):
            return None
        least = float(np.min(gaps))
        for index in np.flatnonzero(gaps <= least * (1.0 + 1.0e-9)):
            exact = _closest_on_model(
                point, candidates=[omitted[int(index)]], bboxes=self.bboxes,
                tolerance=self.tolerance,
            )[1]
            if nearest < 0.0 or exact < nearest:
                nearest = exact
        return nearest


class _OccMirrorTarget:
    """The trimmed OCC faces as a mirror target: exact closest-point projection."""

    def __init__(self, candidates, bboxes, tolerance, role_boundaries, roles=None):
        self.candidates = candidates
        self.bboxes = bboxes
        self.tolerance = tolerance
        self.role_boundaries = role_boundaries
        self._index = None
        # Faces by caller role, so a sample's own role is projected first.
        self._by_role: dict[str, frozenset[int]] = {}
        self._roles = roles
        if roles:
            for face, role in roles.items():
                self._by_role.setdefault(role, set()).add(face)  # type: ignore[attr-defined]
            self._by_role = {role: frozenset(faces) for role, faces in self._by_role.items()}
        # The largest residual the evaluator has recorded so far for this
        # plane: the same running maximum over the hits this target returned.
        self._largest_residual = -1.0

    @property
    def index(self):
        # Built on the first match only: a plane the model does not straddle,
        # or a bad plane or role, is decided before any broad phase.
        if self._index is None:
            self._index = _OccMirrorIndex(self.candidates, self.bboxes, self.tolerance)
        return self._index

    def match(self, surface, mirrored):
        nearby = self.index.candidates_for_points(mirrored)
        preferred = (
            self._by_role.get(self._roles[surface])
            if self._roles and surface in self._roles else None
        )
        found = []
        for point, local in zip(mirrored, nearby):
            row = self.index.closest(
                point, local, preferred=preferred, exit_at=self._largest_residual
            )
            found.append(row)
            if row[2] and not np.isnan(row[0]) and row[0] > self._largest_residual:
                self._largest_residual = row[0]
        return MirrorMatches(
            np.asarray([row[0] for row in found], dtype=float),
            np.asarray([row[1] for row in found], dtype=float),
            hits=[row[2] for row in found],
            context=nearby,
        )

    def role_seam_allows(self, surface, point, mirrored, hits, context):
        return (self.role_boundaries is not None
                and self.role_boundaries.allows(
                    surface, point, mirrored, hits, context, self.tolerance))


#: One-sided small features (R1). A mirrored sample must land on a face of
#: its role that faces the same way, within this |cos| (60 degrees): a hole's
#: wall lying within tolerance of a plate faces 90 degrees away.
FEATURE_NORMAL_ALIGNMENT = 0.5
#: Edges where the faces meet at more than this angle (degrees) are sharp,
#: and so are free and non-manifold edges; samples of every sharp edge must
#: mirror onto some edge of the model.
FEATURE_EDGE_ANGLE_DEG = 30.0
#: Samples along each sharp edge: at least this many, and one per this
#: fraction of the model diagonal.
FEATURE_EDGE_MIN_SAMPLES = 8
FEATURE_EDGE_SPACING_REL = 0.01
FEATURE_EDGE_MAX_SAMPLES = 20000


class _FeatureEdges:
    """The model's sharp edges, sampled, and every edge as a target.

    Built once per model. ``check(plane)`` mirrors the sharp-edge samples and
    counts those farther than tolerance from every edge of the model: a hole
    or boss on one side has sharp edges whose mirror images lie in the middle
    of a face. Seams, splits within one smooth surface and tangent edges are
    not sharp, so a symmetric model with asymmetric face splits passes.
    """

    def __init__(self, surfaces: Iterable[int], *, diagonal: float, tolerance: float) -> None:
        selected = {int(surface) for surface in surfaces}
        uses: dict[int, list[int]] = {}
        for surface in selected:
            try:
                boundary = gmsh.model.getBoundary(
                    [(2, surface)], combined=False, oriented=False, recursive=False)
            except Exception:
                continue
            for dim, curve in boundary:
                if dim == 1:
                    uses.setdefault(abs(int(curve)), []).append(surface)
        self.tolerance = float(tolerance)
        spacing = max(FEATURE_EDGE_SPACING_REL * diagonal, 1.0e-12)
        points: list[np.ndarray] = []
        owners: list[int] = []
        segments_a: list[np.ndarray] = []
        segments_b: list[np.ndarray] = []
        chords: list[np.ndarray] = []
        self.curves: list[int] = []
        for curve, faces in sorted(uses.items()):
            try:
                lo, hi = gmsh.model.getParametrizationBounds(1, curve)
                t0, t1 = float(lo[0]), float(hi[0])
                if not t1 > t0:
                    continue
                polyline, chord = self._polyline(curve, t0, t1)
            except Exception:
                continue
            self.curves.append(curve)
            segments_a.append(polyline[:-1])
            segments_b.append(polyline[1:])
            chords.append(np.full(len(polyline) - 1, chord))
            length = float(np.sum(np.linalg.norm(np.diff(polyline, axis=0), axis=1)))
            count = int(max(FEATURE_EDGE_MIN_SAMPLES, min(200, math.ceil(length / spacing))))
            params = t0 + (np.arange(count) + 0.5) * (t1 - t0) / count
            sharp = self._sharp(curve, faces, params)
            if np.any(sharp):
                values = np.asarray(gmsh.model.getValue(1, curve, params.tolist()), dtype=float).reshape(-1, 3)
                points.append(values[sharp])
                owners.extend([curve] * int(np.count_nonzero(sharp)))
        self.points = np.concatenate(points) if points else np.zeros((0, 3))
        self.owners = owners
        if len(self.points) > FEATURE_EDGE_MAX_SAMPLES:
            keep = np.linspace(0, len(self.points) - 1, FEATURE_EDGE_MAX_SAMPLES).astype(int)
            self.points = self.points[keep]
            self.owners = [owners[i] for i in keep]
        self.a = np.concatenate(segments_a) if segments_a else np.zeros((0, 3))
        self.b = np.concatenate(segments_b) if segments_b else np.zeros((0, 3))
        self.segment_curve = np.concatenate([
            np.full(len(a), curve) for curve, a in zip(self.curves, segments_a)
        ]) if segments_a else np.zeros(0, dtype=int)
        self.segment_chord = np.concatenate(chords) if chords else np.zeros(0)
        self._tree = None

    def _polyline(self, curve: int, t0: float, t1: float) -> tuple[np.ndarray, float]:
        chord = math.inf
        for count in (65, 257, 1025):
            params = np.linspace(t0, t1, count)
            values = np.asarray(gmsh.model.getValue(1, curve, params.tolist()), dtype=float).reshape(-1, 3)
            mids = np.asarray(gmsh.model.getValue(
                1, curve, (0.5 * (params[1:] + params[:-1])).tolist()), dtype=float).reshape(-1, 3)
            chord = float(np.max(np.linalg.norm(mids - 0.5 * (values[1:] + values[:-1]), axis=1)))
            if chord <= 0.125 * self.tolerance:
                break
        return values, 2.0 * chord

    @staticmethod
    def _sharp(curve: int, faces: list[int], params: np.ndarray) -> np.ndarray:
        distinct = sorted(set(faces))
        if len(faces) == 1:
            return np.ones(len(params), dtype=bool)          # free edge
        if len(distinct) == 1:
            return np.zeros(len(params), dtype=bool)         # seam of one face
        if len(faces) > 2:
            return np.ones(len(params), dtype=bool)          # non-manifold
        normals = []
        for face in distinct:
            try:
                uv = gmsh.model.reparametrizeOnSurface(1, curve, params.tolist(), face)
                normal = np.asarray(gmsh.model.getNormal(face, list(uv)), dtype=float).reshape(-1, 3)
            except Exception:
                return np.zeros(len(params), dtype=bool)     # cannot tell: no claim
            normals.append(normal / np.maximum(np.linalg.norm(normal, axis=1), 1e-300)[:, None])
        facing = np.abs(np.einsum("ij,ij->i", normals[0], normals[1]))
        return facing < math.cos(math.radians(FEATURE_EDGE_ANGLE_DEG))

    def _bounds(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per point: the least lower and upper bound of its distance to any
        edge, from the polylines and each curve's own chord error."""
        lower = np.full(len(points), np.inf)
        upper = np.full(len(points), np.inf)
        if not len(self.a) or not len(points):
            return lower, upper
        from scipy.spatial import cKDTree

        if self._tree is None:
            self._tree = cKDTree(0.5 * (self.a + self.b))
        half = 0.5 * float(np.max(np.linalg.norm(self.b - self.a, axis=1)))
        reach = self.tolerance + float(np.max(self.segment_chord)) + half
        rows = self._tree.query_ball_point(points, reach * (1.0 + 1e-9))
        sizes = np.fromiter((len(row) for row in rows), dtype=np.int64, count=len(rows))
        if not sizes.sum():
            return lower, upper
        pid = np.repeat(np.arange(len(points)), sizes)
        sid = np.fromiter((i for row in rows for i in row), dtype=np.int64, count=int(sizes.sum()))
        seg = self.b[sid] - self.a[sid]
        rel = points[pid] - self.a[sid]
        den = np.einsum("ij,ij->i", seg, seg)
        t = np.clip(np.einsum("ij,ij->i", rel, seg) / np.where(den > 0, den, 1.0), 0.0, 1.0)
        dist = np.linalg.norm(rel - t[:, None] * seg, axis=1)
        np.minimum.at(lower, pid, dist - self.segment_chord[sid])
        np.minimum.at(upper, pid, dist + self.segment_chord[sid])
        return lower, upper

    def check(self, plane: str) -> tuple[int, list[dict[str, object]], int]:
        """``(points tested, failure samples (first few), mismatches)``."""
        if not len(self.points):
            return 0, [], 0
        axis = SYMMETRY_AXIS_FOR_PLANE[plane]
        mirrored = self.points.copy()
        mirrored[:, axis] *= -1
        tol = self.tolerance
        lower, upper = self._bounds(mirrored)
        matched = upper <= tol
        for index in np.flatnonzero(~matched & (lower <= tol)):
            matched[index] = self._exact(mirrored[index]) <= tol
        failed = np.flatnonzero(~matched)
        failures = [
            {
                "kind": "edge_mismatch",
                "curve": int(self.owners[index]),
                "point_step_units": [float(value) for value in self.points[index]],
                "mirrored_point_step_units": [float(value) for value in mirrored[index]],
            }
            for index in failed[:AUTO_CUT_MAX_FAILURE_SAMPLES]
        ]
        return len(self.points), failures, len(failed)

    def _exact(self, point: np.ndarray) -> float:
        """OCC's distance from ``point`` to the nearest edge (its projection is
        bounded to each edge); ``0.0`` -- no claim -- if OCC cannot answer."""
        reach = self.tolerance + float(np.max(self.segment_chord))
        rel = np.linalg.norm(0.5 * (self.a + self.b) - point, axis=1)
        half = 0.5 * np.linalg.norm(self.b - self.a, axis=1)
        curves = sorted({int(self.segment_curve[i]) for i in np.flatnonzero(rel - half <= reach)})
        best = math.inf
        for curve in curves:
            try:
                closest, _parametric = gmsh.model.getClosestPoint(1, curve, [float(x) for x in point])
            except Exception:
                return 0.0
            best = min(best, float(np.linalg.norm(np.asarray(closest, dtype=float) - point)))
        return best


#: Exact OCC projections spent, per plane, to pin the reported peak residual
#: and the reported worst off-model distance to real samples.
CERTIFIED_PEAK_CHECKS = 24


class _CertifiedOccMirrorTarget(_OccMirrorTarget):
    """The exact target, asked only what a forward tessellation cannot prove.

    Every mirrored sample is first measured against the tessellated faces
    (``occ_face_index``). A sample within tolerance of a face of its own role,
    by the margin of that triangle's measured chord error, at a point inside
    the face's trim with room to spare, and facing its way, is a hit without
    OCC; a sample farther than tolerance (plus the trim query's nudge and the
    chord error) from every face is a miss without OCC. Every other sample
    takes the exact scan of :class:`_OccMirrorTarget` unchanged, in the
    original order. (The exact scan also counts a hit where the projection
    falls outside the face's parameter box and the trim query's nudge clamps
    it back in; a certified miss is a miss of the trimmed face itself, so the
    fast check is the stricter of the two there.)

    Two reported numbers are then pinned to real OCC projections: the peak
    residual (the certified hits with the largest upper bounds are projected
    exactly, until the next bound cannot beat the peak found, at most
    ``CERTIFIED_PEAK_CHECKS``), and, on a rejected plane, the nearest distance
    of the first failure samples and of the worst-looking misses. A
    certified sample whose exact projection disagrees takes the exact answer.
    """

    def __init__(self, candidates, bboxes, tolerance, role_boundaries, roles, *,
                 samples, axis, face_index, normals=None):
        super().__init__(candidates, bboxes, tolerance, role_boundaries, roles)
        self.samples = samples
        self.axis = axis
        self.face_index = face_index
        # Unit normals of the samples (None for a face sampled on its
        # boundary curves): with them, a hit must also face the same way.
        self.normals = normals
        self._plan: dict[Any, MirrorMatches] | None = None
        self.stats: dict[str, int] = {}

    def match(self, surface, mirrored):
        if self._plan is None:
            self._plan = self._plan_all()
        return self._plan[surface]

    def _exact(self, point):
        local = self.index.candidates_for_points(np.asarray([point]))[0]
        return self.index.closest(point, local)

    def _aligned(self, face, point, normal) -> bool:
        """Whether ``face`` faces along ``normal`` where ``point`` projects (OCC)."""
        try:
            _closest, parametric = gmsh.model.getClosestPoint(2, face, [float(x) for x in point])
            other = np.asarray(gmsh.model.getNormal(face, [float(parametric[0]), float(parametric[1])]),
                               dtype=float)
        except Exception:
            return True
        length = float(np.linalg.norm(other))
        if not np.isfinite(length) or length == 0.0:
            return True
        return abs(float(np.dot(normal, other))) / length >= FEATURE_NORMAL_ALIGNMENT

    def _plan_all(self) -> dict[Any, MirrorMatches]:
        order = sorted(self.samples)
        counts = [len(self.samples[surface]) for surface in order]
        owners = np.repeat(np.arange(len(order)), counts)
        points = np.concatenate([self.samples[surface] for surface in order])
        mirrored = points.copy()
        mirrored[:, self.axis] *= -1
        shape = self.normals is not None
        mirrored_normals = np.full((len(points), 3), np.nan)
        if shape:
            start = 0
            for surface, count in zip(order, counts):
                normals = self.normals.get(surface)
                if normals is not None and len(normals) == count:
                    mirrored_normals[start:start + count] = normals
                start += count
            mirrored_normals[:, self.axis] *= -1
        known_normal = np.all(np.isfinite(mirrored_normals), axis=1)
        total = len(mirrored)
        tol = float(self.tolerance)
        roles = self._roles or {}
        own_role = np.asarray([roles.get(order[i]) for i in owners], dtype=object)
        index = self.face_index
        meshes = index.meshes
        near = index.near(mirrored, tol)
        same_role = np.fromiter(
            (roles.get(int(f)) == own_role[p] for p, f in zip(near.point, near.face)),
            dtype=bool, count=len(near.face),
        )
        hit_cert = near.upper <= tol
        miss_cert = near.lower > tol
        candidate_rows = hit_cert & same_role
        if shape and len(near.face):
            # A certified hit counts only where the hit face (its triangle)
            # faces the mirrored sample's way; otherwise OCC decides.
            facing = np.abs(np.einsum("ij,ij->i", mirrored_normals[near.point], near.normal))
            candidate_rows &= ~known_normal[near.point] | (
                near.normal_ok & (facing >= FEATURE_NORMAL_ALIGNMENT))
        pass_rows = np.flatnonzero(candidate_rows)
        passed = np.zeros(total, dtype=bool)
        passed[near.point[pass_rows]] = True
        estimate = np.full(total, np.inf)
        upper = np.full(total, np.inf)
        hit_face = np.full(total, -1, dtype=np.int64)
        # Per point, the same-role certified hit with the least upper bound.
        if len(pass_rows):
            ranked = pass_rows[np.lexsort((near.upper[pass_rows], near.point[pass_rows]))]
            firsts = ranked[np.r_[True, near.point[ranked][1:] != near.point[ranked][:-1]]]
            pts = near.point[firsts]
            upper[pts] = near.upper[firsts]
            estimate[pts] = np.maximum(near.estimate[firsts], 0.0)
            hit_face[pts] = near.face[firsts]
        blocked = np.zeros(total, dtype=bool)
        blocked[near.point[~miss_cert]] = True
        # Faces the index could not tessellate: only their box can clear them.
        unknown = [face for face in self.candidates if face not in meshes and face in self.bboxes]
        if unknown:
            boxes = np.asarray([self.bboxes[face] for face in unknown], dtype=float)
            deltas = np.maximum(np.maximum(boxes[None, :, :3] - mirrored[:, None, :],
                                           mirrored[:, None, :] - boxes[None, :, 3:]), 0.0)
            gaps = np.sqrt(np.einsum("pfk,pfk->pf", deltas, deltas))
            blocked |= np.any(~(gaps > tol), axis=1)
        missed = ~passed & ~blocked
        # A miss's nearest distance as the exact scan reads it: a face's box
        # gap where that is beyond tolerance, else (approximately) the face.
        approx = np.full(total, np.inf)
        faces_known = [face for face in self.candidates if face in self.bboxes]
        if faces_known:
            boxes = np.asarray([self.bboxes[face] for face in faces_known], dtype=float)
            column = {face: k for k, face in enumerate(faces_known)}
            for start in range(0, total, 2048):
                chunk = mirrored[start:start + 2048]
                deltas = np.maximum(np.maximum(boxes[None, :, :3] - chunk[:, None, :],
                                               chunk[:, None, :] - boxes[None, :, 3:]), 0.0)
                gaps = np.sqrt(np.einsum("pfk,pfk->pf", deltas, deltas))
                if len(near.face):
                    rows = np.flatnonzero((near.point >= start) & (near.point < start + len(chunk)))
                    cols = np.asarray([column.get(int(f), -1) for f in near.face[rows]])
                    keep = cols >= 0
                    pts = near.point[rows][keep] - start
                    cols = cols[keep]
                    within = gaps[pts, cols] <= tol
                    gaps[pts[within], cols[within]] = np.maximum(near.estimate[rows][keep][within], 0.0)
                approx[start:start + len(chunk)] = np.min(gaps, axis=1)

        matched = np.full(total, -1.0)
        nearest = np.full(total, -1.0)
        hits: list[list[int]] = [[] for _ in range(total)]
        exact_done = np.zeros(total, dtype=bool)
        # 1. The undecided samples, exactly as the plain scan, in its order.
        undecided = np.flatnonzero(~passed & ~missed)
        if len(undecided):
            local = self.index.candidates_for_points(mirrored[undecided])
            for row, p in enumerate(undecided):
                preferred = self._by_role.get(roles[order[owners[p]]]) if roles else None
                result = self.index.closest(
                    mirrored[p], local[row], preferred=preferred, exit_at=self._largest_residual
                )
                matched[p], nearest[p], hits[p] = result[0], result[1], list(result[2])
                exact_done[p] = True
                if result[2] and not np.isnan(result[0]) and result[0] > self._largest_residual:
                    self._largest_residual = result[0]
        misaligned = np.zeros(total, dtype=bool)
        if shape:
            # A face of the sample's role that may lie within tolerance and
            # faces its way (by the tessellation) also counts: a sample on a
            # trim curve can project just outside one face and onto the
            # perpendicular face meeting it there.
            facing_rows = np.zeros(total, dtype=bool)
            if len(near.face):
                facing = np.abs(np.einsum("ij,ij->i", mirrored_normals[near.point], near.normal))
                ok_rows = same_role & ~(near.lower > tol) & near.normal_ok & (
                    facing >= FEATURE_NORMAL_ALIGNMENT)
                facing_rows[near.point[ok_rows]] = True
            for p in undecided:
                own = [h for h in hits[p] if roles.get(h) == own_role[p]]
                if own and known_normal[p] and not facing_rows[p]:
                    misaligned[p] = not any(
                        self._aligned(face, mirrored[p], mirrored_normals[p]) for face in own)
        for p in np.flatnonzero(passed):
            matched[p] = nearest[p] = estimate[p]
            hits[p] = [int(hit_face[p])]
        # 2. The peak residual, pinned to exact projections.
        exact_peak = max((matched[p] for p in np.flatnonzero(exact_done) if hits[p]
                          and not np.isnan(matched[p])), default=-1.0)
        checks = 0
        # Ranked by the tessellation's own distance (exact on planes); stop
        # once no remaining sample's upper bound can beat the peak found.
        ranked = sorted(np.flatnonzero(passed), key=lambda q: (-estimate[q], -upper[q]))
        for p in ranked:
            if checks >= CERTIFIED_PEAK_CHECKS:
                break
            if upper[p] <= exact_peak:
                continue
            checks += 1
            result = self._exact(mirrored[p])
            matched[p], nearest[p], hits[p] = result[0], result[1], list(result[2])
            exact_done[p] = True
            if result[2] and not np.isnan(result[0]):
                exact_peak = max(exact_peak, float(result[0]))
        for p in np.flatnonzero(passed & ~exact_done):
            if exact_peak >= 0.0 and matched[p] > exact_peak:
                matched[p] = nearest[p] = exact_peak
        # 3. Misses: exact nearest distances for the reported failures.
        failure_order = []
        for p in range(total):
            if missed[p] and not exact_done[p]:
                failure_order.append(p)
            elif exact_done[p] and (not hits[p] or own_role[p] not in {roles.get(h) for h in hits[p]}
                                    or misaligned[p]):
                failure_order.append(p)
            if len(failure_order) >= DEFAULT_MAX_FAILURE_SAMPLES:
                break
        worst_first = sorted(np.flatnonzero(missed), key=lambda q: -approx[q])[:CERTIFIED_PEAK_CHECKS]
        exact_worst = -1.0
        for p in [*failure_order, *worst_first]:
            if exact_done[p]:
                continue
            result = self._exact(mirrored[p])
            matched[p], nearest[p], hits[p] = result[0], result[1], list(result[2])
            exact_done[p] = True
            if not result[2]:
                exact_worst = max(exact_worst, float(result[1]))
        for p in np.flatnonzero(missed & ~exact_done):
            value = float(approx[p]) if np.isfinite(approx[p]) else exact_worst
            nearest[p] = min(value, exact_worst) if exact_worst >= 0.0 else value
        self.stats = {
            "points": int(total),
            "certified_hits": int(np.count_nonzero(passed & ~exact_done)),
            "certified_misses": int(np.count_nonzero(missed & ~exact_done)),
            "exact": int(np.count_nonzero(exact_done)),
            **({"misaligned": int(np.count_nonzero(misaligned))} if shape else {}),
        }
        plan: dict[Any, MirrorMatches] = {}
        start = 0
        for surface, count in zip(order, counts):
            stop = start + count
            plan[surface] = MirrorMatches(
                matched[start:stop].copy(), nearest[start:stop].copy(),
                hits=[hits[p] for p in range(start, stop)],
                misaligned=misaligned[start:stop].tolist() if shape else None,
            )
            start = stop
        return plan


def evaluate_occ_plane_symmetry(
    plane: str,
    *,
    samples: dict[int, np.ndarray],
    roles: dict[int, str],
    bboxes: dict[int, tuple[float, ...]],
    tolerance: float,
    span_tolerance: float,
    _role_boundaries: _OccTriangleRoleBoundaries | None = None,
    _face_index: occ_face_index.FaceIndex | None = None,
    _stats: dict[str, int] | None = None,
    _normals: Mapping[int, np.ndarray | None] | None = None,
    _feature_edges: _FeatureEdges | None = None,
) -> PlaneSymmetryVerdict:
    """Require both trimmed geometry and caller-supplied roles to mirror.

    ``_face_index`` (the model's forward tessellation) lets most samples be
    decided without OCC; the verdict's rule and tolerances are the same.
    With ``_normals`` and ``_feature_edges`` the plane must also pass the
    one-sided small-feature check: mirrored samples land on faces facing the
    same way, and sharp edges mirror onto edges.
    """
    target: _OccMirrorTarget
    if _face_index is not None and _role_boundaries is None:
        target = _CertifiedOccMirrorTarget(
            sorted(samples), bboxes, tolerance, None, roles,
            samples=samples, axis=SYMMETRY_AXIS_FOR_PLANE[plane], face_index=_face_index,
            normals=_normals,
        )
    else:
        target = _OccMirrorTarget(sorted(samples), bboxes, tolerance, _role_boundaries, roles)
    verdict = evaluate_mirror_symmetry(
        plane, samples=samples, roles=roles, target=target, tolerance=tolerance,
        span_tolerance=span_tolerance, max_failure_samples=AUTO_CUT_MAX_FAILURE_SAMPLES,
    )
    if _stats is not None and isinstance(target, _CertifiedOccMirrorTarget):
        _stats.update(target.stats)
    straddles = verdict.accepted or verdict.points_off_model or verdict.role_mismatches \
        or verdict.feature_mismatches
    if _feature_edges is not None and verdict.feature_mismatches is not None and straddles:
        tested, edge_failures, edge_mismatches = _feature_edges.check(plane)
        mismatches = verdict.feature_mismatches + edge_mismatches
        room = AUTO_CUT_MAX_FAILURE_SAMPLES - len(verdict.failure_samples)
        verdict = replace(
            verdict,
            accepted=verdict.accepted and edge_mismatches == 0,
            reason=(feature_mismatch_reason(mismatches)
                    if verdict.accepted and edge_mismatches else verdict.reason),
            feature_points_tested=int(verdict.feature_points_tested or 0) + tested,
            feature_mismatches=mismatches,
            failure_samples=verdict.failure_samples + tuple(edge_failures[:max(room, 0)]),
        )
        if _stats is not None:
            _stats["edge_points"] = tested
            _stats["edge_mismatches"] = edge_mismatches
    return verdict


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
                    present = set(gmsh.model.occ.getEntities())
                    if (3, container) in present:
                        gmsh.model.occ.remove([(3, container)], recursive=False)
                        present = set(gmsh.model.occ.getEntities())
                    left = [item for item in doomed if item in present]
                    if left:
                        gmsh.model.occ.remove(left, recursive=True)
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
    parent_samples: Mapping[int, np.ndarray] | None = None,
) -> list[int]:
    """Check coverage in the other direction: retained parent onto children.

    Validating child ancestry alone cannot detect a face silently discarded
    by OCC. Use trimmed interior samples, clear of the clipping planes, so
    discarded faces and partially missing parents both trigger recovery.
    ``parent_samples`` are the mirror test's samples of the same parents on
    the same grid: a copy is the same face, so they are its samples too.
    """
    axes = [SYMMETRY_AXIS_FOR_PLANE[plane] for plane in planes]
    missing = []
    for parent, copy in copies.items():
        known = None if parent_samples is None else parent_samples.get(parent)
        points = known if known is not None else sample_occ_surface_points(copy, grid=grid)
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
            raise TriangleCutRefused("triangle surface model contains a nonplanar face")
        boundary = gmsh.model.getBoundary(
            [(2, surface)], combined=False, oriented=False, recursive=False
        )
        edges = tuple(abs(tag) for dim, tag in boundary if dim == 1)
        if len(edges) != 3 or len(set(edges)) != 3:
            raise TriangleCutRefused("triangle surface model contains a nontriangle face")
        face_edges[surface] = edges
        degree: dict[int, int] = {}
        for edge in edges:
            edge_faces.setdefault(edge, []).append(surface)
            if edge not in edge_vertices:
                if gmsh.model.getType(1, edge) != "Line":
                    raise TriangleCutRefused("triangle surface model contains a curved edge")
                ends = gmsh.model.getBoundary([(1, edge)], oriented=False)
                tags = tuple(abs(tag) for dim, tag in ends if dim == 0)
                if len(tags) != 2 or tags[0] == tags[1]:
                    raise TriangleCutRefused("triangle surface model has an invalid edge")
                edge_vertices[edge] = tags
                for tag in tags:
                    if tag not in vertices:
                        vertices[tag] = np.asarray(gmsh.model.getValue(0, tag, []), dtype=float)
            for vertex in edge_vertices[edge]:
                degree[vertex] = degree.get(vertex, 0) + 1
        if len(degree) != 3 or any(value != 2 for value in degree.values()):
            raise TriangleCutRefused("triangle surface model has an invalid face boundary")
        face_vertices[surface] = tuple(sorted(degree))
    if any(len(faces) > 2 for faces in edge_faces.values()):
        raise TriangleCutRefused("triangle surface model has a nonmanifold edge")
    if not vertices or any(not np.all(np.isfinite(point)) for point in vertices.values()):
        raise TriangleCutRefused("triangle surface model has invalid vertices")
    facets = np.asarray([[vertices[v] for v in face_vertices[s]] for s in surfaces])
    lengths = np.linalg.norm(facets - np.roll(facets, 1, axis=1), axis=2)
    areas = np.linalg.norm(np.cross(facets[:, 1] - facets[:, 0],
                                    facets[:, 2] - facets[:, 0]), axis=1)
    if (not np.all(np.isfinite(lengths)) or not np.all(np.isfinite(areas))
            or np.any(lengths <= 0) or np.any(areas <= 0)):
        raise TriangleCutRefused("triangle surface model has a degenerate face")
    origin = facets.min(axis=(0, 1))
    scale = float(np.max(facets.max(axis=(0, 1)) - origin))
    if not np.isfinite(scale) or scale <= 0:
        raise TriangleCutRefused("triangle surface model has invalid extent")
    # Reuse the existing conservative facet intersection and contact checks.
    # Normalize so their numerical guards apply independently of model units.
    from .shell_facets import BoundaryIndex, _contact_only

    normalized = (facets - origin) / scale
    for first, second in BoundaryIndex(normalized).hits(normalized):
        if first < second and not _contact_only(normalized[first], normalized[second]):
            raise TriangleCutRefused("triangle surface model contains intersecting faces")
    return _OccTriangleModel(vertices, face_edges, edge_vertices,
                             {edge: tuple(faces) for edge, faces in edge_faces.items()},
                             face_vertices)


class _OccTriangleRoleBoundaries:
    """Role transitions on the validated triangle model's actual edges."""

    def __init__(self, model: _OccTriangleModel, roles: dict[int, str]) -> None:
        self.roles = roles
        self.boundaries: dict[int, list[tuple[str, np.ndarray, np.ndarray]]] = {}
        self.feature: dict[int, float] = {}
        for surface, edges in model.face_edges.items():
            self.feature[surface] = min(
                float(np.linalg.norm(model.vertices[model.edge_vertices[edge][0]]
                                     - model.vertices[model.edge_vertices[edge][1]]))
                for edge in edges
            )
        for edge, faces in model.edge_faces.items():
            if len(faces) != 2 or roles[faces[0]] == roles[faces[1]]:
                continue
            start, end = (model.vertices[tag] for tag in model.edge_vertices[edge])
            first, second = faces
            self.boundaries.setdefault(first, []).append((roles[second], start, end))
            self.boundaries.setdefault(second, []).append((roles[first], start, end))

    def _distance(self, surface: int, point: np.ndarray, other_roles: set[str]) -> float:
        nearest = float("inf")
        if not np.all(np.isfinite(point)):
            return nearest
        for role, start, end in self.boundaries.get(surface, ()):
            if role not in other_roles:
                continue
            vector = end - start
            squared = float(np.dot(vector, vector))
            if not np.isfinite(squared) or squared <= 0:
                continue
            position = float(np.dot(point - start, vector) / squared)
            if not np.isfinite(position):
                continue
            position = min(1.0, max(0.0, position))
            distance = float(np.linalg.norm(point - (start + position * vector)))
            if distance < nearest:
                nearest = distance
        return nearest

    def allows(
        self, surface: int, point: np.ndarray, mirrored: np.ndarray,
        hits: list[int], candidates: list[int], tolerance: float,
    ) -> bool:
        if not np.isfinite(tolerance) or tolerance <= 0:
            return False
        role = self.roles[surface]
        opposite_roles = {self.roles[hit] for hit in hits}
        for opposite in opposite_roles:
            transition = {opposite}
            source_distance = self._distance(surface, point, transition)
            if not np.isfinite(source_distance):
                continue
            for target in candidates:
                if self.roles[target] != role:
                    continue
                limit = min(tolerance,
                            AUTO_CUT_ROLE_BOUNDARY_EDGE_FRACTION * self.feature[surface],
                            AUTO_CUT_ROLE_BOUNDARY_EDGE_FRACTION * self.feature[target])
                if (source_distance <= limit
                        and self._distance(target, mirrored, transition) <= limit):
                    return True
        return False


def _copy_triangle_faces(surfaces: list[int]) -> list[int]:
    """Copy a batch and clean up any partial result if OCC fails."""
    before = set(gmsh.model.occ.getEntities(2))
    try:
        copied = gmsh.model.occ.copy([(2, surface) for surface in surfaces])
        if len(copied) != len(surfaces) or any(dim != 2 for dim, _ in copied):
            raise RuntimeError("OCC returned incomplete triangle face copies")
        tags = [tag for _, tag in copied]
        if len(set(tags)) != len(surfaces):
            raise RuntimeError("OCC returned duplicate triangle face copies")
        created = set(gmsh.model.occ.getEntities(2)) - before
        if created != {(2, tag) for tag in tags}:
            raise RuntimeError("OCC returned unexpected triangle face copies")
        return tags
    except Exception:
        leaked = [tag for dim, tag in gmsh.model.occ.getEntities(2)
                  if (dim, tag) not in before]
        _remove_temporary_geometry(leaked, None)
        raise


def _verify_triangle_copies(copies: dict[int, int], model: _OccTriangleModel) -> None:
    """Check batch-copy order against each original triangle's vertices."""
    if not copies:
        return
    from .step_mapping import _gmsh_surface_boundaries

    points, _curves = _gmsh_surface_boundaries(list(copies.values()))
    for (parent, _copy), actual in zip(copies.items(), points):
        expected = sorted(tuple(model.vertices[tag]) for tag in model.face_vertices[parent])
        if actual.shape != (3, 3) or sorted(map(tuple, actual)) != expected:
            raise RuntimeError("OCC returned triangle face copies in an unexpected order")


def _cut_occ_geometry_to_positive_side(
    planes: tuple[str, ...],
    *,
    bbox: tuple[float, float, float, float, float, float],
    tolerance: float,
    grid: int,
    triangle_model: _OccTriangleModel | None = None,
    parent_samples: Mapping[int, np.ndarray] | None = None,
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
            raise TriangleCutRefused("triangle surface model has invalid face bounds")
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
    cut_succeeded = False
    try:
        if triangle_model is None:
            for surface in surfaces:
                copies[surface] = gmsh.model.occ.copy([(2, surface)])[0][1]
        elif cut_parents:
            copies = dict(zip(cut_parents, _copy_triangle_faces(cut_parents)))
        gmsh.model.occ.synchronize()
        if triangle_model is not None:
            _verify_triangle_copies(copies, triangle_model)
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
                mapping, copies, planes, tolerance=tolerance, grid=grid,
                parent_samples=parent_samples if triangle_model is None else None,
            )
            if not uncovered:
                break
            if triangle_model is not None:
                # Standalone retry copies lose edge sharing with the untouched
                # inside faces, so the triangle path never retries.
                raise TriangleCutRefused(
                    f"OCC cut lost retained portions of surfaces {uncovered}")
            if attempt:
                raise RuntimeError(f"OCC cut lost retained portions of surfaces {uncovered}")
            # OCC can lose a periodic analytic face when its seam meets the
            # clipping plane. Converting that carrier to rational B-splines
            # avoids the periodic boolean path without approximating the
            # surface. Retry the whole boundary together to share cut edges;
            # adding a repaired face alone would leave a disconnected mesh.
            repaired = uncovered
            gmsh.model.occ.remove([(2, tag) for tag in remaining], recursive=True)
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
            faces = [(2, tag) for tag in sorted(actual)]
            used_curves = {tag for _d, tag in gmsh.model.getBoundary(
                faces, combined=False, oriented=False, recursive=False)}
            used_points = {tag for _d, tag in gmsh.model.getBoundary(
                faces, combined=False, oriented=False, recursive=True)}
            if ({tag for _d, tag in gmsh.model.getEntities(1)} != used_curves
                    or {tag for _d, tag in gmsh.model.getEntities(0)} != used_points):
                raise TriangleCutRefused("triangle cut left stray curves or points")
            remaining = sorted(actual)
            mapping = {surface: ([surface] if surface in inside else mapping.get(surface, []))
                       for surface in surfaces}
        cut_succeeded = True
    finally:
        try:
            _remove_temporary_geometry([*copies.values(), *retry_copies], box)
        except Exception:
            # Raise a cleanup failure after a successful cut (leftover copies
            # would be meshed); never mask an exception already in flight.
            if cut_succeeded:
                raise
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
    known_verdicts: Mapping[str, Mapping[str, Any]] | None = None,
    fast_check: bool = True,
    feature_check: bool = True,
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

    ``known_verdicts`` are verdicts (``to_dict`` form) the caller already has
    for this very geometry -- the same model, placed and healed the same way
    -- keyed by plane; those planes are not tested again.

    ``fast_check`` (the default on the general CAD path) decides most mirror
    samples from a forward tessellation of the faces (``occ_face_index``)
    and leaves the rest to OCC's exact projection; ``False`` projects every
    sample with OCC, exactly as before stage 4b (a portable replay of an
    older package rebuilds that way). ``feature_check`` (with
    ``fast_check``) adds the one-sided small-feature check: mirrored samples
    must land on faces facing the same way, and sharp edges on edges.

    A cut that fails raises ``RuntimeError`` with ``accepted_planes`` (the
    planes the mirror test accepted and the cut was for) and ``report``.
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
    role_boundaries = (_OccTriangleRoleBoundaries(triangle_model, roles)
                       if triangle_model is not None else None)
    # The general CAD path tessellates its faces for the mirror test first:
    # the trim polygons, once they match OCC's area for a face, also answer
    # the sampling's trim queries.
    trims: dict[int, Any] | None = None
    face_index: occ_face_index.FaceIndex | None = None
    if triangle_model is None and fast_check:
        box = gmsh.model.getBoundingBox(-1, -1)
        box_diagonal = float(np.linalg.norm(np.asarray(box[3:]) - np.asarray(box[:3])))
        face_index, trims = occ_face_index.build_face_index(
            surfaces, diagonal=box_diagonal,
            tolerance=TOLERANCES["fitted"].relative * box_diagonal, trims={},
        )
    sampled = {
        surface: _sample_occ_surface(surface, grid=grid, trims=trims) for surface in surfaces
    }
    samples = {surface: points for surface, (points, _uv) in sampled.items() if len(points)}
    normals: dict[int, np.ndarray | None] | None = None
    if face_index is not None and feature_check:
        normals = {}
        for surface in samples:
            uv = sampled[surface][1]
            if uv is None:
                normals[surface] = None
                continue
            try:
                values = np.asarray(gmsh.model.getNormal(surface, uv.ravel().tolist()),
                                    dtype=float).reshape(-1, 3)
            except Exception:
                normals[surface] = None
                continue
            lengths = np.linalg.norm(values, axis=1)
            values = values / np.where(lengths > 0, lengths, 1.0)[:, None]
            values[~(lengths > 0)] = np.nan
            normals[surface] = values
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
    tolerance = replace(TOLERANCES["fitted"], relative=tolerance_rel).resolve(diagonal)
    span_tolerance = AUTO_CUT_SPAN_REL * diagonal
    known = {str(plane): dict(verdict) for plane, verdict in (known_verdicts or {}).items()}
    verdict_options = dict(samples=samples, roles=roles, bboxes=bboxes,
                           tolerance=tolerance, span_tolerance=span_tolerance)
    if role_boundaries is not None:
        verdict_options["_role_boundaries"] = role_boundaries
    oracle: dict[str, dict[str, int]] = {}
    untested = [plane for plane in candidate_planes if plane not in known]
    if face_index is not None and untested:
        verdict_options["_face_index"] = face_index
        if normals is not None:
            verdict_options["_normals"] = normals
            verdict_options["_feature_edges"] = _FeatureEdges(
                samples, diagonal=diagonal, tolerance=float(tolerance))
    tested = {}
    for plane in untested:
        stats: dict[str, int] = {}
        tested[plane] = evaluate_occ_plane_symmetry(plane, **verdict_options, _stats=stats)
        if stats:
            oracle[plane] = stats
    plane_reports = {
        plane: known[plane] if plane in known else tested[plane].to_dict()
        for plane in candidate_planes
    }
    accepted_planes = tuple(
        plane for plane in candidate_planes if plane_reports[plane].get("accepted")
    )
    report: dict[str, object] = {
        "mode": "auto-cut",
        "candidate_planes": list(candidate_planes),
        "tolerance_rel": float(tolerance_rel),
        "tolerance_step_units": float(tolerance),
        "model_diagonal_step_units": diagonal,
        "sample_grid": int(grid),
        **({"reused_verdict_planes": [p for p in candidate_planes if p in known]} if known else {}),
        "sampled_surfaces": len(samples),
        "sample_points": int(len(stacked)),
        "planes": plane_reports,
        "cut_planes": list(accepted_planes),
        **({"mirror_check_points": oracle} if oracle else {}),
    }
    if not accepted_planes:
        report["cut"] = None
        return OccAutoCutResult((), groups, {}, report)
    try:
        return _cut_accepted_planes(
            groups, accepted_planes, report, tolerance=tolerance, grid=grid,
            triangle_model=triangle_model, parent_samples=samples,
        )
    except RuntimeError as exc:
        # The caller learns which planes the refused cut was for, and what the
        # mirror test found, without testing again.
        exc.accepted_planes = accepted_planes  # type: ignore[attr-defined]
        exc.report = report  # type: ignore[attr-defined]
        raise


def _cut_accepted_planes(
    groups: tuple[OccSurfaceGroup, ...],
    accepted_planes: tuple[str, ...],
    report: dict[str, object],
    *,
    tolerance: float,
    grid: int,
    triangle_model: Any,
    parent_samples: Mapping[int, np.ndarray] | None = None,
) -> OccAutoCutResult:
    model_bbox = tuple(float(value) for value in gmsh.model.getBoundingBox(-1, -1))
    cut_options: dict[str, Any] = dict(bbox=model_bbox, tolerance=tolerance, grid=grid)
    if triangle_model is not None:
        cut_options["triangle_model"] = triangle_model
    if parent_samples is not None:
        cut_options["parent_samples"] = parent_samples
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
