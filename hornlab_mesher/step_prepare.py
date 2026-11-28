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
from ._lazy_gmsh import LazyGmsh

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


gmsh: Any = LazyGmsh(globals())


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
    """``_param_inside`` for each (u, v) pair, asking OCC once first
    (``occ_face_index.params_inside``; with ``trim``, the face's verified
    trim polygons answer the points they are certain of).

    ``gmsh.model.isInside`` answers how many of the points handed to it lie
    inside the trim, not which (see ``tessellation.parametric_trim_mask``).
    One call over the whole grid settles an untrimmed face (all inside) and a
    grid that misses the face (none); any other answer, or an exception, falls
    back to exactly the per-point ``_param_inside`` calls this replaces. OCC
    classifies point by point inside that call too, so a mixed grid is not
    split further: halving it re-classifies every point at each level, which
    measured twice as slow as asking per point.
    """

    return occ_face_index.params_inside(surface, uv, trim, _param_inside)


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
        gaps = _box_gaps(np.asarray(point, dtype=float)[None, :], boxes)[0]
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
#: A crease inside a face must keep its jump over this many halvings of the
#: probe interval (a smooth bend's turn halves with each).
FEATURE_CREASE_HALVINGS = 6
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
        self.faces = sorted(selected)
        self.face_boxes = np.asarray([gmsh.model.getBoundingBox(2, face) for face in self.faces],
                                     dtype=float).reshape(-1, 6)
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
        self.spacing = spacing
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
        if gmsh.model.getType(1, curve) == "Line":
            # Exact from its ends (no curve evaluations), in pieces no longer
            # than the edge samples' spacing so the segment search stays local.
            ends = np.asarray(gmsh.model.getValue(1, curve, [t0, t1]), dtype=float).reshape(-1, 3)
            pieces = max(1, math.ceil(float(np.linalg.norm(ends[1] - ends[0])) / self.spacing))
            steps = np.linspace(0.0, 1.0, pieces + 1)[:, None]
            return ends[0] + steps * (ends[1] - ends[0]), 0.0
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
            normals.append(normal)
        lengths = [np.linalg.norm(normal, axis=1) for normal in normals]
        sound = np.isfinite(lengths[0]) & np.isfinite(lengths[1]) & (lengths[0] > 0) & (lengths[1] > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            facing = np.abs(np.einsum("ij,ij->i", normals[0], normals[1])) / (lengths[0] * lengths[1])
        # A pole or apex has no normal: no claim there.
        return sound & (facing < math.cos(math.radians(FEATURE_EDGE_ANGLE_DEG)))

    def _bounds(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per point: the least lower and upper bound of its distance to any
        edge, from the polylines and each curve's own chord error."""
        lower = np.full(len(points), np.inf)
        upper = np.full(len(points), np.inf)
        if not len(self.a) or not len(points):
            return lower, upper
        if self._tree is None:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(0.5 * (self.a + self.b))
        pid, sid, dist = occ_face_index.segment_pairs(
            points, self.a, self.b, self.tolerance + float(np.max(self.segment_chord)), self._tree)
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
        # A sharp edge may mirror onto a crease inside one face (a C0 knot
        # in a spline): that is an edge in all but topology.
        for index in np.flatnonzero(~matched):
            matched[index] = self._on_crease(mirrored[index])
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

    def _on_crease(self, point: np.ndarray) -> bool:
        """Whether ``point`` lies within tolerance of a face (inside its trim)
        that is creased there: normals a little way to either side, in either
        parameter direction, differ by more than the sharp-edge angle, and
        keep doing so as that interval is halved (``FEATURE_CREASE_HALVINGS``
        times, keeping the half that turns more). A C0 crease keeps its jump
        however short the interval; a smooth bend, however tight, turns less
        and less, so it is no crease."""
        if not self.faces:
            return False
        gaps = _box_gaps(point[None, :], self.face_boxes)[0]
        for face in [f for f, gap in zip(self.faces, gaps) if gap <= self.tolerance]:
            try:
                closest, parametric = gmsh.model.getClosestPoint(2, face, [float(x) for x in point])
            except Exception:
                continue
            if float(np.linalg.norm(np.asarray(closest, dtype=float) - point)) > self.tolerance:
                continue
            u, v = float(parametric[0]), float(parametric[1])
            if not _parametric_point_on_face(face, [u, v]):
                continue
            try:
                lo, hi = gmsh.model.getParametrizationBounds(2, face)
                derivative = np.asarray(gmsh.model.getDerivative(2, face, [u, v]), dtype=float)
            except Exception:
                continue
            speed_u = float(np.linalg.norm(derivative[:3]))
            speed_v = float(np.linalg.norm(derivative[3:6]))
            for axis, speed in ((0, speed_u), (1, speed_v)):
                if not (speed > 0 and np.isfinite(speed)):
                    continue
                step = 2.0 * self.tolerance / speed
                low = [u, v]
                high = [u, v]
                low[axis] = max(float(lo[axis]), (u, v)[axis] - step)
                high[axis] = min(float(hi[axis]), (u, v)[axis] + step)
                if self._jump(face, axis, low, high):
                    return True
        return False

    @staticmethod
    def _jump(face: int, axis: int, low: list[float], high: list[float]) -> bool:
        """Whether the normal jumps by more than the sharp-edge angle between
        ``low`` and ``high`` (along parameter ``axis``) at every halving."""
        limit = math.cos(math.radians(FEATURE_EDGE_ANGLE_DEG))

        def turn(a: list[float], b: list[float]) -> float | None:
            try:
                normals = np.asarray(gmsh.model.getNormal(face, [*a, *b]), dtype=float).reshape(2, 3)
            except Exception:
                return None
            lengths = np.linalg.norm(normals, axis=1)
            if not (np.all(np.isfinite(lengths)) and np.all(lengths > 0)):
                return None
            return abs(float(np.dot(normals[0], normals[1]))) / float(lengths[0] * lengths[1])

        facing = turn(low, high)
        for _ in range(FEATURE_CREASE_HALVINGS):
            if facing is None or not facing < limit:
                return False
            middle = list(low)
            middle[axis] = 0.5 * (low[axis] + high[axis])
            first, second = turn(low, middle), turn(middle, high)
            if first is None or second is None:
                return False
            if first <= second:
                high, facing = middle, first
            else:
                low, facing = middle, second
        return facing is not None and facing < limit

    def _exact(self, point: np.ndarray) -> float:
        """OCC's distance from ``point`` to the nearest edge (its projection is
        bounded to each edge); ``0.0`` -- no claim -- if OCC cannot answer."""
        reach = (self.tolerance + float(np.max(self.segment_chord))) * (1.0 + 1e-9)
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


#: Exact OCC projections spent, per rejected plane, on the reported worst
#: off-model distance.
CERTIFIED_PEAK_CHECKS = 24


def _all_inside(face: int, uv: np.ndarray) -> bool:
    """Whether OCC counts every one of these (u, v) inside the face's trim
    (one call); ``False`` when it cannot answer."""
    if not len(uv) or not np.all(np.isfinite(uv)):
        return False
    try:
        return int(gmsh.model.isInside(2, face, uv.ravel().tolist(), parametric=True)) == len(uv)
    except Exception:
        return False


def _box_gaps(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Euclidean gap from each point (P, 3) to each box (F, 6): (P, F)."""
    deltas = np.maximum(np.maximum(boxes[None, :, :3] - points[:, None, :],
                                   points[:, None, :] - boxes[None, :, 3:]), 0.0)
    return np.sqrt(np.einsum("pfk,pfk->pf", deltas, deltas))


class _CertifiedOccMirrorTarget(_OccMirrorTarget):
    """The exact target, asked only what the forward tessellation cannot settle.

    Every mirrored sample is first measured against the tessellated faces
    (``occ_face_index``):

    * a **plane** face of the sample's role is a hit without OCC when the
      triangle's closest point is the perpendicular foot (where OCC projects
      onto a plane), within tolerance and certainly inside the trim -- and,
      with normals, the plane faces the sample's way;
    * a **curved** face the tessellation puts within tolerance is confirmed
      by one batched OCC projection onto that face alone: within tolerance,
      inside the trim, facing the right way;
    * a sample farther than tolerance (plus the trim query's nudge and the
      *measured* chord error) from every face is a miss without OCC;
    * every other sample takes the exact scan of :class:`_OccMirrorTarget`,
      in the original order.

    A miss of the trimmed face is a miss here even where the exact scan's
    trim query would clamp a projection from outside the face's parameter
    box back onto it (the fast check is the stricter there), and stays one
    when its distance is reported.
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

    @staticmethod
    def _facing(normal, other) -> bool:
        length = float(np.linalg.norm(other))
        if not np.isfinite(length) or length == 0.0:
            return True          # OCC cannot tell: no claim
        return abs(float(np.dot(normal, other))) / length >= FEATURE_NORMAL_ALIGNMENT

    def _carrier(self, face, point):
        """OCC's projection of ``point`` onto ``face``'s carrier: (distance,
        normal, whether the projection lies on the trimmed face)."""
        try:
            closest, parametric = gmsh.model.getClosestPoint(2, face, [float(x) for x in point])
            other = np.asarray(gmsh.model.getNormal(face, [float(parametric[0]), float(parametric[1])]),
                               dtype=float)
        except Exception:
            return None
        distance = float(np.linalg.norm(np.asarray(closest, dtype=float) - point))
        return distance, other, _parametric_point_on_face(face, parametric)

    def _confirm(self, face, points, normals):
        """One batched OCC projection of ``points`` onto curved ``face``:
        per point (hit, distance) -- a hit lies within tolerance, inside the
        trim, and (where a normal is given) faces its way."""
        count = len(points)
        hit = np.zeros(count, dtype=bool)
        distance = np.full(count, np.inf)
        try:
            closest, parametric = gmsh.model.getClosestPoint(2, face, points.ravel().tolist())
            closest = np.asarray(closest, dtype=float).reshape(-1, 3)
            uv = np.asarray(parametric, dtype=float).reshape(-1, 2)
        except Exception:
            return hit, distance
        if closest.shape != (count, 3) or uv.shape != (count, 2):
            return hit, distance
        distance = np.linalg.norm(closest - points, axis=1)
        near = np.flatnonzero(distance <= self.tolerance)
        if not len(near):
            return hit, distance
        trim = self.face_index.meshes[face].trim
        inside = np.zeros(len(near), dtype=bool)
        certain = np.zeros(len(near), dtype=bool)
        if trim is not None and trim.ok and trim.verified:
            inside, certain = trim.classify(uv[near])
            sure = np.flatnonzero(inside & certain)
            if len(sure) and not _all_inside(face, uv[near[sure]]):
                certain[:] = False
        for k in np.flatnonzero(~certain):
            inside[k] = _param_inside(face, float(uv[near[k], 0]), float(uv[near[k], 1]))
        hit[near[inside]] = True
        aim = np.flatnonzero(hit & np.all(np.isfinite(normals), axis=1))
        if len(aim):
            try:
                other = np.asarray(gmsh.model.getNormal(face, uv[aim].ravel().tolist()),
                                   dtype=float).reshape(-1, 3)
            except Exception:
                other = np.zeros((len(aim), 3))
            for k, p in enumerate(aim):
                hit[p] = self._facing(normals[p], other[k])
        return hit, distance

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
        matched = np.full(total, -1.0)
        nearest = np.full(total, -1.0)
        hits: list[list[int]] = [[] for _ in range(total)]
        passed = np.zeros(total, dtype=bool)
        # 1. Plane hits, proven by the tessellation.
        plane_rows = np.flatnonzero((near.upper <= tol) & same_role)
        if shape and len(plane_rows):
            facing = np.abs(np.einsum("ij,ij->i", mirrored_normals[near.point[plane_rows]],
                                      near.normal[plane_rows]))
            plane_rows = plane_rows[~known_normal[near.point[plane_rows]]
                                    | (near.normal_ok[plane_rows] & (facing >= FEATURE_NORMAL_ALIGNMENT))]
        # OCC must agree that every foot proven inside a face is inside: one
        # count per face (a count of n for n points proves all n).
        for face in np.unique(near.face[plane_rows]):
            rows = plane_rows[near.face[plane_rows] == face]
            if not _all_inside(int(face), near.uv[rows]):
                plane_rows = plane_rows[near.face[plane_rows] != face]
        # 2. Each point's faces nearest first: a proven plane hit settles it;
        # a curved candidate is confirmed by OCC, in rounds batched per face.
        proven = np.zeros(len(near.face), dtype=bool)
        proven[plane_rows] = True
        usable = np.flatnonzero((proven | (near.candidate & same_role)))
        usable = usable[np.lexsort((near.estimate[usable], near.point[usable]))]
        queue: dict[int, list[int]] = {}
        for row in usable:
            queue.setdefault(int(near.point[row]), []).append(int(row))
        plane_hits = confirmed = 0
        while queue:
            batch: dict[int, list[int]] = {}
            for p, rows in list(queue.items()):
                row = rows.pop(0)
                if not rows:
                    del queue[p]
                if proven[row]:
                    passed[p] = True
                    matched[p] = nearest[p] = max(float(near.upper[row]), 0.0)
                    hits[p] = [int(near.face[row])]
                    plane_hits += 1
                    queue.pop(p, None)
                else:
                    batch.setdefault(int(near.face[row]), []).append(p)
            for face, pts in batch.items():
                pts = np.asarray(pts)
                ok, distance = self._confirm(
                    face, mirrored[pts], mirrored_normals[pts] if shape else np.full((len(pts), 3), np.nan))
                for k, p in enumerate(pts):
                    if ok[k]:
                        passed[p] = True
                        matched[p] = nearest[p] = float(distance[k])
                        hits[p] = [face]
                        confirmed += 1
                        queue.pop(int(p), None)
        # 3. Certified misses.
        blocked = np.zeros(total, dtype=bool)
        blocked[near.point[~(near.lower > tol)]] = True
        # A point the index could not place is never a certified miss.
        blocked |= ~np.all(np.isfinite(mirrored), axis=1)
        # Faces the index could not tessellate: only their box can clear them.
        unknown = [face for face in self.candidates if face not in meshes and face in self.bboxes]
        if unknown:
            boxes = np.asarray([self.bboxes[face] for face in unknown], dtype=float)
            blocked |= np.any(~(_box_gaps(mirrored, boxes) > tol), axis=1)
        missed = ~passed & ~blocked
        # 4. The rest: the exact scan, in its order (every face, no early
        # exit when shapes are checked, so every hit's normal is seen).
        exact_done = np.zeros(total, dtype=bool)
        undecided = np.flatnonzero(~passed & ~missed)
        if len(undecided):
            local = self.index.candidates_for_points(mirrored[undecided])
            for row, p in enumerate(undecided):
                preferred = self._by_role.get(roles[order[owners[p]]]) if roles else None
                result = self.index.closest(
                    mirrored[p], local[row], preferred=preferred,
                    exit_at=-1.0 if shape else self._largest_residual,
                )
                matched[p], nearest[p], hits[p] = result[0], result[1], list(result[2])
                exact_done[p] = True
                if result[2] and not np.isnan(result[0]) and result[0] > self._largest_residual:
                    self._largest_residual = result[0]
        misaligned = np.zeros(total, dtype=bool)
        if shape:
            # Per point, once: the faces of its role the tessellation does
            # not clear of it.
            others: dict[int, set[int]] = {}
            for row in np.flatnonzero(same_role & ~(near.lower > tol)):
                others.setdefault(int(near.point[row]), set()).add(int(near.face[row]))
            for p in undecided:
                own = [h for h in hits[p] if roles.get(h) == own_role[p]]
                if not own or not known_normal[p]:
                    continue
                facing = False
                # Its own hits first; then any face of its role that OCC
                # puts within tolerance, its projection inside that face's
                # trim (a sample on a trim curve can project just outside
                # one face and onto the perpendicular face meeting it there).
                for face in [*own, *sorted(others.get(int(p), set()) - set(own))]:
                    answer = self._carrier(face, mirrored[p])
                    if answer is None:
                        facing = True
                        break
                    distance, other, on_face = answer
                    if (distance <= tol and (face in own or on_face)
                            and self._facing(mirrored_normals[p], other)):
                        facing = True
                        break
                misaligned[p] = not facing
        # 5. Misses: exact nearest distances for the reported failures and
        # the worst-looking ones; a certified miss stays a miss.
        reported = np.flatnonzero(missed)
        if len(reported):
            boxes_all = [face for face in self.candidates if face in self.bboxes]
            approx = np.full(total, np.inf)
            if boxes_all:
                boxes = np.asarray([self.bboxes[face] for face in boxes_all], dtype=float)
                column = {face: k for k, face in enumerate(boxes_all)}
                gaps = _box_gaps(mirrored[reported], boxes)
                position = {int(p): k for k, p in enumerate(reported)}
                for f, q, est in zip(near.face, near.point, near.estimate):
                    k = position.get(int(q))
                    c = column.get(int(f))
                    if k is not None and c is not None and gaps[k, c] <= tol:
                        gaps[k, c] = max(float(est), 0.0)
                approx[reported] = np.min(gaps, axis=1)
            failure_order = []
            for p in range(total):
                if missed[p] or (exact_done[p] and (
                        not hits[p] or own_role[p] not in {roles.get(h) for h in hits[p]}
                        or misaligned[p])):
                    failure_order.append(p)
                if len(failure_order) >= DEFAULT_MAX_FAILURE_SAMPLES:
                    break
            worst_first = sorted(reported, key=lambda q: -approx[q])[:CERTIFIED_PEAK_CHECKS]
            measured = np.zeros(total, dtype=bool)
            exact_worst = -1.0
            for p in [*failure_order, *worst_first]:
                if not missed[p] or measured[p]:
                    continue
                result = self._exact(mirrored[p])
                nearest[p] = float(result[1])
                measured[p] = True
                exact_worst = max(exact_worst, float(result[1]))
            for p in np.flatnonzero(missed & ~measured):
                value = float(approx[p]) if np.isfinite(approx[p]) else exact_worst
                nearest[p] = min(value, exact_worst) if exact_worst >= 0.0 else value
        self.stats = {
            "points": int(total),
            "proven_plane_hits": plane_hits,
            "confirmed_hits": confirmed,
            "certified_misses": int(np.count_nonzero(missed)),
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
    # One batched projection and one inside count settle the usual claims:
    # every sample on the carrier and inside the trim is "on", one farther
    # than tolerance from the carrier is "off" -- the answers the loop below
    # would give. Anything else takes the loop.
    try:
        closest, parametric = gmsh.model.getClosestPoint(2, parent, points.ravel().tolist())
        closest = np.asarray(closest, dtype=float).reshape(-1, 3)
        uv = np.asarray(parametric, dtype=float).reshape(-1, 2)
    except Exception:
        closest = uv = None
    if closest is not None and closest.shape == points.shape and len(uv) == len(points):
        gaps = np.linalg.norm(closest - points, axis=1)
        if np.all(np.isfinite(gaps)):
            if np.any(gaps > tolerance):
                return "off"
            if np.all(np.isfinite(uv)):
                try:
                    count = int(gmsh.model.isInside(2, parent, uv.ravel().tolist(), parametric=True))
                except Exception:
                    count = -1
                if count == len(points):
                    return "on"
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


class _ClaimJudge:
    """``_claim_evidence``, answered from the parents' tessellation when it can.

    The parents' forward tessellation (the mirror test's ``FaceIndex``,
    built before the cut from the very faces the copies copy) certifies a
    claim ``"on"`` when every sample of the child is a certified hit on the
    parent, and ``"off"`` when one is certainly farther than tolerance from
    the trimmed parent -- which ``_claim_evidence`` would also call off. Any
    other claim is judged by ``_claim_evidence`` on the parent's copy.

    It caches each child's samples by tag, so one judge serves one cut
    result: a later cut can reuse those tags for other faces.
    """

    def __init__(self, copies: dict[int, int], *, tolerance: float, grid: int,
                 face_index: occ_face_index.FaceIndex | None) -> None:
        self.copies = copies
        self.tolerance = tolerance
        self.grid = grid
        self.face_index = face_index
        self._near: dict[int, Any] = {}
        self._samples: dict[int, np.ndarray] = {}

    def _child(self, child: int):
        if child not in self._near:
            points = self._points(child)
            near = self.face_index.near(points, self.tolerance) if len(points) else None
            self._near[child] = (len(points), near)
        return self._near[child]

    def carrier_gap(self, parent: int, child: int) -> float:
        """The largest distance from the child's samples to the parent's
        carrier surface (OCC); ``inf`` when OCC cannot answer."""
        points = self._points(child)
        if not len(points):
            return math.inf
        try:
            closest, _parametric = gmsh.model.getClosestPoint(
                2, self.copies[parent], points.ravel().tolist())
        except Exception:
            return math.inf
        gaps = np.linalg.norm(np.asarray(closest, dtype=float).reshape(-1, 3) - points, axis=1)
        return float(np.max(gaps)) if np.all(np.isfinite(gaps)) else math.inf

    def _points(self, child: int) -> np.ndarray:
        if child not in self._samples:
            self._samples[child] = _sample_occ_surface(child, grid=self.grid, trims={})[0]
        return self._samples[child]

    def __call__(self, parent: int, child: int) -> str:
        index = self.face_index
        if index is not None and parent in index.meshes:
            count, near = self._child(child)
            if count and near is not None:
                rows = near.face == parent
                on = np.zeros(count, dtype=bool)
                on[near.point[rows & (near.upper <= self.tolerance)]] = True
                if np.all(on):
                    return "on"
                possible = np.zeros(count, dtype=bool)
                possible[near.point[rows & ~(near.lower > self.tolerance)]] = True
                if not np.all(possible):
                    return "off"
        return _claim_evidence(self.copies[parent], child, tolerance=self.tolerance, grid=self.grid)


def _verify_cut_parentage(
    mapping: dict[int, list[int]],
    copies: dict[int, int],
    remaining: list[int],
    *,
    tolerance: float,
    grid: int,
    judge: _ClaimJudge | None = None,
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
            verdict = (
                judge(parent, child) if judge is not None
                else _claim_evidence(copies[parent], child, tolerance=tolerance, grid=grid)
            )
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


def _solid_cut_applies(volumes, surfaces: list[int], planes: tuple[str, ...],
                       tolerance: float) -> bool:
    """Whether the model can be cut as its solids rather than face by face.

    Every face must bound exactly one solid (no free sheet; no face shared
    by two solids, which copying the solids would split in two), and none
    may lie in a cut plane (there the solid's cap and the face would
    coincide) -- by its box, or by its normal and centroid where OCC pads
    the box beyond the tolerance.
    """
    if not volumes:
        return False
    axes = [SYMMETRY_AXIS_FOR_PLANE[plane] for plane in planes]
    for surface in surfaces:
        if len(gmsh.model.getAdjacencies(2, surface)[0]) != 1:
            return False
        box = gmsh.model.getBoundingBox(2, surface)
        if any(abs(box[axis]) <= tolerance and abs(box[axis + 3]) <= tolerance for axis in axes):
            return False
        if _lies_in_a_cut_plane(surface, axes, tolerance):
            return False
    return True


def _lies_in_a_cut_plane(face: int, axes: list[int], hair: float) -> bool:
    """Whether ``face`` is flat and lies in one of the cut planes: its normal
    along that axis and its centroid on the plane (OCC's padded box cannot
    tell)."""
    if gmsh.model.getType(2, face) != "Plane":
        return False
    try:
        lo, hi = gmsh.model.getParametrizationBounds(2, face)
        normal = np.asarray(gmsh.model.getNormal(
            face, [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])]), dtype=float)
        centre = gmsh.model.occ.getCenterOfMass(2, face)
    except Exception:
        return False
    length = float(np.linalg.norm(normal))
    if not length > 0:
        return False
    return any(abs(abs(normal[axis]) / length - 1.0) <= 1.0e-9 and abs(centre[axis]) <= hair
               for axis in axes)


def _attribute_cut_faces(
    remaining: list[int], parents: list[int], boxes: Mapping[int, tuple[float, ...]],
    planes: tuple[str, ...], *, tolerance: float, judge: _ClaimJudge,
    owners: Mapping[int, str] | None = None,
) -> tuple[dict[int, list[int]], set[tuple[int, int]], list[int]]:
    """Parentage of the faces a solid cut left, from geometry alone.

    A boolean on solids reports no face history. Its caps are found first:
    flat faces lying in a cut plane (no parent lies there: the solid cut is
    not used otherwise). Every other face is claimed by the parents it lies
    on (``judge``: the evidence the face-by-face cut's claims are checked
    with), sure claims before undecided ones, and only by parents whose
    carrier it lies on to a hair -- a piece lies on its own parent's carrier
    exactly, a tangent neighbour only within tolerance. Anything that does
    not settle that way -- no parent, or an undecided claim shared between
    groups -- raises, so the caller cuts face by face instead.
    """
    axes = [SYMMETRY_AXIS_FOR_PLANE[plane] for plane in planes]
    hair = max(1.0e-3 * tolerance, 1.0e-12)
    mapping: dict[int, list[int]] = {parent: [] for parent in parents}
    undecided: set[tuple[int, int]] = set()
    caps: list[int] = []
    for child in remaining:
        box = gmsh.model.getBoundingBox(2, child)
        if _lies_in_a_cut_plane(child, axes, hair):
            caps.append(child)
            continue
        verdicts: dict[int, str] = {}
        for parent in parents:
            other = boxes[parent]
            if any(box[k] > other[k + 3] + tolerance or other[k] > box[k + 3] + tolerance
                   for k in range(3)):
                continue
            verdict = judge(parent, child)
            if verdict != "off":
                verdicts[parent] = verdict
        sure = {parent: verdict for parent, verdict in verdicts.items() if verdict == "on"}
        verdicts = sure or verdicts
        if len(verdicts) > 1:
            verdicts = {parent: verdict for parent, verdict in verdicts.items()
                        if judge.carrier_gap(parent, child) <= hair}
        if not verdicts:
            raise RuntimeError(f"OCC cut left surface {child} with no parent it lies on")
        if owners is not None and any(verdict == "unknown" for verdict in verdicts.values()) \
                and len({owners.get(parent, "") for parent in verdicts}) > 1:
            raise RuntimeError(f"OCC cut left surface {child} claimed undecided across groups")
        for parent, verdict in verdicts.items():
            mapping[parent].append(child)
            if verdict == "unknown":
                undecided.add((parent, child))
    return mapping, undecided, caps


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


def _cut_solids(
    volumes, lo: list[float], hi: list[float], parents: list[int],
    parent_boxes: Mapping[int, tuple[float, ...]], planes: tuple[str, ...], *,
    tolerance: float, grid: int, judge: _ClaimJudge,
    parent_samples: Mapping[int, np.ndarray] | None,
    owners: Mapping[int, str] | None = None,
) -> tuple[dict[int, list[int]], set[tuple[int, int]], list[int]] | None:
    """Keep the solids' positive side, then leave their faces open there.

    The cut works on a copy of the solids; the originals stay as the
    parents the evidence is checked against. When its result is proven
    (every surviving face a cap or claimed as ``_attribute_cut_faces``
    requires, every retained portion covered) the originals are removed and
    the parentage, undecided claims and surviving faces are returned.
    Otherwise nothing of it is left and ``None`` returned: the originals are
    untouched for the face-by-face cut.
    """
    keep = {tag for _dim, tag in gmsh.model.occ.getEntities(2)}
    keep_volumes = {tag for _dim, tag in volumes}
    try:
        solids = gmsh.model.occ.copy(list(volumes))
        box = gmsh.model.occ.addBox(lo[0], lo[1], lo[2], hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
        gmsh.model.occ.synchronize()
        gmsh.model.occ.intersect(solids, [(3, box)], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()
        result = [item for item in gmsh.model.getEntities(3) if item[1] not in keep_volumes]
        if result:
            gmsh.model.occ.remove(result, recursive=False)
            gmsh.model.occ.synchronize()
        remaining = [tag for _dim, tag in sorted(gmsh.model.getEntities(2)) if tag not in keep]
        mapping, undecided, caps = _attribute_cut_faces(
            remaining, parents, parent_boxes, planes, tolerance=tolerance, judge=judge,
            owners=owners)
        if caps:
            gmsh.model.occ.remove([(2, tag) for tag in caps], recursive=True)
            gmsh.model.occ.synchronize()
        remaining = [tag for tag in remaining if tag not in set(caps)]
        if _uncovered_cut_parents(mapping, {parent: parent for parent in parents}, planes,
                                  tolerance=tolerance, grid=grid, parent_samples=parent_samples):
            raise RuntimeError("OCC solid cut lost retained portions")
    except Exception:
        _discard_cut_result(keep, keep_volumes)
        return None
    # Proven: remove the original solids with their own faces.
    gmsh.model.occ.remove(list(volumes), recursive=True)
    gmsh.model.occ.synchronize()
    return mapping, undecided, remaining


def _discard_cut_result(keep: set[int], keep_volumes: set[int]) -> None:
    """Remove every entity a failed solid cut left, keeping the originals and copies."""
    solids = [item for item in gmsh.model.occ.getEntities(3) if item[1] not in keep_volumes]
    if solids:
        gmsh.model.occ.remove(solids, recursive=False)
    faces = [(2, tag) for _dim, tag in gmsh.model.occ.getEntities(2) if tag not in keep]
    if faces:
        gmsh.model.occ.remove(faces, recursive=True)
    gmsh.model.occ.synchronize()


def _cut_occ_geometry_to_positive_side(
    planes: tuple[str, ...],
    *,
    bbox: tuple[float, float, float, float, float, float],
    tolerance: float,
    grid: int,
    triangle_model: _OccTriangleModel | None = None,
    parent_samples: Mapping[int, np.ndarray] | None = None,
    face_index: occ_face_index.FaceIndex | None = None,
    solid_cut: bool = True,
    owners: Mapping[int, str] | None = None,
    report_route: bool = False,
    crossing_faces_only: bool = False,
) -> tuple[dict[int, list[int]], set[tuple[int, int]], dict[str, object]]:
    volumes = gmsh.model.getEntities(3)
    surfaces = [tag for _dim, tag in sorted(gmsh.model.getEntities(2))]
    # A model of solids is cut as its solids (stage 4b): OCC then intersects
    # the half-space with each solid, not every face with every other face
    # -- the face-against-face work made the face-by-face cut take seconds.
    # The face-by-face cut stays for open shells, and is the fallback.
    solid = (solid_cut and triangle_model is None and face_index is not None
             and _solid_cut_applies(volumes, surfaces, planes, tolerance))
    if volumes and not solid:
        gmsh.model.occ.remove(volumes, recursive=False)
        gmsh.model.occ.synchronize()
    parent_boxes = {tag: tuple(gmsh.model.getBoundingBox(2, tag)) for tag in surfaces}
    inside: set[int] = set()
    outside: set[int] = set()
    cut_parents = surfaces
    axes = [SYMMETRY_AXIS_FOR_PLANE[plane] for plane in planes]
    # The face-by-face cut of an open shell can give OCC only the faces that
    # cross a plane (stage 4b speed): a face whose box lies strictly on the
    # kept side is kept as it is, one strictly beyond a plane is dropped.
    # OCC boxes only ever grow a face, so neither can hide a crossing; a
    # face touching a plane is cut. Giving OCC every face made it intersect
    # every face with every other one -- seconds on an open shell.
    crossing_only = (crossing_faces_only and triangle_model is None
                     and all(np.all(np.isfinite(box)) for box in parent_boxes.values()))
    if triangle_model is not None or crossing_only:
        boxes = parent_boxes
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
        repaired: list[int] = []
        mapping: dict[int, list[int]] = {}
        undecided: set[tuple[int, int]] = set()
        disproved = 0
        remaining: list[int] = []
        cut_surfaces = cut_parents
        route = "faces"
        solid_result = None
        if solid:
            # The originals outlive the attempt, so they are the parents
            # (no copies needed unless it falls back).
            route = "solids"
            solid_result = _cut_solids(
                volumes, lo, hi, surfaces, parent_boxes, planes, tolerance=tolerance, grid=grid,
                judge=_ClaimJudge({surface: surface for surface in surfaces}, tolerance=tolerance,
                                  grid=grid, face_index=face_index),
                parent_samples=parent_samples, owners=owners,
            )
            if solid_result is not None:
                mapping, undecided, remaining = solid_result
                cut_surfaces = []
            else:
                # The originals are untouched: cut them face by face, as below.
                route = "solids, then faces"
                gmsh.model.occ.remove(volumes, recursive=False)
                gmsh.model.occ.synchronize()
        if solid_result is None:
            if triangle_model is None:
                for surface in surfaces:
                    copies[surface] = gmsh.model.occ.copy([(2, surface)])[0][1]
            elif cut_parents:
                copies = dict(zip(cut_parents, _copy_triangle_faces(cut_parents)))
            gmsh.model.occ.synchronize()
            if triangle_model is not None:
                _verify_triangle_copies(copies, triangle_model)
        for attempt in range(2) if cut_surfaces else ():
            # A fresh judge each attempt: the retry's children reuse the tags
            # of the attempt-0 children it removed, so samples cached under
            # those tags would describe faces that no longer exist.
            judge = (_ClaimJudge(copies, tolerance=tolerance, grid=grid, face_index=face_index)
                     if triangle_model is None else None)
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
                mapping, copies, remaining, tolerance=tolerance, grid=grid, judge=judge
            )
            uncovered = _uncovered_cut_parents(
                mapping, {parent: copies[parent] for parent in cut_parents}, planes,
                tolerance=tolerance, grid=grid,
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
            if crossing_only:
                # The retry cuts every face, as the plain face cut does: the
                # untouched faces would not share edges with standalone copies.
                gmsh.model.occ.remove([(2, tag) for tag in sorted(inside | outside)],
                                      recursive=True)
                crossing_only = False
                inside, outside = set(), set()
                cut_parents = surfaces
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
        elif crossing_only and solid_result is None:
            if outside:
                gmsh.model.occ.remove([(2, tag) for tag in sorted(outside)], recursive=True)
                gmsh.model.occ.synchronize()
            route += " (crossing faces)"
            remaining = sorted(set(remaining) | inside)
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
        **({"route": route} if report_route else {}),
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
    solid_cut: bool = True,
    crossing_faces_only: bool = False,
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

    ``solid_cut`` (with ``fast_check``) cuts a model of solids as its solids
    -- the same faces, but renumbered, so the mesh is not byte-identical to
    the face-by-face cut's; ``False`` keeps the face-by-face cut.
    ``crossing_faces_only`` lets the face-by-face cut (open shells, and the
    solid cut's fallback) give OCC only the faces that cross a cut plane;
    the others are kept or dropped whole -- the same faces, with the kept
    ones keeping their tags, so again not byte-identical.

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
    # Skipped when there is nothing to test and the cut would not use it.
    trims: dict[int, Any] | None = None
    face_index: occ_face_index.FaceIndex | None = None
    known = {str(plane): dict(verdict) for plane, verdict in (known_verdicts or {}).items()}
    untested = [plane for plane in candidate_planes if plane not in known]
    may_cut_solids = solid_cut and any(
        known[plane].get("accepted") for plane in candidate_planes if plane in known)
    if triangle_model is None and fast_check and (untested or may_cut_solids):
        box = gmsh.model.getBoundingBox(-1, -1)
        box_diagonal = float(np.linalg.norm(np.asarray(box[3:]) - np.asarray(box[:3])))
        face_index, trims = occ_face_index.build_face_index(
            surfaces, diagonal=box_diagonal, tolerance=tolerance_rel * box_diagonal,
            nudge_rel=AUTO_CUT_TRIM_NUDGE_REL, trims={},
        )
    sampled = {
        surface: _sample_occ_surface(surface, grid=grid, trims=trims) for surface in surfaces
    }
    samples = {surface: points for surface, (points, _uv) in sampled.items() if len(points)}
    normals: dict[int, np.ndarray | None] | None = None
    if face_index is not None and feature_check and untested:
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
    verdict_options = dict(samples=samples, roles=roles, bboxes=bboxes,
                           tolerance=tolerance, span_tolerance=span_tolerance)
    if role_boundaries is not None:
        verdict_options["_role_boundaries"] = role_boundaries
    oracle: dict[str, dict[str, int]] = {}
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
            triangle_model=triangle_model, parent_samples=samples, face_index=face_index,
            solid_cut=solid_cut, fast_check=fast_check, crossing_faces_only=crossing_faces_only,
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
    face_index: occ_face_index.FaceIndex | None = None,
    solid_cut: bool = True,
    fast_check: bool = False,
    crossing_faces_only: bool = False,
) -> OccAutoCutResult:
    model_bbox = tuple(float(value) for value in gmsh.model.getBoundingBox(-1, -1))
    cut_options: dict[str, Any] = dict(bbox=model_bbox, tolerance=tolerance, grid=grid)
    owners = {
        int(surface): group.name
        for group in groups
        for surface in group.selector.surface_tags
    }
    if triangle_model is None and fast_check:
        cut_options["report_route"] = True
    if face_index is not None:
        cut_options["face_index"] = face_index
        cut_options["solid_cut"] = solid_cut
        cut_options["owners"] = owners
    if crossing_faces_only:
        cut_options["crossing_faces_only"] = True
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
