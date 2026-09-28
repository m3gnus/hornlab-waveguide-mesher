"""STEP labels onto imported gmsh surfaces: face order, re-anchoring, healing.

Everything here that touches gmsh reads the live model, whose coordinates are
millimetres (OCC's default target unit). The caller owns every label, group
name and role string; this module only matches them.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
import sys
from types import TracebackType

import numpy as np

from .step_prepare import OccSurfaceRole
from .step_text import (
    advanced_face_body_positions_from_text,
    advanced_face_order_from_text,
    advanced_face_placements_from_text,
    advanced_face_surface_kinds_from_text,
    advanced_face_topology_from_text,
    parse_named_shell_faces_from_text,
    parse_styled_face_groups_from_text,
    read_step_text,
    step_body_inventory,
    step_length_unit_mm_from_text,
)

class _LazyGmsh:
    """Import gmsh on first use, never at module import."""

    def __getattr__(self, name: str):
        import gmsh as _gmsh_module

        globals()["gmsh"] = _gmsh_module
        return getattr(_gmsh_module, name)


gmsh = _LazyGmsh()


RIGID_TAG = 1
ANCHOR_MAX_AREA_REL_DIFF = 0.02
ANCHOR_MAX_CENTROID_DISTANCE_MM = 5.0

SurfaceGeometry = tuple[tuple[float, float, float], float]
# Each rung names the Gmsh geometry options a caller must set to 1 for that
# repair. This is a repair-STRATEGY contract, shared by consumers whose STEP
# files are not the same universe as ours, and a consumer reads it as "set each
# of these to 1".
#
# Geometry.OCCMakeSolids therefore does NOT belong here, however tempting the
# pairing looks. Sewing dissolves a solid body, and MakeSolids is what puts the
# volume back -- but whether that rebuild restores the geometry or corrupts it
# is a property of the FILE, not of the rung: on a body with an interior void
# it fills the cavity. Ask hornlab_mesher.step_text.occ_make_solids_is_safe of
# the STEP text instead, and see tests/test_step_import.py for the measurement.
OCC_HEALING_FALLBACKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Start with sewing: it resolves many Fusion periodic-face imports without
    # removing small valid faces. The broader repair remains a last resort.
    ("sew", ("Geometry.OCCSewFaces",)),
    (
        "full",
        (
            "Geometry.OCCFixDegenerated",
            "Geometry.OCCFixSmallEdges",
            "Geometry.OCCFixSmallFaces",
            "Geometry.OCCSewFaces",
        ),
    ),
)


@dataclass(frozen=True)
class StepLabelSelector:
    """Select a STEP group by its exact label.

    Deliberately just a label: the one alias policy that ever existed (a
    caller's legacy left/right names falling back to a generic one) was
    measured never to fire in 471 recorded runs and was deleted, taking the
    matching hook with it rather than leaving a hook with no caller. A caller
    that genuinely needs aliasing should add it back explicitly.
    """

    label: str


@dataclass(frozen=True)
class StepFaceGroup:
    """A caller-named STEP face selection with an opaque caller role."""

    name: str
    selector: StepLabelSelector
    role: OccSurfaceRole
    tag: int = RIGID_TAG
    resolution_mm: float = 0.0

def gmsh_surface_tags() -> list[int]:
    """Return current Gmsh surface tags in deterministic entity order."""
    return [tag for dim, tag in sorted(gmsh.model.getEntities(2))]


def gmsh_surface_geometries(surface_tags: list[int]) -> list[SurfaceGeometry]:
    """Return center-of-mass and area anchors for Gmsh surfaces."""
    return [
        (
            tuple(float(v) for v in gmsh.model.occ.getCenterOfMass(2, tag)),
            float(gmsh.model.occ.getMass(2, tag)),
        )
        for tag in surface_tags
    ]


class StepFaceOrderError(RuntimeError):
    """STEP faces could not be mapped one-to-one onto imported gmsh surfaces."""


#: Vertex coincidence tolerance for matching STEP faces to gmsh surfaces, in
#: model units (mm by default). Well above OCC's reader tolerance and far
#: below any real feature: two distinct vertices of one model are never this
#: close without being the same vertex.
FACE_MATCH_TOLERANCE = 1.0e-2  # gmsh model units (mm unless OCCTargetUnit says otherwise)

# Geometry.OCCTargetUnit -> millimetres; "" is OCC's default target, mm.
_OCC_TARGET_UNIT_MM = {
    "": 1.0, "MM": 1.0, "CM": 10.0, "M": 1000.0, "KM": 1.0e6,
    "UM": 1.0e-3, "MIC": 1.0e-3, "IN": 25.4, "INCH": 25.4, "FT": 304.8,
    "MI": 1609344.0, "MIL": 0.0254,
}


# gmsh surface type name -> step_text surface kind; unknown names never narrow.
_GMSH_SURFACE_KINDS = {
    "Plane": "plane",
    "Cylinder": "cylinder",
    "Cone": "cone",
    "Sphere": "sphere",
    "Torus": "torus",
    "BSpline surface": "bspline",
    "Bezier surface": "bezier",
    "Surface of Revolution": "revolution",
    "Surface of Extrusion": "extrusion",
}


def _gmsh_model_mm_per_step_unit(step_text_value: str) -> float:
    """Scale from STEP file coordinates to the live gmsh model's coordinates."""
    file_mm = step_length_unit_mm_from_text(step_text_value)
    if file_mm is None:
        raise StepFaceOrderError(
            "cannot map STEP faces to gmsh surfaces: the STEP file does not "
            "declare one recognisable length unit"
        )
    target = str(gmsh.option.getString("Geometry.OCCTargetUnit")).strip().upper()
    target_mm = _OCC_TARGET_UNIT_MM.get(target)
    if target_mm is None:
        raise StepFaceOrderError(
            f"cannot map STEP faces to gmsh surfaces: unknown Geometry.OCCTargetUnit {target!r}"
        )
    scaling = float(gmsh.option.getNumber("Geometry.OCCScaling"))
    return file_mm / target_mm * scaling


def _gmsh_surface_boundaries(
    surface_tags: list[int],
) -> tuple[list[np.ndarray], list[frozenset[int]]]:
    """Return each surface's boundary vertex coordinates and boundary curve tags."""
    cache: dict[int, np.ndarray] = {}
    points_out: list[np.ndarray] = []
    curves_out: list[frozenset[int]] = []
    for tag in surface_tags:
        boundary = gmsh.model.getBoundary(
            [(2, int(tag))], combined=False, oriented=False, recursive=True
        )
        points: list[np.ndarray] = []
        for dim, point_tag in boundary:
            if dim != 0:
                continue
            point_tag = abs(int(point_tag))
            if point_tag not in cache:
                cache[point_tag] = np.asarray(gmsh.model.getValue(0, point_tag, []), dtype=np.float64)
            points.append(cache[point_tag])
        points_out.append(np.asarray(points, dtype=np.float64).reshape(-1, 3))
        curves = gmsh.model.getBoundary(
            [(2, int(tag))], combined=False, oriented=False, recursive=False
        )
        curves_out.append(frozenset(abs(int(curve)) for dim, curve in curves if dim == 1))
    return points_out, curves_out


def _neighbours_from_shared(keys: list[frozenset[int]]) -> list[set[int]]:
    """Index i -> indices sharing at least one key with i."""
    owners: dict[int, list[int]] = defaultdict(list)
    for index, members in enumerate(keys):
        for key in members:
            owners[key].append(index)
    out: list[set[int]] = [set() for _ in keys]
    for members in owners.values():
        for index in members:
            out[index].update(members)
    for index, neighbours in enumerate(out):
        neighbours.discard(index)
    return out


def _face_identity_classes(step_text_value: str) -> dict[int, tuple[object, ...]]:
    """Return face id -> everything a label lookup can learn about that face.

    That is its named-shell labels, its style labels and whether its body is
    a solid (``parse_solid_brep_faces``). Two faces with equal classes are
    interchangeable for every lookup this package offers, so geometry that
    cannot tell them apart cannot mislabel anything either. An unnamed body
    carries no label, so which of two unnamed bodies a face belongs to is not
    part of its class.
    """
    named = parse_named_shell_faces_from_text(step_text_value)
    styled = parse_styled_face_groups_from_text(step_text_value)
    labels: dict[int, tuple[set[str], set[str], set[str]]] = defaultdict(
        lambda: (set(), set(), set())
    )
    for name, faces in named.items():
        for face in faces:
            labels[face][0].add(name)
    for name, faces in styled.items():
        for face in faces:
            labels[face][1].add(name)
    for body in step_body_inventory(step_text_value):
        for face in body.face_ids:
            labels[face][2].add(body.kind)
    return {
        face: (frozenset(a), frozenset(b), frozenset(c))
        for face, (a, b, c) in labels.items()
    }


def advanced_face_order_for_surfaces(
    step_path: Path,
    surfaces: list[int] | None = None,
    *,
    model_from_step: object | None = None,
    tolerance: float = FACE_MATCH_TOLERANCE,
    addressed_faces: Iterable[int] = (),
) -> list[int]:
    """Return the STEP ADVANCED_FACE id of each imported gmsh surface.

    The result is aligned with ``surfaces`` (default: every surface of the
    live model, in tag order), so ``dict(zip(result, surfaces))`` is the
    face -> surface map that ``zip(advanced_face_order(path), surfaces)``
    assumed. That older zip holds only when the file's record order happens
    to be the order OCC binds faces in. It often is (Fusion and OCC writers
    emit records in traversal order), but gmsh binds every solid's faces
    before any free face, and the STEP reader's shape fixing may reorder a
    shell: a Fusion export with a surface body written before a solid body,
    or a shell listing its faces in another order, silently mislabels.

    Faces are matched by geometry, in the live model:

    1. A surface's boundary vertices against the vertices each ADVANCED_FACE's
       loops reference, placed by the file's assembly transforms, converted
       to model units and moved by ``model_from_step`` (4x4; pass the
       transform already applied to the imported shapes, if any). A face whose
       vertices all lie on the surface is accepted when no face matches
       exactly -- OCC adds a seam and poles to a sphere written with one.
    2. Shell topology: a surface's candidates are narrowed to faces adjacent
       (sharing an EDGE_CURVE) to the faces its already-matched neighbours
       (sharing a curve) were matched to. This separates coincident faces
       of two touching bodies. So does the surface kind (plane, cylinder...)
       when every candidate's kind is known. Both only intersect: a narrowing
       that would leave no candidate refuses instead of being ignored.
    3. Faces still indistinguishable are ordered by record order *only* when
       they carry identical labels and body kind, so no label lookup can
       differ -- unless the caller names one of them in ``addressed_faces``
       (faces it will select by ADVANCED_FACE id), which makes each face its
       own label. Distinct bodies listed by one representation are ordered
       the way gmsh walks them (solids first, then list order). This is the
       one inference the text cannot prove: it is the observed walk order of
       gmsh's OCC (tested with the bodies apart, where geometry shows it),
       and it applies only to coincident faces of different bodies.

    Anything else raises :class:`StepFaceOrderError`: a count mismatch, a
    surface matching no face, or indistinguishable faces with different
    labels or bodies.
    """
    from scipy.spatial import cKDTree

    text = read_step_text(Path(step_path))
    record_order = advanced_face_order_from_text(text)
    topology = advanced_face_topology_from_text(text)
    placements = advanced_face_placements_from_text(text)
    if surfaces is None:
        surfaces = gmsh_surface_tags()
    surfaces = [int(tag) for tag in surfaces]
    if len(surfaces) != len(record_order):
        raise StepFaceOrderError(
            "cannot map STEP faces to gmsh surfaces: "
            f"{len(record_order)} ADVANCED_FACE records, {len(surfaces)} gmsh surfaces"
        )
    if not surfaces:
        return []
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive and finite, got {tolerance!r}")

    matrix = np.eye(4) if model_from_step is None else np.asarray(model_from_step, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("model_from_step must be a finite 4x4 matrix")
    unplaced = [face for face in record_order if placements.get(face) is None]
    if unplaced:
        raise StepFaceOrderError(
            "cannot map STEP faces to gmsh surfaces: the assembly placement of "
            f"face(s) {['#%d' % face for face in unplaced[:8]]} is not determined by "
            "the file (a part instanced more than once, or an unreadable placement)"
        )
    scale = _gmsh_model_mm_per_step_unit(text)

    face_points: list[np.ndarray] = []
    for face in record_order:
        raw = np.asarray(topology[face].vertices, dtype=np.float64).reshape(-1, 3)
        placed = np.asarray(placements[face], dtype=np.float64)
        raw = raw @ placed[:3, :3].T + placed[:3, 3]
        scaled = raw * scale
        face_points.append(scaled @ matrix[:3, :3].T + matrix[:3, 3])
    surface_points, surface_curves = _gmsh_surface_boundaries(surfaces)
    step_kinds = advanced_face_surface_kinds_from_text(text)
    face_kinds = [step_kinds.get(face) for face in record_order]
    surface_kinds = [_GMSH_SURFACE_KINDS.get(str(gmsh.model.getType(2, tag))) for tag in surfaces]
    face_neighbours = _neighbours_from_shared([topology[face].edges for face in record_order])
    surface_neighbours = _neighbours_from_shared(surface_curves)

    owners = np.concatenate(
        [np.full(len(points), index, dtype=np.int64) for index, points in enumerate(face_points)]
    )
    all_points = np.concatenate(face_points) if len(owners) else np.zeros((0, 3))
    tree = cKDTree(all_points) if len(all_points) else None
    vertexless_faces = {index for index, points in enumerate(face_points) if len(points) == 0}

    def within(a: np.ndarray, b: np.ndarray) -> bool:
        # every point of a lies within tolerance of some point of b
        distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        return bool(np.all(distances.min(axis=1) <= tolerance))

    candidates: list[set[int]] = []
    for points in surface_points:
        if len(points) == 0:
            candidates.append(set(vertexless_faces))
            continue
        touching: set[int] = set()
        if tree is not None:
            for hits in tree.query_ball_point(points, tolerance):
                touching.update(int(owners[hit]) for hit in hits)
        exact = {
            face
            for face in touching
            if within(face_points[face], points) and within(points, face_points[face])
        }
        if exact:
            candidates.append(exact)
            continue
        candidates.append(
            {face for face in touching if within(face_points[face], points)} | vertexless_faces
        )

    unmatched = [surfaces[index] for index, faces in enumerate(candidates) if not faces]
    if unmatched:
        raise StepFaceOrderError(
            "cannot map STEP faces to gmsh surfaces: no ADVANCED_FACE has the "
            f"boundary vertices of surface(s) {unmatched[:8]}"
            + (" and more" if len(unmatched) > 8 else "")
            + "; the imported model does not match the file geometry "
            "(was it moved, scaled or healed after import?)"
        )

    assigned: dict[int, int] = {}
    pending = set(range(len(surfaces)))
    changed = True
    while changed and pending:
        changed = False
        taken = set(assigned.values())
        for index in sorted(pending):
            options = candidates[index] - taken
            if not options:
                raise StepFaceOrderError(
                    "cannot map STEP faces to gmsh surfaces: surface "
                    f"{surfaces[index]} matches only faces already claimed by other surfaces"
                )
            if len(options) > 1:
                kind = surface_kinds[index]
                if kind is not None and all(face_kinds[face] is not None for face in options):
                    narrowed = {face for face in options if face_kinds[face] == kind}
                    if not narrowed:
                        raise StepFaceOrderError(
                            "cannot map STEP faces to gmsh surfaces: surface "
                            f"{surfaces[index]} is a {kind} in the model but none of its "
                            "candidate faces is"
                        )
                    options = narrowed
                for neighbour in surface_neighbours[index]:
                    face = assigned.get(neighbour)
                    if face is None:
                        continue
                    narrowed = options & face_neighbours[face]
                    if not narrowed:
                        raise StepFaceOrderError(
                            "cannot map STEP faces to gmsh surfaces: surface "
                            f"{surfaces[index]} borders a surface matched to face "
                            f"#{record_order[face]}, but none of its candidate faces does"
                        )
                    options = narrowed
            if len(options) == 1:
                assigned[index] = options.pop()
                pending.discard(index)
                taken = set(assigned.values())
                changed = True
            elif options != candidates[index]:
                candidates[index] = options
                changed = True

    if pending:
        classes = _face_identity_classes(text)
        addressed = {int(face) for face in addressed_faces}
        body_positions = advanced_face_body_positions_from_text(text)
        taken = set(assigned.values())
        groups: dict[frozenset[int], list[int]] = defaultdict(list)
        for index in sorted(pending):
            groups[frozenset(candidates[index] - taken)].append(index)
        for faces, members in groups.items():
            if len(faces) != len(members):
                raise StepFaceOrderError(
                    "cannot map STEP faces to gmsh surfaces: surfaces "
                    f"{[surfaces[i] for i in members]} share {len(faces)} candidate faces "
                    "that neither their vertices nor their neighbours tell apart"
                )
            ordered = sorted(faces)
            face_ids = [record_order[face] for face in ordered]
            if len({classes.get(face_id) for face_id in face_ids}) != 1 or (
                addressed & set(face_ids)
            ):
                # Not interchangeable. What is left is the order OCC walks
                # bodies in, which the text fixes only for distinct bodies
                # listed by one representation.
                keys = [body_positions.get(face_id) for face_id in face_ids]
                if (
                    any(key is None for key in keys)
                    or len({key[0] for key in keys}) != 1
                    or len({key[3] for key in keys}) != len(keys)
                ):
                    # (A caller-addressed face in a group of faces of one body
                    # lands here too: nothing orders faces within a body.)
                    raise StepFaceOrderError(
                        "cannot map STEP faces to gmsh surfaces unambiguously: faces "
                        f"{['#%d' % face_id for face_id in face_ids]} have the same boundary "
                        "vertices and neighbours but different labels or bodies, and nothing "
                        "in the file says which imported surface is which"
                    )
                ordered = [face for _key, face in sorted(zip(keys, ordered))]
            # Interchangeable faces keep record order, as before.
            for index, face in zip(sorted(members, key=lambda i: surfaces[i]), ordered):
                assigned[index] = face

    if sorted(assigned.values()) != list(range(len(record_order))):
        raise StepFaceOrderError(
            "cannot map STEP faces to gmsh surfaces: the match is not one-to-one"
        )
    return [record_order[assigned[index]] for index in range(len(surfaces))]


def _coerce_surface_geometry(geom: SurfaceGeometry) -> tuple[np.ndarray, float]:
    center, area = geom
    center_arr = np.asarray(center, dtype=np.float64)
    if center_arr.shape != (3,):
        raise RuntimeError(f"surface geometry center must have 3 coordinates, got {center!r}")
    area_float = float(area)
    if not np.all(np.isfinite(center_arr)) or not np.isfinite(area_float) or area_float <= 0.0:
        raise RuntimeError(f"invalid surface geometry center={center!r} area={area!r}")
    return center_arr, area_float


_AnchorPair = tuple[float, float, float, int, int]


def _anchor_pair(
    reference: list[tuple[np.ndarray, float]],
    healed: list[tuple[np.ndarray, float]],
    ref_index: int,
    healed_index: int,
) -> _AnchorPair:
    ref_center, ref_area = reference[ref_index]
    healed_center, healed_area = healed[healed_index]
    area_rel = abs(healed_area - ref_area) / max(ref_area, 1.0e-12)
    centroid_distance = float(np.linalg.norm(healed_center - ref_center))
    length_scale = max(float(np.sqrt(max(ref_area, healed_area))), 1.0)
    cost = (10.0 * area_rel) + (centroid_distance / length_scale)
    return (cost, area_rel, centroid_distance, ref_index, healed_index)


def _greedy_over(pairs: Iterable[_AnchorPair], n_reference: int) -> list[_AnchorPair]:
    """Take pairs cheapest first, each reference and healed surface once."""
    chosen: list[_AnchorPair] = []
    used_reference: set[int] = set()
    used_healed: set[int] = set()
    for pair in sorted(pairs):
        if pair[3] in used_reference or pair[4] in used_healed:
            continue
        chosen.append(pair)
        used_reference.add(pair[3])
        used_healed.add(pair[4])
        if len(used_reference) == n_reference:
            break
    return chosen


def _greedy_anchor_matches(
    reference: list[tuple[np.ndarray, float]],
    healed: list[tuple[np.ndarray, float]],
) -> list[_AnchorPair]:
    """The cheapest-first matching over ALL pairs, without building all pairs.

    Exactly the result of sorting every (reference, healed) pair by cost and
    taking greedily -- which is what this used to do, at 27 s and 441 MB for
    1500 faces. A pair's cost is at least ``distance / L`` with ``L`` the
    largest length scale in the model, so a pair more than ``c_r * L`` apart
    costs more than the match ``c_r`` its reference already got, and the
    greedy never reaches it with that reference free. The candidate set is
    grown (k nearest, then every pair inside that radius) until it provably
    contains every pair the full greedy could take; the greedy over the
    candidates is then the greedy over everything.
    """
    from scipy.spatial import cKDTree

    n = len(reference)
    if n == 0:
        return []
    healed_centers = np.asarray([center for center, _area in healed])
    reference_centers = np.asarray([center for center, _area in reference])
    largest_length = max(
        1.0,
        float(np.sqrt(max(area for _center, area in reference))),
        float(np.sqrt(max(area for _center, area in healed))),
    )
    tree = cKDTree(healed_centers)
    candidates: set[tuple[int, int]] = set()
    k = min(8, len(healed))
    while True:
        _distances, nearest = tree.query(reference_centers, k=k)
        nearest = np.asarray(nearest).reshape(n, -1)
        candidates.update(
            (ref_index, int(healed_index))
            for ref_index, row in enumerate(nearest)
            for healed_index in row
        )
        while True:
            pairs = [_anchor_pair(reference, healed, r, h) for r, h in candidates]
            chosen = _greedy_over(pairs, n)
            if len(chosen) < n:
                break  # incomplete: widen k
            cost_by_reference = {pair[3]: pair[0] for pair in chosen}
            missing: set[tuple[int, int]] = set()
            for ref_index in range(n):
                # Inclusive radius (plus rounding slack): anything outside it
                # costs strictly more than this reference's match.
                radius = cost_by_reference[ref_index] * largest_length * (1.0 + 1.0e-9) + 1.0e-12
                for healed_index in tree.query_ball_point(reference_centers[ref_index], radius):
                    if (ref_index, healed_index) not in candidates:
                        missing.add((ref_index, int(healed_index)))
            if not missing:
                return chosen
            candidates.update(missing)
        if k >= len(healed):
            return chosen
        k = min(len(healed), k * 4)


def anchor_surface_order(
    healed_tags: list[int],
    healed_geoms: list[SurfaceGeometry],
    reference_geoms: list[SurfaceGeometry],
) -> list[int]:
    """Rebuild STEP face order after OCC healing reorders imported surfaces."""
    if len(healed_tags) != len(healed_geoms):
        raise RuntimeError(
            "cannot anchor healed OCC surfaces: healed tag and geometry counts differ "
            f"({len(healed_tags)} tags vs {len(healed_geoms)} geometries)"
        )
    if len(healed_tags) != len(reference_geoms):
        raise RuntimeError(
            "cannot anchor healed OCC surfaces: surface count mismatch "
            f"({len(reference_geoms)} reference vs {len(healed_tags)} healed)"
        )
    if len(set(healed_tags)) != len(healed_tags):
        raise RuntimeError("cannot anchor healed OCC surfaces: healed surface tags are not unique")
    if not healed_tags:
        return []

    healed = [_coerce_surface_geometry(geom) for geom in healed_geoms]
    reference = [_coerce_surface_geometry(geom) for geom in reference_geoms]

    matches = _greedy_anchor_matches(reference, healed)
    ordered: list[int | None] = [None] * len(reference)
    residuals: dict[int, tuple[float, float, int]] = {}
    used_reference: set[int] = set()
    used_healed: set[int] = set()
    for _cost, area_rel, centroid_distance, ref_index, healed_index in matches:
        ordered[ref_index] = int(healed_tags[healed_index])
        residuals[ref_index] = (area_rel, centroid_distance, healed_index)
        used_reference.add(ref_index)
        used_healed.add(healed_index)

    if any(tag is None for tag in ordered) or len(used_healed) != len(healed_tags):
        raise RuntimeError(
            "cannot anchor healed OCC surfaces: failed to build a one-to-one "
            f"mapping ({len(used_reference)} reference, {len(used_healed)} healed matched)"
        )

    bad_matches = [
        (ref_index, ordered[ref_index], area_rel, centroid_distance)
        for ref_index, (area_rel, centroid_distance, _healed_index) in residuals.items()
        if area_rel > ANCHOR_MAX_AREA_REL_DIFF
        or centroid_distance > ANCHOR_MAX_CENTROID_DISTANCE_MM
    ]
    if bad_matches:
        diagnostics = "; ".join(
            (
                f"ref[{ref_index}] -> surface {tag}: "
                f"area_rel={area_rel:.4g}, centroid_mm={centroid_distance:.4g}"
            )
            for ref_index, tag, area_rel, centroid_distance in bad_matches[:5]
        )
        raise RuntimeError(
            "cannot anchor healed OCC surfaces: implausible geometry residuals; "
            f"{diagnostics}"
        )

    return [int(tag) for tag in ordered]

def run_occ_healing_fallbacks(
    run_attempt: Callable[..., dict[str, object]],
    *,
    original_mesh_error: Exception,
    original_traceback: TracebackType | None,
    surface_order_reference: list[SurfaceGeometry],
) -> tuple[dict[str, object], str, list[dict[str, object]]]:
    """Try OCC repairs without hiding the original unhealed mesh failure."""
    rejection_reasons: list[str] = []
    rejected_attempts: list[dict[str, object]] = []
    for healing_mode, occ_healing_options in OCC_HEALING_FALLBACKS:
        print(
            "gmsh mesh generation failed before healing; retrying with "
            f"OCC {healing_mode} repair. Original gmsh error: {original_mesh_error}",
            file=sys.stderr,
        )
        try:
            healed_state = run_attempt(
                occ_healing_options=occ_healing_options,
                surface_order_reference=surface_order_reference,
            )
        except Exception as exc:  # noqa: BLE001 - every rung failure is a rejection
            # A rung is rejected, not fatal: try the next one. Any Exception,
            # not a list of types: the gmsh Python API raises a bare
            # ``Exception`` for OCC failures (importShapes, intersect, getMass
            # inside a sew rung), and a caller's scope gate is commonly a
            # ValueError subclass. Either escaping used to stop the ladder
            # early AND replace the original unhealed mesh error, so the user
            # saw the wrong failure. Nothing is swallowed: every rejection
            # reason is returned in rejected_attempts and attached to the
            # original error as a note if no rung succeeds. BaseException
            # (KeyboardInterrupt, SystemExit) still propagates.
            reason = (
                f"OCC {healing_mode} repair rejected before meshing "
                f"({type(exc).__name__}): {exc}"
            )
            rejection_reasons.append(reason)
            rejected_attempts.append(
                {
                    "mode": healing_mode,
                    "options": list(occ_healing_options),
                    "reason": reason,
                }
            )
            print(f"{reason}; trying the next healing mode.", file=sys.stderr)
            continue

        healed_mesh_error = healed_state.get("mesh_generation_error")
        if healed_mesh_error is None:
            return healed_state, healing_mode, rejected_attempts
        reason = (
            f"OCC {healing_mode} repair mesh generation failed "
            f"({type(healed_mesh_error).__name__}): {healed_mesh_error}"
        )
        rejection_reasons.append(reason)
        rejected_attempts.append(
            {
                "mode": healing_mode,
                "options": list(occ_healing_options),
                "reason": reason,
            }
        )
        print(
            f"gmsh mesh generation still failed after OCC {healing_mode} repair. "
            f"Healed gmsh error: {healed_mesh_error}",
            file=sys.stderr,
        )

    rejection_summary = "; ".join(rejection_reasons)
    note = f"OCC healing fallback rejection reasons: {rejection_summary}"
    print(
        "gmsh mesh generation failed after all OCC healing fallbacks. "
        f"Original gmsh error: {original_mesh_error}. {note}",
        file=sys.stderr,
    )
    add_note = getattr(original_mesh_error, "add_note", None)
    if callable(add_note):
        add_note(note)
    raise original_mesh_error.with_traceback(original_traceback)
