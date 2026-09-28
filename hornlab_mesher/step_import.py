"""Caller-neutral STEP import, face mapping, and surface-mesh validation.

The caller owns every STEP label, group name, and role string. This
module only reasons about STEP entities, Gmsh surfaces, physical tags, and the
canonical surface-mesh contract.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from types import TracebackType

import meshio
import numpy as np

from .normals import open_shell_bore_alignment
from .step_prepare import OccSurfaceRole, snap_symmetry_plane_vertices
# Re-exported, not merely used: callers have always imported the STEP text
# parsers from this module, and several names below have no other use here.
from .step_text import (  # noqa: F401
    _STEP_CONTROL_RE,
    _STEP_RECORD_RE,
    BODY_ENTITIES,
    SOLID_BODY_ENTITIES,
    SURFACE_BODY_ENTITIES,
    VOIDED_SOLID_BODY_ENTITIES,
    StepBody,
    advanced_face_order,
    advanced_face_body_positions_from_text,
    advanced_face_order_from_text,
    advanced_face_placements_from_text,
    advanced_face_surface_kinds_from_text,
    advanced_face_topology_from_text,
    advanced_face_vertices,
    advanced_face_vertices_from_text,
    blank_step_strings,
    count_step_bodies,
    decode_step_string,
    first_step_string,
    occ_make_solids_is_safe,
    parse_named_shell_faces,
    parse_named_shell_faces_from_text,
    parse_solid_brep_faces,
    parse_styled_face_groups,
    parse_styled_face_groups_from_text,
    read_step_text,
    step_length_unit_mm_from_text,
    record_entity_types,
    step_body_inventory,
    step_records,
    step_refs,
    strip_step_comments,
)


class _LazyGmsh:
    """Import gmsh on first use, never at module import."""

    def __getattr__(self, name: str):
        import gmsh as _gmsh_module

        globals()["gmsh"] = _gmsh_module
        return getattr(_gmsh_module, name)


gmsh = _LazyGmsh()


RIGID_TAG = 1
SPEED_OF_SOUND_M_S = 343.0
FREQUENCY_ELEMENTS_PER_WAVELENGTH = 6.0
WELD_TOLERANCE_MM = 5.0e-3  # 5 micrometres; closes near-duplicate OCC patch nodes
DEGENERATE_MIN_QUALITY = 1.0e-4  # drops needle slivers that make dense solves singular
ANCHOR_MAX_AREA_REL_DIFF = 0.02
ANCHOR_MAX_CENTROID_DISTANCE_MM = 5.0

#: ``postprocess_mesh(symmetry_planes="auto")`` accepts a detected plane only
#: when the triangles along its open rim are this close to perpendicular to it
#: (median |normal . plane normal|; 0.25 is about 14.5 degrees). A cut mesh of
#: a smooth symmetric surface sits near 0 at any sane resolution.
AUTO_MIRROR_MAX_RIM_NORMAL_COMPONENT = 0.25

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


@dataclass(frozen=True)
class StepFaceMapping:
    """Mapped Gmsh surfaces and diagnostics for explicit STEP face groups."""

    surfaces: dict[str, list[int]]
    origins: dict[str, str]
    missing_reasons: dict[str, str]


# The Part 21 text layer lives in ``step_text`` because it must be importable
# where numpy and meshio are not -- an embedded CAD Python runs the same body
# rule. These aliases keep the private spellings this module and its tests have
# always used.
_step_records = step_records
_step_refs = step_refs
_decode_step_string = decode_step_string
_first_step_string = first_step_string
_parse_named_shell_faces = parse_named_shell_faces
_parse_solid_brep_faces = parse_solid_brep_faces
_parse_styled_face_groups = parse_styled_face_groups
_advanced_face_order = advanced_face_order


def map_step_face_groups(
    step_path: Path,
    groups: list[StepFaceGroup],
    *,
    skip_missing_groups: bool = False,
    gmsh_surfaces: list[int] | None = None,
    named_faces: dict[str, list[int]] | None = None,
    styled_faces: dict[str, list[int]] | None = None,
    face_order: list[int] | None = None,
) -> StepFaceMapping:
    """Map explicit caller-selected STEP labels onto imported Gmsh surfaces.

    Labels are compared exactly first and case-insensitively second. What a
    label MEANS is entirely caller-owned; this function only matches the string
    it is handed.

    ``face_order`` is the ADVANCED_FACE id of each entry of ``gmsh_surfaces``.
    When omitted it is matched by geometry in the live gmsh model
    (:func:`advanced_face_order_for_surfaces`), which refuses rather than
    guess; it used to be the file's record order, which is not the order gmsh
    numbers surfaces in.
    """
    if named_faces is None:
        named_faces = _parse_named_shell_faces(step_path)
    if styled_faces is None:
        styled_faces = _parse_styled_face_groups(step_path)
    if gmsh_surfaces is None:
        gmsh_surfaces = _gmsh_surface_tags()
    if face_order is None:
        face_order = advanced_face_order_for_surfaces(step_path, list(gmsh_surfaces))
    face_to_index = {face_id: index for index, face_id in enumerate(face_order)}

    if len(gmsh_surfaces) != len(face_order):
        raise RuntimeError(
            f"STEP has {len(face_order)} ADVANCED_FACE records but gmsh imported "
            f"{len(gmsh_surfaces)} surfaces"
        )

    candidates = (
        ("named shell/surface", named_faces),
        ("appearance/style", styled_faces),
    )

    def lookup_label(label: str) -> tuple[str, list[int]] | None:
        for origin, indexed_faces in candidates:
            if label in indexed_faces:
                return origin, indexed_faces[label]
        folded = label.casefold()
        for origin, indexed_faces in candidates:
            for available_label, faces in indexed_faces.items():
                if available_label.casefold() == folded:
                    return origin, faces
        return None

    def missing_message(label: str) -> str:
        available_named = ", ".join(sorted(named_faces)) or "(none)"
        available_styles = ", ".join(sorted(styled_faces)) or "(none)"
        return (
            f"group {label!r} not found as a named STEP shell/surface "
            f"or face appearance/style. Available shell names: {available_named}. "
            f"Available style names: {available_styles}"
        )

    mapping: dict[str, list[int]] = {}
    origins: dict[str, str] = {}
    missing: dict[str, str] = {}
    for group in groups:
        lookup = lookup_label(group.selector.label)
        if lookup is None:
            reason = missing_message(group.selector.label)
            if not skip_missing_groups:
                raise RuntimeError(reason)
            missing[group.name] = reason
            continue

        origin, face_ids = lookup
        surface_tags: list[int] = []
        for face_id in face_ids:
            if face_id not in face_to_index:
                raise RuntimeError(
                    f"face #{face_id} for group {group.name!r} is not an "
                    "ADVANCED_FACE"
                )
            surface_tags.append(gmsh_surfaces[face_to_index[face_id]])
        mapping[group.name] = surface_tags
        origins[group.name] = origin

    return StepFaceMapping(
        surfaces=mapping,
        origins=origins,
        missing_reasons=missing,
    )

def _gmsh_surface_tags() -> list[int]:
    return [tag for dim, tag in sorted(gmsh.model.getEntities(2))]


def _gmsh_surface_geometries(surface_tags: list[int]) -> list[SurfaceGeometry]:
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
FACE_MATCH_TOLERANCE_MM = 1.0e-2

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
    tolerance_mm: float = FACE_MATCH_TOLERANCE_MM,
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
       of two touching bodies.
    3. Faces still indistinguishable are ordered by record order *only* when
       they carry identical labels and body kind, so no label lookup can
       differ -- unless the caller names one of them in ``addressed_faces``
       (faces it will select by ADVANCED_FACE id), which makes each face its
       own label. Distinct bodies listed by one representation are ordered
       the way gmsh walks them (solids first, then list order).

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
        surfaces = _gmsh_surface_tags()
    surfaces = [int(tag) for tag in surfaces]
    if len(surfaces) != len(record_order):
        raise StepFaceOrderError(
            "cannot map STEP faces to gmsh surfaces: "
            f"{len(record_order)} ADVANCED_FACE records, {len(surfaces)} gmsh surfaces"
        )
    if not surfaces:
        return []
    tolerance = float(tolerance_mm)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"tolerance_mm must be positive and finite, got {tolerance_mm!r}")

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
                    if narrowed:
                        options = narrowed
                for neighbour in surface_neighbours[index]:
                    face = assigned.get(neighbour)
                    if face is None:
                        continue
                    narrowed = options & face_neighbours[face]
                    if narrowed:
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


def _anchor_surface_order(
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

    candidate_pairs: list[tuple[float, float, float, int, int]] = []
    for ref_index, (ref_center, ref_area) in enumerate(reference):
        for healed_index, (healed_center, healed_area) in enumerate(healed):
            area_rel = abs(healed_area - ref_area) / max(ref_area, 1.0e-12)
            centroid_distance = float(np.linalg.norm(healed_center - ref_center))
            length_scale = max(float(np.sqrt(max(ref_area, healed_area))), 1.0)
            cost = (10.0 * area_rel) + (centroid_distance / length_scale)
            candidate_pairs.append((cost, area_rel, centroid_distance, ref_index, healed_index))

    ordered: list[int | None] = [None] * len(reference)
    residuals: dict[int, tuple[float, float, int]] = {}
    used_reference: set[int] = set()
    used_healed: set[int] = set()
    for _cost, area_rel, centroid_distance, ref_index, healed_index in sorted(candidate_pairs):
        if ref_index in used_reference or healed_index in used_healed:
            continue
        ordered[ref_index] = int(healed_tags[healed_index])
        residuals[ref_index] = (area_rel, centroid_distance, healed_index)
        used_reference.add(ref_index)
        used_healed.add(healed_index)
        if len(used_reference) == len(reference):
            break

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


def _named_shell_gmsh_surfaces(
    step_path: Path,
    gmsh_surfaces: list[int],
    face_order: list[int] | None = None,
) -> dict[str, list[int]]:
    """Map each STEP named shell/body to its imported gmsh surface tags."""
    named_faces = _parse_named_shell_faces(step_path)
    if face_order is None:
        face_order = advanced_face_order_for_surfaces(step_path, list(gmsh_surfaces))
    face_to_index = {face_id: index for index, face_id in enumerate(face_order)}
    out: dict[str, list[int]] = {}
    for name, faces in named_faces.items():
        out[name] = sorted(
            {gmsh_surfaces[face_to_index[f]] for f in faces if f in face_to_index}
        )
    return out


def _map_refine_groups_to_gmsh_surfaces(
    step_path: Path,
    refine_specs: list[StepFaceGroup],
    gmsh_surfaces: list[int],
    face_order: list[int] | None = None,
) -> tuple[dict[str, list[int]], dict[str, str]]:
    """Resolve refine group names to gmsh surfaces (case-insensitive lookup).

    Missing refine names are skipped (they are optional overrides, unlike
    sources). Returns ``(name -> surfaces, name -> origin)``. ``face_order``
    is as in :func:`map_step_face_groups`.
    """
    named_faces = _parse_named_shell_faces(step_path)
    styled_faces = _parse_styled_face_groups(step_path)
    if face_order is None:
        face_order = advanced_face_order_for_surfaces(step_path, list(gmsh_surfaces))
    face_to_index = {face_id: index for index, face_id in enumerate(face_order)}

    def _lookup(name: str) -> tuple[str, list[int]] | None:
        for origin, groups in (("named shell/surface", named_faces), ("appearance/style", styled_faces)):
            if name in groups:
                return origin, groups[name]
        lower = name.lower()
        for origin, groups in (("named shell/surface", named_faces), ("appearance/style", styled_faces)):
            for label, faces in groups.items():
                if label.lower() == lower:
                    return origin, faces
        return None

    mapping: dict[str, list[int]] = {}
    origins: dict[str, str] = {}
    for spec in refine_specs:
        lookup = _lookup(spec.name)
        if lookup is None:
            continue
        origin, face_ids = lookup
        surfaces = sorted(
            {gmsh_surfaces[face_to_index[f]] for f in face_ids if f in face_to_index}
        )
        if surfaces:
            mapping[spec.name] = surfaces
            origins[spec.name] = origin
    return mapping, origins

def _mesh_triangle_data(mesh: meshio.Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "triangle" not in mesh.cells_dict:
        raise RuntimeError("mesh has no triangle cells")
    triangles = np.asarray(mesh.cells_dict["triangle"], dtype=np.int64)
    points = np.asarray(mesh.points, dtype=np.float64)
    try:
        tags = np.asarray(mesh.cell_data_dict["gmsh:physical"]["triangle"], dtype=np.int32)
    except KeyError as exc:
        raise RuntimeError("mesh has no gmsh:physical triangle tags") from exc
    return points, triangles, tags


def _triangle_area2(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    return np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def _remove_degenerate_triangles(
    points: np.ndarray,
    triangles: np.ndarray,
    tags: np.ndarray,
    *,
    eps: float = 1e-18,
    min_quality: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop zero-area triangles and, optionally, needle slivers.

    ``min_quality`` is a scale-invariant shape threshold: a triangle whose area
    falls below ``min_quality * longest_edge**2`` is removed. Fine OCC meshes
    carry micrometre-wide needles bridging near-duplicate patch-boundary nodes
    whose quadrature-degenerate rows make the dense metal-bem solve singular
    (LAPACK info > 0). Ported from hornlab_mesher.normals (commit a5539de).
    """
    if len(triangles) == 0:
        return triangles, tags, 0
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    area2 = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    keep = area2 > eps
    if min_quality > 0.0:
        longest_sq = np.maximum(
            np.maximum(
                np.sum((p1 - p0) ** 2, axis=1),
                np.sum((p2 - p1) ** 2, axis=1),
            ),
            np.sum((p0 - p2) ** 2, axis=1),
        )
        keep &= (0.5 * area2) > (min_quality * longest_sq)
    return triangles[keep], tags[keep], int(np.count_nonzero(~keep))


def _weld_near_duplicate_vertices(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    tol_mm: float = WELD_TOLERANCE_MM,
) -> np.ndarray:
    """Remap triangles so vertices closer than ``tol_mm`` coincide.

    Spatial hash with cells of the weld tolerance; clusters merge to the lowest
    vertex index via union-find. Closes the near-duplicate boundary nodes
    (micrometres apart) that OCC leaves between sewn patches on fine meshes,
    which otherwise seed singular slivers and spurious free edges. Ported from
    hornlab_mesher.mesher (commit a8c2648).
    """
    if len(points) == 0 or len(triangles) == 0:
        return triangles
    cells = np.floor(points / tol_mm).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(map(tuple, cells)):
        buckets.setdefault(key, []).append(index)

    parent = np.arange(len(points))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    tol_sq = tol_mm * tol_mm
    neighbor_offsets = [
        (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    ]
    for key, indices in buckets.items():
        candidates: list[int] = []
        for dx, dy, dz in neighbor_offsets:
            candidates.extend(buckets.get((key[0] + dx, key[1] + dy, key[2] + dz), ()))
        for i in indices:
            pi = points[i]
            for j in candidates:
                if j <= i:
                    continue
                delta = points[j] - pi
                if float(delta @ delta) <= tol_sq:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)

    roots = np.fromiter((find(i) for i in range(len(points))), dtype=np.int64, count=len(points))
    if np.array_equal(roots, np.arange(len(points))):
        return triangles
    return roots[triangles]


def _compact_unused_vertices(
    points: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop unreferenced vertices and renumber triangles to the survivors."""
    if len(triangles) == 0:
        return points, triangles
    used = np.unique(triangles)
    if len(used) == len(points):
        return points, triangles
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return points[used], remap[triangles]


def _edge_direction_stats(triangles: np.ndarray) -> dict[str, object]:
    edge_dirs: dict[tuple[int, int], list[int]] = defaultdict(list)
    for tri in np.asarray(triangles, dtype=np.int64):
        for start, end in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a = int(start)
            b = int(end)
            if a == b:
                continue
            if a < b:
                edge_dirs[(a, b)].append(1)
            else:
                edge_dirs[(b, a)].append(-1)

    boundary_edges = 0
    nonmanifold_edges = 0
    inconsistent_edges = 0
    for dirs in edge_dirs.values():
        if len(dirs) == 1:
            boundary_edges += 1
        elif len(dirs) != 2:
            nonmanifold_edges += 1
        elif dirs[0] == dirs[1]:
            inconsistent_edges += 1

    return {
        "n_edges": int(len(edge_dirs)),
        "boundary_edges": int(boundary_edges),
        "free_edges": int(boundary_edges),
        "nonmanifold_edges": int(nonmanifold_edges),
        "inconsistent_edges": int(inconsistent_edges),
    }


def _signed_volume(points: np.ndarray, triangles: np.ndarray) -> float:
    if len(triangles) == 0:
        return 0.0
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    return float(np.sum(p0 * np.cross(p1, p2)) / 6.0)


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


def _signed_volume_noise_floor(points: np.ndarray, triangles: np.ndarray) -> float:
    """Magnitude below which ``_signed_volume`` cannot be read as a sign."""

    if len(triangles) == 0:
        return 0.0
    area = 0.5 * float(np.sum(_triangle_area2(points, triangles)))
    return SIGNED_VOLUME_NOISE_REL * area**1.5


def _source_normal_projections(
    points: np.ndarray,
    triangles: np.ndarray,
    tags: np.ndarray,
    source_specs: list[StepFaceGroup],
) -> dict[str, dict[str, object]]:
    projections: dict[str, dict[str, object]] = {}
    for spec in source_specs:
        mask = tags == spec.tag
        if not np.any(mask):
            continue
        tri = triangles[mask]
        p0 = points[tri[:, 0]]
        p1 = points[tri[:, 1]]
        p2 = points[tri[:, 2]]
        vector = np.sum(np.cross(p1 - p0, p2 - p0), axis=0)
        projections[spec.name] = {
            "tag": int(spec.tag),
            "triangle_count": int(len(tri)),
            "vector_step_units2": [float(v) for v in vector],
            "projection_x_step_units2": float(vector[0]),
            "projection_y_step_units2": float(vector[1]),
            "projection_z_step_units2": float(vector[2]),
        }
    return projections


def _edge_on_expected_plane(
    points: np.ndarray,
    edge: tuple[int, int],
    planes: Iterable[str],
    tol: float,
) -> bool:
    for plane in planes:
        axis = {"x0": 0, "y0": 1, "z0": 2}[plane]
        if all(abs(float(points[vertex, axis])) <= tol for vertex in edge):
            return True
    return False


def _free_edges_on_expected_planes(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    symmetry_planes: tuple[str, ...],
    tolerance: float,
) -> bool:
    """True when every free edge lies wholly on a declared symmetry plane."""
    edge_count: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = tuple(sorted((int(a), int(b))))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    return all(
        _edge_on_expected_plane(points, edge, symmetry_planes, tolerance)
        for edge, count in edge_count.items()
        if count == 1
    )


#: Why :func:`_symmetry_source_projection_detail` declined to judge a component.
#: The distinction is load-bearing: ``no_source`` means there is no evidence and
#: the caller must abstain, while ``no_open_axis`` and ``degenerate_projection``
#: mean a cap exists but this particular measure cannot read it, so a different
#: oracle may still be able to.
_SYMMETRY_PROJECTION_OK = "ok"
_SYMMETRY_PROJECTION_NO_SOURCE = "no_source"
_SYMMETRY_PROJECTION_NO_OPEN_AXIS = "no_open_axis"
_SYMMETRY_PROJECTION_DEGENERATE = "degenerate_projection"
_SYMMETRY_PROJECTION_EMPTY = "empty_component"

#: How far the throat-collar bore fraction must sit from 0.5 before it counts as
#: a reading. At exactly half the collar reports that as much wall area faces
#: the bore as faces away, which is not a winding verdict at any threshold.
#:
#: Measured on flared half-horns, shallow waveguides and straight ducts, a
#: single-cap component reads exactly 1.0 wound correctly and exactly 0.0
#: inverted -- the intermediate values all come from geometries where the
#: radial reference is not the bore axis, and those are shapes to decline
#: rather than close calls to decide. 0.25 is therefore loose, not tight; it
#: exists to reject an even split, not to discriminate near one.
#:
#: It is deliberately NOT waveguide-generator's 0.9 bare-shell threshold
#: (server/mesh/integrity.py). That gate judges a whole finished mesh and can
#: demand near-perfection; this one picks a direction for one component and
#: must not turn a readable component into an unjudged one.
_BORE_ALIGNMENT_MARGIN = 0.25


def _symmetry_source_projection_detail(
    points: np.ndarray,
    triangles: np.ndarray,
    tags: np.ndarray,
    *,
    source_tags: set[int],
    symmetry_planes: tuple[str, ...],
) -> tuple[float | None, str]:
    """:func:`_symmetry_source_normal_projection`, plus why it abstained.

    Collapsing four different outcomes onto a bare ``None`` is what let a
    caller treat "this component has no source at all" and "this component has
    a source I cannot project" as the same situation, and apply one fallback to
    both. They need opposite responses, so the reason travels with the value.

    The value half is identical to :func:`_symmetry_source_normal_projection`,
    which delegates here. That function keeps its exact signature and return
    type because it is re-exported and called directly by the consuming Fusion
    add-in; widening it in place would have broken that consumer silently.
    """
    if len(triangles) == 0 or len(tags) != len(triangles):
        return None, _SYMMETRY_PROJECTION_EMPTY

    cut_axes = {
        {"x0": 0, "y0": 1, "z0": 2}[plane]
        for plane in symmetry_planes
    }
    open_axes = sorted({0, 1, 2} - cut_axes)

    source_mask = np.isin(tags, tuple(source_tags))
    if not np.any(source_mask):
        # Checked before the axis count so that a component with no cap reports
        # the reason that actually governs the caller's decision, whatever its
        # cut geometry happens to be.
        return None, _SYMMETRY_PROJECTION_NO_SOURCE

    if len(cut_axes) != 2 or len(open_axes) != 1:
        return None, _SYMMETRY_PROJECTION_NO_OPEN_AXIS

    source_triangles = triangles[source_mask]
    p0 = points[source_triangles[:, 0]]
    p1 = points[source_triangles[:, 1]]
    p2 = points[source_triangles[:, 2]]
    area_vectors = np.cross(p1 - p0, p2 - p0)
    total_area_vector = np.sum(area_vectors, axis=0)
    total_area = float(np.sum(np.linalg.norm(area_vectors, axis=1)))
    projection = float(total_area_vector[open_axes[0]])
    if total_area <= 0.0 or abs(projection) <= 1.0e-12 * total_area:
        return None, _SYMMETRY_PROJECTION_DEGENERATE
    return projection, _SYMMETRY_PROJECTION_OK


def _symmetry_source_normal_projection(
    points: np.ndarray,
    triangles: np.ndarray,
    tags: np.ndarray,
    *,
    source_tags: set[int],
    symmetry_planes: tuple[str, ...],
) -> float | None:
    """Project a reduced component's source-cap normal onto its open axis.

    Two orthogonal symmetry cuts leave one principal axis unconstrained. The
    positive direction of that remaining axis is the source/aperture winding
    contract used by the Metal solve. The source cap's net area vector is a
    local orientation anchor: it changes sign when the component is flipped,
    but is independent of translation and of any origin chosen for a volume
    sum. A proper rotation that carries the cut planes and remaining axis with
    the mesh preserves the projection.

    ``None`` means the component cannot be judged without guessing: it has no
    tagged source cap, does not have exactly two distinct principal cut
    planes, or its cap has no resolvable projection on the remaining axis.
    :func:`_symmetry_source_projection_detail` says which of those it was.

    Kept as a thin wrapper rather than widened in place: the Fusion add-in
    re-exports this name and calls it directly, so its signature and return
    type are a cross-repository contract.
    """
    projection, _reason = _symmetry_source_projection_detail(
        points,
        triangles,
        tags,
        source_tags=source_tags,
        symmetry_planes=symmetry_planes,
    )
    return projection


def _bore_alignment_verdict(
    points: np.ndarray,
    triangles: np.ndarray,
    tags: np.ndarray,
    *,
    source_tags: set[int],
) -> bool | None:
    """Is this component wound with its walls facing the bore?

    ``True`` keep, ``False`` flip, ``None`` no verdict.

    One cap at a time, and they must agree. ``open_shell_bore_alignment``
    derives its axis and radial origin from the caps it is given, so handing it
    several at once averages them into a reference that need not lie in the
    body at all: measured on a flared half-horn with a 25 mm throat cap and a
    400 mm mouth cap both declared, the combined centroid sat 169 mm off axis
    and an INVERTED mesh read 0.867 -- confident, and wrong. Asked about the
    throat cap alone the same mesh reads 0.0.

    So each declared cap present in the component is asked separately. A cap
    that cannot judge (no collar, a cancelling net area, or a fraction too near
    an even split to be a reading) simply does not vote; the mouth cap of a
    horn usually abstains this way. Disagreement between two caps that both
    voted is itself a reason to decline -- one of them is being measured about
    the wrong axis and this function cannot tell which.
    """
    present = [
        int(tag)
        for tag in np.unique(tags).tolist()
        if int(tag) in source_tags
    ]
    if not present:
        return None
    wall_tags = set(np.unique(tags).tolist()) - source_tags

    votes: set[bool] = set()
    for tag in present:
        alignment = open_shell_bore_alignment(
            points,
            triangles,
            tags,
            source_tags={tag},
            wall_tags=wall_tags,
        )
        if alignment is None or abs(alignment - 0.5) <= _BORE_ALIGNMENT_MARGIN:
            continue
        votes.add(alignment > 0.5)
    if len(votes) != 1:
        return None
    return votes.pop()


#: Which rule :func:`_source_anchored_verdict` used. Each is counted separately
#: by :func:`_repair_triangle_winding`, because they carry different confidence
#: and consumers warn about some and not others.
_SOURCE_RULE_PROJECTION = "projection"
_SOURCE_RULE_NO_SOURCE = "no_source"
_SOURCE_RULE_UNRESOLVED = "unresolved"
_SOURCE_RULE_BORE_ALIGNMENT = "bore_alignment"
_SOURCE_RULE_VOLUME_FALLBACK = "volume_fallback"


def _source_anchored_verdict(
    points: np.ndarray,
    triangles: np.ndarray,
    tags: np.ndarray,
    *,
    source_tags: set[int],
    symmetry_planes: tuple[str, ...],
) -> tuple[bool | None, str]:
    """Orient one symmetry-reduced component from its tagged source cap.

    ``True`` keep, ``False`` flip, ``None`` no verdict -- together with the
    rule that decided, which the caller counts.

    Positive source-normal projection on the one non-cut principal axis is the
    Metal aperture contract, and it is tried first. A component with no source
    cap at all is left unjudged, because nothing in it says which side the
    fluid is on. A component whose cap cannot be projected is judged by its
    throat collar, and the signed volume is only the last resort.
    """
    projection, projection_reason = _symmetry_source_projection_detail(
        points,
        triangles,
        tags,
        source_tags=source_tags,
        symmetry_planes=symmetry_planes,
    )
    if projection is not None:
        return projection > 0.0, _SOURCE_RULE_PROJECTION
    if projection_reason == _SYMMETRY_PROJECTION_NO_SOURCE:
        # No source cap at all, so nothing establishes which side of this
        # surface the fluid is on. Signed volume answers a different question
        # -- which way is out of the region the cut planes cap -- and the two
        # agree only where the meshed region is the solid. For an acoustic bare
        # shell they are opposed: ``normals.open_shell_bore_alignment`` records
        # that the volume "reports the *wrong* sign for correctly wound rollback
        # profiles", and the consuming add-in pins a reduced component whose
        # walls face the bore, and whose volume is therefore negative, as
        # correct (hornlab-fusion-addin,
        # test_symmetry_reduced_source_anchor_wins_when_signed_volume
        # _disagrees). Orienting that outward inverts every normal.
        #
        # Abstention rather than a guess is the decision this contract was
        # built on: hornlab-fusion-addin commit 3cf7d31 deleted a signed-volume
        # predicate here and replaced it with the source anchor. A component
        # can also reach this branch source-less because its STEP label did not
        # match and the caller passed ``skip_missing_groups`` -- recorded in
        # ``missing_reasons``, so not silent, but arriving here
        # indistinguishable from a body that never had a cap. Count it and let
        # the caller say so.
        return None, _SOURCE_RULE_NO_SOURCE
    # This component DOES carry a source cap, but the projection abstained
    # anyway: the cut planes leave no unique non-cut axis to project onto (a
    # single-plane cut, the common Fusion case, or a three-plane octant), or
    # the cap's net normal along that axis is degenerate.
    #
    # Degeneracy first, and on the volume, exactly as before. A component
    # enclosing a volume indistinguishable from zero is not a body this
    # function can orient by any means, and the collar below will still return
    # a confident fraction for one -- measured: the 10 x 10 x 1e-8 sliver in the
    # tests reads 0.0, not None, because the wall normals of a collapsed box
    # are perfectly well defined even though the box is not. Keeping this guard
    # in front preserves an abstention that predates this change rather than
    # quietly spending it.
    volume = _signed_volume(points, triangles)
    if abs(volume) <= _signed_volume_noise_floor(points, triangles):
        return None, _SOURCE_RULE_UNRESOLVED

    # A cap is still an anchor even when it cannot be projected onto a single
    # axis, so ask the throat collar instead of the volume. The collar measure
    # builds both its references out of the cap itself -- the cap's
    # area-weighted centroid and its net area vector -- so it needs no axis
    # from the cut planes and survives translation, rotation and reduced
    # domains. Where the volume oracle and this one disagree, this one is right
    # for an acoustic component: the volume answers "outward from the region
    # the cut planes cap", which inverts a bore-facing acoustic component, and
    # that inversion was live on exactly this path.
    verdict = _bore_alignment_verdict(
        points,
        triangles,
        tags,
        source_tags=source_tags,
    )
    if verdict is not None:
        return verdict, _SOURCE_RULE_BORE_ALIGNMENT
    # The collar could not judge it -- no wall band, a cancelling cap, or a
    # fraction too close to half to be a reading rather than a verdict. Fall
    # back to the signed volume, already known to clear its noise floor. It
    # remains the wrong question for an acoustic component, so it is counted
    # separately and the caller reports it.
    return volume > 0.0, _SOURCE_RULE_VOLUME_FALLBACK


#: How :func:`_repair_triangle_winding` orients a symmetry-reduced component.
#:
#: ``source-anchor`` (the default) trusts the tagged source cap, which is the
#: contract of the Fusion add-in: its reduced components may enclose FLUID
#: (a bore-facing acoustic shell), and for those the signed volume is inverted.
#:
#: ``mirrored-parent`` orients the component as its mirrored, watertight parent
#: would be oriented by the closed branch: outward from the enclosed region.
#: That is the right contract for a caller whose meshed regions are solids and
#: whose full-domain mesh is oriented by signed volume -- because it is the
#: only one under which the reduced domain and the same body meshed whole
#: always agree. A tagged source facing away from the open axis (the rear face
#: of a throat plug, say) is correct in the full domain and would be inverted,
#: with every other triangle of the component, by the source anchor.
REDUCED_ORIENTATION_SOURCE_ANCHOR = "source-anchor"
REDUCED_ORIENTATION_MIRRORED_PARENT = "mirrored-parent"
REDUCED_ORIENTATIONS = (
    REDUCED_ORIENTATION_SOURCE_ANCHOR,
    REDUCED_ORIENTATION_MIRRORED_PARENT,
)


def _repair_triangle_winding(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    tags: np.ndarray | None = None,
    source_tags: set[int] | None = None,
    symmetry_planes: tuple[str, ...] = (),
    tolerance: float = 0.0,
    reduced_orientation: str = REDUCED_ORIENTATION_SOURCE_ANCHOR,
) -> tuple[np.ndarray, dict[str, int]]:
    """Repair manifold winding and orient components with a valid anchor.

    Watertight components retain the signed-volume outwardness contract.

    A symmetry-reduced open component -- every free edge on a cut plane, so
    its mirrored parent is watertight -- is oriented according to
    ``reduced_orientation`` (see ``REDUCED_ORIENTATIONS``):

    * ``source-anchor``, the default, uses the tagged source cap as
      :func:`_source_anchored_verdict` describes: the source projection on the
      one non-cut axis, then the throat collar, then the signed volume, and no
      verdict at all for a component without a cap.
    * ``mirrored-parent`` uses the parent's signed volume, which is the
      component's own origin-based signed volume (see below), and falls back to
      the source anchor only when that volume is within its noise floor.

    Both verdicts are taken for every reduced component, and a disagreement is
    counted in ``symmetry_source_parent_conflicts`` whichever mode wins. The
    two disagree where the source-anchored rules misread a solid -- a tagged
    source facing away from the open axis, or a throat collar read on the
    outside of a wall -- or where the component encloses fluid rather than
    solid. Which of them is right there depends on what the meshed region is,
    and only the caller knows that. A flip against the other contract is never
    silent.
    """
    if reduced_orientation not in REDUCED_ORIENTATIONS:
        raise ValueError(
            f"reduced_orientation must be one of {REDUCED_ORIENTATIONS}, "
            f"not {reduced_orientation!r}"
        )
    repaired = triangles.copy()
    stats = {
        "flipped_consistency": 0,
        "flipped_global": 0,
        "unjudged_symmetry_components": 0,
        "unjudged_symmetry_no_source": 0,
        "symmetry_volume_fallback_flipped": 0,
        "symmetry_volume_fallback_kept": 0,
        "unresolved_symmetry_components": 0,
        "bore_alignment_flipped": 0,
        "bore_alignment_kept": 0,
        "symmetry_parent_volume_flipped": 0,
        "symmetry_parent_volume_kept": 0,
        "symmetry_source_parent_conflicts": 0,
    }
    if len(repaired) == 0:
        return repaired, stats
    component_tags = (
        np.asarray(tags, dtype=np.int32)
        if tags is not None
        else np.full(len(repaired), RIGID_TAG, dtype=np.int32)
    )
    if len(component_tags) != len(repaired):
        raise ValueError("triangle and physical-tag counts differ")
    declared_source_tags = set(source_tags or ())

    edge_to_triangles: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for tri_idx, tri in enumerate(repaired):
        for start, end in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a = int(start)
            b = int(end)
            if a == b:
                continue
            if a < b:
                edge_to_triangles[(a, b)].append((tri_idx, 1))
            else:
                edge_to_triangles[(b, a)].append((tri_idx, -1))

    neighbours: list[list[tuple[int, bool]]] = [[] for _ in range(len(repaired))]
    for uses in edge_to_triangles.values():
        if len(uses) != 2:
            continue
        (tri_a, dir_a), (tri_b, dir_b) = uses
        must_differ = dir_a == dir_b
        neighbours[tri_a].append((tri_b, must_differ))
        neighbours[tri_b].append((tri_a, must_differ))

    flip = np.zeros(len(repaired), dtype=bool)
    seen = np.zeros(len(repaired), dtype=bool)
    components: list[np.ndarray] = []
    for seed in range(len(repaired)):
        if seen[seed]:
            continue
        seen[seed] = True
        queue: deque[int] = deque([seed])
        component: list[int] = []
        while queue:
            tri_idx = queue.popleft()
            component.append(tri_idx)
            for other, must_differ in neighbours[tri_idx]:
                required = bool(flip[tri_idx]) ^ bool(must_differ)
                if seen[other]:
                    continue
                flip[other] = required
                seen[other] = True
                queue.append(other)
        components.append(np.asarray(component, dtype=np.int64))

    if np.any(flip):
        repaired[flip] = repaired[flip][:, [0, 2, 1]]
        stats["flipped_consistency"] = int(np.count_nonzero(flip))

    for component in components:
        component_triangles = repaired[component]
        edge_stats = _edge_direction_stats(component_triangles)
        closed = (
            edge_stats["boundary_edges"] == 0
            and edge_stats["nonmanifold_edges"] == 0
        )
        symmetry_reduced = (
            edge_stats["boundary_edges"] > 0
            and edge_stats["nonmanifold_edges"] == 0
            and _free_edges_on_expected_planes(
                points,
                component_triangles,
                symmetry_planes=symmetry_planes,
                tolerance=tolerance,
            )
        )
        if edge_stats["inconsistent_edges"] != 0:
            continue
        if closed:
            if _signed_volume(points, component_triangles) < 0.0:
                repaired[component] = component_triangles[:, [0, 2, 1]]
                stats["flipped_global"] += int(len(component))
            continue
        if not symmetry_reduced:
            continue

        source_keep, source_rule = _source_anchored_verdict(
            points,
            component_triangles,
            component_tags[component],
            source_tags=declared_source_tags,
            symmetry_planes=symmetry_planes,
        )
        # What the mirrored parent says. Every free edge of this component lies
        # on a cut plane through the origin, so mirroring closes it: the parent
        # is watertight. The cut-plane caps a closed piece would need add
        # nothing to the origin-based volume sum either, because r . n vanishes
        # on a plane through the origin. This component's own signed volume is
        # therefore exactly 1/2^k of its parent's, sign included -- the verdict
        # the closed branch above gives the same body meshed whole.
        volume = _signed_volume(points, component_triangles)
        parent_keep = (
            None
            if abs(volume) <= _signed_volume_noise_floor(points, component_triangles)
            else volume > 0.0
        )
        if (
            source_keep is not None
            and parent_keep is not None
            and source_keep != parent_keep
        ):
            stats["symmetry_source_parent_conflicts"] += 1

        if (
            reduced_orientation == REDUCED_ORIENTATION_MIRRORED_PARENT
            and parent_keep is not None
        ):
            if parent_keep:
                stats["symmetry_parent_volume_kept"] += 1
            else:
                repaired[component] = component_triangles[:, [0, 2, 1]]
                stats["flipped_global"] += int(len(component))
                stats["symmetry_parent_volume_flipped"] += 1
            continue

        if source_rule != _SOURCE_RULE_PROJECTION:
            stats["unjudged_symmetry_components"] += 1
        if source_rule == _SOURCE_RULE_NO_SOURCE:
            stats["unjudged_symmetry_no_source"] += 1
        elif source_rule == _SOURCE_RULE_UNRESOLVED:
            stats["unresolved_symmetry_components"] += 1
        elif source_rule == _SOURCE_RULE_BORE_ALIGNMENT:
            stats["bore_alignment_kept" if source_keep else "bore_alignment_flipped"] += 1
        elif source_rule == _SOURCE_RULE_VOLUME_FALLBACK:
            stats[
                "symmetry_volume_fallback_kept"
                if source_keep
                else "symmetry_volume_fallback_flipped"
            ] += 1
        if source_keep is False:
            repaired[component] = component_triangles[:, [0, 2, 1]]
            stats["flipped_global"] += int(len(component))

    return repaired, stats

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
    edges = np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    if len(edges):
        _unique, first, counts = np.unique(
            edges, axis=0, return_index=True, return_counts=True
        )
        free_rows = first[counts == 1]
    else:
        free_rows = np.zeros(0, dtype=np.int64)
    free_edges = edges[free_rows]
    free_edge_triangles = free_rows // 3

    on_plane = np.stack(
        [np.all(np.abs(points[free_edges, axis]) <= tolerance, axis=1) for axis in range(3)],
        axis=1,
    ) if len(free_edges) else np.zeros((0, 3), dtype=bool)
    planes_per_edge = on_plane.sum(axis=1)
    plane_counts = {
        plane: int(np.count_nonzero(on_plane[:, axis] & (planes_per_edge == 1)))
        for axis, plane in enumerate(("x0", "y0", "z0"))
    }
    shared_plane_edges = int(np.count_nonzero(planes_per_edge > 1))

    # How squarely the surface meets each plane along its rim: the median
    # |normal . plane normal| of the triangles owning the plane's free edges.
    # A mirror-symmetric smooth surface crosses its mirror plane at a right
    # angle (0); a horn mouth that merely ends on the plane meets it at its
    # flare angle. Reported only -- :func:`postprocess_mesh`'s "auto" mode is
    # what acts on it.
    rim_normal_component: dict[str, float | None] = {}
    for axis, plane in enumerate(("x0", "y0", "z0")):
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
    for axis, plane in enumerate(("x0", "y0", "z0")):
        coordinates = points[:, axis]
        negative = int(np.count_nonzero(coordinates < -tolerance))
        positive = int(np.count_nonzero(coordinates > tolerance))
        plane_vertex_side_counts[plane] = {
            "negative": negative,
            "on_plane": int(len(coordinates) - negative - positive),
            "positive": positive,
        }
        one_sided[plane] = not (negative and positive)

    detected = tuple(
        plane for plane in ("x0", "y0", "z0")
        if plane_counts[plane] >= min_edges_per_plane and one_sided[plane]
    )
    rejected_spanning_planes = [
        plane for plane in ("x0", "y0", "z0")
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


# The detector is public because it is the only way to re-read a cut from a
# finished mesh with no knowledge of what was cut, which is what makes an
# auto-cut self-checking: a plane that was cut but does not come back as an
# open rim was capped, and a capped plane meshes as a rigid baffle rather than
# as a symmetry plane.
_detect_symmetry_planes = detect_symmetry_planes

_snap_symmetry_plane_vertices = snap_symmetry_plane_vertices


def _normalize_to_positive_side(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    symmetry_planes: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Reflect the reduced domain onto the positive side of each cut plane.

    The Metal native symmetry solve requires the reduced mesh on the positive
    side of its symmetry planes. A model cut to a negative quadrant in CAD is
    the mirror image of the equivalent positive-quadrant model, so reflecting
    it (and flipping winding once per reflection to keep normals outward)
    changes nothing about the represented full-domain geometry.
    """
    axis_for_plane = {"x0": 0, "y0": 1, "z0": 2}
    reflected_axes: list[str] = []
    points = points.copy()
    triangles = triangles.copy()
    for plane in symmetry_planes:
        axis = axis_for_plane[plane]
        coords = points[:, axis]
        if -float(coords.min()) > float(coords.max()):
            points[:, axis] = -points[:, axis]
            reflected_axes.append("xyz"[axis])
    if len(reflected_axes) % 2 == 1:
        triangles = triangles[:, [0, 2, 1]]
    normalization = {
        "symmetry_planes": list(symmetry_planes),
        "reflected_axes": reflected_axes,
    }
    return points, triangles, normalization


def postprocess_mesh(
    mesh: meshio.Mesh,
    source_specs: list[StepFaceGroup],
    *,
    symmetry_planes: tuple[str, ...] | str,
    tolerance: float,
    symmetry_snap_tolerance: float | None = None,
    reduced_orientation: str = REDUCED_ORIENTATION_SOURCE_ANCHOR,
) -> tuple[meshio.Mesh, dict[str, object], dict[str, object]]:
    """Repair and validate a tagged surface mesh without interpreting roles.

    ``reduced_orientation`` selects how a symmetry-reduced component is
    oriented; see ``REDUCED_ORIENTATIONS`` and :func:`_repair_triangle_winding`.
    The mode used is echoed in the repair report.
    """
    points, triangles, tags = _mesh_triangle_data(mesh)
    # A copy: welding reads and the symmetry snap writes these coordinates,
    # and the caller's mesh must come back as it was handed in.
    points = np.array(points, dtype=np.float64, copy=True)
    before_edge_stats = _edge_direction_stats(triangles)
    before_signed_volume = _signed_volume(points, triangles)
    distinct_before = int(len(np.unique(triangles))) if len(triangles) else 0
    triangles = _weld_near_duplicate_vertices(points, triangles)
    welded_vertices = max(0, distinct_before - (int(len(np.unique(triangles))) if len(triangles) else 0))
    triangles, tags, degenerate_removed = _remove_degenerate_triangles(
        points, triangles, tags, min_quality=DEGENERATE_MIN_QUALITY
    )

    symmetry_detection: dict[str, object] | None = None
    symmetry_tolerance = (
        symmetry_snap_tolerance
        if symmetry_snap_tolerance is not None
        else tolerance
    )
    if symmetry_planes == "auto":
        detected, symmetry_detection = detect_symmetry_planes(
            points,
            triangles,
            tolerance=symmetry_tolerance,
        )
        # An open rim on a coordinate plane is a cut only if the surface
        # meets the plane squarely there. A full open horn whose mouth rim
        # happens to lie on z=0 passes every other test, and mirroring it
        # would solve a horn pair.
        components = symmetry_detection["rim_normal_component"]
        symmetry_planes = tuple(
            plane
            for plane in detected
            if components.get(plane) is not None
            and components[plane] <= AUTO_MIRROR_MAX_RIM_NORMAL_COMPONENT
        )
        symmetry_detection["rejected_non_mirror_planes"] = [
            plane for plane in detected if plane not in symmetry_planes
        ]
        symmetry_detection["max_rim_normal_component"] = AUTO_MIRROR_MAX_RIM_NORMAL_COMPONENT
        symmetry_detection["detected_planes"] = list(symmetry_planes)
    repaired_triangles, repair_stats = _repair_triangle_winding(
        points,
        triangles,
        tags=tags,
        source_tags={spec.tag for spec in source_specs},
        symmetry_planes=symmetry_planes,
        tolerance=symmetry_tolerance,
        reduced_orientation=reduced_orientation,
    )
    repair_stats["welded_vertices"] = int(welded_vertices)
    after_edge_stats = _edge_direction_stats(repaired_triangles)
    if symmetry_snap_tolerance is not None:
        snap_symmetry_plane_vertices(
            points,
            symmetry_planes=symmetry_planes,
            tolerance=symmetry_snap_tolerance,
        )

    points, repaired_triangles, axis_normalization = _normalize_to_positive_side(
        points,
        repaired_triangles,
        symmetry_planes=symmetry_planes,
    )

    # Drop vertices orphaned by welding/degenerate removal so the written node
    # count matches the live mesh the solver assembles.
    points, repaired_triangles = _compact_unused_vertices(points, repaired_triangles)
    after_signed_volume = _signed_volume(points, repaired_triangles)

    repaired_mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", repaired_triangles)],
        cell_data={
            "gmsh:physical": [tags.astype(np.int32, copy=False)],
            "gmsh:geometrical": [tags.astype(np.int32, copy=False)],
        },
        field_data=mesh.field_data,
    )
    topology = _topology_stats(
        points,
        repaired_triangles,
        symmetry_planes=symmetry_planes,
        tolerance=tolerance,
    )
    if symmetry_detection is not None:
        topology["symmetry_plane_detection"] = symmetry_detection
    topology["axis_normalization"] = axis_normalization
    topology["signed_volume_step_units3"] = after_signed_volume
    topology["source_normal_projections"] = _source_normal_projections(
        points,
        repaired_triangles,
        tags,
        source_specs,
    )
    repair = {
        "degenerate_triangles_removed": int(degenerate_removed),
        "reduced_orientation": reduced_orientation,
        **repair_stats,
        "before": {
            **before_edge_stats,
            "signed_volume_step_units3": before_signed_volume,
        },
        "after": {
            **after_edge_stats,
            "signed_volume_step_units3": after_signed_volume,
        },
    }
    return repaired_mesh, repair, topology


def _triangle_edge_lengths(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=np.int64)
    if len(triangles) == 0:
        return np.empty(0, dtype=np.float64)
    edges = triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    edges.sort(axis=1)
    _unique_edges, first_indices = np.unique(edges, axis=0, return_index=True)
    unique_edges = edges[np.sort(first_indices)]
    return np.linalg.norm(
        points[unique_edges[:, 0]] - points[unique_edges[:, 1]],
        axis=1,
    )


def _edge_frequency_stats(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    unit_scale_to_m: float,
    elements_per_wavelength: float,
    speed_of_sound_m_s: float,
) -> dict[str, object]:
    lengths = _triangle_edge_lengths(points, triangles)
    max_edge_step_units = float(np.max(lengths)) if len(lengths) else 0.0
    p95_edge_step_units = float(np.percentile(lengths, 95.0)) if len(lengths) else 0.0
    max_edge_m = max_edge_step_units * unit_scale_to_m
    max_valid_frequency_hz = (
        speed_of_sound_m_s / (elements_per_wavelength * max_edge_m)
        if max_edge_m > 0.0
        else 0.0
    )
    return {
        "max_edge_step_units": max_edge_step_units,
        "max_edge_m": float(max_edge_m),
        "p95_edge_step_units": p95_edge_step_units,
        "p95_edge_m": float(p95_edge_step_units * unit_scale_to_m),
        "max_valid_frequency_hz": float(max_valid_frequency_hz),
    }


def _source_wall_stats(
    points: np.ndarray,
    triangles: np.ndarray,
    tags: np.ndarray,
    spec: SourceSpec,
    *,
    transition_mm: float,
    unit_scale_to_m: float,
    elements_per_wavelength: float,
    speed_of_sound_m_s: float,
) -> dict[str, object] | None:
    """Edge statistics for rigid triangles near a source patch.

    The wave launched by a source travels along the surrounding rigid
    surfaces, so the usable solve band of that source is limited by the
    rigid mesh it traverses, not only by the source patch itself. Rigid
    triangles whose centroid lies within the source refinement transition
    distance are taken as the local wall region.
    """
    source_mask = tags == spec.tag
    rigid_mask = tags == RIGID_TAG
    if not np.any(source_mask) or not np.any(rigid_mask):
        return None
    patch_vertices = points[np.unique(triangles[source_mask])]
    rigid_triangles = triangles[rigid_mask]
    centroids = points[rigid_triangles].mean(axis=1)
    min_distance = np.full(len(centroids), np.inf)
    for start in range(0, len(patch_vertices), 512):
        chunk = patch_vertices[start:start + 512]
        distances = np.linalg.norm(
            centroids[:, None, :] - chunk[None, :, :],
            axis=2,
        ).min(axis=1)
        min_distance = np.minimum(min_distance, distances)
    near_triangles = rigid_triangles[min_distance <= transition_mm]
    if len(near_triangles) == 0:
        return None
    stats = _edge_frequency_stats(
        points,
        near_triangles,
        unit_scale_to_m=unit_scale_to_m,
        elements_per_wavelength=elements_per_wavelength,
        speed_of_sound_m_s=speed_of_sound_m_s,
    )
    return {
        "wall_triangle_count": int(len(near_triangles)),
        "wall_distance_mm": float(transition_mm),
        "wall_max_edge_step_units": float(stats["max_edge_step_units"]),
        "wall_max_edge_m": float(stats["max_edge_m"]),
        "wall_p95_edge_step_units": float(stats["p95_edge_step_units"]),
        "wall_p95_edge_m": float(stats["p95_edge_m"]),
        "wall_max_valid_frequency_hz": float(stats["max_valid_frequency_hz"]),
    }


def mesh_frequency_validation(
    points: np.ndarray,
    triangles: np.ndarray,
    tags: np.ndarray,
    source_specs: list[StepFaceGroup],
    *,
    unit_scale_to_m: float,
    requested_max_frequency_hz: float | None,
    transition_mm: float = 200.0,
    elements_per_wavelength: float = FREQUENCY_ELEMENTS_PER_WAVELENGTH,
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
) -> dict[str, object]:
    """Report conservative global and explicit-group frequency limits."""
    global_stats = _edge_frequency_stats(
        points,
        triangles,
        unit_scale_to_m=unit_scale_to_m,
        elements_per_wavelength=elements_per_wavelength,
        speed_of_sound_m_s=speed_of_sound_m_s,
    )
    edge_limit_m = (
        speed_of_sound_m_s / (elements_per_wavelength * requested_max_frequency_hz)
        if requested_max_frequency_hz is not None
        else None
    )
    warnings: list[str] = []
    global_status = "unknown"
    if requested_max_frequency_hz is not None:
        global_status = "valid"
        if requested_max_frequency_hz > float(global_stats["max_valid_frequency_hz"]):
            global_status = "invalid"
            warnings.append(
                "requested max frequency exceeds conservative global mesh limit "
                f"({requested_max_frequency_hz:.6g} Hz > "
                f"{float(global_stats['max_valid_frequency_hz']):.6g} Hz); "
                "global coarse regions are reported but only active source patches hard-fail"
            )

    per_source: dict[str, dict[str, object]] = {}
    invalid_sources: list[str] = []
    for spec in source_specs:
        mask = tags == spec.tag
        source_triangles = triangles[mask]
        stats = _edge_frequency_stats(
            points,
            source_triangles,
            unit_scale_to_m=unit_scale_to_m,
            elements_per_wavelength=elements_per_wavelength,
            speed_of_sound_m_s=speed_of_sound_m_s,
        )
        wall_stats = _source_wall_stats(
            points,
            triangles,
            tags,
            spec,
            transition_mm=transition_mm,
            unit_scale_to_m=unit_scale_to_m,
            elements_per_wavelength=elements_per_wavelength,
            speed_of_sound_m_s=speed_of_sound_m_s,
        )
        patch_limit = float(stats["max_valid_frequency_hz"])
        effective_limit = patch_limit
        if wall_stats is not None:
            wall_limit = float(wall_stats["wall_max_valid_frequency_hz"])
            if wall_limit > 0.0:
                effective_limit = (
                    min(patch_limit, wall_limit) if patch_limit > 0.0 else wall_limit
                )
        source_status = "unknown"
        if requested_max_frequency_hz is not None:
            source_status = "valid"
            if requested_max_frequency_hz > effective_limit:
                source_status = "invalid"
                invalid_sources.append(spec.name)
                if effective_limit < patch_limit:
                    warnings.append(
                        f"{spec.name} rigid walls within the transition distance are "
                        f"underresolved for {requested_max_frequency_hz:.6g} Hz "
                        f"(wall valid {effective_limit:.6g} Hz, patch valid "
                        f"{patch_limit:.6g} Hz)"
                    )
                else:
                    warnings.append(
                        f"{spec.name} source patch is underresolved for "
                        f"{requested_max_frequency_hz:.6g} Hz "
                        f"(max valid {patch_limit:.6g} Hz)"
                    )
        per_source[spec.name] = {
            "name": spec.name,
            "tag": int(spec.tag),
            "requested_resolution_mm": float(spec.resolution_mm),
            "triangle_count": int(len(source_triangles)),
            "status": source_status,
            "effective_max_valid_frequency_hz": float(effective_limit),
            **stats,
            **(wall_stats or {}),
        }

    status = "unknown"
    if requested_max_frequency_hz is not None:
        status = "invalid" if invalid_sources else "valid"

    return {
        "status": status,
        "scope": "global_warn_source_hard",
        "frequency_policy": "global_warn_source_hard",
        "global_status": global_status,
        "global_max_edge_step_units": float(global_stats["max_edge_step_units"]),
        "global_max_edge_m": float(global_stats["max_edge_m"]),
        "global_p95_edge_step_units": float(global_stats["p95_edge_step_units"]),
        "global_p95_edge_m": float(global_stats["p95_edge_m"]),
        "elements_per_wavelength": float(elements_per_wavelength),
        "speed_of_sound_m_s": float(speed_of_sound_m_s),
        "edge_limit_step_units": (
            None if edge_limit_m is None else float(edge_limit_m / unit_scale_to_m)
        ),
        "edge_limit_m": None if edge_limit_m is None else float(edge_limit_m),
        "max_valid_frequency_hz": float(global_stats["max_valid_frequency_hz"]),
        "global_max_valid_frequency_hz": float(global_stats["max_valid_frequency_hz"]),
        "requested_max_frequency_hz": (
            None if requested_max_frequency_hz is None else float(requested_max_frequency_hz)
        ),
        "invalid_sources": invalid_sources,
        "per_source": per_source,
        "warnings": warnings,
    }

def _topology_stats(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    symmetry_planes: tuple[str, ...],
    tolerance: float,
) -> dict:
    edge_count: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = tuple(sorted((int(a), int(b))))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    free_edges = [edge for edge, count in edge_count.items() if count == 1]
    nonmanifold_edges = [edge for edge, count in edge_count.items() if count > 2]
    edge_direction_stats = _edge_direction_stats(triangles)
    unexpected = []
    samples = []
    for edge in free_edges:
        midpoint = 0.5 * (points[edge[0]] + points[edge[1]])
        if not _edge_on_expected_plane(
            points,
            edge,
            symmetry_planes,
            tolerance,
        ):
            unexpected.append(edge)
            if len(samples) < 20:
                samples.append([float(v) for v in midpoint])

    return {
        "triangles": int(len(triangles)),
        "vertices": int(len(points)),
        "free_edges": int(len(free_edges)),
        "boundary_edges": int(len(free_edges)),
        "nonmanifold_edges": int(len(nonmanifold_edges)),
        "inconsistent_edges": int(edge_direction_stats["inconsistent_edges"]),
        "expected_symmetry_planes": list(symmetry_planes),
        "unexpected_free_edges": int(len(unexpected)),
        "unexpected_free_edge_midpoint_samples": samples,
    }

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


def gmsh_surface_tags() -> list[int]:
    """Return current Gmsh surface tags in deterministic entity order."""
    return _gmsh_surface_tags()


def gmsh_surface_geometries(surface_tags: list[int]) -> list[SurfaceGeometry]:
    """Return center-of-mass and area anchors for Gmsh surfaces."""
    return _gmsh_surface_geometries(surface_tags)


def anchor_surface_order(
    healed_tags: list[int],
    healed_geometries: list[SurfaceGeometry],
    reference_geometries: list[SurfaceGeometry],
) -> list[int]:
    """Recover reference face order after explicit OCC healing."""
    return _anchor_surface_order(
        healed_tags,
        healed_geometries,
        reference_geometries,
    )


def named_shell_gmsh_surfaces(
    step_path: Path,
    gmsh_surfaces: list[int],
    face_order: list[int] | None = None,
) -> dict[str, list[int]]:
    """Map each named STEP shell or body to imported Gmsh surfaces."""
    return _named_shell_gmsh_surfaces(step_path, gmsh_surfaces, face_order)


def map_optional_step_face_groups(
    step_path: Path,
    groups: list[StepFaceGroup],
    gmsh_surfaces: list[int],
    face_order: list[int] | None = None,
) -> tuple[dict[str, list[int]], dict[str, str]]:
    """Map optional caller groups, omitting selectors with no matching label."""
    return _map_refine_groups_to_gmsh_surfaces(
        step_path,
        groups,
        gmsh_surfaces,
        face_order,
    )
