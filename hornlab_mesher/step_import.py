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

from .step_prepare import OccSurfaceRole, snap_symmetry_plane_vertices


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

SurfaceGeometry = tuple[tuple[float, float, float], float]
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


# A record body is any run of characters that are neither a terminator nor a
# quote, interleaved with complete single-quoted STEP strings (in which '' is a
# literal quote). Consuming whole strings is what keeps a ';' *inside* a label
# -- ``STYLED_ITEM('woofer; left', ...)`` -- from truncating the record.
_STEP_RECORD_RE = re.compile(r"#(\d+)\s*=\s*((?:[^;']|'(?:[^']|'')*')*);", flags=re.S)


def _step_records(step_text: str) -> dict[int, str]:
    records: dict[int, str] = {}
    for match in _STEP_RECORD_RE.finditer(step_text):
        records[int(match.group(1))] = " ".join(match.group(2).split())
    return records


def _step_refs(record: str) -> list[int]:
    return [int(value) for value in re.findall(r"#(\d+)", record)]


_STEP_CONTROL_RE = re.compile(r"\\X2\\([0-9A-Fa-f]+)\\X0\\|\\X4\\([0-9A-Fa-f]+)\\X0\\|\\X\\([0-9A-Fa-f]{2})|\\S\\(.)")


def _decode_step_string(value: str) -> str:
    """Decode ISO 10303-21 control directives in a STEP string literal.

    Fusion writes any non-ASCII character in a body or appearance name as an
    escape (``\\X2\\00E5\\X0\\`` for 'a-ring'), so a manifest label carrying one
    could never match the raw literal.
    """

    def replace(match: re.Match[str]) -> str:
        utf16, utf32, byte, shifted = match.groups()
        try:
            if utf16 is not None:
                return bytes.fromhex(utf16).decode("utf-16-be")
            if utf32 is not None:
                return bytes.fromhex(utf32).decode("utf-32-be")
            if byte is not None:
                return bytes([int(byte, 16)]).decode("latin-1")
            if shifted is not None:
                return chr((ord(shifted) + 128) % 0x110000)
        except (ValueError, UnicodeDecodeError):
            return match.group(0)
        return match.group(0)

    return _STEP_CONTROL_RE.sub(replace, value)


def _first_step_string(record: str) -> str | None:
    match = re.search(r"'((?:[^']|'')*)'", record)
    if match is None:
        return None
    return _decode_step_string(match.group(1).replace("''", "'"))


def _parse_named_shell_faces(step_path: Path) -> dict[str, list[int]]:
    """Return STEP shell/surface model name -> ADVANCED_FACE ids.

    Fusion STEP exports commonly encode named surface bodies as
    ``SHELL_BASED_SURFACE_MODEL('name', (#open_shell))``. Gmsh often drops
    those names on import, so we recover them from STEP text and map the face
    order onto imported OCC surface tags.
    """
    records = _step_records(step_path.read_text(encoding="ascii", errors="replace"))
    shell_to_faces: dict[int, list[int]] = {}
    for rec_id, record in records.items():
        if record.startswith(("OPEN_SHELL", "CLOSED_SHELL")):
            shell_to_faces[rec_id] = [
                ref for ref in _step_refs(record)
                if records.get(ref, "").startswith("ADVANCED_FACE")
            ]

    out: dict[str, list[int]] = {}
    for record in records.values():
        if not record.startswith("SHELL_BASED_SURFACE_MODEL"):
            continue
        name = _first_step_string(record)
        if not name:
            continue
        faces: list[int] = []
        for ref in _step_refs(record):
            faces.extend(shell_to_faces.get(ref, []))
        if faces:
            out[name] = faces
    return out


def _parse_solid_brep_faces(step_path: Path) -> set[int]:
    """Return ADVANCED_FACE ids owned by STEP solid BReps.

    Fusion FEM air volumes are exported as ``MANIFOLD_SOLID_BREP`` while the
    exterior BEM acoustic model is normally an open shell.  The main mesher can
    therefore exclude the solid volume from the exterior surface mesh without
    relying on Fusion visibility state or body-name preservation.
    """
    records = _step_records(step_path.read_text(encoding="ascii", errors="replace"))
    shell_faces: dict[int, set[int]] = {}
    for rec_id, record in records.items():
        if record.startswith("CLOSED_SHELL"):
            shell_faces[rec_id] = {
                ref
                for ref in _step_refs(record)
                if records.get(ref, "").startswith("ADVANCED_FACE")
            }
    faces: set[int] = set()
    for record in records.values():
        if not record.startswith(("MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS")):
            continue
        for ref in _step_refs(record):
            faces.update(shell_faces.get(ref, set()))
    return faces


def _parse_styled_face_groups(step_path: Path) -> dict[str, list[int]]:
    """Return STEP presentation/appearance label -> ADVANCED_FACE ids.

    Fusion split faces cannot be named directly in the Browser, but they can
    carry per-face appearance overrides. STEP exports those overrides through
    presentation styles. This parser follows ``STYLED_ITEM`` records to either
    direct ``ADVANCED_FACE`` targets or named shell/surface targets.
    """
    records = _step_records(step_path.read_text(encoding="ascii", errors="replace"))
    shell_faces: dict[int, list[int]] = {}
    model_faces: dict[int, list[int]] = {}
    for rec_id, record in records.items():
        if record.startswith(("OPEN_SHELL", "CLOSED_SHELL")):
            shell_faces[rec_id] = [
                ref for ref in _step_refs(record)
                if records.get(ref, "").startswith("ADVANCED_FACE")
            ]
    for rec_id, record in records.items():
        if record.startswith("SHELL_BASED_SURFACE_MODEL"):
            faces: list[int] = []
            for ref in _step_refs(record):
                faces.extend(shell_faces.get(ref, []))
            if faces:
                model_faces[rec_id] = faces

    def _collect_labels(ref: int, seen: set[int] | None = None) -> set[str]:
        if seen is None:
            seen = set()
        if ref in seen:
            return set()
        seen.add(ref)
        record = records.get(ref, "")
        labels = set()
        label = _first_step_string(record)
        if label:
            labels.add(label)
        for child in _step_refs(record):
            labels.update(_collect_labels(child, seen))
        return labels

    out: dict[str, list[int]] = {}
    for record in records.values():
        if not record.startswith("STYLED_ITEM"):
            continue
        refs = _step_refs(record)
        if len(refs) < 2:
            continue
        target = refs[-1]
        target_record = records.get(target, "")
        if target_record.startswith("ADVANCED_FACE"):
            faces = [target]
        elif target in model_faces:
            faces = model_faces[target]
        elif target in shell_faces:
            faces = shell_faces[target]
        else:
            continue

        labels: set[str] = set()
        styled_name = _first_step_string(record)
        if styled_name:
            labels.add(styled_name)
        for style_ref in refs[:-1]:
            labels.update(_collect_labels(style_ref))
        for label in labels:
            if not label:
                continue
            out.setdefault(label, [])
            out[label].extend(faces)

    return {label: sorted(set(faces)) for label, faces in out.items()}


def _advanced_face_order(step_path: Path) -> list[int]:
    records = _step_records(step_path.read_text(encoding="ascii", errors="replace"))
    return [
        rec_id for rec_id, record in records.items()
        if record.startswith("ADVANCED_FACE")
    ]


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
    """
    if named_faces is None:
        named_faces = _parse_named_shell_faces(step_path)
    if styled_faces is None:
        styled_faces = _parse_styled_face_groups(step_path)
    if face_order is None:
        face_order = _advanced_face_order(step_path)
    face_to_index = {face_id: index for index, face_id in enumerate(face_order)}

    if gmsh_surfaces is None:
        gmsh_surfaces = _gmsh_surface_tags()
    if len(gmsh_surfaces) < len(face_order):
        raise RuntimeError(
            f"STEP has {len(face_order)} ADVANCED_FACE records but gmsh imported "
            f"only {len(gmsh_surfaces)} surfaces"
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


def _named_shell_gmsh_surfaces(step_path: Path, gmsh_surfaces: list[int]) -> dict[str, list[int]]:
    """Map each STEP named shell/body to its imported gmsh surface tags."""
    named_faces = _parse_named_shell_faces(step_path)
    face_order = _advanced_face_order(step_path)
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
) -> tuple[dict[str, list[int]], dict[str, str]]:
    """Resolve refine group names to gmsh surfaces (case-insensitive lookup).

    Missing refine names are skipped (they are optional overrides, unlike
    sources). Returns ``(name -> surfaces, name -> origin)``.
    """
    named_faces = _parse_named_shell_faces(step_path)
    styled_faces = _parse_styled_face_groups(step_path)
    face_order = _advanced_face_order(step_path)
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
    """
    if len(triangles) == 0 or len(tags) != len(triangles):
        return None

    cut_axes = {
        {"x0": 0, "y0": 1, "z0": 2}[plane]
        for plane in symmetry_planes
    }
    open_axes = sorted({0, 1, 2} - cut_axes)
    if len(cut_axes) != 2 or len(open_axes) != 1:
        return None

    source_mask = np.isin(tags, tuple(source_tags))
    if not np.any(source_mask):
        return None

    source_triangles = triangles[source_mask]
    p0 = points[source_triangles[:, 0]]
    p1 = points[source_triangles[:, 1]]
    p2 = points[source_triangles[:, 2]]
    area_vectors = np.cross(p1 - p0, p2 - p0)
    total_area_vector = np.sum(area_vectors, axis=0)
    total_area = float(np.sum(np.linalg.norm(area_vectors, axis=1)))
    projection = float(total_area_vector[open_axes[0]])
    if total_area <= 0.0 or abs(projection) <= 1.0e-12 * total_area:
        return None
    return projection


def _repair_triangle_winding(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    tags: np.ndarray | None = None,
    source_tags: set[int] | None = None,
    symmetry_planes: tuple[str, ...] = (),
    tolerance: float = 0.0,
) -> tuple[np.ndarray, dict[str, int]]:
    """Repair manifold winding and orient components with a valid anchor.

    Watertight components retain the signed-volume outwardness contract.
    Symmetry-reduced open components instead use their tagged source cap:
    positive source-normal projection on the one non-cut principal axis is the
    Metal aperture contract. Reduced components without a usable source anchor
    are deliberately left unjudged and counted rather than guessed.
    """
    repaired = triangles.copy()
    stats = {
        "flipped_consistency": 0,
        "flipped_global": 0,
        "unjudged_symmetry_components": 0,
        "unjudged_symmetry_no_source": 0,
        "symmetry_volume_fallback_flipped": 0,
        "symmetry_volume_fallback_kept": 0,
        "unresolved_symmetry_components": 0,
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

        projection = _symmetry_source_normal_projection(
            points,
            component_triangles,
            component_tags[component],
            source_tags=declared_source_tags,
            symmetry_planes=symmetry_planes,
        )
        if projection is None:
            stats["unjudged_symmetry_components"] += 1
            if not np.any(np.isin(component_tags[component], tuple(declared_source_tags))):
                stats["unjudged_symmetry_no_source"] += 1
            # The source-cap projection abstains whenever a component was not
            # cut on exactly two principal planes -- a single-plane cut, the
            # common Fusion case, is left unjudged by it. Fall back to the
            # signed volume about the origin, which IS a valid oracle here:
            # ``symmetry_reduced`` already established that every free edge
            # lies on a coordinate plane through the origin, so the
            # divergence-theorem cone terms over those rims vanish. (The rule
            # that signed volume cannot orient an open shell applies to an
            # arbitrary rim -- a bare mouth rim off the origin -- not to one
            # pinned to the cut planes, so the general open-shell path below
            # still refuses to guess.)
            volume = _signed_volume(points, component_triangles)
            if volume < 0.0:
                repaired[component] = component_triangles[:, [0, 2, 1]]
                stats["flipped_global"] += int(len(component))
                stats["symmetry_volume_fallback_flipped"] += 1
            elif volume > 0.0:
                stats["symmetry_volume_fallback_kept"] += 1
            else:
                stats["unresolved_symmetry_components"] += 1
            continue
        if projection < 0.0:
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
    edge_count: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = tuple(sorted((int(a), int(b))))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    free_edges = [edge for edge, count in edge_count.items() if count == 1]

    plane_counts = {"x0": 0, "y0": 0, "z0": 0}
    shared_plane_edges = 0
    for edge in free_edges:
        on_planes = [
            plane
            for axis, plane in enumerate(("x0", "y0", "z0"))
            if all(
                abs(float(points[vertex, axis])) <= tolerance
                for vertex in edge
            )
        ]
        if len(on_planes) == 1:
            plane_counts[on_planes[0]] += 1
        elif len(on_planes) > 1:
            shared_plane_edges += 1

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
) -> tuple[meshio.Mesh, dict[str, object], dict[str, object]]:
    """Repair and validate a tagged surface mesh without interpreting roles."""
    points, triangles, tags = _mesh_triangle_data(mesh)
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
        symmetry_planes, symmetry_detection = detect_symmetry_planes(
            points,
            triangles,
            tolerance=symmetry_tolerance,
        )
    repaired_triangles, repair_stats = _repair_triangle_winding(
        points,
        triangles,
        tags=tags,
        source_tags={spec.tag for spec in source_specs},
        symmetry_planes=symmetry_planes,
        tolerance=symmetry_tolerance,
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
        except RuntimeError as exc:
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


def parse_named_shell_faces(step_path: Path) -> dict[str, list[int]]:
    """Return named STEP shell and surface-model face identifiers."""
    return _parse_named_shell_faces(step_path)


def parse_solid_brep_faces(step_path: Path) -> set[int]:
    """Return face identifiers owned by solid STEP B-reps."""
    return _parse_solid_brep_faces(step_path)


def parse_styled_face_groups(step_path: Path) -> dict[str, list[int]]:
    """Return STEP presentation labels and their face identifiers."""
    return _parse_styled_face_groups(step_path)


def advanced_face_order(step_path: Path) -> list[int]:
    """Return STEP ADVANCED_FACE identifiers in file order."""
    return _advanced_face_order(step_path)


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
) -> dict[str, list[int]]:
    """Map each named STEP shell or body to imported Gmsh surfaces."""
    return _named_shell_gmsh_surfaces(step_path, gmsh_surfaces)


def map_optional_step_face_groups(
    step_path: Path,
    groups: list[StepFaceGroup],
    gmsh_surfaces: list[int],
) -> tuple[dict[str, list[int]], dict[str, str]]:
    """Map optional caller groups, omitting selectors with no matching label."""
    return _map_refine_groups_to_gmsh_surfaces(
        step_path,
        groups,
        gmsh_surfaces,
    )
