"""ISO 10303-21 (STEP Part 21) text parsing, standard library only.

Everything here reads STEP *text*. Nothing here imports numpy, meshio or gmsh,
which is the whole point: the same body rule has to be computable inside an
embedded CAD Python that has no third-party packages at all, and
``step_import`` cannot be that home because it imports numpy and meshio at
module scope.

``step_import`` re-exports every public name below, so callers that already
import these parsers from there keep working unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re


# A record body is any run of characters that are neither a terminator nor a
# quote, interleaved with complete single-quoted STEP strings (in which '' is a
# literal quote). Consuming whole strings is what keeps a ';' *inside* a label
# -- ``STYLED_ITEM('woofer; left', ...)`` -- from truncating the record.
_STEP_RECORD_RE = re.compile(r"#(\d+)\s*=\s*((?:[^;']|'(?:[^']|'')*')*);", flags=re.S)

# One alternation, so a '/*' that lives inside a string literal can never open
# a comment and a "'" inside a comment can never open a string. Order matters:
# the string alternative comes first.
_STEP_STRING_OR_COMMENT_RE = re.compile(r"'(?:[^']|'')*'|/\*.*?\*/", flags=re.S)
_STEP_STRING_RE = re.compile(r"'(?:[^']|'')*'", flags=re.S)

_STEP_CONTROL_RE = re.compile(
    r"\\X2\\([0-9A-Fa-f]+)\\X0\\|\\X4\\([0-9A-Fa-f]+)\\X0\\|\\X\\([0-9A-Fa-f]{2})|\\S\\(.)"
)

# A keyword is what immediately precedes '(' in a record whose strings have
# been blanked and whose whitespace has been removed. Removing the whitespace
# first is what tolerates an ISO 10303-21 ed.3 clause 5.6 line break placed
# inside the keyword token itself; blanking the strings first is what stops a
# body *named* "MANIFOLD_SOLID_BREP" from reading as one.
_STEP_KEYWORD_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\(")

# A hollow solid is ONE solid body, so BREP_WITH_VOIDS belongs in
# SOLID_BODY_ENTITIES. It is named separately because "is this a solid body?"
# and "may OCC rebuild this solid?" are different questions with different
# answers for exactly this entity -- see occ_make_solids_is_safe.
VOIDED_SOLID_BODY_ENTITIES: tuple[str, ...] = ("BREP_WITH_VOIDS",)
SOLID_BODY_ENTITIES: tuple[str, ...] = ("MANIFOLD_SOLID_BREP",) + VOIDED_SOLID_BODY_ENTITIES
SURFACE_BODY_ENTITIES: tuple[str, ...] = ("SHELL_BASED_SURFACE_MODEL",)
BODY_ENTITIES: tuple[str, ...] = SOLID_BODY_ENTITIES + SURFACE_BODY_ENTITIES

_SHELL_ENTITIES = ("OPEN_SHELL", "CLOSED_SHELL")


def read_step_text(step_path: Path) -> str:
    """Read a STEP file as text the way every parser here expects it."""
    return step_path.read_text(encoding="ascii", errors="replace")


def strip_step_comments(step_text: str) -> str:
    """Remove ``/* */`` comments, leaving string literals untouched.

    A comment may hold anything, including a complete-looking entity record or
    an unbalanced quote, so structural parsing that does not remove comments
    first is guessing.
    """

    def replace(match: re.Match[str]) -> str:
        text = match.group(0)
        return text if text.startswith("'") else " "

    return _STEP_STRING_OR_COMMENT_RE.sub(replace, step_text)


def blank_step_strings(step_text: str) -> str:
    """Replace every single-quoted literal with an empty one.

    ``''`` inside a literal is an escaped quote and is consumed with it, so a
    label containing quotes cannot leave a dangling delimiter behind.
    """
    return _STEP_STRING_RE.sub("''", step_text)


def step_records(step_text: str) -> dict[int, str]:
    """Return entity id -> record body, whitespace collapsed to single spaces.

    Comments are removed first, so a record spelled out inside ``/* */`` is
    never read as an entity. Every parser below builds on this, which is what
    keeps them agreeing with :func:`step_body_inventory` about what a file
    contains.
    """
    records: dict[int, str] = {}
    for match in _STEP_RECORD_RE.finditer(strip_step_comments(step_text)):
        records[int(match.group(1))] = " ".join(match.group(2).split())
    return records


def step_refs(record: str) -> list[int]:
    """Return every ``#id`` referenced by a record body, in order."""
    return [int(value) for value in re.findall(r"#(\d+)", record)]


def record_entity_types(record: str) -> tuple[str, ...]:
    """Return the entity keywords a record body declares.

    A simple record declares one. A complex (multi-instance) record
    ``(A(...)B(...))`` declares several, and both shapes fall out of the same
    "keyword immediately before an open parenthesis" rule.
    """
    compact = re.sub(r"\s+", "", blank_step_strings(record))
    return tuple(_STEP_KEYWORD_RE.findall(compact))


def decode_step_string(value: str) -> str:
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


def first_step_string(record: str) -> str | None:
    """Return the first string literal of a record body, decoded."""
    match = re.search(r"'((?:[^']|'')*)'", record)
    if match is None:
        return None
    return decode_step_string(match.group(1).replace("''", "'"))


def _shell_faces(records: dict[int, str], *, entities: tuple[str, ...]) -> dict[int, list[int]]:
    return {
        rec_id: [
            ref
            for ref in step_refs(record)
            if records.get(ref, "").startswith("ADVANCED_FACE")
        ]
        for rec_id, record in records.items()
        if record.startswith(entities)
    }


def parse_named_shell_faces_from_text(step_text: str) -> dict[str, list[int]]:
    """Return STEP shell/surface model name -> ADVANCED_FACE ids."""
    records = step_records(step_text)
    shell_to_faces = _shell_faces(records, entities=_SHELL_ENTITIES)

    out: dict[str, list[int]] = {}
    for record in records.values():
        if not record.startswith("SHELL_BASED_SURFACE_MODEL"):
            continue
        name = first_step_string(record)
        if not name:
            continue
        faces: list[int] = []
        for ref in step_refs(record):
            faces.extend(shell_to_faces.get(ref, []))
        if faces:
            out[name] = faces
    return out


def parse_named_shell_faces(step_path: Path) -> dict[str, list[int]]:
    """Return STEP shell/surface model name -> ADVANCED_FACE ids.

    Fusion STEP exports commonly encode named surface bodies as
    ``SHELL_BASED_SURFACE_MODEL('name', (#open_shell))``. Gmsh often drops
    those names on import, so we recover them from STEP text and map the face
    order onto imported OCC surface tags.
    """
    return parse_named_shell_faces_from_text(read_step_text(step_path))


def parse_solid_brep_faces_from_text(step_text: str) -> set[int]:
    """Return ADVANCED_FACE ids owned by STEP solid BReps."""
    records = step_records(step_text)
    shell_faces = {
        rec_id: set(faces)
        for rec_id, faces in _shell_faces(records, entities=("CLOSED_SHELL",)).items()
    }
    faces: set[int] = set()
    for record in records.values():
        if not record.startswith(SOLID_BODY_ENTITIES):
            continue
        for ref in step_refs(record):
            faces.update(shell_faces.get(ref, set()))
    return faces


def parse_solid_brep_faces(step_path: Path) -> set[int]:
    """Return ADVANCED_FACE ids owned by STEP solid BReps.

    Fusion FEM air volumes are exported as ``MANIFOLD_SOLID_BREP`` while the
    exterior BEM acoustic model is normally an open shell.  The main mesher can
    therefore exclude the solid volume from the exterior surface mesh without
    relying on Fusion visibility state or body-name preservation.
    """
    return parse_solid_brep_faces_from_text(read_step_text(step_path))


def parse_styled_face_groups_from_text(step_text: str) -> dict[str, list[int]]:
    """Return STEP presentation/appearance label -> ADVANCED_FACE ids."""
    records = step_records(step_text)
    shell_faces = _shell_faces(records, entities=_SHELL_ENTITIES)
    model_faces: dict[int, list[int]] = {}
    for rec_id, record in records.items():
        if record.startswith("SHELL_BASED_SURFACE_MODEL"):
            faces: list[int] = []
            for ref in step_refs(record):
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
        label = first_step_string(record)
        if label:
            labels.add(label)
        for child in step_refs(record):
            labels.update(_collect_labels(child, seen))
        return labels

    out: dict[str, list[int]] = {}
    for record in records.values():
        if not record.startswith("STYLED_ITEM"):
            continue
        refs = step_refs(record)
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
        styled_name = first_step_string(record)
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


def parse_styled_face_groups(step_path: Path) -> dict[str, list[int]]:
    """Return STEP presentation/appearance label -> ADVANCED_FACE ids.

    Fusion split faces cannot be named directly in the Browser, but they can
    carry per-face appearance overrides. STEP exports those overrides through
    presentation styles. This parser follows ``STYLED_ITEM`` records to either
    direct ``ADVANCED_FACE`` targets or named shell/surface targets.
    """
    return parse_styled_face_groups_from_text(read_step_text(step_path))


def advanced_face_order_from_text(step_text: str) -> list[int]:
    """Return STEP ADVANCED_FACE ids in file (record) order.

    Record order carries no meaning in Part 21, and it is NOT the order in
    which gmsh numbers the imported surfaces: OCC binds faces by walking its
    own shapes, and the STEP reader's shape fixing may reorder a shell's
    faces. Zipping this list with ``gmsh.model.getEntities(2)`` is therefore
    correct only by accident. To map faces onto imported surfaces use
    ``hornlab_mesher.step_import.advanced_face_order_for_surfaces``, which
    matches them by geometry and refuses when it cannot.
    """
    return [
        rec_id
        for rec_id, record in step_records(step_text).items()
        if record.startswith("ADVANCED_FACE")
    ]


def advanced_face_order(step_path: Path) -> list[int]:
    """Return STEP ADVANCED_FACE ids in file (record) order.

    See :func:`advanced_face_order_from_text` for why this is not a surface
    order.
    """
    return advanced_face_order_from_text(read_step_text(step_path))


# Metres per SI prefix; STEP spells the prefix as an enumeration (.MILLI.).
_SI_PREFIX_METRES = {
    "EXA": 1.0e18, "PETA": 1.0e15, "TERA": 1.0e12, "GIGA": 1.0e9, "MEGA": 1.0e6,
    "KILO": 1.0e3, "HECTO": 1.0e2, "DECA": 1.0e1, "DECI": 1.0e-1, "CENTI": 1.0e-2,
    "MILLI": 1.0e-3, "MICRO": 1.0e-6, "NANO": 1.0e-9, "PICO": 1.0e-12,
}
_SI_LENGTH_RE = re.compile(r"SI_UNIT\(\s*(?:\.([A-Z]+)\.|\$)\s*,\s*\.METRE\.\s*\)")
_NUMBER_RE = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"
_LENGTH_MEASURE_RE = re.compile(r"LENGTH_MEASURE\(\s*(" + _NUMBER_RE + r")\s*\)")
_POINT_COORDS_RE = re.compile(r"CARTESIAN_POINT\(\s*'(?:[^']|'')*'\s*,\s*\(([^()]*)\)\s*\)$")
_FACE_BOUNDS_RE = re.compile(r"ADVANCED_FACE\(\s*'(?:[^']|'')*'\s*,\s*\(([^()]*)\)")


def _step_float(token: str) -> float:
    return float(token.strip().replace("D", "E").replace("d", "e"))


def _length_unit_mm(records: dict[int, str], rec_id: int, depth: int = 0) -> float | None:
    record = records.get(rec_id, "")
    if depth > 8 or "LENGTH_UNIT" not in record:
        return None
    si = _SI_LENGTH_RE.search(record)
    if si is not None:
        prefix = si.group(1)
        if prefix is None:
            return 1000.0
        metres = _SI_PREFIX_METRES.get(prefix)
        return None if metres is None else metres * 1000.0
    if "CONVERSION_BASED_UNIT" in record:
        for ref in step_refs(record):
            measure = records.get(ref, "")
            if not measure.startswith("LENGTH_MEASURE_WITH_UNIT"):
                continue
            value = _LENGTH_MEASURE_RE.search(measure)
            unit_refs = step_refs(measure)
            if value is None or not unit_refs:
                return None
            base = _length_unit_mm(records, unit_refs[-1], depth + 1)
            return None if base is None else _step_float(value.group(1)) * base
    return None


def step_length_unit_mm_from_text(step_text: str) -> float | None:
    """Return the file's length unit in millimetres, or ``None`` if unknown.

    ``None`` also means "more than one length unit": a file whose geometry
    contexts disagree about their unit has no single answer, and a caller
    converting coordinates must not pick one.
    """
    records = step_records(step_text)
    # The unit a geometry context assigns, not every unit the file defines:
    # Fusion also declares an unused metre unit beside its millimetre one.
    assigned: set[int] = set()
    for record in records.values():
        _, found, rest = record.partition("GLOBAL_UNIT_ASSIGNED_CONTEXT")
        if found:
            assigned.update(step_refs(rest.partition(")")[0]))
    candidates = [
        rec_id
        for rec_id, record in records.items()
        if "LENGTH_UNIT" in record and (not assigned or rec_id in assigned)
    ]
    units = {_length_unit_mm(records, rec_id) for rec_id in candidates}
    if len(units) != 1 or None in units:
        return None
    return units.pop()


@dataclass(frozen=True)
class StepFaceTopology:
    """What the Part 21 text says about one ADVANCED_FACE's boundary.

    ``vertices`` are coordinates in the file's length unit, in the frame of
    the representation holding the face (see :func:`advanced_face_placements_from_text`).
    ``edges`` are the EDGE_CURVE ids its loops use; two faces sharing one are
    neighbours in the shell.
    """

    vertices: tuple[tuple[float, float, float], ...]
    edges: frozenset[int]


def _record_keyword(record: str) -> str:
    return record.partition("(")[0].strip()


def advanced_face_topology_from_text(step_text: str) -> dict[int, StepFaceTopology]:
    """Return ADVANCED_FACE id -> its boundary vertices and edges, in record order.

    Only the topology is followed (face bounds, loops, oriented edges, edge
    curves, vertices), never a surface or curve definition, whose control
    points are not on the face. A face with no vertex at all -- a closed
    periodic face bounded only by vertex-free loops -- has empty vertices.
    """
    records = step_records(step_text)
    point_cache: dict[int, tuple[float, float, float] | None] = {}

    def point(vertex_id: int) -> tuple[float, float, float] | None:
        if vertex_id in point_cache:
            return point_cache[vertex_id]
        result = None
        refs = step_refs(records.get(vertex_id, ""))
        match = _POINT_COORDS_RE.match(records.get(refs[-1], "")) if refs else None
        if match is not None:
            coords = [part for part in match.group(1).split(",") if part.strip()]
            if len(coords) == 3:
                result = tuple(_step_float(value) for value in coords)
        point_cache[vertex_id] = result
        return result

    out: dict[int, StepFaceTopology] = {}
    for face_id, record in records.items():
        if not record.startswith("ADVANCED_FACE"):
            continue
        bounds = _FACE_BOUNDS_RE.match(record)
        stack = step_refs(bounds.group(1)) if bounds is not None else []
        seen: set[int] = set()
        vertices: dict[int, tuple[float, float, float]] = {}
        edges: set[int] = set()
        while stack:
            rec_id = stack.pop()
            if rec_id in seen:
                continue
            seen.add(rec_id)
            body = records.get(rec_id, "")
            keyword = _record_keyword(body)
            if keyword in ("FACE_BOUND", "FACE_OUTER_BOUND", "EDGE_LOOP", "ORIENTED_EDGE", "VERTEX_LOOP"):
                stack.extend(step_refs(body))
            elif keyword == "EDGE_CURVE":
                edges.add(rec_id)
                # start vertex, end vertex; the third reference is the curve.
                stack.extend(step_refs(body)[:2])
            elif keyword == "VERTEX_POINT":
                coords = point(rec_id)
                if coords is not None:
                    vertices[rec_id] = coords
        out[face_id] = StepFaceTopology(
            vertices=tuple(vertices[key] for key in sorted(vertices)),
            edges=frozenset(edges),
        )
    return out


def advanced_face_vertices_from_text(
    step_text: str,
) -> dict[int, tuple[tuple[float, float, float], ...]]:
    """Return ADVANCED_FACE id -> the coordinates of its boundary vertices.

    Coordinates are in the file's own length unit, in the frame of the
    representation that holds the face. See
    :func:`advanced_face_topology_from_text`.
    """
    return {
        face: topology.vertices
        for face, topology in advanced_face_topology_from_text(step_text).items()
    }


# STEP surface entity -> a kind comparable with gmsh's surface type names.
_STEP_SURFACE_KINDS = {
    "PLANE": "plane",
    "CYLINDRICAL_SURFACE": "cylinder",
    "CONICAL_SURFACE": "cone",
    "SPHERICAL_SURFACE": "sphere",
    "TOROIDAL_SURFACE": "torus",
    "B_SPLINE_SURFACE_WITH_KNOTS": "bspline",
    "B_SPLINE_SURFACE": "bspline",
    "RATIONAL_B_SPLINE_SURFACE": "bspline",
    "BEZIER_SURFACE": "bezier",
    "SURFACE_OF_REVOLUTION": "revolution",
    "SURFACE_OF_LINEAR_EXTRUSION": "extrusion",
}


def advanced_face_surface_kinds_from_text(step_text: str) -> dict[int, str | None]:
    """Return ADVANCED_FACE id -> the kind of its underlying surface, or ``None``.

    Kinds are ``plane``, ``cylinder``, ``cone``, ``sphere``, ``torus``,
    ``bspline``, ``bezier``, ``revolution`` and ``extrusion``; anything else
    (offset, trimmed, degenerate forms) is ``None``, meaning "unknown", never
    "different".
    """
    records = step_records(step_text)
    out: dict[int, str | None] = {}
    for face_id, record in records.items():
        if not record.startswith("ADVANCED_FACE"):
            continue
        refs = step_refs(record)
        surface = records.get(refs[-1], "") if refs else ""
        keywords = record_entity_types(surface) if surface.startswith("(") else (_record_keyword(surface),)
        kinds = {_STEP_SURFACE_KINDS[k] for k in keywords if k in _STEP_SURFACE_KINDS}
        out[face_id] = kinds.pop() if len(kinds) == 1 else None
    return out


Matrix4 = tuple[tuple[float, float, float, float], ...]
_IDENTITY4: Matrix4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _matmul4(a: Matrix4, b: Matrix4) -> Matrix4:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)) for i in range(4)
    )


def _rigid_inverse(m: Matrix4) -> Matrix4:
    rot = [[m[j][i] for j in range(3)] for i in range(3)]  # transpose
    t = [-sum(rot[i][k] * m[k][3] for k in range(3)) for i in range(3)]
    return (
        (rot[0][0], rot[0][1], rot[0][2], t[0]),
        (rot[1][0], rot[1][1], rot[1][2], t[1]),
        (rot[2][0], rot[2][1], rot[2][2], t[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def _matrices_close(a: Matrix4, b: Matrix4, tol: float = 1.0e-9) -> bool:
    scale = max(1.0, *(abs(a[i][3]) for i in range(3)), *(abs(b[i][3]) for i in range(3)))
    return all(
        abs(a[i][j] - b[i][j]) <= tol * (scale if j == 3 else 1.0)
        for i in range(4)
        for j in range(4)
    )


def _vector(records: dict[int, str], rec_id: int | None, default: tuple[float, float, float]) -> tuple[float, float, float] | None:
    if rec_id is None:
        return default
    match = re.search(r"\(([^()]*)\)\s*\)$", records.get(rec_id, ""))
    if match is None:
        return None
    values = [part for part in match.group(1).split(",") if part.strip()]
    if len(values) != 3:
        return None
    return tuple(_step_float(value) for value in values)


def _placement_matrix(records: dict[int, str], rec_id: int) -> Matrix4 | None:
    """AXIS2_PLACEMENT_3D -> the matrix taking local coordinates to its parent's."""
    record = records.get(rec_id, "")
    if _record_keyword(record) != "AXIS2_PLACEMENT_3D":
        return None
    args = blank_step_strings(record).partition("(")[2].rpartition(")")[0].split(",")
    if len(args) != 4:
        return None

    def ref(token: str) -> int | None:
        token = token.strip()
        return int(token[1:]) if token.startswith("#") else None

    origin = _vector(records, ref(args[1]), (0.0, 0.0, 0.0))
    axis = _vector(records, ref(args[2]), (0.0, 0.0, 1.0))
    ref_dir = _vector(records, ref(args[3]), (1.0, 0.0, 0.0))
    if origin is None or axis is None or ref_dir is None:
        return None
    norm = sum(v * v for v in axis) ** 0.5
    if norm == 0.0:
        return None
    z = [v / norm for v in axis]
    dot = sum(a * b for a, b in zip(ref_dir, z))
    x = [a - dot * b for a, b in zip(ref_dir, z)]
    norm = sum(v * v for v in x) ** 0.5
    if norm == 0.0:
        return None
    x = [v / norm for v in x]
    y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]
    return (
        (x[0], y[0], z[0], origin[0]),
        (x[1], y[1], z[1], origin[1]),
        (x[2], y[2], z[2], origin[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def _body_representations(
    records: dict[int, str], step_text: str
) -> dict[int, list[tuple[int, int]]]:
    """Body record id -> every (representation id, item position) listing it."""
    body_ids = {body.record_id for body in step_body_inventory(step_text)}
    out: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for rec_id, record in records.items():
        keyword = _record_keyword(record)
        if not keyword.endswith("REPRESENTATION") or "RELATIONSHIP" in keyword:
            continue
        items = re.match(r"[A-Z_]+\(\s*'(?:[^']|'')*'\s*,\s*\(([^()]*)\)", record)
        if items is None:
            continue
        for position, ref in enumerate(step_refs(items.group(1))):
            if ref in body_ids:
                out[ref].append((rec_id, position))
    return out


def advanced_face_body_positions_from_text(
    step_text: str,
) -> dict[int, tuple[int, int, int, int] | None]:
    """Return ADVANCED_FACE id -> where OCC meets its body, or ``None``.

    The tuple is ``(representation id, kind rank, item position, body id)``:
    gmsh binds every solid's faces before any free face (kind rank 0 for a
    solid body, 1 for a surface body), and within one representation it
    walks bodies in the order the representation lists them. Only faces of
    one body listed by exactly one representation get a position; the key
    orders faces of *different* bodies within one representation, and says
    nothing about faces within one body (the STEP reader may reorder those).
    """
    records = step_records(step_text)
    places = _body_representations(records, step_text)
    out: dict[int, tuple[int, int, int, int] | None] = {}
    for body in step_body_inventory(step_text):
        listed = places.get(body.record_id, [])
        key = None
        if len(listed) == 1:
            rep, position = listed[0]
            key = (rep, 0 if body.kind == "solid" else 1, position, body.record_id)
        for face in body.face_ids:
            out[face] = None if face in out else key
    return out


def advanced_face_placements_from_text(step_text: str) -> dict[int, Matrix4 | None]:
    """Return ADVANCED_FACE id -> the matrix placing its representation in the model.

    An assembly places each part's representation into its parent with a
    ``REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION`` whose
    ``ITEM_DEFINED_TRANSFORMATION`` carries two axis placements; OCC applies
    exactly that chain when it builds the imported compound. The matrix
    takes coordinates in the face's own representation (file length units)
    to the root frame. ``None`` means the text does not determine one
    placement: the representation is instanced more than once, or a
    placement could not be read.
    """
    records = step_records(step_text)
    upward: dict[int, list[tuple[int, Matrix4 | None]]] = defaultdict(list)
    identity_links: dict[int, set[int]] = defaultdict(set)
    for record in records.values():
        if "REPRESENTATION_RELATIONSHIP" not in record:
            continue
        keywords = record_entity_types(record) if record.startswith("(") else (_record_keyword(record),)
        if not any(k.endswith("REPRESENTATION_RELATIONSHIP") for k in keywords):
            continue
        relation = re.search(r"(?<![A-Z_])REPRESENTATION_RELATIONSHIP\(([^()]*)\)", blank_step_strings(record))
        if relation is None:
            relation = re.search(r"SHAPE_REPRESENTATION_RELATIONSHIP\(([^()]*)\)", blank_step_strings(record))
        if relation is None:
            continue
        reps = step_refs(relation.group(1))
        if len(reps) < 2:
            continue
        child, parent = reps[-2], reps[-1]
        transform = re.search(r"REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION\(\s*#(\d+)\s*\)", record)
        if transform is None:
            identity_links[child].add(parent)
            identity_links[parent].add(child)
            continue
        item = records.get(int(transform.group(1)), "")
        matrix: Matrix4 | None = None
        if _record_keyword(item) == "ITEM_DEFINED_TRANSFORMATION":
            placements = step_refs(item)
            if len(placements) >= 2:
                source = _placement_matrix(records, placements[-2])
                target = _placement_matrix(records, placements[-1])
                if source is not None and target is not None:
                    matrix = _matmul4(target, _rigid_inverse(source))
        upward[child].append((parent, matrix))

    def to_root(rep: int, depth: int = 0) -> Matrix4 | None:
        # Every representation identity-linked to ``rep`` shares its frame.
        component = {rep}
        frontier = [rep]
        while frontier:
            current = frontier.pop()
            for other in identity_links.get(current, ()):
                if other not in component:
                    component.add(other)
                    frontier.append(other)
        parents = [link for member in sorted(component) for link in upward.get(member, ())]
        if not parents:
            return _IDENTITY4
        if depth > 32:
            return None
        results: list[Matrix4] = []
        for parent, matrix in parents:
            if matrix is None:
                return None
            above = to_root(parent, depth + 1)
            if above is None:
                return None
            results.append(_matmul4(above, matrix))
        first = results[0]
        return first if all(_matrices_close(first, other) for other in results[1:]) else None

    body_rep = {
        body: {rep for rep, _index in places}
        for body, places in _body_representations(records, step_text).items()
    }

    cache: dict[int, Matrix4 | None] = {}
    out: dict[int, Matrix4 | None] = {}
    for body in step_body_inventory(step_text):
        reps = body_rep.get(body.record_id, set())
        matrix: Matrix4 | None
        if len(reps) != 1:
            matrix = _IDENTITY4 if not reps else None
        else:
            rep = next(iter(reps))
            if rep not in cache:
                cache[rep] = to_root(rep)
            matrix = cache[rep]
        for face in body.face_ids:
            if face in out and out[face] != matrix:
                out[face] = None
            else:
                out[face] = matrix
    for face_id, record in records.items():
        if record.startswith("ADVANCED_FACE") and face_id not in out:
            out[face_id] = _IDENTITY4
    return out


def advanced_face_vertices(step_path: Path) -> dict[int, tuple[tuple[float, float, float], ...]]:
    """Return ADVANCED_FACE id -> boundary vertex coordinates (file units)."""
    return advanced_face_vertices_from_text(read_step_text(step_path))


@dataclass(frozen=True)
class StepBody:
    """One top-level Part 21 body entity.

    ``kind`` is ``"solid"`` or ``"surface"``; it is derived from ``entity`` and
    is what a caller comparing against an OCC volume count actually wants.
    """

    entity: str
    record_id: int
    name: str | None
    kind: str
    face_ids: tuple[int, ...]


def step_body_inventory(step_text: str) -> tuple[StepBody, ...]:
    """Return every top-level body entity in a STEP file, in record order.

    A body is one ``MANIFOLD_SOLID_BREP``, ``BREP_WITH_VOIDS`` or
    ``SHELL_BASED_SURFACE_MODEL`` instance. That definition is computable from
    text on both ends of a CAD round trip, which an OCC entity count is not:
    gmsh has no shell entity, so a one-shell surface body of N faces imports as
    N parentless surfaces.

    Comments are removed and string literals are blanked before the entity
    keyword is read, so neither a record spelled out inside a ``/* */`` comment
    nor a body *named* after an entity type can inflate the result.
    """
    clean = strip_step_comments(step_text)
    records = step_records(clean)
    # A second pass over the same text with strings blanked and *all*
    # whitespace removed. Reading the entity keyword off this copy is what
    # tolerates an ed.3 clause 5.6 line break inside the keyword token, and it
    # is two whole-file regex passes rather than two per record.
    compact = step_records(re.sub(r"\s+", "", blank_step_strings(clean)))

    shell_faces = _shell_faces(records, entities=_SHELL_ENTITIES)
    # A BREP_WITH_VOIDS reaches its void shells through ORIENTED_CLOSED_SHELL,
    # so resolving that one hop is what makes a voided solid report its faces.
    oriented_shells: dict[int, list[int]] = {}
    for rec_id, record in compact.items():
        if not record.startswith("ORIENTED_CLOSED_SHELL"):
            continue
        faces: list[int] = []
        for ref in step_refs(record):
            faces.extend(shell_faces.get(ref, []))
        oriented_shells[rec_id] = faces

    def _body_entity(record: str) -> str | None:
        if record.startswith("("):
            # A complex (multi-instance) record declares several keywords.
            return next(
                (name for name in record_entity_types(record) if name in BODY_ENTITIES),
                None,
            )
        keyword, sep, _rest = record.partition("(")
        return keyword if sep and keyword in BODY_ENTITIES else None

    bodies: list[StepBody] = []
    for rec_id in sorted(compact):
        entity = _body_entity(compact[rec_id])
        if entity is None:
            continue
        record = records.get(rec_id, "")
        faces: list[int] = []
        for ref in step_refs(record):
            faces.extend(shell_faces.get(ref, oriented_shells.get(ref, [])))
        bodies.append(
            StepBody(
                entity=entity,
                record_id=rec_id,
                name=first_step_string(record),
                kind="solid" if entity in SOLID_BODY_ENTITIES else "surface",
                face_ids=tuple(faces),
            )
        )
    return tuple(bodies)


def count_step_bodies(step_text: str) -> int:
    """Return the number of top-level Part 21 body entities.

    Defined as the length of :func:`step_body_inventory` rather than as a
    second, cheaper scan: two body rules that can disagree is the bug this
    module exists to remove.
    """
    return len(step_body_inventory(step_text))


def occ_make_solids_is_safe(step_text: str) -> bool:
    """Return whether OCC may safely rebuild solids for this STEP text.

    ``Geometry.OCCSewFaces`` dissolves a solid body -- gmsh imports it as
    orphan faces with no volume at all -- and ``Geometry.OCCMakeSolids`` is
    what puts the volume back. Whether that rebuild is a *restoration* or a
    *corruption* depends on the file, which is why this is a function of the
    text and must not be an option carried inside a healing rung: a rung is a
    repair strategy, and a consumer that reads a rung as "set each of these to
    1" would have the choice made for it by a repository that never saw its
    files.

    Measured on gmsh 4.15.2, a 40 mm box minus a fully enclosed r=10 sphere --
    which the OCC writer emits as one ``BREP_WITH_VOIDS``:

        unhealed          volumes=1  surfaces=7  mass=59811.21
        sew only          volumes=0  surfaces=7  mass=None
        sew + MakeSolids  volumes=1  surfaces=6  mass=64000.00

    64000 is the solid box. The cavity is filled and its inner shell deleted,
    and because both volume counts read 1 no entity-counting gate can see it.

    Hence:

    * ``False`` if ANY body carries interior voids. A file mixing a hollow
      body with a plain one would have the hollow one silently filled, so one
      voided body disqualifies the whole file.
    * ``False`` if there is no solid body at all. There is nothing to restore,
      and two coincident sheets would sew into a phantom zero-volume solid.
    * ``True`` only for at least one plain ``MANIFOLD_SOLID_BREP`` and no
      voided one.

    Deliberately NOT the same question as :attr:`StepBody.kind`: a hollow body
    *is* one solid body, and it is precisely the body OCC must not
    re-solidify. Conflating the two turns a loud failure into silent geometry
    corruption.
    """
    bodies = step_body_inventory(step_text)
    if any(body.entity in VOIDED_SOLID_BODY_ENTITIES for body in bodies):
        return False
    return any(body.kind == "solid" for body in bodies)
