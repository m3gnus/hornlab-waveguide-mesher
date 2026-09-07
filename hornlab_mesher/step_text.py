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
    """Return entity id -> record body, whitespace collapsed to single spaces."""
    records: dict[int, str] = {}
    for match in _STEP_RECORD_RE.finditer(step_text):
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
    """Return STEP ADVANCED_FACE ids in file order."""
    return [
        rec_id
        for rec_id, record in step_records(step_text).items()
        if record.startswith("ADVANCED_FACE")
    ]


def advanced_face_order(step_path: Path) -> list[int]:
    """Return STEP ADVANCED_FACE ids in file order."""
    return advanced_face_order_from_text(read_step_text(step_path))


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
