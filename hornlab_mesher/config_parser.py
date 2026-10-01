from __future__ import annotations

import json
import logging
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Mapping
from .throat_stretch import canonical_stretch_params, validate_stretch_composition, stretch_coefficients

import numpy as np

try:  # pragma: no cover - exercised only on Python 3.10
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    pass

def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".toml", ".tml"}:
        return tomllib.loads(text)
    if suffix in {".cfg", ".txt"}:
        return parse_text_config(text)
    raise ConfigError(f"unsupported config extension {suffix!r}; use .toml, .json, .cfg, or .txt")


def _maybe_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if not text:
        return value
    try:
        out = float(text)
    except ValueError:
        return text
    if not np.isfinite(out):
        return text
    return int(out) if out.is_integer() else out



def _parse_ath_blocks(
    content: str,
) -> tuple[dict[str, dict[str, str]], dict[str, str], list[tuple[str, ...]]]:
    """Split an ATH text config into top-level blocks, top-level keys and nested blocks.

    ``blocks`` maps each top-level block name to its own ``key = value`` lines.
    ``nested`` lists the full ancestry of every block opened inside another
    one; the contents of a nested block are not kept, because
    no supported item has any -- the caller refuses the nested block by name
    unless its outermost block is one it ignores as a whole.
    """
    blocks: dict[str, dict[str, str]] = {}
    flat: dict[str, str] = {}
    nested: list[tuple[str, ...]] = []
    stack: list[str] = []

    for lineno, raw_line in enumerate(content.splitlines(), 1):
        line = raw_line.split(";", 1)[0].strip()
        if not line:
            continue
        start = line.split("=", 1)
        if len(start) == 2 and start[1].strip() == "{":
            name = start[0].strip()
            if stack:
                nested.append((*stack, name))
            else:
                if name in _PROFILE_BLOCKS and name in blocks:
                    raise ConfigError(
                        f"{name} = {{...}} is a second profile beside {name}; only one profile block is supported"
                    )
                blocks.setdefault(name, {})
            stack.append(name)
            continue
        if line == "}":
            if not stack:
                raise ConfigError(f"line {lineno}: '}}' closes no block")
            stack.pop()
            continue
        if "=" not in line:
            # Script blocks (``Source.Contours``, enclosure plans) hold bare
            # command lines. Outside a block a line without ``=`` sets nothing,
            # and skipping it would hide, for instance, a block opened without
            # its ``=`` whose members would then be read as top-level keys.
            if not stack:
                raise ConfigError(
                    f"line {lineno}: cannot read {line!r}; expected 'key = value' or 'name = {{'"
                )
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if not stack:
            flat[key] = value
        elif len(stack) == 1:
            blocks[stack[0]][key] = value
    if stack:
        raise ConfigError(f"block {stack[-1]!r} is never closed")
    return blocks, flat, nested


def _ath_bool(value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return 1
        if lowered in {"0", "false", "no", "off"}:
            return 0
    return value


# ATH items this importer reads past on purpose. They configure ATH's output
# files, its reports and the solver project it writes, not the horn, so a mesh
# built without them is the mesh the config describes. Everything else that is
# not recognised is refused (``_reject_unrecognised_ath_items``): an item the
# importer merely skipped used to build a simpler horn with nothing to show for
# it. Keep this list to items that cannot change geometry or the source; it is
# documented in ``docs/config-schema.md``.
_IGNORED_ATH_PREFIXES: tuple[str, ...] = (
    # Solver project: frequencies, observation maps and fields, driving values.
    # ``ABEC.SimType`` is the one geometry-relevant member and is read explicitly.
    "ABEC.",
    "Simulation.",
    # Output locations and file switches; the CLI owns output paths.
    "Output.",
    # Lumped-element driver model coupled in the solver project.
    "LE.",
    # Settings for ATH's external Gmsh run; this mesher owns its own meshing.
    "Gmsh.",
)
_IGNORED_ATH_KEYS = frozenset(
    {
        "LE",
        # Items of ATH's global ``ath.cfg`` that sometimes travel in a horn file.
        "OutputRootDir",
        "MeshCmd",
        "GnuplotPath",
        # ATH User Guide 4.1.1: only active for Throat.Profile = 3, which
        # this importer refuses in _reject_unsupported_ath_keys.
        "CircArc.TermAngle",
    }
)
# Block names are compared without their ``:<tag>`` suffix (``GridExport:f360``).
_IGNORED_ATH_BLOCKS = frozenset({"Report", "GridExport", "FRDExport", "RespExport"})

# Top-level keys read directly rather than through a mapping table.
_KNOWN_TOP_LEVEL_KEYS = frozenset({"ABEC.SimType", "Scale", "Throat.Profile"})
_KNOWN_BLOCKS = frozenset({"Mesh", "Morph", "MORPH", "GCurve", "GCURVE", "Mesh.Enclosure", "Source"})
_PROFILE_BLOCKS = ("R-OSSE", "ROSSE", "OSSE")
# Namespaces with their own, older handling: unknown ``Mesh.*`` keys are refused
# by ``_reject_unsupported_ath_keys`` and unknown ``Morph.*``/``GCurve.*`` keys
# are warned about by ``_warn_ignored_ath_keys``.
_SELF_CHECKED_PREFIXES = ("Mesh.", "Morph.", "MORPH.", "GCurve.", "GCURVE.")

_ENCLOSURE_KEY_MAP: tuple[tuple[str, str], ...] = (
    ("Depth", "depth_mm"),
    ("EdgeRadius", "edge_mm"),
    ("EdgeType", "edge_type"),
    ("FrontResolution", "enc_front_resolution"),
    ("BackResolution", "enc_back_resolution"),
)
_ENCLOSURE_KEYS = frozenset({src for src, _dst in _ENCLOSURE_KEY_MAP} | {"Spacing"})
_SOURCE_KEY_MAP: tuple[tuple[str, str], ...] = (
    ("Shape", "sourceShape"),
    ("Radius", "sourceRadius"),
    ("Curv", "sourceCurv"),
    ("Velocity", "sourceVelocity"),
    ("VelocityProfile", "sourceVelocityProfile"),
)
_SOURCE_KEYS = frozenset(src for src, _dst in _SOURCE_KEY_MAP)

# Why the better-known unsupported items are refused, keyed by item name without
# its ``:<tag>``. Anything absent from this table is refused with the generic
# reason only.
_UNSUPPORTED_ATH_REASONS: dict[str, str] = {
    "HornGeometry": "the two-profile (Horn.Part) geometry is not implemented",
    "Horn.Adapter": "the two-profile (Horn.Part) geometry is not implemented",
    "Horn.Part": "the two-profile (Horn.Part) geometry is not implemented",
    "Throat.Ext.Ctrl": "the curved throat adapter is not implemented",
    "Scale.Z": "a separate axial scale is not implemented",
    "Rot": "Rot in this position has not been checked against ATH output",
    "Length": "Length in this position has not been checked against ATH output",
    "Profile": "a profile read from a point file is not implemented",
    "Mesh.Roundover": "the mouth roundover is not implemented",
    "Mesh.Enclosure": (
        "enclosure members inside Mesh and scalar enclosure markers are not supported; "
        "use supported top-level Mesh.Enclosure.* keys or a separate Mesh.Enclosure = {...} block"
    ),
    "Plan": "user-defined enclosure plans are not implemented",
    "Dim": "enclosure dimensions by Dim are not implemented; use Spacing",
    "BEE": "the LF enclosure is not implemented",
    "Source.Array": "source arrays are not implemented",
    "FusionRotaryProfile": "the rotary profile with flange and joint is not implemented",
    "arcterm": "the arc termination is not implemented",
    "trunc": "throat truncation is not implemented",
}


def _ath_base_name(name: str) -> str:
    return name.split(":", 1)[0].strip()


def _ath_item_disabled(name: str) -> bool:
    """ATH scripts park an item by prefixing its name with an underscore."""
    return name.startswith("_")


def _ath_ignore_reason(name: str, *, block: bool) -> str | None:
    base = _ath_base_name(name)
    if _ath_item_disabled(base):
        return "underscore-prefixed ATH item is disabled"
    if not (base.startswith(_IGNORED_ATH_PREFIXES) or base in (_IGNORED_ATH_BLOCKS if block else _IGNORED_ATH_KEYS)):
        return None
    if base.startswith(("ABEC.", "Simulation.")):
        return "ATH solver-run/project settings do not change geometry or the source"
    if base.startswith("Output.") or base == "OutputRootDir":
        return "ATH output paths and file switches are owned by this mesher's CLI"
    if base.startswith("LE.") or (base == "LE" and not block):
        return "ATH's lumped-element driver model is a solver-project setting"
    if base.startswith("Gmsh.") or base == "MeshCmd":
        return "this mesher owns its meshing instead of ATH's external mesh command"
    if not block and base == "GnuplotPath":
        return "ATH's plotting executable does not change geometry or the source"
    if not block and base == "CircArc.TermAngle":
        return "ATH applies it only to Throat.Profile = 3; this importer supports only the OS-SE profile (1)"
    if block and base in _IGNORED_ATH_BLOCKS:
        return "ATH report layout" if base == "Report" else "ATH coordinate/response export settings"
    return None


def _ath_item_ignored(name: str, *, block: bool) -> bool:
    return _ath_ignore_reason(name, block=block) is not None


def _report_ignored_ath_item(name: str, reason: str, *, value: str | None = None, block: bool = False) -> None:
    """One diagnostic format for every deliberately unused ATH item."""
    written = name + (" = {...} (whole subtree)" if block else f" = {value}" if value is not None else "")
    logger.warning("[hornlab-mesher] ignoring %s: %s", written, reason)


def _ath_warning_key_map(namespace: str) -> tuple[tuple[str, str], ...] | None:
    if namespace in {"Morph", "MORPH"}:
        return _MORPH_KEY_MAP
    if namespace in {"GCurve", "GCURVE"}:
        return _GCURVE_KEY_MAP
    return None


def _report_ignored_ath_items(
    flat: Mapping[str, str],
    blocks: Mapping[str, Mapping[str, str]],
    nested: list[tuple[str, ...]],
    *,
    profile_block: str | None,
) -> None:
    """Report skipped parents once; their descendants need no separate warning."""
    for key, value in flat.items():
        if key == "ABEC.SimType":  # This scalar selects geometry topology.
            continue
        reason = _ath_ignore_reason(key, block=False)
        for prefix in (*_SELF_CHECKED_PREFIXES, "Mesh.Enclosure.", "Source."):
            if key.startswith(prefix) and _ath_item_disabled(key[len(prefix):]):
                reason = _ath_ignore_reason(key[len(prefix):], block=False)
        if profile_block == "OSSE":
            if key == "Length":
                reason = "ATH uses the OSSE block's L and ignores top-level Length"
            elif key == "Rot" and "Rot" in blocks["OSSE"]:
                reason = "ATH uses the OSSE block's Rot ahead of top-level Rot"
        if reason:
            _report_ignored_ath_item(key, reason, value=value)
        else:
            namespace, _, member = key.partition(".")
            key_map = _ath_warning_key_map(namespace)
            if key_map is not None:
                _warn_ignored_ath_keys(namespace + ".", {member: value}, key_map)

    skipped: list[tuple[str, ...]] = []
    for name, items in blocks.items():
        reason = _ath_ignore_reason(name, block=True)
        if reason:
            _report_ignored_ath_item(name, reason, block=True)
            skipped.append((name,))
            continue
        key_map = _ath_warning_key_map(name)
        if key_map is not None:
            _warn_ignored_ath_keys(name + ".", items, key_map)
        for key, value in items.items():
            if _ath_item_disabled(key):
                _report_ignored_ath_item(f"{name}.{key}", _ath_ignore_reason(key, block=False), value=value)
    for path in nested:
        if any(path[:len(parent)] == parent for parent in skipped):
            continue
        if _ath_item_disabled(path[-1]):
            _report_ignored_ath_item(".".join(path), _ath_ignore_reason(path[-1], block=True), block=True)
            skipped.append(path)


def _ath_nested_ignored(path: tuple[str, ...]) -> bool:
    return _ath_item_ignored(path[0], block=True) or any(_ath_item_disabled(name) for name in path)


def _ath_refusal_reason(lookup: str, candidates: list[str]) -> str | None:
    reason = _UNSUPPORTED_ATH_REASONS.get(lookup)
    if reason is None and lookup and lookup not in candidates:
        close = get_close_matches(lookup, candidates, n=1, cutoff=0.8)
        # ATH names are case-sensitive. A wrong-case name still deserves a remedy.
        if not close:
            close = [name for name in candidates if name.casefold() == lookup.casefold()][:1]
        if close:
            reason = f"did you mean {close[0]}?"
            if close[0].casefold() == lookup.casefold():
                reason += " (ATH names are case-sensitive)"
    return reason


def _missing_ath_profile_details(flat: Mapping[str, str], blocks: Mapping[str, Mapping[str, str]]) -> str:
    """Name supplied candidates before explaining that no profile was selected."""
    candidates = sorted({*_PROFILE_BLOCKS, "Length", "Coverage.Angle", "Throat.Diameter", "Throat.Angle", "Term.n"})
    details = []
    for name in blocks:
        if not _ath_item_ignored(name, block=True):
            reason = _ath_refusal_reason(_ath_base_name(name), candidates)
            details.append(f"{name} = {{...}} ({reason or 'this block does not select a supported OSSE or R-OSSE profile'})")
    for key in flat:
        if not _ath_item_ignored(key, block=False) or key == "ABEC.SimType":
            reason = _ath_refusal_reason(key, candidates)
            details.append(f"{key} ({reason or 'this key alone does not select a profile; set Length or supply an OSSE or R-OSSE block'})")
    return "supplied ATH profile candidate(s): " + "; ".join(details) + ". " if details else ""


def _reject_unrecognised_ath_items(
    flat: Mapping[str, str],
    blocks: Mapping[str, Mapping[str, str]],
    nested: list[tuple[str, ...]],
    *,
    known_flat: set[str],
    profile_block: str | None,
    known_profile: set[str],
) -> None:
    """Refuse every item that would otherwise be read and then dropped.

    The importer looks up the items it supports by name, so an item it does
    not know never reaches the builder: ``Scale.Z``, a ``Horn.Part`` block, an
    ``arcterm`` inside ``R-OSSE`` or a mistyped ``Coverage.Angel`` all used to
    import cleanly as a simpler horn. ``known_flat`` and ``known_profile`` are
    collected from the mapping tables as ``parse_text_config`` applies them, so
    a key added to a table is recognised here without a second list to update.
    """
    # (item as written, name to look up a reason or a near match for)
    unknown: list[tuple[str, str]] = []

    for key in flat:
        if key in known_flat:
            continue
        if key in {"Mesh.Enclosure", "Enclosure"}:
            unknown.append((key, "Mesh.Enclosure"))
            continue
        if key.startswith("Mesh.Enclosure."):
            # Enclosure members share this final check with the block form.
            member = key[len("Mesh.Enclosure.") :]
            if member in _ENCLOSURE_KEYS or _ath_item_disabled(member):
                continue
            unknown.append((key, member))
            continue
        elif key.startswith(_SELF_CHECKED_PREFIXES):
            continue
        if key.startswith("Source."):
            if key[len("Source.") :] in _SOURCE_KEYS:
                continue
        elif _ath_item_ignored(key, block=False):
            continue
        unknown.append((key, key))

    for name, items in blocks.items():
        if name == profile_block:
            members, known = items, known_profile
        elif name == "Mesh.Enclosure":
            members, known = items, _ENCLOSURE_KEYS
        elif name == "Source":
            members, known = items, _SOURCE_KEYS
        elif name == "Mesh":
            unknown.extend(
                (f"Mesh.{key}", "Mesh.Enclosure") for key in items
                if key == "Enclosure" or key.startswith("Enclosure.")
            )
            continue
        elif name in _KNOWN_BLOCKS or _ath_item_ignored(name, block=True):
            continue
        elif name in _PROFILE_BLOCKS:
            beside = profile_block or "the flat profile keys"
            unknown.append((f"{name} = {{...}} (a second profile beside {beside})", ""))
            continue
        else:
            unknown.append((f"{name} = {{...}}", _ath_base_name(name)))
            continue
        unknown.extend(
            (f"{name}.{key}", key) for key in members if key not in known and not _ath_item_disabled(key)
        )

    for path in nested:
        if _ath_nested_ignored(path):
            continue
        unknown.append((f"{'.'.join(path)} = {{...}}", _ath_base_name(path[-1])))

    if not unknown:
        return
    candidates = sorted(known_flat | known_profile | _ENCLOSURE_KEYS)
    details = []
    for item, lookup in unknown:
        reason = _ath_refusal_reason(lookup, candidates)
        details.append(f"{item} ({reason})" if reason else item)
    raise ConfigError(
        "unsupported item(s) in ATH config: "
        + "; ".join(details)
        + ". The importer would drop them and build a horn that does not match the "
        "config. Remove them, or prefix a name with '_' to disable it."
    )


# Recognised ``Mesh.*`` keys. Kept beside the mapping tables in
# ``parse_ath_config`` that consume them -- if you add a name there, add it
# here or the importer will refuse the very key you just added.
_KNOWN_MESH_KEYS = frozenset(
    {
        "AngularSegments",
        "CornerSegments",
        "LengthSegments",
        "WallThickness",
        "VerticalOffset",
        "Quadrants",
        "ThroatResolution",
        "MouthResolution",
        "RearResolution",
        "SubdomainSlices",
        "InterfaceOffset",
        "InterfaceResolution",
        "SamplingMode",
        "SurfaceFit",
        "ZMapPoints",
        "ZMap",
        # Recognised only to be refused above, with a better message.
        "RearShape",
        "ThroatSegments",
        "ThroatExtSegments",
        # Deferred to the final unread-item check, after C4 composition.
        "Enclosure",
    }
)


# ``Morph.*`` and ``GCurve.*`` names, mapped onto the internal parameter names.
# Both tables carry ATH's own vocabulary; the entries marked below are the only
# deliberate additions. Anything else in these two namespaces is ignored -- ATH
# ignores it as well -- and warned about by ``_warn_ignored_ath_keys``, so the
# tables double as the set of recognised keys.
_MORPH_KEY_MAP: tuple[tuple[str, str], ...] = (
    ("TargetShape", "morphTarget"),
    ("TargetWidth", "morphWidth"),
    ("TargetHeight", "morphHeight"),
    ("CornerRadius", "morphCorner"),
    ("Rate", "morphRate"),
    ("FixedPart", "morphFixed"),
    ("AllowShrinkage", "morphAllowShrinkage"),
    # Not an ATH key -- ATH's morph targets take no exponent -- but this parser
    # also reads configs written for this mesher, whose superellipse target
    # (``Morph.TargetShape = 3``) needs one. Same footing as ``Mesh.SurfaceFit``.
    ("Exponent", "morphExponent"),
)
_GCURVE_KEY_MAP: tuple[tuple[str, str], ...] = (
    ("Type", "gcurveType"),
    ("Width", "gcurveWidth"),
    ("AspectRatio", "gcurveAspectRatio"),
    ("Dist", "gcurveDist"),
    ("Rot", "gcurveRot"),
    ("SF", "gcurveSF"),
    ("SF.a", "gcurveSfA"),
    ("SF.b", "gcurveSfB"),
    ("SF.m1", "gcurveSfM1"),
    ("SF.m2", "gcurveSfM2"),
    ("SF.n1", "gcurveSfN1"),
    ("SF.n2", "gcurveSfN2"),
    ("SF.n3", "gcurveSfN3"),
    ("SE.n", "gcurveSeN"),
)


def _reject_unsupported_ath_keys(
    flat: Mapping[str, str],
    profile_items: Mapping[str, str],
    mesh_items: Mapping[str, str],
    blocks: Mapping[str, Mapping[str, str]] = {},
) -> None:
    """Fail loudly on imported keys that change geometry we cannot build."""
    # Multi-source ATH models (contour-drawn domes, secondary LF sources,
    # per-source velocity profiles) define driving surfaces this mesher cannot
    # build; a silent drop would mesh a default cap source instead of the
    # configured crossover model.
    #
    # Only the indexed ``Source.Velocity.<n>`` form is such a marker. Scalar
    # ``Source.Velocity`` is a single-source item -- it picks the velocity
    # DIRECTION of the one driving surface (1 = normal to the element surface,
    # 2 = axial), per the ATH user guide -- and matching it here rejected two
    # ordinary single-source configs in the reference archive that carry the
    # default ``Source.Velocity = 1``. It is validated with the other
    # ``Source.*`` items instead.
    multi_source_keys = sorted(
        key
        for key in {*flat, *profile_items, *blocks}
        if key == "Source.Contours"
        or key.startswith("LFSource")
        or key.startswith("Source.Velocity.")
    )
    if multi_source_keys:
        raise ConfigError(
            "multi-source ATH configs are not supported by this mesher "
            f"(saw {', '.join(multi_source_keys)}); only the single cap/disc throat source is implemented"
        )
    throat_profile = _maybe_number(profile_items.get("Throat.Profile", flat.get("Throat.Profile")))
    if throat_profile is not None and throat_profile != 1:
        raise ConfigError(
            f"Throat.Profile = {throat_profile} is not supported; only the OS-SE profile (1) is implemented"
        )
    rollback_keys = sorted(
        key
        for key in {*flat, *profile_items, *blocks}
        if key == "Rollback" or key.startswith("Rollback.")
    )
    if rollback_keys:
        raise ConfigError(f"Rollback is not supported by this mesher (saw {', '.join(rollback_keys)})")
    rear_shape = _maybe_number(mesh_items.get("RearShape"))
    if rear_shape is not None and rear_shape != 1:
        raise ConfigError(f"Mesh.RearShape = {rear_shape} is not supported; only the full rear (1) is implemented")
    # The user guide spells the throat-extension slice count ``ThroatSegments``;
    # current ATH builds read ``ThroatExtSegments``.
    for throat_segments in ("ThroatSegments", "ThroatExtSegments"):
        if throat_segments in mesh_items:
            raise ConfigError(f"Mesh.{throat_segments} is not supported; remove it or use Mesh.ZMapPoints")

    # Every other key in this namespace is read through an explicit whitelist,
    # so an unrecognised one was silently discarded and the mesher used its
    # default instead. That is a wrong answer, not a cosmetic gap: mistyping
    # `Mesh.ThroatResolution` as `Mesh.ThroatResolutin` meshed at the default 4
    # rather than the requested 9 -- a 16,730-triangle mesh instead of 15,060,
    # with a clean exit code and no diagnostic. Every downstream solve inherits
    # it. Real ATH configs only ever use the recognised names (checked across
    # the 46 configs in the reference archive), so refusing is safe as well as
    # correct, and it matches how this function already treats every other
    # unsupported key.
    unknown = sorted(
        key
        for key in mesh_items
        if key not in _KNOWN_MESH_KEYS and not key.startswith("Enclosure.") and not _ath_item_disabled(key)
    )
    if unknown:
        details = []
        for key in unknown:
            close = get_close_matches(key, sorted(_KNOWN_MESH_KEYS), n=1, cutoff=0.7)
            details.append(f"Mesh.{key}" + (f" (did you mean Mesh.{close[0]}?)" if close else ""))
        raise ConfigError(
            "unrecognised mesh key(s): "
            + ", ".join(details)
            + ". They would be ignored and the mesher would silently use its "
            "defaults, so the mesh would not match the config."
        )


def _warn_ignored_ath_keys(
    namespace: str,
    items: Mapping[str, str],
    key_map: tuple[tuple[str, str], ...],
) -> None:
    """Say which keys of an ATH namespace were read and then dropped, and why.

    ``Morph.*`` and ``GCurve.*`` names that ATH does not define used to be
    accepted here as aliases for the ones it does -- ``Morph.Width`` for
    ``Morph.TargetWidth``, ``GCurve.Distance`` for ``GCurve.Dist``. ATH itself
    meshes such a config with the key dropped and its default in force, so
    honouring the written value built a different waveguide from the one ATH
    builds off the same file. The aliases are gone; ignoring these keys is what
    ATH does.

    Ignoring them *quietly* is not. Whoever wrote the key meant to change the
    geometry, so name it, name the value going unused, and point at the key
    that would have done the job. Unlike ``Mesh.*``, an unrecognised name here
    is a warning rather than a refusal: ATH meshes these configs, and refusing
    would leave the importer unable to read a file ATH accepts.
    """

    known = sorted(src for src, _dst in key_map)
    for key in sorted(items):
        if key in known or _ath_item_disabled(key):
            continue
        close = get_close_matches(key, known, n=1, cutoff=0.6)
        _report_ignored_ath_item(
            namespace + key,
            "ATH has no such key, so this mesher drops it too; recognized keys and defaults remain in force."
            + (f" Did you mean {namespace}{close[0]}?" if close else ""),
            value=items[key],
        )


def parse_text_config(content: str) -> dict[str, Any]:
    """Parse the text `.cfg` shape used by imported waveguide configs."""
    blocks, flat, nested = _parse_ath_blocks(content)
    # Validate supplied coefficients even in a section ATH would otherwise
    # ignore, except underscore-parked blocks. Field precedence must not
    # conceal an unsupported expression in an enabled section.
    for items in (flat, *(items for name, items in blocks.items() if not _ath_item_disabled(name))):
        stretch_coefficients({key: _maybe_number(items[key]) for key in ("s1", "s2") if key in items})
    formula = None
    profile_block: str | None = None
    profile_items: Mapping[str, str] = {}
    if "R-OSSE" in blocks:
        formula = "R-OSSE"
        profile_block = "R-OSSE"
    elif "ROSSE" in blocks:
        formula = "R-OSSE"
        profile_block = "ROSSE"
    elif "OSSE" in blocks:
        formula = "OSSE"
        profile_block = "OSSE"
    if profile_block is not None:
        profile_items = blocks[profile_block]
    elif any(key in flat for key in ("Coverage.Angle", "Length", "Term.n")):
        formula = "OSSE"
        profile_items = flat
    if formula is None:
        raise ConfigError(
            _missing_ath_profile_details(flat, blocks)
            + "text config must contain an OSSE or R-OSSE block "
            "(ICW is not available via the ATH text format; configure it through the "
            "dict/contract path, e.g. build_from_config({'profile': {'formula': 'ICW', ...}}))"
        )

    # ATH reads Throat.Ext.* and Slot.Length ONLY at top level, even for
    # block-style configs (verified against ath.exe: an R-OSSE block copy is
    # ignored while the top-level key grows the horn). Honouring in-block
    # copies would silently build geometry real ATH ignores, so reject them
    # and merge the top-level values into the profile mapping instead.
    if profile_items is not flat:
        extension_keys = ("Throat.Ext.Length", "Throat.Ext.Angle", "Slot.Length")
        in_block = [key for key in extension_keys if key in profile_items]
        if in_block:
            raise ConfigError(
                f"{', '.join(in_block)} must be top-level keys — ATH ignores them inside "
                f"the {formula} block; move them out of the block"
            )
        merged_profile = dict(profile_items)
        for key in extension_keys:
            if key in flat:
                merged_profile[key] = flat[key]
        profile_items = merged_profile

    # Names each mapping table was applied to, for the final check that nothing
    # in the file went unread. A flat config keeps its profile keys at top
    # level, so there the profile tables recognise top-level keys.
    known_flat: set[str] = set(_KNOWN_TOP_LEVEL_KEYS)
    known_profile: set[str] = {"Throat.Profile"}
    if profile_block is not None:
        known_flat.update(("Throat.Ext.Length", "Throat.Ext.Angle", "Slot.Length"))
        if profile_block == "OSSE":
            # ATH reads the length of a block-defined OSSE horn from the block's
            # L and ignores a top-level Length beside it: the exported grid is
            # byte-identical with and without the key (ath.exe V2026-08c).
            known_flat.update(("Length", "Rot"))
        elif _maybe_number(flat.get("Rot")) == 0:
            # An explicit numeric zero is the same as absent rotation. Do not
            # admit expressions or nonzero R-OSSE rotation without ATH parity.
            known_flat.add("Rot")

    def mapped(items: Mapping[str, str], pairs: tuple[tuple[str, str], ...]) -> dict[str, Any]:
        if items is flat:
            known_flat.update(src for src, _dst in pairs)
        elif items is profile_items:
            known_profile.update(src for src, _dst in pairs)
        out: dict[str, Any] = {}
        for src, dst in pairs:
            if src in items:
                out[dst] = _maybe_number(items[src])
        return out

    def prefixed(prefix: str) -> dict[str, str]:
        return {
            key[len(prefix) :]: value
            for key, value in flat.items()
            if key.startswith(prefix) and len(key) > len(prefix)
        }

    common_profile = mapped(
        profile_items,
        (
            ("r0", "r0"),
            ("Throat.Diameter", "throat_diameter"),
            ("a", "a"),
            ("Coverage.Angle", "a"),
            ("a0", "a0"),
            ("Throat.Angle", "a0"),
            ("k", "k"),
            ("OS.k", "k"),
            ("Term.k", "k"),
            ("q", "q"),
            ("Term.q", "q"),
            ("s1", "s1"),
            ("s2", "s2"),
            ("Throat.Ext.Length", "throatExtLength"),
            ("Throat.Ext.Angle", "throatExtAngle"),
            ("Slot.Length", "slotLength"),
        ),
    )
    if "throat_diameter" in common_profile and "r0" not in common_profile:
        try:
            common_profile["r0"] = float(common_profile.pop("throat_diameter")) / 2.0
        except (TypeError, ValueError):
            common_profile.pop("throat_diameter", None)

    if formula == "OSSE":
        profile = {
            **common_profile,
            "_athLengthMode": "total",
            **mapped(
                profile_items,
                (
                    ("L", "L"),
                    ("Length", "L"),
                    ("n", "n"),
                    ("Term.n", "n"),
                    ("s", "s"),
                    ("Term.s", "s"),
                    # Not an ATH key: ATH V2025-06 ignores OS.h (checked
                    # against ath.exe: an OS.h = 10 grid is identical to one
                    # without it). Waveguide Generator writes it for its
                    # half-sine bulge, so it is honoured here on the same
                    # footing as Morph.Exponent and Mesh.SurfaceFit.
                    ("h", "h"),
                    ("OS.h", "h"),
                    ("Rot", "rot"),
                ),
            ),
        }
    else:
        profile = {
            **common_profile,
            "_athLengthMode": "total",
            **mapped(
                profile_items,
                (
                    ("R", "R"),
                    ("m", "m"),
                    ("b", "b"),
                    ("r", "r"),
                    ("tmax", "tmax"),
                ),
            ),
        }

    if formula == "OSSE":
        if "L" not in profile:
            raise ConfigError("ATH OSSE text configs must set Length")
        if "OSSE" in blocks:
            # V2025-12 honours top-level Rot even with an OSSE block; Length
            # stays subordinate to the block's L.
            if "Rot" not in blocks["OSSE"]:
                profile.update(mapped(flat, (("Rot", "rot"),)))
        # ATH defaults for keys the import may omit (Ath 4.8.2 User Guide 4.1.1).
        # Native TOML/JSON configs keep the package defaults in config_builder.
        profile.setdefault("a0", 0)
        profile.setdefault("s", 0.7)

    mesh_items = {**prefixed("Mesh."), **blocks.get("Mesh", {})}
    mesh = mapped(
        mesh_items,
        (
            ("AngularSegments", "angularSegments"),
            ("CornerSegments", "cornerSegments"),
            ("LengthSegments", "lengthSegments"),
            ("WallThickness", "wallThickness"),
            ("VerticalOffset", "verticalOffset"),
            ("Quadrants", "quadrants"),
            ("ThroatResolution", "throatResolution"),
            ("MouthResolution", "mouthResolution"),
            ("RearResolution", "rearResolution"),
            ("SubdomainSlices", "subdomainSlices"),
            ("InterfaceOffset", "interfaceOffset"),
            ("InterfaceResolution", "interfaceResolution"),
            ("SamplingMode", "samplingMode"),
            # Not an ATH key -- ATH has no equivalent -- but this parser also
            # reads WG-authored text configs, and without a mapping the
            # acoustic patch fit was reachable from TOML only. That put the
            # interpolating fit out of reach of exactly the users most likely
            # to want it: those importing an ATH config.
            ("SurfaceFit", "surfaceFit"),
        ),
    )
    # ATH SubdomainSlices index the segments 0..LengthSegments-1, where the
    # last slice is the mouth; internal indices address grid rings, so shift
    # by one (Ath 4.8.2 User Guide 6.7). An explicit empty value stays empty.
    raw_slices = mesh.get("subdomainSlices")
    if raw_slices is not None and str(raw_slices).strip():
        shifted: list[str] = []
        for part in str(raw_slices).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                shifted.append(str(int(float(part)) + 1))
            except ValueError as exc:
                raise ConfigError(f"Mesh.SubdomainSlices must be integers, got {part!r}") from exc
        mesh["subdomainSlices"] = ",".join(shifted)

    zmap_points = mesh_items.get("ZMapPoints", mesh_items.get("ZMap"))
    if zmap_points is not None:
        mesh["zMapPoints"] = zmap_points
    mesh.setdefault("samplingMode", "zmap" if zmap_points is not None else "ath-default-zmap")
    # ATH defaults for keys the import may omit, verified against ath.exe
    # V2025-12 by byte-identical mesh probes (absent vs explicit value):
    # WallThickness 5, ThroatResolution 4, MouthResolution 8, RearResolution 15.
    mesh.setdefault("wallThickness", 5)
    mesh.setdefault("throatResolution", 4)
    mesh.setdefault("mouthResolution", 8)
    mesh.setdefault("rearResolution", 15)
    morph_items = {
        **prefixed("Morph."),
        **prefixed("MORPH."),
        **blocks.get("Morph", {}),
        **blocks.get("MORPH", {}),
    }
    morph = mapped(morph_items, _MORPH_KEY_MAP)
    if "morphTarget" in morph:
        # ATH default Morph.CornerRadius is 35, not 0 (Ath 4.8.2 User Guide 4.1.2).
        morph.setdefault("morphCorner", 35)
        # ATH's effective Morph.FixedPart default is 0.2, not the 0 its user
        # guide documents: ath.exe V2025-06 builds a byte-identical grid with
        # the key absent and with ``Morph.FixedPart = 0.2``, and an explicit 0
        # differs. The m2-clone reference omits the key and relies on it.
        morph.setdefault("morphFixed", 0.2)
        # ATH morphs Slot.Length like the rest of the horn (ath.exe probe:
        # a 15 mm slot with FixedPart 0 blends from the first slice), unlike
        # this mesher's own contract, which keeps a slot straight.
        morph["_morphKeepsSlot"] = False
    if "morphAllowShrinkage" in morph:
        morph["morphAllowShrinkage"] = _ath_bool(morph["morphAllowShrinkage"])

    gcurve_items = {
        **prefixed("GCurve."),
        **prefixed("GCURVE."),
        **blocks.get("GCurve", {}),
        **blocks.get("GCURVE", {}),
    }
    gcurve = mapped(gcurve_items, _GCURVE_KEY_MAP)

    enc_items = {**prefixed("Mesh.Enclosure."), **blocks.get("Mesh.Enclosure", {})}
    enclosure: dict[str, Any] = {}
    if enc_items:
        enclosure = mapped(enc_items, _ENCLOSURE_KEY_MAP)
        spacing = enc_items.get("Spacing")
        if spacing:
            parts = [part.strip() for part in spacing.split(",")]
            if len(parts) < 4:
                # ATH documents last-value repetition for the resolution lists
                # only; what it does with a short Spacing is not established.
                raise ConfigError(
                    f"Mesh.Enclosure Spacing = {spacing} must list four margins (left, top, right, bottom)"
                )
            if len(parts) >= 4:
                enclosure.update(
                    {
                        "space_l_mm": _maybe_number(parts[0]),
                        "space_t_mm": _maybe_number(parts[1]),
                        "space_r_mm": _maybe_number(parts[2]),
                        "space_b_mm": _maybe_number(parts[3]),
                    }
                )

    source_items = {**prefixed("Source."), **blocks.get("Source", {})}
    source = mapped(source_items, _SOURCE_KEY_MAP)
    if "sourceShape" in source:
        # ATH enum: 1 = spherical cap, 2 = flat disc. Internal enum: 1 = cap, 0 = flat disc.
        ath_shape = source["sourceShape"]
        if ath_shape == 2:
            source["sourceShape"] = 0
        elif ath_shape != 1:
            raise ConfigError(f"Source.Shape = {ath_shape!r} is not supported; use 1 (cap) or 2 (flat disc)")
    if "sourceVelocity" in source:
        # ATH enum: 1 = normal to the element surface, 2 = axial (pistonic
        # motion along z). It selects a boundary condition rather than
        # geometry -- the cap or disc is meshed identically either way -- so
        # the exported ``.msh`` cannot carry it and a solver reading only the
        # mesh would drive an axial model normally. Refuse 2 loudly instead of
        # meshing it into that; the value is kept on the parsed config so a
        # consumer that does model velocity direction sees what was asked for.
        ath_velocity = source["sourceVelocity"]
        if ath_velocity == 2:
            raise ConfigError(
                "Source.Velocity = 2 (axial source motion) is not supported by this mesher; "
                "only 1 (normal to the element surface) is implemented"
            )
        if ath_velocity != 1:
            raise ConfigError(
                f"Source.Velocity = {ath_velocity!r} is not supported; "
                "use 1 (normal to the element surface)"
            )

    nested_names = {path[-1]: {} for path in nested if not _ath_nested_ignored(path)}
    _reject_unsupported_ath_keys(flat, profile_items, mesh_items, {**nested_names, **blocks})

    validate_stretch_composition(
        {**profile, **gcurve,
         "rot": profile.get("rot", _maybe_number(flat.get("Rot", "0")))},
        formula,
        length_supplied="Length" in flat,
    )

    # ABEC.SimType selects the mesh topology: 1 = infinite baffle (the ATH
    # default), 2 = free standing. An enclosure implies a free-standing sim.
    sim_type = _maybe_number(flat.get("ABEC.SimType"))
    if sim_type is None:
        sim_type = 2 if enclosure else 1
    if sim_type not in (1, 2):
        raise ConfigError(
            f"ABEC.SimType = {sim_type!r} is not supported; use 1 (infinite baffle) or 2 (free standing)"
        )
    if sim_type == 1 and enclosure:
        raise ConfigError("ABEC.SimType = 1 (infinite baffle) cannot be combined with Mesh.Enclosure")

    # Last, so that an item with a refusal of its own above keeps that message.
    _reject_unrecognised_ath_items(
        flat,
        blocks,
        nested,
        known_flat=known_flat,
        profile_block=profile_block,
        known_profile=known_profile,
    )

    _report_ignored_ath_items(flat, blocks, nested, profile_block=profile_block)

    config: dict[str, Any] = {"formula": formula, "profile": canonical_stretch_params(profile), "mesh": mesh, "simType": sim_type}
    # Global Scale multiplies every linear geometry dimension after profile
    # evaluation (resolutions and mesh sizes stay in raw millimetres).
    scale = _maybe_number(flat.get("Scale"))
    if scale is not None:
        config["scale"] = scale
    if morph:
        config["morph"] = morph
    if gcurve:
        config["gcurve"] = gcurve
    if enclosure:
        config["enclosure"] = enclosure
    if source:
        config["source"] = source
    return config


parse_legacy_config = parse_text_config
parse_ath_config = parse_text_config
