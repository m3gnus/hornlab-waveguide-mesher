"""Validate supplied native intent before legacy precedence or CAD rewriting."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .config_parser import ConfigError


@dataclass(frozen=True)
class NativeFeature:
    slot: str
    path: tuple[str, ...]
    legacy_marker: str
    mode: str | None = None
    aliases: tuple[tuple[str, ...], ...] = ()
    # Strict payloads are found in any spelling at any depth and must sit at
    # their exact path; others count only at their path or documented aliases.
    # Every payload is strict under OSSE-NATIVE, which has no legacy readers.
    strict: bool = False


_ADAPTER_ALIASES = (("profile", "throat_adapter"),)

# The table owns payload locations, slot occupancy and legacy spellings.
FEATURES = {
    "adapter": NativeFeature("throat", ("throat_adapter",), "OSSE-ADAPTER", "authored", _ADAPTER_ALIASES),
    "controls": NativeFeature("throat", ("throat_adapter",), "OSSE-ADAPTER-CONTROLS", "controls", _ADAPTER_ALIASES),
    "arc": NativeFeature("mouth", ("terminating_arc",), "OSSE-ARC", strict=True),
    "roundover": NativeFeature("mouth", ("mesh", "mouth_roundover_radius_mm"), "OSSE-ROUNDOVER"),
    "axial": NativeFeature("axial", ("axial_scale",), "OSSE-AXIAL"),
    "source_disk": NativeFeature("source", ("source_body",), "SOURCE-DISK", strict=True),
}
LEGACY_MARKERS = {frozenset({name}): feature.legacy_marker for name, feature in FEATURES.items()}
LEGACY_MARKERS[frozenset({"adapter", "axial"})] = "OSSE-ADAPTER-AXIAL"
IMPLEMENTED_COMPOSITES = frozenset({
    frozenset({"arc", "axial"}),
    frozenset({"adapter", "arc"}),
    frozenset({"adapter", "arc", "axial"}),
    frozenset({"roundover", "axial"}),
})
NATIVE_MARKERS = frozenset(LEGACY_MARKERS.values()) | {"OSSE-NATIVE"}

# Even a composite has one model object per geometry, owning its finished meridian.
NATIVE_MODEL_ATTRIBUTES = ("adapter_meridian", "roundover", "axial_model", "controls_meridian", "arc_meridian")


def _fail(message):
    raise ConfigError("Native geometry refused: " + message)


def _numeric(value, name):
    try:
        valid = type(value) in (int, float) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        _fail(f"{name} must be a finite scalar number.")
    return float(value)


def _native_intent(config):
    sections = [config]+[config[key] for key in ("profile", "parameters") if isinstance(config.get(key), Mapping)]
    formulas = [section[key] for section in sections for key in ("formula", "type") if key in section]
    normalized = [str(value).strip().upper() for value in formulas]
    markers = set(normalized).intersection(NATIVE_MARKERS)
    if len(markers) > 1:
        _fail("supplied native formula markers must agree.")
    marker = next(iter(markers), None)
    if marker is not None:
        for value in formulas:
            if not isinstance(value, str) or value.strip().upper() != marker:
                _fail("every supplied formula and type must agree with the native formula marker.")

    def strict(feature):
        return feature.strict or marker == "OSSE-NATIVE"

    payloads = []
    pending = [(config, ())]
    seen = set()
    while pending:
        section, path = pending.pop()
        if not isinstance(section, Mapping) or id(section) in seen:
            continue
        seen.add(id(section))
        for key, value in section.items():
            spelling = str(key).replace("_", "").lower()
            for name, feature in FEATURES.items():
                if spelling != feature.path[-1].replace("_", "").lower():
                    continue
                if not strict(feature) and path+(key,) not in (feature.path, *feature.aliases):
                    continue
                if feature.mode is not None and (not isinstance(value, Mapping) or value.get("mode") != feature.mode):
                    continue
                # Zero roundover radius is existing inactive editor state.
                if name == "roundover" and value == 0:
                    continue
                payloads.append((name, path+(key,)))
            if isinstance(value, Mapping):
                pending.append((value, path+(key,)))
    features = frozenset(name for name, _ in payloads)
    if marker is None:
        # Preserve ordinary parsing, except the existing fail-closed terminals.
        for name in ("source_disk", "arc"):
            if name in features:
                feature = FEATURES[name]
                key = feature.path[-1]
                _fail(f"{key} requires {feature.legacy_marker} and the exact root {key} key.")
        return None, features

    if ("source_disk" in features or marker == "SOURCE-DISK") and (features-{"source_disk"} or marker != "SOURCE-DISK"):
        _fail("SOURCE-DISK must stay alone and cannot compose with native features.")
    # Declared legacy sets never conflict, so this only ever names an extra feature.
    slots = {}
    for name in sorted(features):
        slot = FEATURES[name].slot
        if slot in slots:
            _fail(f"conflict: {slots[slot]} and {name} both own the {slot}.")
        slots[slot] = name

    for name, path in payloads:
        feature = FEATURES[name]
        if strict(feature) and path != feature.path:
            key = feature.path[-1]
            location = "root" if len(feature.path) == 1 else ".".join(feature.path[:-1])
            _fail(f"{key} requires the exact {location} {key} key.")

    if marker != "OSSE-NATIVE":
        declared = next(names for names, spelling in LEGACY_MARKERS.items() if spelling == marker)
        if not features-declared:
            return marker, features
    legacy = LEGACY_MARKERS.get(features)
    if legacy is not None and legacy != marker:
        _fail(f"write {legacy} for this feature set.")
    if len(features) < 2:
        _fail("OSSE-NATIVE requires at least two active features.")
    if features not in IMPLEMENTED_COMPOSITES:
        _fail(f"{', '.join(sorted(features))} is not supported yet.")
    if marker != "OSSE-NATIVE":
        _fail(f"{marker} carries additional native features; write OSSE-NATIVE for compositions without a legacy marker.")
    return marker, features


def native_features(config):
    """Return active features after the shared spelling, slot and support rules."""
    return _native_intent(config)[1]


def validate_native_boundary(config, *, allow_large_mesh=None):
    """Return the native marker after validation, or None for ordinary inputs.

    Every supplied alias is inspected. The legacy parser deliberately retains
    its old coercions for ordinary formulas, but cannot hide a native request.
    """
    marker, _ = _native_intent(config)
    if marker is None:
        return None
    for name in ("profile", "parameters", "mesh", "source", "Source"):
        if name in config and not isinstance(config[name],Mapping):
            _fail(f"{name} must be an object.")
    mesh = config.get("mesh", {})
    for section in (config,mesh):
        if "quadrants" in section:
            value = section["quadrants"]
            if not ((type(value) is int and value == 1234) or (type(value) is str and value == "1234")):
                _fail("quadrants must be exactly 1234 for full-circle coverage.")

    def supplied(names):
        return [(key,section[key]) for section in (config,mesh) for key in names if key in section]

    def agree(values, name):
        if values and any(value != values[0][1] for _,value in values[1:]):
            _fail(f"duplicate {name} controls must agree.")

    for names in (("allow_large_mesh","allowLargeMesh"),("scale_to_metres","scaleToMetres")):
        values = supplied(names)
        for key,value in values:
            if type(value) is not bool:
                _fail(f"{key} must be a boolean.")
        agree(values,names[0])
        if any(name in config for name in names):
            _fail(f"{names[0]} must be supplied inside mesh.")
    if allow_large_mesh is not None and type(allow_large_mesh) is not bool:
        _fail("allow_large_mesh override must be a boolean or None.")
    placement = supplied(("vertical_offset_mm","verticalOffset"))
    for key,value in placement:
        _numeric(value,key)
    agree(placement,"vertical placement")
    for names in (("angular_segments","angularSegments"),("length_segments","lengthSegments"),("max_triangles","maxTriangles")):
        values = supplied(names)
        for key,value in values:
            number = _numeric(value,key)
            if not 0 < number <= 10_000_000 or not number.is_integer():
                _fail(f"{key} must be a positive integer no larger than 10000000.")
        agree(values,names[0])
    return marker
