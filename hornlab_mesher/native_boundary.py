"""Validate supplied native intent before legacy precedence or CAD rewriting."""
from __future__ import annotations

import math
from typing import Mapping

from .config_parser import ConfigError

NATIVE_MARKERS = frozenset({"OSSE-AXIAL", "OSSE-ADAPTER", "OSSE-ROUNDOVER", "SOURCE-DISK"})


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


def validate_native_boundary(config, *, allow_large_mesh=None):
    """Return the native marker after validation, or None for ordinary inputs.

    Every supplied alias is inspected. The legacy parser deliberately retains
    its old coercions for ordinary formulas, but cannot hide a native request.
    """
    sections = [config]+[config[key] for key in ("profile", "parameters") if isinstance(config.get(key), Mapping)]
    formulas = [section[key] for section in sections for key in ("formula", "type") if key in section]
    normalized = [str(value).strip().upper() for value in formulas]
    markers = set(normalized).intersection(NATIVE_MARKERS)
    # Inspect source-body intent before the ordinary parser's early return.
    # No spelling or location may quietly turn an exterior body into a horn.
    payloads = []
    pending = [config]
    seen = set()
    while pending:
        section = pending.pop()
        if not isinstance(section,Mapping) or id(section) in seen:
            continue
        seen.add(id(section))
        for key,value in section.items():
            if str(key).replace("_","").lower()=="sourcebody":
                payloads.append((section,key))
            if isinstance(value,Mapping):
                pending.append(value)
    if payloads and (markers != {"SOURCE-DISK"} or any(section is not config or key!="source_body" for section,key in payloads)):
        _fail("source_body requires a root SOURCE-DISK formula and the exact root source_body key.")
    if not markers:
        return None
    if len(markers) != 1:
        _fail("supplied native formula markers must agree.")
    marker = next(iter(markers))
    for value in formulas:
        if not isinstance(value,str) or value.strip().upper() != marker:
            _fail("every supplied formula and type must agree with the native formula marker.")
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
