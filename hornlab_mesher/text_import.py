"""Versioned geometry interpretation retained by imported text configurations."""

import re
from typing import Any, Mapping

TEXT_IMPORT_VERSION_KEY = "_textImportVersion"
TEXT_IMPORT_V1 = "ath-2026-08c-v1"
TEXT_IMPORT_VERSION = "ath-2026-09a-v2"
NATIVE_GEOMETRY_VERSION = "native-v1"
GEOMETRY_STAMP_LABEL = "Waveguide Generator geometry-interpretation:"
ATH_AXIAL_RULE_KEY = "_athAxialSamplingRule"
ATH_AXIAL_LEGACY = "ath-v2025-06"
ATH_AXIAL_CURRENT = "ath-v2025-12+"


def ath_axial_sampling_rule(content: str) -> tuple[str, str | None]:
    """Interpret ATH's version echo in the leading comment header only.

    Month/suffix variants through June 2025 use the historical rule. Builds
    between June and December 2025 are unverified and use the current rule.
    """
    echo = re.compile(r"^[ \t]*;[ \t]*Ath version (V([0-9]{4})-(0[1-9]|1[0-2])[a-z]?)[ \t]*$", re.IGNORECASE)
    versions: dict[str, tuple[int, int]] = {}
    for line in content.removeprefix("\ufeff").splitlines():
        if line.strip() and not line.lstrip().startswith(";"):
            break
        match = echo.fullmatch(line)
        if match:
            versions[match.group(1).lower()] = (int(match.group(2)), int(match.group(3)))
    if len(versions) > 1:
        raise ValueError("conflicting ATH version echoes")
    version = next(iter(versions), None)
    legacy = version is not None and versions[version] <= (2025, 6)
    return (ATH_AXIAL_LEGACY if legacy else ATH_AXIAL_CURRENT), version


def uses_current_ath_axial_sampling(values: Mapping[str, Any]) -> bool:
    """Only imported geometry can select either ATH axial interpretation."""
    return (uses_v2_text_import_geometry(values)
            and values.get(ATH_AXIAL_RULE_KEY, ATH_AXIAL_CURRENT) == ATH_AXIAL_CURRENT)


def uses_v2_text_import_geometry(values: Mapping[str, Any]) -> bool:
    """Select current import geometry independently of the ATH sampling echo."""
    return (uses_text_import_geometry(values)
            and values.get(TEXT_IMPORT_VERSION_KEY) == TEXT_IMPORT_VERSION)


def text_geometry_version(content: str) -> str | None:
    """Read the saved CFG contract before the lexer discards comments.

    Keep stamp and historical MWG detection aligned with the application's
    ``server/design/textcfg.py`` reader. An explicit stamp owns interpretation.
    """
    stamp = re.compile(r"^;\s*Waveguide Generator geometry-interpretation:\s*(\S+)\s*$", re.IGNORECASE)
    versions: set[str] = set()
    depth = 0
    function_depth = 0
    function_opened = False
    collecting_function = False
    for raw_line in content.splitlines():
        if collecting_function:
            function_opened |= "{" in raw_line
            function_depth += raw_line.count("{") - raw_line.count("}")
            if function_opened and function_depth <= 0:
                collecting_function = False
            continue
        line, separator, comment = raw_line.partition(";")
        # WG collects comments before handling block open/close tokens. An
        # opener's inline comment is top-level; a closer's remains in-block.
        if depth == 0 and separator:
            match = stamp.match(";" + comment.strip())
            if match is not None:
                versions.add(match.group(1))
        line = line.strip()
        if "=" in line:
            value = line.split("=", 1)[1].strip()
            if value.startswith("function anonymous("):
                # Match WG's legacy Function#toString collector. The direct
                # CFG grammar still refuses unsupported function payloads;
                # their body comments cannot select a geometry contract.
                collecting_function = True
                function_opened = "{" in value
                function_depth = value.count("{") - value.count("}")
            elif value == "{":
                depth += 1
        elif line == "}":
            depth = max(0, depth - 1)
    if len(versions) > 1:
        raise ValueError("conflicting geometry interpretation stamps")
    version = next(iter(versions), None)
    if version not in (None, NATIVE_GEOMETRY_VERSION, TEXT_IMPORT_V1, TEXT_IMPORT_VERSION):
        raise ValueError(f"unsupported geometry interpretation {version!r}")
    return version


def text_uses_import_geometry(content: str) -> bool:
    """Select explicit saved interpretation or the historical dialect rule."""
    version = text_geometry_version(content)
    if version is not None:
        return version in (TEXT_IMPORT_V1, TEXT_IMPORT_VERSION)
    return re.search(r";\s*(?:Parameter|MWG) config", content, re.IGNORECASE) is None


def uses_text_import_geometry(values: Mapping[str, Any]) -> bool:
    """Recognize the bounded import contract; never guess a future version."""
    version = values.get(TEXT_IMPORT_VERSION_KEY)
    if version is None:
        return False
    if version not in (TEXT_IMPORT_V1, TEXT_IMPORT_VERSION):
        raise ValueError(f"unsupported text import geometry version {version!r}")
    rule = values.get(ATH_AXIAL_RULE_KEY)
    if rule not in (None, ATH_AXIAL_LEGACY, ATH_AXIAL_CURRENT):
        raise ValueError(f"unsupported ATH axial sampling rule {rule!r}")
    return True
