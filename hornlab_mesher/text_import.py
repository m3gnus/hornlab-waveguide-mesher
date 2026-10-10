"""Versioned geometry interpretation retained by imported text configurations."""

import re
from typing import Any, Mapping

TEXT_IMPORT_VERSION_KEY = "_textImportVersion"
TEXT_IMPORT_VERSION = "ath-2026-08c-v1"
NATIVE_GEOMETRY_VERSION = "native-v1"
GEOMETRY_STAMP_LABEL = "Waveguide Generator geometry-interpretation:"


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
    if version not in (None, NATIVE_GEOMETRY_VERSION, TEXT_IMPORT_VERSION):
        raise ValueError(f"unsupported geometry interpretation {version!r}")
    return version


def text_uses_import_geometry(content: str) -> bool:
    """Select explicit saved interpretation or the historical dialect rule."""
    version = text_geometry_version(content)
    if version is not None:
        return version == TEXT_IMPORT_VERSION
    return re.search(r";\s*(?:Parameter|MWG) config", content, re.IGNORECASE) is None


def uses_text_import_geometry(values: Mapping[str, Any]) -> bool:
    """Recognize the bounded import contract; never guess a future version."""
    version = values.get(TEXT_IMPORT_VERSION_KEY)
    if version is None:
        return False
    if version != TEXT_IMPORT_VERSION:
        raise ValueError(f"unsupported text import geometry version {version!r}")
    return True
