"""Versioned geometry interpretation retained by imported text configurations."""

import re
from typing import Any, Mapping

TEXT_IMPORT_VERSION_KEY = "_textImportVersion"
TEXT_IMPORT_VERSION = "ath-2026-08c-v1"
NATIVE_GEOMETRY_VERSION = "native-v1"
GEOMETRY_STAMP_LABEL = "Waveguide Generator geometry-interpretation:"


def text_uses_import_geometry(content: str) -> bool:
    """Read the saved CFG contract before the lexer discards comments.

    Keep stamp and historical MWG detection aligned with the application's
    ``server/design/textcfg.py`` reader. An explicit stamp owns interpretation.
    """
    stamp = re.compile(r"^;\s*Waveguide Generator geometry-interpretation:\s*(\S+)\s*$", re.IGNORECASE)
    versions = {match.group(1) for line in content.splitlines()
                if (match := stamp.match(line.strip())) is not None}
    if len(versions) > 1:
        raise ValueError("conflicting geometry interpretation stamps")
    version = next(iter(versions), None)
    if version not in (None, NATIVE_GEOMETRY_VERSION, TEXT_IMPORT_VERSION):
        raise ValueError(f"unsupported geometry interpretation {version!r}")
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
