"""Versioned geometry interpretation retained by imported text configurations."""

from typing import Any, Mapping

TEXT_IMPORT_VERSION_KEY = "_textImportVersion"
TEXT_IMPORT_VERSION = "ath-2026-08c-v1"


def uses_text_import_geometry(values: Mapping[str, Any]) -> bool:
    """Recognize the bounded import contract; never guess a future version."""
    version = values.get(TEXT_IMPORT_VERSION_KEY)
    if version is None:
        return False
    if version != TEXT_IMPORT_VERSION:
        raise ValueError(f"unsupported text import geometry version {version!r}")
    return True
