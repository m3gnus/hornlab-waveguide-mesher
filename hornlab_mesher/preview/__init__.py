"""Versioned, render-ready preview geometry.

The preview package is intentionally independent of the solver mesh.  It
reuses the canonical point-grid and enclosure samplers, then owns display
topology, surface roles, and normals as required by ``hornlab.preview/1``.
"""

from .api import (
    PreviewGeometryV1,
    PreviewOptionsV1,
    PreviewSurfaceV1,
    build_preview_geometry,
)

__all__ = [
    "PreviewGeometryV1",
    "PreviewOptionsV1",
    "PreviewSurfaceV1",
    "build_preview_geometry",
]
