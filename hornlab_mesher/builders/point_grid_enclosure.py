"""Compatibility import for the point-grid dispatcher.

The dispatcher used to live in this module. New code should import through
`hornlab_mesher.builders.point_grid` or `point_grid_dispatch`.
"""

from __future__ import annotations

from .point_grid_dispatch import build_point_grid

__all__ = ["build_point_grid"]
