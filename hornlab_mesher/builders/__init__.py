"""Builder entry points for Gmsh/OCC geometry creation.

Builder modules create OCC surfaces and return grouped surface tags. They do
not configure mesh density, write mesh files, or validate final triangle
orientation; `hornlab_mesher.mesher` owns that orchestration boundary.
"""

from __future__ import annotations

from .osse_waveguide import (
    build_osse_waveguide,
    compute_osse_inner_points,
    compute_osse_profile_points,
)
from .point_grid import build_point_grid
from .rosse_waveguide import compute_rosse_profile_points

__all__ = [
    "build_osse_waveguide",
    "build_point_grid",
    "compute_osse_inner_points",
    "compute_osse_profile_points",
    "compute_rosse_profile_points",
]
