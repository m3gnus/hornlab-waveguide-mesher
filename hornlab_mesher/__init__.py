"""Public API for HornLab waveguide mesh generation.

Most callers should use `build_from_config` for config-driven builds or
`build_mesh` with a geometry dataclass for direct Python builds. Lower-level
formula and builder helpers are exported only when they are stable enough for
tests or integration tooling to depend on.
"""

from __future__ import annotations

from .builders import (
    build_osse_waveguide,
    compute_osse_inner_points,
    compute_osse_profile_points,
    compute_rosse_profile_points,
)
from .geometry import (
    CrossSection,
    Enclosure,
    HornEnclosure,
    MeshDensity,
    MeshInfo,
    OsseHornGeometry,
    RosseHornGeometry,
)
from .cost import (
    SolveCostEstimate,
    estimate_solve_cost,
    estimate_triangle_count,
)
from .mesher import MesherError, build_mesh, build_mesh_with_info, load_mesh
from .tags import PhysicalGroup


def build_from_config(*args, **kwargs):
    from .config_builder import build_from_config as _build_from_config

    return _build_from_config(*args, **kwargs)


def build_geometry_params(*args, **kwargs):
    from .config_builder import build_geometry_params as _build_geometry_params

    return _build_geometry_params(*args, **kwargs)


def build_meridian(*args, **kwargs):
    from .config_builder import build_meridian as _build_meridian

    return _build_meridian(*args, **kwargs)


def circsym_rejection_reasons(*args, **kwargs):
    from .config_builder import circsym_rejection_reasons as _circsym_rejection_reasons

    return _circsym_rejection_reasons(*args, **kwargs)


def load_config(*args, **kwargs):
    from .cli import load_config as _load_config

    return _load_config(*args, **kwargs)


__all__ = [
    "CrossSection",
    "Enclosure",
    "HornEnclosure",
    "MeshDensity",
    "MeshInfo",
    "OsseHornGeometry",
    "RosseHornGeometry",
    "PhysicalGroup",
    "MesherError",
    "SolveCostEstimate",
    "estimate_solve_cost",
    "estimate_triangle_count",
    "build_from_config",
    "build_geometry_params",
    "build_meridian",
    "circsym_rejection_reasons",
    "build_mesh",
    "build_mesh_with_info",
    "build_osse_waveguide",
    "compute_osse_inner_points",
    "compute_osse_profile_points",
    "compute_rosse_profile_points",
    "load_config",
    "load_mesh",
]
