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
from .mesher import MesherError, build_mesh, load_mesh
from .tags import PhysicalGroup


def build_from_config(*args, **kwargs):
    from .cli import build_from_config as _build_from_config

    return _build_from_config(*args, **kwargs)


def build_geometry_params(*args, **kwargs):
    from .cli import build_geometry_params as _build_geometry_params

    return _build_geometry_params(*args, **kwargs)


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
    "build_from_config",
    "build_geometry_params",
    "build_mesh",
    "build_osse_waveguide",
    "compute_osse_inner_points",
    "compute_osse_profile_points",
    "compute_rosse_profile_points",
    "load_config",
    "load_mesh",
]
