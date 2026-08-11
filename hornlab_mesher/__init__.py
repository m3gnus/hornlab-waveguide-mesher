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
from .mesh_sizing import (
    ROLE_NEAR_FIELD,
    ROLE_RADIATING,
    ROLE_SHADOW,
    ROLE_SOURCE,
    ROLE_THROAT,
    SPEED_OF_SOUND_M_S,
    VALIDATION_EPW,
    MeshCostEstimate,
    Region,
    estimate_mesh_cost,
    graded_size_mm,
    role_size_mm,
    valid_f_max_hz,
)
from .cad import (
    CadInfo,
    WgLinkIdentity,
    WgLinkInfo,
    read_wglink,
    write_step,
    write_step_from_config,
    write_wglink,
)
from .datums import derive_datums
from .mesher import MesherError, build_mesh, build_mesh_with_info, load_mesh
from .tags import PhysicalGroup
from .step_prepare import (
    DEFAULT_AUTO_CUT_GRID,
    DEFAULT_AUTO_CUT_TOLERANCE_REL,
    DEFAULT_SYMMETRY_SNAP_BAND_MM,
    OccAutoCutResult,
    OccSurfaceGroup,
    OccSurfaceRole,
    OccSurfaceSelector,
    PlaneSymmetryVerdict,
    auto_cut_occ_geometry,
    evaluate_occ_plane_symmetry,
    millimetres_to_step_units,
    remap_surface_tags,
    sample_occ_surface_points,
    snap_symmetry_plane_vertices,
)


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
    "CadInfo",
    "WgLinkIdentity",
    "WgLinkInfo",
    "CrossSection",
    "Enclosure",
    "HornEnclosure",
    "MeshDensity",
    "MeshInfo",
    "OsseHornGeometry",
    "RosseHornGeometry",
    "PhysicalGroup",
    "OccAutoCutResult",
    "OccSurfaceGroup",
    "OccSurfaceRole",
    "OccSurfaceSelector",
    "PlaneSymmetryVerdict",
    "DEFAULT_AUTO_CUT_GRID",
    "DEFAULT_AUTO_CUT_TOLERANCE_REL",
    "DEFAULT_SYMMETRY_SNAP_BAND_MM",
    "MesherError",
    "SolveCostEstimate",
    "MeshCostEstimate",
    "Region",
    "ROLE_NEAR_FIELD",
    "ROLE_RADIATING",
    "ROLE_SHADOW",
    "ROLE_SOURCE",
    "ROLE_THROAT",
    "SPEED_OF_SOUND_M_S",
    "VALIDATION_EPW",
    "estimate_mesh_cost",
    "estimate_solve_cost",
    "estimate_triangle_count",
    "graded_size_mm",
    "role_size_mm",
    "valid_f_max_hz",
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
    "write_step",
    "write_step_from_config",
    "write_wglink",
    "read_wglink",
    "derive_datums",
    "auto_cut_occ_geometry",
    "evaluate_occ_plane_symmetry",
    "millimetres_to_step_units",
    "normalized_arc_positions",
    "remap_surface_tags",
    "resample_grid_onto_existing",
    "resample_point_grid",
    "sample_occ_surface_points",
    "snap_symmetry_plane_vertices",
]


def normalized_arc_positions(*args, **kwargs):
    """Lazily load the SciPy-backed grid resampler."""
    from .grid_resample import normalized_arc_positions as _implementation

    return _implementation(*args, **kwargs)


def resample_point_grid(*args, **kwargs):
    """Lazily load the SciPy-backed grid resampler."""
    from .grid_resample import resample_point_grid as _implementation

    return _implementation(*args, **kwargs)


def resample_grid_onto_existing(*args, **kwargs):
    """Lazily load the SciPy-backed grid resampler."""
    from .grid_resample import resample_grid_onto_existing as _implementation

    return _implementation(*args, **kwargs)
