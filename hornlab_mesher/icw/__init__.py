"""Intrinsic-Curvature Waveguide (ICW) generator.

A gmsh-free geometry kernel: the meridian curvature kappa(s) is a clamped cubic B-spline,
integrated to the tangent angle theta(s) and then to the meridian (x(s), r(s)). One intrinsic
curve supports local curvature control, exact flat-baffle termination (theta=90 deg, kappa=0),
and free-standing rollback (theta>90 deg) without any piecewise body/lip join.

Public API is re-exported from the submodules. The package keeps no gmsh dependency so it can
be lifted out for a standalone optimiser.
"""

from __future__ import annotations

from .core import (
    DEFAULT_DEGREE,
    DEFAULT_SAMPLES,
    ICWCurve,
    ICWSample,
    clamped_uniform_knots,
    kappa_spline,
)
from .solver import (
    FeasibilityReport,
    ICWTargets,
    TerminationMode,
    curve_from_shape_modes,
    n_shape_modes,
    solve_icw,
)
from .seed import (
    fit_error,
    fit_from_points,
    seed_from_osse,
    seed_from_rosse,
)
from .checks import (
    ApertureReport,
    ShellOffsetReport,
    StationReport,
    aperture_report,
    feature_scale_ok,
    hom_cutoff_hz,
    is_monotone_radius,
    meridian_self_intersects,
    shell_offset_report,
    station_report,
)

__all__ = [
    # core
    "ICWCurve",
    "ICWSample",
    "kappa_spline",
    "clamped_uniform_knots",
    "DEFAULT_DEGREE",
    "DEFAULT_SAMPLES",
    # solver
    "TerminationMode",
    "ICWTargets",
    "FeasibilityReport",
    "solve_icw",
    "n_shape_modes",
    "curve_from_shape_modes",
    # seed
    "fit_from_points",
    "fit_error",
    "seed_from_osse",
    "seed_from_rosse",
    # checks
    "is_monotone_radius",
    "meridian_self_intersects",
    "ShellOffsetReport",
    "shell_offset_report",
    "hom_cutoff_hz",
    "ApertureReport",
    "aperture_report",
    "feature_scale_ok",
    "StationReport",
    "station_report",
]
