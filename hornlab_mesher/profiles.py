"""Profile and point-grid facade.

The implementation is split into formulas, morphing, and sampling modules, but
callers import through this facade so the compatibility surface stays stable.
Some underscored helpers remain importable for contract tests that pin imported
ATH/WG behavior, but they are intentionally not advertised through `__all__`.
"""

from __future__ import annotations

from .profile_common import eval_param
from .profile_formulas import (
    build_icw_curve,
    calculate_osse,
    calculate_rosse,
    icw_meridian_points,
    osse_total_length,
    profile_points,
    rosse_total_length,
)
from .profile_morph import (
    _apply_morphing,
    _guiding_curve_target_radius,
    _morph_target_radius_at_angle,
    _rounded_rect_radius,
)
from .profile_sampling import (
    _angle_list,
    build_point_grid,
)

__all__ = [
    "build_icw_curve",
    "build_point_grid",
    "calculate_osse",
    "calculate_rosse",
    "eval_param",
    "icw_meridian_points",
    "osse_total_length",
    "profile_points",
    "rosse_total_length",
]
