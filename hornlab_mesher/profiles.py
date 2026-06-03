"""Profile and point-grid facade.

The implementation is split into formulas, morphing, and sampling modules, but
callers import through this facade so the compatibility surface stays stable.
Some underscored helpers remain importable for contract tests that pin imported
ATH/WG behavior, but they are intentionally not advertised through `__all__`.
"""

from __future__ import annotations

from .profile_common import (
    _DEFAULTS,
    _deg,
    _is_true,
    _normalise_formula,
    _osse_radius,
    _parse_number_list,
    eval_param,
)
from .profile_formulas import calculate_osse, calculate_rosse, osse_total_length, profile_points, rosse_total_length
from .profile_morph import (
    _apply_morphing,
    _circle_morph_target_radius,
    _configured_morph_half_dimension,
    _coverage_angle_from_guiding_curve,
    _guiding_curve_active,
    _guiding_curve_target_radius,
    _guiding_curve_type,
    _invert_osse_coverage_angle,
    _morph_active,
    _morph_factor,
    _morph_target_radius_at_angle,
    _morph_target_shape,
    _rounded_rect_quadrant_angles,
    _rounded_rect_radius,
)
from .profile_sampling import (
    _ATH_T_20,
    _ATH_T_9,
    _angle_list,
    _ath_default_zmap,
    _axial_sample_map,
    _cross_section,
    _custom_zmap,
    _fill_missing_normals,
    _horn_indices,
    _mirror_quadrant_angles,
    _morph_angle_list,
    _normalise3,
    _normalise_ath_angular_segments,
    _normalise_sampling_mode,
    _outer_offset_shell,
    _superellipse_scale,
    _zmap_number_list,
    build_point_grid,
)

__all__ = [
    "build_point_grid",
    "calculate_osse",
    "calculate_rosse",
    "eval_param",
    "osse_total_length",
    "profile_points",
    "rosse_total_length",
]
