"""Shared stretch bounds and canonical identity of an inactive axial map."""

import math
from functools import wraps
from numbers import Real
from typing import Any, Mapping


# At most 900 mm of displacement, far below half an ulp at the largest
# finite float. Thus adding the displacement to any finite x stays finite.
# The atan argument may saturate to infinity; atan still has a finite limit.
STRETCH_COEFFICIENT_MAX = 10.0


def stretch_coefficients(params: Mapping[str, Any]) -> tuple[float, float]:
    """Validate both inputs before canonicalizing even a dormant pair.

    No coefficient expressions are evaluated in this release. Importers must
    preserve the distinction between a numeric token and an expression token.
    """
    from .config_parser import ConfigError

    values = []
    for key in ("s1", "s2"):
        value = params.get(key, 0.0)
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ConfigError(
                f"per-azimuth throat stretch is not supported yet; {key} must be a plain finite number"
            )
        try:
            number = float(value)
        except (ValueError, OverflowError) as exc:
            raise ConfigError(
                f"throat stretch {key} must be finite and >= 0 and <= {STRETCH_COEFFICIENT_MAX:g}"
            ) from exc
        if math.isfinite(number) and number < 0:
            raise ConfigError(
                f"throat stretch {key} must not be negative, got {value!r}: negative values "
                "are not supported (ATH accepts them, but they fold the profile back "
                "through the throat for all but very small magnitudes)"
            )
        if not math.isfinite(number) or not 0 <= number <= STRETCH_COEFFICIENT_MAX:
            raise ConfigError(
                f"throat stretch {key} must be finite and >= 0 and <= {STRETCH_COEFFICIENT_MAX:g}, got {value!r}"
            )
        values.append(number)
    return values[0], values[1]


def stretch_is_inactive(params: Mapping[str, Any]) -> bool:
    s1, s2 = stretch_coefficients(params)
    return s1 == 0.0 or s2 == 0.0


# Canonical composition inputs and their native aliases. These also drive the
# early raw-input check, before normalization can coerce numeric strings.
COMPOSITION_PROFILE_KEYS = {
    "slotLength": ("Slot.Length", ("slot_length_mm", "slotLength")),
    "rot": ("Rot", ("rot_deg", "rot")),
    "throatExtLength": ("Throat.Ext.Length", ("throat_ext_length_mm", "throatExtLength")),
    "throatExtAngle": ("Throat.Ext.Angle", ("throat_ext_angle_deg", "throatExtAngle")),
}
COMPOSITION_GUIDE_KEYS = {
    "gcurveType": ("GCurve.Type", ("gcurve_type", "gcurveType")),
    "gcurveWidth": ("GCurve.Width", ("gcurve_width_mm", "gcurveWidth")),
    "gcurveAspectRatio": ("GCurve.AspectRatio", ("gcurve_aspect_ratio", "gcurveAspectRatio")),
    "gcurveDist": ("GCurve.Dist", ("gcurve_dist", "gcurveDist")),
    "gcurveRot": ("GCurve.Rot", ("gcurve_rot_deg", "gcurveRot")),
    "gcurveSF": ("GCurve.SF", ("gcurve_sf", "gcurveSf", "gcurveSF")),
    "gcurveSf": ("GCurve.SF", ("gcurve_sf", "gcurveSf", "gcurveSF")),
    "gcurveSeN": ("GCurve.SE.n", ("gcurve_se_n", "gcurveSeN")),
    "gcurveSfA": ("GCurve.SF.a", ("gcurve_sf_a", "gcurveSfA")),
    "gcurveSfB": ("GCurve.SF.b", ("gcurve_sf_b", "gcurveSfB")),
    "gcurveSfM1": ("GCurve.SF.m1", ("gcurve_sf_m1", "gcurveSfM1")),
    "gcurveSfM2": ("GCurve.SF.m2", ("gcurve_sf_m2", "gcurveSfM2")),
    "gcurveSfN1": ("GCurve.SF.n1", ("gcurve_sf_n1", "gcurveSfN1")),
    "gcurveSfN2": ("GCurve.SF.n2", ("gcurve_sf_n2", "gcurveSfN2")),
    "gcurveSfN3": ("GCurve.SF.n3", ("gcurve_sf_n3", "gcurveSfN3")),
}


def stretch_input_sections(config: Mapping[str, Any]):
    """Scan supplied native entries even when section precedence ignores them."""
    for section in (config, *(value for key, value in config.items() if key != "icw_seed")):
        if isinstance(section, Mapping):
            yield section


def validate_supplied_stretch(config: Mapping[str, Any], formula: str) -> None:
    from .config_parser import ConfigError

    for section in stretch_input_sections(config):
        if formula in {"OSSE", "R-OSSE"}:
            stretch_coefficients(section)
        elif any(key in section for key in ("s1", "s2")):
            if formula == "ICW":
                message = "OSSE/R-OSSE shape keys are not valid with formula ICW"
            else:
                message = f"formula {formula} does not accept OSSE/R-OSSE profile coefficient keys"
            raise ConfigError(message)


def validate_stretch_composition(params: Mapping[str, Any], formula: str,
                                 *, length_supplied: bool = False) -> None:
    """One refusal boundary for normalized native and imported configurations."""
    from .config_parser import ConfigError

    if stretch_is_inactive(params):
        return
    for key, (name, _) in {**COMPOSITION_PROFILE_KEYS, **COMPOSITION_GUIDE_KEYS}.items():
        if key not in params:
            continue
        value = params[key]
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ConfigError(f"throat stretch does not support per-azimuth {name} yet; must be a plain finite number")
        try:
            finite = math.isfinite(value)
        except (ValueError, OverflowError):
            finite = False
        if not finite:
            raise ConfigError(f"throat stretch {name} must be a plain finite number")
    prefix = any(params.get(k, 0) != 0 for k in ("throatExtLength", "slotLength"))
    rotation = params.get("rot", 0) != 0
    guide = all(params.get(k, 0) != 0 for k in ("gcurveType", "gcurveWidth"))
    if rotation and (formula == "R-OSSE" or prefix):
        raise ConfigError("throat stretch with Rot and a prefix (or R-OSSE Rot) is unverified and not supported")
    if formula == "OSSE" and params.get("slotLength", 0) != 0:
        raise ConfigError("throat stretch with OSSE Slot.Length is unverified and not supported")
    if guide and (prefix or rotation):
        raise ConfigError("throat stretch with GCurve and a prefix or Rot is unverified and not supported")
    if formula == "R-OSSE" and (length_supplied or any(k in params for k in ("L", "L_mm", "Length"))):
        raise ConfigError("throat stretch with R-OSSE top-level Length is unverified and not supported")


def stretch_config_errors(function):
    """Keep public stretch failures within the ConfigError contract."""
    @wraps(function)
    def wrapped(config, *args, **kwargs):
        from .config_parser import ConfigError
        from .mesher import MesherError

        try:
            return function(config, *args, **kwargs)
        except (ValueError, ArithmeticError, MesherError) as exc:
            if isinstance(exc, ConfigError):
                raise
            if isinstance(config, Mapping):
                from .config_builder import _pick, _section

                profile = _section(config, "profile", "parameters")
                coefficients = {key: _pick(profile, config, names=(key,), default=0.0)
                                for key in ("s1", "s2")}
                if not stretch_is_inactive(coefficients):
                    raise ConfigError(f"throat stretch geometry is invalid: {exc}") from exc
            raise
    return wrapped


def canonical_stretch_params(params: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(params)
    if stretch_is_inactive(result):
        result.pop("s1", None)
        result.pop("s2", None)
    return result
