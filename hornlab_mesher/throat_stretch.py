"""Shared stretch bounds and canonical identity of an inactive axial map."""

import math
from functools import wraps
from numbers import Real
from typing import Any, Mapping


# At most 900,000 mm of displacement, far below half an ulp at the largest
# finite float. Thus adding the displacement to any finite x stays finite.
# The atan argument may saturate to infinity; atan still has a finite limit.
STRETCH_COEFFICIENT_MAX = 10_000.0


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
            raise ConfigError(f"throat stretch {key} must be finite and >= 0 and <= 10000") from exc
        if not math.isfinite(number) or not 0 <= number <= STRETCH_COEFFICIENT_MAX:
            raise ConfigError(
                f"throat stretch {key} must be finite and >= 0 and <= {STRETCH_COEFFICIENT_MAX:g}, got {value!r}"
            )
        values.append(number)
    return values[0], values[1]


def stretch_is_inactive(params: Mapping[str, Any]) -> bool:
    s1, s2 = stretch_coefficients(params)
    return s1 == 0.0 or s2 == 0.0


# Quarter-degree sampling is independent of mesh density and includes both
# endpoints and every quadrant boundary. The absolute tolerance absorbs roundoff
# in identities such as sin(p)^2 + cos(p)^2 - 1, far below geometric tolerances.
ZERO_PARAMETER_SAMPLES = 1440
ZERO_PARAMETER_ATOL = 1e-12


def parameter_is_zero(value: Any) -> bool:
    """Evaluate inactivity over a fixed full-circle sample, in parameter units.

    This is a sampled decision, not a symbolic proof. Invalid/nonfinite values
    and missing/empty expression inputs are never classified as zero.
    """
    from .profile_common import eval_param

    if isinstance(value, Real):
        return abs(value) <= ZERO_PARAMETER_ATOL and math.isfinite(value)
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        for i in range(ZERO_PARAMETER_SAMPLES + 1):
            result = eval_param(value, i * (2 * math.pi / ZERO_PARAMETER_SAMPLES))
            if not math.isfinite(result) or abs(result) > ZERO_PARAMETER_ATOL:
                return False
        return True
    except (ValueError, ArithmeticError, TypeError):
        return False


def stretch_input_sections(config: Mapping[str, Any]):
    """Scan supplied native entries even when section precedence ignores them."""
    for section in (config, config.get("profile", {}), config.get("parameters", {})):
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
    prefix = any(not parameter_is_zero(params.get(k, 0)) for k in ("throatExtLength", "slotLength"))
    rotation = not parameter_is_zero(params.get("rot", 0))
    guide = all(not parameter_is_zero(params.get(k, 0)) for k in ("gcurveType", "gcurveWidth"))
    if rotation and (formula == "R-OSSE" or prefix):
        raise ConfigError("throat stretch with Rot and a prefix (or R-OSSE Rot) is unverified and not supported")
    if formula == "OSSE" and not parameter_is_zero(params.get("slotLength", 0)):
        raise ConfigError("throat stretch with OSSE Slot.Length is unverified and not supported")
    if guide and (prefix or rotation):
        raise ConfigError("throat stretch with GCurve and a prefix or Rot is unverified and not supported")
    if formula == "R-OSSE" and length_supplied:
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
            if isinstance(config, Mapping) and any(
                any(k in section for k in ("s1", "s2"))
                for section in stretch_input_sections(config)
            ):
                raise ConfigError(f"throat stretch geometry is invalid: {exc}") from exc
            raise
    return wrapped


def canonical_stretch_params(params: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(params)
    if stretch_is_inactive(result):
        result.pop("s1", None)
        result.pop("s2", None)
    return result
