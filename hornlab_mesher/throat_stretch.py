"""Shared stretch bounds and canonical identity of an inactive axial map."""

import ast
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


def parameter_is_zero(value: Any) -> bool:
    """Recognize numeric zero and algebraically zero ATH expressions.

    Never infer an all-azimuth identity from samples. Simplification preserves
    identical subexpressions, zero products and constant arithmetic, including
    ATH's common ``0*p`` and ``sin(p)^2-sin(p)^2`` spellings. Unknown forms are
    conservatively active.
    """
    if isinstance(value, Real):
        return value == 0
    if not isinstance(value, str):
        return False

    def simplify(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp):
            operand = simplify(node.operand)
            if isinstance(operand, (int, float)):
                if isinstance(node.op, ast.USub):
                    return -operand
                if isinstance(node.op, ast.UAdd):
                    return operand
        if isinstance(node, ast.BinOp):
            left, right = simplify(node.left), simplify(node.right)
            if isinstance(node.op, ast.Sub) and left == right:
                return 0
            if isinstance(node.op, ast.Mult) and (left == 0 or right == 0):
                return 0
            if isinstance(node.op, (ast.Add, ast.Sub)) and right == 0:
                return left
            if isinstance(node.op, ast.Add) and left == 0:
                return right
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.Pow) and abs(right) <= 16:
                    return left ** right
            return (type(node.op).__name__, left, right)
        return ast.dump(node)

    try:
        return simplify(ast.parse(value.replace("^", "**"), mode="eval").body) == 0
    except (SyntaxError, ArithmeticError, ValueError):
        return False


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
            sections = [config]
            if isinstance(config, Mapping):
                sections += [config.get(k, {}) for k in ("profile", "parameters")]
            if any(isinstance(s, Mapping) and any(k in s for k in ("s1", "s2")) for s in sections):
                raise ConfigError(f"throat stretch geometry is invalid: {exc}") from exc
            raise
    return wrapped


def canonical_stretch_params(params: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(params)
    if stretch_is_inactive(result):
        result.pop("s1", None)
        result.pop("s2", None)
    return result
