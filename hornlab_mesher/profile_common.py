from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Mapping


_DEFAULTS = {
    "k": 1.0,
    "n": 4.0,
    "q": 0.995,
    "m": 0.85,
    "r": 0.4,
    "b": 0.2,
}

_EVAL_GLOBALS = {
    "__builtins__": {},
    "abs": abs,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "pi": math.pi,
    "e": math.e,
}

@lru_cache(maxsize=16384)
def _eval_text_param(text: str, p: float) -> float:
    try:
        return float(text)
    except ValueError:
        pass
    expr = text.replace("^", "**")
    try:
        return float(eval(expr, _EVAL_GLOBALS, {"p": p}))
    except Exception as exc:
        raise ValueError(f"invalid parameter expression {text!r}") from exc


def eval_param(value: Any, p: float = 0.0, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float(default)
    return _eval_text_param(text, float(p))


def _deg(value: Any, p: float = 0.0, default: float = 0.0) -> float:
    return math.radians(eval_param(value, p, default))


def _osse_radius(z: float, p: float, params: Mapping[str, Any], *, r0: float, a_deg: float, a0_deg: float) -> float:
    L = eval_param(params.get("L"), p, 120.0)
    k = eval_param(params.get("k"), p, _DEFAULTS["k"])
    n = eval_param(params.get("n"), p, _DEFAULTS["n"])
    q = eval_param(params.get("q"), p, _DEFAULTS["q"])
    s = eval_param(params.get("s"), p, 0.0)
    a = math.radians(a_deg)
    a0 = math.radians(a0_deg)

    base = math.sqrt((k * r0) ** 2 + 2 * k * r0 * z * math.tan(a0) + (z**2) * (math.tan(a) ** 2))
    base += r0 * (1 - k)
    if z <= 0 or n <= 0 or q <= 0 or L <= 0:
        return base
    z_norm = q * z / L
    if z_norm > 1.0:
        term = s * L / q
    else:
        term = (s * L / q) * (1 - (1 - z_norm**n) ** (1 / n))
    return base + term


def _parse_number_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        try:
            parts = list(value)
        except TypeError:
            return []
    out: list[float] = []
    for part in parts:
        if part == "":
            continue
        out.append(float(eval_param(part, 0.0, 0.0)))
    return out



def _normalise_formula(value: Any) -> str:
    raw = str(value or "OSSE").strip().upper().replace("_", "-")
    if raw == "ROSSE":
        raw = "R-OSSE"
    if raw not in {"OSSE", "R-OSSE"}:
        raise ValueError(f"formula must be OSSE or R-OSSE/ROSSE, got {value!r}")
    return raw


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
