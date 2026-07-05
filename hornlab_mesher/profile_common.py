from __future__ import annotations

import ast
import math
import operator
import re
from functools import lru_cache
from typing import Any, Callable, Literal, Mapping


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

_EVAL_NAMES = {
    name: value for name, value in _EVAL_GLOBALS.items() if name != "__builtins__"
}
_EVAL_FUNCTIONS = {
    name: value for name, value in _EVAL_NAMES.items() if callable(value)
}
_EVAL_CONSTANTS = {
    name: value for name, value in _EVAL_NAMES.items() if not callable(value)
}
_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_COMPARE_OPS: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


class _ParamExpressionValidator(ast.NodeVisitor):
    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric literals are allowed")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if type(node.op) not in _BINARY_OPS:
            raise ValueError(f"unsupported operator {type(node.op).__name__}")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if type(node.op) not in _UNARY_OPS:
            raise ValueError(f"unsupported unary operator {type(node.op).__name__}")
        self.visit(node.operand)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise ValueError(f"unsupported boolean operator {type(node.op).__name__}")
        for value in node.values:
            self.visit(value)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_Compare(self, node: ast.Compare) -> None:
        self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            if type(op) not in _COMPARE_OPS:
                raise ValueError(f"unsupported comparison {type(op).__name__}")
            self.visit(comparator)

    def visit_Call(self, node: ast.Call) -> None:
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        if not isinstance(node.func, ast.Name):
            raise ValueError("only direct whitelisted function calls are allowed")
        name = node.func.id
        if "__" in name or name not in _EVAL_FUNCTIONS:
            raise ValueError(f"unknown function {name!r}")
        for arg in node.args:
            self.visit(arg)

    def visit_Name(self, node: ast.Name) -> None:
        name = node.id
        if "__" in name:
            raise ValueError("dunder names are not allowed")
        if name == "p" or name in _EVAL_NAMES:
            return
        raise ValueError(f"unknown name {name!r}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        raise ValueError("attribute access is not allowed")

    def generic_visit(self, node: ast.AST) -> None:
        raise ValueError(f"unsupported expression node {type(node).__name__}")


_PARAM_EXPRESSION_VALIDATOR = _ParamExpressionValidator()


def _eval_ast(node: ast.AST, p: float) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, p)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "p":
            return p
        if node.id in _EVAL_CONSTANTS:
            return _EVAL_CONSTANTS[node.id]
        raise ValueError(f"name {node.id!r} is not a numeric value")
    if isinstance(node, ast.BinOp):
        return _BINARY_OPS[type(node.op)](_eval_ast(node.left, p), _eval_ast(node.right, p))
    if isinstance(node, ast.UnaryOp):
        return _UNARY_OPS[type(node.op)](_eval_ast(node.operand, p))
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result = _eval_ast(node.values[0], p)
            for value in node.values[1:]:
                if not result:
                    return result
                result = _eval_ast(value, p)
            return result
        result = _eval_ast(node.values[0], p)
        for value in node.values[1:]:
            if result:
                return result
            result = _eval_ast(value, p)
        return result
    if isinstance(node, ast.IfExp):
        branch = node.body if _eval_ast(node.test, p) else node.orelse
        return _eval_ast(branch, p)
    if isinstance(node, ast.Compare):
        left = _eval_ast(node.left, p)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_ast(comparator, p)
            if not _COMPARE_OPS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("only direct whitelisted function calls are allowed")
        function = _EVAL_FUNCTIONS[node.func.id]
        return function(*(_eval_ast(arg, p) for arg in node.args))
    raise ValueError(f"unsupported expression node {type(node).__name__}")


@lru_cache(maxsize=16384)
def _eval_text_param(text: str, p: float) -> float:
    try:
        return float(text)
    except ValueError:
        pass
    expr = text.replace("^", "**")
    try:
        if "__" in expr:
            raise ValueError("dunder names are not allowed")
        tree = ast.parse(expr, mode="eval")
        _PARAM_EXPRESSION_VALIDATOR.visit(tree)
        return float(_eval_ast(tree, p))
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


def _parse_number_list(
    value: Any,
    *,
    separators: str = ",",
    flatten: bool = False,
    allow_scalar: bool = False,
    finite_only: bool = False,
    invalid: Literal["raise", "skip", "empty"] = "raise",
    evaluate: bool = True,
) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        if not allow_scalar:
            return []
        number = float(value)
        return [number] if not finite_only or math.isfinite(number) else []
    if isinstance(value, str):
        text = value
        primary = separators[0] if separators else ","
        for separator in separators[1:]:
            text = text.replace(separator, primary)
        parts = [part.strip() for part in text.split(primary)]
    else:
        try:
            parts = list(value)
        except TypeError:
            return []
    out: list[float] = []
    for part in parts:
        if flatten and isinstance(part, (list, tuple)):
            out.extend(
                _parse_number_list(
                    part,
                    separators=separators,
                    flatten=True,
                    allow_scalar=True,
                    finite_only=finite_only,
                    invalid=invalid,
                    evaluate=evaluate,
                )
            )
            continue
        if part == "":
            continue
        try:
            number = float(eval_param(part, 0.0, 0.0) if evaluate else float(part))
        except (TypeError, ValueError):
            if invalid == "empty":
                return []
            if invalid == "skip":
                continue
            raise
        if finite_only and not math.isfinite(number):
            continue
        out.append(number)
    return out



def _normalise_formula(value: Any) -> str:
    raw = str(value or "OSSE").strip().upper().replace("_", "-")
    if raw == "ROSSE":
        raw = "R-OSSE"
    if raw not in {"OSSE", "R-OSSE", "LOOKUP", "ICW"}:
        raise ValueError(
            f"formula must be OSSE, R-OSSE/ROSSE, LOOKUP, or ICW, got {value!r}"
        )
    return raw


_QUADRANTS_LEADING_INT_RE = re.compile(r"[+-]?\d+")


def _quadrants_leading_int(value: Any) -> int:
    """Read Mesh.Quadrants the way Ath does: as a leading integer.

    Ath parses the value with C ``atoi`` / Pascal ``Val`` semantics -- skip leading
    whitespace, take an optional sign and the run of digits that follows, and stop at
    the first non-digit (``0`` when there is no leading digit). Trailing junk is
    ignored, so ``"1234x" -> 1234`` and ``"1,2" -> 1``, while ``"x1234"``, ``""`` and
    ``None`` all read as ``0``. Crucially Ath does NOT reorder digits, so ``"21" ->
    21`` (a quarter model), never the set ``{1, 2}``. Verified against ath.exe under
    Wine over the full 1..1234 value sweep.
    """
    if value is None or isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    match = _QUADRANTS_LEADING_INT_RE.match(str(value).strip())
    return int(match.group(0)) if match else 0


def _normalise_quadrants(value: Any) -> str:
    """Normalise Mesh.Quadrants to Ath's canonical coverage.

    Ath recognises exactly three integer values -- ``1234`` (full model, no
    symmetry), ``12`` (top half, xz mirror) and ``14`` (right half, yz mirror); every
    other value, including the missing/empty/garbage default of ``0``, is a
    quadrant-1 quarter model (both mirrors). We reproduce that mapping verbatim,
    including Ath's atoi parsing quirks (see :func:`_quadrants_leading_int`), so a
    mesher build and a real Ath run agree for every Mesh.Quadrants string. Unlike the
    earlier set-based logic this never raises and never reorders digits -- Ath itself
    silently treats unrecognised values as the quarter default.
    """
    n = _quadrants_leading_int(value)
    if n == 1234:
        return "1234"
    if n == 12:
        return "12"
    if n == 14:
        return "14"
    return "1"


def _symmetry_planes_for_quadrants(value: Any) -> tuple[str, ...]:
    return {
        "1": ("x", "y"),
        "12": ("y",),
        "14": ("x",),
    }.get(_normalise_quadrants(value), ())


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
