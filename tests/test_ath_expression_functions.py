"""Scalar ATH functions and independent V2026-08c expression probes."""
from pathlib import Path
import math

import numpy as np
import pytest

from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.config_parser import ConfigError, parse_text_config
from hornlab_mesher.profile_common import eval_param, _fma_fallback
from hornlab_mesher.profile_formulas import calculate_osse_curve

FIXTURES = Path(__file__).parent / "fixtures/ath_import/ath-v2026-08c"


@pytest.mark.parametrize("name", ["sf1", "sf2", "sf3"])
def test_expression_matches_ath_at_every_exported_station(name):
    params = build_geometry_params(parse_text_config((FIXTURES / f"{name}.cfg").read_text()))[0]
    for block in (FIXTURES / f"{name}.csv").read_text().strip().split("\n\n"):
        points = np.array([[float(v) for v in row.split(";")] for row in block.splitlines()])
        phi = math.atan2(points[-1, 1], points[-1, 0]) % math.tau
        radius = calculate_osse_curve(points[:, 2], phi, params)[1]
        # ATH writes six decimal places, so the recovered azimuth also has
        # rounding error. This checks the function independently of the z-map.
        np.testing.assert_allclose(radius, np.hypot(points[:, 0], points[:, 1]), rtol=0, atol=2e-4)


@pytest.mark.parametrize("m,n1,n2,n3,a,b,p", [
    (5, 3, 7, 2, 1.3, .8, .73), (3, 1.5, 4, 9, .7, 1.2, 2.4),
    (4, 12, 15, 16, 1, .618, 0),
])
def test_sf_argument_order_and_semantics(m, n1, n2, n3, a, b, p):
    expected = (abs(math.cos(m*p/4)/a)**n2 + abs(math.sin(m*p/4)/b)**n3)**(-1/n1)
    assert eval_param(f"sf({m},{n1},{n2},{n3},{a},{b},p)", p) == pytest.approx(expected)


@pytest.mark.parametrize("x,expected", [
    (2.5, 3), (-2.5, -3), (.5, 1), (-.5, -1),
    (math.nextafter(.5, 0), 0), (2.49, 2), (-2.49, -2),
])
def test_round_is_half_away_from_zero(x, expected):
    assert eval_param(f"round({x!r})") == expected


def test_log_is_natural_log():
    assert eval_param("log(10)") == math.log(10)
    assert eval_param("log(exp(2))") == pytest.approx(2)


@pytest.mark.parametrize("expression,expected", [
    ("acosh(1)", 0), ("asinh(0)", 0), ("atanh(0)", 0),
    ("cbrt(-8)", -2), ("ceil(1.2)", 2), ("copysign(2,-1)", -2),
    ("cosh(0)", 1), ("deg(pi)", 180), ("erf(0)", 0), ("erfc(0)", 1),
    ("exp(0)", 1), ("exp2(3)", 8), ("expm1(0)", 0),
    ("fdim(1,2)", 0), ("fdim(3,2)", 1), ("floor(-1.2)", -2),
    ("fma(2,3,4)", 10), ("fmod(-5,2)", -1), ("gamma(5)", 24),
    ("hypot(3,4)", 5), ("ldexp(3,2)", 12), ("lgamma(1)", 0),
    ("log10(100)", 2), ("log1p(0)", 0), ("log2(8)", 3),
    ("logb(8)", 3), ("logb(-.125)", -3), ("nearbyint(2.5)", 2),
    ("nextafter(1,2)", math.nextafter(1, 2)), ("rad(180)", math.pi),
    ("remainder(7,2)", -1), ("scalbn(3,-1)", 1.5),
    ("sinh(0)", 0), ("tanh(0)", 0), ("trunc(-1.7)", -1),
])
def test_scalar_c99_functions(expression, expected):
    assert eval_param(expression) == pytest.approx(expected)


def test_fma_has_single_rounding_on_older_python():
    x, y, z = 1 + 2**-27, 1 - 2**-27, -1.0
    assert _fma_fallback(x, y, z) == -2**-54
    assert eval_param(f"fma({x!r},{y!r},{z!r})") == -2**-54


@pytest.mark.parametrize("name", ["frexp", "modf"])
def test_two_value_functions_refused_by_name(name):
    with pytest.raises(ValueError, match=rf"{name}.*two values"):
        eval_param(f"{name}(2)")
    with pytest.raises(ConfigError, match=rf"{name}.*two values"):
        parse_text_config(f"OSSE = {{\nL = 160\na = 30 + {name}(2)\n}}")


def test_ignored_solver_expressions_remain_ignored():
    parse_text_config("OSSE = {\nL = 160\n}\nABEC.Foo = frexp(2)\n")


@pytest.mark.parametrize("expression", ["sf.__call__(1)", "math.log(10)", "__import__('os')", "log(x=10)"])
def test_new_functions_do_not_expand_expression_syntax(expression):
    with pytest.raises(ValueError):
        eval_param(expression)


@pytest.mark.parametrize('name', ['ldexp', 'scalbn'])
@pytest.mark.parametrize('exponent,expected', [(2.9, 12), (-1.9, 1.5), (-.9, 3)])
def test_binary_exponent_conversion_truncates_toward_zero(name, exponent, expected):
    assert eval_param(f'{name}(3,{exponent})') == expected


@pytest.mark.parametrize('name', ['logb', 'round'])
@pytest.mark.parametrize('value', [math.inf, -math.inf])
def test_nonfinite_scalar_function_limits(name, value):
    # Pass an infinite value through p; expression constants remain bounded.
    assert eval_param(f'{name}(p)', value) == (abs(value) if name == 'logb' else value)
