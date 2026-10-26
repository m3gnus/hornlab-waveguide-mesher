"""Coverage singularities are refused by the shared design resolver."""
from __future__ import annotations

import re

import numpy as np
import pytest

from hornlab_mesher import config_builder as builder
from hornlab_mesher.config_parser import ConfigError, parse_text_config
from hornlab_mesher.preview.api import build_preview_geometry
from hornlab_mesher.profiles import build_point_grid


EXPRESSION = "22 * (abs(cos(p)/1.2)^8 + abs(sin(p)/16)^1)^(-1/2)"


def config(angle=EXPRESSION, *, formula="OSSE", throat=0):
    return {
        "formula": formula, "mode": "bare",
        "profile": {**({"R": 12} if formula == "R-OSSE" else {"L": 0.2, "s": 0}),
                    "r0": 0.1, "a": angle, "a0": throat},
        "source": {"source_shape": 0},
        "mesh": {"angular_segments": 32, "length_segments": 8,
                 "throat_res_mm": 1, "mouth_res_mm": 3},
    }


@pytest.mark.parametrize("formula", ["OSSE", "R-OSSE"])
@pytest.mark.parametrize("entry", ["params", "preview", "plan", "mesh"])
def test_problem_expression_refused_before_any_grid_or_mesh(monkeypatch, tmp_path, formula, entry):
    def forbidden(*args, **kwargs):
        pytest.fail("an invalid design reached a grid or mesh builder")

    # Instrument the actual downstream entry points, including preview's own
    # imported grid aliases, rather than relying only on the error text.
    import hornlab_mesher.preview.api as preview
    import hornlab_mesher.profile_sampling as sampling
    import hornlab_mesher.profiles as profiles
    for module in (builder, preview, sampling, profiles):
        for name in ("build_point_grid", "build_point_grid_arrays", "build_mesh_with_info"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    design = config(formula=formula)
    entries = {
        "params": lambda: builder.build_geometry_params(design),
        "preview": lambda: build_preview_geometry(design),
        "plan": lambda: builder.resolve_geometry(design),
        "mesh": lambda: builder.build_from_config(design, tmp_path / "invalid.msh"),
    }
    with pytest.raises(ConfigError) as error:
        entries[entry]()
    message = str(error.value)
    assert "Mouth coverage angle (a) reaches 94.8°" in message
    assert f"{formula} needs it below 90° everywhere" in message
    assert "Change the expression so it stays below 90°" in message
    p = float(re.search(r"near p = ([\d.]+)°", message).group(1))
    assert min(abs(p - peak) for peak in (54, 126, 234, 306)) < 1
    assert "resolution" not in message


@pytest.mark.parametrize("angle", [0, -1, 90, 100, "90", "45 + 45*sin(p)^2", "45*cos(p)"])
@pytest.mark.parametrize("formula", ["OSSE", "R-OSSE"])
def test_mouth_bounds_are_strict(angle, formula):
    with pytest.raises(ConfigError, match=r"Mouth coverage angle \(a\).*Change the expression"):
        builder.build_geometry_params(config(angle, formula=formula))


@pytest.mark.parametrize("throat", [-90, 90, 100, "89 + 2*sin(p)^2", "-89 - 2*sin(p)^2"])
@pytest.mark.parametrize("formula", ["OSSE", "R-OSSE"])
def test_throat_tangent_singularities_are_refused(throat, formula):
    with pytest.raises(ConfigError, match=r"Throat angle \(a0\).*above -90° and below 90°"):
        builder.build_geometry_params(config(45, formula=formula, throat=throat))


@pytest.mark.parametrize("centre", [54.123, 234.123, 359.973])
@pytest.mark.parametrize("sign,key,baseline", [(1, "a", 89), (-1, "a", 1), (1, "a0", 89)])
def test_refinement_catches_narrow_extremum_between_samples(centre, sign, key, baseline):
    # Every 0.01-degree sample stays inside the bounds, but the true peak
    # crosses them. Include Q3 and the seam, with a reduced mesh domain.
    design = config(45)
    design["mesh"]["quadrants"] = "1"
    design["profile"][key] = f"{baseline} + {sign}*1.1*exp(-((deg(p)-{centre})/.002)^2)"
    with pytest.raises(ConfigError, match=rf"\({key}\) reaches .*near p = {centre:.1f}°"):
        builder.build_geometry_params(design)


@pytest.mark.parametrize("formula", ["OSSE", "R-OSSE"])
def test_peak_invisible_at_tenth_degree_spacing_is_refused(formula):
    # At 54.0 and 54.1 degrees this rounds to 89; the actual peak is 91.
    with pytest.raises(ConfigError, match=r"reaches 91.0° near p = 54.0°"):
        builder.build_geometry_params(config("89 + 2*exp(-((deg(p)-54.05)/.005)^2)", formula=formula))


@pytest.mark.parametrize("key,expression", [
    ("a", "90 - abs(sin(p-.123))"),
    ("a", "89 + exp(-((deg(p)-54.123)/.02)^2)"),
    ("a", "abs(sin(p-.123))"),
    ("a0", "90 - abs(sin(p-.123))"),
    ("a0", "-90 + abs(sin(p-.123))"),
])
@pytest.mark.parametrize("formula", ["OSSE", "R-OSSE"])
def test_off_grid_contact_with_strict_boundary_is_refused(key, expression, formula):
    design = config(45, formula=formula)
    design["profile"][key] = expression
    with pytest.raises(ConfigError, match=rf"\({key}\) reaches"):
        builder.build_geometry_params(design)


@pytest.mark.parametrize("angle", [89.9999999999, "89.9999999999 - abs(sin(p-.123))"])
def test_angle_strictly_below_ninety_is_not_rounded_up(angle):
    assert builder.build_geometry_params(config(angle))[0]["a"] == angle


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "1/(1-cos(p))", "sqrt(-1)"])
def test_invalid_angle_expression_uses_config_error(value):
    with pytest.raises(ConfigError, match=r"Mouth coverage angle \(a\)"):
        builder.build_geometry_params(config(value))


@pytest.mark.parametrize("formula", ["OSSE", "R-OSSE"])
def test_near_ninety_expression_still_builds_grid(formula):
    design = config("88 + sin(p-.123)^2", formula=formula)
    params, _, _ = builder.build_geometry_params(design)
    assert params["a"] == design["profile"]["a"]
    grid = build_point_grid(params)
    assert np.isfinite(grid["inner_points"]).all()


def test_near_ninety_expression_still_builds_mesh(tmp_path):
    result = builder.build_from_config(config("88 + sin(p-.123)^2"), tmp_path / "valid.msh")
    assert result.mesh_path.is_file()
    assert result.n_triangles > 0


@pytest.mark.parametrize("formula", ["OSSE", "R-OSSE"])
@pytest.mark.parametrize("throat", [0, -5, 15.5])
def test_constant_design_grid_is_identical_with_validation(monkeypatch, formula, throat):
    design = config(45, formula=formula, throat=throat)
    checked, _, _ = builder.build_geometry_params(design)
    monkeypatch.setattr(builder, "_validate_coverage_angles", lambda *args: None)
    unchecked, _, _ = builder.build_geometry_params(design)
    assert checked == unchecked
    actual, expected = build_point_grid(checked), build_point_grid(unchecked)
    for key in ("inner_points", "outer_points"):
        np.testing.assert_array_equal(actual[key], expected[key])


def test_text_import_aliases_reach_the_same_check():
    design = parse_text_config(f"Length = 20\nCoverage.Angle = {EXPRESSION}\n")
    with pytest.raises(ConfigError, match=r"Mouth coverage angle \(a\) reaches 94.8°"):
        builder.build_geometry_params(design)
