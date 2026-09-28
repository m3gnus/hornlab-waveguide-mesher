"""Regression tests for the 2026-09-28 geometry review findings.

Where a behaviour is ATH's, the expected numbers come from ath.exe V2025-06
GridExport runs of the configs spelled out here (Scale = 1, profiles file),
so these tests need no reference archive.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from hornlab_mesher import config_builder
from hornlab_mesher.cli import build_geometry_params, load_config
from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.config_parser import ConfigError
from hornlab_mesher.profile_common import eval_param
from hornlab_mesher.profile_formulas import (
    calculate_osse,
    calculate_osse_curve,
    calculate_rosse,
    osse_coverage_inversion,
)
from hornlab_mesher.profile_sampling import build_point_grid_arrays
from hornlab_mesher.profiles import build_point_grid


def _grid(params):
    grid = build_point_grid_arrays(params)
    return grid, grid["inner_grid"]


def _radii(inner):
    return np.hypot(inner[:, :, 0], inner[:, :, 1])


def _ath_text_params(tmp_path, name, text):
    path = tmp_path / f"{name}.cfg"
    path.write_text(text, encoding="utf-8")
    params, formula, _mode = build_geometry_params(load_config(path))
    return params, formula


def _zmap_samples(z_values):
    total = z_values[-1]
    return ",".join(f"{value / total:.12g}" for value in z_values)


# --------------------------------------------------------------------- M5 ---


@pytest.mark.parametrize("expr", ["10**10**7", "pow(10, 10**7)", "9^9^9^9"])
def test_power_in_a_parameter_expression_fails_fast(expr):
    start = time.perf_counter()
    with pytest.raises(ValueError, match="overflows"):
        eval_param(expr, 0.0)
    assert time.perf_counter() - start < 0.5


def test_bounded_power_keeps_ordinary_expressions():
    assert eval_param("2^3", 0.0) == 8.0
    assert eval_param("(-2)^2", 0.0) == 4.0
    assert eval_param("abs(cos(p)/1.8)^3", 0.3) == pytest.approx(
        abs(math.cos(0.3) / 1.8) ** 3, rel=0.0, abs=1.0e-15
    )


# --------------------------------------------------------------------- M1 ---


def _undefined_osse():
    return {
        "type": "OSSE",
        "a": 10.0,
        "a0": -30.0,
        "L": 100.0,
        "r0": 12.7,
        "angularSegments": 16,
        "lengthSegments": 10,
    }


def test_vector_osse_refuses_an_undefined_profile_like_the_scalar_path():
    params = _undefined_osse()
    with pytest.raises(ValueError, match="undefined"):
        calculate_osse(50.0, 0.0, params)
    with pytest.raises(ValueError, match="undefined"):
        calculate_osse_curve(np.linspace(0.0, 100.0, 11), 0.0, params)
    with pytest.raises(ValueError, match="undefined"):
        build_point_grid(params)


def test_resolve_geometry_refuses_non_finite_control_points(monkeypatch):
    real = config_builder.build_point_grid

    def poisoned(params, **kwargs):
        grid = real(params, **kwargs)
        points = list(grid["inner_points"])
        points[4] = float("nan")
        return {**grid, "inner_points": points}

    monkeypatch.setattr(config_builder, "build_point_grid", poisoned)
    config = {
        "formula": "OSSE",
        "profile": {"L": 80.0, "a": 45.0, "a0": 10.0},
        "mesh": {"topology": "legacy", "lengthSegments": 8, "angularSegments": 16},
        "mode": "bare",
    }
    with pytest.raises(ConfigError, match="non-finite"):
        resolve_geometry(config)


# --------------------------------------------------------------------- M6 ---


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"q": 0.0, "s": 0.7}, "q must be > 0"),
        ({"q": -0.5, "s": 0.7}, "q must be > 0"),
        ({"n": 0.0, "s": 0.7}, "n must be > 0"),
        ({"L": 0.0}, "Length must be > 0"),
        ({"L": -20.0}, "Length must be > 0"),
        ({"slotLength": 120.0, "_athLengthMode": "total"}, "Slot.Length"),
        ({"slotLength": 80.0, "_athLengthMode": "total"}, "Slot.Length"),
    ],
)
def test_degenerate_osse_inputs_are_refused(override, match):
    params = {
        "type": "OSSE",
        "L": 80.0,
        "a": 45.0,
        "a0": 10.0,
        "angularSegments": 16,
        "lengthSegments": 8,
        **override,
    }
    with pytest.raises(ValueError, match=match):
        build_point_grid(params)


def test_slot_shorter_than_length_in_total_mode_still_builds():
    grid, inner = _grid(
        {
            "type": "OSSE",
            "L": 80.0,
            "slotLength": 20.0,
            "_athLengthMode": "total",
            "angularSegments": 16,
            "lengthSegments": 8,
        }
    )
    assert float(inner[0, -1, 2]) == pytest.approx(80.0)


def test_negative_morph_rate_is_refused_but_fractional_rates_stay():
    base = {
        "type": "OSSE",
        "L": 80.0,
        "a": 45.0,
        "a0": 10.0,
        "angularSegments": 16,
        "lengthSegments": 8,
        "morphTarget": 2,
        "morphWidth": 300.0,
        "morphHeight": 300.0,
    }
    with pytest.raises(ValueError, match="Morph.Rate"):
        build_point_grid({**base, "morphRate": -1.0})
    # Waveguide Generator documents 0 <= rate < 1 ("0 jumps straight to the target").
    build_point_grid({**base, "morphRate": 0.5})
    build_point_grid({**base, "morphRate": 0.0})


# --------------------------------------------------------------------- H3 ---


def test_config_builds_carry_the_osse_bulge():
    base = {
        "formula": "OSSE",
        "profile": {"L": 100.0, "a": 45.0, "a0": 10.0},
        "mesh": {"lengthSegments": 20, "angularSegments": 16},
    }
    flat, _formula, _mode = build_geometry_params(base)
    bulged, _formula, _mode = build_geometry_params(
        {**base, "profile": {**base["profile"], "h": 10.0}}
    )
    assert bulged["h"] == 10.0
    r_flat = _radii(_grid(flat)[1])
    r_bulged = _radii(_grid(bulged)[1])
    # h * sin(pi t): zero at both ends, the full 10 mm halfway along.
    assert r_bulged[:, 0] == pytest.approx(r_flat[:, 0])
    assert r_bulged[:, -1] == pytest.approx(r_flat[:, -1])
    assert r_bulged[:, 10] - r_flat[:, 10] == pytest.approx(np.full(16, 10.0))


def test_ath_text_os_h_is_honoured(tmp_path):
    params, _formula = _ath_text_params(
        tmp_path, "bulge", "Length = 100\nCoverage.Angle = 45\nOS.h = 10\n"
    )
    assert params["h"] == 10


# --------------------------------------------------------------------- M2 ---


def _gcurve_params(**override):
    return {
        "type": "OSSE",
        "L": 100.0,
        "a0": 10.0,
        "r0": 12.7,
        "s": 0.5,
        "angularSegments": 16,
        "lengthSegments": 20,
        "gcurveType": 1,
        "gcurveWidth": 300.0,
        "gcurveAspectRatio": 0.5,
        "gcurveDist": 0.5,
        "gcurveSeN": 3.0,
        **override,
    }


def test_guiding_curve_is_met_with_an_osse_bulge():
    params = _gcurve_params(h=8.0)
    inversion = osse_coverage_inversion(params, 0.0)
    assert inversion.saturated is None
    assert inversion.target_radius == pytest.approx(150.0)
    # Ring 10 of a uniform 20-segment grid is the GCurve.Dist = 0.5 station.
    radius = _radii(_grid(params)[1])[0, 10]
    assert radius == pytest.approx(150.0, abs=1.0e-4)


def test_guiding_curve_with_a_non_circular_cross_section_is_refused():
    params = _gcurve_params(profileSystem={"crossSection": {"exponent": 2.0, "aspectRatio": 1.5}})
    with pytest.raises(ValueError, match="guiding curve cannot be combined"):
        build_point_grid(params)
    with pytest.raises(ConfigError, match="guiding curve cannot be combined"):
        build_geometry_params(
            {
                "formula": "OSSE",
                "profile": {"L": 100.0},
                "cross_section": {"aspect_ratio": 1.5},
                "gcurve": {"gcurveType": 1, "gcurveWidth": 300.0},
            }
        )


_ROT_GCURVE_CFG = """\
Throat.Profile = 1
Throat.Diameter = 25.4
Throat.Angle = 10
Length = 100
Coverage.Angle = 50
Term.s = 0.5
Term.n = 4.0
Term.q = 0.995
OS.k = 1.0
Mesh.Quadrants = 1234
Mesh.AngularSegments = 48
Mesh.LengthSegments = 20
Mesh.SubdomainSlices =
GCurve.Type = 1
GCurve.Width = 300
GCurve.AspectRatio = 0.5
GCurve.Dist = 1
GCurve.SE.n = 3
Rot = 10
"""


def test_rot_with_a_guiding_curve_matches_ath(tmp_path):
    # ATH applies Rot after the coverage inversion as well, so the rotated
    # mouth misses the guiding curve (150 mm at phi = 0) in ATH too. Pinned,
    # not "fixed": this is ATH's geometry.
    params, _formula = _ath_text_params(tmp_path, "rot_gcurve", _ROT_GCURVE_CFG)
    grid, inner = _grid(params)
    mouth = _radii(inner)[:, -1]
    assert mouth[[0, 6, 12]] == pytest.approx([165.279, 117.991, 91.418], abs=2.0e-3)
    assert float(inner[0, -1, 2]) == pytest.approx(74.639, abs=2.0e-3)


# --------------------------------------------------------------------- M3 ---

_SUPERFORMULA_ASPECT_CFG = """\
Throat.Profile = 1
Throat.Diameter = 25.4
Throat.Angle = 0
Length = 100
Term.s = 0.4
Term.n = 4.0
Term.q = 0.996
OS.k = 1.0
GCurve.Type = 2
GCurve.Width = 200
GCurve.Dist = 0.5
GCurve.AspectRatio = 0.5
GCurve.SF = 1,1,4,2,2,2
Morph.TargetShape = 0
Mesh.Quadrants = 1234
Mesh.AngularSegments = 48
Mesh.LengthSegments = 10
Mesh.SubdomainSlices =
"""


def test_superformula_guiding_curve_aspect_matches_ath(tmp_path):
    # The type-2 aspect ratio scales the superformula point in x/y and assigns
    # its length to the original azimuth. That is not the polar radius of the
    # scaled curve, but it is exactly what ath.exe does: these are its mouth
    # radii (profiles 0, 4, 6, 12 = 0, 30, 45, 90 deg).
    params, _formula = _ath_text_params(tmp_path, "sf_aspect", _SUPERFORMULA_ASPECT_CFG)
    mouth = _radii(_grid(params)[1])[:, -1]
    assert mouth[[0, 4, 6, 12]] == pytest.approx(
        [223.412952, 203.555005, 181.197831, 122.152555], abs=1.0e-3
    )


# --------------------------------------------------------------------- M7 ---


_LOOKUP_PROFILE = [[0.0, 10.0], [50.0, 40.0], [100.0, 120.0]]


@pytest.mark.parametrize(
    "extra",
    [
        {"throat_ext_length_mm": 20.0},
        {"slot_length_mm": 5.0},
        {"throatExtAngle": 5.0},
        {"driver_throat_diameter_mm": 25.4, "waveguide_throat_diameter_mm": 40.0},
        {"a": 45.0},
        {"k": 2.0},
    ],
)
def test_lookup_refuses_keys_it_would_ignore(extra):
    config = {
        "formula": "LOOKUP",
        "profile": {"lookupProfile": _LOOKUP_PROFILE, **extra},
        "mesh": {"lengthSegments": 8, "angularSegments": 16},
    }
    with pytest.raises(ConfigError, match="formula LOOKUP does not support"):
        build_geometry_params(config)


def test_lookup_keeps_a0_for_the_source_cap_and_refuses_a_grid_level_extension():
    config = {
        "formula": "LOOKUP",
        "profile": {"lookupProfile": _LOOKUP_PROFILE, "a0": 12.0},
        "mesh": {"lengthSegments": 8, "angularSegments": 16},
    }
    params, _formula, _mode = build_geometry_params(config)
    build_point_grid(params)
    with pytest.raises(ValueError, match="LOOKUP formula does not support throatExtLength"):
        build_point_grid({**params, "throatExtLength": 20.0})


# --------------------------------------------------------------------- M4 ---


def _rosse(**override):
    return {
        "type": "R-OSSE",
        "R": 150.0,
        "a": 45.0,
        "a0": 15.0,
        "k": 1.5,
        "r": 0.4,
        "m": 0.85,
        "b": 0.3,
        "q": 3.0,
        "r0": 12.7,
        "angularSegments": 16,
        "lengthSegments": 20,
        **override,
    }


def test_rosse_tmax_is_unchanged_by_a_throat_extension():
    # ath.exe (R2/R3 probes, tmax = 0.8): mouth r = 133.947 with or without a
    # 30 mm extension, and the extension adds exactly 30 mm of length.
    _grid_plain, plain = _grid(_rosse(tmax=0.8))
    _grid_ext, extended = _grid(_rosse(tmax=0.8, throatExtLength=30.0))
    assert _radii(plain)[0, -1] == pytest.approx(133.947, abs=2.0e-3)
    assert _radii(extended)[0, -1] == pytest.approx(_radii(plain)[0, -1], abs=1.0e-9)
    assert float(extended[0, -1, 2] - plain[0, -1, 2]) == pytest.approx(30.0, abs=1.0e-9)
    # The scalar oracle agrees at the end of the truncated path.
    x, y = calculate_rosse(0.8, 0.0, _rosse(tmax=0.8, throatExtLength=30.0))
    assert y == pytest.approx(_radii(plain)[0, -1], abs=1.0e-9)


def test_rosse_without_tmax_or_prefix_is_unchanged():
    x0, y0 = calculate_rosse(0.6, 0.3, _rosse())
    x1, y1 = calculate_rosse(0.6, 0.3, _rosse(tmax=1.0))
    assert (x0, y0) == (x1, y1)


def test_rosse_tmax_must_be_positive():
    with pytest.raises(ValueError, match="tmax"):
        build_point_grid(_rosse(tmax=0.0))


# ----------------------------------------------------------------- H1/H2 ---


def _morph_extension_config(**mesh):
    return {
        "formula": "OSSE",
        "mode": "bare",
        "profile": {"L": 80.0, "a": 45.0, "a0": 10.0, "r0": 12.7, "throat_ext_length_mm": 20.0},
        "morph": {"morphTarget": 1, "morphWidth": 400.0, "morphHeight": 200.0, "morphCorner": 20.0},
        "mesh": {
            "lengthSegments": 20,
            "angularSegments": 48,
            "throat_res_mm": 5.0,
            "mouth_res_mm": 15.0,
            "rear_res_mm": 20.0,
            **mesh,
        },
    }


@pytest.mark.parametrize("sampling_mode", ["uniform", "ath-default-zmap"])
def test_straight_throat_extension_stays_round_in_the_solve_grid(sampling_mode):
    config = _morph_extension_config(samplingMode=sampling_mode)
    resolved = resolve_geometry(config)
    points = resolved.geometry.inner_points
    radii = _radii(points)
    in_extension = points[0, :, 2] <= 20.0 + 1.0e-9
    assert in_extension.sum() >= 3
    assert float((radii.max(axis=0) - radii.min(axis=0))[in_extension].max()) < 1.0e-9
    # ... and in the requested (preview) grid, whatever its axial map.
    params, _formula, _mode = build_geometry_params(config)
    requested = _grid(params)[1]
    r_req = _radii(requested)
    ext_req = requested[0, :, 2] <= 20.0 + 1.0e-9
    assert float((r_req.max(axis=0) - r_req.min(axis=0))[ext_req].max()) < 1.0e-9
    assert resolved.sampling_metadata["geometrySampleAxialMap"] == "ath-default-zmap"


def test_solve_grid_morphs_from_the_station_the_preview_snapped_to():
    config = {
        "formula": "OSSE",
        "mode": "bare",
        "profile": {"L": 100.0, "a": 45.0, "a0": 10.0, "r0": 12.7},
        "morph": {
            "morphTarget": 1,
            "morphWidth": 360.0,
            "morphHeight": 220.0,
            "morphCorner": 10.0,
            "morphFixed": 0.33,
            "morphRate": 2.0,
            "morphAllowShrinkage": 1,
        },
        "mesh": {
            "lengthSegments": 20,
            "angularSegments": 48,
            "throat_res_mm": 5.0,
            "mouth_res_mm": 15.0,
            "rear_res_mm": 20.0,
        },
    }
    params, _formula, _mode = build_geometry_params(config)
    requested = build_point_grid_arrays(params)
    # Uniform 20 segments: FixedPart 0.33 snaps to t = 0.35.
    assert requested["morph_start"] == pytest.approx(0.35)
    resolved = resolve_geometry(config)
    points = resolved.geometry.inner_points
    t_fit = points[0, :, 2] / 100.0
    radii = _radii(points)
    spread = radii.max(axis=0) - radii.min(axis=0)
    # A ring sits exactly at the snapped start, nothing before it is morphed,
    # and the first ring after it is.
    at_start = int(np.argmin(np.abs(t_fit - 0.35)))
    assert t_fit[at_start] == pytest.approx(0.35, abs=1.0e-9)
    assert float(spread[: at_start + 1].max()) < 1.0e-9
    assert float(spread[at_start + 1]) > 1.0e-6
    # On both cardinal meridians the solve wall is exactly the preview's
    # analytic surface: raw + ((t - 0.35) / 0.65)^2 * (target - raw mouth).
    phis = np.arctan2(points[:, -1, 1], points[:, -1, 0])
    for phi, target in ((0.0, 180.0), (math.pi / 2.0, 110.0)):
        row = int(np.argmin(np.abs(np.angle(np.exp(1j * (phis - phi))))))
        _z, raw = calculate_osse_curve(t_fit * 100.0, phi, params)
        factor = np.clip((t_fit - 0.35) / 0.65, 0.0, 1.0) ** 2
        expected = raw + (target - raw[-1]) * factor
        assert radii[row] == pytest.approx(expected, abs=1.0e-9)


def test_subdomain_interfaces_land_at_their_requested_axial_position():
    config = {
        "formula": "OSSE",
        "profile": {"L": 100.0, "a": 45.0, "a0": 10.0, "r0": 12.7},
        "mesh": {
            "lengthSegments": 20,
            "angularSegments": 32,
            "subdomainSlices": "5,10,15",
            "interfaceOffset": 5,
            "throat_res_mm": 5.0,
            "mouth_res_mm": 15.0,
            "rear_res_mm": 20.0,
        },
        "enclosure": {"depth_mm": 200.0},
    }
    geometry = resolve_geometry(config).geometry
    z = [float(geometry.inner_points[0, item.slice_index, 2]) for item in geometry.interfaces]
    assert z == pytest.approx([25.0, 50.0, 75.0], abs=1.0e-9)


_ATH_EXTENSION_MORPH_CFG = """\
Throat.Profile = 1
Throat.Diameter = 25.4
Throat.Angle = 0
Length = 100
Coverage.Angle = 60
Term.s = 0.4
Term.n = 4.0
Term.q = 0.996
OS.k = 1.0
Morph.TargetShape = 1
Morph.TargetWidth = 360
Morph.TargetHeight = 220
Morph.CornerRadius = 10
Morph.Rate = 2.0
Morph.AllowShrinkage = 1
Mesh.Quadrants = 1234
Mesh.AngularSegments = 48
Mesh.LengthSegments = 20
Mesh.CornerSegments = 0
Mesh.SubdomainSlices =
Morph.FixedPart = 0.3
Throat.Ext.Length = 20
"""

# ath.exe's own slice positions for that config: ten uniform extension slices,
# then its default z-map over the 100 mm horn.
_ATH_EXTENSION_Z = [
    0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0,
    21.318979, 23.268547, 25.964636, 29.478424, 33.963327, 39.596105,
    46.304895, 54.051285, 62.729392, 71.875, 81.090964, 89.574088,
    97.022622, 103.352953, 108.547453, 112.564261, 115.589943, 117.780636,
    119.218652, 120.0,
]


def test_morph_behind_a_throat_extension_matches_ath(tmp_path):
    params, _formula = _ath_text_params(tmp_path, "ext_morph", _ATH_EXTENSION_MORPH_CFG)
    params = {
        **params,
        "athParitySampling": False,
        "samplingMode": "zmap",
        "zMapPoints": _zmap_samples(_ATH_EXTENSION_Z),
        "zMapKind": "samples",
        "lengthSegments": len(_ATH_EXTENSION_Z) - 1,
    }
    radii = _radii(_grid(params)[1])
    expected = {
        # profile index: radii at rings 10, 17, 20, 25, 30
        0: [12.7, 47.345726, 90.049328, 148.853232, 180.0],
        4: [12.7, 47.345726, 92.03835, 167.44733, 207.230515],
        12: [12.7, 47.345726, 84.936257, 101.054399, 110.0],
    }
    for profile, values in expected.items():
        assert radii[profile, [10, 17, 20, 25, 30]] == pytest.approx(values, abs=1.0e-4)


def test_ath_text_morph_uses_ath_fixed_part_default_and_slot_rule(tmp_path):
    params, _formula = _ath_text_params(
        tmp_path,
        "ath_defaults",
        "Length = 100\nCoverage.Angle = 45\nSlot.Length = 15\nMorph.TargetShape = 1\n",
    )
    assert params["morphFixed"] == 0.2
    assert params["_morphKeepsSlot"] is False
    explicit, _formula = _ath_text_params(
        tmp_path,
        "ath_explicit",
        "Length = 100\nCoverage.Angle = 45\nMorph.TargetShape = 1\nMorph.FixedPart = 0\n",
    )
    assert explicit["morphFixed"] == 0


def _slot_morph(**extra):
    return {
        "type": "OSSE",
        "L": 100.0,
        "a": 45.0,
        "a0": 10.0,
        "slotLength": 15.0,
        "_athLengthMode": "total",
        "morphTarget": 1,
        "morphWidth": 360.0,
        "morphHeight": 220.0,
        "morphCorner": 10.0,
        "morphAllowShrinkage": 1,
        "angularSegments": 48,
        "lengthSegments": 20,
        "samplingMode": "ath-default-zmap",
        **extra,
    }


def test_slot_is_reserved_by_position_on_a_clustered_map():
    grid, inner = _grid(_slot_morph())
    radii = _radii(inner)
    spread = radii.max(axis=0) - radii.min(axis=0)
    in_slot = inner[0, :, 2] <= 15.0 + 1.0e-9
    assert in_slot.sum() >= 5
    assert float(spread[in_slot].max()) < 1.0e-9
    # ATH's own rule morphs the slot; the ATH importer asks for it.
    _grid_ath, inner_ath = _grid(_slot_morph(_morphKeepsSlot=False))
    radii_ath = _radii(inner_ath)
    assert float((radii_ath.max(axis=0) - radii_ath.min(axis=0))[in_slot].max()) > 1.0e-3


def test_slot_reservation_on_a_uniform_map_is_the_historical_ring():
    grid, inner = _grid(_slot_morph(samplingMode="uniform"))
    # ceil(20 * 15 / 100) = 3 -> t = 0.15, which is also the first uniform
    # station at or past the slot end.
    assert grid["morph_start"] == pytest.approx(0.15)


def test_rosse_extension_is_never_morphed():
    params = _rosse(
        throatExtLength=20.0,
        morphTarget=1,
        morphWidth=360.0,
        morphHeight=220.0,
        morphCorner=10.0,
        morphAllowShrinkage=1,
        samplingMode="ath-default-zmap",
        angularSegments=48,
    )
    grid, inner = _grid(params)
    radii = _radii(inner)
    in_extension = inner[0, :, 2] <= 20.0 + 1.0e-9
    assert in_extension.sum() >= 3
    assert float((radii.max(axis=0) - radii.min(axis=0))[in_extension].max()) < 1.0e-9


# --------------------------------------------------------------------- L3 ---


@pytest.mark.parametrize("mode", ["zmap", "z-map", "custom", "Custom_ZMap"])
def test_icw_refuses_every_spelling_of_a_custom_zmap(mode):
    params = {"type": "ICW", "samplingMode": mode, "lengthSegments": 8, "angularSegments": 8}
    with pytest.raises(ValueError, match="ICW does not support"):
        build_point_grid(params)


@pytest.mark.parametrize("key", ["zMapPoints", "zmapPoints", "ZMapPoints"])
def test_icw_refuses_zmap_points_under_any_alias(key):
    params = {"type": "ICW", key: "0.3,0.1", "lengthSegments": 8, "angularSegments": 8}
    with pytest.raises(ValueError, match="ICW does not support"):
        build_point_grid(params)


# --------------------------------------------------------------------- L4 ---


def test_superellipse_exponent_below_two_is_clamped_like_ath():
    # ath.exe builds GCurve.SE.n = 1.5 and 2 identically.
    one_half = _grid(_gcurve_params(gcurveSeN=1.5))[1]
    two = _grid(_gcurve_params(gcurveSeN=2.0))[1]
    assert np.array_equal(one_half, two)


@pytest.mark.parametrize("short", ["1,1", "1,1,4", "1,1,4,2,2"])
def test_short_superformula_list_is_refused(short):
    params = _gcurve_params(gcurveType=2, gcurveSf=short)
    with pytest.raises(ValueError, match="six values"):
        build_point_grid(params)


@pytest.mark.parametrize("sf", [0, "0", "", None])
def test_single_value_superformula_means_not_given(sf):
    # Waveguide Generator's round trip: its .cfg export writes GCurve.SF = 0
    # and a reopened design sends gcurveSf as the string "0", with the
    # superformula held in the SF.a..n3 fields.
    fields = {
        "gcurveSfA": 1.0,
        "gcurveSfB": 1.0,
        "gcurveSfM1": 4.0,
        "gcurveSfM2": 4.0,
        "gcurveSfN1": 4.0,
        "gcurveSfN2": 4.0,
        "gcurveSfN3": 4.0,
    }
    config = {
        "formula": "OSSE",
        "mode": "bare",
        "profile": {"L": 100.0, "a": 45.0, "a0": 10.0, "r0": 12.7},
        "gcurve": {
            "gcurveType": 2,
            "gcurveWidth": 300.0,
            "gcurveAspectRatio": 0.6,
            "gcurveDist": 1.0,
            "gcurveSf": sf,
            **fields,
        },
        "mesh": {"lengthSegments": 20, "angularSegments": 48},
    }
    explicit = {**config, "gcurve": {**config["gcurve"], "gcurveSf": "1,1,4,4,4,4"}}
    got = _grid(build_geometry_params(config)[0])[1]
    expected = _grid(build_geometry_params(explicit)[0])[1]
    assert np.array_equal(got, expected)


def test_termination_parameters_are_free_when_the_term_is_off():
    base = {"type": "OSSE", "L": 80.0, "a": 45.0, "a0": 10.0, "angularSegments": 16, "lengthSegments": 8}
    with_s0 = _grid({**base, "s": 0.0, "n": 0.0, "q": 0.0})[1]
    assert np.array_equal(with_s0, _grid({**base, "s": 0.0})[1])


def test_freeform_overshoot_policy_is_refused_at_config_level():
    config = {
        "formula": "FREEFORM",
        "profile": {
            "profileH": {"points": [[0.0, 12.7], [100.0, 100.0]]},
            "profileV": {"points": [[0.0, 12.7], [100.0, 80.0]]},
            "overshootPolicy": "allow",
        },
    }
    with pytest.raises(ConfigError, match="overshootPolicy was removed"):
        build_geometry_params(config)
