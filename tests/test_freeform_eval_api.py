"""Public production evaluation and separate resolved-curve comparison."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from hornlab_mesher import (
    freeform_compare_profiles,
    freeform_evaluate_profile,
    freeform_insert_anchor,
    freeform_profile_curves,
)
from hornlab_mesher.freeform import _BezierPlane, _PlaneSpline, build_freeform_geometry


def reviewer():
    return {"points": [[0, 12.7], [40, 30], [120, 140]],
            "throatAngleDeg": 15.5, "mouthAngleDeg": 60}


def params(profile):
    return {"profileH": copy.deepcopy(profile), "profileV": copy.deepcopy(profile)}


def corners():
    return {"points": [{"z": 0, "r": 12.7, "join": "corner"},
                       {"z": 40, "r": 30, "join": "corner"},
                       {"z": 120, "r": 140, "join": "corner"}]}


@pytest.mark.parametrize("profile,kind", [
    (reviewer(), _PlaneSpline),
    (corners(), _BezierPlane),
    (freeform_insert_anchor(reviewer(), z=21.407, default_throat_angle_deg=15.5), _BezierPlane),
])
def test_evaluate_equals_production_bit_for_bit(profile, kind):
    config = params(profile)
    # Distinct V ensures plane dispatch is exercised.
    config["profileV"]["points"][-1] = {"z": 120, "r": 130}
    geometry = build_freeform_geometry(config)
    assert type(geometry._profile_h) is kind
    anchors = [row["z"] if isinstance(row, dict) else row[0] for row in profile["points"]]
    z = np.unique(np.r_[np.linspace(0, 120, 137), anchors])
    expected = geometry.evaluate_radii(z)
    for plane, radii in zip(("H", "V"), expected):
        assert np.all(freeform_evaluate_profile(config, plane, z) == radii)
        shaped = z[:6].reshape(2, 3)
        actual = freeform_evaluate_profile(config, plane, shaped)
        assert actual.shape == shaped.shape
        assert np.all(actual == radii[:6].reshape(2, 3))
        scalar = freeform_evaluate_profile(config, plane, 40)
        assert isinstance(scalar, np.ndarray) and scalar.shape == ()
        assert scalar == geometry.evaluate_radii(40)[plane == "V"]
        assert freeform_evaluate_profile(config, plane, []).shape == (0,)


def test_inputs_defaults_and_precedence_match_curve_query():
    profile = reviewer()
    del profile["throatAngleDeg"]
    plain = params(profile)
    config = {"formula": "FREEFORM", "a0_deg": 30, "profile": plain}
    z = [0, 21.407, 40, 80, 120]
    expected = build_freeform_geometry({**plain, "a0": 30}).evaluate_radii(z)[0]
    assert np.all(freeform_evaluate_profile(config, "H", z) == expected)
    assert np.all(freeform_evaluate_profile(plain, "H", z, default_throat_angle_deg=30) == expected)
    assert np.all(freeform_evaluate_profile(config, "H", z, default_throat_angle_deg=60) == expected)
    assert freeform_profile_curves(config)["H"]["anchors"][0]["angleOutDeg"] == pytest.approx(30)
    explicit = params(reviewer())
    assert np.all(freeform_evaluate_profile(explicit, "H", z, default_throat_angle_deg=30) ==
                  build_freeform_geometry(explicit).evaluate_radii(z)[0])
    result = freeform_compare_profiles(config, {**plain, "a0": 30}, samples=33)
    assert result["H"]["curve_deviation_mm"] == result["H"]["evaluator_deviation_mm"] == 0
    assert config["profile"] == plain  # Queries do not rewrite input.


@pytest.mark.parametrize("z", [-1, 120.000000001, np.nan, np.inf, [1, np.nan]])
def test_evaluate_rejects_invalid_z(z):
    with pytest.raises(ValueError, match="z queries"):
        freeform_evaluate_profile(params(reviewer()), "H", z)


def test_evaluate_rejects_invalid_plane():
    with pytest.raises(ValueError, match="plane"):
        freeform_evaluate_profile(params(reviewer()), "horizontal", 20)


@pytest.mark.parametrize("profile", [reviewer(), corners()])
def test_identical_comparison_is_zero_and_reports_actual_grid(profile):
    config = params(profile)
    result = freeform_compare_profiles(config, copy.deepcopy(config), samples=11)
    json.dumps(result, allow_nan=False)
    for plane in ("H", "V"):
        comparison = result[plane]
        assert comparison["curve_deviation_mm"] == comparison["evaluator_deviation_mm"] == 0
        assert comparison["samples"] == sorted(set(np.linspace(0, 120, 11)) | {40.0})
        assert comparison["extents"] == {
            "before": [0, 120], "after": [0, 120], "extents_equal": True,
            "overlap": [0, 120], "comparison_domain": "full"}
        assert "not a certified maximum" in comparison["method"]["estimate"]


def test_exact_insert_separates_curve_and_legacy_evaluator_change():
    before = params(reviewer())
    after = copy.deepcopy(before)
    after["profileH"] = freeform_insert_anchor(reviewer(), z=21.407, default_throat_angle_deg=15.5)
    result = freeform_compare_profiles(before, after)
    assert result["H"]["curve_deviation_mm"] <= 1e-9
    assert 1e-8 < result["H"]["evaluator_deviation_mm"] < 1e-4
    assert result["V"]["curve_deviation_mm"] == result["V"]["evaluator_deviation_mm"] == 0
    assert 21.407 in result["H"]["samples"]
    z = np.asarray(result["H"]["samples"])
    actual = abs(freeform_evaluate_profile(before, "H", z) - freeform_evaluate_profile(after, "H", z))
    assert result["H"]["evaluator_deviation_mm"] == float(actual.max())
    assert result["H"]["evaluator_at_z_mm"] == float(z[np.argmax(actual)])


def test_refit_reviewer_case_is_about_1_207_mm():
    before = params(reviewer())
    after = copy.deepcopy(before)
    radius = float(freeform_evaluate_profile(before, "H", 21.407))
    after["profileH"]["points"].insert(1, [21.407, radius])
    comparison = freeform_compare_profiles(before, after)["H"]
    assert comparison["curve_deviation_mm"] == pytest.approx(1.207, abs=0.001)
    assert comparison["evaluator_deviation_mm"] == pytest.approx(1.207, abs=0.001)
    assert comparison["curve_at_z_mm"] in comparison["samples"]
    # Refinement must improve a coarse common-z grid on this broad peak.
    coarse = freeform_compare_profiles(before, after, samples=7)["H"]
    assert coarse["curve_deviation_mm"] == pytest.approx(comparison["curve_deviation_mm"], abs=1e-7)


@pytest.mark.parametrize("start,stop,overlap", [(20, 160, [20, 120]), (120, 160, [120, 120]),
                                              (140, 160, None)])
def test_different_extents_use_overlap_including_empty_and_touching(start, stop, overlap):
    before = params(corners())
    after = params({"points": [{"z": start, "r": 12.7, "join": "corner"},
                                {"z": stop, "r": 140, "join": "corner"}]})
    result = freeform_compare_profiles(before, after, samples=17)
    json.dumps(result, allow_nan=False)
    for comparison in result.values():
        extents = comparison["extents"]
        assert not extents["extents_equal"]
        assert extents["before"] == [0, 120] and extents["after"] == [start, stop]
        assert extents["overlap"] == overlap
        if overlap is None:
            assert comparison["samples"] == []
            assert comparison["curve_deviation_mm"] is comparison["evaluator_deviation_mm"] is None
            assert comparison["curve_at_z_mm"] is comparison["evaluator_at_z_mm"] is None
            assert extents["comparison_domain"] == "no_overlap"
        else:
            assert min(comparison["samples"]) == overlap[0]
            assert max(comparison["samples"]) == overlap[1]
            assert extents["comparison_domain"] == "overlap_only"


@pytest.mark.parametrize("samples", [True, 1, 2.5, None])
def test_compare_requires_integer_sample_count(samples):
    with pytest.raises(ValueError, match="samples"):
        freeform_compare_profiles(params(reviewer()), params(reviewer()), samples=samples)


def test_fresh_process_without_gmsh_or_grid():
    script = '''
import importlib.abc, sys
class BlockMeshing(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "gmsh" or fullname.startswith("gmsh."):
            raise AssertionError("gmsh import attempted")
sys.meta_path.insert(0, BlockMeshing())
from hornlab_mesher import freeform_compare_profiles, freeform_evaluate_profile
import hornlab_mesher.profile_sampling as sampling
import hornlab_mesher.profiles as profiles
def blocked(*args, **kwargs):
    raise AssertionError("grid construction attempted")
sampling.build_point_grid_arrays = blocked
profiles.build_point_grid = blocked
profiles.build_point_grid_arrays = blocked
p = {"points": [[0, 12.7], [40, 30], [120, 140]]}
config = {"profileH": p, "profileV": p}
assert freeform_evaluate_profile(config, "H", [0, 40, 120]).tolist() == [12.7, 30, 140]
assert freeform_compare_profiles(config, config, samples=17)["H"]["curve_deviation_mm"] == 0
assert "gmsh" not in sys.modules
'''
    subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                   check=True, timeout=30)


@pytest.mark.parametrize('profile', [reviewer(), corners()])
def test_query_shares_production_h_extent_and_relative_boundary_tolerance(profile):
    config = params(profile)
    # Ingest admits H/V end positions within 1e-9 mm; production uses H's span.
    for i, delta in ((0, 5e-10), (-1, -5e-10)):
        row = config['profileV']['points'][i]
        if isinstance(row, dict):
            row['z'] += delta
        else:
            row[0] += delta
    geometry = build_freeform_geometry(config)
    z = np.array([-1e-11, 0, 120, 120 + 1e-11])
    expected = geometry.evaluate_radii(z)
    for plane, radii in zip(('H', 'V'), expected):
        assert np.array_equal(freeform_evaluate_profile(config, plane, z), radii)
        with pytest.raises(ValueError, match='shared'):
            freeform_evaluate_profile(config, plane, 120 + 1e-8)
