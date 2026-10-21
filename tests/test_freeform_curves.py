"""Object joins, local Bézier handles, exact insertion and mesh-entry gates."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from hornlab_mesher import (
    build_from_config, freeform_insert_anchor, freeform_profile_curves,
    write_step_from_config,
)
from hornlab_mesher.freeform import (
    _BezierPlane, _PlaneSpline, _bezier_values, _parse_anchors,
    _parse_endpoint_values, _plane_beziers, _resolve_plane,
    build_freeform_geometry,
)


def reviewer():
    return {"points": [[0, 12.7], [40, 30], [120, 140]],
            "throatAngleDeg": 15.5, "mouthAngleDeg": 60}


def corners():
    return {"points": [{"z": 0, "r": 12.7, "join": "corner"},
                       {"z": 40, "r": 30, "join": "corner"},
                       {"z": 120, "r": 140, "join": "corner"}],
            "throatAngleDeg": 75, "mouthAngleDeg": 10}


def params(profile):
    return {"profileH": profile, "profileV": copy.deepcopy(profile)}


def resolved(profile):
    parsed = _parse_anchors(profile, "H")
    return _resolve_plane("H", parsed, *_parse_endpoint_values(profile, parsed, "H", 15.5),
                          force_extended=True)


def test_legacy_gate_every_example():
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    for path in Path("examples").rglob("*.toml"):
        config = tomllib.loads(path.read_text())
        if config.get("formula") != "FREEFORM":
            continue
        old = build_freeform_geometry(config.get("profile", config))
        new = copy.deepcopy(config.get("profile", config))
        for key in ("profileH", "profileV"):
            new[key]["points"] = [{"z": p[0], "r": p[1], "join": "smooth",
                                   **({"angleDeg": p[2]} if len(p) == 3 else {})}
                                  for p in new[key]["points"]]
        converted = build_freeform_geometry(new)
        z = np.linspace(0, old.length_mm, 2001)
        assert type(converted._profile_h) is _PlaneSpline
        assert all((a == b).all() for a, b in zip(old.evaluate_radii(z), converted.evaluate_radii(z)))
        assert old.report() == converted.report()


def test_default_objects_mixed_rows_keep_legacy_gate():
    p = reviewer()
    p["points"][1] = {"z": 40, "r": 30, "angleDeg": 35}
    g = build_freeform_geometry(params(p))
    assert type(g._profile_h) is _PlaneSpline
    p["points"][1] = [40, 30, 35]
    old = build_freeform_geometry(params(p))
    assert g.report() == old.report()
    assert np.array_equal(g.evaluate_radii(np.linspace(0, 120, 2001)),
                          old.evaluate_radii(np.linspace(0, 120, 2001)))


@pytest.mark.parametrize("profile", [reviewer(),
    {"points": [[0, 12.7], [40, 30, 20], [120, 140]], "mouthAngleDeg": 60},
    {"points": [[0, 12.7], [40, 13], [120, 100]], "throatAngleDeg": 70}])
def test_hermite_bezier_same_mathematical_curve(profile):
    legacy = build_freeform_geometry(params(profile))._profile_h
    bezier = resolved(profile)
    assert isinstance(bezier, _BezierPlane)
    for index, controls in enumerate(_plane_beziers(bezier)):
        s = np.linspace(0, 1, 2001)
        actual = _bezier_values(np.broadcast_to(controls, (len(s), 4, 2)), s)
        expected = legacy.spline(legacy.anchor_u[index] + s*np.diff(legacy.anchor_u)[index])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
        np.testing.assert_allclose(bezier.radii_at_z(expected[:, 0]), expected[:, 1], rtol=0, atol=1e-9)
        np.testing.assert_allclose(legacy.radii_at_z(expected[:, 0]),
                                   bezier.radii_at_z(expected[:, 0]), rtol=0, atol=1e-4)  # legacy inverse-table error, measured <= 5.0e-5 mm


def test_corner_straight_spans_including_endpoints():
    p = corners()
    curve = freeform_profile_curves(params(p), samples_per_span=17)["H"]
    assert all(s["straight"] for s in curve["spans"])
    plane = resolved(p)
    for first, last in zip(plane.anchors[:-1], plane.anchors[1:]):
        z = np.linspace(first[0], last[0], 2001)
        expected = first[1] + (z-first[0])*(last[1]-first[1])/(last[0]-first[0])
        np.testing.assert_allclose(plane.radii_at_z(z), expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("changes,incoming,outgoing", [
    ({"angleDeg": 35}, 35, 35),
    ({"angleDeg": 20, "angleOutDeg": 50}, 20, 50),
    ({"angleOutDeg": 50}, math.degrees(math.atan2(17.3, 40)), 50),
])
def test_corner_side_angle_precedence(changes, incoming, outgoing):
    p = corners()
    p["points"][1].update(changes)
    a = freeform_profile_curves(params(p))["H"]["anchors"][1]
    assert a["angleInDeg"] == pytest.approx(incoming)
    assert a["angleOutDeg"] == pytest.approx(outgoing)


def test_endpoint_angle_beats_corner_and_corner_ignores_block_angle():
    p = corners()
    p["points"][0]["angleDeg"] = 25
    p["points"][-1]["angleDeg"] = 55
    q = freeform_profile_curves(params(p))["H"]
    assert q["anchors"][0]["angleOutDeg"] == pytest.approx(25)
    assert q["anchors"][-1]["angleInDeg"] == pytest.approx(55)
    p["points"][0].pop("angleDeg")
    p["points"][-1].pop("angleDeg")
    g = build_freeform_geometry(params(p))
    assert g.report()["tangentAnglesDeg"]["H"]["throat"] == pytest.approx(math.degrees(math.atan2(17.3, 40)))
    assert all(s["straight"] for s in freeform_profile_curves(params(p))["H"]["spans"])


def test_explicit_handles_win_over_angles_and_are_never_scaled():
    p = corners()
    p["points"][0].update(angleDeg=70, handleOut=[10, 4])
    p["points"][1].update(angleDeg=70, handleIn=[-10, -4], handleOut=[20, 20])
    p["points"][2]["handleIn"] = [-20, -30]
    q = freeform_profile_curves(params(p))["H"]
    assert q["anchors"][0]["angleOutDeg"] == pytest.approx(math.degrees(math.atan2(4, 10)))
    assert q["anchors"][0]["handleOut"] == [10, 4]
    assert q["anchors"][1]["handleIn"] == [-10, -4]
    assert q["anchors"][1]["explicitIn"] is True
    assert q["anchors"][1]["explicitOut"] is True


@pytest.mark.parametrize("index,changes,match", [
    (0, {"handleIn": [-1, 0]}, "handleIn"),
    (2, {"handleOut": [1, 0]}, "handleOut"),
    (0, {"angleOutDeg": 20}, "interior corners"),
    (1, {"join": "smooth", "angleOutDeg": 20}, "interior corners"),
    (1, {"angleOutDeg": 90}, "angleOutDeg"),
    (1, {"angleDeg": 90}, "angleDeg"),
    (1, {"join": "smooth", "handleIn": [-1, 0], "handleOut": [1, 1]}, "collinear and opposite"),
    (1, {"join": "smooth", "handleIn": [-1, 0], "handleOut": [-1, 0]}, "collinear and opposite"),
    (1, {"join": "smooth", "handleIn": [0, 0], "handleOut": [1, 0]}, "nonzero"),
    (1, {"surprise": 1}, "surprise"),
    (1, {"join": "other"}, "join"),
    (1, {"handleIn": [1]}, "vector"),
    (1, {"handleIn": [float("nan"), 1]}, "finite"),
])
def test_object_validation_refusals(index, changes, match):
    p = corners()
    p["points"][index].update(changes)
    with pytest.raises(ValueError, match=match):
        build_freeform_geometry(params(p))


@pytest.mark.parametrize("outgoing,incoming,match", [
    ([100, 0], [100, 0], "folds backward"),
    ([10, -80], [-10, -80], "non-positive radius"),
])
def test_explicit_invalid_span_names_plane_index_and_z_range(outgoing, incoming, match):
    p = {"points": [{"z": 0, "r": 12.7, "handleOut": outgoing},
                    {"z": 40, "r": 30, "handleIn": incoming}]}
    with pytest.raises(ValueError, match=r"profileH span 0 z=\[0, 40\].*"+match):
        build_freeform_geometry(params(p))


def test_only_automatic_handle_shrinks():
    p = {"points": [{"z": 0, "r": 12.7, "handleOut": [5, .01]}, [40, 13]],
         "mouthAngleDeg": 80}
    plane = resolved(p)
    assert (plane.beziers[0, 1] - plane.beziers[0, 0]).tolist() == [5, pytest.approx(.01)]
    assert plane.speed_factors[-1] < 1


def test_both_explicit_handles_allow_radius_overshoot():
    p = {"points": [{"z": 0, "r": 12.7, "handleOut": [10, 100]},
                    {"z": 40, "r": 30, "handleIn": [-10, 100]}]}
    plane = resolved(p)
    assert plane.radii_at_z(np.asarray(20)) > 30
    q = freeform_profile_curves(params(p))["H"]
    assert q["anchors"][0]["handleOut"] == [10, 100]
    assert q["anchors"][1]["handleIn"] == [-10, 100]


@pytest.mark.parametrize("explicit_side", ["out", "in"])
def test_mixed_handles_warn_when_only_overshoot_remains_at_minimum(explicit_side):
    points = [{"z": 0, "r": 20}, {"z": 40, "r": 20}]
    points[0 if explicit_side == "out" else 1][
        "handleOut" if explicit_side == "out" else "handleIn"] = [
            10 if explicit_side == "out" else -10, 100]
    p = {"points": points, "throatAngleDeg": 0, "mouthAngleDeg": 0}
    geometry = build_freeform_geometry(params(p))
    query = freeform_profile_curves(params(p))
    messages = geometry.report()["warnings"]
    assert len(messages) == 2
    for name, message in zip(("H", "V"), messages):
        assert f"profile{name} span 0 z=[0, 40]" in message
        assert "minimum" in message and "overshoots" in message
        assert message in query["warnings"]
    anchors = query["H"]["anchors"]
    assert anchors[0 if explicit_side == "out" else 1][
        "handleOut" if explicit_side == "out" else "handleIn"] == [
            10 if explicit_side == "out" else -10, 100]
    automatic = anchors[1 if explicit_side == "out" else 0][
        "handleIn" if explicit_side == "out" else "handleOut"]
    assert np.linalg.norm(automatic) < 1e-12


@pytest.mark.parametrize("handle,match", [([-100, 0], "folds backward"),
                                        ([10, -100], "non-positive radius")])
def test_mixed_handles_still_refuse_invalid_geometry_at_minimum(handle, match):
    p = {"points": [{"z": 0, "r": 20, "handleOut": handle}, [40, 20]],
         "mouthAngleDeg": 0}
    with pytest.raises(ValueError, match=r"profileH span 0 z=\[0, 40\].*" + match):
        resolved(p)


@pytest.mark.parametrize("profile,z", [(reviewer(), 21.407), (corners(), 21.407),
                                      (reviewer(), .001), (reviewer(), 119.999)])
def test_exact_insert_whole_curve_and_json_roundtrip(profile, z):
    original = copy.deepcopy(profile)
    plane = resolved(profile)
    inserted = freeform_insert_anchor(profile, z=z)
    assert profile == original
    assert len(inserted["points"]) == len(profile["points"]) + 1
    for output in (inserted, json.loads(json.dumps(inserted, allow_nan=False))):
        new_plane = resolved(output)
        for first, last in zip(plane.anchors[:-1], plane.anchors[1:]):
            samples = np.linspace(first[0], last[0], 2001)
            np.testing.assert_allclose(new_plane.radii_at_z(samples), plane.radii_at_z(samples), rtol=0, atol=1e-9)
        new_anchor = next(row for row in output["points"] if row["z"] == z)
        assert new_anchor["join"] == "smooth"
        assert "handleIn" in new_anchor and "handleOut" in new_anchor


def test_reviewer_naive_refit_moves_other_span():
    p = reviewer()
    old = resolved(p)
    r = float(old.radii_at_z(np.asarray(21.407)))
    p["points"].insert(1, [21.407, r])
    refitted = resolved(p)
    difference = float(refitted.radii_at_z(np.asarray(70.5)) - old.radii_at_z(np.asarray(70.5)))
    assert abs(difference) == pytest.approx(1.207, abs=.001)
    inserted = resolved(freeform_insert_anchor(reviewer(), z=21.407))
    assert float(inserted.radii_at_z(np.asarray(70.5))) == pytest.approx(float(old.radii_at_z(np.asarray(70.5))), abs=1e-9)


@pytest.mark.parametrize("z", [-1, 0, 40, 120, 121, float("nan")])
def test_insert_requires_z_strictly_inside_span(z):
    with pytest.raises(ValueError):
        freeform_insert_anchor(reviewer(), z=z)


@pytest.mark.parametrize("profile", [reviewer(), corners(), freeform_insert_anchor(reviewer(), z=21.407)])
def test_curve_query_shape_samples_and_report(profile):
    q = freeform_profile_curves(params(profile), samples_per_span=13)
    assert set(q) == {"H", "V", "warnings"}
    assert json.loads(json.dumps(q, allow_nan=False)) == q
    assert q == freeform_profile_curves({"formula": "FREEFORM", "profile": params(profile)}, samples_per_span=13)
    g = build_freeform_geometry(params(profile))
    for name in ("H", "V"):
        curve = q[name]
        assert set(curve) == {"anchors", "spans", "samples", "inflections"}
        assert len(curve["samples"]) == 1 + (len(profile["points"])-1)*12
        assert curve["inflections"] == g.report()["inflectionSpans"][name]
        assert np.all(np.diff(np.array(curve["samples"])[:, 0]) > 0)
        for a in curve["anchors"]:
            assert set(a) == {"z", "r", "join", "explicitIn", "explicitOut", "handleIn", "handleOut", "angleInDeg", "angleOutDeg"}
            assert [a["z"], a["r"]] in curve["samples"]


@pytest.mark.parametrize("samples", [True, 1, 0, -1, 2.5, "64"])
def test_query_refuses_invalid_sample_count(samples):
    with pytest.raises(ValueError, match="samples_per_span"):
        freeform_profile_curves(params(reviewer()), samples_per_span=samples)


def test_curve_query_does_not_import_gmsh_or_build_grid():
    code = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'gmsh':
        raise AssertionError('curve query imported '+name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from hornlab_mesher import freeform_profile_curves
import hornlab_mesher.profile_sampling as sampling
def no_grid(*args, **kwargs):
    raise AssertionError('curve query built a grid')
sampling.build_point_grid = sampling.build_point_grid_arrays = no_grid
p = {'points': [{'z': 0, 'r': 12.7, 'join': 'corner'}, [100, 50]]}
assert freeform_profile_curves({'profileH': p, 'profileV': p})['H']['samples']
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def coarse_config(profile):
    return {"formula": "FREEFORM", "mode": "bare", "profile": params(profile),
            "mesh": {"angularSegments": 16, "lengthSegments": 8, "samplingMode": "uniform",
                     "throat_res_mm": 30, "mouth_res_mm": 30,
                     "max_triangles": 10000, "allow_large_mesh": True}}


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("entry", ["mesh", "step", "grid"])
def test_corner_wall_refused_at_build_entry(tmp_path, position, entry):
    p = reviewer()
    row = p["points"][position]
    p["points"][position] = {"z": row[0], "r": row[1], "join": "corner"}
    assert freeform_profile_curves(params(p))["H"]["anchors"][position]["join"] == "corner"
    assert build_freeform_geometry(params(p)).length_mm == 120
    with pytest.raises(ValueError, match=r"^FREEFORM corner joins need the corner-wall builder \(not yet available\)$"):
        if entry == "mesh":
            build_from_config(coarse_config(p), tmp_path / "corner.msh")
        elif entry == "step":
            write_step_from_config(coarse_config(p), tmp_path / "corner.step")
        else:
            from hornlab_mesher.profile_sampling import build_point_grid
            build_point_grid({"type": "FREEFORM", **params(p), "lengthSegments": 8, "angularSegments": 16})


def test_handles_only_c1_mesh_builds(tmp_path):
    p = freeform_insert_anchor({"points": [[0, 12.7], [40, 30], [100, 60]]}, z=20)
    result = build_from_config(coarse_config(p), tmp_path / "handles.msh")
    assert Path(result.mesh_path).stat().st_size > 0
    assert result.metadata["freeformReport"]["throatRadiusMm"] == 12.7


def test_extended_report_and_cache_fields():
    from hornlab_mesher.freeform import _freeform_cache_key
    p = corners()
    q = freeform_profile_curves(params(p))
    report = build_freeform_geometry(params(p)).report()
    assert report["straightSpans"] == {"H": [0, 1], "V": [0, 1]}
    assert report["anchorTangents"]["H"] == q["H"]["anchors"]
    original = _freeform_cache_key(params(p))
    for field, value in (("join", "smooth"), ("angleDeg", 25), ("angleOutDeg", 50),
                         ("handleIn", [-5, -2]), ("handleOut", [10, 5])):
        changed = copy.deepcopy(p)
        changed["points"][1][field] = value
        assert _freeform_cache_key(params(changed)) != original


def test_preview_semantic_stations_accept_object_points():
    from hornlab_mesher.preview.horn import _semantic_t_stations
    from hornlab_mesher.profile_sampling import build_point_grid_arrays
    p = freeform_insert_anchor({"points": [[0, 12.7], [40, 30], [100, 60]]}, z=20)
    profile_params = {"type": "FREEFORM", **params(p), "lengthSegments": 8, "angularSegments": 16}
    grid = build_point_grid_arrays(profile_params)
    t = np.asarray(grid["slice_map"])
    indices, inserted, _ = _semantic_t_stations(
        {"params": profile_params, "formula": "FREEFORM", "grid": grid}, t)
    assert "FREEFORM H/V anchors" in inserted
    assert all(np.any(np.abs(t[indices] - target) < 1e-12) for target in (0, .2, .4, 1))


@pytest.mark.parametrize("deviation,allowed", [(0.5e-9, True), (2e-9, False)])
def test_smooth_handle_collinearity_tolerance(deviation, allowed):
    p = {"points": [[0, 12.7],
                    {"z": 40, "r": 30, "handleIn": [-1, 0],
                     "handleOut": [math.cos(deviation), math.sin(deviation)]}, [120, 140]]}
    if allowed:
        _parse_anchors(p, "H")
    else:
        with pytest.raises(ValueError, match="collinear and opposite"):
            _parse_anchors(p, "H")


@pytest.mark.parametrize("field,handle", [("handleIn", [-10, -15]), ("handleOut", [10, 1])])
def test_single_smooth_explicit_handle_keeps_opposite_side_smooth(field, handle):
    p = reviewer()
    p["points"][1] = {"z": 40, "r": 30, field: handle}
    a = freeform_profile_curves(params(p))["H"]["anchors"][1]
    assert a["angleInDeg"] == pytest.approx(a["angleOutDeg"], abs=1e-9)
    for z in (20, 60, 100):
        inserted = resolved(freeform_insert_anchor(p, z=z))
        samples = np.linspace(0, 120, 4001)
        np.testing.assert_allclose(inserted.radii_at_z(samples), resolved(p).radii_at_z(samples), rtol=0, atol=1e-9)


@pytest.mark.parametrize("z", [1e-8, 1e-7, 39.9999999, 119.9999999])
def test_exact_insert_near_anchor_avoids_handle_cancellation(z):
    p = reviewer()
    original = resolved(p)
    inserted = freeform_insert_anchor(p, z=z)
    # Re-parsing the serialized vectors must preserve their collinearity.
    after = resolved(json.loads(json.dumps(inserted)))
    anchor = freeform_profile_curves(params(inserted))["H"]["anchors"][next(
        i for i, row in enumerate(inserted["points"]) if row["z"] == z)]
    assert anchor["angleInDeg"] == pytest.approx(anchor["angleOutDeg"], abs=1e-9)
    samples = np.linspace(0, 120, 4001)
    np.testing.assert_allclose(after.radii_at_z(samples), original.radii_at_z(samples), rtol=0, atol=1e-9)


def test_exact_insert_preserves_subdivision_radius_overshoot():
    p = {"points": [[0, 80], [75, 50], [100, 80]],
         "throatAngleDeg": -60, "mouthAngleDeg": 45}
    build_freeform_geometry(params(p))
    original = resolved(p)
    inserted = freeform_insert_anchor(p, z=10)
    after = resolved(json.loads(json.dumps(inserted, allow_nan=False)))
    for first, last in zip(original.anchors[:-1], original.anchors[1:]):
        samples = np.linspace(first[0], last[0], 2001)
        np.testing.assert_allclose(after.radii_at_z(samples), original.radii_at_z(samples),
                                   rtol=0, atol=1e-9)


@pytest.mark.parametrize("seed", range(20))
def test_exact_insert_seeded_random_legacy_profiles(seed):
    rng = np.random.default_rng(seed)
    count = 3 + seed % 4
    z = np.r_[0., np.cumsum(rng.uniform(15, 50, count - 1))]
    radii = rng.uniform(10, 120, count)
    p = {"points": np.column_stack((z, radii)).tolist(),
         "throatAngleDeg": float(rng.uniform(-65, 65)),
         "mouthAngleDeg": float(rng.uniform(-65, 65))}
    legacy = build_freeform_geometry(params(p))._profile_h
    assert type(legacy) is _PlaneSpline
    original = resolved(p)
    samples = np.unique(np.concatenate([
        np.linspace(first, last, 2001) for first, last in zip(z[:-1], z[1:])]))
    expected = original.radii_at_z(samples)
    for _ in range(5):
        span = int(rng.integers(count - 1))
        insert_z = float(z[span] + rng.uniform(.01, .99) * (z[span+1] - z[span]))
        inserted = freeform_insert_anchor(p, z=insert_z)
        after = resolved(json.loads(json.dumps(inserted, allow_nan=False)))
        assert len(after.anchors) == count + 1
        np.testing.assert_allclose(after.radii_at_z(samples), expected, rtol=0, atol=1e-9)


@pytest.mark.parametrize("fixture_path", [
    "tests/fixtures/morph_identity/baseline.json",
    "tests/fixtures/throat_stretch/base-compatibility.json",
])
def test_legacy_gate_persisted_freeform_fixtures(fixture_path):
    from hornlab_mesher.config_builder import build_geometry_params

    def configurations(value):
        if isinstance(value, dict):
            profile = value.get("profile")
            if isinstance(profile, dict) and "profileH" in profile:
                yield value
            else:
                for child in value.values():
                    yield from configurations(child)
        elif isinstance(value, list):
            for child in value:
                yield from configurations(child)

    configs = list(configurations(json.loads(Path(fixture_path).read_text())))
    assert configs  # Refuse a vacuous pass if a fixture's schema changes.
    for config in configs:
        original, formula, _ = build_geometry_params(config)
        assert formula == "FREEFORM"
        rewritten = copy.deepcopy(original)
        for key in ("profileH", "profileV"):
            rewritten[key]["points"] = [
                {"z": row[0], "r": row[1], "join": "smooth",
                 **({"angleDeg": row[2]} if len(row) == 3 else {})}
                for row in original[key]["points"]]
        old = build_freeform_geometry(original)
        new = build_freeform_geometry(rewritten)
        assert type(new._profile_h) is _PlaneSpline
        assert type(new._profile_v) is _PlaneSpline
        z = np.linspace(old._profile_h.anchors[0, 0], old._profile_h.anchors[-1, 0], 2001)
        assert all((a == b).all() for a, b in zip(old.evaluate_radii(z), new.evaluate_radii(z)))
        assert old.report() == new.report()
