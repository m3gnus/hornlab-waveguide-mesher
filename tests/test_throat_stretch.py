"""C4: degree-mode Desmos points and shared stretched geometry consumers."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from hornlab_mesher.builders.osse_waveguide import _osse_params
from hornlab_mesher.builders.rosse_waveguide import _rosse_params
from hornlab_mesher.config_builder import (
    build_from_config,
    build_geometry_params,
    resolve_geometry,
)
from hornlab_mesher.config_parser import ConfigError, parse_text_config
from hornlab_mesher.geometry import OsseHornGeometry, RosseHornGeometry
from hornlab_mesher.profile_formulas import (
    _icw_cache_key,
    _stretch_x_curve,
    calculate_osse,
    calculate_osse_curve,
    calculate_rosse,
    calculate_rosse_curve,
    osse_total_length,
    rosse_axial_layout,
)
from hornlab_mesher.profile_sampling import build_point_grid
from hornlab_mesher.viewport import build_viewport_geometry_from_config

FIXTURES = Path(__file__).parent / "fixtures" / "throat_stretch"
REFERENCES = json.loads((FIXTURES / "references.json").read_text())
CORPUS = json.loads((FIXTURES / "disabled-corpus.json").read_text())


@pytest.mark.parametrize("reference", REFERENCES, ids=lambda r: r["graph"])
def test_matches_degree_mode_desmos_point_for_point(reference):
    # Stored points transcribe the graph expressions independently of the mesher.
    rows = np.loadtxt(FIXTURES / (reference["graph"] + ".csv"), delimiter=",")
    params = reference["params"]
    scalar, vector = (
        (calculate_osse, calculate_osse_curve)
        if params["type"] == "OSSE"
        else (calculate_rosse, calculate_rosse_curve)
    )
    x, r = vector(rows[:, 0], 0.0, params)
    np.testing.assert_allclose(np.column_stack((x, r)), rows[:, 1:], rtol=0, atol=1e-10)
    actual = np.array([scalar(float(t), 0.0, params) for t in rows[:, 0]])
    np.testing.assert_allclose(actual, rows[:, 1:], rtol=0, atol=1e-10)


@pytest.mark.parametrize("family", ["OSSE", "R-OSSE"])
@pytest.mark.parametrize(
    "prefix",
    [
        {},
        {"throatExtLength": 12, "throatExtAngle": 3},
        {"slotLength": 8},
        {"throatExtLength": 12, "slotLength": 8},
    ],
)
@pytest.mark.parametrize("phi", [0, 0.37, math.pi / 2, 2.1, math.pi])
def test_scalar_vector_and_radius_invariance(family, prefix, phi):
    params = dict(
        next(r["params"] for r in REFERENCES if r["params"]["type"] == family)
    )
    params.update(prefix, s1="0.5 + 0.4*cos(p)^2", s2="0.05 + 0.1*sin(p)^2")
    scalar, vector = (
        (calculate_osse, calculate_osse_curve)
        if family == "OSSE"
        else (calculate_rosse, calculate_rosse_curve)
    )
    t = np.linspace(0, osse_total_length(params, phi) if family == "OSSE" else 1, 401)
    actual = np.column_stack(vector(t, phi, params))
    expected = np.array([scalar(float(u), phi, params) for u in t])
    scale = max(1, np.abs(expected).max())
    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=64 * np.finfo(float).eps * scale
    )
    _, radius = vector(t, phi, {**params, "s1": 0})
    assert np.array_equal(actual[:, 1], radius)


@pytest.mark.parametrize("family", ["OSSE", "R-OSSE"])
@pytest.mark.parametrize("key", ["s1", "s2"])
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), "-1 + sin(p)^2"])
def test_rejects_invalid_coefficients_even_in_prefix(family, key, value):
    params = {"type": family, "L": 120, "R": 140, "throatExtLength": 10, key: value}
    scalar, vector = (
        (calculate_osse, calculate_osse_curve)
        if family == "OSSE"
        else (calculate_rosse, calculate_rosse_curve)
    )
    with pytest.raises(ValueError, match=key):
        scalar(0, 0, params)
    with pytest.raises(ValueError, match=key):
        vector(np.array([0]), 0, params)


@pytest.mark.parametrize("key", ["s1", "s2"])
def test_negative_per_angle_expression_is_rejected_at_the_evaluated_angle(key):
    params = {"L": 120, key: "cos(p)"}
    calculate_osse(10, 0, params)
    with pytest.raises(ValueError, match=key):
        calculate_osse_curve(np.array([10]), math.pi, params)


@pytest.mark.parametrize("family", ["OSSE", "R-OSSE"])
@pytest.mark.parametrize("key", ["s1", "s2"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_config_validation(family, key, value):
    with pytest.raises(ConfigError):
        build_geometry_params({"profile": {"formula": family, key: value}})


@pytest.mark.parametrize("family", ["LOOKUP", "ICW", "FREEFORM"])
@pytest.mark.parametrize("key", ["s1", "s2"])
def test_refuses_stretch_on_other_profile_families(family, key):
    with pytest.raises(ConfigError, match="profile|shape"):
        build_geometry_params({"profile": {"formula": family, key: 0.5}})


def test_axial_map_is_monotone_and_preserves_rosse_foldback():
    x = np.linspace(-400, 400, 4001)
    for s1, s2 in [(0, 0), (0, 10), (1, 0), (0.5, 0.2), (10, 10)]:
        stretched = _stretch_x_curve(x, s1, s2)
        assert np.all(np.diff(stretched) > 0)
        assert np.all(np.diff(stretched) >= np.diff(x) - 1e-12)
    params = REFERENCES[0]["params"]
    t = np.linspace(0, 1, 1001)
    stretched, _ = calculate_rosse_curve(t, 0, params)
    base, _ = calculate_rosse_curve(t, 0, {**params, "s1": 0})
    assert np.any(np.diff(base) < 0)
    assert np.array_equal(np.sign(np.diff(stretched)), np.sign(np.diff(base)))
    osse = REFERENCES[2]["params"]
    assert np.all(
        np.diff(calculate_osse_curve(np.linspace(0, 160, 1001), 0, osse)[0]) > 0
    )


@pytest.mark.parametrize(
    "length_mode,main_length,total", [("main", 160, 180), ("total", 152, 172)]
)
def test_osse_length_prefix_and_rotation_contract(length_mode, main_length, total):
    params = {
        **REFERENCES[2]["params"],
        "throatExtLength": 12,
        "throatExtAngle": 3,
        "slotLength": 8,
        "_athLengthMode": length_mode,
    }
    assert osse_total_length(params) == total  # parameter span, not stretched depth
    stations = np.array([0, 6, 12, 16, 20, 20 + main_length / 2, total])
    x, radius = calculate_osse_curve(stations, 0, params)
    np.testing.assert_array_equal(x[:5], stations[:5])
    for j in [5, 6]:
        u = stations[j] - 20
        assert x[j] == pytest.approx(
            20 + u + 0.5 * math.degrees(math.atan(0.2 * u)), abs=1e-12
        )
    plain_radius = calculate_osse_curve(stations, 0, {**params, "s1": 0})[1]
    assert np.array_equal(radius, plain_radius)
    rotated = np.column_stack(calculate_osse_curve(stations, 0, {**params, "rot": 10}))
    base_x = stations
    angle = math.radians(10)
    rotated_x = base_x * math.cos(angle) - (radius - 10) * math.sin(angle)
    in_main = stations > 20
    rotated_x[in_main] = 20 + _stretch_x_curve(rotated_x[in_main] - 20, 0.5, 0.2)
    expected = np.column_stack(
        (
            rotated_x,
            10 + base_x * math.sin(angle) + (radius - 10) * math.cos(angle),
        )
    )
    np.testing.assert_allclose(rotated, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("tmax", [0.8, 1])
def test_rosse_stretches_main_x_before_prefix_translation(tmax):
    params = {
        **REFERENCES[0]["params"],
        "throatExtLength": 12,
        "slotLength": 8,
        "tmax": tmax,
    }
    layout = rosse_axial_layout(params)
    t = (
        tmax
        * np.array([0, 6, 12, 16, 20, 20 + layout.main_length * tmax])
        / layout.full_length
    )
    x, r = calculate_rosse_curve(t, 0, params)
    np.testing.assert_allclose(x[:5], [0, 6, 12, 16, 20], rtol=0, atol=1e-12)
    main = calculate_rosse(tmax, 0, {**REFERENCES[0]["params"], "tmax": tmax})
    assert x[-1] == pytest.approx(main[0] + 20, abs=1e-10)
    assert r[-1] == pytest.approx(main[1], abs=1e-10)


@pytest.mark.parametrize("block", ["OSSE", "R-OSSE"])
def test_text_and_json_serialization_preserve_coefficients_and_expressions(block):
    length = "L = 160" if block == "OSSE" else "R = 200"
    config = parse_text_config(
        f"{block} = {{\n{length}\ns1 = 0.5 + cos(p)^2\ns2 = 0.2\n}}"
    )
    normalized, _, _ = build_geometry_params(config)
    restored, _, _ = build_geometry_params(json.loads(json.dumps(config)))
    assert normalized == restored
    assert normalized["s1"] == "0.5 + cos(p)^2"
    assert normalized["s2"] == 0.2
    # Consecutive builds cannot return geometry from a different coefficient.
    config["mesh"].update(angularSegments=16, lengthSegments=8)
    old = build_point_grid(build_geometry_params(config)[0])["inner_points"]
    for key in ["s1", "s2"]:
        changed = copy.deepcopy(config)
        changed["profile"][key] = 1.5
        new = build_point_grid(build_geometry_params(changed)[0])["inner_points"]
        assert old != new


@pytest.mark.parametrize(
    "geometry,adapter",
    [(OsseHornGeometry(), _osse_params), (RosseHornGeometry(), _rosse_params)],
)
def test_geometry_identity_serialization_and_builder_mapping(geometry, adapter):
    for key in ["s1", "s2"]:
        changed = replace(geometry, **{key: 0.25})
        assert geometry != changed
        assert hash(geometry) != hash(changed)
        assert asdict(changed)[key] == 0.25
        assert adapter(changed)[key] == 0.25
    assert _icw_cache_key(
        {"icw_seed": {"type": "OSSE", "s1": 0.5, "s2": 0.2}}
    ) != _icw_cache_key({"icw_seed": {"type": "OSSE", "s1": 0.6, "s2": 0.2}})


@pytest.mark.parametrize("case", sorted(CORPUS))
@pytest.mark.parametrize("zero_key", ["s1", "s2"])
def test_disabled_feature_mesh_and_grid_hashes_are_bit_identical(
    case, zero_key, tmp_path
):
    config = CORPUS[case]
    disabled = copy.deepcopy(config)
    disabled["profile"].update(s1="0.5 + cos(p)^2", s2="0.2 + cos(p)^2")
    disabled["profile"][zero_key] = 0
    base_params = build_geometry_params(config)[0]
    disabled_params = build_geometry_params(disabled)[0]
    base_grid, off_grid = (
        build_point_grid(base_params),
        build_point_grid(disabled_params),
    )
    for key in ("inner_points", "outer_points"):
        if base_grid[key] is None:
            assert off_grid[key] is None
        else:
            a, b = np.asarray(base_grid[key]), np.asarray(off_grid[key])
            assert a.tobytes() == b.tobytes()
    # Raw .msh bytes include vertex numbering, triangles and physical groups.
    base = build_from_config(config, tmp_path / "base.msh")
    off = build_from_config(disabled, tmp_path / "off.msh")
    assert (
        hashlib.sha256(base.mesh_path.read_bytes()).digest()
        == hashlib.sha256(off.mesh_path.read_bytes()).digest()
    )
    assert base.n_vertices == off.n_vertices and base.n_triangles == off.n_triangles


@pytest.mark.parametrize("family", ["OSSE", "R-OSSE"])
def test_preview_solve_and_step_use_the_same_stretched_profile(family, tmp_path):
    import gmsh
    import meshio
    from hornlab_mesher.cad import write_step_from_config
    from hornlab_mesher.preview import build_preview_geometry
    from hornlab_mesher.preview.contract import PreviewOptionsV1

    config = copy.deepcopy(CORPUS["osse-bare" if family == "OSSE" else "rosse-bare"])
    config["profile"].update(s1=0.45, s2=0.2)
    config["mesh"]["surface_fit"] = "interpolate"
    viewport = build_viewport_geometry_from_config(config)
    params = viewport["params"]
    grid = viewport["grid"]
    inner = np.asarray(grid["inner_points"]).reshape(grid["grid_n_phi"], -1, 3)
    t = np.asarray(grid["slice_map"])
    scalar = calculate_osse if family == "OSSE" else calculate_rosse
    u = t * osse_total_length(params) if family == "OSSE" else t
    expected = np.array([scalar(float(v), 0, params) for v in u])
    np.testing.assert_allclose(inner[0][:, [2, 0]], expected, rtol=0, atol=1e-10)
    preview = build_preview_geometry(config, PreviewOptionsV1(lod="coarse"))
    preview_inner = next(s for s in preview.surfaces if s.role == "horn.inner")
    preview_points = np.asarray(preview_inner.positions)
    assert float(preview_points[:, 2].max()) == pytest.approx(
        float(inner[:, :, 2].max()), abs=0.1
    )
    resolved = resolve_geometry(config)
    # Check every preview vertex against an independent dense radial inversion
    # of the shared meridian (these fixtures have monotone radius, even where
    # R-OSSE folds back axially).
    dense_u = np.linspace(
        0, osse_total_length(params) if family == "OSSE" else 1, 100001
    )
    vector = calculate_osse_curve if family == "OSSE" else calculate_rosse_curve
    dense_x, dense_r = vector(dense_u, 0, params)
    assert np.all(np.diff(dense_r) > 0)
    preview_r = np.linalg.norm(preview_points[:, :2], axis=1)
    reference_x = np.interp(preview_r, dense_r, dense_x)
    np.testing.assert_allclose(preview_points[:, 2], reference_x, rtol=0, atol=0.002)
    probes = resolved.geometry.inner_points[
        :: max(1, resolved.geometry.inner_points.shape[0] // 4),
        :: max(1, resolved.geometry.inner_points.shape[1] // 12),
    ].reshape(-1, 3)
    mesh = build_from_config(config, tmp_path / "stretched.msh")
    from hornlab_mesher.mesher import _triangles_and_physical_tags
    from hornlab_mesher.tags import PhysicalGroup

    mesh_data = meshio.read(mesh.mesh_path)
    mesh_points = mesh_data.points
    triangles, tags = _triangles_and_physical_tags(mesh_data)
    wall_nodes = np.unique(triangles[tags == PhysicalGroup.RIGID_WALL])
    assert wall_nodes.size > 0
    wall_points_mm = mesh_points[wall_nodes] * 1000
    # Solve writes metres; canonical grids and CAD use mm.
    assert mesh_points[:, 2].max() * 1000 == pytest.approx(
        resolved.geometry.inner_points[:, :, 2].max(), abs=0.1
    )
    step, info = write_step_from_config(config, tmp_path / "stretched.step")
    assert info.body == "surface"
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(str(step), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        surfaces = gmsh.model.getEntities(2)
        for point in np.concatenate((probes, wall_points_mm)):
            distances = [
                np.linalg.norm(
                    np.asarray(gmsh.model.getClosestPoint(2, tag, point.tolist())[0])
                    - point
                )
                for _, tag in surfaces
            ]
            assert min(distances) < 0.1
    finally:
        gmsh.finalize()
