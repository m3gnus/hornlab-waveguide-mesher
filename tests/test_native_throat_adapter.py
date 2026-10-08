"""Analytic and input-boundary checks for circular native adapters."""

import json
import math

import numpy as np
import pytest

from hornlab_mesher.config_builder import build_geometry_params, resolve_geometry
from hornlab_mesher.config_parser import ConfigError
from hornlab_mesher.profile_formulas import calculate_osse, calculate_osse_curve, osse_total_length
from hornlab_mesher.profile_sampling import build_point_grid_arrays
from hornlab_mesher.throat_adapter import CONTROLS, normalize_adapter, resolve_adapter
from hornlab_mesher.viewport import build_viewport_geometry_from_config


def config():
    return {
        "formula": "OSSE-ADAPTER", "mode": "bare",
        "profile": {"L": 120., "r0": 12.7, "a": 40., "a0": 15.5, "k": 1., "s": 0.},
        "source": {"source_shape": 0},
        "mesh": {"angular_segments": 32, "length_segments": 13},
        "throat_adapter": {"mode": "authored", "contract_revision": 1,
                           "driver_exit_diameter_mm": 25.4, "exit_half_angle_deg": 10.,
                           "length_mm": 40., "join_t": .137,
                           "driver_handle_mm": 10., "body_handle_mm": 15.},
    }


def construction(cfg=None):
    params, _, _ = build_geometry_params(cfg or config())
    return params, resolve_adapter(params)


def de_casteljau(controls, u):
    rows = [tuple(row) for row in controls]
    while len(rows) > 1:
        rows = [tuple((1-u)*a+u*b for a, b in zip(left, right)) for left, right in zip(rows[:-1], rows[1:])]
    return rows[0]


def test_independent_cubic_endpoints_and_tangents():
    params, meridian = construction()
    p = meridian.payload
    z, r = meridian.evaluate(np.linspace(0, .5, 101))
    oracle = np.array([de_casteljau(meridian.cubic, u) for u in np.linspace(0, 1, 101)])
    np.testing.assert_allclose(np.column_stack((z, r)), oracle, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(meridian.cubic[0], [0, p["driver_exit_diameter_mm"]/2])
    outgoing = meridian.cubic[1]-meridian.cubic[0]
    assert abs(math.atan2(outgoing[1], outgoing[0])-math.radians(10)) < 1e-14
    tangent = meridian.base_tangent(p["join_t"])
    incoming = meridian.cubic[3]-meridian.cubic[2]
    np.testing.assert_allclose(incoming/np.linalg.norm(incoming), tangent/np.linalg.norm(tangent), atol=1e-14, rtol=0)
    np.testing.assert_allclose(meridian.evaluate(.5), meridian.cubic[-1], atol=1e-12, rtol=0)
    assert osse_total_length(params) == 40+120*(1-.137)


@pytest.mark.parametrize("k", [0.2, 1., 2., 8.])
def test_rational_conic_is_literal_osse_curve(k):
    cfg = config()
    cfg["profile"]["k"] = k
    cfg["throat_adapter"]["body_handle_mm"] = 3.
    _, curve = construction(cfg)
    u = np.linspace(0, 1, 301)
    basis = np.column_stack(((1-u)**2, 2*u*(1-u), u*u))*curve.body_weights
    zr = basis@curve.body_poles/np.sum(basis, axis=1)[:, None]
    original_z = zr[:, 0]-40+120*.137
    r0, a0, a = 12.7, math.radians(15.5), math.radians(40)
    expected = np.sqrt((k*r0)**2+2*k*r0*original_z*math.tan(a0)+original_z**2*math.tan(a)**2)+r0*(1-k)
    np.testing.assert_allclose(zr[:, 1], expected, atol=1e-11, rtol=0)


def test_shared_scalar_vector_sampler_and_exact_join():
    params, curve = construction()
    stations = np.array([0., .071, .5, .793, 1.])
    depth = osse_total_length(params)
    expected = np.column_stack(curve.evaluate(stations))
    scalar = np.array([calculate_osse(float(t)*depth, .713, params) for t in stations])
    vector = np.column_stack(calculate_osse_curve(stations*depth, 1.23, params))
    np.testing.assert_allclose(scalar, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(vector, expected, atol=1e-12, rtol=0)
    grid = build_point_grid_arrays(params)
    assert .5 in grid["slice_map"]
    j = grid["slice_map"].index(.5)
    np.testing.assert_allclose(grid["inner_grid"][:, j, 2], 40, atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(grid["inner_grid"][:, 0, :2], axis=1), 12.7, atol=1e-12, rtol=0)
    assert grid["semantic_stations"] == {"driver": 0., "adapter_join": .5, "mouth": 1.}


def test_scale_placement_and_normalized_persistence():
    cfg = config()
    reference = build_viewport_geometry_from_config(cfg, point_lists=False)
    cfg["scale"] = 1.7
    cfg["mesh"]["vertical_offset_mm"] = 7.
    output = build_viewport_geometry_from_config(cfg, point_lists=False)
    expected = 1.7*reference["grid"]["inner_grid"]
    expected[:, :, 1] += 7
    np.testing.assert_allclose(output["grid"]["inner_grid"], expected, atol=1e-12, rtol=0)
    params, _ = construction(json.loads(json.dumps(cfg)))
    assert params["throat_adapter"] == cfg["throat_adapter"]
    assert params["construction_fingerprint"] != reference["params"]["construction_fingerprint"]
    resolved = resolve_geometry(cfg)
    assert resolved.geometry.source_auto_angle_deg == 10
    assert resolved.geometry.adapter_scale == 1.7
    assert resolved.geometry.adapter_meridian.fingerprint == params["construction_fingerprint"]


@pytest.mark.parametrize("field", CONTROLS)
def test_each_active_control_changes_construction_and_fingerprint(field):
    cfg = config()
    _, first = construction(cfg)
    cfg["throat_adapter"][field] += .01
    _, second = construction(cfg)
    assert first.fingerprint != second.fingerprint
    assert not np.array_equal(first.evaluate(.185), second.evaluate(.185))


@pytest.mark.parametrize("bad", [None, "10", True, float("nan"), float("inf"), [], {}])
def test_malformed_controls_refuse(bad):
    cfg = config()
    cfg["throat_adapter"]["driver_handle_mm"] = bad
    with pytest.raises(ConfigError, match="Curved adapter refused"):
        build_geometry_params(cfg)


@pytest.mark.parametrize("change", [
    {"mode": "future"}, {"mode": []}, {"mode": {}},
    {"contract_revision": 2}, {"contract_revision": True}, {"contract_revision": 1.0},
    {"unrecognized": 0}, {"join_t": 1}, {"join_t": -.1},
    {"driver_handle_mm": 40, "body_handle_mm": 30},
    {"exit_half_angle_deg": -60, "driver_handle_mm": 20},
    {"length_mm": 0}, {"driver_exit_diameter_mm": 0},
])
def test_invalid_payload_refuses(change):
    cfg = config()
    cfg["throat_adapter"].update(change)
    with pytest.raises(ConfigError, match="Curved adapter refused"):
        build_geometry_params(cfg)


@pytest.mark.parametrize("section,key,value", [
    ("profile", "s", 1), ("profile", "h", 1), ("profile", "rot", 1),
    ("profile", "s1", .4), ("profile", "slotLength", 10), ("profile", "throatExtAngle", 5),
    ("profile", "a", "40+sin(p)"), ("profile", "throat_profile", 3),
    ("morph", "morph_target", 1), ("gcurve", "gcurve_type", 1),
    ("cross_section", "aspect_ratio", .9), ("cross_section", "exponent", 3),
    ("mesh", "wall_thickness_mm", 1), ("mesh", "quadrants", "1"),
    ("mesh", "topology_mode", "legacy"), ("mesh", "surface_fit", "approximate"),
    ("source", "source_shape", 1), ("source", "source_curv", 1),
    ("profile", "driver_throat_diameter_mm", 25.4),
])
def test_unsupported_composition_refuses_before_sampling(section, key, value):
    cfg = config()
    cfg.setdefault(section, {})[key] = value
    with pytest.raises(ConfigError, match="Curved adapter refused"):
        build_geometry_params(cfg)


def test_off_has_exact_legacy_parameters_and_grid():
    cfg = config()
    cfg.pop("throat_adapter")
    cfg["formula"] = "OSSE"
    params, _, _ = build_geometry_params(cfg)
    baseline = build_point_grid_arrays(params)
    cfg["throat_adapter"] = {"mode": "off", "driver_handle_mm": "inactive editor value"}
    disabled, _, _ = build_geometry_params(cfg)
    assert disabled == params
    output = build_point_grid_arrays(disabled)
    assert output.keys() == baseline.keys()
    np.testing.assert_array_equal(output["inner_grid"], baseline["inner_grid"])
    assert normalize_adapter({"mode": "off"}) is None


def test_zero_join_and_near_mouth_refusal():
    cfg = config()
    cfg["throat_adapter"].update(join_t=0, body_handle_mm=1)
    _, curve = construction(cfg)
    np.testing.assert_allclose(curve.cubic[-1], [40, 12.7], atol=1e-12, rtol=0)
    cfg["throat_adapter"]["join_t"] = 1-1e-12
    with pytest.raises(ConfigError, match="resolvable retained body"):
        construction(cfg)


@pytest.mark.parametrize("payload", [None, {"mode": "off"}])
def test_format_discriminator_cannot_fall_back(payload):
    cfg = config()
    if payload is None:
        cfg.pop("throat_adapter")
    else:
        cfg["throat_adapter"] = payload
    with pytest.raises(ConfigError, match="requires an active"):
        build_geometry_params(cfg)


def test_active_saved_format_protects_older_readers():
    cfg = config()
    cfg["formula"] = "OSSE"
    with pytest.raises(ConfigError, match="older readers fail closed"):
        build_geometry_params(cfg)
    # Earlier readers only know the common formula discriminator and reject
    # this required feature before evaluating an ordinary body.
    from hornlab_mesher.profile_common import _normalise_formula

    with pytest.raises(ValueError, match="formula must be"):
        _normalise_formula("OSSE-ADAPTER")


@pytest.mark.parametrize("section,key,value", [
    ("profile", "k", 101), ("profile", "L", 10001), ("profile", "r0", 10001),
    ("profile", "a", .1), ("profile", "a0", 40),
    (None, "scale", 101), (None, "scale", 1e-5),
    ("mesh", "vertical_offset_mm", 1e16),
])
def test_unresolved_numeric_domain_refuses(section, key, value):
    cfg = config()
    if section:
        cfg[section][key] = value
    else:
        cfg[key] = value
    with pytest.raises(ConfigError, match="Curved adapter refused"):
        build_geometry_params(cfg)


def test_sub_tolerance_cubic_span_refuses_before_cad():
    cfg = config()
    cfg["scale"] = .001
    cfg["throat_adapter"].update(length_mm=1e-5, driver_handle_mm=2e-6,
                                  body_handle_mm=2e-6, join_t=0, exit_half_angle_deg=0)
    with pytest.raises(ConfigError, match="scaled feature clearance"):
        build_geometry_params(cfg)


def test_adapter_preview_default_budget_and_identity():
    from hornlab_mesher.preview import build_preview_geometry, PreviewOptionsV1

    cfg = config()
    preview = build_preview_geometry(cfg, PreviewOptionsV1(lod="coarse"))
    params, curve = construction(cfg)
    assert preview.metadata["requested_fidelity"]["max_chord_error_mm"] == .008
    assert preview.metadata["construction_fingerprint"] == curve.fingerprint
    assert preview.metadata["throat_adapter"] == params["throat_adapter"]
    assert abs(preview.metadata["source_cap_radius_mm"]-12.7) < 1e-10
    assert "adapter join" in preview.metadata["semantic_stations"]["inserted_first"]
    explicit = build_preview_geometry(cfg, PreviewOptionsV1(lod="coarse", max_chord_error_mm=.02))
    assert explicit.metadata["requested_fidelity"]["max_chord_error_mm"] == .02


@pytest.mark.parametrize("diameter_mm,throat_resolution_mm", [(1., .0002), (25.4, .002)])
def test_scaled_adapter_mesh_preserves_small_and_dense_source_topology(
    tmp_path, diameter_mm, throat_resolution_mm,
):
    import meshio
    from hornlab_mesher.edges import build_edge_table
    from hornlab_mesher.geometry import MeshDensity
    from hornlab_mesher.mesher import build_mesh_with_info
    from hornlab_mesher.tags import PhysicalGroup

    cfg = config()
    cfg["scale"] = .001
    cfg["throat_adapter"].update(driver_exit_diameter_mm=diameter_mm, exit_half_angle_deg=0.)
    resolved = resolve_geometry(cfg)
    path, _ = build_mesh_with_info(
        resolved.geometry,
        MeshDensity(throat_res_mm=throat_resolution_mm, mouth_res_mm=.02),
        tmp_path/"small-adapter.msh", scale_to_metres=False,
    )
    mesh = meshio.read(path)
    points = np.asarray(mesh.points)
    triangles = mesh.get_cells_type("triangle")
    tags = mesh.get_cell_data("gmsh:physical", "triangle")
    source = triangles[tags == int(PhysicalGroup.PRIMARY_SOURCE)]
    assert len(source) > 0
    cross = np.cross(points[source[:, 1]]-points[source[:, 0]],
                     points[source[:, 2]]-points[source[:, 0]])
    radius = .5*diameter_mm*cfg["scale"]
    assert np.all(cross[:, 2] > 0)
    np.testing.assert_allclose(points[np.unique(source), 2], 0., atol=1e-12, rtol=0)
    assert .5*np.linalg.norm(cross, axis=1).sum() == pytest.approx(math.pi*radius**2, rel=.03)

    source_edges = build_edge_table(source)
    rim = source_edges.count == 1
    np.testing.assert_allclose(np.linalg.norm(points[source_edges.lo[rim], :2], axis=1),
                               radius, atol=1e-10, rtol=0)
    # A disk has Euler characteristic one, and every rim edge must also be
    # owned by a wall triangle. No cap tears or unwelded throat edges survive.
    assert len(np.unique(source))-source_edges.n_edges+len(source) == 1
    all_edges = build_edge_table(triangles)
    incidence = dict(zip(zip(all_edges.lo, all_edges.hi), all_edges.count))
    assert all(incidence[(lo, hi)] == 2 for lo, hi in
               zip(source_edges.lo[rim], source_edges.hi[rim]))
    assert np.all(all_edges.count <= 2)
    join_z = cfg["throat_adapter"]["length_mm"]*cfg["scale"]
    join = (np.isclose(points[all_edges.lo, 2], join_z, atol=1e-10, rtol=0)
            & np.isclose(points[all_edges.hi, 2], join_z, atol=1e-10, rtol=0))
    assert np.count_nonzero(join) >= 4
    assert np.all(all_edges.count[join] == 2)
    # The only free boundary is the mouth, never the source or adapter join.
    free = all_edges.count == 1
    assert np.any(free)
    mouth_z = resolved.geometry.adapter_meridian.evaluate(1.)[0]*cfg["scale"]
    np.testing.assert_allclose(points[all_edges.lo[free], 2], mouth_z, atol=1e-10, rtol=0)
    np.testing.assert_allclose(points[all_edges.hi[free], 2], mouth_z, atol=1e-10, rtol=0)


@pytest.mark.parametrize("entry", [calculate_osse, calculate_osse_curve, osse_total_length])
def test_direct_formula_entries_refuse_empty_unknown_adapter(entry):
    params = {"throat_adapter": {}}
    with pytest.raises(ValueError, match="mode must be"):
        if entry is osse_total_length:
            entry(params)
        else:
            entry(0, 0, params)


def test_exact_step_reopens_with_adapter_and_recipe(tmp_path):
    import gmsh
    from hornlab_mesher.cad import write_step_from_config, write_wglink
    from hornlab_mesher.mesher import MesherError
    from hornlab_mesher.native_env import preserve_native_windows_path

    cfg = config()
    params, curve = construction(cfg)
    path, info = write_step_from_config(cfg, tmp_path/"native.step")
    assert info.body == "surface" and info.n_faces == 8 and info.throat_opened
    assert info.construction_fingerprint == curve.fingerprint
    assert info.construction["adapter"] == params["throat_adapter"]
    text = path.read_text(encoding="utf-8")
    assert curve.fingerprint in text and '"join_t":0.137' in text
    with pytest.raises(MesherError, match="supports only"):
        write_wglink(resolve_geometry(cfg).geometry, tmp_path/"native.wglink")
    with preserve_native_windows_path():
        gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.occ.importShapes(str(path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        surfaces = gmsh.model.getEntities(2)
        assert len(surfaces) == 8
        for station in (.185, .5, .83):
            z, r = curve.evaluate(station)
            xyz = np.array([float(r)*math.cos(.61), float(r)*math.sin(.61), float(z)])
            distance = min(np.linalg.norm(np.asarray(gmsh.model.getClosestPoint(2, tag, xyz.tolist())[0])-xyz)
                           for _, tag in surfaces)
            assert distance < 1e-7
    finally:
        with preserve_native_windows_path():
            gmsh.finalize()
