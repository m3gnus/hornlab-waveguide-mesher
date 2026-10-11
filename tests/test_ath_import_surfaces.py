"""Authored lip grids and rear vertex sets from ATH V2026-09a exports."""
from pathlib import Path
from dataclasses import replace

import meshio
import numpy as np
import pytest

from hornlab_mesher.config_parser import ConfigError, parse_text_config
from hornlab_mesher.config_builder import build_geometry_params, resolve_geometry
from hornlab_mesher.profile_sampling import build_point_grid_arrays
from hornlab_mesher.edges import build_edge_table

FIXTURES = Path(__file__).parent / "fixtures/ath_import/ath-v2026-09a"
CASES = ["p1", "segdef", "of20", "onr", "rs1", "odef", "ow0", "ofull", "coreabec"]
REAR_CASES = CASES[2:]


def config(name):
    return parse_text_config((FIXTURES / f"{name}.cfg").read_text())


def reference(name):
    return np.array([[[float(v) for v in row.split(";")] for row in block.splitlines()]
                     for block in (FIXTURES / f"{name}.csv").read_text().strip().split("\n\n")])


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("quadrants", ["1", "1234"])
def test_grid_including_lip_matches_ath(name, quadrants):
    cfg = config(name)
    cfg["mesh"]["quadrants"] = quadrants
    expected = reference(name)
    if quadrants == "1":
        expected = expected[:len(expected)//4 + 1]
    params = build_geometry_params(cfg)[0]
    actual = build_point_grid_arrays(params)
    assert actual["inner_grid"].shape == expected.shape
    assert actual["grid_n_length"] + 1 == expected.shape[1]
    assert len(actual["slice_map"]) == expected.shape[1]
    np.testing.assert_allclose(actual["inner_grid"], expected, rtol=0, atol=2e-4)
    # Both solve routes must keep the measured station-dependent construction.
    for topology in ("acoustic", "legacy"):
        cfg["mesh"]["topology"] = topology
        np.testing.assert_array_equal(resolve_geometry(cfg).geometry.inner_points, actual["inner_grid"])


@pytest.mark.parametrize("name", REAR_CASES)
def test_rear_vertex_set_matches_ath(name):
    resolved = resolve_geometry(config(name))
    geometry = resolved.geometry
    selected = {'of20': [10, 11], 'onr': [7, 10, 11], 'odef': [7, 9],
                'ow0': [9], 'ofull': [7, 9], 'coreabec': [4, 7, 9, 11, 12],
                'rs1': [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 21, 22, 23, 24, 25, 26]}
    assert resolved.sampling_metadata['athImportSurfaces']['rearSelectedRows'] == selected[name]
    # Compare the entire rear/wall set: shifted throat ring, all selected
    # offset rings, unoffset Opus front ring, and fan apex. Exclude the bore,
    # ATH's mouth-disc construction center and source sphere control points.
    actual = np.vstack((geometry.ath_rear_points.reshape(-1, 3), geometry.ath_rear_apex))
    expected = np.loadtxt(FIXTURES / f"{name}-rear.csv", delimiter=";", skiprows=1)
    distance = np.linalg.norm(actual[:, None] - expected[None, :], axis=-1)
    assert distance.min(axis=1).max() <= 2e-3
    assert distance.min(axis=0).max() <= 2e-3  # No unexplained ATH rear vertices.


@pytest.mark.parametrize("name", CASES)
def test_fixture_normalisation(name):
    for extension in ("cfg", "csv"):
        data = (FIXTURES / f"{name}.{extension}").read_bytes()
        assert b"\r" not in data
        assert data.endswith(b"\n") and not data.endswith(b"\n\n")
    if name == "coreabec":
        assert (FIXTURES / f"{name}.cfg").read_text().startswith(
            "; Derived from ATH CORE (C) 2026 Marcel Batík, CC-BY-4.0, at-horns.eu/core.html\n")


def test_core_selection_normals_and_nonmonotonic_lip():
    resolved = resolve_geometry(config("coreabec"))
    g = resolved.geometry
    assert resolved.sampling_metadata["athImportSurfaces"]["rearSelectedRows"] == [4, 7, 9, 11, 12]
    assert g.ath_rear_apex[2] == pytest.approx(-10.1977, abs=1e-4)
    assert np.ptp(g.ath_rear_points[:, 0, 2]) > .1  # Retain the warped throat.
    assert np.any(np.diff(g.inner_points[0, :, 2]) < 0)
    np.testing.assert_allclose(np.linalg.norm(g.ath_normals, axis=-1), 1, atol=1e-12)
    np.testing.assert_array_equal(g.ath_rear_points[:, -1, :2], g.inner_points[:, -1, :2])
    np.testing.assert_allclose(g.ath_rear_points[:, -1, 2], g.inner_points[:, -1, 2] - 40,
                               rtol=0, atol=1e-12)


def assert_watertight_after_symmetry(mesh, geometry):
    from hornlab_mesher.mesher import _validate_closed_shell_contract
    triangles = np.concatenate([cell.data for cell in mesh.cells if cell.type == "triangle"])
    points = np.asarray(mesh.points)
    physical = np.concatenate([mesh.cell_data["gmsh:physical"][i]
                               for i, cell in enumerate(mesh.cells) if cell.type == "triangle"])
    _validate_closed_shell_contract(points, triangles, physical,
        symmetry_snap_axes=geometry.symmetry_planes, symmetry_snap_tol_mm=1e-6, is_open_shell=False)
    # A quadrant boundary is closed by its reflected neighbours, not a
    # fictitious rigid sheet on each cut plane. Weld only coincident cuts.
    for axis in geometry.symmetry_planes:
        reflected = points.copy()
        reflected[:, {"x": 0, "y": 1}[axis]] *= -1
        triangles = np.vstack((triangles, triangles[:, ::-1] + len(points)))
        points = np.vstack((points, reflected))
    _, inverse = np.unique(np.round(points, 9), axis=0, return_inverse=True)
    assert np.all(build_edge_table(inverse[triangles]).count == 2)


def test_core_watertight_quadrant_mesh(tmp_path):
    from hornlab_mesher.mesher import build_mesh
    resolved = resolve_geometry(config("coreabec"))
    path = build_mesh(resolved.geometry, resolved.density, tmp_path / "core.msh", scale_to_metres=False)
    mesh = meshio.read(path)
    triangles = np.concatenate([cell.data for cell in mesh.cells if cell.type == "triangle"])
    assert_watertight_after_symmetry(mesh, resolved.geometry)
    assert np.all(np.isfinite(mesh.points))


@pytest.mark.parametrize("name", ["of20", "rs1"])
@pytest.mark.parametrize("quadrants", ["1", "1234"])
@pytest.mark.parametrize("pathway", ["acoustic", "legacy-wg", "legacy-grid"])
def test_all_shell_topologies_are_watertight(tmp_path, name, quadrants, pathway):
    from hornlab_mesher.mesher import build_mesh
    cfg = config(name)
    cfg["mesh"]["quadrants"] = quadrants
    cfg["mesh"]["topology"] = "acoustic" if pathway == "acoustic" else "legacy"
    resolved = resolve_geometry(cfg)
    g = resolved.geometry
    if pathway == "legacy-grid":
        g = replace(g, preserve_grid=True, wg_topology=False)
    mesh = meshio.read(build_mesh(g, resolved.density, tmp_path / "shell.msh", scale_to_metres=False))
    triangles = np.concatenate([cell.data for cell in mesh.cells if cell.type == "triangle"])
    assert_watertight_after_symmetry(mesh, g)


BASE = "OSSE = {\n L = 100\n r0 = 25\n a0 = 10\n a = 45\n}\nABEC.SimType = 2\n"
LIP = "Mesh.Roundover = {\n Radius = 20\n Segments = 6\n}\n"


@pytest.mark.parametrize("extra,named", [
    ("Mesh.Roundover = 42\n", "Mesh.Roundover"),
    ("Mesh.Roundover = {\n Radius = 20\n MadeUp = 1\n}\n", "Mesh.Roundover.MadeUp"),
    ("Mesh.Roundover = {\n Radius = 0\n}\n", "Mesh.Roundover.Radius"),
    ("Mesh.Roundover = {\n Radius = -1\n}\n", "Mesh.Roundover.Radius"),
    ("Mesh.Roundover = {\n Radius = nan\n}\n", "Mesh.Roundover.Radius"),
    ("Mesh.Roundover = {\n Radius = 20\n Segments = 0\n}\n", "Mesh.Roundover.Segments"),
    ("Mesh.Roundover = {\n Radius = 20\n Segments = 1.5\n}\n", "Mesh.Roundover.Segments"),
    (LIP + "ABEC.SimType = 1\n", "Mesh.Roundover"),
    (LIP + "Mesh.Enclosure = {\n Depth = 100\n}\n", "Mesh.Enclosure"),
    (LIP + "CircSym = 1\n", "CircSym"),
    ("Mesh.RearShape = 3\nABEC.SimType = 1\n", "Mesh.RearShape"),
    ("Mesh.RearShape = 3\nMesh.Enclosure = {\n Depth = 100\n}\n", "Mesh.Enclosure"),
    ("Mesh.RearShape = 2\n", "Mesh.RearShape"),
    ("Mesh.RearShape = 5\n", "Mesh.RearShape"),
    ("Mesh.RearShape = 3\nMesh.RearShape.Step = 0\n", "Mesh.RearShape.Step"),
    ("Mesh.RearShape = 3\nMesh.RearShape.FrontOffset = nan\n", "Mesh.RearShape.FrontOffset"),
    ("Mesh.RearShape.Step = 50\n", "Mesh.RearShape.FrontOffset/Step"),
    ("Mesh.RearCubic = 1\n", "Mesh.RearCubic"),
    ("Mesh.OpusRear = 1\n", "Mesh.OpusRear"),
    ("Mesh.MouthArc = 1\n", "Mesh.MouthArc"),
    ("Mesh.RoundedEdge = 1\n", "Mesh.RoundedEdge"),
])
def test_named_parser_refusals(extra, named):
    with pytest.raises(ConfigError) as error:
        parse_text_config(BASE + extra)
    assert named in str(error.value)


@pytest.mark.parametrize("extra,named", [
    (LIP + "Mesh.LengthSegments = 2\n", "Mesh.LengthSegments"),
    (LIP + "Mesh.WallThickness = 20\n", "Mesh.WallThickness"),
    (LIP + "Mesh.SubdomainSlices = 3\n", "Mesh.SubdomainSlices"),
    ("Mesh.RearShape = 3\nMesh.RearShape.FrontOffset = 500\n", "Mesh.RearShape.FrontOffset"),
])
def test_named_resolution_refusals(extra, named):
    with pytest.raises(ConfigError) as error:
        resolve_geometry(parse_text_config(BASE + extra))
    assert named in str(error.value)


def test_rosse_lip_refused():
    with pytest.raises(ConfigError, match="R-OSSE"):
        parse_text_config("R-OSSE = {\n R = 150\n a = 45\n}\nABEC.SimType = 2\n" + LIP)


@pytest.mark.parametrize("segments", [1, 2])
def test_lip_ring_on_the_mouth_refused_by_name(segments):
    # A 45-degree cone has mouth angle 45 deg, so theta = 0 for row `segments`:
    # ATH V2026-09a repeats the mouth row in its grid, which leaves zero-area faces.
    text = ("OSSE = {\n L = 100\n r0 = 25\n a0 = 45\n a = 45\n k = 1\n n = 4\n q = 0.995\n s = 0\n}\n"
            f"Mesh.Roundover = {{\n Radius = 20\n Segments = {segments}\n}}\n"
            "Mesh.Quadrants = 1\nMesh.AngularSegments = 32\nMesh.LengthSegments = 20\n"
            "Mesh.WallThickness = 5\nABEC.SimType = 2\n")
    with pytest.raises(ConfigError, match=f"Mesh.Roundover lip row {segments} coincides"):
        resolve_geometry(parse_text_config(text))


def test_scalar_zero_is_noop():
    a = resolve_geometry(parse_text_config(BASE)).geometry
    b = resolve_geometry(parse_text_config(BASE + "Mesh.Roundover = 0\n")).geometry
    assert type(a) is type(b)
    np.testing.assert_array_equal(a.inner_points, b.inner_points)


@pytest.mark.parametrize("name", ["p1", "onr", "rs1", "coreabec"])
def test_step_refuses_by_name(tmp_path, name):
    from hornlab_mesher.cad import write_step
    from hornlab_mesher.mesher import MesherError
    cfg = config(name)
    with pytest.raises(MesherError, match="STEP.*Mesh.Roundover / Mesh.RearShape"):
        write_step(resolve_geometry(cfg).geometry, tmp_path / "refused.step")
    assert not (tmp_path / "refused.step").exists()


@pytest.mark.parametrize("name", CASES)
def test_json_roundtrip_retains_geometry(name):
    import json
    cfg = config(name)
    restored = json.loads(json.dumps(cfg))
    a, b = resolve_geometry(cfg).geometry, resolve_geometry(restored).geometry
    np.testing.assert_array_equal(a.inner_points, b.inner_points)
    np.testing.assert_array_equal(a.ath_rear_points, b.ath_rear_points)


@pytest.mark.parametrize("quadrants", ["1", "1234"])
@pytest.mark.parametrize("topology", ["acoustic", "legacy"])
def test_zero_front_offset_shares_the_rim(tmp_path, quadrants, topology):
    from hornlab_mesher.mesher import build_mesh
    cfg = config("of20")
    cfg["mesh"].update(quadrants=quadrants, topology=topology, athRearFrontOffset=0)
    r = resolve_geometry(cfg)
    np.testing.assert_array_equal(r.geometry.ath_rear_points[:, -1], r.geometry.inner_points[:, -1])
    mesh = meshio.read(build_mesh(r.geometry, r.density, tmp_path / "zero-front.msh", scale_to_metres=False))
    assert_watertight_after_symmetry(mesh, r.geometry)


@pytest.mark.parametrize("quadrants", ["1", "1234"])
def test_wall_zero_lip_builds_an_open_mouth(tmp_path, quadrants):
    from hornlab_mesher.mesher import build_mesh
    cfg = config("p1")
    cfg["mesh"]["quadrants"] = quadrants
    r = resolve_geometry(cfg)
    assert r.geometry.ath_rear_points is None
    path = build_mesh(r.geometry, r.density, tmp_path / "bare-lip.msh", scale_to_metres=False)
    mesh = meshio.read(path)
    triangles = np.concatenate([cell.data for cell in mesh.cells if cell.type == "triangle"])
    edges = build_edge_table(triangles)
    assert np.any(edges.count == 1)  # The authored lip rim remains open.
    assert edges.count.max() == 2


def test_full_ring_apex_uses_the_repeated_export_endpoint():
    cfg = config("coreabec")
    cfg["mesh"]["quadrants"] = "1234"
    g = resolve_geometry(cfg).geometry
    assert np.ptp(g.ath_rear_points[:, 0, 2]) > .1
    assert g.ath_rear_apex[2] == g.ath_rear_points[0, 0, 2]


@pytest.mark.parametrize("extra", [LIP + "ABEC.SimProfile = 0\n", "Mesh.RearShape = 3\nABEC.SimProfile = 0\n"])
def test_circular_symmetry_profile_refused(extra):
    with pytest.raises(ConfigError, match="CircSym / ABEC.SimProfile"):
        parse_text_config(BASE + extra)


@pytest.mark.parametrize("name", ["rs1", "ofull", "coreabec"])
def test_constructed_rear_vertices_match_ath(name):
    import gmsh
    from hornlab_mesher.builders.point_grid import build_point_grid
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("ath-import-rear-vertices")
        g = resolve_geometry(config(name)).geometry
        built = build_point_grid(g)
        actual = np.array([gmsh.model.getValue(0, tag, []) for tag in built.metadata["athRearVertexTags"]])
        sizes = np.asarray(gmsh.model.mesh.getSizes([(0, tag) for tag in built.metadata["athRearVertexTags"]]))
        expected_sizes = np.full(g.ath_rear_points.shape[:2], g.ath_rear_resolution_mm)
        if g.ath_rear_shape == 3:
            expected_sizes[:, -1] = 1.2 * g.ath_mouth_resolution_mm
        np.testing.assert_allclose(sizes, np.r_[expected_sizes.reshape(-1), g.ath_rear_resolution_mm], rtol=0, atol=1e-12)
        expected = np.loadtxt(FIXTURES / f"{name}-rear.csv", delimiter=";", skiprows=1)
        distance = np.linalg.norm(actual[:, None] - expected[None, :], axis=-1)
        assert distance.min(axis=1).max() <= 2e-3
        assert distance.min(axis=0).max() <= 2e-3
        assert built.axial_bounds_mm[1] == np.max(g.inner_points[..., 2])
        assert built.axial_bounds_mm[1] > g.inner_points[0, -1, 2]
    finally:
        gmsh.finalize()


def test_native_saved_default_rear_shape_is_still_inactive():
    header = "; Waveguide Generator geometry-interpretation: native-v1\n"
    a = resolve_geometry(parse_text_config(header + BASE)).geometry
    b = resolve_geometry(parse_text_config(header + BASE + "Mesh.RearShape = 1\n")).geometry
    assert type(a) is type(b)
    np.testing.assert_array_equal(a.inner_points, b.inner_points)


def test_parametric_area_forecast_measures_a_patch_not_its_box():
    import gmsh
    from hornlab_mesher.builders.point_grid_surfaces import _GeoSurfaceBuilder
    from hornlab_mesher.builders.point_grid_freestanding import _ath_import_surface_areas
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("area-forecast")
        builder = _GeoSurfaceBuilder()
        points = [builder.add_point(p) for p in ((0, 0, 0), (2, 0, 0), (2, 3, 0), (0, 3, 0))]
        face = builder.surface([builder.line_tags(points[i], points[(i+1) % 4]) for i in range(4)])
        gmsh.model.geo.synchronize()
        assert _ath_import_surface_areas([face])[face[1]] == pytest.approx(6, rel=0, abs=1e-9)
    finally:
        gmsh.finalize()


def test_approximate_surface_fit_refused_by_name():
    cfg = config("coreabec")
    cfg["mesh"]["surfaceFit"] = "approximate"
    with pytest.raises(ConfigError, match="Mesh.SurfaceFit = approximate"):
        resolve_geometry(cfg)


def test_sparse_imported_opus_transport_retains_ath_defaults():
    cfg = config("onr")
    for key in ("wallThickness", "athRearFrontOffset", "athRearStep"):
        cfg["mesh"].pop(key)
    params = build_geometry_params(cfg)[0]
    assert params["wallThickness"] == 5
    assert params["athRearFrontOffset"] == 40
    assert params["athRearStep"] == 25


def test_dictionary_zero_wall_clamp_is_reported(caplog):
    cfg = config("onr")
    cfg["mesh"]["wallThickness"] = 0
    params = build_geometry_params(cfg)[0]
    assert params["wallThickness"] == .1
    assert "clamps Mesh.WallThickness to 0.1 mm" in caplog.text


@pytest.mark.parametrize('name', ['p1', 'onr'])
@pytest.mark.parametrize('extra,key,section,value,named', [
    ('Scale = 2\n', 'scale', None, 2, 'Scale'),
    ('Throat.Ext.Length = 5\n', 'throatExtLength', 'profile', 5, 'Throat.Ext.Length'),
    ('Throat.Ext.Angle = 5\n', 'throatExtAngle', 'profile', 5, 'Throat.Ext.Angle'),
    ('Rot = 5\n', 'rot', 'profile', 5, 'Rot'),
    ('Mesh.VerticalOffset = 5\n', 'verticalOffset', 'mesh', 5, 'Mesh.VerticalOffset'),
    ('Source.Shape = 2\n', 'source_shape', 'source', 2, 'Source.Shape = 2'),
])
def test_unprobed_lip_and_opus_combinations_refused_in_both_transports(name, extra, key, section, value, named):
    text = (FIXTURES / f'{name}.cfg').read_text()
    # ATH V2026-09a m7_rot_in/m7_rot_both establish that OSSE.Rot is
    # ignored. Exercise active top-level Rot for the lip/Opus refusal.
    text += extra
    with pytest.raises(ConfigError) as error:
        parse_text_config(text)
    assert named in str(error.value)
    cfg = config(name)
    target = cfg if section is None else cfg.setdefault(section, {})
    target[key] = value
    for consumer in (build_geometry_params, resolve_geometry):
        with pytest.raises(ConfigError) as error:
            consumer(cfg)
        assert named in str(error.value)


def test_bare_opus_refusal_does_not_log_wall_clamp(caplog):
    cfg = config('onr')
    cfg['mode'] = 'bare'
    cfg['mesh']['wallThickness'] = 0
    caplog.clear()
    with pytest.raises(ConfigError, match='requires a free-standing wall'):
        resolve_geometry(cfg)
    assert 'clamps Mesh.WallThickness' not in caplog.text
