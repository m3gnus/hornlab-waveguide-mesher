"""ATH V2026-09a import witnesses and native authoring boundaries."""
from pathlib import Path
import json

import meshio
import numpy as np
import pytest

from hornlab_mesher.config_parser import ConfigError, parse_text_config
from hornlab_mesher.config_builder import build_geometry_params, resolve_geometry, build_from_config
from hornlab_mesher.profile_sampling import build_point_grid_arrays
from hornlab_mesher.profile_formulas import calculate_osse, calculate_osse_curve
from hornlab_mesher.preview import build_preview_geometry, PreviewOptionsV1
from hornlab_mesher.text_import import TEXT_IMPORT_VERSION_KEY, TEXT_IMPORT_V1, TEXT_IMPORT_VERSION

FIXTURES = Path(__file__).parent / 'fixtures/ath_import/ath-v2026-09a'
GRID_CASES = ['m3_rosse_nomorph', 'm4_enc', 'm5_osse_omit_a_s', 'm5_rosse_omit_all',
              'm6_var_wall', 'm7_rot_none', 'm7_rot_in', 'm7_rot_top', 'm7_rot_both', 'm10_ext_rot']


def config(name):
    return parse_text_config((FIXTURES / f'{name}.cfg').read_text())


def reference(name):
    return np.array([[[float(v) for v in row.split(';')] for row in block.splitlines()]
                     for block in (FIXTURES / f'{name}.csv').read_text().strip().split('\n\n')])


@pytest.mark.parametrize('name', GRID_CASES)
@pytest.mark.parametrize('quadrants', ['1', '1234'])
def test_grid_matches_ath_v2026_09a(name, quadrants):
    cfg = config(name)
    cfg['mesh']['quadrants'] = quadrants
    actual = build_point_grid_arrays(build_geometry_params(cfg)[0])['inner_grid']
    expected = reference(name)
    if quadrants == '1':
        expected = expected[:len(expected)//4 + 1]
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-4)


def test_rosse_morph_refusal_names_measured_radial_change():
    # ATH V2026-09a does change these radii: morph cannot be ignored.
    plain, morphed = reference('m3_rosse_nomorph'), reference('m3_rosse_morph')
    np.testing.assert_allclose(np.linalg.norm(plain[:, -1, :2], axis=1), 150, rtol=0, atol=2e-4)
    radius = np.linalg.norm(morphed[:, -1, :2], axis=1)
    assert radius.min() == pytest.approx(160, abs=2e-4)
    assert radius.max() == pytest.approx(217.766142, abs=2e-4)
    with pytest.raises(ConfigError, match=r'R-OSSE with Morph.TargetShape.*radial profile'):
        config('m3_rosse_morph')
    cfg = config('m3_rosse_nomorph')
    cfg['morph'] = {'morphTarget': 1, 'morphWidth': 320, 'morphHeight': 320, 'morphCorner': 20}
    with pytest.raises(ConfigError, match=r'R-OSSE with Morph.TargetShape'):
        resolve_geometry(cfg)
    cfg.pop(TEXT_IMPORT_VERSION_KEY)
    assert build_point_grid_arrays(build_geometry_params(cfg)[0])['inner_grid'].shape == plain.shape


@pytest.mark.parametrize('name', GRID_CASES + ['m3_rosse_morph'])
def test_normalised_ath_v2026_09a_fixtures(name):
    for suffix in ['cfg', 'csv']:
        data = (FIXTURES / f'{name}.{suffix}').read_bytes()
        assert b'\r' not in data
        assert data.endswith(b'\n') and not data.endswith(b'\n\n')


@pytest.mark.parametrize('name', GRID_CASES)
def test_windows_crlf_cfg_text_imports_identically(name):
    # A Windows user's ATH .cfg has CRLF; text reaching parse_text_config
    # without universal-newline translation must import the same geometry.
    text = (FIXTURES / f'{name}.cfg').read_bytes().decode('utf-8')
    lf, crlf = parse_text_config(text), parse_text_config(text.replace('\n', '\r\n'))
    assert crlf == lf
    grids = [build_point_grid_arrays(build_geometry_params(cfg)[0])['inner_grid'] for cfg in (lf, crlf)]
    assert grids[0].shape == reference(name).shape
    np.testing.assert_array_equal(grids[1], grids[0])


def test_import_spacing_matches_ath_v2026_09a_enclosure_bounds():
    from hornlab_mesher.builders.enclosure import enclosure_box_bounds
    cfg = config('m4_enc')
    geometry = resolve_geometry(cfg).geometry
    points = np.loadtxt(FIXTURES / 'm4_enc-enclosure.csv', delimiter=';', skiprows=1)
    assert (points[:, 0].min(), points[:, 0].max()) == (-196, 186)
    assert (geometry.enclosure.space_l_mm, geometry.enclosure.space_r_mm) == (35, 25)
    # Native left/right margins are -x/+x, including saved native text.
    native_text = '; Waveguide Generator geometry-interpretation: native-v1\n' + (FIXTURES / 'm4_enc.cfg').read_text()
    native = parse_text_config(native_text)
    assert (native['enclosure']['space_l_mm'], native['enclosure']['space_r_mm']) == (25, 35)
    result = enclosure_box_bounds(geometry.inner_points, geometry.enclosure, closed=True)
    assert result['bx0'] == pytest.approx(points[:, 0].min(), abs=2e-3)
    assert result['bx1'] == pytest.approx(points[:, 0].max(), abs=2e-3)


@pytest.mark.parametrize('name,expected', [
    ('m5_osse_omit_a_s', {'a': 45, 's': .5}),
    ('m5_rosse_omit_all', {'R': 200, 'a': 40, 'a0': 0, 'r': .25, 'm': .9, 'b': .4, 'q': 3.5}),
])
def test_sparse_import_defaults_and_native_defaults(name, expected):
    cfg = config(name)
    for key in expected:
        cfg['profile'].pop(key, None)
    params = build_geometry_params(cfg)[0]
    for key, value in expected.items():
        assert params[key] == value
    actual = build_point_grid_arrays(params)['inner_grid']
    np.testing.assert_allclose(actual, reference(name), rtol=0, atol=2e-4)
    cfg.pop(TEXT_IMPORT_VERSION_KEY)
    native = build_geometry_params(cfg)[0]
    assert native['a'] == 60 and native['a0'] == (10 if name.startswith('m5_osse') else 15.5)
    if name.startswith('m5_rosse'):
        assert native['R'] == 150 and native['q'] == 1
        assert not any(key in native for key in ['r', 'm', 'b'])
    else:
        assert native['s'] == 0


def test_osse_block_rotation_is_ignored_with_named_warning(caplog):
    assert config('m7_rot_in') == config('m7_rot_none')
    assert config('m7_rot_both') == config('m7_rot_top')
    assert 'ignoring OSSE.Rot = 10' in caplog.text
    assert 'ignoring Rot = 5' not in caplog.text
    for suffix in ['in', 'both']:
        other = 'none' if suffix == 'in' else 'top'
        np.testing.assert_array_equal(reference('m7_rot_' + suffix), reference('m7_rot_' + other))


@pytest.mark.parametrize('literal', ['12.7', '+12.7', '-12.7', '1.27e1', '.127E+2', '12.'])
def test_literal_osse_block_radius_is_accepted(literal):
    text = (FIXTURES / 'm7_rot_none.cfg').read_text().replace('r0 = 12.7', 'r0 = ' + literal)
    assert parse_text_config(text)['profile']['r0'] == float(literal)


@pytest.mark.parametrize('expression', ['25.4/2', '(12.7)', '2*6.35', '20+4*cos(p)^2'])
@pytest.mark.parametrize('prefix', ['', '; Waveguide Generator geometry-interpretation: ath-2026-09a-v2\n'])
def test_nonliteral_osse_block_radius_is_refused_by_name(expression, prefix):
    text = (FIXTURES / 'm7_rot_none.cfg').read_text().replace('r0 = 12.7', 'r0 = ' + expression)
    with pytest.raises(ConfigError, match=r'OSSE.r0.*literal number'):
        parse_text_config(prefix + text)
    saved = parse_text_config('; Waveguide Generator geometry-interpretation: native-v1\n' + text)
    assert saved['profile']['r0'] == expression
    rosse = (FIXTURES / 'm3_rosse_nomorph.cfg').read_text().replace('r0 = 12.7', 'r0 = ' + expression)
    assert parse_text_config(rosse)['profile']['r0'] == expression


def test_extension_rotation_scalar_vector_and_ath_stations():
    params = build_geometry_params(config('m10_ext_rot'))[0]
    stations = np.array([0, 1, 9, 10, 11, 80, 130.])
    vector = np.column_stack(calculate_osse_curve(stations, 0, params))
    scalar = np.array([calculate_osse(z, 0, params) for z in stations])
    np.testing.assert_allclose(vector, scalar, rtol=0, atol=1e-12)
    unrotated = dict(params, rot=0)
    np.testing.assert_array_equal(vector[:4], np.array([calculate_osse(z, 0, unrotated) for z in stations[:4]]))
    np.testing.assert_array_equal(reference('m10_ext_rot')[0, :11, 2], np.arange(11))


@pytest.mark.parametrize('quadrants', ['1', '14', '1234'])
@pytest.mark.parametrize('topology', ['acoustic', 'legacy', 'faceted'])
def test_warped_rear_mesh_and_preview_share_measured_ring(tmp_path, quadrants, topology):
    cfg = config('m6_var_wall')
    cfg['mesh'].update(quadrants=quadrants, topology='legacy' if topology == 'faceted' else topology,
                       preserve_grid=topology == 'faceted', max_triangles=100000)
    geometry = resolve_geometry(cfg).geometry
    ring = geometry.ath_warped_rear_points
    assert np.ptp(ring[:, 2]) == pytest.approx(.258, abs=2e-3)
    expected = np.loadtxt(FIXTURES / 'm6_var_wall-rear.csv', delimiter=';', skiprows=1)
    if quadrants == '1':
        expected = expected[(expected[:, 0] >= 0) & (expected[:, 1] >= 0)]
    elif quadrants == '14':
        expected = expected[expected[:, 0] >= 0]
    # The shell's denser meridians interpolate the authored ATH rear ring.
    from hornlab_mesher.rear_compatibility import resample_rear_ring
    actual = resample_rear_ring(ring, expected, closed=geometry.closed)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-3)
    preview = build_preview_geometry(cfg, PreviewOptionsV1(lod='coarse'))
    cap = next(s for s in preview.surfaces if s.role == 'wall.rear_cap')
    assert np.ptp(cap.positions[:, 2]) > .25
    np.testing.assert_array_equal(cap.positions[-1], geometry.ath_warped_rear_apex)
    result = build_from_config(cfg, tmp_path / 'warped.msh')
    mesh = meshio.read(result.mesh_path)
    triangles = np.concatenate([cell.data for cell in mesh.cells if cell.type == 'triangle'])
    from hornlab_mesher.mesher import _validate_closed_shell_contract
    physical = np.concatenate([mesh.cell_data['gmsh:physical'][i] for i,cell in enumerate(mesh.cells) if cell.type == 'triangle'])
    _validate_closed_shell_contract(mesh.points, triangles, physical,
        symmetry_snap_axes=geometry.symmetry_planes, symmetry_snap_tol_mm=1e-6, is_open_shell=False)


@pytest.mark.parametrize('suffix', ['none', 'in', 'top', 'both'])
def test_rotation_matches_ath_v2025_12(suffix):
    root = FIXTURES.parent / 'ath-v2025-12'
    name = 'm7_rot_' + suffix
    cfg = parse_text_config((root / f'{name}.cfg').read_text())
    expected = np.array([[[float(v) for v in row.split(';')] for row in block.splitlines()]
                         for block in (root / f'{name}.csv').read_text().strip().split('\n\n')])
    np.testing.assert_allclose(build_point_grid_arrays(build_geometry_params(cfg)[0])['inner_grid'], expected, rtol=0, atol=2e-4)


def test_warped_rear_step_is_a_reopenable_solid(tmp_path):
    import gmsh
    from hornlab_mesher.cad import write_step_from_config
    cfg = config('m6_var_wall')
    path, info = write_step_from_config(cfg, tmp_path / 'warped.step')
    assert info.body == 'solid' and info.throat_opened
    geometry = resolve_geometry(cfg).geometry
    expected = geometry.ath_warped_rear_points[::len(geometry.ath_warped_rear_points)//4]
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.add('warped-step-reopened')
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(3)
        assert len(volumes) == 1
        assert gmsh.model.occ.getMass(*volumes[0]) > 0
        vertices = np.array([gmsh.model.getValue(0, tag, []) for _,tag in gmsh.model.getEntities(0)])
        # Cardinal rear seams retain their measured coordinates after sewing,
        # opening the bore and STEP round trip. OCC control-pole boxes are
        # conservative bounds, not extrema of the actual curved faces.
        distance = np.linalg.norm(expected[:, None]-vertices[None, :], axis=-1)
        assert distance.min(axis=1).max() <= 2e-3
    finally:
        gmsh.finalize()


@pytest.mark.parametrize('quadrants', ['1', '14', '1234'])
def test_warped_preview_vertical_offset_is_a_rigid_translation(quadrants):
    cfg = config('m6_var_wall')
    cfg['mesh']['quadrants'] = quadrants
    original = build_preview_geometry(cfg, PreviewOptionsV1(lod='coarse'))
    cfg['mesh']['verticalOffset'] = 20
    shifted = build_preview_geometry(cfg, PreviewOptionsV1(lod='coarse'))
    for role in ['wall.rear_cap', 'wall.rear_return']:
        a = next(s for s in original.surfaces if s.role == role)
        b = next(s for s in shifted.surfaces if s.role == role)
        np.testing.assert_allclose(b.positions, a.positions + [0,20,0], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(b.indices, a.indices)


@pytest.mark.parametrize('topology', ['acoustic', 'legacy'])
def test_warped_rear_preserves_analytic_body_and_outer_shell(topology, monkeypatch):
    import hornlab_mesher.config_builder as builder
    from hornlab_mesher.rear_compatibility import freestanding_rear_ring
    cfg = config('m6_var_wall')
    cfg['mesh']['topology'] = topology
    warped = resolve_geometry(cfg).geometry
    monkeypatch.setattr(builder, 'freestanding_rear_ring',
                        lambda inner, outer, wall, **kw: freestanding_rear_ring(inner, outer, wall))
    monkeypatch.setattr(builder, '_certify_imported_rear_route_agreement', lambda *a: None)
    body = resolve_geometry(cfg).geometry
    np.testing.assert_array_equal(warped.inner_points, body.inner_points)
    np.testing.assert_array_equal(warped.outer_points, body.outer_points)



def test_warped_rear_options_against_ath_v2026_09a_at_matching_azimuths():
    from hornlab_mesher.rear_compatibility import resample_rear_ring
    from hornlab_mesher.builders.point_grid_freestanding import _restored_outer_throat_points
    geometry = resolve_geometry(config('m6_var_wall')).geometry
    expected = np.loadtxt(FIXTURES / 'm6_var_wall-rear.csv', delimiter=';', skiprows=1)
    analytic = _restored_outer_throat_points(geometry.inner_points, geometry.outer_points,
                                           wall_thickness_mm=geometry.wall_thickness_mm)[:, 0].copy()
    analytic[:, 2] -= geometry.wall_thickness_mm
    a = resample_rear_ring(analytic, expected, closed=True)
    b = resample_rear_ring(geometry.ath_warped_rear_points, expected, closed=True)
    assert np.linalg.norm(a-expected, axis=1).max() == pytest.approx(.398502347, abs=2e-3)
    assert np.linalg.norm(b-expected, axis=1).max() <= 2e-3


V1_CASES = ['m4_enc', 'm5_osse_omit_a_s', 'm5_rosse_omit_all', 'm6_var_wall',
            'm7_rot_in', 'm7_rot_both', 'm10_ext_rot', 'm3_rosse_morph']


def saved_config(name, version=TEXT_IMPORT_V1):
    return parse_text_config('; Waveguide Generator geometry-interpretation: ' + version + '\n'
                             + (FIXTURES / f'{name}.cfg').read_text())


def assert_v1_grid(cfg, key):
    from test_rear_import_compatibility import assert_pinned
    grid = build_point_grid_arrays(build_geometry_params(cfg)[0])['inner_grid']
    assert_pinned(grid, 'ath-mismatches-v1.npz', key)


@pytest.mark.parametrize('name', V1_CASES)
@pytest.mark.parametrize('roundtrip', [False, True])
def test_saved_v1_mismatch_geometry_matches_pinned_contract(name, roundtrip):
    # a515334 geometry, with Term.s=0.5 explicitly supplied for the omitted-s
    # case: that separately qualified default precedes the v2 coefficient rules.
    cfg = saved_config(name)
    if roundtrip:
        cfg = json.loads(json.dumps(cfg))
    assert cfg[TEXT_IMPORT_VERSION_KEY] == TEXT_IMPORT_V1
    assert_v1_grid(cfg, name + '/grid')
    if name == 'm6_var_wall':
        # v1 could not build/save a warped wall through this resolver. Keep
        # its grid and refusal; only v2 has a qualified warped rear cap.
        with pytest.raises(ConfigError, match='nonplanar rear ring'):
            resolve_geometry(cfg)
        with pytest.raises(ValueError, match='nonplanar rear ring'):
            build_preview_geometry(cfg, PreviewOptionsV1(lod='coarse'))
        return
    # These profile-only ATH probes explicitly disable their wall.
    if cfg.get('mesh', {}).get('wallThickness') == 0:
        cfg['mode'] = 'bare'
    geometry = resolve_geometry(cfg).geometry
    from test_rear_import_compatibility import assert_pinned
    for field in ('inner', 'outer'):
        assert_pinned(getattr(geometry, field + '_points'), 'ath-mismatches-v1.npz', name + '/' + field)
    if name == 'm4_enc':
        from hornlab_mesher.builders.enclosure import enclosure_box_bounds
        assert (geometry.enclosure.space_l_mm, geometry.enclosure.space_r_mm) == (25, 35)
        bounds = enclosure_box_bounds(geometry.inner_points, geometry.enclosure, closed=True)
        assert_pinned(np.array([bounds[k] for k in ['bx0', 'bx1', 'by0', 'by1']]),
                      'ath-mismatches-v1.npz', name + '/bounds')


@pytest.mark.parametrize('name,keys', [
    ('m5_osse_omit_a_s', ['a', 's']),
    ('m5_rosse_omit_all', ['R', 'a', 'a0', 'r', 'm', 'b', 'q']),
])
def test_sparse_v1_import_defaults_match_pinned_contract(name, keys):
    cfg = saved_config(name)
    for key in keys:
        cfg['profile'].pop(key, None)
    assert_v1_grid(cfg, name + '/sparse-grid')


@pytest.mark.parametrize('expression', ['25.4/2', '(12.7)', '2*6.35', '20+4*cos(p)^2'])
@pytest.mark.parametrize('roundtrip', [False, True])
def test_saved_v1_nonliteral_osse_radius_matches_pinned_contract(expression, roundtrip):
    text = (FIXTURES / 'm7_rot_none.cfg').read_text().replace('r0 = 12.7', 'r0 = ' + expression)
    cfg = parse_text_config('; Waveguide Generator geometry-interpretation: ' + TEXT_IMPORT_V1 + '\n' + text)
    if roundtrip:
        cfg = json.loads(json.dumps(cfg))
    assert cfg['profile']['r0'] == expression
    assert_v1_grid(cfg, 'r0/' + expression)


@pytest.mark.parametrize('name', GRID_CASES)
def test_saved_v2_cfg_keeps_measured_ath_grid(name):
    cfg = saved_config(name, TEXT_IMPORT_VERSION)
    grid = build_point_grid_arrays(build_geometry_params(cfg)[0])['inner_grid']
    np.testing.assert_allclose(grid, reference(name), rtol=0, atol=2e-4)


@pytest.mark.parametrize('name', ['m7_rot_in', 'm7_rot_both'])
def test_only_v2_warns_that_osse_block_rotation_is_ignored(name, caplog):
    saved_config(name)
    assert 'ignoring OSSE.Rot' not in caplog.text
    saved_config(name, TEXT_IMPORT_VERSION)
    assert 'ignoring OSSE.Rot = 10' in caplog.text


def test_saved_v1_extension_rotation_scalar_and_vector_match_pinned_contract():
    from test_rear_import_compatibility import assert_pinned
    params = build_geometry_params(saved_config('m10_ext_rot'))[0]
    stations = np.array([0, 1, 9, 10, 11, 80, 130.])
    scalar = np.array([[calculate_osse(z, p, params) for z in stations] for p in [0, .3, np.pi/2]])
    vector = np.array([np.column_stack(calculate_osse_curve(stations, p, params)) for p in [0, .3, np.pi/2]])
    assert_pinned(scalar, 'ath-mismatches-v1.npz', 'm10_ext_rot/scalar')
    assert_pinned(vector, 'ath-mismatches-v1.npz', 'm10_ext_rot/vector')


@pytest.mark.parametrize('version', [TEXT_IMPORT_V1, TEXT_IMPORT_VERSION])
@pytest.mark.parametrize('echo', ['V2025-06', 'V2026-09a'])
def test_rosse_morph_refusal_depends_on_geometry_stamp_not_sampling_echo(version, echo):
    text = ('; Ath version ' + echo + '\n; Waveguide Generator geometry-interpretation: ' + version + '\n'
            + (FIXTURES / 'm3_rosse_morph.cfg').read_text())
    if version == TEXT_IMPORT_VERSION:
        with pytest.raises(ConfigError, match=r'R-OSSE with Morph.TargetShape'):
            parse_text_config(text)
    else:
        assert_v1_grid(parse_text_config(text), 'm3_rosse_morph/grid')
