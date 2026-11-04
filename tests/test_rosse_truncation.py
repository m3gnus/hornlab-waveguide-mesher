"""ATH V2026-09a throat cuts, independently witnessed by exported grids."""
from pathlib import Path

import numpy as np
import pytest

from hornlab_mesher.config_builder import build_geometry_params, resolve_geometry
from hornlab_mesher.config_parser import parse_text_config
from hornlab_mesher.profile_formulas import calculate_rosse, calculate_rosse_curve
from hornlab_mesher.profile_sampling import build_point_grid_arrays

FIXTURES = Path(__file__).parent / 'fixtures/ath_import/ath-v2026-09a/rosse-trunc'
CASES = ['trunc-10-default', 'trunc-10-linear', 'trunc-10-nonlinear',
         'trunc-41-stretch-linear', 'trunc-two-roots']


def reference(name):
    return np.array([[[float(v) for v in line.split(';')] for line in block.splitlines()]
                     for block in (FIXTURES / f'{name}.csv').read_text().strip().split('\n\n')])


def config(name='trunc-10-linear'):
    return parse_text_config((FIXTURES / f'{name}.cfg').read_text())


@pytest.mark.parametrize('name', CASES)
def test_throat_cut_matches_every_ath_station_and_azimuth(name):
    cfg = config(name)
    params = build_geometry_params(cfg)[0]
    actual = build_point_grid_arrays(params)['inner_grid']
    expected = reference(name)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=.001)
    # Source stays a flat disc at the enlarged, cut throat.
    geometry = resolve_geometry(cfg).geometry
    np.testing.assert_allclose(np.linalg.norm(geometry.inner_points[:, 0, :2], axis=1),
                               actual[0, 0, 0], rtol=0, atol=1e-12)
    assert np.max(np.abs(geometry.inner_points[:, 0, 2])) < 1e-12
    assert geometry.source_shape == 0
    assert np.max(np.abs(actual[:, 0, 2])) < 1e-12


def test_scalar_array_and_refined_consumers_share_the_same_cut():
    params = build_geometry_params(config())[0]
    stations = np.linspace(0, 1, 109)
    np.testing.assert_allclose(np.column_stack(calculate_rosse_curve(stations, 0, params)),
                               [calculate_rosse(t, 0, params) for t in stations], rtol=0, atol=1e-12)
    refined = build_point_grid_arrays(dict(params, lengthSegments=128))['inner_grid']
    np.testing.assert_allclose(refined[:, [0, -1]], reference('trunc-10-linear')[:, [0, -1]],
                               rtol=0, atol=.001)


def test_old_axial_rule_cannot_acquire_the_new_cut():
    cfg = config()
    cfg['_athAxialSamplingRule'] = 'ath-v2025-06'
    with pytest.raises(ValueError, match='current ATH axial sampling rule'):
        build_geometry_params(cfg)


def test_tmax_failure_stays_refused_instead_of_importing_aths_inconsistent_fallback():
    with pytest.raises(ValueError, match=r'R-OSSE.trunc.*tmax'):
        build_geometry_params(config('trunc-tmax08'))


@pytest.mark.parametrize('change,setting', [
    ({'scale': 2}, 'Scale'), ({'athAxialScale': 2}, 'Scale.Z'),
    ({'profile': {'rot': 3}}, 'Rot'), ({'profile': {'throatExtLength': 10}}, 'Throat.Ext.Length'),
    ({'profile': {'slotLength': 10}}, 'Slot.Length'),
    ({'source': {'sourceShape': 1}}, 'Source.Shape'),
    ({'mesh': {'wallThickness': 5}}, 'Mesh.WallThickness'),
    ({'morph': {'morphTarget': 2}}, 'Morph'),
    ({'cross_section': {'exponent': 4}}, 'cross-section'),
    ({'profile': {'a': '40 + p'}}, 'scalar'),
])
def test_unqualified_compositions_refuse_with_a_remedy(change, setting):
    cfg = config()
    for key, value in change.items():
        if isinstance(value, dict):
            cfg.setdefault(key, {}).update(value)
        else:
            cfg[key] = value
    if 'wallThickness' in change.get('mesh', {}):
        cfg['mode'] = 'freestanding'
    with pytest.raises(ValueError, match=setting.replace('.', r'\.')) as caught:
        build_geometry_params(cfg)
    assert 'Remove' in str(caught.value)


def test_legacy_dictionary_trunc_is_not_reinterpreted():
    cfg = config()
    cfg.pop('athRosseTruncMm')
    base = build_point_grid_arrays(build_geometry_params(cfg)[0])['inner_grid']
    cfg['profile']['trunc'] = 10
    actual = build_point_grid_arrays(build_geometry_params(cfg)[0])['inner_grid']
    assert actual.tobytes() == base.tobytes()


def test_cut_reaches_preview_mesh_and_step(tmp_path):
    import gmsh

    from hornlab_mesher.cad import write_step_from_config
    from hornlab_mesher.config_builder import build_from_config
    from hornlab_mesher.mesher import load_mesh
    from hornlab_mesher.preview import build_preview_geometry

    cfg = config()
    assert cfg['mode'] == 'bare'
    cfg['mesh'].update(throatResolution=8, mouthResolution=20, maxTriangles=50000)
    expected = reference('trunc-10-linear')
    preview = build_preview_geometry(cfg)
    assert any(surface.role == 'horn.inner' for surface in preview.surfaces)
    built = build_from_config(cfg, tmp_path / 'cut.msh')
    info = load_mesh(built.mesh_path)
    assert info.n_triangles > 0
    assert 2 in info.physical_groups
    import meshio

    mesh = meshio.read(built.mesh_path)
    source_faces = np.vstack([cell.data[tags == 2] for cell, tags in
                             zip(mesh.cells, mesh.cell_data['gmsh:physical']) if cell.type == 'triangle'])
    source_points = mesh.points[np.unique(source_faces)]
    assert len(source_points) > 3
    assert np.max(np.abs(source_points[:, 2])) < 1e-12
    unit = 1000 if info.units == 'm' else 1
    assert np.linalg.norm(source_points[:, :2], axis=1).max()*unit == pytest.approx(expected[0, 0, 0], abs=.001)
    path, info = write_step_from_config(cfg, tmp_path / 'cut.step')
    assert info.body == 'surface'
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.occ.importShapes(str(path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        faces = gmsh.model.getEntities(2)
        for point in expected[::2, ::8].reshape(-1, 3):
            assert min(np.linalg.norm(np.asarray(gmsh.model.getClosestPoint(2, tag, point)[0])-point)
                       for _, tag in faces) < .15
    finally:
        gmsh.finalize()
