"""Captured ATH geometry, independent of this implementation's equations."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from hornlab_mesher.config_parser import ConfigError, parse_text_config
from hornlab_mesher.config_builder import build_geometry_params, resolve_geometry, build_from_config
from hornlab_mesher.profile_sampling import build_point_grid_arrays, ACOUSTIC_EXTENSION_SEGMENTS_KEY
from hornlab_mesher.profile_formulas import calculate_rosse, calculate_rosse_curve, rosse_axial_layout
from hornlab_mesher.text_import import TEXT_IMPORT_VERSION_KEY, TEXT_IMPORT_VERSION

ROOT = Path(__file__).parent / 'fixtures/ath_import/ath-v2026-09a/controls'
MANIFEST = json.loads((ROOT/'manifest.json').read_text())
CAPS = json.loads((ROOT/'source-caps.json').read_text())


def config(name):
    return parse_text_config((ROOT/f'{name}.cfg').read_text())


def reference(name):
    return np.array([[[float(v) for v in row.split(';')] for row in block.splitlines()]
                     for block in (ROOT/f'{name}.csv').read_text().strip().split('\n\n')])


def test_native_capture_integrity():
    for name, digest in MANIFEST['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize('name', MANIFEST['grids'])
@pytest.mark.parametrize('quadrants', ['1234', '1'])
def test_imported_wall_matches_ath(name, quadrants):
    cfg = config(name)
    cfg['mesh']['quadrants'] = quadrants
    actual = build_point_grid_arrays(build_geometry_params(cfg)[0])['inner_grid']
    expected = reference(name)
    if quadrants == '1':
        expected = expected[:len(expected)//4+1]
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=MANIFEST['grid_tolerance_mm'])


@pytest.mark.parametrize('name', ['r-rot10', 'r-rotneg', 'r-rotp', 'r-ext-rot',
                                  'r-ext-rot-seg4', 'r-slot-rot', 'r-ext-slot-rot-seg4'])
def test_rotated_rosse_scalar_and_vector_match_the_reference(name):
    params = build_geometry_params(config(name))[0]
    grid = build_point_grid_arrays(params)
    ts = np.asarray(grid['slice_map'])
    expected = reference(name)
    for i, phi in enumerate(grid['angle_list']):
        scalar = np.asarray([calculate_rosse(float(t), float(phi), params) for t in ts])
        vector = np.column_stack(calculate_rosse_curve(ts, float(phi), params))
        np.testing.assert_allclose(vector, scalar, rtol=0, atol=1e-12)
        np.testing.assert_allclose(vector[:,0], expected[i,:,2], rtol=0, atol=2e-4)
        np.testing.assert_allclose(vector[:,1], np.linalg.norm(expected[i,:,:2], axis=1), rtol=0, atol=2e-4)


def test_refinement_subdivides_the_authored_slot_join():
    params = build_geometry_params(config('r-ext-slot-rot-seg4'))[0]
    layout = rosse_axial_layout(params)
    # Last authored segment goes from 7.5 mm to the rotated, shifted body.
    t = 8.75/layout.full_length
    expected = reference('r-ext-slot-rot-seg4')[0,3:5].mean(axis=0)
    np.testing.assert_allclose(calculate_rosse(t,0,params), expected[[2,0]], rtol=0, atol=2e-4)
    refined = build_point_grid_arrays({**params, ACOUSTIC_EXTENSION_SEGMENTS_KEY: 8})['inner_grid']
    np.testing.assert_allclose(refined[0,7], expected, rtol=0, atol=2e-4)


@pytest.mark.parametrize('name', CAPS)
def test_automatic_and_explicit_source_caps_match_ath(name):
    from hornlab_mesher.builders.point_grid_sources import _source_cap_height, _source_cap_radius
    cfg = config(name)
    cfg['mode'] = 'bare'
    cfg['mesh'].update(wallThickness=0, topology='legacy')
    geometry = resolve_geometry(cfg).geometry
    ring = np.asarray(geometry.inner_points)[:,0]
    radius = float(np.mean(np.linalg.norm(ring[:,:2], axis=1)))
    height = _source_cap_height(radius, geometry)
    sphere_radius = _source_cap_radius(radius, geometry)
    sign = -1 if geometry.source_curv == -1 else 1
    pole = float(np.mean(ring[:,2])) + sign*height
    if CAPS[name].get('inconsistent_extension_pole'):
        # ATH's extension source pole is ten millimetres off its sphere.
        # Match its logged wavefront radius and retain a consistent cap.
        assert sphere_radius == pytest.approx(CAPS[name]['radius_mm'], abs=.002)
        assert abs(pole-CAPS[name]['pole_z_mm']) == pytest.approx(10,abs=.002)
        assert abs(abs(CAPS[name]['pole_z_mm']-CAPS[name]['centre_z_mm'])-sphere_radius) > 9.99
        return
    assert pole == pytest.approx(CAPS[name]['pole_z_mm'], abs=.002)
    assert sphere_radius == pytest.approx(CAPS[name]['radius_mm'], abs=.002)
    assert pole-sign*sphere_radius == pytest.approx(CAPS[name]['centre_z_mm'], abs=.002)


@pytest.mark.parametrize('lod', ['coarse', 'fine'])
@pytest.mark.parametrize('name', ['o-source-geo-z2','r-source-geo-rot'])
def test_preview_keeps_authored_source_curvature(lod, name):
    from hornlab_mesher.preview import build_preview_geometry, PreviewOptionsV1
    cfg=config(name)
    cfg['mode']='bare'
    cfg['mesh']['wallThickness']=0
    preview=build_preview_geometry(cfg, PreviewOptionsV1(lod=lod,include_outer=False,
        include_enclosure=False,include_rear_cap=False,include_curvature=False))
    assert preview.metadata['source_cap_radius_mm'] == pytest.approx(CAPS[name]['radius_mm'], abs=.002)
    source=next(s for s in preview.surfaces if s.role=='source_cap')
    assert source.positions[0,2] == pytest.approx(CAPS[name]['pole_z_mm'], abs=.002)


@pytest.mark.parametrize('value', ['0','-1','4.5','nan','inf','4097','4.0000000000000001','1e1000'])
def test_invalid_extension_count_fails_before_sampling(value):
    text=(ROOT/'o-ext.cfg').read_text()+f'\nMesh.ThroatExtSegments = {value}\n'
    with pytest.raises(ConfigError,match='ThroatExtSegments'):
        parse_text_config(text)


def test_unqualified_old_osse_extension_sampling_is_refused():
    with pytest.raises(ConfigError,match='current ATH OSSE sampling'):
        parse_text_config('; Ath version V2025-06\n'+(ROOT/'o-ext-seg4.cfg').read_text())


@pytest.mark.parametrize('rotation', ['nan','inf','1e1000','1e309*sin(p)'])
def test_nonfinite_rosse_rotation_has_a_named_diagnostic(rotation):
    cfg=parse_text_config((ROOT/'r-base.cfg').read_text()+f'Rot = {rotation}\n')
    with pytest.raises(ValueError,match='Rot.*finite'):
        build_point_grid_arrays(build_geometry_params(cfg)[0])


@pytest.mark.parametrize('header', ['; Parameter config\n',
    '; Waveguide Generator geometry-interpretation: ath-2026-08c-v1\n'])
@pytest.mark.parametrize('key', ['Scale.Z = 2','Mesh.ThroatExtSegments = 4','Rot = 10'])
def test_existing_native_and_v1_interpretations_keep_their_boundary(header,key):
    with pytest.raises(ConfigError):
        parse_text_config(header+(ROOT/'r-base.cfg').read_text()+key+'\n')


@pytest.mark.parametrize('key,value', [('athAxialScale',0),('athAxialScale',float('inf'))])
def test_dictionary_transport_validates_scale(key,value):
    cfg=config('o-z2');cfg[key]=value
    with pytest.raises(ConfigError,match='Scale.Z'):
        build_geometry_params(cfg)


def test_legacy_throat_segment_spelling_is_not_an_alias():
    with pytest.raises(ConfigError,match='Mesh.ThroatSegments'):
        parse_text_config((ROOT/'o-ext.cfg').read_text()+'Mesh.ThroatSegments = 4\n')


@pytest.mark.parametrize('name', ['o-z2', 'r-rot10', 'o-ext-seg4'])
def test_controls_reach_mesh_and_reopened_step(tmp_path,name):
    import gmsh
    from hornlab_mesher.cad import write_step_from_config
    from hornlab_mesher.mesher import load_mesh
    cfg=config(name)
    cfg['mode']='bare'
    cfg['mesh'].update(wallThickness=0,throatResolution=6,mouthResolution=20,maxTriangles=100000)
    expected=reference(name)
    built=build_from_config(cfg,tmp_path/'wall.msh')
    mesh=load_mesh(built.mesh_path)
    assert mesh.n_triangles>0
    assert built.physical_groups[2]=='SD1D1001'
    path,info=write_step_from_config(cfg,tmp_path/'wall.step')
    assert info.body=='surface'
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber('General.Terminal',0)
        gmsh.model.occ.importShapes(str(path),highestDimOnly=False)
        gmsh.model.occ.synchronize()
        faces=gmsh.model.getEntities(2)
        assert faces
        for point in expected[::4,::4].reshape(-1,3):
            distance=min(np.linalg.norm(np.asarray(gmsh.model.getClosestPoint(2,tag,point)[0])-point) for _,tag in faces)
            assert distance<.15
    finally:
        gmsh.finalize()
