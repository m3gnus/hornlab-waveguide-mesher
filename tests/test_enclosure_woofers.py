"""Production assembly admission and small, provenance-bearing ATH references."""
from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace
import json
import math
from pathlib import Path

import meshio
import numpy as np
import pytest

from hornlab_mesher import build_from_config
from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.edges import build_edge_table
from hornlab_mesher.enclosure_woofers import (
    EnclosureWoofer, PistonShape, certify_arrays,
)
from hornlab_mesher.mesher import MesherError
from hornlab_mesher.normals import validate_orientation

FIXTURES = Path(__file__).parent/'fixtures'/'assembly_woofers'
BASELINE = json.loads((FIXTURES/'horn_only.json').read_text())
REFERENCE = json.loads((FIXTURES/'ath_reference.json').read_text())
CASES = tuple(BASELINE['cases'])


def assembly_config(case='osse', shape='piston', recess=0.):
    config = copy.deepcopy(BASELINE['cases'][case]['config'])
    mouth = resolve_geometry(config).geometry.inner_points[:, -1]
    config['enclosure']['woofers'] = [
        {'radiator_id':'lf-b', 'x_mm':0., 'y_mm':float(mouth[:,1].min())-65,
         'radius_mm':35., 'shape':{'kind':shape, **({'recess_mm':recess} if shape=='piston' else {})}},
        {'radiator_id':'lf-a', 'x_mm':25., 'y_mm':float(mouth[:,1].max())+65,
         'radius_mm':25., 'shape':{'kind':shape, **({'recess_mm':recess} if shape=='piston' else {})}},
    ]
    return config


def arrays(result):
    mesh = meshio.read(result.mesh_path)
    return (mesh, np.asarray(mesh.points), mesh.cells_dict['triangle'],
            mesh.cell_data_dict['gmsh:physical']['triangle'],
            mesh.cell_data_dict['gmsh:geometrical']['triangle'])


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('shape,recess', [('piston',0.),('piston',2.),('ath_cone',0.)])
def test_formula_assembly(tmp_path, case, shape, recess):
    config = assembly_config(case, shape, recess)
    result = build_from_config(config, tmp_path/'assembly.msh')
    mesh, points, triangles, tags, surfaces = arrays(result)
    report = validate_orientation(points, triangles, tags, require_watertight=True,
                                  require_edge_consistency=True, require_positive_volume=True)
    assert report.watertight and report.signed_volume > 0
    assert result.metadata['tag_namespace'] == 'wg-assembly-v1'
    assert result.metadata['rigid_tag'] == 1
    segments = ('piston',) if shape=='piston' else ('cap','cone')
    source_tags = result.metadata['source_tags']
    assert set(source_tags) == {'horn-throat', *(f'{w}/{s}' for w in ('lf-a','lf-b') for s in segments)}
    assert set(tags) == {1, *source_tags.values()}
    certify_arrays(points, triangles, tags, surfaces, result.metadata['assembly'])
    records = result.metadata['assembly']['patches']
    assert len(records) == 2*len(segments)
    for patch in records:
        tag = source_tags[patch['patch_id']]
        assert patch['physical_tag'] == tag
        assert result.metadata['tag_map'][str(tag)] == {
            'source_id':patch['patch_id'], 'instance_id':None, 'role':'secondary'}
        name = f"wg-assembly-v1|tag={tag}|source_id={patch['patch_id']}|instance_id=null|role=secondary"
        assert mesh.field_data[name].tolist() == [tag,2]
        xyz = points[triangles[tags==tag]]
        normal = np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
        assert np.all(normal[:,2] > 0)
        if shape=='piston':
            assert np.allclose(xyz[:,:,2], result.metadata['assemblyBaffleZMm']-recess, atol=1e-7, rtol=0)
            assert np.linalg.norm(normal,axis=1).sum()/2 == pytest.approx(math.pi*patch['radius_mm']**2, rel=.005)
    if case=='freeform-corners':
        assert config['enclosure']['edge_mm'] == 18
        assert result.metadata['acousticEnclosureEdgeSuppressed'] is False
    # A missing source must still fail the shared closure guard.
    from hornlab_mesher.mesher import _validate_closed_shell_contract
    with pytest.raises(MesherError, match='free edges'):
        _validate_closed_shell_contract(points, triangles[tags!=101], tags[tags!=101],
                                       symmetry_snap_axes=(), symmetry_snap_tol_mm=1e-6, is_open_shell=False)


@pytest.fixture(scope='module')
def horn_only_base(tmp_path_factory):
    from test_freeform_wall_identity import archive_base
    return archive_base(tmp_path_factory, BASELINE['base'])


@pytest.mark.parametrize('case', CASES)
def test_horn_only_byte_identity_to_corner_fix(tmp_path, case, horn_only_base):
    # Compare both trees in this runtime, so the gate also covers WG's pinned
    # Gmsh and future versions without a platform-dependent golden checksum.
    from test_freeform_wall_identity import assert_mesh_and_step_bytes
    row = BASELINE['cases'][case]
    config = copy.deepcopy(row['config'])
    assert_mesh_and_step_bytes(horn_only_base, tmp_path, case, config)
    result = build_from_config(config, tmp_path/'absent.msh')
    assert result.mesh_path.read_bytes() == (tmp_path/'base'/'out.msh').read_bytes()
    assert result.physical_groups == {int(k):v for k,v in row['physical_groups'].items()}
    assert result.metadata.get('tag_namespace') == row['tag_namespace']
    config['enclosure']['woofers'] = []
    empty = build_from_config(config, tmp_path/'empty.msh')
    assert empty.mesh_path.read_bytes() == result.mesh_path.read_bytes()


@pytest.mark.parametrize('shape', ['piston','ath_cone'])
def test_rigid_woofer_still_meshed_and_certified(tmp_path, shape):
    config = assembly_config(shape=shape, recess=2.)
    config['enclosure']['woofers'][0]['active'] = False
    result = build_from_config(config, tmp_path/'rigid.msh')
    _, points, triangles, tags, surfaces = arrays(result)
    assert not any(key.startswith('lf-b/') for key in result.metadata['source_tags'])
    rigid = [p for p in result.metadata['assembly']['patches'] if p['radiator_id']=='lf-b']
    for patch in rigid:
        selected = np.isin(surfaces, patch['surface_tags'])
        assert selected.any() and np.all(tags[selected] == 1)
        assert patch['role'] == 'rigid' and patch['area_mm2'] > 0
    # Rigid patch normals and geometry are also production invariants.
    patch = rigid[0]
    altered = triangles.copy()
    selected = np.isin(surfaces, patch['surface_tags'])
    altered[selected] = altered[selected][:,[0,2,1]]
    with pytest.raises((ValueError, RuntimeError), match='inconsistent|inward'):
        certify_arrays(points, altered, tags, surfaces, result.metadata['assembly'])


@pytest.mark.parametrize('shape,recess', [('piston',0.),('piston',2.),('ath_cone',0.)])
def test_half_domain_clips_centred_sources(tmp_path, shape, recess):
    config = assembly_config(shape=shape, recess=recess)
    rows = config['enclosure']['woofers']
    rows[1]['x_mm'] = 0
    config['mesh']['quadrants'] = '14'
    result = build_from_config(config, tmp_path/'half.msh')
    _, points, triangles, tags, surfaces = arrays(result)
    assert points[:,0].min() >= -1e-8
    assert result.native_symmetry_plane == 'yz'
    edges = build_edge_table(triangles)
    assert np.allclose(points[np.concatenate([edges.lo[edges.count==1],edges.hi[edges.count==1]])][:,0],0,atol=1e-7)
    for patch in result.metadata['assembly']['patches']:
        if shape=='piston':
            assert patch['area_mm2'] == pytest.approx(math.pi*patch['radius_mm']**2/2, rel=1e-7)
        assert np.any(np.isin(surfaces,patch['surface_tags']))


@pytest.mark.parametrize('quadrants', ['1','12','14'])
def test_incompatible_half_refused(tmp_path, quadrants):
    config = assembly_config()
    config['mesh']['quadrants'] = quadrants
    with pytest.raises((ValueError,MesherError), match='require full domain or x=0 half'):
        build_from_config(config,tmp_path/'refused.msh')


@pytest.mark.parametrize('change,message', [
    ({'radius_mm':-1}, 'radius_mm must be positive'),
    ({'x_mm':float('nan')}, 'finite'),
    ({'shape':{'kind':'piston','recess_mm':-1}}, 'recess must be nonnegative'),
    ({'shape':{'kind':'piston','recess_mm':.001}}, 'at least 0.01 mm'),
    ({'shape':{'kind':'mystery'}}, 'shape kind'),
    ({'active':0}, 'active must be boolean'),
    ({'resolution_mm':0}, 'resolution_mm must be positive'),
    ({'radiator_id':''}, 'radiator_id'),
    ({'y_mm':0}, 'not wholly inside'),
    ({'y_mm':-9999}, 'not wholly inside'),
    ({'shape':{'kind':'piston','recess_mm':999}}, 'reaches enclosure back'),
])
def test_invalid_woofer_refused(tmp_path, change, message):
    config = assembly_config()
    config['enclosure']['woofers'][0].update(change)
    with pytest.raises((ValueError, MesherError), match=message):
        build_from_config(config,tmp_path/'invalid.msh')


@pytest.mark.parametrize('gap,allowed', [(.05,False),(.1,True),(.15,True)])
def test_finite_mouth_clearance(tmp_path, gap, allowed):
    config = assembly_config()
    w = config['enclosure']['woofers'][0]
    mouth = resolve_geometry(config).geometry.inner_points[:,-1]
    w['y_mm'] = float(mouth[:,1].min())-w['radius_mm']-gap
    if allowed:
        build_from_config(config,tmp_path/'clear.msh')
    else:
        with pytest.raises(MesherError, match='0.1 mm clearance'):
            build_from_config(config,tmp_path/'close.msh')


@pytest.mark.parametrize('gap,allowed', [(.05,False),(.15,True)])
def test_finite_outer_clearance(tmp_path, gap, allowed):
    config = assembly_config()
    # With sharp edges the flat boundary is exactly the mouth envelope + margin.
    config['enclosure']['edge_mm'] = 0
    from hornlab_mesher.builders.enclosure import enclosure_box_bounds
    geometry = resolve_geometry(config).geometry
    bounds = enclosure_box_bounds(geometry.inner_points,geometry.enclosure,closed=True)
    w = config['enclosure']['woofers'][0]
    w['y_mm'] = bounds['by0']+w['radius_mm']+gap
    if allowed:
        build_from_config(config,tmp_path/'clear.msh')
    else:
        with pytest.raises(MesherError, match='0.1 mm clearance'):
            build_from_config(config,tmp_path/'close.msh')


@pytest.mark.parametrize('gap', [-1.,0.,.05])
def test_finite_interwoofer_clearance(tmp_path, gap):
    config = assembly_config()
    a,b = config['enclosure']['woofers']
    b.update(x_mm=a['x_mm']+a['radius_mm']+b['radius_mm']+gap, y_mm=a['y_mm'])
    with pytest.raises(MesherError, match='apertures overlap'):
        build_from_config(config,tmp_path/'close.msh')


def test_deep_collar_intersection_refused(tmp_path):
    config = assembly_config('lookup',recess=100.)
    config['profile']['lookupProfile'] = [[0,12.7],[30,200],[80,150],[120,100]]
    config['enclosure']['space_l_mm'] = config['enclosure']['space_r_mm'] = 200
    config['enclosure']['woofers'] = [dict(radiator_id='lf-b',x_mm=0.,y_mm=-165.,radius_mm=35.,shape={'kind':'piston','recess_mm':100.})]
    with pytest.raises(MesherError, match='intersects horn wall'):
        build_from_config(config,tmp_path/'intersect.msh')


def test_immutable_typed_contract_and_unique_ids():
    geometry = resolve_geometry(assembly_config()).geometry
    w = geometry.enclosure.woofers[0]
    assert isinstance(w,EnclosureWoofer) and isinstance(w.shape,PistonShape)
    with pytest.raises(FrozenInstanceError):
        w.radius_mm = 42
    with pytest.raises(FrozenInstanceError):
        w.shape.recess_mm = 3
    config = assembly_config()
    config['enclosure']['woofers'][1]['radiator_id'] = 'lf-b'
    with pytest.raises(ValueError,match='unique'):
        resolve_geometry(config)


@pytest.mark.parametrize('recess', [.01,2.])
def test_placement_and_units_once(tmp_path,recess):
    config = assembly_config(recess=recess)
    config['mesh'].update(scale_to_metres=True, verticalOffset=13.)
    result = build_from_config(config,tmp_path/'placed.msh')
    _, points, triangles, tags, _ = arrays(result)
    for w in config['enclosure']['woofers']:
        xyz = points[triangles[tags==result.metadata['source_tags'][w['radiator_id']+'/piston']]]*1000
        assert np.allclose(xyz[:,:,2],result.metadata['assemblyBaffleZMm']-recess,atol=1e-7,rtol=0)
        assert (xyz[:,:,1].min()+xyz[:,:,1].max())/2 == pytest.approx(w['y_mm']+13.,abs=.05)


def corpus_config(case):
    config = assembly_config()
    config['profile'] = dict(L_mm=120.,r0_mm=12.7,a0_deg=10.,a_deg=45.,k=1.,n=4.,q=.995,s=.5)
    config['mesh'].update(angular_segments=32,length_segments=12,max_triangles=30000)
    config['enclosure'].update(depth_mm=200.,space_l_mm=100.,space_r_mm=100.,space_t_mm=220.,space_b_mm=220.,edge_mm=15. if case['case_id']=='b-edge' else 0.)
    if case['case_id']=='b-morph-wide':
        config['morph'] = dict(morph_target=1,morph_width_mm=440.,morph_height_mm=360.,morph_corner_mm=12.,morph_fixed=.3,morph_rate=3.)
    geometry = resolve_geometry({**config,'enclosure':{k:v for k,v in config['enclosure'].items() if k!='woofers'}}).geometry
    mouth = geometry.inner_points[:,-1]
    from hornlab_mesher.builders.enclosure import enclosure_box_bounds
    bounds = enclosure_box_bounds(geometry.inner_points, replace(geometry.enclosure,
        space_l_mm=0,space_r_mm=0,space_t_mm=0,space_b_mm=0,edge_mm=0),closed=True)
    lo,hi = case['enclosure_extents_mm']
    # Resolve ATH's measured overall Dim/envelope into the existing mesher
    # margin contract. This slice deliberately does not parse ATH text.
    config['enclosure'].update(space_l_mm=bounds['bx0']-lo[0],space_r_mm=hi[0]-bounds['bx1'],
                               space_b_mm=bounds['by0']-lo[1],space_t_mm=hi[1]-bounds['by1'])
    config['enclosure']['woofers'] = []
    for expected in case['woofers']:
        sign = 1 if expected['source']=='A' else -1
        radius = expected['radius_mm']
        # These corpus cases use gap 20 except b-pair (35). WG resolves
        # against the cardinal Y mouth sample in ATH's throat-origin frame.
        gap = 35. if case['case_id']=='b-pair' else 20.
        yedge = mouth[np.argmax(sign*mouth[:,1]),1]
        config['enclosure']['woofers'].append(dict(radiator_id='lf-'+expected['source'],x_mm=0.,y_mm=float(yedge)+sign*(gap+radius),radius_mm=radius,shape={'kind':'ath_cone'},resolution_mm=3. if case['case_id'] in ('b-q14','mtm-q14') else expected['resolution_mm'],active=case['case_id']!='b-base'))
    return config


@pytest.mark.parametrize('case', REFERENCE['cases'], ids=lambda c:c['case_id'])
def test_corpus_apertures(tmp_path,case):
    result = build_from_config(corpus_config(case),tmp_path/'corpus.msh')
    _, points, triangles, tags, surfaces = arrays(result)
    assert points.min(axis=0) == pytest.approx(case['enclosure_extents_mm'][0],abs=.1)
    assert points.max(axis=0) == pytest.approx(case['enclosure_extents_mm'][1],abs=.1)
    for expected in case['woofers']:
        patches = [p for p in result.metadata['assembly']['patches'] if p['radiator_id']=='lf-'+expected['source']]
        selected = np.isin(surfaces,[s for p in patches for s in p['surface_tags']])
        vertices = points[np.unique(triangles[selected])]
        rim = vertices[abs(vertices[:,2]-case['baffle_plane_z_mm'])<1e-6]
        assert len(rim)>8
        a = np.column_stack((2*rim[:,0],2*rim[:,1],np.ones(len(rim))))
        cx,cy,c = np.linalg.lstsq(a,np.sum(rim[:,:2]**2,axis=1),rcond=None)[0]
        assert [cx,cy,case['baffle_plane_z_mm']] == pytest.approx(expected['aperture_center_mm'],abs=.1)
        assert math.sqrt(c+cx**2+cy**2) == pytest.approx(expected['radius_mm'],abs=.1)
    if case['case_id'] in ('b-q14','mtm-q14'):
        # Compare the complete emitted WG facets to the exact ATH q14
        # meridian revolved and reflected about X=0, including both X signs.
        from test_source_contour import certified_facet_distance, line_distance, arc_distance
        ref = REFERENCE['q14_meridian_mm']
        for patch in result.metadata['assembly']['patches']:
            xyz = points[triangles[np.isin(surfaces,patch['surface_tags'])]]-patch['center_mm']
            assert xyz[:,:,0].min()<0 and xyz[:,:,0].max()>0
            join = (ref['join_radius'],ref['join_z']-ref['rim_z'])
            if patch['segment_id']=='cone':
                distance = line_distance(join,(ref['radius'],0))
            else:
                distance = arc_distance((0,ref['apex_z']-ref['rim_z']),join,(0,ref['sphere_center_z']-ref['rim_z']),direction='cw')
            certified_facet_distance(xyz,distance,tolerance=.1,n=128)


def test_asymmetric_enclosure_half_refused(tmp_path):
    config = assembly_config()
    config['mesh']['quadrants'] = '14'
    for row in config['enclosure']['woofers']:
        row['x_mm'] = 0
    config['enclosure']['space_l_mm'] = 120
    with pytest.raises((ValueError,MesherError),match='symmetric left/right'):
        build_from_config(config,tmp_path/'asymmetric.msh')


def test_asymmetric_horn_half_refused(tmp_path):
    config = assembly_config()
    config['mesh']['quadrants'] = '14'
    for row in config['enclosure']['woofers']:
        row['x_mm'] = 0
    config['profile']['a_deg'] = '60+5*cos(p)'
    with pytest.raises((ValueError,MesherError),match='horn symmetric'):
        build_from_config(config,tmp_path/'asymmetric.msh')


@pytest.mark.parametrize('shape', ['piston', 'ath_cone'])
def test_woofer_density_preserves_enclosure_roundovers(tmp_path, monkeypatch, shape):
    # Review probe: identical OSSE enclosure, 18 mm edges, radii 35 and 25.
    # Count actual edge-face elements before horn-only export folds CAD tags.
    import gmsh
    from hornlab_mesher import mesher
    configure = mesher.configure_density
    generate = gmsh.model.mesh.generate
    edge_faces, realized = {}, {}

    def capture(built, density):
        edge_faces.clear()
        for role in ('enclosure_edges_front', 'enclosure_edges_back'):
            edge_faces[role] = list(built.mesh_surface_groups[role])
        configure(built, density)

    def measure(dim):
        generate(dim)
        realized.clear()
        for role, faces in edge_faces.items():
            realized[role] = sum(len(gmsh.model.mesh.getElementsByType(2, face)[0])
                                 for face in faces)

    monkeypatch.setattr(mesher, 'configure_density', capture)
    monkeypatch.setattr(gmsh.model.mesh, 'generate', measure)
    config = assembly_config(shape=shape)
    without = copy.deepcopy(config)
    without['enclosure'].pop('woofers')
    counts = []
    for name, recipe in [('horn-only', without), ('woofers', config)]:
        result = build_from_config(recipe, tmp_path/(name+'.msh'))
        counts.append(dict(realized))
        assert all(counts[-1].values())
        print(name, result.n_triangles, counts[-1])
    # Fragmentation can change the neighboring front-panel triangulation.
    # Edge-face triangle differences must stay within the added diaphragm
    # contribution and 2% of the original edges.
    _, _, _, tags, _ = arrays(result)
    woofer_triangles = int(np.isin(tags, [tag for key, tag in result.metadata['source_tags'].items()
                                         if key != 'horn-throat']).sum())
    for role, count in counts[0].items():
        assert abs(counts[1][role]-count) <= min(woofer_triangles, .02*count)


@pytest.mark.parametrize('change', [
    {'shape': None}, {'shape': {}}, {'shape': {'kind': 'piston', 'label': 'LF'}},
    {'label': 'LF'}, {'enabled': True}, {'placement': {}},
])
def test_malformed_woofer_config_raises_value_error(change):
    from hornlab_mesher.enclosure_woofers import woofer_from_config
    row = assembly_config()['enclosure']['woofers'][0]
    row.update(change)
    with pytest.raises(ValueError):
        woofer_from_config(row)


def test_missing_woofer_shape_raises_value_error():
    from hornlab_mesher.enclosure_woofers import woofer_from_config
    row = assembly_config()['enclosure']['woofers'][0]
    row.pop('shape')
    with pytest.raises(ValueError):
        woofer_from_config(row)


def test_certificate_uses_authored_contour(tmp_path, monkeypatch):
    from hornlab_mesher import enclosure_woofers
    from hornlab_mesher.source_contour import cone

    def authored(w):
        contour = cone(w.radius_mm, .3*w.radius_mm, cap_radius_mm=.4*w.radius_mm,
                       cap_height_mm=.1*w.radius_mm, physical_source_id=w.radiator_id, rim_id='rim')
        return replace(contour, segments=(replace(contour.segments[0], id='cap'), contour.segments[1]))

    monkeypatch.setattr(enclosure_woofers, 'ath_contour', authored)
    result = build_from_config(assembly_config(shape='ath_cone'), tmp_path/'authored.msh')
    _, points, triangles, tags, surfaces = arrays(result)
    certify_arrays(points, triangles, tags, surfaces, result.metadata['assembly'])
