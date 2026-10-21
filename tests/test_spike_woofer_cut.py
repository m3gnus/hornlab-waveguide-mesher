"""Feasibility evidence for the shared design-enclosure woofer cut hook."""
from __future__ import annotations

import copy
import json
import math
import platform
from pathlib import Path

import gmsh
import meshio
import numpy as np
import pytest

from hornlab_mesher import build_from_config
from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.edges import build_edge_table
from hornlab_mesher.normals import validate_orientation
from hornlab_mesher.mesher import MesherError
from test_freeform_builds import _owner_config
from test_freeform_corner_wall import case_config

CASES = ('osse', 'rosse', 'freeform-rounded', 'lookup', 'osse-rectangle', 'freeform-corners')


def spike_config(case, recess=0.0):
    if case == 'freeform-rounded':
        config = _owner_config()
    elif case == 'freeform-corners':
        config = case_config('c', 'enclosure')
    else:
        import tomllib
        example = 'rosse-enclosure.toml' if case == 'rosse' else 'osse-freestanding.toml'
        config = tomllib.loads((Path(__file__).parents[1] / 'examples' / example).read_text())
        config['mesh'].pop('wall_thickness_mm', None)
        if case == 'lookup':
            config.update(formula='LOOKUP', profile={'lookupProfile': [[0,12.7],[30,28],[80,85],[120,150]]})
        if case == 'osse-rectangle':
            config['morph'] = {'morph_target':1, 'morph_width_mm':300, 'morph_height_mm':200, 'morph_corner_mm':30}
    config['mode'] = 'enclosure'
    config['mesh'].update(throat_res_mm=8, mouth_res_mm=20, rear_res_mm=25,
                          enc_front_res_mm=12, enc_back_res_mm=30, scale_to_metres=False,
                          source_shape=0, quadrants='1234')
    config['enclosure'] = {'depth_mm':240, 'space_l_mm':80, 'space_r_mm':80,
                           'space_t_mm':150, 'space_b_mm':150, 'edge_mm':18}
    if case == 'freeform-corners':
        config['enclosure']['edge_mm'] = 0
    geometry = resolve_geometry(config).geometry
    mouth = geometry.inner_points[:, -1]
    # Two asymmetric independent radiators, including x=0 across the baffle seam.
    config['enclosure']['spike_woofers'] = [
        {'x_mm':0., 'y_mm':float(mouth[:,1].min())-65, 'radius_mm':35., 'recess_mm':recess},
        {'x_mm':25., 'y_mm':float(mouth[:,1].max())+65, 'radius_mm':25., 'recess_mm':recess},
    ]
    return config


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('recess', [0., 2.])
def test_formula_assembly(tmp_path, case, recess):
    config = spike_config(case, recess)
    result = build_from_config(config, tmp_path / f'{case}-{recess}.msh')
    mesh = meshio.read(result.mesh_path)
    points = np.asarray(mesh.points)
    triangles = mesh.cells_dict['triangle']
    tags = mesh.cell_data_dict['gmsh:physical']['triangle']
    report = validate_orientation(points, triangles, tags, require_watertight=True,
                                  require_edge_consistency=True, require_positive_volume=True,
                                  require_source_normal=True, source_axis='z')
    assert report.watertight and report.inconsistent_edges == 0
    assert set(tags) == {1,2,101,102}
    assert mesh.field_data['woofer-1'].tolist() == [101,2]
    assert mesh.field_data['woofer-2'].tolist() == [102,2]
    assert result.metadata['source_tags'] == {'horn-throat':2,'woofer-1':101,'woofer-2':102}
    assert set(result.metadata['tag_map']) == {'1','2','101','102'}
    xyz = points[triangles]
    normals = np.cross(xyz[:,1]-xyz[:,0], xyz[:,2]-xyz[:,0])
    measures = {}
    z = result.metadata['spikeBaffleZMm']
    for i, row in enumerate(config['enclosure']['spike_woofers']):
        tag = 101+i
        selected = tags == tag
        assert selected.any()
        assert np.all(normals[selected,2] > 0)
        assert np.allclose(xyz[selected,:,2], z-recess, rtol=0, atol=1e-7)
        area = np.linalg.norm(normals[selected],axis=1).sum()/2
        assert area == pytest.approx(math.pi*row['radius_mm']**2, rel=.005)
        edges = build_edge_table(triangles[selected])
        rim = edges.count == 1
        midpoints = (points[edges.lo[rim]] + points[edges.hi[rim]])/2
        chord_error = row['radius_mm'] - np.linalg.norm(midpoints[:,:2] - [row['x_mm'],row['y_mm']], axis=1).min()
        assert chord_error < .1
        assert result.metadata['tag_map'][str(tag)] == {'source_id':f'woofer-{i+1}', 'instance_id':None, 'role':'secondary'}
        measures[str(tag)] = {'triangles':int(selected.sum()),'areaMm2':float(area),'exactAreaMm2':math.pi*row['radius_mm']**2,'maxRimChordErrorMm':float(chord_error)}
    # The existing production closure guard must also detect a missing piston.
    from hornlab_mesher.mesher import _validate_closed_shell_contract
    with pytest.raises(MesherError, match='free edges'):
        _validate_closed_shell_contract(points, triangles[tags!=101], tags[tags!=101],
                                       symmetry_snap_axes=(), symmetry_snap_tol_mm=1e-6, is_open_shell=False)
    evidence = {'case':case,'recessMm':recess,'triangles':len(triangles),'baffleZMm':z,
                'boundaryEdges':report.boundary_edges,'nonmanifoldEdges':report.nonmanifold_edges,
                'inconsistentEdges':report.inconsistent_edges,'signedVolumeMm3':report.signed_volume,
                'sources':measures,'gmsh':gmsh.__version__,'python':platform.python_version()}
    (tmp_path / 'evidence.json').write_text(json.dumps(evidence,indent=2))


@pytest.mark.parametrize('quadrants', ['1','12','14'])
def test_refuse_reduced_assembly(tmp_path, quadrants):
    config = spike_config('osse')
    config['mesh']['quadrants'] = quadrants
    with pytest.raises((ValueError, MesherError), match='requires full domain'):
        build_from_config(config, tmp_path/'refused.msh')


@pytest.mark.parametrize('change,message', [
    ({'radius_mm':-1}, 'radius must be positive'),
    ({'x_mm':float('nan')}, 'must be finite'),
    ({'recess_mm':-1}, 'recess nonnegative'),
    ({'recess_mm':.001}, 'at least 0.01 mm'),
    ({'y_mm':0}, 'not wholly inside'),
    ({'y_mm':-9999}, 'not wholly inside'),
    ({'recess_mm':999}, 'reaches enclosure back'),
])
def test_refuse_invalid_woofer(tmp_path, change, message):
    config = spike_config('osse')
    config['enclosure']['spike_woofers'][0].update(change)
    with pytest.raises((ValueError, MesherError), match=message):
        build_from_config(config,tmp_path/'invalid.msh')


def test_refuse_overlapping_woofers(tmp_path):
    config = spike_config('osse')
    config['enclosure']['spike_woofers'][1] = copy.deepcopy(config['enclosure']['spike_woofers'][0])
    with pytest.raises((ValueError, MesherError), match='apertures overlap'):
        build_from_config(config,tmp_path/'overlap.msh')


def test_corner_rounded_enclosure_failure_precedes_woofer_hook(tmp_path):
    config = spike_config('freeform-corners')
    config['enclosure']['edge_mm'] = 18
    config['enclosure'].pop('spike_woofers')
    with pytest.raises(MesherError, match='corner sewing lost face ancestry'):
        build_from_config(config, tmp_path/'baseline-corner-failure.msh')


@pytest.mark.parametrize('count,recess', [(0,0.),(1,.01),(3,2.)])
def test_variable_woofer_count_and_placement(tmp_path, count, recess):
    config = spike_config('osse', recess)
    rows = config['enclosure']['spike_woofers']
    rows.append({'x_mm':-100., 'y_mm':rows[0]['y_mm'], 'radius_mm':25., 'recess_mm':0})
    config['enclosure']['spike_woofers'] = rows[:count]
    config['mesh'].update(scale_to_metres=True, verticalOffset=13.)
    result = build_from_config(config, tmp_path/f'count-{count}.msh')
    mesh = meshio.read(result.mesh_path)
    tags = mesh.cell_data_dict['gmsh:physical']['triangle']
    triangles = mesh.cells_dict['triangle']
    if not count:
        assert set(tags) == {1,2,3}
        assert 'source_tags' not in result.metadata
        return
    assert set(tags) == {1,2,*range(101,101+count)}
    for i, row in enumerate(rows[:count]):
        xyz = mesh.points[triangles[tags==101+i]] * 1000
        assert np.allclose(xyz[...,2], result.metadata['spikeBaffleZMm']-row['recess_mm'], rtol=0, atol=1e-7)
        lo, hi = xyz.reshape(-1,3).min(axis=0), xyz.reshape(-1,3).max(axis=0)
        # The polygonal rim can miss the exact circle extrema by a chord error.
        assert (lo[0]+hi[0])/2 == pytest.approx(row['x_mm'], abs=.5)
        assert (lo[1]+hi[1])/2 == pytest.approx(row['y_mm']+13, abs=.5)
