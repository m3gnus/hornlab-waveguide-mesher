"""Real corner profiles: the S1a built-boundary checks on production builders."""
from __future__ import annotations
import copy
import json
import math
from dataclasses import replace
from pathlib import Path

import gmsh
import numpy as np
import pytest

from hornlab_mesher import build_from_config, build_geometry_params, write_step
from hornlab_mesher.freeform import build_freeform_geometry, freeform_corner_z
from hornlab_mesher.profile_sampling import build_point_grid_arrays
from hornlab_mesher.geometry import PointGridHornGeometry, HornEnclosure
from hornlab_mesher.corner_wall import (CornerWall, corner_geometry_class, build_corner_wall, measure_corner_mesh,
    mitred_outer_grid, cut_indices, wall_triangles, mesh_sections, mesh_radius, occ_radius)


def case_config(case='a', mode='bare'):
    knots={'a':[0,60,120], 'b':[0,40,80,120], 'c':[0,60,120], 'e':[0,60,120], 'dthroat':[0,5,120], 'dmouth':[0,115,120]}[case]
    ah=[10,30,15] if case=='b' else [10,30]
    av=[7,22] if case=='c' else ah

    def profile(angles):
        radius=12.7
        points=[{'z':0,'r':radius,'join':'corner'}]
        for i,z in enumerate(knots[1:]):
            radius += math.tan(math.radians(angles[i]))*(z-knots[i])
            points.append({'z':z,'r':radius,'join':'corner'})
        return {'points':points}
    h=profile(ah); v=profile(av)
    if case=='e':
        # Exact quadratic as cubic Bezier with monotone z.
        f=lambda z:12.7+math.tan(math.radians(12))*z+0.001*z*z
        slope=lambda z:math.tan(math.radians(12))+0.002*z
        v={'points':[{'z':0,'r':f(0),'handleOut':[40,40*slope(0)]},
                     {'z':120,'r':f(120),'handleIn':[-40,-40*slope(120)]}]}
    stations=[{'t':0,'shape':'circle'},{'t':1,'shape':'ellipse' if case=='e' else 'circle'}]
    if case=='c': stations=[{'t':0,'shape':'circle'},{'t':.4,'shape':'rounded_rectangle','cornerRadiusMm':5.9},{'t':1,'shape':'rounded_rectangle','cornerRadiusMm':5.9}]
    config={'formula':'FREEFORM','mode':mode,'profile':{'profileH':h,'profileV':v,'crossSections':stations},
            'mesh':{'angularSegments':64,'lengthSegments':24,'samplingMode':'uniform','throat_res_mm':8,'mouth_res_mm':8,
                    'rear_res_mm':5,'enc_front_res_mm':8,'enc_back_res_mm':8,'wall_thickness_mm':1,
                    'max_triangles':100000,'allow_large_mesh':True,'source_shape':0,'scale_to_metres':False}}
    if mode=='enclosure': config['enclosure']={'depth':150}
    return config


def raw_geometry(config):
    params, _, mode = build_geometry_params(config)
    grid = build_point_grid_arrays(params)
    kernel = build_freeform_geometry(params)
    return corner_geometry_class(PointGridHornGeometry)(
        inner_points=grid['inner_grid'], outer_points=grid['outer_grid'],
        closed=grid['full_circle'], freeform_report=kernel.report(),
        freeform_wall=CornerWall(kernel), wall_thickness_mm=1.,
        source_shape=0, infinite_baffle=mode == 'infinite-baffle',
        enclosure=HornEnclosure(depth_mm=150, edge_mm=0) if mode == 'enclosure' else None)


@pytest.fixture
def session():
    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.option.setNumber('General.NumThreads', 1)
    gmsh.option.setNumber('Geometry.Tolerance', 1e-8)
    gmsh.option.setNumber('Geometry.ToleranceBoolean', 1e-8)
    gmsh.model.add('corner-test')
    yield
    gmsh.finalize()


def mesh(built, size=2.):
    gmsh.option.setNumber('Mesh.MeshSizeMin', size)
    gmsh.option.setNumber('Mesh.MeshSizeMax', size)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
    gmsh.option.setNumber('Mesh.Algorithm', 1)
    gmsh.model.mesh.generate(2)
    xyz, tri = wall_triangles([tag for _, tag in gmsh.model.getEntities(2)])
    edges = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    assert not np.any(counts > 2)
    return xyz, unique[counts == 1]


@pytest.mark.parametrize('case,mode', [('a','bare'), ('b','freestanding'), ('c','enclosure'),
    ('dthroat','bare'), ('dmouth','infinite-baffle'), ('e','bare')])
def test_real_corner_acceptance(session, tmp_path, case, mode):
    geometry = raw_geometry(case_config(case, mode))
    built = build_corner_wall(geometry)
    xyz, free = mesh(built)
    if mode == 'bare':
        assert len(free) > 0
        assert np.all(np.abs(xyz[free, 2] - 120) < 1e-6)
    else:
        assert len(free) == 0
    measure_corner_mesh(geometry, built)
    records = built.metadata['cornerWall']['corners']
    assert len(records) == (2 if case == 'b' else 1)
    for record in records:
        assert len(record['azimuths']) == 8
        assert max(row['occJumpErrorDeg'] for row in record['azimuths']) <= .5
        assert len(record['sharedEdgeIncidence']) == 4
        assert all(edge['wallFaces'] == 2 for edge in record['sharedEdgeIncidence'])
        for plane in ('H', 'V'):
            assert record['mesh'][plane]['jumpErrorDeg'] <= .5
            assert record['mesh'][plane]['maxRadialDeviationMm'] <= .25
            assert record['occMaxRadialDeviationMm'][plane] <= .25
    if case == 'e':
        assert abs(records[0]['azimuths'][2]['occJumpDeg']) <= .5
    # Sampled maxima, fixed-z half-plane intersections; also retain off-axis mesh error.
    wall_xyz, tri = wall_triangles(built.mesh_surface_groups['inner'])
    off_axis = {}
    for deg in (45, 135, 225, 315):
        phi = math.radians(deg)
        section = mesh_sections(wall_xyz, tri, phi)
        z_grid = np.linspace(*built.axial_bounds_mm, 121)
        expected = lambda z: float(geometry.freeform_wall.radius(phi, z, built.freeform_wall_points_mm))
        off_axis[deg] = {'occ': max(abs(occ_radius(built.mesh_surface_groups['inner'], phi, z, expected(z)) - expected(z)) for z in z_grid),
                         'mesh': max(abs(mesh_radius(section, z) - expected(z)) for z in z_grid)}
    # Ordinary surface/solid STEP writer, not gmsh.write as a proxy.
    path, info = write_step(geometry, tmp_path / f'{case}.step')
    assert path.stat().st_size > 1000
    assert info.body == ('solid' if mode in ('enclosure', 'freestanding') else 'surface')
    if info.body == 'solid':
        assert info.volume_mm3 > 0
    smooth = case_config(case)
    for name in ('profileH', 'profileV'):
        profile = smooth['profile'][name]
        if all(point.get('join') == 'corner' for point in profile['points']):
            rows = profile['points']
            profile['throatAngleDeg'] = math.degrees(math.atan((rows[1]['r'] - rows[0]['r']) / (rows[1]['z'] - rows[0]['z'])))
            profile['mouthAngleDeg'] = math.degrees(math.atan((rows[-1]['r'] - rows[-2]['r']) / (rows[-1]['z'] - rows[-2]['z'])))
        for point in profile['points']:
            point.pop('join', None)
    baseline = raw_geometry(smooth)
    baseline_recipe = baseline.freeform_wall
    baseline = replace(baseline, freeform_wall=None)
    from hornlab_mesher.mesher import _dispatch_builder
    gmsh.clear()
    gmsh.model.add('smooth-station-baseline')
    smooth_built = _dispatch_builder(baseline)
    gmsh.model.occ.synchronize()
    mesh(smooth_built)
    smooth_xyz, smooth_tri = wall_triangles(smooth_built.mesh_surface_groups['inner'])
    smooth_errors = {}
    for deg in (45,135,225,315):
        phi = math.radians(deg)
        section = mesh_sections(smooth_xyz, smooth_tri, phi)
        samples = np.linspace(0,120,121)
        expected = lambda z: float(baseline_recipe.radius(phi,z,baseline.inner_points))
        smooth_errors[deg] = {'occ': max(abs(occ_radius(smooth_built.mesh_surface_groups['inner'],phi,z,expected(z))-expected(z)) for z in samples),
                              'mesh': max(abs(mesh_radius(section,z)-expected(z)) for z in samples)}
    evidence = {'case':case, 'mode':mode, **built.metadata['cornerWall'],
                'offAxisMm':off_axis, 'smoothOffAxisMm':smooth_errors}
    (tmp_path / f'{case}-measurements.json').write_text(json.dumps(evidence, indent=2))
    for deg in off_axis:
        for kind in ('occ','mesh'):
            assert off_axis[deg][kind] <= 1.1 * smooth_errors[deg][kind] + 1e-8, evidence



@pytest.mark.parametrize('corner', [40., 80.])
def test_convex_concave_mitre_keeps_both_side_thickness(corner):
    geometry = raw_geometry(case_config('b', 'freestanding'))
    points = geometry.inner_points
    outer = mitred_outer_grid(points, 1., full_circle=True, corner_z=(40., 80.))
    j = cut_indices(points, (40., 80.))[1 if corner == 40 else 2]
    # Circular meridian: signed normal distance to each offset line is 1 mm.
    for neighbor in (j - 1, j + 1):
        tangent = points[0, neighbor, [2, 0]] - points[0, j, [2, 0]]
        tangent *= 1 if neighbor > j else -1
        normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
        assert np.dot(outer[0, j, [2, 0]] - points[0, j, [2, 0]], normal) == pytest.approx(1, abs=1e-8)


def test_unsafe_concave_mitre_names_corner():
    phi = np.linspace(0, 2*np.pi, 64, endpoint=False)
    z = np.array([0., 1., 2.])
    r = np.array([100., 110., 100.])
    points = np.stack(np.broadcast_arrays(np.cos(phi)[:,None]*r, np.sin(phi)[:,None]*r, z),axis=-1)
    with pytest.raises(ValueError, match=r'corner z=1 mm: unsafe offset mitre'):
        mitred_outer_grid(points, 2, full_circle=True, corner_z=(1.,))


def test_cut_z_union_and_refinement_preserve_full_rings():
    from hornlab_mesher.corner_wall import _refine
    config = case_config('a')
    config['profile']['profileV'] = case_config('b')['profile']['profileV']
    geometry = raw_geometry(config)
    assert geometry.freeform_wall.cuts(geometry.inner_points) == (40,60,80)
    refined = _refine(geometry.inner_points, geometry.freeform_wall)
    assert np.array_equal(refined[:,::2], geometry.inner_points)
    assert [refined[0,j,2] for j in cut_indices(refined,(40,60,80))[1:-1]] == [40,60,80]


@pytest.mark.parametrize('mode', ['bare','enclosure','infinite-baffle','freestanding'])
def test_public_corner_build_report(tmp_path, mode):
    config = case_config('b' if mode == 'freestanding' else 'e', mode)
    if mode == 'enclosure':
        config['enclosure'].update(edge=0)
    result = build_from_config(config, tmp_path / f'{mode}.msh')
    report = result.metadata['freeformReport']['cornerWall']
    assert report['corners']
    for corner in report['corners']:
        assert all(v['jumpErrorDeg'] <= .5 and v['maxRadialDeviationMm'] <= .25 for v in corner['mesh'].values()), corner


def test_measured_refinement_and_named_cap(session, monkeypatch):
    import hornlab_mesher.corner_wall as module
    config = case_config('e')
    config['mesh']['lengthSegments'] = 4
    geometry = raw_geometry(config)
    with monkeypatch.context() as patch:
        patch.setattr(module, 'MAX_CORNER_REFINEMENTS', 0)
        with pytest.raises(ValueError, match=r'corner z=60 mm: OCC slope jump error .* refinement cap'):
            build_corner_wall(geometry)
    gmsh.clear()
    gmsh.model.add('refined-corner')
    built = build_corner_wall(geometry)
    assert built.metadata['cornerWall']['refinements'] > 0
    assert max(row['occJumpErrorDeg'] for row in built.metadata['cornerWall']['corners'][0]['azimuths']) <= .5


@pytest.mark.parametrize('quadrants', ['1','12','14'])
def test_reduced_corner_mesh_and_step(tmp_path, quadrants):
    config = case_config('a')
    config['mesh']['quadrants'] = quadrants
    # Public default source is a rounded cap, so this also covers shared curved rims.
    result = build_from_config(config, tmp_path / f'{quadrants}.msh')
    assert result.n_triangles > 0
    assert len(result.metadata['freeformReport']['cornerWall']['corners'][0]['azimuths']) == 8
    from hornlab_mesher import write_step_from_config
    path, info = write_step_from_config(config, tmp_path / f'{quadrants}.step')
    assert path.stat().st_size > 1000
    assert info.body == 'surface'


def test_refinement_preserves_interface_z(session):
    from hornlab_mesher.geometry import HornInterface
    config = case_config('e', 'enclosure')
    config['mesh']['lengthSegments'] = 4
    geometry = raw_geometry(config)
    index = 1
    wanted_z = geometry.inner_points[0, index, 2]
    geometry = replace(geometry, interfaces=(HornInterface(slice_index=index, offset_mm=2),))
    built = build_corner_wall(geometry)
    assert built.metadata['cornerWall']['refinements'] > 0
    curves = [tag for dim, tag in gmsh.model.getBoundary([(2,t) for t in built.mesh_surface_groups['interface']], oriented=False) if dim == 1]
    # An offset interface spans from its requested z to z + offset.
    assert curves
    curve_z = []
    for tag in curves:
        lo, hi = gmsh.model.getParametrizationBounds(1,tag)
        curve_z.extend(gmsh.model.getValue(1,tag,[u])[2] for u in np.linspace(lo[0],hi[0],9))
    assert min(curve_z) == pytest.approx(wanted_z, abs=1e-8)
    assert max(curve_z) == pytest.approx(wanted_z + 2, abs=1e-8)


def test_real_sharp_corner_mitre_refuses_at_public_entry(tmp_path):
    config = case_config('a','freestanding')
    profile = {'points':[{'z':0,'r':100,'join':'corner'},
                         {'z':1,'r':110,'join':'corner'},
                         {'z':2,'r':100,'join':'corner'}]}
    config['profile'].update(profileH=profile,profileV=copy.deepcopy(profile))
    config['mesh']['wall_thickness_mm'] = 2
    with pytest.raises(ValueError, match=r'corner z=1 mm: unsafe offset mitre'):
        build_from_config(config,tmp_path/'unsafe.msh')


def test_scaled_baffle_cut_keeps_semantic_z(tmp_path):
    config = case_config('a','infinite-baffle')
    config['scale'] = 2
    result = build_from_config(config,tmp_path/'scaled.msh')
    corner = result.metadata['freeformReport']['cornerWall']['corners'][0]
    assert corner['zMm'] == -120
    assert all(edge['wallFaces'] == 2 for edge in corner['sharedEdgeIncidence'])


def test_curvature_stencil_does_not_cross_near_endpoint_corner():
    cut = .001
    r = 12.7 + math.tan(math.radians(10))*cut
    profile = {'points':[{'z':0,'r':12.7,'join':'corner'},
                         {'z':cut,'r':r,'join':'corner'},
                         {'z':120,'r':r+math.tan(math.radians(30))*(120-cut),'join':'corner'}]}
    kernel = build_freeform_geometry({'profileH':profile,'profileV':copy.deepcopy(profile)})
    assert kernel.surface_curvature_report(.1)['ok']


def test_outer_crease_edges_have_two_incident_faces(session):
    built = build_corner_wall(raw_geometry(case_config('b','freestanding')))
    assert len(built.mesh_surface_groups['outer']) == 12
    for record in built.metadata['cornerWall']['corners']:
        assert len(record['outerSharedEdgeIncidence']) == 4
        assert all(edge['wallFaces'] == 2 for edge in record['outerSharedEdgeIncidence'])
