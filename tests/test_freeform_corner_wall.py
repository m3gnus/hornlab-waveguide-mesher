"""Real corner profiles: the S1a built-boundary checks on production builders."""
from __future__ import annotations
import copy
import json
import math
from dataclasses import fields, replace
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
    return corner_geometry_class(PointGridHornGeometry, CornerWall(kernel))(
        inner_points=grid['inner_grid'], outer_points=grid['outer_grid'],
        closed=grid['full_circle'], freeform_report=kernel.report(),
        wall_thickness_mm=1.,
        source_shape=0, infinite_baffle=mode == 'infinite-baffle',
        enclosure=HornEnclosure(depth_mm=150, edge_mm=0) if mode == 'enclosure' else None)


def rounded_enclosure_config(case='c', edge_mm=18, quadrants='1234'):
    config = case_config(case, 'enclosure')
    config['mesh'].update(throat_res_mm=8, mouth_res_mm=20, rear_res_mm=25,
                          enc_front_res_mm=12, enc_back_res_mm=30,
                          scale_to_metres=False, source_shape=0, quadrants=quadrants)
    config['enclosure'] = {'depth_mm':240, 'space_l_mm':80, 'space_r_mm':80,
                           'space_t_mm':150, 'space_b_mm':150, 'edge_mm':edge_mm}
    if edge_mm == 5:
        # Keep the small roundover in the acoustic mesh, rather than letting
        # the normal sub-resolution cosmetic-edge suppression hide it.
        config['mesh']['enc_front_res_mm'] = 5
    return config


@pytest.mark.parametrize('case', ['a', 'c'], ids=['circle', 'rounded-rectangle'])
@pytest.mark.parametrize('edge_mm', [18, 5])
@pytest.mark.parametrize('quadrants', ['1234', '12'])
def test_rounded_corner_enclosure(session, tmp_path, monkeypatch, case, edge_mm, quadrants):
    from hornlab_mesher import corner_wall
    from hornlab_mesher import write_step_from_config
    from test_cad_step import _reimported_solids

    def checked_build(geometry):
        built = build_corner_wall(geometry)
        for side, direction in (('front', -1), ('back', 1)):
            tags = built.mesh_surface_groups[f'enclosure_edges_{side}']
            assert tags
            z = built.enclosure_bounds[f'z_{side}']
            z0, z1 = sorted((z, z + direction * edge_mm))
            for tag in tags:
                lo, hi = map(np.asarray, gmsh.model.getParametrizationBounds(2, tag))
                uv = [lo + np.array([u, v]) * (hi - lo)
                      for u in (.23, .5, .77) for v in (.23, .5, .77)]
                uv = [p for p in uv if gmsh.model.isInside(2, tag, p, parametric=True)]
                assert len(uv) > 0, (side, tag)
                xyz = np.asarray(gmsh.model.getValue(2, tag, np.asarray(uv).ravel())).reshape(-1, 3)
                # Roundovers occupy one radius next to their own end plane.
                # Sample the actual trim: OCC boxes can bound distant poles.
                assert xyz[:, 2].min() >= z0 - 1e-6, (side, tag, xyz)
                assert xyz[:, 2].max() <= z1 + 1e-6, (side, tag, xyz)
        return built

    monkeypatch.setattr(corner_wall, 'build_corner_wall', checked_build)
    config = rounded_enclosure_config(case, edge_mm, quadrants)
    result = build_from_config(config, tmp_path / 'rounded.msh')
    assert result.metadata['acousticEnclosureEdgeSuppressed'] is False
    groups = {tag: set(map(int, gmsh.model.getEntitiesForPhysicalGroup(2, tag)))
              for _, tag in gmsh.model.getPhysicalGroups(2)}
    assert set(groups) == {1, 2, 3}
    assert len(groups[1]) == (8 if quadrants == '1234' else 4)
    assert len(groups[2]) == 1
    assert len(groups[3]) == (44 if quadrants == '1234' else 22)
    assert not (groups[1] & groups[2] or groups[1] & groups[3] or groups[2] & groups[3])
    assert set.union(*groups.values()) == {tag for _, tag in gmsh.model.getEntities(2)}
    # Check raw Gmsh triangles before the public postprocessor can weld nodes.
    xyz, tri = wall_triangles(sorted(set.union(*groups.values())))
    from hornlab_mesher.edges import build_edge_table
    edges = build_edge_table(tri)
    assert not np.any(edges.count > 2)
    free = np.c_[edges.lo[edges.count == 1], edges.hi[edges.count == 1]]
    if quadrants == '1234':
        assert not len(free)
    else:
        assert len(free) > 0
        assert np.all(abs(xyz[free, 1]) < 1e-8)
    source_xyz, source_tri = wall_triangles(sorted(groups[2]))
    source_xyz = source_xyz[np.unique(source_tri)]
    from hornlab_mesher.config_builder import resolve_geometry
    from hornlab_mesher.builders.point_grid_sources import _source_cap_height
    geometry = resolve_geometry(config).geometry
    cap_height = _source_cap_height(12.7, geometry) if geometry.source_shape == 1 else 0
    assert source_xyz[:, 2].min() == pytest.approx(0, abs=1e-8)
    assert source_xyz[:, 2].max() <= cap_height + 1e-8
    assert np.max(np.linalg.norm(source_xyz[:, :2], axis=1)) <= 12.7 + 1e-8
    inner_xyz, inner_tri = wall_triangles(sorted(groups[1]))
    inner_xyz = inner_xyz[np.unique(inner_tri)]
    assert inner_xyz[:, 2].min() == pytest.approx(0, abs=1e-8)
    assert inner_xyz[:, 2].max() == pytest.approx(120, abs=1e-8)
    report = result.metadata['freeformReport']['cornerWall']
    assert len(report['corners']) == 1
    for corner in report['corners']:
        assert len(corner['sharedEdgeIncidence']) == (4 if quadrants == '1234' else 2)
        assert all(edge['wallFaces'] == 2 for edge in corner['sharedEdgeIncidence'])
        assert max(row['occJumpErrorDeg'] for row in corner['azimuths']) <= .5
        for plane in ('H', 'V'):
            assert corner['mesh'][plane]['jumpErrorDeg'] <= .5
            assert corner['mesh'][plane]['maxRadialDeviationMm'] <= .25
            assert corner['occMaxRadialDeviationMm'][plane] <= .25
    path, info = write_step_from_config(config, tmp_path / 'rounded.step')
    assert info.body == 'solid' and info.volume_mm3 > 0
    solids = _reimported_solids(path)
    assert len(solids) == 1 and solids[0][1] > 0


@pytest.mark.parametrize('quadrants', ['1234', '12'])
def test_rounded_corner_enclosure_interface_roles(session, quadrants):
    from hornlab_mesher.config_builder import resolve_geometry
    from hornlab_mesher.geometry import HornInterface

    geometry = resolve_geometry(rounded_enclosure_config(quadrants=quadrants)).geometry
    index = 5
    wanted_z = geometry.inner_points[0, index, 2]
    geometry = replace(geometry, interfaces=(HornInterface(slice_index=index, offset_mm=2),))
    built = build_corner_wall(geometry)
    roles = built.mesh_surface_groups
    assert set(roles['enclosure_edges_front']) <= set(roles['enclosure'])
    assert set(roles['enclosure_edges_back']) <= set(roles['enclosure'])
    assert len(roles['enclosure_edges_front']) == (12 if quadrants == '1234' else 6)
    assert len(roles['enclosure_edges_back']) == (12 if quadrants == '1234' else 6)
    assert built.surface_groups[1] == roles['inner']
    assert built.surface_groups[2] == roles['throat_disc']
    assert built.surface_groups[3] == roles['enclosure']
    assert built.surface_groups[4] == roles['interface']
    assert not (set(roles['interface']) & set(roles['inner'] + roles['enclosure'] + roles['throat_disc']))
    curves = gmsh.model.getBoundary([(2, tag) for tag in roles['interface']],
                                    oriented=False, combined=False)
    z = []
    for dim, tag in curves:
        lo, hi = gmsh.model.getParametrizationBounds(dim, tag)
        z.extend(gmsh.model.getValue(dim, tag, [u])[2] for u in np.linspace(lo[0], hi[0], 9))
    assert min(z) == pytest.approx(wanted_z, abs=1e-8)
    assert max(z) == pytest.approx(wanted_z + 2, abs=1e-8)


@pytest.mark.parametrize('failure', ['lost', 'ambiguous'])
def test_sewing_still_refuses_invalid_face_ancestry(session, monkeypatch, failure):
    from hornlab_mesher.corner_wall import _sew_roles

    faces = [(2, gmsh.model.occ.addRectangle(x, 0, 0, 1, 1)) for x in (0, 1)]
    sew = gmsh.model.occ.addSurfaceLoop

    def invalid_sew(*args, **kwargs):
        shell = sew(*args, **kwargs)
        if failure == 'lost':
            gmsh.model.occ.addRectangle(10, 0, 0, 1, 1)
        else:
            gmsh.model.occ.copy([faces[0]])
        return shell

    monkeypatch.setattr(gmsh.model.occ, 'addSurfaceLoop', invalid_sew)
    with pytest.raises(ValueError, match=f'{failure} face ancestry'):
        _sew_roles({'wall': faces})


def test_sewing_refuses_disjoint_trim_on_same_surface(session, monkeypatch):
    from hornlab_mesher.corner_wall import _sew_roles

    outer = (2, gmsh.model.occ.addRectangle(0, 0, 0, 4, 4))
    hole = (2, gmsh.model.occ.addRectangle(.8, .8, 0, 2.4, 2.4))
    frame, _ = gmsh.model.occ.cut([outer], [hole])
    other = (2, gmsh.model.occ.addRectangle(10, 0, 0, 1, 1))
    sew = gmsh.model.occ.addSurfaceLoop

    def disjoint_sew(*args, **kwargs):
        shell = sew(*args, **kwargs)
        child = (2, gmsh.model.occ.addRectangle(.8, .8, 0, 2.4, 2.4))
        gmsh.model.occ.synchronize()
        # Both parameter rectangles contain both sets of diagonal probes,
        # and the supporting surfaces agree exactly. The trims are disjoint.
        parent_lo, parent_hi = map(np.asarray, gmsh.model.getParametrizationBounds(*frame[0]))
        child_lo, child_hi = map(np.asarray, gmsh.model.getParametrizationBounds(*child))
        for lo, hi in ((parent_lo, parent_hi), (child_lo, child_hi)):
            uv = np.array([lo + f * (hi - lo) for f in (.23, .5, .77)])
            assert np.all((uv >= parent_lo) & (uv <= parent_hi))
            assert np.all((uv >= child_lo) & (uv <= child_hi))
            assert np.allclose(gmsh.model.getValue(*frame[0], uv.ravel()),
                               gmsh.model.getValue(*child, uv.ravel()), rtol=0, atol=1e-12)
        return shell

    monkeypatch.setattr(gmsh.model.occ, 'addSurfaceLoop', disjoint_sew)
    with pytest.raises(ValueError, match='lost face ancestry'):
        _sew_roles({'wall': [*frame, other]})


def test_sewing_refuses_two_parents_matching_one_child(session, monkeypatch):
    from hornlab_mesher.corner_wall import _sew_roles

    faces = [(2, gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)) for _ in range(2)]

    def one_child(tags, *, sewing):
        assert tags == [tag for _, tag in faces] and sewing
        # Coincident parents have identical signatures. Supply exactly one
        # matching child, so neither candidate multiplicity nor loss refuses it.
        gmsh.model.occ.copy([faces[0]])

    monkeypatch.setattr(gmsh.model.occ, 'addSurfaceLoop', one_child)
    with pytest.raises(ValueError, match='ambiguous face ancestry'):
        _sew_roles({'front': [faces[0]], 'back': [faces[1]]})


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
    baseline = PointGridHornGeometry(**{field.name: getattr(baseline, field.name)
                                        for field in fields(baseline)})
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


@pytest.mark.parametrize('quadrants,mode,endpoints_only', [
    (q, mode, endpoint) for q in ('1', '12', '14')
    for mode in ('bare', 'freestanding') for endpoint in (False, True)
] + [('12', 'enclosure', False)])
def test_reduced_corner_mesh_and_step(tmp_path, quadrants, mode, endpoints_only):
    config = case_config('a', mode)
    config['mesh']['quadrants'] = quadrants
    if endpoints_only:
        for name in ('profileH', 'profileV'):
            config['profile'][name]['points'][1].pop('join')
    if mode == 'enclosure':
        config['enclosure']['edge'] = 0
    # Public default source is a rounded cap, so this covers shared curved rims.
    result = build_from_config(config, tmp_path / f'{quadrants}.msh')
    assert result.n_triangles > 0
    corners = result.metadata['freeformReport']['cornerWall']['corners']
    assert len(corners) == (0 if endpoints_only else 1)
    if corners:
        assert len(corners[0]['azimuths']) == 8
    from hornlab_mesher import write_step_from_config
    path, info = write_step_from_config(config, tmp_path / f'{quadrants}.step')
    assert path.stat().st_size > 1000
    assert info.body == ('surface' if mode == 'bare' else 'solid')


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


def angled_config(before, after, mode='bare'):
    config = case_config('a', mode)
    rows = config['profile']['profileH']['points']
    rows[1]['r'] = rows[0]['r'] + 60 * math.tan(math.radians(before))
    rows[2]['r'] = rows[1]['r'] + 60 * math.tan(math.radians(after))
    config['profile']['profileV'] = copy.deepcopy(config['profile']['profileH'])
    return config


def without_corners(config):
    config = copy.deepcopy(config)
    for name in ('profileH', 'profileV'):
        profile = config['profile'][name]
        rows = profile['points']
        for point in rows:
            point.pop('join', None)
        for key, a, b in [('throatAngleDeg', rows[0], rows[1]),
                          ('mouthAngleDeg', rows[-2], rows[-1])]:
            profile[key] = math.degrees(math.atan((b['r'] - a['r']) / (b['z'] - a['z'])))
    return config


def test_tolerance_inflated_crease_public_build(tmp_path):
    config = angled_config(65, 5)
    config['mesh'].update(throat_res_mm=6, mouth_res_mm=6)
    result = build_from_config(config, tmp_path / 'inflated.msh')
    corner = result.metadata['freeformReport']['cornerWall']['corners'][0]
    assert len(corner['sharedEdgeIncidence']) == 4
    assert all(edge['wallFaces'] == 2 for edge in corner['sharedEdgeIncidence'])
    assert all(row['jumpErrorDeg'] <= .5 for row in corner['mesh'].values())


@pytest.mark.parametrize('case,mode', [('a', 'bare'), ('b', 'freestanding')])
def test_corner_build_timing(session, tmp_path, case, mode):
    from time import perf_counter
    config = case_config(case, mode)
    config['mesh'].update(throat_res_mm=3, mouth_res_mm=3, rear_res_mm=3)
    timings = {}
    for label, cfg in [('smooth', without_corners(config)), ('corner', config)]:
        start = perf_counter()
        build_from_config(cfg, tmp_path / f'{label}.msh')
        timings[label] = perf_counter() - start
    (tmp_path / 'timing.json').write_text(json.dumps(timings))
    assert timings['corner'] <= 3 * timings['smooth'] + 2, timings


def test_coarse_corner_respects_user_density(tmp_path):
    config = case_config('a')
    config['mesh'].update(throat_res_mm=20, mouth_res_mm=20)
    smooth = build_from_config(without_corners(config), tmp_path / 'smooth.msh')
    corner = build_from_config(config, tmp_path / 'corner.msh')
    (tmp_path / 'density.json').write_text(json.dumps({'smooth': smooth.n_triangles, 'corner': corner.n_triangles}))
    assert corner.n_triangles <= 1.5 * smooth.n_triangles, (corner.n_triangles, smooth.n_triangles)
    assert all(row['jumpErrorDeg'] <= .5 for record in
        corner.metadata['freeformReport']['cornerWall']['corners'] for row in record['mesh'].values())


@pytest.mark.parametrize('segments', [12, 48, 96])
def test_concave_mitre_trims_swallowtail_at_any_resolution(tmp_path, segments):
    config = angled_config(5, 65, 'freestanding')
    config['mesh'].update(wall_thickness_mm=5, lengthSegments=segments)
    result = build_from_config(config, tmp_path / 'mitre.msh')
    assert result.n_triangles > 0
    assert all(edge['wallFaces'] == 2 for record in
        result.metadata['freeformReport']['cornerWall']['corners']
        for edge in record['outerSharedEdgeIncidence'])


def test_corner_geometry_asdict_is_json_safe_and_equal():
    from dataclasses import asdict, fields
    from hornlab_mesher.config_builder import resolve_geometry
    geometries = [resolve_geometry(case_config('a')).geometry for _ in range(2)]
    dictionaries = [asdict(geometry) for geometry in geometries]
    assert all('freeform_wall' not in {field.name for field in fields(g)} for g in geometries)
    np.testing.assert_equal(dictionaries[0], dictionaries[1])
    assert json.loads(json.dumps(dictionaries[0], default=str)) == json.loads(json.dumps(dictionaries[1], default=str))
    assert replace(geometries[0], source_shape=0).freeform_wall is geometries[0].freeform_wall


def test_close_interior_corners_refused_at_resolve():
    from hornlab_mesher.config_builder import resolve_geometry
    config = case_config('b')
    for name in ('profileH', 'profileV'):
        config['profile'][name]['points'][2]['z'] = 40.3
    with pytest.raises(ValueError, match=r'corners z=40 mm and z=40.3 mm must be at least 1 mm apart'):
        resolve_geometry(config)


def test_mesh_chord_cap_names_corner_and_spacing_cause(tmp_path, monkeypatch):
    import hornlab_mesher.corner_wall as module
    monkeypatch.setattr(module, 'MAX_MERIDIAN_NODES', 2)
    from hornlab_mesher.mesher import MesherError
    with pytest.raises(MesherError, match=r'corner z=60 mm: mesh meridian needs .*chord refinement cap; corners may be too close to each other or to the throat/mouth'):
        build_from_config(case_config('e'), tmp_path / 'cap.msh')


def test_corner_affine_z_inversion_matches_general_solver():
    from hornlab_mesher.freeform import (_corner_bezier_parameters_at_z,
        _bezier_parameters_at_z)
    controls = np.array([[[0, 1], [20, 8], [40, 12], [60, 15]],
                         [[0, 1], [4, 2], [55, 12], [60, 15]]], dtype=float)
    controls = np.repeat(controls, 121, axis=0)
    z = np.tile(np.linspace(0, 60, 121), 2)
    np.testing.assert_allclose(_corner_bezier_parameters_at_z(controls, z),
                               _bezier_parameters_at_z(controls, z), atol=1e-14)
