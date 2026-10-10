import copy
import math
import numpy as np
import pytest
from hornlab_mesher.config_parser import parse_text_config, ConfigError
from hornlab_mesher.config_builder import resolve_geometry, build_geometry_params, build_from_config
from hornlab_mesher.preview import build_preview_geometry, PreviewOptionsV1
from hornlab_mesher.text_import import TEXT_IMPORT_VERSION_KEY, TEXT_IMPORT_VERSION

BASE = '''OSSE = {
 r0 = 19.5
 a0 = 6
 a = 30
 k = 1
 L = 60
 s = 0
 n = 4
 q = .995
}
Throat.Ext.Length = 12
Throat.Ext.Angle = 5.25
Mesh.Quadrants = 1234
Mesh.AngularSegments = 16
Mesh.LengthSegments = 24
Mesh.WallThickness = 5
Mesh.SubdomainSlices =
Source.Shape = 2
Mesh.ThroatResolution = 8
Mesh.MouthResolution = 12
Mesh.RearResolution = 15
ABEC.SimType = 2
'''


def marked(text=BASE):
    cfg=parse_text_config(text)
    cfg[TEXT_IMPORT_VERSION_KEY]=TEXT_IMPORT_VERSION
    return cfg


@pytest.mark.parametrize('angle', [0,5.25,15])
def test_preview_rear_plane_matches_reference_outer_offset_rule(angle):
    cfg=marked(BASE.replace('Throat.Ext.Angle = 5.25',f'Throat.Ext.Angle = {angle}'))
    preview=build_preview_geometry(cfg, PreviewOptionsV1(lod='coarse'))
    roles={s.role:s for s in preview.surfaces}
    expected=-5*(1+math.sin(math.radians(angle)))
    np.testing.assert_allclose(roles['wall.rear_cap'].positions[:,2], expected, rtol=0, atol=1e-7)
    assert np.min(roles['wall.rear_return'].positions[:,2]) == pytest.approx(expected,rel=0,abs=1e-7)


@pytest.mark.parametrize('depth',[0,-1])
def test_imported_explicit_nonpositive_enclosure_depth_is_named_refusal(depth):
    cfg=marked(BASE+f'Mesh.Enclosure = {{\n Depth = {depth}\n}}\n')
    with pytest.raises(ConfigError,match='ATH enclosure depth.*unsupported'):
        build_geometry_params(cfg)


def test_variable_import_rear_ring_is_named_refusal():
    cfg=marked(BASE.replace('Throat.Ext.Angle = 5.25','Throat.Ext.Angle = 5+3*cos(p)^2'))
    with pytest.raises(ConfigError,match='ATH.*nonplanar rear'):
        resolve_geometry(cfg)


def test_native_rear_and_inner_geometry_remain_unchanged():
    cfg=marked()
    native=copy.deepcopy(cfg)
    native.pop(TEXT_IMPORT_VERSION_KEY)
    imported=resolve_geometry(cfg).geometry
    ordinary=resolve_geometry(native).geometry
    np.testing.assert_array_equal(imported.inner_points, ordinary.inner_points)
    np.testing.assert_array_equal(imported.outer_points, ordinary.outer_points)
    assert not hasattr(ordinary,'ath_text_import_rear')
    preview=build_preview_geometry(native,PreviewOptionsV1(lod='coarse'))
    cap=next(s for s in preview.surfaces if s.role=='wall.rear_cap')
    np.testing.assert_allclose(cap.positions[:,2],-5,rtol=0,atol=1e-9)
    zero=copy.deepcopy(native)
    zero['enclosure']={'depth_mm':0}
    assert build_geometry_params(zero)[2]=='freestanding'


def corner_import_config():
    # Public authored corner morph forces the preview's lazy offset wall path.
    return marked(BASE + '''Morph.TargetShape = 1
Morph.TargetWidth = 150
Morph.TargetHeight = 120
Morph.CornerRadius = 12
Morph.FixedPart = 0
Morph.Rate = 3
''')


@pytest.mark.parametrize('lod', ['coarse', 'fine'])
def test_deferred_corner_preview_uses_production_rear_and_preserves_native(lod, monkeypatch):
    import hornlab_mesher.preview.api as api
    from hornlab_mesher.builders.point_grid_freestanding import _restored_outer_throat_points
    cfg = corner_import_config()
    geom = resolve_geometry(cfg).geometry
    outer = _restored_outer_throat_points(
        geom.inner_points, geom.outer_points, wall_thickness_mm=geom.wall_thickness_mm)
    target = outer[:, 0, 2] - geom.wall_thickness_mm
    expected = float(target.min() + (target.max() - target.min()) / 2)
    assert np.max(np.abs(target - expected)) <= min(1e-4, 1e-4 * geom.wall_thickness_mm)
    original_offset, original_resolve = api._outer_offset_shell, api.resolve_geometry
    offsets, resolutions = [], []
    def capture_offset(*args, **kwargs):
        offsets.append(True)
        return original_offset(*args, **kwargs)
    def capture_resolve(*args, **kwargs):
        resolutions.append(True)
        return original_resolve(*args, **kwargs)
    monkeypatch.setattr(api, '_outer_offset_shell', capture_offset)
    monkeypatch.setattr(api, 'resolve_geometry', capture_resolve)
    preview = build_preview_geometry(cfg, PreviewOptionsV1(lod=lod))
    assert offsets  # Actual deferred path, not an eager-wall stand-in.
    assert len(resolutions) == 1
    roles = {s.role: s for s in preview.surfaces}
    cap = roles['wall.rear_cap']
    np.testing.assert_allclose(cap.positions[:, 2], expected, rtol=0, atol=1e-12)
    rear = roles['wall.rear_return'].positions
    rim = rear[rear[:, 2] == expected]
    assert len(rim) > 2
    for point in rim:
        assert np.any(np.all(cap.positions == point, axis=1))
    native = copy.deepcopy(cfg)
    native.pop(TEXT_IMPORT_VERSION_KEY)
    resolutions.clear()
    ordinary = build_preview_geometry(native, PreviewOptionsV1(lod=lod))
    assert not resolutions  # No new production resolution for native rendering.
    native_cap = next(s for s in ordinary.surfaces if s.role == 'wall.rear_cap')
    np.testing.assert_allclose(native_cap.positions[:, 2], -5, rtol=0, atol=1e-9)
    for role in ('horn.inner', 'horn.outer', 'mouth_rim', 'source_cap'):
        surface = next(s for s in ordinary.surfaces if s.role == role)
        np.testing.assert_array_equal(roles[role].positions, surface.positions)
        np.testing.assert_array_equal(roles[role].indices, surface.indices)


@pytest.mark.parametrize('lod', ['coarse', 'fine'])
@pytest.mark.parametrize('include_cap', [True, False])
def test_deferred_corner_import_refuses_warp_even_with_hidden_rear(lod, include_cap):
    cfg = corner_import_config()
    cfg['profile']['throatExtAngle'] = '5+3*cos(p)^2'
    with pytest.raises(ValueError, match='ATH.*nonplanar rear'):
        build_preview_geometry(cfg, PreviewOptionsV1(lod=lod, include_rear_cap=include_cap))


@pytest.mark.parametrize('include_cap',[True,False])
def test_preview_refuses_warped_import_even_when_rear_cap_hidden(include_cap):
    cfg=marked(BASE.replace('Throat.Ext.Angle = 5.25','Throat.Ext.Angle = 5+3*cos(p)^2'))
    with pytest.raises(ValueError,match='ATH.*nonplanar rear'):
        build_preview_geometry(cfg,PreviewOptionsV1(lod='coarse',include_rear_cap=include_cap))


def test_run21_source_and_main_throat_coordinates_are_preserved():
    cfg=marked('''R-OSSE = {
 r0 = 19.5
 a0 = 5.25
 a = 30
 k = 1
 R = 600
 b = .12
 m = .84
 q = 4.2
 r = .31
 tmax = 1
}
'''+ 'Throat.Ext.Length' + BASE.split('Throat.Ext.Length',1)[1])
    # Retain the Run21 body and its 12 mm extension, without moving the bore.
    geom=resolve_geometry(cfg).geometry
    radius=np.hypot(geom.inner_points[0,:,0],geom.inner_points[0,:,1])
    expected_source_radius=19.5-12*math.tan(math.radians(5.25))
    assert radius[0] == pytest.approx(expected_source_radius,abs=1e-9)
    assert geom.inner_points[0,0,2] == pytest.approx(0,abs=1e-9)
    from hornlab_mesher.profile_formulas import calculate_rosse, rosse_total_length
    params,_,_=build_geometry_params(cfg)
    join_z,join_radius=calculate_rosse(12/rosse_total_length(params),0,params)
    assert join_z == pytest.approx(12,abs=1e-8)
    assert join_radius == pytest.approx(19.5,abs=1e-8)
    native=copy.deepcopy(cfg)
    native.pop(TEXT_IMPORT_VERSION_KEY)
    np.testing.assert_array_equal(geom.inner_points,resolve_geometry(native).geometry.inner_points)
    expected_rear=-5*(1+math.sin(math.radians(5.25)))
    assert np.mean(geom.outer_points[:,0,2])-5 == pytest.approx(expected_rear,abs=1e-7)


def test_imported_geometry_variants_keep_fields_replace_and_pickle():
    import dataclasses
    import pickle
    from hornlab_mesher.rear_compatibility import text_import_geometry_class
    from hornlab_mesher.geometry import (PointGridHornGeometry,_StretchedPointGridHornGeometry,
        _MouthFittedPointGridHornGeometry,_MouthFittedStretchedPointGridHornGeometry)
    geom=resolve_geometry(marked()).geometry
    for cls in (PointGridHornGeometry,_StretchedPointGridHornGeometry,
                _MouthFittedPointGridHornGeometry,_MouthFittedStretchedPointGridHornGeometry):
        mapped=text_import_geometry_class(cls)
        assert [f.name for f in dataclasses.fields(mapped)]==[f.name for f in dataclasses.fields(cls)]
        kwargs={'inner_points':geom.inner_points,'outer_points':geom.outer_points,'wall_thickness_mm':5}
        if 'outer_clearance_points_mm' in {f.name for f in dataclasses.fields(cls)}:
            kwargs['outer_clearance_points_mm']=geom.outer_points
        instance=mapped(**kwargs)
        restored=pickle.loads(pickle.dumps(dataclasses.replace(instance,wall_thickness_mm=6)))
        assert type(restored) is mapped
        assert restored.ath_text_import_rear
        assert restored.wall_thickness_mm==6
        np.testing.assert_array_equal(restored.inner_points,geom.inner_points)


@pytest.mark.parametrize('near_planar', [False, True])
def test_actual_mesh_and_reopened_solid_step_match_rear_plane_and_open_bore(tmp_path, near_planar):
    import gmsh
    import meshio
    from hornlab_mesher.cad import write_step_from_config
    from hornlab_mesher.edges import build_edge_table
    cfg=near_planar_config() if near_planar else marked()
    expected=(certified_rear(cfg)[0,2] if near_planar else
              -5*(1+math.sin(math.radians(5.25))))
    result=build_from_config(cfg,tmp_path/'rear.msh')
    mesh=meshio.read(result.mesh_path)
    blocks=[i for i,b in enumerate(mesh.cells) if b.type=='triangle']
    triangles=np.concatenate([mesh.cells[i].data for i in blocks])
    tags=np.concatenate([mesh.cell_data['gmsh:physical'][i] for i in blocks])
    pts=np.asarray(mesh.points)*(1000 if result.units=='m' else 1)
    assert np.min(pts[:,2])==pytest.approx(expected,rel=0,abs=1e-6)
    assert np.all(build_edge_table(triangles).count==2)
    np.testing.assert_allclose(pts[triangles[tags==2],2],0,rtol=0,atol=1e-8)
    path,info=write_step_from_config(cfg,tmp_path/'rear.step')
    assert info.body=='solid' and info.throat_opened
    assert info.bounding_box_mm[0][2]==pytest.approx(expected,rel=0,abs=1e-5)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber('General.Terminal',0)
        gmsh.model.add('rear-step-reopen')
        gmsh.model.occ.importShapes(str(path),highestDimOnly=False)
        gmsh.model.occ.synchronize()
        volumes=gmsh.model.getEntities(3)
        assert len(volumes)==1
        tag=volumes[0][1]
        assert gmsh.model.occ.getBoundingBox(3,tag)[2]==pytest.approx(expected,rel=0,abs=1e-5)
        # OCC bounding boxes include tolerance padding; measure the reopened
        # planar rear face itself instead of relying only on that bound.
        rear_faces=[]
        for _,face in gmsh.model.getEntities(2):
            center=gmsh.model.occ.getCenterOfMass(2,face)
            bounds=gmsh.model.occ.getBoundingBox(2,face)
            if abs(center[2]-expected)<1e-7 and bounds[5]-bounds[2]<1e-4:
                rear_faces.append(face)
        assert rear_faces
        for face in rear_faces:
            lower,upper=gmsh.model.getParametrizationBounds(2,face)
            uv=(np.asarray(lower)+np.asarray(upper))/2
            assert gmsh.model.getValue(2,face,uv.tolist())[2]==pytest.approx(expected,rel=0,abs=1e-7)
        assert not gmsh.model.isInside(3,tag,[0,0,expected+1])
        assert not gmsh.model.isInside(3,tag,[0,0,15])
        assert gmsh.model.isInside(3,tag,[21,0,expected+1])
        # The reopened bore retains the authored conical prefix. These
        # bracketing probes allow the existing fitted-wall tolerance while
        # discriminating a missing extension or any whole-model translation.
        for z in (0, 6, 12):
            radius = 19.5 - (12-z)*math.tan(math.radians(5.25))
            assert not gmsh.model.isInside(3,tag,[radius-0.03,0,z])
            assert gmsh.model.isInside(3,tag,[radius+0.03,0,z])
    finally:
        gmsh.finalize()


@pytest.mark.parametrize('mode',['bare','enclosure','infinite-baffle'])
def test_marker_does_not_change_nonfreestanding_geometry_type(mode):
    cfg=marked()
    cfg['mode']=mode
    if mode=='enclosure':
        cfg['enclosure']={'depth_mm':120,'edge_mm':0}
    native=copy.deepcopy(cfg)
    native.pop(TEXT_IMPORT_VERSION_KEY)
    imported=resolve_geometry(cfg).geometry
    ordinary=resolve_geometry(native).geometry
    assert type(imported) is type(ordinary)
    assert not hasattr(imported,'ath_text_import_rear')


def test_marker_does_not_break_authored_native_adapter():
    cfg={
        'formula':'OSSE-ADAPTER','mode':'bare',
        'profile':{'L':120,'r0':12.7,'a':40,'a0':15.5,'k':1,'s':0},
        'source':{'source_shape':0},'mesh':{'angular_segments':32,'length_segments':13},
        'throat_adapter':{'mode':'authored','contract_revision':1,'driver_exit_diameter_mm':25.4,
            'exit_half_angle_deg':10,'length_mm':40,'join_t':.137,
            'driver_handle_mm':10,'body_handle_mm':15},
    }
    native=resolve_geometry(cfg).geometry
    cfg[TEXT_IMPORT_VERSION_KEY]=TEXT_IMPORT_VERSION
    imported=resolve_geometry(cfg).geometry
    assert type(imported) is type(native)
    np.testing.assert_array_equal(imported.inner_points,native.inner_points)


@pytest.mark.parametrize('near_planar', [False, True])
@pytest.mark.parametrize('pathway',['acoustic','legacy-wg','legacy-grid'])
def test_all_freestanding_builder_paths_use_same_imported_rear_plane(tmp_path,pathway,near_planar):
    import dataclasses
    import meshio
    from hornlab_mesher.mesher import build_mesh
    from hornlab_mesher.edges import build_edge_table
    cfg=near_planar_config() if near_planar else marked()
    if pathway!='acoustic':
        cfg['mesh']['topology']='legacy'
    resolved=resolve_geometry(cfg)
    geom=resolved.geometry
    if pathway=='legacy-grid':
        geom=dataclasses.replace(geom,wg_topology=False,preserve_grid=True)
    path=build_mesh(geom,resolved.density,tmp_path/(pathway+'.msh'),scale_to_metres=False)
    mesh=meshio.read(path)
    triangles=np.concatenate([cell.data for cell in mesh.cells if cell.type=='triangle'])
    expected=(certified_rear(cfg)[0,2] if near_planar else
              -5*(1+math.sin(math.radians(5.25))))
    assert np.min(mesh.points[:,2])==pytest.approx(expected,rel=0,abs=1e-7)
    assert np.all(build_edge_table(triangles).count==2)


@pytest.mark.parametrize('stretch',[(0,0),(.4,.15)])
def test_actual_resolver_retains_stretch_variant_behavior(stretch):
    cfg=marked()
    cfg['profile']['s1'],cfg['profile']['s2']=stretch
    native=copy.deepcopy(cfg)
    native.pop(TEXT_IMPORT_VERSION_KEY)
    imported=resolve_geometry(cfg).geometry
    ordinary=resolve_geometry(native).geometry
    assert getattr(imported,'quadrant_patch_fit',False)==getattr(ordinary,'quadrant_patch_fit',False)
    assert getattr(imported,'ath_text_import_rear',False)
    np.testing.assert_array_equal(imported.inner_points,ordinary.inner_points)
    assert type(imported).__mro__[2] is type(ordinary)


def near_planar_config():
    # Real angular sampling/normal-offset path, not patched ring data. The
    # authored variation is intentionally small but nonzero and must fit the
    # explicit geometric budget; this does not label it floating-point noise.
    return marked(BASE.replace('Throat.Ext.Angle = 5.25',
                               'Throat.Ext.Angle = 5.25+0.001*cos(p)^2'))


def certified_rear(cfg):
    from hornlab_mesher.rear_compatibility import freestanding_rear_ring
    from hornlab_mesher.builders.point_grid_freestanding import _restored_outer_throat_points
    geom=resolve_geometry(cfg).geometry
    outer=_restored_outer_throat_points(geom.inner_points,geom.outer_points,
                                       wall_thickness_mm=geom.wall_thickness_mm)
    rear=freestanding_rear_ring(geom.inner_points,outer,geom.wall_thickness_mm,
                               text_import=True)
    raw_z=outer[:,0,2]-geom.wall_thickness_mm
    assert np.ptp(raw_z)>1e-7
    np.testing.assert_array_equal(rear[:,:2],outer[:,0,:2])
    assert np.ptp(rear[:,2])==0
    assert np.max(np.abs(rear[:,2]-raw_z))<=min(1e-4,1e-4*geom.wall_thickness_mm)
    return rear


@pytest.mark.parametrize('lod', ['coarse','fine','inspection'])
def test_near_planar_preview_uses_full_ring_certified_plane_at_every_lod(lod,monkeypatch):
    cfg=near_planar_config()
    expected=certified_rear(cfg)[0,2]
    import hornlab_mesher.preview.api as api
    original=api.freestanding_rear_ring
    calls=[]
    def capture(*args,**kwargs):
        ring=original(*args,**kwargs)
        calls.append(ring.copy())
        return ring
    monkeypatch.setattr(api,'freestanding_rear_ring',capture)
    preview=build_preview_geometry(cfg,PreviewOptionsV1(lod=lod))
    assert len(calls)==1  # Full canonical authority, never a new LOD plane.
    # Inspection's independently sampled normals differ by ~1e-12 mm.
    # This cross-grid tolerance is 100,000 times smaller than the rear budget.
    assert calls[0][0,2]==pytest.approx(expected,rel=0,abs=1e-9)
    expected=calls[0][0,2]  # Emitted preview must use its full canonical ring exactly.
    roles={surface.role:surface for surface in preview.surfaces}
    np.testing.assert_allclose(roles['wall.rear_cap'].positions[:,2],expected,
                               rtol=0,atol=1e-12)
    assert np.min(roles['wall.rear_return'].positions[:,2])==pytest.approx(expected,rel=0,abs=1e-12)
    native=copy.deepcopy(cfg)
    native.pop(TEXT_IMPORT_VERSION_KEY)
    imported_geom=resolve_geometry(cfg).geometry
    native_geom=resolve_geometry(native).geometry
    np.testing.assert_array_equal(imported_geom.inner_points,native_geom.inner_points)
    np.testing.assert_array_equal(imported_geom.outer_points,native_geom.outer_points)


@pytest.mark.parametrize('wall', [0.5,5.0])
@pytest.mark.parametrize('factor', [0.999,1.001])
def test_rear_minimax_displacement_budget_has_absolute_and_wall_relative_bounds(wall,factor):
    from hornlab_mesher.rear_compatibility import freestanding_rear_ring
    inner=np.zeros((3,2,3))
    outer=np.ones((3,2,3))
    budget=min(1e-4,1e-4*wall)
    raw_z=np.array([-1,0,1])*budget*factor
    outer[:,0,2]=wall+raw_z
    before=outer.copy()
    if factor>1:
        with pytest.raises(ValueError,match='nonplanar rear'):
            freestanding_rear_ring(inner,outer,wall,text_import=True)
    else:
        rear=freestanding_rear_ring(inner,outer,wall,text_import=True)
        assert np.ptp(rear[:,2])==0
        assert np.max(np.abs(rear[:,2]-(outer[:,0,2]-wall)))<=budget
        np.testing.assert_array_equal(rear[:,:2],outer[:,0,:2])
    np.testing.assert_array_equal(outer,before)
    # The native branch remains its original exact mean-inner anchor.
    native=freestanding_rear_ring(inner,outer,wall)
    np.testing.assert_array_equal(native[:,2],np.full(3,-wall))


@pytest.mark.parametrize('value', [np.nan,np.inf,-np.inf])
@pytest.mark.parametrize('field', ['x','z','wall'])
def test_imported_rear_nonfinite_data_is_refused(value,field):
    from hornlab_mesher.rear_compatibility import freestanding_rear_ring
    inner=np.zeros((3,2,3))
    outer=np.ones((3,2,3))
    wall=5.0
    if field=='wall': wall=value
    else: outer[1,0,0 if field=='x' else 2]=value
    with pytest.raises(ValueError,match='finite'):
        freestanding_rear_ring(inner,outer,wall,text_import=True)


@pytest.mark.parametrize('quadrants', [1234,1])
@pytest.mark.parametrize('topology', ['acoustic','legacy'])
def test_scaled_near_planar_sampling_route_disagreement_is_explicit_refusal(quadrants,topology):
    cfg=near_planar_config()
    cfg['scale']=0.1
    cfg['mesh']['wallThickness']=0.5
    cfg['mesh']['quadrants']=quadrants
    cfg['mesh']['topology']=topology
    # Both route-local rings individually fit the geometric budget, yet their
    # planes disagree by ~0.02249mm. No large projection may reconcile them.
    with pytest.raises(ConfigError,match='rear plane disagrees across supported sampling routes'):
        resolve_geometry(cfg)
    for lod in ('coarse','fine','inspection'):
        for include_cap in (True,False):
            with pytest.raises(ValueError,match='rear plane disagrees across supported sampling routes'):
                build_preview_geometry(cfg,PreviewOptionsV1(lod=lod,include_rear_cap=include_cap))


def test_accepted_near_planar_routes_and_preview_share_selected_production_plane():
    quadrants=1234
    cfg=near_planar_config()
    cfg['mesh']['quadrants']=quadrants
    planes=[]
    for topology in ('acoustic','legacy'):
        cfg['mesh']['topology']=topology
        rear=certified_rear(cfg)
        planes.append(rear[0,2])
        for lod in ('coarse','fine','inspection'):
            preview=build_preview_geometry(cfg,PreviewOptionsV1(lod=lod))
            cap=next(s for s in preview.surfaces if s.role=='wall.rear_cap')
            np.testing.assert_allclose(cap.positions[:,2],rear[0,2],rtol=0,atol=1e-12)
    assert abs(planes[0]-planes[1])<=1e-9


@pytest.mark.parametrize('topology', ['acoustic','legacy'])
def test_new_unscaled_quarter_ring_with_inconsistent_planes_is_also_refused(topology):
    cfg=near_planar_config()
    cfg['mesh']['quadrants']=1
    cfg['mesh']['topology']=topology
    with pytest.raises(ConfigError,match='rear plane disagrees across supported sampling routes'):
        resolve_geometry(cfg)


@pytest.mark.parametrize('mode', ['bare','enclosure','infinite-baffle'])
def test_nonfreestanding_import_marker_preserves_actual_preview_surfaces(mode):
    cfg=marked()
    cfg['mode']=mode
    if mode=='enclosure':cfg['enclosure']={'depth_mm':120,'edge_mm':0}
    native=copy.deepcopy(cfg)
    native.pop(TEXT_IMPORT_VERSION_KEY)
    imported=build_preview_geometry(cfg,PreviewOptionsV1(lod='coarse'))
    ordinary=build_preview_geometry(native,PreviewOptionsV1(lod='coarse'))
    assert [s.role for s in imported.surfaces]==[s.role for s in ordinary.surfaces]
    for actual,expected in zip(imported.surfaces,ordinary.surfaces):
        np.testing.assert_array_equal(actual.positions,expected.positions)
        np.testing.assert_array_equal(actual.indices,expected.indices)
