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
    assert np.min(roles['wall.rear_return'].positions[:,2]) == pytest.approx(expected,abs=1e-7)


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


def test_actual_mesh_and_reopened_solid_step_match_rear_plane_and_open_bore(tmp_path):
    import gmsh
    import meshio
    from hornlab_mesher.cad import write_step_from_config
    from hornlab_mesher.edges import build_edge_table
    cfg=marked()
    expected=-5*(1+math.sin(math.radians(5.25)))
    result=build_from_config(cfg,tmp_path/'rear.msh')
    mesh=meshio.read(result.mesh_path)
    blocks=[i for i,b in enumerate(mesh.cells) if b.type=='triangle']
    triangles=np.concatenate([mesh.cells[i].data for i in blocks])
    tags=np.concatenate([mesh.cell_data['gmsh:physical'][i] for i in blocks])
    pts=np.asarray(mesh.points)*(1000 if result.units=='m' else 1)
    assert np.min(pts[:,2])==pytest.approx(expected,abs=1e-6)
    assert np.all(build_edge_table(triangles).count==2)
    np.testing.assert_allclose(pts[triangles[tags==2],2],0,rtol=0,atol=1e-8)
    path,info=write_step_from_config(cfg,tmp_path/'rear.step')
    assert info.body=='solid' and info.throat_opened
    assert info.bounding_box_mm[0][2]==pytest.approx(expected,abs=1e-5)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber('General.Terminal',0)
        gmsh.model.add('rear-step-reopen')
        gmsh.model.occ.importShapes(str(path),highestDimOnly=False)
        gmsh.model.occ.synchronize()
        volumes=gmsh.model.getEntities(3)
        assert len(volumes)==1
        tag=volumes[0][1]
        assert gmsh.model.occ.getBoundingBox(3,tag)[2]==pytest.approx(expected,abs=1e-5)
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
            assert gmsh.model.getValue(2,face,uv.tolist())[2]==pytest.approx(expected,abs=1e-7)
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


@pytest.mark.parametrize('pathway',['acoustic','legacy-wg','legacy-grid'])
def test_all_freestanding_builder_paths_use_same_imported_rear_plane(tmp_path,pathway):
    import dataclasses
    import meshio
    from hornlab_mesher.mesher import build_mesh
    from hornlab_mesher.edges import build_edge_table
    cfg=marked()
    if pathway!='acoustic':
        cfg['mesh']['topology']='legacy'
    resolved=resolve_geometry(cfg)
    geom=resolved.geometry
    if pathway=='legacy-grid':
        geom=dataclasses.replace(geom,wg_topology=False,preserve_grid=True)
    path=build_mesh(geom,resolved.density,tmp_path/(pathway+'.msh'),scale_to_metres=False)
    mesh=meshio.read(path)
    triangles=np.concatenate([cell.data for cell in mesh.cells if cell.type=='triangle'])
    expected=-5*(1+math.sin(math.radians(5.25)))
    assert np.min(mesh.points[:,2])==pytest.approx(expected,abs=1e-7)
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
