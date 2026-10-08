"""A standalone source is a closed exterior body, independently of profiles."""
import json
import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from hornlab_mesher import (
    MeshDensity, StandaloneSourceGeometry, build_from_config, build_geometry_params,
    build_mesh_with_info, derive_datums, write_step, write_step_from_config, write_wglink,
)
from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.mesher import MesherError, TriangleBudgetExceeded
from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry
from hornlab_mesher.preview.dimensions import canonical_dimensions
from hornlab_mesher.source_body import certify_mesh, validate_density


def config(**mesh):
    return {"formula":"SOURCE-DISK","mode":"standalone-source",
        "source_body":{"radius_mm":18.,"outer_radius_mm":22.,"depth_mm":8.},
        "mesh":{"size_mm":3.,**mesh}}


@pytest.mark.parametrize("field",["radius_mm","outer_radius_mm","depth_mm","vertical_offset_mm"])
@pytest.mark.parametrize("value",[True,None,"3",float("nan"),float("inf"),[3]])
def test_shape_is_strict_finite_scalar(field,value):
    with pytest.raises(ValueError):
        StandaloneSourceGeometry(**{field:value})


@pytest.mark.parametrize("values",[{"radius_mm":.099},{"radius_mm":201},{"outer_radius_mm":18.05},
    {"outer_radius_mm":501},{"depth_mm":.09},{"depth_mm":501},{"vertical_offset_mm":10001}])
def test_physical_domain(values):
    with pytest.raises(ValueError):
        StandaloneSourceGeometry(**values)


def test_frozen_explicit_shape_and_density_separation(monkeypatch):
    import hornlab_mesher.config_builder as cb
    monkeypatch.setattr(cb,"build_point_grid",lambda *a,**k:pytest.fail("profile evaluation must not run"))
    model=resolve_geometry(config()).geometry
    assert type(model) is StandaloneSourceGeometry
    with pytest.raises(FrozenInstanceError):
        model.radius_mm=20
    assert not hasattr(model,"inner_points") and not hasattr(model,"build_mode")
    assert resolve_geometry(config(size_mm=2)).geometry==model
    assert resolve_geometry(config(size_mm=2)).geometry.fingerprint==model.fingerprint
    assert model.bounds==((-22.,-22.,-8.),(22.,22.,0.))
    assert build_geometry_params(config())[1:]==("SOURCE-DISK","standalone-source")


@pytest.mark.parametrize("where,key,value",[("root","wall_thickness_mm",0),("root","profile",{}),
    ("root","source",{}),("root","scale",1),("root","axial_scale",None),("root","enclosure",None),
    ("body","curvature",0),("body","weights",[]),("mesh","throat_res_mm",3),("mesh","angular_segments",64),
    ("mesh","surface_fit","auto"),("mesh","topology_mode","acoustic"),("mesh","aperture_res_scale",1)])
def test_unsupported_intent_refused_everywhere(where,key,value,tmp_path):
    c=config();target=c if where=="root" else c["source_body" if where=="body" else "mesh"]
    target[key]=value
    for call in (lambda:build_geometry_params(c),lambda:resolve_geometry(c),lambda:canonical_dimensions(c),
        lambda:build_preview_geometry(c),lambda:write_step_from_config(c,tmp_path/'bad.step')):
        with pytest.raises(ValueError):call()
    assert not (tmp_path/'bad.step').exists()


@pytest.mark.parametrize("where",[None,"profile","parameters","mesh","output","enclosure"])
@pytest.mark.parametrize("key",["source_body","sourceBody","SourceBody"])
def test_orphan_and_misplaced_payload_never_disappears(where,key):
    c={"formula":"OSSE"}
    (c if where is None else c.setdefault(where,{}))[key]={}
    with pytest.raises(ValueError,match="source_body"):
        build_geometry_params(c)


@pytest.mark.parametrize("mesh",[{"size_mm":True},{"size_mm":None},{"size_mm":"3"},{"size_mm":0},
    {"size_mm":51},{"scale_to_metres":1},{"allow_large_mesh":"true"},{"quadrants":1234.0},
    {"quadrants":"1234x"},{"quadrants":"1"},{"vertical_offset_mm":"7"},{"vertical_offset_mm":7,"verticalOffset":8},
    {"max_triangles":None},{"max_triangles":True},{"max_triangles":3.5},
    {"max_triangles":18000,"maxTriangles":18000.0}])
def test_invalid_mesh_intent(mesh):
    with pytest.raises(ValueError):resolve_geometry(config(**mesh))


def test_conflicts_and_api_overrides():
    c=config();c["profile"]={"formula":"OSSE"}
    with pytest.raises(ValueError,match="formula"):resolve_geometry(c)
    with pytest.raises(ValueError,match="boolean"):resolve_geometry(config(),allow_large_mesh=1)
    assert resolve_geometry(config(allow_large_mesh=False),allow_large_mesh=True).density.allow_large_mesh
    c=config(scale_to_metres=False,scaleToMetres=False,vertical_offset_mm=7,verticalOffset=7)
    assert resolve_geometry(c).geometry.vertical_offset_mm==7
    assert resolve_geometry(c).scale_to_metres is False


@pytest.mark.parametrize("change",[{"throat_res_mm":2},{"interface_res_mm":3},{"enc_front_res_mm":3},
    {"aperture_res_scale":2},{"aperture_res_scale":True},{"min_size_mm":1},{"max_size_mm":5},
    {"allow_large_mesh":1},{"max_triangles":None},{"max_triangles":3.0}])
def test_direct_density_controls_are_not_ignored(change):
    density=MeshDensity(throat_res_mm=3,mouth_res_mm=3,rear_res_mm=3)
    with pytest.raises(ValueError):validate_density(StandaloneSourceGeometry(),replace(density,**change))


def test_hard_allocation_limit_cannot_be_bypassed(tmp_path):
    c=config(allow_large_mesh=True,size_mm=.1)
    c['source_body']={'radius_mm':.1,'outer_radius_mm':500.,'depth_mm':500.}
    with pytest.raises(ValueError,match="hard"):
        build_from_config(c,tmp_path/'too-large.msh')
    assert not (tmp_path/'too-large.msh').exists()


@pytest.mark.parametrize("lod",["coarse","fine","inspection"])
def test_preview_bounds_roles_signed_curvature_and_density_identity(lod):
    c=config(vertical_offset_mm=7)
    options=PreviewOptionsV1(lod=lod)
    preview=build_preview_geometry(c,options)
    assert {s.role for s in preview.surfaces}=={'source_body.disk','source_body.annulus','source_body.side','source_body.rear'}
    assert preview.metadata['boundsMm']==((-22.,-15.,-8.),(22.,29.,0.))
    assert preview.metadata['dimensions_mm']=={'source_diameter':[36.,36.],'body_overall':[44.,44.,8.]}
    assert all('mouth' not in key.lower() and 'throat' not in key.lower() for key in preview.metadata['datums'])
    other=build_preview_geometry(config(vertical_offset_mm=7,size_mm=2),options)
    assert preview.metadata==other.metadata
    for s,t in zip(preview.surfaces,other.surfaces):
        assert np.array_equal(s.positions,t.positions) and np.array_equal(s.indices,t.indices)
        assert s.metadata['orientation']=='exterior'
        fidelity=preview.metadata['fidelity'][s.role]
        assert fidelity['max_chord_error_mm_achieved']<=fidelity['max_chord_error_mm_requested']
        if s.role=='source_body.side':
            np.testing.assert_allclose(s.curvature_mean,-1/44)
            np.testing.assert_allclose(s.curvature_principal,-1/22)
        else:assert not np.any(s.curvature_mean)


def test_preview_selection_budgets_and_owned_snapshot(monkeypatch):
    from hornlab_mesher.preview import source_body
    c=config()
    expected=build_preview_geometry(c)
    original=source_body.build
    def mutate(params,options):
        c['source_body']['radius_mm']=1
        c['mesh']['vertical_offset_mm']=10
        return original(params,options)
    monkeypatch.setattr(source_body,'build',mutate)
    actual=build_preview_geometry(c)
    assert actual.metadata==expected.metadata
    for s,t in zip(actual.surfaces,expected.surfaces):assert np.array_equal(s.positions,t.positions)
    empty=build_preview_geometry(config(),PreviewOptionsV1(include_outer=False,include_source_cap=False,include_rear_cap=False,max_vertices=8))
    assert empty.surfaces==[]
    with pytest.raises(ValueError,match="vertex"):
        build_preview_geometry(config(),PreviewOptionsV1(max_vertices=8))
    with pytest.raises(ValueError,match="precision"):
        build_preview_geometry(config(vertical_offset_mm=10000),PreviewOptionsV1(max_chord_error_mm=1e-6))


@pytest.mark.parametrize("r,R,d,size,y,metres",[(5,8,3,1,0,True),(18,22,8,3,7,False),(40,50,12,5,-11,True),(.1,.2,.1,.1,0,False)])
def test_actual_mesh_step_datums_and_identity(r,R,d,size,y,metres,tmp_path,monkeypatch):
    import gmsh
    import hornlab_mesher.mesher as mesher
    c=config(size_mm=size,vertical_offset_mm=y,scale_to_metres=metres)
    c['source_body']={'radius_mm':r,'outer_radius_mm':R,'depth_mm':d}
    monkeypatch.setattr(mesher,'_weld_near_duplicate_vertices',lambda *a,**k:pytest.fail("exact source body must not approximately weld"))
    result=build_from_config(c,tmp_path/'body.msh')
    assert result.physical_groups=={1:'SD1G0',2:'SD1D1001'}
    assert result.native_check_open_edges and result.native_symmetry_plane is None
    assert result.units==('m' if metres else 'mm')
    metadata=result.metadata['sourceBody']
    assert metadata['edgeIncidence']==2 and metadata['connectedComponents']==1
    assert metadata['sourceAreaRelativeError']<.01 and metadata['volumeRelativeError']<.01
    assert metadata['effectiveMeshSizeMm']==min(size,r/8)
    path,info=write_step_from_config(c,tmp_path/'body.step')
    assert info.body=='solid' and info.n_faces==4 and not info.throat_opened
    assert math.isclose(info.volume_mm3,math.pi*R*R*d,rel_tol=1e-8)
    assert info.source_body['constructionFingerprint']==metadata['constructionFingerprint']
    assert 'hornlab-source-body' in path.read_text()
    assert info.bounding_box_mm==((-R,y-R,-d),(R,y+R,0.))
    model=resolve_geometry(c).geometry
    assert derive_datums(model,None)==model.datums()
    with pytest.raises(MesherError,match='source-only'):
        write_wglink(model,tmp_path/'body.wglink')
    assert not (tmp_path/'body.wglink').exists()
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal',0);gmsh.open(str(path))
        volumes=gmsh.model.getEntities(3)
        assert len(volumes)==1
        faces=[abs(t) for _,t in gmsh.model.getBoundary(volumes,oriented=False)]
        assert len(faces)==4
        source=[t for t in faces if abs(gmsh.model.occ.getMass(2,t)-math.pi*r*r)<1e-7*r*r and abs(gmsh.model.occ.getCenterOfMass(2,t)[2])<1e-7]
        assert len(source)==1
        rim={abs(t) for _,t in gmsh.model.getBoundary([(2,source[0])],oriented=False)}
        assert len([t for t in faces if t!=source[0] and rim.intersection({abs(e) for _,e in gmsh.model.getBoundary([(2,t)],oriented=False)})])==1
    finally:gmsh.finalize()


def test_candidate_budget_and_certificate_refusal_preserve_target(tmp_path,monkeypatch):
    import hornlab_mesher.source_body as sb
    path=tmp_path/'existing.msh';path.write_text('keep me')
    with pytest.raises(TriangleBudgetExceeded):build_from_config(config(max_triangles=1300),path)
    assert path.read_text()=='keep me'
    def reject(*a,**k):raise MesherError('candidate certificate refusal')
    monkeypatch.setattr(sb,'certify_mesh',reject)
    with pytest.raises(MesherError,match='certificate'):build_from_config(config(),path)
    assert path.read_text()=='keep me'


def test_direct_api_default_and_caller_options_restored(tmp_path):
    import gmsh
    gmsh.initialize()
    try:
        originals={'Mesh.MeshSizeMin':.7,'Mesh.MeshSizeMax':8.,'Mesh.MeshSizeFromPoints':1.,'Mesh.MeshSizeExtendFromBoundary':0.,'General.Terminal':0.}
        for name,value in originals.items():gmsh.option.setNumber(name,value)
        _,info=build_mesh_with_info(StandaloneSourceGeometry(),output_path=tmp_path/'direct.msh')
        assert info.metadata['sourceBody']['requestedMeshSizeMm']==3
        assert gmsh.isInitialized()
        for name,value in originals.items():assert gmsh.option.getNumber(name)==value
    finally:gmsh.finalize()


@pytest.mark.parametrize("config_format",["json","toml"])
def test_actual_cli_json_mesh_and_step_summary(tmp_path,capsys,config_format):
    from hornlab_mesher.cli import main
    path=tmp_path/('source.'+config_format)
    path.write_text(json.dumps(config()) if config_format=='json' else 'formula=\"SOURCE-DISK\"\nmode=\"standalone-source\"\n[source_body]\nradius_mm=18\nouter_radius_mm=22\ndepth_mm=8\n[mesh]\nsize_mm=3\n')
    assert main([str(path),'-o',str(tmp_path/'cli.msh'),'--print-summary'])==0
    summary=json.loads(capsys.readouterr().out)
    assert summary['mode']=='standalone-source' and summary['metadata']['sourceBody']['edgeIncidence']==2
    assert main([str(path),'--step',str(tmp_path/'cli.step'),'--print-summary'])==0
    summary=json.loads(capsys.readouterr().out)
    assert summary['body']=='solid' and not summary['throat_opened']
    assert summary['source_body']['formula']=='SOURCE-DISK'


def test_canonical_dimensions_no_horn_surrogates():
    assert canonical_dimensions(config(vertical_offset_mm=7))=={'source_diameter':[36.,36.],'body_overall':[44.,44.,8.]}


def test_horn_only_helpers_refuse_the_explicit_source_body():
    from hornlab_mesher import profile_formulas as pf
    from hornlab_mesher.profile_sampling import build_point_grid,build_point_grid_arrays
    params=build_geometry_params(config())[0]
    calls=[lambda:pf.calculate_osse(0,0,params),lambda:pf.calculate_osse_curve([0],0,params),
        lambda:pf.osse_length_config(params),lambda:pf.osse_total_length(params),lambda:pf.rosse_total_length(params),
        lambda:pf.rosse_axial_layout(params),lambda:pf.calculate_rosse(0,0,params),lambda:pf.calculate_rosse_curve([0],0,params),
        lambda:build_point_grid(params),lambda:build_point_grid_arrays(params)]
    for call in calls:
        with pytest.raises(ValueError,match='standalone source'):call()


@pytest.mark.parametrize("kind",["hole","bad_tag","source_plane","source_normal","side_normal","outside_envelope","extra_point"])
def test_candidate_certificate_detects_actual_corruption(kind,tmp_path):
    import meshio
    model=StandaloneSourceGeometry()
    _,info=build_mesh_with_info(model,output_path=tmp_path/'valid.msh',scale_to_metres=False)
    mesh=meshio.read(info.path)
    triangles=mesh.cells[0].data.copy()
    tags=mesh.cell_data['gmsh:physical'][0].copy()
    points=mesh.points.copy()
    if kind=='hole':
        triangles=triangles[:-1];tags=tags[:-1]
    elif kind=='bad_tag':tags[0]=3
    elif kind=='source_plane':points[triangles[np.flatnonzero(tags==2)[0],0],2]+=.01
    elif kind=='source_normal':
        i=np.flatnonzero(tags==2)[0];triangles[i]=triangles[i,[0,2,1]]
    elif kind=='outside_envelope':points[triangles[0,0],2]=1
    elif kind=='extra_point':points=np.vstack((points,[0.,0.,0.]))
    else:
        side=(np.ptp(points[triangles][:,:,2],axis=1)>0)&(tags==1)
        i=np.flatnonzero(side)[0];triangles[i]=triangles[i,[0,2,1]]
    corrupted=meshio.Mesh(points,[('triangle',triangles)],cell_data={'gmsh:physical':[tags]},field_data=mesh.field_data)
    path=tmp_path/'corrupt.msh';meshio.write(path,corrupted,file_format='gmsh22',binary=False)
    with pytest.raises(MesherError,match='Standalone source mesh refused'):
        certify_mesh(model,path,'mm')


@pytest.mark.parametrize("r,R,d,size,y",[(200,500,500,50,10000),(200,200.1,.1,50,-10000),(18,18.1,.1,3,7)])
def test_actual_bounded_large_and_thin_domains(r,R,d,size,y,tmp_path):
    c=config(size_mm=size,vertical_offset_mm=y,allow_large_mesh=True)
    c['source_body']={'radius_mm':r,'outer_radius_mm':R,'depth_mm':d}
    result=build_from_config(c,tmp_path/'domain.msh')
    assert result.n_triangles<200000
    assert result.metadata['sourceBody']['edgeIncidence']==2


@pytest.mark.parametrize("relative_delta",[0.,1e-9])
@pytest.mark.parametrize("route",["config","direct"])
def test_equal_area_source_and_annulus_keep_the_actual_inner_disk(relative_delta,route,tmp_path):
    import gmsh
    import meshio
    r=5.;R=math.sqrt(2)*r*(1+relative_delta);d=3.
    model=StandaloneSourceGeometry(r,R,d)
    c=config();c['source_body']={'radius_mm':r,'outer_radius_mm':R,'depth_mm':d}
    msh=tmp_path/'equal.msh';step=tmp_path/'equal.step'
    # A valid build replaces an existing target only with its final product.
    msh.write_text('old mesh');step.write_text('old step')
    if route=='config':
        result=build_from_config(c,msh)
        _,cad=write_step_from_config(c,step)
        metadata=result.metadata['sourceBody']
    else:
        _,info=build_mesh_with_info(model,output_path=msh)
        _,cad=write_step(model,step)
        metadata=info.metadata['sourceBody']
    assert metadata['edgeIncidence']==2 and metadata['connectedComponents']==1
    assert metadata['sourceRimErrorMm']<2e-7 and metadata['sourceAreaRelativeError']<.01
    mesh=meshio.read(msh)
    tags=mesh.cell_data['gmsh:physical'][0];tri=mesh.cells[0].data
    points=mesh.points*1000
    assert set(tags)=={1,2}
    source_points=points[tri[tags==2]]
    assert np.max(np.hypot(source_points[:,:,0],source_points[:,:,1]))<=r+2e-7
    assert cad.body=='solid' and cad.n_faces==4 and not cad.throat_opened
    assert math.isclose(cad.volume_mm3,math.pi*R*R*d,rel_tol=1e-8)
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal',0);gmsh.open(str(step))
        volumes=gmsh.model.getEntities(3);assert len(volumes)==1
        faces=[abs(t) for _,t in gmsh.model.getBoundary(volumes,oriented=False)]
        assert len(faces)==4
        # Identify the reopened inner disk by one boundary curve and its
        # radius, independently of either planar face's area or fragment map.
        disks=[]
        for face in faces:
            curves=[abs(t) for _,t in gmsh.model.getBoundary([(2,face)],oriented=False)]
            if len(curves)!=1 or gmsh.model.getType(2,face)!='Plane':continue
            if abs(gmsh.model.occ.getCenterOfMass(2,face)[2])>1e-7:continue
            if math.isclose(gmsh.model.occ.getMass(1,curves[0]),2*math.pi*r,rel_tol=1e-8):
                disks.append((face,curves[0]))
        assert len(disks)==1
        face,rim=disks[0]
        assert math.isclose(gmsh.model.occ.getMass(2,face),math.pi*r*r,rel_tol=1e-8)
        annulus=[t for t in faces if t!=face and rim in {abs(e) for _,e in gmsh.model.getBoundary([(2,t)],oriented=False)}]
        assert len(annulus)==1
        assert math.isclose(gmsh.model.occ.getMass(2,annulus[0]),math.pi*(R*R-r*r),rel_tol=1e-8)
    finally:gmsh.finalize()
