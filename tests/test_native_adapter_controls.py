"""Literal cubic controls, deliberate join corners and complete native routes."""
import copy
import math

import numpy as np
import pytest

from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.config_parser import ConfigError
from hornlab_mesher.preview import build_preview_geometry, PreviewOptionsV1


def config(points=None):
    return {"formula":"OSSE-ADAPTER-CONTROLS","mode":"bare","scale":.25,
        "profile":{"L_mm":137,"r0_mm":17,"a_deg":37,"a0_deg":6,"k":1.25,"s":0},
        "throat_adapter":{"mode":"controls","contract_revision":1,"join_t":0,
            "control_points_mm":points or [[-31.7,16.2],[-22.1,16.9],[-9.3,20.4]]},
        "source":{"source_shape":0},"mesh":{"wall_thickness_mm":0,
            "throat_res_mm":1,"mouth_res_mm":1.5,"rear_res_mm":2,
            "max_triangles":250000,"allow_large_mesh":True,"scale_to_metres":False}}


@pytest.mark.parametrize("mode",["Controls","CONTROLS"])
def test_misplaced_case_variant_controls_intent_refuses(mode):
    c={"formula":"OSSE","extra":{"throat_adapter":{"mode":mode,"contract_revision":1}}}
    with pytest.raises(ConfigError):resolve_geometry(c)


@pytest.mark.parametrize("field,value",[("s",1),("sourceShape",1),("wallThickness",2),("type","OSSE")])
def test_direct_model_active_normalized_intent_refuses(field,value):
    from hornlab_mesher.adapter_controls import ControlsMeridian
    from hornlab_mesher.config_builder import build_geometry_params
    params=build_geometry_params(config())[0];params[field]=value
    with pytest.raises(ConfigError):ControlsMeridian.from_params(params)


def test_interior_envelope_local_sizing_stays_within_fixed_resource_budget():
    gmsh=pytest.importorskip("gmsh")
    from hornlab_mesher.builders.adapter_controls import build_controls,configure_density
    c=config([[-30,12],[-20,55],[-8,48]])
    c["profile"].update(L_mm=40,a_deg=8,a0_deg=2)
    resolved=resolve_geometry(c)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.model.add("interior-envelope-estimate")
        built=build_controls(resolved.geometry);configure_density(built,resolved.density)
        assert built.metadata["meshTriangleEstimate"]<=250000
    finally:gmsh.finalize()


@pytest.mark.parametrize("field,value",[("s",False),("sourceShape","0"),("wallThickness",None),
    ("sourceRadius",3),("sourceCurv",1),("throatExtLength",2),("throatExtAngle",5),
    ("rot",4),("h",1),("morphTarget",1),("gcurveType",1),("zMapPoints",[]),
    ("subdomainSlices","1"),("interfaceOffset",0),("lookupProfile",[]),
    ("profileSystem",{"crossSection":{"exponent":4,"aspectRatio":1}}),
    ("quadrants","12"),("angularSegments",True),("throatResolution","1"),
    ("controlsFingerprint","stale"),("formula","OSSE"),("axialScale",1)])
def test_direct_model_normalized_controls_and_types_are_strict(field,value):
    from hornlab_mesher.adapter_controls import ControlsMeridian
    from hornlab_mesher.config_builder import build_geometry_params
    params=build_geometry_params(config())[0];params[field]=value
    with pytest.raises(ConfigError):ControlsMeridian.from_params(params)


def test_direct_model_diagnostics_and_density_do_not_change_identity():
    from hornlab_mesher.adapter_controls import ControlsMeridian
    from hornlab_mesher.config_builder import build_geometry_params
    params=build_geometry_params(config())[0];original=ControlsMeridian.from_params(params)
    params.update(angularSegments=128,lengthSegments=80,throatResolution=.5,mouthResolution=.75,rearResolution=1)
    changed=ControlsMeridian.from_params(params)
    assert changed.fingerprint==original.fingerprint
    np.testing.assert_array_equal(changed.cubic_poles,original.cubic_poles)
    c=config();c["mesh"].update(scale_to_metres=True);c["output"]={"path":"elsewhere.msh"}
    assert resolve_geometry(c).geometry.controls_meridian.fingerprint==original.fingerprint


@pytest.mark.parametrize("points",[[[-30,12],[-20,55],[-8,48]],[[-30,5],[-5,-1],[-20,10]],
    [[-30,12],[-10,14],[-14,20]]])
def test_whole_interval_meridian_speed_bound_encloses_derivatives(points):
    m=resolve_geometry(config(points)).geometry.controls_meridian
    for branch in ("cubic","body"):
        for lo,hi in ((0,.1),(.1,.3),(.3,.7),(.7,1),(0,1)):
            bound=m.sizing_speed_bound(branch,lo,hi)
            speed=np.linalg.norm(m.branch(branch,np.linspace(lo,hi,101))[1],axis=-1)
            # A wide hull can contain zero despite a strictly forward curve;
            # sizing subdivides such unresolved intervals before use.
            assert 0<=bound<=speed.min()
            p=m.branch(branch,np.asarray([lo,(lo+hi)/2,hi]))[0]
            assert np.min(np.linalg.norm(np.diff(p,axis=0),axis=1)/((hi-lo)/2))>=bound
    if points[0][1]==12 and points[1][1]==55:
        assert m.sizing_speed_bound("cubic",0,.1)>2*m.interval_bounds("cubic",0,.1)[4]


@pytest.mark.parametrize("points",[None,[[-30,12],[-10,14],[-14,20]],[[-30,5],[-5,-1],[-20,10]],
    [[-30,14],[-20,11],[-8,19]]])
def test_literal_independent_controls_and_whole_curve_certificates(points):
    c=config(points);m=resolve_geometry(c).geometry.controls_meridian
    poles=np.asarray(c["throat_adapter"]["control_points_mm"]+[[0,17]],dtype=float)
    poles[:,0]-=poles[0,0];poles*=.25
    u=np.asarray([0,.07,.37,.5,.81,1]);v=1-u
    expected=v[:,None]**3*poles[0]+3*v[:,None]**2*u[:,None]*poles[1]+3*v[:,None]*u[:,None]**2*poles[2]+u[:,None]**3*poles[3]
    np.testing.assert_allclose(m.cubic(u)[0],expected,rtol=0,atol=1e-12)
    assert m.minimum_z_derivative>1e-4 and m.minimum_radius>1e-4
    assert m.source_angle_deg==pytest.approx(math.degrees(math.atan2(poles[1,1]-poles[0,1],poles[1,0]-poles[0,0])))


@pytest.mark.parametrize("points",[[[-30,1],[0,-8],[-30,-8]],[[-30,1],[-20,-8],[-10,-8]],
    [[-30,0],[-20,4],[-10,8]]])
def test_whole_curve_invalids_refuse(points):
    with pytest.raises(ConfigError):resolve_geometry(config(points))


def test_control_mutations_are_independent_and_authority_is_immutable():
    c=config();m=resolve_geometry(c).geometry.controls_meridian
    baseline=m.cubic(.37)[0]
    for row in range(3):
        for col in range(2):
            other=copy.deepcopy(c);other["throat_adapter"]["control_points_mm"][row][col]+=.73
            changed=resolve_geometry(other).geometry.controls_meridian
            assert changed.fingerprint!=m.fingerprint
            assert not np.array_equal(changed.cubic(.37)[0],baseline)
    for a in (m.cubic_poles,m.body_poles,m.body_weights):
        with pytest.raises(ValueError):a.setflags(write=True)
    identity=m.identity;identity["controls"]["control_points_mm"][2][1]=900
    assert m.identity["controls"]["control_points_mm"][2][1]==20.4


@pytest.mark.parametrize("lod",["coarse","fine","inspection"])
def test_preview_keeps_both_join_sides(lod):
    preview=build_preview_geometry(config(),PreviewOptionsV1(lod=lod))
    wall=next(s for s in preview.surfaces if s.role=="horn.inner")
    branches=wall.metadata["branchRanges"]
    left,right=branches;az=wall.metadata["angularSegments"]
    a=left["vertexEnd"]-az;b=right["vertexStart"]
    np.testing.assert_array_equal(wall.positions[a:a+az].astype(np.float32),wall.positions[b:b+az].astype(np.float32))
    assert np.max(np.linalg.norm(wall.normals[a:a+az]-wall.normals[b:b+az],axis=1))>.1
    tri=wall.indices.reshape(-1,3)
    assert not np.any((tri.min(axis=1)<b)&(tri.max(axis=1)>=b))
    assert preview.metadata["fidelity"]["horn.inner"]["max_chord_error_mm"]<=.008


@pytest.mark.parametrize("patch",[{"contract_revision":True},{"join_t":.01},{"mode":"off"},
    {"control_points_mm":[[-30,1],[-20,2],[-10,3],[0,4]]},{"driver_handle_mm":2}])
def test_exact_payload_is_strict(patch):
    c=config();c["throat_adapter"].update(patch)
    with pytest.raises(ConfigError):resolve_geometry(c)


def test_actual_small_mesh_has_complete_facet_certificate_and_shared_source(tmp_path,monkeypatch):
    meshio=pytest.importorskip("meshio");gmsh=pytest.importorskip("gmsh")
    from hornlab_mesher.config_builder import build_from_config
    from hornlab_mesher import mesher
    gmsh.initialize(interruptible=False)
    try:
        gmsh.model.add("caller-mesh");point=gmsh.model.occ.addPoint(3,4,5);gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MinimumCirclePoints",23)
        result=build_from_config(config(),tmp_path/"controls.msh")
        assert gmsh.model.getCurrent()=="caller-mesh" and gmsh.model.getEntities(0)==[(0,point)]
        assert gmsh.option.getNumber("Mesh.MinimumCirclePoints")==23
        path=tmp_path/"sentinel.msh";path.write_bytes(b"existing")
        calls=[]
        def fail(*args):calls.append(1);raise RuntimeError("injected native builder failure")
        monkeypatch.setattr(mesher,"_dispatch_builder",fail)
        with pytest.raises(mesher.MesherError,match="injected native builder"):build_from_config(config(),path)
        assert calls==[1] and path.read_bytes()==b"existing"
        assert gmsh.model.getCurrent()=="caller-mesh" and gmsh.model.getEntities(0)==[(0,point)]
        assert gmsh.option.getNumber("Mesh.MinimumCirclePoints")==23
    finally:gmsh.finalize()
    assert result.metadata["controlsMeshChordBoundMm"]<=.01
    mesh=meshio.read(result.mesh_path);t=mesh.get_cells_type("triangle");p=mesh.points
    g=mesh.get_cell_data("gmsh:physical","triangle");s=t[g==2]
    area=np.linalg.norm(np.cross(p[s[:,1]]-p[s[:,0]],p[s[:,2]]-p[s[:,0]]),axis=1).sum()/2
    assert area/(math.pi*(16.2*.25)**2)>.96
    assert set(mesh.get_cell_data("gmsh:geometrical","triangle")[g==1])==set(sum(result.metadata["controlsBranchSurfaceTags"].values(),[]))
    edges=np.sort(np.concatenate((t[:,[0,1]],t[:,[1,2]],t[:,[2,0]])),axis=1)
    unique,count=np.unique(edges,axis=0,return_counts=True)
    assert count.max()==2
    np.testing.assert_allclose(p[unique[count==1]][...,2],(.25*(31.7+137)),rtol=0,atol=1e-7)
    join=unique[np.all(np.abs(p[unique][...,2]-.25*31.7)<1e-7,axis=1)]
    assert len(join)>=16
    source_edges=np.sort(np.concatenate((s[:,[0,1]],s[:,[1,2]],s[:,[2,0]])),axis=1)
    source_unique,source_count=np.unique(source_edges,axis=0,return_counts=True)
    rim_nodes=np.unique(source_unique[source_count==1])
    assert len(rim_nodes)>=16
    np.testing.assert_allclose(np.hypot(p[rim_nodes,0],p[rim_nodes,1]),16.2*.25,rtol=0,atol=1e-7)
    assert np.all(np.cross(p[s[:,1]]-p[s[:,0]],p[s[:,2]]-p[s[:,0]])[:,2]>0)


@pytest.mark.parametrize("opened",[True,False])
def test_actual_step_reopens_shared_zero_volume_shell(tmp_path,opened):
    import collections
    gmsh=pytest.importorskip("gmsh")
    from hornlab_mesher.cad import write_step_from_config
    path,info=write_step_from_config(config(),tmp_path/"controls.step",open_throat=opened)
    assert info.body=="surface" and info.volume_mm3 is None and info.n_faces==(8 if opened else 9)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal",0);gmsh.open(str(path))
        assert not gmsh.model.getEntities(3)
        counts=collections.Counter(abs(t) for _,f in gmsh.model.getEntities(2)
            for d,t in gmsh.model.getBoundary([(2,f)],combined=False,oriented=True) if d==1)
        assert collections.Counter(counts.values())==({1:8,2:12} if opened else {1:4,2:16})
    finally:gmsh.finalize()


def test_configured_density_evaluates_on_real_curves_in_subprocess():
    import subprocess,sys,json
    code="""import json,gmsh
from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.builders.adapter_controls import build_controls,configure_density
r=resolve_geometry(json.loads(CONFIG));gmsh.initialize(interruptible=False)
gmsh.option.setNumber('General.Terminal',0);gmsh.model.add('curve-regression')
b=build_controls(r.geometry);configure_density(b,r.density);gmsh.model.mesh.generate(1)
assert len(gmsh.model.mesh.getNodes()[0])>16
assert all(gmsh.model.mesh.getElements(1,c)[0].size for _,c in gmsh.model.getEntities(1))
gmsh.finalize()
""".replace("CONFIG",repr(json.dumps(config())))
    result=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize("section,key,value",[("root","axial_scale",1),("root","terminating_arc",None),
    ("root","parameters",{}),("profile","n",4),("profile","s",1),("profile","formula","OSSE"),
    ("mesh","wall_thickness_mm",1),("mesh","topology_mode","legacy"),("mesh","surface_fit","approximate"),
    ("mesh","allow_large_mesh","false"),("mesh","scale_to_metres",1),("mesh","max_triangles",10.),
    ("mesh","throat_res_mm",True),("mesh","quadrants",12),("source","source_shape",1),
    ("source","source_auto_angle_deg",0),("source","source_curv",0),("root","scale",float("nan")),
    ("root","vertical_offset_mm",100001)])
def test_unsupported_original_controls_refuse(section,key,value):
    c=config();target=c if section=="root" else c[section];target[key]=value
    with pytest.raises((ConfigError,ValueError)):resolve_geometry(c)


@pytest.mark.parametrize("value",[True,False,"0.25",None,float("nan"),float("inf"),10**400])
def test_original_root_scale_requires_finite_numeric_scalar(value):
    c=config();c["scale"]=value
    with pytest.raises(ConfigError):resolve_geometry(c)


@pytest.mark.parametrize("value",["false",0,1,None])
def test_native_cad_full_model_flag_requires_boolean(value,tmp_path):
    from hornlab_mesher.cad import write_step_from_config
    from hornlab_mesher.mesher import MesherError
    with pytest.raises(MesherError,match="must be a boolean"):write_step_from_config(config(),tmp_path/"bad.step",full_model=value)


@pytest.mark.parametrize("value",[None,[],[[-30,1],[-20,2]],[[True,1],[-20,2],[-10,3]],
    [[-30,float("nan")],[-20,2],[-10,3]],[[-30,float("inf")],[-20,2],[-10,3]],
    [[-30,1],[-20,"2"],[-10,3]],[[-30,1],[-20,2],[-10,10**400]]])
def test_noncanonical_control_coordinates_refuse(value):
    c=config();c["throat_adapter"]["control_points_mm"]=value
    with pytest.raises((ConfigError,ValueError)):resolve_geometry(c)


@pytest.mark.parametrize("section",["profile","mesh","source","extra"])
def test_misplaced_payload_cannot_disappear(section):
    c=config();payload=c.pop("throat_adapter");c["formula"]="OSSE"
    c.setdefault(section,{})["throat_adapter"]=payload
    with pytest.raises(ConfigError):resolve_geometry(c)


def test_deeply_nested_intent_cannot_become_ordinary_body():
    c=config();c.pop("throat_adapter");c["formula"]="OSSE"
    c["extra"]=[[{"controlPointsMm":None}]]
    with pytest.raises(ConfigError):resolve_geometry(c)


@pytest.mark.parametrize("patch",[{"interfaces":("active",)},{"interface_offset_mm":1},
    {"preserve_grid":True},{"wg_topology":False},{"source_auto_angle_deg":0},
    {"source_shape":1},{"surface_fit":"approximate"},{"topology_mode":"legacy"},
    {"vertical_offset_mm":2},{"controls_meridian":None},{"controls_meridian":object()}])
def test_direct_geometry_refuses_before_dispatch(patch):
    from dataclasses import replace
    with pytest.raises(ValueError):replace(resolve_geometry(config()).geometry,**patch)


@pytest.mark.parametrize("patch",[{"allow_large_mesh":"false"},{"allow_large_mesh":1},
    {"max_triangles":None},{"max_triangles":True},{"max_triangles":10.5},{"max_triangles":10**400},
    {"throat_res_mm":"1"},{"mouth_res_mm":float("nan")},{"rear_res_mm":False},
    {"min_size_mm":.01},{"interface_res_mm":1},{"aperture_res_scale":2}])
def test_direct_density_refuses_before_native_allocation(patch,tmp_path,monkeypatch):
    from dataclasses import replace
    from hornlab_mesher.geometry import MeshDensity
    from hornlab_mesher import mesher
    def forbidden(*args):raise AssertionError("native allocation was reached")
    monkeypatch.setattr(mesher,"_dispatch_builder",forbidden)
    with pytest.raises(ValueError):mesher.build_mesh_with_info(resolve_geometry(config()).geometry,replace(MeshDensity(),**patch),tmp_path/"bad.msh")
    assert not (tmp_path/"bad.msh").exists()


@pytest.mark.parametrize("name,value",[("include_inner",1),("include_curvature","false"),
    ("max_vertices",7),("max_vertices",1000001),("max_vertices",10**400),
    ("max_chord_error_mm",float("nan")),("max_chord_error_mm",5e-324),
    ("max_normal_step_deg",1e-308),("max_normal_step_deg",10**400),("lod","unknown")])
def test_preview_options_are_strict_and_bounded(name,value):
    with pytest.raises(ValueError):build_preview_geometry(config(),PreviewOptionsV1(**{name:value}))


@pytest.mark.parametrize("value",[None,0,1,"false"])
def test_direct_cad_flag_is_boolean(value,tmp_path):
    from hornlab_mesher.cad import write_step
    from hornlab_mesher.mesher import MesherError
    with pytest.raises(MesherError,match="must be a boolean"):
        write_step(resolve_geometry(config()).geometry,tmp_path/"bad.step",open_throat=value)
    assert not (tmp_path/"bad.step").exists()


def test_model_replacement_cannot_diverge_from_certified_identity():
    from dataclasses import replace
    m=resolve_geometry(config()).geometry.controls_meridian
    with pytest.raises(ValueError,match="certified construction"):replace(m,offset=5)
    with pytest.raises(ValueError,match="certified construction"):replace(m,_cubic_bytes=bytes(len(m._cubic_bytes)))


def test_direct_constructor_cannot_reseal_forged_geometry():
    from dataclasses import fields
    m=resolve_geometry(config()).geometry.controls_meridian
    args={f.name:getattr(m,f.name) for f in fields(m)}
    cube=m.cubic_poles.copy();cube[2,1]+=.2
    args["_cubic_bytes"]=cube.tobytes()
    values=[args[f.name] for f in fields(m) if f.name!="_authority_seal"]
    args["_authority_seal"]=m._digest(values)
    with pytest.raises(ValueError,match="certified construction"):type(m)(**args)


def test_unrequested_wall_does_not_allocate_its_vertex_grids(monkeypatch):
    from hornlab_mesher.adapter_controls import ControlsMeridian
    original=ControlsMeridian.branch
    def checked(self,name,u):
        assert np.asarray(u).size<=2,"unrequested wall grid was evaluated"
        return original(self,name,u)
    monkeypatch.setattr(ControlsMeridian,"branch",checked)
    preview=build_preview_geometry(config(),PreviewOptionsV1(include_inner=False,include_source_cap=False,max_vertices=8))
    assert preview.surfaces==[]


def test_collinear_controls_preserve_straight_segment_and_join_corner():
    E=20.;r0=17.;angle=12.3;slope=math.tan(math.radians(angle))
    points=[[-E,r0-E*slope],[-2*E/3,r0-2*E*slope/3],[-E/3,r0-E*slope/3]]
    m=resolve_geometry(config(points)).geometry.controls_meridian
    u=np.asarray([0,.07,.37,.81,1]);p,d,e=m.cubic(u)
    np.testing.assert_allclose(p[:,0],.25*E*u,rtol=0,atol=1e-13)
    np.testing.assert_allclose(p[:,1],.25*(r0-E*slope+E*slope*u),rtol=0,atol=1e-13)
    np.testing.assert_allclose(e,0,rtol=0,atol=1e-13)
    assert m.source_angle_deg==pytest.approx(angle)
    assert m.join_jump_deg==pytest.approx(6-angle)


def test_exact_extrema_can_exceed_mouth_radius():
    c=config([[-30,12],[-20,55],[-8,48]]);c["profile"].update(L_mm=40,a_deg=8,a0_deg=2)
    m=resolve_geometry(c).geometry.controls_meridian
    radius=m.cubic(np.linspace(0,1,10001))[0][:,1].max()
    assert m.reach>float(m.body(1)[0][1])
    assert m.reach>=radius and m.reach-radius<1e-6


def test_external_gmsh_model_and_options_survive_native_cad_success_and_failure(tmp_path,monkeypatch):
    gmsh=pytest.importorskip("gmsh")
    from hornlab_mesher.cad import write_step_from_config
    from hornlab_mesher.mesher import MesherError
    gmsh.initialize(interruptible=False)
    try:
        gmsh.model.add("caller-owned");point=gmsh.model.occ.addPoint(3,4,5);gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MinimumCirclePoints",23)
        write_step_from_config(config(),tmp_path/"ok.step")
        assert gmsh.model.getCurrent()=="caller-owned" and gmsh.model.getEntities(0)==[(0,point)]
        assert gmsh.option.getNumber("Mesh.MinimumCirclePoints")==23
        path=tmp_path/"sentinel.step";path.write_bytes(b"existing")
        def fail(*args):raise RuntimeError("injected transport failure")
        monkeypatch.setattr(gmsh,"write",fail)
        with pytest.raises(MesherError,match="injected transport"):write_step_from_config(config(),path)
        assert path.read_bytes()==b"existing"
        assert gmsh.model.getCurrent()=="caller-owned" and gmsh.model.getEntities(0)==[(0,point)]
        assert not list(tmp_path.glob(".sentinel.step.*"))
    finally:gmsh.finalize()
