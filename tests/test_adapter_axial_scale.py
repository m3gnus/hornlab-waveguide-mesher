"""Physical affine construction, strict intent and actual analytic exports."""
import copy
import json
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from hornlab_mesher.adapter_axial import PhysicalAdapter, FORMULA
from hornlab_mesher.config_builder import build_geometry_params, resolve_geometry, build_from_config
from hornlab_mesher.config_parser import ConfigError, load_config
from hornlab_mesher.preview import build_preview_geometry, PreviewOptionsV1
from hornlab_mesher.preview.dimensions import canonical_dimensions


def config():
    return {"formula":FORMULA,"scale":2.,"axial_scale":1.,"vertical_offset_mm":7.,
        "profile":{"L_mm":120.,"r0_mm":12.7,"a_deg":40.,"a0_deg":15.5,"k":1.},
        "throat_adapter":{"mode":"authored","contract_revision":1,
            "driver_exit_diameter_mm":25.4,"exit_half_angle_deg":10.,"length_mm":40.,
            "join_t":.137,"driver_handle_mm":10.,"body_handle_mm":15.},
        "mesh":{"throat_res_mm":5.,"mouth_res_mm":15.,"rear_res_mm":10.}}


def model(c=None):
    return resolve_geometry(c or config()).geometry.adapter_meridian


def casteljau(controls,u):
    values = np.array(controls,dtype=float)
    while len(values)>1:
        values=(1-u)*values[:-1]+u*values[1:]
    return values[0]


@pytest.mark.parametrize("xy,axis,offset",[(2,1,7),(1,1.2,-9),(.3,.5,0),(2,2,7),(.1,1,10000),(1,.1,-10000)])
def test_independent_affine_cubic_conic_and_actual_tangents(xy,axis,offset):
    c=config();c.update(scale=xy,axial_scale=axis,vertical_offset_mm=offset)
    m=model(c);p=c["profile"];a=c["throat_adapter"]
    L,r0,k=p["L_mm"],p["r0_mm"],p["k"]
    ta,t0=math.tan(math.radians(p["a_deg"])),math.tan(math.radians(p["a0_deg"]))
    zJ=L*a["join_t"]
    root=lambda z:math.sqrt((k*r0)**2+2*k*r0*z*t0+z*z*ta*ta)
    rJ=root(zJ)+r0*(1-k);slope=(k*r0*t0+zJ*ta*ta)/root(zJ)
    alpha=math.radians(a["exit_half_angle_deg"])
    controls=np.array([[0,a["driver_exit_diameter_mm"]/2],
        [a["driver_handle_mm"]*math.cos(alpha),a["driver_exit_diameter_mm"]/2+a["driver_handle_mm"]*math.sin(alpha)],
        [a["length_mm"]-a["body_handle_mm"]/math.hypot(1,slope),rJ-a["body_handle_mm"]*slope/math.hypot(1,slope)],
        [a["length_mm"],rJ]])
    for u in (.019,.217,.619,.983):
        np.testing.assert_allclose(m.cubic_values(u)[0],casteljau(controls,u)*[axis,xy],rtol=0,atol=2e-12)
    t=np.array([a["join_t"],.551,.777,1.])
    expected=np.array([[axis*(a["length_mm"]+L*v-zJ),xy*(root(L*v)+r0*(1-k))] for v in t])
    np.testing.assert_allclose(m.body_values(t)[0],expected,rtol=0,atol=2e-12)
    np.testing.assert_array_equal(m.cubic[-1],m.body_poles[0])
    outgoing=m.cubic[-1]-m.cubic[-2];incoming=m.body_poles[1]-m.body_poles[0]
    np.testing.assert_allclose(outgoing/np.linalg.norm(outgoing),incoming/np.linalg.norm(incoming),atol=1e-13,rtol=0)
    assert m.source_angle==pytest.approx(math.degrees(math.atan2(xy*math.sin(alpha),axis*math.cos(alpha))))
    assert m.body_poles[-1,0]==pytest.approx(axis*(a["length_mm"]+L*(1-a["join_t"])))


def test_identity_immutable_and_density_independent():
    c=config();a=model(c);c["mesh"].update(length_segments=97,angular_segments=96,mouth_res_mm=9.)
    b=model(c)
    assert a==b and hash(a)==hash(b) and a.fingerprint==b.fingerprint
    a.identity["adapter"]["join_t"]=.9
    a.payload["join_t"]=.8
    assert a==b
    with pytest.raises(FrozenInstanceError):a.offset=3
    for arr in (a.cubic,a.body_poles,a.body_weights):
        with pytest.raises(ValueError):arr.setflags(write=True)
    for field in ("scale","axial_scale","vertical_offset_mm"):
        changed=copy.deepcopy(c);changed[field]+=.01
        assert model(changed).fingerprint!=a.fingerprint


@pytest.mark.parametrize("angle",[-20.,0.,17.])
@pytest.mark.parametrize("join",[0.,.23])
def test_safe_authored_exit_angles_and_join_zero(angle,join):
    c=config();c["throat_adapter"].update(exit_half_angle_deg=angle,join_t=join,body_handle_mm=4.)
    m=model(c)
    assert m.source_angle==pytest.approx(math.degrees(math.atan(2*math.tan(math.radians(angle)))))


@pytest.mark.parametrize("lod",["coarse","fine","inspection"])
def test_certified_physical_preview_positions_normals_curvature_and_dimensions(lod):
    c=config();m=model(c)
    p=build_preview_geometry(c,PreviewOptionsV1(lod=lod))
    json.dumps(p.metadata,allow_nan=False)
    wall,cap=p.surfaces
    nphi=p.metadata["actual_segment_counts"]["horn_phi"]
    points=wall.positions.reshape(-1,nphi,3)
    ns=wall.normals.reshape(-1,nphi,3)
    assert set(s.role for s in p.surfaces)=={"horn.inner","source_cap"}
    np.testing.assert_array_equal(p.metadata["boundsMm"],m.bounds)
    assert canonical_dimensions(c)==m.dimensions==p.metadata["dimensions_mm"]
    np.testing.assert_allclose(cap.normals,[np.array([0,0,1.])]*len(cap.positions),atol=0,rtol=0)
    assert np.max(abs(cap.positions[:,2]))==0
    assert np.mean(cap.positions[:,1])==pytest.approx(7.,abs=1e-12)
    # An independent inverse-transpose normal at the driver is distinct from
    # uniform scaling; this also checks the role's inward radial sign.
    alpha=math.radians(10.)
    expected=np.array([-math.cos(alpha)/2,0,math.sin(alpha)])
    expected/=np.linalg.norm(expected)
    np.testing.assert_allclose(ns[0,0],expected,atol=2e-14,rtol=0)
    f=p.metadata["fidelity"]["horn.inner"]
    assert f["max_chord_error_mm"]<=.008 and f["max_normal_step_deg"]<=3
    # Finite differences at an actual retained-body preview station provide a
    # second derivative/curvature oracle independent of the model derivative.
    row=len(points)-4;z=points[row,0,2];A=c["throat_adapter"]["length_mm"]
    old_z=z-A+120*.137;h=.001
    radius=lambda zz:2*math.sqrt(12.7**2+2*12.7*zz*math.tan(math.radians(15.5))+zz**2*math.tan(math.radians(40))**2)
    r=radius(old_z);dr=(radius(old_z+h)-radius(old_z-h))/(2*h)
    ddr=(radius(old_z+h)-2*r+radius(old_z-h))/(h*h)
    km=-ddr/(1+dr*dr)**1.5;kp=1/(r*math.sqrt(1+dr*dr))
    assert wall.curvature_mean[row*nphi]==pytest.approx((km+kp)/2,abs=2e-8)
    assert wall.metadata["join_curvature"]=="retained-body-one-sided"
    normal_dots=np.sum(ns[:-1]*ns[1:],axis=-1)
    assert math.degrees(np.max(np.arccos(np.clip(normal_dots,-1,1))))<=f["max_normal_step_deg"]+1e-7


def test_cubic_interior_envelope_does_not_assume_mouth_largest():
    c=config();c["profile"].update(L_mm=20.,a_deg=6.,a0_deg=0.)
    c["throat_adapter"].update(driver_exit_diameter_mm=60.,exit_half_angle_deg=30.,
        length_mm=40.,driver_handle_mm=20.,body_handle_mm=5.,join_t=0.)
    m=model(c)
    assert m.reach>m.cubic[0,1] and m.reach>m.body_poles[-1,1]
    samples=m.cubic_values(np.linspace(0,1,10001))[0][:,1]
    assert 0<=m.reach-float(max(samples))<1e-5
    assert canonical_dimensions(c)["horn_overall"][0]==2*m.reach


def test_actual_triangle_interiors_satisfy_geometric_distance_bound():
    from scipy.optimize import minimize_scalar
    c=config();c.update(scale=1.3,axial_scale=.8,vertical_offset_mm=-11.)
    c["profile"].update(k=2.7,a_deg=51.,a0_deg=8.)
    c["throat_adapter"].update(join_t=.193,driver_handle_mm=7.,body_handle_mm=11.,exit_half_angle_deg=-12.)
    p=build_preview_geometry(c,PreviewOptionsV1(max_chord_error_mm=.02,max_normal_step_deg=5.))
    wall=p.surfaces[0];m=model(c)
    triangles=wall.positions[wall.indices.reshape(-1,3)]
    rng=np.random.default_rng(8821)
    selected=triangles[rng.choice(len(triangles),256,replace=False)]
    worst=0.
    for weights in ((1/3,1/3,1/3),(.11,.27,.62)):
        interiors=np.einsum('tvc,v->tc',selected,weights)
        for point in interiors:
            zr=np.array([point[2],math.hypot(point[0],point[1]-m.offset)])
            if point[2]<m.cubic[-1,0]:
                lo,hi=0.,1.
                curve=lambda u:casteljau(m.cubic,u)
            else:
                lo,hi=m.payload["join_t"],1.
                body=m.identity["body"];L,r0,k=body["L"],body["r0"],body["k"]
                ta,t0=math.tan(math.radians(body["a"])),math.tan(math.radians(body["a0"]))
                curve=lambda t:np.array([m.axial_scale*(m.payload["length_mm"]+L*(t-m.payload["join_t"])),
                    m.xy_scale*(math.sqrt((k*r0)**2+2*k*r0*L*t*t0+(L*t*ta)**2)+r0*(1-k))])
            result=minimize_scalar(lambda u:float(np.sum((curve(u)-zr)**2)),bounds=(lo,hi),method="bounded",options={"xatol":1e-14})
            worst=max(worst,math.sqrt(result.fun))
    assert worst<=p.metadata["fidelity"]["horn.inner"]["max_chord_error_mm"]


@pytest.mark.parametrize("kwargs",[{"max_vertices":100},{"max_chord_error_mm":1e-12},
    {"min_silhouette_segments":4097},{"max_vertices":True},{"max_normal_step_deg":0},
    {"min_silhouette_segments":3.5},{"max_chord_error_mm":True},{"lod":"unknown"},
    {"lod":[]},{"include_inner":1},{"include_curvature":"yes"},{"max_vertices":1000001},
    {"min_silhouette_segments":0}])
def test_preview_refuses_uncertified_or_malformed_requests(kwargs):
    with pytest.raises(ValueError):build_preview_geometry(config(),PreviewOptionsV1(**kwargs))


@pytest.mark.parametrize("kwargs",[
    {"max_normal_step_deg":1e-308},
    {"max_normal_step_deg":1e-320},
    {"max_normal_step_deg":5e-324},
    {"max_normal_step_deg":10**400},
    {"max_chord_error_mm":10**400},
],ids=["tiny-normal","subnormal-normal","minimum-normal","oversized-normal","oversized-chord"])
def test_extreme_numeric_preview_requests_refuse_before_sampling(monkeypatch,kwargs):
    import hornlab_mesher.preview.adapter_axial as preview
    calls=[]
    original_ceil=preview.math.ceil
    original_arange=preview.np.arange
    def ceil(value):
        calls.append("ceil")
        return original_ceil(value)
    def arange(*args,**options):
        calls.append("allocate")
        return original_arange(*args,**options)
    monkeypatch.setattr(preview.math,"ceil",ceil)
    monkeypatch.setattr(preview.np,"arange",arange)
    with pytest.raises(ValueError):
        build_preview_geometry(config(),PreviewOptionsV1(**kwargs))
    assert calls==[]


def test_empty_preview_selection_and_curvature_disabled():
    p=build_preview_geometry(config(),PreviewOptionsV1(include_inner=False,include_source_cap=False))
    assert not p.surfaces
    p=build_preview_geometry(config(),PreviewOptionsV1(include_curvature=False))
    assert all(s.curvature_mean is None for s in p.surfaces)


@pytest.mark.parametrize("field",["scale","axial_scale","vertical_offset_mm"])
@pytest.mark.parametrize("value",[True,"1",None,[],float("nan"),float("inf")])
def test_strict_transform_numbers(field,value):
    c=config();c[field]=value
    with pytest.raises(ConfigError):resolve_geometry(c)


@pytest.mark.parametrize("xy,axis",[(.009,1),(10.01,1),(1,.009),(1,10.01),(.1,1.01),(1,.099)])
def test_transform_domain(xy,axis):
    c=config();c.update(scale=xy,axial_scale=axis)
    with pytest.raises(ConfigError):resolve_geometry(c)


@pytest.mark.parametrize("section,key,value",[("root","mode","freestanding"),("mesh","wall_thickness_mm",1),
    ("profile","s",.01),("source","source_shape",1),("source","source_curv",1),
    ("source","source_radius_mm",10),("mesh","topology_mode","legacy"),("mesh","surface_fit","approximate"),
    ("root","parameters",{}),("root","Source",{}),("root","mouth_roundover_radius_mm",0),
    ("root","gcurve",{}),("root","morph",{}),("profile","L",120),("profile","n",4),
    ("profile","throat_adapter",{}),("mesh","axial_scale",1),("mesh","throat_res_mm",True),
    ("mesh","angular_segments",True),("mesh","length_segments",2.5),("mesh","allow_large_mesh",1),
    ("mesh","scale_to_metres",1),("root","quadrants","1234junk"),("mesh","quadrants",12),
    ("root","output",{"unknown":3}),("mesh","surface_fit",[]),("mesh","surface_fit",{}),
    ("mesh","topology_mode",[]),("root","output_path",[]),("profile",None,3.)])
def test_original_supplied_intent_refused(section,key,value):
    c=config();(c if section=="root" else c.setdefault(section,{}))[key]=value
    with pytest.raises(ConfigError):resolve_geometry(c)


@pytest.mark.parametrize("payload",[None,{}, {"mode":"off"}])
def test_missing_or_inactive_adapter_refused(payload):
    c=config();c["throat_adapter"]=payload
    with pytest.raises(ConfigError):resolve_geometry(c)


def test_unknown_nonstring_adapter_key_is_config_refusal():
    c=config();c["throat_adapter"][None]=3.
    with pytest.raises(ConfigError,match="unsupported adapter"):resolve_geometry(c)


def test_missing_axis_and_mixed_formula_aliases_refused():
    c=config();del c["axial_scale"]
    with pytest.raises(ConfigError):resolve_geometry(c)
    c=config();c["profile"]["formula"]="OSSE-ADAPTER"
    with pytest.raises(ConfigError):resolve_geometry(c)
    c=config();c["formula"]="OSSE-ADAPTER"
    with pytest.raises(ConfigError):resolve_geometry(c)


def test_profile_formula_alias_retains_required_transform():
    c=config();del c["formula"];c["profile"]["type"]=FORMULA.lower()
    assert model(c)==model()


def test_body_only_helpers_and_unqualified_bundle_refuse(tmp_path):
    from hornlab_mesher.profile_formulas import calculate_osse,calculate_osse_curve,osse_total_length
    from hornlab_mesher.profile_sampling import build_point_grid_arrays
    from hornlab_mesher.viewport import build_viewport_geometry_from_config
    from hornlab_mesher.cad import write_wglink
    from hornlab_mesher.mesher import MesherError
    params=build_geometry_params(config())[0]
    for fn,args in ((calculate_osse,(1,0,params)),(calculate_osse_curve,([1],0,params)),
        (osse_total_length,(params,)),(build_point_grid_arrays,(params,)),(build_viewport_geometry_from_config,(config(),))):
        with pytest.raises(ValueError,match="adapter axial scale"):fn(*args)
    target=tmp_path/"refused.wglink"
    with pytest.raises(MesherError,match="analytic recipe"):write_wglink(resolve_geometry(config()).geometry,target)
    assert not target.exists()


def test_step_rewrites_cannot_hide_invalid_original_coverage(tmp_path,monkeypatch):
    import hornlab_mesher.cad as cad
    c=config();c["quadrants"]=1
    monkeypatch.setattr(cad,"write_step",lambda *a,**k:pytest.fail("terminal must not run"))
    with pytest.raises(ConfigError):cad.write_step_from_config(c,tmp_path/"refused.step")
    assert not (tmp_path/"refused.step").exists()


def test_nonboolean_api_budget_override_refused():
    with pytest.raises(ConfigError,match="boolean"):resolve_geometry(config(),allow_large_mesh=1)


def test_native_composition_never_retries_with_approximate_shape(monkeypatch,tmp_path):
    import hornlab_mesher.config_builder as cb
    from hornlab_mesher.mesher import MesherError
    calls=[]
    def refuse(*args,**kwargs):
        calls.append(args[0]);raise MesherError("deliberate exact construction refusal")
    monkeypatch.setattr(cb,"build_mesh_with_info",refuse)
    with pytest.raises(MesherError,match="deliberate"):cb.build_from_config(config(),tmp_path/"refused.msh")
    assert len(calls)==1


@pytest.mark.parametrize("mutation",[{"driver_handle_mm":39.},{"body_handle_mm":100.},
    {"length_mm":.00001,"driver_handle_mm":.000002,"body_handle_mm":.000002,"join_t":0.}])
def test_control_crossing_and_unresolved_physical_spans_refuse(mutation):
    c=config();c["throat_adapter"].update(mutation)
    with pytest.raises(ConfigError):resolve_geometry(c)


@pytest.mark.parametrize("metres",[False,True])
def test_actual_mesh_source_shared_edges_and_transformed_extent(tmp_path,metres):
    pytest.importorskip("gmsh");meshio=pytest.importorskip("meshio")
    c=config();c["mesh"]["scale_to_metres"]=metres
    result=build_from_config(c,tmp_path/"physical.msh")
    assert result.physical_groups=={1:"SD1G0",2:"SD1D1001"}
    assert result.metadata["construction_fingerprint"]==model(c).fingerprint
    mesh=meshio.read(str(result.mesh_path));points=mesh.points*(1000 if metres else 1)
    triangles=mesh.cells_dict["triangle"];tags=mesh.cell_data_dict["gmsh:physical"]["triangle"]
    source=triangles[tags==2]
    normals=np.cross(points[source[:,1]]-points[source[:,0]],points[source[:,2]]-points[source[:,0]])
    assert np.all(normals[:,2]>0)
    assert np.max(abs(points[np.unique(source),2]))<1e-9
    edges=np.sort(np.concatenate((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    assert counts.max()==2
    free=unique[counts==1]
    assert len(free)>0
    np.testing.assert_allclose(points[free,2],model(c).body_poles[-1,0],atol=1e-8,rtol=0)
    assert np.max(points[:,2])==pytest.approx(model(c).body_poles[-1,0])
    assert np.max(abs(points[np.unique(source),0]))==pytest.approx(25.4,rel=1e-7)


@pytest.mark.parametrize("opened",[True,False])
def test_actual_reopened_step_exact_recipe_source_and_datums(tmp_path,opened):
    gmsh=pytest.importorskip("gmsh")
    from hornlab_mesher.cad import write_step, _realized_bundle_geometry
    from hornlab_mesher.datums import derive_datums
    c=config();c["vertical_offset_mm"]=-7.
    resolved=resolve_geometry(c);m=resolved.geometry.adapter_meridian
    path,info=write_step(resolved.geometry,tmp_path/"physical.step",open_throat=opened)
    assert info.body=="surface" and info.volume_mm3 is None
    assert info.construction["axial_scale"]==1 and info.construction["scale"]==2
    np.testing.assert_array_equal(info.bounding_box_mm,m.bounds)
    text=path.read_text();recipe=json.loads(text.split("/* native-construction ")[1].split(" */")[0])
    assert recipe["formula"]==FORMULA and recipe["construction_fingerprint"]==m.fingerprint
    datums=derive_datums(resolved.geometry,_realized_bundle_geometry(resolved.geometry))
    assert datums["WG_THROAT_PLANE"]["origin_mm"]==[0,-7.,0]
    assert datums["WG_ADAPTER_JOIN_PLANE"]["origin_mm"]==[0,-7.,40.]
    assert datums["WG_MOUTH_PLANE"]["origin_mm"][2]==pytest.approx(143.56)
    initialized=not gmsh.isInitialized()
    if initialized:gmsh.initialize()
    try:
        gmsh.model.add("physical-reopen")
        gmsh.model.occ.importShapes(str(path));gmsh.model.occ.synchronize()
        assert not gmsh.model.getEntities(3)
        box=gmsh.model.getBoundingBox(-1,-1)
        np.testing.assert_allclose(np.array(box).reshape(2,3),m.bounds,atol=1e-5,rtol=0)
        if not opened:
            plane=[tag for dim,tag in gmsh.model.getEntities(2) if gmsh.model.getType(dim,tag)=="Plane"]
            assert len(plane)==1
            assert gmsh.model.occ.getMass(2,plane[0])==pytest.approx(math.pi*25.4**2,rel=1e-8)
            assert gmsh.model.occ.getCenterOfMass(2,plane[0])==pytest.approx((0.,-7.,0.),abs=1e-8)
    finally:
        if initialized:gmsh.finalize()
        else:gmsh.model.remove()


def test_json_toml_native_roundtrip_preserves_authority(tmp_path):
    c=config();json_path=tmp_path/"model.json";json_path.write_text(json.dumps(c))
    toml_path=tmp_path/"model.toml"
    lines=[]
    for key,value in c.items():
        if not isinstance(value,dict):lines.append(f"{key} = {json.dumps(value)}")
    for key,values in c.items():
        if isinstance(values,dict):
            lines.append(f"[{key}]")
            lines.extend(f"{name} = {json.dumps(value)}" for name,value in values.items())
    toml_path.write_text("\n".join(lines))
    assert model(load_config(json_path))==model(load_config(toml_path))==model(c)
