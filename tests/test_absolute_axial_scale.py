"""Public-boundary and continuous physical-model checks."""
import copy
import math

import numpy as np
import pytest

from hornlab_mesher.axial_scale import AxialModel, FIT_TOL_MM
from hornlab_mesher.config_builder import build_geometry_params, resolve_geometry
from hornlab_mesher.config_parser import ConfigError
from hornlab_mesher.preview import build_preview_geometry, PreviewOptionsV1
from hornlab_mesher.preview.dimensions import canonical_dimensions
from hornlab_mesher.datums import derive_datums
from hornlab_mesher.profile_formulas import calculate_osse, calculate_osse_curve, osse_total_length
from hornlab_mesher.profile_sampling import build_point_grid_arrays


def config():
    return {"formula":"OSSE-AXIAL","axial_scale":1.2,"scale":2,
        "profile":{"L_mm":120,"r0_mm":18,"a_deg":40,"a0_deg":8,"k":1,"s":0},
        "mesh":{"wall_thickness_mm":0,"vertical_offset_mm":7},"source":{"source_shape":0}}


def test_absolute_override_literal_formula_and_datums():
    c = config()
    resolved = resolve_geometry(c)
    model = resolved.geometry.axial_model
    z = np.linspace(0,144,501)
    authored = z/1.2
    expected = 2*np.sqrt(18**2+2*18*math.tan(math.radians(8))*authored+math.tan(math.radians(40))**2*authored**2)
    np.testing.assert_allclose(model.body(z)[0][:,1],expected,atol=1e-12)
    assert model.length == 144
    assert model.r0 == 36
    assert resolved.geometry.source_auto_angle_deg == pytest.approx(math.degrees(math.atan(2/1.2*math.tan(math.radians(8)))))
    d = derive_datums(resolved.geometry,None)
    assert d["WG_THROAT_PLANE"]["origin_mm"] == [0,7,0]
    assert d["WG_MOUTH_PLANE"]["origin_mm"] == [0,7,144]
    assert model.bounds[0][1]+model.bounds[1][1] == pytest.approx(14)
    assert canonical_dimensions(c)["horn_overall"][2] == 144


def test_density_independent_identity_and_rigid_placement():
    a = config()
    b = copy.deepcopy(a)
    b["mesh"].update(throat_res_mm=.5,mouth_res_mm=1.5,angular_segments=97,length_segments=71,scale_to_metres=False)
    m1,m2 = resolve_geometry(a).geometry.axial_model,resolve_geometry(b).geometry.axial_model
    assert m1 == m2 and m1.fingerprint == m2.fingerprint
    b["mesh"]["vertical_offset_mm"] = 8
    assert resolve_geometry(b).geometry.axial_model.fingerprint != m1.fingerprint


@pytest.mark.parametrize("value",[None,False,0,-1,"1",float("nan"),float("inf"),1e300])
def test_axis_refuses_invalid_values(value):
    c=config();c["axial_scale"]=value
    with pytest.raises(ConfigError):resolve_geometry(c)


@pytest.mark.parametrize("value",[None,False,0,1,2])
def test_ordinary_formula_refuses_every_supplied_axis(value):
    c=config();c["formula"]="OSSE";c["axial_scale"]=value
    with pytest.raises(ConfigError):resolve_geometry(c)


@pytest.mark.parametrize("section,key,value",[
    ("profile","formula","R-OSSE"),("profile","type","FREEFORM"),("root","type","ICW"),
    ("root","quadrants",1),("mesh","quadrants","1234garbage"),("root","vertical_offset_mm",8),
    ("profile","scale",2),("profile","a_deg","40"),("profile","s",.1),("profile","throatExtLength",5),
    ("root","cross_section",{}),("root","enclosure",{}),("mesh","wall_thickness_mm",1),
    ("source","source_shape",1),("source","source_radius_mm",50),("source","source_curv",1),
    ("mesh","topology_mode","legacy"),("mesh","surface_fit","approximate"),
    ("mesh","angular_segments",0),("mesh","length_segments",1.5),("mesh","max_triangles",False),
    ("mesh","allow_large_mesh",1),("mesh","scale_to_metres","false"),("mesh","mouth_res_mm",0),
])
def test_supplied_intent_and_unsupported_compositions_refuse(section,key,value):
    c=config();(c if section=="root" else c[section])[key]=value
    with pytest.raises(ConfigError):resolve_geometry(c)


@pytest.mark.parametrize("section,key,value",[("profile","L_mm",.1),("profile","r0_mm",.1),("profile","k",.1),
    ("profile","a_deg",76),("profile","a0_deg",40),("root","axial_scale",.01),("root","scale",11),
    ("mesh","vertical_offset_mm",10001)])
def test_numerical_domain_refusals(section,key,value):
    c=config();(c if section=="root" else c[section])[key]=value
    with pytest.raises(ConfigError):resolve_geometry(c)


@pytest.mark.parametrize("value",[0,1,"true",[],{}])
def test_override_is_exact_boolean_or_none(value):
    with pytest.raises(ConfigError):resolve_geometry(config(),allow_large_mesh=value)


@pytest.mark.parametrize("axis,xy,a,a0,k",[(.1,1,75,0,.5),(1,.1,5,0,10),(.5,1,35,8,1),(2,1,45,10,2),(1.2,2,40,8,1)])
def test_whole_interval_hermite_bound_dense_independent_interpolant(axis,xy,a,a0,k):
    c=config();c.update(axial_scale=axis,scale=xy);c["profile"].update(a_deg=a,a0_deg=a0,k=k)
    m=resolve_geometry(c).geometry.axial_model
    knots=m.stations(FIT_TOL_MM,fit=True)
    for lo,hi in zip(knots[:-1],knots[1:]):
        u=np.linspace(0,1,33);h=hi-lo
        p,d=m.body(np.array([lo,hi]))
        fitted=(2*u**3-3*u*u+1)[:,None]*p[0]+(u**3-2*u*u+u)[:,None]*h*d[0]+(-2*u**3+3*u*u)[:,None]*p[1]+(u**3-u*u)[:,None]*h*d[1]
        err=np.linalg.norm(fitted-m.body(lo+u*h)[0],axis=1).max()
        certificate=m.fourth_bound(lo,hi)*h**4/384
        assert err <= certificate+1e-11
        assert certificate <= FIT_TOL_MM


@pytest.mark.parametrize("lod",["coarse","fine"])
def test_preview_complete_chord_normal_and_transformed_positions(lod):
    c=config();p=build_preview_geometry(c,PreviewOptionsV1(lod=lod,max_chord_error_mm=.008,max_normal_step_deg=3))
    assert {s.role for s in p.surfaces} == {"horn.inner","source_cap"}
    model=resolve_geometry(c).geometry.axial_model
    for s in p.surfaces:
        assert p.metadata["fidelity"][s.role]["max_chord_error_mm_achieved"] <= .008
        xyz=s.positions;tri=xyz[s.indices.reshape(-1,3)]
        cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        ns=s.normals[s.indices.reshape(-1,3)].mean(axis=1)
        assert np.all(np.sum(cross*ns,axis=1)>0)
        # Check actual shipped vertex normals along every triangle edge.
        tri_ns=s.normals[s.indices.reshape(-1,3)]
        for i,j in ((0,1),(1,2),(2,0)):
            angles=np.degrees(np.arccos(np.clip(np.sum(tri_ns[:,i]*tri_ns[:,j],axis=1),-1,1)))
            assert angles.max() <= 3.01
        if s.role=="horn.inner":
            r=np.hypot(xyz[:,0],xyz[:,1]-7)
            expected=model.body(xyz[:,2])[0][:,1]
            assert np.max(abs(r-expected)) < 2e-5
        else:assert np.all(xyz[:,2]==0)


def test_preview_preflight_budget_and_precision_refusal():
    with pytest.raises(ValueError,match="vertex budget"):
        build_preview_geometry(config(),PreviewOptionsV1(max_vertices=8,max_chord_error_mm=.008))
    with pytest.raises(ValueError,match="binary32"):
        build_preview_geometry(config(),PreviewOptionsV1(max_chord_error_mm=1e-10))


def test_body_only_consumers_refuse():
    p,_,_=build_geometry_params(config())
    for fn,args in ((calculate_osse,(0,0,p)),(calculate_osse_curve,([0,1],0,p)),(osse_total_length,(p,)),(build_point_grid_arrays,(p,))):
        with pytest.raises(ValueError,match="absolute axial scale"):fn(*args)


def test_missing_axis_and_default_source_and_wall():
    c=config();del c["axial_scale"]
    with pytest.raises(ConfigError):resolve_geometry(c)
    c=config();del c["source"];del c["mesh"]["wall_thickness_mm"]
    assert resolve_geometry(c).mode=="bare"


@pytest.mark.parametrize("section",["profile","mesh","source"])
def test_misplaced_axis_refuses_even_for_ordinary_formula(section):
    c=config();c["formula"]="OSSE";del c["axial_scale"];c[section]["axial_scale"]=1
    with pytest.raises(ConfigError,match="root"):resolve_geometry(c)


def test_no_approximate_retry_for_native_build(monkeypatch,tmp_path):
    import hornlab_mesher.config_builder as cb
    from hornlab_mesher.mesher import MesherError
    calls=[]
    def refuse(*args,**kwargs):
        calls.append(args[0]);raise MesherError("deliberate construction refusal")
    monkeypatch.setattr(cb,"build_mesh_with_info",refuse)
    with pytest.raises(MesherError,match="deliberate"):cb.build_from_config(config(),tmp_path/"refused.msh")
    assert len(calls)==1


def test_bundle_refuses_before_creating_destination(tmp_path):
    from hornlab_mesher.cad import write_wglink
    from hornlab_mesher.mesher import MesherError
    target=tmp_path/"refused.wglink"
    with pytest.raises(MesherError,match="analytic recipe"):
        write_wglink(resolve_geometry(config()).geometry,target)
    assert not target.exists()


@pytest.mark.parametrize("open_throat",[True,False])
def test_real_step_surface_recipe_bounds_and_source_rim(tmp_path,open_throat):
    gmsh=pytest.importorskip("gmsh")
    from hornlab_mesher.cad import write_step
    c=config();resolved=resolve_geometry(c);model=resolved.geometry.axial_model
    path,info=write_step(resolved.geometry,tmp_path/"axial.step",open_throat=open_throat)
    text=path.read_text(encoding="utf-8")
    assert "hornlab-absolute-axial-scale" in text and model.fingerprint in text
    assert info.body=="surface" and info.volume_mm3 is None
    np.testing.assert_array_equal(info.bounding_box_mm,model.bounds)
    assert info.absolute_axial_scale["sourceAngleDeg"]==pytest.approx(resolved.geometry.source_auto_angle_deg)
    initialized_here=not gmsh.isInitialized()
    if initialized_here:gmsh.initialize()
    try:
        gmsh.model.add("reopen-axial")
        gmsh.model.occ.importShapes(str(path));gmsh.model.occ.synchronize()
        assert not gmsh.model.getEntities(3)
        faces=gmsh.model.getEntities(2)
        assert len(faces)==(1 if open_throat else 2)
        area=[gmsh.model.occ.getMass(d,t) for d,t in faces]
        if not open_throat:
            assert min(area)==pytest.approx(math.pi*model.r0**2,rel=1e-7)
        box=gmsh.model.getBoundingBox(-1,-1)
        assert box[2]==pytest.approx(0,abs=1e-5) and box[5]==pytest.approx(144,abs=1e-5)
    finally:
        if initialized_here:gmsh.finalize()
        else:gmsh.model.remove()


@pytest.mark.parametrize("section,value",[("root",1),("mesh","1234garbage"),("mesh",123)])
def test_config_step_refuses_invalid_coverage_before_full_model_rewrite(monkeypatch,tmp_path,section,value):
    import hornlab_mesher.cad as cad
    c=config();(c if section=="root" else c[section])["quadrants"]=value
    def unexpected(*args,**kwargs):raise AssertionError("terminal writer must not run")
    monkeypatch.setattr(cad,"write_step",unexpected)
    with pytest.raises(ConfigError,match="quadrants"):
        cad.write_step_from_config(c,tmp_path/"refused.step")


@pytest.mark.parametrize("lod,expected",[("coarse",.15),("fine",.05),("inspection",.025)])
def test_preview_default_physical_target_and_empty_inclusion(lod,expected):
    p=build_preview_geometry(config(),PreviewOptionsV1(lod=lod))
    assert all(f["max_chord_error_mm_requested"]==expected for f in p.metadata["fidelity"].values())
    p=build_preview_geometry(config(),PreviewOptionsV1(lod=lod,include_inner=False,include_source_cap=False))
    assert not p.surfaces


@pytest.mark.parametrize("section",["profile","mesh","source"])
def test_active_malformed_section_refuses(section):
    c=config();c[section]=[]
    with pytest.raises(ConfigError,match="object"):resolve_geometry(c)


@pytest.mark.parametrize("marker",["OSSE-ADAPTER","OSSE-ROUNDOVER"])
def test_other_native_markers_preflight_before_cad_rewrites(monkeypatch,tmp_path,marker):
    import hornlab_mesher.config_builder as cb
    import hornlab_mesher.cad as cad
    c={"formula":marker,"quadrants":1234,"vertical_offset_mm":7,
        "mesh":{"quadrants":"1234","vertical_offset_mm":7}}
    seen=[]
    def validate(supplied):
        seen.append(copy.deepcopy(supplied));raise ConfigError("original native controls refused")
    monkeypatch.setattr(cb,"build_geometry_params",validate)
    with pytest.raises(ConfigError,match="original native"):
        cad.write_step_from_config(c,tmp_path/"refused.step")
    assert seen==[c]


@pytest.mark.parametrize("section",["root","mesh"])
def test_nonstring_unknown_keys_raise_config_error(section):
    c=config();(c if section=="root" else c[section])[None]=1
    with pytest.raises(ConfigError,match="unsupported"):resolve_geometry(c)


@pytest.mark.parametrize("name",["formula","type"])
def test_profile_marker_alias_is_normalized_without_losing_transform(name):
    c=config();del c["formula"];c["profile"][name]="osse-axial"
    assert resolve_geometry(c).geometry.axial_model==resolve_geometry(config()).geometry.axial_model


def test_nearly_equal_angles_keep_convexity_bound_nonnegative():
    c=config();c.update(scale=1,axial_scale=1)
    c["profile"].update(r0_mm=113.72002711670157,k=9.117836156021417,
        a_deg=62.74394217570826,a0_deg=62.743942175708256)
    model=resolve_geometry(c).geometry.axial_model
    assert model.second_bound(0,model.length)>=0


def test_long_small_throat_body_fit_stays_certified():
    c=config();c.update(scale=1,axial_scale=1)
    c["profile"].update(L_mm=2000,r0_mm=1,a_deg=75,a0_deg=0,k=.5)
    model=resolve_geometry(c).geometry.axial_model
    stations=model.stations(FIT_TOL_MM,fit=True)
    assert stations[0]==0 and stations[-1]==2000 and len(stations)<8192
    for lo,hi in zip(stations[:-1],stations[1:]):
        assert model.fourth_bound(lo,hi)*(hi-lo)**4/384<=FIT_TOL_MM


def test_disabled_preview_surfaces_do_not_allocate_full_surface_grids(monkeypatch):
    import hornlab_mesher.preview.axial_scale as preview
    stack=preview.np.stack
    surface_allocations=[]
    def observe(*args,**kwargs):
        result=stack(*args,**kwargs)
        if result.ndim==3:surface_allocations.append(result.shape)
        return result
    monkeypatch.setattr(preview.np,"stack",observe)
    result=build_preview_geometry(config(),PreviewOptionsV1(include_inner=False,include_source_cap=False,
        max_vertices=8,max_chord_error_mm=.008,max_normal_step_deg=3))
    assert not result.surfaces
    assert not surface_allocations


def test_signed_curvature_matches_independent_fundamental_forms():
    c=config()
    preview=build_preview_geometry(c,PreviewOptionsV1(lod="fine",include_curvature=True))
    body=next(s for s in preview.surfaces if s.role=="horn.inner")
    assert np.any(body.curvature_principal<0)
    xy,axis=c["scale"],c["axial_scale"]
    authored=c["profile"];r0=authored["r0_mm"];k=authored["k"]
    ta=math.tan(math.radians(authored["a_deg"]));t0=math.tan(math.radians(authored["a0_deg"]))
    def surface(z,az):
        u=z/axis
        r=xy*(math.sqrt((k*r0)**2+2*k*r0*t0*u+ta*ta*u*u)+r0*(1-k))
        return np.array([r*math.cos(az),r*math.sin(az)+7,z])
    for index in np.linspace(0,len(body.positions)-1,37,dtype=int):
        xyz=body.positions[index];z=xyz[2];az=math.atan2(xyz[1]-7,xyz[0])
        h,g=1e-3,1e-3
        centre=surface(z,az)
        dz=(surface(z+h,az)-surface(z-h,az))/(2*h)
        dp=(surface(z,az+g)-surface(z,az-g))/(2*g)
        dzz=(surface(z+h,az)-2*centre+surface(z-h,az))/(h*h)
        dpp=(surface(z,az+g)-2*centre+surface(z,az-g))/(g*g)
        dzp=(surface(z+h,az+g)-surface(z+h,az-g)-surface(z-h,az+g)+surface(z-h,az-g))/(4*h*g)
        normal=np.cross(dz,dp);normal/=np.linalg.norm(normal)
        E,F,G=np.dot(dz,dz),np.dot(dz,dp),np.dot(dp,dp)
        L,M,N=np.dot(dzz,normal),np.dot(dzp,normal),np.dot(dpp,normal)
        H=(E*N-2*F*M+G*L)/(2*(E*G-F*F))
        K=(L*N-M*M)/(E*G-F*F)
        root=math.sqrt(max(0,H*H-K));positive,negative=H+root,H-root
        largest=positive if abs(positive)>=abs(negative) else negative
        assert body.curvature_mean[index]==pytest.approx(H,rel=2e-4,abs=1e-6)
        assert body.curvature_principal[index]==pytest.approx(largest,rel=2e-4,abs=1e-6)
    cap=next(s for s in preview.surfaces if s.role=="source_cap")
    assert np.all(cap.curvature_mean==0) and np.all(cap.curvature_principal==0)


def test_equal_opposite_principal_curvatures_choose_positive_tie(monkeypatch):
    import hornlab_mesher.preview.axial_scale as preview
    from hornlab_mesher.axial_scale import AxialModel
    # At this exact model's throat r=2, r'=0 and r''=1/2: the two
    # signed principal curvatures are exactly -1/2 and +1/2.
    model=AxialModel(10,2,1,0,1,1,1,0)
    monkeypatch.setattr(AxialModel,"from_params",classmethod(lambda cls,params:model))
    result=preview.build({},PreviewOptionsV1(include_curvature=True,include_source_cap=False))
    body=result.surfaces[0]
    throat=body.positions[:,2]==0
    assert np.any(throat)
    assert np.all(body.curvature_mean[throat]==0)
    assert np.all(body.curvature_principal[throat]==.5)


def test_native_first_dispatch_owns_input_before_caller_mutation(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import hornlab_mesher.axial_scale as axial
    supplied=config()
    options=PreviewOptionsV1(lod="fine",include_curvature=True)
    expected=build_preview_geometry(supplied,options)
    entered,resume=threading.Event(),threading.Event()
    real=axial.configuration
    seen=[]
    def delayed(snapshot,resolver):
        seen.append(snapshot)
        if len(seen)==1:
            entered.set()
            assert resume.wait(10)
        return real(snapshot,resolver)
    monkeypatch.setattr(axial,"configuration",delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending=pool.submit(build_preview_geometry,supplied,options)
        try:
            assert entered.wait(10)
            supplied["profile"]["L_mm"]=240
            supplied["mesh"]["vertical_offset_mm"]=19
        finally:
            resume.set()
        result=pending.result(timeout=10)
    assert seen[0] is not supplied
    assert seen[0]["profile"] is not supplied["profile"]
    assert seen[0]["mesh"] is not supplied["mesh"]
    assert result.metadata==expected.metadata
    assert [s.role for s in result.surfaces]==[s.role for s in expected.surfaces]
    for actual,baseline in zip(result.surfaces,expected.surfaces):
        for field in ("positions","indices","normals","curvature_mean","curvature_principal"):
            np.testing.assert_array_equal(getattr(actual,field),getattr(baseline,field))


@pytest.mark.parametrize("section,canonical",[("parameters","profile"),("Source","source")])
@pytest.mark.parametrize("masked",[False,True])
@pytest.mark.parametrize("entrypoint",[build_geometry_params,resolve_geometry,canonical_dimensions])
def test_ordinary_misplaced_axis_aliases_refuse_before_construction(section,canonical,masked,entrypoint):
    c=config();c["formula"]="OSSE";del c["axial_scale"]
    c[section]=copy.deepcopy(c[canonical])
    if not masked:del c[canonical]
    c[section]["axial_scale"]=2
    with pytest.raises(ConfigError,match="root"):
        entrypoint(c)


@pytest.mark.parametrize("section,canonical",[("parameters","profile"),("Source","source")])
def test_ordinary_aliases_without_axis_retain_geometry(section,canonical):
    baseline=config();baseline.update(formula="OSSE",mode="bare");del baseline["axial_scale"]
    aliased=copy.deepcopy(baseline);aliased[section]=aliased.pop(canonical)
    assert build_geometry_params(aliased)==build_geometry_params(baseline)
    np.testing.assert_array_equal(resolve_geometry(aliased).geometry.inner_points,
        resolve_geometry(baseline).geometry.inner_points)
    assert canonical_dimensions(aliased)==canonical_dimensions(baseline)


@pytest.mark.parametrize("marker",["OSSE-ADAPTER","OSSE-ROUNDOVER"])
def test_other_native_markers_preflight_before_dimension_rewrites(monkeypatch,marker):
    import hornlab_mesher.preview.dimensions as dimensions
    c={"formula":marker,"quadrants":1234,"vertical_offset_mm":10001,
        "mesh":{"quadrants":"1234","vertical_offset_mm":10001}}
    original=copy.deepcopy(c)
    seen=[]
    def validate(supplied):
        seen.append(copy.deepcopy(supplied))
        raise ConfigError("original native placement refused")
    def unexpected(*args,**kwargs):
        raise AssertionError("rewritten geometry must not resolve")
    monkeypatch.setattr(dimensions,"build_geometry_params",validate)
    monkeypatch.setattr(dimensions,"resolve_geometry",unexpected)
    with pytest.raises(ConfigError,match="original native placement"):
        dimensions.canonical_dimensions(c)
    assert seen==[original] and c==original


@pytest.mark.parametrize("controls",[
    {"output":{"path":"configured.msh"}},
    {"output":{"output_path":"configured.msh"}},
    {"path":"configured.msh"},
    {"output_path":"configured.msh"},
    {"output":{"path":None},"path":"configured.msh"},
    {"output":{"path":"configured.msh","output_path":"ignored.msh"},
        "path":"ignored-root.msh","output_path":"ignored-alias.msh"},
])
@pytest.mark.parametrize("route",["mesh","override","step"])
def test_cli_output_controls_preserve_native_model(monkeypatch,tmp_path,controls,route):
    import json
    from types import SimpleNamespace
    import hornlab_mesher.cli as cli
    import hornlab_mesher.config_builder as builder
    import hornlab_mesher.cad as cad
    c=config();c.update(controls)
    authored=tmp_path/"axial.json"
    authored.write_text(json.dumps(c),encoding="utf-8")
    expected=resolve_geometry(config()).geometry.axial_model
    seen=[]
    def mesh_writer(geometry,density,destination,**kwargs):
        assert geometry.axial_model==expected
        assert geometry.axial_model.fingerprint==expected.fingerprint
        seen.append(str(destination))
        raise ConfigError("validated native mesh reached")
    def step_writer(geometry,destination,**kwargs):
        assert geometry.axial_model==expected
        assert geometry.axial_model.fingerprint==expected.fingerprint
        seen.append(str(destination))
        return destination,SimpleNamespace(body="surface",n_faces=1,volume_mm3=None,
            units="mm",bounding_box_mm=expected.bounds,throat_opened=True)
    monkeypatch.setattr(builder,"build_mesh_with_info",mesh_writer)
    monkeypatch.setattr(cad,"write_step",step_writer)
    args=[str(authored)]
    if route=="override":args.extend(["-o","override.msh"])
    if route=="step":args.extend(["--step",str(tmp_path/"axial.step")])
    if route=="step":
        assert cli.main(args)==0
        assert seen==[str(tmp_path/"axial.step")]
    else:
        # Stop at the actual terminal writer after public CLI/config resolution.
        assert cli.main(args)==2
        assert seen==["override.msh" if route=="override" else "configured.msh"]


@pytest.mark.parametrize("controls",[
    {"output":[]},{"output":"mesh.msh"},{"path":1},{"output_path":False},
    {"output":{"path":[]}},{"output":{"unknown_geometry":1}},
    {"unknown_geometry":1},
])
def test_native_output_controls_validate_shape_and_keep_unknown_refusal(controls):
    c=config();c.update(controls)
    with pytest.raises(ConfigError):resolve_geometry(c)
