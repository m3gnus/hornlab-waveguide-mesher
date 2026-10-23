"""Composable native OSSE: one rule table and every implemented composite."""
from __future__ import annotations

import copy
import itertools
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
from identity_harness import release_tree
from test_source_contour import arc_distance

from hornlab_mesher.axial_scale import AxialModel
from hornlab_mesher.config_builder import build_geometry_params, resolve_geometry
from hornlab_mesher.config_parser import ConfigError
from hornlab_mesher.datums import derive_datums
from hornlab_mesher.native_boundary import (
    FEATURES,
    IMPLEMENTED_COMPOSITES,
    LEGACY_MARKERS,
    validate_native_boundary,
)
from hornlab_mesher.preview import build_preview_geometry
from hornlab_mesher.preview.dimensions import canonical_dimensions

SCALE, AXIAL = 1.6, .7


def composite(end=100., scale=SCALE, axial=AXIAL):
    return {"formula":"OSSE-NATIVE","mode":"bare","scale":scale,"axial_scale":axial,
        "profile":{"L_mm":120.,"r0_mm":12.7,"a_deg":40.,"a0_deg":15.5,"k":1.,"s":0},
        "terminating_arc":{"contract_revision":1,"join_t":.05,"end_tangent_deg":end},
        "mesh":{"wall_thickness_mm":0,"vertical_offset_mm":7.,"allow_large_mesh":True},
        "source":{"source_shape":0}}


def model(c=None):
    return resolve_geometry(composite() if c is None else c).geometry.arc_meridian


# One payload per feature, at the location the rule table owns.
PAYLOADS = {
    "adapter":{"throat_adapter":{"mode":"authored","contract_revision":1,"driver_exit_diameter_mm":25.4,
        "exit_half_angle_deg":10.,"length_mm":40.,"join_t":.137,"driver_handle_mm":10.,"body_handle_mm":15.}},
    "controls":{"throat_adapter":{"mode":"controls","contract_revision":1,"join_t":0,
        "control_points_mm":[[-31.7,16.2],[-22.1,16.9],[-9.3,20.4]]}},
    "arc":{"terminating_arc":{"contract_revision":1,"join_t":.05,"end_tangent_deg":100.}},
    "roundover":{"mesh":{"mouth_roundover_radius_mm":5.}},
    "axial":{"axial_scale":1.2},
    "source_disk":{"source_body":{"radius_mm":18.,"outer_radius_mm":22.,"depth_mm":8.}},
}


def spelled(marker, names):
    c={"formula":marker,"profile":{"L_mm":120.,"r0_mm":12.7,"a_deg":40.,"a0_deg":15.5,"k":1.,"s":0},"mesh":{}}
    for name in names:
        for key,value in copy.deepcopy(PAYLOADS[name]).items():
            if isinstance(value,dict) and isinstance(c.get(key),dict):
                c[key].update(value)
            else:
                c[key]=value
    return c


def expected_refusal(marker, names):
    """The contract's verdict for one spelling, in rule order; None accepts."""
    if "source_disk" in names or marker == "SOURCE-DISK":
        return "SOURCE-DISK must stay alone"
    slots=[FEATURES[name].slot for name in names]
    if len(set(slots)) < len(slots):
        return "conflict"
    if LEGACY_MARKERS.get(names) == marker:
        return None
    if names in LEGACY_MARKERS:
        return f"write {LEGACY_MARKERS[names]}"
    if names not in IMPLEMENTED_COMPOSITES:
        return "not supported yet"
    return None if marker == "OSSE-NATIVE" else "write OSSE-NATIVE"


# Authored and controls adapters share one payload key; they cannot co-occur.
SETS=[frozenset(s) for n in (2,3) for s in itertools.combinations(sorted(PAYLOADS),n)
      if not {"adapter","controls"} <= set(s)]


@pytest.mark.parametrize("names",SETS,ids=["+".join(sorted(s)) for s in SETS])
def test_rule_matrix_in_every_spelling(names):
    markers={"OSSE-NATIVE"}|{FEATURES[name].legacy_marker for name in names}|set(LEGACY_MARKERS.values())
    for marker in sorted(markers):
        c=spelled(marker,names)
        refusal=expected_refusal(marker,names)
        if refusal is None:
            assert validate_native_boundary(c) == marker
        else:
            with pytest.raises(ConfigError,match=refusal):
                validate_native_boundary(c)
    assert sum(expected_refusal(marker,names) is None for marker in markers) == (
        names in LEGACY_MARKERS or names in IMPLEMENTED_COMPOSITES)


def test_implemented_composites_are_exactly_this_release_line():
    assert IMPLEMENTED_COMPOSITES == {frozenset(s) for s in (
        {"arc","axial"},{"adapter","arc"},{"adapter","arc","axial"},{"roundover","axial"},
        {"adapter","roundover"},{"adapter","roundover","axial"})}


@pytest.mark.parametrize("name",sorted(PAYLOADS))
def test_single_feature_must_keep_its_legacy_marker(name):
    with pytest.raises(ConfigError,match=f"write {FEATURES[name].legacy_marker}|SOURCE-DISK must stay alone"):
        validate_native_boundary(spelled("OSSE-NATIVE",{name}))


def test_composite_requires_two_features_and_exact_locations():
    with pytest.raises(ConfigError,match="at least two"):
        validate_native_boundary(spelled("OSSE-NATIVE",set()))
    c=composite();c["profile"]["axial_scale"]=c.pop("axial_scale")
    with pytest.raises(ConfigError,match="exact root axial_scale"):
        build_geometry_params(c)
    c=composite();c["profile"]["formula"]="OSSE-ARC"
    with pytest.raises(ConfigError,match="must agree"):
        build_geometry_params(c)


@pytest.mark.parametrize("end",[60.,100.,130.])
def test_circle_osculates_the_mapped_conic_and_is_never_mapped(end):
    c=composite(end);m=model(c)
    params=build_geometry_params(c)[0]
    body=AxialModel.from_params(params)
    assert (body.xy_scale,body.axial_scale) == (SCALE,AXIAL)
    assert m.join_z == pytest.approx(.05*120*AXIAL,rel=1e-15)
    z=np.linspace(0,m.join_z,401)
    np.testing.assert_allclose(m.body(z)[0],body.body(z)[0],rtol=0,atol=1e-12)
    J,tangent=body.body(m.join_z)
    np.testing.assert_allclose(m.arc(m.beta),J,rtol=0,atol=1e-12)
    d=float(tangent[1]);dd=body.second_bound(m.join_z,m.join_z)
    assert math.tan(m.beta) == pytest.approx(d,rel=1e-13)
    assert m.radius == pytest.approx((1+d*d)**1.5/dd,rel=1e-13)
    # Literal mapped conic: authored z = physical z/axial, radius times scale.
    A=12.7**2;B=12.7*math.tan(math.radians(15.5));C=math.tan(math.radians(40))**2
    u=z/AXIAL
    np.testing.assert_allclose(m.body(z)[0][:,1],SCALE*np.sqrt(A+2*B*u+C*u*u),rtol=1e-14,atol=1e-12)
    P=m.arc(np.linspace(m.beta,m.end,401))
    distance=np.hypot(P[:,0]-m.center_z,P[:,1]-m.center_r)
    np.testing.assert_allclose(distance,m.radius,rtol=1e-9,atol=0)
    # An unmapped circle in the authored frame would not osculate after mapping.
    uniform=model(composite(end,axial=SCALE))
    assert m.radius != pytest.approx(uniform.radius,rel=1e-3)


def test_identity_formula_fingerprint_and_metadata():
    c=composite();m=model(c);r=resolve_geometry(c)
    assert r.formula == "OSSE-NATIVE"
    assert m.identity["formula"] == "OSSE-NATIVE"
    assert m.identity["authoredBody"]["axial_scale"] == AXIAL
    assert r.sampling_metadata["construction_fingerprint"] == m.fingerprint
    params=build_geometry_params(c)[0]
    assert params["type"] == "OSSE-NATIVE" and params["terminatingArcFingerprint"] == m.fingerprint
    # The same mapped body under a different axial factor is a different design.
    assert model(composite(axial=.8)).fingerprint != m.fingerprint
    # Equal factors keep their composite spelling and identity, distinct from OSSE-ARC.
    legacy=composite(axial=SCALE);legacy["formula"]="OSSE-ARC";legacy.pop("axial_scale")
    equal=model(composite(axial=SCALE));arc=model(legacy)
    np.testing.assert_array_equal(equal.poles,arc.poles)
    assert equal.radius == arc.radius and equal.fingerprint != arc.fingerprint


def test_preview_dimensions_and_datums():
    c=composite();m=model(c);g=resolve_geometry(c).geometry
    preview=build_preview_geometry(c)
    assert preview.metadata["formula"] == "OSSE-NATIVE"
    assert preview.metadata["construction_fingerprint"] == m.fingerprint
    inner=next(s for s in preview.surfaces if s.role=="horn.inner")
    pts=np.asarray(inner.positions,dtype=np.float64)
    r=np.hypot(pts[:,0],pts[:,1]-m.offset)
    assert r.max() == pytest.approx(m.reach,rel=1e-6)
    dims=canonical_dimensions(c)
    assert dims["mouth_opening"] == [2*m.reach]*2
    np.testing.assert_allclose(dims["horn_overall"],np.ptp(m.bounds,axis=0))
    d=derive_datums(g,None)
    assert d["WG_ARC_JOIN_PLANE"]["origin_mm"][2] == m.join_z
    assert d["WG_BODY_MOUTH_REFERENCE_PLANE"]["origin_mm"][2] == pytest.approx(120*AXIAL)
    assert d["WG_MOUTH_PLANE"]["origin_mm"] == [0.,7.,float(m.opening[0])]


def test_msh_groups_vertex_oracle_and_arc_certificate(tmp_path):
    meshio=pytest.importorskip("meshio");pytest.importorskip("gmsh")
    from hornlab_mesher.config_builder import build_from_config
    c=composite(60.);c["mesh"].update(scale_to_metres=False,throat_res_mm=2,mouth_res_mm=2)
    m=model(c);result=build_from_config(c,tmp_path/"composite.msh")
    assert result.metadata["construction_fingerprint"] == m.fingerprint
    assert result.metadata["terminatingArc"]["formula"] == "OSSE-NATIVE"
    assert result.metadata["terminatingArcMeshChordBoundMm"] <= .01
    mesh=meshio.read(result.mesh_path);p=np.asarray(mesh.points)
    tri=mesh.get_cells_type("triangle");tags=mesh.get_cell_data("gmsh:physical","triangle")
    assert set(tags) == {1,2}
    wall=np.unique(tri[tags==1]);r=np.hypot(p[:,0],p[:,1]-7)
    body=wall[r[wall]<=m.poles[-1,1]+1e-8];arc=wall[r[wall]>m.poles[-1,1]+1e-8]
    assert len(body) and len(arc)
    np.testing.assert_allclose(r[body],m.body(p[body,2])[0][:,1],atol=1e-9,rtol=0)
    np.testing.assert_allclose(np.hypot(p[arc,2]-m.center_z,r[arc]-m.center_r),m.radius,atol=1e-9,rtol=0)


def test_step_recipe_carries_the_axial_scale(tmp_path):
    gmsh=pytest.importorskip("gmsh")
    from hornlab_mesher.cad import write_step_from_config
    c=composite(60.);m=model(c)
    path,info=write_step_from_config(c,tmp_path/"composite.step")
    assert info.terminating_arc["fingerprint"] == m.fingerprint
    np.testing.assert_array_equal(info.bounding_box_mm,m.bounds)
    text=path.read_text()
    # Composites carry one header recipe; the legacy arc trailer stays OSSE-ARC's.
    assert "hornlab-terminating-arc" not in text and text.count("/* native-composite ") == 1
    header=text.split("HEADER;\n",1)[1]
    assert header.startswith("/* native-composite ")
    recipe=json.loads(header.split("/* native-composite ")[1].split(" */")[0])
    assert recipe == {"construction_fingerprint":m.fingerprint,**m.identity}
    assert info.construction == m.identity and info.construction_fingerprint == m.fingerprint
    assert recipe["formula"] == "OSSE-NATIVE"
    assert (recipe["authoredBody"]["scale"],recipe["authoredBody"]["axial_scale"]) == (SCALE,AXIAL)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal",0);gmsh.open(str(path))
        for _,face in gmsh.model.getEntities(2):
            lo,hi=gmsh.model.getParametrizationBounds(2,face)
            x,y,z=gmsh.model.getValue(2,face,((np.asarray(lo)+np.asarray(hi))/2).tolist())
            r=math.hypot(x,y-m.offset)
            if r<=m.poles[-1,1]+1e-8:
                assert r == pytest.approx(m.body(z)[0][1],abs=1e-9)
            else:
                assert math.hypot(z-m.center_z,r-m.center_r) == pytest.approx(m.radius,abs=1e-9)
    finally:
        gmsh.finalize()


OLD_READER = '''
import json, sys
from pathlib import Path
import hornlab_mesher
assert Path(hornlab_mesher.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
from hornlab_mesher import build_from_config, write_step_from_config
from hornlab_mesher.config_builder import build_geometry_params
config = json.loads(sys.argv[2])
out = Path(sys.argv[3])
for name, call in (("params", lambda: build_geometry_params(config)),
                   ("msh", lambda: build_from_config(config, out / "old.msh")),
                   ("step", lambda: write_step_from_config(config, out / "old.step"))):
    try:
        call()
    except Exception as exc:
        print(name, type(exc).__name__, str(exc).replace("\\n", " "))
    else:
        print(name, "ACCEPTED")
print("files", sorted(p.name for p in out.iterdir()))
'''


@pytest.fixture(scope="module",params=["v0.2.6","v0.2.7"])
def old_reader_tree(request,tmp_path_factory):
    return request.param,release_tree(tmp_path_factory,request.param)


def assert_old_reader_refuses(old_reader_tree,config,tmp_path):
    tag,tree=old_reader_tree
    result=subprocess.run([sys.executable,"-c",OLD_READER,str(tree),json.dumps(config),str(tmp_path)],
        cwd=tree,env={**os.environ,"PYTHONPATH":str(tree)},
        capture_output=True,text=True,check=True,timeout=120)
    lines=result.stdout.splitlines()
    assert [line.split()[0] for line in lines] == ["params","msh","step","files"],(tag,result.stdout)
    assert all(line.split()[1] == "ConfigError" for line in lines[:3]),(tag,result.stdout)
    assert lines[-1] == "files []",(tag,result.stdout)


def test_released_readers_fail_closed(old_reader_tree,tmp_path):
    assert_old_reader_refuses(old_reader_tree,composite(),tmp_path)


def legacy_acceptance_cases():
    """Inputs whose v0.2.6 acceptance the rule table must not change."""
    from test_native_single_feature_identity import identity_cases
    cases=dict(identity_cases())
    for section in ("profile","parameters"):
        moved=copy.deepcopy(cases["OSSE-ADAPTER"])
        moved.setdefault(section,{})["throat_adapter"]=moved.pop("throat_adapter")
        yield f"adapter-{section}-alias",moved,section == "profile"
    nested=copy.deepcopy(cases["OSSE-ADAPTER"]);nested["extra"]={"axial_scale":1}
    yield "unrelated-nested-axial-key",nested,True
    padded=copy.deepcopy(cases["OSSE-ARC-forward"]);padded["formula"]=" osse-arc "
    yield "padded-arc-marker",padded,False


@pytest.mark.parametrize("name,config,accepted",list(legacy_acceptance_cases()),
    ids=[name for name,_,_ in legacy_acceptance_cases()])
def test_rule_table_keeps_legacy_acceptance(name,config,accepted):
    if accepted:
        build_geometry_params(config)
    else:
        with pytest.raises(ConfigError):
            build_geometry_params(config)


# Slice 2: Adapter x Arc (x Axial). The adapter keeps its OSSE-ADAPTER-AXIAL
# construction; the retained conic runs from its join to the circle join.
ADAPTER=dict(PAYLOADS["adapter"]["throat_adapter"],join_t=.02)
ADAPTER_SETS={"adapter+arc":None,"adapter+arc+axial":AXIAL}


def adapter_arc(axial=AXIAL,end=60.,scale=SCALE):
    c=composite(end,scale=scale,axial=axial)
    if axial is None:
        c.pop("axial_scale")
    c["throat_adapter"]=dict(ADAPTER)
    c["mesh"].update(scale_to_metres=False,throat_res_mm=4,mouth_res_mm=4)
    return c


def physical_adapter(c):
    from hornlab_mesher.adapter_axial import PhysicalAdapter
    params=build_geometry_params(c)[0]
    axis=c.get("axial_scale",c["scale"])
    # A roundover's wall is its own; the adapter is the bare OSSE-ADAPTER-AXIAL cubic.
    return PhysicalAdapter.from_params(dict(params,type="OSSE",wallThickness=0,adapterAxialScale=axis))


def monotone_residual(curve,z,r,lo=0.,hi=1.):
    """Radial residual within the axial domain; endpoint distance outside it."""
    z,r=np.broadcast_arrays(np.atleast_1d(z).astype(float),np.atleast_1d(r).astype(float))
    lo,hi=np.full_like(z,lo),np.full_like(z,hi)
    for _ in range(80):
        mid=(lo+hi)/2
        below=curve(mid)[:,0] < z
        lo,hi=np.where(below,mid,lo),np.where(below,hi,mid)
    point=curve((lo+hi)/2)
    return np.hypot(point[:,0]-z,point[:,1]-r)


def cubic_residual(m,z,r):
    """Radial distance to the throat cubic at the same z; z is monotone in u."""
    return monotone_residual(lambda u:m.throat.cubic_values(u)[0],z,r)


def meridian_residual(m,z,r):
    """Residual to the retained cubic, conic or finite circular arc."""
    z,r=np.atleast_1d(z).astype(float),np.atleast_1d(r).astype(float)
    out=np.full(z.shape,np.inf)
    cubic=z <= m.conic_start_z+1e-9
    out[cubic]=cubic_residual(m,np.minimum(z[cubic],m.conic_start_z),r[cubic])
    conic=(z >= m.conic_start_z-1e-9)&(z <= m.join_z+1e-9)
    out[conic]=np.minimum(out[conic],np.abs(m.body(z[conic])[0][:,1]-r[conic]))
    circle=arc_distance(m.arc(m.beta),m.arc(m.end),(m.center_z,m.center_r),direction="ccw")
    return np.minimum(out,circle(z,r))


@pytest.mark.parametrize("end",[60.,100.,130.,160.])
@pytest.mark.parametrize("name",sorted(ADAPTER_SETS))
def test_meridian_residual_checks_arc_span(name,end):
    m=model(adapter_arc(ADAPTER_SETS[name],end))
    points=m.arc(np.linspace(m.beta,m.end,401))
    # Include both endpoints and, at 160 degrees, the return behind the join.
    if end == 160. and ADAPTER_SETS[name] is None:
        assert np.any(points[:,0] < m.join_z)
    np.testing.assert_allclose(meridian_residual(m,*points.T),0,atol=1e-9,rtol=0)
    assert meridian_residual(m,*m.arc(m.end+.05))[0] > 1e-4


@pytest.mark.parametrize("end",[60.,100.,130.])
@pytest.mark.parametrize("name",sorted(ADAPTER_SETS))
def test_adapter_arc_meridian_certificates(name,end):
    axial=ADAPTER_SETS[name];axis=SCALE if axial is None else axial
    c=adapter_arc(axial,end);m=model(c);ref=physical_adapter(c)
    assert validate_native_boundary(c) == "OSSE-NATIVE"
    assert (ref.xy_scale,ref.axial_scale) == (SCALE,axis)
    # The throat is the OSSE-ADAPTER-AXIAL cubic, ending on the conic's first pole.
    np.testing.assert_array_equal(m.throat.cubic,ref.cubic)
    np.testing.assert_array_equal(m.poles[0],m.throat.cubic[-1])
    assert m.conic_start_z == pytest.approx(40*axis,rel=1e-15)
    assert m.source_radius == 25.4/2*SCALE
    # The retained conic is the mapped adapter body, adapter join to circle join.
    t=np.linspace(ADAPTER["join_t"],.05,201)
    p,d,dd=ref.body_values(t)
    np.testing.assert_allclose(m.body(p[:,0])[0],p,rtol=0,atol=1e-11)
    assert m.join_z == pytest.approx(float(p[-1,0]),abs=1e-12)
    # Literal mapped conic: authored body coordinate is (z-shift)/axial.
    u=(p[:,0]-m.shift)/axis
    A,B,C=12.7**2,12.7*math.tan(math.radians(15.5)),math.tan(math.radians(40))**2
    np.testing.assert_allclose(m.body(p[:,0])[0][:,1],SCALE*np.sqrt(A+2*B*u+C*u*u),rtol=1e-14,atol=1e-12)
    # G1 at the adapter join.
    cd=m.throat.cubic_values(np.array([1.]))[1][0];bd=m.body(m.conic_start_z)[1]
    assert math.atan2(cd[1],cd[0]) == pytest.approx(math.atan2(bd[1],bd[0]),abs=1e-12)
    # The circle osculates the mapped conic: point, tangent and curvature.
    np.testing.assert_allclose(m.arc(m.beta),p[-1],rtol=0,atol=1e-11)
    assert math.tan(m.beta) == pytest.approx(d[-1,1]/d[-1,0],rel=1e-12)
    curvature=(d[-1,0]*dd[-1,1]-d[-1,1]*dd[-1,0])/np.linalg.norm(d[-1])**3
    assert 1/m.radius == pytest.approx(curvature,rel=1e-12)
    P=m.arc(np.linspace(m.beta,m.end,401))
    np.testing.assert_allclose(np.hypot(P[:,0]-m.center_z,P[:,1]-m.center_r),m.radius,rtol=1e-9,atol=0)
    # The retained conic is an exact rational quadratic of the same body.
    s=np.linspace(0,1,101)[:,None];w=m.weights
    basis=np.hstack(((1-s)**2*w[0],2*s*(1-s)*w[1],s*s*w[2]))
    curve=basis@m.poles/basis.sum(axis=1,keepdims=True)
    assert np.max(np.abs(m.body(curve[:,0])[0][:,1]-curve[:,1])) <= 1e-9


def test_adapter_arc_identity_and_uniform_axial_spelling():
    plain=model(adapter_arc(None));uniform=model(adapter_arc(SCALE));axial=model(adapter_arc())
    assert plain.identity["construction"] == "circular-cubic-conic-osculating-arc-v1"
    assert plain.identity["adapter"] == {**ADAPTER,"contract_revision":1}
    assert plain.identity["source"] == "flat-driver-rim-v1"
    assert plain.identity["formula"] == axial.identity["formula"] == "OSSE-NATIVE"
    assert "axial_scale" not in plain.identity["authoredBody"]
    assert axial.identity["authoredBody"]["axial_scale"] == AXIAL
    # An authored axial factor equal to scale builds the same horn, but it is a
    # different feature set and therefore a different design identity.
    np.testing.assert_array_equal(plain.poles,uniform.poles)
    np.testing.assert_array_equal(plain.throat.cubic,uniform.throat.cubic)
    assert plain.radius == uniform.radius and plain.fingerprint != uniform.fingerprint
    assert len({plain.fingerprint,uniform.fingerprint,axial.fingerprint}) == 3
    params=build_geometry_params(adapter_arc())[0]
    assert params["type"] == "OSSE-NATIVE" and params["terminatingArcFingerprint"] == axial.fingerprint
    assert "adapterAxialScale" not in params and "construction_fingerprint" not in params


def test_adapter_arc_refusals():
    late=adapter_arc();late["throat_adapter"]["join_t"]=.05
    with pytest.raises(ConfigError,match="exceed the adapter join_t"):
        build_geometry_params(late)
    aliased=adapter_arc();aliased["profile"]["throat_adapter"]=aliased.pop("throat_adapter")
    with pytest.raises(ConfigError,match="exact root throat_adapter"):
        build_geometry_params(aliased)
    for marker in ("OSSE-ADAPTER","OSSE-ARC","OSSE-ADAPTER-AXIAL"):
        legacy=adapter_arc();legacy["formula"]=marker
        with pytest.raises(ConfigError,match="write OSSE-NATIVE"):
            build_geometry_params(legacy)
    controls=adapter_arc();controls["throat_adapter"]=copy.deepcopy(PAYLOADS["controls"]["throat_adapter"])
    with pytest.raises(ConfigError,match="not supported yet"):
        build_geometry_params(controls)
    # A returning circle that would reach back into a wide throat cubic.
    wide=adapter_arc(None,180.,1.)
    wide["profile"].update(L_mm=30.,a_deg=60.,k=2.)
    wide["throat_adapter"].update(driver_exit_diameter_mm=400.,length_mm=80.,join_t=.3)
    wide["terminating_arc"]["join_t"]=.4
    with pytest.raises(ConfigError,match="clear the throat adapter"):
        build_geometry_params(wide)
    wall=adapter_arc();wall["mesh"]["wall_thickness_mm"]=1
    with pytest.raises(ConfigError):
        build_geometry_params(wall)


@pytest.mark.parametrize("name",sorted(ADAPTER_SETS))
def test_adapter_arc_design_file_round_trip(name):
    # A design file stores the config as JSON; reading it back is the same design.
    c=adapter_arc(ADAPTER_SETS[name]);text=json.dumps(c,sort_keys=True)
    again=json.loads(text)
    assert json.dumps(again,sort_keys=True) == text
    first,second=build_geometry_params(c),build_geometry_params(again)
    assert json.dumps(first,sort_keys=True) == json.dumps(second,sort_keys=True)
    assert model(again).fingerprint == model(c).fingerprint
    assert validate_native_boundary(again) == "OSSE-NATIVE"


@pytest.mark.parametrize("name",sorted(ADAPTER_SETS))
def test_adapter_arc_preview_dimensions_and_datums(name):
    c=adapter_arc(ADAPTER_SETS[name],100.);m=model(c);g=resolve_geometry(c).geometry
    preview=build_preview_geometry(c)
    assert preview.metadata["formula"] == "OSSE-NATIVE"
    assert preview.metadata["construction_fingerprint"] == m.fingerprint
    assert preview.metadata["source_radius_mm"] == m.source_radius
    assert preview.metadata["source_auto_angle_deg"] == m.source_angle_deg
    inner=next(s for s in preview.surfaces if s.role == "horn.inner")
    pts=np.asarray(inner.positions,dtype=np.float64);r=np.hypot(pts[:,0],pts[:,1]-m.offset)
    assert pts[:,2].min() == 0 and r[pts[:,2] == 0] == pytest.approx(m.source_radius,rel=1e-6)
    assert r.max() == pytest.approx(m.envelope_radius,rel=1e-6)
    # Every rendered vertex lies on the exact meridian within binary32 placement.
    assert np.max(meridian_residual(m,pts[:,2],r)) <= 1e-4
    fidelity=preview.metadata["fidelity"]["horn.inner"]
    assert fidelity["max_chord_error_mm"] <= .008 and fidelity["measurement_complete"]
    cap=next(s for s in preview.surfaces if s.role == "source_cap")
    capr=np.hypot(np.asarray(cap.positions)[:,0],np.asarray(cap.positions)[:,1]-m.offset)
    assert capr.max() == pytest.approx(m.source_radius,rel=1e-6)
    dims=canonical_dimensions(c)
    assert dims["mouth_opening"] == [2*m.reach]*2
    np.testing.assert_allclose(dims["horn_overall"],np.ptp(m.bounds,axis=0))
    d=derive_datums(g,None);axis=c.get("axial_scale",SCALE)
    assert d["WG_ADAPTER_JOIN_PLANE"]["origin_mm"] == [0.,7.,m.conic_start_z]
    assert d["WG_ARC_JOIN_PLANE"]["origin_mm"][2] == m.join_z
    assert d["WG_BODY_MOUTH_REFERENCE_PLANE"]["origin_mm"][2] == pytest.approx(40*axis+120*(1-ADAPTER["join_t"])*axis)
    assert d["WG_MOUTH_PLANE"]["origin_mm"] == [0.,7.,float(m.opening[0])]


@pytest.mark.parametrize("name",sorted(ADAPTER_SETS))
def test_adapter_arc_msh_groups_vertex_oracle_and_certificate(name,tmp_path):
    meshio=pytest.importorskip("meshio");pytest.importorskip("gmsh")
    from hornlab_mesher.config_builder import build_from_config
    c=adapter_arc(ADAPTER_SETS[name]);m=model(c)
    result=build_from_config(c,tmp_path/"composite.msh")
    assert result.metadata["construction_fingerprint"] == m.fingerprint
    assert result.metadata["terminatingArc"]["adapterJoinMm"] == m.poles[0].tolist()
    assert result.metadata["terminatingArcMeshChordBoundMm"] <= .01
    mesh=meshio.read(result.mesh_path);p=np.asarray(mesh.points)
    tri=mesh.get_cells_type("triangle");tags=mesh.get_cell_data("gmsh:physical","triangle")
    assert set(tags) == {1,2}
    wall=np.unique(tri[tags == 1]);r=np.hypot(p[:,0],p[:,1]-7)
    assert np.max(meridian_residual(m,p[wall,2],r[wall])) <= 1e-9
    for owned in (p[wall,2] < m.conic_start_z-1e-6,(p[wall,2] > m.conic_start_z+1e-6)&(p[wall,2] < m.join_z-1e-6),
                  r[wall] > m.poles[-1,1]+1e-6):
        assert owned.sum() > 10
    source=tri[tags == 2];v=p[source]
    np.testing.assert_allclose(v[:,:,2],0,atol=1e-12)
    cross=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0])
    assert np.all(cross[:,2] > 0)
    from hornlab_mesher.edges import build_edge_table
    rim=build_edge_table(source);rim_nodes=rim.lo[rim.count == 1]
    # The source disk ends exactly on the driver rim the cubic starts from.
    np.testing.assert_allclose(np.hypot(p[rim_nodes,0],p[rim_nodes,1]-7),m.source_radius,atol=1e-10,rtol=0)
    assert .98*math.pi*m.source_radius**2 < .5*cross[:,2].sum() <= math.pi*m.source_radius**2
    # Welded everywhere except the open mouth rim.
    edges=build_edge_table(tri)
    assert set(edges.count) == {1,2}
    np.testing.assert_allclose(r[edges.lo[edges.count == 1]],m.reach,atol=1e-9,rtol=0)


@pytest.mark.parametrize("name",sorted(ADAPTER_SETS))
def test_adapter_arc_step_header_recipe_and_reopened_faces(name,tmp_path):
    gmsh=pytest.importorskip("gmsh")
    from hornlab_mesher.cad import write_step_from_config
    c=adapter_arc(ADAPTER_SETS[name]);m=model(c)
    path,info=write_step_from_config(c,tmp_path/"composite.step")
    text=path.read_text()
    recipe=json.loads(text.split("HEADER;\n/* native-composite ",1)[1].split(" */")[0])
    assert recipe == {"construction_fingerprint":m.fingerprint,**m.identity}
    assert "native-construction" not in text and "hornlab-terminating-arc" not in text
    assert info.construction_fingerprint == m.fingerprint and info.terminating_arc["fingerprint"] == m.fingerprint
    assert info.datums["WG_ADAPTER_JOIN_PLANE"]["origin_mm"][2] == m.conic_start_z
    assert info.body == "surface" and info.throat_opened
    np.testing.assert_array_equal(info.bounding_box_mm,m.bounds)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal",0);gmsh.open(str(path))
        faces=gmsh.model.getEntities(2)
        assert len(faces) == 4*(3 if m.beta < math.pi/2 < m.end else 2)+4
        for _,face in faces:
            lo,hi=gmsh.model.getParametrizationBounds(2,face)
            x,y,z=gmsh.model.getValue(2,face,((np.asarray(lo)+np.asarray(hi))/2).tolist())
            assert meridian_residual(m,z,math.hypot(x,y-m.offset))[0] <= 1e-9
    finally:
        gmsh.finalize()


@pytest.mark.parametrize("name",sorted(ADAPTER_SETS))
def test_released_readers_refuse_adapter_arc(name,old_reader_tree,tmp_path):
    assert_old_reader_refuses(old_reader_tree,adapter_arc(ADAPTER_SETS[name]),tmp_path)


# Slice 3: Roundover x Axial. The body is mapped first; the offset wall and the
# lip are then built in finished millimetres against the mapped body.
def roundover_axial(axial=AXIAL,scale=1.):
    return {"formula":"OSSE-NATIVE","mode":"freestanding","scale":scale,"axial_scale":axial,
        "profile":{"L_mm":12.,"r0_mm":1.27,"a_deg":45,"a0_deg":10,"k":1,"s":0},
        "mesh":{"wall_thickness_mm":.2,"mouth_roundover_radius_mm":1.,"scale_to_metres":False,
            "throat_res_mm":.2,"mouth_res_mm":.4,"rear_res_mm":.4,"max_triangles":100000}}


def roundover_model(c):
    return resolve_geometry(c).geometry.roundover


@pytest.mark.parametrize("axial,scale",[(AXIAL,1.),(1.6,1.2),(1.,.8)])
def test_roundover_axial_maps_the_body_and_builds_the_lip_in_mm(axial,scale):
    from hornlab_mesher.mouth_roundover import AxialRoundover
    c=roundover_axial(axial,scale);m=roundover_model(c)
    assert type(m) is AxialRoundover and (m.xy_scale,m.axial_scale) == (scale,axial)
    params=build_geometry_params(c)[0]
    assert params["type"] == "OSSE-NATIVE" and params["mouthRoundoverFingerprint"] == m.fingerprint
    body=AxialModel.from_params(dict(params,wallThickness=0))
    assert m.length == body.length == pytest.approx(12*axial,rel=1e-15)
    z=np.linspace(0,m.length,401)
    np.testing.assert_allclose(m.body(z)[0],body.body(z)[0],rtol=0,atol=1e-12)
    np.testing.assert_allclose(m.body(z)[1],body.body(z)[1],rtol=1e-13,atol=0)
    # Literal mapped conic: authored z = physical z/axial, radius times scale.
    u=z/axial;A=1.27**2;B=1.27*math.tan(math.radians(10));C=1.
    np.testing.assert_allclose(m.body(z)[0][:,1],scale*np.sqrt(A+2*B*u+C*u*u),rtol=1e-13,atol=1e-13)
    # The outer wall is the finished-mm normal offset of the mapped inner wall.
    inner,tangent=m.body(z);outer=m.body(z,True)[0]
    normal=np.column_stack((-tangent[:,1],tangent[:,0]))/np.hypot(1,tangent[:,1])[:,None]
    np.testing.assert_allclose(outer,inner+.2*normal,rtol=0,atol=1e-12)
    # The lip: radius 1 mm (never mapped), tangent to the mapped mouth, ending
    # half a turn later on the outer wall's lip circle.
    mouth=inner[-1];assert m.beta == pytest.approx(math.atan(tangent[-1,1]),abs=1e-15)
    np.testing.assert_allclose(m.arc(0),mouth,rtol=0,atol=1e-12)
    assert np.hypot(*(m.center-mouth)) == pytest.approx(1.,rel=1e-14)
    P=m.arc(np.linspace(0,m.sweep,201))
    np.testing.assert_allclose(np.hypot(P[:,0]-m.center[0],P[:,1]-m.center[1]),1.,rtol=1e-13)
    np.testing.assert_allclose(m.arc(m.sweep,True),m.center+[0,.8],rtol=0,atol=1e-12)


def test_roundover_axial_identity_and_uniform_spelling():
    legacy=roundover_axial();legacy.pop("axial_scale");legacy["formula"]="OSSE-ROUNDOVER"
    plain=roundover_model(legacy);equal=roundover_model(roundover_axial(1.))
    z=np.linspace(0,plain.length,101)
    np.testing.assert_allclose(equal.body(z)[0],plain.body(z)[0],rtol=1e-15,atol=1e-15)
    assert equal.fingerprint != plain.fingerprint and plain.formula == "OSSE-ROUNDOVER"
    assert roundover_model(roundover_axial(.8)).fingerprint != roundover_model(roundover_axial()).fingerprint
    r=resolve_geometry(roundover_axial())
    assert r.formula == "OSSE-NATIVE"
    meta=r.sampling_metadata
    assert meta["construction_fingerprint"] == meta["mouthRoundover"]["fingerprint"]
    assert (meta["mouthRoundover"]["formula"],meta["mouthRoundover"]["axialScale"]) == ("OSSE-NATIVE",AXIAL)
    m=r.geometry.roundover
    assert m.identity == {"contractRevision":1,"formula":"OSSE-NATIVE","construction":"circular-osse-axial-lip-v1",
        "features":["axial","roundover"],"authoredBody":{"L":12.,"r0":1.27,"a":45.,"a0":10.,"k":1.,
            "scale":1.,"axial_scale":AXIAL},"verticalOffsetMm":0.,"source":"flat-throat-rim-v1",
        "roundover":{"radiusMm":1.,"wallThicknessMm":.2}}
    # As for the arc composites, the fingerprint is the hash of the header recipe identity.
    import hashlib
    assert m.fingerprint == hashlib.sha256(json.dumps(m.identity,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def test_roundover_axial_refusals():
    for marker in ("OSSE-ROUNDOVER","OSSE-AXIAL"):
        c=roundover_axial();c["formula"]=marker
        with pytest.raises(ConfigError,match="write OSSE-NATIVE"):
            build_geometry_params(c)
    c=roundover_axial(.05)
    with pytest.raises(ConfigError,match="axial_scale/scale"):
        build_geometry_params(c)
    # The mapped body keeps the axial feature's own domain, mouth limit included.
    c=roundover_axial(1.,10.);c["profile"].update(L_mm=2000.,r0_mm=10.)
    with pytest.raises(ConfigError,match="mouth radius exceeds 10000"):
        build_geometry_params(c)
    c=roundover_axial();c["profile"]["axial_scale"]=c.pop("axial_scale")
    with pytest.raises(ConfigError,match="exact root axial_scale"):
        build_geometry_params(c)
    c=roundover_axial();c["throat_adapter"]=copy.deepcopy(PAYLOADS["controls"]["throat_adapter"])
    with pytest.raises(ConfigError,match="not supported yet"):
        build_geometry_params(c)
    c=roundover_axial();c["profile"]["formula"]="OSSE-ROUNDOVER"
    with pytest.raises(ConfigError,match="must agree"):
        build_geometry_params(c)


def test_roundover_axial_design_file_round_trip():
    c=roundover_axial();again=json.loads(json.dumps(c,sort_keys=True))
    assert json.dumps(build_geometry_params(c),sort_keys=True) == json.dumps(build_geometry_params(again),sort_keys=True)
    assert roundover_model(again).fingerprint == roundover_model(c).fingerprint


def test_roundover_axial_preview_dimensions_and_datums():
    from types import SimpleNamespace
    c=roundover_axial();m=roundover_model(c);g=resolve_geometry(c).geometry
    preview=build_preview_geometry(c)
    assert preview.metadata["formula"] == "OSSE-NATIVE"
    assert preview.metadata["mouthRoundover"]["fingerprint"] == m.fingerprint
    for s in preview.surfaces:
        assert preview.metadata["fidelity"][s.role]["max_chord_error_mm"] <= .008
    inner=next(s for s in preview.surfaces if s.role == "horn.inner")
    pts=np.asarray(inner.positions,dtype=np.float64)
    radius=np.hypot(pts[:,0],pts[:,1])
    # The lip curls back below the mouth plane, always outside the mouth radius.
    body=(pts[:,2] <= m.length)&(radius <= float(m.body(m.length)[0][1])+1e-6)
    assert body.sum() > 100
    np.testing.assert_allclose(radius[body],m.body(pts[body,2])[0][:,1],atol=1e-5,rtol=0)
    dims=canonical_dimensions(c);box=np.asarray(m.bounds)
    assert dims["mouth_opening"] == [2*float(m.body(m.length)[0][1])]*2
    np.testing.assert_allclose(dims["horn_overall"],box[1]-box[0])
    d=derive_datums(g,SimpleNamespace(source_axis="z"))
    mouth=np.asarray(d["WG_MOUTH_OUTLINE_INNER"]["points_mm"])
    np.testing.assert_allclose(np.hypot(mouth[:,0],mouth[:,1]),m.body(m.length)[0][1],rtol=1e-12)
    np.testing.assert_allclose(mouth[:,2],m.length)
    outer=np.asarray(d["WG_MOUTH_OUTLINE_OUTER"]["points_mm"])
    np.testing.assert_allclose(outer[:,2],m.body(m.length,True)[0][0])


def test_roundover_axial_msh_and_lip_certificate(tmp_path):
    meshio=pytest.importorskip("meshio");pytest.importorskip("gmsh")
    from hornlab_mesher.config_builder import build_from_config
    from hornlab_mesher.edges import build_edge_table
    c=roundover_axial();m=roundover_model(c)
    result=build_from_config(c,tmp_path/"roundover.msh")
    assert result.metadata["construction_fingerprint"] == m.fingerprint
    assert result.metadata["mouthRoundoverMeshChordBoundMm"] <= .01
    mesh=meshio.read(result.mesh_path);p=np.asarray(mesh.points)
    tri=mesh.get_cells_type("triangle");tags=mesh.get_cell_data("gmsh:physical","triangle")
    assert set(tags) == {1,2}
    assert set(build_edge_table(tri).count) == {2}
    r=np.hypot(p[:,0],p[:,1]);wall=np.unique(tri[tags == 1])
    distance=np.hypot(p[wall,2]-m.center[0],r[wall]-m.center[1])
    for radius in (1.,.8):
        assert np.sum(np.abs(distance-radius) < 1e-9) > 50
    inner=wall[(p[wall,2] > 1e-6)&(p[wall,2] < m.length-1e-6)&(r[wall] < m.body(p[wall,2])[0][:,1]+1e-6)]
    assert len(inner) > 50
    # The inner body is a certified cubic Hermite fit (FIT_TOL_MM).
    np.testing.assert_allclose(r[inner],m.body(p[inner,2])[0][:,1],atol=1e-4+1e-9,rtol=0)


def test_roundover_axial_step_header_recipe(tmp_path):
    pytest.importorskip("gmsh")
    from hornlab_mesher.cad import write_step_from_config
    c=roundover_axial();m=roundover_model(c)
    path,info=write_step_from_config(c,tmp_path/"roundover.step")
    text=path.read_text()
    recipe=json.loads(text.split("HEADER;\n/* native-composite ",1)[1].split(" */")[0])
    assert recipe == {"construction_fingerprint":m.fingerprint,**m.identity}
    assert "hornlab-mouth-roundover" not in text
    assert info.body == "solid" and info.construction_fingerprint == m.fingerprint
    assert info.mouth_roundover["formula"] == "OSSE-NATIVE"
    np.testing.assert_array_equal(info.bounding_box_mm,m.bounds)


def test_released_readers_refuse_roundover_axial(old_reader_tree,tmp_path):
    assert_old_reader_refuses(old_reader_tree,roundover_axial(),tmp_path)


# Slice 4: Adapter x Roundover (x Axial). The OSSE-ADAPTER-AXIAL cubic, the
# mapped conic from its join, then the offset wall, lip, rim and rear closure
# in finished mm; a closed solid.
LIP_SETS={"adapter+roundover":None,"adapter+roundover+axial":AXIAL}
LIP_ADAPTER={"mode":"authored","contract_revision":1,"driver_exit_diameter_mm":2.,"exit_half_angle_deg":5.,
    "length_mm":3.,"join_t":.1,"driver_handle_mm":.8,"body_handle_mm":1.2}


def adapter_roundover(axial=AXIAL,scale=1.):
    """A small design that meshes quickly; dimensions follow slice 3's."""
    c=roundover_axial(axial,scale)
    if axial is None:
        c.pop("axial_scale")
    c["throat_adapter"]=dict(LIP_ADAPTER)
    return c


def converging_adapter_roundover():
    """A converging driver exit: the outer wall starts in front of the throat
    plane, so the rear return crosses it. Placed off axis."""
    c=adapter_roundover(None);c["throat_adapter"]["exit_half_angle_deg"]=-15.
    c["vertical_offset_mm"]=-12.
    return c


# Designs that are previewed, meshed and exported.
LIP_CASES={"adapter+roundover":lambda:adapter_roundover(None),"adapter+roundover+axial":adapter_roundover,
    "converging-placed":converging_adapter_roundover}


def large_adapter_roundover(axial=AXIAL):
    c={"formula":"OSSE-NATIVE","mode":"freestanding","scale":SCALE,"axial_scale":axial,
        "profile":dict(composite()["profile"]),"throat_adapter":dict(PAYLOADS["adapter"]["throat_adapter"]),
        "mesh":{"wall_thickness_mm":5.,"mouth_roundover_radius_mm":25.,"vertical_offset_mm":7.}}
    if axial is None:
        c.pop("axial_scale")
    return c


def lip_residual(m,z,r):
    """Distance in (z,r) to the closed meridian of the opened solid.

    Throat cubic, its offset, retained conic and its offset are z-monotone;
    the lips are circles; the rim, rear return, rear plate and opened bore
    are segments. The fitted walls are within FIT_TOL_MM of these curves.
    """
    z,r=np.atleast_1d(z).astype(float),np.atleast_1d(r).astype(float)
    ro=float(m.rear_start[1]);rd=m.source_radius;cz,cr=m.center
    pieces=[monotone_residual(lambda u:m.cubic(u)[0],z,r),
        monotone_residual(lambda u:m.cubic(u,True)[0],z,r),
        np.where((z >= m.start-1e-9)&(z <= m.length+1e-9),np.abs(m.body(np.clip(z,m.start,m.length))[0][:,1]-r),np.inf),
        monotone_residual(lambda t:m.body(t,True)[0],z,r,m.start,m.length),
        arc_distance(m.arc(0),m.arc(m.sweep),m.center,direction="ccw")(z,r),
        arc_distance(m.arc(0,True),m.arc(m.sweep,True),m.center,direction="ccw")(z,r),
        np.where((r >= cr+m.radius-m.wall-1e-9)&(r <= cr+m.radius+1e-9),np.abs(z-cz),np.inf),
        np.where((z >= -m.wall-1e-9)&(z <= m.rear_start[0]+1e-9),np.abs(r-ro),np.inf),
        np.where(r <= ro+1e-9,np.abs(z+m.wall),np.inf),
        np.where((z >= -m.wall-1e-9)&(z <= 1e-9),np.abs(r-rd),np.inf),
        np.where(r <= rd+1e-9,np.abs(z),np.inf)]
    return np.min(pieces,axis=0)


def solid_volume(m,opened=True):
    """2*pi times the first moment of the closed meridian loop (Green's theorem)."""
    u=np.linspace(0,1,20001);z=np.linspace(m.start,m.length,20001);phi=np.linspace(0,m.sweep,20001)
    rear=[[-m.wall,m.rear_start[1]]]+([[-m.wall,m.source_radius],[0.,m.source_radius]] if opened else [[-m.wall,0.],[0.,0.]])
    P=np.vstack((m.cubic(u)[0],m.body(z)[0][1:],m.arc(phi)[1:],m.arc(phi[::-1],True),
        m.body(z[::-1],True)[0][1:],m.cubic(u[::-1],True)[0][1:],rear))
    zz,rr=P[:,0],P[:,1]
    return abs(2*math.pi*np.sum((rr[:-1]**2+rr[:-1]*rr[1:]+rr[1:]**2)/6*np.diff(zz)))


@pytest.mark.parametrize("axial",[None,AXIAL,SCALE],ids=["plain","axial","uniform-axial"])
@pytest.mark.parametrize("size",["small","large"])
def test_adapter_roundover_meridian_certificates(size,axial):
    from hornlab_mesher.mouth_roundover import FIT_TOL_MM,AdapterRoundover
    c=adapter_roundover(axial) if size == "small" else large_adapter_roundover(axial)
    scale=c["scale"];axis=scale if axial is None else axial
    m=roundover_model(c);ref=physical_adapter(c)
    assert validate_native_boundary(c) == "OSSE-NATIVE" and type(m) is AdapterRoundover
    assert (m.xy_scale,m.axial_scale) == (ref.xy_scale,ref.axial_scale) == (scale,axis)
    # The throat is the OSSE-ADAPTER-AXIAL cubic; the conic starts on its exact end.
    np.testing.assert_array_equal(m.throat.cubic,ref.cubic)
    assert m.start == ref.cubic[-1,0] == pytest.approx(c["throat_adapter"]["length_mm"]*axis,rel=1e-15)
    assert m.source_radius == c["throat_adapter"]["driver_exit_diameter_mm"]/2*scale
    # The retained conic is the mapped adapter body, adapter join to body mouth.
    t=np.linspace(c["throat_adapter"]["join_t"],1,201)
    p,d,dd=ref.body_values(t)
    np.testing.assert_allclose(m.body(p[:,0])[0],p,rtol=0,atol=1e-11)
    assert m.length == pytest.approx(float(p[-1,0]),rel=1e-14)
    profile=c["profile"];u=(p[:,0]-m.shift)/axis;r0=profile["r0_mm"]
    A,B,C=r0**2,r0*math.tan(math.radians(profile["a0_deg"])),math.tan(math.radians(profile["a_deg"]))**2
    np.testing.assert_allclose(m.body(p[:,0])[0][:,1],scale*np.sqrt(A+2*B*u+C*u*u),rtol=1e-13,atol=1e-12)
    # G1 at the adapter join, on the inner wall and on its offset.
    cd=m.cubic(np.array([1.]))[1][0];bd=m.body(m.start)[1]
    assert math.atan2(cd[1],cd[0]) == pytest.approx(math.atan2(bd[1],bd[0]),abs=1e-9)
    np.testing.assert_allclose(m.cubic(np.array([1.]),True)[0][0],m.body(m.start,True)[0],rtol=0,atol=1e-9)
    od=m.cubic(np.array([1.]),True)[1][0];obd=m.body(m.start,True)[1]
    assert math.atan2(od[1],od[0]) == pytest.approx(math.atan2(obd[1],obd[0]),abs=1e-9)
    # The outer cubic is the exact normal offset: distance wall, along the normal,
    # with derivative c'*(1-wall*kappa) and the fold guard's margin.
    s=np.linspace(0,1,4001);inner,tangent=m.cubic(s);outer,outer_tangent=m.cubic(s,True)
    np.testing.assert_allclose(np.linalg.norm(outer-inner,axis=1),m.wall,rtol=1e-12)
    np.testing.assert_allclose(np.sum((outer-inner)*tangent,axis=1),0,atol=1e-12*m.length)
    factor=np.sum(outer_tangent*tangent,axis=1)/np.sum(tangent*tangent,axis=1)
    assert factor.min() >= .6
    cross=outer_tangent[:,0]*tangent[:,1]-outer_tangent[:,1]*tangent[:,0]
    np.testing.assert_allclose(cross,0,atol=1e-9*np.max(np.abs(tangent))**2)
    # Each offset derivative is the exact derivative of the offset point.
    h=1e-6;numeric=(m.cubic(s[1:-1]+h,True)[0]-m.cubic(s[1:-1]-h,True)[0])/(2*h)
    np.testing.assert_allclose(numeric,outer_tangent[1:-1],rtol=0,atol=1e-6*np.max(np.abs(outer_tangent)))
    # The lip: radius in finished mm, tangent to the mapped mouth, half a turn.
    mouth,mouth_tangent=m.body(m.length)
    np.testing.assert_allclose(m.arc(0),mouth,rtol=0,atol=1e-12*m.length)
    assert m.beta == pytest.approx(math.atan(mouth_tangent[1]),abs=1e-15)
    assert np.hypot(*(m.center-mouth)) == pytest.approx(m.radius,rel=1e-13)
    np.testing.assert_allclose(m.arc(m.sweep,True),m.center+[0,m.radius-m.wall],rtol=0,atol=1e-12*m.length)
    # The certified Hermite fit of the offset cubic stays within FIT_TOL_MM,
    # and its whole-interval fourth-derivative bound dominates the curve's.
    stations=m.cubic_fit_stations()
    P,D=m.cubic(stations,True)
    worst=0.
    for j,(lo,hi) in enumerate(zip(stations[:-1],stations[1:])):
        w=np.linspace(0,1,65)[:,None];h=hi-lo
        hermite=((2*w**3-3*w**2+1)*P[j]+(w**3-2*w**2+w)*h*D[j]+(-2*w**3+3*w**2)*P[j+1]+(w**3-w**2)*h*D[j+1])
        worst=max(worst,float(np.max(np.linalg.norm(hermite-m.cubic(lo+w[:,0]*h,True)[0],axis=1))))
        e=h/8;q=lo+np.linspace(2*e,h-2*e,5)
        fourth=(m.cubic(q-2*e,True)[0]-4*m.cubic(q-e,True)[0]+6*m.cubic(q,True)[0]
                -4*m.cubic(q+e,True)[0]+m.cubic(q+2*e,True)[0])/e**4
        assert np.max(np.linalg.norm(fourth,axis=1)) <= m.offset_bounds(lo,hi)[1]*1.001
    assert worst <= m.fit_tolerance <= FIT_TOL_MM


def test_adapter_roundover_identity_and_uniform_axial_spelling():
    import hashlib
    plain=roundover_model(adapter_roundover(None));uniform=roundover_model(adapter_roundover(1.))
    axial=roundover_model(adapter_roundover())
    assert plain.identity["construction"] == "circular-cubic-conic-lip-v1"
    assert plain.identity["formula"] == axial.identity["formula"] == "OSSE-NATIVE"
    assert plain.identity["adapter"] == {**LIP_ADAPTER,"contract_revision":1}
    assert (plain.identity["features"],axial.identity["features"]) == (["adapter","roundover"],["adapter","axial","roundover"])
    assert plain.identity["authoredBody"] == {"L":12.,"r0":1.27,"a":45.,"a0":10.,"k":1.,"scale":1.}
    # An authored axial factor equal to scale builds the same horn as a
    # different feature set, so it keeps its own identity.
    np.testing.assert_array_equal(plain.throat.cubic,uniform.throat.cubic)
    assert plain.coefficients == uniform.coefficients and plain.center.tolist() == uniform.center.tolist()
    assert len({plain.fingerprint,uniform.fingerprint,axial.fingerprint}) == 3
    for m in (plain,axial):
        assert m.fingerprint == hashlib.sha256(json.dumps(m.identity,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    # Without the adapter the same body is Roundover x Axial, another design.
    assert roundover_model(roundover_axial()).fingerprint != axial.fingerprint
    params=build_geometry_params(adapter_roundover())[0]
    assert params["type"] == "OSSE-NATIVE" and params["mouthRoundoverFingerprint"] == axial.fingerprint
    assert params["throat_adapter"] == {**LIP_ADAPTER,"contract_revision":1}
    r=resolve_geometry(adapter_roundover());meta=r.sampling_metadata
    assert r.formula == "OSSE-NATIVE" and meta["construction_fingerprint"] == axial.fingerprint
    lip=meta["mouthRoundover"]
    assert (lip["formula"],lip["xyScale"],lip["axialScale"]) == ("OSSE-NATIVE",1.,AXIAL)
    assert lip["adapterJoinMm"] == axial.throat.cubic[-1].tolist() and lip["sourceRadiusMm"] == 1.
    assert lip["sourceAngleDeg"] == axial.throat.source_angle and lip["mouthZMm"] == axial.length


def test_composite_authored_fingerprints():
    # Only normalized authored numbers enter these recipes, never libm results
    # or mapped geometry. These goldens therefore apply on every platform.
    assert {"arc+axial":model(composite()).fingerprint,"adapter+arc":model(adapter_arc(None)).fingerprint,
            "adapter+arc+axial":model(adapter_arc()).fingerprint,
            "roundover+axial":roundover_model(roundover_axial()).fingerprint,
            "adapter+roundover":roundover_model(adapter_roundover(None)).fingerprint,
            "adapter+roundover+axial":roundover_model(adapter_roundover()).fingerprint,
            "large+adapter+roundover+axial":roundover_model(large_adapter_roundover()).fingerprint} == {
        "arc+axial":"a8c66f0dc5e026f13a68524e8c3a752f14f0b39a623397a70a31974a14d2f847",
        "adapter+arc":"a8e47944f065418e3c4453e66aba77debdd1b157fff9613fa7f20a74a5797d9b",
        "adapter+arc+axial":"86285a52ae5879a9bc00effab0fa00d836c71b5873fd3d94e5090771667ef85d",
        "roundover+axial":"51c4094ef8686eaeabb4749c75d13a02c8ed9a5afd6567e03c41c3d27ba7011a",
        "adapter+roundover":"015d08118f1a627ed45d9b7fecfd8523dc777819491c1ceb96f5c06ed5c08061",
        "adapter+roundover+axial":"086b246a81eae2f34a225b997309a50ee2611d8e1f39b8169b4cd1bb057b6301",
        "large+adapter+roundover+axial":"f1a1830f7458be8f4384b0236d3854bb6aa5d882d9b14096dff75b3a9cddb060"}


@pytest.mark.parametrize("config",[composite(),adapter_arc(None),adapter_arc(),roundover_axial(),
    adapter_roundover(None),adapter_roundover(),large_adapter_roundover()])
@pytest.mark.parametrize("function",["tan","radians"])
def test_composite_identity_ignores_libm_roundoff(config,function,monkeypatch):
    from dataclasses import replace
    resolve=roundover_model if "mouth_roundover_radius_mm" in config["mesh"] else model
    before=resolve(config)
    original=getattr(math,function)
    calls=[]
    def perturbed(value):
        calls.append(value)
        return math.nextafter(original(value),math.inf)
    monkeypatch.setattr(math,function,perturbed)
    after=resolve(config)
    assert calls
    assert after.identity == before.identity
    assert after.fingerprint == before.fingerprint
    # The physical map can round away a one-ULP libm change. Also inject the
    # reported final-slope difference directly, independent of that rounding.
    changed=replace(before,throat_slope=math.nextafter(before.throat_slope,math.inf))
    assert changed.fingerprint == before.fingerprint


def test_monotone_residual_measures_outside_endpoint_distance():
    # A non-default interval also exercises the retained outer conic's domain.
    curve=lambda t:np.stack((t,2*t),axis=-1)
    z=np.array([-100.,2.,3.,4.,100.]);r=np.array([4.,4.,6.,8.,8.])
    np.testing.assert_allclose(monotone_residual(curve,z,r,2.,4.),[102.,0.,0.,0.,96.],atol=1e-12)


@pytest.mark.parametrize("name",sorted(LIP_CASES))
@pytest.mark.parametrize("outer",[False,True],ids=["inner","outer"])
def test_lip_residual_checks_arc_span(name,outer):
    m=roundover_model(LIP_CASES[name]())
    # This point is on the forward semicircle, before the mouth join.
    assert lip_residual(m,*m.arc(-m.beta,outer))[0] > 1e-4
    points=m.arc(np.linspace(0,m.sweep,401),outer)
    # The mouth join and rim endpoints belong to the actual lip span.
    np.testing.assert_allclose(lip_residual(m,*points.T),0,atol=1e-9,rtol=0)


@pytest.mark.parametrize("name",sorted(LIP_CASES))
def test_lip_residual_rejects_points_outside_axial_domain(name):
    m=roundover_model(LIP_CASES[name]())
    # Matching an endpoint radius must never hide a thousand mm of axial error.
    curves=[lambda t:m.cubic(t)[0],lambda t:m.cubic(t,True)[0],
            lambda t:m.body(m.start+t*(m.length-m.start))[0],
            lambda t:m.body(m.start+t*(m.length-m.start),True)[0]]
    radii=np.concatenate([curve(np.array([0.,1.]))[:,1] for curve in curves])
    for z in (-1013.8,1013.8):
        assert np.min(lip_residual(m,np.full_like(radii,z),radii)) > 900.


def test_adapter_roundover_refusals():
    for marker in ("OSSE-ROUNDOVER","OSSE-ADAPTER","OSSE-ADAPTER-AXIAL","OSSE-AXIAL"):
        c=adapter_roundover();c["formula"]=marker
        with pytest.raises(ConfigError,match="write OSSE-NATIVE"):
            build_geometry_params(c)
    c=adapter_roundover();c["throat_adapter"]=copy.deepcopy(PAYLOADS["controls"]["throat_adapter"])
    with pytest.raises(ConfigError,match="not supported yet"):
        build_geometry_params(c)
    c=adapter_roundover();c["terminating_arc"]=copy.deepcopy(PAYLOADS["arc"]["terminating_arc"])
    with pytest.raises(ConfigError,match="conflict"):
        build_geometry_params(c)
    c=adapter_roundover();c["profile"]["throat_adapter"]=c.pop("throat_adapter")
    with pytest.raises(ConfigError,match="exact root throat_adapter"):
        build_geometry_params(c)
    c=adapter_roundover();c["mode"]="bare"
    with pytest.raises(ConfigError,match="freestanding"):
        build_geometry_params(c)
    # The adapter keeps its own domain and refusal prefix.
    c=adapter_roundover();c["throat_adapter"]["driver_handle_mm"]=10.
    with pytest.raises(ConfigError,match="Adapter axial scale refused"):
        build_geometry_params(c)
    # A driver as wide as the mouth, with the lip's centre behind the adapter join.
    c=adapter_roundover()
    c["throat_adapter"].update(driver_exit_diameter_mm=24.,exit_half_angle_deg=0.,length_mm=8.,join_t=.95,
        driver_handle_mm=2.,body_handle_mm=2.)
    with pytest.raises(ConfigError,match="clear the throat adapter"):
        build_geometry_params(c)
    # A converging exit and a wide mouth turn the wall too far to certify the offset.
    c=adapter_roundover();c["throat_adapter"]["exit_half_angle_deg"]=-20.;c["profile"]["a_deg"]=75.
    with pytest.raises(ConfigError,match="85 degrees"):
        build_geometry_params(c)
    # A tight throat bend inside a thick wall.
    c=adapter_roundover(None);c["mesh"]["mouth_roundover_radius_mm"]=1.5
    c["throat_adapter"].update(exit_half_angle_deg=-20.,driver_handle_mm=.5,body_handle_mm=.5)
    with pytest.raises(ConfigError,match="throat curvature fold"):
        build_geometry_params(c)
    c=adapter_roundover();c["throat_adapter"]["exit_half_angle_deg"]=80.
    with pytest.raises(ConfigError,match="within 75 degrees"):
        build_geometry_params(c)


@pytest.mark.parametrize("name",sorted(LIP_SETS))
def test_adapter_roundover_design_file_round_trip(name):
    c=adapter_roundover(LIP_SETS[name]);text=json.dumps(c,sort_keys=True)
    again=json.loads(text)
    assert json.dumps(build_geometry_params(again),sort_keys=True) == json.dumps(build_geometry_params(c),sort_keys=True)
    assert roundover_model(again).fingerprint == roundover_model(c).fingerprint
    assert validate_native_boundary(again) == "OSSE-NATIVE"


@pytest.mark.parametrize("name",sorted(LIP_CASES))
def test_adapter_roundover_preview_dimensions_and_datums(name):
    from types import SimpleNamespace
    c=LIP_CASES[name]();m=roundover_model(c);g=resolve_geometry(c).geometry
    preview=build_preview_geometry(c)
    assert preview.metadata["formula"] == "OSSE-NATIVE"
    assert preview.metadata["mouthRoundover"]["fingerprint"] == m.fingerprint
    assert preview.metadata["source_radius_mm"] == m.source_radius
    assert preview.metadata["source_auto_angle_deg"] == m.throat.source_angle
    for s in preview.surfaces:
        assert preview.metadata["fidelity"][s.role]["max_chord_error_mm"] <= .008
    for role in ("horn.inner","horn.outer","mouth_rim","wall.rear_return","wall.rear_cap"):
        pts=np.asarray(next(s for s in preview.surfaces if s.role == role).positions,dtype=np.float64)
        # Every rendered vertex lies on the closed meridian within binary32 placement.
        assert np.max(lip_residual(m,pts[:,2],np.hypot(pts[:,0],pts[:,1]-m.offset))) <= 1e-5,role
    inner=np.asarray(next(s for s in preview.surfaces if s.role == "horn.inner").positions,dtype=np.float64)
    radius=np.hypot(inner[:,0],inner[:,1]-m.offset)
    assert inner[:,2].min() == 0 and radius[inner[:,2] == 0] == pytest.approx(m.source_radius,rel=1e-6)
    throat=inner[:,2] < m.start
    assert throat.sum() > 100
    assert np.max(monotone_residual(lambda u:m.cubic(u)[0],inner[throat,2],radius[throat])) <= 1e-5
    for role,r in (("source_cap",m.source_radius),("wall.rear_cap",float(m.rear_start[1]))):
        cap=np.asarray(next(s for s in preview.surfaces if s.role == role).positions)
        assert np.hypot(cap[:,0],cap[:,1]-m.offset).max() == pytest.approx(r,rel=1e-6)
    dims=canonical_dimensions(c);box=np.asarray(m.bounds)
    assert dims["mouth_opening"] == [2*float(m.body(m.length)[0][1])]*2
    np.testing.assert_allclose(dims["horn_overall"],box[1]-box[0])
    d=derive_datums(g,SimpleNamespace(source_axis="z"))
    assert d["WG_ADAPTER_JOIN_PLANE"]["origin_mm"] == [0.,m.offset,m.start]
    assert d["WG_THROAT_PLANE"]["origin_mm"][2] == 0
    mouth=np.asarray(d["WG_MOUTH_OUTLINE_INNER"]["points_mm"])
    np.testing.assert_allclose(np.hypot(mouth[:,0],mouth[:,1]-m.offset),m.body(m.length)[0][1],rtol=1e-12)
    np.testing.assert_allclose(mouth[:,2],m.length)


@pytest.mark.parametrize("name",sorted(LIP_CASES))
def test_adapter_roundover_msh_closed_body_oracle_and_certificate(name,tmp_path):
    meshio=pytest.importorskip("meshio");pytest.importorskip("gmsh")
    from hornlab_mesher.config_builder import build_from_config
    from hornlab_mesher.edges import build_edge_table
    from hornlab_mesher.mouth_roundover import FIT_TOL_MM
    c=LIP_CASES[name]();m=roundover_model(c)
    if name == "converging-placed":
        assert m.rear_start[0] > 0 and m.offset == -12.
    result=build_from_config(c,tmp_path/"adapter-roundover.msh")
    assert result.metadata["construction_fingerprint"] == m.fingerprint
    assert result.metadata["mouthRoundoverMeshChordBoundMm"] <= .01
    mesh=meshio.read(result.mesh_path);p=np.asarray(mesh.points)
    tri=mesh.get_cells_type("triangle");tags=mesh.get_cell_data("gmsh:physical","triangle")
    assert set(tags) == {1,2}
    # A closed, consistently oriented body (what the BEMPP closed-body path solves).
    edges=build_edge_table(tri)
    assert set(edges.count) == {2}
    assert len(np.unique(tri))-edges.n_edges+len(tri) == 2
    directed={tuple(e) for e in np.concatenate((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]))}
    assert len(directed) == 3*len(tri) and all((b,a) in directed for a,b in directed)
    v=p[tri];signed=np.sum(np.einsum("ij,ij->i",v[:,0],np.cross(v[:,1],v[:,2])))/6
    assert signed == pytest.approx(solid_volume(m,opened=False),rel=1e-2)
    r=np.hypot(p[:,0],p[:,1]-m.offset);nodes=np.unique(tri)
    assert np.max(lip_residual(m,p[nodes,2],r[nodes])) <= FIT_TOL_MM+1e-9
    # The exact throat cubic (an OCC Bezier) and the exact lips carry vertices.
    wall=np.unique(tri[tags == 1])
    on_cubic=wall[(p[wall,2] > 1e-6)&(p[wall,2] < m.start-1e-6)&(r[wall] < m.throat.cubic_reach+1e-6)]
    exact=cubic_residual(m,p[on_cubic,2],r[on_cubic]) <= 1e-9
    assert exact.sum() > 50
    distance=np.hypot(p[wall,2]-m.center[0],r[wall]-m.center[1])
    for radius in (m.radius,m.radius-m.wall):
        assert np.sum(np.abs(distance-radius) < 1e-9) > 50
    # The source disk sits on the driver rim the cubic starts from.
    source=tri[tags == 2];rim=build_edge_table(source)
    np.testing.assert_allclose(p[source][:,:,2],0,atol=1e-12)
    np.testing.assert_allclose(r[rim.lo[rim.count == 1]],m.source_radius,atol=1e-10,rtol=0)


@pytest.mark.parametrize("name",sorted(LIP_CASES))
def test_adapter_roundover_step_closed_solid_recipe_and_reopened_faces(name,tmp_path):
    gmsh=pytest.importorskip("gmsh")
    from hornlab_mesher.cad import write_step_from_config
    from hornlab_mesher.mouth_roundover import FIT_TOL_MM
    c=LIP_CASES[name]();m=roundover_model(c)
    path,info=write_step_from_config(c,tmp_path/"adapter-roundover.step")
    text=path.read_text()
    recipe=json.loads(text.split("HEADER;\n/* native-composite ",1)[1].split(" */")[0])
    assert recipe == {"construction_fingerprint":m.fingerprint,**m.identity}
    assert text.count("/* native-") == 1 and "hornlab-mouth-roundover" not in text
    assert info.body == "solid" and info.throat_opened and info.construction_fingerprint == m.fingerprint
    assert info.mouth_roundover["formula"] == "OSSE-NATIVE" and info.mouth_roundover["fingerprint"] == m.fingerprint
    np.testing.assert_array_equal(info.bounding_box_mm,m.bounds)
    assert info.volume_mm3 == pytest.approx(solid_volume(m),rel=1e-5)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal",0);gmsh.open(str(path))
        assert len(gmsh.model.getEntities(3)) == 1
        volume=gmsh.model.getEntities(3)[0][1]
        assert gmsh.model.occ.getMass(3,volume) == pytest.approx(info.volume_mm3,rel=1e-9)
        faces=gmsh.model.getEntities(2)
        assert len(faces) == info.n_faces
        for _,face in faces:
            # A plane's parameter box is untrimmed; its boundary curves are not.
            points=[]
            if gmsh.model.getType(2,face) != "Plane":
                lo,hi=(np.asarray(v) for v in gmsh.model.getParametrizationBounds(2,face))
                points+=[gmsh.model.getValue(2,face,(lo+w*(hi-lo)).tolist()) for w in (.25,.5,.75)]
            for _,curve in gmsh.model.getBoundary([(2,face)],oriented=False):
                lo,hi=gmsh.model.getParametrizationBounds(1,abs(curve))
                points+=[gmsh.model.getValue(1,abs(curve),[lo[0]+w*(hi[0]-lo[0])]) for w in (0.,.3,.7,1.)]
            x,y,z=np.asarray(points).T
            assert np.max(lip_residual(m,z,np.hypot(x,y-m.offset))) <= FIT_TOL_MM+1e-9
    finally:
        gmsh.finalize()


@pytest.mark.parametrize("name",sorted(LIP_SETS))
def test_released_readers_refuse_adapter_roundover(name,old_reader_tree,tmp_path):
    assert_old_reader_refuses(old_reader_tree,adapter_roundover(LIP_SETS[name]),tmp_path)
