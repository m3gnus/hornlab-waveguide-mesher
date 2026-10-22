"""Composable native OSSE: one rule table, and the implemented Arc x Axial composite."""
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
from test_native_single_feature_identity import native_base_tree  # noqa: F401

from hornlab_mesher.axial_scale import AxialModel
from hornlab_mesher.config_builder import build_geometry_params, resolve_geometry
from hornlab_mesher.config_parser import ConfigError
from hornlab_mesher.datums import derive_datums
from hornlab_mesher.native_boundary import (
    FEATURES,
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
    if names != frozenset({"arc","axial"}):
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
        names in LEGACY_MARKERS or names == frozenset({"arc","axial"}))


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
    recipe=json.loads(path.read_text().split("/* hornlab-terminating-arc ")[1].split(" */")[0])
    assert recipe == m.metadata()["terminatingArc"]
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


def test_v026_reader_fails_closed(native_base_tree, tmp_path):  # noqa: F811
    result=subprocess.run([sys.executable,"-c",OLD_READER,str(native_base_tree),json.dumps(composite()),str(tmp_path)],
        cwd=native_base_tree,env={**os.environ,"PYTHONPATH":str(native_base_tree)},
        capture_output=True,text=True,check=True,timeout=120)
    lines=result.stdout.splitlines()
    assert [line.split()[0] for line in lines] == ["params","msh","step","files"],result.stdout
    assert all(line.split()[1] == "ConfigError" for line in lines[:3]),result.stdout
    assert lines[-1] == "files []",result.stdout


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
