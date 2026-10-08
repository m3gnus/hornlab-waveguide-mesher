"""Exact circular termination, public intent and complete terminal contracts."""
import copy
import json
import math

import numpy as np
import pytest

from hornlab_mesher.config_builder import build_geometry_params,resolve_geometry
from hornlab_mesher.config_parser import ConfigError
from hornlab_mesher.datums import derive_datums
from hornlab_mesher.preview import build_preview_geometry,PreviewOptionsV1
from hornlab_mesher.preview.dimensions import canonical_dimensions


def config(end=100.):
    return {"formula":"OSSE-ARC","mode":"bare","scale":1.,
        "profile":{"L_mm":120.,"r0_mm":12.7,"a_deg":40.,"a0_deg":15.5,"k":1.,"s":0},
        "terminating_arc":{"contract_revision":1,"join_t":.05,"end_tangent_deg":end},
        "mesh":{"wall_thickness_mm":0,"vertical_offset_mm":7.,"allow_large_mesh":True},
        "source":{"source_shape":0}}


def model(c=None):
    return resolve_geometry(config() if c is None else c).geometry.arc_meridian


@pytest.mark.parametrize("end",[30.,90.,100.,155.])
def test_literal_conic_and_osculating_circle(end):
    m=model(config(end))
    z=np.linspace(0,m.join_z,1001)
    A=(12.7)**2;B=12.7*math.tan(math.radians(15.5));C=math.tan(math.radians(40))**2
    expected=np.sqrt(A+2*B*z+C*z*z)
    np.testing.assert_allclose(m.body(z)[0][:,1],expected,rtol=0,atol=1e-12)
    J=m.body(m.join_z)[0]
    root=math.sqrt(A+2*B*m.join_z+C*m.join_z*m.join_z)
    d=(B+C*m.join_z)/root
    dd=(A*C-B*B)/root**3
    R=(1+d*d)**1.5/dd
    assert m.radius == pytest.approx(R,rel=1e-14)
    np.testing.assert_allclose(m.arc(m.beta),J,rtol=0,atol=1e-12)
    np.testing.assert_allclose(np.array([math.cos(m.beta),math.sin(m.beta)]),[1/math.hypot(1,d),d/math.hypot(1,d)],atol=1e-15)
    assert dd/(1+d*d)**1.5 == pytest.approx(1/R,rel=1e-14)
    phi=np.linspace(m.beta,m.end,401)
    P=m.arc(phi)
    np.testing.assert_allclose(np.hypot(P[:,0]-m.center_z,P[:,1]-m.center_r),R,atol=1e-12)
    assert np.all(np.diff(P[:,1])>0)
    # Independent rational Bernstein evaluation of the exact prefix.
    u=np.linspace(0,1,1001)
    basis=np.column_stack(((1-u)**2,2*u*(1-u),u*u))*m.weights
    conic=basis@m.poles/basis.sum(axis=1)[:,None]
    np.testing.assert_allclose(conic[:,1],np.sqrt(A+2*B*conic[:,0]+C*conic[:,0]**2),atol=1e-12,rtol=0)


def test_backward_endpoint_and_nominal_reference_do_not_define_envelope():
    c=config(155);g=resolve_geometry(c).geometry;m=g.arc_meridian
    assert .1 < m.opening[0] < m.join_z
    assert m.bounds[1][2] > m.join_z
    d=derive_datums(g,None)
    assert d["WG_MOUTH_PLANE"]["origin_mm"] == [0.,7.,float(m.opening[0])]
    assert d["WG_ARC_JOIN_PLANE"]["origin_mm"][2] == m.join_z
    assert d["WG_BODY_MOUTH_REFERENCE_PLANE"]["origin_mm"][2] == 120
    assert d["WG_BODY_MOUTH_REFERENCE_PLANE"]["nominal"] is True
    dims=canonical_dimensions(c)
    np.testing.assert_allclose(dims["horn_overall"],np.ptp(m.bounds,axis=0))
    assert dims["mouth_opening"] == [2*m.reach]*2


def test_immutable_authority_detached_identity_and_input_snapshot():
    c=config();m=model(c)
    for array in (m.poles,m.weights,resolve_geometry(c).geometry.inner_points):
        with pytest.raises(ValueError):array.setflags(write=True)
    original=m.fingerprint
    c["terminating_arc"]["join_t"] = .1
    detached=m.identity;detached["arc"]["joinT"] = .5
    metadata=m.metadata();metadata["terminatingArc"]["body"]["lengthMm"] = 1
    assert m.fingerprint == original
    assert m.identity["arc"]["joinT"] == .05
    with pytest.raises(Exception):m.length=4


def test_density_units_routing_and_placement_identity():
    a=config();b=copy.deepcopy(a)
    b["mesh"].update(throat_res_mm=1.3,mouth_res_mm=4.8,rear_res_mm=3.,length_segments=19,angular_segments=47,scale_to_metres=False)
    b["output"]={"path":"example.msh"}
    assert model(a)==model(b)
    b["mesh"]["vertical_offset_mm"]=8
    assert model(a).fingerprint != model(b).fingerprint
    b=config();b["vertical_offset_mm"]=7
    assert model(a).fingerprint == model(b).fingerprint
    b=config();b["scale"]=2
    mb=model(b);ma=model(a)
    assert mb.radius == pytest.approx(2*ma.radius)
    np.testing.assert_allclose(mb.opening,2*ma.opening)


@pytest.mark.parametrize("lod",["coarse","fine","inspection"])
@pytest.mark.parametrize("end",[30.,100.,155.])
def test_preview_normals_signed_curvature_and_continuous_fidelity(lod,end):
    c=config(end);m=model(c)
    preview=build_preview_geometry(c,PreviewOptionsV1(lod=lod))
    wall=next(s for s in preview.surfaces if s.role=="horn.inner")
    assert preview.metadata["construction_fingerprint"] == m.fingerprint
    np.testing.assert_allclose(preview.metadata["dimensions_mm"]["horn_overall"],np.ptp(m.bounds,axis=0))
    assert preview.metadata["fidelity"]["horn.inner"]["max_chord_error_mm_achieved"] <= .008
    assert preview.metadata["fidelity"]["horn.inner"]["max_normal_step_deg_achieved"] <= 3
    radius=np.hypot(wall.positions[:,0],wall.positions[:,1]-m.offset)
    arc=radius>m.poles[-1,1]+1e-8
    phi=np.arctan2(wall.positions[arc,2]-m.center_z,m.center_r-radius[arc])
    az=np.arctan2(wall.positions[arc,1]-m.offset,wall.positions[arc,0])
    expected=np.column_stack((-np.cos(phi)*np.cos(az),-np.cos(phi)*np.sin(az),np.sin(phi)))
    np.testing.assert_allclose(wall.normals[arc],expected,atol=1e-12)
    k1=np.full(len(phi),-1/m.radius);k2=np.cos(phi)/radius[arc]
    np.testing.assert_allclose(wall.curvature_mean[arc],(k1+k2)/2,atol=1e-13)
    np.testing.assert_allclose(wall.curvature_principal[arc],np.where(abs(k1)>abs(k2),k1,k2),atol=1e-13)
    if end>90:
        assert np.all(wall.curvature_mean[arc][phi>math.pi/2]<0)
    triangles=wall.positions[wall.indices.reshape(-1,3)]
    normals=wall.normals[wall.indices.reshape(-1,3)].mean(axis=1)
    assert np.min(np.einsum('ij,ij->i',np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),normals))>0


@pytest.mark.parametrize("section,key,value",[
    ("root","type","OSSE"),("profile","formula","ICW"),("parameters","type","R-OSSE"),
    ("root","axial_scale",1),("root","throat_adapter",{"mode":"off"}),("root","source_body",None),
    ("root","cross_section",{}),("root","enclosure",{}),("root","parameters",{}),
    ("profile","s",.2),("profile","n",4),("profile","q",.9),("profile","a_deg","40"),
    ("profile","throatExtLength",0),("mesh","mouth_roundover_radius_mm",0),
    ("mesh","wall_thickness_mm",1),("source","source_shape",1),("source","source_radius_mm",12),
    ("source","source_curv",1),("mesh","surface_fit","approximate"),("mesh","topology_mode","legacy"),
    ("mesh","quadrants","1234x"),("root","quadrants",True),("root","vertical_offset_mm",8),
    ("mesh","length_segments",False),("mesh","angular_segments",1.5),("mesh","max_triangles",10**400),
    ("mesh","allow_large_mesh",1),("mesh","scale_to_metres","false"),("mesh","mouth_res_mm",0),
    ("profile","L_mm",.1),("profile","r0_mm",.5),("profile","k",.1),
    ("profile","a_deg",76),("profile","a0_deg",40),("root","scale",.001),
    ("mesh","vertical_offset_mm",10001),
])
@pytest.mark.parametrize("entry",[resolve_geometry,build_preview_geometry,canonical_dimensions])
def test_original_unsupported_intent_refuses(section,key,value,entry):
    c=config();(c if section=="root" else c.setdefault(section,{}))[key]=value
    with pytest.raises(ConfigError):entry(c)


@pytest.mark.parametrize("value",[None,False,{},[],{"mode":"off"},
    {"contract_revision":1.,"join_t":.05,"end_tangent_deg":100}])
def test_payload_shape_and_exact_revision(value):
    c=config();c["terminating_arc"]=value
    with pytest.raises(ConfigError):resolve_geometry(c)


@pytest.mark.parametrize("key,value",[("join_t",0),("join_t",-1),("join_t",1.001),
    ("join_t",1e-8),("join_t",True),("join_t",".05"),("join_t",float("nan")),
    ("end_tangent_deg",27),("end_tangent_deg",180),("end_tangent_deg",181),
    ("end_tangent_deg",float("inf")),("end_tangent_deg",10**400)])
def test_coupled_geometric_domain_refuses(key,value):
    c=config();c["terminating_arc"][key]=value
    with pytest.raises(ConfigError):resolve_geometry(c)


@pytest.mark.parametrize("key",["terminating_arc","terminatingArc","Terminating_Arc"])
@pytest.mark.parametrize("section",["root","profile","parameters","mesh","source","output"])
@pytest.mark.parametrize("value",[None,{}])
def test_every_supplied_arc_intent_requires_exact_root_and_marker(key,section,value):
    c=config();c["formula"]="OSSE";c.pop("terminating_arc")
    (c if section=="root" else c.setdefault(section,{}))[key]=value
    with pytest.raises(ConfigError):resolve_geometry(c)


def test_missing_payload_unknown_fields_and_body_only_consumers_refuse():
    c=config();c.pop("terminating_arc")
    with pytest.raises(ConfigError):resolve_geometry(c)
    c=config();c["terminating_arc"]["radius_mm"]=10
    with pytest.raises(ConfigError):resolve_geometry(c)
    from hornlab_mesher.profile_formulas import calculate_osse,calculate_osse_curve,osse_total_length
    from hornlab_mesher.profile_sampling import build_point_grid_arrays
    p=build_geometry_params(config())[0]
    for fn,args in [(calculate_osse,(0,0,p)),(calculate_osse_curve,([0,1],0,p)),(osse_total_length,(p,)),(build_point_grid_arrays,(p,))]:
        with pytest.raises(ValueError,match="terminating arcs"):fn(*args)
    from hornlab_mesher.profile_common import _normalise_formula
    with pytest.raises(ValueError):_normalise_formula("OSSE-ARC")


@pytest.mark.parametrize("key,value",[("max_chord_error_mm",1e-10),("max_chord_error_mm",10**400),
    ("max_normal_step_deg",1e-308),("max_normal_step_deg",1e-320),("max_normal_step_deg",5e-324),
    ("max_normal_step_deg",10**400),("max_vertices",8),("min_silhouette_segments",10**400)])
def test_unachievable_preview_refuses_before_allocation(key,value):
    with pytest.raises(ValueError):build_preview_geometry(config(),PreviewOptionsV1(**{key:value}))


def test_actual_small_mesh_groups_source_join_topology_and_vertex_oracle(tmp_path):
    meshio=pytest.importorskip("meshio");pytest.importorskip("gmsh")
    from hornlab_mesher.config_builder import build_from_config
    c=config(32);c["mesh"].update(scale_to_metres=False,throat_res_mm=2,mouth_res_mm=2)
    m=model(c);result=build_from_config(c,tmp_path/"arc.msh")
    assert result.metadata["construction_fingerprint"] == m.fingerprint
    assert result.metadata["terminatingArcMeshChordBoundMm"] <= .01
    assert result.metadata["terminatingArcMeshCertificateScope"] == "circle-facets-to-analytic-surface-only"
    mesh=meshio.read(result.mesh_path);p=np.asarray(mesh.points);tri=mesh.get_cells_type("triangle")
    tags=mesh.get_cell_data("gmsh:physical","triangle")
    assert set(tags)=={1,2}
    src=tri[tags==2];wall=tri[tags==1]
    cross=np.cross(p[src[:,1]]-p[src[:,0]],p[src[:,2]]-p[src[:,0]])
    assert np.all(cross[:,2]>0)
    np.testing.assert_allclose(p[np.unique(src),2],0,atol=1e-10)
    r=np.hypot(p[:,0],p[:,1]-7)
    body_nodes=np.unique(wall)[r[np.unique(wall)]<=m.poles[-1,1]+1e-8]
    np.testing.assert_allclose(r[body_nodes],m.body(p[body_nodes,2])[0][:,1],atol=1e-9,rtol=0)
    arc_nodes=np.unique(wall)[r[np.unique(wall)]>m.poles[-1,1]+1e-8]
    np.testing.assert_allclose(np.hypot(p[arc_nodes,2]-m.center_z,r[arc_nodes]-m.center_r),m.radius,atol=1e-9,rtol=0)
    edges=np.sort(np.concatenate((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    assert np.max(counts)==2
    boundary=unique[counts==1]
    np.testing.assert_allclose(p[np.unique(boundary),2],m.opening[0],atol=1e-9)
    assert len(np.intersect1d(np.unique(src),np.unique(wall)))>=12
    join_nodes=np.flatnonzero(abs(p[:,2]-m.join_z)<1e-9)
    assert len(join_nodes)>=12
    join_edges=unique[np.all(np.isin(unique,join_nodes),axis=1)]
    assert len(join_edges)>=12
    assert np.all(counts[np.all(np.isin(unique,join_nodes),axis=1)]==2)


@pytest.mark.parametrize("end",[32.,100.,155.])
def test_actual_step_reopens_exact_surface_recipe_datums_and_bundle_refusal(tmp_path,end):
    gmsh=pytest.importorskip("gmsh")
    from hornlab_mesher.cad import write_step_from_config,write_wglink
    from hornlab_mesher.mesher import MesherError
    c=config(end);m=model(c)
    path,info=write_step_from_config(c,tmp_path/"arc.step")
    assert info.body=="surface" and info.volume_mm3 is None
    assert info.n_faces==(8 if end<=90 else 12)
    assert info.terminating_arc["fingerprint"]==m.fingerprint
    assert info.datums==derive_datums(resolve_geometry(c).geometry,None)
    np.testing.assert_array_equal(info.bounding_box_mm,m.bounds)
    text=path.read_text();recipe=text.split("/* hornlab-terminating-arc ")[1].split(" */")[0]
    assert json.loads(recipe)==m.metadata()["terminatingArc"]
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal",0);gmsh.open(str(path))
        assert not gmsh.model.getEntities(3)
        assert len(gmsh.model.getEntities(2))==info.n_faces
        for _,face in gmsh.model.getEntities(2):
            lo,hi=gmsh.model.getParametrizationBounds(2,face)
            uv=(np.asarray(lo)+np.asarray(hi))/2
            x,y,z=gmsh.model.getValue(2,face,uv.tolist())
            r=math.hypot(x,y-m.offset)
            if r<=m.poles[-1,1]+1e-8:
                assert r==pytest.approx(m.body(z)[0][1],abs=1e-9)
            else:
                assert math.hypot(z-m.center_z,r-m.center_r)==pytest.approx(m.radius,abs=1e-9)
    finally:gmsh.finalize()
    with pytest.raises(MesherError,match="terminating arcs"):
        write_wglink(resolve_geometry(c).geometry,tmp_path/"refused.wglink")
    assert not (tmp_path/"refused.wglink").exists()


def test_budget_and_terminal_failures_preserve_atomic_output_and_external_session(tmp_path,monkeypatch):
    gmsh=pytest.importorskip("gmsh")
    from hornlab_mesher.config_builder import build_from_config
    from hornlab_mesher.cad import write_step_from_config
    from hornlab_mesher.mesher import MesherError,TriangleBudgetExceeded
    c=config(32);c["mesh"].update(allow_large_mesh=False,max_triangles=10)
    target=tmp_path/"preserved.msh";target.write_bytes(b"existing mesh")
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal",0)
        before=gmsh.option.getNumber("Geometry.Tolerance")
        with pytest.raises(TriangleBudgetExceeded):build_from_config(c,target)
        assert target.read_bytes()==b"existing mesh"
        assert gmsh.isInitialized()
        assert gmsh.option.getNumber("Geometry.Tolerance")==before
        step=tmp_path/"preserved.step";step.write_bytes(b"existing step")
        def fail(*args):raise RuntimeError("injected export failure")
        monkeypatch.setattr(gmsh,"write",fail)
        with pytest.raises(MesherError,match="injected export failure"):write_step_from_config(config(32),step)
        assert step.read_bytes()==b"existing step"
        assert gmsh.isInitialized()
        assert gmsh.option.getNumber("Geometry.Tolerance")==before
    finally:gmsh.finalize()


def test_competing_authorities_refuse_before_native_construction(tmp_path):
    from hornlab_mesher.builders.point_grid_dispatch import build_point_grid
    from hornlab_mesher.cad import write_step
    from hornlab_mesher.mesher import MesherError
    geometry=resolve_geometry(config(32)).geometry
    # A direct caller can combine different dataclass layers only by making
    # their own subclass. The dispatch and CAD boundary still reject it.
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class Competing(type(geometry)):
        axial_model: object=None
    fields={key:value for key,value in geometry.__dict__.items()}
    bad=Competing(**fields,axial_model=object())
    with pytest.raises(ValueError,match="combined active native"):build_point_grid(bad)
    with pytest.raises(MesherError,match="combined active native"):write_step(bad,tmp_path/"bad.step")
    assert not (tmp_path/"bad.step").exists()


def test_unresolved_nearly_straight_body_refuses_as_configuration_error():
    c=config();c["profile"].update(a_deg=40.,a0_deg=math.nextafter(40.,-math.inf))
    with pytest.raises(ConfigError):resolve_geometry(c)


@pytest.mark.parametrize("patch",[{"vertical_offset_mm":8},{"source_shape":1},{"wall_thickness_mm":1},
    {"closed":False},{"source_radius_mm":12.7}])
def test_direct_geometry_cannot_override_immutable_native_intent(patch,tmp_path):
    from dataclasses import replace
    from hornlab_mesher.builders.point_grid_dispatch import build_point_grid
    g=replace(resolve_geometry(config(32)).geometry,**patch)
    with pytest.raises(ValueError,match="canonical placement"):build_point_grid(g)


@pytest.mark.parametrize("value",[0,1,"false",None])
def test_direct_mesh_units_flag_requires_boolean(value,tmp_path):
    from hornlab_mesher.mesher import build_mesh_with_info,MesherError
    with pytest.raises(MesherError,match="must be a boolean"):
        build_mesh_with_info(resolve_geometry(config(32)).geometry,output_path=tmp_path/"bad.msh",scale_to_metres=value)


@pytest.mark.parametrize("opened",[True,False])
def test_step_preserves_raw_and_reopened_shared_shell_edges(tmp_path,opened):
    import collections,re
    gmsh=pytest.importorskip("gmsh")
    from hornlab_mesher.cad import write_step_from_config
    c=config(100);m=model(c)
    path,info=write_step_from_config(c,tmp_path/"sewn.step",open_throat=opened)
    text=path.read_text()
    entities={int(i):(kind,value) for i,kind,value in re.findall(r'#(\d+)\s*=\s*(\w+)\s*\((.*?)\)\s*;',text,re.S)}
    edge_ids={i for i,(kind,_) in entities.items() if kind=="EDGE_CURVE"}
    count=collections.Counter()
    for kind,value in entities.values():
        if kind=="ORIENTED_EDGE":count.update(i for i in map(int,re.findall(r'#(\d+)',value)) if i in edge_ids)
    assert collections.Counter(count.values())==({1:8,2:20} if opened else {1:4,2:24})
    assert "MANIFOLD_SOLID_BREP" not in text and "CLOSED_SHELL" not in text
    assert text.count("OPEN_SHELL(")==1
    # Reconstruct the three temporary OCC carrier wrappers. Transport must
    # preserve every other STEP entity, and unexpected carriers must refuse.
    from hornlab_mesher.cad import _arc_surface_shell_transport
    from hornlab_mesher.mesher import MesherError
    raw=text.replace("MANIFOLD_SURFACE_SHAPE_REPRESENTATION(","ADVANCED_BREP_SHAPE_REPRESENTATION(")
    raw=re.sub(r"SHELL_BASED_SURFACE_MODEL\('',\(#(\d+)\)\)",r"MANIFOLD_SOLID_BREP('',#\1)",raw)
    raw=raw.replace("OPEN_SHELL(","CLOSED_SHELL(")
    assert _arc_surface_shell_transport(raw,n_faces=info.n_faces,open_throat=opened)==text
    bad=raw.replace("MANIFOLD_SOLID_BREP('',#","MANIFOLD_SOLID_BREP('unexpected',#",1)
    with pytest.raises(MesherError,match="carrier references"):
        _arc_surface_shell_transport(bad,n_faces=info.n_faces,open_throat=opened)
    with pytest.raises(MesherError,match="exactly one"):
        _arc_surface_shell_transport(raw+"\n#999999 = MANIFOLD_SOLID_BREP('',#16);",n_faces=info.n_faces,open_throat=opened)
    with pytest.raises(MesherError,match="every exact face"):
        _arc_surface_shell_transport(raw,n_faces=info.n_faces+1,open_throat=opened)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal",0);gmsh.open(str(path))
        assert not gmsh.model.getEntities(3)
        incidence=collections.Counter()
        for _,face in gmsh.model.getEntities(2):
            incidence.update(abs(edge) for dim,edge in gmsh.model.getBoundary([(2,face)],oriented=True,combined=False) if dim==1)
        assert collections.Counter(incidence.values())==({1:8,2:20} if opened else {1:4,2:24})
        join=[]
        for edge,n in incidence.items():
            low,high=gmsh.model.getParametrizationBounds(1,edge)
            x,y,z=gmsh.model.getValue(1,edge,[(low[0]+high[0])/2])
            if abs(z-m.join_z)<1e-8 and abs(math.hypot(x,y-m.offset)-m.poles[-1,1])<1e-8:
                join.append((edge,n))
        assert len(join)==4 and all(n==2 for _,n in join)
    finally:gmsh.finalize()


@pytest.mark.parametrize("name,value",[("allow_large_mesh","false"),("allow_large_mesh",1),
    ("max_triangles",10.5),("max_triangles",True),("max_triangles",10**400),("max_triangles",None),
    ("throat_res_mm","2"),("mouth_res_mm",False),("rear_res_mm",float("nan")),
    ("min_size_mm",1),("interface_res_mm",2),("aperture_res_scale",2)])
def test_direct_native_density_refuses_before_construction(name,value,tmp_path,monkeypatch):
    from dataclasses import replace
    from hornlab_mesher.geometry import MeshDensity
    from hornlab_mesher import mesher
    def forbidden(*args):raise AssertionError("native construction was reached")
    monkeypatch.setattr(mesher,"_dispatch_builder",forbidden)
    density=replace(MeshDensity(),**{name:value})
    with pytest.raises(ValueError):mesher.build_mesh_with_info(resolve_geometry(config(32)).geometry,density,tmp_path/"bad.msh")


@pytest.mark.parametrize("patch",[{"interfaces":("active",)},{"interface_offset_mm":3},
    {"topology_mode":"legacy"},{"preserve_grid":True},{"surface_fit":"approximate"},
    {"wg_topology":False}])
def test_direct_unsupported_controls_refuse_before_occ(patch,monkeypatch):
    from dataclasses import replace
    from hornlab_mesher.builders import terminating_arc
    g=replace(resolve_geometry(config(32)).geometry,**patch)
    def forbidden():raise AssertionError("OCC was reached")
    monkeypatch.setattr(terminating_arc,"require_gmsh",forbidden)
    with pytest.raises(ValueError):terminating_arc.build_arc(g)


def test_valid_returning_arc_mesh_passes_and_inverted_body_collar_refuses(tmp_path):
    meshio=pytest.importorskip("meshio");pytest.importorskip("gmsh")
    from hornlab_mesher.config_builder import build_from_config
    from hornlab_mesher.normals import validate_orientation,MeshOrientationError
    c=config(178);c.update(scale=2);c["profile"].update(L_mm=40,r0_mm=15,a_deg=60,a0_deg=0,k=.5)
    c["terminating_arc"]["join_t"]=.005;c["mesh"].update(vertical_offset_mm=3,scale_to_metres=False)
    result=build_from_config(c,tmp_path/"return.msh")
    mesh=meshio.read(result.mesh_path);p=np.asarray(mesh.points);tri=mesh.get_cells_type("triangle")
    tags=mesh.get_cell_data("gmsh:physical","triangle");surface=mesh.get_cell_data("gmsh:geometrical","triangle")
    body_tags=result.metadata["terminatingArcBodySurfaceTags"]
    body=np.isin(surface,body_tags)
    report=validate_orientation(p,tri,tags,require_open_shell_bore_normal=True,open_shell_bore_wall_mask=body)
    assert report.open_shell_bore_alignment==1
    reversed_tri=tri.copy();reversed_tri[body]=reversed_tri[body][:,::-1]
    with pytest.raises(MeshOrientationError,match="face the bore"):
        validate_orientation(p,reversed_tri,tags,require_positive_volume=False,require_open_shell_bore_normal=True,
            open_shell_bore_wall_mask=body)


@pytest.mark.parametrize("throat_res",[5.,.15])
def test_locally_sized_circle_certifies_practical_expanding_arc(tmp_path,throat_res):
    import collections
    gmsh=pytest.importorskip("gmsh");meshio=pytest.importorskip("meshio")
    from hornlab_mesher.config_builder import build_from_config
    c=config(75);c["scale"]=.1;c["terminating_arc"]["join_t"]=.237
    c["mesh"].update(throat_res_mm=throat_res,mouth_res_mm=25,rear_res_mm=15,scale_to_metres=False)
    m=model(c)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("Mesh.MinimumCirclePoints",23)
        result=build_from_config(c,tmp_path/"expanding.msh")
        assert gmsh.option.getNumber("Mesh.MinimumCirclePoints")==23
    finally:gmsh.finalize()
    assert result.metadata["terminatingArcMeshChordBoundMm"]<=.01
    mesh=meshio.read(result.mesh_path);p=np.asarray(mesh.points);tri=mesh.get_cells_type("triangle")
    tags=mesh.get_cell_data("gmsh:physical","triangle")
    source=tri[tags==2];wall=tri[tags==1]
    area=np.linalg.norm(np.cross(p[source[:,1]]-p[source[:,0]],p[source[:,2]]-p[source[:,0]]),axis=1).sum()/2
    assert area/(math.pi*m.r0**2)>.96
    def edges(faces):
        return collections.Counter(tuple(sorted((int(a),int(b)))) for face in faces for a,b in
            ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])))
    source_edges=edges(source);wall_edges=edges(wall)
    rim=[edge for edge,n in source_edges.items() if n==1]
    assert rim and all(wall_edges[edge]==1 for edge in rim)
    rim_nodes=np.unique(np.asarray(rim))
    assert len(rim_nodes)>=16
    if throat_res<1:assert len(rim_nodes)>16
    np.testing.assert_allclose(np.hypot(p[rim_nodes,0],p[rim_nodes,1]-m.offset),m.r0,rtol=0,atol=1e-10)
    np.testing.assert_allclose(p[rim_nodes,2],0,rtol=0,atol=1e-10)
    angle=np.sort(np.mod(np.arctan2(p[rim_nodes,1]-m.offset,p[rim_nodes,0]),2*math.pi))
    gap=np.diff(np.r_[angle,angle[0]+2*math.pi])
    assert gap.max()<=math.pi/8+1e-10


@pytest.mark.parametrize("name,value",[("include_inner","false"),("include_source_cap",1),
    ("include_curvature",None),("max_vertices",10**400)])
def test_native_preview_direct_flags_and_resource_ceiling_are_strict(name,value):
    with pytest.raises(ValueError):build_preview_geometry(config(32),PreviewOptionsV1(**{name:value}))


@pytest.mark.parametrize("value",["false",0,1,None])
def test_native_step_open_throat_flag_requires_boolean(value,tmp_path):
    from hornlab_mesher.cad import write_step
    from hornlab_mesher.mesher import MesherError
    with pytest.raises(MesherError,match="open_throat must be a boolean"):
        write_step(resolve_geometry(config(32)).geometry,tmp_path/"bad.step",open_throat=value)


def test_native_class_cannot_clear_its_exact_authority():
    from dataclasses import replace
    geometry=resolve_geometry(config(32)).geometry
    with pytest.raises(ValueError,match="requires its immutable"):
        replace(geometry,arc_meridian=None)


def test_native_class_cannot_override_source_angle(monkeypatch):
    from dataclasses import replace
    from hornlab_mesher.builders import terminating_arc
    geometry=resolve_geometry(config(32)).geometry
    bad=replace(geometry,source_auto_angle_deg=1.)
    def forbidden():raise AssertionError("OCC was reached")
    monkeypatch.setattr(terminating_arc,"require_gmsh",forbidden)
    with pytest.raises(ValueError,match="canonical placement"):
        terminating_arc.build_arc(bad)
