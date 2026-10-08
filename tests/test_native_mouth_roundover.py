"""Native circular lip contracts and fail-closed transport boundaries."""
import copy
import math

import numpy as np
import pytest

from hornlab_mesher.config_builder import build_geometry_params, resolve_geometry
from hornlab_mesher.config_parser import ConfigError
from hornlab_mesher.mouth_roundover import Roundover, FIT_TOL_MM
from hornlab_mesher.preview import build_preview_geometry, PreviewOptionsV1


def config():
    return {"formula":"OSSE-ROUNDOVER", "mode":"freestanding",
            "profile":{"L_mm":120,"r0_mm":12.7,"a_deg":45,"a0_deg":10,"k":1,"s":0},
            "mesh":{"wall_thickness_mm":5,"mouth_roundover_radius_mm":25,"scale_to_metres":False}}


def model(c=None):
    return Roundover.from_params(build_geometry_params(c or config())[0])


@pytest.mark.parametrize("radius",[-1,False,True,float("nan"),float("inf"),"25",{},10**400])
def test_radius_refused(radius):
    c = config()
    c["mesh"]["mouth_roundover_radius_mm"] = radius
    with pytest.raises(ConfigError,match="Mouth roundover refused"):
        build_geometry_params(c)


@pytest.mark.parametrize("section,key,value",[
    ("profile","s",0.3),("profile","a_deg",80),("profile","a0_deg",45),
    ("profile","r0_mm",0),("profile","k",20),("profile","L_mm",3000),
    ("profile","r0_mm","12.7"),("profile","s1",1),
    ("profile","slot_length_mm",1),("profile","throat_ext_length_mm",1),
    ("mesh","wall_thickness_mm",25),("mesh","wall_thickness_mm",0),
    ("mesh","quadrants","1"),("mesh","surface_fit","approximate"),
    ("mesh","topology_mode","legacy"),("source","source_shape",1),
    ("source","source_radius_mm",40),("source","source_curv",1),
    ("root","mode","bare"),("root","mode","infinite-baffle"),
    ("root","scale",-1),("root","scale",11),("root","vertical_offset_mm",10001),
    ("root","enclosure",{"depth_mm":50}),("root","cross_section",{"aspect_ratio":2}),
    ("root","morph",{}),("root","gcurve",{}),("root","throat_adapter",{}),
])
def test_unsupported_combinations_refuse(section,key,value):
    c = config()
    (c if section == "root" else c.setdefault(section,{}))[key] = value
    with pytest.raises((ConfigError,ValueError)):
        build_geometry_params(c)


@pytest.mark.parametrize("section",["profile","mesh","source"])
def test_malformed_section_refused(section):
    c = config()
    c[section] = "bad"
    with pytest.raises(ConfigError):
        build_geometry_params(c)


def test_marker_old_reader_refuses_and_plain_formula_cannot_drop_lip():
    from hornlab_mesher.profile_common import _normalise_formula
    with pytest.raises(ValueError):
        _normalise_formula("OSSE-ROUNDOVER")
    c = config()
    c["formula"] = "OSSE"
    with pytest.raises(ConfigError,match="requires formula"):
        build_geometry_params(c)
    c["formula"] = "OSSE-ROUNDOVER"
    c["mesh"]["mouth_roundover_radius_mm"] = 0
    with pytest.raises(ConfigError,match="positive"):
        build_geometry_params(c)


def test_absent_and_off_have_identical_params_and_geometry():
    c = config()
    c["formula"] = "OSSE"
    c["mesh"].pop("mouth_roundover_radius_mm")
    off = copy.deepcopy(c)
    off["mesh"]["mouth_roundover_radius_mm"] = 0
    assert build_geometry_params(c) == build_geometry_params(off)
    a,b = resolve_geometry(c).geometry,resolve_geometry(off).geometry
    np.testing.assert_array_equal(a.inner_points,b.inner_points)
    np.testing.assert_array_equal(a.outer_points,b.outer_points)


@pytest.mark.parametrize("angle,radius,wall,scale,k",[(20,10,3,1,1),(45,25,5,1,1),(65,40,10,2,10)])
def test_canonical_join_circle_envelope_and_finished_scale(angle,radius,wall,scale,k):
    c = config()
    c.update(scale=scale,vertical_offset_mm=7)
    c["profile"].update(a_deg=angle,k=k)
    c["mesh"].update(mouth_roundover_radius_mm=radius,wall_thickness_mm=wall)
    m = model(c)
    mouth,tangent = m.body(m.length)
    beta = math.atan2(tangent[1],tangent[0])
    np.testing.assert_allclose(m.arc(0),mouth,atol=1e-12)
    np.testing.assert_allclose(m.arc(0,True),m.body(m.length,True)[0],atol=1e-12)
    phi = np.linspace(0,m.sweep,101)
    for outer,r in [(False,radius),(True,radius-wall)]:
        np.testing.assert_allclose(np.linalg.norm(m.arc(phi,outer)-m.center,axis=1),r,atol=1e-12)
    assert m.bounds[1][2] == pytest.approx(m.length+radius*(1-math.sin(beta)))
    assert m.arc(m.sweep)[0] == pytest.approx(m.center[0])
    assert m.arc(m.sweep)[1] == pytest.approx(m.center[1]+radius)
    assert m.radius == radius
    assert m.length == 120*scale


@pytest.mark.parametrize("outer",[False,True])
def test_complete_body_hermite_certificate_and_dense_independent_remainder(outer):
    m = model()
    stations = m.fit_stations(outer)
    assert len(stations) < 100
    for lo,hi in zip(stations[:-1],stations[1:]):
        bound = m.fourth_bound(lo,hi,outer)*(hi-lo)**4/384
        assert bound <= FIT_TOL_MM
        p,d = m.body(np.array([lo,hi]),outer)
        t = np.linspace(0,1,31)[:,None]
        h = hi-lo
        hermite = (2*t**3-3*t**2+1)*p[0]+(t**3-2*t**2+t)*h*d[0]+(-2*t**3+3*t**2)*p[1]+(t**3-t**2)*h*d[1]
        expected = m.body(lo+t[:,0]*h,outer)[0]
        assert np.max(np.linalg.norm(hermite-expected,axis=1)) <= bound+1e-11


@pytest.mark.parametrize("lod",["coarse","fine","inspection"])
def test_preview_bound_normals_unique_roles_and_true_dimensions(lod):
    c = config()
    p = build_preview_geometry(c,PreviewOptionsV1(lod=lod))
    m = model(c)
    assert p.metadata["mouthRoundoverChordBoundMm"] <= 0.008
    assert p.metadata["dimensions_status"] == "current"
    np.testing.assert_allclose(p.metadata["dimensions_mm"]["horn_overall"],np.ptp(m.bounds,axis=0))
    roles = [s.role for s in p.surfaces]
    assert len(roles) == len(set(roles))
    rim = next(s for s in p.surfaces if s.role == "mouth_rim")
    np.testing.assert_array_equal(rim.normals,np.tile([0,0,-1],(len(rim.positions),1)))
    inner = next(s for s in p.surfaces if s.role == "horn.inner")
    assert np.min(abs(inner.positions[:,2]-m.length)) < 1e-12
    assert np.max(inner.positions[:,2]) == pytest.approx(m.bounds[1][2])
    for surface in p.surfaces:
        xyz = surface.positions[surface.indices.reshape(-1,3)]
        n = surface.normals[surface.indices.reshape(-1,3)].mean(axis=1)
        assert np.min(np.einsum('ij,ij->i',np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),n)) >= -1e-9


@pytest.mark.parametrize("option",[{"max_vertices":8},{"max_chord_error_mm":1e-10},{"max_normal_step_deg":0}])
def test_unachievable_preview_explicitly_refuses(option):
    with pytest.raises(ValueError):
        build_preview_geometry(config(),PreviewOptionsV1(**option))


def test_vertex_budget_refuses_before_surface_outer_product(monkeypatch):
    from hornlab_mesher.preview import mouth_roundover
    def forbidden(*args,**kwargs):
        raise AssertionError("surface allocation reached")
    monkeypatch.setattr(mouth_roundover.np,"broadcast_to",forbidden)
    with pytest.raises(ValueError,match="vertex budget"):
        build_preview_geometry(config(),PreviewOptionsV1(max_vertices=8,min_silhouette_segments=4096))


def test_canonical_mesh_placement_and_root_alias_agree():
    a,b = config(),config()
    a["vertical_offset_mm"] = 7
    b["mesh"]["vertical_offset_mm"] = 7
    assert model(a).fingerprint == model(b).fingerprint
    np.testing.assert_array_equal(resolve_geometry(a).geometry.inner_points,resolve_geometry(b).geometry.inner_points)


def test_sampling_counts_do_not_change_identity_or_datums():
    a,b = config(),config()
    b["mesh"].update(length_segments=80,angular_segments=128)
    assert model(a).fingerprint == model(b).fingerprint
    from hornlab_mesher.datums import derive_datums
    from hornlab_mesher.geometry import BuiltGeometry
    g = resolve_geometry(a).geometry
    built = BuiltGeometry(surface_groups={},axial_bounds_mm=(-5,model(a).bounds[1][2]),source_axis="z")
    datums = derive_datums(g,built)
    assert datums["WG_MOUTH_PLANE"]["origin_mm"][2] == pytest.approx(model(a).length)


@pytest.mark.parametrize("entry",[resolve_geometry,build_preview_geometry])
@pytest.mark.parametrize("patch",[
    {"profile":{"formula":"R-OSSE"}}, {"type":"ICW"},
    {"profile":{"type":"FREEFORM"}},
    {"quadrants":1,"mesh":{"quadrants":"1234"}},
    {"vertical_offset_mm":20,"mesh":{"vertical_offset_mm":7}},
    {"quadrants":"1234garbage"}, {"mesh":{"quadrants":"1234garbage"}},
])
def test_conflicting_or_malformed_supplied_intent_refused(entry,patch):
    c = config()
    for key,value in patch.items():
        if isinstance(value,dict):
            c.setdefault(key,{}).update(value)
        else:
            c[key] = value
    with pytest.raises(ConfigError,match="Native geometry refused"):
        entry(c)


@pytest.mark.parametrize("key",["length_segments","angular_segments"])
@pytest.mark.parametrize("value",[-1,0,1.5,True,"32",None,float("nan"),float("inf"),[],{}])
def test_invalid_supplied_sampling_counts_refused(key,value):
    c = config()
    c["mesh"][key] = value
    with pytest.raises(ConfigError,match="Native geometry refused"):
        resolve_geometry(c)


def test_matching_discriminators_coverage_and_placement_preserve_identity():
    a,b = config(),config()
    a["vertical_offset_mm"] = 7
    b.update(type="osse-roundover",quadrants=1234,vertical_offset_mm=7)
    b["profile"].update(formula="OSSE-ROUNDOVER",type="OSSE-ROUNDOVER")
    b["mesh"].update(quadrants="1234",vertical_offset_mm=7,length_segments=80,angular_segments=128)
    assert model(a).fingerprint == model(b).fingerprint
    np.testing.assert_array_equal(resolve_geometry(a).geometry.inner_points,resolve_geometry(b).geometry.inner_points)


@pytest.mark.parametrize("key",["allow_large_mesh","scale_to_metres"])
@pytest.mark.parametrize("value",["not-approved","millimetres","true",[0],{},1,0,None])
@pytest.mark.parametrize("entry",[resolve_geometry,build_preview_geometry])
def test_active_boolean_controls_require_actual_booleans(key,value,entry):
    c = config()
    c["mesh"][key] = value
    with pytest.raises(ConfigError,match=f"{key} must be a boolean"):
        entry(c)


@pytest.mark.parametrize("value",["not-approved",[0],{},1,0])
def test_active_large_mesh_override_requires_boolean(value,tmp_path):
    from hornlab_mesher.config_builder import build_from_config
    with pytest.raises(ConfigError,match="override must be a boolean"):
        resolve_geometry(config(),allow_large_mesh=value)
    with pytest.raises(ConfigError,match="override must be a boolean"):
        build_from_config(config(),tmp_path/"refused.msh",allow_large_mesh=value)


@pytest.mark.parametrize("value",[False,True])
def test_active_boolean_controls_and_override_retain_explicit_values(value):
    c = config()
    c["mesh"].update(allow_large_mesh=value,scale_to_metres=value)
    resolved = resolve_geometry(c)
    assert resolved.density.allow_large_mesh is value
    assert resolved.scale_to_metres is value
    assert resolve_geometry(c,allow_large_mesh=not value).density.allow_large_mesh is not value


def test_body_only_consumers_refuse_active_geometry():
    from hornlab_mesher.profile_formulas import calculate_osse,calculate_osse_curve,osse_total_length
    from hornlab_mesher.profile_sampling import build_point_grid_arrays
    p = build_geometry_params(config())[0]
    for fn,args in [(calculate_osse,(0,0,p)),(calculate_osse_curve,([0,1],0,p)),(build_point_grid_arrays,(p,)),(osse_total_length,(p,))]:
        with pytest.raises(ValueError,match="mouth roundover requires"):
            fn(*args)


def test_real_step_solid_recipe_identity_and_bundle_refusal(tmp_path):
    pytest.importorskip("gmsh")
    from hornlab_mesher.cad import write_step_from_config,write_wglink
    from hornlab_mesher.mesher import MesherError
    c = config()
    path,info = write_step_from_config(c,tmp_path/"lip.step")
    assert info.body == "solid" and info.volume_mm3 > 0 and info.throat_opened
    assert info.mouth_roundover["fingerprint"] == model(c).fingerprint
    np.testing.assert_allclose(info.bounding_box_mm,model(c).bounds)
    assert "hornlab-mouth-roundover" in path.read_text()
    with pytest.raises(MesherError,match="qualified bundle transport"):
        write_wglink(resolve_geometry(c).geometry,tmp_path/"lip.wglink")


@pytest.mark.parametrize("allow_large",[False,True])
def test_lip_budget_estimate_runs_before_mesh_generation(monkeypatch,tmp_path,allow_large):
    gmsh = pytest.importorskip("gmsh")
    from hornlab_mesher.config_builder import build_from_config
    from hornlab_mesher.mesher import MesherError,TriangleBudgetExceeded
    calls = []
    def generate(*args):
        calls.append(args)
        raise RuntimeError("generation boundary reached")
    monkeypatch.setattr(gmsh.model.mesh,"generate",generate)
    c = config()
    c["mesh"]["allow_large_mesh"] = allow_large
    if allow_large:
        with pytest.raises(MesherError,match="generation boundary reached"):
            build_from_config(c,tmp_path/"lip.msh")
        assert len(calls) == 1
    else:
        with pytest.raises(TriangleBudgetExceeded,match="circular mouth lip"):
            build_from_config(c,tmp_path/"lip.msh")
        assert not calls


@pytest.mark.parametrize("location", ["root", "profile"])
@pytest.mark.parametrize("inactive", [None, {"mode": "off"}, {"mode": "off", "length_mm": "editor draft"}])
def test_inactive_adapter_preserves_roundover_identity(location, inactive):
    c = config()
    (c if location == "root" else c["profile"])["throat_adapter"] = inactive
    assert build_geometry_params(c) == build_geometry_params(config())
    assert resolve_geometry(c).geometry.roundover == model()


@pytest.mark.parametrize("payload", [{}, "off", {"mode": "off", "unknown": 1}, {"mode": "authored"}])
def test_malformed_adapter_cannot_hide_behind_roundover(payload):
    c = config()
    c["throat_adapter"] = payload
    with pytest.raises(ConfigError, match="roundover refused"):
        build_geometry_params(c)


def test_duplicate_inactive_adapter_refused():
    c = config()
    c["throat_adapter"] = c["profile"]["throat_adapter"] = {"mode": "off"}
    with pytest.raises(ConfigError, match="only one location"):
        build_geometry_params(c)


def test_active_adapter_and_roundover_refused_at_config_and_direct_boundaries(tmp_path):
    from types import SimpleNamespace
    from hornlab_mesher.builders.point_grid_dispatch import build_point_grid
    from hornlab_mesher.cad import write_step
    from hornlab_mesher.mesher import MesherError
    from hornlab_mesher.profile_formulas import calculate_osse, calculate_osse_curve, osse_total_length
    from hornlab_mesher.profile_sampling import build_point_grid_arrays
    c = config()
    c["throat_adapter"] = {"mode": "authored", "driver_exit_diameter_mm": 25.4,
        "exit_half_angle_deg": 10, "length_mm": 40, "join_t": .137,
        "driver_handle_mm": 10, "body_handle_mm": 15}
    with pytest.raises(ConfigError, match="combined active"):
        build_geometry_params(c)
    mixed = SimpleNamespace(roundover=object(), adapter_meridian=object())
    with pytest.raises(ValueError, match="combined active"):
        build_point_grid(mixed)
    output = tmp_path / "unsupported.step"
    with pytest.raises(MesherError, match="combined active"):
        write_step(mixed, output)
    assert not output.exists()
    # A body-only consumer must refuse the lip before taking an adapter shortcut.
    p = build_geometry_params(config())[0]
    p["throatAdapter"] = c["throat_adapter"]
    for fn, args in [(calculate_osse, (0, 0, p)), (calculate_osse_curve, ([0, 1], 0, p)),
                     (build_point_grid_arrays, (p,)), (osse_total_length, (p,))]:
        with pytest.raises(ValueError, match="mouth roundover requires"):
            fn(*args)


def test_zero_roundover_radius_preserves_active_adapter_identity():
    c = {"formula": "OSSE-ADAPTER", "mode": "bare",
         "profile": {"L": 120., "r0": 12.7, "a": 40., "a0": 15.5, "k": 1., "s": 0.},
         "source": {"source_shape": 0},
         "mesh": {"angular_segments": 32, "length_segments": 13},
         "throat_adapter": {"mode": "authored", "driver_exit_diameter_mm": 25.4,
             "exit_half_angle_deg": 10., "length_mm": 40., "join_t": .137,
             "driver_handle_mm": 10., "body_handle_mm": 15.}}
    off = copy.deepcopy(c)
    off["mesh"]["mouth_roundover_radius_mm"] = 0
    assert build_geometry_params(off) == build_geometry_params(c)
    a, b = resolve_geometry(off).geometry, resolve_geometry(c).geometry
    assert a.adapter_meridian.fingerprint == b.adapter_meridian.fingerprint
    np.testing.assert_array_equal(a.inner_points, b.inner_points)


def test_dense_native_source_and_final_lip_keep_shared_topology_and_certificate(tmp_path):
    """A supported .004 mm source must survive the legacy .005 mm weld boundary."""
    meshio = pytest.importorskip("meshio")
    pytest.importorskip("gmsh")
    from hornlab_mesher.config_builder import build_from_config
    from hornlab_mesher.tags import PhysicalGroup
    from hornlab_mesher.edges import build_edge_table
    c = {"formula": "OSSE-ROUNDOVER", "mode": "freestanding",
         "profile": {"L_mm": 1, "r0_mm": 1, "a_deg": 10, "a0_deg": 0, "k": 1, "s": 0},
         "mesh": {"wall_thickness_mm": .1, "mouth_roundover_radius_mm": .2,
                  "scale_to_metres": False, "allow_large_mesh": True,
                  "throat_res_mm": .004, "mouth_res_mm": .05, "rear_res_mm": .05}}
    m = model(c)
    result = build_from_config(c, tmp_path / "dense-source-lip.msh")
    mesh = meshio.read(result.mesh_path)
    points = np.asarray(mesh.points)
    triangles = mesh.get_cells_type("triangle")
    tags = mesh.get_cell_data("gmsh:physical", "triangle")
    edges = build_edge_table(triangles)
    assert np.all(edges.count == 2)  # source/body/lip/rear interfaces all sewn
    assert len(np.unique(triangles)) - edges.n_edges + len(triangles) == 2
    source = triangles[tags == int(PhysicalGroup.PRIMARY_SOURCE)]
    vertices = points[source]
    cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    assert np.all(cross[:, 2] > 0)
    np.testing.assert_allclose(vertices[:, :, 2], 0, atol=1e-12)
    assert .5 * cross[:, 2].sum() == pytest.approx(math.pi * m.r0**2, rel=.001)
    source_edges = build_edge_table(source)
    assert len(np.unique(source)) - source_edges.n_edges + len(source) == 1
    rim = source_edges.count == 1
    np.testing.assert_allclose(np.linalg.norm(points[source_edges.lo[rim], :2], axis=1),
                               m.r0, atol=1e-10, rtol=0)
    # Recompute the continuous torus bound from the emitted triangle vertices,
    # independently of the live Gmsh model used before postprocessing.
    vertices = points[triangles]
    radial = np.hypot(vertices[:, :, 0], vertices[:, :, 1])
    circle_distance = np.hypot(vertices[:, :, 2] - m.center[0], radial - m.center[1])
    largest = 0.
    for radius in (m.radius, m.radius - m.wall):
        on_lip = np.all(np.abs(circle_distance - radius) < 1e-9, axis=1)
        assert on_lip.sum() > 100
        v, r = vertices[on_lip], radial[on_lip]
        phi = np.arctan2(v[:, :, 2] - m.center[0], m.center[1] - r)
        az = np.arctan2(v[:, :, 1], v[:, :, 0])
        az = az[:, :1] + (az - az[:, :1] + math.pi) % (2 * math.pi) - math.pi
        dp, da = np.ptp(phi, axis=1), np.ptp(az, axis=1)
        residual = np.max(np.abs(circle_distance[on_lip] - radius), axis=1)
        bound = (radius * dp**2 + 2 * radius * dp * da + (m.center[1] + radius) * da**2) / 8 + residual
        largest = max(largest, float(np.max(bound)))
    assert largest <= .01
    assert largest == pytest.approx(result.metadata["mouthRoundoverMeshChordBoundMm"], abs=1e-12)


@pytest.mark.parametrize("root_key", ["formula", "type"])
@pytest.mark.parametrize("profile_key", ["formula", "type"])
@pytest.mark.parametrize("radius", [None, 0])
def test_hidden_lip_marker_cannot_fall_back_to_ordinary_body(root_key, profile_key, radius):
    c = config()
    c.pop("formula")
    c[root_key] = "OSSE"
    c["profile"][profile_key] = "osse-roundover"
    if radius is None:
        c["mesh"].pop("mouth_roundover_radius_mm")
    else:
        c["mesh"]["mouth_roundover_radius_mm"] = radius
    with pytest.raises(ConfigError, match="Native geometry refused"):
        build_geometry_params(c)


@pytest.mark.parametrize("location", ["root", "mesh", "both"])
@pytest.mark.parametrize("offset", [-7, 7])
def test_canonical_dimensions_preserve_supported_placement_aliases(location, offset):
    from hornlab_mesher.preview.dimensions import canonical_dimensions, dimension_metadata
    c = config()
    if location in {"root", "both"}:
        c["vertical_offset_mm"] = offset
    if location in {"mesh", "both"}:
        c["mesh"]["vertical_offset_mm"] = offset
    original = copy.deepcopy(c)
    assert canonical_dimensions(c) == canonical_dimensions(config())
    measured = dimension_metadata(c, "fine")
    assert measured["dimensions_status"] == "current"
    assert c == original


def test_native_lip_preview_preserves_metadata_schema_and_signed_curvature():
    p = build_preview_geometry(config())
    m = model()
    metadata = p.metadata
    for key in ("actual_segment_counts", "timings_ms", "requested_fidelity", "vertex_accounting",
                "canonical_reference", "angular_sampling", "semantic_stations", "nesting",
                "normal_convention", "normal_method_notes", "guiding_curve_probe"):
        assert key in metadata
    assert set(metadata["timings_ms"]) == {"canonical_sampling", "surface_assembly_and_fidelity", "total"}
    assert all(value >= 0 for value in metadata["timings_ms"].values())
    assert metadata["actual_segment_counts"]["horn_phi"] == metadata["canonical_reference"]["azimuth_rows"]
    assert metadata["requested_fidelity"]["max_chord_error_mm"] == .008
    assert metadata["requested_fidelity"]["max_vertices"] is None
    for surface in p.surfaces:
        assert metadata["vertex_accounting"][surface.role]["vertices"] == len(surface.positions)
        assert len(surface.curvature_mean) == len(surface.positions)
    rear = next(s for s in p.surfaces if s.role == "wall.rear_return")
    radius = float(m.body(0, True)[0][1])
    np.testing.assert_allclose(rear.curvature_mean, -.5/radius, atol=1e-12)
    np.testing.assert_allclose(rear.curvature_principal, -1/radius, atol=1e-12)
    for outer, role in ((False, "horn.inner"), (True, "horn.outer")):
        s = next(s for s in p.surfaces if s.role == role)
        rad = m.radius-m.wall if outer else m.radius
        terminal_radius = float(m.center[1]+rad)
        terminal = np.isclose(np.hypot(s.positions[:, 0], s.positions[:, 1]), terminal_radius, atol=1e-10, rtol=0)
        assert terminal.sum() == metadata["actual_segment_counts"]["horn_phi"]
        sign = 1 if outer else -1
        np.testing.assert_allclose(s.curvature_mean[terminal], sign*.5*(1/rad+1/terminal_radius), atol=1e-12)
        np.testing.assert_allclose(s.curvature_principal[terminal], sign/rad, atol=1e-12)


def test_lip_triangle_certificate_unwraps_terminal_pi_roundoff(monkeypatch):
    from types import SimpleNamespace
    from hornlab_mesher.builders import mouth_roundover
    m = model()
    theta = np.array([math.pi-0.001, math.pi+1e-13, math.pi-0.001])
    az = np.array([0., 0., .001])
    zr = m.arc(theta-m.beta)
    xyz = np.column_stack((zr[:, 1]*np.cos(az), zr[:, 1]*np.sin(az), zr[:, 0]))
    mesh = SimpleNamespace(getNodes=lambda: (np.array([1, 2, 3]), xyz.ravel(), []),
        getElements=lambda *_: ([2], [np.array([1])], [np.array([1, 2, 3])]))
    fake = SimpleNamespace(model=SimpleNamespace(mesh=mesh))
    monkeypatch.setattr(mouth_roundover, "require_gmsh", lambda: fake)
    built = SimpleNamespace(mesh_surface_groups={"lip_inner": [1], "lip_outer": []}, metadata={})
    mouth_roundover.certify_lip_mesh(m, built)
    assert 0 < built.metadata["mouthRoundoverMeshChordBoundMm"] < .001


def test_canonical_measurement_does_not_hide_conflicting_raw_placement():
    from hornlab_mesher.preview.dimensions import canonical_dimensions
    c = config()
    c["vertical_offset_mm"] = 7
    c["mesh"]["vertical_offset_mm"] = 9
    with pytest.raises(ConfigError, match="must agree"):
        canonical_dimensions(c)


def test_native_lip_first_parameter_read_uses_owned_snapshot(monkeypatch):
    import hornlab_mesher.preview.api as api
    c = config()
    expected = build_preview_geometry(c)
    real = api.build_geometry_params
    seen = []
    def mutate_caller_before_read(snapshot):
        seen.append(snapshot)
        c["profile"]["L_mm"] = 240
        return real(snapshot)
    monkeypatch.setattr(api, "build_geometry_params", mutate_caller_before_read)
    actual = build_preview_geometry(c)
    assert seen[0] is not c
    assert actual.metadata["dimensions_mm"] == expected.metadata["dimensions_mm"]
    assert actual.metadata["mouthRoundover"]["fingerprint"] == expected.metadata["mouthRoundover"]["fingerprint"]
    assert len(actual.surfaces) == len(expected.surfaces)
    for left, right in zip(actual.surfaces, expected.surfaces):
        assert left.role == right.role
        for field in ("positions", "normals", "indices", "curvature_mean", "curvature_principal"):
            np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
