"""Actual nonlinear horn attachment and independent meridian/topology evidence."""

from copy import deepcopy
from dataclasses import replace
import json

import meshio
import numpy as np
import pytest
from scipy.interpolate import make_interp_spline, BSpline
from scipy.optimize import minimize_scalar

from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.general_horn import GeneralHornWall, FEATURE
from hornlab_mesher.source_assembly import SourceAssembly, assembly_channels
from hornlab_mesher.source_contour import ContourDrive, flat
from hornlab_mesher.phase_plug import PhasePlug, validate_passages
from hornlab_mesher.assembly_artifact import export_assembly


def config(family):
    base = {"formula": family, "mode": "bare", "profile": {}, "mesh": {
        "angular_segments": 64, "length_segments": 32,
        "throat_res_mm": 2, "mouth_res_mm": 3, "rear_res_mm": 3}}
    if family == "OSSE":
        base["profile"] = {"L_mm": 24, "r0_mm": 4, "a_deg": 32,
            "a0_deg": 6, "k": 1.25, "s": 0.7, "n": 4, "q": 0.995}
    elif family == "ROSSE":
        base["profile"] = {"R_mm": 22, "r0_mm": 4, "a_deg": 40,
            "a0_deg": 6, "k": 1, "q": 1, "tmax": 1}
    elif family == "ICW":
        base["profile"] = {"r0_mm": 4, "a0_deg": 18,
            "termination": "flat_baffle", "L_mm": 24, "R_mm": 19}
    else:
        profile = {"points": [[0, 4], [12, 9], [24, 18]],
                   "throatAngleDeg": 6, "mouthAngleDeg": 50}
        base["profile"] = {"profileH": profile, "profileV": deepcopy(profile),
            "crossSections": [{"t": 0, "shape": "circle"}, {"t": 1, "shape": "circle"}]}
    return base


def case(family, plugs=True, woofer=False):
    hf = replace(flat(4), physical_source_id="hf/source", rim_id="hf.rim")
    lf = replace(flat(5), physical_source_id="lf", rim_id="lf.rim") if woofer else None
    model = SourceAssembly.attach(hf, config(family), width_mm=110, height_mm=100,
        depth_mm=80, front_z_mm=7, horn_xy_mm=(-12, 10), woofer=lf,
        woofer_xy_mm=(25, -20) if woofer else (0, 0),
        aperture_radius_mm=7 if woofer else 0,
        phase_plugs=(PhasePlug("core", 2, 6, 0, 1, 0, 1.4),
                     PhasePlug("vane", 2, 6, 2, 2.5, 2.4, 2.9)) if plugs else ())
    drives = [ContourDrive("hf", (("piston", -0.5),), "axial")]
    if lf:
        drives.append(ContourDrive("lf", (("piston", 0),), "normal"))
    return model, drives


@pytest.mark.parametrize("family", ["OSSE", "ROSSE", "ICW", "FREEFORM"])
def test_shared_resolved_axial_fit_and_roundtrip(family):
    model, drives = case(family)
    g = resolve_geometry(config(family)).geometry
    xyz = g.inner_points
    meridian = np.column_stack((np.linalg.norm(xyz[0, :, :2], axis=-1),
                               xyz[0, :, 2] - xyz[0, 0, 2]))
    # Independently reconstruct original cardinal axial fit using the actual
    # resolved station chord lengths; do not call the new wall's fit helper.
    lengths = np.linalg.norm(np.diff(xyz, axis=1), axis=-1).mean(axis=0)
    params = np.r_[0, np.cumsum(lengths)] / lengths.sum()
    if g.surface_fit == "interpolate":
        oracle = make_interp_spline(params, meridian, k=min(3, len(meridian) - 1))
    else:
        degree = min(3, len(meridian) - 1)
        knots = np.r_[np.zeros(degree), np.linspace(0, 1, len(meridian) - degree + 1), np.ones(degree)]
        oracle = BSpline(knots, meridian, degree)
    samples = np.linspace(0, 1, 1001)
    np.testing.assert_allclose(model.horn_wall.evaluate(samples), oracle(samples), atol=1e-8, rtol=0)
    midpoint = model.horn_wall.evaluate(.5)
    a, b = model.horn_wall.evaluate([0, 1])
    direction, offset = b - a, midpoint - a
    assert abs(direction[0] * offset[1] - direction[1] * offset[0]) / np.linalg.norm(direction) > .1
    restored = SourceAssembly.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored == model
    assert restored.geometry_sha256 == model.geometry_sha256
    assert len(assembly_channels(model, drives)) == 1
    preview = model.preview(radial_steps=64, azimuth_steps=48)["horn-wall"]
    assert model.rigid_distance("horn-wall", preview).max() < 1e-6
    assert min(validate_passages(model).values()) > .1


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    root = tmp_path_factory.mktemp("general-horn")
    result = []
    for family in ["OSSE", "ROSSE", "ICW", "FREEFORM"]:
        model, drives = case(family, woofer=family == "ICW")
        directory = root / family
        manifest = export_assembly(model, drives, directory, mesh_size_mm=3)
        result.append((model, directory, manifest))
    return result


def test_actual_mesh_nonlinear_wall_source_ids_and_full_facet_clearance(bundles):
    for model, directory, manifest in bundles:
        assert FEATURE in manifest["required_features"]
        assert manifest["shared_join_edge_counts"] == [1] * (len(model.parts) * 2)
        assert manifest["passage_quality"]["certified_clearance_mm"] > 0
        mesh = meshio.read(directory / "preview.msh")
        triangles = mesh.get_cells_type("triangle")
        tags = mesh.get_cell_data("gmsh:physical", "triangle")
        xyz = mesh.points[triangles]
        assert len(manifest["channels"]) == len(model.parts)
        assert manifest["channels"][0]["patch_weights"] == {"hf%2Fsource/piston": -.5}
        local = xyz - model.parts[0][1]
        for patch in manifest["patches"]:
            points = xyz[tags == patch["mesh_tag"]]
            assert model.patch_distance(patch["id"], points).max() < 1e-6
        # Find actual wall facets by finite vertices then independently compare
        # centroid/edge midpoint radius at Z with the saved spline equations.
        on_wall = model.horn_wall.distance(xyz, model.parts[0][1]).max(axis=1) < 1e-6
        assert on_wall.any()
        facets = local[on_wall]
        midpoints = np.concatenate([facets.mean(axis=1), ((facets + np.roll(facets, 1, axis=1)) / 2).reshape(-1, 3)])
        spline = BSpline(model.horn_wall.knots, model.horn_wall.poles_mm, model.horn_wall.degree)
        for point in midpoints[::max(1, len(midpoints) // 60)]:
            rz = [np.hypot(*point[:2]), point[2]]
            distance = min(minimize_scalar(lambda t: np.linalg.norm(spline(t) - rz),
                bounds=(a, b), method="bounded", options={"xatol": 1e-12}).fun
                for a, b in zip(np.unique(spline.t), np.unique(spline.t)[1:]))
            assert distance < manifest["passage_quality"]["surface_tolerance_mm"]


def test_reopened_step_exact_wall_and_shell_inventory(bundles):
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        for model, directory, manifest in bundles:
            gmsh.clear()
            gmsh.model.occ.importShapes(str(directory / "geometry.step"), highestDimOnly=False)
            gmsh.model.occ.synchronize()
            assert not gmsh.model.getEntities(3)
            faces = [tag for _, tag in gmsh.model.getEntities(2)]
            matches = []
            for tag in faces:
                lo, hi = gmsh.model.getParametrizationBounds(2, tag)
                point = np.asarray(gmsh.model.getValue(2, tag, (np.asarray(lo) + hi) / 2))
                if model.horn_wall.distance(point, model.parts[0][1]) < 1e-6:
                    matches.append(tag)
            assert len(matches) == 1
            face = matches[0]
            from hornlab_mesher.general_horn import wall_face_area
            assert wall_face_area(gmsh, face, model.horn_wall) == pytest.approx(model.horn_wall.area_mm2, rel=1e-9)
            lo, hi = gmsh.model.getParametrizationBounds(2, face)
            uv = np.asarray([(u, v) for u in np.linspace(lo[0], hi[0], 11) for v in np.linspace(lo[1], hi[1], 17)])
            xyz = np.asarray(gmsh.model.getValue(2, face, uv.reshape(-1))).reshape(-1, 3)
            assert model.horn_wall.distance(xyz, model.parts[0][1]).max() < 1e-6
            assert (directory / "geometry.step").read_text().count("SHELL_BASED_SURFACE_MODEL(") == 3
    finally:
        gmsh.finalize()


@pytest.mark.parametrize("kind", ["noncircular", "reduced", "wall", "rim", "rollback", "wrong-version", "datum"])
def test_incompatible_composition_is_explicit(kind):
    c = config("OSSE")
    if kind == "noncircular":
        c["cross_section"] = {"exponent": 2, "aspect_ratio": 1.2}
    elif kind == "reduced":
        c["mesh"]["quadrants"] = "1"
    elif kind == "wall":
        c["mode"] = "freestanding"
        c["mesh"]["wall_thickness_mm"] = 1
    if kind in ("noncircular", "reduced", "wall"):
        with pytest.raises(ValueError, match="requires"):
            GeneralHornWall.from_config(c)
    else:
        model, _ = case("OSSE")
        if kind == "rim":
            with pytest.raises(ValueError, match="source rim"):
                replace(model, horn=replace(flat(3), physical_source_id="hf", rim_id="hf.rim"))
        elif kind == "rollback":
            with pytest.raises(ValueError, match="simple"):
                GeneralHornWall(1, (0, 0, .3, .6, 1, 1), ((4, 0), (5, 4), (3, 4), (4, 0)), "ROSSE")
        elif kind == "wrong-version":
            with pytest.raises(ValueError, match="version"):
                GeneralHornWall.from_dict({**model.horn_wall.to_dict(), "version": 2})
        else:
            with pytest.raises(ValueError, match="datums"):
                replace(model, mouth_radius_mm=model.mouth_radius_mm + 1)


def test_general_wall_clearance_checks_interior_bend_not_endpoints():
    # Both endpoints clear the linear vane by 2 mm, but the actual spline
    # bows to it halfway. A conical endpoint check would admit intersection.
    wall = GeneralHornWall(3, (0, 0, 0, 0, 1, 1, 1, 1),
        ((4, 0), (1.5, 4), (1.5, 8), (7, 12)), "FREEFORM")
    body = PhasePlug("unsafe", .5, 11.5, 0, 2.5, 0, 2.5)
    with pytest.raises(ValueError, match="outside"):
        wall.minimum_body_clearance(body)


@pytest.mark.parametrize("family", ["OSSE", "ROSSE", "ICW", "FREEFORM"])
def test_existing_tensor_wall_circle_correction_is_bounded(family):
    import gmsh
    from hornlab_mesher.builders.point_grid_surfaces import _add_occ_bspline_patch_wall_surfaces
    model, _ = case(family, plugs=False)
    resolved = resolve_geometry(config(family)).geometry
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("original-general-horn-wall")
        faces = _add_occ_bspline_patch_wall_surfaces(resolved.inner_points,
            closed=True, surface_fit=resolved.surface_fit)
        gmsh.model.occ.synchronize()
        maximum = 0.
        for _, face in faces:
            low, high = gmsh.model.getParametrizationBounds(2, face)
            for t in np.linspace(0, 1, 101):
                uv = np.asarray([(u, low[1] + t * (high[1] - low[1])) for u in np.linspace(low[0], high[0], 37)])
                xyz = np.asarray(gmsh.model.getValue(2, face, uv.reshape(-1))).reshape(-1, 3)
                rz = np.column_stack((np.linalg.norm(xyz[:, :2], axis=1), xyz[:, 2] - resolved.inner_points[0, 0, 2]))
                expected = model.horn_wall.evaluate(t)
                maximum = max(maximum, float(np.linalg.norm(rz - expected, axis=1).max()))
        print(f"{family}: original angular tensor wall to exact circle correction {maximum:.12f} mm")
        # Fixed geometric tolerance, independent of mesh density. Cardinal axial
        # profile remains exactly the existing axial fit; circles are exact.
        assert maximum <= model.horn_wall.circle_correction_bound_mm + 1e-8
        assert model.horn_wall.circle_correction_bound_mm <= .15
    finally:
        gmsh.finalize()


def test_source_rim_is_actual_first_ring_with_throat_extension():
    c = config("OSSE")
    c["profile"].update(throat_ext_length_mm=5, throat_ext_angle_deg=10)
    resolved = resolve_geometry(c).geometry.inner_points
    radius = float(np.linalg.norm(resolved[0, 0, :2]))
    source = replace(flat(radius), physical_source_id="hf", rim_id="rim")
    model = SourceAssembly.attach(source, c, width_mm=110, height_mm=100, depth_mm=80)
    assert model.horn_wall.throat_radius_mm == pytest.approx(radius, abs=1e-9)
    with pytest.raises(ValueError, match="source rim"):
        replace(model, horn=replace(flat(radius + 1), physical_source_id="hf", rim_id="rim"))


def test_rotational_circle_morph_attaches_without_key_based_refusal():
    c = config("OSSE")
    c["morph"] = {"morphTarget": 2, "morphWidth": 60, "morphHeight": 60}
    wall = GeneralHornWall.from_config(c)
    assert wall.mouth_radius_mm == 30


def test_large_approximate_circle_correction_is_refused():
    c = config("FREEFORM")
    for name in ("profileH", "profileV"):
        c["profile"][name]["points"] = [[z * 10, r * 10] for z, r in c["profile"][name]["points"]]
    c["mesh"].update(angular_segments=16, throat_res_mm=20, mouth_res_mm=30, rear_res_mm=30)
    with pytest.raises(ValueError, match="circle fit correction"):
        GeneralHornWall.from_config(c)


def test_discontinuous_recipe_cannot_pass_simple_meridian_admission():
    with pytest.raises(ValueError, match="continuity"):
        GeneralHornWall(1, (0, 0, .5, .5, 1, 1), ((4, 0), (8, 10), (5, 0), (9, 10)), "OSSE")


def test_projection_solver_tolerance_is_not_a_geometry_certificate(monkeypatch):
    from types import SimpleNamespace
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "linprog", lambda *a, **k: SimpleNamespace(success=True, x=np.asarray([0., 0., 1e-5])))
    with pytest.raises(ValueError, match="simple"):
        GeneralHornWall(1, (0, 0, .3, .6, 1, 1), ((4, 0), (5, 4), (3, 4), (4, 0)), "ROSSE")


def test_c0_cubic_span_certificate_uses_its_own_left_derivative():
    wall = GeneralHornWall(3, (0, 0, 0, 0, .5, .5, .5, 1, 1, 1, 1),
        ((4, 0), (5, 1), (6, 2), (7, 3), (12, 4), (15, 6), (20, 10)), "FREEFORM")
    np.testing.assert_array_equal(wall.beziers[0], ((4, 0), (5, 1), (6, 2), (7, 3)))
    assert wall.distance(np.asarray([[5.5, 0, 1.5]]), (0, 0, 0))[0] <= 2e-7
