"""Imported physical curves and native authoring keep distinct contracts."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from hornlab_mesher.config_builder import build_geometry_params, resolve_geometry
from hornlab_mesher.config_parser import parse_text_config
from hornlab_mesher.profile_formulas import calculate_osse, calculate_osse_curve
from hornlab_mesher.profile_sampling import build_point_grid_arrays
from hornlab_mesher.text_import import TEXT_IMPORT_VERSION_KEY, TEXT_IMPORT_VERSION

BASE = """OSSE = {
L = 137
r0 = 17
a = 37
a0 = 6
k = 1.25
s = .7
n = 4
q = .995
}
Mesh.AngularSegments = 32
Mesh.LengthSegments = 32
Mesh.WallThickness = 0
Source.Shape = 2
"""
FIXTURE = json.loads((Path(__file__).parent / "fixtures/text_import_geometry.json").read_text())


def params(text=BASE):
    return build_geometry_params(parse_text_config(text))[0]


def inner(config):
    return build_point_grid_arrays(build_geometry_params(config)[0])["inner_grid"]


def test_import_version_survives_copy_and_json_transport():
    imported = parse_text_config(BASE + "Slot.Length = 15\n")
    assert imported[TEXT_IMPORT_VERSION_KEY] == TEXT_IMPORT_VERSION
    for config in (imported, copy.deepcopy(imported), json.loads(json.dumps(imported))):
        assert build_geometry_params(config)[0][TEXT_IMPORT_VERSION_KEY] == TEXT_IMPORT_VERSION
        assert np.array_equal(inner(config), inner(imported))
    config = copy.deepcopy(imported)
    config[TEXT_IMPORT_VERSION_KEY] = "unknown-future-version"
    with pytest.raises(ValueError, match="unsupported text import geometry version"):
        build_geometry_params(config)


@pytest.mark.parametrize("case", FIXTURE["slot_cases"])
def test_imported_slot_matches_independent_physical_stations(case):
    p = params(case["text"])
    zr = np.asarray(case["points_zr_mm"])
    x, r = calculate_osse_curve(zr[:, 0], 0., p)
    assert np.allclose(x, zr[:, 0], rtol=0, atol=FIXTURE["tolerance_mm"])
    assert np.allclose(r, zr[:, 1], rtol=0, atol=FIXTURE["tolerance_mm"])
    scalar = np.asarray([calculate_osse(float(z), 0., p) for z in zr[:, 0]])
    assert np.allclose(scalar, np.column_stack((x, r)), rtol=0, atol=1e-12)


def test_imported_slot_join_is_continuous_with_the_throat_slope():
    p = params(BASE + "Slot.Length = 15\n")
    slope = math.tan(math.radians(6))
    rho = 17 + 15 * slope
    assert calculate_osse(15, 0., p)[1] == rho
    step = 1e-5
    left = (rho-calculate_osse(15-step, 0., p)[1])/step
    right = (calculate_osse(15+step, 0., p)[1]-rho)/step
    assert left == pytest.approx(slope, abs=1e-6)
    assert right == pytest.approx(slope, abs=1e-6)


def test_native_slot_remains_cylindrical_and_additive():
    imported = parse_text_config(BASE + "Slot.Length = 15\n")
    native = copy.deepcopy(imported)
    native.pop(TEXT_IMPORT_VERSION_KEY)
    native["profile"].pop("_athLengthMode")
    p = build_geometry_params(native)[0]
    assert calculate_osse(15, 0., p)[1] == 17
    assert calculate_osse(5, 0., p)[1] == 17
    assert inner(native)[0, -1, 2] == 152
    assert inner(imported)[0, -1, 2] == 137


@pytest.mark.parametrize("width,height", [(360, 0), (360,360), (410,310), (20,20)])
def test_imported_circle_ignores_dimensions_and_preserves_raw_radius(width, height):
    config = parse_text_config(BASE + f"Morph.TargetShape = 2\nMorph.TargetWidth = {width}\nMorph.TargetHeight = {height}\n")
    grid = inner(config)
    assert np.hypot(grid[:, -1, 0], grid[:, -1, 1]) == pytest.approx(164.2197938615493, abs=1e-6)
    raw = inner(parse_text_config(BASE))
    assert np.allclose(grid, raw, rtol=0, atol=1e-12)


def test_imported_circle_uses_largest_raw_mouth_radius_for_nonuniform_profile():
    text = BASE.replace("a = 37", "a = 37 + 5*cos(p)^2")
    grid = inner(parse_text_config(text + "Morph.TargetShape = 2\nMorph.TargetWidth = 410\nMorph.TargetHeight = 310\n"))
    assert np.hypot(grid[:, -1, 0], grid[:, -1, 1]) == pytest.approx(183.54782172, abs=1e-6)


def test_native_circle_honors_explicit_dimensions():
    config = parse_text_config(BASE + "Morph.TargetShape = 2\nMorph.TargetWidth = 360\nMorph.TargetHeight = 360\n")
    config.pop(TEXT_IMPORT_VERSION_KEY)
    assert np.hypot(inner(config)[:, -1, 0], inner(config)[:, -1, 1]) == pytest.approx(180, abs=1e-12)


@pytest.mark.parametrize("flag", ["", "Morph.AllowShrinkage = 0\n"])
def test_imported_inward_no_shrink_target_has_named_refusal(flag):
    config = parse_text_config(BASE + "Morph.TargetShape = 1\nMorph.TargetWidth = 360\nMorph.TargetHeight = 280\nMorph.CornerRadius = 35\n" + flag)
    with pytest.raises(ValueError, match=r"Imported Morph.AllowShrinkage=0.*Morph.AllowShrinkage=1"):
        inner(config)
    with pytest.raises(ValueError, match="no qualified construction"):
        resolve_geometry(config)
    config.pop(TEXT_IMPORT_VERSION_KEY)
    native = inner(config)
    assert np.max(np.abs(native[:, -1, 1])) == pytest.approx(164.2197938615493, abs=1e-6)


def test_imported_explicit_shrink_targets_and_native_nonshrinking_controls():
    config = parse_text_config(BASE + "Morph.TargetShape = 1\nMorph.TargetWidth = 360\nMorph.TargetHeight = 280\nMorph.CornerRadius = 35\nMorph.AllowShrinkage = 1\n")
    assert np.max(np.abs(inner(config)[:, -1, :2]), axis=0) == pytest.approx([180,140], abs=1e-12)
    control = parse_text_config(BASE + "Morph.TargetShape = 1\nMorph.TargetWidth = 400\nMorph.TargetHeight = 380\nMorph.CornerRadius = 35\n")
    native = copy.deepcopy(control)
    native.pop(TEXT_IMPORT_VERSION_KEY)
    assert np.array_equal(inner(control), inner(native))


def test_imported_slot_with_guide_has_named_refusal():
    p = params(BASE + "Slot.Length = 15\nGCurve.Type = 1\nGCurve.Width = 240\nGCurve.Dist = .8\n")
    with pytest.raises(ValueError, match="Imported Slot.Length.*active guiding curve"):
        calculate_osse(137, 0, p)
    with pytest.raises(ValueError, match="no qualified construction"):
        calculate_osse_curve([0,137], 0, p)
    p.pop(TEXT_IMPORT_VERSION_KEY)
    assert math.isfinite(calculate_osse(137, 0, p)[1])


@pytest.mark.parametrize("controls,mouth_radius", [
    ("Slot.Length = 15\n", 146.879189),
    ("Morph.TargetShape = 2\nMorph.TargetWidth = 360\nMorph.TargetHeight = 360\n", 164.219794),
])
def test_imported_profile_reaches_real_mesh_and_reimported_step(tmp_path, controls, mouth_radius):
    import gmsh
    from hornlab_mesher.cad import write_step_from_config
    from hornlab_mesher.config_builder import build_from_config
    from hornlab_mesher.mesher import load_mesh

    config = parse_text_config(BASE + controls)
    config["mode"] = "bare"
    config["mesh"].update(throatResolution=5, mouthResolution=20, maxTriangles=18000)
    resolved = resolve_geometry(config)
    assert np.max(np.hypot(resolved.geometry.inner_points[:, -1, 0],
                           resolved.geometry.inner_points[:, -1, 1])) == pytest.approx(mouth_radius, abs=1e-6)
    built = build_from_config(config, tmp_path / "profile.msh")
    mesh = load_mesh(built.mesh_path)
    assert mesh.n_triangles > 0
    assert {1,2} <= set(mesh.physical_groups)
    assert mesh.bounding_box[1][2] == pytest.approx(.137, abs=1e-7)
    assert mesh.bounding_box[1][0] == pytest.approx(mouth_radius/1000, abs=1e-7)
    path, info = write_step_from_config(config, tmp_path / "profile.step")
    assert info.body == "surface"
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("imported-profile-regression")
        gmsh.model.occ.importShapes(str(path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        assert gmsh.model.getEntities(2)
        box = gmsh.model.getBoundingBox(-1, -1)
        assert box[5] == pytest.approx(137, abs=1e-3)
        # OCC boxes can bound the control hull rather than the trimmed surface.
        # Probe the physical rim instead of treating that hull as geometry.
        for phi in (0., math.pi/2, math.pi, 3*math.pi/2):
            point = np.asarray([mouth_radius*math.cos(phi), mouth_radius*math.sin(phi), 137.])
            distance = min(np.linalg.norm(np.asarray(gmsh.model.getClosestPoint(2, tag, point)[0])-point)
                           for _, tag in gmsh.model.getEntities(2))
            assert distance < 1e-4
    finally:
        gmsh.finalize()
