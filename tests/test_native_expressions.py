"""Expression-shaped native bodies retain the existing meridian owners."""
import math
from copy import deepcopy

import numpy as np
import pytest

from hornlab_mesher.config_builder import build_geometry_params, resolve_geometry
from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry
from hornlab_mesher.profile_common import eval_param


def recipe(feature):
    config = {
        "formula": feature, "mode": "bare", "scale": 1,
        "profile": {"L_mm": 12, "r0_mm": 1.27, "a_deg": 40,
                    "a0_deg": 10, "k": 1, "s": 0},
        "source": {"source_shape": 0},
        "mesh": {"wall_thickness_mm": 0, "throat_res_mm": 1,
                 "mouth_res_mm": 2, "rear_res_mm": 2, "scale_to_metres": False},
    }
    if "ADAPTER" in feature:
        config["throat_adapter"] = {
            "mode": "authored", "contract_revision": 1,
            "driver_exit_diameter_mm": 2, "exit_half_angle_deg": 5,
            "length_mm": 3, "join_t": .02, "driver_handle_mm": .8, "body_handle_mm": 1.2,
        }
    if feature == "OSSE-ADAPTER-CONTROLS":
        config["throat_adapter"] = {"mode": "controls", "contract_revision": 1,
                                    "join_t": 0, "control_points_mm": [[-3, 1], [-2, 1.1], [-1, 1.2]]}
    if "AXIAL" in feature:
        config["axial_scale"] = 1.2
    if feature == "OSSE-ARC":
        config["terminating_arc"] = {"contract_revision": 1, "join_t": .1, "end_tangent_deg": 100}
    if feature == "OSSE-ROUNDOVER":
        config["mode"] = "freestanding"
        config["mesh"].update(wall_thickness_mm=.2, mouth_roundover_radius_mm=1, max_triangles=60000)
    return config


FAMILIES = ["OSSE-ADAPTER", "OSSE-ADAPTER-CONTROLS", "OSSE-ADAPTER-AXIAL",
            "OSSE-AXIAL", "OSSE-ARC", "OSSE-ROUNDOVER"]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("field,expression", [
    ("L_mm", "12 + cos(2*p)"), ("r0_mm", "1.27 + .05*sin(p)"),
    ("a_deg", "40 + 2*cos(2*p)"), ("a0_deg", "10 + sin(p)"), ("k", "1 + .05*cos(p)"),
])
def test_native_body_expressions_use_the_scalar_construction_per_meridian(family, field, expression):
    config = recipe(family)
    config["profile"][field] = expression
    params, formula, mode = build_geometry_params(config)
    from hornlab_mesher.native_expressions import meridian_branches
    assert params["nativeExpressionConfig"] == config
    assert formula == family
    resolved = resolve_geometry(config)
    assert resolved.mode == mode
    surface = resolved.geometry.expression_surface
    for i in (0, 7, 16, 32, 48, 63):
        phi = 2*math.pi*i/64
        scalar = deepcopy(config)
        scalar["profile"][field] = eval_param(expression, phi)
        expected = meridian_branches(build_geometry_params(scalar)[0], 32)
        for branch, (_, zr) in zip(surface.branches, expected):
            actual = branch[1][i]
            np.testing.assert_allclose(actual[:, 0], zr[:, 1]*math.cos(phi), atol=1e-12)
            np.testing.assert_allclose(actual[:, 1], zr[:, 1]*math.sin(phi), atol=1e-12)
            np.testing.assert_array_equal(actual[:, 2], zr[:, 0])


@pytest.mark.parametrize("family", FAMILIES)
def test_native_expression_preview_has_complete_roles_and_keeps_input(family):
    config = recipe(family)
    config["profile"]["a_deg"] = "40 + cos(2*p)"
    before = deepcopy(config)
    preview = build_preview_geometry(config, PreviewOptionsV1(lod="coarse", include_curvature=False))
    roles = {surface.role for surface in preview.surfaces}
    assert {"horn.inner", "source_cap"} <= roles
    if family == "OSSE-ROUNDOVER":
        assert {"horn.outer", "mouth_rim", "wall.rear_return", "wall.rear_cap"} <= roles
    assert config == before
    for surface in preview.surfaces:
        assert np.isfinite(surface.positions).all()


@pytest.mark.parametrize("outer", [False, True])
def test_roundover_expression_preview_lip_orientation_winding_and_curvature(outer):
    from hornlab_mesher.mouth_roundover import Roundover
    from hornlab_mesher.native_expressions import scalar_config

    config = recipe("OSSE-ROUNDOVER")
    config["profile"]["a_deg"] = "40 + .1*cos(2*p)"
    preview = build_preview_geometry(config, PreviewOptionsV1(lod="coarse"))
    role = "horn.outer" if outer else "horn.inner"
    surface = next(part for part in preview.surfaces if part.role == role)
    model = Roundover.from_params(build_geometry_params(scalar_config(config, 0))[0])
    radius = model.radius - (model.wall if outer else 0)
    # At this cardinal meridian the expression's angular derivative is zero,
    # so the scalar owner's meridian tangent gives the exact surface normal.
    for fraction in (.25, .5, .75):
        z, r = model.arc(fraction*model.sweep, outer)
        index = np.argmin(np.linalg.norm(surface.positions - [r, 0, z], axis=1))
        x, _, z = surface.positions[index]
        theta = math.atan2(z-model.center[0], model.center[1]-x)
        expected = np.array([math.cos(theta), 0, -math.sin(theta)]) * (1 if outer else -1)
        assert surface.normals[index] @ expected > .99999
        sign = 1 if outer else -1
        assert sign*surface.curvature_mean[index] > 0
        assert sign*surface.curvature_principal[index] > .9/radius
        triangles = surface.indices.reshape(-1, 3)
        adjacent = triangles[np.any(triangles == index, axis=1)]
        a, b, c = surface.positions[adjacent].transpose(1, 0, 2)
        assert np.all(np.cross(b-a, c-a) @ expected > 0)


def test_native_expression_domain_error_names_the_meridian_and_remedy():
    config = recipe("OSSE-AXIAL")
    config["profile"]["a_deg"] = "40 + 50*sin(p)"
    with pytest.raises(ValueError, match="azimuth.*Edit.*expression"):
        build_geometry_params(config)


@pytest.mark.parametrize("family", FAMILIES)
def test_native_expression_solve_mesh_and_step(family, tmp_path):
    config = recipe(family)
    config["profile"]["a_deg"] = "40 + cos(2*p)"
    _assert_mesh_and_step(config, tmp_path)


def _assert_mesh_and_step(config, tmp_path):
    import meshio

    from hornlab_mesher.cad import write_step_from_config
    from hornlab_mesher.config_builder import build_from_config

    result = build_from_config(config, tmp_path / "expression.msh")
    mesh = meshio.read(result.mesh_path)
    assert np.isfinite(mesh.points).all()
    assert set(mesh.cell_data_dict["gmsh:physical"]["triangle"]) == {1, 2}
    path, info = write_step_from_config(config, tmp_path / "expression.step")
    assert "ISO-10303-21" in path.read_text()
    assert "native-osse-meridian-grid-v1" in path.read_text()
    closed = config["mode"] == "freestanding"
    assert info.body == ("solid" if closed else "surface")
    if closed:
        triangles = mesh.cells_dict["triangle"]
        edges = np.sort(np.vstack((triangles[:, :2], triangles[:, 1:], triangles[:, [2, 0]])), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        assert np.all(counts == 2)


@pytest.mark.parametrize("mouth", ["arc", "roundover"])
def test_composite_expression_solve_mesh_and_step(mouth, tmp_path):
    config = recipe("OSSE-ARC" if mouth == "arc" else "OSSE-ROUNDOVER")
    config.update(formula="OSSE-NATIVE", axial_scale=1.2,
                  throat_adapter=recipe("OSSE-ADAPTER")["throat_adapter"])
    config["profile"]["a_deg"] = "40 + cos(2*p)"
    _assert_mesh_and_step(config, tmp_path)


def test_varying_throat_and_length_solve_mesh_and_step(tmp_path):
    config = recipe("OSSE-AXIAL")
    config["profile"].update(r0_mm="1.27 + .05*cos(2*p)", L_mm="12 + cos(2*p)")
    _assert_mesh_and_step(config, tmp_path)


@pytest.mark.parametrize("family", FAMILIES)
def test_constant_body_expressions_keep_literal_geometry_bytes(family):
    literal = recipe(family)
    expressions = deepcopy(literal)
    expressions["profile"] = {key: f"{value} + 0" if key != "s" else value
                              for key, value in literal["profile"].items()}
    assert build_geometry_params(expressions) == build_geometry_params(literal)
    old, new = resolve_geometry(literal).geometry, resolve_geometry(expressions).geometry
    assert old.inner_points.tobytes() == new.inner_points.tobytes()
    if old.outer_points is not None:
        assert old.outer_points.tobytes() == new.outer_points.tobytes()


def test_preview_density_ignores_mesh_sampling_controls_and_measures_fidelity():
    config = recipe("OSSE-AXIAL")
    config["profile"]["a_deg"] = "40 + cos(2*p)"
    options = PreviewOptionsV1(lod="coarse", include_curvature=False)
    original = build_preview_geometry(config, options)
    config["mesh"].update(angular_segments=24, length_segments=10)
    changed = build_preview_geometry(config, options)
    for before, after in zip(original.surfaces, changed.surfaces):
        assert before.positions.tobytes() == after.positions.tobytes()
    measured = changed.metadata["fidelity"]["horn.inner"]
    assert measured["max_chord_error_mm_achieved"] > 0
    assert measured["measurement_complete"]


@pytest.mark.parametrize("section", ["root", "profile"])
@pytest.mark.parametrize("field,canonical,expression", [
    ("L", "L_mm", "12 + cos(p)"), ("r0", "r0_mm", "1.27 + .05*cos(p)"),
    ("a", "a_deg", "40 + cos(p)"), ("a0", "a0_deg", "10 + cos(p)"), ("k", "k", "1 + .01*cos(p)"),
])
def test_authored_adapter_reuses_its_existing_body_aliases(section, field, canonical, expression):
    config = recipe("OSSE-ADAPTER")
    del config["profile"][canonical]
    target = config if section == "root" else config["profile"]
    target[field] = expression
    assert "nativeExpressionConfig" in build_geometry_params(config)[0]
    assert resolve_geometry(config).geometry.expression_surface is not None


def test_existing_recipe_keeps_dormant_body_alias_text_and_literal_geometry():
    literal = recipe("OSSE-ADAPTER")
    dormant = deepcopy(literal)
    dormant["profile"]["a"] = "1/0 + sin(p)"
    assert build_geometry_params(dormant) == build_geometry_params(literal)
    assert resolve_geometry(dormant).geometry.inner_points.tobytes() == resolve_geometry(literal).geometry.inner_points.tobytes()
