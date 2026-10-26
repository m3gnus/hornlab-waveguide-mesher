"""The flush route is selected by the existing enclosure clearance owner."""
import numpy as np
import pytest

from hornlab_mesher.builders.enclosure import (
    enclosure_box_bounds,
    measure_enclosure_box_bounds,
)
from hornlab_mesher.config_builder import build_geometry_params, resolve_enclosure_bounds, resolve_geometry
from hornlab_mesher.config_parser import ConfigError
from hornlab_mesher.geometry import HornEnclosure
from test_point_grid_contract import _ASRO2_PARAMS, _boundary_edges, _triangles_and_tags, rectangular_enclosure


def _points(half_width=150.0, half_height=200.0):
    mouth = np.array([[half_width, 0], [0, half_height],
                      [-half_width, 0], [0, -half_height]])
    points = np.zeros((4, 2, 3), dtype=float)
    points[:, -1, :2] = mouth
    points[:, -1, 2] = 100
    return points


@pytest.mark.parametrize("clearance,flush", [(0.0, True), (0.05, True),
                                              (0.1 - 1e-8, True),
                                              (0.1, False), (1.0, False)])
@pytest.mark.parametrize("dim", [False, True])
def test_shared_clearance_decides_flush(clearance, flush, dim):
    enclosure = HornEnclosure(
        depth_mm=200, space_l_mm=clearance, space_r_mm=clearance,
        space_t_mm=clearance, space_b_mm=clearance,
        dim_mm=(150 + clearance, 200 + clearance, 150 + clearance,
                200 + clearance) if dim else None,
    )
    bounds = enclosure_box_bounds(_points(), enclosure, closed=True, warn_prefix=None)
    assert bounds.get("flush_mount", False) is flush
    if not flush:
        assert bounds == measure_enclosure_box_bounds(
            _points(), enclosure, closed=True, warn_prefix=None
        )


def test_previously_buildable_fractional_zero_margin_keeps_bounds():
    points = _points(150.25, 200.25)
    enclosure = HornEnclosure(depth_mm=280, space_l_mm=0, space_r_mm=0,
                              space_t_mm=0, space_b_mm=0)
    bounds = enclosure_box_bounds(points, enclosure, closed=True, warn_prefix=None)
    assert bounds["front_ring_min_mm"] == pytest.approx(0.75)
    assert bounds == measure_enclosure_box_bounds(points, enclosure, closed=True,
                                                  warn_prefix=None)


def test_flush_refuses_a_horn_wider_behind_the_rim_before_meshing(monkeypatch, tmp_path):
    import hornlab_mesher.config_builder as builder

    config = {
        "formula": "OSSE", "mode": "enclosure",
        "profile": {"L": 120, "r0": 12.7, "a": 60, "a0": 15.5, "k": 1, "s": 0},
        "morph": {"morphTarget": 2, "morphWidth": 200, "morphHeight": 200,
                  "morphAllowShrinkage": 1, "morphFixed": 0.5},
        "mesh": {"quadrants": "1", "topology": "legacy", "angularSegments": 32,
                 "lengthSegments": 24, "surfaceFit": "interpolate", "wallThickness": 0},
        "enclosure": {"depth": 280, "edge": 0,
                      **{f"space_{side}": 0 for side in "lrtb"}},
    }
    monkeypatch.setattr(builder, "build_mesh_with_info", lambda *a, **k: pytest.fail("mesh started"))
    with pytest.raises(ConfigError, match="wider behind the mouth.*Add an enclosure margin"):
        builder.resolve_geometry(config)
    with pytest.raises(ConfigError, match="wider behind the mouth"):
        builder.build_from_config(config, tmp_path / "refused.msh")
    # The new restriction only applies on the flush branch.
    config["enclosure"].update({f"space_{side}": 100 for side in "lrtb"})
    assert builder.resolve_enclosure_bounds(config, warn_prefix=None).get("flush_mount") is None


@pytest.mark.parametrize("active", [True, False])
def test_flush_with_configured_woofer_is_refused_before_meshing(active, monkeypatch, tmp_path):
    import hornlab_mesher.config_builder as builder
    from test_enclosure_woofers import assembly_config

    config = assembly_config()
    config["enclosure"]["woofers"][0]["active"] = active
    mouth = builder.resolve_geometry(config).geometry.inner_points[:, -1, :]
    config["enclosure"]["dim_mm"] = (
        float(mouth[:, 0].max()), float(mouth[:, 1].max()) + 150,
        float(-mouth[:, 0].min()) + 80, float(-mouth[:, 1].min()) + 150,
    )
    monkeypatch.setattr(builder, "build_mesh_with_info", lambda *a, **k: pytest.fail("mesh started"))
    with pytest.raises(ConfigError, match="no front baffle for a woofer"):
        builder.resolve_geometry(config)
    with pytest.raises(ConfigError, match="no front baffle for a woofer"):
        builder.build_from_config(config, tmp_path / "refused.msh")


def test_flush_bounds_include_the_fitted_mouth_rim():
    config = rectangular_enclosure(margin=0.0, quadrants="1")
    bounds = resolve_enclosure_bounds(config, warn_prefix=None)
    assert bounds["flush_mount"] is True
    # A fitted surface can extend beyond its input points. The enclosure's
    # real side follows that surface, not the nominal 150 x 200 mm endpoints.
    assert bounds["bx1"] == pytest.approx(150.034255, abs=1e-5)
    assert bounds["by1"] == pytest.approx(200.501229, abs=1e-5)
    assert bounds["bx0"] == bounds["by0"] == 0


@pytest.mark.parametrize("quadrants", ["1", "12", "14", "1234"])
@pytest.mark.parametrize("preserve", [False, True])
def test_flush_rear_cap_shares_all_side_curves(tmp_path, monkeypatch, quadrants, preserve):
    gmsh = pytest.importorskip("gmsh")
    from hornlab_mesher.builders import enclosure as enclosure_builder
    from hornlab_mesher.config_builder import build_from_config

    original = enclosure_builder._build_flush_enclosure

    def checked(curves, bounds):
        built = original(curves, bounds)
        gmsh.model.occ.synchronize()
        sides = [(2, tag) for tag in built["sides"]]
        rear_curves = set(enclosure_builder._boundary_curves_at_z_extreme(
            sides, want_min_z=True,
        ))
        cap_curves = {abs(tag) for dim, tag in gmsh.model.getBoundary(
            [(2, tag) for tag in built["back"]], combined=False, oriented=False,
        ) if dim == 1}
        assert len(rear_curves) == len(curves)
        assert rear_curves <= cap_curves
        return built

    monkeypatch.setattr(enclosure_builder, "_build_flush_enclosure", checked)
    config = rectangular_enclosure(margin=0.0, quadrants=quadrants)
    config["mesh"].update(angularSegments="12", lengthSegments="8",
                          topology="legacy", preserve_grid=preserve,
                          surfaceFit="interpolate")
    assert build_from_config(config, tmp_path / "joined.msh").n_triangles > 0


def test_flush_rear_cap_handles_mouth_that_crosses_a_sector_axis(tmp_path):
    from hornlab_mesher.config_builder import build_from_config

    config = {
        "formula": "OSSE", "mode": "enclosure",
        "profile": {"L": 120, "r0": 12.7, "a0": 15.5,
                    "a": "45+35*sin(4*p)", "k": 1, "s": 0},
        "mesh": {"quadrants": "1234", "topology": "legacy",
                 "angular_segments": 16, "length_segments": 16,
                 "wall_thickness_mm": 0, "surface_fit": "interpolate"},
        "enclosure": {"depth": 280, "edge": 0,
                      **{f"space_{side}": 0 for side in "lrtb"}},
    }
    mouth = resolve_geometry(config).geometry.inner_points[:, -1, :]
    config["enclosure"]["dim_mm"] = (
        float(mouth[:, 0].max()), float(mouth[:, 1].max()),
        float(-mouth[:, 0].min()), float(-mouth[:, 1].min()),
    )
    assert resolve_enclosure_bounds(config, warn_prefix=None)["flush_mount"]
    assert build_from_config(config, tmp_path / "fold.msh").n_triangles > 0


@pytest.mark.parametrize("quadrant_patch_fit", [False, True])
@pytest.mark.parametrize("closed", [False, True])
def test_flush_back_stays_behind_interpolated_wall_overshoot(quadrant_patch_fit, closed):
    """A monotone sampled wall can interpolate behind its lowest station."""
    gmsh = pytest.importorskip("gmsh")
    from hornlab_mesher.builders.point_grid_surfaces import (
        _add_occ_bspline_patch_wall_surfaces, _bspline_patch_phi_groups,
    )

    phi = (np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False) if closed
           else np.linspace(0.0, np.pi / 2.0, 9))
    radius = np.linspace(10.0, 100.0, 4)
    points = np.empty((len(phi), len(radius), 3))
    points[:, :, 0] = np.cos(phi)[:, None] * radius
    points[:, :, 1] = np.sin(phi)[:, None] * radius
    points[:, :, 2] = [0.0, 1.0, 99.0, 100.0]
    enclosure = HornEnclosure(
        depth_mm=1.0, depth_margin_mm=1.0, edge_mm=0.0,
        space_l_mm=0.0, space_r_mm=0.0, space_t_mm=0.0, space_b_mm=0.0,
        dim_mm=(100.0, 100.0, 100.0, 100.0),
    )
    bounds = enclosure_box_bounds(
        points, enclosure, closed=closed,
        symmetry_planes=() if closed else ("x", "y"),
        surface_fit="interpolate", warn_prefix=None,
    )
    assert bounds["flush_mount"]

    initialized_here = not gmsh.isInitialized()
    if initialized_here:
        gmsh.initialize()
    gmsh.model.add("flush-rear-overshoot")
    try:
        wall = _add_occ_bspline_patch_wall_surfaces(
            points, closed=closed,
            phi_groups=_bspline_patch_phi_groups(len(phi), closed=closed),
            surface_fit="interpolate", quadrant_patch_fit=quadrant_patch_fit,
        )
        gmsh.model.occ.synchronize()
        uv = np.column_stack((np.full(401, 0.5), np.linspace(0.0, 1.0, 401)))
        actual = np.asarray(gmsh.model.getValue(2, wall[0][1], uv.ravel())).reshape(-1, 3)
        assert actual[:, 2].min() < -3.0  # The old sampled bound was -1 mm.
        assert bounds["z_back"] <= actual[:, 2].min() - 1.0 + 1e-8
    finally:
        gmsh.model.remove()
        if initialized_here:
            gmsh.finalize()


def test_auto_fit_keeps_the_original_horn_fit_for_flush():
    flush = rectangular_enclosure(margin=0.0, quadrants="1")
    flush["mesh"]["surfaceFit"] = "auto"
    assert resolve_geometry(flush).geometry.surface_fit == "interpolate"
    accepted = rectangular_enclosure(margin=0.0, quadrants="1", fractional=True)
    accepted["mesh"]["surfaceFit"] = "auto"
    assert resolve_geometry(accepted).geometry.surface_fit == "interpolate"


@pytest.mark.parametrize("quadrants", ["1", "12", "14", "1234"])
def test_auto_fit_retry_preserves_flush_wall(tmp_path, monkeypatch, quadrants):
    import hornlab_mesher.config_builder as builder
    from hornlab_mesher.builders._occ import sample_patch_boundary
    from hornlab_mesher.builders.point_grid_surfaces import _bspline_patch_phi_groups
    from hornlab_mesher.mesher import MesherError

    config = rectangular_enclosure(margin=0.0, quadrants=quadrants)
    config["mesh"]["surfaceFit"] = "auto"
    original_build = builder.build_mesh_with_info
    attempted = []

    def fail_first(geometry, *args, **kwargs):
        attempted.append(geometry)
        if len(attempted) == 1:
            raise MesherError("simulated first-fit mesh failure")
        return original_build(geometry, *args, **kwargs)

    monkeypatch.setattr(builder, "build_mesh_with_info", fail_first)
    result = builder.build_from_config(config, tmp_path / "retry.msh")
    assert result.metadata["surfaceFit"] == "interpolate"
    assert [geometry.surface_fit for geometry in attempted] == ["interpolate", "interpolate"]
    from hornlab_mesher.geometry import _RetryInterpolatingFit
    assert isinstance(attempted[1].surface_fit, _RetryInterpolatingFit)
    np.testing.assert_array_equal(attempted[0].inner_points, attempted[1].inner_points)
    group = _bspline_patch_phi_groups(
        len(attempted[0].inner_points), closed=attempted[0].closed,
        n_sectors=1 if len(attempted[0].symmetry_planes) >= 2 else 2,
    )[0]
    u = np.linspace(0.0, 1.0, 257)
    rims = [sample_patch_boundary(g.inner_points, group, u, surface_fit=g.surface_fit)
            for g in attempted]
    np.testing.assert_allclose(rims[0], rims[1], atol=1e-10, rtol=0.0)


def test_native_roundover_with_enclosure_keeps_existing_refusal():
    config = {
        "formula": "OSSE-ROUNDOVER", "mode": "enclosure",
        "profile": {"L_mm": 120, "r0_mm": 12.7, "a_deg": 45,
                    "a0_deg": 10, "k": 1, "s": 0},
        "mesh": {"wall_thickness_mm": 5, "mouth_roundover_radius_mm": 22},
        "enclosure": {"depth_mm": 300, "space_l_mm": 0, "space_r_mm": 0,
                      "space_t_mm": 0, "space_b_mm": 0},
    }
    with pytest.raises(ConfigError, match="Mouth roundover refused: unsupported root controls: enclosure"):
        build_geometry_params(config)


@pytest.mark.parametrize("quadrants,cut_axes", [
    ("1", (0, 1)), ("12", (1,)), ("14", (0,)), ("1234", ()),
])
@pytest.mark.parametrize("shape", ["rectangle", "circle"])
def test_flush_mesh_welds_to_mouth_and_is_open_only_on_cut_planes(
    tmp_path, quadrants, cut_axes, shape,
):
    import meshio
    from hornlab_mesher.config_builder import build_from_config

    config = rectangular_enclosure(margin=0.0, quadrants=quadrants)
    if shape == "circle":
        config["morph"].update(morphTarget=2, morphWidth=300, morphHeight=300)
    result = build_from_config(config, tmp_path / "flush.msh")
    mesh = meshio.read(result.mesh_path)
    triangles, tags = _triangles_and_tags(mesh)
    assert {1, 2, 3}.issubset(set(tags))
    boundary = _boundary_edges(triangles)
    assert bool(boundary) is bool(cut_axes)
    for a, b in boundary:
        assert any(abs(mesh.points[a, axis]) < 1e-7 and
                   abs(mesh.points[b, axis]) < 1e-7 for axis in cut_axes)
    edges = np.sort(np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]],
                               triangles[:, [2, 0]])), axis=1)
    assert np.unique(edges, axis=0, return_counts=True)[1].max() <= 2


@pytest.mark.parametrize("shape", ["rectangle", "circle"])
@pytest.mark.parametrize("fit,preserve", [("auto", False), ("approximate", False),
                                         ("interpolate", False), ("approximate", True)])
def test_flush_mesh_supports_wall_fit_modes(tmp_path, shape, fit, preserve):
    import meshio
    from hornlab_mesher.config_builder import build_from_config

    config = rectangular_enclosure(margin=0.0, quadrants="1")
    config["mesh"].update(angularSegments="12", lengthSegments="8",
                          surfaceFit=fit, topology="legacy", preserve_grid=preserve)
    if shape == "circle":
        config["morph"].update(morphTarget=2, morphWidth=300, morphHeight=300)
    result = build_from_config(config, tmp_path / "fit.msh")
    mesh = meshio.read(result.mesh_path)
    triangles, tags = _triangles_and_tags(mesh)
    assert {1, 2, 3}.issubset(set(tags))
    boundary = _boundary_edges(triangles)
    assert boundary
    assert all(any(abs(mesh.points[a, axis]) < 1e-7 and
                   abs(mesh.points[b, axis]) < 1e-7 for axis in (0, 1))
               for a, b in boundary)


@pytest.mark.parametrize("quadrants", ["1", "1234"])
@pytest.mark.parametrize("plan_type", [2, 3])
def test_flush_uses_mouth_rim_for_nonrectangular_enclosure_plans(tmp_path, quadrants, plan_type):
    import meshio
    from hornlab_mesher.config_builder import build_from_config

    config = rectangular_enclosure(margin=0.0, quadrants=quadrants)
    config["morph"].update(morphTarget=2, morphWidth=300, morphHeight=300)
    config["enclosure"]["plan_type"] = plan_type
    result = build_from_config(config, tmp_path / "plan.msh")
    mesh = meshio.read(result.mesh_path)
    triangles, _ = _triangles_and_tags(mesh)
    boundary = _boundary_edges(triangles)
    if quadrants == "1234":
        assert not boundary
    else:
        assert boundary
        assert all(any(abs(mesh.points[a, axis]) < 1e-7 and
                       abs(mesh.points[b, axis]) < 1e-7 for axis in (0, 1))
                   for a, b in boundary)


def test_flush_nonplanar_mouth_still_has_planar_rear(tmp_path):
    import meshio
    from hornlab_mesher.config_builder import build_from_config

    config = {
        "formula": "OSSE", "mode": "enclosure",
        "profile": {"L": "120+20*sin(p)^2", "r0": 12.7, "a": 45,
                    "a0": 15.5, "k": 1, "s": 0},
        "mesh": {"quadrants": "1", "angularSegments": 32, "lengthSegments": 16,
                 "surfaceFit": "interpolate", "topology": "legacy"},
        "morph": {"morphTarget": 2, "morphWidth": 300, "morphHeight": 300},
        "enclosure": {"depth": 280, "dim_mm": (150, 150, 150, 150)},
    }
    bounds = resolve_enclosure_bounds(config, warn_prefix=None)
    assert bounds["flush_mount"] is True
    result = build_from_config(config, tmp_path / "nonplanar.msh")
    mesh = meshio.read(result.mesh_path)
    triangles, _ = _triangles_and_tags(mesh)
    assert all(any(abs(mesh.points[a, axis]) < 1e-7 and
                   abs(mesh.points[b, axis]) < 1e-7 for axis in (0, 1))
               for a, b in _boundary_edges(triangles))
    assert mesh.points[:, 2].min() == pytest.approx(bounds["z_back"] / 1000, abs=1e-6)


def test_flush_refuses_a_deep_rollback_that_is_wider_behind_the_rim():
    profile = dict(_ASRO2_PARAMS, tmax=1.3)
    config = {
        "formula": "ROSSE", "mode": "enclosure", "profile": profile,
        "mesh": {"quadrants": "1", "angularSegments": 32, "lengthSegments": 16,
                 "surfaceFit": "interpolate"},
        "enclosure": {"depth": 1, **{f"space_{side}": 0 for side in "lrtb"}},
    }
    geometry = resolve_geometry(config).geometry
    mouth = geometry.inner_points[:, -1, :]
    xmax, ymax = float(mouth[:, 0].max()), float(mouth[:, 1].max())
    config["enclosure"]["dim_mm"] = (xmax, ymax, xmax, ymax)
    with pytest.raises(ConfigError, match="wider behind the mouth"):
        resolve_enclosure_bounds(config, warn_prefix=None)


@pytest.mark.parametrize("case", ["a", "c"])
def test_flush_freeform_corner_wall_uses_the_actual_rim(tmp_path, case):
    import meshio
    from hornlab_mesher.builders._occ import patch_boundary_bounds
    from hornlab_mesher.builders.point_grid_surfaces import _bspline_patch_phi_groups, _snap_open_symmetry_grid
    from hornlab_mesher.corner_wall import _angular_poles
    from hornlab_mesher.config_builder import build_from_config
    from test_freeform_corner_wall import case_config

    config = case_config(case, "enclosure")
    config["mesh"]["quadrants"] = "1"
    geometry = resolve_geometry(config).geometry
    assert geometry.freeform_wall is not None
    mouth = geometry.inner_points[:, -1, :]
    xmax, ymax = float(mouth[:, 0].max()), float(mouth[:, 1].max())
    config["enclosure"].update(dim_mm=(xmax, ymax, xmax, ymax), edge=0)
    bounds = resolve_enclosure_bounds(config, warn_prefix=None)
    assert bounds["flush_mount"] is True
    points = _snap_open_symmetry_grid(
        geometry.inner_points, closed=geometry.closed, symmetry_planes=geometry.symmetry_planes,
    )
    groups = _bspline_patch_phi_groups(len(points), closed=False, n_sectors=1)
    poles = _angular_poles(points, groups, geometry.freeform_wall)
    _, fitted_max = patch_boundary_bounds(poles, groups[0], surface_fit="approximate")
    assert [bounds["bx1"], bounds["by1"]] == pytest.approx(fitted_max[:2])
    result = build_from_config(config, tmp_path / "freeform.msh")
    mesh = meshio.read(result.mesh_path)
    triangles, _ = _triangles_and_tags(mesh)
    assert all(any(abs(mesh.points[a, axis]) < 1e-7 and
                   abs(mesh.points[b, axis]) < 1e-7 for axis in (0, 1))
               for a, b in _boundary_edges(triangles))


def _flush_corpus_config(family, quadrants, fit):
    from copy import deepcopy

    from test_general_horn_attachment import config as corpus_config

    config = deepcopy(corpus_config(family))
    config["mode"] = "enclosure"
    config["mesh"].update(surface_fit=fit, quadrants="1234")
    config["enclosure"] = {"depth": 280, "edge": 0}
    rim = resolve_geometry(config).geometry.inner_points[:, -1, :]
    x_max, y_max = float(np.abs(rim[:, 0]).max()), float(np.abs(rim[:, 1]).max())
    config["enclosure"]["dim_mm"] = (x_max, y_max, x_max, y_max)
    config["mesh"]["quadrants"] = quadrants
    return config


@pytest.mark.parametrize("quadrants", ["1", "12", "14", "1234"])
@pytest.mark.parametrize("family,fit", [
    ("OSSE", "interpolate"), ("OSSE", "approximate"),
    ("ROSSE", "interpolate"), ("ROSSE", "approximate"),
    ("ICW", "interpolate"), ("ICW", "approximate"),
    ("FREEFORM", "approximate"),
])
def test_ordinary_flush_designs_pass_the_flush_refusals(family, fit, quadrants):
    # The identity-corpus horns, flush-mounted, are neither wider behind the
    # rim nor folded: the design-time refusals must leave them buildable.
    config = _flush_corpus_config(family, quadrants, fit)
    assert resolve_enclosure_bounds(config, warn_prefix=None)["flush_mount"] is True


@pytest.mark.parametrize("quadrants", ["1", "1234"])
@pytest.mark.parametrize("margin,fractional", [(0.0, True), (0.1, False), (1.0, False)])
def test_flush_refusals_never_run_for_enclosures_0_3_6_accepted(
    monkeypatch, quadrants, margin, fractional,
):
    import hornlab_mesher.builders.enclosure as enclosure_builder

    def refuse(*args, **kwargs):
        raise AssertionError("an accepted baffle reached the flush refusals")

    monkeypatch.setattr(enclosure_builder, "_validate_flush_enclosure", refuse)
    config = rectangular_enclosure(margin=margin, quadrants=quadrants, fractional=fractional)
    assert "flush_mount" not in resolve_enclosure_bounds(config, warn_prefix=None)
    assert resolve_geometry(config).geometry.enclosure is not None
