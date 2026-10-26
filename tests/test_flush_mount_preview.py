"""The preview must carry the solver-grid enclosure decision across LODs."""

import numpy as np
import pytest
from test_point_grid_contract import rectangular_enclosure

from hornlab_mesher.config_builder import resolve_enclosure_bounds
from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry
from hornlab_mesher.viewport import build_viewport_geometry_from_config


def _enclosure_with_resolved_flush():
    # Inject a flush decision without depending on the display grid's sampling.
    config = rectangular_enclosure(margin=1.0)
    config["mesh"]["verticalOffset"] = 7.25
    bounds = dict(resolve_enclosure_bounds(config, warn_prefix=None))
    bounds["flush_mount"] = True
    return config, bounds


def test_viewport_keeps_resolved_flush_bounds_and_rigid_offset():
    config, canonical = _enclosure_with_resolved_flush()
    viewport = build_viewport_geometry_from_config(
        config, enclosure_validated=True, enclosure_bounds=canonical
    )
    bounds = viewport["enclosure"]["bounds"]
    assert bounds["flush_mount"] is True
    for key in ("bx0", "bx1", "x_min", "x_max", "z_front", "z_back"):
        assert bounds[key] == pytest.approx(canonical[key])
    for key in ("by0", "by1", "cy", "y_min", "y_max"):
        assert bounds[key] == pytest.approx(canonical[key] + 7.25)
    # A placement belongs only to the viewport, not the resolved input.
    assert viewport["enclosure"]["bounds"] is not canonical


@pytest.mark.parametrize("lod", ["coarse", "fine"])
def test_adaptive_preview_propagates_original_flush_decision(lod, monkeypatch):
    from hornlab_mesher.preview import api

    config, canonical = _enclosure_with_resolved_flush()
    real_viewport = api.build_viewport_geometry_from_config
    observed = []

    monkeypatch.setattr(api, "resolve_enclosure_bounds", lambda *_a, **_kw: canonical)

    def capture(*args, **kwargs):
        assert kwargs["enclosure_bounds"] is canonical
        output = real_viewport(*args, **kwargs)
        observed.append(output["enclosure"]["bounds"])
        return output

    monkeypatch.setattr(api, "build_viewport_geometry_from_config", capture)
    result = build_preview_geometry(config, PreviewOptionsV1(lod=lod))
    assert result.surfaces and observed
    assert all(bounds["flush_mount"] for bounds in observed)
    assert all(bounds["bx1"] == pytest.approx(canonical["bx1"]) for bounds in observed)
    assert all(bounds["by1"] == pytest.approx(canonical["by1"] + 7.25) for bounds in observed)


@pytest.mark.parametrize("quadrants", ["1", "12", "14", "1234"])
@pytest.mark.parametrize("shape", ["rectangle", "circle"])
def test_flush_preview_shares_the_fitted_mouth_and_continues_to_the_rear(quadrants, shape):
    from scipy.spatial import cKDTree

    from hornlab_mesher.builders._occ import sample_patch_boundary
    from hornlab_mesher.builders.point_grid_surfaces import (
        _bspline_patch_phi_groups,
        _open_sector_count,
    )
    from hornlab_mesher.config_builder import resolve_geometry

    config = rectangular_enclosure(margin=0, quadrants=quadrants)
    if shape == "circle":
        config["morph"].update(morphTarget=2, morphWidth=300, morphHeight=300)
        config["enclosure"]["plan_type"] = 2
    bounds = resolve_enclosure_bounds(config, warn_prefix=None)
    assert bounds["flush_mount"]
    preview = build_preview_geometry(config, PreviewOptionsV1(lod="coarse"))
    surfaces = {surface.role: surface for surface in preview.surfaces}
    assert "enclosure.front" not in surfaces
    assert {"enclosure.side", "enclosure.rear", "horn.inner"} <= surfaces.keys()
    for role in ("enclosure.side", "enclosure.rear"):
        surface = surfaces[role]
        triangles = surface.positions[surface.indices.reshape(-1, 3)]
        assert len(triangles) > 0
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                        triangles[:, 2] - triangles[:, 0]), axis=1)
        assert np.min(areas) > 1e-9
    side = surfaces["enclosure.side"]
    assert side.closed_phi == (quadrants == "1234")
    n = len(side.positions) // 2
    front, rear = side.positions[:n], side.positions[n:]
    assert np.min(np.linalg.norm(np.diff(front, axis=0), axis=1)) > 1e-7
    if side.closed_phi:
        assert np.linalg.norm(front[-1] - front[0]) > 1e-7
    else:
        columns = side.indices.reshape(-1, 3) % n
        assert not np.any(np.any(columns == 0, axis=1) & np.any(columns == n - 1, axis=1))
    np.testing.assert_allclose(front, surfaces["horn.inner"].positions[-n:], atol=1e-9)
    np.testing.assert_allclose(rear[:, :2], front[:, :2], atol=1e-9)
    np.testing.assert_allclose(rear[:, 2], bounds["z_back"], atol=1e-9)

    geometry = resolve_geometry(config).geometry
    groups = _bspline_patch_phi_groups(
        len(geometry.inner_points), closed=geometry.closed,
        n_sectors=_open_sector_count(
            closed=geometry.closed, symmetry_planes=geometry.symmetry_planes,
        ),
    )
    solver_rim = np.vstack([sample_patch_boundary(
        geometry.inner_points, group, np.linspace(0, 1, 4097),
        surface_fit=geometry.surface_fit, preserve_grid=geometry.preserve_grid,
    ) for group in groups])
    # Every preview mouth vertex lies on a solver rim curve. Discretizing the
    # reference here adds at most a small fraction of a millimetre of error.
    assert np.max(cKDTree(solver_rim).query(front)[0]) < 0.3


def test_flush_preview_accepts_nonplanar_mouth_and_has_planar_rear():
    config = {
        "formula": "OSSE", "mode": "enclosure",
        "profile": {"L": "120+20*sin(p)^2", "r0": 12.7,
                    "a0": 15.5, "a": 45, "k": 1, "s": 0},
        "mesh": {"quadrants": "1234", "topology": "legacy",
                 "angular_segments": 32, "length_segments": 16,
                 "wall_thickness_mm": 0},
        "morph": {"morphTarget": 2, "morphWidth": 300, "morphHeight": 300},
        "enclosure": {"depth": 280, "edge": 0, "dim_mm": (150, 150, 150, 150)},
    }
    preview = build_preview_geometry(config, PreviewOptionsV1(lod="coarse"))
    side = next(surface for surface in preview.surfaces if surface.role == "enclosure.side")
    n = len(side.positions) // 2
    assert np.ptp(side.positions[:n, 2]) > 1.0
    assert np.ptp(side.positions[n:, 2]) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("quadrants", ["1", "1234"])
def test_direct_viewport_flush_rings_share_the_solver_mouth_and_offset(quadrants):
    from scipy.spatial import cKDTree

    from hornlab_mesher.builders._occ import sample_patch_boundary
    from hornlab_mesher.builders.point_grid_surfaces import (
        _bspline_patch_phi_groups,
        _open_sector_count,
    )
    from hornlab_mesher.config_builder import resolve_geometry

    config = rectangular_enclosure(margin=0, quadrants=quadrants)
    config["mesh"]["verticalOffset"] = 7.25
    viewport = build_viewport_geometry_from_config(config, point_lists=False)
    enclosure = viewport["enclosure"]
    mouth = viewport["grid"]["inner_grid"][:, -1, :]
    front = np.asarray(enclosure["front_outer_points"]).reshape(-1, 3)
    rear = np.asarray(enclosure["back_outer_points"]).reshape(-1, 3)
    np.testing.assert_allclose(front, mouth, atol=1e-9)
    np.testing.assert_allclose(rear[:, :2], front[:, :2], atol=1e-9)
    np.testing.assert_allclose(rear[:, 2], enclosure["bounds"]["z_back"], atol=1e-9)
    geometry = resolve_geometry(config).geometry
    groups = _bspline_patch_phi_groups(
        len(geometry.inner_points), closed=geometry.closed,
        n_sectors=_open_sector_count(
            closed=geometry.closed, symmetry_planes=geometry.symmetry_planes,
        ),
    )
    solver_rim = np.vstack([sample_patch_boundary(
        geometry.inner_points, group, np.linspace(0, 1, 4097),
        surface_fit=geometry.surface_fit, preserve_grid=geometry.preserve_grid,
    ) for group in groups])
    solver_rim[:, 1] += 7.25
    assert np.max(cKDTree(solver_rim).query(mouth)[0]) < 0.3


def test_freeform_corner_preview_uses_the_corner_wall_mouth_fit():
    from scipy.spatial import cKDTree
    from test_freeform_corner_wall import case_config

    from hornlab_mesher.builders._occ import sample_patch_boundary
    from hornlab_mesher.builders.point_grid_surfaces import _bspline_patch_phi_groups
    from hornlab_mesher.config_builder import resolve_geometry
    from hornlab_mesher.corner_wall import _angular_poles

    config = case_config("c", "enclosure")
    geometry = resolve_geometry(config).geometry
    mouth = geometry.inner_points[:, -1, :]
    config["enclosure"]["dim_mm"] = (
        float(mouth[:, 0].max()), float(mouth[:, 1].max()),
        float(-mouth[:, 0].min()), float(-mouth[:, 1].min()),
    )
    geometry = resolve_geometry(config).geometry
    assert resolve_enclosure_bounds(config, warn_prefix=None)["flush_mount"]
    preview = build_preview_geometry(config, PreviewOptionsV1(lod="coarse"))
    side = next(surface for surface in preview.surfaces if surface.role == "enclosure.side")
    front = side.positions[:len(side.positions) // 2]
    groups = _bspline_patch_phi_groups(len(geometry.inner_points), closed=True)
    poles = _angular_poles(geometry.inner_points, groups, geometry.freeform_wall)
    solver_rim = np.vstack([sample_patch_boundary(
        poles, group, np.linspace(0, 1, 4097), surface_fit="approximate",
    ) for group in groups])
    assert np.max(cKDTree(solver_rim).query(front)[0]) < 0.3


def test_flush_preview_keeps_the_morph_start_station():
    from hornlab_mesher.config_builder import resolve_geometry

    config = rectangular_enclosure(margin=0, quadrants="1")
    config["morph"]["morphFixed"] = 0.4
    resolved = resolve_geometry(config)
    assert np.min(np.abs(np.asarray(resolved.flush_slice_map) - 0.4)) < 1e-9
    preview = build_preview_geometry(config, PreviewOptionsV1(lod="coarse"))
    assert "morph start" in preview.metadata["semantic_stations"]["inserted_first"]


def test_preserved_grid_flush_preview_keeps_control_vertices_and_mesh_wall(tmp_path):
    import meshio
    from scipy.spatial import cKDTree
    from test_point_grid_contract import _triangles_and_tags
    from test_preview_enclosure_edges import distance_to_triangles

    from hornlab_mesher.config_builder import build_from_config, resolve_geometry

    config = rectangular_enclosure(margin=0, quadrants="1")
    config["mesh"].update(angularSegments="12", lengthSegments="8", surfaceFit="approximate",
                          topology="legacy", preserve_grid=True)
    geometry = resolve_geometry(config).geometry
    preview = build_preview_geometry(config, PreviewOptionsV1(lod="fine"))
    inner = next(surface for surface in preview.surfaces if surface.role == "horn.inner")
    distances = cKDTree(inner.positions).query(geometry.inner_points.reshape(-1, 3))[0]
    assert np.max(distances) < 1e-7

    result = build_from_config(config, tmp_path / "preserved.msh")
    mesh = meshio.read(result.mesh_path)
    triangles, tags = _triangles_and_tags(mesh)
    wall = mesh.points[triangles[tags == 1]] * 1000.0
    selected = np.linspace(0, len(inner.positions) - 1, min(100, len(inner.positions)), dtype=int)
    mesh_distance = distance_to_triangles(inner.positions[selected], wall)
    assert max(mesh_distance) < 0.05
    fidelity = preview.metadata["fidelity"]["horn.inner"]
    if not fidelity["measurement_complete"]:
        assert fidelity["max_chord_error_mm_achieved"] is None


@pytest.mark.parametrize("angular,length", [(512, 8), (12, 600)])
def test_preserved_grid_preview_keeps_dense_control_vertices(angular, length):
    from scipy.spatial import cKDTree

    from hornlab_mesher.config_builder import resolve_geometry

    config = rectangular_enclosure(margin=0, quadrants="1")
    config["mesh"].update(angularSegments=str(angular), lengthSegments=str(length),
                          surfaceFit="approximate", topology="legacy", preserve_grid=True)
    geometry = resolve_geometry(config).geometry
    preview = build_preview_geometry(config, PreviewOptionsV1(lod="coarse"))
    inner = next(surface for surface in preview.surfaces if surface.role == "horn.inner")
    distances = cKDTree(inner.positions).query(geometry.inner_points.reshape(-1, 3))[0]
    assert np.max(distances) < 1e-7


@pytest.mark.parametrize("quadrants", ["1", "12", "14", "1234"])
@pytest.mark.parametrize("lod", ["coarse", "fine"])
def test_non_star_flush_caps_follow_the_ordered_rims(quadrants, lod):
    from scipy.spatial import cKDTree
    from test_preview_enclosure_edges import sampling_sensitive_enclosure

    from hornlab_mesher.config_builder import resolve_geometry

    config = sampling_sensitive_enclosure()
    config["mesh"].update(angular_segments=16, surface_fit="interpolate")
    config["profile"]["a"] = "45+20*sin(5*p)"
    full_rim = resolve_geometry(config).geometry.inner_points[:, -1, :]
    extent = float(np.max(np.abs(full_rim[:, :2])))
    config["enclosure"]["dim_mm"] = (extent,) * 4
    config["mesh"]["quadrants"] = quadrants
    preview = build_preview_geometry(config, PreviewOptionsV1(lod=lod))
    surfaces = {surface.role: surface for surface in preview.surfaces}
    side = surfaces["enclosure.side"]
    n_phi = len(side.positions) // 2
    horn = surfaces["horn.inner"]
    source_rim = horn.positions[:n_phi]
    rear_rim = side.positions[n_phi:]
    assert np.max(cKDTree(surfaces["source_cap"].positions).query(source_rim)[0]) < 1e-8
    assert np.max(cKDTree(surfaces["enclosure.rear"].positions).query(rear_rim)[0]) < 1e-8
    for role, ring in (("source_cap", source_rim), ("enclosure.rear", rear_rim)):
        polygon = ring[:, :2]
        if quadrants != "1234":
            polygon = np.vstack((polygon, (0.0, 0.0)))
        polygon_area = 0.5 * abs(np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1)
            - polygon[:, 1] * np.roll(polygon[:, 0], -1)
        ))
        surface = surfaces[role]
        triangles = surface.positions[surface.indices.reshape(-1, 3), :2]
        ab, ac = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        triangle_area = 0.5 * np.sum(np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]))
        assert triangle_area == pytest.approx(polygon_area, rel=1e-9)
    for surface in surfaces.values():
        triangles = surface.positions[surface.indices.reshape(-1, 3)]
        area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                        triangles[:, 2] - triangles[:, 0]), axis=1)
        assert np.min(area2) > 1e-10
    assert not preview.metadata["fidelity"]["source_cap"]["measurement_complete"]


def test_non_star_flush_original_full_domain_fixture():
    from test_preview_enclosure_edges import sampling_sensitive_enclosure

    from hornlab_mesher.config_builder import resolve_geometry

    config = sampling_sensitive_enclosure()
    config["mesh"].update(angular_segments=16, surface_fit="interpolate")
    config["profile"]["a"] = "45+35*sin(4*p)"
    rim = resolve_geometry(config).geometry.inner_points[:, -1, :]
    config["enclosure"]["dim_mm"] = (float(rim[:, 0].max()), float(rim[:, 1].max()),
                                     float(-rim[:, 0].min()), float(-rim[:, 1].min()))
    preview = build_preview_geometry(config, PreviewOptionsV1(lod="fine"))
    assert {"horn.inner", "source_cap", "enclosure.side", "enclosure.rear"} <= {
        surface.role for surface in preview.surfaces
    }


@pytest.mark.parametrize("surface_fit", ["interpolate", "approximate"])
@pytest.mark.parametrize("quadrants", ["1", "1234"])
def test_flush_preview_wall_samples_match_occ_patches(surface_fit, quadrants):
    import gmsh
    from test_preview_enclosure_edges import sampling_sensitive_enclosure

    from hornlab_mesher.builders._occ import (
        _patch_boundary_spline,
        add_bspline_patch,
        grid_v_parameters,
    )
    from hornlab_mesher.builders.point_grid_surfaces import (
        _bspline_patch_phi_groups,
        _open_sector_count,
        _snap_open_symmetry_grid,
    )
    from hornlab_mesher.config_builder import resolve_geometry
    from hornlab_mesher.preview.enclosure import (
        _fit_flush_grid,
        _flush_azimuth_parameters,
    )

    config = sampling_sensitive_enclosure()
    config["mesh"].update(quadrants=quadrants, surface_fit=surface_fit)
    resolved = resolve_geometry(config)
    geometry = resolved.geometry
    assert resolve_enclosure_bounds(config, warn_prefix=None)["flush_mount"]
    points = _snap_open_symmetry_grid(
        geometry.inner_points, closed=geometry.closed,
        symmetry_planes=geometry.symmetry_planes,
    )
    display = points.copy()
    _, groups, patch_indices, mapped, _, _ = _flush_azimuth_parameters(display[:, -1, :], geometry)
    _fit_flush_grid(display, resolved, resolved.flush_slice_map)
    degree_v = min(3, points.shape[1] - 1)
    v_params = grid_v_parameters(
        points, closed=geometry.closed if getattr(geometry, "quadrant_patch_fit", False) else None,
    )
    if surface_fit == "interpolate":
        v = v_params
    else:
        transposed = np.ascontiguousarray(points.transpose(1, 0, 2))
        spline = _patch_boundary_spline(
            transposed, transposed[:, -1, :], list(range(len(transposed))), -1, surface_fit,
        )
        v = np.asarray([np.mean(spline.t[i + 1:i + degree_v + 1])
                        for i in range(points.shape[1])])
    assert groups == _bspline_patch_phi_groups(
        len(points), closed=geometry.closed,
        n_sectors=_open_sector_count(closed=geometry.closed, symmetry_planes=geometry.symmetry_planes),
    )
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("flush-preview-wall")
        faces = [add_bspline_patch(points, group, degree_v=degree_v,
                                   surface_fit=surface_fit, v_params=v_params)
                 for group in groups]
        gmsh.model.occ.synchronize()
        for patch_index, face in enumerate(faces):
            mask = patch_indices == patch_index
            if not np.any(mask):
                continue
            lower, upper = map(np.asarray, gmsh.model.getParametrizationBounds(2, face))
            u = lower[0] + mapped[mask] * (upper[0] - lower[0])
            U, V = np.meshgrid(u, v)
            expected = np.asarray(gmsh.model.getValue(
                2, face, np.column_stack((U.ravel(), V.ravel())).ravel(),
            )).reshape(len(v), len(u), 3).transpose(1, 0, 2)
            np.testing.assert_allclose(display[mask], expected, rtol=0, atol=1e-8)
    finally:
        gmsh.finalize()
