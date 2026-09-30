"""A thick normal offset must expose its exterior instead of internal loops."""

from __future__ import annotations

import numpy as np
import pytest

from hornlab_mesher.freeform import validate_outer_offset_grid
from hornlab_mesher.offset_envelope import (
    _equivalent_azimuth_rows,
    _ray_offset_radius,
    _triangles,
    regularize_outer_offset,
)
from hornlab_mesher.profile_sampling import _outer_offset_shell, build_point_grid_arrays


def _grooved_params(**overrides):
    return {
        "type": "OSSE",
        "L": 160.0,
        "r0": 12.0,
        "a0": 0.0,
        "k": 1.0,
        "s": 0.3,
        "n": 4.0,
        "q": 0.8,
        "gcurveType": 1,
        "gcurveWidth": "100 + 200*sin(2*p)^2",
        "gcurveSeN": 2,
        "gcurveDist": 0.7,
        "angularSegments": 64,
        "lengthSegments": 32,
        "samplingMode": "uniform",
        "wallThickness": 5.0,
        **overrides,
    }


def _grooved_config():
    return {
        "formula": "OSSE",
        "profile": {
            "L_mm": 160,
            "r0_mm": 12,
            "a0_deg": 0,
            "k": 1,
            "s": 0.3,
            "n": 4,
            "q": 0.8,
        },
        "gcurve": {
            "gcurveType": 1,
            "gcurveWidth": "100 + 200*sin(2*p)^2",
            "gcurveSeN": 2,
            "gcurveDist": 0.7,
        },
        "mesh": {
            "angular_segments": 64,
            "length_segments": 32,
            "samplingMode": "uniform",
            "wall_thickness_mm": 5,
            "throat_res_mm": 5,
            "mouth_res_mm": 10,
            "rear_res_mm": 15,
            "surface_fit": "interpolate",
            "max_triangles": 25000,
        },
    }


def _raw_offset(grid, wall):
    inner = grid["inner_grid"]
    return _outer_offset_shell(
        inner,
        wall,
        full_circle=grid["full_circle"],
        t_coordinates=np.asarray(grid["slice_map"]),
        phi_coordinates=np.broadcast_to(
            np.asarray(grid["angle_list"]), inner.shape[1::-1]
        ),
    )


def _distance_to_surface(point, triangles):
    """Independent closest-point calculation, without offset-ray intersections."""
    distance = np.inf
    for index in range(3):
        a = triangles[:, index]
        edge = triangles[:, (index + 1) % 3] - a
        squared = np.sum(edge * edge, axis=1)
        t = np.sum((point - a) * edge, axis=1) / np.where(squared > 0, squared, 1)
        closest = a + np.clip(t, 0, 1)[:, None] * edge
        distance = min(distance, np.linalg.norm(point - closest, axis=1).min())
    a = triangles[:, 0]
    ab = triangles[:, 1] - a
    ac = triangles[:, 2] - a
    normals = np.cross(ab, ac)
    norm = np.linalg.norm(normals, axis=1)
    normals /= np.where(norm > 0, norm, 1)[:, None]
    height = np.sum((point - a) * normals, axis=1)
    projected = point - height[:, None] * normals
    inside = norm > 0
    for index in range(3):
        edge = triangles[:, (index + 1) % 3] - triangles[:, index]
        inward = np.cross(edge, projected - triangles[:, index])
        inside &= np.sum(inward * normals, axis=1) >= -1e-10
    return min(distance, float(np.min(np.abs(height[inside]), initial=np.inf)))


@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
@pytest.mark.parametrize(
    "triangle, z, expected",
    [
        # A face interior, where the closest point is not on an edge.
        ([[10, -4, -4], [10, 4, -4], [10, 0, 4]], 0, 12.0),
        # Outside the face, a finite edge cylinder provides the boundary.
        ([[10, 0, -4], [10, 0, 4], [10, 4, 0]], 0, 12.0),
        # Beyond the edge endpoint, the vertex sphere provides the boundary.
        ([[10, 0, 0], [10, 4, 0], [10, 2, -4]], 1, 10 + np.sqrt(3)),
        # A collinear triangle still has a well-defined capsule offset.
        ([[10, 0, -4], [10, 0, 0], [10, 0, 4]], 0, 12.0),
    ],
)
def test_radial_offset_matches_analytic_face_edge_and_vertex(
    triangle, z, expected, scale
):
    actual = _ray_offset_radius(
        np.array([triangle], dtype=float) * scale,
        z * scale,
        np.array([1.0, 0.0, 0.0]),
        2.0 * scale,
    )
    assert actual == pytest.approx(expected * scale, rel=1e-12)


def test_the_farthest_component_is_the_exterior():
    triangles = np.array(
        [[[6, -4, -4], [6, 4, -4], [6, 0, 4]], [[10, -4, -4], [10, 4, -4], [10, 0, 4]]],
        dtype=float,
    )
    assert _ray_offset_radius(triangles, 0.0, np.array([1.0, 0.0, 0.0]), 2.0) == 12.0


@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
def test_accelerated_envelope_matches_exhaustive_intersections(scale):
    rng = np.random.default_rng(314159)
    angles = np.cumsum(rng.uniform(0.5, 1.5, 12))
    angles *= 2 * np.pi / (angles[-1] + 1.0)
    angles += 0.37
    stations = np.cumsum(rng.uniform(0.2, 2.0, 9))
    z = stations[None, :] * (1 + 0.07 * np.cos(angles)[:, None])
    r = 8 + z * (1 + 0.2 * np.sin(3 * angles)[:, None])
    inner = (
        np.stack((r * np.cos(angles)[:, None], r * np.sin(angles)[:, None], z), axis=2)
        * scale
    )
    wall = 2.0 * scale
    raw = _outer_offset_shell(inner, wall, full_circle=True)
    result = regularize_outer_offset(inner, raw, wall, full_circle=True)
    triangles = _triangles(inner, True)
    for row, angle in enumerate(angles):
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        for column in range(inner.shape[1]):
            point = result[row, column]
            expected = _ray_offset_radius(triangles, point[2], direction, wall)
            assert np.linalg.norm(point[:2]) == pytest.approx(
                expected, abs=2e-9 * scale, rel=1e-11
            )


@pytest.mark.parametrize("quadrants", ["1234", "12", "1"])
def test_nonleading_groove_fold_is_replaced_by_a_regular_exterior(quadrants):
    grid = build_point_grid_arrays(_grooved_params(quadrants=quadrants))
    inner, outer = grid["inner_grid"], grid["outer_grid"]
    raw = _raw_offset(grid, 5.0)
    with pytest.raises(ValueError, match="axial interval [4-9]"):
        validate_outer_offset_grid(inner, raw, full_circle=grid["full_circle"])
    assert grid["outer_offset_fold"] is None
    validate_outer_offset_grid(inner, outer, full_circle=grid["full_circle"])
    assert not np.allclose(raw, outer)
    # Every repaired vertex remains one requested thickness from the acoustic
    # triangulation; a radial expansion or smoothing shortcut fails this check.
    triangles = _triangles(inner, grid["full_circle"])
    for point in outer.reshape(-1, 3)[::29]:
        assert _distance_to_surface(point, triangles) == pytest.approx(5.0, abs=2e-7)
    acoustic_only = build_point_grid_arrays(
        _grooved_params(quadrants=quadrants, wallThickness=0.0)
    )
    np.testing.assert_array_equal(inner, acoustic_only["inner_grid"])


def test_rotation_and_scale_do_not_change_the_envelope():
    grid = build_point_grid_arrays(
        _grooved_params(angularSegments=32, lengthSegments=16)
    )
    raw = _raw_offset(grid, 5.0)
    angle = 0.41
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    expected = (
        regularize_outer_offset(grid["inner_grid"], raw, 5.0, full_circle=True)
        @ rotation.T
    )
    for scale in (0.01, 10.0):
        actual = regularize_outer_offset(
            grid["inner_grid"] @ rotation.T * scale,
            raw @ rotation.T * scale,
            5.0 * scale,
            full_circle=True,
        )
        np.testing.assert_allclose(actual / scale, expected, atol=2e-8, rtol=1e-10)


@pytest.mark.parametrize("axis", [0, 1])
def test_a_symmetric_grooved_shell_remains_reflection_symmetric(axis):
    grid = build_point_grid_arrays(_grooved_params())
    reflection = np.eye(3)
    reflection[axis, axis] = -1
    # Match the actual reflected input positions, including the seam, without
    # presuming a row permutation from the implementation under test.
    inner = grid["inner_grid"]
    mirrored = inner @ reflection
    matching = np.argmin(
        np.linalg.norm(mirrored[:, None, 0] - inner[None, :, 0], axis=2), axis=1
    )
    np.testing.assert_allclose(mirrored, inner[matching], atol=1e-10)
    np.testing.assert_allclose(
        grid["outer_grid"] @ reflection, grid["outer_grid"][matching], atol=2e-8
    )


@pytest.mark.parametrize("perturbation", [None, "surface", "target_z"])
def test_rotational_reuse_requires_the_whole_surface_and_target_stations(perturbation):
    grid = build_point_grid_arrays(_grooved_params())
    inner, outer = grid["inner_grid"].copy(), _raw_offset(grid, 5.0)
    if perturbation == "surface":
        inner[3, 5, 0] += 1e-6
    if perturbation == "target_z":
        outer[3, 5, 2] += 1e-6
    representatives = _equivalent_azimuth_rows(
        inner,
        outer[:, :, 2],
        np.asarray(grid["angle_list"]),
        full_circle=True,
        margin=1e-11,
    )
    if perturbation is None:
        assert len(np.unique(representatives)) == len(inner) // 4
    else:
        np.testing.assert_array_equal(representatives, np.arange(len(inner)))


def test_a_leading_meridian_loop_also_uses_the_exterior():
    phi = np.arange(32) * (2 * np.pi / 32)
    z = np.linspace(0, 30, 61)
    radius = 12 + 0.2 * z**2
    inner = np.stack(
        (
            np.cos(phi)[:, None] * radius,
            np.sin(phi)[:, None] * radius,
            np.broadcast_to(z, (len(phi), len(z))),
        ),
        axis=2,
    )
    raw = _outer_offset_shell(inner, 5.0, full_circle=True)
    with pytest.raises(ValueError, match="axial interval 0"):
        validate_outer_offset_grid(inner, raw, full_circle=True)
    repaired = regularize_outer_offset(inner, raw, 5.0, full_circle=True)
    validate_outer_offset_grid(inner, repaired, full_circle=True)
    triangles = _triangles(inner, True)
    for point in repaired[0, ::5]:
        assert _distance_to_surface(point, triangles) == pytest.approx(5.0, abs=1e-8)


def test_rotation_reuse_checks_stations_after_rollback_remapping():
    phi = np.arange(16) * (2 * np.pi / 16)
    z = np.arange(4.0)
    r = 12 + 2 * z
    inner = np.stack(
        (
            np.cos(phi)[:, None] * r,
            np.sin(phi)[:, None] * r,
            np.broadcast_to(z, (len(phi), len(z))),
        ),
        axis=2,
    )
    outer = inner.copy()
    outer[:, :, 2] = [0, 1, 1, 3]
    margin = 64 * np.finfo(float).eps * np.max(np.abs(inner))
    outer[:, 2, 2] += 0.75 * margin
    outer[4, 2, 2] += 0.5 * margin
    result = regularize_outer_offset(inner, outer, 1.0, full_circle=True)
    triangles = _triangles(inner, True)
    for row in (0, 4):
        point = result[row, 2]
        expected = _ray_offset_radius(
            triangles, point[2], np.array([np.cos(phi[row]), np.sin(phi[row]), 0]), 1.0
        )
        assert np.linalg.norm(point[:2]) == pytest.approx(expected, abs=1e-10)


def test_rotational_reuse_shares_the_station_that_the_radius_was_evaluated_at():
    grid = build_point_grid_arrays(_grooved_params())
    inner, raw = grid["inner_grid"], _raw_offset(grid, 5.0)
    margin = 64 * np.finfo(float).eps * np.max(np.abs(inner))
    raw[16, 8, 2] += 2 * margin
    result = regularize_outer_offset(inner, raw, 5.0, full_circle=True)
    assert result[16, 8, 2] == result[0, 8, 2]
    triangles = _triangles(inner, True)
    assert _distance_to_surface(result[16, 8], triangles) == pytest.approx(
        5.0, abs=1e-8
    )


def test_near_critical_throat_offsets_do_not_split_symmetric_rollback_decisions():
    params = dict(
        type="OSSE",
        L=140,
        r0=12.7,
        a0=0,
        a=70,
        k=1,
        s=0.4,
        n=4,
        q=0.8,
        angularSegments=64,
        lengthSegments=64,
        samplingMode="uniform",
        wallThickness=0,
    )
    grid = build_point_grid_arrays(params)
    inner = grid["inner_grid"]
    unit_offset = _raw_offset(grid, 1.0)
    delta_inner = inner[0, 1, 2] - inner[0, 0, 2]
    delta_offset = (unit_offset[0, 1, 2] - inner[0, 1, 2]) - (
        unit_offset[0, 0, 2] - inner[0, 0, 2]
    )
    wall = -delta_inner / delta_offset
    result = build_point_grid_arrays({**params, "wallThickness": wall})
    assert result["outer_offset_fold"] is None
    validate_outer_offset_grid(inner, result["outer_grid"], full_circle=True)


def test_healthy_normal_offsets_are_unchanged(monkeypatch):
    import hornlab_mesher.profile_sampling as sampling

    def unexpected(*args, **kwargs):
        pytest.fail("healthy shell should retain its original normal offset")

    monkeypatch.setattr(sampling, "regularize_outer_offset", unexpected)
    grid = build_point_grid_arrays(
        _grooved_params(gcurveWidth=200.0, wallThickness=1.0)
    )
    np.testing.assert_array_equal(grid["outer_grid"], _raw_offset(grid, 1.0))


def test_failed_envelope_validation_retains_the_fold_diagnostic(monkeypatch):
    import hornlab_mesher.profile_sampling as sampling

    monkeypatch.setattr(
        sampling, "regularize_outer_offset", lambda inner, outer, *args, **kwargs: outer
    )
    assert (
        "normal flip" in build_point_grid_arrays(_grooved_params())["outer_offset_fold"]
    )


def test_the_repaired_grooved_shell_builds_a_bounded_manifold_mesh(tmp_path):
    import meshio
    from hornlab_mesher.config_builder import build_from_config
    from hornlab_mesher.mesher import _triangles_and_physical_tags
    from hornlab_mesher.normals import validate_orientation

    result = build_from_config(_grooved_config(), tmp_path / "grooved.msh")
    assert not result.metadata.get("outerOffsetFold")
    mesh = meshio.read(result.mesh_path)
    triangles, tags = _triangles_and_physical_tags(mesh)
    report = validate_orientation(
        mesh.points,
        triangles,
        tags,
        require_watertight=True,
        require_edge_consistency=True,
    )
    assert report.watertight and report.edge_consistent
    assert result.n_triangles < 25000


@pytest.mark.parametrize("lod", ["coarse", "fine"])
@pytest.mark.parametrize("rounded_morph", [False, True])
def test_folded_preview_repairs_only_the_selected_grid(lod, rounded_morph, monkeypatch):
    import hornlab_mesher.profile_sampling as sampling
    import hornlab_mesher.preview.api as preview_api
    from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry

    # This test isolates the render repair. Settled canonical measurements
    # deliberately repair solve/CAD geometry too, covered by the C2 tests.
    monkeypatch.setattr(preview_api, "dimension_metadata", lambda *args: {})
    config = _grooved_config()
    if rounded_morph:
        config["morph"] = {
            "morphTarget": 1,
            "morphWidth": 300,
            "morphHeight": 300,
            "morphCorner": 30,
            "morphFixed": 0.6,
        }
    repaired_shapes = []
    original = sampling.regularize_outer_offset

    def tracked(inner, outer, wall, *, full_circle):
        repaired_shapes.append(inner.shape)
        return original(inner, outer, wall, full_circle=full_circle)

    monkeypatch.setattr(sampling, "regularize_outer_offset", tracked)
    result = build_preview_geometry(config, PreviewOptionsV1(lod=lod))
    outer = next(surface for surface in result.surfaces if surface.role == "horn.outer")
    assert len(repaired_shapes) == 1
    # The master is deliberately denser than the render grid. Repairing it
    # first discarded most of the expensive result on every drag frame.
    assert repaired_shapes[0][0] * repaired_shapes[0][1] == len(outer.positions)


def test_deferred_repair_is_independent_of_vertical_placement():
    from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry

    config = _grooved_config()
    config["morph"] = {
        "morphTarget": 1,
        "morphWidth": 300,
        "morphHeight": 300,
        "morphCorner": 30,
        "morphFixed": 0.6,
    }
    centered = build_preview_geometry(config, PreviewOptionsV1(lod="coarse"))
    config["mesh"]["vertical_offset_mm"] = 80.0
    translated = build_preview_geometry(config, PreviewOptionsV1(lod="coarse"))
    original = next(
        surface for surface in centered.surfaces if surface.role == "horn.outer"
    )
    moved = next(
        surface for surface in translated.surfaces if surface.role == "horn.outer"
    )
    np.testing.assert_allclose(
        np.asarray(moved.positions) - [0.0, 80.0, 0.0], original.positions, atol=2e-8
    )
