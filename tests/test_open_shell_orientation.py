from __future__ import annotations

from collections import defaultdict

import meshio
import numpy as np
import pytest

from hornlab_mesher import (
    CrossSection,
    MeshDensity,
    OsseHornGeometry,
    build_from_config,
    build_mesh,
)


def _triangles_and_tags(path):
    mesh = meshio.read(path)
    return (
        np.asarray(mesh.points, dtype=np.float64),
        np.asarray(mesh.cells_dict["triangle"], dtype=np.int64),
        np.asarray(
            mesh.cell_data_dict["gmsh:physical"]["triangle"], dtype=np.int32
        ),
    )


def _assert_open_wall_points_into_bore(path):
    """Independent throat-axis check, separate from mesher orientation code."""

    points, triangles, tags = _triangles_and_tags(path)
    wall = triangles[tags == 1]
    corners = points[wall]
    area_vectors = np.cross(
        corners[:, 1] - corners[:, 0],
        corners[:, 2] - corners[:, 0],
    )
    centroids = np.mean(corners, axis=1)
    radii = np.linalg.norm(centroids[:, :2], axis=1)
    throat = radii <= np.quantile(radii, 0.1)
    toward_axis = -centroids[throat, :2]
    toward_axis /= np.linalg.norm(toward_axis, axis=1)[:, None]
    projections = np.sum(area_vectors[throat, :2] * toward_axis, axis=1)
    assert float(np.sum(projections) / np.sum(np.abs(projections))) > 0.99
    assert float(np.mean(projections > 0.0)) > 0.99

    source = triangles[tags == 2]
    source_corners = points[source]
    source_z = np.sum(
        np.cross(
            source_corners[:, 1] - source_corners[:, 0],
            source_corners[:, 2] - source_corners[:, 0],
        )[:, 2]
    )
    assert source_z > 0.0


BASE_MESH = {
    "angular_segments": 32,
    "length_segments": 16,
    "throat_res_mm": 5.0,
    "mouth_res_mm": 12.0,
    "rear_res_mm": 12.0,
    "quadrants": "1234",
    "max_triangles": 100_000,
    "allow_large_mesh": True,
    "scale_to_metres": False,
}


@pytest.mark.parametrize(
    "config",
    [
        {
            "formula": "OSSE",
            "mode": "bare",
            "profile": {
                "L_mm": 150.0,
                "r0_mm": 12.7,
                "a_deg": 45.0,
                "a0_deg": 0.0,
                "q": 1.0,
            },
            "mesh": {
                **BASE_MESH,
                # This density reproducibly took the detached-cap/inverted
                # branch before open-shell parameterisation anchoring.
                "throat_res_mm": 6.0,
                "mouth_res_mm": 6.0,
            },
        },
        {
            "formula": "OSSE",
            "mode": "bare",
            "profile": {
                "L_mm": 150.0,
                "r0_mm": 12.7,
                "a_deg": 45.0,
                "a0_deg": 0.0,
                "q": 1.0,
            },
            "mesh": {
                **BASE_MESH,
                "topology": "legacy",
                "preserve_grid": True,
            },
        },
        {
            "formula": "R-OSSE",
            "mode": "bare",
            "profile": {
                "R_mm": 150.0,
                "r0_mm": 12.7,
                "a_deg": 45.0,
                "a0_deg": 10.0,
                "k": 1.0,
                "q": 1.0,
                "m": 0.85,
                "r": 0.4,
                "b": 0.2,
            },
            "mesh": BASE_MESH,
        },
        {
            "formula": "ICW",
            "mode": "bare",
            "profile": {
                "formula": "ICW",
                "r0_mm": 12.7,
                "a0_deg": 12.0,
                "termination": "rollback",
                "theta1_deg": 160.0,
                "R_mm": 110.0,
                "depth": 90.0,
            },
            "mesh": BASE_MESH,
        },
        {
            "formula": "OSSE",
            "mode": "bare",
            "profile": {
                "L_mm": 100.0,
                "r0_mm": 15.0,
                "a_deg": 40.0,
                "a0_deg": 8.0,
            },
            "cross_section": {"exponent": 4.0, "aspect_ratio": 1.3},
            "mesh": BASE_MESH,
        },
    ],
    ids=("osse-detached-cap", "legacy-faceted", "rosse-rollback", "icw-rollback", "superellipse"),
)
def test_bare_point_grid_builders_orient_wall_into_bore(tmp_path, config):
    output = tmp_path / "bare.msh"

    build_from_config(config, output)

    _assert_open_wall_points_into_bore(output)


def test_direct_axisymmetric_builder_orients_wall_into_bore(tmp_path):
    output = tmp_path / "direct.msh"

    build_mesh(
        OsseHornGeometry(
            L_mm=100.0,
            r0_mm=15.0,
            a_deg=40.0,
            a0_deg=8.0,
            cross_section=CrossSection(exponent=4.0, aspect_ratio=1.3),
            n_phi=40,
            n_axial=14,
        ),
        MeshDensity(
            throat_res_mm=5.0,
            mouth_res_mm=12.0,
            max_triangles=100_000,
            allow_large_mesh=True,
        ),
        output,
        scale_to_metres=False,
    )

    _assert_open_wall_points_into_bore(output)


@pytest.mark.parametrize("quadrants", ("1", "12", "14"))
def test_curved_source_reduced_bare_mesh_keeps_axis_winding(tmp_path, quadrants):
    """A biased reduced cap centroid must not become an inferred frame."""

    output = tmp_path / f"reduced-{quadrants}.msh"
    build_from_config(
        {
            "formula": "OSSE",
            "mode": "bare",
            "profile": {
                "L_mm": 100.0,
                "r0_mm": 12.7,
                "a_deg": 45.0,
                "a0_deg": 15.5,
            },
            "mesh": {**BASE_MESH, "quadrants": quadrants},
        },
        output,
    )

    _assert_open_wall_points_into_bore(output)


def test_closed_freestanding_orientation_contract_is_unchanged(tmp_path):
    output = tmp_path / "freestanding.msh"
    build_from_config(
        {
            "formula": "OSSE",
            "mode": "freestanding",
            "profile": {
                "L_mm": 80.0,
                "r0_mm": 12.7,
                "a_deg": 40.0,
                "a0_deg": 8.0,
            },
            "mesh": {
                **BASE_MESH,
                "wall_thickness_mm": 5.0,
            },
        },
        output,
    )
    points, triangles, tags = _triangles_and_tags(output)

    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    assert float(np.sum(p0 * np.cross(p1, p2)) / 6.0) > 0.0

    edge_directions: dict[tuple[int, int], list[int]] = defaultdict(list)
    for triangle in triangles:
        for start, end in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            a, b = int(start), int(end)
            edge_directions[tuple(sorted((a, b)))].append(1 if a < b else -1)
    assert all(
        len(directions) == 2 and directions[0] != directions[1]
        for directions in edge_directions.values()
    )

    source = triangles[tags == 2]
    source_corners = points[source]
    assert (
        np.sum(
            np.cross(
                source_corners[:, 1] - source_corners[:, 0],
                source_corners[:, 2] - source_corners[:, 0],
            )[:, 2]
        )
        > 0.0
    )
