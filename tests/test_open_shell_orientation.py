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
from hornlab_mesher.normals import (
    MeshOrientationError,
    open_shell_bore_alignment,
    validate_orientation,
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
    points, triangles, tags = _triangles_and_tags(output)
    baseline = open_shell_bore_alignment(points, triangles, tags)
    assert baseline == 1.0

    # The cap centroid of a sector is off the horn's centreline. Exercise the
    # detector itself on that asymmetric frame, including the rigid transforms
    # and post-build vertical placement it claims not to depend on.
    translated = points + np.array([173.0, -91.0, 247.0])
    vertically_offset = points + np.array([0.0, 321.5, 0.0])
    rotated = points[:, [2, 0, 1]]  # proper 3-cycle rotation: det = +1
    assert open_shell_bore_alignment(translated, triangles, tags) == baseline
    assert open_shell_bore_alignment(vertically_offset, triangles, tags) == baseline
    assert open_shell_bore_alignment(rotated, triangles, tags) == baseline


# The reported geometry, at the densities whose Gmsh cap welding used to decide
# the branch. Under the old repair path 6.0/4.5/3.0 came out with the wall sheet
# inverted and 5.0/4.0/3.4 came out correct, giving a factor 3.5 on throat
# impedance across a mesh-density sweep of one horn.
REPORTED_SWEEP_DENSITIES = (6.0, 5.0, 4.5, 4.0, 3.4, 3.0)


def test_bare_shell_orientation_is_density_independent(tmp_path):
    """One horn, many densities: the winding must not depend on the mesh."""

    signed_volumes = {}
    for throat_res_mm in REPORTED_SWEEP_DENSITIES:
        output = tmp_path / f"sweep-{throat_res_mm}.msh"
        build_from_config(
            {
                "formula": "OSSE",
                "mode": "bare",
                "profile": {
                    "L_mm": 150.0,
                    "r0_mm": 12.7,
                    "a_deg": 45.0,
                    "a0_deg": 0.0,
                    "k": 1.0,
                    "n": 4.0,
                    "q": 1.0,
                    "s": 0.0,
                },
                "mesh": {
                    **BASE_MESH,
                    "angular_segments": 96,
                    "length_segments": 32,
                    "throat_res_mm": throat_res_mm,
                    "mouth_res_mm": 6.0,
                    "rear_res_mm": 6.0,
                },
            },
            output,
        )

        _assert_open_wall_points_into_bore(output)

        points, triangles, _tags = _triangles_and_tags(output)
        signed_volumes[throat_res_mm] = float(
            np.sum(
                points[triangles[:, 0]]
                * np.cross(points[triangles[:, 1]], points[triangles[:, 2]])
            )
        )

    # The reported symptom, pinned directly: the signed-volume sign correlated
    # 11/11 with the branch on this profile. It is only a symptom -- rollback
    # profiles hold the same wall contract with the opposite sign -- so it is
    # asserted as a per-geometry constant, never as the contract itself.
    signs = {np.sign(volume) for volume in signed_volumes.values()}
    assert len(signs) == 1, f"signed volume changed sign across densities: {signed_volumes}"


def test_inverted_bare_wall_fails_validation(tmp_path):
    """The validator must reject the broken branch, not merely avoid emitting it."""

    output = tmp_path / "bare.msh"
    build_from_config(
        {
            "formula": "OSSE",
            "mode": "bare",
            "profile": {"L_mm": 150.0, "r0_mm": 12.7, "a_deg": 45.0, "a0_deg": 0.0},
            "mesh": BASE_MESH,
        },
        output,
    )
    points, triangles, tags = _triangles_and_tags(output)

    assert (
        validate_orientation(
            points,
            triangles,
            tags,
            require_open_shell_bore_normal=True,
        ).open_shell_bore_alignment
        == 1.0
    )

    wall = tags == 1
    inverted = triangles.copy()
    inverted[wall] = inverted[wall][:, [0, 2, 1]]

    with pytest.raises(MeshOrientationError, match="do not face the bore"):
        validate_orientation(
            points,
            inverted,
            tags,
            require_edge_consistency=False,
            require_open_shell_bore_normal=True,
        )


@pytest.mark.parametrize(
    ("formula", "profile"),
    (
        (
            "R-OSSE",
            {
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
        ),
        (
            "ICW",
            {
                "formula": "ICW",
                "r0_mm": 12.7,
                "a0_deg": 12.0,
                "termination": "rollback",
                "theta1_deg": 160.0,
                "R_mm": 110.0,
                "depth": 90.0,
            },
        ),
    ),
    ids=("rosse", "icw"),
)
def test_rollback_bore_alignment_disagrees_with_signed_volume(
    tmp_path, formula, profile
):
    """Signed volume is positive on a correctly wound rollback; the check is not."""

    output = tmp_path / f"{formula.lower()}-rollback.msh"
    build_from_config(
        {
            "formula": formula,
            "mode": "bare",
            "profile": profile,
            "mesh": BASE_MESH,
        },
        output,
    )
    points, triangles, tags = _triangles_and_tags(output)

    _assert_open_wall_points_into_bore(output)
    assert (
        float(
            np.sum(
                points[triangles[:, 0]]
                * np.cross(points[triangles[:, 1]], points[triangles[:, 2]])
            )
        )
        > 0.0
    )
    # Moving the sum to a material point on the throat wall does not rescue
    # the sign: both rollback families remain positive when correctly wound.
    wall_vertices = np.unique(triangles[tags == 1])
    throat_vertex = points[
        wall_vertices[
            np.argmin(np.linalg.norm(points[wall_vertices, :2], axis=1))
        ]
    ]
    shifted = points - throat_vertex
    assert (
        float(
            np.sum(
                shifted[triangles[:, 0]]
                * np.cross(
                    shifted[triangles[:, 1]], shifted[triangles[:, 2]]
                )
            )
        )
        > 0.0
    )
    assert open_shell_bore_alignment(points, triangles, tags) == 1.0


def test_unjudgeable_bore_alignment_fails_closed():
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    cases = (
        (
            np.array([[0, 1, 2]], dtype=np.int64),
            np.array([1], dtype=np.int32),
        ),
        (
            np.array([[3, 4, 5]], dtype=np.int64),
            np.array([2], dtype=np.int32),
        ),
        (
            np.array([[0, 1, 2], [3, 4, 5], [3, 5, 4]], dtype=np.int64),
            np.array([1, 2, 2], dtype=np.int32),
        ),
        (
            np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
            np.array([1, 2], dtype=np.int32),
        ),
    )

    for triangles, tags in cases:
        assert open_shell_bore_alignment(points, triangles, tags) is None
        with pytest.raises(MeshOrientationError, match="no measurable throat collar"):
            validate_orientation(
                points,
                triangles,
                tags,
                require_edge_consistency=False,
                require_positive_volume=False,
                require_source_normal=False,
                require_open_shell_bore_normal=True,
            )


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
