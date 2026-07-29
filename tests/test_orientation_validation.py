from __future__ import annotations

import numpy as np
import pytest
import meshio

from hornlab_mesher.mesher import _postprocess_mesh
from hornlab_mesher.normals import (
    MeshOrientationError,
    repair_orientation,
    validate_orientation,
)
from hornlab_mesher.tags import PhysicalGroup


def _tetrahedron() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ],
        dtype=np.int64,
    )
    tags = np.full(len(triangles), int(PhysicalGroup.RIGID_WALL), dtype=np.int32)
    return points, triangles, tags


def _source_disc() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    tags = np.full(len(triangles), int(PhysicalGroup.PRIMARY_SOURCE), dtype=np.int32)
    return points, triangles, tags


def test_validate_orientation_reports_watertight_consistent_positive_volume():
    points, triangles, tags = _tetrahedron()

    report = validate_orientation(
        points,
        triangles,
        tags,
        require_watertight=True,
        require_edge_consistency=True,
        require_source_normal=False,
    )

    assert report.watertight
    assert report.edge_consistent
    assert report.signed_volume > 0.0


def test_validate_orientation_rejects_negative_signed_volume():
    points, triangles, tags = _tetrahedron()
    flipped = triangles[:, [0, 2, 1]]

    with pytest.raises(MeshOrientationError, match="signed volume is negative"):
        validate_orientation(
            points,
            flipped,
            tags,
            require_watertight=True,
            require_edge_consistency=True,
            require_source_normal=False,
        )


def test_validate_orientation_rejects_boundary_edges_when_watertight_required():
    points, triangles, tags = _tetrahedron()

    with pytest.raises(MeshOrientationError, match="not watertight"):
        validate_orientation(
            points,
            triangles[:-1],
            tags[:-1],
            require_watertight=True,
            require_edge_consistency=True,
            require_source_normal=False,
        )


def test_validate_orientation_rejects_inconsistent_shared_edges():
    points, triangles, tags = _tetrahedron()
    inconsistent = triangles.copy()
    inconsistent[0] = inconsistent[0, [0, 2, 1]]

    with pytest.raises(MeshOrientationError, match="inconsistent shared edges"):
        validate_orientation(
            points,
            inconsistent,
            tags,
            require_watertight=True,
            require_edge_consistency=True,
            require_source_normal=False,
            require_positive_volume=False,
        )


def test_validate_orientation_rejects_reversed_primary_source_normal():
    points, triangles, tags = _source_disc()

    report = validate_orientation(
        points,
        triangles,
        tags,
        require_positive_volume=False,
    )
    assert report.source_normal_projection > 0.0

    with pytest.raises(MeshOrientationError, match="primary source normals"):
        validate_orientation(
            points,
            triangles[:, [0, 2, 1]],
            tags,
            require_positive_volume=False,
        )


def test_repair_orientation_is_legacy_opt_in():
    points, triangles, tags = _tetrahedron()
    flipped = triangles[:, [0, 2, 1]]

    repaired, stats = repair_orientation(points, flipped, tags)

    assert stats["flipped_global"] == len(flipped)
    assert np.array_equal(repaired, triangles)


def test_repair_orientation_restores_shared_edge_consistency():
    points, triangles, tags = _tetrahedron()
    inconsistent = triangles.copy()
    inconsistent[0] = inconsistent[0, [0, 2, 1]]

    repaired, stats = repair_orientation(points, inconsistent, tags)
    report = validate_orientation(
        points,
        repaired,
        tags,
        require_watertight=True,
        require_edge_consistency=True,
        require_source_normal=False,
    )

    assert stats["flipped_consistency"] > 0
    assert report.edge_consistent
    assert report.signed_volume > 0.0


@pytest.mark.parametrize("reverse_wall", [False, True])
def test_repair_orientation_anchors_detached_open_wall_to_parameterisation(
    reverse_wall,
):
    """A detached cap must not leave an open wall at signed-volume mercy."""

    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    throat = np.column_stack(
        (np.cos(angles), np.sin(angles), np.zeros_like(angles))
    )
    mouth = np.column_stack(
        (2.0 * np.cos(angles), 2.0 * np.sin(angles), np.ones_like(angles))
    )
    # Deliberately duplicate the source rim so source and wall are separate
    # edge-connected components, matching the failing Gmsh cap-weld branch.
    source_rim = throat.copy()
    source_center = np.array([[0.0, 0.0, 0.0]])
    points = np.vstack((throat, mouth, source_rim, source_center))

    wall: list[list[int]] = []
    references: list[np.ndarray] = []
    expected_normals: list[np.ndarray] = []
    for i in range(len(angles)):
        ni = (i + 1) % len(angles)
        wall.extend(
            (
                [i, ni + len(angles), ni],
                [i, i + len(angles), ni + len(angles)],
            )
        )
        references.append(
            np.mean(points[[i, ni, i + len(angles), ni + len(angles)]], axis=0)
        )
        phi = angles[i] + np.pi / len(angles)
        expected_normals.append(np.array([-np.cos(phi), -np.sin(phi), 1.0]))

    center_index = len(points) - 1
    source = [
        [center_index, 2 * len(angles) + i, 2 * len(angles) + (i + 1) % len(angles)]
        for i in range(len(angles))
    ]
    triangles = np.asarray([*wall, *source], dtype=np.int64)
    tags = np.asarray(
        [int(PhysicalGroup.RIGID_WALL)] * len(wall)
        + [int(PhysicalGroup.PRIMARY_SOURCE)] * len(source),
        dtype=np.int32,
    )
    if reverse_wall:
        triangles[: len(wall)] = triangles[: len(wall), [0, 2, 1]]

    repaired, _stats = repair_orientation(
        points,
        triangles,
        tags,
        open_shell_wall_points_mm=np.asarray(references),
        open_shell_wall_normals=np.asarray(expected_normals),
    )

    wall_triangles = repaired[tags == int(PhysicalGroup.RIGID_WALL)]
    corners = points[wall_triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    centroids = np.mean(corners, axis=1)
    assert np.all(np.sum(normals[:, :2] * -centroids[:, :2], axis=1) > 0.0)

    source_triangles = repaired[tags == int(PhysicalGroup.PRIMARY_SOURCE)]
    source_corners = points[source_triangles]
    source_projection = np.sum(
        np.cross(
            source_corners[:, 1] - source_corners[:, 0],
            source_corners[:, 2] - source_corners[:, 0],
        )[:, 2]
    )
    assert source_projection > 0.0


def test_postprocess_normalizes_generated_inward_winding(tmp_path):
    points, triangles, tags = _tetrahedron()
    flipped = triangles[:, [0, 2, 1]]
    raw_path = tmp_path / "raw_inward.msh"
    out_path = tmp_path / "outward.msh"

    meshio.write(
        raw_path,
        meshio.Mesh(
            points=points,
            cells=[("triangle", flipped)],
            cell_data={
                "gmsh:physical": [tags],
                "gmsh:geometrical": [tags],
            },
            field_data={
                "SD1G0": np.array([int(PhysicalGroup.RIGID_WALL), 2], dtype=np.int32),
            },
        ),
        file_format="gmsh22",
        binary=False,
    )

    _postprocess_mesh(raw_path, out_path, source_axis="z", scale_to_metres=False)

    processed = meshio.read(out_path)
    repaired = np.asarray(processed.cells_dict["triangle"], dtype=np.int64)
    report = validate_orientation(
        np.asarray(processed.points, dtype=np.float64),
        repaired,
        np.asarray(processed.cell_data_dict["gmsh:physical"]["triangle"], dtype=np.int32),
        require_watertight=True,
        require_edge_consistency=True,
        require_source_normal=False,
    )
    assert report.signed_volume > 0.0


def test_postprocess_rejects_inconsistent_open_mesh(tmp_path):
    """Boundary edges must not mask an unrepairable winding contradiction."""

    radius = 2.0
    half_width = 0.35
    points = np.asarray(
        [
            (
                (radius + u * np.cos(t / 2.0)) * np.cos(t),
                (radius + u * np.cos(t / 2.0)) * np.sin(t),
                u * np.sin(t / 2.0),
            )
            for t in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
            for u in (half_width, -half_width)
        ],
        dtype=np.float64,
    )
    # Three strips with the final endpoints identified in reverse form a
    # Möbius band: it is open, but its shared edges cannot all be consistently
    # wound. The repair pass necessarily leaves one contradictory edge.
    triangles = np.asarray(
        [
            [0, 2, 3],
            [0, 3, 1],
            [2, 4, 5],
            [2, 5, 3],
            [4, 1, 0],
            [4, 0, 5],
        ],
        dtype=np.int64,
    )
    tags = np.full(len(triangles), int(PhysicalGroup.RIGID_WALL), dtype=np.int32)
    raw_path = tmp_path / "raw-mobius.msh"
    out_path = tmp_path / "processed-mobius.msh"
    meshio.write(
        raw_path,
        meshio.Mesh(
            points=points,
            cells=[("triangle", triangles)],
            cell_data={
                "gmsh:physical": [tags],
                "gmsh:geometrical": [tags],
            },
            field_data={
                "SD1G0": np.array(
                    [int(PhysicalGroup.RIGID_WALL), 2], dtype=np.int32
                ),
            },
        ),
        file_format="gmsh22",
        binary=False,
    )

    with pytest.raises(MeshOrientationError, match="inconsistent shared edges"):
        _postprocess_mesh(
            raw_path,
            out_path,
            source_axis="z",
            scale_to_metres=False,
        )

    assert not out_path.exists()
