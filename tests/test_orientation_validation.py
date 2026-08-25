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


from hornlab_mesher.step_import import (  # noqa: E402
    _repair_triangle_winding,
    _signed_volume,
    _signed_volume_noise_floor,
)


def _half_box_cut_on_x() -> tuple[np.ndarray, np.ndarray]:
    """A box cut by the single plane x=0, wound outward."""
    points = np.array(
        [
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 2, 1], [0, 3, 2],      # z = 0
            [4, 5, 6], [4, 6, 7],      # z = 10
            [0, 1, 5], [0, 5, 4],      # y = 0
            [3, 7, 6], [3, 6, 2],      # y = 10
            [1, 2, 6], [1, 6, 5],      # x = 10
        ],
        dtype=np.int64,
    )
    return points, triangles


def test_single_plane_cut_orientation_is_resolved_by_signed_volume():
    """The Fusion-common one-plane cut used to be left unjudged.

    Every free edge of a symmetry-reduced component lies on a coordinate plane
    through the origin, so the divergence-theorem cone terms vanish and the
    signed volume about the origin is a valid oracle -- unlike for an arbitrary
    open rim.
    """
    points, triangles = _half_box_cut_on_x()
    inverted = triangles[:, [0, 2, 1]]

    repaired, stats = _repair_triangle_winding(
        points,
        inverted,
        tags=np.zeros(len(inverted), dtype=np.int64),
        source_tags=set(),
        symmetry_planes=("x0",),
        tolerance=1e-6,
    )
    assert stats["unjudged_symmetry_components"] == 1
    assert stats["symmetry_volume_fallback_flipped"] == 1
    assert np.array_equal(np.sort(repaired, axis=1), np.sort(triangles, axis=1))
    # and the repaired winding is the outward one
    normals = np.cross(
        points[repaired[:, 1]] - points[repaired[:, 0]],
        points[repaired[:, 2]] - points[repaired[:, 0]],
    )
    centroids = points[repaired].mean(axis=1) - np.array([5.0, 5.0, 5.0])
    assert np.all(np.einsum("ij,ij->i", normals, centroids) > 0.0)


def test_a_near_degenerate_component_is_left_unresolved_not_flipped():
    """A sliver encloses nothing, so its signed-volume sign is float noise.

    Exactly 0.0 is not the only unresolved case. Collapsing the half box to a
    thickness of 1e-9 leaves a signed volume many orders below the component's
    own scale, and reading that sign would flip real normals on rounding.
    """
    points, triangles = _half_box_cut_on_x()
    flattened = points.copy()
    flattened[:, 0] *= 1.0e-9  # 10 x 10 x 1e-8 sliver on the x=0 cut plane

    volume = _signed_volume(flattened, triangles)
    assert volume != 0.0, "the hazard is a nonzero volume, not an exact zero"

    repaired, stats = _repair_triangle_winding(
        flattened,
        triangles,
        tags=np.zeros(len(triangles), dtype=np.int64),
        source_tags=set(),
        symmetry_planes=("x0",),
        tolerance=1e-6,
    )
    assert stats["unresolved_symmetry_components"] == 1
    assert stats["symmetry_volume_fallback_flipped"] == 0
    assert stats["symmetry_volume_fallback_kept"] == 0
    assert np.array_equal(repaired, triangles)


def test_a_healthy_reduced_shell_clears_the_noise_floor_by_orders_of_magnitude():
    """The guard must not start abstaining on the geometry it is meant to judge."""
    points, triangles = _half_box_cut_on_x()
    floor = _signed_volume_noise_floor(points, triangles)
    assert abs(_signed_volume(points, triangles)) > 1.0e6 * floor


def test_already_outward_single_plane_cut_is_left_alone():
    points, triangles = _half_box_cut_on_x()
    repaired, stats = _repair_triangle_winding(
        points,
        triangles,
        tags=np.zeros(len(triangles), dtype=np.int64),
        source_tags=set(),
        symmetry_planes=("x0",),
        tolerance=1e-6,
    )
    assert stats["symmetry_volume_fallback_kept"] == 1
    assert stats["symmetry_volume_fallback_flipped"] == 0
    assert np.array_equal(repaired, triangles)
