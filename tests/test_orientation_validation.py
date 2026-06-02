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
