"""Compatibility of the generated and imported mesh repair callers."""

import numpy as np

from hornlab_mesher import mesh_repair, mesher, normals


def test_weld_keeps_transitive_lowest_index_across_cells():
    points = np.asarray([
        [0.008, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.004, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    triangles = np.asarray([[0, 3, 4], [1, 3, 4], [2, 3, 4]])
    expected = [[0, 3, 4], [0, 3, 4], [0, 3, 4]]
    for weld in (mesher._weld_near_duplicate_vertices, mesh_repair._weld_near_duplicate_vertices):
        assert weld(points, triangles).tolist() == expected


def test_weld_preserves_the_callers_nonfinite_distance_behavior():
    points = np.asarray([[np.nan, 0.0, 0.0], [np.nan, 0.0, 0.0], [1.0, 0.0, 0.0]])
    triangles = np.asarray([[0, 1, 2]])
    with np.errstate(invalid="ignore", over="ignore"):
        assert mesher._weld_near_duplicate_vertices(points, triangles).tolist() == [[0, 0, 2]]
        assert mesh_repair._weld_near_duplicate_vertices(points, triangles).tolist() == [[0, 1, 2]]


def test_empty_triangles_keep_each_callers_compaction_and_filter_contract():
    points = np.asarray([[1.0, 2.0, 3.0]])
    triangles = np.empty((0, 3), dtype=np.int64)
    tags = np.empty(0, dtype=np.int32)

    imported_points, imported_triangles = mesh_repair._compact_unused_vertices(points, triangles)
    generated_points, generated_triangles = mesher._compact_unused_vertices(points, triangles)
    assert imported_points is points and imported_triangles is triangles
    assert generated_points.shape == (0, 3) and generated_triangles.shape == (0, 3)

    imported, imported_tags, removed = mesh_repair._remove_degenerate_triangles(points, triangles, tags)
    generated, generated_tags, generated_removed = normals.remove_degenerate_triangles(points, triangles, tags)
    assert imported is triangles and imported_tags is tags and removed == 0
    assert generated is not triangles and generated_tags is not tags and generated_removed == 0
    _, extra_tags, _ = normals.remove_degenerate_triangles(points, triangles, np.asarray([7]))
    assert extra_tags.size == 0


def test_filter_keeps_triangle_order_and_matching_tags():
    points = np.asarray([[0., 0., 0.], [2., 0., 0.], [1., 0.0001, 0.], [0., 2., 0.]])
    triangles = np.asarray([[0, 1, 3], [0, 1, 2], [0, 0, 3], [1, 3, 0]], dtype=np.int64)
    tags = np.asarray([10, 20, 30, 40], dtype=np.int32)
    for remove in (mesh_repair._remove_degenerate_triangles, normals.remove_degenerate_triangles):
        kept, kept_tags, removed = remove(points, triangles, tags, min_quality=1.0e-4)
        assert kept.tolist() == [[0, 1, 3], [1, 3, 0]]
        assert kept_tags.tolist() == [10, 40] and kept_tags.dtype == np.int32
        assert removed == 2
        _, paired_tags, _ = remove(
            points, triangles, np.column_stack((tags, tags + 1)), min_quality=1.0e-4
        )
        assert paired_tags.tolist() == [[10, 11], [40, 41]]
