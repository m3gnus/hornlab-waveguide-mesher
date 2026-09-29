"""The vectorised edge table must agree with the dict passes it replaced."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from hornlab_mesher.edges import build_edge_table


def _soup(seed: int, n_vertices: int = 30, n_triangles: int = 120) -> np.ndarray:
    """Random triangles including collapsed edges, repeats and 3+-face edges."""

    rng = np.random.default_rng(seed)
    return rng.integers(0, n_vertices, size=(n_triangles, 3)).astype(np.int64)


def _reference_edge_dirs(triangles):
    dirs = defaultdict(list)
    for tri in triangles:
        for start, end in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(start), int(end)
            if a == b:
                continue
            if a < b:
                dirs[(a, b)].append(1)
            else:
                dirs[(b, a)].append(-1)
    return dirs


def _reference_uses(triangles):
    uses = defaultdict(list)
    for tri_idx, tri in enumerate(triangles):
        for start, end in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(start), int(end)
            if a == b:
                continue
            uses[(min(a, b), max(a, b))].append((tri_idx, 1 if a < b else -1))
    return uses


@pytest.mark.parametrize("seed", range(6))
def test_edges_match_the_dict_reference_in_order(seed):
    triangles = _soup(seed)
    table = build_edge_table(triangles, drop_degenerate=True)
    reference = _reference_uses(triangles)
    assert table.n_edges == len(reference)
    assert list(zip(table.lo.tolist(), table.hi.tolist())) == list(reference)
    for edge_id, edge in enumerate(reference):
        assert int(table.count[edge_id]) == len(reference[edge])
        mine = [
            (int(t), int(d))
            for t, d, e in zip(table.tri, table.direction, table.edge)
            if e == edge_id
        ]
        assert mine == reference[edge]
        assert int(table.tri[table.first_use[edge_id]]) == reference[edge][0][0]


def _flipped_grid(seed: int, n: int = 7):
    """Open triangulated grid, some faces reversed, one fully collapsed triangle.

    Manifold (every edge has one or two uses) so the validator reaches its
    counts instead of refusing; flips give inconsistent edges and the open
    rim gives boundary edges.
    """

    rng = np.random.default_rng(seed)
    idx = np.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
    faces = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            faces += [[a, b, c], [a, c, d]]
    faces = np.array(faces, dtype=np.int64)
    flip = rng.random(len(faces)) < 0.3
    faces[flip] = faces[flip][:, ::-1]
    faces = np.vstack([faces, [[3, 3, 3]]])  # degenerate: contributes no edges
    points = rng.normal(size=((n + 1) * (n + 1), 3))
    return points, faces


@pytest.mark.parametrize("seed", range(6))
def test_orientation_counts_match_the_dict_reference(seed):
    from hornlab_mesher.normals import validate_orientation

    points, triangles = _flipped_grid(seed)
    tags = np.full(len(triangles), 1, dtype=np.int32)
    reference = _reference_edge_dirs(triangles)
    boundary = sum(1 for d in reference.values() if len(d) == 1)
    nonmanifold = sum(1 for d in reference.values() if len(d) not in (1, 2))
    inconsistent = sum(1 for d in reference.values() if len(d) == 2 and d[0] == d[1])
    assert nonmanifold == 0 and boundary > 0 and inconsistent > 0
    report = validate_orientation(
        points,
        triangles,
        tags,
        require_source_normal=False,
        require_positive_volume=False,
    )
    assert report.boundary_edges == boundary
    assert report.nonmanifold_edges == nonmanifold
    assert report.inconsistent_edges == inconsistent
    assert report.n_edges == len(reference)


@pytest.mark.parametrize("seed", range(6))
def test_nonmanifold_count_matches_the_dict_reference(seed):
    from hornlab_mesher.normals import MeshOrientationError, validate_orientation

    triangles = _soup(seed)
    points = np.random.default_rng(seed + 100).normal(size=(30, 3))
    tags = np.full(len(triangles), 1, dtype=np.int32)
    reference = _reference_edge_dirs(triangles)
    nonmanifold = sum(1 for d in reference.values() if len(d) not in (1, 2))
    assert nonmanifold > 0
    with pytest.raises(MeshOrientationError, match=rf"has {nonmanifold} nonmanifold"):
        validate_orientation(points, triangles, tags, require_source_normal=False)


def test_repair_matches_a_consistent_mesh_and_flips_a_reversed_one():
    from hornlab_mesher.normals import repair_orientation

    # Tetrahedron with one face wound the wrong way.
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]], dtype=np.int64)
    faces[2] = faces[2][::-1]
    tags = np.full(4, 1, dtype=np.int32)
    repaired, stats = repair_orientation(points, faces, tags)
    assert stats["flipped_consistency"] in (1, 3)
    table = build_edge_table(repaired, drop_degenerate=True)
    sums = np.bincount(table.edge, weights=table.direction.astype(float))
    assert np.all(sums == 0.0)
    volume = float(
        np.sum(points[repaired[:, 0]] * np.cross(points[repaired[:, 1]], points[repaired[:, 2]]))
    )
    assert volume > 0.0
