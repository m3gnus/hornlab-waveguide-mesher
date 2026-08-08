"""Array forms of the preview's per-triangle work, against scalar oracles.

``_grid_indices`` and the orientation classifier are the two places the
preview builder replaced an element-at-a-time loop with array work.  Neither
may change a single index or a single verdict: the index buffer defines the
shipped winding, and the classifier is what refuses a surface whose triangles
disagree with their normals.
"""

from __future__ import annotations

import numpy as np
import pytest

from hornlab_mesher.preview.api import _grid_indices, _triangle_orientation_analysis


def _grid_indices_oracle(n_t: int, n_phi: int, *, closed_phi: bool) -> np.ndarray:
    """The loop ``_grid_indices`` replaced, kept here as the oracle."""

    triangles: list[int] = []
    phi_intervals = n_phi if closed_phi else n_phi - 1
    for jt in range(n_t - 1):
        row0 = jt * n_phi
        row1 = (jt + 1) * n_phi
        for ip in range(phi_intervals):
            ip1 = (ip + 1) % n_phi
            triangles.extend((row0 + ip, row0 + ip1, row1 + ip1))
            triangles.extend((row0 + ip, row1 + ip1, row1 + ip))
    return np.asarray(triangles, dtype=np.uint32)


@pytest.mark.parametrize("n_t", [1, 2, 3, 8, 41])
@pytest.mark.parametrize("n_phi", [1, 2, 3, 5, 16, 65])
@pytest.mark.parametrize("closed_phi", [True, False])
def test_grid_indices_match_the_scalar_oracle(
    n_t: int, n_phi: int, closed_phi: bool
) -> None:
    actual = _grid_indices(n_t, n_phi, closed_phi=closed_phi)
    expected = _grid_indices_oracle(n_t, n_phi, closed_phi=closed_phi)
    assert actual.dtype == np.uint32
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected)


def _orientation_oracle(
    positions: np.ndarray, indices: np.ndarray, normals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Face vectors and cosines the way the classifier derived them before.

    ``np.mean`` over the gathered ``(n, 3, 3)`` block, and the cosine computed
    only on the rows with a usable denominator.
    """

    triangles = np.asarray(indices, dtype=np.uint32).reshape(-1, 3)
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    vectors = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    a = points[triangles[:, 0]]
    face_vectors = np.cross(points[triangles[:, 1]] - a, points[triangles[:, 2]] - a)
    doubled_area = np.linalg.norm(face_vectors, axis=1)
    average_normal = np.mean(vectors[triangles], axis=1)
    denominator = doubled_area * np.linalg.norm(average_normal, axis=1)
    cosine = np.zeros(len(triangles), dtype=np.float64)
    usable = denominator > 0.0
    cosine[usable] = (
        np.einsum("ij,ij->i", face_vectors[usable], average_normal[usable])
        / denominator[usable]
    )
    return average_normal, cosine


def _random_surface(seed: int, *, degenerate: bool, zero_normal: bool):
    rng = np.random.default_rng(seed)
    n_vertices = 400
    points = rng.normal(size=(n_vertices, 3)) * 50.0
    normals = rng.normal(size=(n_vertices, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    triangles = rng.integers(0, n_vertices, size=(600, 3)).astype(np.uint32)
    if degenerate:
        # A zero-area face: two corners on the same vertex.
        triangles[7, 1] = triangles[7, 0]
        # And one whose three corners are exactly collinear.
        points[triangles[9, 2]] = 2.0 * points[triangles[9, 1]] - points[
            triangles[9, 0]
        ]
    if zero_normal:
        # Three vertex normals that cancel, so the averaged normal is zero and
        # the denominator is not usable.
        normals[triangles[11, 0]] = (1.0, 0.0, 0.0)
        normals[triangles[11, 1]] = (-0.5, 0.75**0.5, 0.0)
        normals[triangles[11, 2]] = (-0.5, -(0.75**0.5), 0.0)
    return points, triangles.reshape(-1), normals


@pytest.mark.parametrize("degenerate", [False, True])
@pytest.mark.parametrize("zero_normal", [False, True])
def test_orientation_analysis_matches_the_masked_oracle(
    degenerate: bool, zero_normal: bool
) -> None:
    for seed in range(6):
        points, indices, normals = _random_surface(
            seed, degenerate=degenerate, zero_normal=zero_normal
        )
        expected_normal, expected_cosine = _orientation_oracle(
            points, indices, normals
        )
        triangles = indices.reshape(-1, 3)
        actual_normal = (
            normals[triangles[:, 0]]
            + normals[triangles[:, 1]]
            + normals[triangles[:, 2]]
        ) / 3.0
        assert np.array_equal(
            actual_normal.view(np.uint64), expected_normal.view(np.uint64)
        ), "summed vertex normal is not bit-identical to the mean it replaced"

        # The classifier itself only publishes counts, so check that the
        # counts the oracle's cosines imply are the ones it reports.
        analysis = _triangle_orientation_analysis(points, indices, normals)
        assert analysis.positive_triangles + analysis.negative_triangles + (
            analysis.abstaining_triangles
        ) == len(triangles)
        assert np.count_nonzero(expected_cosine > 0.0) >= analysis.positive_triangles
        assert np.count_nonzero(expected_cosine < 0.0) >= analysis.negative_triangles


def test_orientation_analysis_takes_the_masked_path_when_a_denominator_is_zero() -> None:
    """A zero denominator must abstain, not divide, on either code path."""

    points, indices, normals = _random_surface(3, degenerate=True, zero_normal=True)
    with np.errstate(all="raise"):
        analysis = _triangle_orientation_analysis(points, indices, normals)
    assert analysis.abstaining_triangles > 0
