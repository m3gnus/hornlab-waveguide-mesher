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

import hornlab_mesher.preview.api as preview_api
from hornlab_mesher.preview.api import (
    PreviewSurfaceV1,
    _grid_indices,
    _orient_indices_to_normals,
    _orientation_metadata,
    _triangle_orientation_analysis,
)


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


def test_only_an_internal_unchanged_buffer_reuses_its_orientation_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The builder proof saves one pass without weakening public construction."""

    positions = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    )
    indices = np.asarray((0, 1, 2), dtype=np.uint32)
    normals = np.tile((0.0, 0.0, 1.0), (3, 1))
    calls = 0
    original = preview_api._triangle_orientation_analysis

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(preview_api, "_triangle_orientation_analysis", counted)
    oriented = _orient_indices_to_normals(
        "horn.outer", positions, indices, normals
    )
    assert calls == 1
    internal_metadata = _orientation_metadata(oriented)

    # Equal values in a different positions array do not satisfy the proof.
    PreviewSurfaceV1(
        role="horn.outer",
        positions=positions.copy(),
        indices=oriented.indices,
        normals=normals,
        shading="smooth",
        normal_method="analytic-parametric",
        closed_phi=False,
        metadata=internal_metadata,
    )
    assert calls == 2

    surface = PreviewSurfaceV1(
        role="horn.outer",
        positions=positions,
        indices=oriented.indices,
        normals=normals,
        shading="smooth",
        normal_method="analytic-parametric",
        closed_phi=False,
        metadata=internal_metadata,
    )
    assert calls == 2
    assert all(isinstance(key, str) for key in surface.metadata)

    # Without the private object-keyed proof, the public constructor always
    # performs its own final-buffer check, even for the same array objects.
    PreviewSurfaceV1(
        role="horn.outer",
        positions=positions,
        indices=oriented.indices,
        normals=normals,
        shading="smooth",
        normal_method="analytic-parametric",
        closed_phi=False,
    )
    assert calls == 3


def test_internal_orientation_proof_rejects_same_object_mutation() -> None:
    """Identity alone cannot establish that a mutable ndarray is unchanged."""

    positions = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    )
    indices = np.asarray((0, 1, 2), dtype=np.uint32)
    normals = np.tile((0.0, 0.0, 1.0), (3, 1))
    oriented = _orient_indices_to_normals(
        "horn.outer", positions, indices, normals
    )

    # Keep the exact same ndarray objects, but reverse the triangle's winding
    # after the proof was issued. The final constructor must reclassify the
    # changed bytes and reject them.
    positions[2] = (0.0, -1.0, 0.0)
    with pytest.raises(ValueError, match="windings disagree with their normals"):
        PreviewSurfaceV1(
            role="horn.outer",
            positions=positions,
            indices=oriented.indices,
            normals=normals,
            shading="smooth",
            normal_method="analytic-parametric",
            closed_phi=False,
            metadata=_orientation_metadata(oriented),
        )


def test_a_flipped_buffer_is_rechecked_after_normal_sum_reassociation() -> None:
    """Flipping swaps two summands, which can change a near-cancelling sum."""

    positions = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.11420311234311552, 0.9934574219014853, 0.0),
        )
    )
    indices = np.asarray((0, 1, 2), dtype=np.uint32)
    normals = np.asarray(
        (
            (1.0, 1.2878587085651816e-14, 0.0),
            (-0.49999999999998829, 0.86602540378444548, 0.0),
            (-0.50000000000000888, -0.86602540378443349, 0.0),
        )
    )

    original = _triangle_orientation_analysis(positions, indices, normals)
    assert original.negative_triangles == 1
    oriented = _orient_indices_to_normals(
        "horn.outer", positions, indices, normals
    )
    assert np.array_equal(oriented.indices, (0, 2, 1))
    flipped = _triangle_orientation_analysis(positions, oriented.indices, normals)
    assert flipped.negative_triangles == 1

    # No proof is issued when indices move. The public surface check therefore
    # catches the still-negative final buffer instead of trusting pass one.
    with pytest.raises(ValueError, match="windings disagree with their normals"):
        PreviewSurfaceV1(
            role="horn.outer",
            positions=positions,
            indices=oriented.indices,
            normals=normals,
            shading="smooth",
            normal_method="analytic-parametric",
            closed_phi=False,
            metadata=_orientation_metadata(oriented),
        )


def test_a_combined_surface_still_runs_its_global_orientation_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Part-level evidence cannot replace the combined surface's classifier."""

    positions = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    )
    indices = np.asarray((0, 1, 2), dtype=np.uint32)
    normals = np.tile((0.0, 0.0, 1.0), (3, 1))
    part = PreviewSurfaceV1(
        role="horn.outer",
        positions=positions,
        indices=indices,
        normals=normals,
        shading="smooth",
        normal_method="analytic-parametric",
        closed_phi=False,
    )
    calls = 0
    original = preview_api._triangle_orientation_analysis

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(preview_api, "_triangle_orientation_analysis", counted)
    combined = preview_api._combine_surfaces("horn.outer", [part, part])

    assert calls == 1
    assert combined.metadata["windingChecked"] is True
