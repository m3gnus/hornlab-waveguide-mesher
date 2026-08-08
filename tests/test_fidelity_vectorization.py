"""Bit-exact oracles for preview-fidelity array rewrites."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hornlab_mesher.preview.fidelity import (
    _angle_degrees,
    _bilinear_coarse_points,
    _phi_derivative,
    resample_grid_vectors,
)


def test_squared_distance_selects_the_same_bit_exact_norm_winner() -> None:
    deltas = np.random.default_rng(716_064).normal(size=(41, 29, 3))
    norms = np.linalg.norm(deltas, axis=-1)
    squared = (deltas * deltas).sum(axis=-1)
    expected_index = int(np.argmax(norms))
    actual_index = int(np.argmax(squared))

    assert actual_index == expected_index
    assert float(np.sqrt(squared.reshape(-1)[actual_index])).hex() == float(
        norms.reshape(-1)[expected_index]
    ).hex()


def _bilinear_oracle(
    source: np.ndarray,
    out_shape: tuple[int, int],
    *,
    closed_phi: bool,
) -> np.ndarray:
    out_t, out_phi = out_shape
    src_t, src_phi, _ = source.shape
    t_coords = np.linspace(0.0, src_t - 1.0, out_t)
    phi_coords = (
        np.arange(out_phi, dtype=np.float64) * src_phi / out_phi
        if closed_phi
        else np.linspace(0.0, src_phi - 1.0, out_phi)
    )
    result = np.empty((out_t, out_phi, 3), dtype=np.float64)
    for jt, t_coord in enumerate(t_coords):
        t0 = min(int(math.floor(t_coord)), src_t - 1)
        t1 = min(t0 + 1, src_t - 1)
        wt = t_coord - t0
        for ip, phi_coord in enumerate(phi_coords):
            phi_floor = math.floor(phi_coord)
            p0 = int(phi_floor)
            if closed_phi:
                p0 %= src_phi
                p1 = (p0 + 1) % src_phi
            else:
                p0 = min(p0, src_phi - 1)
                p1 = min(p0 + 1, src_phi - 1)
            wp = phi_coord - phi_floor
            a = source[t0, p0] * (1.0 - wp) + source[t0, p1] * wp
            b = source[t1, p0] * (1.0 - wp) + source[t1, p1] * wp
            result[jt, ip] = a * (1.0 - wt) + b * wt
    return result


@pytest.mark.parametrize("closed_phi", [False, True])
@pytest.mark.parametrize("out_shape", [(3, 4), (8, 11), (13, 7)])
def test_bilinear_grid_rewrites_match_the_nested_loops(
    closed_phi: bool, out_shape: tuple[int, int]
) -> None:
    source = np.random.default_rng(260808).normal(size=(5, 7, 3))
    expected = _bilinear_oracle(source, out_shape, closed_phi=closed_phi)

    points = _bilinear_coarse_points(
        source, out_shape, closed_phi=closed_phi
    )
    vectors = resample_grid_vectors(
        source, out_shape, closed_phi=closed_phi
    )
    expected_vectors = expected / np.linalg.norm(expected, axis=-1, keepdims=True)

    assert np.array_equal(points.view(np.uint64), expected.view(np.uint64))
    assert np.array_equal(
        vectors.view(np.uint64), expected_vectors.view(np.uint64)
    )


def _closed_phi_derivative_oracle(
    samples: np.ndarray, phi: np.ndarray | None
) -> np.ndarray:
    n_t, n_phi = samples.shape[:2]
    if phi is None:
        step = math.tau / n_phi
        return (
            np.roll(samples, -1, axis=1) - np.roll(samples, 1, axis=1)
        ) / (2.0 * step)
    unwrapped = np.unwrap(phi, axis=1)
    previous_phi = np.roll(unwrapped, 1, axis=1)
    previous_phi[:, 0] -= math.tau
    next_phi = np.roll(unwrapped, -1, axis=1)
    next_phi[:, -1] += math.tau
    h_previous = unwrapped - previous_phi
    h_next = next_phi - unwrapped
    coefficient_previous = -h_next / (
        h_previous * (h_previous + h_next)
    )
    coefficient_center = (h_next - h_previous) / (h_previous * h_next)
    coefficient_next = h_previous / (h_next * (h_previous + h_next))
    trailing = (1,) * (samples.ndim - 2)
    return (
        coefficient_previous.reshape((n_t, n_phi) + trailing)
        * np.roll(samples, 1, axis=1)
        + coefficient_center.reshape((n_t, n_phi) + trailing) * samples
        + coefficient_next.reshape((n_t, n_phi) + trailing)
        * np.roll(samples, -1, axis=1)
    )


@pytest.mark.parametrize("components", [None, 3])
@pytest.mark.parametrize("with_coordinates", [False, True])
@pytest.mark.parametrize("fortran_order", [False, True])
def test_closed_phi_derivative_matches_the_roll_oracle(
    components: int | None, with_coordinates: bool, fortran_order: bool
) -> None:
    rng = np.random.default_rng(715_447)
    shape = (6, 17) if components is None else (6, 17, components)
    samples = rng.normal(size=shape)
    steps = rng.uniform(0.02, 0.08, size=(shape[0], shape[1]))
    phi = np.cumsum(steps, axis=1)
    phi -= phi[:, :1]
    if fortran_order:
        samples = np.asfortranarray(samples)
        phi = np.asfortranarray(phi)
    coordinates = phi if with_coordinates else None

    actual = _phi_derivative(
        samples, closed_phi=True, phi_coordinates=coordinates
    )
    expected = _closed_phi_derivative_oracle(samples, coordinates)

    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_scalar_angle_pipeline_matches_the_numpy_pipeline() -> None:
    rng = np.random.default_rng(936_055)
    left = rng.normal(size=(1024, 3))
    right = rng.normal(size=(1024, 3))
    left /= np.linalg.norm(left, axis=1, keepdims=True)
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    dots = np.sum(left * right, axis=-1)
    expected = float(
        np.degrees(np.arccos(np.clip(np.min(dots), -1.0, 1.0)))
    )

    assert _angle_degrees(left, right).hex() == expected.hex()
