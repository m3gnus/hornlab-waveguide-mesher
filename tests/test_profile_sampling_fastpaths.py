"""Exact-output checks for the profile-sampling preview fast paths."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hornlab_mesher.config_builder import build_geometry_params
import hornlab_mesher.profile_sampling as sampling


def _legacy_mirror(q1: np.ndarray) -> np.ndarray:
    q = [float(value) for value in q1]
    full: list[float] = []
    full.extend(q)
    full.extend(math.pi - value for value in reversed(q[:-1]))
    full.extend(math.pi + value for value in q[1:])
    full.extend(math.tau - value for value in reversed(q[1:-1]))
    return np.asarray(full, dtype=np.float64)


def _legacy_freeform_quadrant_angles(
    *,
    half_width: float,
    half_height: float,
    corner_radius: float,
    side1_segments: int,
    side2_segments: int,
    arc_subdivision: int,
    collapse_transition_intervals: float,
) -> np.ndarray:
    a = float(half_width)
    b = float(half_height)
    corner = min(max(float(corner_radius), 0.0), a, b)
    theta1 = math.atan2(b - corner, a)
    theta2 = math.atan2(b, a - corner)
    arc_segments = 3 * max(1, int(arc_subdivision))
    side1_segments = int(side1_segments)
    side2_segments = int(side2_segments)
    total_segments = side1_segments + arc_segments + side2_segments
    base_layout = sampling._rounded_rect_quadrant_layout(
        side1_segments + side2_segments + 3, a, b, corner
    )
    uniform_angles = np.linspace(0.0, math.pi / 2.0, total_segments + 1)
    if corner >= b and corner >= a:
        return uniform_angles

    span1 = theta1
    span2 = math.pi / 2.0 - theta2
    arc_segments = (
        3 if base_layout is None else int(base_layout.arc_segments)
    ) * max(1, int(arc_subdivision))
    angles: list[float] = []
    if side1_segments:
        angles.extend(np.linspace(0.0, theta1, side1_segments + 1).tolist())
    else:
        angles.append(theta1)
    cx = a - corner
    cy = b - corner
    for index in range(1, arc_segments + 1):
        arc_phi = index * math.pi / (2.0 * arc_segments)
        angles.append(
            math.atan2(
                cy + corner * math.sin(arc_phi),
                cx + corner * math.cos(arc_phi),
            )
        )
    if side2_segments:
        angles.extend(
            theta2
            + (math.pi / 2.0 - theta2) * index / side2_segments
            for index in range(1, side2_segments + 1)
        )
    structural_angles = np.asarray(angles, dtype=np.float64)
    transition_span = (
        max(1.0, float(collapse_transition_intervals))
        * math.pi
        / (2.0 * total_segments)
    )
    progress = min(1.0, max(0.0, min(span1, span2) / transition_span))
    blend = progress * progress * (3.0 - 2.0 * progress)
    return uniform_angles + blend * (structural_angles - uniform_angles)


def _legacy_phi_derivative(
    points: np.ndarray, *, full_circle: bool, phi_coordinates: np.ndarray | None
) -> np.ndarray:
    n_t, n_phi = points.shape[:2]
    phi = None
    if phi_coordinates is not None:
        candidate = np.asarray(phi_coordinates, dtype=np.float64)
        if candidate.shape == (n_t, n_phi) and np.all(np.isfinite(candidate)):
            phi = candidate
    if full_circle:
        if phi is None:
            step = math.tau / n_phi
            return (np.roll(points, -1, axis=1) - np.roll(points, 1, axis=1)) / (
                2.0 * step
            )
        unwrapped = np.unwrap(phi, axis=1)
        previous = np.roll(unwrapped, 1, axis=1)
        previous[:, 0] -= math.tau
        following = np.roll(unwrapped, -1, axis=1)
        following[:, -1] += math.tau
        h_previous = unwrapped - previous
        h_next = following - unwrapped
        if np.any(h_previous <= 0.0) or np.any(h_next <= 0.0):
            step = math.tau / n_phi
            return (np.roll(points, -1, axis=1) - np.roll(points, 1, axis=1)) / (
                2.0 * step
            )
        weight_previous = (
            -h_next / (h_previous * (h_previous + h_next))
        )[..., None]
        weight_center = (
            (h_next - h_previous) / (h_previous * h_next)
        )[..., None]
        weight_next = (
            h_previous / (h_next * (h_previous + h_next))
        )[..., None]
        return (
            weight_previous * np.roll(points, 1, axis=1)
            + weight_center * points
            + weight_next * np.roll(points, -1, axis=1)
        )
    if phi is None or np.any(np.diff(phi, axis=1) <= 0.0):
        phi = np.broadcast_to(
            np.linspace(0.0, 1.0, n_phi, dtype=np.float64), (n_t, n_phi)
        )
    derivative = np.empty_like(points)
    edge_order = 2 if n_phi >= 3 else 1
    for row in range(n_t):
        derivative[row] = np.gradient(
            points[row], phi[row], axis=0, edge_order=edge_order
        )
    return derivative


def test_mirror_quadrant_angles_matches_the_list_oracle_exactly() -> None:
    rng = np.random.default_rng(20260808)
    for count in (0, 1, 2, 3, 8, 17, 65):
        q1 = np.sort(rng.uniform(0.0, math.pi / 2.0, count))
        if count:
            q1[0] = 0.0
            q1[-1] = math.pi / 2.0
        assert np.array_equal(sampling._mirror_quadrant_angles(q1), _legacy_mirror(q1))


def test_freeform_angle_basis_matches_the_per_ring_oracle_exactly() -> None:
    rng = np.random.default_rng(20260808)
    corners = (0.0, -3.0, 1.0e-12, 5.9, 40.0)
    for index in range(1000):
        a = float(10.0 ** rng.uniform(-3.0, 3.0))
        b = float(10.0 ** rng.uniform(-3.0, 3.0))
        corner = min(a, b) if index % 7 == 0 else corners[index % len(corners)]
        kwargs = {
            "half_width": a,
            "half_height": b,
            "corner_radius": corner,
            "side1_segments": int(rng.integers(1, 20)),
            "side2_segments": int(rng.integers(1, 20)),
            "arc_subdivision": int(rng.integers(1, 14)),
            "collapse_transition_intervals": float(rng.choice((1.0, 4.0, 8.5))),
        }
        expected = _legacy_freeform_quadrant_angles(**kwargs)
        actual = sampling._freeform_rounded_rect_quadrant_angles(**kwargs)
        assert np.array_equal(actual, expected), kwargs


@pytest.mark.parametrize("full_circle", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_grid_phi_derivative_matches_the_full_grid_oracle_exactly(
    shared: bool, full_circle: bool
) -> None:
    rng = np.random.default_rng(20260808)
    points = rng.normal(size=(9, 33, 3))
    increments = rng.uniform(0.2, 1.2, points.shape[1])
    base = np.cumsum(increments)
    base = (base - base[0]) * math.tau / (base[-1] + increments[-1])
    phi = np.broadcast_to(base, points.shape[:2])
    if not shared:
        phi = phi.copy()
        phi += np.linspace(0.0, 0.02, points.shape[0])[:, None] * np.sin(base)
    expected = _legacy_phi_derivative(
        points, full_circle=full_circle, phi_coordinates=phi
    )
    actual = sampling._grid_phi_derivative(
        points, full_circle=full_circle, phi_coordinates=phi
    )
    assert np.array_equal(actual, expected)


def test_grid_surface_normals_matches_np_cross_exactly() -> None:
    rng = np.random.default_rng(20260808)
    points = rng.normal(size=(7, 24, 3))
    t = np.linspace(0.0, 1.0, points.shape[0]) ** 1.7
    phi = np.broadcast_to(
        np.linspace(0.0, math.tau, points.shape[1], endpoint=False),
        points.shape[:2],
    )
    d_t = np.gradient(points, t, axis=0, edge_order=2)
    d_phi = _legacy_phi_derivative(
        points, full_circle=True, phi_coordinates=phi
    )
    expected = np.cross(d_phi, d_t)
    actual = sampling._grid_surface_normals(
        points,
        full_circle=True,
        t_coordinates=t,
        phi_coordinates=phi,
    )
    assert np.array_equal(actual, expected)


def _outer_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, 13) ** 1.5
    phi = np.linspace(0.0, math.tau, 32, endpoint=False)
    phi_grid = np.broadcast_to(phi, (len(t), len(phi)))
    radius = 12.7 + 80.0 * t[:, None]
    grid = np.empty((len(t), len(phi), 3), dtype=np.float64)
    grid[..., 0] = radius * np.cos(phi_grid)
    grid[..., 1] = radius * np.sin(phi_grid)
    grid[..., 2] = np.broadcast_to(120.0 * t[:, None], radius.shape)
    return grid.transpose(1, 0, 2).copy(), t, phi_grid


def test_outer_offset_skips_the_missing_normal_walk_on_a_regular_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inner, t, phi = _outer_fixture()

    def unexpected(*_args, **_kwargs):
        raise AssertionError("regular grid should have no missing normals")

    monkeypatch.setattr(sampling, "_fill_missing_normals", unexpected)
    outer = sampling._outer_offset_shell(
        inner,
        5.0,
        full_circle=True,
        t_coordinates=t,
        phi_coordinates=phi,
    )
    assert outer.shape == inner.shape
    assert np.all(np.isfinite(outer))


def _freeform_config(quadrants: str) -> dict:
    return {
        "formula": "FREEFORM",
        "mode": "bare",
        "profile": {
            "profileH": {
                "points": [[0.0, 12.7], [60.0, 80.0], [120.0, 160.0]],
                "throatAngleDeg": 15.5,
                "mouthAngleDeg": 70.0,
            },
            "profileV": {
                "points": [[0.0, 12.7], [60.0, 60.0], [120.0, 110.0]],
                "throatAngleDeg": 15.5,
                "mouthAngleDeg": 60.0,
            },
            "crossSections": [
                {"t": 0.0, "shape": "circle"},
                {
                    "t": 0.4,
                    "shape": "rounded_rectangle",
                    "cornerRadiusMm": 5.9,
                },
                {
                    "t": 1.0,
                    "shape": "rounded_rectangle",
                    "cornerRadiusMm": 5.9,
                },
            ],
        },
        "mesh": {
            "angularSegments": 32,
            "lengthSegments": 8,
            "samplingMode": "uniform",
            "quadrants": quadrants,
        },
    }


@pytest.mark.parametrize("quadrants", ["1", "12", "14", "1234"])
@pytest.mark.parametrize(
    ("endpoint", "replacement", "cardinal"),
    [(0, 1.0e-6, "0"), (-1, math.pi / 2.0 - 1.0e-6, "1.5708")],
)
def test_freeform_cardinal_check_guards_every_reduced_domain(
    monkeypatch: pytest.MonkeyPatch,
    quadrants: str,
    endpoint: int,
    replacement: float,
    cardinal: str,
) -> None:
    params, formula, _mode = build_geometry_params(_freeform_config(quadrants))
    assert formula == "FREEFORM"
    original = sampling._freeform_rounded_rect_quadrant_angles

    def malformed(**kwargs):
        angles = original(**kwargs)
        angles[endpoint] = replacement
        return angles

    monkeypatch.setattr(
        sampling, "_freeform_rounded_rect_quadrant_angles", malformed
    )
    expected_cardinal = (
        "-1.5708" if quadrants == "14" and endpoint == -1 else cardinal
    )
    with pytest.raises(
        ValueError, match=rf"required cardinal {expected_cardinal} rad"
    ):
        sampling._freeform_raw_radial_grid(params, n_length=8)


def test_batched_cardinal_check_still_guards_mirrored_meridians(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params, formula, _mode = build_geometry_params(_freeform_config("1234"))
    assert formula == "FREEFORM"
    original = sampling._mirror_quadrant_angles

    def malformed(q1: np.ndarray) -> np.ndarray:
        full = original(q1)
        pi_index = 2 * len(q1) - 2
        full[pi_index] += 1.0e-6
        return full

    monkeypatch.setattr(sampling, "_mirror_quadrant_angles", malformed)
    with pytest.raises(ValueError, match=r"required cardinal 3\.14159 rad"):
        sampling._freeform_raw_radial_grid(params, n_length=8)
