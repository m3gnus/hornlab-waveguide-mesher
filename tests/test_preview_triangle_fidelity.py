"""Preview fidelity is measured against the triangles the preview emits.

A grid quad is drawn as two planar triangles split on its ``(0,0)-(1,1)``
diagonal. The old measurements compared the true surface with a bilinear patch
through the quad's corners (``estimate_grid_fidelity``) or with the chords of
the grid lines only (``adaptive_grid_indices``). A twisted patch such as
``P(u, v) = (100u, 100v, 4uv)`` mm is straight along both parameters, so both
reported ~0 mm while its emitted triangles are ~1 mm off in the middle.
"""

from __future__ import annotations

import numpy as np
import pytest

from hornlab_mesher.preview.fidelity import (
    adaptive_grid_indices,
    analytic_grid_normals,
    emitted_triangle_errors,
    estimate_grid_fidelity,
)


def _twisted_patch(n_u: int, n_v: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, n_u)
    v = np.linspace(0.0, 1.0, n_v)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return np.stack((100.0 * uu, 100.0 * vv, 4.0 * uu * vv), axis=2)


def _brute_force_triangle_error(points: np.ndarray, t_indices, phi_indices) -> float:
    """Every true sample against a dense lattice over every emitted triangle."""

    from scipy.spatial import cKDTree

    steps = 240
    a, b = np.meshgrid(np.arange(steps + 1), np.arange(steps + 1), indexing="ij")
    keep = a + b <= steps
    weights = np.stack(
        (steps - a[keep] - b[keep], a[keep], b[keep]), axis=1
    ).astype(np.float64) / steps
    rows, cols = list(t_indices), list(phi_indices)
    dense = []
    for i0, i1 in zip(rows[:-1], rows[1:]):
        for j0, j1 in zip(cols[:-1], cols[1:]):
            p00, p01, p10, p11 = points[i0, j0], points[i0, j1], points[i1, j0], points[i1, j1]
            for triangle in ((p00, p01, p11), (p00, p11, p10)):
                dense.append(weights @ np.stack(triangle))
    distances, _ = cKDTree(np.concatenate(dense)).query(points.reshape(-1, 3))
    return float(distances.max())


def test_twisted_patch_estimate_reports_the_triangle_error():
    reference = _twisted_patch(41, 41)
    coarse = reference[::40, ::40]  # one quad: the four corners
    normals = analytic_grid_normals(reference, closed_phi=False)

    measured = estimate_grid_fidelity(coarse, reference, normals, closed_phi=False)

    # The bilinear patch through the corners IS this surface, so the old
    # measurement reported machine epsilon. The two emitted planes miss the
    # middle by 1 mm vertically, ~0.9993 mm along their normals.
    # Closed form: the centre (50, 50, 1) against the plane z = 0.04 x of the
    # (0,0,0)-(0,100,0)-(100,100,4) triangle.
    assert measured["max_chord_error_mm"] == pytest.approx(
        10000.0 / np.hypot(400.0, 10000.0), rel=1.0e-12
    )


def test_twisted_patch_estimate_with_parameter_coordinates():
    reference = _twisted_patch(21, 21)
    coarse = reference[::20, ::20]
    normals = analytic_grid_normals(reference, closed_phi=False)
    axis = np.linspace(0.0, 1.0, 21)
    measured = estimate_grid_fidelity(
        coarse,
        reference,
        normals,
        closed_phi=False,
        coarse_t=np.array([0.0, 1.0]),
        coarse_phi=np.broadcast_to(np.array([0.0, 1.0]), (2, 2)),
        reference_t=axis,
        reference_phi=np.broadcast_to(axis, (21, 21)),
    )
    assert measured["max_chord_error_mm"] > 0.99


def test_adaptive_refinement_splits_a_twisted_quad_until_its_triangles_fit():
    points = _twisted_patch(33, 33)
    normals = analytic_grid_normals(points, closed_phi=False)
    target = 0.05

    t_indices, phi_indices, achieved = adaptive_grid_indices(
        points,
        normals,
        [0, 32],
        [0, 16, 32],
        max_chord_error_mm=target,
        max_normal_step_deg=180.0,
        max_vertices=None,
        closed_phi=False,
    )

    distances, *_ = emitted_triangle_errors(
        points, t_indices, phi_indices, closed_phi=False
    )
    assert float(distances.max()) <= target
    assert achieved["measurement_complete"] is True
    assert achieved["max_chord_error_mm"] == pytest.approx(float(distances.max()))
    assert achieved["vertex_cap_limited"] is False
    # Directional chords alone accept the 2x3 seed as exact (every grid line
    # of this patch is straight). Splitting across the longer extent keeps
    # the quads square rather than slicing the patch into slivers.
    assert len(t_indices) > 2 and len(phi_indices) > 3
    assert abs(len(t_indices) - len(phi_indices)) <= 1


def test_emitted_triangle_errors_match_brute_force_on_a_curved_grid():
    u = np.linspace(0.0, 1.0, 13)
    v = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    radius = 10.0 + 30.0 * uu**2
    points = np.stack(
        (radius * np.cos(vv) * (1.0 + 0.2 * np.cos(4 * vv)), radius * np.sin(vv), 80.0 * uu),
        axis=2,
    )
    t_indices, phi_indices = [0, 5, 12], list(range(0, 24, 6))
    distances, *_ = emitted_triangle_errors(
        points, t_indices, phi_indices, closed_phi=True
    )
    closed_phi = phi_indices + [phi_indices[0]]
    # Brute force over the closed seam too, with the wrap quad's columns 18 -> 0.
    wrapped = np.concatenate((points, points[:, :1]), axis=1)
    exact = _brute_force_triangle_error(wrapped, t_indices, closed_phi[:-1] + [24])
    # Each sample is measured against its own quad, so it can only read high,
    # and the lattice spacing reads the brute force a hair high as well.
    assert float(distances.max()) == pytest.approx(exact, rel=5.0e-3)
