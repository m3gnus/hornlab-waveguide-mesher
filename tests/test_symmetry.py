"""The one mirror-symmetry check (``hornlab_mesher.symmetry``)."""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from hornlab_mesher.symmetry import (
    TOLERANCES,
    MirrorMatches,
    PlaneSymmetryVerdict,
    RingMirrorTarget,
    evaluate_enclosure_symmetry,
    evaluate_mirror_symmetry,
    evaluate_point_grid_symmetry,
    native_symmetry_plane_for_quadrants,
    snap_symmetry_plane_vertices,
)


def test_import_needs_neither_gmsh_nor_scipy():
    code = (
        "import sys, hornlab_mesher.symmetry\n"
        "print('gmsh' in sys.modules, 'scipy' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.split() == ["False", "False"]


def test_old_import_paths_name_the_same_objects():
    import hornlab_mesher
    from hornlab_mesher import step_prepare, symmetry

    assert step_prepare.PlaneSymmetryVerdict is symmetry.PlaneSymmetryVerdict
    assert hornlab_mesher.PlaneSymmetryVerdict is symmetry.PlaneSymmetryVerdict
    assert step_prepare.snap_symmetry_plane_vertices is symmetry.snap_symmetry_plane_vertices
    assert step_prepare.DEFAULT_AUTO_CUT_TOLERANCE_REL == TOLERANCES["fitted"].relative
    assert step_prepare.AUTO_CUT_SPAN_REL == symmetry.SPAN_TOLERANCE_REL
    assert step_prepare.DEFAULT_SYMMETRY_SNAP_BAND_MM == symmetry.DEFAULT_SYMMETRY_SNAP_BAND_MM
    assert step_prepare.SYMMETRY_AXIS_FOR_PLANE is symmetry.SYMMETRY_AXIS_FOR_PLANE


@pytest.mark.parametrize("diagonal", [0.0, 1.0e-9, 1.0e-3, 1.0, 182.42, 1.0e4])
def test_tolerance_classes(diagonal):
    # exact: the design check's rule, max(1e-7, 2e-4 * max(diagonal, 1e-7)).
    assert TOLERANCES["exact"].resolve(diagonal) == max(1.0e-7, 2.0e-4 * max(diagonal, 1.0e-7))
    # fitted: the CAD check's rule, 5e-4 * diagonal, no floor.
    assert TOLERANCES["fitted"].resolve(diagonal) == 5.0e-4 * diagonal


def _ellipse_grid(*, n_phi=64, n_axial=9, a=40.0, b=25.0, rotation_deg=0.0, phase=0.0):
    phi = phase + 2.0 * math.pi * np.arange(n_phi) / n_phi
    z = np.linspace(0.0, 80.0, n_axial)
    grow = 1.0 + z / 80.0
    x = a * np.cos(phi)[:, None] * grow[None, :]
    y = b * np.sin(phi)[:, None] * grow[None, :]
    turn = math.radians(rotation_deg)
    xr = x * math.cos(turn) - y * math.sin(turn)
    yr = x * math.sin(turn) + y * math.cos(turn)
    return np.stack([xr, yr, np.broadcast_to(z, xr.shape)], axis=2)


def _reference_ring_distance(point, ring):
    """Independent point-to-closed-polyline distance, one segment at a time."""
    best = math.inf
    for index in range(len(ring)):
        start, end = ring[index], ring[(index + 1) % len(ring)]
        delta = end - start
        length_sq = float(delta @ delta)
        t = 0.0 if length_sq <= 1.0e-24 else min(1.0, max(0.0, float((point - start) @ delta) / length_sq))
        best = min(best, float(np.linalg.norm(point - (start + t * delta))))
    return best


def test_symmetric_grids_accept_both_planes():
    inner = _ellipse_grid()
    outer = _ellipse_grid(a=46.0, b=31.0)
    verdicts = evaluate_point_grid_symmetry({"inner": inner, "outer": outer})
    assert set(verdicts) == {"x0", "y0"}
    for plane, verdict in verdicts.items():
        assert isinstance(verdict, PlaneSymmetryVerdict)
        assert verdict.plane == plane
        assert verdict.accepted
        assert verdict.points_tested == inner.size // 3 + outer.size // 3
        assert verdict.points_off_model == 0
        assert verdict.role_mismatches == 0
        assert 0.0 <= verdict.max_residual < 1.0e-12
        assert verdict.worst_off_model_distance == -1.0


def test_azimuth_samples_that_do_not_mirror_still_judge_the_surface():
    # Offset azimuths: no sample mirrors onto another sample, yet the
    # surface is symmetric; the ring test measures to the polyline.
    grid = _ellipse_grid(n_phi=128, phase=0.013)
    verdicts = evaluate_point_grid_symmetry({"inner": grid})
    assert verdicts["x0"].accepted and verdicts["y0"].accepted


def test_rotated_grid_rejects_with_its_true_worst_distance():
    grid = _ellipse_grid(rotation_deg=25.0)
    verdicts = evaluate_point_grid_symmetry({"inner": grid}, planes=("y0",))
    verdict = verdicts["y0"]
    assert not verdict.accepted
    mirrored = grid.copy()
    mirrored[..., 1] *= -1.0
    reference = np.array([
        [_reference_ring_distance(mirrored[i, j], grid[:, j]) for j in range(grid.shape[1])]
        for i in range(grid.shape[0])
    ])
    tolerance = TOLERANCES["exact"].resolve(float(np.linalg.norm(np.ptp(grid.reshape(-1, 3), axis=0))))
    assert verdict.tolerance == tolerance
    assert verdict.points_off_model == int(np.count_nonzero(reference > tolerance))
    assert verdict.worst_off_model_distance == pytest.approx(float(reference.max()), abs=1e-12)
    assert verdict.worst_surface == "inner"
    assert len(verdict.failure_samples) == 5
    assert all(sample["kind"] == "off_model" for sample in verdict.failure_samples)


def test_one_asymmetric_surface_rejects_the_plane():
    inner = _ellipse_grid()
    outer = _ellipse_grid(a=46.0, b=31.0)
    outer[:, -1, 0] += 3.0  # the outer mouth ring shifted along +x
    verdicts = evaluate_point_grid_symmetry({"inner": inner, "outer": outer})
    assert verdicts["y0"].accepted
    assert not verdicts["x0"].accepted
    assert verdicts["x0"].worst_surface == "outer"
    assert verdicts["x0"].worst_off_model_distance > 1.0


def test_explicit_tolerance_overrides_the_class():
    grid = _ellipse_grid()
    grid[:, :, 0] += 1.0e-5
    assert evaluate_point_grid_symmetry({"inner": grid})["x0"].accepted
    strict = evaluate_point_grid_symmetry({"inner": grid}, tolerance_mm=1.0e-6)["x0"]
    assert not strict.accepted
    assert strict.tolerance == 1.0e-6


@pytest.mark.parametrize("bad", [np.zeros((4, 3)), np.zeros((4, 3, 2))])
def test_grid_shape_is_checked(bad):
    with pytest.raises(ValueError, match="shaped"):
        evaluate_point_grid_symmetry({"inner": bad})


def test_non_finite_grid_is_refused():
    grid = _ellipse_grid()
    grid[3, 2, 1] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_point_grid_symmetry({"inner": grid})


def test_empty_grid_set_is_refused():
    with pytest.raises(ValueError, match="no grid"):
        evaluate_point_grid_symmetry({})


class _TableTarget:
    """A fake target: fixed hits per sample index, and a seam allowance list."""

    def __init__(self, table, seams=()):
        self.table = table
        self.seams = set(seams)
        self.seam_calls = []

    def match(self, surface, mirrored):
        rows = self.table[surface]
        return MirrorMatches(
            np.asarray([row[0] for row in rows], dtype=float),
            np.asarray([row[1] for row in rows], dtype=float),
            hits=[row[2] for row in rows],
            context=[(surface, index) for index in range(len(rows))],
        )

    def role_seam_allows(self, surface, point, mirrored, hits, context):
        self.seam_calls.append(context)
        return context in self.seams


def _straddling(n):
    return np.column_stack([np.linspace(-1.0, 1.0, n), np.zeros(n), np.zeros(n)])


def test_kernel_counts_roles_misses_and_seams_in_sample_order():
    samples = {1: _straddling(4), 2: _straddling(2)}
    roles = {1: "a", 2: "b", 3: "a"}
    target = _TableTarget(
        {
            1: [
                (0.01, 0.01, [3]),       # same role: fine
                (-1.0, 0.7, []),         # off the model
                (0.02, 0.02, [2]),       # other role: mismatch
                (0.03, 0.03, [2]),       # other role, but on a seam
            ],
            2: [(0.04, 0.04, [2]), (-1.0, 0.9, [])],
        },
        seams={(1, 3)},
    )
    verdict = evaluate_mirror_symmetry(
        "x0", samples=samples, roles=roles, target=target, tolerance=0.05, span_tolerance=0.1,
    )
    assert not verdict.accepted
    assert verdict.points_tested == 6
    assert verdict.points_off_model == 2
    assert verdict.role_mismatches == 1
    assert verdict.max_residual == 0.04
    assert verdict.worst_residual_surface == 2
    assert verdict.worst_off_model_distance == 0.9
    assert verdict.worst_surface == 2
    assert [sample["kind"] for sample in verdict.failure_samples] == [
        "off_model", "role_mismatch", "off_model",
    ]
    assert verdict.failure_samples[1]["mirrored_roles"] == ["b"]
    assert target.seam_calls == [(1, 2), (1, 3)]
    assert verdict.reason == "2 of 6 sampled points do not mirror onto the model"


def test_kernel_requires_a_straddle_unless_told_not_to():
    samples = {1: np.column_stack([np.linspace(0.0, 1.0, 3), np.zeros(3), np.zeros(3)])}
    target = _TableTarget({1: [(0.0, 0.0, [1])] * 3})
    kept = evaluate_mirror_symmetry(
        "x0", samples=samples, roles={1: "a"}, target=target, tolerance=0.1, span_tolerance=0.01,
    )
    assert not kept.accepted
    assert kept.reason == "model does not straddle the plane; there is no half to remove"
    free = evaluate_mirror_symmetry(
        "x0", samples=samples, roles={1: "a"}, target=target, tolerance=0.1,
        span_tolerance=0.01, require_straddle=False,
    )
    assert free.accepted


def test_kernel_refuses_surfaces_without_roles():
    with pytest.raises(ValueError, match="missing roles"):
        evaluate_mirror_symmetry(
            "x0", samples={1: _straddling(2)}, roles={}, target=_TableTarget({}),
            tolerance=0.1, span_tolerance=0.0,
        )


def test_ring_target_reports_every_distance():
    grid = _ellipse_grid(n_phi=16, n_axial=3)
    mirrored = grid.reshape(-1, 3).copy()
    mirrored[:, 0] += 0.5
    matches = RingMirrorTarget({"inner": grid}, tolerance=0.1).match("inner", mirrored)
    assert matches.hits is None
    assert matches.nearest_distance.shape == (16 * 3,)
    assert np.all(matches.matched_distance[matches.nearest_distance > 0.1] == -1.0)


def test_enclosure_reasons_in_order():
    reasons = evaluate_enclosure_symmetry(
        resolutions={"front_resolution": [20.0, 24.0, 28.0, 32.0], "back_resolution": [8.0] * 4},
        spacing_lr=(20.0, 30.0),
        spacing_tb=(25.0, 25.0),
        spacing_tolerance_mm=0.01,
    )
    assert reasons == {
        "x0": [
            "enclosure front_resolution quadrant sizes differ across the yz plane",
            "enclosure space_l=20 mm differs from space_r=30 mm",
        ],
        "y0": ["enclosure front_resolution quadrant sizes differ across the xz plane"],
    }


def test_enclosure_spacing_tolerance_and_non_finite():
    assert evaluate_enclosure_symmetry(spacing_lr=(25.0, 25.005), spacing_tolerance_mm=0.01) == {
        "x0": [], "y0": [],
    }
    assert evaluate_enclosure_symmetry(spacing_lr=(25.0, 25.005))["x0"]
    assert evaluate_enclosure_symmetry(spacing_tb=(None, 25.0))["y0"] == [
        "enclosure space_t/space_b spacing is not a finite scalar"
    ]
    assert evaluate_enclosure_symmetry(spacing_tb=(math.inf, math.inf))["y0"]


def test_enclosure_dim_is_compared_exactly():
    assert evaluate_enclosure_symmetry(dim_mm=(200.0, 150.0, 200.0, 150.0)) == {"x0": [], "y0": []}
    reasons = evaluate_enclosure_symmetry(dim_mm=(200.0, 150.0, 200.0 + 1e-9, 170.0))
    assert reasons["x0"] == ["enclosure Dim +X=200 mm differs from -X=200 mm"]
    assert reasons["y0"] == ["enclosure Dim +Y=150 mm differs from -Y=170 mm"]


def test_enclosure_checks_only_what_is_passed():
    assert evaluate_enclosure_symmetry() == {"x0": [], "y0": []}


def test_snap_accepts_both_plane_spellings():
    points = np.array([[5e-5, -5e-5, 1e-5], [2e-4, 1.0, -2e-4]])
    snap_symmetry_plane_vertices(points, symmetry_planes=("x", "y0"), tolerance=1e-4)
    assert points.tolist() == [[0.0, 0.0, 1e-5], [2e-4, 1.0, -2e-4]]
    snap_symmetry_plane_vertices(points, symmetry_planes=("z",), tolerance=1e-4)
    assert points[0, 2] == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        snap_symmetry_plane_vertices(points, symmetry_planes=("x0",), tolerance=-1.0)
    with pytest.raises(KeyError):
        snap_symmetry_plane_vertices(points, symmetry_planes=("w",), tolerance=1.0)


@pytest.mark.parametrize(
    "quadrants, plane",
    [
        ("1", "yz+xz"), (1, "yz+xz"), ("12", "xz"), (12, "xz"), ("14", "yz"),
        ("1234", None), (1234, None), ("21", "yz+xz"), ("", "yz+xz"), (None, "yz+xz"),
        ("1234x", None), ("x1234", "yz+xz"),
    ],
)
def test_native_plane_for_quadrants(quadrants, plane):
    assert native_symmetry_plane_for_quadrants(quadrants) == plane
