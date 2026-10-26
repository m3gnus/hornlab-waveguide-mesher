"""Acoustic B-spline surface fitting.

``occ.addBSplineSurface`` treats its points as *poles*, so handing it the
sampled profile grid meshed a surface that sat systematically inside the
sampled one -- an error floor no element size could reach. These cover the
interpolating fit that removes it.
"""

from __future__ import annotations

import numpy as np
import pytest

from hornlab_mesher.builders._occ import (
    SURFACE_FIT_APPROXIMATE,
    SURFACE_FIT_INTERPOLATE,
    angular_interpolating_surface_poles,
    grid_v_parameters,
    interpolating_surface_poles,
)
from hornlab_mesher.geometry import PointGridHornGeometry


def _revolved_grid(n_phi: int = 24, n_len: int = 12) -> np.ndarray:
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    z = np.linspace(0.0, 40.0, n_len)
    grid = np.empty((n_phi, n_len, 3), dtype=np.float64)
    for i, angle in enumerate(phi):
        radius = 10.0 + 0.6 * z
        grid[i, :, 0] = radius * np.cos(angle)
        grid[i, :, 1] = radius * np.sin(angle)
        grid[i, :, 2] = z
    return grid


# --------------------------------------------------------------------------
# B-spline surface fitting
# --------------------------------------------------------------------------


def test_interpolating_poles_reproduce_the_sampled_grid_exactly():
    """The whole point of the fit: the surface passes through its samples.

    A pole fit does not, and that miss is the error floor the acoustic mesh
    was converging onto.
    """
    from scipy.interpolate import BSpline

    grid = _revolved_grid()
    columns = list(range(grid.shape[0])) + [0]
    patch = np.ascontiguousarray(grid[columns, :, :].transpose(1, 0, 2))

    poles, (knots_u, mults_u), (knots_v, mults_v) = interpolating_surface_poles(
        patch, degree_u=3, degree_v=3, v_params=grid_v_parameters(grid)
    )
    assert poles.shape == patch.shape

    full_u = np.repeat(knots_u, mults_u)
    full_v = np.repeat(knots_v, mults_v)
    # A clamped knot vector carries n + degree + 1 entries.
    assert len(full_u) == poles.shape[1] + 3 + 1
    assert len(full_v) == poles.shape[0] + 3 + 1

    u_params = np.linspace(full_u[0], full_u[-1], patch.shape[1])
    v_params = np.linspace(full_v[0], full_v[-1], patch.shape[0])
    worst = 0.0
    for vi, v in enumerate(v_params):
        row = BSpline(full_v, poles, 3)(v)
        for ui, u in enumerate(u_params):
            worst = max(worst, float(np.linalg.norm(BSpline(full_u, row, 3)(u) - patch[vi, ui])))
    assert worst < 1e-9, worst


def test_angular_only_fit_reproduces_ring_samples_without_fitting_axially():
    """Outer shells can fix ring sag without interpolating their rear corner."""

    from scipy.interpolate import BSpline

    grid = _revolved_grid()
    columns = list(range(grid.shape[0] // 4 + 1))
    patch = np.ascontiguousarray(grid[columns, :, :].transpose(1, 0, 2))
    poles, (knots_u, mults_u) = angular_interpolating_surface_poles(
        patch, degree_u=3
    )

    full_u = np.repeat(knots_u, mults_u)
    u_params = np.linspace(full_u[0], full_u[-1], patch.shape[1])
    actual = np.stack([BSpline(full_u, row, 3)(u_params) for row in poles])
    assert poles.shape == patch.shape
    assert np.allclose(actual, patch, rtol=0.0, atol=1.0e-10)


def test_seam_poles_coincide_so_a_closed_patch_still_closes():
    grid = _revolved_grid()
    columns = list(range(grid.shape[0])) + [0]
    patch = np.ascontiguousarray(grid[columns, :, :].transpose(1, 0, 2))
    poles, _, _ = interpolating_surface_poles(
        patch, degree_u=3, degree_v=3, v_params=grid_v_parameters(grid)
    )
    assert np.allclose(poles[:, 0], poles[:, -1], atol=0.0)


def test_patches_cut_from_one_grid_share_a_v_knot_vector():
    """Neighbouring patches must trace the *same* seam curve.

    Deriving v from each patch's own columns gives each a slightly different
    parameterisation, which tears the shell open along every seam.
    """
    grid = _revolved_grid()
    shared = grid_v_parameters(grid)
    left = np.ascontiguousarray(grid[list(range(0, 13)), :, :].transpose(1, 0, 2))
    right = np.ascontiguousarray(grid[list(range(12, 24)) + [0], :, :].transpose(1, 0, 2))

    _, _, (knots_left, mults_left) = interpolating_surface_poles(
        left, degree_u=3, degree_v=3, v_params=shared
    )
    _, _, (knots_right, mults_right) = interpolating_surface_poles(
        right, degree_u=3, degree_v=3, v_params=shared
    )
    assert np.allclose(knots_left, knots_right)
    assert mults_left == mults_right


def test_degenerate_rings_fall_back_to_a_uniform_parameterisation():
    """A collapsed apex ring must not produce a singular collocation system."""
    grid = _revolved_grid(n_phi=16, n_len=6)
    grid[:, 0, :] = grid[0, 0, :]  # collapse the first ring to a single point
    patch = np.ascontiguousarray(grid[list(range(16)) + [0], :, :].transpose(1, 0, 2))
    poles, _, _ = interpolating_surface_poles(patch, degree_u=3, degree_v=3)
    assert np.all(np.isfinite(poles))


def test_surface_fit_defaults_to_the_interpolating_fit():
    """``auto`` is the default and resolves to the fit that is measurably truer.

    Over twelve ATH reference configs the interpolating fit moves the median
    ``rms x triangles`` against the analytic surface from 0.93x of ATH's to
    1.24x, at a triangle count that moves by at most 4%.
    """

    geometry = PointGridHornGeometry(inner_points=_revolved_grid())
    assert geometry.surface_fit == SURFACE_FIT_INTERPOLATE


def test_auto_resolves_to_the_approximating_fit_on_freeform():
    """The one family the interpolating fit cannot mesh keeps the old one."""

    geometry = PointGridHornGeometry(
        inner_points=_revolved_grid(), freeform_report={"axis": "z"}
    )
    assert geometry.surface_fit == SURFACE_FIT_APPROXIMATE


def test_naming_a_fit_explicitly_still_pins_it():
    for mode in (SURFACE_FIT_APPROXIMATE, SURFACE_FIT_INTERPOLATE):
        geometry = PointGridHornGeometry(
            inner_points=_revolved_grid(), surface_fit=mode
        )
        assert geometry.surface_fit == mode


def test_surface_fit_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="surface_fit"):
        PointGridHornGeometry(inner_points=_revolved_grid(), surface_fit="curvature")


def test_config_reads_both_surface_fit_spellings():
    from hornlab_mesher.config_builder import ConfigError, _mesh_surface_fit

    assert _mesh_surface_fit({}) == "auto"
    assert _mesh_surface_fit({"surface_fit": "approximate"}) == SURFACE_FIT_APPROXIMATE
    assert _mesh_surface_fit({"surface_fit": "interpolate"}) == SURFACE_FIT_INTERPOLATE
    assert _mesh_surface_fit({"surfaceFit": "Interpolate"}) == SURFACE_FIT_INTERPOLATE
    with pytest.raises(ConfigError, match="surface_fit"):
        _mesh_surface_fit({"surface_fit": "nearest"})


def test_interpolate_is_refused_on_freeform_profiles():
    """FREEFORM creases make the interpolating fit unmeshable, not just worse.

    examples/freeform-bare ran past ten minutes inside gmsh's 2D mesher with
    this enabled. Refusing is the honest outcome: a silent fallback would hide
    that the requested fit was not the one used.
    """
    with pytest.raises(ValueError, match="FREEFORM"):
        PointGridHornGeometry(
            inner_points=_revolved_grid(),
            surface_fit=SURFACE_FIT_INTERPOLATE,
            freeform_report={"freeformProfileDeviationMm": 0.0},
        )


def test_freeform_still_builds_with_the_default_fit():
    geometry = PointGridHornGeometry(
        inner_points=_revolved_grid(),
        freeform_report={"freeformProfileDeviationMm": 0.0},
    )
    assert geometry.surface_fit == SURFACE_FIT_APPROXIMATE


# --------------------------------------------------------------------------
# Reduced-domain seams
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source_shape", [0, 1], ids=["flat-disc", "rounded-cap"])
@pytest.mark.parametrize("quadrants", [1234, 12, 14, 1])
@pytest.mark.parametrize(
    ("mode", "extra"),
    [
        ("freestanding", {"wall_thickness_mm": 6.0}),
        ("bare", {"wall_thickness_mm": 0.0}),
        ("infinite-baffle", {"wall_thickness_mm": 0.0}),
        ("enclosure", {"wall_thickness_mm": 0.0}),
    ],
)
def test_the_interpolating_fit_still_welds_a_reduced_domain(
    tmp_path, quadrants: int, mode: str, extra: dict, source_shape: int
):
    """A symmetry-cut shell must close under either fit.

    The open sector source cap authors its own throat rim and relies on
    ``mesh.removeDuplicateNodes`` welding it to the wall's edge, which works
    only while the two are the same curve. A plain pole B-spline through the
    ring points is the wall's edge under ``approximate`` and is not under
    ``interpolate``, so every reduced-domain build -- all three cut domains, in
    all four modes -- used to fail its own closed/open-shell contract at the
    source. The full domain never did, because a closed wall fills its cap on
    its own boundary curves.
    """

    config = {
        "formula": "OSSE",
        "mode": mode,
        "profile": {"L_mm": 120.0, "r0_mm": 12.7, "a_deg": 60.0, "a0_deg": 15.5,
                    "k": 1.0, "n": 4.0, "q": 0.995, "s": 0.0},
        "mesh": {
            "angular_segments": 64, "length_segments": 32, "throat_res_mm": 4.0,
            "mouth_res_mm": 26.0, "rear_res_mm": 25.0, "quadrants": quadrants,
            "surface_fit": SURFACE_FIT_INTERPOLATE, **extra,
        },
        # Both source shapes matter: a flat disc fills an open throat as a
        # sector on a re-authored ring curve, a rounded cap fans radial curves
        # to a pole off one. Each authors the rim, and each has to author the
        # wall's rim.
        "source": {"source_shape": source_shape, "source_radius": -1},
    }
    if mode == "enclosure":
        config["enclosure"] = {"depth_mm": 180.0}

    from hornlab_mesher import build_from_config
    from hornlab_mesher.config_builder import resolve_geometry

    # The shape has to reach the builder, or half this matrix is the same case
    # twice over.
    assert resolve_geometry(config).geometry.source_shape == source_shape

    result = build_from_config(config, tmp_path / f"{mode}-{quadrants}-{source_shape}.msh")
    assert result.n_triangles > 0


@pytest.mark.parametrize(
    "source_shape,source_curv",
    [(0, 0), (1, 1), (1, -1)],
    ids=["flat-disc", "rounded-cap", "concave-cap"],
)
@pytest.mark.parametrize("quadrants", [1, 12, 14])
@pytest.mark.parametrize(
    "formula,profile",
    [
        (
            "OSSE",
            {"L_mm": 120.0, "r0_mm": 12.0, "a_deg": 45.0, "a0_deg": 4.0},
        ),
        (
            "R-OSSE",
            {
                "R_mm": 120.0, "r0_mm": 12.0, "a_deg": 45.0, "a0_deg": 4.0,
                "tmax": 1.0, "m": 0.85, "r": 0.4, "b": 0.2, "k": 2.0, "q": 3.4,
            },
        ),
    ],
    ids=["OSSE", "R-OSSE"],
)
def test_approximate_fit_welds_a_freestanding_axisymmetric_throat(
    tmp_path, formula, profile, quadrants, source_shape, source_curv
):
    """H4: freestanding + reduced domain + ``approximate`` used to tear the seam.

    An axisymmetric grid's freestanding inner wall always builds with the
    interpolating u-fit (to keep the meridian circular), regardless of
    ``surface_fit``. The open-sector throat cap has to match that actual
    decision rather than re-derive its own from ``surface_fit`` alone, or the
    wall and cap rims mesh different node rings and the closed-shell contract
    refuses the build with a free-edge error at the throat. Covers every
    reduced domain (q1/q12/q14) and source shape (flat/rounded/concave) for
    both axisymmetric formulas.
    """

    from hornlab_mesher import build_from_config

    config = {
        "formula": formula,
        "mode": "freestanding",
        "profile": profile,
        "mesh": {
            "angular_segments": 32, "length_segments": 12,
            "throat_res_mm": 6.0, "mouth_res_mm": 20.0, "rear_res_mm": 30.0,
            "wall_thickness_mm": 5.0, "quadrants": quadrants,
            "surface_fit": SURFACE_FIT_APPROXIMATE, "max_triangles": 40000,
        },
        "source": {"source_shape": source_shape, "source_curv": source_curv},
    }
    result = build_from_config(
        config, tmp_path / f"{formula}-{quadrants}-{source_shape}-{source_curv}.msh"
    )
    assert result.n_triangles > 0


def test_the_throat_boundary_curve_reproduces_the_patch_edge():
    """The rim the cap authors has to trace the wall patch's own v = 0 isocurve."""

    from scipy.interpolate import BSpline

    from hornlab_mesher.builders._occ import (
        grid_v_parameters,
        interpolating_surface_poles,
        throat_boundary_curve,
    )

    grid = _revolved_grid(n_phi=20, n_len=9)
    columns = list(range(grid.shape[0]))
    patch = np.ascontiguousarray(grid[columns, :, :].transpose(1, 0, 2))

    poles, (knots_u, mults_u), (knots_v, mults_v) = interpolating_surface_poles(
        patch, degree_u=3, degree_v=3, v_params=grid_v_parameters(grid)
    )
    rim_poles, (rim_knots, rim_mults), rim_degree = throat_boundary_curve(grid, columns)

    assert rim_degree == 3
    assert np.allclose(rim_knots, knots_u)
    assert rim_mults == mults_u
    # The surface's v = 0 isocurve carries its first pole row, so the two must
    # be the same curve pole for pole.
    assert np.allclose(rim_poles, poles[0], atol=1e-12)

    # And that row does not depend on the v parameterisation, which is why one
    # rim curve serves both the patch that derives v from its own columns and
    # the patch handed a shared v.
    own_v_poles, _, _ = interpolating_surface_poles(patch, degree_u=3, degree_v=3)
    assert np.allclose(rim_poles, own_v_poles[0], atol=1e-12)

    # And that curve passes through the sampled throat ring.
    full = np.repeat(rim_knots, rim_mults)
    curve = BSpline(full, rim_poles, rim_degree)
    for index, u in enumerate(np.linspace(full[0], full[-1], len(columns))):
        assert np.allclose(curve(u), grid[columns[index], 0, :], atol=1e-9)


@pytest.mark.parametrize("surface_fit", [SURFACE_FIT_INTERPOLATE, SURFACE_FIT_APPROXIMATE])
@pytest.mark.parametrize("station_index", [0, -1])
@pytest.mark.parametrize("n_columns", [2, 3, 7, 13])
def test_sample_patch_boundary_matches_occ(surface_fit, station_index, n_columns):
    """Both axial ends follow the actual wall edge, including its fit bias."""
    gmsh = pytest.importorskip("gmsh")
    from hornlab_mesher.builders._occ import add_bspline_patch, sample_patch_boundary
    from hornlab_mesher.builders.enclosure import _boundary_curves_at_z_extreme

    grid = _revolved_grid(n_phi=32, n_len=9)
    # Distort angular spacing differently on each axial ring. This detects a
    # boundary fit that incorrectly uses only the mouth ring's chord lengths.
    grid[:, :, 0] *= 1.0 + 0.17 * (
        np.arange(len(grid), dtype=np.float64)[:, None] / len(grid)
    ) ** 2 * (grid[:, :, 2] / 40.0)
    columns = list(range(n_columns))
    samples = np.unique(np.r_[np.linspace(0.0, 1.0, 257), 0.01234, 0.934567])

    gmsh.initialize(interruptible=False)
    try:
        gmsh.model.add("patch-boundary-fit")
        surface = add_bspline_patch(
            grid, columns, degree_v=3, surface_fit=surface_fit,
            v_params=grid_v_parameters(grid),
        )
        gmsh.model.occ.synchronize()
        edges = _boundary_curves_at_z_extreme(
            [(2, surface)], want_min_z=(station_index == 0),
        )
        assert len(edges) == 1
        lower, upper = gmsh.model.getParametrizationBounds(1, edges[0])
        actual_u = lower[0] + samples * (upper[0] - lower[0])
        actual = np.asarray(gmsh.model.getValue(1, edges[0], actual_u)).reshape(-1, 3)
        result = sample_patch_boundary(
            grid, columns, samples, station_index=station_index, surface_fit=surface_fit,
        )
        np.testing.assert_allclose(result, actual, rtol=0.0, atol=1e-8)
    finally:
        gmsh.finalize()


@pytest.mark.parametrize("station_index", [0, -1])
def test_sample_patch_boundary_preserves_the_grid_edges(station_index):
    from hornlab_mesher.builders._occ import sample_patch_boundary

    grid = _revolved_grid(n_phi=20, n_len=9)
    columns = [0, 3, 7, 12, 18]
    nodes = grid[columns, station_index, :]
    values = np.arange(len(columns), dtype=np.float64) / (len(columns) - 1)
    actual = sample_patch_boundary(
        grid, columns, values, station_index=station_index, preserve_grid=True,
    )
    np.testing.assert_array_equal(actual, nodes)
    midpoints = (values[:-1] + values[1:]) / 2.0
    actual = sample_patch_boundary(
        grid, columns, midpoints, station_index=station_index, preserve_grid=True,
    )
    np.testing.assert_allclose(actual, (nodes[:-1] + nodes[1:]) / 2.0, atol=1e-12)


def test_sample_patch_boundary_rejects_non_boundary_or_invalid_parameters():
    from hornlab_mesher.builders._occ import sample_patch_boundary

    grid = _revolved_grid(n_phi=8, n_len=5)
    columns = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="throat or mouth"):
        sample_patch_boundary(grid, columns, np.array([0.5]), station_index=2)
    for values in ([float("nan")], [-0.01], [1.01], [[0.5]]):
        with pytest.raises(ValueError, match="parameters"):
            sample_patch_boundary(grid, columns, np.asarray(values))
    with pytest.raises(ValueError, match="surface_fit"):
        sample_patch_boundary(grid, columns, np.array([0.5]), surface_fit="auto")


@pytest.mark.parametrize("surface_fit", [SURFACE_FIT_INTERPOLATE, SURFACE_FIT_APPROXIMATE])
def test_patch_boundary_xy_bounds_include_fitted_overshoot(surface_fit):
    """Bounds follow the wall fit, including extrema between control points."""
    from hornlab_mesher.builders._occ import patch_boundary_xy_bounds, sample_patch_boundary

    grid = _revolved_grid(n_phi=12, n_len=6)
    columns = list(range(6))
    grid[columns, -1, 0] = [0, 1, 1, 1, 0, 0]
    grid[columns, -1, 1] = [0, 1, 2, 3, 4, 5]

    minimum, maximum = patch_boundary_xy_bounds(grid, columns, surface_fit=surface_fit)
    dense = sample_patch_boundary(
        grid, columns, np.linspace(0.0, 1.0, 20001), surface_fit=surface_fit,
    )[:, :2]
    sampled_min, sampled_max = np.min(dense, axis=0), np.max(dense, axis=0)
    assert np.all(minimum <= sampled_min + 1e-12)
    assert np.all(maximum >= sampled_max - 1e-12)
    np.testing.assert_allclose(minimum, sampled_min, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(maximum, sampled_max, rtol=0.0, atol=1e-5)
    if surface_fit == SURFACE_FIT_INTERPOLATE:
        assert minimum[0] < -0.3 and maximum[0] > 1.08
    else:
        assert minimum[0] == pytest.approx(0.0) and maximum[0] == pytest.approx(1.0)


@pytest.mark.parametrize("surface_fit", [SURFACE_FIT_INTERPOLATE, SURFACE_FIT_APPROXIMATE])
def test_patch_boundary_xy_bounds_handle_constant_coordinate(surface_fit):
    from hornlab_mesher.builders._occ import patch_boundary_xy_bounds

    grid = _revolved_grid(n_phi=12, n_len=6)
    grid[:, :, 0] = 42.0
    minimum, maximum = patch_boundary_xy_bounds(grid, list(range(6)), surface_fit=surface_fit)
    assert np.all(np.isfinite(minimum)) and np.all(np.isfinite(maximum))
    assert minimum[0] == pytest.approx(42.0, abs=1e-10)
    assert maximum[0] == pytest.approx(42.0, abs=1e-10)


def test_patch_boundary_xy_bounds_preserved_grid_uses_original_vertices():
    from hornlab_mesher.builders._occ import patch_boundary_xy_bounds

    grid = _revolved_grid(n_phi=20, n_len=9)
    columns = [0, 3, 7, 12, 18]
    ring = grid[columns, -1, :2]
    minimum, maximum = patch_boundary_xy_bounds(grid, columns, preserve_grid=True)
    np.testing.assert_array_equal(minimum, np.min(ring, axis=0))
    np.testing.assert_array_equal(maximum, np.max(ring, axis=0))


@pytest.mark.parametrize("surface_fit", [SURFACE_FIT_INTERPOLATE, SURFACE_FIT_APPROXIMATE])
def test_patch_boundary_xy_bounds_two_point_linear_patch(surface_fit):
    from hornlab_mesher.builders._occ import patch_boundary_xy_bounds

    grid = _revolved_grid(n_phi=8, n_len=5)
    grid[[0, 1], -1, :2] = [[-4.0, 5.0], [12.0, -3.0]]
    minimum, maximum = patch_boundary_xy_bounds(grid, [0, 1], surface_fit=surface_fit)
    np.testing.assert_allclose(minimum, [-4.0, -3.0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(maximum, [12.0, 5.0], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("surface_fit", [SURFACE_FIT_INTERPOLATE, SURFACE_FIT_APPROXIMATE])
def test_patch_boundary_bounds_include_nonplanar_fitted_z_overshoot(surface_fit):
    from hornlab_mesher.builders._occ import (
        patch_boundary_bounds, patch_boundary_xy_bounds, sample_patch_boundary,
    )

    grid = _revolved_grid(n_phi=12, n_len=6)
    columns = list(range(6))
    grid[columns, -1, 2] = [0, 1, 1, 1, 0, 0]
    minimum, maximum = patch_boundary_bounds(grid, columns, surface_fit=surface_fit)
    dense = sample_patch_boundary(
        grid, columns, np.linspace(0.0, 1.0, 20001), surface_fit=surface_fit,
    )
    sampled_min, sampled_max = np.min(dense, axis=0), np.max(dense, axis=0)
    assert np.all(minimum <= sampled_min + 1e-12)
    assert np.all(maximum >= sampled_max - 1e-12)
    np.testing.assert_allclose(minimum, sampled_min, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(maximum, sampled_max, rtol=0.0, atol=1e-5)
    if surface_fit == SURFACE_FIT_INTERPOLATE:
        assert minimum[2] < -0.3 and maximum[2] > 1.08
    else:
        assert minimum[2] == pytest.approx(0.0) and maximum[2] == pytest.approx(1.0)
    xy_minimum, xy_maximum = patch_boundary_xy_bounds(grid, columns, surface_fit=surface_fit)
    np.testing.assert_array_equal(xy_minimum, minimum[:2])
    np.testing.assert_array_equal(xy_maximum, maximum[:2])


@pytest.mark.parametrize("surface_fit", [SURFACE_FIT_INTERPOLATE, SURFACE_FIT_APPROXIMATE])
def test_patch_boundary_bounds_handle_planar_z(surface_fit):
    from hornlab_mesher.builders._occ import patch_boundary_bounds

    grid = _revolved_grid(n_phi=12, n_len=6)
    minimum, maximum = patch_boundary_bounds(grid, list(range(6)), surface_fit=surface_fit)
    assert np.all(np.isfinite(minimum)) and np.all(np.isfinite(maximum))
    assert minimum[2] == pytest.approx(40.0, abs=1e-10)
    assert maximum[2] == pytest.approx(40.0, abs=1e-10)


def test_patch_boundary_bounds_preserved_grid_handles_nonplanar_edges():
    from hornlab_mesher.builders._occ import patch_boundary_bounds

    grid = _revolved_grid(n_phi=12, n_len=6)
    columns = list(range(6))
    grid[columns, -1, 2] = [0, 1, 1, 1, 0, -2]
    minimum, maximum = patch_boundary_bounds(grid, columns, preserve_grid=True)
    np.testing.assert_array_equal(minimum, np.min(grid[columns, -1, :], axis=0))
    np.testing.assert_array_equal(maximum, np.max(grid[columns, -1, :], axis=0))


@pytest.mark.parametrize("scale", [0.1, 1.0, 12.7])
def test_coarse_small_interpolating_shell_remains_manifold(tmp_path, caplog, scale):
    """Curve sampling must preserve topology even below the default mesh size."""

    pytest.importorskip("gmsh")
    from hornlab_mesher.config_builder import build_from_config

    config = {
        "formula": "OSSE",
        "mode": "bare",
        "profile": {"r0": scale, "a": 2.0, "a0": 0.0, "L": 4.0 * scale},
        "mesh": {
            "angularSegments": 12,
            "lengthSegments": 4,
            "scaleToMetres": False,
        },
        "source": {"sourceShape": 0},
    }

    result = build_from_config(config, tmp_path / "auto.msh")
    assert result.n_triangles > 0

    pinned = {**config, "mesh": {**config["mesh"], "surface_fit": "interpolate"}}
    interpolated = build_from_config(pinned, tmp_path / "pinned.msh")
    assert interpolated.n_triangles > 0
    assert "falling back" not in caplog.text


def _small_shell_config(surface_fit="auto"):
    return {
        "formula": "OSSE", "mode": "bare",
        "profile": {"r0": 1.0, "a": 2.0, "a0": 0.0, "L": 4.0},
        "mesh": {"angularSegments": 12, "lengthSegments": 4,
                 "scaleToMetres": False, "surface_fit": surface_fit},
        "source": {"sourceShape": 0},
    }


def test_interpolating_mesh_preserves_callers_curve_sampling_option(tmp_path):
    gmsh = pytest.importorskip("gmsh")
    from hornlab_mesher.config_builder import build_from_config

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("Mesh.MinimumCurvePoints", 11)
        result = build_from_config(_small_shell_config("interpolate"), tmp_path / "caller.msh")
        assert result.n_triangles > 0
        assert gmsh.isInitialized()
        assert gmsh.option.getNumber("Mesh.MinimumCurvePoints") == 11
    finally:
        gmsh.finalize()


@pytest.mark.parametrize("measure_profile_deviation", [False, True])
def test_auto_fit_recovers_from_failure_but_explicit_fit_propagates(
    tmp_path, monkeypatch, caplog, measure_profile_deviation
):
    pytest.importorskip("gmsh")
    import hornlab_mesher.config_builder as builder
    from hornlab_mesher.mesher import MesherError

    original = builder.build_mesh_with_info
    attempts = []

    def fail_interpolation(geometry, *args, **kwargs):
        assert kwargs["measure_profile_deviation"] is measure_profile_deviation
        attempts.append(geometry.surface_fit)
        if geometry.surface_fit == "interpolate":
            raise MesherError("injected interpolation failure")
        return original(geometry, *args, **kwargs)

    monkeypatch.setattr(builder, "build_mesh_with_info", fail_interpolation)
    result = builder.build_from_config(
        _small_shell_config(), tmp_path / "recovered.msh",
        measure_profile_deviation=measure_profile_deviation,
    )
    assert result.n_triangles > 0
    assert attempts == ["interpolate", "approximate"]
    assert "falling back" in caplog.text
    attempts.clear()
    with pytest.raises(MesherError, match="injected interpolation failure"):
        builder.build_from_config(
            _small_shell_config("interpolate"), tmp_path / "explicit.msh",
            measure_profile_deviation=measure_profile_deviation,
        )
    assert attempts == ["interpolate"]
    assert not (tmp_path / "explicit.msh").exists()
