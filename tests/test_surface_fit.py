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


def test_surface_fit_defaults_to_the_approximating_pole_fit():
    geometry = PointGridHornGeometry(inner_points=_revolved_grid())
    assert geometry.surface_fit == SURFACE_FIT_APPROXIMATE


def test_surface_fit_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="surface_fit"):
        PointGridHornGeometry(inner_points=_revolved_grid(), surface_fit="curvature")


def test_config_reads_both_surface_fit_spellings():
    from hornlab_mesher.config_builder import ConfigError, _mesh_surface_fit

    assert _mesh_surface_fit({}) == SURFACE_FIT_APPROXIMATE
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
