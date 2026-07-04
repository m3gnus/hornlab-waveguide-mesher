"""Tests for the selectable ICW gene/nullspace basis.

The default ``gene_basis="svd"`` must keep the historical SVD gauge exactly. The opt-in
``gene_basis="local"`` gives a deterministic, station-ordered basis for saved genomes and
warm-starts.
"""

from __future__ import annotations

import numpy as np
import pytest

from hornlab_mesher.icw import (
    ICWTargets,
    TerminationMode,
    curve_from_shape_modes,
    gene_basis_diagnostics,
)
from hornlab_mesher.icw.solver import (
    DEFAULT_DEGREE,
    _build_linsys,
    _knots_for_targets,
    build_linear_constraints,
    feasible_subspace,
)


def _flat_targets() -> ICWTargets:
    return ICWTargets(
        mode=TerminationMode.FLAT_BAFFLE,
        r0=12.7,
        theta0_deg=18.0,
        x_target=120.0,
        r_mouth=116.0,
    )


def _coverage_targets() -> ICWTargets:
    return ICWTargets(
        mode=TerminationMode.FLAT_BAFFLE,
        r0=12.7,
        theta0_deg=14.5,
        x_target=160.0,
        coverage_angle_deg=45.0,
    )


def _lin(targets: ICWTargets, n_coeff: int, gene_basis: str):
    knots = _knots_for_targets(targets, n_coeff, DEFAULT_DEGREE)
    return knots, _build_linsys(
        targets, knots, n_coeff, DEFAULT_DEGREE, gene_basis=gene_basis
    )


def test_default_svd_basis_is_historical_feasible_subspace_basis() -> None:
    targets = _flat_targets()
    n_coeff = 12
    knots, lin = _lin(targets, n_coeff, gene_basis="svd")
    C, d = build_linear_constraints(targets, 1.0, knots, n_coeff, DEFAULT_DEGREE)
    _a0, Phi_ref = feasible_subspace(C, d)

    assert np.array_equal(lin.Phi, Phi_ref)


@pytest.mark.parametrize(
    ("targets", "n_coeff"),
    [
        (_flat_targets(), 12),
        (_coverage_targets(), 16),
    ],
)
def test_local_basis_spans_nullspace_and_is_station_ordered(
    targets: ICWTargets, n_coeff: int
) -> None:
    knots, lin = _lin(targets, n_coeff, gene_basis="local")
    C, d = build_linear_constraints(targets, 1.0, knots, n_coeff, DEFAULT_DEGREE)
    _a0, Phi_svd = feasible_subspace(C, d)
    Phi = lin.Phi
    D = Phi_svd.shape[1]

    assert Phi.shape == (n_coeff, D)
    assert np.linalg.norm(C @ Phi) < 1e-8
    assert np.allclose(Phi.T @ Phi, np.eye(D), atol=1e-10)

    _a0, Phi_via_feasible = feasible_subspace(
        C, d, gene_basis="local", knots=knots, degree=DEFAULT_DEGREE
    )
    assert np.array_equal(Phi, Phi_via_feasible)

    diag = gene_basis_diagnostics(targets, n_coeff=n_coeff, gene_basis="local")
    centroids = diag["centroid_sigma"]
    assert centroids.shape == (D,)
    assert np.all(np.diff(centroids) >= -1e-12)
    assert diag["gene_roughness"].shape == (D,)


def test_local_basis_is_bitwise_deterministic() -> None:
    targets = _coverage_targets()
    n_coeff = 16
    _knots_a, lin_a = _lin(targets, n_coeff, gene_basis="local")
    _knots_b, lin_b = _lin(targets, n_coeff, gene_basis="local")

    assert np.array_equal(lin_a.Phi, lin_b.Phi)


def test_local_gene_basis_preserves_bcs_and_coverage_plateau_for_random_genes() -> None:
    targets = _coverage_targets()
    rng = np.random.default_rng(20260704)

    for b_gene in [
        rng.normal(scale=0.03, size=1),
        rng.normal(scale=0.03, size=2),
        rng.normal(scale=0.03, size=3),
    ]:
        curve, report = curve_from_shape_modes(b_gene, targets, gene_basis="local")
        assert report.feasible, report.violations

        assert abs(float(curve.kappa(np.array([1.0]))[0])) < 1e-8
        dkappa1 = float(curve.kappa_spline()(np.array([1.0]), nu=1)[0])
        assert abs(dkappa1) < 1e-7
        assert np.degrees(curve.theta_end()) == pytest.approx(90.0, abs=1e-6)

        hold_sigma = np.linspace(targets.hold_start, targets.hold_end, 401)
        assert np.max(np.abs(curve.kappa(hold_sigma))) < 1e-8
        sample = curve.sample(2001)
        sigma_mid = 0.5 * (targets.hold_start + targets.hold_end)
        theta_mid = float(np.interp(sigma_mid, sample.sigma, sample.theta))
        assert np.degrees(theta_mid) == pytest.approx(targets.coverage_angle_deg, abs=1e-6)


def test_local_gene_perturbation_is_near_its_basis_centroid() -> None:
    targets = _flat_targets()
    n_coeff = 16
    diag = gene_basis_diagnostics(targets, n_coeff=n_coeff, gene_basis="local")
    j = 2

    baseline, report_base = curve_from_shape_modes(
        np.zeros(0), targets, n_coeff=n_coeff, gene_basis="local"
    )
    perturbed, report_pert = curve_from_shape_modes(
        np.array([0.0, 0.0, 0.05]), targets, n_coeff=n_coeff, gene_basis="local"
    )
    assert report_base.feasible, report_base.violations
    assert report_pert.feasible, report_pert.violations

    sigma = np.linspace(0.0, 1.0, 2001)
    delta_kappa = perturbed.kappa(sigma) - baseline.kappa(sigma)
    weights = np.abs(delta_kappa)
    delta_centroid = float(np.sum(sigma * weights) / np.sum(weights))

    assert abs(delta_centroid - diag["centroid_sigma"][j]) < 0.2
