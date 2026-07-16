from __future__ import annotations

import pytest

from hornlab_mesher import (
    cost,
    estimate_solve_cost,
    estimate_triangle_count,
)
from hornlab_mesher.config_builder import BuildResult


def test_triangle_count_uniform_plate():
    # 100x100 mm plate at 5 mm: 2.3 * 10000 / 25 = 920
    assert estimate_triangle_count([(10_000.0, 5.0)]) == 920


def test_triangle_count_sums_per_region():
    fine = (10_000.0, 5.0)
    coarse = (10_000.0, 20.0)
    assert estimate_triangle_count([fine, coarse]) == pytest.approx(977, abs=2)


def test_matrix_ram_matches_measured_study_points():
    assert cost.matrix_ram_bytes(8000) / 1e9 == pytest.approx(1.024, rel=1e-3)
    assert cost.matrix_ram_bytes(28178) / 1e9 == pytest.approx(12.7, rel=2e-2)
    assert cost.matrix_ram_bytes(37665) / 1e9 == pytest.approx(22.7, rel=2e-2)


def test_solve_time_calibration_reproduces_anchor_points():
    assert cost.solve_seconds_per_freq(8000) == pytest.approx(1.0, rel=1e-6)
    assert cost.solve_seconds_per_freq(28178) == pytest.approx(21.0, rel=1e-6)
    assert cost.solve_seconds_per_freq(16000) > 2.0 * cost.solve_seconds_per_freq(8000)


def test_feasibility_bands():
    assert cost.feasibility_from_ram_gb(0.5) == "ok"
    assert cost.feasibility_from_ram_gb(12.0) == "caution"
    assert cost.feasibility_from_ram_gb(30.0) == "warn"
    assert cost.feasibility_from_ram_gb(45.0) == "infeasible"


def test_estimate_solve_cost_serializes_and_scales_with_freq_count():
    est = estimate_solve_cost(8000, freq_count=60)
    payload = est.to_dict()
    assert payload["n_triangles"] == 8000
    assert payload["ram_gb"] == pytest.approx(1.024, rel=1e-3)
    assert payload["feasibility"] == "ok"
    assert payload["solve_seconds_total"] == pytest.approx(
        payload["solve_seconds_per_freq"] * 60, rel=1e-3
    )


def test_build_result_carries_cost_and_raw_mesh_report():
    mesh_report = {
        "SD1G0": {"max_edge_mm": 8.0},
        "SD1D1001": {"max_edge_mm": 3.6},
    }
    result = BuildResult(
        mesh_path="/tmp/x.msh",
        formula="osse",
        mode="freestanding",
        n_vertices=100,
        n_triangles=8000,
        units="m",
        physical_groups={1: "SD1G0", 2: "SD1D1001"},
        mesh_report=mesh_report,
        solve_cost=estimate_solve_cost(8000).to_dict(),
    )
    d = result.as_dict()
    assert "valid_f_max_hz" not in d
    assert d["mesh_report"] == mesh_report
    assert d["solve_cost"]["feasibility"] == "ok"
    assert d["solve_cost"]["n_triangles"] == 8000
