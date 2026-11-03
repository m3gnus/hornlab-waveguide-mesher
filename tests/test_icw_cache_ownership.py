"""Mutable public ICW curves never expose the memo's owned curve or arrays."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_mesher import icw
from hornlab_mesher import profile_formulas as formulas
from hornlab_mesher.profile_sampling import build_point_grid_arrays
from hornlab_mesher.profiles import build_icw_curve

DIRECT = {
    "type": "ICW",
    "r0": 12.7,
    "a0": 15.5,
    "icw_coeffs": [0, 0, 0, 0, 0, 0],
    "icw_S": 80,
}


@pytest.mark.parametrize("mode", ["direct", "seed", "targets"])
def test_public_curves_own_their_fields_and_arrays(monkeypatch, mode):
    calls = []

    def make_curve(*args, **kwargs):
        calls.append(1)
        return icw.ICWCurve(np.zeros(6), S=80, r0=12.7, theta0=0.2)

    monkeypatch.setattr(icw, "seed_from_osse", make_curve)
    monkeypatch.setattr(
        icw,
        "solve_icw",
        lambda *a, **kw: (make_curve(), SimpleNamespace(feasible=True)),
    )
    params = (
        DIRECT
        if mode == "direct"
        else {"type": "ICW", "icw_seed": {"type": "OSSE", "L": 80}}
        if mode == "seed"
        else {"type": "ICW", "L": 80, "R": 50}
    )
    monkeypatch.setattr(formulas, "_ICW_CURVE_CACHE", formulas.OrderedDict())
    first = build_icw_curve(params)
    pristine = first.sample(17)
    second = build_icw_curve(params)
    assert first is not second
    assert not np.shares_memory(first.coeffs, second.coeffs)
    assert not np.shares_memory(first.knots, second.knots)
    # Keep the kernel's mutable API: a consumer can still experiment locally.
    first.r0 = 99
    first.S = 7
    first.theta0 = 0.9
    first.x0 = 5
    first.degree = 1
    first.coeffs[:] = 0.1
    first.knots[:] = -1
    second.r0 = -1
    second.coeffs[:] = -0.1
    third = build_icw_curve(params)
    sample = third.sample(17)
    for field in ("sigma", "s", "x", "r", "theta", "kappa"):
        np.testing.assert_array_equal(getattr(sample, field), getattr(pristine, field))
    assert len(calls) == (0 if mode == "direct" else 1)


def test_public_curve_mutation_cannot_change_later_point_grids(monkeypatch):
    monkeypatch.setattr(formulas, "_ICW_CURVE_CACHE", formulas.OrderedDict())
    params = {**DIRECT, "angularSegments": 8, "lengthSegments": 4}
    before = build_point_grid_arrays(params)["inner_grid"]
    curve = build_icw_curve(params)
    curve.r0 = 99
    curve.coeffs[:] = 0.01
    after = build_point_grid_arrays(params)["inner_grid"]
    np.testing.assert_array_equal(after, before)


def test_existing_direct_fixture_mesh_bytes_survive_public_mutation(
    tmp_path, monkeypatch
):
    from hornlab_mesher.config_builder import build_from_config, build_geometry_params

    monkeypatch.setattr(formulas, "_ICW_CURVE_CACHE", formulas.OrderedDict())
    fixture = Path(__file__).parent / "fixtures/throat_stretch/base-compatibility.json"
    config = json.loads(fixture.read_text())["resolved"]["ICW"]["config"]
    before = build_from_config(config, tmp_path / "before.msh")
    params, _, _ = build_geometry_params(config)
    curve = build_icw_curve(params)
    curve.r0 = -1
    curve.S = 1
    curve.coeffs[:] = 1
    curve.knots[:] = -1
    after = build_from_config(config, tmp_path / "after.msh")
    assert after.mesh_path.read_bytes() == before.mesh_path.read_bytes()
