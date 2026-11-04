"""The grid sampler reuses H/V radii while retaining the analytic owner."""

import numpy as np
import pytest
from test_freeform_curves import corners
from test_profile_sampling_fastpaths import _freeform_config

from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.freeform import FreeformGeometry, build_freeform_geometry
from hornlab_mesher.profile_sampling import build_point_grid_arrays


@pytest.mark.parametrize("extended", [False, True])
def test_grid_evaluates_the_two_meridians_once(monkeypatch, extended):
    params = build_geometry_params(_freeform_config("1234"))[0]
    if extended:
        params["profileH"], params["profileV"] = corners(), corners()
    # Finish ingest before counting spline queries during sampling.
    build_freeform_geometry(params)
    original = FreeformGeometry.evaluate_radii
    queries = []

    def counted(self, z):
        queries.append(np.asarray(z).shape)
        return original(self, z)

    monkeypatch.setattr(FreeformGeometry, "evaluate_radii", counted)
    grid = build_point_grid_arrays(params)
    assert queries == [(len(grid["slice_map"]),)]


@pytest.mark.parametrize("quadrants", ["1", "12", "14", "1234"])
@pytest.mark.parametrize("target", [0, 1, 2, 3])
@pytest.mark.parametrize("rate", [3.0, "3 + 0.1*cos(p)"])
@pytest.mark.parametrize("extended", [False, True])
def test_sampled_rings_match_public_analytic_radii_exactly(quadrants, target, rate, extended):
    params = build_geometry_params(_freeform_config(quadrants))[0]
    if extended:
        params["profileH"], params["profileV"] = corners(), corners()
        for profile in (params["profileH"], params["profileV"]):
            for point in profile["points"]:
                point["z"] += 37.0
    params.update(morphTarget=target, morphCorner=10.0, morphWidth=360.0,
                  morphHeight=260.0, morphRate=rate, morphFixed=0.2)
    if target in {2, 3}:
        params["crossSections"] = [{"t": 0.0, "shape": "circle"},
                                   {"t": 1.0, "shape": "ellipse"}]
    geometry = build_freeform_geometry(params)
    grid = build_point_grid_arrays(params)
    inner = grid["inner_grid"]
    phi = np.asarray(grid["phi_grid"])
    for j, t in enumerate(grid["slice_map"]):
        radius = geometry.cross_section_radius(phi[:, j], float(t))
        expected_x = radius * np.cos(phi[:, j])
        expected_y = radius * np.sin(phi[:, j])
        assert inner[:, j, 0].tobytes() == expected_x.tobytes()
        assert inner[:, j, 1].tobytes() == expected_y.tobytes()
