"""``build_point_grid_arrays`` is the same grid without the list round trip.

``build_point_grid``'s ``inner_points``/``outer_points`` are a published flat
list contract that in-process callers immediately reshape back into arrays.
The array form must produce exactly those numbers, in exactly that layout, or
the preview and the exported mesh stop being the same geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

from hornlab_mesher.profile_sampling import build_point_grid, build_point_grid_arrays


def _rosse_params(**overrides):
    params = {
        "type": "R-OSSE",
        "R": 140.0,
        "r0": 12.7,
        "a0": 15.5,
        "a": 25.0,
        "k": 2.0,
        "m": 0.85,
        "b": 0.2,
        "r": 0.4,
        "q": 3.4,
        "tmax": 1.0,
        "angularSegments": 24,
        "lengthSegments": 20,
        "wallThickness": 5.0,
    }
    params.update(overrides)
    return params


_CASES = {
    "rosse": _rosse_params(),
    "rosse_no_wall": _rosse_params(wallThickness=0.0),
    "rosse_offset": _rosse_params(verticalOffset=12.0),
    "rosse_morphed": _rosse_params(
        morphTarget=1.0, morphWidth=300.0, morphHeight=200.0, morphCorner=40.0
    ),
    "osse": {
        "type": "OSSE",
        "L": 130.0,
        "a": 45.0,
        "a0": 10.0,
        "r0": 12.7,
        "k": 7.0,
        "s": 0.85,
        "n": 4.0,
        "q": 0.991,
        "angularSegments": 24,
        "lengthSegments": 20,
        "wallThickness": 5.0,
    },
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_array_grid_carries_the_same_values_as_the_list_grid(case: str) -> None:
    params = _CASES[case]
    listed = build_point_grid(params)
    arrays = build_point_grid_arrays(params)

    n_phi = int(listed["grid_n_phi"])
    n_length = int(listed["grid_n_length"])
    inner = arrays["inner_grid"]
    assert inner.dtype == np.float64
    assert inner.shape == (n_phi, n_length + 1, 3)
    assert inner.reshape(-1).tolist() == listed["inner_points"]

    outer = arrays["outer_grid"]
    if listed["outer_points"] is None:
        assert outer is None
    else:
        assert outer.shape == (n_phi, n_length + 1, 3)
        assert outer.reshape(-1).tolist() == listed["outer_points"]

    for key, value in arrays.items():
        if key in {"inner_grid", "outer_grid"}:
            continue
        assert listed[key] == value, f"{case}: {key} differs between the two forms"


def test_the_list_form_keeps_its_published_key_order() -> None:
    listed = build_point_grid(_rosse_params())
    assert list(listed)[:2] == ["inner_points", "outer_points"]
    assert "inner_grid" not in listed
    assert "outer_grid" not in listed
