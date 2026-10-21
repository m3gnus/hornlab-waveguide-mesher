"""A non-circular cross-section must leave the throat circular.

The throat is where the compression driver bolts on, so the cross-section
exponent and aspect ratio may only reshape the horn past it.
"""

from __future__ import annotations

import numpy as np
import pytest

from hornlab_mesher.profile_sampling import build_point_grid_arrays


_OSSE = {
    "type": "OSSE", "L": 150.0, "r0": 25.0, "a0": 10.0, "a": 40.0, "k": 1.0,
    "n": 4.0, "q": 0.995, "s": 0.5,
    "angularSegments": 64, "lengthSegments": 20, "quadrants": "1234",
    "wallThickness": 0.0, "encDepth": 0.0,
}
_ROSSE = {
    **_OSSE, "type": "R-OSSE", "R": 150.0, "r": 0.4, "b": 0.2, "m": 0.85, "q": 3.4,
}


def _section(exponent, aspect_ratio):
    return {"profileSystem": {"crossSection": {"exponent": exponent, "aspectRatio": aspect_ratio}}}


def _radii(params):
    grid = build_point_grid_arrays(params)["inner_grid"]
    return np.hypot(grid[..., 0], grid[..., 1])


@pytest.mark.parametrize("base", [_OSSE, _ROSSE], ids=["OSSE", "R-OSSE"])
@pytest.mark.parametrize("exponent, aspect_ratio", [(4.0, 1.0), (2.0, 1.5), (6.0, 0.7)])
def test_non_circular_section_keeps_the_throat_circle(base, exponent, aspect_ratio):
    circular = _radii(base)
    shaped = _radii({**base, **_section(exponent, aspect_ratio)})
    np.testing.assert_allclose(shaped[:, 0], circular[:, 0], rtol=0.0, atol=1.0e-9)
    # The section still shapes the mouth.
    assert np.ptp(shaped[:, -1]) > 10.0


def test_throat_extension_and_slot_stay_round():
    params = {**_OSSE, "throatExtLength": 10.0, "throatExtAngle": 5.0, "slotLength": 8.0}
    circular = _radii(params)
    shaped = _radii({**params, **_section(4.0, 1.3)})
    # Uniform map: the 18 mm prefix of 168 mm ends before ring 3.
    np.testing.assert_allclose(shaped[:, :3], circular[:, :3], rtol=0.0, atol=1.0e-9)
    assert np.ptp(shaped[:, 3]) > 0.0
    assert np.ptp(shaped[:, -1]) > 10.0


def test_section_scales_only_growth_past_the_throat():
    circular = _radii(_OSSE)
    shaped = _radii({**_OSSE, **_section(2.0, 1.5)})
    # phi = 0 carries the full aspect ratio, phi = 90 deg none.
    r_t = circular[0, 0]
    np.testing.assert_allclose(shaped[0], r_t + (circular[0] - r_t) * 1.5, rtol=1.0e-12)
    np.testing.assert_allclose(shaped[16], circular[16], rtol=1.0e-12)
