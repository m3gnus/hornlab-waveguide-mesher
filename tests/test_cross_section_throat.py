"""A non-circular cross-section must leave the throat circular.

The throat is where the compression driver bolts on, so the cross-section
exponent and aspect ratio may only reshape the horn past it.
"""

from __future__ import annotations

import math

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


_SLOTTED_OSSE = {
    "type": "OSSE", "L": 100.0, "r0": 12.7, "a0": 10.0, "a": 45.0, "k": 1.0, "s": 0.0,
    "slotLength": 12.0, "angularSegments": 16, "quadrants": "1234",
    "wallThickness": 0.0, "encDepth": 0.0,
}


@pytest.mark.parametrize("length_segments", [4, 16, 64, 112])
def test_slotted_osse_mouth_does_not_depend_on_surface_sampling(length_segments):
    """The mouth of a slotted, non-round OSSE horn is fixed by the design alone.

    Independent derivation. A native slot is a cylinder of radius ``r0``, so the
    reference radius where the slot ends is ``r_t = r0`` exactly. With
    ``k = 1`` and ``s = 0`` the OS-SE body radius at the mouth (main-section
    ``z = L``) is ``R = sqrt(r0^2 + 2 r0 L tan(a0) + L^2 tan(a)^2)``
    = sqrt(12.7^2 + 2*12.7*100*tan 10deg + 100^2*tan^2 45deg) = 103.000... mm.
    At phi = 0 an aspect ratio of 2 gives the superellipse factor ``s(0) = 2``,
    so the mouth half-width is ``r_t + (R - r_t) * 2`` for every sampling. The
    old code interpolated ``r_t`` between the last slot station and the first
    flare station, so it moved with Surface sampling (189.27 mm at 4 segments,
    192.94 mm at 16, 193.30 mm at 112).
    """
    r0, L = 12.7, 100.0
    R = math.sqrt(r0**2 + 2.0 * r0 * L * math.tan(math.radians(10.0)) + L**2)
    expected = r0 + (R - r0) * 2.0
    params = {**_SLOTTED_OSSE, **_section(2.0, 2.0), "lengthSegments": length_segments}
    grid = build_point_grid_arrays(params)["inner_grid"]
    assert grid[0, -1, 0] == pytest.approx(expected, abs=1.0e-9)


@pytest.mark.parametrize("base", [_OSSE, _ROSSE], ids=["OSSE", "R-OSSE"])
@pytest.mark.parametrize("length_segments", [5, 13, 40])
def test_prefix_reference_is_the_slot_radius_at_any_sampling(base, length_segments):
    """Every flare station scales its growth from the exact slot-end radius.

    Independent derivation. The native extension tapers back from ``r0`` to the
    driver and the native slot is a cylinder of radius ``r0`` (geometry
    contract, "Throat extension"), so the radius where the slot ends is
    ``r_t = r0``. The circular build's radii are unaffected by the section, so
    a station past the prefix must satisfy
    ``r'(t) = r0 + (r(t) - r0) * s(phi)`` whatever the axial station count.
    """
    params = {
        **base, "throatExtLength": 10.0, "throatExtAngle": 5.0, "slotLength": 8.0,
        "lengthSegments": length_segments,
    }
    circular = _radii(params)
    shaped = _radii({**params, **_section(2.0, 1.5)})
    r0 = params["r0"]
    # phi = 0: s(phi) = aspect ratio. Compare every station past the prefix.
    past_prefix = circular[0] > r0 + 1.0e-6
    np.testing.assert_allclose(
        shaped[0, past_prefix], r0 + (circular[0, past_prefix] - r0) * 1.5,
        rtol=0.0, atol=1.0e-9,
    )


_MOUTH_CASES = {
    "OSSE extension only": {**_OSSE, "throatExtLength": 10.0, "throatExtAngle": 5.0},
    "OSSE slot only": {**_OSSE, "slotLength": 12.0},
    "OSSE h bulge": {**_OSSE, "throatExtLength": 10.0, "throatExtAngle": 5.0,
                     "slotLength": 8.0, "h": 3.0},
    "R-OSSE extension only": {**_ROSSE, "throatExtLength": 10.0, "throatExtAngle": 5.0},
    "R-OSSE slot only": {**_ROSSE, "slotLength": 12.0},
    "R-OSSE tmax 0.8": {**_ROSSE, "throatExtLength": 10.0, "throatExtAngle": 5.0,
                        "slotLength": 8.0, "tmax": 0.8},
}


@pytest.mark.parametrize("name", list(_MOUTH_CASES))
def test_non_round_mouth_is_the_same_at_every_surface_sampling(name):
    """The mouth ring of a non-round horn with a throat prefix is sampling-free.

    The mouth station (t = 1) is always sampled, and its radius before the
    section is applied does not depend on the other stations. ``r_t`` is a
    property of the design (the meridian at the prefix end), so the scaled
    mouth ``r_t + (r_mouth - r_t) * s(phi)`` must be identical, to rounding,
    for any number of axial stations: extension only, slot only, an OSSE
    ``h`` bulge (which is nonzero at the prefix end) and a truncated R-OSSE
    (``tmax`` < 1, where the prefix fraction is of the truncated length).
    """
    base = {**_MOUTH_CASES[name], **_section(3.0, 1.5)}
    mouths = [
        build_point_grid_arrays({**base, "lengthSegments": n})["inner_grid"][:, -1]
        for n in (5, 13, 40)
    ]
    for mouth in mouths[1:]:
        np.testing.assert_allclose(mouth, mouths[0], rtol=0.0, atol=1.0e-9)


@pytest.mark.parametrize("length_segments", [5, 13, 40])
def test_h_bulge_counts_in_the_prefix_reference(length_segments):
    """OSSE's ``h`` bulge is part of the radius at the prefix end.

    Independent derivation. The sampler adds ``h sin(pi t)`` over the whole
    composite length (contract, OSSE ``h``), and the native extension and slot
    end at radius ``r0``, so at the slot end ``t_p = (ext + slot) / total`` the
    reference is ``r_t = r0 + h sin(pi t_p)`` = 25 + 3 sin(pi * 18 / 168).
    Every station past the prefix then satisfies
    ``r'(t) = r_t + (r(t) - r_t) * s(phi)`` at phi = 0 with s = 1.5.
    """
    params = {**_MOUTH_CASES["OSSE h bulge"], "lengthSegments": length_segments}
    grid = build_point_grid_arrays(params)["inner_grid"]
    circular = np.hypot(grid[..., 0], grid[..., 1])
    shaped = _radii({**params, **_section(2.0, 1.5)})
    r_t = 25.0 + 3.0 * math.sin(math.pi * 18.0 / 168.0)
    past_prefix = grid[0, :, 2] > 18.0 + 1.0e-9
    np.testing.assert_allclose(
        shaped[0, past_prefix], r_t + (circular[0, past_prefix] - r_t) * 1.5,
        rtol=0.0, atol=1.0e-9,
    )
