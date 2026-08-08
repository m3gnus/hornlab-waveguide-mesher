"""The scalar profile formulas are the differential oracle for the array ones.

``calculate_rosse`` and ``calculate_osse`` evaluate one point; the grid
builders call ``calculate_rosse_curve`` and ``calculate_osse_curve``, which
evaluate a whole meridian with one parameter resolution.  The array forms are
the ones a preview build actually runs, so every case here re-derives the
expected values from the scalar functions rather than from stored numbers.

Bound: the two paths agree bit for bit except where a squaring is involved.
NumPy squares an array by multiplication, which is the correctly rounded
square; CPython's ``x ** 2`` goes through the platform ``pow``, which is
occasionally one ulp wide of it.  That one ulp is the *only* difference, but
it does not stay one ulp of the answer: ``x`` near the R-OSSE curve's inner
extremum is a difference of two nearly equal square roots, so a value that
cancels down to ~1e-12 mm carries the rounding of terms hundreds of
millimetres wide.  The bound is therefore stated against the profile's own
scale rather than per value.  The measured worst case over these fixtures and
2,000 randomised parameter sets is 13 eps; ``_TOLERANCE_EPS`` leaves five
times that, and is still ~1e-12 mm on a 400 mm horn, so any algorithmic
divergence blows straight past it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hornlab_mesher.profile_common import _osse_radius, _osse_radius_curve
from hornlab_mesher.profile_formulas import (
    _circular_arc_radius,
    _circular_arc_radius_curve,
    calculate_osse,
    calculate_osse_curve,
    calculate_rosse,
    calculate_rosse_curve,
)
from hornlab_mesher.profile_morph import (
    _morph_factor,
    _morph_factors,
    _rounded_rect_radii,
    _rounded_rect_radius,
)

_TOLERANCE_EPS = 64.0
_EPS = float(np.finfo(np.float64).eps)


def _assert_agrees(actual: np.ndarray, expected: list[float], label: str) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(expected, dtype=np.float64)
    assert actual.shape == reference.shape
    finite = np.isfinite(reference)
    assert np.array_equal(np.isfinite(actual), finite), f"{label}: finiteness differs"
    a = actual[finite]
    b = reference[finite]
    scale = max(1.0, float(np.abs(b).max(initial=0.0)))
    deviation = np.abs(a - b)
    worst = float(deviation.max(initial=0.0)) / (_EPS * scale)
    assert worst <= _TOLERANCE_EPS, (
        f"{label}: array path is {worst:.1f} eps of the {scale:.3f} mm profile "
        f"scale from the scalar oracle (bound {_TOLERANCE_EPS:.0f} eps)"
    )


_ROSSE_SEED = {
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
}

_ROSSE_CASES = {
    "seed": {},
    "throat_extension": {"throatExtLength": 12.0, "throatExtAngle": 3.0},
    "slot": {"slotLength": 8.0},
    "extension_and_slot": {
        "throatExtLength": 12.0,
        "throatExtAngle": 3.0,
        "slotLength": 8.0,
    },
    "unit_k": {"k": 1.0},
    "steep": {"a": 70.0, "a0": 4.0, "q": 0.5, "m": 0.2, "r": 1.4, "b": 1.0},
    "azimuthal_expressions": {
        "a": "25 + 10*cos(p)^2",
        "m": "0.85 - 0.1*sin(p)",
        "R": "140 + 20*cos(2*p)",
    },
}

_OSSE_SEED = {
    "type": "OSSE",
    "L": 130.0,
    "a": 45.0,
    "a0": 10.0,
    "r0": 12.7,
    "k": 7.0,
    "s": 0.85,
    "n": 4.0,
    "q": 0.991,
    "throatProfile": 1.0,
}

_OSSE_CASES = {
    "seed": {},
    "throat_extension": {"throatExtLength": 15.0, "throatExtAngle": 4.0},
    "slot": {"slotLength": 6.0},
    "extension_and_slot": {
        "throatExtLength": 15.0,
        "throatExtAngle": 4.0,
        "slotLength": 6.0,
    },
    "rotated": {"rot": 7.5},
    "saturating_superellipse": {"q": 4.0, "n": 2.0, "s": 1.0},
    "no_superellipse_term": {"n": 0.0},
    "circular_arc": {"throatProfile": 3.0, "circArcRadius": 400.0},
    "circular_arc_tangent": {"throatProfile": 3.0, "circArcTermAngle": 12.0},
    "azimuthal_expressions": {
        "a": "45 + 12*cos(p)^2",
        "L": "130 + 15*sin(p)^2",
        "s": "0.85 - 0.2*cos(p)",
    },
}

_AZIMUTHS = (0.0, 0.37, math.pi / 4.0, math.pi / 2.0, 2.1, math.pi)


@pytest.mark.parametrize("case", sorted(_ROSSE_CASES))
@pytest.mark.parametrize("phi", _AZIMUTHS)
def test_rosse_curve_matches_the_scalar_oracle(case: str, phi: float) -> None:
    params = {**_ROSSE_SEED, **_ROSSE_CASES[case]}
    t_values = np.linspace(0.0, 1.0, 401)
    z, radius = calculate_rosse_curve(t_values, phi, params)
    scalar = [calculate_rosse(float(t), phi, params) for t in t_values]
    _assert_agrees(z, [point[0] for point in scalar], f"rosse/{case}/z")
    _assert_agrees(radius, [point[1] for point in scalar], f"rosse/{case}/r")


@pytest.mark.parametrize("case", sorted(_OSSE_CASES))
@pytest.mark.parametrize("phi", _AZIMUTHS)
def test_osse_curve_matches_the_scalar_oracle(case: str, phi: float) -> None:
    params = {**_OSSE_SEED, **_OSSE_CASES[case]}
    total = 130.0 + 15.0 + 6.0
    z_values = np.linspace(0.0, total, 401)
    z, radius = calculate_osse_curve(z_values, phi, params)
    scalar = [calculate_osse(float(value), phi, params) for value in z_values]
    _assert_agrees(z, [point[0] for point in scalar], f"osse/{case}/z")
    _assert_agrees(radius, [point[1] for point in scalar], f"osse/{case}/r")


def test_rosse_curve_covers_randomised_parameter_sets() -> None:
    rng = np.random.default_rng(20260808)
    t_values = np.linspace(0.0, 1.0, 97)
    for _ in range(60):
        params = {
            "type": "R-OSSE",
            "R": float(rng.uniform(60.0, 400.0)),
            "r0": float(rng.uniform(6.0, 30.0)),
            "a0": float(rng.uniform(1.0, 30.0)),
            "a": float(rng.uniform(20.0, 75.0)),
            "k": float(rng.uniform(0.5, 4.0)),
            "m": float(rng.uniform(0.05, 0.95)),
            "b": float(rng.uniform(0.0, 1.5)),
            "r": float(rng.uniform(0.05, 2.0)),
            "q": float(rng.uniform(0.3, 6.0)),
            "throatExtLength": float(rng.choice([0.0, 0.0, 9.0])),
            "throatExtAngle": float(rng.uniform(0.0, 5.0)),
            "slotLength": float(rng.choice([0.0, 0.0, 5.0])),
        }
        phi = float(rng.uniform(0.0, math.tau))
        z, radius = calculate_rosse_curve(t_values, phi, params)
        scalar = [calculate_rosse(float(t), phi, params) for t in t_values]
        _assert_agrees(z, [point[0] for point in scalar], "rosse/random/z")
        _assert_agrees(radius, [point[1] for point in scalar], "rosse/random/r")


def test_osse_radius_curve_matches_the_scalar_oracle() -> None:
    params = {"L": 130.0, "k": 7.0, "n": 4.0, "q": 0.991, "s": 0.85}
    # Spans below, at, and past the superellipse term's saturation point.
    z_values = np.concatenate(
        [np.linspace(-5.0, 0.0, 11), np.linspace(0.0, 300.0, 401)]
    )
    actual = _osse_radius_curve(z_values, 0.0, params, r0=12.7, a_deg=45.0, a0_deg=10.0)
    expected = [
        _osse_radius(float(z), 0.0, params, r0=12.7, a_deg=45.0, a0_deg=10.0)
        for z in z_values
    ]
    _assert_agrees(actual, expected, "osse_radius")


def test_circular_arc_radius_curve_matches_the_scalar_oracle() -> None:
    params = {"circArcRadius": 400.0, "circArcTermAngle": 1.0}
    # Runs past both ends of the arc so the off-arc fallback is exercised.
    z_values = np.linspace(-200.0, 600.0, 401)
    actual = _circular_arc_radius_curve(
        z_values, 0.0, params, r0_main=12.7, mouth_radius=140.0, length=130.0
    )
    expected = [
        _circular_arc_radius(
            float(z), 0.0, params, r0_main=12.7, mouth_radius=140.0, length=130.0
        )
        for z in z_values
    ]
    _assert_agrees(actual, expected, "circular_arc")


_MORPH_CASES = {
    "rounded_rect": {
        "morphTarget": 1.0,
        "morphWidth": 300.0,
        "morphHeight": 200.0,
        "morphCorner": 40.0,
        "morphRate": 3.0,
        "morphFixed": 0.0,
    },
    "square_rate": {"morphTarget": 1.0, "morphWidth": 300.0, "morphRate": 2.0},
    "root_rate": {"morphTarget": 1.0, "morphWidth": 300.0, "morphRate": 0.5},
    "late_start": {"morphTarget": 1.0, "morphWidth": 300.0, "morphFixed": 0.6},
    "superellipse": {"morphTarget": 2.0, "morphWidth": 300.0, "morphHeight": 200.0},
    "inactive": {"morphTarget": 0.0, "morphWidth": 300.0},
    "azimuthal_rate": {
        "morphTarget": 1.0,
        "morphWidth": 300.0,
        "morphRate": "3 + 2*cos(p)^2",
    },
}


@pytest.mark.parametrize("case", sorted(_MORPH_CASES))
@pytest.mark.parametrize("phi", _AZIMUTHS)
@pytest.mark.parametrize("morph_start", [None, 0.0, 0.35])
def test_morph_factors_match_the_scalar_oracle(
    case: str, phi: float, morph_start: float | None
) -> None:
    params = _MORPH_CASES[case]
    t_values = np.linspace(0.0, 1.0, 401)
    actual = _morph_factors(t_values, phi, params, morph_start=morph_start)
    expected = [
        _morph_factor(float(t), phi, params, morph_start=morph_start)
        for t in t_values
    ]
    # ``** 2`` and ``** 0.5`` are the two exponents NumPy reroutes away from
    # the platform pow, so those two rates are exactly where a factor can land
    # one ulp off; the bound covers them.
    _assert_agrees(actual, expected, f"morph/{case}")


def test_morph_factors_leave_a_dormant_rate_expression_unevaluated() -> None:
    """The array path keeps the scalar path's lazy pre-morph behavior."""

    params = {
        "morphTarget": 1.0,
        "morphWidth": 300.0,
        "morphRate": "unsupported_function(p)",
    }
    t_values = np.linspace(0.0, 1.0, 17)

    actual = _morph_factors(t_values, 0.0, params, morph_start=1.0)
    expected = [
        _morph_factor(float(t), 0.0, params, morph_start=1.0)
        for t in t_values
    ]

    assert np.array_equal(actual, expected)


def test_rosse_curve_rejects_an_impossible_throat_extension() -> None:
    params = {**_ROSSE_SEED, "throatExtLength": 400.0, "throatExtAngle": 45.0}
    with pytest.raises(ValueError):
        calculate_rosse(0.5, 0.0, params)
    with pytest.raises(ValueError):
        calculate_rosse_curve(np.linspace(0.0, 1.0, 9), 0.0, params)


def test_rosse_curve_handles_a_degenerate_zero_length_design() -> None:
    params = {**_ROSSE_SEED, "R": 12.7, "throatExtLength": 0.0, "slotLength": 0.0}
    t_values = np.linspace(0.0, 1.0, 17)
    z, radius = calculate_rosse_curve(t_values, 0.0, params)
    scalar = [calculate_rosse(float(t), 0.0, params) for t in t_values]
    _assert_agrees(z, [point[0] for point in scalar], "degenerate/z")
    _assert_agrees(radius, [point[1] for point in scalar], "degenerate/r")


# --- FREEFORM cross-sections ------------------------------------------------
#
# ``_rounded_rect_radius`` answers one azimuth; a FREEFORM ring needs the whole
# row, and the semi-axes and corner radius are fixed for the ring.  Unlike the
# OSSE/R-OSSE curves above there is no squaring on this path -- every branch is
# a quotient, a product or one ``sqrt`` -- so the bound here is exact equality
# rather than a tolerance.

_ROUNDED_RECT_GEOMETRIES = {
    # a sharp rectangle: the corner-arc branch is unreachable
    "sharp": (130.0, 80.0, 0.0),
    "typical": (130.0, 80.0, 12.0),
    # r == min(a, b): the flat-side branches are unreachable, all arc
    "stadium": (100.0, 100.0, 100.0),
    "near_sharp": (100.0, 100.0, 1.0e-12),
    "tall_sliver": (5.0, 200.0, 4.9),
    "wide_sliver": (200.0, 5.0, 5.0),
    # a negative radius must clamp to zero exactly as the scalar clamps it
    "negative_corner": (80.0, 80.0, -3.0),
    "tiny": (1.0e-6, 1.0e-6, 1.0e-7),
}


def _rounded_rect_angles() -> np.ndarray:
    """Several turns, both cardinals, and either side of a degenerate axis."""

    rng = np.random.default_rng(20260808)
    return np.concatenate(
        (
            np.linspace(-4.0 * math.pi, 4.0 * math.pi, 2001),
            # the two branches that return before dividing
            np.array(
                [
                    0.0,
                    math.pi / 2.0,
                    math.pi,
                    3.0 * math.pi / 2.0,
                    -math.pi / 2.0,
                    math.tau,
                ]
            ),
            # just inside and just outside the 1e-9 degeneracy cutoff
            np.array(
                [
                    1.0e-12,
                    -1.0e-12,
                    1.0e-10,
                    1.0e-8,
                    math.pi / 2.0 - 1.0e-12,
                    math.pi / 2.0 + 1.0e-12,
                    math.pi / 2.0 - 1.0e-8,
                ]
            ),
            rng.uniform(-10.0, 10.0, 2000),
        )
    )


@pytest.mark.parametrize("case", sorted(_ROUNDED_RECT_GEOMETRIES))
def test_rounded_rect_radii_match_the_scalar_oracle(case: str) -> None:
    half_width, half_height, corner_radius = _ROUNDED_RECT_GEOMETRIES[case]
    angles = _rounded_rect_angles()
    actual = _rounded_rect_radii(angles, half_width, half_height, corner_radius)
    expected = np.array(
        [
            _rounded_rect_radius(float(angle), half_width, half_height, corner_radius)
            for angle in angles
        ]
    )
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected), (
        f"{case}: {int(np.count_nonzero(actual != expected))} of {angles.size} "
        f"azimuths differ from the scalar oracle, worst "
        f"{float(np.abs(actual - expected).max()):.3e} mm"
    )


def test_rounded_rect_radii_keep_the_input_shape() -> None:
    """A FREEFORM ring grid arrives 2-D; the scalar path flattened and rebuilt."""

    angles = np.linspace(0.0, math.tau, 24, endpoint=False).reshape(4, 6)
    actual = _rounded_rect_radii(angles, 130.0, 80.0, 12.0)
    assert actual.shape == angles.shape
    expected = np.array(
        [_rounded_rect_radius(float(a), 130.0, 80.0, 12.0) for a in angles.reshape(-1)]
    ).reshape(angles.shape)
    assert np.array_equal(actual, expected)


def test_rounded_rect_radii_do_not_warn_on_a_degenerate_axis() -> None:
    """The clamped denominators exist so a cardinal azimuth cannot overflow."""

    angles = np.array([0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0])
    with np.errstate(all="raise"):
        actual = _rounded_rect_radii(angles, 130.0, 80.0, 12.0)
    assert np.array_equal(actual, np.array([130.0, 80.0, 130.0, 80.0]))
