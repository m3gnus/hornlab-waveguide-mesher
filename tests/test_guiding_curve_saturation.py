"""The guiding-curve coverage solver must say so when it cannot reach the curve.

The inversion bisects a fixed ``[0.5, 89]`` degree bracket. Outside that
bracket it converges onto a bracket end and returns geometry whose mouth is not
on the guiding curve at all -- and because the coverage angle is then pinned,
every other parameter appears to stop affecting the mouth. R-OSSE already
raises for the analogous unreachable-mouth case (``_rosse_length``); OSSE used
to clamp in silence.
"""

from __future__ import annotations

import math

import pytest

import hornlab_mesher.preview.api as api
from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry
from hornlab_mesher.preview.api import _guiding_curve_warnings
from hornlab_mesher.profile_common import _osse_radius
from hornlab_mesher.profile_morph import (
    _COVERAGE_ANGLE_MAX,
    _COVERAGE_ANGLE_MIN,
    _invert_osse_coverage_angle,
)
from hornlab_mesher.profile_formulas import (
    calculate_osse,
    osse_coverage_angle,
    osse_coverage_inversion,
    osse_coverage_saturation,
    osse_coverage_saturation_probe,
)

#: A preview-API config (not a raw params dict) for the metadata contract.
OSSE_GUIDING_CURVE_CONFIG = {
    "formula": "OSSE",
    "mode": "freestanding",
    "profile": {
        "L_mm": 120.0,
        "r0_mm": 12.7,
        "a0_deg": 15.5,
        "a_deg": 55.0,
        "k": 1.0,
        "q": 0.995,
    },
    "mesh": {"wall_thickness_mm": 6.0},
}


def _params(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "type": "OSSE",
        "L": 400.0,
        "a": 45.0,
        "a0": 10.0,
        "r0": 25.4,
        "k": 7.0,
        "s": 0.85,
        "n": 4.0,
        "q": 0.991,
        "gcurveType": 1,
        "gcurveWidth": 1000.0,
        "gcurveAspectRatio": 1.0,
        # Exponent 2 is a true circle; the default of 3 is a squircle whose
        # diagonals sit ~12% outside the nominal width.
        "gcurveSeN": 2.0,
        "gcurveDist": 1.0,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("length", [60.0, 150.0, 400.0, 600.0])
def test_reachable_guiding_curve_is_met_and_not_flagged(length: float) -> None:
    params = _params(L=length)
    assert osse_coverage_saturation(params, 0.0) is None
    # 24 bisection steps over the 88.5 degree bracket leave a sub-micron residual.
    assert calculate_osse(length, 0.0, params)[1] == pytest.approx(500.0, abs=1e-4)


def test_reachable_solve_is_unchanged_by_the_bracket_probe() -> None:
    # The probe must not perturb the solved angle: these are the values the
    # plain bisection produced before the guard existed.
    for length, expected in ((150.0, 74.56), (400.0, 44.39), (600.0, 22.41)):
        angle = osse_coverage_angle(_params(L=length), 0.0)
        assert angle == pytest.approx(expected, abs=0.01)


def test_length_past_the_reachable_range_is_reported_not_silently_clamped() -> None:
    params = _params(L=900.0)
    solved = osse_coverage_inversion(params, 0.0)
    assert solved is not None
    assert solved.saturated == "min"
    assert solved.angle_deg == pytest.approx(0.5)
    # The mouth really does miss the requested 500 mm radius.
    assert solved.target_radius == pytest.approx(500.0)
    assert solved.achieved_radius > 560.0
    assert calculate_osse(900.0, 0.0, params)[1] == pytest.approx(
        solved.achieved_radius, abs=1e-6
    )

    reason = osse_coverage_saturation(params, 0.0)
    assert reason is not None
    assert "unreachable" in reason
    assert "shorten the horn" in reason


def test_mouth_too_wide_for_any_coverage_angle_reports_the_other_direction() -> None:
    params = _params(L=10.0, gcurveWidth=3000.0)
    solved = osse_coverage_inversion(params, 0.0)
    assert solved is not None
    assert solved.saturated == "max"
    assert solved.angle_deg == pytest.approx(89.0)

    reason = osse_coverage_saturation(params, 0.0)
    assert reason is not None
    assert "lengthen the horn" in reason


def test_no_guiding_curve_never_reports_saturation() -> None:
    assert osse_coverage_inversion(_params(gcurveType=0), 0.0) is None
    assert osse_coverage_saturation(_params(gcurveType=0), 0.0) is None
    assert osse_coverage_saturation(_params(gcurveWidth=0.0), 0.0) is None


def test_saturation_is_detected_per_azimuth() -> None:
    # An elliptical guide whose narrow axis is reachable at this length and
    # whose wide axis is not. A phi=0-only probe would miss half of these.
    params = _params(L=700.0, gcurveWidth="760 + 700*sin(p)^2")
    assert osse_coverage_saturation(params, 0.0) is not None
    assert osse_coverage_saturation(params, math.pi / 2.0) is None


class TestPreviewWarnings:
    def test_reachable_curve_emits_no_warning(self) -> None:
        assert _guiding_curve_warnings(_params(L=400.0), "OSSE") == []

    def test_unreachable_curve_emits_exactly_one_warning(self) -> None:
        warnings = _guiding_curve_warnings(_params(L=900.0), "OSSE")
        assert len(warnings) == 1
        # A rotationally symmetric miss must not name one arbitrary azimuth.
        assert "every probed azimuth" in warnings[0]

    def test_partial_miss_names_the_worst_azimuth(self) -> None:
        warnings = _guiding_curve_warnings(
            _params(L=700.0, gcurveWidth="760 + 700*sin(p)^2"), "OSSE"
        )
        assert len(warnings) == 1
        assert "phi=0.0 deg" in warnings[0]

    def test_non_osse_formulas_are_skipped(self) -> None:
        # R-OSSE and FREEFORM reject guiding curves in config validation; the
        # preview must not second-guess that with its own message.
        assert _guiding_curve_warnings(_params(L=900.0), "R-OSSE") == []

    def test_no_guiding_curve_emits_no_warning(self) -> None:
        assert _guiding_curve_warnings(_params(L=900.0, gcurveType=0), "OSSE") == []


class TestProbeResolution:
    """The azimuth screen has to be fine enough for its silence to mean anything.

    The screen used to walk 15 degree steps because every azimuth cost a full
    coverage inversion. That is coarse enough to miss real geometry, and the
    docstring said so with a worked example. Screening with the bracket probe
    instead of the inversion is ~7x cheaper per azimuth, which pays for a
    1 degree step.
    """

    #: The example the old 15 degree comment named as undetectable: saturated
    #: at every odd multiple of 7.5 degrees, reachable at every multiple of 15.
    SPIKY = "1000 - 900*sin(12*p)^2"

    def test_the_example_the_old_step_admitted_it_missed_is_now_caught(self) -> None:
        warnings = _guiding_curve_warnings(_params(gcurveWidth=self.SPIKY), "OSSE")
        assert len(warnings) == 1

    def test_a_15_degree_step_really_would_have_missed_it(self) -> None:
        # Guards the premise: without this the test above could pass for
        # reasons unrelated to the step size.
        coarse = tuple(math.radians(deg) for deg in range(0, 360, 15))
        assert all(
            osse_coverage_saturation_probe(_params(gcurveWidth=self.SPIKY), phi) is None
            for phi in coarse
        )

    def test_step_divides_the_circle_and_does_not_probe_360_twice(self) -> None:
        step = api._GUIDING_CURVE_PROBE_STEP_DEG
        azimuths = api._GUIDING_CURVE_PROBE_AZIMUTHS
        assert len(azimuths) == round(360.0 / step)
        assert azimuths[0] == 0.0
        assert math.degrees(azimuths[-1]) == pytest.approx(360.0 - step)

    def test_preview_publishes_the_resolution_it_actually_screened_at(self) -> None:
        # The absence of a warning cannot be a guarantee at any fixed step, so
        # the step is reported rather than implied.
        preview = build_preview_geometry(
            OSSE_GUIDING_CURVE_CONFIG, PreviewOptionsV1(lod="coarse")
        )
        probe = preview.metadata["guiding_curve_probe"]
        assert probe["step_deg"] == api._GUIDING_CURVE_PROBE_STEP_DEG
        assert probe["azimuths"] == len(api._GUIDING_CURVE_PROBE_AZIMUTHS)
        assert probe["best_effort"] is True


class TestSaturationProbe:
    """The cheap screen must agree exactly with the full inversion."""

    def test_probe_matches_the_inversion_on_saturated_geometry(self) -> None:
        for params in (
            _params(L=900.0),
            _params(L=10.0, gcurveWidth=3000.0),
            _params(L=400.0, gcurveWidth=100.0, gcurveDist=0.5),
        ):
            assert osse_coverage_saturation_probe(params, 0.0) == (
                osse_coverage_inversion(params, 0.0)
            )

    def test_probe_reports_nothing_where_the_curve_is_reachable(self) -> None:
        # None means "nothing to report", not "solved": the probe never
        # bisects, so it has no angle to hand back.
        for length in (60.0, 150.0, 400.0, 600.0):
            params = _params(L=length)
            assert osse_coverage_inversion(params, 0.0) is not None
            assert osse_coverage_saturation_probe(params, 0.0) is None

    def test_probe_reports_nothing_without_a_guiding_curve(self) -> None:
        assert osse_coverage_saturation_probe(_params(gcurveType=0), 0.0) is None
        assert osse_coverage_saturation_probe(_params(gcurveWidth=0.0), 0.0) is None

    def test_probe_does_not_raise_where_the_radius_is_undefined(self) -> None:
        # The negative-a0 radicand case: the probe visits bracket ends the
        # bisection never reaches, so it must swallow the raise like the
        # inversion does, and claim nothing off the back of a NaN.
        params = _params(L=100.0, a0=-15.0, k=1.0, s=0.0, q=0.99, gcurveWidth=1000.0)
        assert osse_coverage_saturation_probe(params, 0.0) is None

    def test_probe_agrees_with_the_inversion_across_a_partial_sweep(self) -> None:
        params = _params(L=700.0, gcurveWidth="760 + 700*sin(p)^2")
        sides = set()
        for index in range(0, 360, 3):
            phi = math.radians(index)
            probed = osse_coverage_saturation_probe(params, phi)
            solved = osse_coverage_inversion(params, phi)
            assert solved is not None
            if solved.saturated is None:
                assert probed is None
            else:
                assert probed == solved
                sides.add(solved.saturated)
        # The fixture has to actually exercise both verdicts to be worth much.
        assert sides == {"min"}


class TestBracketEdges:
    """Cases from the adversarial review of the bracket probe."""

    @pytest.mark.parametrize("angle", [_COVERAGE_ANGLE_MIN, _COVERAGE_ANGLE_MAX])
    def test_target_exactly_on_a_bracket_end_is_reachable_there(
        self, angle: float
    ) -> None:
        # A target equal to r(0.5) is met AT 0.5 degrees, and likewise at 89;
        # only a target strictly outside the bracket is unreachable. Inclusive
        # comparisons would warn about geometry that is exactly right. Driven
        # through the inversion directly because no guiding-curve width lands
        # on a bracket end to the last bit once it has been through the
        # superellipse.
        params = _params(L=400.0)
        exact = _osse_radius(400.0, 0.0, params, r0=25.4, a_deg=angle, a0_deg=10.0)
        solved = _invert_osse_coverage_angle(
            exact, 400.0, 0.0, params, a0_deg=10.0, r0_main=25.4
        )
        assert solved.saturated is None
        assert solved.angle_deg == pytest.approx(angle, abs=1e-4)

    @pytest.mark.parametrize("nudge", [-1e-6, 1e-6])
    def test_a_target_just_outside_the_bracket_is_still_flagged(self, nudge: float) -> None:
        # The strict comparison must not blunt the guard: a hair outside the
        # bracket is still unreachable.
        params = _params(L=400.0)
        angle = _COVERAGE_ANGLE_MIN if nudge < 0 else _COVERAGE_ANGLE_MAX
        exact = _osse_radius(400.0, 0.0, params, r0=25.4, a_deg=angle, a0_deg=10.0)
        solved = _invert_osse_coverage_angle(
            exact * (1.0 + nudge), 400.0, 0.0, params, a0_deg=10.0, r0_main=25.4
        )
        assert solved.saturated == ("min" if nudge < 0 else "max")

    def test_undefined_radius_in_the_bracket_does_not_raise_or_misdiagnose(self) -> None:
        # A negative a0 makes the OSSE radicand negative at small coverage
        # angles, where math.sqrt raises rather than returning a non-finite
        # value. The probe visits angles the plain bisection never reached, so
        # it must not turn that into a crash -- and must not claim saturation
        # off the back of a NaN.
        params = _params(L=100.0, a0=-15.0, k=1.0, s=0.0, q=0.99, gcurveWidth=1000.0)
        solved = osse_coverage_inversion(params, 0.0)
        assert solved is not None
        assert solved.saturated is None
        assert osse_coverage_saturation(params, 0.0) is None
        # It still finds the real solution further up the bracket.
        assert calculate_osse(100.0, 0.0, params)[1] == pytest.approx(500.0, abs=1e-3)

    def test_message_does_not_call_a_mid_horn_radius_the_mouth_radius(self) -> None:
        # gcurveDist < 1 solves the inversion partway along the horn. The v1
        # schema still defaults it to 0.5, so this is the common case.
        params = _params(L=400.0, gcurveWidth=100.0, gcurveDist=0.5)
        reason = osse_coverage_saturation(params, 0.0)
        assert reason is not None
        assert "guiding-curve distance (z=200.0 mm)" in reason
        assert "mouth radius" not in reason
        # The real mouth is nowhere near the quoted number.
        assert calculate_osse(400.0, 0.0, params)[1] > 250.0

    def test_message_says_mouth_radius_when_the_curve_is_at_the_mouth(self) -> None:
        reason = osse_coverage_saturation(_params(L=900.0), 0.0)
        assert reason is not None
        assert "the mouth radius is" in reason
