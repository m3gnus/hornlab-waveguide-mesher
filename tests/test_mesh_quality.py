"""Element shape and chord deviation must be measured, and must be reported.

Nothing in the pipeline gated either. A sliver with finite area and a facet that
chords a whole roundover both passed every check, because the only shape test
anywhere was ``areas <= 1e-15``.

The thresholds these tests pin come from a measured population, not a
preference; ``docs/mesh-quality.md`` carries the distributions and, more
importantly, the null result that constrains what this gate may claim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hornlab_mesher.geometry import MeshDensity, OsseHornGeometry
from hornlab_mesher.mesher import MesherError, build_mesh_with_info
from hornlab_mesher.quality import (
    FAIL_CHORD_DEVIATION_MM,
    FAIL_P1_ANGLE_DEG,
    SLIVER_ANGLE_DEG,
    WARN_P1_ANGLE_DEG,
    chord_deviation_report,
    element_shape_report,
    evaluate_quality_gate,
    mesh_quality_report,
)


def _regular_strip(count: int, aspect: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """A strip of ``count`` well-shaped quads, split into triangles."""

    xs = np.arange(count + 1, dtype=float)
    points = np.array(
        [[x, y * aspect, 0.0] for y in (0.0, 1.0) for x in xs], dtype=float
    )
    top = count + 1
    triangles = []
    for index in range(count):
        triangles.append([index, index + 1, top + index])
        triangles.append([index + 1, top + index + 1, top + index])
    return points, np.asarray(triangles, dtype=np.int64)


def _with_slivers(count: int, slivers: int) -> tuple[np.ndarray, np.ndarray]:
    """The same strip, with ``slivers`` of its quads squashed almost flat."""

    points, triangles = _regular_strip(count)
    top = count + 1
    for index in range(slivers):
        points[top + index, 1] = 0.002
    return points, triangles


class TestElementShape:
    def test_a_finite_area_sliver_is_measured_where_area_alone_sees_nothing(self):
        """The defect this module exists for: area is finite, shape is not."""

        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0e-4, 0.0]], dtype=float
        )
        triangles = np.array([[0, 1, 2]], dtype=np.int64)
        # The old guard's only test passes: the area is far above 1e-15.
        area = 0.5 * abs(
            np.cross(points[1] - points[0], points[2] - points[0])[2]
        )
        assert area > 1.0e-15

        report = element_shape_report(points, triangles)
        assert report.triangle_count == 1
        assert report.min_angle_deg < 0.05
        assert report.min_radius_ratio < 1.0e-3
        assert report.sliver_count == 1

    def test_the_percentile_and_the_extreme_rank_meshes_differently(self):
        """The reason the gate is on a percentile: they disagree, and it matters.

        One bad triangle in a large clean mesh and a mesh that is a fifth
        slivers have the same worst triangle. Only the percentile separates
        them, and the measured population separates the same way -- the ATH
        archive's worst single triangle, at 2.05 degrees, belongs to a mesh that
        converges without trouble.
        """

        one_bad = element_shape_report(*_with_slivers(200, 1))
        many_bad = element_shape_report(*_with_slivers(200, 40))

        assert one_bad.min_angle_deg == pytest.approx(
            many_bad.min_angle_deg, rel=0.05
        )
        assert one_bad.p1_angle_deg > 40.0
        assert many_bad.p1_angle_deg < SLIVER_ANGLE_DEG
        assert many_bad.sliver_fraction > 10.0 * max(one_bad.sliver_fraction, 1e-9)

    def test_degenerate_faces_are_excluded_rather_than_dragging_the_statistics(self):
        """A zero-area face has no shape, and folding it in reports a lie.

        The collinear triple matters: a face with a repeated corner is already
        excluded by its zero side length, so it never reaches the area test and
        cannot show that the area test does anything.
        """

        points, triangles = _regular_strip(20)
        collinear = np.vstack(
            [points, [[10.0, 5.0, 0.0], [11.0, 5.0, 0.0], [12.0, 5.0, 0.0]]]
        )
        base = len(points)
        degenerate = np.vstack([triangles, [[base, base + 1, base + 2]]])
        report = element_shape_report(collinear, degenerate)
        assert report.excluded_count == 1
        assert report.triangle_count == len(triangles)
        assert report.min_angle_deg > 10.0

    def test_the_worst_elements_are_located_on_the_model(self):
        """A count without a place is not actionable; the gate quotes z and r."""

        points, triangles = _with_slivers(60, 5)
        points[:, 2] += 0.25
        report = element_shape_report(points, triangles, axis=2)
        assert report.worst
        assert report.worst[0].min_angle_deg == report.min_angle_deg
        assert report.worst[0].z == pytest.approx(0.25, abs=1e-9)
        assert report.worst[0].radius > 0.0


class TestChordDeviation:
    def test_a_flat_mesh_deviates_from_nothing(self):
        report = chord_deviation_report(*_regular_strip(20), vertex_units="mm")
        assert report.edge_count > 0
        assert report.max_deviation_mm == pytest.approx(0.0, abs=1e-9)

    def test_one_facet_across_a_roundover_is_measured_though_its_shape_is_perfect(self):
        """The second defect, and the reason one measure cannot carry both.

        Three equilateral triangles folded through a right angle: every element
        shape statistic is as good as a triangle gets, and the faceting still
        stands millimetres off the arc it is standing in for.
        """

        span = 26.0
        height = span * math.sqrt(3.0) / 2.0
        # Two equilateral facets meeting at 90 degrees about the shared edge.
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [span, 0.0, 0.0],
                [0.5 * span, -height, 0.0],
                [0.5 * span, 0.0, height],
            ],
            dtype=float,
        )
        triangles = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int64)

        shape = element_shape_report(points, triangles)
        assert shape.min_angle_deg > 55.0          # equilateral, by construction
        assert shape.min_radius_ratio > 0.95
        assert shape.sliver_count == 0

        chord = chord_deviation_report(points, triangles, vertex_units="mm")
        assert chord.max_turn_deg == pytest.approx(90.0, abs=1.0)
        assert chord.max_deviation_mm > FAIL_CHORD_DEVIATION_MM

    def test_deviation_falls_as_the_turn_is_resolved(self):
        """The remedy has to move the number, or the number is not measuring it."""

        def two_facets(turn_deg: float, span: float) -> tuple[np.ndarray, np.ndarray]:
            half = math.radians(turn_deg) / 2.0
            points = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.5, -span * math.cos(half), span * math.sin(half)],
                    [0.5, span * math.cos(half), span * math.sin(half)],
                ],
                dtype=float,
            )
            return points, np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int64)

        coarse = chord_deviation_report(*two_facets(90.0, 13.0), vertex_units="mm")
        finer = chord_deviation_report(*two_facets(45.0, 6.5), vertex_units="mm")
        # Halving the element halves the span and the turn together, and the
        # sagitta goes as their product, so the small-angle limit is 4x. At a
        # right angle tan(theta/4) is no longer linear and the exact factor is
        # 3.19; bracketing it pins the trend without pretending it is 4.
        ratio = coarse.max_deviation_mm / finer.max_deviation_mm
        assert 3.0 < ratio < 4.5

    def test_units_are_honoured_rather_than_assumed(self):
        points, triangles = _regular_strip(6)
        points[:, 2] = np.where(points[:, 1] > 0.5, 0.4, 0.0)
        in_metres = chord_deviation_report(points, triangles, vertex_units="m")
        in_millimetres = chord_deviation_report(points, triangles, vertex_units="mm")
        assert in_metres.max_deviation_mm == pytest.approx(
            1000.0 * in_millimetres.max_deviation_mm
        )

    def test_an_unknown_unit_is_refused_rather_than_guessed(self):
        with pytest.raises(ValueError, match="vertex_units"):
            chord_deviation_report(*_regular_strip(4), vertex_units="furlong")


class TestGate:
    def test_a_clean_mesh_raises_nothing(self):
        report = mesh_quality_report(*_regular_strip(200), vertex_units="mm")
        verdict = evaluate_quality_gate(report, strict=True)
        assert verdict.passed
        assert not verdict.warnings

    def test_a_sliver_rimmed_mesh_warns_and_says_where(self):
        report = mesh_quality_report(*_with_slivers(200, 40), vertex_units="mm")
        verdict = evaluate_quality_gate(report)
        assert verdict.passed          # never fatal outside strict mode
        assert any("1st-percentile" in message for message in verdict.warnings)
        assert any("z=" in message and "r=" in message for message in verdict.warnings)

    def test_strict_mode_fails_what_report_mode_only_warns_about(self):
        report = mesh_quality_report(*_with_slivers(200, 40), vertex_units="mm")
        assert evaluate_quality_gate(report, strict=False).passed
        strict = evaluate_quality_gate(report, strict=True)
        assert not strict.passed
        assert strict.failures

    def test_one_bad_triangle_does_not_fail_a_mesh_that_is_otherwise_clean(self):
        """The gate must read the percentile; on the extreme these two agree.

        Both meshes have the same worst triangle. A gate on the worst one
        rejects both, which is the ranking the measured population shows to be
        wrong -- the ATH archive's worst single triangle, at 2.05 degrees,
        belongs to a mesh that solves without trouble.
        """

        one_bad = mesh_quality_report(*_with_slivers(200, 1), vertex_units="mm")
        many_bad = mesh_quality_report(*_with_slivers(200, 40), vertex_units="mm")
        assert one_bad["element_shape"]["min_angle_deg"] == pytest.approx(
            many_bad["element_shape"]["min_angle_deg"], rel=0.05
        )

        assert evaluate_quality_gate(one_bad, strict=True).passed
        assert not evaluate_quality_gate(many_bad, strict=True).passed

    def test_the_warn_band_sits_above_the_gate(self):
        """The two tiers must be distinct, or the second one is decoration."""

        assert WARN_P1_ANGLE_DEG > FAIL_P1_ANGLE_DEG


class TestBuildIntegration:
    def _geometry(self) -> OsseHornGeometry:
        return OsseHornGeometry(L_mm=120.0, r0_mm=12.7, a_deg=60.0, a0_deg=15.5)

    def test_a_build_records_its_own_quality(self, tmp_path):
        _path, info = build_mesh_with_info(
            self._geometry(),
            MeshDensity(throat_res_mm=6.0, mouth_res_mm=12.0),
            tmp_path / "mesh.msh",
        )
        quality = info.metadata["quality"]
        assert quality["measured"]
        assert quality["element_shape"]["triangle_count"] == info.n_triangles
        assert quality["element_shape"]["p1_angle_deg"] > 0.0
        assert quality["chord_deviation"]["edge_count"] > 0

    def test_an_unknown_gate_mode_is_refused(self, tmp_path):
        with pytest.raises(MesherError, match="quality_gate"):
            build_mesh_with_info(
                self._geometry(),
                MeshDensity(throat_res_mm=6.0, mouth_res_mm=12.0),
                tmp_path / "mesh.msh",
                quality_gate="warn",
            )
