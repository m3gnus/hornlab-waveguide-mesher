"""Acoustic control-grid fit over a rounded-rectangle morph corner.

ATH pins the corner arc at three intervals per quadrant, so its chord is a
constant ``2*R*sin(15 deg)`` regardless of the angular budget. The acoustic fit
demands every azimuth chord stay within twice the local mm target, so before
this was fixed any ``mouthResolution < R*sin(15 deg)`` refined to the segment
cap and failed with "requested mm resolution needs more than 2048 internal
geometry samples" -- a message that blamed the resolution instead of the corner.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hornlab_mesher.config_builder import (
    _build_acoustic_sampling_grid,
    _corner_arc_edge_mask,
    _mesh_density_from_config,
    _reshape_grid,
    build_from_config,
    build_geometry_params,
)
from hornlab_mesher.config_parser import ConfigError
from hornlab_mesher.profile_morph import (
    _rounded_rect_quadrant_angles,
    rounded_rect_corner_arc_span,
)
from hornlab_mesher.profile_sampling import ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY
from hornlab_mesher.profiles import build_point_grid

# The ATH corner chord that no angular refinement could ever shrink.
ATH_CORNER_CHORD_FACTOR = 2.0 * math.sin(math.pi / 12.0)


def _config(**mesh_overrides) -> dict:
    morph = mesh_overrides.pop("morph", None)
    config = {
        "formula": "OSSE",
        "mode": "freestanding",
        "profile": {
            "formula": "OSSE",
            "r0": 12.7,
            "a": "45 - 16*cos(1*p)^2 - 40*sin(p*1)^16",
            "a0": 15.5,
            "k": 2.0,
            "q": 0.993,
            "L": 310.0,
            "n": 5.0,
            "s": 0.8,
            "h": 0.0,
        },
        "cross_section": {"exponent": 2.0, "aspectRatio": 1.0},
        "mesh": {
            "quadrants": 1234,
            "angularSegments": 80,
            "lengthSegments": 20,
            "cornerSegments": 4,
            "throatResolution": 6.0,
            "mouthResolution": 15.0,
            "rearResolution": 40.0,
        },
        "source": {"sourceShape": 0, "sourceRadius": -1.0, "sourceCurv": 0},
    }
    config["mesh"].update(mesh_overrides)
    if morph is not None:
        config["morph"] = morph
    return config


def _rounded_rect_morph(corner_mm: float) -> dict:
    return {
        "morphTarget": 1,
        "morphWidth": 0,
        "morphHeight": 0,
        "morphCorner": corner_mm,
        "morphRate": 3.0,
    }


def _acoustic_grid(config: dict):
    params, _, _ = build_geometry_params(config)
    density = _mesh_density_from_config(config, allow_large_mesh=True)
    return _build_acoustic_sampling_grid(params, density, topology_mode="acoustic"), density


def _angular_ratios(grid, density) -> np.ndarray:
    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    points = _reshape_grid(grid["inner_points"], n_phi, n_length, "inner_points")
    ring_h = density.throat_res_mm + (
        density.mouth_res_mm - density.throat_res_mm
    ) * np.linspace(0.0, 1.0, n_length + 1)
    delta = np.diff(points, axis=0)
    if bool(grid.get("full_circle", True)):
        delta = np.concatenate((delta, points[:1] - points[-1:]), axis=0)
    return np.linalg.norm(delta, axis=2) / (2.0 * ring_h[None, :])


# --- the public/ATH sampler must not move -----------------------------------


@pytest.mark.parametrize(
    "points_per_quadrant,half_width,half_height,corner,corner_segments",
    [
        (26, 208.0, 180.0, 20.0, 4),  # ordinary rounded rectangle
        (26, 200.0, 100.0, 0.0, 4),  # sharp rectangle (uniform azimuth)
        (26, 200.0, 100.0, 100.0, 4),  # one collapsed side (stadium)
        (10, 150.0, 150.0, 150.0, 1),  # fully round target
        (1, 100.0, 60.0, 60.0, 1),  # degenerate single-interval quadrant
    ],
)
def test_default_subdivision_reproduces_the_ath_sampler_exactly(
    points_per_quadrant, half_width, half_height, corner, corner_segments
):
    baseline = _rounded_rect_quadrant_angles(
        points_per_quadrant, half_width, half_height, corner, corner_segments
    )
    explicit = _rounded_rect_quadrant_angles(
        points_per_quadrant,
        half_width,
        half_height,
        corner,
        corner_segments,
        arc_subdivision=1,
    )
    assert np.array_equal(baseline, explicit)


@pytest.mark.parametrize("subdivision", [2, 3, 5, 8, 17])
def test_subdivision_keeps_the_four_canonical_ath_corner_profiles(subdivision):
    """3*k equal intervals put ATH's tangency + 30/60 deg samples back exactly."""

    baseline = _rounded_rect_quadrant_angles(26, 208.0, 180.0, 20.0, 4)
    refined = _rounded_rect_quadrant_angles(
        26, 208.0, 180.0, 20.0, 4, arc_subdivision=subdivision
    )
    for angle in baseline:
        assert np.any(refined == angle), f"canonical angle {angle!r} lost"
    assert len(refined) == len(baseline) + 3 * (subdivision - 1)
    assert np.all(np.diff(refined) > 0.0)


def test_public_point_grid_ignores_the_acoustic_override_by_default():
    """The viewport/ATH path calls build_point_grid directly; it must not move."""

    params, _, _ = build_geometry_params(_config(morph=_rounded_rect_morph(20.0)))
    assert ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY not in params
    grid = build_point_grid(params)
    # 80 angular + 4 corner segments -> 21 points per quadrant -> 84 profiles.
    assert int(grid["grid_n_phi"]) == 84


# --- the fit now converges over the corner ----------------------------------


@pytest.mark.parametrize("corner_mm", [20.0, 40.0, 60.0])
def test_corner_below_the_ath_chord_floor_now_fits(corner_mm):
    """Just under the old hard floor: previously a ConfigError, now a valid fit."""

    mouth_res = round(corner_mm * math.sin(math.pi / 12.0) - 0.1, 3)
    assert corner_mm * ATH_CORNER_CHORD_FACTOR > 2.0 * mouth_res  # the old failure
    config = _config(mouthResolution=mouth_res, morph=_rounded_rect_morph(corner_mm))
    (grid, _), density = _acoustic_grid(config)

    ratios = _angular_ratios(grid, density)
    assert float(ratios.max()) <= 1.0

    corner_edges = _corner_arc_edge_mask(grid)
    assert corner_edges is not None and bool(corner_edges.any())
    assert float(ratios[corner_edges].max()) <= 1.0


def test_corner_fit_survives_the_full_wg_corner_radius_range_at_default_resolution():
    """WG's Corner Radius slider reaches 100 mm; its default mouth res is 15 mm."""

    config = _config(mouthResolution=15.0, morph=_rounded_rect_morph(100.0))
    (grid, _), density = _acoustic_grid(config)
    assert float(_angular_ratios(grid, density).max()) <= 1.0


@pytest.mark.parametrize("quadrants", [1234, 1, 12, 14])
def test_corner_fit_holds_on_symmetry_reduced_domains(quadrants):
    config = _config(
        quadrants=quadrants, mouthResolution=5.0, morph=_rounded_rect_morph(20.0)
    )
    (grid, _), density = _acoustic_grid(config)
    assert float(_angular_ratios(grid, density).max()) <= 1.0


@pytest.mark.parametrize(
    "half_width,half_height,corner",
    [
        (200.0, 100.0, 0.0),  # sharp rectangle
        (150.0, 150.0, 150.0),  # fully round target: both walls collapse
    ],
)
def test_targets_without_a_fixed_arc_report_no_span(half_width, half_height, corner):
    """Uniform-azimuth targets have no fixed arc, so nothing needs classifying."""

    assert rounded_rect_corner_arc_span(26, half_width, half_height, corner) is None


def test_sharp_rectangle_fit_is_unclassified_and_converges():
    config = _config(mouthResolution=5.0, morph=_rounded_rect_morph(0.0))
    (grid, _), density = _acoustic_grid(config)
    assert _corner_arc_edge_mask(grid) is None
    assert float(_angular_ratios(grid, density).max()) <= 1.0


def test_expression_valued_morph_params_report_no_corner_span():
    """Morph parameters are evaluated at every azimuth.

    A single first-quadrant span cannot describe an expression-valued corner, and
    a wrong span would route refinement to the wrong channel -- so the fit must
    fall back to treating every interval as ordinary.
    """

    config = _config(
        morph={
            "morphTarget": 1,
            "morphWidth": 0,
            "morphHeight": 0,
            "morphCorner": "20 + 10*cos(2*p)",
            "morphRate": 3.0,
        }
    )
    params, _, _ = build_geometry_params(config)
    grid = build_point_grid({**params, "samplingMode": "ath-default-zmap"})
    assert grid["morph_corner_arc_span"] is None
    assert _corner_arc_edge_mask(grid) is None


@pytest.mark.parametrize("seed", [(16, 8), (100, 20), (400, 160), (1200, 600)])
def test_fitted_grid_is_essentially_independent_of_the_requested_segments(seed):
    """mm targets decide the control net; segments are only the probe density.

    Chord error falls as 1/n but sagitta error falls as 1/n^2, so extrapolating a
    sagitta violation linearly overshot badly from coarse seeds (a 16x8 request
    used to fit a 2048x144 grid while 200x60 fit 328x151), and oversized requests
    were kept verbatim.
    """

    angular, length = seed
    config = _config(angularSegments=angular, lengthSegments=length)
    (grid, meta), _ = _acoustic_grid(config)
    assert 200 <= int(grid["grid_n_phi"]) <= 320
    assert 120 <= int(grid["grid_n_length"]) <= 200
    assert meta["geometrySamplePhiProfiles"] == int(grid["grid_n_phi"])
    assert meta["geometrySampleControlPoints"] == int(grid["grid_n_phi"]) * (
        int(grid["grid_n_length"]) + 1
    )


def test_metadata_reports_the_effective_grid_not_just_the_nominal_segments():
    config = _config(mouthResolution=5.0, morph=_rounded_rect_morph(20.0))
    (grid, meta), _ = _acoustic_grid(config)
    assert meta["geometrySampleCornerArcSubdivision"] >= 2
    assert meta["geometrySampleAxialRings"] == int(grid["grid_n_length"]) + 1
    # The nominal segment count does not describe the real azimuth profile count.
    assert meta["geometrySamplePhiProfiles"] != meta["geometrySampleAngularSegments"]


def test_stadium_target_keeps_the_fixed_arc_and_still_fits():
    """One collapsed wall leaves ATH's fixed arc in place, so it needs the fix too."""

    config = _config(
        mouthResolution=5.0,
        morph={
            "morphTarget": 1,
            "morphWidth": 400.0,
            "morphHeight": 240.0,
            "morphCorner": 120.0,
            "morphRate": 3.0,
        },
    )
    (grid, _), density = _acoustic_grid(config)
    corner_edges = _corner_arc_edge_mask(grid)
    assert corner_edges is not None and bool(corner_edges.any())
    assert float(_angular_ratios(grid, density).max()) <= 1.0


def test_corner_mask_excludes_the_wall_interval_at_the_tangency_point():
    config = _config(mouthResolution=5.0, morph=_rounded_rect_morph(20.0))
    (grid, _), _ = _acoustic_grid(config)
    mask = _corner_arc_edge_mask(grid)
    angles = np.asarray(grid["angle_list"], dtype=np.float64)
    theta1, theta2 = grid["morph_corner_arc_span"]
    folded = np.arctan2(np.abs(np.sin(angles)), np.abs(np.cos(angles)))
    # Every flagged interval lies wholly inside the arc span, and at least one
    # wall interval touching a tangency point is left unflagged.
    for index in np.flatnonzero(mask):
        for end in (index, (index + 1) % len(angles)):
            assert theta1 - 1.0e-9 <= folded[end] <= theta2 + 1.0e-9
    assert not bool(mask.all())


def test_end_to_end_mesh_builds_for_a_previously_rejected_morph(tmp_path):
    config = _config(mouthResolution=5.0, morph=_rounded_rect_morph(20.0))
    result = build_from_config(config, tmp_path / "morph-corner.msh", allow_large_mesh=True)
    assert result.mesh_path.exists()


# --- a genuinely unfittable geometry still fails, and says why ---------------


def test_unfittable_geometry_is_stopped_by_the_effective_grid_cap():
    """The cap now bounds the grid that is actually produced, and names it."""

    config = _config(throatResolution=0.02, mouthResolution=0.02)
    params, _, _ = build_geometry_params(config)
    density = _mesh_density_from_config(config, allow_large_mesh=True)
    with pytest.raises(ConfigError) as excinfo:
        _build_acoustic_sampling_grid(params, density, topology_mode="acoustic")
    message = str(excinfo.value)
    assert "0.02 mm" in message
    assert "control points exceeds the limit" in message


def test_effective_caps_bound_the_grid_that_is_actually_built():
    """Capping the two inputs never bounded the real control net.

    ``points_per_quadrant = ceil((angularSegments + cornerSegments)/4)`` is
    mirrored over four quadrants, so the old per-input 2048 cap still permitted
    4096 azimuth profiles.
    """

    config = _config(throatResolution=0.05, mouthResolution=0.05)
    params, _, _ = build_geometry_params(config)
    density = _mesh_density_from_config(config, allow_large_mesh=True)
    with pytest.raises(ConfigError) as excinfo:
        _build_acoustic_sampling_grid(params, density, topology_mode="acoustic")
    message = str(excinfo.value)
    assert "too large" in message
    assert any(
        token in message
        for token in ("azimuth profiles", "axial rings", "control points")
    )


def test_sagitta_stays_off_for_morph_because_a_sharp_target_has_a_real_vertex():
    """Regression guard for the exclusion at the top of the fit.

    A sharp rectangle (WG's default Corner Radius of 0) has a genuine 90 degree
    vertex whose three-point sagitta decays as 1/n, not 1/n^2. Enforcing the
    smooth-curvature limit there stalls against the segment cap rather than
    converging, so morphed targets stay chord-constrained.
    """

    config = _config(mouthResolution=5.0, morph=_rounded_rect_morph(0.0))
    (grid, _), density = _acoustic_grid(config)
    assert float(_angular_ratios(grid, density).max()) <= 1.0

    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    points = _reshape_grid(grid["inner_points"], n_phi, n_length, "inner_points")
    ring_h = density.throat_res_mm + (
        density.mouth_res_mm - density.throat_res_mm
    ) * np.linspace(0.0, 1.0, n_length + 1)
    padded = np.concatenate((points[-1:], points, points[:1]), axis=0)
    previous, current, following = padded[:-2], padded[1:-1], padded[2:]
    chord = following - previous
    chord_sq = np.sum(chord * chord, axis=2)
    alpha = np.divide(
        np.sum((current - previous) * chord, axis=2),
        chord_sq,
        out=np.zeros_like(chord_sq),
        where=chord_sq > 1.0e-18,
    )
    sagitta = np.linalg.norm(current - (previous + alpha[:, :, None] * chord), axis=2)
    # The vertex genuinely violates the smooth-curvature limit; the fit accepts
    # it rather than chasing an unreachable target.
    assert float(np.max(sagitta / (0.05 * ring_h[None, :]))) > 1.0
