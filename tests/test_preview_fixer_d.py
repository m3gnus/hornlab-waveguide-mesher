"""Regressions for the accepted M1/M2 preview fidelity findings."""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry
from hornlab_mesher.preview.api import (
    _MAX_ARC_INTERVALS,
    _adaptive_lod_config,
    _intervals_for_arc,
    _validate_finite_metadata,
)
from hornlab_mesher.preview.fidelity import (
    _axis_interval_error,
    adaptive_grid_indices,
    analytic_grid_normals,
)
from tests.test_preview_api import OSSE_FREESTANDING, ROSSE_ENCLOSURE


def test_m1_zmap_points_uppercase_alias_is_preserved():
    points = [0.0, 0.02, 0.2, 1.0]
    config = {
        "formula": "OSSE",
        "mesh": {"sampling_mode": "uniform", "ZMapPoints": points},
    }

    adapted = _adaptive_lod_config(config, 32, 3, power=2.0)

    assert adapted["mesh"]["ZMapPoints"] == points
    assert "z_map_points" not in adapted["mesh"]


@pytest.mark.parametrize("cap", [6, 12])
def test_m1_too_small_vertex_caps_are_rejected_before_topology_crashes(cap):
    with pytest.raises(ValueError, match="max_vertices"):
        build_preview_geometry(
            ROSSE_ENCLOSURE,
            PreviewOptionsV1(lod="coarse", max_vertices=cap),
        )


def test_m1_vertex_cap_accounts_for_every_enclosure_surface():
    cap = 200
    preview = build_preview_geometry(
        ROSSE_ENCLOSURE,
        PreviewOptionsV1(lod="coarse", max_vertices=cap),
    )

    assert {surface.role for surface in preview.surfaces} >= {
        "enclosure.front",
        "enclosure.roundover",
        "enclosure.side",
        "enclosure.rear",
    }
    assert all(len(surface.positions) <= cap for surface in preview.surfaces)


@pytest.mark.parametrize(("plan_type", "plan_n"), [(2, 2.0), (3, 6.0)])
def test_m1_ellipse_and_superellipse_plans_adapt_and_measure_emitted_plan(
    plan_type, plan_n
):
    config = copy.deepcopy(ROSSE_ENCLOSURE)
    config["enclosure"].update({"plan_type": plan_type, "plan_n": plan_n})

    preview = build_preview_geometry(
        config,
        PreviewOptionsV1(
            lod="coarse", max_chord_error_mm=0.15, max_normal_step_deg=6.0
        ),
    )
    achieved = preview.metadata["fidelity"]["enclosure.side"]

    assert achieved["silhouette_segments_achieved"] > 28
    assert achieved["reference_density_multiplier"] == 8
    assert achieved["max_chord_error_mm_achieved"] <= 0.15
    assert achieved["max_normal_step_deg_achieved"] <= 6.0


@pytest.mark.parametrize(("plan_type", "plan_n"), [(2, 2.0), (3, 6.0)])
def test_unbuildable_enclosure_plans_are_previewed_with_a_warning(plan_type, plan_n):
    """The ellipse/superellipse plans still draw, but say they cannot be built.

    ``build_enclosure_box`` refuses both of them -- NotImplementedError in the
    closed domain, and the open-domain route takes plan_type=1 only. Without a
    warning the viewport is the only thing the user sees before the build.
    """

    config = copy.deepcopy(ROSSE_ENCLOSURE)
    config["enclosure"].update({"plan_type": plan_type, "plan_n": plan_n})

    preview = build_preview_geometry(config, PreviewOptionsV1(lod="coarse"))

    assert any(
        surface.role.startswith("enclosure.") for surface in preview.surfaces
    ), "the preview should still render the shape"
    warning = [
        text
        for text in preview.metadata["warnings"]
        if f"plan_type={plan_type}" in text
    ]
    assert len(warning) == 1, preview.metadata["warnings"]
    assert "not" in warning[0] and "buildable" in warning[0]
    assert "plan_type=1" in warning[0]


def test_buildable_enclosure_plan_is_previewed_without_the_warning():
    preview = build_preview_geometry(ROSSE_ENCLOSURE, PreviewOptionsV1(lod="coarse"))

    assert not [
        text for text in preview.metadata["warnings"] if "buildable" in text
    ]


def test_unbuildable_enclosure_plan_warns_even_when_not_rendered():
    """The toggle hides the surfaces; it does not make the config buildable."""

    config = copy.deepcopy(ROSSE_ENCLOSURE)
    config["enclosure"].update({"plan_type": 2, "plan_n": 2.0})

    preview = build_preview_geometry(
        config, PreviewOptionsV1(lod="coarse", include_enclosure=False)
    )

    assert not [
        surface for surface in preview.surfaces if surface.role.startswith("enclosure.")
    ]
    assert any("plan_type=2" in text for text in preview.metadata["warnings"])


def test_amended_m1_4_analytic_parametric_definition_is_published():
    preview = build_preview_geometry(
        OSSE_FREESTANDING,
        PreviewOptionsV1(
            lod="coarse",
            include_outer=False,
            include_source_cap=False,
            include_rear_cap=False,
        ),
    )

    inner = preview.surfaces[0]
    note = preview.metadata["normal_method_notes"]["analytic-parametric"]
    assert inner.normal_method == "analytic-parametric"
    assert "true analytic/canonical surface" in note
    assert "never mesh-derived" in note


def test_m2_nonuniform_true_coordinates_drive_chords_and_open_grid_normals():
    t = np.asarray([0.0, 0.1, 0.45, 1.0])
    base_phi = np.asarray([-1.0, -0.7, -0.1, 0.25, 1.0])
    phi_drift = np.asarray([0.0, 0.3, 0.2, 0.4, 0.0])
    phi = base_phi[None, :] + t[:, None] * phi_drift[None, :]
    points = np.empty((len(t), phi.shape[1], 3), dtype=np.float64)
    points[:, :, 0] = phi
    points[:, :, 1] = t[:, None]
    points[:, :, 2] = t[:, None] * phi

    normals = analytic_grid_normals(
        points,
        closed_phi=False,
        t_coordinates=t,
        phi_coordinates=phi,
    )
    expected = np.stack(
        (-t[:, None] * np.ones_like(phi), -phi, np.ones_like(phi)), axis=2
    )
    expected /= np.linalg.norm(expected, axis=2, keepdims=True)
    assert np.allclose(normals, expected, atol=1.0e-12)

    linear = np.zeros((3, 3, 3), dtype=np.float64)
    linear[:, :, 2] = np.asarray([0.0, 0.1, 1.0])[:, None]
    constant_normals = np.broadcast_to([1.0, 0.0, 0.0], linear.shape)
    chord, _normal, _split, measured = _axis_interval_error(
        linear,
        constant_normals,
        0,
        2,
        axis=0,
        coordinates=np.asarray([0.0, 0.1, 1.0]),
    )
    assert measured is True
    assert chord < 1.0e-14


def test_m1_chamfer_side_duplicates_vertices_and_keeps_distinct_face_normals():
    config = copy.deepcopy(ROSSE_ENCLOSURE)
    config["enclosure"]["edge_type"] = 2
    preview = build_preview_geometry(config, PreviewOptionsV1(lod="coarse"))
    side = next(surface for surface in preview.surfaces if surface.role == "enclosure.side")

    assert side.shading == "flat"
    assert side.normal_method == "exact-planar"
    groups: dict[tuple[float, float, float], list[np.ndarray]] = {}
    for position, normal in zip(side.positions, side.normals, strict=True):
        groups.setdefault(tuple(position), []).append(normal)
    seam_normals = [
        values
        for values in groups.values()
        if len(values) > 1 and len(np.unique(np.asarray(values), axis=0)) > 1
    ]
    assert seam_normals


def test_m1_flat_auto_cap_metadata_is_strict_json_safe_and_guarded():
    config = copy.deepcopy(OSSE_FREESTANDING)
    config["profile"]["a0_deg"] = 0.0
    preview = build_preview_geometry(config, PreviewOptionsV1(lod="coarse"))

    assert preview.metadata["source_cap_radius_mm"] is None
    json.dumps(preview.metadata, allow_nan=False)
    with pytest.raises(ValueError, match="must be finite or null"):
        _validate_finite_metadata({"nested": {"bad": math.inf}})


def test_m1_achieved_silhouette_is_per_surface_and_reports_cap_limiting():
    requested = 128
    preview = build_preview_geometry(
        ROSSE_ENCLOSURE,
        PreviewOptionsV1(
            lod="fine", min_silhouette_segments=requested, max_vertices=200
        ),
    )

    for role, fidelity in preview.metadata["fidelity"].items():
        assert isinstance(fidelity["silhouette_segments_achieved"], int), role
        if fidelity["silhouette_segments_achieved"] < requested:
            assert fidelity["vertex_cap_limited"] is True, role


def test_m2_interval_without_reference_interior_is_explicitly_unmeasured():
    points = np.zeros((2, 3, 3), dtype=np.float64)
    points[:, :, 0] = np.arange(3, dtype=np.float64)
    points[1, :, 1] = 1.0
    normals = np.broadcast_to([0.0, 0.0, 1.0], points.shape)

    _t, _phi, achieved = adaptive_grid_indices(
        points,
        normals,
        [0, 1],
        [0, 1, 2],
        max_chord_error_mm=0.1,
        max_normal_step_deg=10.0,
        max_vertices=None,
        closed_phi=False,
    )

    assert achieved["max_chord_error_mm"] is None
    assert achieved["measurement_complete"] is False
    assert achieved["unmeasured_intervals"] == 3
    assert achieved["vertex_cap_limited"] is True


def test_m1_pathological_requests_are_preflight_clamped():
    assert (
        _intervals_for_arc(200.0, 90.0, 1.0e-300, 1.0e-300, 1)
        == _MAX_ARC_INTERVALS
    )
    preview = build_preview_geometry(
        OSSE_FREESTANDING,
        PreviewOptionsV1(
            lod="coarse",
            min_silhouette_segments=100_000,
            max_vertices=200,
            include_outer=False,
            include_source_cap=False,
            include_rear_cap=False,
        ),
    )
    assert any("canonical azimuth reference clamped" in item for item in preview.metadata["warnings"])
    assert len(preview.surfaces[0].positions) <= 200
