"""Contract tests for the additive ``hornlab.preview/1`` API."""

from __future__ import annotations

import time

import numpy as np
import pytest

from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry


OSSE_FREESTANDING = {
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

ROSSE_ENCLOSURE = {
    "formula": "R-OSSE",
    "mode": "enclosure",
    "profile": {
        "R_mm": 150.0,
        "r0_mm": 12.7,
        "a0_deg": 15.5,
        "a_deg": 55.0,
        "k": 1.0,
        "q": 1.0,
        "m": 0.85,
        "r": 0.35,
        "b": 0.4,
        "tmax": 1.0,
    },
    "enclosure": {
        "depth_mm": 150.0,
        "edge_mm": 18.0,
        "edge_type": 1,
    },
}

ICW_FLAT_BAFFLE = {
    "formula": "ICW",
    "mode": "freestanding",
    "profile": {
        "r0_mm": 12.7,
        "a0_deg": 18.0,
        "termination": "flat_baffle",
        "L_mm": 120.0,
        "R_mm": 110.0,
    },
    "mesh": {"wall_thickness_mm": 6.0},
    # Exercise the other source-cap contract as well as ICW termination.
    "source": {"source_shape": 0},
}

FREEFORM_FREESTANDING = {
    "formula": "FREEFORM",
    "mode": "freestanding",
    "profile": {
        "profileH": {
            "points": [[0.0, 12.7], [60.0, 80.0], [120.0, 160.0]],
            "throatAngleDeg": 15.5,
            "mouthAngleDeg": 70.0,
        },
        "profileV": {
            "points": [[0.0, 12.7], [60.0, 60.0], [120.0, 110.0]],
            "throatAngleDeg": 15.5,
            "mouthAngleDeg": 60.0,
        },
        "crossSections": [
            {"t": 0.0, "shape": "circle"},
            {
                "t": 0.4,
                "shape": "rounded_rectangle",
                "cornerRadiusMm": 5.9,
            },
            {
                "t": 1.0,
                "shape": "rounded_rectangle",
                "cornerRadiusMm": 5.9,
            },
        ],
    },
    "mesh": {"wall_thickness_mm": 6.0},
}


FAMILIES = [
    pytest.param(
        OSSE_FREESTANDING,
        {"horn.inner", "horn.outer", "mouth_rim", "source_cap", "wall.rear_cap"},
        id="osse-freestanding",
    ),
    pytest.param(
        ROSSE_ENCLOSURE,
        {
            "horn.inner",
            "mouth_rim",
            "source_cap",
            "enclosure.front",
            "enclosure.roundover",
            "enclosure.side",
            "enclosure.rear",
        },
        id="rosse-enclosure-roundovers",
    ),
    pytest.param(
        ICW_FLAT_BAFFLE,
        {"horn.inner", "horn.outer", "mouth_rim", "source_cap", "wall.rear_cap"},
        id="icw-flat-baffle",
    ),
    pytest.param(
        FREEFORM_FREESTANDING,
        {"horn.inner", "horn.outer", "mouth_rim", "source_cap", "wall.rear_cap"},
        id="freeform",
    ),
]


@pytest.mark.parametrize("config,expected_roles", FAMILIES)
def test_fine_preview_contract_for_all_required_families(config, expected_roles):
    preview = build_preview_geometry(config)
    by_role = {surface.role: surface for surface in preview.surfaces}

    assert set(by_role) == expected_roles
    assert preview.metadata["api_version"] == "hornlab.preview/1"
    assert preview.metadata["units"] == "mm"
    assert preview.metadata["actual_segment_counts"]["horn_phi"] >= 96
    assert preview.metadata["actual_segment_counts"]["horn_axial"] >= 48
    assert set(preview.metadata["timings_ms"]) == {
        "canonical_sampling",
        "surface_assembly_and_fidelity",
        "total",
    }

    for surface in preview.surfaces:
        assert surface.positions.dtype == np.float64
        assert surface.normals.dtype == np.float64
        assert surface.indices.dtype == np.uint32
        assert surface.positions.shape == surface.normals.shape
        assert surface.positions.ndim == 2 and surface.positions.shape[1] == 3
        assert np.all(np.isfinite(surface.positions))
        assert np.all(np.isfinite(surface.normals))
        assert np.allclose(np.linalg.norm(surface.normals, axis=1), 1.0, atol=1.0e-3)
        assert surface.indices.ndim == 1 and surface.indices.size % 3 == 0
        assert int(surface.indices.max()) < len(surface.positions)

        # Closed-phi surfaces use modulo indices. There is no duplicated wrap row
        # (hard-boundary duplication happens between separate role arrays only).
        if surface.closed_phi:
            assert len(np.unique(surface.positions, axis=0)) == len(surface.positions)

        if surface.shading == "flat":
            assert surface.normal_method == "exact-planar"
            normal = surface.normals[0]
            assert np.array_equal(surface.normals, np.broadcast_to(normal, surface.normals.shape))
            distances = (surface.positions - surface.positions[0]) @ normal
            assert np.max(np.abs(distances)) < 1.0e-10
        else:
            assert surface.normal_method == "analytic-parametric"

    for role, achieved in preview.metadata["fidelity"].items():
        assert role in by_role
        assert 0.0 < achieved["max_chord_error_mm"] < 5.0
        assert 0.0 <= achieved["max_normal_step_deg"] <= 180.0
        assert achieved["reference_density_multiplier"] == 4
        assert achieved["max_chord_error_mm_achieved"] <= achieved[
            "max_chord_error_mm_requested"
        ]
        assert achieved["max_normal_step_deg_achieved"] <= achieved[
            "max_normal_step_deg_requested"
        ]
        assert achieved["vertex_cap_limited"] is False


def test_rounded_source_cap_is_an_analytic_sphere_not_a_cone_fan():
    preview = build_preview_geometry(OSSE_FREESTANDING)
    cap = next(surface for surface in preview.surfaces if surface.role == "source_cap")
    radius = preview.metadata["source_cap_radius_mm"]
    sphere_center = cap.positions[0] - cap.normals[0] * radius
    radial_deviation = np.abs(np.linalg.norm(cap.positions - sphere_center, axis=1) - radius)

    assert cap.shading == "smooth"
    assert len(cap.positions) > preview.metadata["actual_segment_counts"]["horn_phi"] + 1
    assert float(radial_deviation.max()) < preview.metadata["fidelity"]["source_cap"][
        "max_chord_error_mm"
    ]
    expected_radial = (cap.positions - sphere_center) / radius
    assert np.allclose(cap.normals, expected_radial, atol=1.0e-12)


def test_enclosure_roundover_meets_existing_and_fine_interval_floors():
    preview = build_preview_geometry(ROSSE_ENCLOSURE)
    roundover = next(
        surface for surface in preview.surfaces if surface.role == "enclosure.roundover"
    )

    # Canonical viewport currently has two intervals per quarter. Fine preview
    # deliberately raises that stage-1 floor to twelve per the review contract.
    assert preview.metadata["actual_segment_counts"]["enclosure_roundover_quarter"] >= 12
    assert len(roundover.indices) // 3 >= 2 * 12 * 32 * 2


def test_flat_icw_source_cap_has_exact_plane_and_normal():
    preview = build_preview_geometry(ICW_FLAT_BAFFLE)
    cap = next(surface for surface in preview.surfaces if surface.role == "source_cap")

    assert cap.shading == "flat"
    assert cap.normal_method == "exact-planar"
    assert np.array_equal(cap.normals, np.tile([0.0, 0.0, 1.0], (len(cap.positions), 1)))
    assert np.ptp(cap.positions[:, 2]) < 1.0e-12


def test_identical_calls_have_byte_identical_arrays():
    first = build_preview_geometry(OSSE_FREESTANDING)
    second = build_preview_geometry(OSSE_FREESTANDING)

    assert [surface.role for surface in first.surfaces] == [
        surface.role for surface in second.surfaces
    ]
    for left, right in zip(first.surfaces, second.surfaces, strict=True):
        assert left.positions.tobytes() == right.positions.tobytes()
        assert left.normals.tobytes() == right.normals.tobytes()
        assert left.indices.tobytes() == right.indices.tobytes()


def test_stage_two_tolerance_options_are_honored_and_reported():
    preview = build_preview_geometry(
        OSSE_FREESTANDING,
        PreviewOptionsV1(max_chord_error_mm=0.05, max_normal_step_deg=3.0),
    )

    assert preview.metadata["warnings"] == []
    requested = preview.metadata["requested_fidelity"]
    assert requested["max_chord_error_mm"] == 0.05
    assert requested["max_normal_step_deg"] == 3.0
    for achieved in preview.metadata["fidelity"].values():
        assert achieved["max_chord_error_mm_achieved"] <= 0.05
        assert achieved["max_normal_step_deg_achieved"] <= 3.0


def test_surface_include_options_do_not_recompute_or_leak_omitted_roles():
    preview = build_preview_geometry(
        ROSSE_ENCLOSURE,
        PreviewOptionsV1(
            lod="coarse",
            include_enclosure=False,
            include_source_cap=False,
            include_rear_cap=False,
        ),
    )

    assert [surface.role for surface in preview.surfaces] == ["horn.inner"]


def test_coarse_floor_roundover_floor_and_target_fidelity():
    preview = build_preview_geometry(
        ROSSE_ENCLOSURE, PreviewOptionsV1(lod="coarse")
    )

    assert preview.metadata["actual_segment_counts"]["horn_phi"] >= 64
    assert preview.metadata["actual_segment_counts"]["horn_axial"] >= 12
    assert (
        preview.metadata["actual_segment_counts"]["enclosure_roundover_quarter"]
        >= 6
    )
    for achieved in preview.metadata["fidelity"].values():
        assert not achieved["vertex_cap_limited"]
        assert achieved["max_chord_error_mm_achieved"] <= achieved[
            "max_chord_error_mm_requested"
        ]
        assert achieved["max_normal_step_deg_achieved"] <= achieved[
            "max_normal_step_deg_requested"
        ]


def test_freeform_corner_arcs_are_dense_and_flat_rows_remain_sparser():
    preview = build_preview_geometry(FREEFORM_FREESTANDING)
    sampling = preview.metadata["angular_sampling"]

    assert sampling["strategy"] == "stable-union-corner-grid"
    assert sampling["corner_arc_rows"] >= 4 * 12
    assert 0 < sampling["flat_side_rows"] < sampling["corner_arc_rows"]
    assert preview.metadata["fidelity"]["horn.inner"][
        "max_normal_step_deg_achieved"
    ] <= 3.0


def test_default_osse_coarse_vertices_are_an_exact_subset_of_fine():
    coarse = build_preview_geometry(
        OSSE_FREESTANDING, PreviewOptionsV1(lod="coarse")
    )
    fine = build_preview_geometry(OSSE_FREESTANDING, PreviewOptionsV1(lod="fine"))
    coarse_inner = next(s for s in coarse.surfaces if s.role == "horn.inner")
    fine_inner = next(s for s in fine.surfaces if s.role == "horn.inner")

    fine_vertices = {tuple(position) for position in fine_inner.positions}
    assert all(tuple(position) in fine_vertices for position in coarse_inner.positions)


def test_tiny_per_body_cap_degrades_to_valid_cap_limited_surfaces():
    cap = 200
    preview = build_preview_geometry(
        ROSSE_ENCLOSURE,
        PreviewOptionsV1(lod="fine", max_vertices=cap),
    )

    assert preview.surfaces
    for surface in preview.surfaces:
        assert len(surface.positions) <= cap
        assert surface.indices.size % 3 == 0
        assert int(surface.indices.max()) < len(surface.positions)
        accounting = preview.metadata["vertex_accounting"][surface.role]
        assert accounting["vertices"] == len(surface.positions)
        assert accounting["vertex_cap_limited"] is True


def test_freeform_semantic_station_availability_is_explicit():
    preview = build_preview_geometry(FREEFORM_FREESTANDING)
    semantic = preview.metadata["semantic_stations"]

    assert "FREEFORM cross-section stations" in semantic["inserted_first"]
    assert "FREEFORM H/V anchors" in semantic["inserted_first"]
    assert "corner tangencies/cardinals" in semantic["inserted_first"]
    assert isinstance(semantic["unavailable_additively"], list)


def test_fine_osse_machine_local_performance_guard():
    # Warm imports/allocator state; this is deliberately a generous local guard
    # against accidental density explosions, not a micro-benchmark.
    build_preview_geometry(OSSE_FREESTANDING, PreviewOptionsV1(lod="fine"))
    started = time.perf_counter()
    build_preview_geometry(OSSE_FREESTANDING, PreviewOptionsV1(lod="fine"))
    elapsed = time.perf_counter() - started

    assert elapsed < 0.150, f"fine OSSE preview took {elapsed * 1000.0:.1f} ms"
