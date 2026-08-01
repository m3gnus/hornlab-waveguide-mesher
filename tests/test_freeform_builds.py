"""Stage-M3 FREEFORM acoustic-loop, guard, and end-to-end build coverage."""

from __future__ import annotations

import copy

import meshio
import numpy as np
import pytest

from hornlab_mesher import build_from_config
from hornlab_mesher.config_builder import (
    MAX_EFFECTIVE_AXIAL_RINGS,
    MAX_EFFECTIVE_CONTROL_POINTS,
    MAX_EFFECTIVE_PHI_PROFILES,
    circsym_rejection_reasons,
)
from hornlab_mesher.config_parser import ConfigError
from hornlab_mesher.freeform import validate_outer_offset_grid
from hornlab_mesher.normals import validate_orientation


def _owner_config(*, quadrants: str = "1234") -> dict:
    return {
        "formula": "FREEFORM",
        "mode": "bare",
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
                {"t": 0.4, "shape": "rounded_rectangle", "cornerRatio": 0.12},
                {"t": 1.0, "shape": "rounded_rectangle", "cornerRatio": 0.12},
            ],
        },
        "mesh": {
            "angularSegments": 32,
            "lengthSegments": 8,
            "samplingMode": "uniform",
            "quadrants": quadrants,
            "throat_res_mm": 8.0,
            "mouth_res_mm": 20.0,
            "rear_res_mm": 20.0,
            "enc_front_res_mm": 25.0,
            "enc_back_res_mm": 25.0,
            "max_triangles": 100_000,
            "allow_large_mesh": True,
        },
    }


def _assert_closed_valid_mesh(result, *, require_positive_volume: bool) -> None:
    mesh = meshio.read(result.mesh_path)
    points = np.asarray(mesh.points, dtype=np.float64)
    triangles = np.asarray(mesh.cells_dict["triangle"], dtype=np.int64)
    tags = np.asarray(
        mesh.cell_data_dict["gmsh:physical"]["triangle"], dtype=np.int32
    )
    report = validate_orientation(
        points,
        triangles,
        tags,
        source_axis="z",
        require_watertight=True,
        require_edge_consistency=True,
        require_positive_volume=require_positive_volume,
        require_source_normal=True,
    )
    assert report.watertight
    assert report.inconsistent_edges == 0


@pytest.mark.parametrize(
    "mode,extra,require_positive_volume",
    [
        ("freestanding", {"wall_thickness_mm": 1.0}, True),
        ("enclosure", None, True),
        ("infinite-baffle", None, False),
    ],
)
def test_owner_freeform_closed_acoustic_builds_are_watertight(
    tmp_path, mode, extra, require_positive_volume
):
    config = _owner_config()
    config["mode"] = mode
    if extra:
        config["mesh"].update(extra)
    if mode == "enclosure":
        config["enclosure"] = {"depth": 150.0}

    result = build_from_config(config, tmp_path / f"freeform-{mode}.msh")

    _assert_closed_valid_mesh(
        result, require_positive_volume=require_positive_volume
    )
    assert result.metadata["freeformProfileDeviationMm"] < 0.25
    assert result.metadata["freeformReport"]["throatRadiusMm"] == pytest.approx(12.7)


def test_owner_freeform_quadrant_reduced_freestanding_build_succeeds(tmp_path):
    config = _owner_config(quadrants="1")
    config["mode"] = "freestanding"
    config["mesh"]["wall_thickness_mm"] = 1.0

    result = build_from_config(config, tmp_path / "freeform-quarter.msh")

    assert result.n_triangles > 0
    assert result.native_symmetry_plane == "yz+xz"
    assert result.metadata["freeformProfileDeviationMm"] < 0.25


def test_small_corner_large_mouth_acoustic_fit_stays_within_caps(tmp_path):
    config = _owner_config()
    config["profile"]["profileH"]["points"][-1][1] = 200.0
    config["profile"]["profileV"]["points"][-1][1] = 140.0
    config["profile"]["crossSections"] = [
        {"t": 0.0, "shape": "circle"},
        {"t": 0.5, "shape": "superellipse", "exponent": 16.0},
        {"t": 1.0, "shape": "rounded_rectangle", "cornerRatio": 0.02},
    ]
    config["mesh"]["mouth_res_mm"] = 15.0

    result = build_from_config(config, tmp_path / "freeform-small-corner.msh")

    metadata = result.metadata
    assert metadata["geometrySamplePhiProfiles"] <= MAX_EFFECTIVE_PHI_PROFILES
    assert metadata["geometrySampleAxialRings"] <= MAX_EFFECTIVE_AXIAL_RINGS
    assert metadata["geometrySampleControlPoints"] <= MAX_EFFECTIVE_CONTROL_POINTS


def test_tight_corner_thick_wall_fails_surface_curvature_guard(tmp_path):
    config = _owner_config()
    config["mode"] = "freestanding"
    config["mesh"]["wall_thickness_mm"] = 6.0
    config["profile"]["crossSections"] = [
        {"t": 0.0, "shape": "circle"},
        {"t": 0.5, "shape": "superellipse", "exponent": 16.0},
        {"t": 1.0, "shape": "rounded_rectangle", "cornerRatio": 0.02},
    ]

    with pytest.raises(
        ConfigError,
        match=r"surface-curvature guard near t=.*phi=.*wallThickness\*kappa_i",
    ):
        build_from_config(config, tmp_path / "must-not-build.msh")


def test_generated_outer_offset_grid_guard_rejects_a_normal_flip():
    phi = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    z = np.linspace(0.0, 40.0, 5)
    radius = 12.0 + 0.5 * z
    inner = np.empty((phi.size, z.size, 3), dtype=np.float64)
    inner[:, :, 0] = np.cos(phi)[:, None] * radius[None, :]
    inner[:, :, 1] = np.sin(phi)[:, None] * radius[None, :]
    inner[:, :, 2] = z[None, :]
    outer = inner.copy()
    outer[:, :, :2] *= (radius + 1.0)[None, :, None] / radius[None, :, None]
    outer[[3, 4]] = outer[[4, 3]]

    with pytest.raises(ValueError, match="outer offset grid has a normal flip"):
        validate_outer_offset_grid(inner, outer, full_circle=True)


def test_freeform_circsym_eligibility_tracks_actual_azimuthal_span():
    axisymmetric = _owner_config()
    axisymmetric["profile"]["profileV"] = copy.deepcopy(
        axisymmetric["profile"]["profileH"]
    )
    axisymmetric["profile"].pop("crossSections")
    assert circsym_rejection_reasons(axisymmetric) == []

    non_axisymmetric = copy.deepcopy(axisymmetric)
    non_axisymmetric["profile"]["profileV"]["points"][-1][1] -= 10.0
    reasons = circsym_rejection_reasons(non_axisymmetric)
    assert reasons
    assert "varies with azimuth" in reasons[0]
    assert "radius span" in reasons[0]
