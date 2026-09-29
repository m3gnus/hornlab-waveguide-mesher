"""Previews that used to be refused now degrade with a warning.

Each case here raised ``ValueError`` from ``build_preview_geometry`` at 0.2.x:
a folded outer wall (``horn.outer: inconsistent local orientation``), an
enclosure on a reduced-quadrant model (``enclosure.front: ...``), and, without
raising, a tight normal target that asked for ~950k vertices.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry

_FRAME = {
    "mode": "freestanding",
    "mesh": {
        "quadrants": 1234,
        "wallThickness": 5.0,
        "surfaceFit": "interpolate",
        "scaleToMetres": True,
        "verticalOffset": 0.0,
    },
    "cross_section": {"exponent": 2.0, "aspectRatio": 1.0},
    "morph": {},
    "gcurve": {},
    "source": {},
}
ROSSE_EXTENSION = {
    **_FRAME,
    "formula": "R-OSSE",
    "profile": {"formula": "R-OSSE", "throatExtLength": "30", "throatExtAngle": "5"},
}
ROSSE_SLOT = {
    **_FRAME,
    "formula": "R-OSSE",
    "profile": {"formula": "R-OSSE", "slotLength": "15"},
}
FREEFORM_MORPH = {
    **_FRAME,
    "formula": "FREEFORM",
    "profile": {
        "formula": "FREEFORM",
        "profileH": {"points": [[0.0, 10.0], [100.0, 50.0]]},
        "profileV": {"points": [[0.0, 10.0], [100.0, 40.0]]},
        "crossSections": [
            {"t": 0.0, "shape": "ellipse"},
            {"t": 1.0, "shape": "ellipse"},
        ],
        "inflectionPolicy": "warn",
    },
    "mesh": {**_FRAME["mesh"], "surfaceFit": "approximate"},
    "morph": {"morphTarget": "1", "morphCorner": "20", "morphRate": "3"},
}


def _outer(geometry):
    return next(s for s in geometry.surfaces if s.role == "horn.outer")


@pytest.mark.parametrize(
    ("config", "lod"),
    [
        (ROSSE_EXTENSION, "fine"),
        (ROSSE_SLOT, "fine"),
        (FREEFORM_MORPH, "coarse"),
        (FREEFORM_MORPH, "fine"),
    ],
    ids=["rosse-extension-fine", "rosse-slot-fine", "freeform-morph-coarse", "freeform-morph-fine"],
)
def test_a_folded_outer_wall_is_shown_with_a_warning(config, lod):
    options = PreviewOptionsV1(lod=lod, include_curvature=False)

    geometry = build_preview_geometry(copy.deepcopy(config), options)

    outer = _outer(geometry)
    assert outer.metadata["foldedTriangles"] > 0
    assert outer.metadata["disagreeingTriangles"] == 0
    assert any("outer wall folds over itself" in w for w in geometry.metadata["warnings"])
    # ``PreviewSurfaceV1`` re-runs the winding check on the shipped buffers and
    # refuses any disagreeing triangle, so reaching here means the fold was
    # wound to its normals. It is a small share of the wall, not the wall.
    assert outer.metadata["foldedTriangles"] < 0.05 * len(outer.indices) / 3
    assert outer.metadata["windingChecked"] is True

    # The acoustic surface is exactly the one the same design has without a wall.
    bare = copy.deepcopy(config)
    bare["mesh"] = {**bare["mesh"], "wallThickness": 0.0}
    inner = next(s for s in geometry.surfaces if s.role == "horn.inner")
    reference = next(
        s
        for s in build_preview_geometry(bare, options).surfaces
        if s.role == "horn.inner"
    )
    assert np.array_equal(inner.positions, reference.positions)


def test_a_healthy_outer_wall_carries_no_fold_report():
    geometry = build_preview_geometry(
        copy.deepcopy(ROSSE_EXTENSION), PreviewOptionsV1(lod="coarse")
    )

    assert "foldedTriangles" not in _outer(geometry).metadata
    assert not any("folds" in w for w in geometry.metadata["warnings"])


def _enclosure_config(quadrants):
    return {
        "formula": "OSSE",
        "mode": "enclosure",
        "profile": {"formula": "OSSE", "L": 120, "a": 45, "a0": 10, "r0": 12.7},
        "mesh": {"quadrants": quadrants, "wallThickness": 0.0},
        "enclosure": {
            "depth_mm": 200,
            "edge_radius_mm": 20,
            "space_l": 20,
            "space_t": 20,
            "space_r": 20,
            "space_b": 20,
        },
    }


@pytest.mark.parametrize("quadrants", [1, 12, 14])
def test_a_reduced_quadrant_enclosure_previews_the_horn_alone(quadrants):
    geometry = build_preview_geometry(
        _enclosure_config(quadrants), PreviewOptionsV1(lod="coarse")
    )

    roles = {s.role for s in geometry.surfaces}
    assert "horn.inner" in roles
    assert not any(role.startswith("enclosure.") for role in roles)
    assert any("enclosure not drawn" in w for w in geometry.metadata["warnings"])


def test_the_full_enclosure_is_still_drawn():
    geometry = build_preview_geometry(
        _enclosure_config(1234), PreviewOptionsV1(lod="coarse")
    )

    assert "enclosure.front" in {s.role for s in geometry.surfaces}
    assert not any("enclosure not drawn" in w for w in geometry.metadata["warnings"])


_OSSE = {
    "formula": "OSSE",
    "mode": "freestanding",
    "profile": {"formula": "OSSE", "L": 120, "a": 45, "a0": 10, "r0": 12.7},
    "mesh": {"quadrants": 1234, "wallThickness": 3.0},
}


def test_a_tight_normal_target_is_bounded_by_default():
    geometry = build_preview_geometry(
        copy.deepcopy(_OSSE),
        PreviewOptionsV1(lod="coarse", max_normal_step_deg=0.35),
    )

    assert geometry.metadata["requested_fidelity"]["max_vertices"] == 200_000
    assert any("bounded at 200000 vertices" in w for w in geometry.metadata["warnings"])
    assert max(len(s.positions) for s in geometry.surfaces) <= 200_000


def test_an_explicit_vertex_bound_and_the_lod_presets_are_left_alone():
    explicit = build_preview_geometry(
        copy.deepcopy(_OSSE),
        PreviewOptionsV1(lod="coarse", max_normal_step_deg=0.35, max_vertices=5_000),
    )
    assert explicit.metadata["requested_fidelity"]["max_vertices"] == 5_000
    assert not any("bounded at" in w for w in explicit.metadata["warnings"])

    for lod in ("coarse", "fine", "inspection"):
        preset = build_preview_geometry(
            copy.deepcopy(_OSSE), PreviewOptionsV1(lod=lod, include_curvature=False)
        )
        assert preset.metadata["requested_fidelity"]["max_vertices"] is None
