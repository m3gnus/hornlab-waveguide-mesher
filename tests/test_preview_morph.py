"""Rounded-rectangle morph and tolerant winding regressions."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from hornlab_mesher.preview.api import (
    PreviewOptionsV1,
    PreviewSurfaceV1,
    _configuration_has_corners,
    build_preview_geometry,
)


ROUNDED_RECT_MORPH = {
    "formula": "R-OSSE",
    "mode": "freestanding",
    "profile": {
        "r0": 12.7,
        "a0": 12.0,
        "a": 40.0,
        "k": 1.0,
        "r": 0.3,
        "m": 0.8,
        "b": 0.3,
        "q": 0.995,
        "R": 150.0,
    },
    "mesh": {"wallThickness": 5.0},
    "morph": {
        "morphTarget": 1.0,
        "morphWidth": 300.0,
        "morphHeight": 200.0,
    },
}


@pytest.mark.parametrize(
    "config",
    [
        {"formula": "OSSE", "profile": {}, "morph": {"morph_target": 1}},
        {"formula": "OSSE", "profile": {}, "MORPH": {"morphTarget": 1}},
        {"formula": "OSSE", "profile": {}, "morphTarget": 1},
        {"formula": "OSSE", "profile": {"morph_target": 1}},
    ],
)
def test_corner_detection_accepts_canonical_and_legacy_morph_locations(config):
    assert _configuration_has_corners(config) is True


def test_only_rounded_rectangle_morph_target_has_corners():
    assert not _configuration_has_corners(
        {"formula": "OSSE", "profile": {}, "morph": {"morphTarget": 0}}
    )
    assert not _configuration_has_corners(
        {"formula": "OSSE", "profile": {}, "morph": {"morphTarget": 2}}
    )


@pytest.mark.parametrize("lod", ["coarse", "fine", "inspection"])
@pytest.mark.parametrize("corner_radius", [None, 40.0])
def test_rounded_rectangle_morph_builds_at_every_lod(lod, corner_radius):
    config = copy.deepcopy(ROUNDED_RECT_MORPH)
    if corner_radius is not None:
        config["morph"]["morphCorner"] = corner_radius

    preview = build_preview_geometry(config, PreviewOptionsV1(lod=lod))

    assert {surface.role for surface in preview.surfaces} == {
        "horn.inner",
        "horn.outer",
        "mouth_rim",
        "source_cap",
        "wall.rear_cap",
    }
    assert preview.metadata["angular_sampling"]["strategy"] == (
        "stable-union-corner-grid"
    )
    assert all(surface.metadata["disagreeingTriangles"] == 0 for surface in preview.surfaces)


def test_rounded_rectangle_morph_builds_with_implicit_extents():
    config = copy.deepcopy(ROUNDED_RECT_MORPH)
    config["morph"] = {"morphTarget": 1.0}

    preview = build_preview_geometry(config, PreviewOptionsV1(lod="coarse"))

    assert preview.surfaces


def test_rounded_rectangle_morph_builds_when_shrinkage_sharpens_the_corner():
    """A target smaller than the mouth leaves a genuinely sharp corner.

    Offsetting a sharp corner has no defined direction there, so the outer shell
    carries two full-area facets tipped a few degrees past perpendicular. That is
    a singularity in the surface, not an inverted patch, and it must not cost the
    user every surface in the preview.
    """

    config = copy.deepcopy(ROUNDED_RECT_MORPH)
    config["morph"]["morphAllowShrinkage"] = 1

    preview = build_preview_geometry(config, PreviewOptionsV1(lod="fine"))

    outer = next(surface for surface in preview.surfaces if surface.role == "horn.outer")
    assert outer.metadata["disagreeingTriangles"] == 0
    assert 0 < outer.metadata["orientationSingularTriangles"] <= 8


def _tiled_plane(rows: int, columns: int) -> tuple[np.ndarray, np.ndarray]:
    """A +z plane fine enough that one extra triangle is a negligible area."""

    grid = np.stack(
        np.meshgrid(np.arange(columns + 1.0), np.arange(rows + 1.0), indexing="xy"),
        axis=-1,
    ).reshape(-1, 2)
    positions = np.column_stack((grid, np.zeros(len(grid))))
    stride = columns + 1
    triangles = []
    for row in range(rows):
        for column in range(columns):
            corner = row * stride + column
            triangles += [corner, corner + 1, corner + stride + 1]
            triangles += [corner, corner + stride + 1, corner + stride]
    return positions, np.asarray(triangles, dtype=np.uint32)


@pytest.mark.parametrize(
    "tilt, expect_raise",
    [
        # Barely past perpendicular over a negligible area: a singularity.
        (-0.13, False),
        # Pointing decisively the wrong way over the same area: a real fault,
        # and the whole reason this check exists.
        (-0.97, True),
    ],
)
def test_a_negligible_minority_is_forgiven_only_when_it_is_shallow(tilt, expect_raise):
    positions, indices = _tiled_plane(20, 20)
    normals = np.tile((0.0, 0.0, 1.0), (len(positions), 1))
    depth = float(np.sqrt(1.0 - tilt * tilt))
    apex = len(positions)
    scale = 0.5
    positions = np.vstack((
        positions,
        ((0.0, 0.0, 0.0), (scale, 0.0, 0.0), (0.0, scale * tilt, scale * depth)),
    ))
    normals = np.vstack((normals, np.tile((0.0, 0.0, 1.0), (3, 1))))
    indices = np.concatenate((indices, np.asarray((apex, apex + 1, apex + 2), dtype=np.uint32)))

    def build() -> PreviewSurfaceV1:
        return PreviewSurfaceV1(
            role="horn.outer",
            positions=positions,
            indices=indices,
            normals=normals,
            shading="smooth",
            normal_method="analytic-parametric",
            closed_phi=False,
        )

    if expect_raise:
        with pytest.raises(ValueError, match="disagree with their normals"):
            build()
        return
    surface = build()
    assert surface.metadata["disagreeingTriangles"] == 0
    assert surface.metadata["orientationSingularTriangles"] == 1


def test_full_area_inverted_triangle_still_raises():
    positions = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    normals = np.tile((0.0, 0.0, 1.0), (4, 1))
    indices = np.asarray((0, 1, 2, 0, 3, 2), dtype=np.uint32)

    with pytest.raises(ValueError, match="non-degenerate triangle"):
        PreviewSurfaceV1(
            role="horn.outer",
            positions=positions,
            indices=indices,
            normals=normals,
            shading="smooth",
            normal_method="analytic-parametric",
            closed_phi=False,
        )


def test_zero_area_sliver_abstains_and_is_reported():
    positions = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    normals = np.tile((0.0, 0.0, 1.0), (4, 1))
    indices = np.asarray((0, 1, 2, 0, 2, 3, 0, 0, 1), dtype=np.uint32)

    surface = PreviewSurfaceV1(
        role="horn.outer",
        positions=positions,
        indices=indices,
        normals=normals,
        shading="smooth",
        normal_method="analytic-parametric",
        closed_phi=False,
    )

    assert surface.metadata["degenerateTriangles"] == 1
    assert surface.metadata["orientationAbstainingTriangles"] == 1
    assert surface.metadata["disagreeingTriangles"] == 0
