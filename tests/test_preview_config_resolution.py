"""The preview resolves its config through the same resolver as the solved build.

``build_geometry_params`` accepts the formula at the top level or under the
profile section (``profile`` or ``parameters``), as ``formula`` or ``type``, and
normalises spellings such as ``r_osse``/``ROSSE``. The preview used to re-read
``config["formula"]`` itself, so a documented alias built an ICW preview with a
z-map ICW refuses, took FREEFORM off its corner sampling (a 22 degree normal
step against a 3 degree target), and dropped R-OSSE's doubled reference.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry

from test_preview_api import (
    FREEFORM_FREESTANDING,
    ICW_FLAT_BAFFLE,
    OSSE_FREESTANDING,
    ROSSE_ENCLOSURE,
)


def _formula_in_profile(config, *, key="formula", section="profile"):
    moved = copy.deepcopy(config)
    formula = moved.pop("formula")
    profile = moved.pop("profile")
    profile[key] = formula
    moved[section] = profile
    return moved


def _assert_same_preview(left, right):
    assert [surface.role for surface in left.surfaces] == [
        surface.role for surface in right.surfaces
    ]
    for a, b in zip(left.surfaces, right.surfaces):
        np.testing.assert_array_equal(a.positions, b.positions)
        np.testing.assert_array_equal(a.indices, b.indices)
    for key in ("formula", "mode", "actual_segment_counts", "canonical_reference",
                "fidelity", "angular_sampling", "semantic_stations", "warnings"):
        assert left.metadata[key] == right.metadata[key], key


@pytest.mark.parametrize(
    "config",
    [ICW_FLAT_BAFFLE, FREEFORM_FREESTANDING, OSSE_FREESTANDING, ROSSE_ENCLOSURE],
    ids=["icw", "freeform", "osse", "rosse"],
)
@pytest.mark.parametrize(
    "key,section",
    [("formula", "profile"), ("type", "profile"), ("formula", "parameters")],
)
def test_profile_formula_alias_previews_exactly_like_top_level(config, key, section):
    options = PreviewOptionsV1(lod="coarse")
    expected = build_preview_geometry(config, options)
    aliased = build_preview_geometry(
        _formula_in_profile(config, key=key, section=section), options
    )
    _assert_same_preview(expected, aliased)


def test_icw_preview_with_profile_formula_builds():
    preview = build_preview_geometry(_formula_in_profile(ICW_FLAT_BAFFLE))
    assert preview.metadata["formula"] == "ICW"


def test_freeform_preview_with_profile_formula_meets_its_normal_target():
    preview = build_preview_geometry(_formula_in_profile(FREEFORM_FREESTANDING))
    inner = preview.metadata["fidelity"]["horn.inner"]
    assert preview.metadata["angular_sampling"]["strategy"] == "stable-union-corner-grid"
    assert inner["max_normal_step_deg_achieved"] <= inner[
        "max_normal_step_deg_requested"
    ] * (1.0 + 1.0e-9)


@pytest.mark.parametrize("spelling", ["r_osse", "rosse", "ROSSE", "r-osse"])
def test_rosse_spellings_keep_the_rosse_reference(spelling):
    # Coarse is where R-OSSE builds its doubled reference outright, so a
    # spelling the preview failed to recognise shows up as half the rows.
    options = PreviewOptionsV1(lod="coarse")
    expected = build_preview_geometry(ROSSE_ENCLOSURE, options)
    config = copy.deepcopy(ROSSE_ENCLOSURE)
    config["formula"] = spelling
    _assert_same_preview(expected, build_preview_geometry(config, options))


# --- Reduced-domain source cap ---------------------------------------------

def _rounded_cap_preview(quadrants):
    config = copy.deepcopy(OSSE_FREESTANDING)
    config["source"] = {"source_shape": 1}
    config["mesh"]["quadrants"] = quadrants
    preview = build_preview_geometry(config, PreviewOptionsV1(lod="coarse"))
    by_role = {surface.role: surface for surface in preview.surfaces}
    return preview, by_role["source_cap"], by_role["horn.inner"]


@pytest.mark.parametrize("quadrants", [1, 12, 14])
def test_reduced_source_cap_is_centred_on_the_axis_like_the_full_model(quadrants):
    """The solved build pins an open ring's cap centre to the axis.

    The preview took the mean of the (arc-shaped) throat ring instead: a
    quarter model put the pole at (8.06, 8.06) mm with a 0.68 mm cap on an
    18.8 mm sphere, against 1.73 mm on 47.5 mm for the same horn in full.
    """

    full, full_cap, _ = _rounded_cap_preview(1234)
    reduced, cap, inner = _rounded_cap_preview(quadrants)

    pole = cap.positions[0]
    assert abs(pole[0]) < 1.0e-9 and abs(pole[1]) < 1.0e-9
    assert pole[2] == pytest.approx(full_cap.positions[0][2], abs=1.0e-9)
    for key in ("source_cap_height_mm", "source_cap_radius_mm"):
        assert reduced.metadata[key] == pytest.approx(full.metadata[key], rel=1.0e-9)

    # The cap's outer ring is the throat ring the wall starts from.
    n_phi = len(inner.positions) // (reduced.metadata["actual_segment_counts"]["horn_axial"] + 1)
    throat = inner.positions[:n_phi]
    rim = cap.positions[-n_phi:]
    np.testing.assert_allclose(rim, throat, atol=1.0e-6)

    # The open ring's angular step is measured on its real span (a quarter
    # turn here), so the cap reports the same normal step as the full model's
    # equally spaced ring rather than a half-turn guess.
    assert reduced.metadata["fidelity"]["source_cap"][
        "max_normal_step_deg_achieved"
    ] == pytest.approx(
        full.metadata["fidelity"]["source_cap"]["max_normal_step_deg_achieved"],
        rel=1.0e-6,
    )


# --- Constant-radius runs in the semantic stations ---------------------------

def test_flat_radius_run_contributes_only_its_two_ends():
    from hornlab_mesher.preview.api import _radial_extrema

    # throat extension (flat), flare, a rollback turn, then flat again
    radius = np.array([5.0, 5.0, 5.0, 5.0, 6.0, 8.0, 9.0, 8.5, 8.5, 8.5])
    assert _radial_extrema(radius).tolist() == [3, 6, 7]


def test_float_noise_is_not_a_turn():
    from hornlab_mesher.preview.api import _radial_extrema

    radius = 20.0 + np.array([0.0, 1e-15, -1e-15, 2e-15, 0.0, 1.0, 2.0])
    assert _radial_extrema(radius).tolist() == [4]


def test_throat_extension_does_not_inflate_the_axial_grid():
    """A 30 mm straight extension used to force 100 axial rows instead of 58.

    The extension is real geometry with a crease at its end, so it may earn a
    few rows over the plain horn's 54; it must not earn one per master row.
    """

    with_extension = copy.deepcopy(OSSE_FREESTANDING)
    with_extension["profile"]["throat_ext_length_mm"] = 30.0
    base = build_preview_geometry(OSSE_FREESTANDING).metadata
    extended = build_preview_geometry(with_extension).metadata
    assert (
        extended["actual_segment_counts"]["horn_axial"]
        <= base["actual_segment_counts"]["horn_axial"] + 6
    )
