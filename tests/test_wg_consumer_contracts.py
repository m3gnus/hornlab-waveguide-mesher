"""Contracts of mesher functions Waveguide Generator calls in production.

These functions had no test in this repository -- they were exercised only by
WG's integration suite, so a change here went green in mesher CI and failed at
the next pin bump. Each test pins the behaviour WG reads (see
docs/public-api.md, "Integration API"), not the implementation.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from hornlab_mesher.step_import import (
    ANCHOR_MAX_AREA_REL_DIFF,
    ANCHOR_MAX_CENTROID_DISTANCE_MM,
    FREQUENCY_ELEMENTS_PER_WAVELENGTH,
    RIGID_TAG,
    SPEED_OF_SOUND_M_S,
    StepFaceGroup,
    StepLabelSelector,
    anchor_surface_order,
    gmsh_surface_geometries,
    gmsh_surface_tags,
    mesh_frequency_validation,
)
from hornlab_mesher.step_prepare import OccSurfaceRole

SOURCE_TAG = 2


def _limit_hz(edge_mm: float) -> float:
    return SPEED_OF_SOUND_M_S / (FREQUENCY_ELEMENTS_PER_WAVELENGTH * edge_mm * 1.0e-3)


def _frequency_mesh():
    """A 3-4-5 mm source triangle, a nearby 10 mm rigid one, a far 50 mm one."""

    points = np.array(
        [
            # source: legs 3 and 4 mm, hypotenuse 5 mm
            [0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0],
            # near rigid wall: legs 10 mm, hypotenuse 14.142 mm, ~12 mm away
            [10.0, 10.0, 0.0], [20.0, 10.0, 0.0], [10.0, 20.0, 0.0],
            # far rigid wall: legs 50 mm, hypotenuse 70.71 mm, ~500 mm away
            [500.0, 0.0, 0.0], [550.0, 0.0, 0.0], [500.0, 50.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)
    tags = np.array([SOURCE_TAG, RIGID_TAG, RIGID_TAG], dtype=np.int32)
    return points, triangles, tags


def _source(name="throat", tag=SOURCE_TAG, resolution_mm=2.5):
    return StepFaceGroup(
        name=name,
        selector=StepLabelSelector(name),
        role=OccSurfaceRole("source"),
        tag=tag,
        resolution_mm=resolution_mm,
    )


def _validate(requested, **kwargs):
    points, triangles, tags = _frequency_mesh()
    return mesh_frequency_validation(
        points,
        triangles,
        tags,
        [_source()],
        unit_scale_to_m=1.0e-3,
        requested_max_frequency_hz=requested,
        transition_mm=100.0,
        **kwargs,
    )


# --- mesh_frequency_validation ----------------------------------------------

def test_frequency_limits_follow_six_elements_per_wavelength():
    report = _validate(None)

    assert report["status"] == "unknown"
    assert report["global_status"] == "unknown"
    assert report["frequency_policy"] == "global_warn_source_hard"
    assert report["warnings"] == [] and report["invalid_sources"] == []
    assert report["edge_limit_m"] is None and report["edge_limit_step_units"] is None
    assert report["elements_per_wavelength"] == 6.0
    assert report["speed_of_sound_m_s"] == 343.0

    far = 50.0 * math.sqrt(2.0)
    assert report["global_max_edge_step_units"] == pytest.approx(far)
    assert report["global_max_edge_m"] == pytest.approx(far * 1.0e-3)
    assert report["global_max_valid_frequency_hz"] == pytest.approx(_limit_hz(far))
    assert report["max_valid_frequency_hz"] == report["global_max_valid_frequency_hz"]

    source = report["per_source"]["throat"]
    assert source["tag"] == SOURCE_TAG
    assert source["triangle_count"] == 1
    assert source["requested_resolution_mm"] == 2.5
    assert source["status"] == "unknown"
    assert source["max_edge_step_units"] == pytest.approx(5.0)
    assert source["max_valid_frequency_hz"] == pytest.approx(_limit_hz(5.0))
    # Only the rigid wall within ``transition_mm`` limits the source; the far
    # coarse wall does not.
    near = 10.0 * math.sqrt(2.0)
    assert source["wall_triangle_count"] == 1
    assert source["wall_distance_mm"] == 100.0
    assert source["wall_max_edge_step_units"] == pytest.approx(near)
    assert source["wall_max_valid_frequency_hz"] == pytest.approx(_limit_hz(near))
    # WG's solver reads this one: the lower of the patch and its walls.
    assert source["effective_max_valid_frequency_hz"] == pytest.approx(_limit_hz(near))
    json.dumps(report, allow_nan=False)


def test_coarse_walls_elsewhere_warn_but_do_not_fail_the_source():
    near_limit = _limit_hz(10.0 * math.sqrt(2.0))
    requested = 0.9 * near_limit  # above the far wall's limit, below the near one's

    report = _validate(requested)

    assert report["status"] == "valid"
    assert report["global_status"] == "invalid"
    assert report["invalid_sources"] == []
    assert report["per_source"]["throat"]["status"] == "valid"
    assert len(report["warnings"]) == 1
    assert "exceeds conservative global mesh limit" in report["warnings"][0]
    assert report["edge_limit_m"] == pytest.approx(
        SPEED_OF_SOUND_M_S / (FREQUENCY_ELEMENTS_PER_WAVELENGTH * requested)
    )
    assert report["edge_limit_step_units"] == pytest.approx(report["edge_limit_m"] * 1.0e3)


def test_an_underresolved_wall_next_to_the_source_fails_it():
    requested = 1.1 * _limit_hz(10.0 * math.sqrt(2.0))  # patch alone would pass

    report = _validate(requested)

    assert report["status"] == "invalid"
    assert report["invalid_sources"] == ["throat"]
    assert report["per_source"]["throat"]["status"] == "invalid"
    assert any(
        "throat rigid walls within the transition distance are underresolved" in warning
        for warning in report["warnings"]
    )


def test_an_underresolved_source_patch_fails_it():
    requested = 1.1 * _limit_hz(5.0)
    points, triangles, tags = _frequency_mesh()
    report = mesh_frequency_validation(
        points,
        triangles,
        tags,
        [_source()],
        unit_scale_to_m=1.0e-3,
        requested_max_frequency_hz=requested,
        transition_mm=1.0,  # no wall within reach
    )

    source = report["per_source"]["throat"]
    assert "wall_triangle_count" not in source
    assert source["effective_max_valid_frequency_hz"] == pytest.approx(_limit_hz(5.0))
    assert report["invalid_sources"] == ["throat"]
    assert any("throat source patch is underresolved" in w for w in report["warnings"])


def test_a_source_with_no_triangles_is_invalid_when_a_band_is_requested():
    points, triangles, tags = _frequency_mesh()
    report = mesh_frequency_validation(
        points,
        triangles,
        tags,
        [_source(name="missing", tag=77)],
        unit_scale_to_m=1.0e-3,
        requested_max_frequency_hz=1000.0,
    )

    missing = report["per_source"]["missing"]
    assert missing["triangle_count"] == 0
    assert missing["effective_max_valid_frequency_hz"] == 0.0
    assert report["invalid_sources"] == ["missing"]


def test_units_follow_unit_scale_to_m():
    points, triangles, tags = _frequency_mesh()
    in_metres = mesh_frequency_validation(
        points * 1.0e-3,
        triangles,
        tags,
        [_source()],
        unit_scale_to_m=1.0,
        requested_max_frequency_hz=None,
        transition_mm=0.1,  # compared in the points' own units
    )
    assert in_metres["global_max_valid_frequency_hz"] == pytest.approx(
        _validate(None)["global_max_valid_frequency_hz"]
    )
    assert in_metres["global_max_edge_step_units"] == pytest.approx(0.05 * math.sqrt(2.0))


# --- anchor_surface_order ----------------------------------------------------

_REFERENCE = [
    ((0.0, 0.0, 0.0), 100.0),
    ((50.0, 0.0, 0.0), 200.0),
    ((0.0, 80.0, 0.0), 300.0),
    ((0.0, 0.0, 120.0), 300.0),  # same area as the previous one: centroid decides
]


def test_anchor_recovers_the_reference_order_from_a_permutation():
    permutation = [2, 0, 3, 1]
    healed_tags = [11, 12, 13, 14]
    healed_geometries = [_REFERENCE[index] for index in permutation]

    order = anchor_surface_order(healed_tags, healed_geometries, _REFERENCE)

    # order[i] is the healed tag carrying reference face i.
    assert order == [12, 14, 11, 13]


def test_anchor_tolerates_healing_noise_inside_its_limits():
    small = 0.5 * ANCHOR_MAX_AREA_REL_DIFF
    shift = 0.5 * ANCHOR_MAX_CENTROID_DISTANCE_MM
    healed = [((x + shift, y, z), area * (1.0 + small)) for (x, y, z), area in _REFERENCE]

    assert anchor_surface_order([5, 6, 7, 8], healed, _REFERENCE) == [5, 6, 7, 8]


@pytest.mark.parametrize(
    "healed",
    [
        [((0.0, 0.0, 0.0), 100.0 * (1.0 + 2.5 * ANCHOR_MAX_AREA_REL_DIFF)), *_REFERENCE[1:]],
        [((2.0 * ANCHOR_MAX_CENTROID_DISTANCE_MM, 0.0, 0.0), 100.0), *_REFERENCE[1:]],
    ],
    ids=["area", "centroid"],
)
def test_anchor_refuses_an_implausible_match(healed):
    with pytest.raises(RuntimeError, match="implausible geometry residuals"):
        anchor_surface_order([1, 2, 3, 4], healed, _REFERENCE)


@pytest.mark.parametrize(
    "tags,geometries,message",
    [
        ([1, 2, 3], _REFERENCE[:3], "surface count mismatch"),
        ([1, 2, 3, 4], _REFERENCE[:3], "healed tag and geometry counts differ"),
        ([1, 1, 2, 3], _REFERENCE, "not unique"),
        ([1, 2, 3, 4], [((0.0, 0.0, 0.0), 0.0), *_REFERENCE[1:]], "invalid surface geometry"),
        ([1, 2, 3, 4], [((math.nan, 0.0, 0.0), 1.0), *_REFERENCE[1:]], "invalid surface geometry"),
    ],
)
def test_anchor_refuses_inconsistent_input(tags, geometries, message):
    with pytest.raises(RuntimeError, match=message):
        anchor_surface_order(tags, geometries, _REFERENCE)


def test_anchor_of_nothing_is_nothing():
    assert anchor_surface_order([], [], []) == []


# --- gmsh_surface_tags / gmsh_surface_geometries, end to end -----------------

@pytest.fixture
def gmsh_session():
    import gmsh

    initialized_here = not gmsh.isInitialized()
    if initialized_here:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.clear()
    try:
        yield gmsh
    finally:
        gmsh.option.setNumber("Geometry.OCCSewFaces", 0)
        gmsh.clear()
        if initialized_here:
            gmsh.finalize()


def test_surface_anchors_survive_a_healed_reimport(gmsh_session, tmp_path):
    """The WG sequence: anchor an unhealed import, re-import healed, re-anchor."""

    gmsh = gmsh_session
    step = tmp_path / "box.step"
    gmsh.model.add("writer")
    gmsh.model.occ.addBox(0.0, 0.0, 0.0, 10.0, 20.0, 30.0)
    gmsh.model.occ.synchronize()
    gmsh.write(str(step))
    gmsh.clear()

    gmsh.open(str(step))
    reference_tags = gmsh_surface_tags()
    reference = gmsh_surface_geometries(reference_tags)
    assert reference_tags == sorted(tag for _dim, tag in gmsh.model.getEntities(2))
    assert sorted(round(area) for _center, area in reference) == [200, 200, 300, 300, 600, 600]
    for center, _area in reference:
        assert len(center) == 3
    gmsh.clear()

    gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
    gmsh.open(str(step))
    healed_tags = gmsh_surface_tags()
    healed = dict(zip(healed_tags, gmsh_surface_geometries(healed_tags)))

    order = anchor_surface_order(list(healed), list(healed.values()), reference)

    assert sorted(order) == sorted(healed_tags)
    for tag, (center, area) in zip(order, reference):
        healed_center, healed_area = healed[tag]
        assert healed_area == pytest.approx(area, rel=1.0e-9)
        np.testing.assert_allclose(healed_center, center, atol=1.0e-9)


# --- config_builder / viewport / preview --------------------------------------

_OSSE = {
    "formula": "OSSE",
    "mode": "freestanding",
    "profile": {"L_mm": 120.0, "r0_mm": 12.7, "a0_deg": 15.5, "a_deg": 55.0},
    "mesh": {"wall_thickness_mm": 6.0, "angular_segments": 32, "length_segments": 12},
}


def test_resolve_geometry_carries_what_the_export_hash_and_cad_plan_read():
    from hornlab_mesher.config_builder import resolve_geometry
    from hornlab_mesher.geometry import PointGridHornGeometry

    first = resolve_geometry(_OSSE)
    second = resolve_geometry(_OSSE)

    assert isinstance(first.geometry, PointGridHornGeometry)
    assert first.formula == "OSSE" and first.mode == "freestanding"
    angular = int(first.sampling_metadata["geometrySampleAngularSegments"])
    length = int(first.sampling_metadata["geometrySampleLengthSegments"])
    assert angular >= 4 and length >= 2
    # WG hashes ``resolved.geometry`` for export identity: same config, same geometry.
    np.testing.assert_array_equal(first.geometry.inner_points, second.geometry.inner_points)


def test_build_point_grid_flat_lists_reshape_to_the_published_counts():
    from hornlab_mesher.config_builder import build_geometry_params, build_point_grid

    params, formula, mode = build_geometry_params(_OSSE)
    assert (formula, mode) == ("OSSE", "freestanding")
    grid = build_point_grid(params)
    n_phi, n_length = int(grid["grid_n_phi"]), int(grid["grid_n_length"])
    for key in ("inner_points", "outer_points"):
        points = np.asarray(grid[key], dtype=np.float64)
        assert points.size == n_phi * (n_length + 1) * 3, key
        assert np.all(np.isfinite(points))


def test_corner_arc_subdivision_key_only_adds_samples():
    """WG's export sizing densifies the corner arcs through this private key."""

    from hornlab_mesher.config_builder import build_geometry_params, build_point_grid
    from hornlab_mesher.profile_sampling import ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY

    config = dict(_OSSE, morph={"morphTarget": 1, "morphWidth": 320.0,
                               "morphHeight": 220.0, "morphCorner": 30.0})
    params, _formula, _mode = build_geometry_params(config)
    base = build_point_grid(params)
    denser = build_point_grid({**params, ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY: 3})
    assert int(denser["grid_n_phi"]) > int(base["grid_n_phi"])
    assert int(denser["grid_n_length"]) == int(base["grid_n_length"])


def test_viewport_inner_points_reshape_phi_major():
    from hornlab_mesher.viewport import build_viewport_geometry_from_config

    geometry = build_viewport_geometry_from_config(_OSSE)
    grid = geometry["grid"]
    n_phi, n_length = int(grid["grid_n_phi"]), int(grid["grid_n_length"])
    points = np.asarray(grid["inner_points"], dtype=np.float64)
    assert n_phi >= 4 and n_length >= 1
    assert points.size == n_phi * (n_length + 1) * 3
    rings = points.reshape(n_phi, n_length + 1, 3)
    # phi-major: each row is one meridian running throat -> mouth.
    assert np.all(np.diff(rings[:, :, 2], axis=1) > 0.0)
    assert np.all(np.isfinite(rings))


def test_preview_entry_points_wg_imports_from_preview_api():
    import hornlab_mesher.preview.api as api

    for name in ("build_preview_geometry", "PreviewOptionsV1", "PreviewGeometryV1",
                 "PreviewSurfaceV1", "_lod_config", "_guiding_curve_warnings"):
        assert hasattr(api, name), name
    # WG builds these options field by field.
    options = api.PreviewOptionsV1(
        lod="coarse",
        include_inner=True,
        include_outer=True,
        include_enclosure=True,
        include_source_cap=True,
        include_rear_cap=True,
        include_curvature=False,
    )
    geometry = api.build_preview_geometry(_OSSE, options)
    assert geometry.metadata["api_version"] == "hornlab.preview/1"
    json.dumps(geometry.metadata, allow_nan=False)


def test_lod_config_overwrites_the_mesh_segment_counts_wg_documents():
    """WG's preview translation relies on these four fields being overwritten."""

    from hornlab_mesher.preview.api import _lod_config

    config = {"mesh": {"angularSegments": 7, "lengthSegments": 5, "wall_thickness_mm": 4.0}}
    result = _lod_config(config, 64, 24)

    assert result["mesh"]["angular_segments"] == 64
    assert result["mesh"]["length_segments"] == 24
    assert "angularSegments" not in result["mesh"]
    assert "lengthSegments" not in result["mesh"]
    assert result["mesh"]["wall_thickness_mm"] == 4.0
    assert config["mesh"]["angularSegments"] == 7  # the caller's config is untouched



# --- root sizing exports ------------------------------------------------------

def test_root_sizing_exports_price_the_imported_mesh_regions():
    """WG's imported-CAD path prices regions through the package root."""

    import hornlab_mesher
    from hornlab_mesher import Region, estimate_mesh_cost, estimate_solve_cost, mesh_sizing

    assert hornlab_mesher.Region is mesh_sizing.Region
    assert hornlab_mesher.estimate_mesh_cost is mesh_sizing.estimate_mesh_cost
    regions = [
        Region(40_000.0, 10.0, label="rigid", role="shadow"),
        Region(500.0, 2.0, label="throat", role="source"),
    ]

    estimate = estimate_mesh_cost(regions).to_dict()

    expected = sum(region.triangle_count() for region in regions)
    assert estimate["n_triangles"] == pytest.approx(expected, abs=1.0)
    assert set(estimate["per_role_triangles"]) == {"rigid", "throat"}
    assert estimate["per_role_valid_f_max_hz"]["throat"] > estimate[
        "per_role_valid_f_max_hz"
    ]["rigid"]
    json.dumps(estimate, allow_nan=False)
    measured = estimate_solve_cost(int(estimate["n_triangles"])).to_dict()
    assert measured["n_triangles"] == int(estimate["n_triangles"])
