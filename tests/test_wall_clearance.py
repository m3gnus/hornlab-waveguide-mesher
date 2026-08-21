"""The free-standing outer wall must not chord through the acoustic surface.

A flat facet on a curved shell sits inside the surface it approximates. On a
free-standing waveguide the whole outer wall is meshed at one rear resolution,
so near the throat -- where the shell is only tens of millimetres from the axis
-- a coarse facet departs from the shell by more than a thin wall is thick, and
the rear shell comes out the other side of the bore. The mesh is watertight and
manifold throughout; only the geometry is impossible.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hornlab_mesher.density import (
    _WALL_CLEARANCE_FRACTION,
    _WALL_CLEARANCE_SIZE_OVERSHOOT,
    _wall_clearance_axial_ramp,
    _wall_clearance_chord_mm,
    _wall_clearance_size_formula,
)


def _flaring_shell() -> tuple[np.ndarray, np.ndarray]:
    """Radii and axial positions shaped like an R-OSSE outer shell.

    A nearly cylindrical throat extension, then a flare, then a mouth roundover
    that turns back on itself in z.
    """

    axial = np.asarray(
        [-0.5, 2.4, 5.2, 10.7, 25.0, 50.0, 100.0, 200.0, 283.0, 340.0, 374.0, 330.0]
    )
    radius = np.asarray(
        [23.4, 23.6, 23.9, 24.4, 30.0, 45.0, 90.0, 200.0, 330.0, 430.0, 480.0, 595.0]
    )
    return radius, axial


def _ramp_values(radius, axial, wall_mm):
    base, slope, intercept = _wall_clearance_axial_ramp(
        radius, axial, wall_mm=wall_mm
    )
    return np.maximum(base, intercept + slope * np.asarray(axial, dtype=float))


def test_chord_bound_is_the_exact_sagitta_relation() -> None:
    """``h = 2 sqrt(2 R d - d^2)``, not the small-angle approximation."""

    wall, radius = 5.0, 23.4
    sagitta = _WALL_CLEARANCE_FRACTION * wall
    expected = (
        2.0
        * math.sqrt(2.0 * radius * sagitta - sagitta * sagitta)
        / _WALL_CLEARANCE_SIZE_OVERSHOOT
    )
    assert float(_wall_clearance_chord_mm(radius, wall)) == pytest.approx(expected)
    # The small-angle form sqrt(8 R d) is close but always larger, so using it
    # would let a slightly-too-coarse facet through.
    assert expected < math.sqrt(8.0 * radius * sagitta)


def test_chord_bound_grows_with_radius_and_wall() -> None:
    radii = np.asarray([10.0, 25.0, 100.0, 600.0])
    bound = _wall_clearance_chord_mm(radii, 5.0)
    assert np.all(np.diff(bound) > 0.0)
    assert np.all(_wall_clearance_chord_mm(radii, 10.0) > bound)
    # A degenerate radius must not produce a negative or complex chord.
    assert float(_wall_clearance_chord_mm(0.0, 5.0)) == 0.0


@pytest.mark.parametrize("wall_mm", [1.0, 5.0, 10.0, 25.0])
def test_ramp_never_exceeds_the_bound_it_approximates(wall_mm) -> None:
    radius, axial = _flaring_shell()
    values = _ramp_values(radius, axial, wall_mm)
    assert np.all(values <= _wall_clearance_chord_mm(radius, wall_mm) + 1.0e-9)


@pytest.mark.parametrize(
    "radius, axial",
    [
        (np.asarray([20.0]), np.asarray([0.0])),
        (np.asarray([20.0, 20.0]), np.asarray([0.0, 50.0])),
        (np.asarray([20.0, 20.0]), np.asarray([0.0, 0.0])),
        (np.asarray([30.0, 20.0]), np.asarray([0.0, 50.0])),
    ],
    ids=["single ring", "cylindrical", "zero axial span", "narrowing"],
)
def test_degenerate_shells_fall_back_to_a_flat_bound(radius, axial) -> None:
    base, slope, intercept = _wall_clearance_axial_ramp(radius, axial, wall_mm=5.0)
    values = np.maximum(base, intercept + slope * axial)
    assert np.all(np.isfinite(values))
    assert np.all(values <= _wall_clearance_chord_mm(radius, 5.0) + 1.0e-9)


def test_ramp_beats_a_flat_bound_on_a_flaring_shell() -> None:
    """The whole point of a ramp: do not hold the mouth to the throat's size.

    A shell that flares from 23 mm to 595 mm may carry far coarser facets at its
    mouth than at its throat, and pinning the lot at the throat's bound is what
    makes a safe mesh unaffordable.
    """

    radius, axial = _flaring_shell()
    base, slope, intercept = _wall_clearance_axial_ramp(radius, axial, wall_mm=5.0)
    assert slope > 0.0
    at_mouth = intercept + slope * 283.0
    assert at_mouth > 3.0 * base


def test_a_mouth_that_turns_back_does_not_pin_the_shell() -> None:
    """The cheapest admissible line, not simply the last hull edge.

    An R-OSSE mouth roundover reaches its maximum z and then curls back, so the
    largest-z ring is not the largest-radius ring. The lower hull of (z, bound)
    then ends on a near-vertical edge, and extending that one holds the entire
    shell at the throat's bound -- costing more than twice the triangles while
    looking, from the formula alone, like a generous ramp.
    """

    radius, axial = _flaring_shell()
    values = _ramp_values(radius, axial, 5.0)
    base = float(np.min(_wall_clearance_chord_mm(radius, 5.0)))
    # Rings out at the flare must be allowed to grow well past the floor.
    assert float(np.max(values)) > 3.0 * base


def test_formula_is_parseable_by_gmsh_and_matches_the_fit() -> None:
    gmsh = pytest.importorskip("gmsh")
    radius, axial = _flaring_shell()
    base, slope, intercept = _wall_clearance_axial_ramp(radius, axial, wall_mm=5.0)
    formula = _wall_clearance_size_formula(
        "z",
        rear_res_mm=40.0,
        base_mm=base,
        slope_per_mm=slope,
        intercept_mm=intercept,
    )
    # A negative intercept must not be emitted as "z - -1081", which the
    # MathEval parser rejects outright.
    assert "- -" not in formula
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("wall-clearance-formula")
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
        gmsh.model.occ.synchronize()
        field = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(field, "F", formula)
        gmsh.model.mesh.field.setAsBackgroundMesh(field)
        gmsh.model.mesh.generate(2)
    finally:
        gmsh.finalize()


def test_no_wall_means_no_cap() -> None:
    """Every other builder must keep the sizing it has always had."""

    assert (
        _wall_clearance_size_formula(
            "z", rear_res_mm=40.0, base_mm=40.0, slope_per_mm=0.0, intercept_mm=40.0
        )
        == "min(40, max(40 + (0)*z, 40))"
    )


def test_cap_is_inactive_on_a_shell_that_never_needed_it() -> None:
    """A wide throat with a thick wall must mesh exactly as before."""

    radius = np.asarray([200.0, 300.0, 600.0])
    axial = np.asarray([0.0, 50.0, 200.0])
    base, slope, intercept = _wall_clearance_axial_ramp(radius, axial, wall_mm=20.0)
    values = np.maximum(base, intercept + slope * axial)
    # Every bound is already coarser than a 40 mm rear resolution, so the
    # min() in the formula leaves the requested size untouched.
    assert float(np.min(values)) > 40.0


def _thin_walled_freestanding_config(wall_mm: float) -> dict:
    return {
        "formula": "OSSE",
        "profile": {"L_mm": 120.0, "r0_mm": 12.0, "a_deg": 45.0, "a0_deg": 4.0},
        "mesh": {
            "angular_segments": 32,
            "length_segments": 12,
            "throat_res_mm": 6.0,
            "mouth_res_mm": 20.0,
            # Coarse enough, against a thin wall, to chord through the bore.
            "rear_res_mm": 40.0,
            "wall_thickness_mm": wall_mm,
            "quadrants": 1234,
        },
    }


def test_thin_wall_shell_facets_stay_inside_the_wall(tmp_path) -> None:
    """The build-level regression: a thin wall must not license coarse facets.

    Measured on the mesh rather than on the formula, because Gmsh treats a size
    field as a target and not as a maximum edge length -- the whole reason the
    bound carries an overshoot allowance.
    """

    pytest.importorskip("gmsh")
    meshio = pytest.importorskip("meshio")
    from hornlab_mesher.config_builder import build_from_config

    result = build_from_config(
        _thin_walled_freestanding_config(3.0), tmp_path / "thin.msh"
    )
    clearance = (result.metadata or {}).get("outerWallClearance")
    assert clearance is not None
    assert clearance["capActive"] is True
    assert clearance["cappedSizeAtMinRadiusMm"] < clearance[
        "requestedRearResolutionMm"
    ]

    mesh = meshio.read(str(result.mesh_path))
    points = np.vstack([block.data for block in mesh.cells if block.type == "triangle"])
    corners = np.asarray(mesh.points, dtype=float)[points] * 1000.0
    edges = np.max(
        [
            np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1),
            np.linalg.norm(corners[:, 2] - corners[:, 1], axis=1),
            np.linalg.norm(corners[:, 0] - corners[:, 2], axis=1),
        ],
        axis=0,
    )
    centroid = corners.mean(axis=1)
    radius = np.hypot(centroid[:, 0], centroid[:, 1])
    # Near the throat the shell is tightest and the cap bites hardest. Without
    # it every one of these facets is free to reach the 40 mm rear resolution.
    throat = radius < 2.0 * clearance["minOuterRadiusMm"]
    assert np.any(throat)
    assert float(np.max(edges[throat])) < clearance["requestedRearResolutionMm"]


def test_thick_wall_leaves_the_requested_resolution_alone(tmp_path) -> None:
    pytest.importorskip("gmsh")
    from hornlab_mesher.config_builder import build_from_config

    thin = build_from_config(
        _thin_walled_freestanding_config(3.0), tmp_path / "thin.msh"
    )
    # Well inside the throat's curvature radius: a wall thicker than that
    # folds the offset shell, which is a different defect and would make this
    # comparison rest on degenerate geometry.
    thick = build_from_config(
        _thin_walled_freestanding_config(10.0), tmp_path / "thick.msh"
    )
    # A thicker wall can afford coarser facets, so it must not cost more.
    assert thick.n_triangles < thin.n_triangles
