"""Cap-loop detection on OCC boundary curves.

``extreme_boundary_loop_curves`` decides which boundary curves form the planar
cap that closes a shell at its throat or its rear. It takes its extreme over
*axis-planar* curves only: a wall-running curve that overshoots the cap plane by
any amount at all would otherwise set the target, and then no curve matches it,
so the search returns no loop, hence no cap, hence a mesh that fails the
closed-shell contract.

Every build calls this, including the default ``approximate`` surface fit, so
these also pin that the axis-planar restriction leaves the default path's output
exactly where it was.
"""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager

import gmsh
import numpy as np
import pytest

from hornlab_mesher.builders import _occ as occ_module
from hornlab_mesher.builders import point_grid_sources as sources_module
from hornlab_mesher.builders._occ import extreme_boundary_loop_curves
from hornlab_mesher.geometry import (
    HornEnclosure,
    MeshDensity,
    PointGridHornGeometry,
)
from hornlab_mesher.mesher import build_mesh


@contextmanager
def _gmsh_session(name: str):
    initialized_here = False
    if not gmsh.isInitialized():
        gmsh.initialize()
        initialized_here = True
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.add(name)
        yield gmsh
    finally:
        gmsh.clear()
        if initialized_here:
            gmsh.finalize()


def _plane_surface_through(points: list[tuple[float, float, float]]) -> list[tuple[int, int]]:
    """One OCC plane surface through ``points``, closed back to the first."""

    tags = [gmsh.model.occ.addPoint(x, y, z) for x, y, z in points]
    lines = [
        gmsh.model.occ.addLine(tags[i], tags[(i + 1) % len(tags)])
        for i in range(len(tags))
    ]
    loop = gmsh.model.occ.addCurveLoop(lines)
    surface = int(gmsh.model.occ.addPlaneSurface([loop]))
    gmsh.model.occ.synchronize()
    return [(2, surface)]


#: ``getBoundingBox`` pads its answer by about 1e-7, so an exactly planar curve
#: reports a nonzero span. That padding is also why ``eps`` in the production
#: selector carries a 1e-6 floor rather than trusting the axial span alone.
_BBOX_PAD = 1.0e-6


def _curve_axial_bounds(curve_tag: int) -> tuple[float, float]:
    box = gmsh.model.getBoundingBox(1, curve_tag)
    return float(min(box[2], box[5])), float(max(box[2], box[5]))


# ---------------------------------------------------------------------------
# The pre-change selection, kept as an executable reference
# ---------------------------------------------------------------------------


def _whole_boundary_extreme_loop_curves(
    dimtags: list[tuple[int, int]],
    *,
    source_axis: str = "z",
    use_min: bool = True,
) -> list[int]:
    """The selection this module used before it restricted itself to flat curves.

    Identical to ``extreme_boundary_loop_curves`` except that the target comes
    from the extreme of *every* boundary curve rather than of the axis-planar
    ones. Kept here so the default-path tests below can assert the two agree
    rather than merely assert the new one returns something plausible.
    """

    boundary = gmsh.model.getBoundary(dimtags, oriented=False, combined=False)
    curve_tags: list[int] = []
    seen: set[int] = set()
    for dim, tag in boundary:
        if int(dim) != 1:
            continue
        curve_tag = int(tag)
        if curve_tag in seen:
            continue
        seen.add(curve_tag)
        curve_tags.append(curve_tag)
    if not curve_tags:
        return []

    axis_idx = {"x": 0, "y": 1, "z": 2}.get(source_axis, 2)
    bounds: dict[int, tuple[float, float]] = {}
    lo_all = float("inf")
    hi_all = float("-inf")
    for curve_tag in curve_tags:
        box = gmsh.model.getBoundingBox(1, curve_tag)
        lo = float(min(box[axis_idx], box[axis_idx + 3]))
        hi = float(max(box[axis_idx], box[axis_idx + 3]))
        bounds[curve_tag] = (lo, hi)
        lo_all = min(lo_all, lo)
        hi_all = max(hi_all, hi)
    if not math.isfinite(lo_all):
        return []

    target = lo_all if use_min else hi_all
    eps = max(1e-6, abs(hi_all - lo_all) * 1e-3)
    return [
        curve_tag
        for curve_tag, (lo, hi) in bounds.items()
        if abs(lo - target) <= eps and abs(hi - target) <= eps
    ]


# ---------------------------------------------------------------------------
# Axis-planar selection
# ---------------------------------------------------------------------------


def test_an_overshooting_wall_curve_no_longer_hides_the_cap_loop():
    """The regression the axis-planar restriction exists to prevent.

    The z = 0 rim is the only curve that can bound a planar cap, but a wall
    curve dips to z = -1. Taking the extreme over the whole boundary aims at
    z = -1, which no curve lies flat on, so the search fails closed and the
    shell is left with no cap at all.
    """

    with _gmsh_session("overshooting-wall-curve"):
        surface = _plane_surface_through(
            [
                (0.0, 0.0, 0.0),
                (10.0, 0.0, 0.0),  # rim, flat at z = 0
                (12.0, 0.0, -1.0),  # wall curve overshooting below the rim
                (12.0, 0.0, 20.0),
                (0.0, 0.0, 20.0),
            ]
        )

        selected = extreme_boundary_loop_curves(surface, use_min=True)
        assert len(selected) == 1
        lo, hi = _curve_axial_bounds(selected[0])
        assert lo == pytest.approx(0.0, abs=_BBOX_PAD)
        assert hi == pytest.approx(0.0, abs=_BBOX_PAD)

        assert _whole_boundary_extreme_loop_curves(surface, use_min=True) == []


def test_use_max_takes_the_highest_axis_planar_curve():
    with _gmsh_session("highest-axis-planar-curve"):
        surface = _plane_surface_through(
            [
                (0.0, 0.0, 0.0),
                (10.0, 0.0, 0.0),
                (12.0, 0.0, 21.0),  # wall curve overshooting above the mouth rim
                (12.0, 0.0, 20.0),
                (0.0, 0.0, 20.0),  # mouth rim, flat at z = 20
            ]
        )

        selected = extreme_boundary_loop_curves(surface, use_min=False)
        assert len(selected) == 1
        lo, hi = _curve_axial_bounds(selected[0])
        assert lo == pytest.approx(20.0, abs=_BBOX_PAD)
        assert hi == pytest.approx(20.0, abs=_BBOX_PAD)

        assert _whole_boundary_extreme_loop_curves(surface, use_min=False) == []


def test_no_axis_planar_curve_yields_no_loop():
    """The fallback branch, which is a no-op by construction.

    Nothing here can bound a planar cap. With ``use_min`` the fallback target is
    the minimum ``lo`` over every curve, so any curve it accepts has
    ``hi - target <= eps`` and therefore ``hi - lo <= eps`` -- which would have
    put it in ``flat`` and taken the other branch. The fallback can only ever
    return an empty loop, i.e. exactly the fail-closed behaviour that preceded
    the axis-planar restriction. It must do that rather than raise, and rather
    than invent a loop out of slanted curves.
    """

    with _gmsh_session("no-axis-planar-curve"):
        surface = _plane_surface_through(
            [(0.0, 0.0, 0.0), (10.0, 0.0, 5.0), (5.0, 0.0, 20.0)]
        )

        assert extreme_boundary_loop_curves(surface, use_min=True) == []
        assert extreme_boundary_loop_curves(surface, use_min=False) == []
        # Same answer either way, so nothing downstream can tell the fallback
        # from the pre-change selection.
        assert _whole_boundary_extreme_loop_curves(surface, use_min=True) == []
        assert _whole_boundary_extreme_loop_curves(surface, use_min=False) == []


def test_a_surface_flat_in_the_axis_returns_its_whole_boundary():
    """A degenerate span must not collapse the tolerance to zero.

    Every curve of a z = 0 disc is axis-planar, so all of them belong to the
    loop. The 1e-6 floor under ``eps`` is what keeps this from depending on
    exact float equality.
    """

    with _gmsh_session("flat-in-axis"):
        surface = _plane_surface_through(
            [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 8.0, 0.0), (0.0, 8.0, 0.0)]
        )

        selected = extreme_boundary_loop_curves(surface, use_min=True)
        assert len(selected) == 4
        assert extreme_boundary_loop_curves(surface, use_min=False) == selected


def test_selection_is_taken_on_the_named_axis():
    """``source_axis`` picks the axis, so an x-flat rim is found on x."""

    with _gmsh_session("named-axis"):
        surface = _plane_surface_through(
            [
                (0.0, 0.0, 0.0),
                (0.0, 10.0, 0.0),  # rim, flat at x = 0
                (-1.0, 12.0, 0.0),  # wall curve overshooting below the rim
                (20.0, 12.0, 0.0),
                (20.0, 0.0, 0.0),
            ]
        )

        selected = extreme_boundary_loop_curves(surface, source_axis="x", use_min=True)
        assert len(selected) == 1
        box = gmsh.model.getBoundingBox(1, selected[0])
        assert float(box[0]) == pytest.approx(0.0, abs=_BBOX_PAD)
        assert float(box[3]) == pytest.approx(0.0, abs=_BBOX_PAD)


# ---------------------------------------------------------------------------
# The default build path is unchanged
# ---------------------------------------------------------------------------


def _horn_grid(n_phi: int = 48, n_len: int = 20, *, closed: bool = True) -> np.ndarray:
    span = 2.0 * np.pi if closed else 0.5 * np.pi
    phi = (
        np.linspace(0.0, span, n_phi, endpoint=False)
        if closed
        else np.linspace(0.0, span, n_phi)
    )
    z = np.linspace(0.0, 60.0, n_len)
    radius = 12.7 + 0.9 * z
    grid = np.empty((len(phi), n_len, 3), dtype=np.float64)
    for i, angle in enumerate(phi):
        grid[i, :, 0] = radius * np.cos(angle)
        grid[i, :, 1] = radius * np.sin(angle)
        grid[i, :, 2] = z
    return grid


def _outer_grid(inner: np.ndarray, thickness: float = 5.0) -> np.ndarray:
    outer = np.array(inner, dtype=np.float64, copy=True)
    radial = outer[:, :, :2]
    norms = np.linalg.norm(radial, axis=-1, keepdims=True)
    outer[:, :, :2] = radial * (1.0 + thickness / np.maximum(norms, 1.0e-9))
    return outer


def _default_path_cases() -> dict[str, PointGridHornGeometry]:
    inner = _horn_grid()
    return {
        "freestanding": PointGridHornGeometry(
            inner_points=inner, outer_points=_outer_grid(inner)
        ),
        "enclosure": PointGridHornGeometry(
            inner_points=inner,
            enclosure=HornEnclosure(
                depth_mm=140.0,
                space_l_mm=20.0,
                space_t_mm=20.0,
                space_r_mm=20.0,
                space_b_mm=20.0,
                edge_mm=8.0,
                edge_type=1,
                plan_type=1,
                depth_margin_mm=5.0,
            ),
        ),
        "bare": PointGridHornGeometry(inner_points=inner, wall_thickness_mm=0.0),
        "infinite-baffle": PointGridHornGeometry(
            inner_points=inner, infinite_baffle=True
        ),
        "quarter-freestanding": PointGridHornGeometry(
            inner_points=_horn_grid(n_phi=16, closed=False),
            outer_points=_outer_grid(_horn_grid(n_phi=16, closed=False)),
            closed=False,
        ),
    }


_DENSITY = MeshDensity(throat_res_mm=6.0, mouth_res_mm=18.0, rear_res_mm=20.0)


def test_default_path_selects_the_same_curves_as_the_whole_boundary_extreme(
    tmp_path, monkeypatch
):
    """Every call a default build makes must land on the same curves as before.

    The default ``approximate`` fit did not ask for this change, so if any real
    call disagreed the change would be moving meshes nobody opted into.
    """

    calls: list[tuple[list[int], list[int]]] = []
    real = occ_module.extreme_boundary_loop_curves

    def recording(dimtags, *, source_axis="z", use_min=True):
        new = real(dimtags, source_axis=source_axis, use_min=use_min)
        old = _whole_boundary_extreme_loop_curves(
            dimtags, source_axis=source_axis, use_min=use_min
        )
        calls.append((sorted(new), sorted(old)))
        return new

    monkeypatch.setattr(occ_module, "extreme_boundary_loop_curves", recording)
    monkeypatch.setattr(sources_module, "extreme_boundary_loop_curves", recording)

    for label, geometry in _default_path_cases().items():
        build_mesh(geometry, _DENSITY, tmp_path / f"{label}.msh")

    assert calls, "no build reached the cap-loop search"
    disagreements = [(new, old) for new, old in calls if new != old]
    assert not disagreements, disagreements


@pytest.mark.parametrize("label", ["freestanding", "bare"])
def test_default_path_mesh_is_byte_identical_under_either_selection(
    label, tmp_path, monkeypatch
):
    """Stronger than curve equality: the written mesh does not move a byte."""

    geometry = _default_path_cases()[label]
    shipped = build_mesh(geometry, _DENSITY, tmp_path / f"{label}-shipped.msh")

    monkeypatch.setattr(
        occ_module,
        "extreme_boundary_loop_curves",
        _whole_boundary_extreme_loop_curves,
    )
    monkeypatch.setattr(
        sources_module,
        "extreme_boundary_loop_curves",
        _whole_boundary_extreme_loop_curves,
    )
    previous = build_mesh(geometry, _DENSITY, tmp_path / f"{label}-previous.msh")

    assert hashlib.sha256(shipped.read_bytes()).hexdigest() == hashlib.sha256(
        previous.read_bytes()
    ).hexdigest()
