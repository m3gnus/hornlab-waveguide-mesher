"""A mouth ring in the throat plane must not be taken as part of the throat loop.

An R-OSSE with ``b = 1`` rolls back until its mouth ring lies in the throat
plane. Both rings are flat and both sit on the extreme plane, so the boundary
search returned them together as one loop and OCC failed with "Could not
create wire" for the full solve mesh and for every STEP export (STEP reopens a
reduced design to the full model).
"""

from __future__ import annotations

import copy
import math

import gmsh
import pytest

from hornlab_mesher import build_from_config, write_step_from_config
from hornlab_mesher.builders._occ import extreme_boundary_loop_curves
from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.profile_formulas import calculate_rosse

COPLANAR_ROSSE = {
    "formula": "R-OSSE",
    "mode": "freestanding",
    "profile": {
        "R_mm": 150.0,
        "r0_mm": 12.7,
        "a_deg": 45,
        "a0_deg": 10,
        "k": 1.0,
        "r": 0.4,
        "m": 0.8,
        "b": 1.0,
        "q": 3.4,
    },
    "mesh": {
        "angular_segments": 32,
        "length_segments": 24,
        "wall_thickness_mm": 5.0,
        "throat_res_mm": 6.0,
        "mouth_res_mm": 20.0,
        "rear_res_mm": 25.0,
    },
}


def _config(quadrants: str, mode: str = "freestanding") -> dict:
    config = copy.deepcopy(COPLANAR_ROSSE)
    config["mode"] = mode
    config["mesh"]["quadrants"] = quadrants
    if mode == "bare":
        del config["mesh"]["wall_thickness_mm"]
    return config


def test_fixture_mouth_lies_in_the_throat_plane() -> None:
    params = build_geometry_params(_config("1234"))[0]
    z_mouth, r_mouth = calculate_rosse(1.0, 0.0, params)
    assert z_mouth == pytest.approx(0.0, abs=1e-9)
    assert r_mouth == pytest.approx(150.0)


@pytest.mark.parametrize("quadrants", ["1234", "14", "1"])
def test_coplanar_freestanding_horn_meshes(quadrants: str, tmp_path) -> None:
    result = build_from_config(_config(quadrants), str(tmp_path / "horn.msh"))
    assert result is not None
    assert (tmp_path / "horn.msh").stat().st_size > 0


@pytest.mark.parametrize("quadrants", ["1234", "14", "1"])
@pytest.mark.parametrize("mode", ["freestanding", "bare"])
def test_coplanar_horn_exports_step(quadrants: str, mode: str, tmp_path) -> None:
    path, info = write_step_from_config(_config(quadrants, mode), str(tmp_path / "horn.step"))
    assert info.body == ("solid" if mode == "freestanding" else "surface")


def _ring(radius: float, z: float) -> list[int]:
    """A circle as four arcs whose end points are separate, coincident entities."""
    occ = gmsh.model.occ
    curves = []
    for quarter in range(4):
        start, end = quarter * math.pi / 2, (quarter + 1) * math.pi / 2
        curves.append(occ.addCircle(0.0, 0.0, z, radius, angle1=start, angle2=end))
    return curves


@pytest.fixture
def gmsh_model():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("rings")
    yield gmsh.model
    gmsh.finalize()


def _selected(curves: list[int], *, use_min: bool) -> list[int]:
    gmsh.model.occ.synchronize()
    # A wire per ring gives the search a boundary to read without fusing the
    # rings' end points.
    surfaces = []
    for ring in (curves[i : i + 4] for i in range(0, len(curves), 4)):
        loop = gmsh.model.occ.addCurveLoop(ring)
        surfaces.append((2, gmsh.model.occ.addPlaneSurface([loop])))
    gmsh.model.occ.synchronize()
    return sorted(extreme_boundary_loop_curves(surfaces, source_axis="z", use_min=use_min))


def _reach(curve: int) -> float:
    box = gmsh.model.getBoundingBox(1, curve)
    return max(abs(box[0]), abs(box[1]), abs(box[3]), abs(box[4]))


def test_two_rings_in_the_extreme_plane_yield_one_ring(gmsh_model) -> None:
    _ring(12.7, 0.0)
    _ring(150.0, 0.0)
    inner = _selected(list(range(1, 9)), use_min=True)
    assert len(inner) == 4
    assert all(_reach(curve) == pytest.approx(12.7, abs=1e-6) for curve in inner)


def test_max_plane_keeps_the_outer_ring(gmsh_model) -> None:
    _ring(12.7, 0.0)
    _ring(150.0, 0.0)
    outer = _selected(list(range(1, 9)), use_min=False)
    assert len(outer) == 4
    assert all(_reach(curve) == pytest.approx(150.0, abs=1e-6) for curve in outer)


@pytest.mark.parametrize("use_min", [True, False])
def test_a_single_ring_is_returned_whole(gmsh_model, use_min: bool) -> None:
    _ring(12.7, 0.0)
    _ring(150.0, 40.0)
    assert len(_selected(list(range(1, 9)), use_min=use_min)) == 4
