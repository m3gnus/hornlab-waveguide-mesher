from __future__ import annotations

import sys
import types

import pytest

from hornlab_mesher.density import (
    configure_density,
    _enclosure_resolution_formula,
    _parse_quadrant_resolutions,
)
from hornlab_mesher.geometry import BuiltGeometry, MeshDensity


def test_quadrant_resolution_parsing_matches_wg_contract():
    assert _parse_quadrant_resolutions("5", 9.0) == [5.0, 5.0, 5.0, 5.0]
    assert _parse_quadrant_resolutions("6,7,8,9", 9.0) == [6.0, 7.0, 8.0, 9.0]
    assert _parse_quadrant_resolutions("6,7", 9.0) == [6.0, 7.0, 9.0, 9.0]
    assert _parse_quadrant_resolutions("", 9.0) == [9.0, 9.0, 9.0, 9.0]


def test_enclosure_resolution_formula_hits_wg_corner_targets():
    formula = _enclosure_resolution_formula(
        [2.0, 3.0, 4.0, 5.0],
        [12.0, 13.0, 14.0, 15.0],
        bx0=-10.0,
        bx1=10.0,
        by0=-20.0,
        by1=20.0,
        z_front=100.0,
        z_back=40.0,
    )

    def eval_formula(x, y, z):
        return float(eval(formula, {"__builtins__": {}}, {"x": x, "y": y, "z": z}))

    assert eval_formula(10.0, 20.0, 100.0) == 2.0
    assert eval_formula(-10.0, 20.0, 100.0) == 3.0
    assert eval_formula(-10.0, -20.0, 100.0) == 4.0
    assert eval_formula(10.0, -20.0, 100.0) == 5.0
    assert eval_formula(10.0, 20.0, 40.0) == 12.0
    assert eval_formula(-10.0, 20.0, 40.0) == 13.0
    assert eval_formula(-10.0, -20.0, 40.0) == 14.0
    assert eval_formula(10.0, -20.0, 40.0) == 15.0


class _FakeField:
    def __init__(self):
        self._next = 1
        self.kinds = {}
        self.strings = {}
        self.numbers = {}
        self.number_lists = {}
        self.background = None

    def add(self, kind):
        tag = self._next
        self._next += 1
        self.kinds[tag] = str(kind)
        return tag

    def setString(self, tag, key, value):
        self.strings[(int(tag), str(key))] = str(value)

    def setNumber(self, tag, key, value):
        self.numbers[(int(tag), str(key))] = float(value)

    def setNumbers(self, tag, key, values):
        self.number_lists[(int(tag), str(key))] = [int(value) for value in values]

    def setAsBackgroundMesh(self, tag):
        self.background = int(tag)


class _FakeModel:
    def __init__(self, bboxes):
        self._bboxes = dict(bboxes)
        self.mesh = types.SimpleNamespace(field=_FakeField())

    def getBoundary(self, dimtags, oriented=False, combined=False):
        return [(1, 1000 + int(tag)) for dim, tag in dimtags if int(dim) == 2]

    def getBoundingBox(self, dim, tag):
        assert int(dim) == 2
        return self._bboxes[int(tag)]


def _restriction_formulas(fake_gmsh):
    field = fake_gmsh.model.mesh.field
    out = {}
    for tag, kind in field.kinds.items():
        if kind != "Restrict":
            continue
        in_field = int(field.numbers[(tag, "InField")])
        surfaces = tuple(field.number_lists.get((tag, "SurfacesList"), ()))
        out[surfaces] = field.strings[(in_field, "F")]
    return out


def test_enclosure_density_refines_panels_and_roundover_without_global_box_fillet_size(monkeypatch):
    fake_gmsh = types.SimpleNamespace(
        model=_FakeModel(
            {
                1: (-20.0, -20.0, 100.0, 20.0, 20.0, 100.0),  # front panel
                2: (-20.0, -20.0, 40.0, 20.0, 20.0, 40.0),  # back panel
                3: (20.0, -20.0, 82.0, 38.0, 20.0, 100.0),  # front roundover
                4: (20.0, -20.0, 40.0, 38.0, 20.0, 58.0),  # back roundover
                5: (38.0, -20.0, 58.0, 38.0, 20.0, 82.0),  # side wall
            }
        ),
        option=types.SimpleNamespace(setNumber=lambda *_args: None),
    )
    monkeypatch.setitem(sys.modules, "gmsh", fake_gmsh)

    geometry = BuiltGeometry(
        surface_groups={},
        axial_bounds_mm=(0.0, 100.0),
        mesh_surface_groups={
            "enclosure": [1, 2, 3, 4, 5],
            "enclosure_edges_front": [3],
            "enclosure_edges_back": [4],
        },
        enclosure_bounds={
            "bx0": -20.0,
            "bx1": 38.0,
            "by0": -20.0,
            "by1": 20.0,
            "z_front": 100.0,
            "z_back": 40.0,
            "clamped_edge": 18.0,
            "edge_depth": 18.0,
        },
    )

    configure_density(
        geometry,
        MeshDensity(
            throat_res_mm=8.0,
            mouth_res_mm=26.0,
            rear_res_mm=30.0,
            max_frequency_hz=5_000.0,
        ),
    )

    formulas = _restriction_formulas(fake_gmsh)

    assert "11.4333333333" in formulas[(1,)]
    assert "11.4333333333" in formulas[(2,)]
    assert formulas[(3,)] == "6"
    assert formulas[(4,)] == "6"
    assert formulas[(1, 2, 3, 4, 5)] != "6"
    assert geometry.metadata == {}


def test_enclosure_density_caps_large_high_frequency_triangle_estimate(monkeypatch):
    fake_gmsh = types.SimpleNamespace(
        model=_FakeModel(
            {
                1: (-150.0, -150.0, 100.0, 150.0, 150.0, 100.0),  # front panel
                2: (-150.0, -150.0, 0.0, 150.0, 150.0, 0.0),  # back panel
                3: (150.0, -150.0, 82.0, 168.0, 150.0, 100.0),  # front roundover
                4: (150.0, -150.0, 0.0, 168.0, 150.0, 18.0),  # back roundover
                5: (168.0, -150.0, 18.0, 168.0, 150.0, 82.0),  # side wall
            }
        ),
        option=types.SimpleNamespace(setNumber=lambda *_args: None),
    )
    monkeypatch.setitem(sys.modules, "gmsh", fake_gmsh)

    geometry = BuiltGeometry(
        surface_groups={},
        axial_bounds_mm=(0.0, 100.0),
        mesh_surface_groups={
            "enclosure": [1, 2, 3, 4, 5],
            "enclosure_edges_front": [3],
            "enclosure_edges_back": [4],
        },
        enclosure_bounds={
            "bx0": -150.0,
            "bx1": 168.0,
            "by0": -150.0,
            "by1": 150.0,
            "z_front": 100.0,
            "z_back": 0.0,
            "clamped_edge": 18.0,
            "edge_depth": 18.0,
        },
    )

    configure_density(
        geometry,
        MeshDensity(
            throat_res_mm=30.0,
            mouth_res_mm=30.0,
            rear_res_mm=30.0,
            max_frequency_hz=20_000.0,
        ),
    )

    metadata = geometry.metadata
    assert metadata["enclosureMeshCapped"] is True
    assert metadata["enclosureMeshTriangleCeiling"] == 18_000
    assert metadata["enclosureMeshTriangleEstimatePre"] > 18_000
    assert metadata["enclosureMeshTriangleEstimatePost"] == pytest.approx(18_000, abs=2)
    assert metadata["enclosureMeshCapScale"] > 1.0

    raw_panel_target = 343_000.0 / (6.0 * 20_000.0)
    expected_capped_target = raw_panel_target * metadata["enclosureMeshCapScale"]
    formulas = _restriction_formulas(fake_gmsh)

    front_panel_value = float(
        eval(
            formulas[(1,)],
            {"__builtins__": {}},
            {"x": 0.0, "y": 0.0, "z": 100.0},
        )
    )
    assert front_panel_value == pytest.approx(expected_capped_target)
    assert float(formulas[(3,)]) == pytest.approx(expected_capped_target)
