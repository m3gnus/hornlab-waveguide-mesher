from __future__ import annotations

import sys
import types

import pytest

from hornlab_mesher.density import (
    configure_density,
    _enclosure_resolution_formula,
    _parse_quadrant_resolutions,
)
from hornlab_mesher.geometry import BuiltGeometry, MeshDensity, validate_mesh_density


def test_quadrant_resolution_parsing_matches_wg_contract():
    assert _parse_quadrant_resolutions("5", 9.0) == [5.0, 5.0, 5.0, 5.0]
    assert _parse_quadrant_resolutions("6,7,8,9", 9.0) == [6.0, 7.0, 8.0, 9.0]
    assert _parse_quadrant_resolutions("6,7", 9.0) == [6.0, 7.0, 9.0, 9.0]
    assert _parse_quadrant_resolutions("", 9.0) == [9.0, 9.0, 9.0, 9.0]


@pytest.mark.parametrize(
    "density",
    (
        MeshDensity(throat_res_mm=0.0),
        MeshDensity(mouth_res_mm=float("nan")),
        MeshDensity(rear_res_mm=-1.0),
        MeshDensity(aperture_res_scale=0.5),
        MeshDensity(max_triangles=0),
    ),
)
def test_mesh_density_validation_rejects_invalid_controls(density):
    with pytest.raises(ValueError):
        validate_mesh_density(density)


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
    def __init__(self, bboxes, masses=None):
        self._bboxes = dict(bboxes)
        self._masses = dict(masses or {})
        self.mesh = types.SimpleNamespace(field=_FakeField())

    def getBoundary(self, dimtags, oriented=False, combined=False):
        return [(1, 1000 + int(tag)) for dim, tag in dimtags if int(dim) == 2]

    def getBoundingBox(self, dim, tag):
        assert int(dim) == 2
        return self._bboxes[int(tag)]

    def getMass(self, dim, tag):
        assert int(dim) == 2
        return self._masses[int(tag)]


def _restriction_formulas(fake_gmsh):
    field = fake_gmsh.model.mesh.field
    out = {}
    for tag, kind in field.kinds.items():
        if kind != "Restrict":
            continue
        in_field = int(field.numbers[(tag, "InField")])
        formula = field.strings.get((in_field, "F"))
        if formula is None:
            # Not a MathEval restriction (e.g. the roundover-seam distance
            # grading wraps a Threshold field); formula assertions only
            # target the MathEval size formulas.
            continue
        surfaces = tuple(field.number_lists.get((tag, "SurfacesList"), ()))
        out[surfaces] = formula
    return out


def _fake_panel_enclosure_geometry(*, symmetry_snap_axes=()):
    return BuiltGeometry(
        surface_groups={},
        axial_bounds_mm=(0.0, 100.0),
        mesh_surface_groups={"enclosure": [1, 2]},
        enclosure_bounds={
            "bx0": -150.0,
            "bx1": 150.0,
            "by0": -150.0,
            "by1": 150.0,
            "z_front": 100.0,
            "z_back": 0.0,
        },
        symmetry_snap_axes=tuple(symmetry_snap_axes),
    )


def test_aperture_density_coarsens_surface_not_rim_curves(monkeypatch):
    fake_gmsh = types.SimpleNamespace(
        model=_FakeModel(
            {
                1: (-20.0, -20.0, -80.0, 20.0, 20.0, 0.0),
                2: (-20.0, -20.0, 0.0, 20.0, 20.0, 0.0),
            }
        ),
        option=types.SimpleNamespace(setNumber=lambda *_args: None),
    )
    monkeypatch.setitem(sys.modules, "gmsh", fake_gmsh)

    geometry = BuiltGeometry(
        surface_groups={},
        axial_bounds_mm=(-80.0, 0.0),
        mesh_surface_groups={
            "inner": [1],
            "mouth_aperture": [2],
        },
    )

    configure_density(
        geometry,
        MeshDensity(
            throat_res_mm=5.0,
            mouth_res_mm=10.0,
            rear_res_mm=30.0,
            aperture_res_scale=2.5,
        ),
    )

    formulas = _restriction_formulas(fake_gmsh)
    assert formulas[(2,)] == "25"
    assert geometry.metadata["apertureMeshResolutionScale"] == pytest.approx(2.5)
    assert geometry.metadata["apertureMeshRimSizeMm"] == pytest.approx(10.0)
    assert geometry.metadata["apertureMeshInteriorSizeMm"] == pytest.approx(25.0)

    field = fake_gmsh.model.mesh.field
    aperture_restrict = next(
        tag
        for tag, kind in field.kinds.items()
        if kind == "Restrict"
        and tuple(field.number_lists.get((tag, "SurfacesList"), ())) == (2,)
    )
    assert (aperture_restrict, "CurvesList") not in field.number_lists


def test_enclosure_density_uses_only_user_mm_targets(monkeypatch):
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
            enc_front_res_mm=40.0,
            enc_back_res_mm=50.0,
            allow_large_mesh=True,
        ),
    )

    formulas = _restriction_formulas(fake_gmsh)

    assert float(
        eval(formulas[(1,)], {"__builtins__": {}}, {"x": 0, "y": 0, "z": 100})
    ) == pytest.approx(40.0)
    assert float(
        eval(formulas[(2,)], {"__builtins__": {}}, {"x": 0, "y": 0, "z": 40})
    ) == pytest.approx(50.0)
    assert formulas[(3,)] == "40"
    assert formulas[(4,)] == "50"
    assert formulas[(1, 2, 3, 4, 5)] not in {"40", "50"}
    assert geometry.metadata["meshTriangleEstimate"] > 0


def test_large_triangle_estimate_refuses_instead_of_rewriting_sizes(monkeypatch):
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

    density = MeshDensity(
        throat_res_mm=2.0,
        mouth_res_mm=2.0,
        rear_res_mm=2.0,
        enc_front_res_mm=2.0,
        enc_back_res_mm=2.0,
    )
    with pytest.raises(
        ValueError, match=r"estimated mesh size .*largest estimated contribution is"
    ):
        configure_density(geometry, density)
    assert geometry.metadata["meshTriangleEstimate"] > 27_000
    assert geometry.metadata["meshTriangleDominantRegion"] == "front enclosure panel"
    assert geometry.metadata["meshTriangleDominantTargetMm"] == pytest.approx(2.0)
    assert "enclosureMeshCapped" not in geometry.metadata

    allowed_geometry = BuiltGeometry(
        surface_groups={},
        axial_bounds_mm=geometry.axial_bounds_mm,
        mesh_surface_groups=geometry.mesh_surface_groups,
        enclosure_bounds=geometry.enclosure_bounds,
    )
    configure_density(
        allowed_geometry,
        MeshDensity(
            throat_res_mm=2.0,
            mouth_res_mm=2.0,
            rear_res_mm=2.0,
            enc_front_res_mm=2.0,
            enc_back_res_mm=2.0,
            allow_large_mesh=True,
        ),
    )
    formulas = _restriction_formulas(fake_gmsh)
    assert float(formulas[(3,)]) == pytest.approx(2.0)
    assert float(
        eval(
            formulas[(1,)],
            {"__builtins__": {}},
            {"x": 0.0, "y": 0.0, "z": 100.0},
        )
    ) == pytest.approx(2.0)
    assert "enclosureMeshCapped" not in allowed_geometry.metadata


def test_global_gmsh_options_are_defensive_not_hidden_refinement(monkeypatch):
    option_values: dict[str, float] = {}
    fake_gmsh = types.SimpleNamespace(
        model=_FakeModel(
            {
                1: (-150.0, -150.0, 100.0, 150.0, 150.0, 100.0),
                2: (-150.0, -150.0, 0.0, 150.0, 150.0, 0.0),
            },
            masses={1: 90_000.0, 2: 90_000.0},
        ),
        option=types.SimpleNamespace(
            setNumber=lambda name, value: option_values.__setitem__(
                str(name), float(value)
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "gmsh", fake_gmsh)
    geometry = BuiltGeometry(
        surface_groups={},
        axial_bounds_mm=(0.0, 100.0),
        mesh_surface_groups={"inner": [1], "throat_disc": [2]},
    )

    configure_density(
        geometry,
        MeshDensity(
            throat_res_mm=5.0,
            mouth_res_mm=20.0,
            rear_res_mm=30.0,
            allow_large_mesh=True,
        ),
    )
    assert option_values["Mesh.MeshSizeMin"] == pytest.approx(1.25)
    assert option_values["Mesh.MeshSizeMax"] == pytest.approx(30.0)
    assert option_values["Mesh.MeshSizeFromPoints"] == 0.0
    assert option_values["Mesh.MeshSizeFromCurvature"] == 0.0
    assert option_values["Mesh.MeshSizeExtendFromBoundary"] == 0.0


def test_triangle_estimate_and_limit_are_quadrant_consistent(monkeypatch):
    density = MeshDensity(
        throat_res_mm=30.0,
        mouth_res_mm=30.0,
        rear_res_mm=30.0,
        allow_large_mesh=True,
    )

    full_gmsh = types.SimpleNamespace(
        model=_FakeModel(
            {
                1: (-150.0, -150.0, 100.0, 150.0, 150.0, 100.0),
                2: (-150.0, -150.0, 0.0, 150.0, 150.0, 0.0),
            },
            masses={1: 90_000.0, 2: 90_000.0},
        ),
        option=types.SimpleNamespace(setNumber=lambda *_args: None),
    )
    monkeypatch.setitem(sys.modules, "gmsh", full_gmsh)
    full_geometry = _fake_panel_enclosure_geometry()
    configure_density(full_geometry, density)

    quarter_gmsh = types.SimpleNamespace(
        model=_FakeModel(
            {
                1: (0.0, 0.0, 100.0, 150.0, 150.0, 100.0),
                2: (0.0, 0.0, 0.0, 150.0, 150.0, 0.0),
            },
            masses={1: 22_500.0, 2: 22_500.0},
        ),
        option=types.SimpleNamespace(setNumber=lambda *_args: None),
    )
    monkeypatch.setitem(sys.modules, "gmsh", quarter_gmsh)
    quarter_geometry = _fake_panel_enclosure_geometry(symmetry_snap_axes=("x", "y"))
    configure_density(quarter_geometry, density)

    full = full_geometry.metadata
    quarter = quarter_geometry.metadata
    assert full["meshEffectiveTriangleLimit"] == 18_000
    assert quarter["meshEffectiveTriangleLimit"] == 4_500
    assert quarter["meshTriangleEstimateFullDomain"] == pytest.approx(
        full["meshTriangleEstimateFullDomain"],
        abs=4,
    )

    full_formula = _restriction_formulas(full_gmsh)[(1,)]
    quarter_formula = _restriction_formulas(quarter_gmsh)[(1,)]
    full_panel_value = float(
        eval(full_formula, {"__builtins__": {}}, {"x": 0.0, "y": 0.0, "z": 100.0})
    )
    quarter_panel_value = float(
        eval(quarter_formula, {"__builtins__": {}}, {"x": 75.0, "y": 75.0, "z": 100.0})
    )
    assert quarter_panel_value == pytest.approx(full_panel_value)
