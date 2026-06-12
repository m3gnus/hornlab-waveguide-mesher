from __future__ import annotations

import math

import pytest

from hornlab_mesher.cli import build_from_config, build_geometry_params, load_config


ROSSE_CFG = """
ABEC.SimType = 2
R-OSSE = {
  R = 160 * (abs(cos(p)/1.8)^3 + abs(sin(p)/1)^4)^(-1/7)
  r = 0.35
  b = 0.4
  m = 0.84
  a = 22
  a0 = 15.5
  r0 = 12.7
  k = 4
  q = 4
}
Mesh = {
  AngularSegments = 80
  LengthSegments = 20
  WallThickness = 6
}
"""

OSSE_CFG = """
ABEC.SimType = 2
OSSE = {
  Length = 80
  Throat.Diameter = 20
  Coverage.Angle = 45 * cos(p)
  Throat.Angle = 10
  OS.k = 1
  Term.n = 4
  Term.q = 0.995
}
MORPH = {
  TargetShape = 1
}
Mesh = {
  AngularSegments = 12
  LengthSegments = 4
  WallThickness = 5
}
"""

OSSE_ENCLOSURE_CFG = """
OSSE = {
  Length = 80
  Throat.Diameter = 20
  Coverage.Angle = 45
  Throat.Angle = 10
  OS.k = 1
  Term.n = 4
  Term.q = 0.995
}
Mesh = {
  AngularSegments = 12
  LengthSegments = 4
  WallThickness = 0
}
Mesh.Enclosure = {
  Depth = 120
  EdgeRadius = 6
}
"""


def test_build_geometry_params_accepts_rosse_alias():
    params, formula, mode = build_geometry_params(
        {
            "formula": "ROSSE",
            "profile": {"R_mm": 130.0, "r0_mm": 12.7},
            "mesh": {"angular_segments": 16, "length_segments": 6},
        }
    )

    assert formula == "R-OSSE"
    assert mode == "freestanding"
    assert params["type"] == "R-OSSE"
    assert params["R"] == 130.0
    assert params["wallThickness"] == 6.0


def test_build_geometry_params_rejects_wrong_formula_profile_keys():
    with pytest.raises(ValueError, match="R-OSSE-only"):
        build_geometry_params({"formula": "OSSE", "profile": {"R_mm": 130.0}})

    with pytest.raises(ValueError, match="OSSE-only"):
        build_geometry_params({"formula": "R-OSSE", "profile": {"L_mm": 120.0}})


def test_build_geometry_params_rejects_rosse_guiding_curve():
    with pytest.raises(ValueError, match="guiding curves"):
        build_geometry_params(
            {
                "formula": "R-OSSE",
                "profile": {"R_mm": 130.0},
                "gcurve": {"gcurveType": 1, "gcurveWidth": 100.0},
            }
        )


def test_load_config_accepts_ath_cfg_fixture(tmp_path):
    cfg_path = tmp_path / "rosse-simple.cfg"
    cfg_path.write_text(ROSSE_CFG, encoding="utf-8")

    config = load_config(cfg_path)
    params, formula, mode = build_geometry_params(config)

    assert formula == "R-OSSE"
    assert mode == "freestanding"
    assert params["type"] == "R-OSSE"
    assert "cos(p)" in params["R"]
    assert params["angularSegments"] == 80
    assert params["samplingMode"] == "ath-default-zmap"
    assert params["_athLengthMode"] == "total"


def test_rosse_config_preserves_throat_extension_keys(tmp_path):
    cfg_path = tmp_path / "rosse-extension.cfg"
    cfg_path.write_text(
        """
R-OSSE = {
  R = 150
  r0 = 10
  a = 45
  a0 = 12
  k = 1
  q = 1
  Throat.Ext.Length = 12
  Throat.Ext.Angle = 20
  Slot.Length = 5
}
Mesh = {
  AngularSegments = 16
  LengthSegments = 8
}
""",
        encoding="utf-8",
    )

    config = load_config(cfg_path)
    params, formula, _mode = build_geometry_params(config)

    assert formula == "R-OSSE"
    assert params["throatExtLength"] == 12
    assert params["throatExtAngle"] == 20
    assert params["slotLength"] == 5


def test_driver_adapter_derives_extension_length_from_inch_diameters():
    params, formula, _mode = build_geometry_params(
        {
            "formula": "R-OSSE",
            "profile": {
                "R": 150.0,
                "driver_throat_diameter_in": 1.0,
                "waveguide_throat_diameter_in": 1.4,
                "throatExtAngle": 15.0,
            },
            "mesh": {"angular_segments": 16, "length_segments": 8},
        }
    )

    expected_length = (0.5 * (1.4 - 1.0) * 25.4) / math.tan(math.radians(15.0))
    assert formula == "R-OSSE"
    assert math.isclose(params["r0"], 12.7)
    assert math.isclose(params["throatExtLength"], expected_length)
    assert params["throatExtAngle"] == 15.0


def test_driver_adapter_derives_extension_angle_from_length():
    params, _formula, _mode = build_geometry_params(
        {
            "formula": "OSSE",
            "profile": {
                "L": 100.0,
                "driver_throat_diameter_mm": 25.4,
                "waveguide_throat_diameter_mm": 35.56,
                "throatExtLength": 20.0,
            },
            "mesh": {"angular_segments": 16, "length_segments": 8},
        }
    )

    expected_angle = math.degrees(math.atan((17.78 - 12.7) / 20.0))
    assert math.isclose(params["r0"], 12.7)
    assert params["throatExtLength"] == 20.0
    assert math.isclose(params["throatExtAngle"], expected_angle)


def test_load_config_accepts_ath_txt_extension(tmp_path):
    txt_path = tmp_path / "osse-simple.txt"
    txt_path.write_text(OSSE_CFG, encoding="utf-8")

    config = load_config(txt_path)
    params, formula, mode = build_geometry_params(config)

    assert formula == "OSSE"
    assert mode == "freestanding"
    assert "cos(p)" in params["a"]
    assert params["morphTarget"] == 1


def test_parse_ath_config_preserves_gcurve_metadata(tmp_path):
    cfg_path = tmp_path / "gcurve.cfg"
    cfg_path.write_text(
        """
OSSE = {
  Length = 80
}
GCurve.Type = 2
GCurve.Width = 75
GCurve.SF = 1,1,8,0.6,5,2
Mesh.LengthSegments = 4
Mesh.SubdomainSlices = 2,4
Mesh.InterfaceOffset = 10
Mesh.InterfaceResolution = 12
""",
        encoding="utf-8",
    )

    config = load_config(cfg_path)
    params, formula, _mode = build_geometry_params(config)

    assert formula == "OSSE"
    assert params["gcurveType"] == 2
    assert params["gcurveWidth"] == 75
    assert params["gcurveSF"] == "1,1,8,0.6,5,2"
    assert params["subdomainSlices"] == "2,4"
    assert params["interfaceOffset"] == 10
    assert params["interfaceResolution"] == 12


def test_parse_ath_config_preserves_interface_arrays(tmp_path):
    cfg_path = tmp_path / "interfaces.cfg"
    cfg_path.write_text(
        """
OSSE = {
  Length = 80
}
Mesh.SubdomainSlices = 2,4
Mesh.InterfaceOffset = 6,9
Mesh.InterfaceResolution = 12
""",
        encoding="utf-8",
    )

    config = load_config(cfg_path)
    params, formula, _mode = build_geometry_params(config)

    assert formula == "OSSE"
    assert params["subdomainSlices"] == "2,4"
    assert params["interfaceOffset"] == "6,9"
    assert params["samplingMode"] == "ath-default-zmap"


def test_parse_ath_config_preserves_zmap_points(tmp_path):
    cfg_path = tmp_path / "zmap.cfg"
    cfg_path.write_text(
        """
OSSE = {
  Length = 80
}
Mesh.LengthSegments = 4
Mesh.ZMapPoints = 0.5,0.1,0.75,0.7
""",
        encoding="utf-8",
    )

    config = load_config(cfg_path)
    params, formula, _mode = build_geometry_params(config)

    assert formula == "OSSE"
    assert config["mesh"]["samplingMode"] == "zmap"
    assert config["mesh"]["zMapPoints"] == "0.5,0.1,0.75,0.7"
    assert params["samplingMode"] == "zmap"
    assert params["zMapPoints"] == "0.5,0.1,0.75,0.7"


def test_build_from_config_accepts_ath_morph(tmp_path):
    result = build_from_config(
        {
            "formula": "OSSE",
            "profile": {
                "L": 80.0,
                "r0": 12.7,
                "a": 45.0,
                "a0": 10.0,
            },
            "morph": {
                "morphTarget": 1,
                "morphWidth": 320,
                "morphHeight": 320,
            },
            "mesh": {
                "angularSegments": 12,
                "lengthSegments": 4,
            },
        },
        tmp_path / "morph.msh",
    )

    assert result.n_vertices > 0


def test_build_from_config_accepts_ath_gcurve(tmp_path):
    result = build_from_config(
        {
            "formula": "OSSE",
            "profile": {
                "L": 80.0,
                "r0": 12.7,
                "a": 45.0,
                "a0": 10.0,
            },
            "gcurve": {
                "gcurveType": 2,
                "gcurveWidth": 75,
            },
            "mesh": {
                "angularSegments": 12,
                "lengthSegments": 4,
            },
        },
        tmp_path / "gcurve.msh",
    )

    assert result.n_vertices > 0


def test_build_from_config_osse_freestanding(tmp_path):
    result = build_from_config(
        {
            "formula": "OSSE",
            "mode": "freestanding",
            "profile": {
                "L_mm": 80.0,
                "r0_mm": 10.0,
                "a_deg": 45.0,
                "a0_deg": 10.0,
            },
            "mesh": {
                "angular_segments": 12,
                "length_segments": 4,
                "wall_thickness_mm": 5.0,
                "throat_res_mm": 8.0,
                "mouth_res_mm": 30.0,
                "rear_res_mm": 30.0,
            },
        },
        tmp_path / "osse.msh",
    )

    assert result.formula == "OSSE"
    assert result.mode == "freestanding"
    assert result.n_vertices > 0
    assert result.n_triangles > 0
    assert {1, 2}.issubset(result.physical_groups)


def test_build_from_config_rosse_enclosure(tmp_path):
    result = build_from_config(
        {
            "formula": "ROSSE",
            "profile": {
                "R_mm": 90.0,
                "r0_mm": 10.0,
                "a_deg": 50.0,
                "a0_deg": 12.0,
                "k": 1.0,
                "q": 1.0,
            },
            "mesh": {
                "angular_segments": 16,
                "length_segments": 6,
                "throat_res_mm": 8.0,
                "mouth_res_mm": 35.0,
                "rear_res_mm": 40.0,
            },
            "enclosure": {
                "depth_mm": 120.0,
                "space_l_mm": 20.0,
                "space_t_mm": 20.0,
                "space_r_mm": 20.0,
                "space_b_mm": 20.0,
                "edge_mm": 6.0,
            },
        },
        tmp_path / "rosse-enclosure.msh",
    )

    assert result.formula == "R-OSSE"
    assert result.mode == "enclosure"
    assert result.n_vertices > 0
    assert result.n_triangles > 0
    assert {1, 2}.issubset(result.physical_groups)


def test_build_from_ath_enclosure_fixture(tmp_path):
    cfg_path = tmp_path / "osse-with-enclosure.cfg"
    cfg_path.write_text(OSSE_ENCLOSURE_CFG, encoding="utf-8")

    result = build_from_config(
        load_config(cfg_path),
        tmp_path / "osse-with-enclosure.msh",
    )

    assert result.formula == "OSSE"
    assert result.mode == "enclosure"
    assert result.n_vertices > 0
    assert result.n_triangles > 0
    assert {1, 2}.issubset(result.physical_groups)


ATH_FLAT_OSSE_CFG = """
Throat.Diameter = 36
Coverage.Angle = 50
Length = 150
Term.n = 4
Term.q = 0.996
OS.k = 0.9
"""


def test_ath_text_import_injects_ath_defaults(tmp_path):
    cfg_path = tmp_path / "flat-osse.cfg"
    # SimType 2 keeps freestanding mode so the wall-thickness default is
    # observable (infinite-baffle mode forces the wall to zero).
    cfg_path.write_text(ATH_FLAT_OSSE_CFG + "ABEC.SimType = 2\n", encoding="utf-8")

    params, formula, _mode = build_geometry_params(load_config(cfg_path))

    assert formula == "OSSE"
    assert params["a0"] == 0
    assert params["s"] == 0.7
    assert params["wallThickness"] == 5
    assert params["throatResolution"] == 5
    assert params["mouthResolution"] == 8
    assert params["rearResolution"] == 10


def test_toml_dict_configs_keep_package_defaults():
    params, _formula, _mode = build_geometry_params(
        {
            "formula": "OSSE",
            "profile": {"L_mm": 120.0, "r0_mm": 12.7},
            "mesh": {"angular_segments": 16, "length_segments": 6},
        }
    )

    assert params["a0"] == 15.5
    assert params["s"] == 0.0
    assert params["wallThickness"] == 6.0
    assert params["throatResolution"] == 4.0
    assert params["mouthResolution"] == 26.0


def test_ath_text_import_maps_term_k_alias(tmp_path):
    cfg_path = tmp_path / "term-k.cfg"
    cfg_path.write_text(ATH_FLAT_OSSE_CFG.replace("OS.k = 0.9", "Term.k = 0.7"), encoding="utf-8")

    params, _formula, _mode = build_geometry_params(load_config(cfg_path))

    assert params["k"] == 0.7


def test_ath_text_import_requires_length(tmp_path):
    cfg_path = tmp_path / "no-length.cfg"
    cfg_path.write_text(ATH_FLAT_OSSE_CFG.replace("Length = 150\n", ""), encoding="utf-8")

    with pytest.raises(ValueError, match="Length"):
        load_config(cfg_path)


def test_ath_text_import_defaults_morph_corner_radius(tmp_path):
    cfg_path = tmp_path / "morph-default-corner.cfg"
    cfg_path.write_text(ATH_FLAT_OSSE_CFG + "Morph.TargetShape = 1\n", encoding="utf-8")

    params, _formula, _mode = build_geometry_params(load_config(cfg_path))

    assert params["morphTarget"] == 1
    assert params["morphCorner"] == 35


def test_ath_text_import_normalizes_boolean_shrinkage(tmp_path):
    cfg_path = tmp_path / "morph-shrinkage.cfg"
    cfg_path.write_text(
        ATH_FLAT_OSSE_CFG + "Morph.TargetShape = 1\nMorph.AllowShrinkage = yes\n",
        encoding="utf-8",
    )

    params, _formula, _mode = build_geometry_params(load_config(cfg_path))

    assert params["morphAllowShrinkage"] == 1


def test_ath_text_import_translates_source_shape_enum(tmp_path):
    cfg_path = tmp_path / "source-flat.cfg"
    cfg_path.write_text(ATH_FLAT_OSSE_CFG + "Source.Shape = 2\n", encoding="utf-8")

    params, _formula, _mode = build_geometry_params(load_config(cfg_path))

    assert params["sourceShape"] == 0

    bad_path = tmp_path / "source-bad.cfg"
    bad_path.write_text(ATH_FLAT_OSSE_CFG + "Source.Shape = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Source.Shape"):
        load_config(bad_path)


def test_ath_text_import_rejects_unsupported_geometry_keys(tmp_path):
    for extra, match in (
        ("Throat.Profile = 3\n", "Throat.Profile"),
        ("Rollback = 1\n", "Rollback"),
        ("Rollback.StartAt = 0.5\n", "Rollback"),
        ("Mesh.RearShape = 2\n", "RearShape"),
        ("Mesh.ThroatSegments = 8\n", "ThroatSegments"),
    ):
        cfg_path = tmp_path / "unsupported.cfg"
        cfg_path.write_text(ATH_FLAT_OSSE_CFG + extra, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_config(cfg_path)

    ok_path = tmp_path / "supported.cfg"
    ok_path.write_text(ATH_FLAT_OSSE_CFG + "Throat.Profile = 1\nMesh.RearShape = 1\n", encoding="utf-8")
    params, _formula, _mode = build_geometry_params(load_config(ok_path))
    assert params["a0"] == 0


def test_ath_sim_type_selects_mesh_topology_mode(tmp_path):
    base = ATH_FLAT_OSSE_CFG

    cfg_path = tmp_path / "default-simtype.cfg"
    cfg_path.write_text(base, encoding="utf-8")
    _params, _formula, mode = build_geometry_params(load_config(cfg_path))
    assert mode == "infinite-baffle"

    cfg_path.write_text(base + "ABEC.SimType = 1\n", encoding="utf-8")
    _params, _formula, mode = build_geometry_params(load_config(cfg_path))
    assert mode == "infinite-baffle"

    cfg_path.write_text(base + "ABEC.SimType = 2\n", encoding="utf-8")
    _params, _formula, mode = build_geometry_params(load_config(cfg_path))
    assert mode == "freestanding"

    cfg_path.write_text(base + "ABEC.SimType = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SimType"):
        load_config(cfg_path)


def test_toml_dict_config_without_mode_stays_freestanding():
    _params, _formula, mode = build_geometry_params(
        {
            "formula": "OSSE",
            "profile": {"L_mm": 80.0, "r0_mm": 10.0},
            "mesh": {"angular_segments": 12, "length_segments": 4},
        }
    )
    assert mode == "freestanding"


def test_infinite_baffle_build_closes_mouth_with_interface(tmp_path):
    cfg_path = tmp_path / "ib.cfg"
    cfg_path.write_text(
        ATH_FLAT_OSSE_CFG + "Mesh.AngularSegments = 12\nMesh.LengthSegments = 4\n",
        encoding="utf-8",
    )

    result = build_from_config(load_config(cfg_path), tmp_path / "ib.msh")

    assert result.mode == "infinite-baffle"
    assert result.physical_groups[1] == "SD1G0"
    assert result.physical_groups[2] == "SD1D1001"
    assert result.physical_groups[4] == "I1-2"

    import meshio

    mesh = meshio.read(result.mesh_path)
    assert float(mesh.points[:, 2].min()) >= -1.0e-9
