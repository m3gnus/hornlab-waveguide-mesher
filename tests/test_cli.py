from __future__ import annotations

import math

import pytest

from hornlab_mesher.cli import build_from_config, build_geometry_params, load_config
from hornlab_mesher.config_parser import ConfigError


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
    # ATH slice indices shift by one onto grid rings (last slice = mouth).
    assert params["subdomainSlices"] == "3,5"
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
    # ATH slice indices shift by one onto grid rings (last slice = mouth).
    assert params["subdomainSlices"] == "3,5"
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
    # Closed mouth -> the metal solver keeps the strict cut-plane open-edge guard.
    assert result.native_check_open_edges is True
    assert result.as_dict()["native_check_open_edges"] is True


def test_build_from_config_osse_bare_relaxes_open_edge_guard(tmp_path):
    result = build_from_config(
        {
            "formula": "OSSE",
            "mode": "bare",
            "profile": {
                "L_mm": 80.0,
                "r0_mm": 10.0,
                "a_deg": 45.0,
                "a0_deg": 10.0,
            },
            "mesh": {
                "angular_segments": 12,
                "length_segments": 4,
                "throat_res_mm": 8.0,
                "mouth_res_mm": 30.0,
            },
        },
        tmp_path / "osse_bare.msh",
    )

    assert result.mode == "bare"
    # A bare horn radiates from an open mouth: its mirror-reduced rim is a real
    # free edge off the symmetry planes, so the open-edge guard must be relaxed.
    assert result.native_check_open_edges is False
    assert result.as_dict()["native_check_open_edges"] is False


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
    assert result.native_check_open_edges is True


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


def test_multi_source_ath_configs_are_rejected(tmp_path):
    # Contour/secondary-source models would silently mesh a default cap source
    # instead of the configured crossover model (Boundary Lab 2-way example).
    for snippet in (
        "Source.Contours = {\n1 0 0\n2 10 0\n}\n",
        "LFSource.B = {\nSID = 4\n}\n",
        "Source.Velocity = 2\n",
    ):
        cfg_path = tmp_path / "multi-source.cfg"
        cfg_path.write_text(ATH_FLAT_OSSE_CFG + "ABEC.SimType = 2\n" + snippet, encoding="utf-8")
        with pytest.raises(ConfigError, match="multi-source"):
            load_config(cfg_path)


def test_explicit_mode_contradicting_enclosure_depth_raises():
    config = {
        "formula": "OSSE",
        "mode": "infinite-baffle",
        "profile": {"r0": 12.7, "a": 45.0, "L": 90.0},
        "enclosure": {"depth_mm": 200.0},
    }
    with pytest.raises(ConfigError, match="contradicts"):
        build_geometry_params(config)


def test_enclosure_depth_without_mode_still_implies_enclosure():
    config = {
        "formula": "OSSE",
        "profile": {"r0": 12.7, "a": 45.0, "L": 90.0},
        "enclosure": {"depth_mm": 200.0},
    }
    _params, _formula, mode = build_geometry_params(config)
    assert mode == "enclosure"


def test_ath_text_import_injects_ath_defaults(tmp_path):
    cfg_path = tmp_path / "flat-osse.cfg"
    # SimType 2 keeps freestanding mode so the wall-thickness default is
    # observable (infinite-baffle mode forces the wall to zero).
    cfg_path.write_text(ATH_FLAT_OSSE_CFG + "ABEC.SimType = 2\n", encoding="utf-8")

    params, formula, _mode = build_geometry_params(load_config(cfg_path))

    assert formula == "OSSE"
    assert params["a0"] == 0
    assert params["s"] == 0.7
    # ATH defaults verified against ath.exe V2025-12 (byte-identical mesh for
    # absent vs explicit value): wall 5, throat 4, mouth 8, rear 15.
    assert params["wallThickness"] == 5
    assert params["throatResolution"] == 4
    assert params["mouthResolution"] == 8
    assert params["rearResolution"] == 15


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


def test_auto_source_cap_is_flat_at_zero_throat_angle(tmp_path):
    # ATH matches the auto source cap to the throat opening angle; the m2
    # config omits Throat.Angle (default 0), so the source must stay flat.
    cfg_path = tmp_path / "flat-source.cfg"
    cfg_path.write_text(
        ATH_FLAT_OSSE_CFG + "Mesh.AngularSegments = 12\nMesh.LengthSegments = 4\n",
        encoding="utf-8",
    )

    result = build_from_config(load_config(cfg_path), tmp_path / "flat-source.msh")

    import meshio
    import numpy as np

    mesh = meshio.read(result.mesh_path)
    source_tag = next(tag for tag, name in result.physical_groups.items() if name == "SD1D1001")
    source_vertex_ids = set()
    for block, data in zip(mesh.cells, mesh.cell_data.get("gmsh:physical", [])):
        if block.type != "triangle":
            continue
        source_vertex_ids.update(block.data[data == source_tag].ravel().tolist())
    source_z = mesh.points[sorted(source_vertex_ids), 2]
    assert float(np.max(np.abs(source_z))) < 1.0e-9


def test_build_result_reports_symmetry_hint_for_quadrant_grids(tmp_path):
    base = {
        "formula": "OSSE",
        "profile": {"L_mm": 80.0, "r0_mm": 10.0, "a_deg": 40.0, "a0_deg": 0.0},
        "mesh": {"angular_segments": 16, "length_segments": 4, "wall_thickness_mm": 0.0, "mode": "bare"},
    }

    full = build_from_config(base, tmp_path / "full.msh")
    assert full.quadrants == "1234"
    assert full.native_symmetry_plane is None

    quarter_cfg = {**base, "mesh": {**base["mesh"], "quadrants": "1"}}
    quarter = build_from_config(quarter_cfg, tmp_path / "quarter.msh")
    assert quarter.quadrants == "1"
    assert quarter.native_symmetry_plane == "yz+xz"
    assert quarter.as_dict()["native_symmetry_plane"] == "yz+xz"


def test_build_result_mesh_report_carries_validity_frequencies(tmp_path):
    result = build_from_config(
        {
            "formula": "OSSE",
            "profile": {"L_mm": 80.0, "r0_mm": 10.0, "a_deg": 40.0, "a0_deg": 0.0},
            "mesh": {"angular_segments": 16, "length_segments": 4},
        },
        tmp_path / "report.msh",
    )

    assert set(result.mesh_report) == set(result.physical_groups.values())
    for stats in result.mesh_report.values():
        assert stats["median_edge_mm"] > 0.0
        assert stats["max_edge_mm"] >= stats["median_edge_mm"]
        # valid f = c / (epw * max_edge) with the 6 e/w, 343 m/s defaults
        assert math.isclose(
            stats["valid_f_max_hz"], 343000.0 / (6.0 * stats["max_edge_mm"]), rel_tol=1.0e-9
        )


def test_frequency_aware_sizing_clamps_coarse_mm_resolutions(tmp_path):
    # Bare mode keeps SD1G0 to the inner wall so the throat/mouth roles are
    # observable without the deliberately coarser rear/outer grading.
    base = {
        "formula": "OSSE",
        "mode": "bare",
        "profile": {"L_mm": 100.0, "r0_mm": 12.7, "a_deg": 45.0, "a0_deg": 0.0},
        "mesh": {
            "angular_segments": 32,
            "length_segments": 8,
            "throat_res_mm": 5.0,
            "mouth_res_mm": 30.0,
            "rear_res_mm": 30.0,
        },
    }

    coarse = build_from_config(base, tmp_path / "coarse.msh")
    banded_cfg = {**base, "mesh": {**base["mesh"], "max_frequency_hz": 10000.0}}
    banded = build_from_config(banded_cfg, tmp_path / "banded.msh")

    # Mouth-role ceiling at 10 kHz / 6 e/w is 5.717 mm; the 30 mm wall must
    # refine. Throat role (8 e/w -> 4.29 mm) clamps the 5 mm throat too.
    assert banded.n_triangles > 2.0 * coarse.n_triangles
    wall = banded.mesh_report["SD1G0"]
    assert wall["median_edge_mm"] < 7.0
    assert wall["valid_f_max_hz"] > 0.5 * 10000.0
    assert banded.mesh_report["SD1D1001"]["median_edge_mm"] < 4.5


def test_frequency_aware_rear_grading_keeps_freestanding_meshes_small(tmp_path):
    base = {
        "formula": "OSSE",
        "profile": {"L_mm": 100.0, "r0_mm": 12.7, "a_deg": 45.0, "a0_deg": 0.0},
        "mesh": {
            "angular_segments": 32,
            "length_segments": 8,
            "throat_res_mm": 5.0,
            "mouth_res_mm": 30.0,
            "rear_res_mm": 30.0,
            "max_frequency_hz": 10000.0,
        },
    }

    graded = build_from_config(base, tmp_path / "graded.msh")
    flat_cfg = {**base, "mesh": {**base["mesh"], "rear_epw": 6.0}}
    flat = build_from_config(flat_cfg, tmp_path / "flat.msh")

    # The shadowed rear/outer surfaces at 2.5 e/w (22.9 mm ceiling) must cost
    # markedly fewer elements than forcing the strict 6 e/w target there.
    assert graded.n_triangles < 0.75 * flat.n_triangles


def test_freestanding_half_models_build_with_single_cut_plane(tmp_path):
    base = {
        "formula": "OSSE",
        "profile": {"L_mm": 80.0, "r0_mm": 10.0, "a_deg": 40.0, "a0_deg": 0.0},
        "mesh": {"angular_segments": 16, "length_segments": 4},
    }
    results = {
        q: build_from_config(
            {**base, "mesh": {**base["mesh"], "quadrants": q}}, tmp_path / f"fs-{q}.msh"
        )
        for q in ("1", "12", "14", "1234")
    }
    for r in results.values():
        assert r.n_triangles > 0
        # Closed (freestanding) modes keep the strict cut-plane open-edge guard.
        assert r.native_check_open_edges is True

    # Half-models map to a single mirror plane; the solver reflects the modeled
    # half across it. Quadrants 12 mirror about xz, 14 about yz.
    assert results["12"].native_symmetry_plane == "xz"
    assert results["14"].native_symmetry_plane == "yz"
    assert results["1"].native_symmetry_plane == "yz+xz"
    assert results["1234"].native_symmetry_plane is None

    # A half spans one mirror; its count sits between the quarter and the full
    # model and the two halves match each other.
    assert results["1"].n_triangles < results["12"].n_triangles < results["1234"].n_triangles
    assert results["12"].n_triangles == results["14"].n_triangles
