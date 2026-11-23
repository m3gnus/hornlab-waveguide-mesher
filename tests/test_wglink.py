from __future__ import annotations

from dataclasses import replace
import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest

from hornlab_mesher.builders.enclosure import enclosure_box_bounds
from hornlab_mesher.cad import (
    SOURCE_INTERFACE_FEATURE,
    THROAT_ORIGIN_FRAME_FEATURE,
    WGLINK_BUNDLE_VERSION,
    WgLinkSourceInterface,
    read_wglink,
    write_wglink,
)
from hornlab_mesher import cad as cad_module
from hornlab_mesher.datums import derive_datums
from hornlab_mesher.geometry import (
    BuiltGeometry,
    HornEnclosure,
    PointGridHornGeometry,
)
from hornlab_mesher.mesher import MesherError


def _inner_grid(*, nonplanar_mouth: bool = False) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    points = np.empty((len(angles), 3, 3), dtype=np.float64)
    for station, (radius, z) in enumerate(((12.7, 0.0), (30.0, 55.0), (60.0, 100.0))):
        points[:, station, 0] = radius * np.cos(angles)
        aspect = 1.0 if station == 0 else 0.8
        points[:, station, 1] = aspect * radius * np.sin(angles)
        points[:, station, 2] = z
    if nonplanar_mouth:
        points[0, -1, 2] += 0.05
        points[4, -1, 2] -= 0.03
    return points


def _outer_grid(inner: np.ndarray) -> np.ndarray:
    outer = np.array(inner, copy=True)
    radial = np.linalg.norm(outer[..., :2], axis=2)
    scale = np.divide(radial + 6.0, radial, out=np.ones_like(radial), where=radial > 0)
    outer[..., :2] *= scale[..., None]
    return outer


def _freestanding(**kwargs) -> PointGridHornGeometry:
    inner = _inner_grid(nonplanar_mouth=kwargs.pop("nonplanar_mouth", False))
    return PointGridHornGeometry(
        inner_points=inner,
        outer_points=_outer_grid(inner),
        **kwargs,
    )


def _enclosure_geometry(
    *,
    edge_type: int | None = None,
    plan_type: int = 1,
    vertical_offset_mm: float = 0.0,
):
    edge_type_kwargs = {} if edge_type is None else {"edge_type": edge_type}
    return PointGridHornGeometry(
        inner_points=_inner_grid(),
        enclosure=HornEnclosure(
            depth_mm=20.0,
            space_l_mm=10.0,
            space_r_mm=14.0,
            space_t_mm=12.0,
            space_b_mm=8.0,
            edge_mm=50.0,
            plan_type=plan_type,
            **edge_type_kwargs,
        ),
        vertical_offset_mm=vertical_offset_mm,
    )


def _built(geometry: PointGridHornGeometry) -> BuiltGeometry:
    bounds = None
    if geometry.enclosure is not None:
        bounds = enclosure_box_bounds(
            geometry.inner_points,
            geometry.enclosure,
            closed=geometry.closed,
            symmetry_planes=geometry.symmetry_planes,
            surface_fit=geometry.surface_fit,
            preserve_grid=geometry.preserve_grid,
        )
    return BuiltGeometry(
        surface_groups={},
        axial_bounds_mm=(0.0, 100.0),
        source_axis="z",
        enclosure_bounds=bounds,
    )


def _fake_step(monkeypatch) -> list:
    """A bundle (format 1.2) writes no STEP: it may only check the solid builds.

    The OCC check is stubbed (it stops before writing, returning None); writing
    a STEP is a failure. Returns the stops requested.
    """

    stops: list = []

    def no_step(*_args, **_kwargs):
        raise AssertionError("write_wglink must not write a STEP")

    def checked_only(_geometry, output_path=None, *, stop=None, **_kwargs):
        if stop is None:
            raise AssertionError("write_wglink must not write a STEP")
        stops.append(stop)
        return None

    monkeypatch.setattr("hornlab_mesher.cad.write_step", no_step)
    monkeypatch.setattr("hornlab_mesher.cad._write_step_once", checked_only)
    return stops


def test_freestanding_datums_use_inner_and_outer_realized_rings():
    geometry = _freestanding()
    datums = derive_datums(geometry, _built(geometry))

    assert datums["rim_planar"] is True
    assert datums["WG_MOUTH_PLANE"]["exact"] is True
    assert np.asarray(datums["WG_MOUTH_OUTLINE_INNER"]["points_mm"]) == pytest.approx(
        geometry.inner_points[:, -1, :]
    )
    assert np.asarray(datums["WG_MOUTH_OUTLINE_OUTER"]["points_mm"]) == pytest.approx(
        geometry.outer_points[:, -1, :]
    )


def test_nonplanar_rim_is_flagged_without_an_exact_mouth_plane():
    geometry = _freestanding(nonplanar_mouth=True)
    datums = derive_datums(geometry, _built(geometry))

    assert datums["rim_planar"] is False
    assert "WG_MOUTH_PLANE" not in datums


def test_enclosure_datums_use_clamped_bounds_and_distinct_baffle_outlines():
    geometry = _enclosure_geometry()
    built = _built(geometry)
    datums = derive_datums(geometry, built)
    bounds = built.enclosure_bounds
    assert bounds is not None

    assert bounds["enc_depth"] == pytest.approx(101.0)
    assert bounds["z_back"] == pytest.approx(bounds["z_front"] - 101.0)
    assert bounds["clamped_edge"] < geometry.enclosure.edge_mm
    assert datums["WG_BAFFLE_PLANE"]["origin_mm"][2] == bounds["z_front"]
    assert datums["WG_ENC_BACK_PLANE"]["origin_mm"][2] == bounds["z_back"]
    face = np.asarray(datums["WG_BAFFLE_OUTLINE_FACE"]["points_mm"])
    envelope = np.asarray(datums["WG_BAFFLE_OUTLINE_ENVELOPE"]["points_mm"])
    assert np.ptp(face[:, 0]) < np.ptp(envelope[:, 0])
    assert np.ptp(face[:, 1]) < np.ptp(envelope[:, 1])


def test_vertical_offset_does_not_move_the_link_frame():
    """Format 1.3: the throat centre is the link origin whatever the offset.

    Mesh.VerticalOffset used to be baked into every link-local coordinate, so
    an 80 mm offset design put its throat 80 mm above the Fusion origin and
    needed a datum pair (WG_GEOM_MIDPLANE_Y, WG_AXIS) just to find it again.
    """
    geometry = _freestanding(vertical_offset_mm=37.5)
    datums = derive_datums(geometry, _built(geometry), link_frame=True)

    assert datums["WG_AXIS"]["origin_mm"] == [0.0, 0.0, 0.0]
    assert datums["WG_GEOM_MIDPLANE_Y"]["origin_mm"][1] == 0.0
    assert datums["WG_SOLVER_CUT_PLANE_Y"]["origin_mm"][1] == 0.0
    assert datums["WG_THROAT_PLANE"]["origin_mm"] == pytest.approx([0.0, 0.0, 0.0])
    mouth = np.asarray(datums["WG_MOUTH_OUTLINE_INNER"]["points_mm"])
    assert mouth[:, 1] == pytest.approx(geometry.inner_points[:, -1, 1])


def test_default_datums_keep_the_mesh_placement_that_write_step_writes():
    """``derive_datums`` without ``link_frame`` is the frame a default STEP is in.

    ``CadInfo.datums`` of a placed STEP comes from it, so it must keep the
    offset (and the throat's own station) the STEP keeps.
    """

    raised = _freestanding(vertical_offset_mm=37.5)
    raised = replace(raised, inner_points=raised.inner_points + np.asarray([0.0, 0.0, 7.0]))
    datums = derive_datums(raised, _built(raised))

    assert datums["WG_AXIS"]["origin_mm"] == [0.0, 37.5, 0.0]
    assert datums["WG_GEOM_MIDPLANE_Y"]["origin_mm"][1] == 37.5
    assert datums["WG_SOLVER_CUT_PLANE_Y"]["origin_mm"][1] == 0.0
    assert datums["WG_THROAT_PLANE"]["origin_mm"] == pytest.approx([0.0, 37.5, 7.0])
    linked = derive_datums(raised, _built(raised), link_frame=True)
    assert linked["WG_THROAT_PLANE"]["origin_mm"] == pytest.approx([0.0, 0.0, 0.0])


def test_point_grid_payload_shares_the_throat_origin_datum_frame(monkeypatch, tmp_path):
    """ONE frame per bundle: the grid, datums and parameters share the throat
    origin, and Mesh.VerticalOffset moves none of them (format 1.3).

    Found on a real Tritonia export: with the offset baked in, the throat sat
    80 mm above the Fusion origin and the return path had to undo it.
    """
    _fake_step(monkeypatch)
    geometry = _freestanding(vertical_offset_mm=37.5)
    result = write_wglink(geometry, tmp_path / "horn.wglink")

    grid = json.loads(result.point_grid_path.read_text())
    assert grid["frame"] == "link-local"
    assert grid["vertical_offset_mm"] == 0.0
    inner = np.asarray(grid["inner_points"])
    assert np.mean(inner[:, 0, :], axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert inner == pytest.approx(geometry.inner_points)
    mouth = np.asarray(
        result.manifest["datums"]["WG_MOUTH_OUTLINE_INNER"]["points_mm"]
    )
    assert inner[:, -1, :] == pytest.approx(mouth)
    params = {entry["name"]: entry["value"] for entry in result.manifest["parameters"]}
    assert params["wg_waveguide_vertical_offset"] == 0.0
    assert THROAT_ORIGIN_FRAME_FEATURE in result.manifest["required_features"]
    assert result.manifest["wglink_version"] == WGLINK_BUNDLE_VERSION == "1.3"


def test_a_throat_off_z_zero_is_moved_to_the_link_origin(monkeypatch, tmp_path):
    """Every artifact in the bundle moves by one axial shift to put the throat at z = 0."""

    _fake_step(monkeypatch)
    centred = _enclosure_geometry()
    raised = replace(centred, inner_points=centred.inner_points + np.asarray([0.0, 0.0, 7.0]))
    reference = write_wglink(centred, tmp_path / "centred.wglink", check_points=[[1.0, 2.0, 3.0]])
    moved = write_wglink(raised, tmp_path / "raised.wglink", check_points=[[1.0, 2.0, 10.0]])

    grid = json.loads(moved.point_grid_path.read_text())
    expected = json.loads(reference.point_grid_path.read_text())
    np.testing.assert_allclose(grid["inner_points"], expected["inner_points"], atol=1e-9)
    np.testing.assert_allclose(grid["ring_z_mm"], expected["ring_z_mm"], atol=1e-9)
    np.testing.assert_allclose(grid["check_points"], expected["check_points"], atol=1e-9)
    assert grid["ring_z_mm"][0] == pytest.approx(0.0, abs=1e-9)
    values = {e["name"]: e["value"] for e in moved.manifest["parameters"]}
    reference_values = {e["name"]: e["value"] for e in reference.manifest["parameters"]}
    assert values == pytest.approx(reference_values)
    for name in ("WG_THROAT_PLANE", "WG_BAFFLE_PLANE", "WG_ENC_BACK_PLANE", "WG_AXIS"):
        assert moved.manifest["datums"][name]["origin_mm"] == pytest.approx(
            reference.manifest["datums"][name]["origin_mm"], abs=1e-9
        ), name
    np.testing.assert_allclose(
        moved.manifest["datums"]["WG_BAFFLE_OUTLINE_FACE"]["points_mm"],
        reference.manifest["datums"]["WG_BAFFLE_OUTLINE_FACE"]["points_mm"],
        atol=1e-9,
    )


def test_freestanding_parameter_table_has_no_enclosure_parameters(
    monkeypatch, tmp_path
):
    _fake_step(monkeypatch)
    result = write_wglink(_freestanding(), tmp_path / "horn.wglink")

    names = {entry["name"] for entry in result.manifest["parameters"]}
    assert not any("_enc_" in name for name in names)
    assert "enclosure" not in result.manifest
    assert "enclosure" not in read_wglink(result.path)


def test_source_interface_v1_round_trips_as_an_additive_sources_table(
    monkeypatch, tmp_path
):
    _fake_step(monkeypatch)
    source = WgLinkSourceInterface(
        id="source-hf",
        role="HF",
        required=True,
        default_drive_channel_id="drive-hf",
        patch_policy="single-connected",
        expected_connected_components=1,
        suggested_resolution_mm=4.0,
    )

    result = write_wglink(
        _freestanding(),
        tmp_path / "horn.wglink",
        interface_sources=[source],
    )
    manifest = read_wglink(result.path)

    assert SOURCE_INTERFACE_FEATURE in manifest["required_features"]
    assert manifest["interface"] == {
        "sources": [
            {
                "id": "source-hf",
                "role": "HF",
                "required": True,
                "default_drive_channel_id": "drive-hf",
                "patch_policy": "single-connected",
                "expected_connected_components": 1,
                "suggested_resolution_mm": 4.0,
            }
        ]
    }


def test_source_interface_feature_and_nonempty_table_are_atomic(monkeypatch, tmp_path):
    _fake_step(monkeypatch)
    result = write_wglink(
        _freestanding(),
        tmp_path / "horn.wglink",
        interface_sources=[
            {
                "id": "source-hf",
                "role": "HF",
                "required": True,
                "default_drive_channel_id": "drive-hf",
                "patch_policy": "single-connected",
                "expected_connected_components": 1,
                "suggested_resolution_mm": 4,
            }
        ],
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["required_features"].remove(SOURCE_INTERFACE_FEATURE)
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MesherError, match="required exactly when"):
        read_wglink(result.path)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"id": " source-hf"}, "non-empty trimmed string"),
        ({"required": 1}, "required must be boolean"),
        ({"patch_policy": "nearest"}, "patch_policy"),
        ({"expected_connected_components": 2}, "must be 1"),
        ({"suggested_resolution_mm": 0}, "must be positive"),
    ],
)
def test_source_interface_rejects_ambiguous_policy(
    monkeypatch, tmp_path, override, message
):
    _fake_step(monkeypatch)
    source = {
        "id": "source-hf",
        "role": "HF",
        "required": True,
        "default_drive_channel_id": "drive-hf",
        "patch_policy": "single-connected",
        "expected_connected_components": 1,
        "suggested_resolution_mm": 4,
        **override,
    }

    with pytest.raises(MesherError, match=message):
        write_wglink(
            _freestanding(),
            tmp_path / "bad.wglink",
            interface_sources=[source],
        )


@pytest.mark.parametrize(
    ("edge_type", "expected_edge_type"),
    [(None, 1), (2, 2)],
)
def test_enclosure_edge_treatment_round_trips(
    monkeypatch, tmp_path, edge_type, expected_edge_type
):
    _fake_step(monkeypatch)
    geometry = _enclosure_geometry(edge_type=edge_type)
    result = write_wglink(geometry, tmp_path / "horn.wglink")
    expected = {"edge_type": expected_edge_type, "plan_type": 1}

    assert geometry.enclosure.edge_type == expected_edge_type
    assert result.manifest["enclosure"] == expected
    assert read_wglink(result.path)["enclosure"] == expected


def test_tilted_planar_rim_has_consistent_planarity_metadata(monkeypatch, tmp_path):
    _fake_step(monkeypatch)
    geometry = _freestanding()
    geometry.inner_points[:, -1, 2] += 0.1 * geometry.inner_points[:, -1, 0]
    result = write_wglink(geometry, tmp_path / "horn.wglink")

    grid = json.loads(result.point_grid_path.read_text())
    assert grid["ring_planar"][-1] is True
    assert grid["all_rings_planar"] is True
    assert result.manifest["datums"]["rim_planar"] is True
    assert result.manifest["datums"]["WG_MOUTH_PLANE"]["exact"] is True


@pytest.mark.parametrize("plan_type", [2, 3])
def test_wglink_rejects_unbuildable_enclosure_plans(plan_type, tmp_path):
    with pytest.raises(MesherError, match=rf"plan_type={plan_type}.*plan_type=1"):
        write_wglink(_enclosure_geometry(plan_type=plan_type), tmp_path / "bad.wglink")


@pytest.mark.parametrize("plan_type", [1, 2, 3])
@pytest.mark.parametrize("nonplanar", [False, True])
def test_flush_bundle_exposes_the_fitted_mouth_without_a_baffle(
    plan_type, nonplanar, monkeypatch, tmp_path,
):
    from dataclasses import replace

    from hornlab_mesher.builders._occ import sample_patch_boundary
    from hornlab_mesher.builders.point_grid_surfaces import _bspline_patch_phi_groups
    from scipy.spatial import cKDTree

    _fake_step(monkeypatch)
    geometry = _enclosure_geometry(plan_type=plan_type, vertical_offset_mm=11.0)
    geometry = replace(geometry, enclosure=replace(geometry.enclosure, space_r_mm=0))
    if nonplanar:
        geometry = replace(geometry, inner_points=_inner_grid(nonplanar_mouth=True))
    result = write_wglink(geometry, tmp_path / "flush.wglink")
    manifest = read_wglink(result.path)
    datums = manifest["datums"]
    assert manifest["enclosure"]["plan_type"] == plan_type
    assert "WG_BAFFLE_PLANE" not in datums
    assert "WG_BAFFLE_OUTLINE_FACE" not in datums
    assert "WG_BAFFLE_OUTLINE_ENVELOPE" not in datums
    assert ("WG_MOUTH_PLANE" in datums) is not nonplanar
    actual = np.asarray(datums["WG_MOUTH_OUTLINE_INNER"]["points_mm"])
    assert len(actual) == len(geometry.inner_points)
    groups = _bspline_patch_phi_groups(len(geometry.inner_points), closed=True)
    expected = np.vstack([
        sample_patch_boundary(geometry.inner_points, group,
                              np.linspace(0, 1, 65, endpoint=False),
                              surface_fit=geometry.surface_fit)
        for group in groups
    ])
    # The offset (11 mm) is not part of the link frame.
    np.testing.assert_allclose(datums["WG_MOUTH_OUTLINE_OUTER"]["points_mm"], expected, atol=1e-10)
    dense = np.vstack([
        sample_patch_boundary(geometry.inner_points, group,
                              np.linspace(0, 1, 16385),
                              surface_fit=geometry.surface_fit)
        for group in groups
    ])
    assert cKDTree(dense).query(actual)[0].max() < 0.01
    # The n_phi entries still correspond to the original grid rays.
    original = geometry.inner_points[:, -1, :2]
    on_ray = actual[:, :2].copy()
    cross = original[:, 0] * on_ray[:, 1] - original[:, 1] * on_ray[:, 0]
    assert np.max(np.abs(cross)) < 0.01
    assert datums["WG_ENC_BACK_PLANE"]["origin_mm"][2] == pytest.approx(
        _built(geometry).enclosure_bounds["z_back"]
    )


def test_flush_preserved_grid_datum_keeps_every_corner(monkeypatch, tmp_path):
    from dataclasses import replace

    _fake_step(monkeypatch)
    geometry = _enclosure_geometry(vertical_offset_mm=11.0)
    geometry = replace(geometry, preserve_grid=True,
                       enclosure=replace(geometry.enclosure, space_r_mm=0))
    datums = write_wglink(geometry, tmp_path / "flush.wglink").manifest["datums"]
    expected = geometry.inner_points[:, -1, :].copy()
    actual = np.asarray(datums["WG_MOUTH_OUTLINE_INNER"]["points_mm"])
    np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize("plan_type", [2, 3])
def test_flush_nonrectangular_plan_writes_real_bundle(plan_type, tmp_path):
    """The real OCC solid check runs (verify_solid) and these enclosures pass it."""
    from hornlab_mesher.config_builder import resolve_geometry
    from test_point_grid_contract import rectangular_enclosure

    config = rectangular_enclosure(margin=0, quadrants="1234")
    config["enclosure"]["plan_type"] = plan_type
    result = write_wglink(resolve_geometry(config).geometry, tmp_path / "flush.wglink")
    manifest = read_wglink(result.path)
    assert manifest["enclosure"]["plan_type"] == plan_type
    assert manifest["body"]["file"] == "point-grid.json"


@pytest.mark.parametrize(
    "geometry, mode",
    [
        (PointGridHornGeometry(inner_points=_inner_grid()), "BARE"),
        (
            PointGridHornGeometry(inner_points=_inner_grid(), infinite_baffle=True),
            "INFINITE-BAFFLE",
        ),
    ],
)
def test_wglink_rejects_modes_without_material(geometry, mode, tmp_path):
    with pytest.raises(MesherError, match=rf"{mode} is not supported"):
        write_wglink(geometry, tmp_path / "bad.wglink")


def test_bundle_round_trip_realized_parameters_identity_and_checksums(monkeypatch, tmp_path):
    _fake_step(monkeypatch)
    geometry = _enclosure_geometry(vertical_offset_mm=11.0)
    object.__setattr__(geometry, "source_radius_mm", 999.0)
    built = _built(geometry)
    identity = {
        "bundle": {"id": "caller-bundle", "created_at": "verbatim"},
        "design": {"id": "caller-design", "design_hash": "sha256:design"},
        "export": {"id": "caller-export", "sequence": 19},
    }
    result = write_wglink(
        geometry,
        tmp_path / "horn.wglink",
        built_geometry=built,
        identity=identity,
        instance_slug="A-1",
        check_points=[[1.0, 2.0, 3.0]],
    )
    manifest = read_wglink(result.path)

    assert result.manifest["wglink_version"] == "1.3"
    assert manifest["wglink_version"] == "1.3"
    assert set(path.name for path in result.path.iterdir()) == {
        "wglink.json",
        "point-grid.json",
    }
    assert manifest["body"] == {
        "file": "point-grid.json",
        "kind": "point-grid-loft",
        "semantic_face_names": False,
    }
    assert set(manifest["files"]) == {"point-grid.json"}
    assert manifest["bundle"] == identity["bundle"]
    assert manifest["design"] == identity["design"]
    assert manifest["export"] == identity["export"]
    grid = json.loads(result.point_grid_path.read_text())
    assert grid["n_phi"] == geometry.inner_points.shape[0]
    assert grid["n_length"] == geometry.inner_points.shape[1]
    # Check points ship in the link frame, which the 11 mm offset does not move.
    assert grid["check_points"] == [[1.0, 2.0, 3.0]]
    parameter_entries = {entry["name"]: entry for entry in manifest["parameters"]}
    params = {name: entry["value"] for name, entry in parameter_entries.items()}
    bounds = built.enclosure_bounds
    assert bounds is not None
    assert params["wg_a_1_throat_dia"] == pytest.approx(25.4)
    assert params["wg_a_1_enc_w"] == pytest.approx(bounds["bx1"] - bounds["bx0"])
    assert params["wg_a_1_enc_depth"] == pytest.approx(bounds["enc_depth"])
    assert params["wg_a_1_enc_edge"] == pytest.approx(bounds["clamped_edge"])
    assert params["wg_a_1_enc_x0"] == pytest.approx(bounds["bx0"])
    assert params["wg_a_1_enc_y0"] == pytest.approx(bounds["by0"])
    assert params["wg_a_1_enc_y0"] < 0.0
    assert params["wg_a_1_enc_z_front"] == pytest.approx(bounds["z_front"])

    for name in ("wg_a_1_enc_x0", "wg_a_1_enc_y0", "wg_a_1_enc_z_front"):
        assert parameter_entries[name]["role"] == "interface"
        assert parameter_entries[name]["unit"] == "mm"
    assert {
        name: entry["position_axis"]
        for name, entry in parameter_entries.items()
        if "position_axis" in entry
    } == {
        "wg_a_1_vertical_offset": "y",
        "wg_a_1_enc_x0": "x",
        "wg_a_1_enc_y0": "y",
        "wg_a_1_enc_z_front": "z",
    }

    datums = manifest["datums"]
    envelope = np.asarray(datums["WG_BAFFLE_OUTLINE_ENVELOPE"]["points_mm"])
    assert params["wg_a_1_enc_x0"] == pytest.approx(float(np.min(envelope[:, 0])))
    assert params["wg_a_1_enc_y0"] == pytest.approx(float(np.min(envelope[:, 1])))
    assert params["wg_a_1_enc_x0"] + params["wg_a_1_enc_w"] == pytest.approx(
        float(np.max(envelope[:, 0]))
    )
    assert params["wg_a_1_enc_y0"] + params["wg_a_1_enc_h"] == pytest.approx(
        float(np.max(envelope[:, 1]))
    )
    assert params["wg_a_1_enc_z_front"] == pytest.approx(
        datums["WG_BAFFLE_PLANE"]["origin_mm"][2]
    )
    assert params["wg_a_1_enc_z_front"] - params["wg_a_1_enc_depth"] == pytest.approx(
        datums["WG_ENC_BACK_PLANE"]["origin_mm"][2]
    )

    with result.point_grid_path.open("ab") as stream:
        stream.write(b"X")
    with pytest.raises(MesherError, match="checksum validation failed.*point-grid.json"):
        read_wglink(result.path)


def test_reader_rejects_same_size_point_grid_corruption(monkeypatch, tmp_path):
    _fake_step(monkeypatch)
    result = write_wglink(_freestanding(), tmp_path / "horn.wglink")
    payload = bytearray(result.point_grid_path.read_bytes())
    offset = payload.index(b'"units": "mm"') + len(b'"units": "')
    payload[offset] = ord("c")
    result.point_grid_path.write_bytes(payload)

    with pytest.raises(MesherError, match="checksum validation failed.*point-grid.json"):
        read_wglink(result.path)


def test_reader_rejects_unchecksummed_extra_file(monkeypatch, tmp_path):
    _fake_step(monkeypatch)
    result = write_wglink(_freestanding(), tmp_path / "horn.wglink")
    (result.path / "swapped.step").write_bytes(b"not declared")

    with pytest.raises(MesherError, match="unchecksummed.*swapped.step"):
        read_wglink(result.path)


def test_reader_rejects_nested_unchecksummed_file(monkeypatch, tmp_path):
    _fake_step(monkeypatch)
    result = write_wglink(_freestanding(), tmp_path / "horn.wglink")
    nested = result.path / "extras" / "payload.bin"
    nested.parent.mkdir()
    nested.write_bytes(b"undeclared")

    with pytest.raises(MesherError, match=r"unchecksummed.*extras/payload\.bin"):
        read_wglink(result.path)


def test_writer_refuses_an_existing_bundle_and_leaves_it_intact(monkeypatch, tmp_path):
    _fake_step(monkeypatch)
    target = tmp_path / "horn.wglink"
    write_wglink(_freestanding(), target, identity={"export": {"sequence": 1}})

    with pytest.raises(MesherError, match="already exists"):
        write_wglink(_freestanding(), target, identity={"export": {"sequence": 2}})

    assert read_wglink(target)["export"]["sequence"] == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == ["horn.wglink"]


def test_publication_is_one_rename_that_leaves_nothing_beside_the_bundle(
    monkeypatch, tmp_path
):
    _fake_step(monkeypatch)
    target = tmp_path / "exports" / "horn.wglink"

    write_wglink(_freestanding(), target)

    assert sorted(path.name for path in target.parent.iterdir()) == ["horn.wglink"]
    assert read_wglink(target)["body"]["file"] == "point-grid.json"


def test_a_destination_that_appears_while_staging_is_not_replaced(tmp_path):
    target = tmp_path / "horn.wglink"
    staging = tmp_path / ".horn.wglink.staged"
    staging.mkdir()
    (staging / "generation.txt").write_text("new", encoding="utf-8")
    target.mkdir()
    (target / "generation.txt").write_text("raced", encoding="utf-8")

    with pytest.raises(MesherError, match="already exists"):
        cad_module._publish_bundle(staging, target)

    assert (target / "generation.txt").read_text(encoding="utf-8") == "raced"
    assert (staging / "generation.txt").read_text(encoding="utf-8") == "new"


def test_writer_fsyncs_staged_members_through_writable_descriptors(
    monkeypatch, tmp_path
):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "member.bin").write_bytes(b"payload")
    synced: list[int] = []

    monkeypatch.setattr(cad_module, "_fsync_directory", lambda _path: None)
    monkeypatch.setattr(
        cad_module.os,
        "fsync",
        lambda descriptor: synced.append(os.write(descriptor, b"")),
    )

    cad_module._sync_staged_bundle(staged)

    assert synced == [0]


def test_writer_rejects_config_like_enclosure_bounds(monkeypatch, tmp_path):
    _fake_step(monkeypatch)
    geometry = _enclosure_geometry()
    built = _built(geometry)
    assert built.enclosure_bounds is not None
    built.enclosure_bounds["enc_depth"] = geometry.enclosure.depth_mm

    with pytest.raises(MesherError, match="do not match.*enc_depth"):
        write_wglink(geometry, tmp_path / "horn.wglink", built_geometry=built)


@pytest.mark.parametrize("supplied_bounds", [False, True])
def test_writer_accepts_flush_enclosure_with_validated_bounds(monkeypatch, tmp_path, supplied_bounds):
    from dataclasses import replace

    _fake_step(monkeypatch)
    geometry = _enclosure_geometry()
    geometry = replace(geometry, enclosure=replace(geometry.enclosure, space_r_mm=0))
    built = _built(geometry) if supplied_bounds else None
    if built is not None:
        assert built.enclosure_bounds["flush_mount"] is True
    result = write_wglink(geometry, tmp_path / "flush.wglink", built_geometry=built)
    assert result.path.is_dir()
    assert read_wglink(result.path)["enclosure"]["plan_type"] == 1


def test_writer_rejects_unvalidated_flush_bounds(monkeypatch, tmp_path):
    from dataclasses import replace
    from hornlab_mesher.builders.enclosure import measure_enclosure_box_bounds

    _fake_step(monkeypatch)
    geometry = _enclosure_geometry()
    geometry = replace(geometry, enclosure=replace(geometry.enclosure, space_r_mm=0))
    built = _built(geometry)
    built.enclosure_bounds = measure_enclosure_box_bounds(
        geometry.inner_points, geometry.enclosure, closed=True)
    assert "flush_mount" not in built.enclosure_bounds
    output = tmp_path / "unvalidated.wglink"
    with pytest.raises(MesherError, match="enclosure_bounds do not match"):
        write_wglink(geometry, output, built_geometry=built)
    assert not output.exists()


def test_writer_rejects_flush_bounds_without_topology_marker(monkeypatch, tmp_path):
    from dataclasses import replace

    _fake_step(monkeypatch)
    geometry = _enclosure_geometry()
    geometry = replace(geometry, enclosure=replace(geometry.enclosure, space_r_mm=0))
    built = _built(geometry)
    assert built.enclosure_bounds.pop("flush_mount") is True
    with pytest.raises(MesherError, match="flush_mount"):
        write_wglink(geometry, tmp_path / "missing-marker.wglink", built_geometry=built)


def test_bundle_is_not_staged_in_a_private_temporary_directory(monkeypatch, tmp_path):
    """``mkdtemp`` gives Windows a non-inheriting owner-only DACL that the
    published files keep; the staging directory becomes the bundle."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("private tempfile staging used for a published bundle")

    monkeypatch.setattr(cad_module.tempfile, "mkdtemp", refuse)
    monkeypatch.setattr(cad_module.tempfile, "NamedTemporaryFile", refuse)
    _fake_step(monkeypatch)
    result = write_wglink(_freestanding(), tmp_path / "horn.wglink")
    assert read_wglink(result.path)["body"]["file"] == "point-grid.json"


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_published_bundle_has_ordinary_umask_modes(tmp_path):
    previous = os.umask(0o022)
    try:
        result = write_wglink(_freestanding(), tmp_path / "horn.wglink")
    finally:
        os.umask(previous)
    assert stat.S_IMODE(os.stat(result.path).st_mode) == 0o755
    for member in result.path.iterdir():
        assert stat.S_IMODE(os.stat(member).st_mode) == 0o644, member.name


def _write_manifest(bundle: Path, manifest: object) -> Path:
    bundle.mkdir()
    (bundle / "wglink.json").write_text(json.dumps(manifest), encoding="utf-8")
    return bundle


def test_reader_rejects_a_bundle_with_no_step_or_point_grid(tmp_path):
    bundle = _write_manifest(
        tmp_path / "empty.wglink",
        {"wglink_version": "1.0", "required_features": [], "files": {}},
    )
    with pytest.raises(MesherError, match="names no body file"):
        read_wglink(bundle)


def test_reader_rejects_a_bundle_missing_its_point_grid(monkeypatch, tmp_path):
    _fake_step(monkeypatch)
    result = write_wglink(_freestanding(), tmp_path / "horn.wglink")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    del manifest["files"]["point-grid.json"]
    result.point_grid_path.unlink()
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(MesherError, match="required file point-grid.json"):
        read_wglink(result.path)


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ([], "must be a JSON object"),
        ({"wglink_version": "1.0", "required_features": "checksummed-files-v1"}, "array of strings"),
    ],
)
def test_reader_type_checks_the_top_level(tmp_path, manifest, message):
    bundle = _write_manifest(tmp_path / "bad.wglink", manifest)
    with pytest.raises(MesherError, match=message):
        read_wglink(bundle, verify_checksums=False)


def test_reader_still_reads_a_format_1_1_bundle_with_its_step(tmp_path):
    """Bundles published before 1.2 carry a STEP body; they stay readable."""

    import hashlib

    bundle = tmp_path / "old.wglink"
    bundle.mkdir()
    step = b"ISO-10303-21;\nEND-ISO-10303-21;\n"
    grid = b"{}\n"
    (bundle / "waveguide.step").write_bytes(step)
    (bundle / "point-grid.json").write_bytes(grid)

    def entry(data: bytes) -> dict:
        return {"sha256": "sha256:" + hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}

    (bundle / "wglink.json").write_text(json.dumps({
        "wglink_version": "1.1",
        "required_features": ["checksummed-files-v1", "link-local-frame-v1"],
        "body": {"file": "waveguide.step", "kind": "solid"},
        "files": {"waveguide.step": entry(step), "point-grid.json": entry(grid)},
    }), encoding="utf-8")

    assert read_wglink(bundle)["body"]["file"] == "waveguide.step"


def test_the_bundle_checks_its_solid_builds_unless_told_it_already_did(monkeypatch, tmp_path):
    stops = _fake_step(monkeypatch)
    write_wglink(_freestanding(), tmp_path / "checked.wglink")
    assert stops == ["solid"]
    stops.clear()
    write_wglink(_freestanding(), tmp_path / "trusted.wglink", verify_solid=False)
    assert stops == []  # no envelope to certify, nothing built


def test_a_solid_that_cannot_be_built_refuses_the_bundle(monkeypatch, tmp_path):
    def broken(_geometry, output_path=None, *, stop=None, **_kwargs):
        raise MesherError("sewing the waveguide surfaces did not produce a closed solid")

    monkeypatch.setattr("hornlab_mesher.cad._write_step_once", broken)
    target = tmp_path / "broken.wglink"
    with pytest.raises(MesherError, match="closed solid"):
        write_wglink(_freestanding(), target)
    assert not target.exists()
