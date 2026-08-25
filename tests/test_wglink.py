from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from hornlab_mesher.builders.enclosure import enclosure_box_bounds
from hornlab_mesher.cad import (
    SOURCE_INTERFACE_FEATURE,
    CadInfo,
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


def _assert_private_publish_lock(target: Path) -> None:
    lock_path = cad_module._publish_lock_path(target)
    state_path = cad_module._private_state_path(target)
    metadata = os.lstat(lock_path)
    assert stat.S_ISREG(metadata.st_mode)
    assert metadata.st_nlink == 1
    assert metadata.st_size == 1
    assert lock_path.read_bytes() == b"\0"
    state_metadata = os.lstat(state_path)
    assert stat.S_ISREG(state_metadata.st_mode)
    assert state_metadata.st_nlink == 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["schema"] == 1
    assert len(state["secret"]) == 64
    assert state["active_token"] is None
    if os.name == "posix":
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_uid == os.getuid()
        assert stat.S_IMODE(state_metadata.st_mode) == 0o600
        assert state_metadata.st_uid == os.getuid()
        root_metadata = os.lstat(lock_path.parent)
        assert stat.S_IMODE(root_metadata.st_mode) == 0o700
        assert root_metadata.st_uid == os.getuid()
    elif sys.platform == "win32":
        cad_module._windows_verify_owner_only_dacl(lock_path.parent, directory=True)
        cad_module._windows_verify_owner_only_dacl(lock_path, directory=False)
        cad_module._windows_verify_owner_only_dacl(state_path, directory=False)


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
        )
    return BuiltGeometry(
        surface_groups={},
        axial_bounds_mm=(0.0, 100.0),
        source_axis="z",
        enclosure_bounds=bounds,
    )


def _fake_step(monkeypatch):
    def write_step(geometry, output_path, *, open_throat=True):
        path = Path(output_path)
        path.write_bytes(b"ISO-10303-21;\nFAKE STEP\nEND-ISO-10303-21;\n")
        return path, CadInfo(
            path=path,
            body="solid",
            n_faces=12,
            volume_mm3=123.5,
            bounding_box_mm=((-70.0, -60.0, -6.0), (70.0, 60.0, 100.0)),
            throat_opened=bool(open_throat),
        )

    monkeypatch.setattr("hornlab_mesher.cad.write_step", write_step)


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


def test_vertical_offset_has_distinct_geometry_and_solver_planes():
    geometry = _freestanding(vertical_offset_mm=37.5)
    datums = derive_datums(geometry, _built(geometry))

    assert datums["WG_AXIS"]["origin_mm"] == [0.0, 37.5, 0.0]
    assert datums["WG_GEOM_MIDPLANE_Y"]["origin_mm"][1] == 37.5
    assert datums["WG_SOLVER_CUT_PLANE_Y"]["origin_mm"][1] == 0.0
    mouth = np.asarray(datums["WG_MOUTH_OUTLINE_INNER"]["points_mm"])
    assert mouth[:, 1] == pytest.approx(geometry.inner_points[:, -1, 1] + 37.5)


def test_point_grid_payload_shares_the_datum_frame(monkeypatch, tmp_path):
    """ONE frame per bundle (plan D1): the grid ships already placed.

    Found on a real Tritonia export: the STEP body and datums sat at
    y = vertical_offset while point-grid.json stayed centred, so an adapter
    lofting the grid and mating against the datums would misassemble by the
    whole offset (80 mm on the reference design).
    """
    _fake_step(monkeypatch)
    geometry = _freestanding(vertical_offset_mm=37.5)
    result = write_wglink(geometry, tmp_path / "horn.wglink")

    grid = json.loads(result.point_grid_path.read_text())
    assert grid["frame"] == "link-local"
    inner = np.asarray(grid["inner_points"])
    assert inner[:, -1, 1] == pytest.approx(geometry.inner_points[:, -1, 1] + 37.5)
    mouth = np.asarray(
        result.manifest["datums"]["WG_MOUTH_OUTLINE_INNER"]["points_mm"]
    )
    assert inner[:, -1, :] == pytest.approx(mouth)
    # The stored geometry itself stays unshifted.
    assert float(np.mean(geometry.inner_points[:, -1, 1])) == pytest.approx(
        float(np.mean(inner[:, -1, 1])) - 37.5
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

    assert result.manifest["wglink_version"] == "1.1"
    assert manifest["wglink_version"] == "1.1"
    assert set(path.name for path in result.path.iterdir()) == {
        "wglink.json",
        "waveguide.step",
        "point-grid.json",
    }
    assert manifest["bundle"] == identity["bundle"]
    assert manifest["design"] == identity["design"]
    assert manifest["export"] == identity["export"]
    assert result.cad_info.path == result.step_path
    grid = json.loads(result.point_grid_path.read_text())
    assert grid["n_phi"] == geometry.inner_points.shape[0]
    assert grid["n_length"] == geometry.inner_points.shape[1]
    # Check points ship placed into the link-local frame (vertical offset 11).
    assert grid["check_points"] == [[1.0, 13.0, 3.0]]
    parameter_entries = {entry["name"]: entry for entry in manifest["parameters"]}
    params = {name: entry["value"] for name, entry in parameter_entries.items()}
    bounds = built.enclosure_bounds
    assert bounds is not None
    assert params["wg_a_1_throat_dia"] == pytest.approx(25.4)
    assert params["wg_a_1_enc_w"] == pytest.approx(bounds["bx1"] - bounds["bx0"])
    assert params["wg_a_1_enc_depth"] == pytest.approx(bounds["enc_depth"])
    assert params["wg_a_1_enc_edge"] == pytest.approx(bounds["clamped_edge"])
    assert params["wg_a_1_enc_x0"] == pytest.approx(bounds["bx0"])
    assert params["wg_a_1_enc_y0"] == pytest.approx(
        bounds["by0"] + geometry.vertical_offset_mm
    )
    assert params["wg_a_1_enc_z_front"] == pytest.approx(bounds["z_front"])

    for name in ("wg_a_1_enc_x0", "wg_a_1_enc_y0", "wg_a_1_enc_z_front"):
        assert parameter_entries[name]["role"] == "interface"
        assert parameter_entries[name]["unit"] == "mm"

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

    with result.step_path.open("ab") as stream:
        stream.write(b"X")
    with pytest.raises(MesherError, match="checksum validation failed.*waveguide.step"):
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


def test_writer_atomically_replaces_live_bundle(monkeypatch, tmp_path):
    _fake_step(monkeypatch)
    target = tmp_path / "horn.wglink"
    write_wglink(_freestanding(), target, identity={"export": {"sequence": 1}})

    result = write_wglink(
        _freestanding(), target, identity={"export": {"sequence": 2}}
    )

    assert read_wglink(target)["export"]["sequence"] == 2
    assert result.path == target
    assert not set(tmp_path.glob(".horn.wglink.*"))


def test_writer_replaces_live_bundle_without_posix_exchange(monkeypatch, tmp_path):
    target = tmp_path / "horn.wglink"
    staging = tmp_path / ".horn.wglink.staged"
    target.mkdir()
    staging.mkdir()
    (target / "generation.txt").write_text("old", encoding="utf-8")
    (staging / "generation.txt").write_text("new", encoding="utf-8")

    cad_module._publish_bundle_without_exchange(staging, target)

    assert (target / "generation.txt").read_text(encoding="utf-8") == "new"
    assert not staging.exists()
    assert not list(tmp_path.glob("*.previous"))
    assert not cad_module._transaction_path(target).exists()
    _assert_private_publish_lock(target)


def test_non_posix_directory_replacement_restores_live_bundle_on_failure(
    monkeypatch, tmp_path
):
    target = tmp_path / "horn.wglink"
    staging = tmp_path / ".horn.wglink.staged"
    target.mkdir()
    staging.mkdir()
    (target / "generation.txt").write_text("old", encoding="utf-8")
    (staging / "generation.txt").write_text("new", encoding="utf-8")
    original_rename = cad_module._rename_open_directory

    def fail_staged_publish(source, destination_root, name):
        if source.path == staging and destination_root.path / name == target:
            raise OSError("injected publish failure")
        return original_rename(source, destination_root, name)

    monkeypatch.setattr(cad_module, "_rename_open_directory", fail_staged_publish)

    with pytest.raises(OSError, match="injected publish failure"):
        cad_module._publish_bundle_without_exchange(staging, target)

    assert (target / "generation.txt").read_text(encoding="utf-8") == "old"
    assert (staging / "generation.txt").read_text(encoding="utf-8") == "new"
    assert not list(tmp_path.glob("*.previous"))


def test_failed_recovery_keeps_token_for_later_retry(monkeypatch, tmp_path):
    target = tmp_path / "horn.wglink"
    staging = tmp_path / ".horn.wglink.staged"
    target.mkdir()
    staging.mkdir()
    (target / "generation.txt").write_text("old", encoding="utf-8")
    (staging / "generation.txt").write_text("new", encoding="utf-8")
    original_rename = cad_module._rename_open_directory
    moves_blocked = True

    def fail_moves_to_live_name(source, destination_root, name):
        destination = destination_root.path / name
        if (
            moves_blocked
            and destination == target
            and (source.path == staging or source.path.name.endswith(".previous"))
        ):
            raise OSError("injected sharing violation")
        return original_rename(source, destination_root, name)

    monkeypatch.setattr(
        cad_module, "_rename_open_directory", fail_moves_to_live_name
    )

    with pytest.raises(OSError, match="injected sharing violation"):
        cad_module._publish_bundle_without_exchange(staging, target)

    state_path = cad_module._private_state_path(target)
    failed_state = json.loads(state_path.read_text(encoding="utf-8"))
    record = json.loads(
        cad_module._transaction_record_path(target).read_text(encoding="utf-8")
    )
    backup = tmp_path / record["backup"]
    assert failed_state["active_token"] == record["token"]
    assert not target.exists()
    assert not staging.exists()
    assert (backup / "generation.txt").read_text(encoding="utf-8") == "old"

    moves_blocked = False
    with cad_module._bundle_publish_lock(target) as state:
        cad_module._recover_directory_replacement(target, state)

    assert (target / "generation.txt").read_text(encoding="utf-8") == "old"
    assert not backup.exists()
    assert not cad_module._transaction_path(target).exists()
    _assert_private_publish_lock(target)


def test_non_posix_publication_recovers_after_process_dies_between_renames(tmp_path):
    target = tmp_path / "horn.wglink"
    interrupted_staging = tmp_path / ".horn.wglink.interrupted"
    next_staging = tmp_path / ".horn.wglink.next"
    target.mkdir()
    interrupted_staging.mkdir()
    next_staging.mkdir()
    (target / "generation.txt").write_text("old", encoding="utf-8")
    (interrupted_staging / "generation.txt").write_text("interrupted", encoding="utf-8")
    (next_staging / "generation.txt").write_text("next", encoding="utf-8")
    crash_script = """
import os
import sys
from pathlib import Path
from hornlab_mesher import cad

target = Path(sys.argv[1])
staging = Path(sys.argv[2])
original_rename = cad._rename_open_directory

def die_after_first_rename(source, destination_root, name):
    original_path = source.path
    result = original_rename(source, destination_root, name)
    if original_path == target and name.endswith(".previous"):
        os._exit(91)
    return result

cad._rename_open_directory = die_after_first_rename
cad._publish_bundle_without_exchange(staging, target)
"""

    crashed = subprocess.run(
        [sys.executable, "-c", crash_script, str(target), str(interrupted_staging)],
        check=False,
    )

    assert crashed.returncode == 91
    assert not target.exists()
    assert interrupted_staging.exists()
    record = json.loads(
        cad_module._transaction_record_path(target).read_text(encoding="utf-8")
    )
    assert (tmp_path / record["backup"]).exists()

    with cad_module._bundle_publish_lock(target) as state:
        cad_module._recover_directory_replacement(target, state)

    assert (target / "generation.txt").read_text(encoding="utf-8") == "old"
    assert not interrupted_staging.exists()
    assert not list(tmp_path.glob("*.previous"))
    assert not cad_module._transaction_path(target).exists()

    cad_module._publish_bundle_without_exchange(next_staging, target)

    assert (target / "generation.txt").read_text(encoding="utf-8") == "next"
    assert not interrupted_staging.exists()
    assert not next_staging.exists()
    assert not list(tmp_path.glob("*.previous"))
    _assert_private_publish_lock(target)


@pytest.mark.parametrize("replaced_role", ["backup", "staging"])
def test_recovery_rejects_replaced_owned_directory_after_first_rename(
    tmp_path, replaced_role
):
    target = tmp_path / "horn.wglink"
    staging = tmp_path / ".horn.wglink.interrupted"
    target.mkdir()
    staging.mkdir()
    (target / "generation.txt").write_text("old", encoding="utf-8")
    (staging / "generation.txt").write_text("new", encoding="utf-8")
    crash_script = """
import os
import sys
from pathlib import Path
from hornlab_mesher import cad

target = Path(sys.argv[1])
staging = Path(sys.argv[2])
original_rename = cad._rename_open_directory

def die_after_first_rename(source, destination_root, name):
    original_path = source.path
    result = original_rename(source, destination_root, name)
    if original_path == target and name.endswith(".previous"):
        os._exit(94)
    return result

cad._rename_open_directory = die_after_first_rename
cad._publish_bundle_without_exchange(staging, target)
"""
    crashed = subprocess.run(
        [sys.executable, "-c", crash_script, str(target), str(staging)],
        check=False,
    )
    assert crashed.returncode == 94
    record = json.loads(
        cad_module._transaction_record_path(target).read_text(encoding="utf-8")
    )
    backup = tmp_path / record["backup"]
    replaced = backup if replaced_role == "backup" else staging
    preserved = tmp_path / f"preserved-real-{replaced_role}"
    replaced.replace(preserved)
    replaced.mkdir()
    (replaced / "attacker.txt").write_text("must survive", encoding="utf-8")

    with cad_module._bundle_publish_lock(target) as state:
        with pytest.raises(MesherError, match="identity changed"):
            cad_module._recover_directory_replacement(target, state)

    assert not target.exists()
    assert (replaced / "attacker.txt").read_text(encoding="utf-8") == "must survive"
    expected = "old" if replaced_role == "backup" else "new"
    assert (preserved / "generation.txt").read_text(encoding="utf-8") == expected
    untouched = staging if replaced_role == "backup" else backup
    untouched_expected = "new" if replaced_role == "backup" else "old"
    assert (untouched / "generation.txt").read_text(encoding="utf-8") == untouched_expected


def test_recovery_never_deletes_replaced_backup_after_second_rename(tmp_path):
    target = tmp_path / "horn.wglink"
    staging = tmp_path / ".horn.wglink.interrupted"
    target.mkdir()
    staging.mkdir()
    (target / "generation.txt").write_text("old", encoding="utf-8")
    (staging / "generation.txt").write_text("new", encoding="utf-8")
    crash_script = """
import os
import sys
from pathlib import Path
from hornlab_mesher import cad

target = Path(sys.argv[1])
staging = Path(sys.argv[2])
original_rename = cad._rename_open_directory

def die_after_second_rename(source, destination_root, name):
    original_path = source.path
    result = original_rename(source, destination_root, name)
    if original_path == staging and destination_root.path / name == target:
        os._exit(95)
    return result

cad._rename_open_directory = die_after_second_rename
cad._publish_bundle_without_exchange(staging, target)
"""
    crashed = subprocess.run(
        [sys.executable, "-c", crash_script, str(target), str(staging)],
        check=False,
    )
    assert crashed.returncode == 95
    record = json.loads(
        cad_module._transaction_record_path(target).read_text(encoding="utf-8")
    )
    backup = tmp_path / record["backup"]
    preserved_backup = tmp_path / "preserved-real-backup"
    backup.replace(preserved_backup)
    backup.mkdir()
    (backup / "attacker.txt").write_text("must not be deleted", encoding="utf-8")

    with cad_module._bundle_publish_lock(target) as state:
        with pytest.raises(MesherError, match="identity changed"):
            cad_module._recover_directory_replacement(target, state)

    assert (target / "generation.txt").read_text(encoding="utf-8") == "new"
    assert (backup / "attacker.txt").read_text(encoding="utf-8") == "must not be deleted"
    assert (preserved_backup / "generation.txt").read_text(encoding="utf-8") == "old"


def test_recovery_never_deletes_backup_path_swapped_after_validation(
    monkeypatch, tmp_path
):
    target = tmp_path / "horn.wglink"
    staging = tmp_path / ".horn.wglink.interrupted"
    target.mkdir()
    staging.mkdir()
    (target / "generation.txt").write_text("old", encoding="utf-8")
    (staging / "generation.txt").write_text("new", encoding="utf-8")

    with cad_module._bundle_publish_lock(target) as state:
        token = "a" * 32
        backup = tmp_path / f".{target.name}.publish.{token}.previous"
        staging_identity = cad_module._directory_identity(staging)
        backup_identity = cad_module._directory_identity(target)
        cad_module._set_active_transaction(target, state, token)
        record = cad_module._write_transaction_record(
            target,
            staging,
            backup,
            token,
            staging_identity,
            backup_identity,
            state,
        )
        cad_module._install_transaction_marker(
            target, target, token, "backup", backup_identity, state
        )
        cad_module._install_transaction_marker(
            staging, target, token, "staging", staging_identity, state
        )
        cad_module._update_transaction_phase(target, record, state, "marked")
        cad_module._rename_owned_directory(
            target,
            backup,
            backup_identity,
            marker=(target, token, "backup", state),
        )
        cad_module._rename_owned_directory(
            staging,
            target,
            staging_identity,
            marker=(target, token, "staging", state),
        )

        preserved_backup = tmp_path / "preserved-real-backup"
        decoy = tmp_path / "customer-data"
        decoy.mkdir()
        (decoy / "must-survive.txt").write_text("unrelated", encoding="utf-8")
        original_rmtree = cad_module.shutil.rmtree

        def swap_at_path_based_delete(path, *args, **kwargs):
            if Path(path) == backup:
                backup.replace(preserved_backup)
                decoy.replace(backup)
            return original_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(cad_module.shutil, "rmtree", swap_at_path_based_delete)
        cad_module._recover_directory_replacement(target, state)

    assert (target / "generation.txt").read_text(encoding="utf-8") == "new"
    assert (decoy / "must-survive.txt").read_text(encoding="utf-8") == "unrelated"
    assert not preserved_backup.exists()


def test_owned_rename_mutates_open_identity_not_swapped_path(monkeypatch, tmp_path):
    source = tmp_path / "authenticated"
    destination = tmp_path / "published"
    decoy = tmp_path / "customer-data"
    preserved = tmp_path / "pinned-authenticated-object"
    source.mkdir()
    decoy.mkdir()
    (source / "owned.txt").write_text("transaction", encoding="utf-8")
    (decoy / "must-survive.txt").write_text("unrelated", encoding="utf-8")
    expected = cad_module._directory_identity(source)
    original_new_handle = cad_module._new_directory_handle

    def inject_swap_after_handle_validation(
        path, *, rename_source=False, child_access=False
    ):
        handle = original_new_handle(
            path,
            rename_source=rename_source,
            child_access=child_access,
        )
        if path == source and rename_source:
            original_rename_into = handle.rename_into

            if sys.platform == "win32":
                # The Windows handle is opened without FILE_SHARE_DELETE, so
                # the OS itself pins the path-object binding: the attacker's
                # path swap cannot even start while the validated handle is
                # open.  Assert the sharing violation, then let the real
                # handle-based rename proceed.
                def rename_pinned_object(destination_root, name):
                    with pytest.raises(PermissionError):
                        source.replace(preserved)
                    original_rename_into(destination_root, name)

            else:

                def rename_pinned_object(destination_root, name):
                    source.replace(preserved)
                    decoy.replace(source)
                    preserved.replace(destination_root.path / name)
                    handle.path = destination_root.path / name

            handle.rename_into = rename_pinned_object
        return handle

    monkeypatch.setattr(
        cad_module, "_new_directory_handle", inject_swap_after_handle_validation
    )

    cad_module._rename_owned_directory(source, destination, expected)

    assert (destination / "owned.txt").read_text(encoding="utf-8") == "transaction"
    if sys.platform == "win32":
        assert not source.exists()
        assert (decoy / "must-survive.txt").read_text(encoding="utf-8") == "unrelated"
    else:
        assert (source / "must-survive.txt").read_text(encoding="utf-8") == "unrelated"


def test_recovery_resumes_authenticated_quarantine_cleanup(monkeypatch, tmp_path):
    target = tmp_path / "horn.wglink"
    staging = tmp_path / ".horn.wglink.interrupted"
    target.mkdir()
    staging.mkdir()
    (target / "generation.txt").write_text("old", encoding="utf-8")
    (staging / "generation.txt").write_text("new", encoding="utf-8")

    with cad_module._bundle_publish_lock(target) as state:
        token = "b" * 32
        backup = tmp_path / f".{target.name}.publish.{token}.previous"
        staging_identity = cad_module._directory_identity(staging)
        backup_identity = cad_module._directory_identity(target)
        cad_module._set_active_transaction(target, state, token)
        record = cad_module._write_transaction_record(
            target,
            staging,
            backup,
            token,
            staging_identity,
            backup_identity,
            state,
        )
        cad_module._install_transaction_marker(
            target, target, token, "backup", backup_identity, state
        )
        cad_module._install_transaction_marker(
            staging, target, token, "staging", staging_identity, state
        )
        cad_module._update_transaction_phase(target, record, state, "marked")
        cad_module._rename_owned_directory(
            target,
            backup,
            backup_identity,
            marker=(target, token, "backup", state),
        )
        cad_module._rename_owned_directory(
            staging,
            target,
            staging_identity,
            marker=(target, token, "staging", state),
        )

        original_remove_contents = cad_module._remove_directory_contents
        cleanup = cad_module._transaction_cleanup_path(target, "backup")
        fail_cleanup = True

        def interrupt_quarantine_cleanup(path):
            if fail_cleanup and path == cleanup:
                raise OSError("injected cleanup interruption")
            return original_remove_contents(path)

        monkeypatch.setattr(
            cad_module, "_remove_directory_contents", interrupt_quarantine_cleanup
        )
        with pytest.raises(OSError, match="injected cleanup interruption"):
            cad_module._recover_directory_replacement(target, state)

        assert state["active_token"] == token
        assert cleanup.is_dir()
        assert not backup.exists()
        assert (target / "generation.txt").read_text(encoding="utf-8") == "new"

        fail_cleanup = False
        cad_module._recover_directory_replacement(target, state)

    assert (target / "generation.txt").read_text(encoding="utf-8") == "new"
    assert not cleanup.exists()
    assert not cad_module._transaction_path(target).exists()
    _assert_private_publish_lock(target)


def test_windows_directory_identity_distinguishes_full_128_bit_file_id():
    shared_low_bits = bytes.fromhex("0011223344556677")
    first = cad_module._windows_file_identity(
        42, shared_low_bits + bytes.fromhex("8899aabbccddeeff")
    )
    second = cad_module._windows_file_identity(
        42, shared_low_bits + bytes.fromhex("8899aabbccddee00")
    )

    assert first != second
    assert len(bytes.fromhex(first["file_id"])) == 16
    assert len(bytes.fromhex(second["file_id"])) == 16


def test_publication_fails_closed_when_identity_backend_is_unsupported(
    monkeypatch, tmp_path
):
    target = tmp_path / "horn.wglink"
    staging = tmp_path / ".horn.wglink.staged"
    target.mkdir()
    staging.mkdir()
    (target / "generation.txt").write_text("old", encoding="utf-8")
    (staging / "generation.txt").write_text("new", encoding="utf-8")
    original_new_handle = cad_module._new_directory_handle

    def reject_staging_identity(
        path, *, rename_source=False, child_access=False
    ):
        if path == staging:
            raise MesherError("filesystem has no stable 128-bit directory ID")
        return original_new_handle(
            path,
            rename_source=rename_source,
            child_access=child_access,
        )

    monkeypatch.setattr(cad_module, "_new_directory_handle", reject_staging_identity)

    with pytest.raises(MesherError, match="no stable 128-bit directory ID"):
        cad_module._publish_bundle_without_exchange(staging, target)

    assert (target / "generation.txt").read_text(encoding="utf-8") == "old"
    assert (staging / "generation.txt").read_text(encoding="utf-8") == "new"
    assert not cad_module._transaction_path(target).exists()


def test_non_posix_publication_serializes_concurrent_processes(tmp_path):
    target = tmp_path / "horn.wglink"
    staging_a = tmp_path / ".horn.wglink.writer-a"
    staging_b = tmp_path / ".horn.wglink.writer-b"
    backed_up = tmp_path / "writer-a-backed-up"
    release_a = tmp_path / "release-writer-a"
    writer_b_ready = tmp_path / "writer-b-ready"
    output_lock_decoy = tmp_path / ".horn.wglink.publish.lock"
    target.mkdir()
    staging_a.mkdir()
    staging_b.mkdir()
    (target / "generation.txt").write_text("old", encoding="utf-8")
    (staging_a / "generation.txt").write_text("writer-a", encoding="utf-8")
    (staging_b / "generation.txt").write_text("writer-b", encoding="utf-8")
    writer_a_script = """
import sys
import time
from pathlib import Path
from hornlab_mesher import cad

target, staging, backed_up, release = map(Path, sys.argv[1:])
original_rename = cad._rename_open_directory

def pause_after_first_rename(source, destination_root, name):
    original_path = source.path
    result = original_rename(source, destination_root, name)
    if original_path == target and name.endswith(".previous"):
        backed_up.write_text("ready", encoding="utf-8")
        while not release.exists():
            time.sleep(0.01)
    return result

cad._rename_open_directory = pause_after_first_rename
cad._publish_bundle_without_exchange(staging, target)
"""
    writer_b_script = """
import sys
from pathlib import Path
from hornlab_mesher import cad

target, staging, ready = map(Path, sys.argv[1:])
ready.write_text("ready", encoding="utf-8")
cad._publish_bundle_without_exchange(staging, target)
"""
    writer_a = subprocess.Popen(
        [
            sys.executable,
            "-c",
            writer_a_script,
            str(target),
            str(staging_a),
            str(backed_up),
            str(release_a),
        ]
    )
    writer_b = None
    try:
        for _ in range(500):
            if backed_up.exists():
                break
            assert writer_a.poll() is None
            time.sleep(0.01)
        assert backed_up.exists()
        assert not target.exists()

        # The old output-directory lock pathname is attacker-replaceable and
        # must have no bearing on the live per-user mutex.
        output_lock_decoy.write_bytes(b"replacement while writer A holds lock")

        writer_b = subprocess.Popen(
            [
                sys.executable,
                "-c",
                writer_b_script,
                str(target),
                str(staging_b),
                str(writer_b_ready),
            ]
        )
        for _ in range(500):
            if writer_b_ready.exists():
                break
            assert writer_b.poll() is None
            time.sleep(0.01)
        assert writer_b_ready.exists()
        time.sleep(0.1)
        assert writer_b.poll() is None

        release_a.write_text("release", encoding="utf-8")
        assert writer_a.wait(timeout=5) == 0
        assert writer_b.wait(timeout=5) == 0
    finally:
        release_a.touch(exist_ok=True)
        if writer_a.poll() is None:
            writer_a.kill()
            writer_a.wait()
        if writer_b is not None and writer_b.poll() is None:
            writer_b.kill()
            writer_b.wait()

    assert (target / "generation.txt").read_text(encoding="utf-8") == "writer-b"
    assert not staging_a.exists()
    assert not staging_b.exists()
    assert not list(tmp_path.glob("*.previous"))
    assert not cad_module._transaction_path(target).exists()
    assert output_lock_decoy.read_bytes() == b"replacement while writer A holds lock"
    _assert_private_publish_lock(target)


def test_non_posix_publication_preserves_unowned_matching_directory(tmp_path):
    target = tmp_path / "horn.wglink"
    staging = tmp_path / ".horn.wglink.staged"
    unrelated = tmp_path / ".horn.wglink.customer.previous"
    target.mkdir()
    staging.mkdir()
    unrelated.mkdir()
    (target / "generation.txt").write_text("old", encoding="utf-8")
    (staging / "generation.txt").write_text("new", encoding="utf-8")
    (unrelated / "keep.txt").write_text("owned elsewhere", encoding="utf-8")

    cad_module._publish_bundle_without_exchange(staging, target)

    assert (target / "generation.txt").read_text(encoding="utf-8") == "new"
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "owned elsewhere"
    assert not cad_module._transaction_path(target).exists()


def test_non_posix_publication_rejects_unowned_state_when_target_is_missing(tmp_path):
    target = tmp_path / "horn.wglink"
    staging = tmp_path / ".horn.wglink.staged"
    unrelated = tmp_path / ".horn.wglink.customer.previous"
    staging.mkdir()
    unrelated.mkdir()
    (staging / "generation.txt").write_text("new", encoding="utf-8")
    (unrelated / "keep.txt").write_text("owned elsewhere", encoding="utf-8")

    with pytest.raises(MesherError, match="unowned.*recovery was refused"):
        cad_module._publish_bundle_without_exchange(staging, target)

    assert staging.exists()
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "owned elsewhere"


def test_non_posix_publication_rejects_untrusted_transaction_record(tmp_path):
    target = tmp_path / "horn.wglink"
    staging = tmp_path / ".horn.wglink.staged"
    target.mkdir()
    staging.mkdir()
    (target / "generation.txt").write_text("old", encoding="utf-8")
    (staging / "generation.txt").write_text("new", encoding="utf-8")
    transaction_path = cad_module._transaction_path(target)
    cad_module._create_transaction_directory(transaction_path)
    record_path = cad_module._transaction_record_path(target)
    token = "0" * 32
    payload = json.dumps(
        {
            "schema": cad_module._TRANSACTION_SCHEMA,
            "target": target.name,
            "token": token,
            "staging": staging.name,
            "backup": f".{target.name}.publish.{token}.previous",
            "staging_identity": cad_module._directory_identity(staging),
            "backup_identity": cad_module._directory_identity(target),
            "phase": "marked",
            "committed_role": None,
            "mac": "0" * 64,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    cad_module._atomic_write_private_payload(
        record_path,
        payload,
        replace=False,
    )

    with pytest.raises(MesherError, match="invalid or unauthenticated"):
        cad_module._publish_bundle_without_exchange(staging, target)

    assert (target / "generation.txt").read_text(encoding="utf-8") == "old"
    assert (staging / "generation.txt").read_text(encoding="utf-8") == "new"
    assert record_path.exists()


def test_non_posix_publication_rejects_chosen_key_forged_record(tmp_path):
    target = tmp_path / "horn.wglink"
    staging = tmp_path / ".horn.wglink.next"
    target.mkdir()
    staging.mkdir()
    (target / "generation.txt").write_text("old", encoding="utf-8")
    (staging / "generation.txt").write_text("new", encoding="utf-8")

    # This is the former output-directory lock location.  A publisher must not
    # trust either its attacker-chosen key or a record MACed with that key.
    chosen_key = b"A" * 32
    (tmp_path / ".horn.wglink.publish.lock").write_bytes(chosen_key)
    token = "1" * 32
    victim = tmp_path / f".horn.wglink.publish.{token}.previous"
    victim.mkdir()
    (victim / "keep.txt").write_text("do not delete", encoding="utf-8")
    record = {
        "schema": cad_module._TRANSACTION_SCHEMA,
        "target": target.name,
        "token": token,
        "staging": staging.name,
        "backup": victim.name,
        "staging_identity": cad_module._directory_identity(staging),
        "backup_identity": cad_module._directory_identity(victim),
        "phase": "marked",
        "committed_role": None,
    }
    record["mac"] = cad_module._transaction_mac(chosen_key, record)
    cad_module._create_transaction_directory(cad_module._transaction_path(target))
    cad_module._atomic_write_private_payload(
        cad_module._transaction_record_path(target),
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        replace=False,
    )

    with pytest.raises(MesherError, match="invalid or unauthenticated"):
        cad_module._publish_bundle_without_exchange(staging, target)

    assert (target / "generation.txt").read_text(encoding="utf-8") == "old"
    assert (staging / "generation.txt").read_text(encoding="utf-8") == "new"
    assert (victim / "keep.txt").read_text(encoding="utf-8") == "do not delete"


def test_non_posix_publication_rejects_captured_record_replay(tmp_path):
    target = tmp_path / "horn.wglink"
    staging = tmp_path / ".horn.wglink.writer-a"
    paused = tmp_path / "writer-a-paused"
    release = tmp_path / "release-writer-a"
    target.mkdir()
    staging.mkdir()
    (target / "generation.txt").write_text("old", encoding="utf-8")
    (staging / "generation.txt").write_text("writer-a", encoding="utf-8")
    writer_script = """
import sys
import time
from pathlib import Path
from hornlab_mesher import cad

target, staging, paused, release = map(Path, sys.argv[1:])
original_rename = cad._rename_open_directory

def pause_after_first_rename(source, destination_root, name):
    original_path = source.path
    result = original_rename(source, destination_root, name)
    if original_path == target and name.endswith(".previous"):
        paused.write_text("ready", encoding="utf-8")
        while not release.exists():
            time.sleep(0.01)
    return result

cad._rename_open_directory = pause_after_first_rename
cad._publish_bundle_without_exchange(staging, target)
"""
    writer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            writer_script,
            str(target),
            str(staging),
            str(paused),
            str(release),
        ]
    )
    try:
        for _ in range(500):
            if paused.exists():
                break
            assert writer.poll() is None
            time.sleep(0.01)
        assert paused.exists()
        captured = cad_module._transaction_record_path(target).read_bytes()
        captured_record = json.loads(captured.decode("utf-8"))
        release.write_text("release", encoding="utf-8")
        assert writer.wait(timeout=5) == 0
    finally:
        release.touch(exist_ok=True)
        if writer.poll() is None:
            writer.kill()
            writer.wait()

    victim = tmp_path / captured_record["backup"]
    victim.mkdir()
    (victim / "keep.txt").write_text("recreated after publish", encoding="utf-8")
    cad_module._create_transaction_directory(cad_module._transaction_path(target))
    cad_module._atomic_write_private_payload(
        cad_module._transaction_record_path(target), captured, replace=False
    )
    next_staging = tmp_path / ".horn.wglink.next"
    next_staging.mkdir()
    (next_staging / "generation.txt").write_text("next", encoding="utf-8")

    with pytest.raises(MesherError, match="stale or replayed"):
        cad_module._publish_bundle_without_exchange(next_staging, target)

    assert (target / "generation.txt").read_text(encoding="utf-8") == "writer-a"
    assert (next_staging / "generation.txt").read_text(encoding="utf-8") == "next"
    assert (victim / "keep.txt").read_text(
        encoding="utf-8"
    ) == "recreated after publish"


@pytest.mark.parametrize("crash_mode", ["before", "partial"])
def test_non_posix_publication_recovers_crash_while_writing_record(
    tmp_path, crash_mode
):
    target = tmp_path / "horn.wglink"
    interrupted = tmp_path / ".horn.wglink.interrupted"
    next_staging = tmp_path / ".horn.wglink.next"
    target.mkdir()
    interrupted.mkdir()
    next_staging.mkdir()
    (target / "generation.txt").write_text("old", encoding="utf-8")
    (interrupted / "generation.txt").write_text("interrupted", encoding="utf-8")
    (next_staging / "generation.txt").write_text("next", encoding="utf-8")
    crash_script = """
import os
import sys
from pathlib import Path
from hornlab_mesher import cad

target, staging = map(Path, sys.argv[1:3])
mode = sys.argv[3]
original_write_record = cad._write_transaction_record

def crash_while_writing_record(*args, **kwargs):
    def crashing_write(descriptor, payload):
        if mode == "partial":
            os.write(descriptor, payload[: max(1, len(payload) // 2)])
        os._exit(92)
    cad._write_all = crashing_write
    return original_write_record(*args, **kwargs)

cad._write_transaction_record = crash_while_writing_record
cad._publish_bundle_without_exchange(staging, target)
"""
    crashed = subprocess.run(
        [
            sys.executable,
            "-c",
            crash_script,
            str(target),
            str(interrupted),
            crash_mode,
        ],
        check=False,
    )

    assert crashed.returncode == 92
    assert (target / "generation.txt").read_text(encoding="utf-8") == "old"
    assert interrupted.exists()
    transaction_path = cad_module._transaction_path(target)
    assert transaction_path.is_dir()
    assert not cad_module._transaction_record_path(target).exists()

    cad_module._publish_bundle_without_exchange(next_staging, target)

    assert (target / "generation.txt").read_text(encoding="utf-8") == "next"
    assert interrupted.exists()
    assert not next_staging.exists()
    assert not transaction_path.exists()
    _assert_private_publish_lock(target)


def test_coordination_write_all_retries_short_writes(monkeypatch, tmp_path):
    path = tmp_path / "coordination.bin"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    original_write = os.write
    calls = 0

    def short_write(fd, payload):
        nonlocal calls
        calls += 1
        return original_write(fd, payload[:3])

    monkeypatch.setattr(cad_module.os, "write", short_write)
    try:
        cad_module._write_all(descriptor, b"complete coordination payload")
    finally:
        os.close(descriptor)

    assert calls > 1
    assert path.read_bytes() == b"complete coordination payload"


def test_private_state_replacements_use_write_through_namespace_move(
    monkeypatch, tmp_path
):
    state_path = tmp_path / "state.json"
    moves: list[tuple[Path, Path, bool]] = []
    original_move = cad_module._move_path_write_through

    def capture_move(source, destination, *, replace):
        moves.append((source, destination, replace))
        return original_move(source, destination, replace=replace)

    monkeypatch.setattr(cad_module, "_move_path_write_through", capture_move)

    cad_module._atomic_write_private_payload(state_path, b"first", replace=False)
    cad_module._atomic_write_private_payload(state_path, b"second", replace=True)

    assert state_path.read_bytes() == b"second"
    assert [(destination, replace) for _, destination, replace in moves] == [
        (state_path, False),
        (state_path, True),
    ]
    assert all(source.parent == state_path.parent for source, _, _ in moves)


def test_private_state_update_survives_partial_write_crash(tmp_path):
    target = tmp_path / "horn.wglink"
    interrupted = tmp_path / ".horn.wglink.interrupted"
    next_staging = tmp_path / ".horn.wglink.next"
    target.mkdir()
    interrupted.mkdir()
    next_staging.mkdir()
    (target / "generation.txt").write_text("old", encoding="utf-8")
    (interrupted / "generation.txt").write_text("interrupted", encoding="utf-8")
    (next_staging / "generation.txt").write_text("next", encoding="utf-8")
    with cad_module._bundle_publish_lock(target):
        pass
    state_path = cad_module._private_state_path(target)
    original_state = state_path.read_bytes()
    crash_script = """
import os
import sys
from pathlib import Path
from hornlab_mesher import cad

target, staging = map(Path, sys.argv[1:])

def crash_during_state_write(descriptor, payload):
    os.write(descriptor, payload[: max(1, len(payload) // 2)])
    os._exit(93)

cad._write_all = crash_during_state_write
cad._publish_bundle_without_exchange(staging, target)
"""

    crashed = subprocess.run(
        [sys.executable, "-c", crash_script, str(target), str(interrupted)],
        check=False,
    )

    assert crashed.returncode == 93
    assert state_path.read_bytes() == original_state
    assert cad_module._private_state_temps(state_path)
    assert (target / "generation.txt").read_text(encoding="utf-8") == "old"

    cad_module._publish_bundle_without_exchange(next_staging, target)

    assert (target / "generation.txt").read_text(encoding="utf-8") == "next"
    assert interrupted.exists()
    assert not cad_module._private_state_temps(state_path)
    _assert_private_publish_lock(target)


def test_publish_lock_rejects_hardlink_redirection(tmp_path):
    target = tmp_path / "horn.wglink"
    staging = tmp_path / ".horn.wglink.staged"
    target.mkdir()
    staging.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("do not touch", encoding="utf-8")
    if os.name == "posix":
        victim.chmod(0o600)
    os.link(victim, cad_module._publish_lock_path(target))

    with pytest.raises(MesherError, match="exactly one link"):
        cad_module._publish_bundle_without_exchange(staging, target)

    assert victim.read_text(encoding="utf-8") == "do not touch"
    assert target.exists()
    assert staging.exists()


def test_windows_private_lock_root_never_reacls_existing_ancestors(
    monkeypatch, tmp_path
):
    app_root = tmp_path / "HornLab" / "WaveguideMesher"
    app_root.mkdir(parents=True)
    applied: list[tuple[Path, bool]] = []
    verified: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        cad_module,
        "_windows_apply_owner_only_dacl",
        lambda path, *, directory: applied.append((path, directory)),
    )
    monkeypatch.setattr(
        cad_module,
        "_windows_verify_owner_only_dacl",
        lambda path, *, directory: verified.append((path, directory)),
    )

    root = cad_module._windows_private_lock_root(tmp_path)
    assert root == app_root / "lock-state-v1"
    assert applied == [(root, True)]
    assert verified == [(root, True)]

    assert cad_module._windows_private_lock_root(tmp_path) == root
    assert applied == [(root, True)]
    assert verified == [(root, True), (root, True)]


class _FakeWindowsDll:
    """A stand-in for ``ctypes.WinDLL`` results; attributes are the fakes."""

    def __init__(self, **functions):
        for name, implementation in functions.items():
            setattr(self, name, implementation)


def _install_fake_windll(monkeypatch, dlls):
    import ctypes

    monkeypatch.setattr(
        ctypes,
        "WinDLL",
        lambda name, *args, **kwargs: dlls[name],
        raising=False,
    )


def _set_pointer(byref_argument, pointer_type, value):
    import ctypes

    ctypes.cast(byref_argument, ctypes.POINTER(pointer_type))[0] = value


def _fake_verify_security_environment(monkeypatch, *, stamped_owner_marker):
    """Drive the real ``_windows_verify_owner_only_dacl`` ctypes code.

    SIDs are represented by their pointer values; the fake ``EqualSid``
    compares those values.  The DACL and its single ACE are real memory so
    the production casts and ``from_address`` reads operate on the layout
    Windows would hand back.  ``stamped_owner_marker`` selects which SID the
    fake ``GetNamedSecurityInfoW`` reports as the object owner.
    """
    import ctypes
    from ctypes import wintypes

    file_all_access = 0x001F01FF
    acl_bytes = bytearray(24)
    acl_bytes[0] = 2  # AclRevision
    acl_bytes[2:4] = (24).to_bytes(2, "little")  # AclSize
    acl_bytes[4:6] = (1).to_bytes(2, "little")  # AceCount
    acl_bytes[8] = 0  # ACCESS_ALLOWED_ACE_TYPE
    acl_bytes[9] = 0x03  # OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE
    acl_bytes[10:12] = (16).to_bytes(2, "little")  # AceSize
    acl_bytes[12:16] = file_all_access.to_bytes(4, "little")  # Mask
    acl_buffer = ctypes.create_string_buffer(bytes(acl_bytes), len(acl_bytes))
    acl_address = ctypes.addressof(acl_buffer)
    descriptor_buffer = ctypes.create_string_buffer(16)

    ace_sid = acl_address + 16  # the SID embedded in the single ACE
    token_user_sid = ace_sid  # the DACL grants FILE_ALL_ACCESS to TokenUser
    token_owner_sid = 0xAD314  # BUILTIN\Administrators for an admin token
    third_party_sid = 0xBAD514
    owners = {
        "token_user": token_user_sid,
        "token_owner": token_owner_sid,
        "third_party": third_party_sid,
    }
    stamped_owner = owners[stamped_owner_marker]

    def get_named_security_info(
        path, object_type, information, owner, group, dacl, sacl, descriptor
    ):
        assert object_type == 1  # SE_FILE_OBJECT
        assert information == 0x1 | 0x4  # OWNER | DACL
        _set_pointer(owner, ctypes.c_void_p, stamped_owner)
        _set_pointer(dacl, ctypes.c_void_p, acl_address)
        _set_pointer(descriptor, ctypes.c_void_p, ctypes.addressof(descriptor_buffer))
        return 0

    def equal_sid(first, second):
        def sid_value(sid):
            return sid if isinstance(sid, int) else int(sid.value or 0)

        return 1 if sid_value(first) == sid_value(second) else 0

    def get_security_descriptor_control(descriptor, control, revision):
        _set_pointer(control, wintypes.WORD, 0x1000)  # SE_DACL_PROTECTED
        _set_pointer(revision, wintypes.DWORD, 1)
        return 1

    def get_ace(dacl, index, ace):
        assert index == 0
        _set_pointer(ace, ctypes.c_void_p, acl_address + 8)
        return 1

    freed: list[object] = []
    advapi32 = _FakeWindowsDll(
        GetNamedSecurityInfoW=get_named_security_info,
        EqualSid=equal_sid,
        GetSecurityDescriptorControl=get_security_descriptor_control,
        GetAce=get_ace,
    )
    kernel32 = _FakeWindowsDll(LocalFree=lambda handle: freed.append(handle))
    _install_fake_windll(monkeypatch, {"advapi32": advapi32, "kernel32": kernel32})
    monkeypatch.setattr(
        cad_module, "_windows_current_user_sid", lambda: (token_user_sid, None)
    )
    monkeypatch.setattr(
        cad_module,
        "_windows_current_token_owner_sid",
        lambda: (token_owner_sid, None),
    )
    return acl_buffer, descriptor_buffer


def test_windows_apply_owner_only_dacl_sets_token_user_as_owner(
    monkeypatch, tmp_path
):
    import ctypes
    from ctypes import wintypes

    token_user_sid = 0x51DF00D
    dacl_address = 0xDAC1
    sid_text = ctypes.create_unicode_buffer("S-1-5-21-1111-2222-3333-1001")
    descriptor_buffer = ctypes.create_string_buffer(16)
    sddl_strings: list[str] = []
    set_calls: list[tuple] = []
    verified: list[tuple[Path, bool]] = []

    def convert_sid_to_string_sid(sid, string_pointer):
        assert sid == token_user_sid
        _set_pointer(string_pointer, ctypes.c_void_p, ctypes.addressof(sid_text))
        return 1

    def convert_sddl_to_descriptor(sddl, revision, descriptor, size):
        sddl_strings.append(sddl)
        _set_pointer(
            descriptor, ctypes.c_void_p, ctypes.addressof(descriptor_buffer)
        )
        _set_pointer(size, wintypes.DWORD, 16)
        return 1

    def get_security_descriptor_dacl(descriptor, present, dacl, defaulted):
        _set_pointer(present, wintypes.BOOL, 1)
        _set_pointer(dacl, ctypes.c_void_p, dacl_address)
        _set_pointer(defaulted, wintypes.BOOL, 0)
        return 1

    def set_named_security_info(
        path, object_type, information, owner, group, dacl, sacl
    ):
        set_calls.append((path, object_type, information, owner, group, dacl, sacl))
        return 0

    advapi32 = _FakeWindowsDll(
        ConvertSidToStringSidW=convert_sid_to_string_sid,
        ConvertStringSecurityDescriptorToSecurityDescriptorW=(
            convert_sddl_to_descriptor
        ),
        GetSecurityDescriptorDacl=get_security_descriptor_dacl,
        SetNamedSecurityInfoW=set_named_security_info,
    )
    kernel32 = _FakeWindowsDll(LocalFree=lambda handle: None)
    _install_fake_windll(monkeypatch, {"advapi32": advapi32, "kernel32": kernel32})
    monkeypatch.setattr(
        cad_module, "_windows_current_user_sid", lambda: (token_user_sid, None)
    )
    monkeypatch.setattr(
        cad_module,
        "_windows_verify_owner_only_dacl",
        lambda path, *, directory: verified.append((path, directory)),
    )

    target = tmp_path / "lock-state-v1"
    cad_module._windows_apply_owner_only_dacl(target, directory=True)

    assert sddl_strings == ["D:P(A;OICI;FA;;;S-1-5-21-1111-2222-3333-1001)"]
    assert len(set_calls) == 1
    path, object_type, information, owner, group, dacl, sacl = set_calls[0]
    assert path == str(target)
    assert object_type == 1  # SE_FILE_OBJECT
    owner_security_information = 0x00000001
    dacl_security_information = 0x00000004
    protected_dacl_security_information = 0x80000000
    assert information == (
        owner_security_information
        | dacl_security_information
        | protected_dacl_security_information
    )
    assert owner == token_user_sid
    assert group is None
    assert int(dacl.value) == dacl_address
    assert sacl is None
    assert verified == [(target, True)]


def test_windows_verify_accepts_token_user_owner(monkeypatch, tmp_path):
    keepalive = _fake_verify_security_environment(
        monkeypatch, stamped_owner_marker="token_user"
    )

    cad_module._windows_verify_owner_only_dacl(
        tmp_path / "lock-state-v1", directory=True
    )
    del keepalive


def test_windows_verify_accepts_token_owner_stamped_by_admin_token(
    monkeypatch, tmp_path
):
    keepalive = _fake_verify_security_environment(
        monkeypatch, stamped_owner_marker="token_owner"
    )

    cad_module._windows_verify_owner_only_dacl(
        tmp_path / "lock-state-v1", directory=True
    )
    del keepalive


def test_windows_verify_rejects_third_party_owner(monkeypatch, tmp_path):
    keepalive = _fake_verify_security_environment(
        monkeypatch, stamped_owner_marker="third_party"
    )

    with pytest.raises(MesherError, match="different owner"):
        cad_module._windows_verify_owner_only_dacl(
            tmp_path / "lock-state-v1", directory=True
        )
    del keepalive


def test_publish_lock_rejects_symlink_redirection(tmp_path):
    target = tmp_path / "horn.wglink"
    staging = tmp_path / ".horn.wglink.staged"
    target.mkdir()
    staging.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("do not touch", encoding="utf-8")
    try:
        cad_module._publish_lock_path(target).symlink_to(victim)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(MesherError, match="coordination"):
        cad_module._publish_bundle_without_exchange(staging, target)

    assert victim.read_text(encoding="utf-8") == "do not touch"
    assert target.exists()
    assert staging.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits are not portable")
def test_publish_lock_rejects_non_private_mode(tmp_path):
    target = tmp_path / "horn.wglink"
    staging = tmp_path / ".horn.wglink.staged"
    target.mkdir()
    staging.mkdir()
    lock_path = cad_module._publish_lock_path(target)
    lock_path.write_bytes(b"\0")
    lock_path.chmod(0o644)

    with pytest.raises(MesherError, match="mode 0600"):
        cad_module._publish_bundle_without_exchange(staging, target)

    assert target.exists()
    assert staging.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership is not portable")
def test_publish_lock_rejects_different_owner(monkeypatch, tmp_path):
    target = tmp_path / "horn.wglink"
    staging = tmp_path / ".horn.wglink.staged"
    target.mkdir()
    staging.mkdir()
    lock_path = cad_module._publish_lock_path(target)
    lock_path.write_bytes(b"existing")
    lock_path.chmod(0o600)
    actual_uid = os.getuid()
    monkeypatch.setattr(cad_module.os, "getuid", lambda: actual_uid + 1)

    with pytest.raises(MesherError, match="different owner"):
        cad_module._publish_bundle_without_exchange(staging, target)

    assert target.exists()
    assert staging.exists()


@pytest.mark.skipif(os.name != "posix", reason="requires unlinking an open file")
def test_publish_lock_rejects_inode_split(monkeypatch, tmp_path):
    target = tmp_path / "horn.wglink"
    staging = tmp_path / ".horn.wglink.staged"
    target.mkdir()
    staging.mkdir()
    lock_path = cad_module._publish_lock_path(target)
    original_lock = cad_module._lock_file

    def split_lock_inode(descriptor):
        original_lock(descriptor)
        lock_path.unlink()
        lock_path.write_bytes(b"replacement")
        lock_path.chmod(0o600)

    monkeypatch.setattr(cad_module, "_lock_file", split_lock_inode)

    with pytest.raises(MesherError, match="exactly one link|changed while open"):
        cad_module._publish_bundle_without_exchange(staging, target)

    assert target.exists()
    assert staging.exists()


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
