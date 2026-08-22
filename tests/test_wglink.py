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
    metadata = os.lstat(lock_path)
    assert stat.S_ISREG(metadata.st_mode)
    assert metadata.st_nlink == 1
    assert metadata.st_size == 32
    if os.name == "posix":
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_uid == os.getuid()


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
    remnants = set(tmp_path.glob(".horn.wglink.*"))
    if os.name == "posix":
        assert not remnants
    else:
        assert remnants == {cad_module._publish_lock_path(target)}
        _assert_private_publish_lock(target)


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
    backup = tmp_path / ".horn.wglink.publish.00000000000000000000000000000000.previous"
    path_type = type(target)
    original_replace = path_type.replace

    def fail_staged_publish(path, destination):
        if path == staging and Path(destination) == target:
            raise OSError("injected publish failure")
        return original_replace(path, destination)

    monkeypatch.setattr(path_type, "replace", fail_staged_publish)

    with pytest.raises(OSError, match="injected publish failure"):
        cad_module._replace_directories_with_rollback(staging, target, backup)

    assert (target / "generation.txt").read_text(encoding="utf-8") == "old"
    assert (staging / "generation.txt").read_text(encoding="utf-8") == "new"
    assert not list(tmp_path.glob("*.previous"))


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
path_type = type(target)
original_replace = path_type.replace

def die_after_first_rename(path, destination):
    result = original_replace(path, destination)
    if path == target and Path(destination).name.endswith(".previous"):
        os._exit(91)
    return result

path_type.replace = die_after_first_rename
cad._publish_bundle_without_exchange(staging, target)
"""

    crashed = subprocess.run(
        [sys.executable, "-c", crash_script, str(target), str(interrupted_staging)],
        check=False,
    )

    assert crashed.returncode == 91
    assert not target.exists()
    assert interrupted_staging.exists()
    record = json.loads(cad_module._transaction_path(target).read_text(encoding="utf-8"))
    assert (tmp_path / record["backup"]).exists()

    with cad_module._bundle_publish_lock(target) as lock_secret:
        cad_module._recover_directory_replacement(target, lock_secret)

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


def test_non_posix_publication_serializes_concurrent_processes(tmp_path):
    target = tmp_path / "horn.wglink"
    staging_a = tmp_path / ".horn.wglink.writer-a"
    staging_b = tmp_path / ".horn.wglink.writer-b"
    backed_up = tmp_path / "writer-a-backed-up"
    release_a = tmp_path / "release-writer-a"
    writer_b_ready = tmp_path / "writer-b-ready"
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
path_type = type(target)
original_replace = path_type.replace

def pause_after_first_rename(path, destination):
    result = original_replace(path, destination)
    if path == target and Path(destination).name.endswith(".previous"):
        backed_up.write_text("ready", encoding="utf-8")
        while not release.exists():
            time.sleep(0.01)
    return result

path_type.replace = pause_after_first_rename
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
    record_path = cad_module._transaction_path(target)
    token = "0" * 32
    record_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "target": target.name,
                "token": token,
                "staging": staging.name,
                "backup": f".{target.name}.publish.{token}.previous",
                "mac": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    if os.name == "posix":
        record_path.chmod(0o600)

    with pytest.raises(MesherError, match="invalid or unauthenticated"):
        cad_module._publish_bundle_without_exchange(staging, target)

    assert (target / "generation.txt").read_text(encoding="utf-8") == "old"
    assert (staging / "generation.txt").read_text(encoding="utf-8") == "new"
    assert record_path.exists()


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


def test_writer_fsyncs_staged_members_through_writable_descriptors(monkeypatch, tmp_path):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "member.bin").write_bytes(b"payload")
    synced: list[int] = []

    monkeypatch.setattr(cad_module, "_fsync_directory", lambda _path: None)
    monkeypatch.setattr(cad_module.os, "fsync", lambda descriptor: synced.append(os.write(descriptor, b"")))

    cad_module._sync_staged_bundle(staged)

    assert synced == [0]


def test_writer_rejects_config_like_enclosure_bounds(monkeypatch, tmp_path):
    _fake_step(monkeypatch)
    geometry = _enclosure_geometry()
    built = _built(geometry)
    assert built.enclosure_bounds is not None
    built.enclosure_bounds["enc_depth"] = geometry.enclosure.depth_mm

    with pytest.raises(MesherError, match="do not match.*enc_depth"):
        write_wglink(
            geometry, tmp_path / "horn.wglink", built_geometry=built
        )
