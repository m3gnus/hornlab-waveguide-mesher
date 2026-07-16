from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
import pytest

from hornlab_mesher import MesherError, build_from_config
from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.profiles import build_point_grid


def _base_config(mode: str, **mesh_overrides):
    mesh = {
        "mode": mode,
        "angular_segments": 32,
        "length_segments": 16,
        "wall_thickness_mm": 6.0 if mode == "freestanding" else 0.0,
        "throat_res_mm": 8.0,
        "mouth_res_mm": 20.0,
        "rear_res_mm": 20.0,
        "allow_large_mesh": True,
        **mesh_overrides,
    }
    return {
        "formula": "OSSE",
        "mode": mode,
        "profile": {
            "L_mm": 80.0,
            "r0_mm": 10.0,
            "a_deg": 40.0,
            "a0_deg": 0.0,
        },
        "mesh": mesh,
    }


@pytest.mark.parametrize("mode", ["bare", "freestanding", "infinite-baffle"])
def test_acoustic_topology_count_is_stable_against_geometry_sampling(tmp_path, mode):
    counts = []
    for length_segments, angular_segments in ((8, 16), (16, 32), (64, 128)):
        result = build_from_config(
            _base_config(
                mode,
                length_segments=length_segments,
                angular_segments=angular_segments,
            ),
            tmp_path / f"{mode}-{length_segments}-{angular_segments}.msh",
        )
        counts.append(result.n_triangles)

    assert max(counts) / min(counts) < 1.15


@pytest.mark.parametrize("mode", ["bare", "freestanding", "infinite-baffle"])
def test_acoustic_topology_count_follows_mm_targets(tmp_path, mode):
    counts = []
    for size_mm in (5.0, 10.0, 20.0):
        result = build_from_config(
            _base_config(
                mode,
                throat_res_mm=size_mm,
                mouth_res_mm=size_mm,
                rear_res_mm=size_mm,
                aperture_res_scale=1.0,
            ),
            tmp_path / f"{mode}-{size_mm:g}.msh",
        )
        counts.append(result.n_triangles)

    assert counts[0] > counts[1] > counts[2]
    # Doubling h should approach quartering area-controlled triangle count;
    # conformity and closure regions make the ratio approximate.
    assert 2.5 < counts[0] / counts[1] < 5.5
    assert 2.0 < counts[1] / counts[2] < 5.5


def test_acoustic_control_grid_preserves_mouth_geometry_at_coarse_input_sampling(
    tmp_path,
):
    config = _base_config(
        "bare",
        angular_segments=8,
        length_segments=4,
        throat_res_mm=8.0,
        mouth_res_mm=10.0,
    )
    params, _formula, _mode = build_geometry_params(config)
    reference = build_point_grid({**params, "angularSegments": 256})
    n_phi = int(reference["grid_n_phi"])
    n_length = int(reference["grid_n_length"])
    reference_points = np.asarray(reference["inner_points"], dtype=float).reshape(
        n_phi, n_length + 1, 3
    )
    target_radius_mm = float(
        np.mean(np.linalg.norm(reference_points[:, -1, :2], axis=1))
    )

    result = build_from_config(config, tmp_path / "geometry-fidelity.msh")
    points_mm = np.asarray(meshio.read(result.mesh_path).points, dtype=float) * 1000.0
    mouth_points = points_mm[
        np.isclose(points_mm[:, 2], np.max(points_mm[:, 2]), atol=1.0e-7)
    ]
    mouth_radii_mm = np.linalg.norm(mouth_points[:, :2], axis=1)

    assert result.metadata["geometrySampleAngularSegments"] > 8
    assert result.metadata["geometrySampleLengthSegments"] > 4
    assert np.max(np.abs(mouth_radii_mm - target_radius_mm)) < 0.5


def _enclosure_config(
    *, edge_mm: float, edge_type: int, size_mm: float, space_mm: float
):
    config = _base_config(
        "enclosure",
        quadrants="1",
        angular_segments=16,
        length_segments=8,
        mouth_res_mm=size_mm,
        rear_res_mm=size_mm,
        enc_front_res_mm=size_mm,
        enc_back_res_mm=size_mm,
    )
    config["enclosure"] = {
        "depth": 100.0,
        "space_l": space_mm,
        "space_t": space_mm,
        "space_r": space_mm,
        "space_b": space_mm,
        "edge": edge_mm,
        "edgeType": edge_type,
    }
    return config


@pytest.mark.parametrize("edge_type", [1, 2])
def test_sub_resolution_cosmetic_edges_use_sharp_acoustic_topology(tmp_path, edge_type):
    sharp = build_from_config(
        _enclosure_config(
            edge_mm=0.0, edge_type=edge_type, size_mm=40.0, space_mm=25.0
        ),
        tmp_path / f"sharp-{edge_type}.msh",
    )
    tiny = build_from_config(
        _enclosure_config(
            edge_mm=1.0, edge_type=edge_type, size_mm=40.0, space_mm=25.0
        ),
        tmp_path / f"tiny-{edge_type}.msh",
    )
    margin_clamped = build_from_config(
        _enclosure_config(
            edge_mm=18.0, edge_type=edge_type, size_mm=40.0, space_mm=0.25
        ),
        tmp_path / f"clamped-{edge_type}.msh",
    )
    margin_sharp = build_from_config(
        _enclosure_config(
            edge_mm=0.0, edge_type=edge_type, size_mm=40.0, space_mm=0.25
        ),
        tmp_path / f"clamped-sharp-{edge_type}.msh",
    )

    assert tiny.metadata["acousticEnclosureEdgeSuppressed"] is True
    assert tiny.n_triangles == sharp.n_triangles
    assert margin_clamped.metadata["acousticEnclosureEdgeSuppressed"] is True
    assert margin_clamped.n_triangles == margin_sharp.n_triangles


@pytest.mark.parametrize("edge_type", [1, 2])
def test_resolvable_cosmetic_edges_are_retained_and_mm_controlled(tmp_path, edge_type):
    sharp = build_from_config(
        _enclosure_config(
            edge_mm=0.0, edge_type=edge_type, size_mm=10.0, space_mm=25.0
        ),
        tmp_path / f"retained-sharp-{edge_type}.msh",
    )
    retained = build_from_config(
        _enclosure_config(
            edge_mm=20.0, edge_type=edge_type, size_mm=10.0, space_mm=25.0
        ),
        tmp_path / f"retained-{edge_type}.msh",
    )

    assert retained.metadata["acousticEnclosureEdgeSuppressed"] is False
    assert retained.metadata["acousticEnclosureFeatureLengthMm"] >= 10.0
    assert 0.5 < retained.n_triangles / sharp.n_triangles < 2.0


def test_realized_triangle_limit_catches_legacy_topology_overrun(tmp_path):
    config = _base_config(
        "bare",
        topology="legacy",
        preserve_grid=True,
        angular_segments=64,
        length_segments=32,
        throat_res_mm=100.0,
        mouth_res_mm=100.0,
        rear_res_mm=100.0,
        max_triangles=500,
        allow_large_mesh=False,
    )
    output = Path(tmp_path) / "over-limit.msh"

    with pytest.raises(MesherError, match="generated mesh contains .* exceeding"):
        build_from_config(config, output)
    assert not output.exists()


def test_failed_triangle_limit_preserves_existing_output(tmp_path):
    config = _base_config(
        "bare",
        topology="legacy",
        preserve_grid=True,
        angular_segments=64,
        length_segments=32,
        throat_res_mm=100.0,
        mouth_res_mm=100.0,
        rear_res_mm=100.0,
        max_triangles=500,
        allow_large_mesh=False,
    )
    output = Path(tmp_path) / "existing.msh"
    original = b"known-good-existing-mesh"
    output.write_bytes(original)

    with pytest.raises(MesherError, match="generated mesh contains .* exceeding"):
        build_from_config(config, output)

    assert output.read_bytes() == original
    assert not list(Path(tmp_path).glob(".existing.msh.*.tmp"))


def test_sub_resolution_freestanding_wall_fails_before_invalid_shell_build(tmp_path):
    config = _base_config("freestanding", wall_thickness_mm=0.1)

    with pytest.raises(ValueError, match="below the stable acoustic feature floor"):
        build_from_config(config, tmp_path / "too-thin-wall.msh")

    assert not (tmp_path / "too-thin-wall.msh").exists()
