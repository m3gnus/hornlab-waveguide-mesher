from __future__ import annotations

import math

import meshio
import numpy as np
import pytest

from hornlab_mesher.experimental.cabinet import (
    _grid_from_payload,
    build_horn_in_box_mesh,
    build_mesh_via_hornlab,
    measure_horn_mouth,
    waveguide_payload_to_mesher_config,
)
from hornlab_mesher.tags import PhysicalGroup


def _raw_point_grid_payload(*, enc_depth: float = 0.0) -> dict:
    n_phi = 16
    n_length = 5
    points = np.empty((n_phi, n_length + 1, 3), dtype=float)
    for i in range(n_phi):
        phi = math.tau * i / n_phi
        for j in range(n_length + 1):
            t = j / n_length
            radius = 10.0 + 30.0 * t
            points[i, j] = (radius * math.cos(phi), radius * math.sin(phi), 120.0 * t)
    payload = {
        "inner_points": points.reshape(-1).tolist(),
        "grid_n_phi": n_phi,
        "grid_n_length": n_length,
        "full_circle": True,
        "source_shape": 2,
        "throat_res": 8.0,
        "mouth_res": 18.0,
        "rear_res": 24.0,
        "enc_depth": enc_depth,
    }
    if enc_depth > 0:
        payload.update(
            {
                "enc_space_l": 20.0,
                "enc_space_r": 20.0,
                "enc_space_t": 20.0,
                "enc_space_b": 20.0,
                "enc_edge": 0.0,
                "enc_front_resolution": "18,18,18,18",
                "enc_back_resolution": "24,24,24,24",
            }
        )
    return payload


def _formula_payload() -> dict:
    return {
        "formula_type": "OSSE",
        "L": 100.0,
        "r0": 10.0,
        "a": 35.0,
        "a0": 10.0,
        "k": 1.0,
        "n": 3.0,
        "q": 0.95,
        "s": 0.0,
        "n_angular": 16,
        "n_length": 6,
        "source_shape": 2,
        "throat_res": 8.0,
        "mouth_res": 18.0,
        "rear_res": 24.0,
        "enc_depth": 160.0,
        "enc_space_l": 20.0,
        "enc_space_r": 20.0,
        "enc_space_t": 20.0,
        "enc_space_b": 20.0,
        "enc_edge": 0.0,
        "enc_plan_type": 1,
        "enc_front_resolution": "18,18,18,18",
        "enc_back_resolution": "24,24,24,24",
    }


def _triangle_tags(path):
    mesh = meshio.read(path)
    tags = []
    for index, block in enumerate(mesh.cells):
        if block.type not in ("triangle", "triangle3"):
            continue
        tags.extend(mesh.cell_data["gmsh:physical"][index].astype(int).tolist())
    return tags


def _node_z_extent(msh_text: str) -> float:
    lines = msh_text.splitlines()
    start = lines.index("$Nodes")
    header = lines[start + 1].split()
    z_values = []
    if len(header) == 1:
        node_count = int(header[0])
        for line in lines[start + 2 : start + 2 + node_count]:
            parts = line.split()
            z_values.append(float(parts[3]))
    else:
        block_count = int(header[0])
        index = start + 2
        for _ in range(block_count):
            block_header = lines[index].split()
            index += 1
            node_count = int(block_header[3])
            index += node_count
            for line in lines[index : index + node_count]:
                parts = line.split()
                z_values.append(float(parts[2]))
            index += node_count
    return max(z_values) - min(z_values)


def test_formula_only_osse_horn_in_box_builds(tmp_path):
    result = build_mesh_via_hornlab(_formula_payload())

    assert "$Nodes" in result["msh_text"]
    assert "$Elements" in result["msh_text"]
    assert result["stats"]["nodeCount"] > 0
    assert result["stats"]["elementCount"] > 0
    assert result["stats"]["source"] == "hornlab_waveguide_mesher_experimental_cabinet"
    assert result["stats"]["units"] == "mm"
    assert _node_z_extent(result["msh_text"]) > 50.0
    assert {"1", "2"}.issubset(result["stats"]["tagCounts"])
    # Size/cost/trustworthy-band forecast carried for the optimizer/BIGMEH.
    solve_cost = result["stats"]["solveCost"]
    assert solve_cost["n_triangles"] == result["stats"]["elementCount"]
    assert solve_cost["ram_bytes"] > 0
    assert solve_cost["feasibility"] in {"ok", "caution", "warn", "infeasible"}
    assert result["stats"]["validFreqMaxHz"] is None or result["stats"]["validFreqMaxHz"] > 0
    assert result["stats"]["meshReport"]


def test_raw_point_grid_payload_builds(tmp_path):
    output = build_horn_in_box_mesh(_raw_point_grid_payload(enc_depth=0.0), tmp_path / "raw.msh", verbose=False)
    tags = set(_triangle_tags(output))

    assert output.is_file()
    assert int(PhysicalGroup.RIGID_WALL) in tags
    assert int(PhysicalGroup.PRIMARY_SOURCE) in tags


def test_measure_horn_mouth_uses_raw_grid():
    width, height = measure_horn_mouth(_raw_point_grid_payload())

    assert width == 80.0
    assert height == 80.0


def test_mouth_scaling_is_throat_pinned_and_reaches_targets():
    payload = _raw_point_grid_payload()
    payload["h_scale_target"] = 100.0
    payload["v_scale_target"] = 60.0

    width, height = measure_horn_mouth(payload)

    assert width == 100.0
    assert height == 60.0


def test_cross_section_passthrough_affects_formula_grid():
    payload = _formula_payload()
    payload["cross_section"] = {"exponent": 4.0, "aspect_ratio": 1.5}

    config = waveguide_payload_to_mesher_config(payload)

    assert config["cross_section"] == {"exponent": 4.0, "aspectRatio": 1.5}

    default_payload = {
        key: value
        for key, value in _formula_payload().items()
        if not key.startswith("enc_")
    }
    aspect_payload = dict(default_payload)
    aspect_payload["cross_section"] = {"exponent": 2.0, "aspect_ratio": 1.5}

    default_width, default_height = measure_horn_mouth(default_payload)
    aspect_width, aspect_height = measure_horn_mouth(aspect_payload)

    assert default_width == pytest.approx(default_height)
    assert aspect_width != aspect_height


def test_raw_point_grid_grid_closed_legacy_key_controls_closure():
    payload = _raw_point_grid_payload()
    payload.pop("full_circle")
    payload["grid_closed"] = False

    _inner_points, _outer_points, closed, _planes, _voffset = _grid_from_payload(payload)

    assert closed is False

    payload.pop("grid_closed")
    _inner_points, _outer_points, closed, _planes, _voffset = _grid_from_payload(payload)

    assert closed is True


def test_bigmeh_extension_tags_are_available_without_reassigning_wg_tags():
    assert int(PhysicalGroup.ENCLOSURE_WALL) == 3
    assert int(PhysicalGroup.INTERFACE) == 4
    assert int(PhysicalGroup.MID_CHAMBER) == 8
    assert int(PhysicalGroup.PORT_INTERIOR) == 9
    assert int(PhysicalGroup.MID_PORT_EXIT_LEFT) == 10
    assert int(PhysicalGroup.MID_PORT_EXIT_RIGHT) == 11


def _cone_grid(angles: np.ndarray, *, n_length: int = 5, mouth_z: float = 120.0) -> np.ndarray:
    points = np.empty((len(angles), n_length + 1, 3), dtype=float)
    for i, phi in enumerate(angles):
        for j in range(n_length + 1):
            t = j / n_length
            radius = 10.0 + 30.0 * t
            points[i, j] = (radius * math.cos(phi), radius * math.sin(phi), mouth_z * t)
    return points


def _box_payload(points: np.ndarray, *, full_circle: bool, spaces: tuple[float, float, float, float], enc_edge: float, enc_depth: float = 200.0) -> dict:
    space_l, space_r, space_t, space_b = spaces
    return {
        "inner_points": points.reshape(-1).tolist(),
        "grid_n_phi": points.shape[0],
        "grid_n_length": points.shape[1] - 1,
        "full_circle": full_circle,
        "source_shape": 2,
        "throat_res": 8.0,
        "mouth_res": 18.0,
        "rear_res": 24.0,
        "enc_depth": enc_depth,
        "enc_space_l": space_l,
        "enc_space_r": space_r,
        "enc_space_t": space_t,
        "enc_space_b": space_b,
        "enc_edge": enc_edge,
        "enc_front_resolution": "18,18,18,18",
        "enc_back_resolution": "24,24,24,24",
    }


def _side_wall_roundover(msh_path, *, z_front: float = 120.0) -> float:
    """Front roundover radius measured as z_front minus the +x side wall's top."""
    mesh = meshio.read(msh_path)
    pts = np.asarray(mesh.points, dtype=float)
    bx1 = float(pts[:, 0].max())
    at_wall = pts[np.abs(pts[:, 0] - bx1) < 1.0e-6]
    return z_front - float(at_wall[:, 2].max())


def test_enclosure_rejects_wall_crossing_baffle_outside_mouth(tmp_path):
    # A rolled-back lip that stays radially inside the mouth ring protrudes
    # through the baffle hole and is legitimate (R-OSSE). But a bulbous lip
    # that curls back IN crosses the front-baffle plane radially outside the
    # ring: the baffle would bisect the wall, the box still welds watertight,
    # and the solve is silently wrong — the builder must fail explicitly.
    angles = np.array([math.tau * i / 16 for i in range(16)])
    radii = [10.0, 16.0, 22.0, 28.0, 32.0, 42.0, 35.0]
    zs = [0.0, 24.0, 48.0, 72.0, 95.0, 106.0, 100.0]
    points = np.empty((len(angles), len(radii), 3), dtype=float)
    for i, phi in enumerate(angles):
        for j, (radius, z) in enumerate(zip(radii, zs)):
            points[i, j] = (radius * math.cos(phi), radius * math.sin(phi), z)
    payload = _box_payload(points, full_circle=True, spaces=(20.0, 20.0, 20.0, 20.0), enc_edge=0.0)
    with pytest.raises(Exception, match="front-baffle"):
        build_horn_in_box_mesh(payload, tmp_path / "curl-in.msh", verbose=False)


def test_quarter_roundover_matches_full_build(tmp_path):
    # The roundover clamp must describe the mirror-completed physical box:
    # edge=80 with 100 mm spacings survives on the full build (limits: margin
    # 100-5, half-extent 139.9), so the quarter build of the same design must
    # not clamp it to its open-box half-extent (69.9 pre-fix).
    full_angles = np.array([math.tau * i / 16 for i in range(16)])
    quarter_angles = np.array([(math.pi / 2.0) * i / 16 for i in range(17)])
    spaces = (100.0, 100.0, 100.0, 100.0)

    full_payload = _box_payload(_cone_grid(full_angles), full_circle=True, spaces=spaces, enc_edge=80.0)
    build_horn_in_box_mesh(full_payload, tmp_path / "full.msh", verbose=False)
    r_full = _side_wall_roundover(tmp_path / "full.msh")

    quarter_payload = _box_payload(_cone_grid(quarter_angles), full_circle=False, spaces=spaces, enc_edge=80.0)
    build_horn_in_box_mesh(quarter_payload, tmp_path / "quarter.msh", verbose=False)
    r_quarter = _side_wall_roundover(tmp_path / "quarter.msh")

    assert abs(r_full - 80.0) < 1.0e-3
    assert abs(r_quarter - r_full) < 1.0e-6


def test_quarter_unused_cut_side_spacing_keeps_roundover(tmp_path):
    # Spacings on the cut-plane sides are never applied on a reduced domain;
    # space_l = space_b = 0 must not force a sharp box (the pre-fix margin
    # clamp minimised over all four spacings).
    quarter_angles = np.array([(math.pi / 2.0) * i / 16 for i in range(17)])
    payload = _box_payload(
        _cone_grid(quarter_angles),
        full_circle=False,
        spaces=(0.0, 25.0, 25.0, 0.0),
        enc_edge=18.0,
    )
    build_horn_in_box_mesh(payload, tmp_path / "quarter-rounded.msh", verbose=False)
    assert abs(_side_wall_roundover(tmp_path / "quarter-rounded.msh") - 18.0) < 1.0e-3
