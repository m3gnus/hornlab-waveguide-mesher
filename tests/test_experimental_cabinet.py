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

    _inner_points, _outer_points, closed = _grid_from_payload(payload)

    assert closed is False

    payload.pop("grid_closed")
    _inner_points, _outer_points, closed = _grid_from_payload(payload)

    assert closed is True


def test_bigmeh_extension_tags_are_available_without_reassigning_wg_tags():
    assert int(PhysicalGroup.ENCLOSURE_WALL) == 3
    assert int(PhysicalGroup.INTERFACE) == 4
    assert int(PhysicalGroup.MID_CHAMBER) == 8
    assert int(PhysicalGroup.PORT_INTERIOR) == 9
    assert int(PhysicalGroup.MID_PORT_EXIT_LEFT) == 10
    assert int(PhysicalGroup.MID_PORT_EXIT_RIGHT) == 11
