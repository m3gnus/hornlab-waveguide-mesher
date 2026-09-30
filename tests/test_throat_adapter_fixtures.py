"""Validate archived ATH exports; no mesher adapter implementation is exercised."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PureWindowsPath

import pytest


ROOT = Path(__file__).parent / "fixtures" / "throat_adapter" / "ath-v2025-12"
INDEX = json.loads((ROOT / "index.json").read_text(encoding="utf-8"))
CASES = INDEX["cases"]
CTRL_CASES = [
    case for case in CASES
    if not case["case_id"].startswith("base-")
    and not case["case_id"].endswith("-noctrl")
]


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_portable(value):
    if isinstance(value, dict):
        for key, item in value.items():
            assert key not in {"Output.DestDir", "ath_sha256", "ath_executable_hash"}
            _assert_portable(key)
            _assert_portable(item)
    elif isinstance(value, list):
        for item in value:
            _assert_portable(item)
    elif isinstance(value, str):
        value = value.strip().strip("\"'")
        assert not value.startswith(("/", "\\", "~/", "file:"))
        assert not PureWindowsPath(value).drive
    elif isinstance(value, (float, int)):
        assert math.isfinite(value)


def _profiles(case):
    raw = (ROOT / case["csv"]).read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    assert hashlib.sha256(raw).hexdigest() == case["csv_sha256"]
    blocks = raw.decode("ascii").strip().split("\n\n")
    profiles = []
    for block in blocks:
        points = []
        for line in block.splitlines():
            fields = line.split(INDEX["delimiter"])
            assert len(fields) == len(INDEX["columns"])
            point = tuple(float(field) for field in fields)
            assert all(math.isfinite(v) for v in point)
            points.append(point)
        profiles.append(points)
    return profiles


def _de_casteljau(controls, u):
    """Independent recursive linear interpolation, in the (z,r) plane."""
    points = [tuple(point) for point in controls]
    while len(points) > 1:
        points = [
            tuple((1 - u) * a + u * b for a, b in zip(left, right))
            for left, right in zip(points, points[1:])
        ]
    return points[0]


def _assert_adapter(profiles, parameters, meridians):
    config = parameters["config"]
    adapter = parameters["adapter"]
    controls = adapter["control_points_join_local_mm"]
    assert len(controls) == 4 and all(len(point) == 2 for point in controls)
    raw = [float(v) for v in config["Throat.Ext.Ctrl"].split(",")]
    assert len(raw) == 6
    assert controls[:3] == [raw[:2], raw[2:4], raw[4:]]
    assert controls[3][0] == 0
    extension = float(config["Throat.Ext.Length"])
    translation = adapter["axial_translation_mm"]
    assert translation == extension - raw[0]
    start = adapter["first_station"]
    assert start == (1 if extension else 0)
    segments = adapter["segments"]
    assert segments == int(config.get("Mesh.ThroatExtSegments", "10")) - start
    endpoint_offset = adapter["endpoint_axial_offset_mm"]
    assert endpoint_offset == (
        float(config["Slot.Length"]) if "R-OSSE" in config else 0.0
    )
    scale = float(config["Scale"]) * float(config["GridExport:c5"]["Scale"])
    for profile, phi_deg in zip(profiles, meridians):
        assert len(profile) > start + segments
        phi = math.radians(phi_deg)
        for i in range(segments + 1):
            z, radius = _de_casteljau(controls, i / segments)
            # R1 with a slot exports its far endpoint instead of the cubic join.
            if i == segments:
                z += endpoint_offset
            expected = (
                scale * radius * math.cos(phi),
                scale * radius * math.sin(phi),
                scale * (z + translation),
            )
            assert math.dist(profile[start + i], expected) <= INDEX["tolerance_mm"]


def test_fixture_index_and_portability():
    assert INDEX["schema_version"] == 1
    assert INDEX["ath_version"] == "V2025-12"
    assert INDEX["export_date"] == "2026-09-30"
    assert INDEX["columns"] == ["x", "y", "z"]
    assert INDEX["units"] == "mm" and INDEX["delimiter"] == ";"
    assert INDEX["header"] is False and INDEX["tolerance_mm"] == 0.001
    ids = [case["case_id"] for case in CASES]
    assert ids and len(ids) == len(set(ids))
    expected_files = {"README.md", "index.json"}
    for case in CASES:
        name = case["case_id"]
        assert case["csv"] == f"{name}.csv"
        assert case["parameters"] == f"{name}.json"
        expected_files.update([case["csv"], case["parameters"]])
    assert {path.name for path in ROOT.iterdir()} == expected_files
    for path in ROOT.iterdir():
        raw = path.read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n")
        if path.suffix == ".json":
            _assert_portable(_json(path))
    assert sum(path.stat().st_size for path in ROOT.iterdir()) < 1_500_000


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_fixture_case_integrity(case):
    profiles = _profiles(case)
    counts = case["station_counts"]
    angles = case["meridians_deg"]
    assert len(profiles) == len(counts) == len(angles)
    assert len(set(angles)) == len(angles) and {0, 45, 90} <= set(angles)
    assert all(isinstance(n, int) and n > 0 for n in counts)
    assert [len(profile) for profile in profiles] == counts
    assert sum(counts) == case["point_count"]
    parameters = _json(ROOT / case["parameters"])
    assert parameters["case_id"] == case["case_id"]
    assert parameters["ath_version"] == INDEX["ath_version"]
    assert parameters["expected_import"] in {"qualified", "refuse"}
    config = parameters["config"]
    kind = "R1" if "R-OSSE" in config else "O1"
    assert parameters["contract_interpretation"] == f"ATH-V2025-12-C5-{kind}"
    syntax = "R-OSSE block" if kind == "R1" else (
        "OSSE block" if "OSSE" in config else "flat OSSE"
    )
    assert parameters["profile_syntax"] == syntax
    if "Throat.Ext.Ctrl" in config:
        assert parameters["role"] == "ctrl" and "adapter" in parameters
        baseline_id = parameters["baseline_case_id"]
        baseline = next(c for c in CASES if c["case_id"] == baseline_id)
        baseline_parameters = _json(ROOT / baseline["parameters"])
        assert baseline_parameters["role"] == "baseline"
        assert baseline_parameters["contract_interpretation"] == parameters["contract_interpretation"]
        assert baseline_parameters["config"] == {
            key: value for key, value in config.items() if key != "Throat.Ext.Ctrl"
        }
    else:
        assert parameters["role"] == "baseline" and "adapter" not in parameters


@pytest.mark.parametrize("case", CTRL_CASES, ids=lambda case: case["case_id"])
def test_exported_adapter_matches_declared_cubic(case):
    _assert_adapter(
        _profiles(case), _json(ROOT / case["parameters"]), case["meridians_deg"]
    )
