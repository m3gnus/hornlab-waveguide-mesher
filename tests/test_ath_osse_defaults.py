"""ATH's defaults for OSSE items a text config may omit, measured against ATH.

The Ath 4.8.2 User Guide gives 0.7 as the default termination ``s``
(``Term.s``). The ATH binaries do not use it: V2025-12 and V2026-09a build an
OSSE block without ``s`` exactly as one with ``s = 0.5``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.config_parser import parse_text_config
from hornlab_mesher.profile_formulas import calculate_osse, calculate_osse_curve

ATH_FIXTURES = Path(__file__).parent / "fixtures" / "ath_import" / "ath-v2025-12"


def _ath_profiles(name: str) -> list[np.ndarray]:
    text = (ATH_FIXTURES / f"{name}.csv").read_text(encoding="utf-8")
    return [
        np.array([[float(value) for value in line.split(";")] for line in block.splitlines()])
        for block in text.strip().split("\n\n")
    ]


def test_ath_default_osse_termination_s_is_one_half() -> None:
    omitted = (ATH_FIXTURES / "osse-block-without-s.csv").read_text(encoding="utf-8")
    assert omitted == (ATH_FIXTURES / "osse-block-s-0.5.csv").read_text(encoding="utf-8")
    assert omitted != (ATH_FIXTURES / "osse-block-s-0.7.csv").read_text(encoding="utf-8")


@pytest.mark.parametrize(("name", "s"), [
    ("osse-block-without-s", 0.5),
    ("osse-block-s-0.5", 0.5),
    ("osse-block-s-0.7", 0.7),
])
def test_import_matches_ath_for_osse_termination_s(name: str, s: float) -> None:
    config = parse_text_config((ATH_FIXTURES / f"{name}.cfg").read_text(encoding="utf-8"))
    assert config["profile"]["s"] == s
    params = build_geometry_params(config)[0]
    profiles = _ath_profiles(name)
    assert len(profiles) == 8
    for index, points in enumerate(profiles):
        phi = 2.0 * np.pi * index / len(profiles)
        z = points[:, 2]
        expected_radius = np.hypot(points[:, 0], points[:, 1])
        assert z[-1] == pytest.approx(80.0)
        scalar = np.array([calculate_osse(float(station), phi, params) for station in z])
        vector = np.column_stack(calculate_osse_curve(z, phi, params))
        for actual in (scalar, vector):
            np.testing.assert_allclose(actual[:, 0], z, rtol=0, atol=1e-9)
            np.testing.assert_allclose(actual[:, 1], expected_radius, rtol=0, atol=5e-6)


def test_omitted_flat_term_s_matches_explicit_one_half() -> None:
    flat = "Throat.Diameter = 20\nCoverage.Angle = 30\nLength = 80\n"
    assert parse_text_config(flat) == parse_text_config(flat + "Term.s = 0.5\n")
