"""Source caps keep the rigid placement applied to the preview throat ring."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry
from hornlab_mesher.preview.source_cap import _source_cap


REPRODUCTIONS = json.loads(
    (Path(__file__).parent / "fixtures/preview-offset/reproductions.json").read_text()
)


@pytest.mark.parametrize("case", REPRODUCTIONS)
@pytest.mark.parametrize("lod", ["coarse", "fine"])
@pytest.mark.parametrize("quadrants", ["14", "12", "1"])
def test_offset_source_previews_across_families_modes_and_symmetry(case, lod, quadrants):
    config = copy.deepcopy(REPRODUCTIONS[case])
    config["mesh"]["quadrants"] = quadrants
    preview = build_preview_geometry(config, PreviewOptionsV1(lod=lod))
    cap = next(s for s in preview.surfaces if s.role == "source_cap")
    assert cap.metadata["disagreeingTriangles"] == 0
    assert cap.metadata["windingChecked"] is True


@pytest.mark.parametrize("quadrants", ["1", "12", "14", "1234"])
@pytest.mark.parametrize("offset", [-7.0, 7.0])
@pytest.mark.parametrize("shape,curv", [(0, 1), (1, 1), (1, -1)])
def test_source_cap_is_a_rigid_translation_in_both_symmetry_planes(quadrants, offset, shape, curv):
    spans = {"1": (0, np.pi / 2), "12": (0, np.pi),
             "14": (-np.pi / 2, np.pi / 2), "1234": (0, 2 * np.pi)}
    closed = quadrants == "1234"
    phi = np.linspace(*spans[quadrants], 65, endpoint=not closed)
    ring = np.column_stack((12.7 * np.cos(phi), 12.7 * np.sin(phi), np.zeros(len(phi))))
    inner = ring[:, None, :]
    params = {"sourceShape": shape, "sourceCurv": curv, "sourceRadius": 32.0}
    reference, fidelity, details = _source_cap(
        inner, params, "OSSE", 8, closed_phi=closed, include_curvature=True
    )
    translation = np.array([0.0, offset, 0.0])
    placed, placed_fidelity, placed_details = _source_cap(
        inner + translation, {**params, "verticalOffset": offset}, "OSSE", 8,
        closed_phi=closed, include_curvature=True,
    )
    np.testing.assert_allclose(placed.positions, reference.positions + translation, rtol=0, atol=1e-12)
    np.testing.assert_allclose(placed.normals, reference.normals, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(placed.indices, reference.indices)
    np.testing.assert_allclose(placed.curvature_mean, reference.curvature_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(placed.curvature_principal, reference.curvature_principal, rtol=0, atol=1e-12)
    assert placed_details == pytest.approx(details)
    if fidelity is not None:
        assert placed_fidelity == pytest.approx(fidelity)
    if shape == 1:
        np.testing.assert_allclose(placed.positions[0, :2], [0.0, offset], rtol=0, atol=1e-12)
        np.testing.assert_allclose(placed.positions[-len(ring):], ring + translation, rtol=0, atol=1e-12)


@pytest.mark.parametrize("offset", [0.0, -0.0])
@pytest.mark.parametrize("quadrants", ["1", "12", "14"])
def test_zero_offset_cap_is_bytewise_the_unshifted_cap(quadrants, offset):
    spans = {"1": (0, np.pi / 2), "12": (0, np.pi), "14": (-np.pi / 2, np.pi / 2)}
    phi = np.linspace(*spans[quadrants], 65)
    ring = np.column_stack((12.7 * np.cos(phi), 12.7 * np.sin(phi), np.zeros(len(phi))))
    params = {"sourceShape": 1, "sourceCurv": 1, "sourceRadius": 32.0}
    reference, _, _ = _source_cap(ring[:, None, :], params, "OSSE", 8,
                                  closed_phi=False, include_curvature=True)
    placed, _, _ = _source_cap(ring[:, None, :], {**params, "verticalOffset": offset},
                               "OSSE", 8, closed_phi=False, include_curvature=True)
    assert placed.positions.tobytes() == reference.positions.tobytes()
    assert placed.normals.tobytes() == reference.normals.tobytes()
