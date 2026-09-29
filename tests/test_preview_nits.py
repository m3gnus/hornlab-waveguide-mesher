"""Fold diagnostics and cooperative tight-normal refinement limits."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry
from hornlab_mesher.preview import api, fidelity, intersections
from hornlab_mesher.preview.contract import (
    _orient_indices_to_normals,
    _orientation_metadata,
)
from hornlab_mesher.preview.primitives import _grid_indices
from tests.test_preview_followups import FREEFORM_MORPH, ROSSE_EXTENSION, ROSSE_SLOT, _OSSE


def test_crossing_outside_the_fold_neighbour_ring_is_detected():
    outer = np.array([
        [10., 0., 0.], [12., 0., 0.], [10., 2., 0.],
        [6., 0., 0.], [6., 2., 0.],
        [0., 0., 0.], [2., 0., 0.], [0., 2., 0.],
    ])
    faces = np.array([[0, 1, 2], [2, 3, 4], [3, 4, 5], [5, 6, 7]], dtype=np.uint32)
    folded = np.array([True, False, False, False])
    inner = np.array([[0.5, 0.5, -1.], [0.5, 0.5, 1.], [1.5, 0.5, 0.]])
    inner_faces = np.array([0, 1, 2], dtype=np.uint32)
    ring = np.any(np.isin(faces, faces[folded].ravel()), axis=1)
    assert ring.tolist() == [True, True, False, False]
    assert not intersections.folded_wall_crosses_inner(
        outer, faces[ring], np.ones(2, dtype=bool), inner, inner_faces
    )
    assert intersections.folded_wall_crosses_inner(outer, faces, folded, inner, inner_faces)


def test_whole_wall_broad_phase_discards_unreachable_facets(monkeypatch):
    outer = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
    faces = np.arange(3000, dtype=np.uint32).reshape(-1, 3)
    points = np.tile(outer, (1000, 1)) + np.repeat(np.arange(1000) * 10., 3)[:, None]
    inner = outer + 20000.

    def unexpected_narrow_phase(*args):
        pytest.fail("disjoint surface boxes must not reach the narrow phase")

    monkeypatch.setattr(intersections, "_triangles_intersect", unexpected_narrow_phase)
    assert not intersections.folded_wall_crosses_inner(
        points, faces, np.ones(1000, dtype=bool), inner, np.array([0, 1, 2])
    )


@pytest.mark.parametrize("reverse_all", [False, True])
def test_fold_ids_survive_rewinding_in_either_prevailing_orientation(reverse_all):
    positions = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
    normals = np.tile([0., 0., 1.], (3, 1))
    faces = np.array([[0, 1, 2], [0, 2, 1], [0, 1, 2]], dtype=np.uint32)
    if reverse_all:
        faces = faces[:, (0, 2, 1)]
    result = _orient_indices_to_normals(
        "horn.outer", positions, faces, normals, wind_folds_individually=True
    )
    metadata = _orientation_metadata(result)
    assert metadata["foldedTriangleIndices"] == [1]
    assert metadata["foldedTriangles"] == 1
    np.testing.assert_array_equal(result.indices.reshape(-1, 3), [[0, 1, 2]] * 3)


@pytest.mark.parametrize("config", [ROSSE_EXTENSION, ROSSE_SLOT])
def test_surface_fold_ids_and_warning_describe_the_shipped_fold(config):
    geometry = build_preview_geometry(copy.deepcopy(config), PreviewOptionsV1(lod="fine"))
    outer = next(s for s in geometry.surfaces if s.role == "horn.outer")
    ids = outer.metadata["foldedTriangleIndices"]
    assert ids == sorted(set(ids))
    assert len(ids) == outer.metadata["foldedTriangles"]
    assert 0 <= ids[0] <= ids[-1] < len(outer.indices) // 3
    raw = _grid_indices(
        geometry.metadata["actual_segment_counts"]["horn_axial"] + 1,
        geometry.metadata["actual_segment_counts"]["horn_phi"],
        closed_phi=True,
    ).reshape(-1, 3)
    points = outer.positions[raw]
    face_normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    dots = np.einsum("ij,ij->i", face_normals, outer.normals[raw].mean(axis=1))
    majority = 1 if np.count_nonzero(dots > 0) >= np.count_nonzero(dots < 0) else -1
    assert np.all(dots[ids] * majority < 0)
    shipped = outer.positions[outer.indices.reshape(-1, 3)[ids]]
    shipped_normals = np.cross(shipped[:, 1] - shipped[:, 0], shipped[:, 2] - shipped[:, 0])
    assert np.all(np.einsum("ij,ij->i", shipped_normals,
                           outer.normals[outer.indices.reshape(-1, 3)[ids]].mean(axis=1)) > 0)
    assert geometry.metadata["surface_metadata"]["horn.outer"]["foldedTriangleIndices"] == ids
    z = shipped[..., 2]
    warning = next(w for w in geometry.metadata["warnings"] if "outer wall folds" in w)
    assert f"R-OSSE wall at z={z.min():.3g} to {z.max():.3g} mm" in warning


def _old_twisted_freeform_layout(monkeypatch):
    """Put back the pre-fix FREEFORM rectangle-morph azimuth layout.

    The continuous layout no longer folds these fixtures' outer walls, so the fold
    paths are exercised on the old, twisted layout instead.
    """

    import hornlab_mesher.profile_sampling as sampling

    monkeypatch.setattr(
        sampling, "_blend_toward_uniform_layout", lambda angles, factor: (angles, True)
    )


def test_freeform_fold_warning_has_its_own_design_and_location(monkeypatch):
    _old_twisted_freeform_layout(monkeypatch)
    config = copy.deepcopy(FREEFORM_MORPH)
    config["morph"].update(morphCorner="25", morphFixed="0.2")
    geometry = build_preview_geometry(config, PreviewOptionsV1(lod="coarse"))
    warning = next(w for w in geometry.metadata["warnings"] if "outer wall folds" in w)
    assert "FREEFORM wall at z=" in warning
    assert "throat-extension" not in warning
    assert "slot" not in warning


@pytest.mark.parametrize("vertex_cap", [None, 5000])
def test_expired_tight_normal_budget_degrades_with_incomplete_fidelity(monkeypatch, vertex_cap):
    monkeypatch.setattr(api, "_TIGHT_NORMAL_REFINEMENT_SECONDS", 0.0)
    geometry = build_preview_geometry(
        copy.deepcopy(_OSSE),
        PreviewOptionsV1(lod="coarse", max_normal_step_deg=0.35, max_vertices=vertex_cap),
    )
    assert geometry.metadata["refinement_budget"] == {"seconds": 0.0, "exhausted": True}
    assert any("time budget" in w for w in geometry.metadata["warnings"])
    record = geometry.metadata["fidelity"]["horn.inner"]
    assert record["refinement_time_limited"] is True
    assert record["measurement_complete"] is False
    assert record["max_chord_error_mm_achieved"] is None
    assert record["unmeasured_intervals"] > 0
    assert record["vertex_cap_limited"] is True
    assert all(s.metadata["windingChecked"] for s in geometry.surfaces)
    assert max(len(s.positions) for s in geometry.surfaces) <= (vertex_cap or 200_000)


@pytest.mark.parametrize("lod", ["coarse", "fine"])
def test_lod_defaults_ignore_the_tight_normal_time_budget(monkeypatch, lod):
    baseline = build_preview_geometry(copy.deepcopy(_OSSE), PreviewOptionsV1(lod=lod))
    monkeypatch.setattr(api, "_TIGHT_NORMAL_REFINEMENT_SECONDS", 0.0)
    actual = build_preview_geometry(copy.deepcopy(_OSSE), PreviewOptionsV1(lod=lod))
    assert "refinement_budget" not in actual.metadata
    assert actual.metadata["fidelity"] == baseline.metadata["fidelity"]
    assert actual.metadata["warnings"] == baseline.metadata["warnings"]
    for a, b in zip(actual.surfaces, baseline.surfaces, strict=True):
        for name in ("positions", "indices", "normals", "curvature_mean", "curvature_principal"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize("phase", ["directional", "triangles"])
def test_deadline_interrupts_refinement_after_measurement(monkeypatch, phase):
    t, p = np.meshgrid(np.linspace(0., 1., 8), np.linspace(0., 1., 9), indexing="ij")
    points = np.stack((100 * t, 100 * p, 4 * t * p), axis=-1)
    normals = np.broadcast_to([0., 0., 1.], points.shape)
    clock = {"now": 0.}
    monkeypatch.setattr(fidelity, "time", SimpleNamespace(perf_counter=lambda: clock["now"]))
    calls = []
    if phase == "directional":
        original = fidelity._IntervalErrors.__call__

        def measured(*args, **kwargs):
            calls.append(1)
            result = original(*args, **kwargs)
            clock["now"] = 2.
            return result

        monkeypatch.setattr(fidelity._IntervalErrors, "__call__", measured)
    else:
        original = fidelity.emitted_triangle_errors

        def measured(*args, **kwargs):
            calls.append(1)
            result = original(*args, **kwargs)
            clock["now"] = 2.
            return result

        monkeypatch.setattr(fidelity, "emitted_triangle_errors", measured)
    ti, pi, achieved = fidelity.adaptive_grid_indices(
        points, normals, [0, 7], [0, 4, 8], max_chord_error_mm=0.001,
        max_normal_step_deg=180., max_vertices=50, closed_phi=False,
        refinement_deadline=1.,
    )
    assert len(calls) == 1
    assert ti[0] == 0 and ti[-1] == 7
    assert pi[0] == 0 and pi[-1] == 8
    assert len(ti) * len(pi) <= 50
    assert achieved["refinement_time_limited"] is True
    assert achieved["measurement_complete"] is False
    assert achieved["max_chord_error_mm"] is None


def test_source_cap_refinement_shares_the_deadline(monkeypatch):
    clock = {"now": 0.}
    timer = SimpleNamespace(perf_counter=lambda: clock["now"])
    monkeypatch.setattr(api, "time", timer)
    monkeypatch.setattr(fidelity, "time", timer)
    original = api._source_cap
    calls = []

    def cap_uses_remaining_budget(*args, **kwargs):
        calls.append(1)
        surface, measured, details = original(*args, **kwargs)
        measured["max_normal_step_deg"] = 180.
        clock["now"] = 6.
        return surface, measured, details

    monkeypatch.setattr(api, "_source_cap", cap_uses_remaining_budget)
    geometry = build_preview_geometry(
        copy.deepcopy(_OSSE),
        PreviewOptionsV1(lod="coarse", max_normal_step_deg=0.35, max_vertices=1000),
    )
    assert len(calls) == 1
    assert geometry.metadata["refinement_budget"]["exhausted"] is True
    cap = geometry.metadata["fidelity"]["source_cap"]
    assert cap["refinement_time_limited"] is True
    assert cap["measurement_complete"] is True
    assert cap["max_normal_step_deg_achieved"] == 180.
    assert sum("time budget" in w for w in geometry.metadata["warnings"]) == 1


def test_freeform_fixed_morph_crossing_away_from_the_throat_refuses(monkeypatch):
    _old_twisted_freeform_layout(monkeypatch)
    config = copy.deepcopy(FREEFORM_MORPH)
    config["morph"].update(morphCorner="25", morphFixed="0.2")
    with pytest.raises(ValueError, match="horn.outer: inconsistent local orientation"):
        build_preview_geometry(config, PreviewOptionsV1(lod="fine", include_curvature=False))
