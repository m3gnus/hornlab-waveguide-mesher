"""Triangle role mismatches are waived only at matching local boundaries."""

import numpy as np
import pytest

import hornlab_mesher.step_prepare as sp


def _boundaries(*, other_source_role="wall", target_scale=1.0):
    # Two role boundaries at x = +/- 1. The negative-side boundary is
    # displaced by 0.002 in z; all faces share their role-transition edge.
    vertices = {
        1: (1, 0, 0), 2: (1, 1, 0), 3: (1, 0, 1), 4: (1, 0, -1),
        5: (-1, 0, 0.002), 6: (-1, target_scale, 0.002),
        7: (-1, 0, target_scale + 0.002), 8: (-1, 0, -target_scale + 0.002),
    }
    vertices = {tag: np.asarray(point, dtype=float) for tag, point in vertices.items()}
    edge_vertices = {1: (1, 2), 2: (2, 3), 3: (3, 1), 4: (2, 4), 5: (4, 1),
                     6: (5, 6), 7: (6, 7), 8: (7, 5), 9: (6, 8), 10: (8, 5)}
    face_edges = {1: (1, 2, 3), 2: (1, 4, 5), 3: (6, 7, 8), 4: (6, 9, 10)}
    edge_faces = {1: (1, 2), 2: (1,), 3: (1,), 4: (2,), 5: (2,),
                  6: (3, 4), 7: (3,), 8: (3,), 9: (4,), 10: (4,)}
    face_vertices = {1: (1, 2, 3), 2: (1, 2, 4), 3: (5, 6, 7), 4: (5, 6, 8)}
    model = sp._OccTriangleModel(vertices, face_edges, edge_vertices, edge_faces, face_vertices)
    roles = {1: "source", 2: other_source_role, 3: "source", 4: "wall"}
    return sp._OccTriangleRoleBoundaries(model, roles), roles


def _allows(boundaries, *, source_z=0.001, mirrored_z=0.001,
            hits=(4,), candidates=(3, 4), tolerance=0.01):
    return boundaries.allows(
        1, np.array([1.0, 0.025, source_z]), np.array([-1.0, 0.025, mirrored_z]),
        list(hits), list(candidates), tolerance,
    )


def test_mismatch_is_waived_near_the_same_role_boundary_on_both_sides():
    boundaries, _ = _boundaries()
    assert _allows(boundaries)


@pytest.mark.parametrize("source_z,mirrored_z", [(0.01, 0.002), (0.001, 0.02)])
def test_mismatch_away_from_either_boundary_is_not_waived(source_z, mirrored_z):
    boundaries, _ = _boundaries()
    assert not _allows(boundaries, source_z=source_z, mirrored_z=mirrored_z)


def test_boundary_limit_is_capped_by_geometric_tolerance_and_both_local_edges():
    boundaries, _ = _boundaries()
    assert not _allows(boundaries, tolerance=0.0005)
    assert not _allows(boundaries, source_z=0.006, mirrored_z=0.008)
    small_target, _ = _boundaries(target_scale=0.1)
    assert not _allows(small_target, source_z=0.001, mirrored_z=0.001)


def test_waiver_requires_a_matching_role_face_and_the_same_opposite_role():
    boundaries, _ = _boundaries()
    assert not _allows(boundaries, candidates=(4,))
    different_boundary, _ = _boundaries(other_source_role="port")
    assert not _allows(different_boundary)


def test_nonfinite_boundary_queries_never_waive_a_mismatch():
    boundaries, _ = _boundaries()
    assert not _allows(boundaries, source_z=float("nan"))
    assert not _allows(boundaries, mirrored_z=float("nan"))
    assert not _allows(boundaries, tolerance=float("inf"))


def test_mirror_verdict_applies_boundary_rule_only_to_triangle_models(monkeypatch):
    boundaries, roles = _boundaries()
    samples = {1: np.array([[1.0, 0.25, 0.001]]),
               2: np.array([[1.0, 0.25, -0.5]]),
               3: np.array([[-1.0, 0.25, 0.5]]),
               4: np.array([[-1.0, 0.25, -0.5]])}

    class MirrorIndex:
        def __init__(self, *_args):
            pass

        def candidates_for_points(self, points):
            return [[1, 2, 3, 4] for _ in points]

        def closest(self, point, _candidates):
            if point[0] < 0:
                hit = 4
            else:
                hit = 1 if point[2] > 0 else 2
            return 0.0, 0.0, [hit]

    monkeypatch.setattr(sp, "_OccMirrorIndex", MirrorIndex)
    kwargs = dict(samples=samples, roles=roles, bboxes={}, tolerance=0.01, span_tolerance=0.1)
    strict = sp.evaluate_occ_plane_symmetry("x0", **kwargs)
    tolerant = sp.evaluate_occ_plane_symmetry("x0", **kwargs, _role_boundaries=boundaries)
    assert not strict.accepted and strict.role_mismatches == 1
    assert tolerant.accepted and tolerant.role_mismatches == 0
    samples[1][0, 2] = 0.01
    assert not sp.evaluate_occ_plane_symmetry("x0", **kwargs, _role_boundaries=boundaries).accepted
