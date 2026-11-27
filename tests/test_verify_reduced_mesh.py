"""The one reduced-mesh check (``hornlab_mesher.symmetry.verify_reduced_mesh``)."""

from __future__ import annotations

import numpy as np
import pytest

from hornlab_mesher.symmetry import (
    FAILURE_CAPPED,
    FAILURE_INVERTED,
    FAILURE_LEAK,
    FAILURE_MISSING_RIM,
    FAILURE_STRADDLING,
    FAILURE_WRONG_SIDE,
    GLOBAL_FAILURES,
    SYMMETRY_SNAP_TOLERANCE_MM,
    reduced_mesh_orientation,
    verify_reduced_mesh,
)

# Each face of the box [0, 1] x [0, 1] x [1, 2] as (origin, u, v) with u x v
# its outward normal.
_FACES = {
    "x0": ((0, 0, 1), (0, 0, 1), (0, 1, 0)),
    "x1": ((1, 0, 1), (0, 1, 0), (0, 0, 1)),
    "y0": ((0, 0, 1), (1, 0, 0), (0, 0, 1)),
    "y1": ((0, 1, 1), (0, 0, 1), (1, 0, 0)),
    "zlow": ((0, 0, 1), (0, 1, 0), (1, 0, 0)),
    "zhigh": ((0, 0, 2), (1, 0, 0), (0, 1, 0)),
}


def _box(n: int = 4, *, omit: tuple[str, ...] = (), scale: float = 1.0):
    """A closed box meshed n x n per face, wound outward; ``omit`` opens faces.

    Returns the points, the triangles and, per face name, its triangle rows.
    """

    index: dict[tuple[float, float, float], int] = {}
    points: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    rows: dict[str, list[int]] = {}

    def vertex(point: np.ndarray) -> int:
        key = tuple(round(float(value), 9) for value in point)
        if key not in index:
            index[key] = len(points)
            points.append(key)
        return index[key]

    for name, (origin, u, v) in _FACES.items():
        if name in omit:
            continue
        o, du, dv = (np.asarray(item, dtype=float) for item in (origin, u, v))
        rows[name] = []
        for i in range(n):
            for j in range(n):
                a = vertex(o + du * i / n + dv * j / n)
                b = vertex(o + du * (i + 1) / n + dv * j / n)
                c = vertex(o + du * (i + 1) / n + dv * (j + 1) / n)
                d = vertex(o + du * i / n + dv * (j + 1) / n)
                rows[name] += [len(triangles), len(triangles) + 1]
                triangles += [(a, b, c), (a, c, d)]
    return (
        scale * np.asarray(points, dtype=float),
        np.asarray(triangles, dtype=np.int64),
        rows,
    )


def _quarter(**kwargs):
    points, triangles, _rows = _box(omit=("x0", "y0"), **kwargs)
    return points, triangles


def _verify(points, triangles, planes=("x0", "y0"), **kwargs):
    kwargs.setdefault("closed", True)
    kwargs.setdefault("expected_volume_sign", 1)
    return verify_reduced_mesh(points, triangles, planes, **kwargs)


def test_an_open_quarter_is_verified():
    verdict = _verify(*_quarter())
    assert verdict.ok
    assert verdict.failure_codes == () and verdict.failed_planes == ()
    assert verdict.detected_planes == ("x0", "y0")
    assert verdict.off_plane_open_edge_count == 0
    assert verdict.orientation["checked"] is True
    assert verdict.orientation["inverted_component_count"] == 0
    record = verdict.to_dict()
    assert record["verified"] is True
    assert record["reason"] == "every cut plane is open in the mesh"
    assert record["capped_planes"] == [] and record["straddling_planes"] == []


def test_a_cut_plane_without_its_open_rim_implicates_that_plane():
    points, triangles, _rows = _box(omit=("x0",), scale=10.0)
    verdict = _verify(points, triangles)
    assert not verdict.ok
    # A closed plane is both: no rim, and the cap lying in it.
    assert verdict.failure_codes == (FAILURE_MISSING_RIM, FAILURE_CAPPED)
    assert verdict.failed_planes == ("y0",)
    assert verdict.missing_rim_planes == ("y0",)
    assert verdict.capped_planes == ("y0",)
    assert "capped" in verdict.reason


def test_a_face_lying_in_an_open_cut_plane_caps_it():
    points, triangles, rows = _box(omit=("y0",), scale=10.0)
    # The x0 face is open except for one triangle in its middle: the rim is
    # there, and so is a wall on the mirror.
    keep = rows["x0"][10]
    drop = [row for row in rows["x0"] if row != keep]
    triangles = np.delete(triangles, drop, axis=0)
    verdict = _verify(points, triangles)
    assert "x0" in verdict.detected_planes
    assert verdict.failure_codes == (FAILURE_CAPPED,)
    assert verdict.failed_planes == ("x0",)
    assert dict(verdict.cap_face_counts) == {"x0": 1}
    assert verdict.capped_planes == ("x0",)
    assert verdict.to_dict()["capped_planes"] == ["x0"]


def test_a_mesh_across_a_cut_plane_implicates_it():
    points, triangles = _quarter()
    points = points - np.asarray([0.5, 0.0, 0.0])
    verdict = _verify(points, triangles)
    assert verdict.failure_codes[0] == FAILURE_STRADDLING
    assert verdict.straddling_planes == ("x0",)
    assert verdict.failed_planes == ("x0",)
    assert verdict.negative_side_planes == ("x0",)
    assert "spans both sides of x0" in verdict.reason


def test_a_domain_kept_on_the_negative_side_implicates_its_plane():
    points, triangles = _quarter()
    points = points * np.asarray([-1.0, 1.0, 1.0])
    verdict = _verify(points, triangles)
    assert verdict.failure_codes == (FAILURE_WRONG_SIDE,)
    assert verdict.wrong_side_planes == ("x0",)
    assert verdict.failed_planes == ("x0",)
    assert "negative side of x0" in verdict.reason


def test_an_off_plane_hole_fails_a_closed_model_and_names_no_plane():
    points, triangles, rows = _box(omit=("x0", "y0"))
    triangles = np.delete(triangles, rows["zhigh"][10], axis=0)
    verdict = _verify(points, triangles)
    assert verdict.failure_codes == (FAILURE_LEAK,)
    assert set(verdict.failure_codes) <= GLOBAL_FAILURES
    assert verdict.failed_planes == ()
    assert verdict.off_plane_open_edge_count == 3
    assert "leaks" in verdict.reason
    for sample in verdict.off_plane_open_edge_samples:
        assert sample[2] == pytest.approx(2.0)
    # The winding is not read off a leaking surface.
    assert verdict.orientation["checked"] is True
    assert FAILURE_INVERTED not in verdict.failure_codes


def test_an_open_model_owns_its_rim():
    """A bare horn's mouth is an open rim off every cut plane, and has no volume."""

    points, triangles, rows = _box(omit=("x0", "y0"))
    triangles = np.delete(triangles, rows["zhigh"], axis=0)
    open_model = _verify(points, triangles, closed=False)
    assert open_model.ok
    assert open_model.off_plane_open_edge_count > 0
    assert open_model.orientation["checked"] is False
    assert not _verify(points, triangles).ok


def test_a_reversed_winding_fails_unless_the_model_expects_it():
    points, triangles = _quarter()
    reversed_ = triangles[:, [0, 2, 1]]
    verdict = _verify(points, reversed_)
    assert verdict.failure_codes == (FAILURE_INVERTED,)
    assert verdict.failed_planes == ()
    assert "wound against the model they mirror" in verdict.reason
    # An infinite-baffle model is wound inward on purpose.
    assert _verify(points, reversed_, expected_volume_sign=-1).ok
    assert not _verify(points, triangles, expected_volume_sign=-1).ok


def test_a_full_domain_is_verified_and_reports_the_planes_it_finds_open():
    points, triangles = _quarter()
    verdict = _verify(points, triangles, planes=())
    assert verdict.ok
    assert verdict.reason == "full domain: no cut plane to verify"
    assert verdict.detected_planes == ("x0", "y0")
    assert verdict.orientation["checked"] is False


def test_the_tolerance_is_the_snap_band_in_millimetres():
    points, triangles = _quarter()
    nudged = points.copy()
    nudged[nudged[:, 0] == 0.0, 0] = 0.5 * SYMMETRY_SNAP_TOLERANCE_MM
    assert _verify(nudged, triangles).ok
    nudged[nudged[:, 0] == 0.5 * SYMMETRY_SNAP_TOLERANCE_MM, 0] = 2.0 * SYMMETRY_SNAP_TOLERANCE_MM
    assert not _verify(nudged, triangles).ok
    assert _verify(nudged, triangles, snap_tolerance_mm=1.0e-3).ok


def test_mode_arguments_are_explicit_and_checked():
    points, triangles = _quarter()
    with pytest.raises(TypeError):
        verify_reduced_mesh(points, triangles, ("x0",))  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="expected_volume_sign"):
        _verify(points, triangles, expected_volume_sign=0)
    with pytest.raises(ValueError, match="unknown symmetry planes"):
        _verify(points, triangles, planes=("x1",))


def test_orientation_is_judged_per_component():
    points, triangles = _quarter()
    record = reduced_mesh_orientation(points, triangles, ("x0", "y0"), expected_volume_sign=1)
    assert record["checked"] is True
    assert record["component_count"] == 1
    assert record["inverted_component_count"] == 0
    # Origin-based, the open quarter still reads its closed piece's volume.
    assert record["signed_volume_mm3"] == pytest.approx(1.0)

    inverted = reduced_mesh_orientation(
        points, triangles[:, [0, 2, 1]], ("x0", "y0"), expected_volume_sign=1
    )
    assert inverted["inverted_component_count"] == 1
    assert inverted["inverted_triangle_count"] == len(triangles)

    # One inverted body cannot hide behind a larger correct one.
    big_points, big = _quarter(scale=10.0)
    both = reduced_mesh_orientation(
        np.vstack([big_points, points]),
        np.vstack([big, triangles[:, [0, 2, 1]] + len(big_points)]),
        ("x0", "y0"),
        expected_volume_sign=1,
    )
    assert both["signed_volume_mm3"] > 0.0
    assert both["component_count"] == 2
    assert both["inverted_component_count"] == 1

    # A full domain has no mirror to be wound against.
    assert reduced_mesh_orientation(
        points, triangles[:, [0, 2, 1]], (), expected_volume_sign=1
    )["checked"] is False


def test_the_record_keeps_the_keys_its_readers_use():
    record = _verify(*_quarter()).to_dict()
    assert {
        "cut_planes", "verified", "reason", "tolerance_mm", "free_edge_count",
        "off_plane_free_edge_count", "off_plane_free_edge_samples", "detected_planes",
        "capped_planes", "straddling_planes", "wrong_side_planes", "detection",
        "failure_codes", "failed_planes", "reduced_orientation", "closed",
        "expected_volume_sign", "missing_rim_planes", "cap_face_counts",
    } == set(record)
    assert record["detection"]["plane_vertex_side_counts"]["x0"]["negative"] == 0


def test_a_snap_sliver_in_a_cut_plane_is_not_a_cap():
    """At most SYMMETRY_SLIVER_MAX_AREA_MM2: a flattened seam facet, as the builder drops."""

    from hornlab_mesher.symmetry import SYMMETRY_SLIVER_MAX_AREA_MM2

    points, triangles, rows = _box(omit=("y0",))  # 1 mm box: 1/32 mm^2 triangles
    keep = rows["x0"][10]
    triangles = np.delete(triangles, [row for row in rows["x0"] if row != keep], axis=0)
    verdict = _verify(points, triangles)
    assert FAILURE_CAPPED not in verdict.failure_codes
    assert dict(verdict.cap_face_counts) == {}
    assert SYMMETRY_SLIVER_MAX_AREA_MM2 == 0.25


def test_exempt_triangles_are_left_out_like_the_closed_shell_contract():
    """An internal interface skirt has a free rim by construction (tag INTERFACE)."""

    points, triangles = _quarter()
    skirt_points = np.asarray([(0.2, 0.2, 1.5), (0.8, 0.2, 1.5), (0.2, 0.8, 1.5)])
    all_points = np.vstack([points, skirt_points])
    skirt = np.asarray([[len(points), len(points) + 1, len(points) + 2]])
    all_triangles = np.vstack([triangles, skirt])
    assert _verify(all_points, all_triangles).failure_codes == (FAILURE_LEAK,)
    exempt = np.zeros(len(all_triangles), dtype=bool)
    exempt[-1] = True
    verdict = _verify(all_points, all_triangles, exempt_triangles=exempt)
    assert verdict.ok
    assert verdict.orientation["component_count"] == 1
    with pytest.raises(ValueError, match="one flag per triangle"):
        _verify(all_points, all_triangles, exempt_triangles=exempt[:-1])


def test_an_unreferenced_vertex_is_not_mesh_on_the_wrong_side():
    points, triangles = _quarter()
    stray = np.vstack([points, [(-5.0, -5.0, 1.5)]])
    assert _verify(stray, triangles).ok


def test_a_mesh_lying_in_the_plane_is_not_called_the_negative_side():
    flat = np.asarray([(0.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)])
    verdict = _verify(flat, np.asarray([[0, 1, 2]]), planes=("x0",), closed=False)
    assert FAILURE_WRONG_SIDE not in verdict.failure_codes
    assert verdict.failed_planes == ("x0",)
