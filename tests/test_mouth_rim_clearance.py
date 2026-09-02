"""The acoustic surface must not chord across a rollback and out through the shell.

The mouth-side sibling of ``test_wall_clearance.py``. That guard bounds the
outer shell's AZIMUTHAL chord, which is tight near the throat and never bites
at the mouth. An R-OSSE rollback turns the MERIDIAN through more than a right
angle inside a few millimetres, and a single element sized by
``mouth_res_mm`` chords the whole of it -- so the bore, not the shell, is what
passes through the wall.

Measured on a stock R-OSSE (R 150, r0 12.7, a 60, a0 15.5) with a 3 mm wall:
341 self-intersecting triangle pairs at r 138-145 mm, z 36-37 mm, every one of
them an acoustic-surface facet crossing the outer shell behind it. Making the
shell FINER made it worse (0 pairs at 15 mm rear resolution, 341 at 7 mm, 719
at 5 mm), which is what rules out the rear guard's remedy: a finer shell only
resolves more of a crossing it did not cause.
"""

from __future__ import annotations

import numpy as np
import pytest

from hornlab_mesher.builders.point_grid_freestanding import (
    _meridian_curvature_radius_mm,
)
from hornlab_mesher.density import (
    _MOUTH_CLEARANCE_FRACTION,
    _mouth_clearance_radial_ramp,
)


def _rosse_config(**mesh: float) -> dict:
    return {
        "formula": "R-OSSE",
        "mode": "freestanding",
        "profile": {
            "R_mm": 150.0,
            "r0_mm": 12.7,
            "a_deg": 60.0,
            "a0_deg": 15.5,
            "k": 1.0,
            "q": 0.995,
        },
        "cross_section": {"exponent": 2.0, "aspect_ratio": 1.0},
        "mesh": {
            "throat_res_mm": 4.0,
            "mouth_res_mm": 26.0,
            "rear_res_mm": 15.0,
            "wall_thickness_mm": 3.0,
            "quadrants": 1234,
            **mesh,
        },
    }


def _osse_config(**mesh: float) -> dict:
    """The same horn without a rollback: nothing for this bound to do."""

    return {
        "formula": "OSSE",
        "mode": "freestanding",
        "profile": {"L_mm": 120.0, "r0_mm": 12.7, "a_deg": 60.0, "a0_deg": 15.5},
        "cross_section": {"exponent": 2.0, "aspect_ratio": 1.0},
        "mesh": {
            "throat_res_mm": 4.0,
            "mouth_res_mm": 26.0,
            "rear_res_mm": 15.0,
            "wall_thickness_mm": 3.0,
            "quadrants": 1234,
            **mesh,
        },
    }


def _segment_crosses_triangle(start, end, triangle) -> bool:
    """Proper crossing only: the segment must pierce the triangle's interior."""

    first = triangle[1] - triangle[0]
    second = triangle[2] - triangle[0]
    direction = end - start
    normal = np.cross(first, second)
    denominator = float(np.dot(normal, direction))
    if abs(denominator) < 1.0e-15:
        return False
    along = float(np.dot(normal, triangle[0] - start)) / denominator
    if not (1.0e-9 < along < 1.0 - 1.0e-9):
        return False
    point = start + along * direction - triangle[0]
    d00 = float(np.dot(first, first))
    d01 = float(np.dot(first, second))
    d11 = float(np.dot(second, second))
    determinant = d00 * d11 - d01 * d01
    if abs(determinant) < 1.0e-20:
        return False
    d20 = float(np.dot(point, first))
    d21 = float(np.dot(point, second))
    u = (d11 * d20 - d01 * d21) / determinant
    v = (d00 * d21 - d01 * d20) / determinant
    return u > 1.0e-9 and v > 1.0e-9 and u + v < 1.0 - 1.0e-9


def _crossing_pairs(points_mm: np.ndarray, triangles: np.ndarray) -> int:
    """Count triangle pairs that pass through each other.

    Deliberately not the same implementation as the consumer's detector: this
    one only has to be right, not fast, and an independent one cannot inherit
    the consumer's blind spots. Sweep on x, reject by box, then test every edge
    of each triangle against the other's interior.
    """

    corners = points_mm[triangles]
    lower = corners.min(axis=1)
    upper = corners.max(axis=1)
    order = np.argsort(lower[:, 0], kind="stable")
    found = 0
    for position, left in enumerate(order):
        for right in order[position + 1 :]:
            if lower[right, 0] > upper[left, 0]:
                break
            if np.any(upper[right, 1:] < lower[left, 1:]) or np.any(
                lower[right, 1:] > upper[left, 1:]
            ):
                continue
            for first, second in ((left, right), (right, left)):
                edges = corners[first]
                if any(
                    _segment_crosses_triangle(
                        edges[index], edges[(index + 1) % 3], corners[second]
                    )
                    for index in range(3)
                ):
                    found += 1
                    break
    return found


def _mouth_band(points_mm: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Triangles near the mouth rim, where the rollback lives."""

    centroid = points_mm[triangles].mean(axis=1)
    radius = np.hypot(centroid[:, 0], centroid[:, 1])
    return triangles[radius > 0.85 * float(radius.max())]


def _built_mesh(config: dict, path):
    meshio = pytest.importorskip("meshio")
    from hornlab_mesher.config_builder import build_from_config

    result = build_from_config(config, path)
    mesh = meshio.read(str(result.mesh_path))
    triangles = np.vstack(
        [block.data for block in mesh.cells if block.type == "triangle"]
    )
    return result, np.asarray(mesh.points, dtype=float) * 1000.0, triangles


@pytest.mark.parametrize("rear_res_mm", [7.0, 15.0])
def test_a_thin_walled_rollback_does_not_cross_itself(tmp_path, rear_res_mm) -> None:
    """The regression. Both resolutions, because only one of them used to fail.

    At 15 mm the shell was coarse enough to chord across its own rim and land
    outside the bore's chord, so the mesh happened to come out clean while the
    acoustic surface was still 3 mm out of place. At 7 mm the same defect
    produced 341 crossing pairs. A guard that only fixed the second would be
    resting on that accident.
    """

    pytest.importorskip("gmsh")
    result, points, triangles = _built_mesh(
        _rosse_config(rear_res_mm=rear_res_mm), tmp_path / f"rosse-{rear_res_mm}.msh"
    )
    # The mesh first, so this test fails on the defect and not on a metadata
    # key that happens to be missing with the guard removed.
    band = _mouth_band(points, triangles)
    assert len(band) > 0
    assert _crossing_pairs(points, band) == 0

    clearance = (result.metadata or {}).get("mouthRimClearance")
    assert clearance is not None
    assert clearance["capActive"] is True
    assert clearance["cappedSizeAtRimMm"] < clearance["requestedMouthResolutionMm"]


def test_the_cap_stays_local_to_the_rollback(tmp_path) -> None:
    """It must not pay for the rim out of the whole bore.

    Every station constrains the fitted ramp, so one station that needs nothing
    can still flatten it. The stock R-OSSE throat turns at 14.7 mm, which
    licenses only 8.7 mm of chord against a 3 mm wall -- but the throat-to-mouth
    interpolation already asks for 4 mm there, so the bound was never in danger.
    Letting it into the fit dropped the slope from 0.95 to 0.034 mm per mm and
    pinned the entire bore at the rim's size: 1,672 triangles became 8,072.
    """

    pytest.importorskip("gmsh")
    _capped, points, triangles = _built_mesh(
        _rosse_config(), tmp_path / "capped.msh"
    )
    centroid = points[triangles].mean(axis=1)
    radius = np.hypot(centroid[:, 0], centroid[:, 1])
    edges = np.max(
        [
            np.linalg.norm(points[triangles[:, 1]] - points[triangles[:, 0]], axis=1),
            np.linalg.norm(points[triangles[:, 2]] - points[triangles[:, 1]], axis=1),
            np.linalg.norm(points[triangles[:, 0]] - points[triangles[:, 2]], axis=1),
        ],
        axis=0,
    )
    # The flare below the rollback keeps the size the interpolation asked for.
    flare = (radius > 60.0) & (radius < 110.0) & (centroid[:, 2] > 20.0)
    assert np.any(flare)
    assert float(np.max(edges[flare])) > 15.0


def test_a_horn_without_a_rollback_is_left_alone(tmp_path) -> None:
    """No rollback, no cap, and no field that could perturb the sizing."""

    pytest.importorskip("gmsh")
    result, _points, _triangles = _built_mesh(_osse_config(), tmp_path / "osse.msh")
    clearance = (result.metadata or {}).get("mouthRimClearance")
    assert clearance is not None
    assert "capActive" not in clearance


def test_meridian_curvature_sees_what_ring_curvature_cannot() -> None:
    """A rim ring is a 150 mm circle while its meridian turns at 4 mm."""

    angle = np.linspace(0.0, 0.5 * np.pi, 24)
    # One meridian: a quarter-circle roundover of radius 4 mm centred at
    # (146, 36), swept to a full ring at radius ~150 mm.
    meridian = np.stack(
        (146.0 + 4.0 * np.cos(angle), np.zeros_like(angle), 36.0 + 4.0 * np.sin(angle)),
        axis=1,
    )
    radius = _meridian_curvature_radius_mm(meridian[None, :, :])
    interior = radius[0, 1:-1]
    assert np.allclose(interior, 4.0, rtol=1.0e-6)
    # The ends carry no triple and must not invent a constraint.
    assert np.isinf(radius[0, 0]) and np.isinf(radius[0, -1])


def test_the_ramp_is_tightest_at_the_rim_and_relaxes_inward() -> None:
    """``max(base, intercept + slope * -radius)`` must fall as radius grows."""

    curvature = np.asarray([80.0, 40.0, 20.0, 10.0, 5.0, 3.6])
    radial = np.asarray([130.0, 136.0, 141.0, 145.0, 148.0, 150.0])
    base, slope, intercept = _mouth_clearance_radial_ramp(
        curvature, radial, wall_mm=3.0, size_fallback_mm=26.0
    )
    values = np.maximum(base, intercept + slope * (-radial))
    assert slope > 0.0
    assert np.all(np.diff(values) <= 1.0e-9)
    # Admissible everywhere: the ramp never licenses more than the bound.
    sagitta = _MOUTH_CLEARANCE_FRACTION * 3.0
    bound = 2.0 * np.sqrt(2.0 * sagitta * curvature - sagitta * sagitta) / 1.25
    assert np.all(values <= bound + 1.0e-9)
