"""Enclosure baffle and edge-treatment contracts for ``hornlab.preview/1``.

The reference here is the geometry the solver actually builds. ``build_sector``
draws a chamfered box edge with straight ``line()`` roundover curves, so each
quadrant of a chamfer is two planar side parallelograms plus one planar corner
triangle -- never a curved surface being approximated. These tests measure the
emitted band against that analytic surface rather than against a golden
tessellation, so a future retessellation is free to change triangle counts and
still has to land on the right planes.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from hornlab_mesher.preview.api import (
    PreviewOptionsV1,
    build_preview_geometry,
    build_viewport_geometry_from_config,
)

BASE_ENCLOSURE = {
    "formula": "R-OSSE",
    "mode": "enclosure",
    "profile": {
        "R_mm": 150.0,
        "r0_mm": 12.7,
        "a0_deg": 15.5,
        "a_deg": 55.0,
        "k": 1.0,
        "q": 1.0,
        "m": 0.85,
        "r": 0.35,
        "b": 0.4,
        "tmax": 1.0,
    },
    "enclosure": {"depth_mm": 150.0, "edge_mm": 18.0, "edge_type": 2},
}


def config(**enclosure):
    merged = copy.deepcopy(BASE_ENCLOSURE)
    merged["enclosure"].update(enclosure)
    return merged


def preview(cfg, lod="fine"):
    geometry = build_preview_geometry(cfg, PreviewOptionsV1(lod=lod))
    return {surface.role: surface for surface in geometry.surfaces}


def triangles(surface):
    points = np.asarray(surface.positions, dtype=np.float64).reshape(-1, 3)
    return points[np.asarray(surface.indices, dtype=np.int64).reshape(-1, 3)]


def exact_chamfer_triangles(cfg):
    """The chamfer the solver builds, straight from the box bounds."""

    enclosure = build_viewport_geometry_from_config(cfg)["enclosure"]
    bounds = enclosure["bounds"]
    depth = float(enclosure["edge_depth"])
    bx0, bx1, by0, by1 = (float(bounds[key]) for key in ("bx0", "bx1", "by0", "by1"))
    ax, ay = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
    faces = []
    for z_face, sign_z in ((float(bounds["z_front"]), -1.0), (float(bounds["z_back"]), 1.0)):
        z_outer = z_face + sign_z * depth
        for sign_x, sign_y in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            ox = bx1 if sign_x > 0 else bx0
            oy = by1 if sign_y > 0 else by0
            fx, fy = ox - sign_x * depth, oy - sign_y * depth
            for quad in (
                ((fx, ay, z_face), (ox, ay, z_outer), (ox, fy, z_outer), (fx, fy, z_face)),
                ((ax, fy, z_face), (ax, oy, z_outer), (fx, oy, z_outer), (fx, fy, z_face)),
            ):
                faces.append((quad[0], quad[1], quad[2]))
                faces.append((quad[0], quad[2], quad[3]))
            faces.append(((fx, fy, z_face), (ox, fy, z_outer), (fx, oy, z_outer)))
    return np.asarray(faces, dtype=np.float64)


def distance_to_triangles(points, faces):
    """Exact point-to-triangle-soup distance, one point at a time."""

    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    ab, ac = b - a, c - a
    worst = np.empty(len(points))
    for index, point in enumerate(points):
        ap = point - a
        denominator = np.einsum("ij,ij->i", ab, ab) * np.einsum("ij,ij->i", ac, ac) - (
            np.einsum("ij,ij->i", ab, ac) ** 2
        )
        safe = np.where(np.abs(denominator) < 1.0e-30, 1.0, denominator)
        v = (
            np.einsum("ij,ij->i", ac, ac) * np.einsum("ij,ij->i", ab, ap)
            - np.einsum("ij,ij->i", ab, ac) * np.einsum("ij,ij->i", ac, ap)
        ) / safe
        w = (
            np.einsum("ij,ij->i", ab, ab) * np.einsum("ij,ij->i", ac, ap)
            - np.einsum("ij,ij->i", ab, ac) * np.einsum("ij,ij->i", ab, ap)
        ) / safe
        v, w = np.clip(v, 0.0, 1.0), np.clip(w, 0.0, 1.0)
        total = v + w
        outside = total > 1.0
        v = np.where(outside, v / np.where(total == 0.0, 1.0, total), v)
        w = np.where(outside, w / np.where(total == 0.0, 1.0, total), w)
        closest = a + v[:, None] * ab + w[:, None] * ac
        best = np.linalg.norm(point - closest, axis=1)
        for start, end in ((a, b), (b, c), (c, a)):
            edge = end - start
            t = np.clip(
                np.einsum("ij,ij->i", point - start, edge)
                / np.maximum(np.einsum("ij,ij->i", edge, edge), 1.0e-30),
                0.0,
                1.0,
            )
            best = np.minimum(
                best, np.linalg.norm(point - (start + t[:, None] * edge), axis=1)
            )
        worst[index] = best.min()
    return worst


def test_enclosure_baffle_is_one_annulus_starting_at_the_mouth():
    """No mouth_rim, and no seam partway across a flat continuous baffle.

    An enclosure horn has no wall end face -- wall thickness is forced to zero
    and the geometry model forbids outer points -- so splitting the baffle at a
    midpoint and naming the inner half ``mouth_rim`` invented a role. The
    renderer paints that role in the horn material and outlines it in edge mode,
    which drew a colour seam and a feature line across flat geometry.
    """

    surfaces = preview(config())
    assert "mouth_rim" not in surfaces

    baffle = surfaces["enclosure.front"]
    positions = np.asarray(baffle.positions, dtype=np.float64).reshape(2, -1, 3)
    mouth = np.asarray(surfaces["horn.inner"].positions, dtype=np.float64).reshape(
        -1, positions.shape[1], 3
    )[-1]
    # The baffle starts exactly on the horn mouth, so the two weld rather than
    # overlap or leave a gap.
    assert np.allclose(positions[0], mouth)
    # ... and it is planar all the way out.
    assert np.ptp(positions[..., 2]) == pytest.approx(0.0, abs=1.0e-9)


@pytest.mark.parametrize("lod", ["coarse", "fine", "inspection"])
def test_chamfer_band_lands_on_the_surface_the_solver_builds(lod):
    cfg = config(edge_type=2)
    band = triangles(preview(cfg, lod=lod)["enclosure.roundover"])
    # Sample the emitted band densely: vertices alone would pass even if whole
    # facets were ruled the wrong way between them.
    weights = np.asarray([[1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2],
                          [0.2, 0.2, 0.6], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5],
                          [0.5, 0.0, 0.5], [1.0, 0.0, 0.0]], dtype=np.float64)
    samples = np.einsum("kw,twj->ktj", weights, band).reshape(-1, 3)
    deviation = distance_to_triangles(samples, exact_chamfer_triangles(cfg))
    # The ray-aligned first ring used to rule the corner facets against mouth
    # stations instead of the plan's own corners, putting the band 2.6 mm off
    # this surface at every corner.
    assert deviation.max() < 0.01


def test_chamfer_band_is_faceted_not_smoothed():
    """Flat shading, and every triangle inside a single facet plane.

    A chamfer's tangent breaks are the geometry, so smooth normals interpolated
    across them read as blotches however finely the band is cut.
    """

    band_surface = preview(config(edge_type=2))["enclosure.roundover"]
    assert band_surface.shading == "flat"
    assert band_surface.normal_method == "exact-planar"

    normals = np.asarray(band_surface.normals, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(band_surface.indices, dtype=np.int64).reshape(-1, 3)
    band = triangles(band_surface)
    geometric = np.cross(band[:, 1] - band[:, 0], band[:, 2] - band[:, 0])
    lengths = np.linalg.norm(geometric, axis=1)
    real = lengths > 1.0e-9
    geometric = geometric[real] / lengths[real, None]
    # Each triangle's own plane matches the normal shipped for all three of its
    # vertices, so the facet is exact rather than an average across a crease.
    for corner in range(3):
        shipped = normals[faces[real][:, corner]]
        assert np.all(np.einsum("ij,ij->i", geometric, shipped) > 1.0 - 1.0e-6)


def test_chamfer_plan_does_not_spend_samples_on_straight_chords():
    """Refining fidelity must not multiply collinear chamfer vertices.

    ``sample_rounded_rect`` walks an ``edge_type == 2`` corner along the straight
    chord, so the plan is a polygon the emitted ring reproduces exactly. Sizing
    it with the fillet's 90 degree arc model bought 32 collinear samples per
    corner at fine LOD -- and, zippered against the mouth ring, one sliver per
    sample.
    """

    counts = {
        lod: len(preview(config(edge_type=2), lod=lod)["enclosure.roundover"].positions)
        for lod in ("coarse", "fine", "inspection")
    }
    assert len(set(counts.values())) == 1, counts

    fillet = {
        lod: len(preview(config(edge_type=1), lod=lod)["enclosure.roundover"].positions)
        for lod in ("coarse", "fine")
    }
    # A fillet is genuinely curved and must still refine with fidelity.
    assert fillet["fine"] > fillet["coarse"]


@pytest.mark.parametrize("lod", ["coarse", "fine", "inspection"])
def test_chamfer_reports_no_discretisation_error(lod):
    """A chamfer discretises nothing, so it may not report a fillet's error.

    The published number drove nothing useful and understated the real problem:
    the band was reported to within a few hundredths of a millimetre while
    sitting 2.6 mm off the solver's surface.
    """

    geometry = build_preview_geometry(config(edge_type=2), PreviewOptionsV1(lod=lod))
    record = geometry.metadata["fidelity"]["enclosure.roundover"]
    assert record["max_chord_error_mm_achieved"] == pytest.approx(0.0, abs=1.0e-9)
    assert record["max_normal_step_deg_achieved"] == pytest.approx(0.0, abs=1.0e-9)
    assert record["max_chord_error_mm_achieved"] <= record["max_chord_error_mm"]

    fillet = build_preview_geometry(config(edge_type=1), PreviewOptionsV1(lod=lod))
    # A fillet still measures itself: zeroing every edge treatment would hide
    # real approximation error.
    assert (
        fillet.metadata["fidelity"]["enclosure.roundover"]["max_chord_error_mm_achieved"]
        > 0.0
    )
