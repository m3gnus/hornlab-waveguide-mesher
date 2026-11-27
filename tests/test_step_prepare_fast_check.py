"""Stage 4b: the certified fast mirror check and the one-sided small-feature check.

The fast check decides most mirror samples from a forward tessellation of the
faces and asks OCC only about the rest. It must never accept what the exact
scan rejects, and keeps the verdict's rule, tolerances and report meaning. The
small-feature check (R1) rejects holes, bosses and edges on one side only,
however small, and must not reject symmetric models with asymmetric face
splits.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import gmsh
import numpy as np
import pytest

from hornlab_mesher import (
    OccSurfaceGroup,
    OccSurfaceRole,
    OccSurfaceSelector,
    auto_cut_occ_geometry,
    occ_face_index,
    step_prepare,
)
from hornlab_mesher.symmetry import TOLERANCES

PLANES = ("x0", "y0", "z0")
EXACT_FIELDS = ("accepted", "reason", "points_tested", "points_off_model", "role_mismatches",
                "tolerance_step_units", "axis_min_step_units", "axis_max_step_units")


@contextmanager
def _session():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("fast-check-test")
    try:
        yield
    finally:
        gmsh.finalize()


def _one_role():
    surfaces = [tag for _dim, tag in gmsh.model.getEntities(2)]
    return [OccSurfaceGroup("all", OccSurfaceSelector(surfaces), OccSurfaceRole("opaque"))]


def _two_roles():
    surfaces = [tag for _dim, tag in gmsh.model.getEntities(2)]
    top = max(surfaces, key=lambda tag: gmsh.model.getBoundingBox(2, tag)[5])
    return [
        OccSurfaceGroup("top", OccSurfaceSelector([top]), OccSurfaceRole("source")),
        OccSurfaceGroup("rest", OccSurfaceSelector([t for t in surfaces if t != top]),
                        OccSurfaceRole("rigid")),
    ]


def _build(name):
    occ = gmsh.model.occ
    if name == "box":
        occ.addBox(-40, -30, -20, 80, 60, 40)
    elif name == "offset-box":
        occ.addBox(-35, -30, -20, 80, 60, 40)
    elif name == "cylinder":
        occ.addCylinder(0, 0, -30, 0, 0, 60, 25)
    elif name == "sphere":
        occ.addSphere(0, 0, 0, 30)
    elif name == "cone":
        occ.addCone(0, 0, -20, 0, 0, 50, 30, 8)
    elif name == "torus":
        occ.addTorus(0, 0, 0, 40, 9)
    elif name == "bored-box":
        box = occ.addBox(-40, -30, 0, 80, 60, 50)
        bore = occ.addCylinder(12, 0, -1, 0, 0, 52, 6)
        occ.cut([(3, box)], [(3, bore)])
    elif name == "two-bores":
        box = occ.addBox(-40, -30, 0, 80, 60, 50)
        a = occ.addCylinder(12, 0, -1, 0, 0, 52, 6)
        b = occ.addCylinder(-12, 0, -1, 0, 0, 52, 6)
        occ.cut([(3, box)], [(3, a), (3, b)])
    elif name == "filleted-box":
        box = occ.addBox(-40, -30, -20, 80, 60, 40)
        occ.synchronize()
        occ.fillet([box], [tag for _d, tag in gmsh.model.getEntities(1)], [6.0])
    elif name == "tilted-box":
        box = occ.addBox(-40, -30, -20, 80, 60, 40)
        occ.rotate([(3, box)], 0, 0, 0, 0, 0, 1, 0.02)
    else:  # pragma: no cover - test bug
        raise AssertionError(name)
    occ.synchronize()


MODELS = ["box", "offset-box", "cylinder", "sphere", "cone", "torus", "bored-box", "two-bores",
          "filleted-box", "tilted-box"]


@pytest.mark.parametrize("roles", ["one", "two"])
@pytest.mark.parametrize("model", MODELS)
def test_the_fast_check_gives_the_exact_scans_verdicts(model, roles):
    """Same accepted planes, reasons and counts as projecting every sample."""

    reports = {}
    for fast in (False, True):
        with _session():
            _build(model)
            groups = _one_role() if roles == "one" else _two_roles()
            result = auto_cut_occ_geometry(groups, planes=PLANES, fast_check=fast,
                                           feature_check=False)
            result.report["group_geometry"] = _group_geometry(result)
            reports[fast] = result
    exact, quick = reports[False], reports[True]
    assert quick.planes == exact.planes
    assert quick.report["group_geometry"] == exact.report["group_geometry"]
    assert quick.report["sample_points"] == exact.report["sample_points"]
    for plane in PLANES:
        a, b = exact.report["planes"][plane], quick.report["planes"][plane]
        for field in EXACT_FIELDS:
            assert a[field] == b[field], (plane, field)
        if a["max_residual_step_units"] is not None:
            # Pinned to an exact projection of a real sample: never above the
            # exact scan's peak, and within the tessellation's bound of it.
            assert b["max_residual_step_units"] <= a["max_residual_step_units"] + 1e-12
            assert a["max_residual_step_units"] - b["max_residual_step_units"] <= \
                0.25 * a["tolerance_step_units"]


def _group_geometry(result):
    """Each group's faces as (type, area, box), sorted: tag numbers aside."""
    return {
        group.name: sorted(
            (gmsh.model.getType(2, tag), round(gmsh.model.occ.getMass(2, tag), 6),
             tuple(round(v, 6) for v in gmsh.model.getBoundingBox(2, tag)))
            for tag in group.selector.surface_tags)
        for group in result.groups
    }


def _audit(model):
    """Per-sample: (false hits, extra misses) of the certified target."""
    _build(model)
    surfaces = [tag for _dim, tag in gmsh.model.getEntities(2)]
    box = gmsh.model.getBoundingBox(-1, -1)
    diagonal = float(np.linalg.norm(np.asarray(box[3:]) - box[:3]))
    index, trims = occ_face_index.build_face_index(
        surfaces, diagonal=diagonal, tolerance=5e-4 * diagonal, trims={})
    samples = {s: step_prepare.sample_occ_surface_points(s, grid=7, trims=trims) for s in surfaces}
    samples = {s: p for s, p in samples.items() if len(p)}
    stacked = np.concatenate(list(samples.values()))
    tolerance = TOLERANCES["fitted"].resolve(float(np.linalg.norm(stacked.max(0) - stacked.min(0))))
    bboxes = {s: tuple(gmsh.model.getBoundingBox(2, s)) for s in samples}
    roles = {s: "r" for s in samples}
    false_hits = extra_misses = certified = 0
    for plane in PLANES:
        axis = step_prepare.SYMMETRY_AXIS_FOR_PLANE[plane]
        plain = step_prepare._OccMirrorTarget(sorted(samples), bboxes, tolerance, None, None)
        fast = step_prepare._CertifiedOccMirrorTarget(
            sorted(samples), bboxes, tolerance, None, roles, samples=samples, axis=axis,
            face_index=index)
        for surface in sorted(samples):
            mirrored = samples[surface].copy()
            mirrored[:, axis] *= -1
            a, b = plain.match(surface, mirrored), fast.match(surface, mirrored)
            for i in range(len(mirrored)):
                false_hits += bool(b.hits[i]) and not a.hits[i]
                extra_misses += bool(a.hits[i]) and not b.hits[i]
        certified += fast.stats["certified_hits"] + fast.stats["certified_misses"]
    return false_hits, extra_misses, certified


@pytest.mark.parametrize("model", MODELS)
def test_no_sample_is_certified_a_hit_that_the_exact_scan_misses(model):
    with _session():
        false_hits, extra_misses, certified = _audit(model)
    assert false_hits == 0
    assert extra_misses == 0
    assert certified > 0


def test_a_known_residual_is_reported_exactly():
    """One side 0.4 tolerances longer than its mirror: the reported peak
    residual is that offset, from an exact projection."""

    diagonal = math.sqrt(80 ** 2 + 60 ** 2 + 40 ** 2)
    offset = 0.4 * TOLERANCES["fitted"].relative * diagonal
    with _session():
        gmsh.model.occ.addBox(-40, -30, -20, 80 + offset, 60, 40)
        gmsh.model.occ.synchronize()
        result = auto_cut_occ_geometry(_one_role(), planes=("x0",))
    verdict = result.report["planes"]["x0"]
    assert verdict["accepted"] is True
    assert verdict["max_residual_step_units"] == pytest.approx(offset, rel=1e-3)


def test_trim_polygons_give_occs_samples_bit_for_bit():
    with _session():
        for model in ("bored-box", "torus", "sphere", "cone", "filleted-box"):
            gmsh.model.remove()
            gmsh.model.add(model)
            _build(model)
            surfaces = [tag for _dim, tag in gmsh.model.getEntities(2)]
            box = gmsh.model.getBoundingBox(-1, -1)
            diagonal = float(np.linalg.norm(np.asarray(box[3:]) - box[:3]))
            _index, trims = occ_face_index.build_face_index(
                surfaces, diagonal=diagonal, tolerance=5e-4 * diagonal, trims={})
            for surface in surfaces:
                plain = step_prepare.sample_occ_surface_points(surface, grid=7)
                fast = step_prepare.sample_occ_surface_points(surface, grid=7, trims=trims)
                assert plain.tobytes() == fast.tobytes(), (model, surface)


def test_a_face_whose_trim_cannot_be_read_falls_back_to_occ(monkeypatch):
    """No polygons: the face is never a certified hit, and the verdicts stay."""

    monkeypatch.setattr(occ_face_index.FaceTrim, "build",
                        classmethod(lambda cls, surface: cls(surface, [], math.inf, False)))
    for model in ("bored-box", "offset-box"):
        results = {}
        for fast in (False, True):
            with _session():
                _build(model)
                results[fast] = auto_cut_occ_geometry(_one_role(), planes=PLANES,
                                                      fast_check=fast, feature_check=False)
        assert results[True].planes == results[False].planes
        for plane in PLANES:
            for field in EXACT_FIELDS:
                assert (results[True].report["planes"][plane][field]
                        == results[False].report["planes"][plane][field])


def test_point_triangle_distance_matches_a_dense_search():
    rng = np.random.default_rng(7)
    triangles = rng.normal(size=(4000, 3, 3))
    triangles[:50, 2] = triangles[:50, 1]           # degenerate ones too
    points = rng.normal(size=(4000, 3)) * 2.0
    exact = occ_face_index.point_triangle_distance(points, triangles)
    best = np.full(len(points), np.inf)
    for i in range(41):
        for j in range(41 - i):
            u, v = i / 40.0, j / 40.0
            q = triangles[:, 0] + u * (triangles[:, 1] - triangles[:, 0]) + v * (
                triangles[:, 2] - triangles[:, 0])
            best = np.minimum(best, np.linalg.norm(points - q, axis=1))
    assert np.all(exact <= best + 1e-12)
    assert np.all(best - exact <= 0.1 * (1.0 + np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
                                         + np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)))


# -- R1: one-sided small features ------------------------------------------------


def _plate(holes=(), bosses=(), thickness=0.4):
    occ = gmsh.model.occ
    plate = occ.addBox(-200.0, -150.0, 0.0, 400.0, 300.0, thickness)
    if holes:
        tools = [(3, occ.addCylinder(x, y, -1.0, 0.0, 0.0, thickness + 2.0, r)) for x, y, r in holes]
        occ.cut([(3, plate)], tools)
    if bosses:
        tools = [(3, occ.addCylinder(x, y, thickness, 0.0, 0.0, h, r)) for x, y, r, h in bosses]
        occ.fuse([(3, plate)], tools)
    occ.synchronize()
    return _one_role()


# The 7 x 7 floor grid's sample columns and rows on the 400 x 300 plate.
_XS = [-200 + (i + 0.5) * 400 / 7 for i in range(7)]
_YS = [-150 + (j + 0.5) * 300 / 7 for j in range(7)]


@pytest.mark.parametrize("radius", [0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("where", [(_XS[4], _YS[5]), (_XS[5], _YS[5]), (85.0, 65.0), (150.0, -20.0)])
def test_a_one_sided_hole_of_any_size_is_not_cut(radius, where):
    with _session():
        result = auto_cut_occ_geometry(_plate(holes=[(where[0], where[1], radius)]),
                                       planes=("x0", "y0"))
        assert result.planes == ()
        for plane in ("x0", "y0"):
            assert result.report["planes"][plane]["feature_mismatches"] > 0


def test_mirrored_holes_are_cut():
    holes = [(sx * 85.0, sy * 65.0, 2.0) for sx in (1, -1) for sy in (1, -1)]
    with _session():
        result = auto_cut_occ_geometry(_plate(holes=holes), planes=("x0", "y0"))
        assert result.planes == ("x0", "y0")
        for plane in ("x0", "y0"):
            assert result.report["planes"][plane]["feature_mismatches"] == 0


@pytest.mark.parametrize("height", [0.05, 0.2, 3.0])
def test_a_one_sided_boss_below_the_tolerance_is_not_cut(height):
    """A boss lower than the mirror tolerance (0.25 mm here) is within it by
    distance alone; its walls and sharp edges still have no mirror."""

    with _session():
        result = auto_cut_occ_geometry(_plate(bosses=[(85.0, 65.0, 1.5, height)]),
                                       planes=("x0", "y0"))
        assert result.planes == ()


def test_mirrored_bosses_are_cut():
    bosses = [(sx * 85.0, sy * 65.0, 1.5, 0.2) for sx in (1, -1) for sy in (1, -1)]
    with _session():
        result = auto_cut_occ_geometry(_plate(bosses=bosses), planes=("x0", "y0"))
        assert result.planes == ("x0", "y0")


def test_a_hole_in_an_open_sheet_is_not_cut():
    """No wall to land anywhere: only the hole's free edge, whose mirror lies
    inside the sheet, gives it away."""

    with _session():
        occ = gmsh.model.occ
        sheet = occ.addRectangle(-200.0, -150.0, 0.0, 400.0, 300.0)
        disk = occ.addDisk(85.0, 65.0, 0.0, 3.0, 3.0)
        occ.cut([(2, sheet)], [(2, disk)])
        occ.synchronize()
        result = auto_cut_occ_geometry(_one_role(), planes=("x0", "y0"))
        assert result.planes == ()
        kinds = {sample["kind"] for plane in ("x0", "y0")
                 for sample in result.report["planes"][plane]["failure_samples"]}
        assert "edge_mismatch" in kinds


def test_a_face_split_on_one_side_only_is_still_cut():
    """A split line inside one flat face is not a feature: the mirror test
    must not reject a symmetric model for it. Two parallel sheets, the top
    one made of two coplanar faces split at x = 15."""

    with _session():
        occ = gmsh.model.occ
        left = occ.addRectangle(-40, -30, 20, 55, 60)
        right = occ.addRectangle(15, -30, 20, 25, 60)
        occ.fragment([(2, left)], [(2, right)])
        occ.addRectangle(-40, -30, -20, 80, 60)
        occ.synchronize()
        assert len(gmsh.model.getEntities(2)) == 3
        result = auto_cut_occ_geometry(_one_role(), planes=("x0", "y0"))
        assert result.planes == ("x0", "y0")
        for plane in ("x0", "y0"):
            assert result.report["planes"][plane]["feature_mismatches"] == 0


def test_the_feature_check_keeps_the_seams_of_periodic_faces():
    """A cylinder's seam lies on one side; seams are not features."""

    with _session():
        _build("cylinder")
        result = auto_cut_occ_geometry(_one_role(), planes=("x0", "y0"))
        assert result.planes == ("x0", "y0")
