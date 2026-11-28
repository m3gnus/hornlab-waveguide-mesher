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
        surfaces, diagonal=diagonal, tolerance=5e-4 * diagonal, nudge_rel=1e-3, trims={})
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
        certified += (fast.stats["proven_plane_hits"] + fast.stats["confirmed_hits"]
                      + fast.stats["certified_misses"])
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
                surfaces, diagonal=diagonal, tolerance=5e-4 * diagonal, nudge_rel=1e-3, trims={})
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


# -- the cut: solids cut as solids ---------------------------------------------


def _topology():
    return (len(gmsh.model.getEntities(3)), len(gmsh.model.getEntities(2)),
            len(gmsh.model.getEntities(1)), len(gmsh.model.getEntities(0)))


@pytest.mark.parametrize("model", ["box", "cylinder", "two-bores", "filleted-box", "cone"])
@pytest.mark.parametrize("planes", [("x0",), ("x0", "y0")])
def test_a_solid_is_cut_as_a_solid_into_the_same_open_faces(model, planes):
    """Same kept faces, same shared curves and points, no cap, no copy left."""

    results = {}
    for fast in (False, True):
        with _session():
            _build(model)
            result = auto_cut_occ_geometry(_one_role(), planes=planes, fast_check=fast,
                                           feature_check=False)
            results[fast] = (result.planes, _group_geometry(result), _topology(),
                             (result.report.get("cut") or {}).get("route"))
    assert results[True][0] == results[False][0] == planes
    assert results[True][1] == results[False][1]
    assert results[True][2] == results[False][2]
    assert results[True][3] == "solids"
    assert results[False][3] is None


def test_an_open_shell_is_cut_face_by_face():
    with _session():
        gmsh.model.occ.addRectangle(-40, -30, 20, 80, 60)
        gmsh.model.occ.addRectangle(-40, -30, -20, 80, 60)
        gmsh.model.occ.synchronize()
        result = auto_cut_occ_geometry(_one_role(), planes=("x0",))
        assert result.planes == ("x0",)
        assert result.report["cut"]["route"] == "faces"


def test_a_solid_with_a_face_in_the_cut_plane_is_cut_face_by_face():
    with _session():
        occ = gmsh.model.occ
        left = occ.addBox(-40, -30, -20, 40, 60, 40)
        right = occ.addBox(0, -30, -20, 40, 60, 40)
        occ.fragment([(3, left)], [(3, right)])
        occ.synchronize()
        result = auto_cut_occ_geometry(_one_role(), planes=("x0",), feature_check=False)
        assert (result.report.get("cut") or {}).get("route") in (None, "faces")


def test_a_solid_cut_that_cannot_be_proven_falls_back_to_the_face_cut(monkeypatch):
    def unproven(*_args, **_kwargs):
        raise RuntimeError("OCC cut left surface 999 with no parent")

    with _session():
        _build("two-bores")
        expected = auto_cut_occ_geometry(_one_role(), planes=("x0", "y0"), fast_check=False)
        expected_geometry, expected_topology = _group_geometry(expected), _topology()
    monkeypatch.setattr(step_prepare, "_attribute_cut_faces", unproven)
    with _session():
        _build("two-bores")
        result = auto_cut_occ_geometry(_one_role(), planes=("x0", "y0"), feature_check=False)
        assert result.report["cut"]["route"] == "solids, then faces"
        assert _group_geometry(result) == expected_geometry
        assert _topology() == expected_topology


def test_a_cap_is_never_kept_and_a_stray_face_is_never_kept(monkeypatch):
    """Faces no parent claims: the cap (in the plane) is removed; any other
    face makes the solid cut fall back."""

    real = step_prepare._ClaimJudge.__call__
    calls = []

    def judge(self, parent, child):
        calls.append(child)
        return real(self, parent, child)

    monkeypatch.setattr(step_prepare._ClaimJudge, "__call__", judge)
    with _session():
        _build("box")
        result = auto_cut_occ_geometry(_one_role(), planes=("x0",), feature_check=False)
        assert result.report["cut"]["route"] == "solids"
        kept = [tag for group in result.groups for tag in group.selector.surface_tags]
        assert all(abs(gmsh.model.getBoundingBox(2, tag)[0]) > 1e-9
                   or abs(gmsh.model.getBoundingBox(2, tag)[3]) > 1e-9 for tag in kept)
        assert len(kept) == 5          # the x0 cap is gone
        assert gmsh.model.getEntities(3) == []


@pytest.mark.parametrize(("name", "call"), [("copy", "solids"), ("intersect", 1), ("synchronize", 4)])
def test_a_failure_inside_the_solid_cut_leaves_the_face_cut_as_before(monkeypatch, name, call):
    """An OCC failure while cutting the copied solids discards that attempt;
    the originals are then cut face by face, exactly as without the fast check."""

    with _session():
        _build("two-bores")
        expected = auto_cut_occ_geometry(_one_role(), planes=("x0",), fast_check=False)
        expected_geometry, expected_topology = _group_geometry(expected), _topology()
    real = getattr(gmsh.model.occ, name)
    count = {"n": 0}

    def failing(*args, **kwargs):
        count["n"] += 1
        if count["n"] == call or (call == "solids" and args and args[0] and args[0][0][0] == 3):
            raise RuntimeError(f"injected {name} failure")
        return real(*args, **kwargs)

    with _session():
        _build("two-bores")
        monkeypatch.setattr(gmsh.model.occ, name, failing)
        result = auto_cut_occ_geometry(_one_role(), planes=("x0",), feature_check=False)
        monkeypatch.undo()
        assert result.report["cut"]["route"] == "solids, then faces"
        assert _group_geometry(result) == expected_geometry
        assert _topology() == expected_topology


def test_a_face_within_tolerance_of_a_neighbour_keeps_only_its_own_parent():
    """A fillet narrower than the mirror tolerance lies within tolerance of the
    faces it joins; each piece of it still belongs to the fillet alone, so a
    group never gains a neighbour's face."""

    def build():
        occ = gmsh.model.occ
        box = occ.addBox(-200, -150, -100, 400, 300, 200)
        occ.synchronize()
        along_x = [tag for _d, tag in gmsh.model.getEntities(1)
                   if abs(gmsh.model.getBoundingBox(1, tag)[3] - gmsh.model.getBoundingBox(1, tag)[0]) > 300]
        occ.fillet([box], along_x, [0.2])
        occ.synchronize()
        surfaces = [t for _d, t in gmsh.model.getEntities(2)]
        fillets = [t for t in surfaces if gmsh.model.getType(2, t) != "Plane"]
        assert len(fillets) == 4
        return [
            OccSurfaceGroup("fillets", OccSurfaceSelector(fillets), OccSurfaceRole("src")),
            OccSurfaceGroup("rest", OccSurfaceSelector([t for t in surfaces if t not in fillets]),
                            OccSurfaceRole("rigid")),
        ]

    results = {}
    for fast in (False, True):
        with _session():
            result = auto_cut_occ_geometry(build(), planes=("x0",), fast_check=fast,
                                           feature_check=False)
            results[fast] = (_group_geometry(result), result.report["cut"]["surfaces_split"],
                             result.report["cut"].get("route"))
    assert results[True][0] == results[False][0]
    assert results[True][1] == results[False][1]
    assert results[True][2] == "solids"


def test_a_caller_can_keep_the_face_by_face_cut():
    """``solid_cut=False``: the fast check still decides, the cut is the
    face-by-face one -- the same tags and parentage as without the fast check."""

    results = {}
    for fast in (False, True):
        with _session():
            _build("two-bores")
            result = auto_cut_occ_geometry(_one_role(), planes=("x0", "y0"), fast_check=fast,
                                           feature_check=False, solid_cut=False)
            results[fast] = ([list(g.selector.surface_tags) for g in result.groups],
                             result.parent_to_children, result.report["cut"].get("route"))
    assert results[True][:2] == results[False][:2]
    assert results[True][2] == "faces"


# -- round 2: what the fast path may prove ----------------------------------------


def _ridge_sheet(height):
    """Opus's probe: a flat open sheet whose right part is a dense-knot cubic
    B-spline carrying one narrow ridge (no edges) that fits between the
    chord probes of a coarse cell, on the +x side only."""
    occ = gmsh.model.occ
    knots_n, column = 144, 83
    a = occ.addRectangle(-50, -30, 0, 50, 60)
    uc = (12 * 6 + 10) / knots_n
    m = 50.0 * 5.5 / 7.0
    s = (m - 50.0 * uc) / (1.0 - uc)
    length = 50.0 - s
    knots = [0.0] * 4 + [i / knots_n for i in range(1, knots_n)] + [1.0] * 4
    b = occ.addRectangle(0, -30, 0, s, 60)
    count_u = knots_n + 3
    greville = [(knots[i + 1] + knots[i + 2] + knots[i + 3]) / 3 for i in range(count_u)]
    ys = [-30.0, -1.0, -0.4, 0.4, 1.0, 30.0]
    tags = [occ.addPoint(s + length * greville[i], y, height if (i == column and r in (2, 3)) else 0.0)
            for r, y in enumerate(ys) for i in range(count_u)]
    spline = occ.addBSplineSurface(
        tags, count_u, degreeU=3, degreeV=1,
        knotsU=[0.0] + [i / knots_n for i in range(1, knots_n)] + [1.0],
        knotsV=[(y + 30) / 60 for y in ys], multiplicitiesU=[4] + [1] * (knots_n - 1) + [4],
        multiplicitiesV=[2, 1, 1, 1, 1, 2])
    occ.remove([(0, t) for t in tags])
    occ.fragment([(2, a), (2, b), (2, spline)], [])
    occ.synchronize()


@pytest.mark.parametrize("height", [0.3, 5.0])
def test_a_narrow_spline_ridge_between_chord_probes_is_not_accepted(height):
    """The chord error is measured, not proven: a hit on a curved face is
    confirmed by OCC, so the hidden ridge is rejected like the exact scan."""
    verdicts = {}
    for fast in (False, True):
        with _session():
            _ridge_sheet(height)
            result = auto_cut_occ_geometry(_one_role(), planes=("x0",), fast_check=fast)
            verdicts[fast] = result.report["planes"]["x0"]
    assert verdicts[False]["accepted"] is False
    assert verdicts[True]["accepted"] is False
    assert verdicts[True]["points_off_model"] == verdicts[False]["points_off_model"]


def test_a_crease_inside_one_face_mirrors_a_sharp_edge():
    """Opus's probe: one side kinked by two planes meeting at an edge, the
    other by a C0 knot inside one spline face -- the same shape."""
    def build():
        occ = gmsh.model.occ

        def height(x):
            return 0.0 if x <= 25 else x - 25.0

        def quad(x0, x1):
            pts = [occ.addPoint(x0, -30, height(abs(x0))), occ.addPoint(x1, -30, height(abs(x1))),
                   occ.addPoint(x1, 30, height(abs(x1))), occ.addPoint(x0, 30, height(abs(x0)))]
            lines = [occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
            return occ.addPlaneSurface([occ.addCurveLoop(lines)])

        a, b = quad(0, 25), quad(25, 50)
        tags = [occ.addPoint(x, y, height(abs(x))) for y in (-30, 30) for x in (-50, -25, 0)]
        c = occ.addBSplineSurface(tags, 3, degreeU=1, degreeV=1, knotsU=[0, 0.5, 1], knotsV=[0, 1],
                                  multiplicitiesU=[2, 1, 2], multiplicitiesV=[2, 2])
        occ.fragment([(2, a), (2, b), (2, c)], [])
        occ.synchronize()

    with _session():
        build()
        exact = auto_cut_occ_geometry(_one_role(), planes=("x0",), fast_check=False)
    with _session():
        build()
        fast = auto_cut_occ_geometry(_one_role(), planes=("x0",))
    assert exact.planes == fast.planes == ("x0",)
    assert fast.report["planes"]["x0"]["feature_mismatches"] == 0


def test_a_small_hole_inside_one_cell_never_makes_that_cell_inside():
    """Neon P1-2: a trim curve wholly inside a cell, with a polygon margin
    wider than half a cell, must still mark the cell."""
    outer = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    hole = np.array([[5.48, 5.48], [5.52, 5.48], [5.52, 5.52], [5.48, 5.52]])
    trim = occ_face_index.FaceTrim(1, [outer, hole], 0.55, True)
    grid = np.linspace(0.0, 10.0, 11)
    inside, keep = occ_face_index._trim_cells(trim, grid, grid)
    assert not inside[5, 5]
    assert keep[5, 5]


def test_a_face_without_an_area_from_occ_has_no_inside_cells():
    with _session():
        gmsh.model.occ.addRectangle(-10, -10, 0, 20, 20)
        gmsh.model.occ.synchronize()
        trim = occ_face_index.FaceTrim.build(1)
        mesh = occ_face_index.build_face_mesh(1, trim, cell_size=1.0, delta_target=0.01,
                                              area=None, nudge_rel=1e-3)
        assert mesh is not None and not np.any(mesh.inside)
        assert trim.ok is False


@pytest.mark.parametrize("model", ["holed-plate", "bored-box", "cylinder", "filleted-box"])
def test_the_shape_check_never_trusts_a_triangle_normal(model, monkeypatch):
    """Neon P1-4: the small-feature decision comes from OCC normals (a plane
    triangle's normal only lets a proven plane hit skip OCC); with every
    triangle's normal turned sideways no verdict changes."""
    def reference():
        with _session():
            if model == "holed-plate":
                groups = _plate(holes=[(85.0, 65.0, 2.0)])
            else:
                _build(model)
                groups = _one_role()
            return auto_cut_occ_geometry(groups, planes=PLANES).report["planes"]

    expected = reference()
    real_init = occ_face_index.FaceIndex.__init__

    def sideways(self, meshes):
        real_init(self, meshes)
        turned = np.cross(self.normals, [0.3, 0.4, 0.5])
        self.normals = turned / np.maximum(np.linalg.norm(turned, axis=1), 1e-12)[:, None]

    monkeypatch.setattr(occ_face_index.FaceIndex, "__init__", sideways)
    got = reference()
    for plane in PLANES:
        for field in EXACT_FIELDS + ("feature_mismatches",):
            assert got[plane].get(field) == expected[plane].get(field), (plane, field)


def test_a_certified_miss_stays_a_miss_when_its_distance_is_reported(monkeypatch):
    """Opus P3-4: re-measuring a reported miss exactly (where the exact
    scan's clamp could call it a hit) only sets its distance."""
    def build():
        _build("offset-box")
        return _one_role()

    with _session():
        expected = auto_cut_occ_geometry(build(), planes=("x0",)).report["planes"]["x0"]
    real = step_prepare._CertifiedOccMirrorTarget._exact

    def hit_anyway(self, point):
        matched, nearest, hits = real(self, point)
        return 0.0, nearest, hits or [self.candidates[0]]

    monkeypatch.setattr(step_prepare._CertifiedOccMirrorTarget, "_exact", hit_anyway)
    with _session():
        got = auto_cut_occ_geometry(build(), planes=("x0",)).report["planes"]["x0"]
    assert got["points_off_model"] == expected["points_off_model"] > 0
    assert got["accepted"] is False


def test_a_strip_narrower_than_the_tolerance_is_cut_as_a_solid_like_face_by_face():
    """Opus P2-1 / Neon P1-5: a source strip narrower than the tolerance on a
    box top: the solid cut gives the face cut's groups (no refusal)."""
    def build(width):
        occ = gmsh.model.occ
        box = occ.addBox(-50, -30, -20, 100, 60, 40)
        r = occ.addRectangle(-50, 5, 20, 100, width)
        r2 = occ.addRectangle(-50, -5 - width, 20, 100, width)
        occ.fragment([(3, box)], [(2, r), (2, r2)])
        occ.synchronize()
        surfaces = [t for _d, t in gmsh.model.getEntities(2)]
        strips = [t for t in surfaces
                  if gmsh.model.getBoundingBox(2, t)[4] - gmsh.model.getBoundingBox(2, t)[1] < 2 * width + 1e-6
                  and abs(gmsh.model.getBoundingBox(2, t)[2] - 20) < 1e-6]
        return [OccSurfaceGroup("strip", OccSurfaceSelector(strips), OccSurfaceRole("source")),
                OccSurfaceGroup("rest", OccSurfaceSelector([t for t in surfaces if t not in strips]),
                                OccSurfaceRole("rigid"))]

    for width in (0.03, 0.06):
        results = {}
        for solid in (False, True):
            with _session():
                result = auto_cut_occ_geometry(build(width), planes=("x0", "y0"), solid_cut=solid)
                results[solid] = (result.planes, _group_geometry(result), result.report["cut"]["route"])
        assert results[True][:2] == results[False][:2]
        assert results[True][2] == "solids"


def test_a_thin_walled_solid_keeps_no_cap_on_the_plane():
    """Neon P1-5: a plate thinner than the tolerance -- its cap lies within
    tolerance of both walls -- is still left open on the plane."""
    def build():
        gmsh.model.occ.addBox(-250.0, -150.0, 0.0, 500.0, 300.0, 0.2)
        gmsh.model.occ.synchronize()
        return _one_role()

    results = {}
    for solid in (False, True):
        with _session():
            result = auto_cut_occ_geometry(build(), planes=("x0",), solid_cut=solid,
                                           feature_check=False)
            results[solid] = (result.planes, _group_geometry(result), _topology())
    assert results[True] == results[False]
    assert results[True][0] == ("x0",)


def test_the_tessellation_never_proves_a_hit_on_a_curved_face():
    """Opus P1-1 at the primitive: over the hidden ridge the tessellation
    reads a gap of ~0 while OCC measures ~3 tolerances; no curved triangle
    may certify a hit (only a candidate for OCC)."""
    with _session():
        _ridge_sheet(0.3)
        surfaces = [t for _d, t in gmsh.model.getEntities(2)]
        box = gmsh.model.getBoundingBox(-1, -1)
        diagonal = float(np.linalg.norm(np.subtract(box[3:], box[:3])))
        tolerance = 5e-4 * diagonal
        index, _trims = occ_face_index.build_face_index(
            surfaces, diagonal=diagonal, tolerance=tolerance, nudge_rel=1e-3, trims={})
        ridge = next(t for t in surfaces if gmsh.model.getType(2, t) != "Plane")
        point = np.array([[50.0 * 5.5 / 7.0, 0.0, 0.0]])
        near = index.near(point, tolerance)
        rows = near.face == ridge
        assert np.any(rows)
        assert np.all(np.isinf(near.upper[rows]))
        closest, _uv = gmsh.model.getClosestPoint(2, ridge, point.ravel().tolist())
        assert float(np.linalg.norm(np.asarray(closest) - point[0])) > tolerance


def test_the_face_cut_retry_judges_its_own_children(monkeypatch):
    """The periodic-seam retry's children reuse the tags of the attempt-0
    children it removed; a claim judge carried over from attempt 0 answered
    from those faces' samples, disproved true claims and refused the cut
    (the 34d775978b42 family, two roles). Force the retry on a cylinder."""
    real = step_prepare._uncovered_cut_parents
    calls = []

    def lose_one_on_the_first_attempt(mapping, copies, planes, **kwargs):
        calls.append(1)
        got = real(mapping, copies, planes, **kwargs)
        return got or (sorted(mapping)[:1] if len(calls) == 1 else got)

    monkeypatch.setattr(step_prepare, "_uncovered_cut_parents", lose_one_on_the_first_attempt)
    with _session():
        gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, 30, 15)
        gmsh.model.occ.synchronize()
        surfaces = [t for _d, t in gmsh.model.getEntities(2)]
        low = min(surfaces, key=lambda t: gmsh.model.getBoundingBox(2, t)[2])
        groups = [
            OccSurfaceGroup("src", OccSurfaceSelector([low]), OccSurfaceRole("source")),
            OccSurfaceGroup("rest", OccSurfaceSelector([t for t in surfaces if t != low]),
                            OccSurfaceRole("rigid")),
        ]
        result = auto_cut_occ_geometry(groups, planes=("x0",), solid_cut=False)
        assert len(calls) == 2
        assert result.planes == ("x0",)
        assert result.report["cut"]["route"] == "faces"


# -- round 3 (reviews of ce1dadd) ----------------------------------------------


def _bent_sheet(radius, hole=True):
    """Opus bendhole.py: an open sheet with a quarter-round bend, and a
    one-sided 1.2 mm hole drilled through the bend at x = +100."""
    occ = gmsh.model.occ
    p0 = occ.addPoint(-300, -400, 0)
    p1 = occ.addPoint(-300, -radius, 0)
    centre = occ.addPoint(-300, -radius, radius)
    p2 = occ.addPoint(-300, 0, radius)
    p3 = occ.addPoint(-300, 0, 300)
    curves = [occ.addLine(p0, p1), occ.addCircleArc(p1, centre, p2), occ.addLine(p2, p3)]
    occ.extrude([(1, c) for c in curves], 600, 0, 0)
    occ.synchronize()
    if hole:
        angle = math.pi / 4
        cy, cz = -radius + radius * math.sin(angle), radius - radius * math.cos(angle)
        ny, nz = math.sin(angle), -math.cos(angle)
        drill = occ.addCylinder(100, cy - 2 * ny, cz - 2 * nz, 0, 4 * ny, 4 * nz, 0.6)
        occ.cut([(2, t) for _d, t in gmsh.model.getEntities(2)], [(3, drill)])
        occ.synchronize()
    surfaces = [t for _d, t in gmsh.model.getEntities(2)]
    return [OccSurfaceGroup("all", OccSurfaceSelector(surfaces), OccSurfaceRole("wall"))]


@pytest.mark.parametrize("radius", [2.0, 20.0])
def test_a_one_sided_hole_on_a_tight_bend_is_rejected(radius):
    """A smooth bend tighter than ~7.6 tolerances turns by more than 30
    degrees over the crease probe, but it is no crease: the hole's edges
    mirror onto no edge (Opus P2-R2, Neon P1-1). The bare sheet passes."""
    with _session():
        report = auto_cut_occ_geometry(_bent_sheet(radius), planes=("x0",)).report
    assert report["planes"]["x0"]["accepted"] is False
    assert report["mirror_check_points"]["x0"]["edge_mismatches"] > 0
    with _session():
        report = auto_cut_occ_geometry(_bent_sheet(radius, hole=False), planes=("x0",)).report
    assert report["planes"]["x0"]["accepted"] is True


def test_a_wrong_facing_hit_is_not_excused_by_another_faces_carrier():
    """Neon P1-2: a low one-sided boss's walls mirror onto the plate,
    facing the wrong way. A rib wall of the same role on the same carrier
    plane, 0.1 mm beyond the mirrored samples, must not excuse them: its
    trimmed face is not within tolerance."""
    with _session():
        occ = gmsh.model.occ
        plate = occ.addBox(-60, -30, -2, 120, 60, 2)
        boss = occ.addBox(45, 0, 0, 5, 5, 0.05)
        ribs = [occ.addBox(50, 5.1, 0, 2, 10, 10), occ.addBox(-52, 5.1, 0, 2, 10, 10)]
        occ.fuse([(3, plate)], [(3, boss)] + [(3, r) for r in ribs])
        occ.synchronize()
        surfaces = [t for _d, t in gmsh.model.getEntities(2)]
        groups = [OccSurfaceGroup("all", OccSurfaceSelector(surfaces), OccSurfaceRole("wall"))]
        report = auto_cut_occ_geometry(groups, planes=("x0",)).report
    stats = report["mirror_check_points"]["x0"]
    assert report["planes"]["x0"]["accepted"] is False
    # Every sample the tessellation left to the exact scan is a boss wall's,
    # and every one of them faces the wrong way.
    assert stats["exact"] > 0
    assert stats["misaligned"] == stats["exact"]


def test_the_solid_cut_refuses_faces_in_a_cut_plane_or_shared_by_two_solids(monkeypatch):
    """Neon P1-3: a face in a cut plane must keep the face cut even where
    OCC pads its box beyond the tolerance, and so must a face two solids
    share (copying the solids would split it in two)."""
    tolerance = 0.01
    with _session():
        occ = gmsh.model.occ
        occ.addBox(-10, -5, 0, 10, 10, 5)
        occ.addBox(0, -5, 0, 10, 10, 5)
        occ.synchronize()
        surfaces = [t for _d, t in gmsh.model.getEntities(2)]
        real = gmsh.model.getBoundingBox

        def padded(dim, tag):
            box = list(real(dim, tag))
            return tuple([v - 0.1 for v in box[:3]] + [v + 0.1 for v in box[3:]])

        monkeypatch.setattr(gmsh.model, "getBoundingBox", padded)
        assert not step_prepare._solid_cut_applies(
            gmsh.model.getEntities(3), surfaces, ("x0",), tolerance)
        monkeypatch.setattr(gmsh.model, "getBoundingBox", real)
    with _session():
        occ = gmsh.model.occ
        low = occ.addBox(-10, -5, 0, 20, 10, 5)
        high = occ.addBox(-10, -5, 5, 20, 10, 5)
        occ.fragment([(3, low)], [(3, high)])
        occ.synchronize()
        surfaces = [t for _d, t in gmsh.model.getEntities(2)]
        assert any(len(gmsh.model.getAdjacencies(2, s)[0]) == 2 for s in surfaces)
        assert not step_prepare._solid_cut_applies(
            gmsh.model.getEntities(3), surfaces, ("x0",), tolerance)


def test_a_plane_hit_needs_the_perpendicular_foot_itself():
    """Neon P2-1: a point 0.25 above a plane triangle whose foot is 1e-4
    outside it is no proven hit; comparing distances let the lateral offset
    through as its square (2e-8)."""
    triangle = np.array([[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]]])
    mesh = occ_face_index.FaceMesh(
        1, triangle, np.array([True]), np.zeros(1), np.zeros(1),
        np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]), None, True)
    index = occ_face_index.FaceIndex({1: mesh})
    near = index.near(np.array([[50.0, -1.0e-4, 0.25], [50.0, 1.0e-4, 0.25]]), 0.3)
    upper = {int(p): float(u) for p, u in zip(near.point, near.upper)}
    assert math.isinf(upper[0])
    assert upper[1] == pytest.approx(0.25)


def test_a_non_finite_mirror_sample_is_never_a_certified_miss():
    """Neon P2-2: a sample the index cannot place is left to the exact scan."""
    with _session():
        gmsh.model.occ.addBox(-10, -5, 0, 20, 10, 5)
        gmsh.model.occ.synchronize()
        surfaces = [t for _d, t in gmsh.model.getEntities(2)]
        bboxes = {t: tuple(gmsh.model.getBoundingBox(2, t)) for t in surfaces}
        samples = {t: step_prepare.sample_occ_surface_points(t, grid=5) for t in surfaces}
        samples[surfaces[0]] = np.vstack([samples[surfaces[0]], [np.nan, 0.0, 1.0]])
        index, _trims = occ_face_index.build_face_index(
            surfaces, diagonal=25.0, tolerance=0.0125, nudge_rel=1e-3, trims={})
        target = step_prepare._CertifiedOccMirrorTarget(
            surfaces, bboxes, 0.0125, None, {t: "wall" for t in surfaces},
            samples=samples, axis=0, face_index=index)
        target.match(surfaces[0], None)
    assert target.stats["certified_misses"] == 0
    assert target.stats["exact"] >= 1


def test_trim_cells_marks_wide_margins_in_one_pass():
    """Neon P2-3: a p-curve margin as wide as the face on a 1024 x 1024 grid
    marks every cell it reaches without one pass per cell of reach."""
    import time
    from types import SimpleNamespace

    loop = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
    u = np.linspace(0.0, 1.0, 1025)

    def classify_grid(uu, vv):
        shape = (len(uu), len(vv))
        return np.ones(shape, dtype=bool), np.ones(shape, dtype=bool)

    trim = SimpleNamespace(ok=True, margin=0.6, loops=[loop], classify_grid=classify_grid)
    start = time.perf_counter()
    inside, keep = occ_face_index._trim_cells(trim, u, u)
    assert time.perf_counter() - start < 5.0
    assert not inside.any()
    assert keep.all()


@pytest.mark.parametrize("fraction", [1.0, 4.0])
def test_a_coarse_curved_tessellation_still_confirms_hits_with_occ(monkeypatch, fraction):
    """The curved tessellation only bounds misses: a curved face its bound
    does not clear is a candidate OCC confirms, however coarse the cells, so
    a coarser tessellation moves no hit to the exact scan."""
    def build():
        gmsh.model.occ.addTorus(0, 0, 0, 30, 8)
        gmsh.model.occ.addCylinder(0, 0, -40, 0, 0, 20, 12)
        gmsh.model.occ.synchronize()
        surfaces = [t for _d, t in gmsh.model.getEntities(2)]
        return [OccSurfaceGroup("all", OccSurfaceSelector(surfaces), OccSurfaceRole("wall"))]

    monkeypatch.setattr(occ_face_index, "DELTA_TARGET_FRACTION", fraction)
    reports = {}
    for fast in (False, True):
        with _session():
            result = auto_cut_occ_geometry(build(), planes=("x0", "y0"), fast_check=fast)
            reports[fast] = result.report
    for plane in ("x0", "y0"):
        for field in EXACT_FIELDS:
            assert reports[True]["planes"][plane][field] == reports[False]["planes"][plane][field]
        stats = reports[True]["mirror_check_points"][plane]
        assert stats["exact"] == 0
        assert stats["confirmed_hits"] > 0


def test_the_batched_claim_evidence_answers_as_the_point_loop(monkeypatch):
    """``_claim_evidence`` settles most claims with one projection and one
    inside count; every (parent, child) pair of a real cut must get the
    answer the point-by-point loop gives."""
    with _session():
        gmsh.model.occ.addTorus(0, 0, 0, 30, 8)
        gmsh.model.occ.addBox(-20, -20, 10, 40, 40, 10)
        gmsh.model.occ.synchronize()
        surfaces = [t for _d, t in gmsh.model.getEntities(2)]
        copies = {s: gmsh.model.occ.copy([(2, s)])[0][1] for s in surfaces}
        box = gmsh.model.occ.addBox(0, -100, -100, 200, 200, 200)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.intersect([(2, s) for s in surfaces], [(3, box)])
        gmsh.model.occ.synchronize()
        children = [t for _d, t in gmsh.model.getEntities(2) if t not in copies.values()]
        tolerance = 0.05
        batched = {(p, c): step_prepare._claim_evidence(copy, c, tolerance=tolerance, grid=7)
                   for p, copy in copies.items() for c in children}
        real = gmsh.model.getClosestPoint

        def one_at_a_time(dim, tag, coords):
            if len(coords) > 3:
                raise RuntimeError("batch refused")
            return real(dim, tag, coords)

        monkeypatch.setattr(gmsh.model, "getClosestPoint", one_at_a_time)
        looped = {(p, c): step_prepare._claim_evidence(copy, c, tolerance=tolerance, grid=7)
                  for p, copy in copies.items() for c in children}
    assert batched == looped
    assert set(batched.values()) >= {"on", "off"}


def _open_shell():
    """An open shell (faces, no solid): a box with a round bore, its lid
    removed, plus a fin wholly on the kept side and one wholly beyond."""
    occ = gmsh.model.occ
    body = occ.addBox(-30, -20, 0, 60, 40, 25)
    bore = occ.addCylinder(0, 0, -1, 0, 0, 27, 8)
    occ.cut([(3, body)], [(3, bore)])
    occ.synchronize()
    volumes = gmsh.model.getEntities(3)
    top = max((t for _d, t in gmsh.model.getEntities(2)), key=lambda t: gmsh.model.getBoundingBox(2, t)[5]
              + gmsh.model.getBoundingBox(2, t)[2])
    occ.remove(volumes, recursive=False)
    occ.remove([(2, top)], recursive=True)
    occ.addRectangle(5, 30, 0, 10, 5)
    occ.addRectangle(-15, 30, 0, 10, 5)
    occ.synchronize()
    surfaces = [t for _d, t in gmsh.model.getEntities(2)]
    return [OccSurfaceGroup("all", OccSurfaceSelector(surfaces), OccSurfaceRole("wall"))]


def _free_edges_off_planes(planes):
    gmsh.option.setNumber("Mesh.MeshSizeMax", 4.0)
    gmsh.model.mesh.generate(2)
    nodes, coords, _ = gmsh.model.mesh.getNodes()
    xyz = dict(zip(nodes, np.asarray(coords).reshape(-1, 3)))
    tri = np.asarray(gmsh.model.mesh.getElementsByType(2)[1]).reshape(-1, 3)
    edges = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    axes = [step_prepare.SYMMETRY_AXIS_FOR_PLANE[p] for p in planes]
    off = sum(1 for a, b in edges[counts == 1]
              if not any(abs(xyz[a][k]) < 1e-7 and abs(xyz[b][k]) < 1e-7 for k in axes))
    return off, int(counts.max())


@pytest.mark.parametrize("force_retry", [False, True])
def test_an_open_shell_cut_on_its_crossing_faces_keeps_the_same_faces(monkeypatch, force_retry):
    """The crossing-faces cut gives OCC only the faces that cross the plane:
    the same kept faces, curves and points as cutting every face, and the
    mesh is as watertight. A retry (periodic seams) cuts every face again."""
    calls: list[int] = []
    if force_retry:
        real = step_prepare._uncovered_cut_parents

        def lose_one_first(mapping, copies, planes, **kwargs):
            calls.append(1)
            got = real(mapping, copies, planes, **kwargs)
            if got or len(calls) > 1:
                return got
            # The same face either way: the bottom (it crosses the plane).
            return [min(copies, key=lambda p: gmsh.model.getBoundingBox(2, copies[p])[5])]

        monkeypatch.setattr(step_prepare, "_uncovered_cut_parents", lose_one_first)
    results = {}
    for crossing in (False, True):
        calls.clear()
        with _session():
            result = auto_cut_occ_geometry(_open_shell(), planes=("x0",), crossing_faces_only=crossing)
            results[crossing] = (result.planes, _group_geometry(result), _topology(),
                                 _free_edges_off_planes(result.planes), result.report["cut"]["route"])
    assert results[True][:4] == results[False][:4]
    assert results[True][0] == ("x0",)
    assert results[True][4] == ("faces" if force_retry else "faces (crossing faces)")
    assert results[False][4] == "faces"
