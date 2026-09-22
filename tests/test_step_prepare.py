from __future__ import annotations

from contextlib import contextmanager

import gmsh
import numpy as np
import pytest

import hornlab_mesher.step_prepare as step_prepare
from hornlab_mesher import (
    DEFAULT_AUTO_CUT_TOLERANCE_REL,
    OccSurfaceGroup,
    OccSurfaceRole,
    OccSurfaceSelector,
    auto_cut_occ_geometry,
    millimetres_to_step_units,
    snap_symmetry_plane_vertices,
)


@contextmanager
def _gmsh_session():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("step-prepare-test")
    try:
        yield
    finally:
        gmsh.finalize()


def _box_surfaces(x0, y0, z0, dx, dy, dz):
    volume = gmsh.model.occ.addBox(x0, y0, z0, dx, dy, dz)
    gmsh.model.occ.synchronize()
    return [
        tag
        for dim, tag in gmsh.model.getBoundary(
            [(3, volume)], combined=False, oriented=False
        )
        if dim == 2
    ]


def _groups_for_roles(surfaces, roles):
    grouped = {}
    for surface in surfaces:
        grouped.setdefault(roles[surface], []).append(surface)
    return [
        OccSurfaceGroup(
            f"group-{index}",
            OccSurfaceSelector(group_surfaces),
            OccSurfaceRole(role),
        )
        for index, (role, group_surfaces) in enumerate(grouped.items())
    ]


def test_auto_cut_uses_opaque_roles_and_remaps_groups():
    with _gmsh_session():
        surfaces = _box_surfaces(-1.0, -0.5, 0.0, 2.0, 2.0, 2.0)
        selected = next(
            surface
            for surface in surfaces
            if abs(gmsh.model.occ.getCenterOfMass(2, surface)[1] + 0.5) < 1.0e-9
        )
        groups = [
            OccSurfaceGroup(
                "painted",
                OccSurfaceSelector([selected]),
                OccSurfaceRole("opaque-caller-role"),
            ),
            OccSurfaceGroup(
                "everything-else",
                OccSurfaceSelector(set(surfaces) - {selected}),
                OccSurfaceRole("another-opaque-role"),
            ),
        ]

        result = auto_cut_occ_geometry(
            groups, grid=5, tolerance_rel=DEFAULT_AUTO_CUT_TOLERANCE_REL
        )

        assert result.planes == ("x0",)
        assert result.group("painted").role.name == "opaque-caller-role"
        assert result.group("painted").selector.surface_tags
        assert result.report["planes"]["x0"]["role_mismatches"] == 0
        assert len(gmsh.model.getEntities(2)) == 5


def test_auto_cut_rejects_geometry_that_does_not_mirror():
    with _gmsh_session():
        surfaces = _box_surfaces(-1.0, -0.5, 0.0, 3.0, 2.0, 2.0)
        result = auto_cut_occ_geometry(
            [
                OccSurfaceGroup(
                    "all",
                    OccSurfaceSelector(surfaces),
                    OccSurfaceRole("opaque"),
                )
            ],
            grid=5,
        )

        assert result.planes == ()
        assert result.parent_to_children == {}
        assert result.report["planes"]["x0"]["points_off_model"] > 0
        assert len(gmsh.model.getEntities(2)) == len(surfaces)


def test_param_inside_fails_closed_when_occ_cannot_judge_trim_membership(monkeypatch):
    monkeypatch.setattr(
        gmsh.model,
        "isInside",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("OCC failed")),
    )

    assert step_prepare._param_inside(7, 0.25, 0.75) is False


def test_auto_cut_does_not_accept_samples_when_trim_membership_fails(monkeypatch):
    with _gmsh_session():
        surfaces = _box_surfaces(-1.0, -1.0, -1.0, 2.0, 2.0, 2.0)
        monkeypatch.setattr(
            gmsh.model,
            "isInside",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("OCC failed")),
        )

        result = auto_cut_occ_geometry(
            [
                OccSurfaceGroup(
                    "all", OccSurfaceSelector(surfaces), OccSurfaceRole("opaque")
                )
            ],
            grid=5,
            planes=("x0", "y0"),
        )

        assert result.planes == ()
        assert result.report["planes"]["x0"]["points_off_model"] > 0
        assert result.report["planes"]["y0"]["points_off_model"] > 0


def test_auto_cut_requires_roles_to_mirror_even_when_geometry_does():
    with _gmsh_session():
        right = _box_surfaces(0.2, -0.5, 0.0, 0.8, 1.0, 1.0)
        left = _box_surfaces(-1.0, -0.5, 0.0, 0.8, 1.0, 1.0)
        surfaces = right + left
        roles = {surface: "ordinary" for surface in surfaces}
        right_front = next(
            surface
            for surface in right
            if abs(gmsh.model.occ.getCenterOfMass(2, surface)[1] + 0.5) < 1.0e-9
        )
        roles[right_front] = "special"

        result = auto_cut_occ_geometry(_groups_for_roles(surfaces, roles), grid=5)

        x_verdict = result.report["planes"]["x0"]
        assert x_verdict["accepted"] is False
        assert x_verdict["points_off_model"] == 0
        assert x_verdict["role_mismatches"] > 0
        assert any(
            sample["kind"] == "role_mismatch"
            for sample in x_verdict["failure_samples"]
        )


def test_auto_cut_does_not_treat_untrimmed_hole_as_symmetric():
    with _gmsh_session():
        box = gmsh.model.occ.addBox(-1.0, -0.5, 0.0, 2.0, 1.0, 1.0)
        bore = gmsh.model.occ.addCylinder(0.5, 0.0, -0.1, 0.0, 0.0, 1.2, 0.2)
        gmsh.model.occ.cut([(3, box)], [(3, bore)])
        gmsh.model.occ.synchronize()
        surfaces = [tag for dim, tag in gmsh.model.getEntities(2) if dim == 2]

        result = auto_cut_occ_geometry(
            [
                OccSurfaceGroup(
                    "all",
                    OccSurfaceSelector(surfaces),
                    OccSurfaceRole("opaque"),
                )
            ],
            grid=9,
        )

        assert result.planes == ("y0",)
        assert result.report["planes"]["x0"]["points_off_model"] > 0


def test_auto_cut_only_considers_the_planes_the_caller_supports():
    with _gmsh_session():
        # Symmetric about all three coordinate planes, so the unrestricted
        # cutter would halve the model on a plane the caller cannot mirror.
        surfaces = _box_surfaces(-1.0, -1.0, -1.0, 2.0, 2.0, 2.0)
        groups = [
            OccSurfaceGroup(
                "all", OccSurfaceSelector(surfaces), OccSurfaceRole("opaque")
            )
        ]

        result = auto_cut_occ_geometry(groups, grid=5, planes=("x0", "y0"))

        assert result.planes == ("x0", "y0")
        assert result.report["candidate_planes"] == ["x0", "y0"]
        assert set(result.report["planes"]) == {"x0", "y0"}
        assert result.report["cut"]["planes"] == ["x0", "y0"]


def test_auto_cut_refuses_an_unknown_plane_name():
    with _gmsh_session():
        surfaces = _box_surfaces(-1.0, -1.0, -1.0, 2.0, 2.0, 2.0)
        groups = [
            OccSurfaceGroup(
                "all", OccSurfaceSelector(surfaces), OccSurfaceRole("opaque")
            )
        ]
        with pytest.raises(ValueError, match="unknown symmetry planes"):
            auto_cut_occ_geometry(groups, grid=5, planes=("x0", "xy"))


def test_auto_cut_rejects_overlapping_group_selectors():
    with _gmsh_session():
        surfaces = _box_surfaces(-1.0, -1.0, -1.0, 2.0, 2.0, 2.0)
        duplicate = surfaces[0]
        groups = [
            OccSurfaceGroup(
                "a", OccSurfaceSelector([duplicate]), OccSurfaceRole("one")
            ),
            OccSurfaceGroup(
                "b", OccSurfaceSelector([duplicate]), OccSurfaceRole("two")
            ),
        ]
        with pytest.raises(ValueError, match="more than one group"):
            auto_cut_occ_geometry(groups)


def test_auto_cut_never_hands_a_removed_face_the_piece_that_reused_its_tag():
    """A wholly removed face maps to nothing, even when its tag is reused.

    Gmsh frees the tags of the removed objects before numbering the new
    pieces, so a face split into two gets the lowest free tags -- here the
    removed half's. Its out_map entry then names the new piece as if the
    removed face had survived unchanged. The cylinder wall's seam lies on +x,
    so the cut splits it in two; the -x box is numbered last and dropped
    whole, and its source face held the tag the first new piece reuses.
    """
    with _gmsh_session():
        cylinder = gmsh.model.occ.addCylinder(0.0, 0.0, -1.0, 0.0, 0.0, 2.0, 1.0)
        gmsh.model.occ.synchronize()
        wall = [
            tag
            for dim, tag in gmsh.model.getBoundary(
                [(3, cylinder)],
                combined=False,
                oriented=False,
            )
            if dim == 2
        ]
        right = _box_surfaces(2.0, -0.5, -0.5, 1.0, 1.0, 1.0)
        left = _box_surfaces(-3.0, -0.5, -0.5, 1.0, 1.0, 1.0)

        def facing(surfaces, x):
            return next(
                surface
                for surface in surfaces
                if abs(gmsh.model.occ.getCenterOfMass(2, surface)[0] - x) < 1.0e-9
            )

        # The two inner faces are one source, mirror images across x0. The
        # wall's two new pieces take the two lowest freed tags, and the left
        # source face holds one of them.
        source = [facing(right, 2.0), facing(left, -2.0)]
        survivors = max(wall + right)
        assert source[1] in (survivors + 1, survivors + 2) and min(left) > survivors
        groups = [
            OccSurfaceGroup(
                "rigid",
                OccSurfaceSelector(set(wall + right + left) - set(source)),
                OccSurfaceRole("rigid"),
            ),
            OccSurfaceGroup("mf", OccSurfaceSelector(source), OccSurfaceRole("mf")),
        ]

        result = auto_cut_occ_geometry(groups, grid=5, planes=("x0",))

        assert result.planes == ("x0",)
        assert result.report["cut"]["surfaces_split"] >= 1
        for surface in left:
            assert result.parent_to_children[surface] == [], surface
        assert list(result.group("mf").selector.surface_tags) == [source[0]]
        rigid = set(result.group("rigid").selector.surface_tags)
        assert not rigid & {source[0]}
        remaining = {tag for _dim, tag in gmsh.model.getEntities(2)}
        assert rigid | {source[0]} == remaining


def test_a_quarter_cut_drops_the_reused_tag_claims_of_both_removed_faces():
    """Both planes are one intersection; the wall seam is turned so it splits."""
    with _gmsh_session():
        cylinder = gmsh.model.occ.addCylinder(0.0, 0.0, -1.0, 0.0, 0.0, 2.0, 1.0)
        gmsh.model.occ.rotate([(3, cylinder)], 0, 0, 0, 0, 0, 1, np.pi / 4)
        gmsh.model.occ.synchronize()
        wall = [
            tag
            for dim, tag in gmsh.model.getBoundary(
                [(3, cylinder)], combined=False, oriented=False
            )
            if dim == 2
        ]
        boxes = [
            _box_surfaces(x, y, -0.5, 1.0, 1.0, 1.0)
            for x, y in ((2.0, 2.0), (-3.0, 2.0), (2.0, -3.0), (-3.0, -3.0))
        ]
        everything = wall + [surface for box in boxes for surface in box]
        groups = [
            OccSurfaceGroup("all", OccSurfaceSelector(everything), OccSurfaceRole("rigid"))
        ]

        result = auto_cut_occ_geometry(groups, grid=5, planes=("x0", "y0"))

        assert result.planes == ("x0", "y0")
        assert result.report["cut"]["surfaces_split"] >= 1
        for box in boxes[1:]:
            for surface in box:
                assert result.parent_to_children[surface] == [], surface
        remaining = {tag for _dim, tag in gmsh.model.getEntities(2)}
        assert set(result.group("all").selector.surface_tags) == remaining


def test_auto_cut_keeps_the_genuine_claims_of_overlapping_surfaces():
    """Overlapping inputs really share a child; only the removed one loses it."""
    with _gmsh_session():
        outer = gmsh.model.occ.addRectangle(-2.0, -1.0, 0.0, 4.0, 2.0)
        right = gmsh.model.occ.addRectangle(0.5, -0.5, 0.0, 1.0, 1.0)
        left = gmsh.model.occ.addRectangle(-1.5, -0.5, 0.0, 1.0, 1.0)
        gmsh.model.occ.synchronize()
        groups = [
            OccSurfaceGroup("outer", OccSurfaceSelector([outer]), OccSurfaceRole("rigid")),
            OccSurfaceGroup(
                "inner", OccSurfaceSelector([right, left]), OccSurfaceRole("rigid")
            ),
        ]

        result = auto_cut_occ_geometry(groups, grid=5, planes=("x0",))

        assert result.planes == ("x0",)
        assert result.parent_to_children[right] == [right]
        assert result.parent_to_children[left] == []
        assert list(result.group("inner").selector.surface_tags) == [right]
        assert right in result.group("outer").selector.surface_tags


@pytest.mark.parametrize("copies", [2, 3])
def test_auto_cut_accepts_coincident_surfaces_that_straddle_the_plane(copies):
    with _gmsh_session():
        tags = [
            gmsh.model.occ.addRectangle(-1.0, -1.0, 0.0, 2.0, 2.0) for _ in range(copies)
        ]
        gmsh.model.occ.synchronize()
        groups = [OccSurfaceGroup("all", OccSurfaceSelector(tags), OccSurfaceRole("rigid"))]

        result = auto_cut_occ_geometry(groups, grid=5, planes=("x0",))

        assert result.planes == ("x0",)
        children = [result.parent_to_children[tag] for tag in tags]
        assert all(len(kids) == 1 for kids in children)
        assert len(result.group("all").selector.surface_tags) == 1


def test_auto_cut_refuses_a_surviving_surface_that_no_parent_claims(monkeypatch):
    with _gmsh_session():
        surfaces = _box_surfaces(-1.0, -1.0, -1.0, 2.0, 2.0, 2.0)
        intersect = gmsh.model.occ.intersect

        def forgetful(*args, **kwargs):
            out, out_map = intersect(*args, **kwargs)
            return out, [list(children) for children in out_map[:-2]] + [[], []]

        monkeypatch.setattr(gmsh.model.occ, "intersect", forgetful)
        groups = [
            OccSurfaceGroup("all", OccSurfaceSelector(surfaces), OccSurfaceRole("rigid"))
        ]
        with pytest.raises(RuntimeError, match="with no parent"):
            auto_cut_occ_geometry(groups, grid=5, planes=("x0",))


def _cut_groups(groups, planes=("x0",), grid=5):
    result = auto_cut_occ_geometry(groups, grid=grid, planes=planes)
    return result, {group.name: list(group.selector.surface_tags) for group in result.groups}


@pytest.mark.parametrize("scale", [1.0, 1.0e3, 1.0e6])
def test_auto_cut_keeps_a_strip_that_survives_just_past_the_plane(scale):
    """A partition seam just past x0 leaves face 1 a real, kept strip."""
    with _gmsh_session():
        eps = 1.0e-6
        tags = [
            gmsh.model.occ.addRectangle(-scale, -scale, 0, (1 + eps) * scale, 2 * scale),
            gmsh.model.occ.addRectangle(eps * scale, -scale, 0, (1 - eps) * scale, 2 * scale),
        ]
        gmsh.model.occ.synchronize()

        result, groups = _cut_groups(
            [OccSurfaceGroup("all", OccSurfaceSelector(tags), OccSurfaceRole("rigid"))]
        )

        assert result.planes == ("x0",)
        assert all(len(result.parent_to_children[tag]) == 1 for tag in tags)
        assert len(groups["all"]) == 2


@pytest.mark.parametrize("scale", [1.0, 1.0e3, 1.0e6])
def test_auto_cut_keeps_mirrored_inclined_sheets_that_meet_past_the_plane(scale):
    with _gmsh_session():
        occ = gmsh.model.occ
        tags = []
        for sign in (1, -1):
            points = [
                occ.addPoint(sign * x * scale, y * scale, x * scale)
                for x, y in ((-1, -1), (1.0e-6, -1), (1.0e-6, 1), (-1, 1))
            ]
            lines = [occ.addLine(points[i], points[(i + 1) % 4]) for i in range(4)]
            tags.append(occ.addPlaneSurface([occ.addCurveLoop(lines)]))
        occ.synchronize()

        result, groups = _cut_groups(
            [OccSurfaceGroup("all", OccSurfaceSelector(tags), OccSurfaceRole("rigid"))]
        )

        assert result.planes == ("x0",)
        assert groups["all"]


def _cylinder_and_sources(scale, make_source):
    occ = gmsh.model.occ
    occ.addCylinder(0, 0, -scale, 0, 0, 2 * scale, scale)
    occ.synchronize()
    rigid = [tag for _dim, tag in gmsh.model.getEntities(2)]
    sources = [make_source(sign) for sign in (1, -1)]
    occ.synchronize()
    return rigid, sources


def _rigid_and_source_groups(rigid, sources):
    return [
        OccSurfaceGroup("rigid", OccSurfaceSelector(rigid), OccSurfaceRole("rigid")),
        OccSurfaceGroup("source", OccSurfaceSelector(sources), OccSurfaceRole("source")),
    ]


@pytest.mark.parametrize("width", [1.0, 100.0])
def test_a_removed_bspline_source_whose_box_reaches_the_kept_side_gets_nothing(width):
    """The negative patch lies at x <= -2, but its OCC box reaches x = +1."""
    with _gmsh_session():
        occ = gmsh.model.occ

        def patch(sign):
            # x(u) = -sign * (-5 + 12u - 12u^2): the sign = -1 patch sits at x <= -2.
            points = [
                occ.addPoint(-sign * x, y, z)
                for y in (-width / 2, width / 2)
                for x, z in ((-5, -1), (1, 0), (-5, 1))
            ]
            return occ.addBSplineSurface(points, 3, degreeU=2, degreeV=1)

        rigid, sources = _cylinder_and_sources(1.0, patch)
        assert gmsh.model.getBoundingBox(2, sources[1])[3] > 0.5

        result, groups = _cut_groups(_rigid_and_source_groups(rigid, sources), grid=7)

        assert result.planes == ("x0",)
        assert result.parent_to_children[sources[1]] == []
        assert groups["source"] == [sources[0]]
        assert not set(groups["rigid"]) & set(groups["source"])


@pytest.mark.parametrize("scale", [1.0, 0.1, 0.01, 0.001])
def test_a_removed_source_touching_the_plane_gets_nothing_at_any_scale(scale):
    with _gmsh_session():

        def rectangle(sign):
            x = 0.0 if sign > 0 else -3.0
            return gmsh.model.occ.addRectangle(x * scale, -0.5 * scale, 2 * scale, 3 * scale, scale)

        rigid, sources = _cylinder_and_sources(scale, rectangle)

        result, groups = _cut_groups(_rigid_and_source_groups(rigid, sources))

        assert result.planes == ("x0",)
        assert result.parent_to_children[sources[1]] == []
        assert groups["source"] == [sources[0]]


def _inject_claim(monkeypatch, parent_index, child_of_index):
    """Make out_map also list parent ``parent_index`` as a parent of the
    first child of input ``child_of_index`` -- the shape of gmsh's stale claim."""
    intersect = gmsh.model.occ.intersect

    def stale(*args, **kwargs):
        out, out_map = intersect(*args, **kwargs)
        out_map = [list(children) for children in out_map]
        out_map[parent_index] = out_map[parent_index] + [out_map[child_of_index][0]]
        return out, out_map

    monkeypatch.setattr(gmsh.model.occ, "intersect", stale)


def test_a_claim_on_a_piece_off_the_parents_surface_is_dropped(monkeypatch):
    with _gmsh_session():
        low = gmsh.model.occ.addRectangle(-1.0, -1.0, 0.0, 2.0, 2.0)
        high = gmsh.model.occ.addRectangle(-1.0, -1.0, 1.0, 2.0, 2.0)
        gmsh.model.occ.synchronize()
        _inject_claim(monkeypatch, 0, 1)

        result, _groups = _cut_groups(
            [
                OccSurfaceGroup("low", OccSurfaceSelector([low]), OccSurfaceRole("a")),
                OccSurfaceGroup("high", OccSurfaceSelector([high]), OccSurfaceRole("b")),
            ]
        )

        assert len(result.parent_to_children[low]) == 1
        assert not set(result.parent_to_children[low]) & set(result.parent_to_children[high])
        assert result.report["cut"]["parentage_claims_disproved"] == 1


def test_a_claim_on_a_coplanar_piece_outside_the_parents_trim_is_dropped(monkeypatch):
    with _gmsh_session():
        outer = gmsh.model.occ.addRectangle(-2.0, -1.0, 0.0, 4.0, 2.0)
        # The pads lie on the outer face's own plane, so only the trim can
        # tell the removed left pad's claim apart.
        right = gmsh.model.occ.addRectangle(1.0, -0.5, 0.0, 0.5, 1.0)
        left = gmsh.model.occ.addRectangle(-1.5, -0.5, 0.0, 0.5, 1.0)
        gmsh.model.occ.synchronize()
        _inject_claim(monkeypatch, 2, 0)  # the removed left face claims outer's piece

        result, groups = _cut_groups(
            [
                OccSurfaceGroup("outer", OccSurfaceSelector([outer]), OccSurfaceRole("rigid")),
                OccSurfaceGroup("pads", OccSurfaceSelector([right, left]), OccSurfaceRole("pad")),
            ]
        )

        assert result.parent_to_children[left] == []
        assert groups["pads"] == [right]
        assert result.report["cut"]["parentage_claims_disproved"] == 1


def test_a_sample_outside_the_trim_but_near_its_edge_does_not_disprove_a_claim(monkeypatch):
    """A kept strip narrower than the tolerance is all edge; trim noise there is doubt."""
    with _gmsh_session():
        eps = 1.0e-6
        tags = [
            gmsh.model.occ.addRectangle(-1.0, -1.0, 0, 1 + eps, 2.0),
            gmsh.model.occ.addRectangle(eps, -1.0, 0, 1 - eps, 2.0),
        ]
        gmsh.model.occ.synchronize()
        evidence = step_prepare._claim_evidence

        def noisy_trim(parent, child, **kwargs):
            with monkeypatch.context() as patch:
                patch.setattr(gmsh.model, "isInside", lambda *_a, **_k: 0)
                return evidence(parent, child, **kwargs)

        monkeypatch.setattr(step_prepare, "_claim_evidence", noisy_trim)

        result, groups = _cut_groups(
            [OccSurfaceGroup("all", OccSurfaceSelector(tags), OccSurfaceRole("rigid"))]
        )

        assert result.parent_to_children[tags[0]]
        assert len(groups["all"]) == 2


def _undecided_for(monkeypatch, parent_of):
    evidence = step_prepare._claim_evidence

    def unsure(parent, child, **kwargs):
        verdict = evidence(parent, child, **kwargs)
        return "unknown" if parent == parent_of["copy"] else verdict

    monkeypatch.setattr(step_prepare, "_claim_evidence", unsure)
    copy = gmsh.model.occ.copy

    def remember(dimtags):
        out = copy(dimtags)
        if dimtags[0][1] == parent_of["original"]:
            parent_of["copy"] = out[0][1]
        return out

    monkeypatch.setattr(gmsh.model.occ, "copy", remember)


def test_an_undecided_claim_into_another_roles_surface_is_refused(monkeypatch):
    with _gmsh_session():
        low = gmsh.model.occ.addRectangle(-1.0, -1.0, 0.0, 2.0, 2.0)
        high = gmsh.model.occ.addRectangle(-1.0, -1.0, 1.0, 2.0, 2.0)
        gmsh.model.occ.synchronize()
        _inject_claim(monkeypatch, 0, 1)
        _undecided_for(monkeypatch, {"original": low, "copy": None})
        groups = [
            OccSurfaceGroup("low", OccSurfaceSelector([low]), OccSurfaceRole("a")),
            OccSurfaceGroup("high", OccSurfaceSelector([high]), OccSurfaceRole("b")),
        ]
        with pytest.raises(RuntimeError, match="cannot confirm"):
            auto_cut_occ_geometry(groups, grid=5, planes=("x0",))


def test_an_undecided_claim_within_one_role_is_kept(monkeypatch):
    with _gmsh_session():
        low = gmsh.model.occ.addRectangle(-1.0, -1.0, 0.0, 2.0, 2.0)
        high = gmsh.model.occ.addRectangle(-1.0, -1.0, 1.0, 2.0, 2.0)
        gmsh.model.occ.synchronize()
        _inject_claim(monkeypatch, 0, 1)
        _undecided_for(monkeypatch, {"original": low, "copy": None})

        result, _groups = _cut_groups(
            [OccSurfaceGroup("all", OccSurfaceSelector([low, high]), OccSurfaceRole("rigid"))]
        )

        assert len(result.parent_to_children[low]) == 2
        assert result.report["cut"]["parentage_claims_undecided"] >= 1


def test_the_cut_leaves_no_parent_copies_behind():
    with _gmsh_session():
        surfaces = _box_surfaces(-1.0, -1.0, -1.0, 2.0, 2.0, 2.0)
        result, groups = _cut_groups(
            [OccSurfaceGroup("all", OccSurfaceSelector(surfaces), OccSurfaceRole("rigid"))]
        )
        remaining = {tag for _dim, tag in gmsh.model.getEntities(2)}
        assert remaining == set(groups["all"])
        assert len(remaining) == 5


def test_snap_band_uses_step_units_conversion():
    assert millimetres_to_step_units(1.0e-4, 1.0e-3) == pytest.approx(1.0e-4)
    assert millimetres_to_step_units(1.0e-4, 1.0) == pytest.approx(1.0e-7)
    points = np.asarray([[5.0e-8, 1.0, 2.0], [2.0e-7, 1.0, 2.0]])
    snap_symmetry_plane_vertices(
        points,
        symmetry_planes=("x0",),
        tolerance=millimetres_to_step_units(1.0e-4, 1.0),
    )
    assert points[:, 0].tolist() == [0.0, 2.0e-7]


def test_importing_the_package_never_imports_gmsh_or_scipy():
    """Every gmsh/scipy import in this package is lazy, by contract.

    Breaking it is invisible locally and lethal downstream: gmsh's C++
    runtime installs signal handlers on load, and on Linux that rearms
    SIGPIPE's default action inside any host process that merely imports
    hornlab_mesher — WG v2's ubuntu CI died with pytest exit 141 when a
    test wrote to a closed websocket. Run in a subprocess so this file's
    own imports cannot contaminate the check.
    """
    import subprocess
    import sys

    probe = (
        "import sys; import hornlab_mesher; "
        "import hornlab_mesher.step_prepare; "
        "bad = [m for m in ('gmsh', 'scipy') if m in sys.modules]; "
        "raise SystemExit(', '.join(bad) if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"importing hornlab_mesher pulled in: {result.stderr.strip()}"
    )
