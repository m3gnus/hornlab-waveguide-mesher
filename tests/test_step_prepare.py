from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

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
        assert {tag for _, tag in gmsh.model.getEntities(2)} == {
            tag for group in result.groups for tag in group.selector.surface_tags
        }


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
        with pytest.raises(RuntimeError, match="with no parent") as refused:
            auto_cut_occ_geometry(groups, grid=5, planes=("x0",))
        # The caller learns which planes the refused cut was for, and why.
        assert refused.value.accepted_planes == ("x0",)
        assert refused.value.report["planes"]["x0"]["accepted"] is True


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


def test_an_undecided_claim_into_another_groups_surface_is_refused(monkeypatch):
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


def test_an_undecided_claim_within_one_group_is_kept(monkeypatch):
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


def test_an_undecided_claim_between_two_sources_of_one_role_is_refused(monkeypatch):
    """Two strips narrower than the tolerance: the removed one's stale claim on
    the kept one lies outside its trim but within tolerance of its edge, so the
    real evidence is undecided. Both are sources, and still two sources."""
    with _gmsh_session():
        removed = gmsh.model.occ.addRectangle(-1.0e-4, -1.0, 0.0, 1.0e-4, 2.0)
        kept = gmsh.model.occ.addRectangle(0.0, -1.0, 0.0, 1.0e-4, 2.0)
        rigid = gmsh.model.occ.addRectangle(-1.0, -1.0, 1.0, 2.0, 2.0)
        gmsh.model.occ.synchronize()
        _inject_claim(monkeypatch, 0, 1)
        groups = [
            OccSurfaceGroup("lf", OccSurfaceSelector([removed]), OccSurfaceRole("source")),
            OccSurfaceGroup("mf", OccSurfaceSelector([kept]), OccSurfaceRole("source")),
            OccSurfaceGroup("rigid", OccSurfaceSelector([rigid]), OccSurfaceRole("rigid")),
        ]
        with pytest.raises(RuntimeError, match="cannot confirm"):
            auto_cut_occ_geometry(groups, grid=5, planes=("x0",))


def _fail_on_call(monkeypatch, owner, name, call):
    real = getattr(owner, name)
    count = {"n": 0}

    def failing(*args, **kwargs):
        count["n"] += 1
        if count["n"] == call:
            raise RuntimeError(f"injected {name} failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, failing)


@pytest.mark.parametrize(
    ("owner", "name", "call"),
    [
        ("occ", "copy", 1),
        ("occ", "copy", 2),
        ("occ", "synchronize", 1),   # after the copies
        ("occ", "addBox", 1),
        ("occ", "synchronize", 2),   # after the box
        ("occ", "intersect", 1),
        ("occ", "synchronize", 3),   # after the intersect
        ("mesher", "_verify_cut_parentage", 1),
    ],
)
def test_no_temporary_geometry_survives_a_failure_during_the_cut(monkeypatch, owner, name, call):
    with _gmsh_session():
        tags = [gmsh.model.occ.addRectangle(-1.0, -1.0, z, 2.0, 2.0) for z in (0.0, 1.0)]
        gmsh.model.occ.synchronize()
        copies = []
        copy = gmsh.model.occ.copy

        def recording(dimtags):
            out = copy(dimtags)
            copies.extend(tag for _dim, tag in out)
            return out

        monkeypatch.setattr(gmsh.model.occ, "copy", recording)
        target = step_prepare if owner == "mesher" else gmsh.model.occ
        _fail_on_call(monkeypatch, target, name, call)
        groups = [OccSurfaceGroup("all", OccSurfaceSelector(tags), OccSurfaceRole("rigid"))]

        with pytest.raises(RuntimeError, match="injected"):
            auto_cut_occ_geometry(groups, grid=5, planes=("x0",))

        monkeypatch.undo()
        gmsh.model.occ.synchronize()
        present = set(gmsh.model.occ.getEntities())
        assert not {(2, tag) for tag in copies} & present
        assert not [entity for entity in present if entity[0] == 3]
        surfaces = [tag for dim, tag in present if dim == 2]
        assert len(surfaces) == len(tags)


def test_the_cut_leaves_no_parent_copies_behind():
    with _gmsh_session():
        surfaces = _box_surfaces(-1.0, -1.0, -1.0, 2.0, 2.0, 2.0)
        result, groups = _cut_groups(
            [OccSurfaceGroup("all", OccSurfaceSelector(surfaces), OccSurfaceRole("rigid"))]
        )
        remaining = {tag for _dim, tag in gmsh.model.getEntities(2)}
        assert remaining == set(groups["all"])
        assert len(remaining) == 5


@pytest.mark.parametrize("planes", [("x0", "y0"), ("x0",), ("y0",)])
def test_periodic_cylinder_cut_preserves_each_parent_and_has_no_leaks(planes):
    # A synthetic closed cylinder with 3D-only STEP curves. The seam starts
    # on y0 and winds by 1e-5 mm at the other end (radial error < 1e-10 mm).
    # OCC's analytic boolean loses the entire wall, despite valid full meshing.
    path = Path(__file__).parent / "fixtures/step_prepare/periodic-cylinder.step"
    with _gmsh_session():
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        assert len(gmsh.model.getEntities(3)) == 1
        surfaces = [tag for _dim, tag in gmsh.model.getEntities(2)]
        areas = {tag: gmsh.model.occ.getMass(2, tag) for tag in surfaces}
        gmsh.option.setNumber("Mesh.MeshSizeMin", 4.0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 6.0)
        gmsh.model.mesh.generate(2)
        full_triangles = np.asarray(gmsh.model.mesh.getElementsByType(2)[1]).reshape(-1, 3)
        full_edges = np.sort(np.concatenate([
            full_triangles[:, [0, 1]], full_triangles[:, [1, 2]], full_triangles[:, [2, 0]],
        ]), axis=1)
        assert np.all(np.unique(full_edges, axis=0, return_counts=True)[1] == 2)
        gmsh.model.mesh.clear()
        groups = [
            OccSurfaceGroup(
                f"face-{tag}", OccSurfaceSelector([tag]),
                OccSurfaceRole("wall" if gmsh.model.getType(2, tag) == "Cylinder" else f"source-{tag}"),
            )
            for tag in surfaces
        ]

        result = auto_cut_occ_geometry(groups, planes=planes)

        assert result.planes == planes
        for index, parent in enumerate(surfaces):
            children = result.parent_to_children[parent]
            assert children
            def area(child):
                if gmsh.model.getType(2, child) != "Plane":
                    return gmsh.model.occ.getMass(2, child)
                # Independently integrate the disk boundary: OCC's mass
                # quadrature on rational cut curves can be off by > 0.1%.
                integral = 0.0
                x, weights = np.polynomial.legendre.leggauss(80)
                for _dim, curve in gmsh.model.getBoundary(
                    [(2, child)], combined=False, oriented=True
                ):
                    lo, hi = gmsh.model.getParametrizationBounds(1, abs(curve))
                    params = (lo + (hi - lo) * (x + 1) / 2).tolist()
                    xyz = np.asarray(gmsh.model.getValue(1, abs(curve), params)).reshape(-1, 3)
                    derivative = np.asarray(gmsh.model.getDerivative(1, abs(curve), params)).reshape(-1, 3)
                    integral += np.sign(curve) * float(
                        np.sum((xyz[:, 0] * derivative[:, 1] - xyz[:, 1] * derivative[:, 0]) * weights)
                    ) * float(hi[0] - lo[0]) / 4
                return abs(integral)

            assert sum(area(child) for child in children) == pytest.approx(
                areas[parent] / 2 ** len(planes), rel=5.0e-4
            )
            group = result.group(f"face-{parent}")
            assert set(group.selector.surface_tags) == set(children)
            assert group.role == groups[index].role
            gmsh.model.addPhysicalGroup(2, children, 101 + index)
        assert {tag for _dim, tag in gmsh.model.getEntities(2)} == {
            child for children in result.parent_to_children.values() for child in children
        }
        assert not gmsh.model.getEntities(3)
        # Point alignment must survive a caller synchronizing before meshing.
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        nodes, coords, _ = gmsh.model.mesh.getNodes()
        points = dict(zip(nodes, np.asarray(coords).reshape(-1, 3)))
        triangles = np.asarray(gmsh.model.mesh.getElementsByType(2)[1]).reshape(-1, 3)
        edges = np.sort(np.concatenate([
            triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]],
        ]), axis=1)
        edges, counts = np.unique(edges, axis=0, return_counts=True)
        assert max(counts) == 2
        free = edges[counts == 1]
        assert len(free)
        for edge in free:
            assert any(
                all(abs(points[node][step_prepare.SYMMETRY_AXIS_FOR_PLANE[plane]]) < 1.0e-7 for node in edge)
                for plane in planes
            )
        for index, parent in enumerate(surfaces):
            assert set(gmsh.model.getEntitiesForPhysicalGroup(2, 101 + index)) == set(
                result.parent_to_children[parent]
            )


@pytest.mark.parametrize(("name", "call"), [("convertToNURBS", 1), ("intersect", 2)])
def test_periodic_cut_retry_cleans_up_on_failure(monkeypatch, name, call):
    with _gmsh_session():
        path = Path(__file__).parent / "fixtures/step_prepare/periodic-cylinder.step"
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        surfaces = [tag for _dim, tag in gmsh.model.getEntities(2)]
        copies = []
        copy = gmsh.model.occ.copy

        def recording(dimtags):
            out = copy(dimtags)
            copies.extend(tag for _dim, tag in out)
            return out

        monkeypatch.setattr(gmsh.model.occ, "copy", recording)
        _fail_on_call(monkeypatch, gmsh.model.occ, name, call)
        with pytest.raises(RuntimeError, match="injected"):
            auto_cut_occ_geometry([
                OccSurfaceGroup("all", OccSurfaceSelector(surfaces), OccSurfaceRole("rigid"))
            ], planes=("x0", "y0"))

        monkeypatch.undo()
        gmsh.model.occ.synchronize()
        present = set(gmsh.model.occ.getEntities())
        assert not {(2, tag) for tag in copies} & present
        assert not gmsh.model.getEntities(3)


def test_periodic_cut_refuses_a_retry_that_still_loses_the_wall(monkeypatch):
    with _gmsh_session():
        path = Path(__file__).parent / "fixtures/step_prepare/periodic-cylinder.step"
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        surfaces = [tag for _dim, tag in gmsh.model.getEntities(2)]
        monkeypatch.setattr(gmsh.model.occ, "convertToNURBS", lambda _entities: None)
        with pytest.raises(RuntimeError, match="lost retained portions"):
            auto_cut_occ_geometry([
                OccSurfaceGroup("all", OccSurfaceSelector(surfaces), OccSurfaceRole("rigid"))
            ], planes=("x0", "y0"))


@pytest.mark.parametrize("carrier_y", [0.0, 0.01])
@pytest.mark.parametrize("nurbs", [False, True])
def test_cut_boundary_alignment_requires_a_carrier_on_the_plane(carrier_y, nurbs):
    with _gmsh_session():
        face = gmsh.model.occ.addRectangle(0, carrier_y, 0, 1, 1)
        if nurbs:
            gmsh.model.occ.convertToNURBS([(2, face)])
        gmsh.model.occ.synchronize()
        vertex = next(
            tag for _dim, tag in gmsh.model.getEntities(0)
            if np.allclose(gmsh.model.getValue(0, tag, []), [0, carrier_y, 0])
        )
        gmsh.model.setCoordinates(vertex, 0, carrier_y - 0.005, 0)

        count = step_prepare._align_cut_boundary_vertices(
            [face], ("y0",), tolerance=0.1, noise=1.0e-12
        )

        assert count == (1 if carrier_y == 0 else 0)
        gmsh.model.occ.synchronize()
        assert gmsh.model.getValue(0, vertex, [])[1] == pytest.approx(
            0 if carrier_y == 0 else carrier_y - 0.005
        )


def test_cut_coverage_detects_a_partially_missing_parent():
    with _gmsh_session():
        parent = gmsh.model.occ.addRectangle(-1, -1, 0, 2, 2)
        whole_half = gmsh.model.occ.addRectangle(0, -1, 0, 1, 2)
        partial_half = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        for child, missing in [(whole_half, []), (partial_half, [parent])]:
            assert step_prepare._uncovered_cut_parents(
                {parent: [child]}, {parent: parent}, ("x0",), tolerance=1.0e-6, grid=7
            ) == missing


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


def test_auto_cut_refuses_surfaces_it_would_cut_without_testing():
    """The cut intersects every surface; one left out of the groups used to be
    halved unexamined and then vanish from the returned groups."""
    with _gmsh_session():
        tested = _box_surfaces(-1.0, -0.5, 0.0, 2.0, 2.0, 2.0)
        _box_surfaces(-1.0, -0.5, 5.0, 2.0, 2.0, 2.0)
        groups = [OccSurfaceGroup("wall", OccSurfaceSelector(tested), OccSurfaceRole("rigid"))]
        with pytest.raises(ValueError, match="surfaces in no group"):
            auto_cut_occ_geometry(groups, grid=5)


def _thin_plate_with_one_sided_hole():
    """A 400 x 300 mm plate, 0.4 mm thick, with a 20 mm hole at (+85, +65).

    The hole is on one side of both mirror planes. Its wall is so short that
    its mirror image lies within the mirror tolerance of the plate's faces,
    and on the fixed 7 x 7 grid no sample of the large faces falls inside the
    hole or its mirror image: the floor grid alone calls the plate symmetric.
    """

    plate = gmsh.model.occ.addBox(-200.0, -150.0, 0.0, 400.0, 300.0, 0.4)
    hole = gmsh.model.occ.addCylinder(85.0, 65.0, -1.0, 0.0, 0.0, 2.4, 10.0)
    gmsh.model.occ.cut([(3, plate)], [(3, hole)])
    gmsh.model.occ.synchronize()
    surfaces = [tag for _dim, tag in gmsh.model.getEntities(2)]
    return [OccSurfaceGroup("all", OccSurfaceSelector(surfaces), OccSurfaceRole("opaque"))]


def test_a_small_one_sided_hole_between_floor_samples_is_not_cut_as_symmetric():
    with _gmsh_session():
        result = auto_cut_occ_geometry(_thin_plate_with_one_sided_hole(), planes=("x0", "y0"))
        assert result.planes == ()
        for plane in ("x0", "y0"):
            verdict = result.report["planes"][plane]
            assert verdict["feature_mismatches"] > 0
            assert verdict["points_off_model"] == 0 and verdict["role_mismatches"] == 0
            assert "no mirror image of the same shape" in verdict["reason"]


def test_the_floor_grid_alone_misses_the_one_sided_hole():
    """Without the small-feature check (the pre-4b exact scan, which old
    replays rebuild with) the floor grid accepts both planes here."""

    with _gmsh_session():
        result = auto_cut_occ_geometry(
            _thin_plate_with_one_sided_hole(), planes=("x0", "y0"), fast_check=False
        )
        assert result.planes == ("x0", "y0")
        assert "feature_mismatches" not in result.report["planes"]["x0"]


def test_known_verdicts_are_reused_not_tested_again(monkeypatch):
    with _gmsh_session():
        groups = _thin_plate_with_one_sided_hole()
        first = auto_cut_occ_geometry(groups, planes=("x0", "y0"))
    with _gmsh_session():
        groups = _thin_plate_with_one_sided_hole()
        tested = []
        real = step_prepare.evaluate_occ_plane_symmetry

        def counting(plane, **kwargs):
            tested.append(plane)
            return real(plane, **kwargs)

        monkeypatch.setattr(step_prepare, "evaluate_occ_plane_symmetry", counting)
        again = auto_cut_occ_geometry(
            groups, planes=("x0", "y0"), known_verdicts=first.report["planes"]
        )
    assert tested == []
    assert again.report["planes"] == first.report["planes"]
    assert again.report["reused_verdict_planes"] == ["x0", "y0"]
    assert again.planes == first.planes


def _plain_scan(monkeypatch):
    """Turn the three mirror-test speed-ups off: one isInside per point, every
    face projected in candidate order, every omitted gap measured singly."""

    def one_at_a_time(surface, uv, trim=None):
        return [step_prepare._param_inside(surface, uv[2 * i], uv[2 * i + 1])
                for i in range(len(uv) // 2)]

    monkeypatch.setattr(step_prepare, "_params_inside", one_at_a_time)
    monkeypatch.setattr(
        step_prepare._OccMirrorIndex, "_nearest_with_omitted_gaps", lambda *_args: None
    )
    real_init = step_prepare._OccMirrorTarget.__init__

    def without_roles(self, candidates, bboxes, tolerance, role_boundaries, roles=None):
        real_init(self, candidates, bboxes, tolerance, role_boundaries, None)

    monkeypatch.setattr(step_prepare._OccMirrorTarget, "__init__", without_roles)


def _role_model():
    """A box with a one-sided bore, two roles: some planes pass, some fail."""

    box = gmsh.model.occ.addBox(-40.0, -30.0, 0.0, 80.0, 60.0, 50.0)
    bore = gmsh.model.occ.addCylinder(12.0, 0.0, -1.0, 0.0, 0.0, 52.0, 6.0)
    gmsh.model.occ.cut([(3, box)], [(3, bore)])
    gmsh.model.occ.synchronize()
    surfaces = [tag for _dim, tag in gmsh.model.getEntities(2)]
    top = max(surfaces, key=lambda tag: gmsh.model.getBoundingBox(2, tag)[5])
    return [
        OccSurfaceGroup("top", OccSurfaceSelector([top]), OccSurfaceRole("source")),
        OccSurfaceGroup("rest", OccSurfaceSelector([t for t in surfaces if t != top]),
                        OccSurfaceRole("rigid")),
    ]


@pytest.mark.parametrize("model", ["role-model", "holed-plate", "box"])
def test_the_mirror_test_speed_ups_change_no_number(model, monkeypatch):
    """Batched trim queries, own-role-first early exit and the bulk gap screen
    give the report the plain scan gives, byte for byte (the exact path that
    ``fast_check=False`` keeps for old replays)."""

    import json

    def build():
        if model == "role-model":
            return _role_model()
        if model == "holed-plate":
            return _thin_plate_with_one_sided_hole()
        surfaces = _box_surfaces(-1.0, -1.0, -1.0, 2.0, 2.0, 2.0)
        return [OccSurfaceGroup("all", OccSurfaceSelector(surfaces), OccSurfaceRole("opaque"))]

    with _gmsh_session():
        fast = auto_cut_occ_geometry(build(), planes=("x0", "y0", "z0"), fast_check=False)
    with monkeypatch.context() as plain:
        _plain_scan(plain)
        with _gmsh_session():
            slow = auto_cut_occ_geometry(build(), planes=("x0", "y0", "z0"), fast_check=False)
    assert json.dumps(fast.report, sort_keys=True) == json.dumps(slow.report, sort_keys=True)
    assert fast.planes == slow.planes
    assert [list(g.selector.surface_tags) for g in fast.groups] == [
        list(g.selector.surface_tags) for g in slow.groups
    ]


def test_batched_trim_queries_keep_the_samples_bit_for_bit(monkeypatch):
    with _gmsh_session():
        _thin_plate_with_one_sided_hole()
        surfaces = [tag for _dim, tag in gmsh.model.getEntities(2)]
        fast = {tag: step_prepare.sample_occ_surface_points(tag, grid=7) for tag in surfaces}
        calls = []
        real = gmsh.model.isInside

        def counted(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(gmsh.model, "isInside", counted)
        batched = sum(1 for _ in [step_prepare.sample_occ_surface_points(t, grid=7) for t in surfaces])
        batched_calls = len(calls)
        monkeypatch.setattr(
            step_prepare, "_params_inside",
            lambda surface, uv, trim=None: [step_prepare._param_inside(surface, uv[2 * i], uv[2 * i + 1])
                                 for i in range(len(uv) // 2)],
        )
        calls.clear()
        slow = {tag: step_prepare.sample_occ_surface_points(tag, grid=7) for tag in surfaces}
    assert batched == len(surfaces)
    for tag in surfaces:
        assert fast[tag].tobytes() == slow[tag].tobytes()
    assert batched_calls < len(calls)
