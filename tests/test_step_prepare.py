from __future__ import annotations

from contextlib import contextmanager

import gmsh
import numpy as np
import pytest

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
