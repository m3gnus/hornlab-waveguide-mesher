"""Placed half-model caps and walls agree with the native solve and STEP.

STEP exports a full body; the reduced solve and preview are compared in that
body's placed frame. Infinite-baffle builds place the mouth at z=0, so their
preview's throat-origin coordinates need the established axial registration.
"""

from __future__ import annotations

import copy

import gmsh
import meshio
import numpy as np
import pytest

from hornlab_mesher import build_from_config, write_step_from_config
from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.mesher import _triangles_and_physical_tags
from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry
from hornlab_mesher.preview.fidelity import _point_triangle_squared
from hornlab_mesher.tags import PhysicalGroup

from test_preview_source_offset import REPRODUCTIONS

# What this file guards is PLACEMENT: before the fix an open-ring preview cap sat
# on the untranslated axis, 7 mm (the vertical offset used here) from the solve and
# STEP cap. The bound is set from that defect, not from the measured residuals:
# far below the 7 mm offset, and above the fitting differences between the
# analytic preview, the reduced solve surface and the full STEP surface, which
# exist on the base independently of placement (up to about 0.6 mm on rounded
# caps of ICW and FREEFORM at this size) and are outside this test's subject.
# Exact placement is pinned separately, to 1e-12 mm, in test_preview_source_offset.
PLACEMENT_BOUND_MM = 1.0
OFFSET_MM = 7.0


def _config(family, mode, shape):
    if family in {"OSSE", "R-OSSE"}:
        config = copy.deepcopy(REPRODUCTIONS[f"{family}-{mode}"])
    else:
        profile = {
            "ICW": {"formula": "ICW", "r0": 12.7, "a0": 5,
                    "icw_coeffs": [0, 0, 0, 0, 0, 0], "icw_S": 80},
            "FREEFORM": {
                "formula": "FREEFORM",
                "profileH": {"points": [[0, 12.7], [40, 40], [80, 75]],
                             "throatAngleDeg": 5, "mouthAngleDeg": 45},
                "profileV": {"points": [[0, 12.7], [40, 35], [80, 60]],
                             "throatAngleDeg": 5, "mouthAngleDeg": 35},
            },
        }[family]
        config = {"profile": profile, "mode": mode,
                  "mesh": {"quadrants": "14", "vertical_offset_mm": 7,
                           "wall_thickness_mm": 3},
                  "enclosure": {"depth_mm": 110, "edge_mm": 5}}
    if mode != "enclosure":
        config.pop("enclosure", None)
    # A dense interpolating CAD lattice qualifies the analytic preview rather
    # than the unrelated approximation error of a very coarse control lattice.
    config["mesh"].update(angularSegments=64, lengthSegments=96,
                          surface_fit="interpolate", scale_to_metres=False,
                          throat_res_mm=1.5, mouth_res_mm=3, rear_res_mm=5,
                          allow_large_mesh=True, max_triangles=100_000)
    if family == "FREEFORM":
        config["mesh"]["surface_fit"] = "approximate"
    config["source"] = {"source_shape": shape}
    return config


def _sample(points, count=40):
    assert len(points) > 0, "parity must exercise actual surface points"
    return points[np.unique(np.linspace(0, len(points) - 1, min(count, len(points)), dtype=int))]


def _step_distances(points, surfaces):
    assert surfaces, "STEP contains no surfaces to compare"
    return np.array([
        min(np.linalg.norm(np.asarray(gmsh.model.getClosestPoint(2, tag, point.tolist())[0]) - point)
            for tag in surfaces)
        for point in points
    ])


def _mesh_distances(points, corners):
    assert len(corners) > 0, "solve contains no triangles for this role"
    return np.array([
        np.sqrt(np.min(_point_triangle_squared(
            np.broadcast_to(point, corners[:, 0].shape),
            corners[:, 0], corners[:, 1], corners[:, 2],
        ))) for point in points
    ])


@pytest.mark.parametrize("family", ["OSSE", "R-OSSE", "FREEFORM", "ICW"])
@pytest.mark.parametrize("mode", ["bare", "freestanding", "enclosure", "infinite-baffle"])
@pytest.mark.parametrize("shape", [0, 1], ids=["flat", "rounded"])
@pytest.mark.parametrize("quadrants", ["12", "14"])
def test_placed_half_preview_matches_solve_and_step(tmp_path, record_property, family, mode, shape, quadrants):
    config = _config(family, mode, shape)
    config["mesh"]["quadrants"] = quadrants
    resolved = resolve_geometry(config)
    assert resolved.geometry.vertical_offset_mm == 7
    build = build_from_config(config, tmp_path / "solve.msh")
    assert build.n_triangles > 0 and build.units == "mm"
    mesh = meshio.read(build.mesh_path)
    triangles, tags = _triangles_and_physical_tags(mesh)
    corners = mesh.points[triangles]
    groups = {"source_cap": PhysicalGroup.PRIMARY_SOURCE,
              "horn.inner": PhysicalGroup.RIGID_WALL}
    step, _ = write_step_from_config(config, tmp_path / "body.step", open_throat=False)

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(str(step), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
        for role, tag in groups.items():
            nodes = _sample(mesh.points[np.unique(triangles[tags == tag])])
            error = float(np.max(_step_distances(nodes, surfaces)))
            record_property("solve_to_step." + role, error)
            assert error < PLACEMENT_BOUND_MM, (family, mode, role, error)

        for lod in ("coarse", "fine"):
            preview = build_preview_geometry(config, PreviewOptionsV1(lod=lod))
            by_role = {s.role: s for s in preview.surfaces}
            for role, tag in groups.items():
                points = _sample(by_role[role].positions).copy()
                if mode == "infinite-baffle":
                    points[:, 2] -= np.mean(resolved.geometry.inner_points[:, -1, 2])
                # Includes the source pole, its interior and rim, and the
                # wall's full axial/azimuthal span. All checks are nonvacuous.
                step_error = float(np.max(_step_distances(points, surfaces)))
                mesh_error = float(np.max(_mesh_distances(points, corners[tags == tag])))
                record_property(lod + ".preview_to_step." + role, step_error)
                record_property(lod + ".preview_to_mesh." + role, mesh_error)
                assert step_error < PLACEMENT_BOUND_MM, (family, mode, lod, role, step_error)
                assert mesh_error < PLACEMENT_BOUND_MM, (family, mode, lod, role, mesh_error)
            if shape == 1:
                cap = by_role["source_cap"]
                np.testing.assert_allclose(cap.positions[0, :2], [0, 7], atol=1e-10)
    finally:
        gmsh.finalize()
