"""Shared OCC point-grid surfaces for expression-shaped native meridians."""
from __future__ import annotations

import numpy as np

from ..geometry import BuiltGeometry
from ..normals import open_shell_wall_orientation_references
from ..tags import PhysicalGroup
from ._occ import make_planar_fill_from_boundary, require_gmsh
from .point_grid_surfaces import _add_occ_bspline_patch_wall_surfaces


def build(geometry):
    surface = geometry.expression_surface
    gmsh = require_gmsh()
    groups = []
    roles = []
    lip = surface.lip_size_mm is not None
    for index, (role, grid) in enumerate(surface.branches):
        groups.append(_add_occ_bspline_patch_wall_surfaces(
            grid, closed=True, surface_fit="interpolate", interpolate_u=True))
        if lip and role in {"horn.inner", "horn.outer"} and surface.branches[index+1][0] != role:
            role = "lip_inner" if role == "horn.inner" else "lip_outer"
        roles.append(role)
    gmsh.model.occ.synchronize()
    source = make_planar_fill_from_boundary(groups[0], source_axis="z", use_min=True, closed=True)
    if not source:
        raise ValueError("Native OSSE source closure failed. Increase angular samples or simplify the body expressions.")
    if geometry.outer_points is not None:
        rear = make_planar_fill_from_boundary(groups[-1], source_axis="z", use_min=True, closed=True)
        if not rear:
            raise ValueError("Native OSSE rear closure failed. Increase angular samples or simplify the body expressions.")
        groups.append(rear)
        roles.append("wall.rear_cap")
    groups.append(source)
    roles.append("source_cap")
    mapped = {}
    for role, group in zip(roles, groups):
        tags = {int(t) for d, t in group if d == 2}
        mapped.setdefault(role, set()).update(tags)

    def join_roles(names):
        selected = [(name, sorted(mapped[name])) for name in names if name in mapped]
        flat = [(2, tag) for _, tags in selected for tag in tags]
        _, mapping = gmsh.model.occ.fragment(flat[:1], flat[1:])
        cursor = 0
        for name, tags in selected:
            mapped[name] = {int(tag) for parts in mapping[cursor:cursor+len(tags)] for dim, tag in parts if dim == 2}
            cursor += len(tags)

    # Join adjacent spans on each skin. A single global boolean also compares
    # the near-parallel offset skins, which is needlessly expensive for cubic
    # adapter offsets. The rim is the shared seam between the two skin joins.
    join_roles(("horn.inner", "lip_inner", "source_cap", "mouth_rim"))
    if lip:
        join_roles(("horn.outer", "lip_outer", "wall.rear_return", "wall.rear_cap", "mouth_rim"))
    gmsh.model.occ.synchronize()
    rigid = sorted(set().union(*(tags for role, tags in mapped.items() if role != "source_cap")))
    inner = geometry.inner_points
    points = np.concatenate([grid.reshape(-1, 3) for _, grid in surface.branches])
    refs = (None, None) if geometry.outer_points is not None else open_shell_wall_orientation_references(inner, closed=True)
    return BuiltGeometry(
        surface_groups={int(PhysicalGroup.RIGID_WALL): rigid,
                        int(PhysicalGroup.PRIMARY_SOURCE): sorted(mapped["source_cap"])},
        axial_bounds_mm=(float(points[:, 2].min()), float(points[:, 2].max())), source_axis="z",
        mesh_surface_groups={"inner": sorted(mapped["horn.inner"]),
                             "outer": sorted(mapped.get("horn.outer", [])),
                             "lip": sorted(set().union(*(mapped.get(role, set()) for role in ("lip_inner", "lip_outer", "mouth_rim")))),
                             "rear": sorted(mapped.get("wall.rear_return", set()) | mapped.get("wall.rear_cap", set())),
                             "throat_disc": sorted(mapped["source_cap"])},
        open_shell_wall_points_mm=refs[0], open_shell_wall_normals=refs[1],
        metadata={"nativeExpressionBody": True,
                  **({"mouthRoundoverMeshSizeMm": surface.lip_size_mm} if lip else {})})
