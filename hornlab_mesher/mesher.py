from __future__ import annotations

import math
import tempfile
import threading
from dataclasses import replace
from pathlib import Path

import meshio
import numpy as np

from .builders import (
    build_osse_waveguide,
    build_point_grid,
)
from .builders._occ import add_physical_groups
from .density import (
    _parse_quadrant_resolutions,
    configure_density,
    effective_triangle_limit,
)
from .geometry import (
    BuiltGeometry,
    HornGeometry,
    MeshDensity,
    MeshInfo,
    OsseHornGeometry,
    PointGridHornGeometry,
    validate_mesh_density,
)
from .normals import (
    MeshOrientationError,
    remove_degenerate_triangles,
    repair_orientation,
    validate_orientation,
)
from .tags import PHYSICAL_NAMES, PhysicalGroup


_GMSH_LOCK = threading.RLock()


class MesherError(Exception):
    pass


def _acoustic_geometry(
    geometry: HornGeometry,
    density: MeshDensity,
) -> tuple[HornGeometry, dict[str, object]]:
    """Apply semantic acoustic LOD without changing manufacturing geometry."""

    if not isinstance(geometry, PointGridHornGeometry) or geometry.enclosure is None:
        return geometry, {}
    if geometry.topology_mode == "legacy" or geometry.preserve_grid:
        return geometry, {"meshTopologyMode": "legacy"}

    from .builders.enclosure import enclosure_box_bounds

    enclosure = geometry.enclosure
    bounds = enclosure_box_bounds(
        geometry.inner_points,
        enclosure,
        closed=geometry.closed,
        symmetry_planes=tuple(geometry.symmetry_planes),
    )
    clamped_edge = float(bounds.get("clamped_edge", 0.0) or 0.0)
    edge_depth = float(bounds.get("edge_depth", 0.0) or 0.0)
    front = _parse_quadrant_resolutions(density.enc_front_res_mm, density.mouth_res_mm)
    back = _parse_quadrant_resolutions(density.enc_back_res_mm, density.mouth_res_mm)
    adjacent_h = min(*front, *back)
    feature_length = (
        0.5 * math.pi * clamped_edge
        if int(enclosure.edge_type) == 1
        else math.hypot(clamped_edge, edge_depth)
    )
    suppressed = clamped_edge > 0.0 and feature_length < adjacent_h
    metadata: dict[str, object] = {
        "meshTopologyMode": "acoustic",
        "acousticEnclosureRequestedEdgeMm": float(enclosure.edge_mm),
        "acousticEnclosureClampedEdgeMm": clamped_edge,
        "acousticEnclosureFeatureLengthMm": feature_length,
        "acousticEnclosureAdjacentResolutionMm": adjacent_h,
        "acousticEnclosureEdgeSuppressed": suppressed,
    }
    if not suppressed:
        return geometry, metadata
    # The rounded-rectangle sharp path is the robust semantic representation
    # for both suppressed fillets and suppressed chamfers.
    return replace(
        geometry,
        enclosure=replace(enclosure, edge_mm=0.0, edge_type=1),
    ), metadata


def _dispatch_builder(geometry: HornGeometry) -> BuiltGeometry:
    if isinstance(geometry, OsseHornGeometry):
        return build_osse_waveguide(geometry)
    if isinstance(geometry, PointGridHornGeometry):
        return build_point_grid(geometry)
    raise TypeError(f"unsupported geometry type: {type(geometry)!r}")


def build_mesh(
    geometry: HornGeometry,
    density: MeshDensity | str | Path | None = None,
    output_path: str | Path | None = None,
    scale_to_metres: bool = True,
) -> Path:
    """Build a tagged, validated Gmsh ``.msh`` file."""

    return build_mesh_with_info(
        geometry,
        density,
        output_path,
        scale_to_metres=scale_to_metres,
    )[0]


def build_mesh_with_info(
    geometry: HornGeometry,
    density: MeshDensity | str | Path | None = None,
    output_path: str | Path | None = None,
    scale_to_metres: bool = True,
) -> tuple[Path, MeshInfo]:
    """Build a ``.msh`` file and return it with the inspection info.

    Equivalent to ``build_mesh`` followed by ``load_mesh``, but the info is
    collected from the post-processed arrays at write time so the file is not
    read back.
    """

    if isinstance(density, (str, Path)) and output_path is None:
        output_path = density
        density = None
    mesh_density = density if isinstance(density, MeshDensity) else MeshDensity()
    validate_mesh_density(mesh_density)

    if output_path is None:
        handle = tempfile.NamedTemporaryFile(
            prefix="hornlab-mesher-", suffix=".msh", delete=False
        )
        out_path = Path(handle.name)
        handle.close()
    else:
        out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import gmsh

    with _GMSH_LOCK:
        initialized_here = False
        raw_path: Path | None = None
        staged_path: Path | None = None
        try:
            if not gmsh.isInitialized():
                # interruptible=False skips gmsh's SIGINT handler, which only
                # the main thread may install — required for worker threads.
                gmsh.initialize(interruptible=False)
                initialized_here = True
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Geometry.Tolerance", 1e-8)
            gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-8)
            gmsh.clear()
            gmsh.model.add("HornLabMesher")

            acoustic_geometry, acoustic_metadata = _acoustic_geometry(
                geometry, mesh_density
            )
            built = _dispatch_builder(acoustic_geometry)
            built.metadata.update(acoustic_metadata)
            gmsh.model.occ.synchronize()

            configure_density(built, mesh_density)
            add_physical_groups(built.surface_groups)

            gmsh.option.setNumber("Mesh.Algorithm", int(built.mesh_algorithm or 1))
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.removeDuplicateNodes()

            with tempfile.NamedTemporaryFile(
                prefix="hornlab-mesher-raw-", suffix=".msh", delete=False
            ) as tmp:
                raw_path = Path(tmp.name)
            gmsh.write(str(raw_path))

            with tempfile.NamedTemporaryFile(
                dir=out_path.parent,
                prefix=f".{out_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                staged_path = Path(tmp.name)

            is_infinite_baffle = bool(getattr(geometry, "infinite_baffle", False))
            info = _postprocess_mesh(
                raw_path,
                staged_path,
                built.source_axis,
                scale_to_metres,
                symmetry_snap_axes=built.symmetry_snap_axes,
                symmetry_snap_tol_mm=built.symmetry_snap_tol_mm,
                vertical_offset_mm=float(
                    getattr(geometry, "vertical_offset_mm", 0.0) or 0.0
                ),
                # Coupled infinite-baffle meshes are interior-domain BIE
                # surfaces. Keep one consistent negative-volume winding:
                # source/wall normals point into the cavity and aperture
                # normals point -Z. Rayleigh exterior evaluation is selected by
                # the aperture tag, not by changing triangle winding.
                require_positive_volume=not is_infinite_baffle,
                infinite_baffle=is_infinite_baffle,
            )
            limit = effective_triangle_limit(built, mesh_density)
            if limit is not None:
                built.metadata["meshTriangleCount"] = int(info.n_triangles)
                built.metadata["meshTriangleLimitExceeded"] = bool(
                    info.n_triangles > limit
                )
            if (
                limit is not None
                and not mesh_density.allow_large_mesh
                and info.n_triangles > limit
            ):
                raise MesherError(
                    f"generated mesh contains {info.n_triangles:,} triangles, exceeding "
                    f"the effective limit {limit:,}; increase the relevant mm resolution, "
                    "raise max_triangles, or set allow_large_mesh=true explicitly"
                )
            if built.metadata:
                info.metadata.update(built.metadata)
            _validate_physical_tags(set(info.physical_groups))
            staged_path.replace(out_path)
            staged_path = None
            info = replace(info, path=out_path)
            return out_path, info
        except Exception as exc:
            raise MesherError(f"mesh build failed: {exc}") from exc
        finally:
            if raw_path is not None:
                raw_path.unlink(missing_ok=True)
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)
            if initialized_here and gmsh.isInitialized():
                gmsh.finalize()


def load_mesh(path: str | Path) -> MeshInfo:
    """Load and inspect a HornLab ``.msh`` file without creating a BEM grid."""

    mesh_path = Path(path)
    mesh = meshio.read(mesh_path)
    triangles, phys = _triangles_and_physical_tags(mesh)
    if len(triangles) == 0:
        raise MesherError("mesh contains no triangles")
    names = _physical_names(mesh)
    return MeshInfo(
        path=mesh_path,
        n_vertices=int(len(mesh.points)),
        n_triangles=int(len(triangles)),
        physical_groups=names,
        bounding_box=(np.min(mesh.points, axis=0), np.max(mesh.points, axis=0)),
        units="m" if _looks_like_metres(mesh.points) else "mm",
    )


def _postprocess_mesh(
    raw_path: Path,
    out_path: Path,
    source_axis: str,
    scale_to_metres: bool,
    *,
    symmetry_snap_axes: tuple[str, ...] = (),
    symmetry_snap_tol_mm: float = 1.0e-6,
    vertical_offset_mm: float = 0.0,
    require_positive_volume: bool = True,
    infinite_baffle: bool = False,
) -> MeshInfo:
    mesh = meshio.read(raw_path)
    triangles, phys = _triangles_and_physical_tags(mesh)
    if len(triangles) == 0:
        raise MesherError("gmsh produced no triangle elements")

    points = np.asarray(mesh.points, dtype=np.float64)
    # Snap BEFORE welding: snapping near-plane vertices onto the symmetry
    # planes can create coincident nodes, and only a subsequent weld folds
    # them back into single vertices (welding first would leave the
    # near-identical rows that make dense BEM solves singular).
    _snap_symmetry_planes(points, symmetry_snap_axes, symmetry_snap_tol_mm)
    # Gmsh stitches adjacent OCC patch boundaries with near-duplicate nodes
    # (micrometres apart on fine grids); welding them prevents overlapping
    # elements whose near-identical rows make dense BEM solves singular.
    triangles = _weld_near_duplicate_vertices(points, triangles, tol_mm=5.0e-3)
    triangles, phys = _remove_symmetry_plane_slivers(
        points, triangles, phys, symmetry_snap_axes
    )
    triangles, phys, _ = remove_degenerate_triangles(
        points, triangles, phys, min_quality=1.0e-4
    )
    if len(triangles) == 0:
        raise MesherError("gmsh produced only degenerate triangle elements")
    # Gmsh/OCC can emit an otherwise valid canonical surface with the whole
    # triangle set inward-wound, most visibly on non-monotonic freestanding
    # R-OSSE point grids. The mesher owns output orientation, so normalize its
    # generated triangles before the validation/write boundary.
    triangles, _ = repair_orientation(
        points,
        triangles,
        phys,
        source_axis=source_axis,
    )
    report = validate_orientation(
        points,
        triangles,
        phys,
        source_axis=source_axis,
        require_watertight=False,
        require_edge_consistency=False,
        require_positive_volume=require_positive_volume,
        require_source_normal=True,
    )
    if infinite_baffle:
        _validate_infinite_baffle_contract(
            points,
            triangles,
            phys,
            report=report,
            symmetry_snap_axes=symmetry_snap_axes,
            symmetry_snap_tol_mm=symmetry_snap_tol_mm,
        )
    if not report.edge_consistent and report.watertight:
        raise MeshOrientationError(
            f"watertight mesh has {report.inconsistent_edges} inconsistent shared edges"
        )
    points, triangles = _compact_unused_vertices(points, triangles)
    if vertical_offset_mm:
        # Mesh.VerticalOffset: rigid +y placement of the finished reduced/full
        # model, applied here (still in millimetres) after every cut-plane snap
        # and the enclosure build have run at the origin. The declared symmetry
        # plane stays at y=0, so a y-cut mesh reconstructs about y=0 as ATH does.
        points[:, 1] += float(vertical_offset_mm)
    edge_stats = _edge_stats_by_tag(points, triangles, phys)
    if scale_to_metres:
        points = points * 0.001

    used_tags = sorted({int(tag) for tag in phys.tolist()})
    physical_names = {
        int(tag): PHYSICAL_NAMES.get(int(tag), f"SD1D{1000 + int(tag) - 1}")
        for tag in used_tags
    }
    out_mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", triangles.astype(np.int64))],
        cell_data={
            "gmsh:physical": [phys.astype(np.int32)],
            "gmsh:geometrical": [phys.astype(np.int32)],
        },
        field_data={
            name: np.array([tag, 2], dtype=np.int32)
            for tag, name in physical_names.items()
        },
    )
    meshio.write(out_path, out_mesh, file_format="gmsh22", binary=False)
    return MeshInfo(
        path=out_path,
        n_vertices=int(len(points)),
        n_triangles=int(len(triangles)),
        physical_groups=physical_names,
        bounding_box=(np.min(points, axis=0), np.max(points, axis=0)),
        # Generated arrays have known units. Keep load_mesh's conservative
        # magnitude heuristic for arbitrary third-party files, but never use it
        # to mislabel a small unscaled build as metres.
        units="m" if scale_to_metres else "mm",
        edge_stats_mm=edge_stats,
    )


def _edge_uses(
    triangles: np.ndarray,
    phys: np.ndarray,
) -> dict[tuple[int, int], list[int]]:
    uses: dict[tuple[int, int], list[int]] = {}
    for tri, raw_tag in zip(triangles, phys):
        tag = int(raw_tag)
        for start, end in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = sorted((int(start), int(end)))
            if a != b:
                uses.setdefault((a, b), []).append(tag)
    return uses


def _tag_axis_projections(
    points: np.ndarray,
    triangles: np.ndarray,
    phys: np.ndarray,
    *,
    tag: int,
    axis_idx: int,
) -> np.ndarray:
    mask = phys == int(tag)
    if not np.any(mask):
        return np.empty((0,), dtype=np.float64)
    p0 = points[triangles[mask, 0]]
    p1 = points[triangles[mask, 1]]
    p2 = points[triangles[mask, 2]]
    return np.cross(p1 - p0, p2 - p0)[:, axis_idx]


def _validate_infinite_baffle_contract(
    points: np.ndarray,
    triangles: np.ndarray,
    phys: np.ndarray,
    *,
    report: object,
    symmetry_snap_axes: tuple[str, ...],
    symmetry_snap_tol_mm: float,
) -> None:
    """Enforce the coupled interior-BEM/Rayleigh mesh contract at write time."""

    wall_tag = int(PhysicalGroup.RIGID_WALL)
    source_tag = int(PhysicalGroup.PRIMARY_SOURCE)
    aperture_tag = int(PhysicalGroup.MOUTH_APERTURE)
    used_tags = {int(value) for value in phys.tolist()}
    missing = sorted({wall_tag, source_tag, aperture_tag} - used_tags)
    if missing:
        raise MesherError(
            "infinite-baffle mesh is missing required physical tag(s): "
            + ", ".join(str(tag) for tag in missing)
        )

    scale = max(float(np.ptp(points, axis=0).max()), 1.0)
    area_eps = max(1.0e-12, scale * scale * 1.0e-12)
    volume_eps = max(1.0e-12, scale * scale * scale * 1.0e-12)
    source_face_projections = _tag_axis_projections(
        points, triangles, phys, tag=source_tag, axis_idx=2
    )
    aperture_face_projections = _tag_axis_projections(
        points, triangles, phys, tag=aperture_tag, axis_idx=2
    )
    source_projection = float(np.sum(source_face_projections))
    aperture_projection = float(np.sum(aperture_face_projections))
    if source_projection <= area_eps:
        raise MesherError(
            "infinite-baffle primary source must have positive driven area with +Z normals "
            f"(projection {source_projection:.6g})"
        )
    if np.any(source_face_projections < -area_eps):
        raise MesherError(
            "infinite-baffle primary source contains triangle normals opposite +Z"
        )
    if aperture_projection >= -area_eps:
        raise MesherError(
            "infinite-baffle mouth aperture must have nonzero area with -Z normals "
            f"(projection {aperture_projection:.6g})"
        )
    if np.any(aperture_face_projections > area_eps):
        raise MesherError(
            "infinite-baffle mouth aperture contains triangle normals opposite -Z"
        )
    if float(getattr(report, "signed_volume")) >= -volume_eps:
        raise MesherError(
            "infinite-baffle interior domain must have negative signed volume "
            f"(got {float(getattr(report, 'signed_volume')):.6g})"
        )
    if int(getattr(report, "nonmanifold_edges")):
        raise MesherError(
            "infinite-baffle mesh is nonmanifold: "
            f"{int(getattr(report, 'nonmanifold_edges'))} edge(s)"
        )
    if int(getattr(report, "inconsistent_edges")):
        raise MesherError(
            "infinite-baffle mesh has inconsistent winding on "
            f"{int(getattr(report, 'inconsistent_edges'))} shared edge(s)"
        )

    aperture_nodes = np.unique(triangles[phys == aperture_tag])
    plane_tol = max(float(symmetry_snap_tol_mm), 1.0e-7)
    if np.any(np.abs(points[aperture_nodes, 2]) > plane_tol):
        span = float(np.ptp(points[aperture_nodes, 2]))
        raise MesherError(
            "infinite-baffle mouth aperture must be coplanar at z=0 "
            f"(z span {span:.6g} mm)"
        )

    edge_uses = _edge_uses(triangles, phys)
    shared_rim = [
        edge
        for edge, tags in edge_uses.items()
        if len(tags) == 2 and set(tags) == {wall_tag, aperture_tag}
    ]
    if not shared_rim:
        raise MesherError(
            "infinite-baffle wall and mouth aperture do not share a welded rim"
        )
    for edge in shared_rim:
        if np.any(np.abs(points[list(edge), 2]) > plane_tol):
            raise MesherError("infinite-baffle wall/aperture rim is not on z=0")

    boundary_edges = [edge for edge, tags in edge_uses.items() if len(tags) == 1]
    axes = tuple(str(axis) for axis in symmetry_snap_axes)
    if not axes:
        if boundary_edges:
            raise MesherError(
                "full infinite-baffle domain must be watertight; "
                f"found {len(boundary_edges)} open edge(s)"
            )
        return

    axis_idx = {"x": 0, "y": 1, "z": 2}
    unknown = [axis for axis in axes if axis not in axis_idx]
    if unknown:
        raise MesherError(f"unknown infinite-baffle cut plane axis/axes: {unknown!r}")
    for edge in boundary_edges:
        if not any(
            abs(float(points[edge[0], axis_idx[axis]])) <= plane_tol
            and abs(float(points[edge[1], axis_idx[axis]])) <= plane_tol
            for axis in axes
        ):
            raise MesherError(
                "reduced infinite-baffle domain has an open edge away from its declared cut planes"
            )


def _weld_near_duplicate_vertices(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    tol_mm: float = 5.0e-3,
) -> np.ndarray:
    """Remap triangle indices so vertices closer than ``tol_mm`` coincide.

    Uses a spatial hash with cells of the weld tolerance; clusters merge to
    the lowest vertex index. Orphaned duplicate vertices become unused and
    are dropped by the final compaction. Returns the remapped triangles.
    """

    if len(points) == 0 or len(triangles) == 0:
        return triangles
    cells = np.floor(points / tol_mm).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(map(tuple, cells)):
        buckets.setdefault(key, []).append(index)

    parent = np.arange(len(points))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    tol_sq = tol_mm * tol_mm
    neighbor_offsets = [
        (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    ]
    for key, indices in buckets.items():
        candidates: list[int] = []
        for dx, dy, dz in neighbor_offsets:
            candidates.extend(buckets.get((key[0] + dx, key[1] + dy, key[2] + dz), ()))
        for i in indices:
            pi = points[i]
            for j in candidates:
                if j <= i:
                    continue
                delta = points[j] - pi
                if float(delta @ delta) <= tol_sq:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)

    roots = np.fromiter(
        (find(i) for i in range(len(points))), dtype=np.int64, count=len(points)
    )
    if np.array_equal(roots, np.arange(len(points))):
        return triangles
    return roots[triangles]


def _compact_unused_vertices(
    points: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    used = np.unique(triangles)
    if len(used) == len(points):
        return points, triangles
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return points[used], remap[triangles]


_SYMMETRY_SLIVER_MAX_AREA_MM2 = 0.25


def _remove_symmetry_plane_slivers(
    points: np.ndarray,
    triangles: np.ndarray,
    phys: np.ndarray,
    symmetry_snap_axes: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Drop snap-flattened sliver triangles lying entirely in a symmetry plane.

    Snapping near-plane vertices exactly onto the symmetry planes can flatten
    seam slivers into the plane. Solvers reject in-plane triangles (the plane
    is an image plane, not a physical boundary), so remove the negligible-area
    artifacts and fail loudly if a substantial triangle ends up in-plane,
    which would indicate a real geometry defect.
    """

    if not symmetry_snap_axes or len(triangles) == 0:
        return triangles, phys
    axis_index = {"x": 0, "y": 1, "z": 2}
    keep = np.ones(len(triangles), dtype=bool)
    corners = points[triangles]
    for axis in symmetry_snap_axes:
        idx = axis_index.get(axis)
        if idx is None:
            continue
        in_plane = np.all(np.abs(corners[:, :, idx]) <= 1.0e-9, axis=1)
        if not np.any(in_plane):
            continue
        areas = 0.5 * np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            axis=1,
        )
        large = in_plane & (areas > _SYMMETRY_SLIVER_MAX_AREA_MM2)
        if np.any(large):
            raise MesherError(
                f"{int(np.sum(large))} non-sliver triangle(s) lie entirely in the "
                f"{axis}=0 symmetry plane; the geometry is defective"
            )
        keep &= ~in_plane
    if np.all(keep):
        return triangles, phys
    return triangles[keep], phys[keep]


def _edge_stats_by_tag(
    points_mm: np.ndarray,
    triangles: np.ndarray,
    phys: np.ndarray,
) -> dict[int, dict[str, float]]:
    """Per physical tag edge-length statistics in millimetres."""

    stats: dict[int, dict[str, float]] = {}
    for tag in sorted({int(value) for value in phys.tolist()}):
        tris = triangles[phys == tag]
        if len(tris) == 0:
            continue
        edges = np.concatenate(
            [
                np.linalg.norm(points_mm[tris[:, 0]] - points_mm[tris[:, 1]], axis=1),
                np.linalg.norm(points_mm[tris[:, 1]] - points_mm[tris[:, 2]], axis=1),
                np.linalg.norm(points_mm[tris[:, 2]] - points_mm[tris[:, 0]], axis=1),
            ]
        )
        stats[int(tag)] = {
            "min_edge_mm": float(np.min(edges)),
            "p05_edge_mm": float(np.percentile(edges, 5)),
            "median_edge_mm": float(np.median(edges)),
            "p95_edge_mm": float(np.percentile(edges, 95)),
            "max_edge_mm": float(np.max(edges)),
        }
    return stats


def _snap_symmetry_planes(
    points: np.ndarray, axes: tuple[str, ...], tolerance: float
) -> None:
    if not axes or tolerance <= 0.0:
        return
    axis_indices = {"x": 0, "y": 1, "z": 2}
    for axis in axes:
        idx = axis_indices.get(str(axis))
        if idx is None:
            continue
        mask = np.abs(points[:, idx]) <= float(tolerance)
        points[mask, idx] = 0.0


def _triangles_and_physical_tags(mesh: meshio.Mesh) -> tuple[np.ndarray, np.ndarray]:
    triangles: list[np.ndarray] = []
    tags: list[np.ndarray] = []
    physical_data = mesh.cell_data.get("gmsh:physical") or mesh.cell_data.get(
        "physical"
    )
    for idx, cell_block in enumerate(mesh.cells):
        if cell_block.type not in ("triangle", "triangle3"):
            continue
        triangles.append(np.asarray(cell_block.data, dtype=np.int64))
        if physical_data is not None and idx < len(physical_data):
            tags.append(np.asarray(physical_data[idx], dtype=np.int32))
        else:
            tags.append(
                np.full(
                    len(cell_block.data), int(PhysicalGroup.RIGID_WALL), dtype=np.int32
                )
            )
    if not triangles:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int32)
    return np.vstack(triangles), np.concatenate(tags)


def _physical_names(mesh: meshio.Mesh) -> dict[int, str]:
    _, used = _triangles_and_physical_tags(mesh)
    used_tags = {int(tag) for tag in used.tolist()}
    out: dict[int, str] = {}
    for name, raw in mesh.field_data.items():
        if len(raw) >= 2 and int(raw[1]) == 2 and int(raw[0]) in used_tags:
            out[int(raw[0])] = str(name)
    return out


def _looks_like_metres(points: np.ndarray) -> bool:
    span = np.max(points, axis=0) - np.min(points, axis=0)
    return bool(np.max(span) < 10.0)


def _validate_physical_tags(tags: set[int]) -> None:
    if int(PhysicalGroup.PRIMARY_SOURCE) not in tags:
        raise MesherError("mesh has no primary source physical group (tag 2)")
    if int(PhysicalGroup.RIGID_WALL) not in tags:
        raise MesherError("mesh has no rigid wall physical group (tag 1)")
