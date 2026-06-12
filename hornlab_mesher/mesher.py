from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import meshio
import numpy as np

from .builders import (
    build_osse_waveguide,
    build_point_grid,
)
from .builders._occ import add_physical_groups
from .density import configure_density
from .geometry import (
    BuiltGeometry,
    HornGeometry,
    MeshDensity,
    MeshInfo,
    OsseHornGeometry,
    PointGridHornGeometry,
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

    if output_path is None:
        handle = tempfile.NamedTemporaryFile(prefix="hornlab-mesher-", suffix=".msh", delete=False)
        out_path = Path(handle.name)
        handle.close()
    else:
        out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import gmsh

    with _GMSH_LOCK:
        initialized_here = False
        try:
            if not gmsh.isInitialized():
                gmsh.initialize()
                initialized_here = True
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Geometry.Tolerance", 1e-8)
            gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-8)
            gmsh.clear()
            gmsh.model.add("HornLabMesher")

            built = _dispatch_builder(geometry)
            gmsh.model.occ.synchronize()

            configure_density(built, mesh_density)
            add_physical_groups(built.surface_groups)

            gmsh.option.setNumber("Mesh.Algorithm", int(built.mesh_algorithm or 1))
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.removeDuplicateNodes()

            with tempfile.NamedTemporaryFile(prefix="hornlab-mesher-raw-", suffix=".msh", delete=False) as tmp:
                raw_path = Path(tmp.name)
            gmsh.write(str(raw_path))

            info = _postprocess_mesh(
                raw_path,
                out_path,
                built.source_axis,
                scale_to_metres,
                symmetry_snap_axes=built.symmetry_snap_axes,
                symmetry_snap_tol_mm=built.symmetry_snap_tol_mm,
            )
            raw_path.unlink(missing_ok=True)
            _validate_physical_tags(set(info.physical_groups))
            return out_path, info
        except Exception as exc:
            raise MesherError(f"mesh build failed: {exc}") from exc
        finally:
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
) -> MeshInfo:
    mesh = meshio.read(raw_path)
    triangles, phys = _triangles_and_physical_tags(mesh)
    if len(triangles) == 0:
        raise MesherError("gmsh produced no triangle elements")

    points = np.asarray(mesh.points, dtype=np.float64)
    _snap_symmetry_planes(points, symmetry_snap_axes, symmetry_snap_tol_mm)
    triangles, phys = _remove_symmetry_plane_slivers(points, triangles, phys, symmetry_snap_axes)
    triangles, phys, _ = remove_degenerate_triangles(points, triangles, phys, min_quality=1.0e-4)
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
        require_positive_volume=True,
        require_source_normal=True,
    )
    if report.watertight and not report.edge_consistent:
        raise MeshOrientationError(
            f"watertight mesh has {report.inconsistent_edges} inconsistent shared edges"
        )
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
        units="m" if _looks_like_metres(points) else "mm",
        edge_stats_mm=edge_stats,
    )


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
            "median_edge_mm": float(np.median(edges)),
            "p95_edge_mm": float(np.percentile(edges, 95)),
            "max_edge_mm": float(np.max(edges)),
        }
    return stats


def _snap_symmetry_planes(points: np.ndarray, axes: tuple[str, ...], tolerance: float) -> None:
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
    physical_data = mesh.cell_data.get("gmsh:physical") or mesh.cell_data.get("physical")
    for idx, cell_block in enumerate(mesh.cells):
        if cell_block.type not in ("triangle", "triangle3"):
            continue
        triangles.append(np.asarray(cell_block.data, dtype=np.int64))
        if physical_data is not None and idx < len(physical_data):
            tags.append(np.asarray(physical_data[idx], dtype=np.int32))
        else:
            tags.append(np.full(len(cell_block.data), int(PhysicalGroup.RIGID_WALL), dtype=np.int32))
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
