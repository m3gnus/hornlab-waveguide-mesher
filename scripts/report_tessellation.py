#!/usr/bin/env python3
"""Measure preview and solve tessellation for mesher configurations.

See docs/tessellation-report.md for input formats and measurement limits.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import meshio
import numpy as np
from scipy.spatial import cKDTree

from hornlab_mesher.config_builder import build_from_config
from hornlab_mesher.config_parser import load_config
from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry
from hornlab_mesher.preview.api import _lod_config
from hornlab_mesher.preview.fidelity import (
    _point_segment_squared,
    _point_triangle_squared,
)
from hornlab_mesher.preview.primitives import _grid_indices
from hornlab_mesher.quality import element_shape_report, mesh_quality_report
from hornlab_mesher.viewport import build_viewport_geometry_from_config


def _load(path: Path) -> dict[str, Any]:
    config = load_config(path)
    if isinstance(config.get("config"), dict):
        config = config["config"]
    return config


def _json_value(value: Any) -> Any:
    """Convert NumPy metadata without masking unsupported or non-finite data."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _shape(points: np.ndarray, triangles: np.ndarray) -> dict[str, Any]:
    return asdict(element_shape_report(points, triangles, sliver_angle_deg=15.0, sample_limit=0))


def _reference_meridian(config: dict[str, Any], stations: int = 4097) -> tuple[np.ndarray | None, str | None, float]:
    """Sample a meridian when the canonical surface appears axisymmetric.

    The preview sampler is not used: this is a separately sampled canonical
    point grid. A non-axisymmetric or native-only case uses a separate surface
    approximation instead of pretending one meridian describes its wall.
    """
    def grid_at(n: int):
        sampled = _lod_config(config, 64 if n == 33 else 16, n - 1)
        mesh = sampled["mesh"]
        mesh["wall_thickness_mm"] = 0.0
        for alias in ("wallThickness", "wall_thickness", "WallThickness"):
            mesh.pop(alias, None)
        # Keep native sampling modes and their input validation intact.
        return build_viewport_geometry_from_config(sampled, point_lists=False)["grid"]

    try:
        coarse = grid_at(33)
        points = np.asarray(coarse["inner_grid"], dtype=float)
        offset = float(coarse.get("vertical_offset_mm", 0.0) or 0.0)
        radius = np.hypot(points[:, :, 0], points[:, :, 1] - offset)
        span = max(float(np.ptp(points)), 1.0)
        if np.max(np.ptp(radius, axis=0)) > span * 1e-8 or np.max(np.ptp(points[:, :, 2], axis=0)) > span * 1e-8:
            return None, "surface is not axisymmetric; meridian measurement unavailable", offset
        dense = grid_at(stations)
        points = np.asarray(dense["inner_grid"], dtype=float)[0]
        offset = float(dense.get("vertical_offset_mm", 0.0) or 0.0)
        radius = np.hypot(points[:, 0], points[:, 1] - offset)
        return np.column_stack((points[:, 2], radius)), None, offset
    except (ValueError, TypeError, KeyError, NotImplementedError) as exc:
        return None, f"canonical meridian unavailable: {type(exc).__name__}: {exc}", 0.0


def _reference_surface(config: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Build a dense reference without using the fine preview tessellation.

    The distance to this triangulation is an approximation, not an exact
    closest-point certificate. Native bodies with their own preview builders
    use their inspection LOD when the point-grid evaluator does not apply.
    """
    try:
        sampled = _lod_config(config, 256, 256)
        mesh = sampled["mesh"]
        mesh["wall_thickness_mm"] = 0.0
        for alias in ("wallThickness", "wall_thickness", "WallThickness"):
            mesh.pop(alias, None)
        grid = build_viewport_geometry_from_config(sampled, point_lists=False)["grid"]
        points = np.ascontiguousarray(np.transpose(grid["inner_grid"], (1, 0, 2)))
        faces = _grid_indices(*points.shape[:2], closed_phi=bool(grid.get("full_circle", True))).reshape(-1, 3)
        points = points.reshape(-1, 3)
        method = "dense canonical point-grid triangulation; nearest 16 candidate facets"
    except (ValueError, TypeError, KeyError, NotImplementedError):
        try:
            inspection = build_preview_geometry(config, PreviewOptionsV1(lod="inspection", include_curvature=False))
            inner = next(s for s in inspection.surfaces if s.role in ("horn.inner", "inner"))
            points = inner.positions
            faces = inner.indices.reshape(-1, 3)
            method = "inspection preview triangulation; nearest 16 candidate facets"
        except (ValueError, TypeError, KeyError, NotImplementedError, StopIteration) as exc:
            return None, f"surface approximation unavailable: {type(exc).__name__}: {exc}"
    if not len(faces):
        return None, "surface approximation has no triangles"
    corners = points[np.asarray(faces, dtype=np.int64)]
    return {"corners": corners, "tree": cKDTree(corners.mean(axis=1)), "method": method}, None


def _sampled_surface_distance(points: np.ndarray, reference: dict[str, Any]) -> np.ndarray:
    """Approximate closest distance to the dense reference facets in batches."""
    result = np.empty(len(points), dtype=float)
    corners = reference["corners"]
    count = min(16, len(corners))
    for start in range(0, len(points), 1024):
        query = points[start:start + 1024]
        _, nearest = reference["tree"].query(query, k=count)
        nearest = np.asarray(nearest).reshape(len(query), count)
        candidate = corners[nearest]
        squared = _point_triangle_squared(query[:, None, :], candidate[:, :, 0], candidate[:, :, 1], candidate[:, :, 2])
        result[start:start + len(query)] = np.sqrt(np.min(squared, axis=1))
    return result


def _meridian_distance(points: np.ndarray, reference: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray]:
    query = np.column_stack((points[:, 2], np.hypot(points[:, 0], points[:, 1] - offset)))
    _, nearest = cKDTree(reference).query(query)
    # Both adjacent segments must be considered; this is exact locally for the
    # nearest reference vertex and uses the preview's shared segment routine.
    best = np.full(len(query), np.inf)
    best_segment = np.zeros(len(query), dtype=np.int64)
    for shift in (-1, 0):
        index = np.clip(nearest + shift, 0, len(reference) - 2)
        distance = _point_segment_squared(query, reference[index], reference[index + 1])
        improve = distance < best
        best[improve] = distance[improve]
        best_segment[improve] = index[improve]
    return np.sqrt(best), best_segment


def _surface_stats(points: np.ndarray, triangles: np.ndarray, reference: np.ndarray | None, offset: float, surface_reference: dict[str, Any] | None = None) -> dict[str, Any]:
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    result: dict[str, Any] = {
        "vertices": len(points),
        "triangles": len(triangles),
        "quality": mesh_quality_report(points, triangles, vertex_units="mm", sliver_angle_deg=15.0, sample_limit=0),
    }
    if reference is None or not len(triangles):
        result["meridian_deviation"] = None
        if surface_reference is not None and len(triangles):
            centroids = points[triangles].mean(axis=1)
            distance = _sampled_surface_distance(centroids, surface_reference)
            ref_z = surface_reference["corners"][:, :, 2]
            z_min, z_max = float(np.min(ref_z)), float(np.max(ref_z))
            z_span = max(z_max - z_min, np.finfo(float).eps)
            deciles = np.clip((10.0 * (centroids[:, 2] - z_min) / z_span).astype(int), 0, 9)
            regions = []
            for decile in range(10):
                selected = deciles == decile
                if not np.any(selected):
                    continue
                shape = _shape(points, triangles[selected])
                regions.append({
                    "z_decile": decile,
                    "triangles": int(np.count_nonzero(selected)),
                    "fraction_below_15_deg": shape["sliver_fraction"],
                    "p1_radius_ratio": shape["p1_radius_ratio"],
                    "centroid_deviation_max_mm": float(np.max(distance[selected])),
                })
            result["sampled_surface_deviation"] = {
                "method": surface_reference["method"],
                "max_mm": float(np.max(distance)),
                "rms_mm": float(np.sqrt(np.mean(distance * distance))),
                "by_z_decile": regions,
            }
        return result
    centroid = points[triangles].mean(axis=1)
    distance, segments = _meridian_distance(centroid, reference, offset)
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(reference, axis=0), axis=1))))
    if arc[-1] == 0.0:
        result["meridian_deviation"] = None
        return result
    deciles = np.minimum((10.0 * arc[segments] / arc[-1]).astype(int), 9)
    regions = []
    for decile in range(10):
        select = deciles == decile
        if not np.any(select):
            continue
        shape = _shape(points, triangles[select])
        regions.append({
            "arc_decile": decile,
            "triangles": int(np.count_nonzero(select)),
            "fraction_below_15_deg": shape["sliver_fraction"],
            "p1_radius_ratio": shape["p1_radius_ratio"],
            "centroid_deviation_max_mm": float(np.max(distance[select])),
        })
    result["meridian_deviation"] = {
        "method": "triangle centroids to sampled canonical axisymmetric meridian",
        "reference_stations": len(reference),
        "max_mm": float(np.max(distance)),
        "rms_mm": float(np.sqrt(np.mean(distance * distance))),
        "by_arc_decile": regions,
    }
    return result


def _measure(config: dict[str, Any], *, solve: bool, meridian: bool) -> dict[str, Any]:
    reference, why, offset = (None, "not requested", 0.0)
    if meridian:
        reference, why, offset = _reference_meridian(config)
    started = time.perf_counter()
    preview = build_preview_geometry(config, PreviewOptionsV1(lod="fine", include_curvature=False))
    preview_ms = (time.perf_counter() - started) * 1000.0
    surface_reference = None
    if meridian and reference is None:
        surface_reference, surface_why = _reference_surface(config)
        if surface_why:
            why = f"{why}; {surface_why}"
    surfaces = {}
    for surface in preview.surfaces:
        # The reference describes the horn's acoustic wall, not other roles.
        is_inner = surface.role in ("horn.inner", "inner")
        ref = reference if is_inner else None
        sample_ref = surface_reference if is_inner else None
        surfaces[surface.role] = _surface_stats(surface.positions, surface.indices, ref, offset, sample_ref)
    result: dict[str, Any] = {
        "preview_ms": preview_ms,
        "preview_surfaces": surfaces,
        "preview_fidelity": preview.metadata.get("fidelity", {}),
        "meridian_unavailable_reason": why,
    }
    if solve:
        with tempfile.TemporaryDirectory() as directory:
            mesh_path = Path(directory) / "mesh.msh"
            started = time.perf_counter()
            try:
                built = build_from_config(config, mesh_path)
            except Exception as exc:  # noqa: BLE001 - preserve the preview and report the solve refusal
                result["solve"] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                return result
            elapsed = (time.perf_counter() - started) * 1000.0
            mesh = meshio.read(mesh_path)
        points = np.asarray(mesh.points, dtype=float) * (1000.0 if built.units == "m" else 1.0)
        triangles = np.concatenate([block.data for block in mesh.cells if block.type == "triangle"])
        tag_blocks = mesh.cell_data.get("gmsh:physical", [])
        tags = np.concatenate([tag_blocks[i] for i, block in enumerate(mesh.cells) if block.type == "triangle"])
        regions = {}
        for tag in np.unique(tags):
            faces = triangles[tags == tag]
            regions[str(int(tag))] = _surface_stats(points, faces, None, offset)
        acoustic = None
        selection = None
        if reference is not None:
            node_distance, _ = _meridian_distance(points, reference, offset)
        elif surface_reference is not None:
            node_distance = _sampled_surface_distance(points, surface_reference)
        else:
            node_distance = None
        if node_distance is not None:
            # OCC's acoustic wall nodes lie on the fitted wall. Identify the
            # wall using all three nodes, not a physical tag that can also
            # contain the shell or baffle. This is an explicitly reported
            # selection, not a claim that every physical-group facet was tested.
            tolerance_mm = 0.05
            selected = np.all(node_distance[triangles] <= tolerance_mm, axis=1)
            selection = {
                "method": "all three nodes within tolerance of sampled wall",
                "node_tolerance_mm": tolerance_mm,
                "selected_triangles": int(np.count_nonzero(selected)),
                "unselected_triangles": int(np.count_nonzero(~selected)),
            }
            if np.any(selected):
                acoustic = _surface_stats(points, triangles[selected], reference, offset, surface_reference)
        result["solve"] = {
            "status": "ok",
            "build_ms": elapsed,
            "triangles": len(triangles),
            "physical_regions": regions,
            "acoustic_wall_selection": selection,
            "acoustic_wall": acoustic,
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="mesher JSON/TOML/ATH config files")
    parser.add_argument("--output", required=True, type=Path, help="JSON report")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--no-meridian", action="store_true", help="skip independently sampled meridians")
    args = parser.parse_args(argv)
    results = []
    failed = 0
    for index, path in enumerate(args.inputs):
        print(path.name, flush=True)
        try:
            result = _measure(_load(path), solve=not args.preview_only, meridian=not args.no_meridian)
            partial = result.get("solve", {}).get("status") == "error"
            failed += int(partial)
            if partial:
                print(f"  SOLVE ERROR: {result['solve']['error']}", file=sys.stderr, flush=True)
            results.append({"index": index, "name": path.name, "status": "partial" if partial else "ok", **result})
        except Exception as exc:  # noqa: BLE001 - report each failed case and return a failing exit status
            failed += 1
            results.append({"index": index, "name": path.name, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    report = {"format": "tessellation-report/1", "cases": results, "failures": failed}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False, default=_json_value) + "\n", encoding="utf-8")
    print(f"{len(results)} cases, {failed} failures", flush=True)
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
