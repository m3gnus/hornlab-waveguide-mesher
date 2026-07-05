"""Experimental cabinet compatibility bridge.

This module provides a compatibility shim for the deleted in-tree mesher
bridge import path used by downstream cabinet callers. It accepts the same
Waveguide Generator-style payload dictionaries, but it uses the standalone
mesher's native Python point-grid and Gmsh builders.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping

import meshio
import numpy as np

from hornlab_mesher.config_builder import _mesh_report, _number_list
from hornlab_mesher.cost import estimate_solve_cost, worst_valid_f_max_hz
from hornlab_mesher.geometry import HornEnclosure, MeshDensity, PointGridHornGeometry
from hornlab_mesher.mesher import build_mesh_with_info
from hornlab_mesher.viewport import build_viewport_geometry_from_config

HORNLAB_MESHER_AVAILABLE = True


def _clean_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _number(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if np.isfinite(out) else fallback


def _first_number(value: Any, fallback: float = 0.0) -> float:
    numbers = _number_list(value)
    return float(numbers[0]) if numbers else float(fallback)


def _normalize_formula(value: Any) -> str:
    raw = str(value or "R-OSSE").strip().upper().replace("_", "-")
    if raw == "ROSSE":
        return "R-OSSE"
    if raw not in {"OSSE", "R-OSSE", "LOOKUP"}:
        raise ValueError(
            f"formula_type '{value}' is not supported for formula-only cabinet builds. "
            "Provide inner_points/grid_n_phi/grid_n_length for precomputed payloads."
        )
    return raw


def _normalize_source_shape(value: Any) -> Any:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return value
    # WG legacy: 1 = rounded cap, 2 = flat disc. Mesher: 1 = rounded, 0 = flat.
    return 0 if numeric == 2 else numeric


def waveguide_payload_to_mesher_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a WG-style request payload into mesher public config."""

    formula = _normalize_formula(payload.get("formula_type") or payload.get("formula"))
    profile: dict[str, Any] = _clean_dict(
        {
            "formula": formula,
            "r0": payload.get("r0"),
            "a": payload.get("a"),
            "a0": payload.get("a0"),
            "k": payload.get("k"),
            "q": payload.get("q"),
            "throatExtLength": payload.get("throat_ext_length"),
            "throatExtAngle": payload.get("throat_ext_angle"),
            "slotLength": payload.get("slot_length"),
        }
    )
    if formula == "OSSE":
        profile.update(
            _clean_dict(
                {
                    "L": payload.get("L"),
                    "n": payload.get("n"),
                    "s": payload.get("s"),
                    "h": payload.get("h"),
                    "rot": payload.get("rot"),
                    "throatProfile": payload.get("throat_profile"),
                    "circArcRadius": payload.get("circ_arc_radius"),
                    "circArcTermAngle": payload.get("circ_arc_term_angle"),
                }
            )
        )
    elif formula == "LOOKUP":
        # LOOKUP carries a precomputed [z, r] profile (PCHIP-fit upstream)
        # instead of analytic formula coefficients.
        profile.update(
            _clean_dict(
                {
                    "lookupProfile": payload.get("lookup_profile")
                    or payload.get("lookupProfile"),
                }
            )
        )
    else:
        profile.update(
            _clean_dict(
                {
                    "R": payload.get("R"),
                    "r": payload.get("r"),
                    "b": payload.get("b"),
                    "m": payload.get("m"),
                    "tmax": payload.get("tmax"),
                }
            )
        )

    raw_cross_section = payload.get("cross_section")
    if isinstance(raw_cross_section, Mapping):
        cross_section = {
            "exponent": _number(
                raw_cross_section.get(
                    "exponent",
                    raw_cross_section.get("cross_section_exponent", 2.0),
                ),
                2.0,
            ),
            "aspectRatio": _number(
                raw_cross_section.get(
                    "aspectRatio",
                    raw_cross_section.get("aspect_ratio", 1.0),
                ),
                1.0,
            ),
        }
    else:
        cross_section = {"exponent": 2.0, "aspectRatio": 1.0}

    enc_depth = _number(payload.get("enc_depth"), 0.0)
    config: dict[str, Any] = {
        "formula": formula,
        "mode": "enclosure" if enc_depth > 0.0 else "freestanding",
        "profile": profile,
        "mesh": _clean_dict(
            {
                "angularSegments": payload.get("n_angular"),
                "lengthSegments": payload.get("n_length"),
                "cornerSegments": payload.get("corner_segments"),
                "quadrants": payload.get("quadrants", 1234),
                "wallThickness": payload.get("wall_thickness"),
                "throatResolution": payload.get("throat_res"),
                "mouthResolution": payload.get("mouth_res"),
                "rearResolution": payload.get("rear_res"),
                "encFrontResolution": _first_number(payload.get("enc_front_resolution")),
                "encBackResolution": _first_number(payload.get("enc_back_resolution")),
            }
        ),
        "cross_section": cross_section,
        "morph": _clean_dict(
            {
                "morphTarget": payload.get("morph_target"),
                "morphWidth": payload.get("morph_width"),
                "morphHeight": payload.get("morph_height"),
                "morphCorner": payload.get("morph_corner"),
                "morphRate": payload.get("morph_rate"),
                "morphFixed": payload.get("morph_fixed"),
                "morphAllowShrinkage": payload.get("morph_allow_shrinkage"),
            }
        ),
        "gcurve": _clean_dict(
            {
                "gcurveType": payload.get("gcurve_type"),
                "gcurveWidth": payload.get("gcurve_width"),
                "gcurveAspectRatio": payload.get("gcurve_aspect_ratio"),
                "gcurveDist": payload.get("gcurve_dist"),
                "gcurveRot": payload.get("gcurve_rot"),
                "gcurveSf": payload.get("gcurve_sf"),
                "gcurveSeN": payload.get("gcurve_se_n"),
                "gcurveSfA": payload.get("gcurve_sf_a"),
                "gcurveSfB": payload.get("gcurve_sf_b"),
                "gcurveSfM1": payload.get("gcurve_sf_m1"),
                "gcurveSfM2": payload.get("gcurve_sf_m2"),
                "gcurveSfN1": payload.get("gcurve_sf_n1"),
                "gcurveSfN2": payload.get("gcurve_sf_n2"),
                "gcurveSfN3": payload.get("gcurve_sf_n3"),
            }
        ),
        "source": _clean_dict(
            {
                "sourceShape": _normalize_source_shape(payload.get("source_shape")),
                "sourceRadius": payload.get("source_radius"),
                "sourceCurv": payload.get("source_curv"),
            }
        ),
    }

    if enc_depth > 0.0:
        config["enclosure"] = _clean_dict(
            {
                "depth": enc_depth,
                "space_l": payload.get("enc_space_l"),
                "space_t": payload.get("enc_space_t"),
                "space_r": payload.get("enc_space_r"),
                "space_b": payload.get("enc_space_b"),
                "edge": payload.get("enc_edge"),
                "edgeType": payload.get("enc_edge_type"),
                "planType": payload.get("enc_plan_type"),
                "planN": payload.get("enc_plan_n"),
                "depth_margin": payload.get("enc_depth_margin"),
                "frontMeshSize": _first_number(payload.get("enc_front_resolution")),
                "backMeshSize": _first_number(payload.get("enc_back_resolution")),
            }
        )

    return config


def _payload_has_point_grid(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("inner_points") is not None
        and payload.get("grid_n_phi") is not None
        and payload.get("grid_n_length") is not None
    )


def _reshape_grid(raw: Any, n_phi: int, n_length: int, name: str) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    expected = int(n_phi) * (int(n_length) + 1) * 3
    if arr.size != expected:
        raise ValueError(f"{name} has {arr.size} values; expected {expected}.")
    return arr.reshape(int(n_phi), int(n_length) + 1, 3)


def _enclosure_from_payload(payload: Mapping[str, Any]) -> HornEnclosure | None:
    depth = _number(payload.get("enc_depth"), 0.0)
    if depth <= 0.0:
        return None
    return HornEnclosure(
        depth_mm=depth,
        space_l_mm=_number(payload.get("enc_space_l"), 25.0),
        space_t_mm=_number(payload.get("enc_space_t"), 25.0),
        space_r_mm=_number(payload.get("enc_space_r"), 25.0),
        space_b_mm=_number(payload.get("enc_space_b"), 25.0),
        edge_mm=_number(payload.get("enc_edge"), 18.0),
        edge_type=int(_number(payload.get("enc_edge_type"), 1.0)),
        plan_type=int(_number(payload.get("enc_plan_type"), 1.0)),
        plan_n=_number(payload.get("enc_plan_n"), 2.0),
        depth_margin_mm=_number(payload.get("enc_depth_margin"), 1.0),
        front_mesh_size_mm=_first_number(payload.get("enc_front_resolution")),
        back_mesh_size_mm=_first_number(payload.get("enc_back_resolution")),
    )


def _density_from_payload(payload: Mapping[str, Any]) -> MeshDensity:
    return MeshDensity(
        throat_res_mm=_number(payload.get("throat_res"), 5.0),
        mouth_res_mm=_number(payload.get("mouth_res"), 8.0),
        rear_res_mm=_number(payload.get("rear_res"), 25.0),
        enc_front_res_mm=payload.get("enc_front_resolution"),
        enc_back_res_mm=payload.get("enc_back_resolution"),
    )


def _apply_mouth_scaling(
    inner_points: np.ndarray,
    outer_points: np.ndarray | None,
    h_target: float,
    v_target: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    if h_target <= 0.0 and v_target <= 0.0:
        return inner_points, outer_points

    mouth = inner_points[:, -1, :]
    current_h = float(np.ptp(mouth[:, 0]))
    current_v = float(np.ptp(mouth[:, 1]))
    scale_h = h_target / current_h if h_target > 0.0 and current_h > 0.0 else 1.0
    scale_v = v_target / current_v if v_target > 0.0 and current_v > 0.0 else 1.0

    z_min = float(np.min(inner_points[..., 2]))
    z_max = float(np.max(inner_points[..., 2]))
    z_span = z_max - z_min
    if z_span <= 1.0e-12:
        return inner_points, outer_points

    def scale(points: np.ndarray) -> np.ndarray:
        out = np.array(points, dtype=np.float64, copy=True)
        t = np.clip((out[..., 2] - z_min) / z_span, 0.0, 1.0)
        out[..., 0] *= 1.0 + t * (scale_h - 1.0)
        out[..., 1] *= 1.0 + t * (scale_v - 1.0)
        return out

    return scale(inner_points), scale(outer_points) if outer_points is not None else None


def _grid_from_payload(
    payload: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray | None, bool, tuple[str, ...], float]:
    """Return (inner, outer, closed, symmetry_planes, vertical_offset_mm).

    Symmetry planes and the vertical offset used to be dropped here, so an
    open half grid got the default ("x", "y") planes (crushing the 180-degree
    ray onto the axis) and Mesh.VerticalOffset never reached the build.
    """
    if _payload_has_point_grid(payload):
        n_phi = int(payload["grid_n_phi"])
        n_length = int(payload["grid_n_length"])
        inner_points = _reshape_grid(payload["inner_points"], n_phi, n_length, "inner_points")
        outer_points = None
        if payload.get("outer_points") is not None:
            outer_points = _reshape_grid(payload["outer_points"], n_phi, n_length, "outer_points")
        closed = bool(payload.get("full_circle", payload.get("grid_closed", True)))
        raw_planes = payload.get("symmetry_planes")
        if raw_planes is not None:
            planes = tuple(str(p) for p in raw_planes)
        elif payload.get("quadrants") is not None:
            planes = _symmetry_planes_for_quadrants(payload.get("quadrants"))
        else:
            planes = ("x", "y") if not closed else ()
        offset = _number(payload.get("vertical_offset"), 0.0)
        return inner_points, outer_points, closed, planes, offset

    config = waveguide_payload_to_mesher_config(payload)
    viewport = build_viewport_geometry_from_config(config)
    grid = viewport.get("grid") or {}
    n_phi = int(grid.get("grid_n_phi") or 0)
    n_length = int(grid.get("grid_n_length") or 0)
    inner_points = _reshape_grid(grid.get("inner_points"), n_phi, n_length, "inner_points")
    outer_points = None
    if grid.get("outer_points") is not None:
        outer_points = _reshape_grid(grid.get("outer_points"), n_phi, n_length, "outer_points")
    planes = tuple(str(p) for p in (grid.get("symmetry_planes") or ()))
    offset = _number(grid.get("vertical_offset_mm"), 0.0)
    return inner_points, outer_points, bool(grid.get("full_circle", True)), planes, offset


def _geometry_from_payload(payload: Mapping[str, Any]) -> PointGridHornGeometry:
    inner_points, outer_points, closed, symmetry_planes, vertical_offset_mm = _grid_from_payload(payload)
    inner_points, outer_points = _apply_mouth_scaling(
        inner_points,
        outer_points,
        _number(payload.get("h_scale_target"), 0.0),
        _number(payload.get("v_scale_target"), 0.0),
    )
    enclosure = _enclosure_from_payload(payload)
    if enclosure is not None:
        outer_points = None
    return PointGridHornGeometry(
        inner_points=inner_points,
        outer_points=outer_points,
        wall_thickness_mm=_number(payload.get("wall_thickness"), 6.0),
        source_shape=int(_number(_normalize_source_shape(payload.get("source_shape")), 1.0)),
        source_radius_mm=_number(payload.get("source_radius"), -1.0),
        source_curv=int(_number(payload.get("source_curv"), 0.0)),
        source_auto_angle_deg=_finite_float_or_none(payload.get("a0")),
        preserve_grid=bool(payload.get("grid_preserve_rings", False)),
        closed=closed,
        symmetry_planes=symmetry_planes,
        vertical_offset_mm=vertical_offset_mm,
        enclosure=enclosure,
    )


def _finite_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _triangles_and_tags(mesh: meshio.Mesh) -> tuple[np.ndarray, np.ndarray]:
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
            tags.append(np.ones(len(cell_block.data), dtype=np.int32))
    if not triangles:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int32)
    return np.vstack(triangles), np.concatenate(tags)


def _canonical_mesh_from_msh(path: Path) -> dict[str, Any]:
    mesh = meshio.read(path)
    triangles, tags = _triangles_and_tags(mesh)
    tag_counts = {
        str(int(tag)): int(np.count_nonzero(tags == int(tag)))
        for tag in sorted({int(tag) for tag in tags.tolist()})
    }
    return {
        "vertices": np.asarray(mesh.points, dtype=np.float64).reshape(-1).tolist(),
        "indices": triangles.reshape(-1).astype(int).tolist(),
        "surfaceTags": tags.astype(int).tolist(),
        "metadata": {
            "units": "m" if _looks_like_metres(np.asarray(mesh.points, dtype=np.float64)) else "mm",
            "unitScaleToMeter": 1.0 if _looks_like_metres(np.asarray(mesh.points, dtype=np.float64)) else 0.001,
            "tagCounts": tag_counts,
            "generatedBy": "hornlab-waveguide-mesher",
        },
    }


def _looks_like_metres(points: np.ndarray) -> bool:
    span = np.max(points, axis=0) - np.min(points, axis=0)
    return bool(np.max(span) < 10.0)


def _load_payload(params: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(params, (str, Path)):
        return json.loads(Path(params).read_text(encoding="utf-8"))
    return dict(params)


def measure_horn_mouth(params: Mapping[str, Any] | str | Path) -> tuple[float, float]:
    """Return post-scaling mouth width and height in millimetres."""

    payload = _load_payload(params)
    inner_points, outer_points, _closed, _planes, _voffset = _grid_from_payload(payload)
    inner_points, _outer_points = _apply_mouth_scaling(
        inner_points,
        outer_points,
        _number(payload.get("h_scale_target"), 0.0),
        _number(payload.get("v_scale_target"), 0.0),
    )
    mouth = inner_points[:, -1, :]
    return (float(np.ptp(mouth[:, 0])), float(np.ptp(mouth[:, 1])))


def build_mesh_via_hornlab(
    payload: Mapping[str, Any],
    *,
    include_canonical: bool = False,
    cancellation_callback: Any | None = None,
) -> dict[str, Any]:
    """Build a ``.msh`` through the standalone mesher experimental cabinet path."""

    if cancellation_callback is not None:
        cancellation_callback()
    geometry = _geometry_from_payload(payload)
    density = _density_from_payload(payload)

    with tempfile.TemporaryDirectory(prefix="hornlab-cabinet-") as tmp_dir:
        mesh_path = Path(tmp_dir) / "cabinet.msh"
        out_path, info = build_mesh_with_info(
            geometry,
            density,
            mesh_path,
            # Legacy consumers (downstream BEM/solver pipelines) load
            # .msh files with a 0.001 scale factor, so this bridge preserves
            # the old bridge's millimetre output contract. Callers can opt
            # into metres by passing scale_to_metres=true in the payload.
            scale_to_metres=bool(payload.get("scale_to_metres", False)),
        )
        if cancellation_callback is not None:
            cancellation_callback()
        msh_text = out_path.read_text(encoding="utf-8", errors="replace")
        canonical = _canonical_mesh_from_msh(out_path)

    mesh_report = _mesh_report(info.physical_groups, info.edge_stats_mm, density)
    stats = {
        "nodeCount": int(info.n_vertices),
        "elementCount": int(info.n_triangles),
        "tagCounts": canonical["metadata"]["tagCounts"],
        "units": info.units,
        "source": "hornlab_waveguide_mesher_experimental_cabinet",
        "generatedBy": "hornlab-waveguide-mesher",
        # Mesh validity + dense-BEM solve cost, so downstream callers get the
        # same size/cost/trustworthy-band forecast as the build_from_config path.
        "meshReport": mesh_report,
        "validFreqMaxHz": worst_valid_f_max_hz(mesh_report),
        "solveCost": estimate_solve_cost(info.n_triangles).to_dict(),
    }
    result: dict[str, Any] = {"msh_text": msh_text, "stats": stats}
    if include_canonical:
        result["canonical_mesh"] = canonical
    return result


def build_horn_in_box_mesh(
    params: Mapping[str, Any] | str | Path,
    out_path: str | Path,
    *,
    verbose: bool = True,
) -> Path:
    """Build a horn-in-cabinet mesh and write the ``.msh`` text to *out_path*."""

    payload = _load_payload(params)
    result = build_mesh_via_hornlab(payload)
    path = Path(out_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result["msh_text"], encoding="utf-8")
    if verbose:
        stats = result["stats"]
        print(
            f"[experimental.cabinet] wrote {path} "
            f"({stats.get('nodeCount', '?')} nodes, "
            f"{stats.get('elementCount', '?')} elements)",
            flush=True,
        )
    return path


__all__ = [
    "HORNLAB_MESHER_AVAILABLE",
    "build_horn_in_box_mesh",
    "build_mesh_via_hornlab",
    "measure_horn_mouth",
    "waveguide_payload_to_mesher_config",
]
