from __future__ import annotations

"""Config normalization and config-driven mesh build orchestration.

This module owns the conversion from external TOML/JSON/imported ATH config
names into profile parameters, `PointGridHornGeometry`, `MeshDensity`, and the
final `BuildResult`. The CLI imports these helpers but does not own this
translation layer.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config_parser import ConfigError
from .geometry import HornEnclosure, HornInterface, MeshDensity, PointGridHornGeometry
from .mesher import build_mesh, load_mesh
from .profiles import build_point_grid, eval_param


@dataclass(frozen=True)
class BuildResult:
    mesh_path: Path
    formula: str
    mode: str
    n_vertices: int
    n_triangles: int
    units: str
    physical_groups: dict[int, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mesh_path": str(self.mesh_path),
            "formula": self.formula,
            "mode": self.mode,
            "n_vertices": self.n_vertices,
            "n_triangles": self.n_triangles,
            "units": self.units,
            "physical_groups": {str(k): v for k, v in self.physical_groups.items()},
        }


def _number_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            try:
                number = float(item)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                out.append(number)
        return out
    if isinstance(value, (int, float)):
        number = float(value)
        return [number] if np.isfinite(number) else []
    text = str(value).strip()
    if not text:
        return []
    parts = [part.strip() for part in text.split(",")]
    out = []
    for part in parts:
        if not part:
            continue
        try:
            number = float(part)
        except ValueError:
            continue
        if np.isfinite(number):
            out.append(number)
    return out


def _first_number(*sources: Mapping[str, Any], names: tuple[str, ...], default: float) -> float:
    value = _pick(*sources, names=names, default=default)
    numbers = _number_list(value)
    return float(numbers[0]) if numbers else float(default)


def _section(config: Mapping[str, Any], *names: str) -> Mapping[str, Any]:
    for name in names:
        value = config.get(name)
        if isinstance(value, Mapping):
            return value
    return {}


def _pick(*sources: Mapping[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for source in sources:
        for name in names:
            if name in source and source[name] is not None:
                return source[name]
    return default


def _float(*sources: Mapping[str, Any], names: tuple[str, ...], default: float) -> float:
    value = _pick(*sources, names=names, default=default)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{names[0]} must be numeric, got {value!r}") from exc
    if not np.isfinite(out):
        raise ConfigError(f"{names[0]} must be finite, got {value!r}")
    return out


def _scalar_or_expr(*sources: Mapping[str, Any], names: tuple[str, ...], default: Any) -> Any:
    value = _pick(*sources, names=names, default=default)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            out = float(stripped)
        except ValueError:
            return stripped
        return int(out) if out.is_integer() else out
    return value


def _int(*sources: Mapping[str, Any], names: tuple[str, ...], default: int) -> int:
    value = _pick(*sources, names=names, default=default)
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{names[0]} must be an integer, got {value!r}") from exc
    return out


def _bool(*sources: Mapping[str, Any], names: tuple[str, ...], default: bool) -> bool:
    value = _pick(*sources, names=names, default=default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _normalise_formula(value: Any) -> str:
    raw = str(value or "OSSE").strip().upper().replace("_", "-")
    if raw == "ROSSE":
        raw = "R-OSSE"
    if raw not in {"OSSE", "R-OSSE"}:
        raise ConfigError(f"formula must be OSSE or R-OSSE/ROSSE, got {value!r}")
    return raw


def _normalise_mode(config: Mapping[str, Any], mesh: Mapping[str, Any], enclosure: Mapping[str, Any]) -> str:
    raw = str(_pick(config, mesh, names=("mode",), default="")).strip().lower().replace("_", "-")
    enc_depth = _float(enclosure, mesh, config, names=("depth_mm", "depth", "encDepth"), default=0.0)
    if raw in {"enclosure", "enclosed"} or enc_depth > 0:
        return "enclosure"
    if raw in {"bare", "inner", "open"}:
        return "bare"
    if raw in {"", "free-standing", "freestanding", "free"}:
        return "freestanding"
    raise ConfigError(f"mode must be freestanding, enclosure, or bare, got {raw!r}")


def _enclosure_from_config(
    config: Mapping[str, Any],
    mesh: Mapping[str, Any],
    enclosure: Mapping[str, Any],
) -> HornEnclosure | None:
    depth = _float(enclosure, mesh, config, names=("depth_mm", "depth", "encDepth"), default=0.0)
    if depth <= 0.0:
        return None
    return HornEnclosure(
        depth_mm=depth,
        space_l_mm=_float(enclosure, names=("space_l_mm", "space_l", "left_margin_mm"), default=25.0),
        space_t_mm=_float(enclosure, names=("space_t_mm", "space_t", "top_margin_mm"), default=25.0),
        space_r_mm=_float(enclosure, names=("space_r_mm", "space_r", "right_margin_mm"), default=25.0),
        space_b_mm=_float(enclosure, names=("space_b_mm", "space_b", "bottom_margin_mm"), default=25.0),
        edge_mm=_float(enclosure, names=("edge_mm", "edge", "encEdge"), default=18.0),
        edge_type=_int(enclosure, names=("edge_type", "edgeType", "encEdgeType"), default=1),
        plan_type=_int(enclosure, names=("plan_type", "planType", "encPlanType"), default=1),
        plan_n=_float(enclosure, names=("plan_n", "planN", "encPlanN"), default=2.0),
        depth_margin_mm=_float(enclosure, names=("depth_margin_mm", "depth_margin", "encDepthMargin"), default=1.0),
        front_mesh_size_mm=_first_number(
            enclosure,
            names=("front_mesh_size_mm", "frontMeshSize", "enc_front_resolution", "encFrontResolution"),
            default=0.0,
        ),
        back_mesh_size_mm=_first_number(
            enclosure,
            names=("back_mesh_size_mm", "backMeshSize", "enc_back_resolution", "encBackResolution"),
            default=0.0,
        ),
    )


def build_geometry_params(config: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
    profile = _section(config, "profile", "parameters")
    mesh = _section(config, "mesh")
    enclosure = _section(config, "enclosure")
    cross = _section(config, "cross_section", "crossSection")
    morph = _section(config, "morph", "MORPH")
    gcurve = _section(config, "gcurve", "GCurve", "GCURVE")
    source = _section(config, "source", "Source")

    formula = _normalise_formula(_pick(config, profile, names=("formula", "type"), default="OSSE"))
    mode = _normalise_mode(config, mesh, enclosure)
    enc_depth = 0.0
    enclosure_obj = _enclosure_from_config(config, mesh, enclosure)
    if enclosure_obj is not None:
        enc_depth = enclosure_obj.depth_mm
    elif mode == "enclosure":
        raise ConfigError("enclosure mode requires enclosure.depth_mm > 0")

    default_wall = 0.0 if mode in {"bare", "enclosure"} else 6.0
    wall_thickness = _float(
        mesh,
        config,
        names=("wall_thickness_mm", "wall_thickness", "wallThickness"),
        default=default_wall,
    )
    if mode == "bare":
        wall_thickness = 0.0
    if mode == "enclosure":
        wall_thickness = 0.0
    z_map_points = _pick(mesh, config, names=("z_map_points", "zMapPoints", "zmapPoints", "ZMapPoints"), default=None)
    default_sampling_mode = "zmap" if z_map_points is not None else "uniform"

    common: dict[str, Any] = {
        "type": formula,
        "r0": _scalar_or_expr(profile, config, names=("r0_mm", "r0"), default=12.7),
        "a": _scalar_or_expr(profile, config, names=("a_deg", "a"), default=60.0),
        "a0": _scalar_or_expr(profile, config, names=("a0_deg", "a0"), default=15.5),
        "k": _scalar_or_expr(profile, config, names=("k",), default=1.0),
        "q": _scalar_or_expr(profile, config, names=("q",), default=1.0 if formula == "R-OSSE" else 0.995),
        "angularSegments": _int(mesh, config, names=("angular_segments", "angularSegments"), default=64),
        "cornerSegments": _int(mesh, config, names=("corner_segments", "cornerSegments"), default=0),
        "lengthSegments": _int(mesh, config, names=("length_segments", "lengthSegments"), default=32),
        "samplingMode": _pick(
            mesh,
            config,
            names=("sampling_mode", "samplingMode"),
            default=default_sampling_mode,
        ),
        "athParitySampling": _bool(
            mesh,
            config,
            names=("ath_parity_sampling", "athParitySampling"),
            default=False,
        ),
        "zMapPoints": z_map_points,
        "wallThickness": wall_thickness,
        "encDepth": enc_depth,
        "morphTarget": _scalar_or_expr(morph, config, names=("morph_target", "morphTarget"), default=0),
        "morphWidth": _scalar_or_expr(morph, config, names=("morph_width_mm", "morphWidth"), default=0),
        "morphHeight": _scalar_or_expr(morph, config, names=("morph_height_mm", "morphHeight"), default=0),
        "morphCorner": _scalar_or_expr(morph, config, names=("morph_corner_mm", "morphCorner"), default=0),
        "morphRate": _scalar_or_expr(morph, config, names=("morph_rate", "morphRate"), default=3.0),
        "morphFixed": _scalar_or_expr(morph, config, names=("morph_fixed", "morphFixed"), default=0),
        "morphAllowShrinkage": _scalar_or_expr(
            morph, config, names=("morph_allow_shrinkage", "morphAllowShrinkage"), default=0
        ),
        "gcurveType": _scalar_or_expr(gcurve, config, names=("gcurve_type", "gcurveType"), default=0),
        "gcurveWidth": _scalar_or_expr(gcurve, config, names=("gcurve_width_mm", "gcurveWidth"), default=0),
        "gcurveAspectRatio": _scalar_or_expr(
            gcurve, config, names=("gcurve_aspect_ratio", "gcurveAspectRatio"), default=1
        ),
        "gcurveDist": _scalar_or_expr(gcurve, config, names=("gcurve_dist", "gcurveDist"), default=0),
        "gcurveRot": _scalar_or_expr(gcurve, config, names=("gcurve_rot_deg", "gcurveRot"), default=0),
        "gcurveSF": _pick(gcurve, config, names=("gcurve_sf", "gcurveSf", "gcurveSF"), default=""),
        "gcurveSf": _pick(gcurve, config, names=("gcurve_sf", "gcurveSf", "gcurveSF"), default=""),
        "gcurveSeN": _scalar_or_expr(gcurve, config, names=("gcurve_se_n", "gcurveSeN"), default=3),
        "gcurveSfA": _scalar_or_expr(gcurve, config, names=("gcurve_sf_a", "gcurveSfA"), default=1),
        "gcurveSfB": _scalar_or_expr(gcurve, config, names=("gcurve_sf_b", "gcurveSfB"), default=1),
        "gcurveSfM1": _scalar_or_expr(gcurve, config, names=("gcurve_sf_m1", "gcurveSfM1"), default=4),
        "gcurveSfM2": _scalar_or_expr(gcurve, config, names=("gcurve_sf_m2", "gcurveSfM2"), default=None),
        "gcurveSfN1": _scalar_or_expr(gcurve, config, names=("gcurve_sf_n1", "gcurveSfN1"), default=2),
        "gcurveSfN2": _scalar_or_expr(gcurve, config, names=("gcurve_sf_n2", "gcurveSfN2"), default=2),
        "gcurveSfN3": _scalar_or_expr(gcurve, config, names=("gcurve_sf_n3", "gcurveSfN3"), default=2),
        "quadrants": str(_pick(mesh, config, names=("quadrants",), default="1234")),
        "throatResolution": _float(mesh, config, names=("throat_res_mm", "throatResolution"), default=4.0),
        "mouthResolution": _float(mesh, config, names=("mouth_res_mm", "mouthResolution"), default=26.0),
        "rearResolution": _float(mesh, config, names=("rear_res_mm", "rearResolution"), default=25.0),
        "subdomainSlices": _scalar_or_expr(mesh, config, names=("subdomain_slices", "subdomainSlices"), default=""),
        "interfaceOffset": _scalar_or_expr(mesh, config, names=("interface_offset_mm", "interfaceOffset"), default=0.0),
        "interfaceResolution": _float(
            mesh,
            config,
            names=("interface_res_mm", "interfaceResolution"),
            default=12.0,
        ),
        "sourceShape": _scalar_or_expr(source, config, names=("source_shape", "sourceShape"), default=1),
        "sourceRadius": _scalar_or_expr(source, config, names=("source_radius_mm", "sourceRadius"), default=-1),
        "sourceCurv": _scalar_or_expr(source, config, names=("source_curv", "sourceCurv"), default=0),
        "profileSystem": {
            "crossSection": {
                "exponent": _float(cross, profile, config, names=("exponent", "cross_section_exponent"), default=2.0),
                "aspectRatio": _float(cross, profile, config, names=("aspect_ratio", "aspectRatio"), default=1.0),
            },
        },
    }

    if formula == "OSSE":
        common.update(
            {
                "L": _scalar_or_expr(profile, config, names=("L_mm", "L"), default=120.0),
                "n": _scalar_or_expr(profile, config, names=("n",), default=4.0),
                "s": _scalar_or_expr(profile, config, names=("s",), default=0.0),
                "throatExtLength": _scalar_or_expr(
                    profile, config, names=("throat_ext_length_mm", "throatExtLength"), default=0.0
                ),
                "throatExtAngle": _scalar_or_expr(
                    profile, config, names=("throat_ext_angle_deg", "throatExtAngle"), default=0.0
                ),
                "slotLength": _scalar_or_expr(profile, config, names=("slot_length_mm", "slotLength"), default=0.0),
                "rot": _scalar_or_expr(profile, config, names=("rot_deg", "rot"), default=0.0),
            }
        )
    else:
        common.update(
            {
                "R": _scalar_or_expr(profile, config, names=("R_mm", "R"), default=150.0),
                "tmax": _scalar_or_expr(profile, config, names=("tmax",), default=1.0),
            }
        )
        for key in ("m", "r", "b"):
            value = _pick(profile, config, names=(key,), default=None)
            if value is not None:
                common[key] = _scalar_or_expr(profile, config, names=(key,), default=None)

    return common, formula, mode


def _interfaces_from_params(params: Mapping[str, Any], n_length: int) -> tuple[HornInterface, ...]:
    offsets = _number_list(params.get("interfaceOffset"))
    if not offsets:
        return ()

    slices = [int(round(value)) for value in _number_list(params.get("subdomainSlices"))]
    if not slices:
        return ()

    interfaces: list[HornInterface] = []
    last_ring = int(n_length)
    for slice_index, offset in zip(slices, offsets):
        if offset <= 0.0:
            continue
        # Imported text configs address grid slices; keep valid indices and ignore
        # out-of-range declarations rather than guessing a different topology.
        if 0 <= int(slice_index) <= last_ring:
            interfaces.append(HornInterface(slice_index=int(slice_index), offset_mm=float(offset)))
    return tuple(interfaces)


def _reshape_grid(raw: Any, n_phi: int, n_length: int, name: str) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    expected = n_phi * (n_length + 1) * 3
    if arr.size != expected:
        raise ConfigError(f"{name} has {arr.size} values; expected {expected}")
    return arr.reshape(n_phi, n_length + 1, 3)


def build_from_config(
    config: Mapping[str, Any],
    output_path: str | Path,
) -> BuildResult:
    params, formula, mode = build_geometry_params(config)
    mesh = _section(config, "mesh")
    enclosure = _section(config, "enclosure")
    enclosure_obj = _enclosure_from_config(config, mesh, enclosure)

    grid = build_point_grid(params)

    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    inner_points = _reshape_grid(grid["inner_points"], n_phi, n_length, "inner_points")
    outer_points = None
    if grid.get("outer_points") is not None and enclosure_obj is None:
        outer_points = _reshape_grid(grid["outer_points"], n_phi, n_length, "outer_points")

    interface_offsets = _number_list(params.get("interfaceOffset"))
    geometry = PointGridHornGeometry(
        inner_points=inner_points,
        outer_points=outer_points,
        wall_thickness_mm=float(params["wallThickness"] or 0.0),
        preserve_grid=_bool(mesh, names=("preserve_grid", "preserveGrid"), default=False),
        closed=bool(grid.get("full_circle", True)),
        source_shape=int(float(params.get("sourceShape", 1) or 1)),
        source_radius_mm=float(params.get("sourceRadius", -1) or -1),
        source_curv=int(float(params.get("sourceCurv", 0) or 0)),
        source_auto_angle_deg=float(eval_param(params.get("a0"), 0.0, 15.5)),
        interface_offset_mm=float(interface_offsets[0] if interface_offsets else 0.0),
        interfaces=_interfaces_from_params(params, n_length),
        enclosure=enclosure_obj,
    )
    density = MeshDensity(
        throat_res_mm=_float(mesh, names=("throat_res_mm", "throat_res", "throatResolution"), default=4.0),
        mouth_res_mm=_float(mesh, names=("mouth_res_mm", "mouth_res", "mouthResolution"), default=26.0),
        rear_res_mm=_float(mesh, names=("rear_res_mm", "rear_res", "rearResolution"), default=25.0),
        enc_front_res_mm=_pick(
            mesh,
            enclosure,
            names=("enc_front_res_mm", "enc_front_resolution", "encFrontResolution"),
            default=None,
        ),
        enc_back_res_mm=_pick(
            mesh,
            enclosure,
            names=("enc_back_res_mm", "enc_back_resolution", "encBackResolution"),
            default=None,
        ),
        interface_res_mm=_float(mesh, names=("interface_res_mm", "interface_res", "interfaceResolution"), default=12.0),
    )
    scale_to_metres = _bool(mesh, names=("scale_to_metres", "scaleToMetres"), default=True)
    mesh_path = build_mesh(geometry, density, output_path, scale_to_metres=scale_to_metres)
    info = load_mesh(mesh_path)
    return BuildResult(
        mesh_path=mesh_path,
        formula=formula,
        mode=mode,
        n_vertices=info.n_vertices,
        n_triangles=info.n_triangles,
        units=info.units,
        physical_groups=info.physical_groups,
    )


__all__ = [
    "BuildResult",
    "build_from_config",
    "build_geometry_params",
    "_bool",
    "_enclosure_from_config",
    "_first_number",
    "_float",
    "_int",
    "_interfaces_from_params",
    "_normalise_formula",
    "_normalise_mode",
    "_number_list",
    "_pick",
    "_reshape_grid",
    "_scalar_or_expr",
    "_section",
]
