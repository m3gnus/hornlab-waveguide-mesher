from __future__ import annotations

"""Config normalization and config-driven mesh build orchestration.

This module owns the conversion from external TOML/JSON/imported ATH config
names into profile parameters, `PointGridHornGeometry`, `MeshDensity`, and the
final `BuildResult`. The CLI imports these helpers but does not own this
translation layer.
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import cost
from .config_parser import ConfigError
from .geometry import (
    HornEnclosure,
    HornInterface,
    MeshDensity,
    PointGridHornGeometry,
    validate_mesh_density,
)
from .mesher import build_mesh_with_info
from .profile_common import (
    _normalise_quadrants as _normalise_quadrants_common,
    _parse_number_list,
    _symmetry_planes_for_quadrants as _symmetry_planes_for_quadrants_common,
)
from .profiles import build_point_grid, eval_param, profile_points
from .builders.point_grid_freestanding import (
    _outer_wall_axial_ring_indices,
    _restored_outer_throat_points,
)
from .builders.point_grid_sources import (
    SOURCE_SHAPE_ROUNDED_CAP,
    _source_cap_height,
    _source_cap_radius,
)
from .builders.point_grid_surfaces import _rear_rim_points
from .tags import PhysicalGroup

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildResult:
    mesh_path: Path
    formula: str
    mode: str
    n_vertices: int
    n_triangles: int
    units: str
    physical_groups: dict[int, str]
    # Quadrant coverage of the built grid and the solver symmetry flag a
    # reduced mesh requires (hornlab-metal-bem SolveConfig.native_symmetry_plane).
    quadrants: str = "1234"
    native_symmetry_plane: str | None = None
    # Whether the metal solver's cut-plane open-edge guard applies. Bare horns
    # are open shells whose free mouth rims are legitimate solve boundaries, so
    # the guard must be relaxed (hornlab-metal-bem
    # SolveConfig.native_check_open_edges=False). Closed/coupled modes cap the
    # mouth and keep the strict check.
    native_check_open_edges: bool = True
    # Realized per-group geometric edge statistics in millimetres.
    mesh_report: dict[str, dict[str, float]] = field(default_factory=dict)
    # Dense-BEM cost for the realized triangle count. See hornlab_mesher.cost.
    solve_cost: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mesh_path": str(self.mesh_path),
            "formula": self.formula,
            "mode": self.mode,
            "n_vertices": self.n_vertices,
            "n_triangles": self.n_triangles,
            "units": self.units,
            "physical_groups": {str(k): v for k, v in self.physical_groups.items()},
            "quadrants": self.quadrants,
            "native_symmetry_plane": self.native_symmetry_plane,
            "native_check_open_edges": self.native_check_open_edges,
            "mesh_report": self.mesh_report,
            "solve_cost": self.solve_cost,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class MeridianBuildResult:
    """CircSym meridian arrays derived from the public mesher config.

    Coordinates are in metres and follow hornlab-metal-bem's MeridianMesh
    contract: nodes are ``(rho, z)``, segments index adjacent nodes,
    ``physical_tags`` is one tag per segment, and normals are cylindrical
    ``(n_rho, n_z)``.
    """

    nodes: np.ndarray
    segments: np.ndarray
    physical_tags: np.ndarray
    normals: np.ndarray
    baffle_z: float | None
    formula: str
    mode: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_metal_meridian(self, meridian_cls: Any | None = None) -> Any:
        """Materialise a hornlab-metal-bem MeridianMesh from these arrays."""

        if meridian_cls is None:
            try:
                from hornlab_metal_bem import MeridianMesh as meridian_cls
            except ImportError as exc:  # pragma: no cover - dependency-owned
                raise RuntimeError(
                    "hornlab-metal-bem is required to materialise a CircSym MeridianMesh."
                ) from exc
        return meridian_cls(
            nodes=self.nodes,
            segments=self.segments,
            physical_tags=self.physical_tags,
            normals=self.normals,
        )


def _number_list(value: Any) -> list[float]:
    return _parse_number_list(
        value,
        allow_scalar=True,
        finite_only=True,
        invalid="skip",
        evaluate=False,
    )


def _first_number(
    *sources: Mapping[str, Any], names: tuple[str, ...], default: float
) -> float:
    value = _pick(*sources, names=names, default=default)
    numbers = _number_list(value)
    return float(numbers[0]) if numbers else float(default)


def _section(config: Mapping[str, Any], *names: str) -> Mapping[str, Any]:
    for name in names:
        value = config.get(name)
        if isinstance(value, Mapping):
            return value
    return {}


def _pick(
    *sources: Mapping[str, Any], names: tuple[str, ...], default: Any = None
) -> Any:
    for source in sources:
        for name in names:
            if name in source and source[name] is not None:
                return source[name]
    return default


def _float(
    *sources: Mapping[str, Any], names: tuple[str, ...], default: float
) -> float:
    value = _pick(*sources, names=names, default=default)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{names[0]} must be numeric, got {value!r}") from exc
    if not np.isfinite(out):
        raise ConfigError(f"{names[0]} must be finite, got {value!r}")
    return out


def _optional_float(
    *sources: Mapping[str, Any], names: tuple[str, ...]
) -> float | None:
    value = _pick(*sources, names=names, default=None)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{names[0]} must be numeric, got {value!r}") from exc
    if not np.isfinite(out):
        raise ConfigError(f"{names[0]} must be finite, got {value!r}")
    return out


def _numeric_param(value: Any, *, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"{name} must be numeric when deriving a driver adapter, got {value!r}"
        ) from exc
    if not np.isfinite(out):
        raise ConfigError(f"{name} must be finite, got {value!r}")
    return out


def _scalar_or_expr(
    *sources: Mapping[str, Any], names: tuple[str, ...], default: Any
) -> Any:
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
    if raw not in {"OSSE", "R-OSSE", "LOOKUP", "ICW"}:
        raise ConfigError(
            f"formula must be OSSE, R-OSSE/ROSSE, LOOKUP, or ICW, got {value!r}"
        )
    return raw


def _has_any(*sources: Mapping[str, Any], names: tuple[str, ...]) -> bool:
    for source in sources:
        for name in names:
            if name in source and source[name] is not None:
                return True
    return False


def _validate_formula_specific_keys(
    formula: str,
    profile: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    if formula == "OSSE":
        names = ("R_mm", "R", "tmax", "m", "r", "b")
        if _has_any(profile, config, names=names):
            raise ConfigError(
                "R-OSSE-only profile keys are not valid with formula OSSE"
            )
        return

    if formula == "LOOKUP":
        # LOOKUP carries only a precomputed profile; analytic coefficients of
        # either formula family are out of place.
        names = (
            "L_mm",
            "L",
            "n",
            "s",
            "rot_deg",
            "rot",
            "R_mm",
            "R",
            "tmax",
            "m",
            "r",
            "b",
        )
        if _has_any(profile, config, names=names):
            raise ConfigError("formula LOOKUP does not accept OSSE/R-OSSE profile keys")
        return

    if formula == "ICW":
        # ICW accepts its own intrinsic-curvature keys (throat r0/a0, targets
        # L/R/theta1/x_aperture/depth/x_setback, kappa0, n_coeff, termination,
        # and the seed/direct inputs). Both L and R are legitimate ICW size
        # targets, so unlike OSSE/R-OSSE neither is rejected here. OSSE-only
        # shape coefficients (n, s, rot) and R-OSSE-only shape coefficients
        # (m, r, b, tmax) have no meaning on an ICW curve and are rejected at
        # the TOP LEVEL -- they may still appear nested inside icw_seed (a
        # separate OSSE/R-OSSE profile dict), which _has_any does not scan.
        # Coverage/manufacturability keys (coverage_angle, hold_*, kappa_abs_max,
        # dkappa_ds_abs_max, theta_max_deg, pin_mouth_radius) are valid ICW
        # top-level keys and are intentionally not part of this reject set.
        names = ("n", "s", "rot_deg", "rot", "m", "r", "b", "tmax")
        if _has_any(profile, config, names=names):
            raise ConfigError("OSSE/R-OSSE shape keys are not valid with formula ICW")
        return

    names = ("L_mm", "L", "n", "s", "rot_deg", "rot")
    if _has_any(profile, config, names=names):
        raise ConfigError("OSSE-only profile keys are not valid with formula R-OSSE")


def _static_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        raise ConfigError(f"numeric config value must be finite, got {value!r}")
    return out


def _gcurve_type_width(
    gcurve: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[int | None, float | None]:
    raw_type = _pick(gcurve, config, names=("gcurve_type", "gcurveType"), default=0)
    raw_width = _pick(
        gcurve, config, names=("gcurve_width_mm", "gcurveWidth"), default=0
    )
    type_value = _static_float_or_none(raw_type)
    width = _static_float_or_none(raw_width)
    curve_type = int(round(type_value)) if type_value is not None else None
    return curve_type, width


def _has_gcurve_keys(gcurve: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    names = (
        "gcurve_type",
        "gcurveType",
        "gcurve_width_mm",
        "gcurveWidth",
        "gcurve_aspect_ratio",
        "gcurveAspectRatio",
        "gcurve_dist",
        "gcurveDist",
        "gcurve_rot_deg",
        "gcurveRot",
        "gcurve_sf",
        "gcurveSf",
        "gcurveSF",
        "gcurve_se_n",
        "gcurveSeN",
        "gcurve_sf_a",
        "gcurveSfA",
        "gcurve_sf_b",
        "gcurveSfB",
        "gcurve_sf_m1",
        "gcurveSfM1",
        "gcurve_sf_m2",
        "gcurveSfM2",
        "gcurve_sf_n1",
        "gcurveSfN1",
        "gcurve_sf_n2",
        "gcurveSfN2",
        "gcurve_sf_n3",
        "gcurveSfN3",
    )
    return _has_any(gcurve, config, names=names)


def _gcurve_could_be_active(
    gcurve: Mapping[str, Any], config: Mapping[str, Any]
) -> bool:
    curve_type, width = _gcurve_type_width(gcurve, config)
    if curve_type is None or width is None:
        return _has_gcurve_keys(gcurve, config)
    return curve_type in {1, 2} and width > 0.0


def _validate_static_gcurve_type(
    gcurve: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    curve_type, _width = _gcurve_type_width(gcurve, config)
    if curve_type is not None and curve_type not in {0, 1, 2}:
        raise ConfigError(f"unsupported GCurve type {curve_type}")


def _validate_formula_features(
    formula: str,
    gcurve: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    _validate_static_gcurve_type(gcurve, config)
    if formula != "OSSE" and _gcurve_could_be_active(gcurve, config):
        raise ConfigError("guiding curves are only supported with formula OSSE")


def _enc_depth_mm(
    config: Mapping[str, Any],
    mesh: Mapping[str, Any],
    enclosure: Mapping[str, Any],
    formula: str,
) -> float:
    """Resolve the enclosure depth (mm); 0.0 means "no enclosure".

    A bare ``depth`` is read straight from the ``enclosure`` / ``mesh`` sections, where it is
    unambiguous. At the TOP LEVEL of the config a bare ``depth`` is NOT treated as enclosure depth
    for an ICW profile -- there it is the rollback axial target (a profile param), and letting it
    trip enclosure mode silently wrapped a free-standing rollback ICW in a box. ICW enclosures must
    therefore name the depth explicitly (``encDepth`` / ``depth_mm``) or nest it under the
    ``enclosure`` section; other formulas keep the historical top-level bare-``depth`` fallback.
    """
    sectioned = _optional_float(
        enclosure, mesh, names=("depth_mm", "depth", "encDepth")
    )
    if sectioned is not None:
        return sectioned
    top_names = (
        ("depth_mm", "encDepth")
        if formula == "ICW"
        else ("depth_mm", "depth", "encDepth")
    )
    return _float(config, names=top_names, default=0.0)


def _normalise_mode(
    config: Mapping[str, Any],
    mesh: Mapping[str, Any],
    enclosure: Mapping[str, Any],
    formula: str = "OSSE",
) -> str:
    raw = (
        str(_pick(config, mesh, names=("mode",), default=""))
        .strip()
        .lower()
        .replace("_", "-")
    )
    enc_depth = _enc_depth_mm(config, mesh, enclosure, formula)
    if raw in {"enclosure", "enclosed"}:
        return "enclosure"
    if enc_depth > 0:
        if raw == "":
            # An enclosure depth implies enclosure mode when no mode is given.
            return "enclosure"
        raise ConfigError(
            f"mode {raw!r} contradicts the configured enclosure depth {enc_depth:g} mm; "
            "drop the enclosure or use mode='enclosure'"
        )
    if raw in {"bare", "inner", "open"}:
        return "bare"
    if raw in {"infinite-baffle", "infinitebaffle", "ib", "baffle"}:
        return "infinite-baffle"
    if raw == "":
        # Imported ATH text configs carry ABEC.SimType (1 = infinite baffle,
        # 2 = free standing); native configs without a mode stay freestanding.
        sim_type = _pick(config, mesh, names=("simType", "sim_type"), default=None)
        if sim_type is not None:
            try:
                sim_int = int(float(sim_type))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"simType must be 1 or 2, got {sim_type!r}") from exc
            if sim_int == 1:
                return "infinite-baffle"
            if sim_int == 2:
                return "freestanding"
            raise ConfigError(f"simType must be 1 or 2, got {sim_type!r}")
        return "freestanding"
    if raw in {"free-standing", "freestanding", "free"}:
        return "freestanding"
    raise ConfigError(
        f"mode must be freestanding, enclosure, bare, or infinite-baffle, got {raw!r}"
    )


def _enclosure_from_config(
    config: Mapping[str, Any],
    mesh: Mapping[str, Any],
    enclosure: Mapping[str, Any],
    formula: str = "OSSE",
) -> HornEnclosure | None:
    depth = _enc_depth_mm(config, mesh, enclosure, formula)
    if depth <= 0.0:
        return None
    return HornEnclosure(
        depth_mm=depth,
        space_l_mm=_float(
            enclosure, names=("space_l_mm", "space_l", "left_margin_mm"), default=25.0
        ),
        space_t_mm=_float(
            enclosure, names=("space_t_mm", "space_t", "top_margin_mm"), default=25.0
        ),
        space_r_mm=_float(
            enclosure, names=("space_r_mm", "space_r", "right_margin_mm"), default=25.0
        ),
        space_b_mm=_float(
            enclosure, names=("space_b_mm", "space_b", "bottom_margin_mm"), default=25.0
        ),
        edge_mm=_float(enclosure, names=("edge_mm", "edge", "encEdge"), default=18.0),
        edge_type=_int(
            enclosure, names=("edge_type", "edgeType", "encEdgeType"), default=1
        ),
        plan_type=_int(
            enclosure, names=("plan_type", "planType", "encPlanType"), default=1
        ),
        plan_n=_float(enclosure, names=("plan_n", "planN", "encPlanN"), default=2.0),
        depth_margin_mm=_float(
            enclosure,
            names=("depth_margin_mm", "depth_margin", "encDepthMargin"),
            default=1.0,
        ),
        front_mesh_size_mm=_first_number(
            enclosure,
            names=(
                "front_mesh_size_mm",
                "frontMeshSize",
                "enc_front_resolution",
                "encFrontResolution",
            ),
            default=0.0,
        ),
        back_mesh_size_mm=_first_number(
            enclosure,
            names=(
                "back_mesh_size_mm",
                "backMeshSize",
                "enc_back_resolution",
                "encBackResolution",
            ),
            default=0.0,
        ),
    )


def _diameter_radius_mm(
    *sources: Mapping[str, Any],
    mm_names: tuple[str, ...],
    inch_names: tuple[str, ...],
) -> float | None:
    diameter_mm = _optional_float(*sources, names=mm_names)
    if diameter_mm is not None:
        return 0.5 * diameter_mm
    diameter_in = _optional_float(*sources, names=inch_names)
    if diameter_in is not None:
        return 0.5 * diameter_in * 25.4
    return None


def _apply_driver_adapter(
    common: dict[str, Any],
    profile: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    driver_radius = _diameter_radius_mm(
        profile,
        config,
        mm_names=(
            "driver_throat_diameter_mm",
            "driver_throat_diameter",
            "driverThroatDiameterMm",
            "driverThroatDiameter",
        ),
        inch_names=("driver_throat_diameter_in", "driverThroatDiameterIn"),
    )
    waveguide_radius = _diameter_radius_mm(
        profile,
        config,
        mm_names=(
            "waveguide_throat_diameter_mm",
            "waveguide_throat_diameter",
            "waveguideThroatDiameterMm",
            "waveguideThroatDiameter",
        ),
        inch_names=("waveguide_throat_diameter_in", "waveguideThroatDiameterIn"),
    )
    if driver_radius is None and waveguide_radius is None:
        return
    if driver_radius is None or waveguide_radius is None:
        raise ConfigError(
            "driver adapter requires both driver and waveguide throat diameters"
        )
    if driver_radius <= 0.0 or waveguide_radius <= 0.0:
        raise ConfigError("driver and waveguide throat diameters must be > 0")
    if waveguide_radius < driver_radius:
        raise ConfigError(
            "driver adapter cannot shrink from waveguide throat to driver throat"
        )

    delta_radius = waveguide_radius - driver_radius
    # Both formulas anchor r0 (Throat.Diameter) at the MAIN waveguide throat
    # and taper the extension BACK to the driver end (r0 - ext*tan == driver
    # radius) — the ATH convention. Setting r0 to the driver radius here (the
    # old forward-expansion assumption) built a horn whose requested waveguide
    # throat diameter appeared nowhere in the geometry.
    common["r0"] = waveguide_radius
    if delta_radius <= 1.0e-12:
        common["throatExtLength"] = 0.0
        common["throatExtAngle"] = 0.0
        return

    ext_len = _numeric_param(common.get("throatExtLength", 0.0), name="throatExtLength")
    ext_angle_deg = _numeric_param(
        common.get("throatExtAngle", 0.0), name="throatExtAngle"
    )
    if ext_len <= 0.0 and abs(ext_angle_deg) <= 1.0e-12:
        raise ConfigError("driver adapter requires throatExtLength or throatExtAngle")
    if ext_len <= 0.0:
        tan_angle = np.tan(np.deg2rad(ext_angle_deg))
        if tan_angle <= 1.0e-12:
            raise ConfigError(
                "throatExtAngle must be > 0 when deriving driver adapter length"
            )
        common["throatExtLength"] = delta_radius / tan_angle
        return
    if abs(ext_angle_deg) <= 1.0e-12:
        common["throatExtAngle"] = float(np.rad2deg(np.arctan(delta_radius / ext_len)))
        return

    derived_radius = driver_radius + ext_len * np.tan(np.deg2rad(ext_angle_deg))
    if abs(derived_radius - waveguide_radius) > 1.0e-6:
        raise ConfigError(
            "driver adapter throatExtLength/throatExtAngle do not reach waveguide throat diameter"
        )


def _param_is_nonzero(value: Any, *, name: str) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    try:
        out = float(value)
    except (TypeError, ValueError):
        return True
    if not np.isfinite(out):
        raise ConfigError(f"{name} must be finite, got {value!r}")
    return abs(out) > 1.0e-12


def _reject_icw_throat_extension(common: Mapping[str, Any]) -> None:
    names = [
        key
        for key in ("throatExtLength", "throatExtAngle")
        if _param_is_nonzero(common.get(key), name=key)
    ]
    if names:
        joined = "/".join(names)
        raise ConfigError(f"formula ICW does not support throat extension ({joined})")


def build_geometry_params(config: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
    profile = _section(config, "profile", "parameters")
    mesh = _section(config, "mesh")
    enclosure = _section(config, "enclosure")
    cross = _section(config, "cross_section", "crossSection")
    morph = _section(config, "morph", "MORPH")
    gcurve = _section(config, "gcurve", "GCurve", "GCURVE")
    source = _section(config, "source", "Source")

    formula = _normalise_formula(
        _pick(config, profile, names=("formula", "type"), default="OSSE")
    )
    _validate_formula_specific_keys(formula, profile, config)
    _validate_formula_features(formula, gcurve, config)
    mode = _normalise_mode(config, mesh, enclosure, formula)
    enc_depth = 0.0
    enclosure_obj = _enclosure_from_config(config, mesh, enclosure, formula)
    if enclosure_obj is not None:
        enc_depth = enclosure_obj.depth_mm
    elif mode == "enclosure":
        raise ConfigError("enclosure mode requires enclosure.depth_mm > 0")

    default_wall = 0.0 if mode in {"bare", "enclosure", "infinite-baffle"} else 6.0
    wall_thickness = _float(
        mesh,
        config,
        names=("wall_thickness_mm", "wall_thickness", "wallThickness"),
        default=default_wall,
    )
    if mode in {"bare", "enclosure", "infinite-baffle"}:
        wall_thickness = 0.0
    z_map_points = _pick(
        mesh,
        config,
        names=("z_map_points", "zMapPoints", "zmapPoints", "ZMapPoints"),
        default=None,
    )
    default_sampling_mode = "zmap" if z_map_points is not None else "uniform"

    common: dict[str, Any] = {
        "type": formula,
        "lookupProfile": _pick(
            profile, config, names=("lookupProfile", "lookup_profile"), default=None
        ),
        "r0": _scalar_or_expr(profile, config, names=("r0_mm", "r0"), default=12.7),
        "a": _scalar_or_expr(profile, config, names=("a_deg", "a"), default=60.0),
        "a0": _scalar_or_expr(profile, config, names=("a0_deg", "a0"), default=15.5),
        "k": _scalar_or_expr(profile, config, names=("k",), default=1.0),
        "q": _scalar_or_expr(
            profile, config, names=("q",), default=1.0 if formula == "R-OSSE" else 0.995
        ),
        "throatExtLength": _scalar_or_expr(
            profile,
            config,
            names=("throat_ext_length_mm", "throatExtLength"),
            default=0.0,
        ),
        "throatExtAngle": _scalar_or_expr(
            profile,
            config,
            names=("throat_ext_angle_deg", "throatExtAngle"),
            default=0.0,
        ),
        "slotLength": _scalar_or_expr(
            profile, config, names=("slot_length_mm", "slotLength"), default=0.0
        ),
        "angularSegments": _int(
            mesh, config, names=("angular_segments", "angularSegments"), default=64
        ),
        "cornerSegments": _int(
            mesh, config, names=("corner_segments", "cornerSegments"), default=0
        ),
        "lengthSegments": _int(
            mesh, config, names=("length_segments", "lengthSegments"), default=32
        ),
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
        "morphTarget": _scalar_or_expr(
            morph, config, names=("morph_target", "morphTarget"), default=0
        ),
        "morphWidth": _scalar_or_expr(
            morph, config, names=("morph_width_mm", "morphWidth"), default=0
        ),
        "morphHeight": _scalar_or_expr(
            morph, config, names=("morph_height_mm", "morphHeight"), default=0
        ),
        "morphCorner": _scalar_or_expr(
            morph, config, names=("morph_corner_mm", "morphCorner"), default=0
        ),
        "morphRate": _scalar_or_expr(
            morph, config, names=("morph_rate", "morphRate"), default=3.0
        ),
        "morphFixed": _scalar_or_expr(
            morph, config, names=("morph_fixed", "morphFixed"), default=0
        ),
        "morphAllowShrinkage": _scalar_or_expr(
            morph,
            config,
            names=("morph_allow_shrinkage", "morphAllowShrinkage"),
            default=0,
        ),
        "gcurveType": _scalar_or_expr(
            gcurve, config, names=("gcurve_type", "gcurveType"), default=0
        ),
        "gcurveWidth": _scalar_or_expr(
            gcurve, config, names=("gcurve_width_mm", "gcurveWidth"), default=0
        ),
        "gcurveAspectRatio": _scalar_or_expr(
            gcurve,
            config,
            names=("gcurve_aspect_ratio", "gcurveAspectRatio"),
            default=1,
        ),
        "gcurveDist": _scalar_or_expr(
            gcurve, config, names=("gcurve_dist", "gcurveDist"), default=0
        ),
        "gcurveRot": _scalar_or_expr(
            gcurve, config, names=("gcurve_rot_deg", "gcurveRot"), default=0
        ),
        "gcurveSF": _pick(
            gcurve, config, names=("gcurve_sf", "gcurveSf", "gcurveSF"), default=""
        ),
        "gcurveSf": _pick(
            gcurve, config, names=("gcurve_sf", "gcurveSf", "gcurveSF"), default=""
        ),
        "gcurveSeN": _scalar_or_expr(
            gcurve, config, names=("gcurve_se_n", "gcurveSeN"), default=3
        ),
        "gcurveSfA": _scalar_or_expr(
            gcurve, config, names=("gcurve_sf_a", "gcurveSfA"), default=1
        ),
        "gcurveSfB": _scalar_or_expr(
            gcurve, config, names=("gcurve_sf_b", "gcurveSfB"), default=1
        ),
        "gcurveSfM1": _scalar_or_expr(
            gcurve, config, names=("gcurve_sf_m1", "gcurveSfM1"), default=4
        ),
        "gcurveSfM2": _scalar_or_expr(
            gcurve, config, names=("gcurve_sf_m2", "gcurveSfM2"), default=None
        ),
        "gcurveSfN1": _scalar_or_expr(
            gcurve, config, names=("gcurve_sf_n1", "gcurveSfN1"), default=2
        ),
        "gcurveSfN2": _scalar_or_expr(
            gcurve, config, names=("gcurve_sf_n2", "gcurveSfN2"), default=2
        ),
        "gcurveSfN3": _scalar_or_expr(
            gcurve, config, names=("gcurve_sf_n3", "gcurveSfN3"), default=2
        ),
        "quadrants": _normalised_quadrants(
            _pick(mesh, config, names=("quadrants",), default="1234")
        ),
        "scale": _float(config, profile, names=("scale", "Scale"), default=1.0),
        "verticalOffset": _float(
            mesh, config, names=("vertical_offset_mm", "verticalOffset"), default=0.0
        ),
        "throatResolution": _float(
            mesh, config, names=("throat_res_mm", "throatResolution"), default=4.0
        ),
        "mouthResolution": _float(
            mesh, config, names=("mouth_res_mm", "mouthResolution"), default=26.0
        ),
        "rearResolution": _float(
            mesh, config, names=("rear_res_mm", "rearResolution"), default=15.0
        ),
        "subdomainSlices": _scalar_or_expr(
            mesh, config, names=("subdomain_slices", "subdomainSlices"), default=""
        ),
        # None (not 0.0) when omitted: an omitted offset with SubdomainSlices
        # set takes ATH's 5 mm default, while an explicit 0 disables interfaces.
        "interfaceOffset": _scalar_or_expr(
            mesh, config, names=("interface_offset_mm", "interfaceOffset"), default=None
        ),
        "interfaceResolution": _optional_float(
            mesh,
            config,
            names=("interface_res_mm", "interfaceResolution"),
        ),
        "sourceShape": _scalar_or_expr(
            source, config, names=("source_shape", "sourceShape"), default=1
        ),
        "sourceRadius": _scalar_or_expr(
            source, config, names=("source_radius_mm", "sourceRadius"), default=-1
        ),
        "sourceCurv": _scalar_or_expr(
            source, config, names=("source_curv", "sourceCurv"), default=0
        ),
        "profileSystem": {
            "crossSection": {
                "exponent": _float(
                    cross,
                    profile,
                    config,
                    names=("exponent", "cross_section_exponent"),
                    default=2.0,
                ),
                "aspectRatio": _float(
                    cross,
                    profile,
                    config,
                    names=("aspect_ratio", "aspectRatio"),
                    default=1.0,
                ),
            },
        },
    }
    length_mode = _pick(
        profile,
        config,
        names=("_athLengthMode", "athLengthMode", "length_mode", "lengthMode"),
        default=None,
    )
    if length_mode is not None:
        common["_athLengthMode"] = length_mode
    _apply_driver_adapter(common, profile, config)
    if formula == "ICW":
        _reject_icw_throat_extension(common)

    if formula == "OSSE":
        common.update(
            {
                "L": _scalar_or_expr(
                    profile, config, names=("L_mm", "L"), default=120.0
                ),
                "n": _scalar_or_expr(profile, config, names=("n",), default=4.0),
                "s": _scalar_or_expr(profile, config, names=("s",), default=0.0),
                "rot": _scalar_or_expr(
                    profile, config, names=("rot_deg", "rot"), default=0.0
                ),
            }
        )
    elif formula == "LOOKUP":
        # No analytic coefficients: the precomputed lookupProfile (threaded
        # into common above) fully defines the radial profile.
        pass
    elif formula == "ICW":
        # ICW reads its targets/seed straight off the params dict in
        # build_icw_curve. r0/a0 are already in ``common``; thread the rest
        # through. Only keys actually present are forwarded so the kernel keeps
        # applying its own defaults for the optional targets.
        common["termination"] = _pick(
            profile, config, names=("termination",), default="flat_baffle"
        )
        termination = str(common["termination"] or "flat_baffle").strip().lower()
        if termination == "flat_baffle":
            common["L"] = _scalar_or_expr(
                profile, config, names=("L_mm", "L"), default=120.0
            )
            common["R"] = _scalar_or_expr(
                profile, config, names=("R_mm", "R"), default=150.0
            )
        else:
            # Rollback reads R with r_aperture as a fallback; materialising the
            # flat-baffle defaults here would silently override an explicit
            # r_aperture with R=150. Thread L/R only when actually configured.
            for key, src_names in (("L", ("L_mm", "L")), ("R", ("R_mm", "R"))):
                if _pick(profile, config, names=src_names, default=None) is not None:
                    common[key] = _scalar_or_expr(
                        profile, config, names=src_names, default=None
                    )
        for key, src_names in (
            ("kappa0", ("kappa0",)),
            ("r_aperture", ("r_aperture",)),
            ("n_coeff", ("n_coeff",)),
            ("theta1", ("theta1_deg", "theta1")),
            ("x_aperture", ("x_aperture",)),
            ("depth", ("depth",)),
            ("x_setback", ("x_setback",)),
            ("coverage_angle", ("coverage_angle", "coverage_angle_deg")),
            ("hold_start", ("hold_start",)),
            ("hold_end", ("hold_end",)),
            ("kappa_abs_max", ("kappa_abs_max",)),
            ("dkappa_ds_abs_max", ("dkappa_ds_abs_max",)),
            ("theta_max_deg", ("theta_max_deg",)),
            ("icw_S", ("icw_S",)),
        ):
            value = _pick(profile, config, names=src_names, default=None)
            if value is not None:
                common[key] = _scalar_or_expr(
                    profile, config, names=src_names, default=None
                )
        pin_mouth_radius = _pick(
            profile, config, names=("pin_mouth_radius", "pinMouthRadius"), default=None
        )
        if pin_mouth_radius is not None:
            common["pin_mouth_radius"] = _bool(
                profile,
                config,
                names=("pin_mouth_radius", "pinMouthRadius"),
                default=False,
            )
        # Seed / direct-coefficient inputs are passed through verbatim (nested
        # dict / list), not coerced to scalars.
        seed = _pick(profile, config, names=("icw_seed",), default=None)
        if seed is not None:
            common["icw_seed"] = seed
        coeffs = _pick(profile, config, names=("icw_coeffs",), default=None)
        if coeffs is not None:
            common["icw_coeffs"] = coeffs
    else:
        common.update(
            {
                "R": _scalar_or_expr(
                    profile, config, names=("R_mm", "R"), default=150.0
                ),
                "tmax": _scalar_or_expr(profile, config, names=("tmax",), default=1.0),
            }
        )
        for key in ("m", "r", "b"):
            value = _pick(profile, config, names=(key,), default=None)
            if value is not None:
                common[key] = _scalar_or_expr(
                    profile, config, names=(key,), default=None
                )

    return common, formula, mode


def _normalised_quadrants(value: Any) -> str:
    """Normalise Mesh.Quadrants to Ath's canonical coverage ({1, 12, 14, 1234}).

    Delegates to the shared parser, which follows Ath's atoi rule: it reads a leading
    integer and maps 1234/12/14 to full/half-y/half-x and every other value —
    including junk, empties and permutations like "21" — to a quarter model. It never
    raises, matching Ath, which silently treats unrecognised values as the quarter
    default (a well-defined Q1 grid) rather than erroring or building the degenerate
    open full-circle grid this mesher's earlier set-based logic produced.
    """
    return _normalise_quadrants_common(value)


def _native_symmetry_plane_for_quadrants(quadrants: str) -> str | None:
    """Map grid quadrant coverage to the metal solver's symmetry-plane flag.

    Quadrant 1 spans x >= 0, y >= 0 and mirrors about both the yz and xz
    planes; "12" spans y >= 0 (xz mirror); "14" spans x >= 0 (yz mirror).
    """

    return {
        "1": "yz+xz",
        "12": "xz",
        "14": "yz",
    }.get(_normalised_quadrants(quadrants))


def _native_symmetry_plane_for_mode(mode: str, quadrants: str) -> str | None:
    return _native_symmetry_plane_for_quadrants(quadrants)


def _symmetry_planes_for_quadrants(quadrants: str) -> tuple[str, ...]:
    """Map grid quadrant coverage to open-grid snap axes.

    Mirrors :func:`_native_symmetry_plane_for_quadrants` but in the mesher's
    ``"x"``/``"y"`` snap-axis convention (``"x"`` is the x=0 / yz plane, ``"y"``
    is the y=0 / xz plane): quadrant 1 is bounded by both planes, "12" by the
    xz plane (snap ``"y"``), "14" by the yz plane (snap ``"x"``). Full coverage
    ("1234") returns no planes.
    """

    return _symmetry_planes_for_quadrants_common(quadrants)


def _native_check_open_edges_for_mode(mode: str) -> bool:
    """Whether the metal solver's cut-plane open-edge guard applies to ``mode``.

    ``bare`` horns radiate from an open mouth, so their free mouth rims are
    legitimate solve boundaries and the guard must be disabled. Coupled
    infinite-baffle, freestanding, and enclosure modes cap the mouth, so any
    reduced-domain open edges lie on cut planes and the strict check stays on.
    """

    return mode != "bare"


def _mesh_report(
    info_physical_groups: Mapping[int, str],
    edge_stats_mm: Mapping[int, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for tag, stats in edge_stats_mm.items():
        name = info_physical_groups.get(int(tag), str(tag))
        entry = {key: float(value) for key, value in stats.items()}
        report[name] = entry
    return report


_REMOVED_MESH_KEYS: dict[str, str] = {
    "max_frequency_hz": "use throat_res_mm, mouth_res_mm, and rear_res_mm",
    "maxFrequencyHz": "use throat_res_mm, mouth_res_mm, and rear_res_mm",
    "maxFrequency": "use throat_res_mm, mouth_res_mm, and rear_res_mm",
    "f_max_hz": "use throat_res_mm, mouth_res_mm, and rear_res_mm",
    "fMaxHz": "use throat_res_mm, mouth_res_mm, and rear_res_mm",
    "elements_per_wavelength": "use the semantic *_res_mm values",
    "elementsPerWavelength": "use the semantic *_res_mm values",
    "throat_epw": "use throat_res_mm",
    "throatEpw": "use throat_res_mm",
    "mouth_epw": "use mouth_res_mm",
    "mouthEpw": "use mouth_res_mm",
    "rear_epw": "use rear_res_mm",
    "rearEpw": "use rear_res_mm",
    "interface_epw": "use interface_res_mm",
    "interfaceEpw": "use interface_res_mm",
    "aperture_epw": "use aperture_res_scale and mouth_res_mm",
    "apertureEpw": "use aperture_res_scale and mouth_res_mm",
    "speed_of_sound_m_s": "mesh sizing is millimetre-only",
    "speedOfSound": "mesh sizing is millimetre-only",
    "curvature_segments": "curvature sizing is disabled; use the local *_res_mm values",
    "curvatureSegments": "curvature sizing is disabled; use the local *_res_mm values",
}


def _reject_removed_mesh_keys(
    config: Mapping[str, Any], mesh: Mapping[str, Any]
) -> None:
    for mapping in (mesh, config):
        for key, migration in _REMOVED_MESH_KEYS.items():
            if key in mapping:
                raise ConfigError(
                    f"mesh key {key!r} was removed by the millimetre-only mesh contract; "
                    f"{migration}"
                )


def _mesh_topology_mode(mesh: Mapping[str, Any]) -> str:
    raw = _pick(
        mesh, names=("topology", "topology_mode", "topologyMode"), default="acoustic"
    )
    mode = str(raw or "acoustic").strip().lower()
    if mode not in {"acoustic", "legacy"}:
        raise ConfigError("mesh topology must be 'acoustic' or 'legacy'")
    preserve = _bool(mesh, names=("preserve_grid", "preserveGrid"), default=False)
    if preserve and mode != "legacy":
        raise ConfigError(
            "preserve_grid pins sampled CAD faces and is only available with mesh.topology='legacy'"
        )
    return mode


def _mesh_density_from_config(
    config: Mapping[str, Any],
    *,
    allow_large_mesh: bool | None = None,
) -> MeshDensity:
    mesh = _section(config, "mesh")
    enclosure = _section(config, "enclosure")
    _reject_removed_mesh_keys(config, mesh)
    configured_allow_large = _bool(
        mesh,
        names=("allow_large_mesh", "allowLargeMesh"),
        default=False,
    )
    density = MeshDensity(
        throat_res_mm=_float(
            mesh,
            config,
            names=("throat_res_mm", "throat_res", "throatResolution"),
            default=4.0,
        ),
        mouth_res_mm=_float(
            mesh,
            config,
            names=("mouth_res_mm", "mouth_res", "mouthResolution"),
            default=26.0,
        ),
        rear_res_mm=_float(
            mesh,
            config,
            names=("rear_res_mm", "rear_res", "rearResolution"),
            default=15.0,
        ),
        aperture_res_scale=_float(
            mesh,
            config,
            names=(
                "aperture_res_scale",
                "apertureResolutionScale",
                "aperture_cap_coarsening",
                "apertureCapCoarsening",
            ),
            default=1.5,
        ),
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
        interface_res_mm=_optional_float(
            mesh,
            names=("interface_res_mm", "interface_res", "interfaceResolution"),
        ),
        max_triangles=_int(
            mesh,
            names=("max_triangles", "maxTriangles"),
            default=18_000,
        ),
        allow_large_mesh=(
            configured_allow_large
            if allow_large_mesh is None
            else bool(allow_large_mesh)
        ),
    )
    try:
        validate_mesh_density(density)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return density


def _num_or_default(value: Any, default: float) -> float:
    """Numeric param with an explicit unset check.

    ``0`` is a meaningful value for several params (``sourceShape = 0`` is the
    flat disc), so only ``None``/blank counts as unset — never falsiness.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return float(default)
    return float(value)


def _interfaces_from_params(
    params: Mapping[str, Any], n_length: int
) -> tuple[HornInterface, ...]:
    slices = [
        int(round(value)) for value in _number_list(params.get("subdomainSlices"))
    ]
    if not slices:
        return ()

    offsets = _number_list(params.get("interfaceOffset"))
    if not offsets:
        # ATH defaults Mesh.InterfaceOffset to 5 mm when SubdomainSlices are
        # set; an omitted offset used to silently drop the interfaces entirely.
        offsets = [5.0]
    if len(offsets) == 1 and len(slices) > 1:
        offsets = offsets * len(slices)
    if len(offsets) != len(slices):
        raise ConfigError(
            f"Mesh.SubdomainSlices lists {len(slices)} slices but Mesh.InterfaceOffset "
            f"lists {len(offsets)} offsets; give one offset or one per slice"
        )

    interfaces: list[HornInterface] = []
    last_ring = int(n_length)
    for slice_index, offset in zip(slices, offsets):
        if offset <= 0.0:
            continue
        # Imported text configs address grid slices; keep valid indices and ignore
        # out-of-range declarations rather than guessing a different topology.
        if 0 <= int(slice_index) <= last_ring:
            interfaces.append(
                HornInterface(slice_index=int(slice_index), offset_mm=float(offset))
            )
        else:
            logger.warning(
                "[hornlab-mesher] ignoring out-of-range Mesh.SubdomainSlices index %d "
                "(grid has rings 0..%d)",
                int(slice_index),
                last_ring,
            )
    return tuple(interfaces)


def _reshape_grid(raw: Any, n_phi: int, n_length: int, name: str) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    expected = n_phi * (n_length + 1) * 3
    if arr.size != expected:
        raise ConfigError(f"{name} has {arr.size} values; expected {expected}")
    return arr.reshape(n_phi, n_length + 1, 3)


def _axisymmetric_rejection_reasons(
    params: Mapping[str, Any],
    *,
    mode: str,
    enclosure_obj: HornEnclosure | None,
) -> list[str]:
    reasons: list[str] = []
    cross = params.get("profileSystem")
    cross_section = (
        cross.get("crossSection")
        if isinstance(cross, Mapping) and isinstance(cross.get("crossSection"), Mapping)
        else {}
    )
    exponent = float(cross_section.get("exponent", 2.0))
    aspect = float(cross_section.get("aspectRatio", 1.0))
    if not math.isclose(exponent, 2.0, rel_tol=0.0, abs_tol=1.0e-9):
        reasons.append(f"CrossSection exponent is {exponent:g}, not 2")
    if not math.isclose(aspect, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        reasons.append(f"CrossSection aspectRatio is {aspect:g}, not 1")

    morph_target = _static_float_or_none(params.get("morphTarget", 0))
    if morph_target is None or not math.isclose(
        morph_target, 0.0, rel_tol=0.0, abs_tol=1.0e-9
    ):
        reasons.append(f"morphTarget is {params.get('morphTarget')!r}, not 0")

    if enclosure_obj is not None or mode == "enclosure":
        depth = getattr(enclosure_obj, "depth_mm", params.get("encDepth", None))
        reasons.append(f"enclosure depth is {float(depth):g} mm")
    if _number_list(params.get("subdomainSlices")) or _number_list(
        params.get("interfaceOffset")
    ):
        reasons.append(
            "Mesh.SubdomainSlices/Mesh.InterfaceOffset subdomain interfaces are not "
            "implemented by the CircSym meridian builder"
        )
    return reasons


def _validate_mode_contract(params: Mapping[str, Any], mode: str) -> None:
    if mode == "freestanding":
        thickness = float(params.get("wallThickness") or 0.0)
        if not math.isfinite(thickness) or thickness <= 0.0:
            raise ConfigError(
                "freestanding mode requires Mesh.WallThickness/wall_thickness_mm > 0; "
                "use mode='bare' for an inner-only open horn"
            )


def _radial_profile_from_grid(points: np.ndarray, *, name: str) -> np.ndarray:
    radial = np.linalg.norm(points[:, :, :2], axis=2)
    z_values = points[:, :, 2]
    radial_span = np.ptp(radial, axis=0)
    z_span = np.ptp(z_values, axis=0)
    if np.any(radial_span > 1.0e-6) or np.any(z_span > 1.0e-6):
        max_radial = float(np.max(radial_span)) if radial_span.size else 0.0
        max_z = float(np.max(z_span)) if z_span.size else 0.0
        raise ConfigError(
            "CircSym requires a circular waveguide: "
            f"{name} varies with azimuth (max radius span {max_radial:.6g} mm, "
            f"max z span {max_z:.6g} mm)"
        )
    return np.column_stack((np.mean(radial, axis=0), np.mean(z_values, axis=0)))


def _append_meridian_points(
    nodes: list[tuple[float, float]],
    tags: list[int],
    points: np.ndarray,
    *,
    tag: int,
) -> int:
    added = 0
    for raw_point in np.asarray(points, dtype=np.float64):
        point = (float(raw_point[0]), float(raw_point[1]))
        if not nodes:
            nodes.append(point)
            continue
        prev = nodes[-1]
        if math.isclose(
            prev[0], point[0], rel_tol=0.0, abs_tol=1.0e-9
        ) and math.isclose(prev[1], point[1], rel_tol=0.0, abs_tol=1.0e-9):
            continue
        nodes.append(point)
        tags.append(int(tag))
        added += 1
    return added


def _subdivision_count_between(
    start: tuple[float, float],
    end: tuple[float, float],
    target_segment_mm: float,
) -> int:
    distance = math.hypot(
        float(end[0]) - float(start[0]), float(end[1]) - float(start[1])
    )
    if distance <= 1.0e-9:
        return 0
    return max(1, int(math.ceil(distance / max(float(target_segment_mm), 1.0e-9))))


def _append_subdivided_meridian_points(
    nodes: list[tuple[float, float]],
    tags: list[int],
    points: np.ndarray,
    *,
    tag: int,
    target_segment_mm: float,
) -> int:
    added = 0
    for raw_point in np.asarray(points, dtype=np.float64):
        point = (float(raw_point[0]), float(raw_point[1]))
        if not nodes:
            nodes.append(point)
            continue
        start = nodes[-1]
        n_segments = _subdivision_count_between(start, point, target_segment_mm)
        if n_segments == 0:
            continue
        for step in range(1, n_segments + 1):
            alpha = step / n_segments
            interp = (
                float(start[0] + alpha * (point[0] - start[0])),
                float(start[1] + alpha * (point[1] - start[1])),
            )
            prev = nodes[-1]
            if math.isclose(
                prev[0], interp[0], rel_tol=0.0, abs_tol=1.0e-9
            ) and math.isclose(prev[1], interp[1], rel_tol=0.0, abs_tol=1.0e-9):
                continue
            nodes.append(interp)
            tags.append(int(tag))
            added += 1
    return added


def _semicircular_meridian_join_points(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    incoming_tangent: tuple[float, float],
    target_segment_mm: float,
    minimum_segments: int = 2,
) -> np.ndarray:
    """ATH-compatible rounded closure between inner and outer mouth points.

    The free-standing CircSym body closes its wall thickness with the
    semicircle whose diameter is the inner-to-outer mouth chord.  Choose the
    half-circle that continues the incoming inner-wall tangent, then retain
    ATH's five-segment minimum while refining further when the frequency
    budget requires it.
    """

    p0 = np.asarray(start, dtype=np.float64)
    p1 = np.asarray(end, dtype=np.float64)
    midpoint = 0.5 * (p0 + p1)
    radius_vector = p0 - midpoint
    radius = float(np.linalg.norm(radius_vector))
    if radius <= 1.0e-12:
        return np.asarray([p0, p1], dtype=np.float64)

    tangent_vector = np.asarray(incoming_tangent, dtype=np.float64)
    arc_direction = np.asarray(
        [radius_vector[1], -radius_vector[0]],
        dtype=np.float64,
    )
    if float(np.dot(arc_direction, tangent_vector)) < 0.0:
        arc_direction *= -1.0

    arc_length = math.pi * radius
    segment_count = max(
        int(minimum_segments),
        int(math.ceil(arc_length / max(float(target_segment_mm), 1.0e-9))),
    )
    angles = np.linspace(0.0, math.pi, segment_count + 1, dtype=np.float64)
    points = (
        midpoint[None, :]
        + np.cos(angles)[:, None] * radius_vector[None, :]
        + np.sin(angles)[:, None] * arc_direction[None, :]
    )
    points[0] = p0
    points[-1] = p1
    return points


def _profile_arc_length_mm(params: Mapping[str, Any]) -> float:
    base_segments = max(64, int(params.get("lengthSegments") or 32))
    sample_count = max(513, min(4097, 8 * base_segments + 1))
    profile = np.asarray(
        profile_points(params, sample_count, phi=0.0), dtype=np.float64
    )
    if profile.ndim != 2 or profile.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(profile, axis=0), axis=1)))


def _resample_polyline_by_arc(points: np.ndarray, segment_count: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return pts.copy()
    count = max(1, int(segment_count))
    lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(np.sum(lengths))
    if total <= 1.0e-12:
        return np.vstack([pts[0], pts[-1]])
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    samples = np.linspace(0.0, total, count + 1, dtype=np.float64)
    out = np.empty((count + 1, pts.shape[1]), dtype=np.float64)
    for axis in range(pts.shape[1]):
        out[:, axis] = np.interp(samples, cumulative, pts[:, axis])
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def _resample_polyline_by_local_size(
    points: np.ndarray,
    start_size_mm: float,
    end_size_mm: float,
) -> np.ndarray:
    """Arc-length resample a polyline using a linearly graded mm target."""

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return pts.copy()
    lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(np.sum(lengths))
    if total <= 1.0e-12:
        return np.vstack([pts[0], pts[-1]])
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    t = cumulative / total
    sizes = float(start_size_mm) + (float(end_size_mm) - float(start_size_mm)) * t
    sizes = np.maximum(sizes, 1.0e-9)
    density_coordinate = np.concatenate(
        [[0.0], np.cumsum(lengths * 0.5 * (1.0 / sizes[:-1] + 1.0 / sizes[1:]))]
    )
    segment_count = max(1, int(math.ceil(float(density_coordinate[-1]))))
    samples = np.linspace(0.0, float(density_coordinate[-1]), segment_count + 1)
    arc_samples = np.interp(samples, density_coordinate, cumulative)
    out = np.empty((segment_count + 1, pts.shape[1]), dtype=np.float64)
    for axis in range(pts.shape[1]):
        out[:, axis] = np.interp(arc_samples, cumulative, pts[:, axis])
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def _circsym_resolution_budget(
    params: Mapping[str, Any],
    density: MeshDensity,
) -> dict[str, float | int]:
    arc_length_mm = _profile_arc_length_mm(params)
    finest = min(float(density.throat_res_mm), float(density.mouth_res_mm))
    # Internal geometry sampling is deliberately denser than the requested
    # final meridian. It improves profile fitting without creating BEM nodes.
    geometry_segments = max(
        64,
        min(4096, int(math.ceil(4.0 * arc_length_mm / max(finest, 1.0e-9)))),
    )
    return {
        "throat_target_mm": float(density.throat_res_mm),
        "mouth_target_mm": float(density.mouth_res_mm),
        "outer_target_mm": float(density.rear_res_mm),
        "aperture_target_mm": float(density.mouth_res_mm * density.aperture_res_scale),
        "profile_arc_length_mm": float(arc_length_mm),
        "geometry_segments": int(geometry_segments),
    }


def _source_cap_polyline(
    *,
    throat_radius_mm: float,
    throat_z_mm: float,
    geometry: PointGridHornGeometry,
    target_segment_mm: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    shape = int(geometry.source_shape)
    if shape != SOURCE_SHAPE_ROUNDED_CAP or throat_radius_mm <= 1.0e-9:
        n_segments = max(
            1, int(math.ceil(throat_radius_mm / max(target_segment_mm, 1.0e-9)))
        )
        rho = np.linspace(0.0, throat_radius_mm, n_segments + 1, dtype=np.float64)
        z = np.full_like(rho, float(throat_z_mm))
        return np.column_stack((rho, z)), {
            "source_cap_height_mm": 0.0,
            "source_cap_radius_mm": math.inf,
            "source_cap_center_z_mm": None,
            "source_cap_segments": int(n_segments),
        }

    cap_height = _source_cap_height(throat_radius_mm, geometry)
    if cap_height <= 1.0e-12:
        n_segments = max(
            1, int(math.ceil(throat_radius_mm / max(target_segment_mm, 1.0e-9)))
        )
        rho = np.linspace(0.0, throat_radius_mm, n_segments + 1, dtype=np.float64)
        z = np.full_like(rho, float(throat_z_mm))
        return np.column_stack((rho, z)), {
            "source_cap_height_mm": 0.0,
            "source_cap_radius_mm": math.inf,
            "source_cap_center_z_mm": None,
            "source_cap_segments": int(n_segments),
        }

    radius = max(
        _source_cap_radius(throat_radius_mm, geometry), throat_radius_mm * 1.001
    )
    sign = -1.0 if int(geometry.source_curv) == -1 else 1.0
    center_z = float(throat_z_mm) + sign * (cap_height - radius)
    rim_angle = math.asin(max(-1.0, min(1.0, throat_radius_mm / radius)))
    arc_length = abs(radius * rim_angle)
    n_segments = max(1, int(math.ceil(arc_length / max(target_segment_mm, 1.0e-9))))
    rho = np.linspace(0.0, throat_radius_mm, n_segments + 1, dtype=np.float64)
    z = center_z + sign * np.sqrt(np.maximum(0.0, radius * radius - rho * rho))
    z[-1] = float(throat_z_mm)
    return np.column_stack((rho, z)), {
        "source_cap_height_mm": float(cap_height),
        "source_cap_radius_mm": float(radius),
        "source_cap_center_z_mm": float(center_z),
        "source_cap_segments": int(n_segments),
    }


def _polyline_normals(nodes: np.ndarray, segments: np.ndarray) -> np.ndarray:
    p0 = nodes[segments[:, 0]]
    p1 = nodes[segments[:, 1]]
    delta = p1 - p0
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths <= 1.0e-15):
        raise ConfigError("CircSym meridian contains a zero-length segment")
    return np.column_stack((-delta[:, 1], delta[:, 0])) / lengths[:, None]


def circsym_rejection_reasons(
    config: Mapping[str, Any],
    *,
    freq_max_hz: float | None = None,
) -> list[str]:
    """Reasons ``config`` cannot be solved as a CircSym body-of-revolution.

    Empty list => the config is CircSym-eligible (circular cross-section, no
    morph/enclosure). Non-empty => the full-3D solver is
    required. This is the *authoritative* eligibility gate for auto-mode: it
    defers to :func:`build_meridian`, so it agrees with the real solve path by
    construction -- covering the static cross-section/morph/enclosure checks
    *and* the runtime azimuthal-variation guard that a cheap param-only check
    would miss (e.g. a non-unity superellipse guiding curve).
    Cheap: a meridian build is ~1 ms and never touches gmsh; auto-mode discards
    the returned mesh and rebuilds it inside the solve path (a negligible
    double-build) so the eligibility check can stay a pure predicate.
    """
    try:
        build_meridian(config, freq_max_hz=freq_max_hz)
    except ConfigError as exc:
        return [str(exc)]
    return []


def build_meridian(
    config: Mapping[str, Any],
    *,
    freq_max_hz: float | None = None,
) -> MeridianBuildResult:
    """Build a tagged CircSym body-of-revolution meridian from public config.

    CircSym resolution is controlled only by the mesh section's millimetre
    values.
    """

    if freq_max_hz is not None:
        raise ConfigError(
            "freq_max_hz was removed by the millimetre-only mesh contract; "
            "use throat_res_mm, mouth_res_mm, and rear_res_mm"
        )

    params, formula, mode = build_geometry_params(config)
    _validate_mode_contract(params, mode)
    mesh = _section(config, "mesh")
    topology_mode = _mesh_topology_mode(mesh)
    density = _mesh_density_from_config(config)
    enclosure = _section(config, "enclosure")
    enclosure_obj = _enclosure_from_config(config, mesh, enclosure, formula)

    reasons = _axisymmetric_rejection_reasons(
        params,
        mode=mode,
        enclosure_obj=enclosure_obj,
    )
    if reasons:
        raise ConfigError(
            "CircSym requires a circular waveguide: " + "; ".join(reasons)
        )

    resolution = _circsym_resolution_budget(params, density)
    params = dict(params)
    params["lengthSegments"] = int(resolution["geometry_segments"])
    target_throat_mm = float(resolution["throat_target_mm"])
    target_mouth_mm = float(resolution["mouth_target_mm"])
    target_outer_mm = float(resolution["outer_target_mm"])
    target_aperture_mm = float(resolution["aperture_target_mm"])

    grid = build_point_grid(params)
    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    inner_points = _reshape_grid(grid["inner_points"], n_phi, n_length, "inner_points")
    inner_profile = _radial_profile_from_grid(inner_points, name="inner profile")
    inner_profile = _resample_polyline_by_local_size(
        inner_profile,
        target_throat_mm,
        target_mouth_mm,
    )

    outer_profile: np.ndarray | None = None
    rear_profile: np.ndarray | None = None
    if grid.get("outer_points") is not None and mode != "infinite-baffle":
        outer_points = _reshape_grid(
            grid["outer_points"], n_phi, n_length, "outer_points"
        )
        outer_points = _restored_outer_throat_points(
            inner_points,
            outer_points,
            wall_thickness_mm=float(params["wallThickness"] or 0.0),
        )
        outer_profile_grid = _radial_profile_from_grid(
            outer_points, name="outer profile"
        )
        rear_z = float(
            np.mean(inner_points[:, 0, 2]) - float(params["wallThickness"] or 0.0)
        )
        rear_points = _rear_rim_points(outer_points, rear_z=rear_z)
        rear_profile = _radial_profile_from_grid(
            rear_points[:, np.newaxis, :],
            name="rear cap",
        )[0:1, :]
        outer_indices = _outer_wall_axial_ring_indices(inner_points)
        topology_rows = [rear_profile[0], outer_profile_grid[0]]
        topology_rows.extend(outer_profile_grid[index] for index in outer_indices)
        outer_profile = _resample_polyline_by_local_size(
            np.asarray(topology_rows, dtype=np.float64),
            target_outer_mm,
            target_outer_mm,
        )

    baffle_z_mm: float | None = None
    if mode == "infinite-baffle":
        mouth_z = float(inner_profile[-1, 1])
        inner_profile = inner_profile.copy()
        inner_profile[:, 1] -= mouth_z

    throat_radius = float(inner_profile[0, 0])
    mouth_radius = float(inner_profile[-1, 0])
    throat_z = float(inner_profile[0, 1])
    source_geometry = PointGridHornGeometry(
        inner_points=inner_points,
        wall_thickness_mm=float(params["wallThickness"] or 0.0),
        source_shape=int(_num_or_default(params.get("sourceShape"), 1)),
        source_radius_mm=_num_or_default(params.get("sourceRadius"), -1),
        source_curv=int(_num_or_default(params.get("sourceCurv"), 0)),
        source_auto_angle_deg=float(eval_param(params.get("a0"), 0.0, 15.5)),
        infinite_baffle=(mode == "infinite-baffle"),
    )
    source_shape = int(source_geometry.source_shape)
    if source_shape not in {0, 1}:
        raise ConfigError(
            f"CircSym source_shape={source_shape} is not supported; "
            "expected 0 (flat) or 1 (rounded)"
        )
    if not math.isfinite(throat_radius) or throat_radius <= 1.0e-9:
        raise ConfigError(
            "CircSym driven source must have positive swept area; "
            f"throat radius is {throat_radius:.6g} mm"
        )
    cap_points, cap_meta = _source_cap_polyline(
        throat_radius_mm=throat_radius,
        throat_z_mm=throat_z,
        geometry=source_geometry,
        target_segment_mm=target_throat_mm,
    )

    nodes_mm: list[tuple[float, float]] = []
    tags: list[int] = []
    source_segment_count = _append_subdivided_meridian_points(
        nodes_mm,
        tags,
        cap_points,
        tag=int(PhysicalGroup.PRIMARY_SOURCE),
        target_segment_mm=target_throat_mm,
    )
    inner_segment_count = _append_subdivided_meridian_points(
        nodes_mm,
        tags,
        inner_profile,
        tag=int(PhysicalGroup.RIGID_WALL),
        target_segment_mm=max(target_throat_mm, target_mouth_mm),
    )
    aperture_segment_count = 0
    if mode == "infinite-baffle":
        disc_points = np.asarray(
            [[mouth_radius, 0.0], [0.0, 0.0]],
            dtype=np.float64,
        )
        aperture_segment_count = _append_subdivided_meridian_points(
            nodes_mm,
            tags,
            disc_points,
            tag=int(PhysicalGroup.MOUTH_APERTURE),
            target_segment_mm=target_aperture_mm,
        )

    mouth_rim_segment_count = 0
    outer_segment_count = 0
    rear_cap_segment_count = 0
    if outer_profile is not None and rear_profile is not None:
        outer_reversed = outer_profile[::-1]
        if nodes_mm and outer_reversed.shape[0] > 0:
            incoming = (
                float(nodes_mm[-1][0] - nodes_mm[-2][0]),
                float(nodes_mm[-1][1] - nodes_mm[-2][1]),
            )
            rounded_mouth = _semicircular_meridian_join_points(
                nodes_mm[-1],
                (float(outer_reversed[0, 0]), float(outer_reversed[0, 1])),
                incoming_tangent=incoming,
                target_segment_mm=min(target_mouth_mm, target_outer_mm),
                minimum_segments=5 if topology_mode == "legacy" else 2,
            )
            mouth_rim_segment_count = _append_subdivided_meridian_points(
                nodes_mm,
                tags,
                rounded_mouth,
                tag=int(PhysicalGroup.RIGID_WALL),
                target_segment_mm=min(target_mouth_mm, target_outer_mm),
            )
        outer_segment_count = (
            mouth_rim_segment_count
            + _append_subdivided_meridian_points(
                nodes_mm,
                tags,
                outer_reversed,
                tag=int(PhysicalGroup.RIGID_WALL),
                target_segment_mm=target_outer_mm,
            )
        )
        rear_axis = np.asarray([[0.0, float(rear_profile[0, 1])]], dtype=np.float64)
        if nodes_mm:
            rear_cap_segment_count = _subdivision_count_between(
                nodes_mm[-1],
                (float(rear_axis[0, 0]), float(rear_axis[0, 1])),
                target_outer_mm,
            )
        _append_subdivided_meridian_points(
            nodes_mm,
            tags,
            rear_axis,
            tag=int(PhysicalGroup.RIGID_WALL),
            target_segment_mm=target_outer_mm,
        )

    nodes = np.asarray(nodes_mm, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[0] < 2:
        raise ConfigError("CircSym meridian requires at least two nodes")
    if len(tags) != nodes.shape[0] - 1:
        raise ConfigError("CircSym meridian tag count does not match segment count")
    segments = np.column_stack(
        (
            np.arange(nodes.shape[0] - 1, dtype=np.int32),
            np.arange(1, nodes.shape[0], dtype=np.int32),
        )
    )
    tags_arr = np.asarray(tags, dtype=np.int32)
    normals = _polyline_normals(nodes, segments)
    source_mask = tags_arr == int(PhysicalGroup.PRIMARY_SOURCE)
    source_delta = nodes[segments[source_mask, 1]] - nodes[segments[source_mask, 0]]
    source_length = np.linalg.norm(source_delta, axis=1)
    source_rho = 0.5 * (
        nodes[segments[source_mask, 0], 0] + nodes[segments[source_mask, 1], 0]
    )
    source_swept_area_mm2 = float(np.sum(2.0 * math.pi * source_rho * source_length))
    if source_segment_count <= 0 or source_swept_area_mm2 <= 1.0e-12:
        raise ConfigError("CircSym driven source has zero swept surface measure")

    nodes_m = nodes * 0.001
    baffle_z = None if baffle_z_mm is None else float(baffle_z_mm) * 0.001
    metadata = {
        "generatedBy": "hornlab-waveguide-mesher",
        "units": "m",
        "formula": formula,
        "mode": mode,
        "segmentCount": int(segments.shape[0]),
        "nodeCount": int(nodes.shape[0]),
        "throatTargetSegmentM": target_throat_mm * 0.001,
        "mouthTargetSegmentM": target_mouth_mm * 0.001,
        "outerTargetSegmentM": target_outer_mm * 0.001,
        "apertureTargetSegmentM": target_aperture_mm * 0.001,
        "profileArcLengthM": float(resolution["profile_arc_length_mm"]) * 0.001,
        "geometrySampleSegments": int(params["lengthSegments"]),
        "sourceSegmentCount": int(source_segment_count),
        "sourceSweptAreaM2": source_swept_area_mm2 * 1.0e-6,
        "innerSegmentCount": int(inner_segment_count),
        "outerSegmentCount": int(outer_segment_count),
        "mouthRimSegmentCount": int(mouth_rim_segment_count),
        "rearCapSegmentCount": int(rear_cap_segment_count),
        "apertureTag": (
            int(PhysicalGroup.MOUTH_APERTURE) if mode == "infinite-baffle" else None
        ),
        "apertureSegmentCount": int(aperture_segment_count),
        "wallSegmentCount": int(
            np.count_nonzero(tags_arr == int(PhysicalGroup.RIGID_WALL))
        ),
        "throatRadiusM": throat_radius * 0.001,
        "mouthRadiusM": mouth_radius * 0.001,
        "sourceCapHeightM": float(cap_meta["source_cap_height_mm"]) * 0.001,
        "sourceCapRadiusM": (
            None
            if not math.isfinite(float(cap_meta["source_cap_radius_mm"]))
            else float(cap_meta["source_cap_radius_mm"]) * 0.001
        ),
        "sourceCapCenterZM": (
            None
            if cap_meta["source_cap_center_z_mm"] is None
            else float(cap_meta["source_cap_center_z_mm"]) * 0.001
        ),
        "baffleZM": baffle_z,
        "closedOnAxis": bool(nodes[0, 0] <= 1.0e-9 and nodes[-1, 0] <= 1.0e-9),
    }
    return MeridianBuildResult(
        nodes=nodes_m,
        segments=segments,
        physical_tags=tags_arr,
        normals=normals,
        baffle_z=baffle_z,
        formula=formula,
        mode=mode,
        metadata=metadata,
    )


def _build_acoustic_sampling_grid(
    params: Mapping[str, Any],
    density: MeshDensity,
    *,
    topology_mode: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Fit geometry from a control grid finer than the requested final mesh.

    OCC B-spline surfaces approximate their control points; they do not
    interpolate every sample. Keep angular chords within twice the local mesh
    target, angular sagitta within 5% of it, and axial chords within half the
    target. Coarse viewport/export sampling therefore cannot visibly shrink the
    acoustic geometry.
    """

    working = dict(params)
    if topology_mode != "acoustic":
        grid = build_point_grid(working)
        return grid, {
            "geometrySampleAngularSegments": int(working["angularSegments"]),
            "geometrySampleLengthSegments": int(working["lengthSegments"]),
        }

    if not working.get("zMapPoints"):
        working["samplingMode"] = "ath-default-zmap"
    enforce_angular_sagitta = (
        _static_float_or_none(working.get("morphTarget", 0)) == 0.0
        and _static_float_or_none(working.get("gcurveType", 0)) == 0.0
    )

    max_segments = 2048
    for _attempt in range(10):
        grid = build_point_grid(working)
        n_phi = int(grid["grid_n_phi"])
        n_length = int(grid["grid_n_length"])
        # The outer freestanding grid is derived from the inner fit and can
        # contain intentional closure transitions. The inner acoustic boundary
        # is the fidelity authority; both surfaces receive the same sampling.
        surfaces = [
            _reshape_grid(grid["inner_points"], n_phi, n_length, "inner_points")
        ]

        ring_t = np.linspace(0.0, 1.0, n_length + 1, dtype=np.float64)
        ring_h = (
            float(density.throat_res_mm)
            + (float(density.mouth_res_mm) - float(density.throat_res_mm)) * ring_t
        )
        angular_ratio = 0.0
        axial_ratio = 0.0
        for points in surfaces:
            angular_delta = np.diff(points, axis=0)
            if bool(grid.get("full_circle", True)):
                angular_delta = np.concatenate(
                    (angular_delta, points[:1, :, :] - points[-1:, :, :]), axis=0
                )
            if angular_delta.size:
                angular_lengths = np.linalg.norm(angular_delta, axis=2)
                angular_ratio = max(
                    angular_ratio,
                    float(np.max(angular_lengths / (2.0 * ring_h[None, :]))),
                )
                angular_points = points
                if bool(grid.get("full_circle", True)):
                    angular_points = np.concatenate(
                        (points[-1:, :, :], points, points[:1, :, :]), axis=0
                    )
                if enforce_angular_sagitta and angular_points.shape[0] >= 3:
                    previous = angular_points[:-2, :, :]
                    current = angular_points[1:-1, :, :]
                    following = angular_points[2:, :, :]
                    chord = following - previous
                    chord_sq = np.sum(chord * chord, axis=2)
                    alpha = np.divide(
                        np.sum((current - previous) * chord, axis=2),
                        chord_sq,
                        out=np.zeros_like(chord_sq),
                        where=chord_sq > 1.0e-18,
                    )
                    projection = previous + alpha[:, :, None] * chord
                    sagitta = np.linalg.norm(current - projection, axis=2)
                    angular_ratio = max(
                        angular_ratio,
                        float(np.max(sagitta / (0.05 * ring_h[None, :]))),
                    )

            axial_delta = np.diff(points, axis=1)
            if axial_delta.size:
                axial_lengths = np.linalg.norm(axial_delta, axis=2)
                axial_h = 0.5 * (ring_h[:-1] + ring_h[1:])
                axial_ratio = max(
                    axial_ratio,
                    float(np.max(axial_lengths / (0.5 * axial_h[None, :]))),
                )

        if angular_ratio <= 1.0 and axial_ratio <= 1.0:
            return grid, {
                "geometrySampleAngularSegments": int(working["angularSegments"]),
                "geometrySampleLengthSegments": int(working["lengthSegments"]),
            }

        current_angular = int(working["angularSegments"])
        current_length = int(working["lengthSegments"])
        current_corner = int(working.get("cornerSegments") or 1)
        if angular_ratio > 1.0:
            working["angularSegments"] = min(
                max_segments,
                max(
                    current_angular + 1,
                    int(math.ceil(current_angular * angular_ratio * 1.05)),
                ),
            )
            working["cornerSegments"] = min(
                max_segments,
                max(
                    current_corner + 1,
                    int(math.ceil(current_corner * angular_ratio * 1.05)),
                ),
            )
        if axial_ratio > 1.0:
            working["lengthSegments"] = min(
                max_segments,
                max(
                    current_length + 1,
                    int(math.ceil(current_length * axial_ratio * 1.05)),
                ),
            )
        if (
            int(working["angularSegments"]) == current_angular
            and int(working["lengthSegments"]) == current_length
            and int(working["cornerSegments"]) == current_corner
        ):
            break

    raise ConfigError(
        "requested mm resolution needs more than 2048 internal geometry samples; "
        "use a coarser throat/mouth resolution"
    )


def build_from_config(
    config: Mapping[str, Any],
    output_path: str | Path,
    *,
    allow_large_mesh: bool | None = None,
) -> BuildResult:
    params, formula, mode = build_geometry_params(config)
    _validate_mode_contract(params, mode)
    mesh = _section(config, "mesh")
    enclosure = _section(config, "enclosure")
    topology_mode = _mesh_topology_mode(mesh)
    density = _mesh_density_from_config(config, allow_large_mesh=allow_large_mesh)
    enclosure_obj = _enclosure_from_config(config, mesh, enclosure, formula)

    wall_thickness_mm = float(params.get("wallThickness") or 0.0)
    acoustic_wall_floor_mm = 0.025 * min(
        float(density.throat_res_mm),
        float(density.mouth_res_mm),
        float(density.rear_res_mm),
    )
    if (
        topology_mode == "acoustic"
        and mode == "freestanding"
        and 0.0 < wall_thickness_mm < acoustic_wall_floor_mm
    ):
        raise ConfigError(
            f"wall_thickness_mm={wall_thickness_mm:g} is below the stable acoustic "
            f"feature floor {acoustic_wall_floor_mm:g} mm for the requested mesh; "
            "set wall thickness to 0 for bare mode or use a resolvable thickness"
        )

    grid, geometry_sampling_metadata = _build_acoustic_sampling_grid(
        params,
        density,
        topology_mode=topology_mode,
    )

    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    inner_points = _reshape_grid(grid["inner_points"], n_phi, n_length, "inner_points")
    outer_points = None
    if grid.get("outer_points") is not None and enclosure_obj is None:
        outer_points = _reshape_grid(
            grid["outer_points"], n_phi, n_length, "outer_points"
        )

    interface_offsets = _number_list(params.get("interfaceOffset"))
    interfaces = _interfaces_from_params(params, n_length)
    # ATH builds free-standing subdomain models (mouth interface I1-2 plus an
    # SD2 exterior); this mesher only builds interfaces for enclosure models, so
    # an explicit request on other modes must fail loudly instead of silently
    # dropping the subdomain topology.
    if mode != "enclosure" and (
        _number_list(params.get("subdomainSlices")) or interface_offsets
    ):
        raise ConfigError(
            "Mesh.SubdomainSlices/Mesh.InterfaceOffset request subdomain interfaces, "
            "which are only supported for enclosure builds "
            "(ATH's free-standing two-subdomain construction is not implemented)"
        )
    quadrants = _normalised_quadrants(params.get("quadrants"))
    native_plane = _native_symmetry_plane_for_mode(mode, quadrants)
    geometry = PointGridHornGeometry(
        inner_points=inner_points,
        outer_points=outer_points,
        topology_mode=topology_mode,
        # ATH does not scale Mesh.WallThickness by global Scale; the rear-cap
        # depth follows the unscaled wall offset.
        wall_thickness_mm=float(params["wallThickness"] or 0.0),
        preserve_grid=(
            topology_mode == "legacy"
            and _bool(mesh, names=("preserve_grid", "preserveGrid"), default=False)
        ),
        closed=bool(grid.get("full_circle", True)),
        symmetry_planes=_symmetry_planes_for_quadrants(quadrants),
        vertical_offset_mm=float(grid.get("vertical_offset_mm", 0.0) or 0.0),
        source_shape=int(_num_or_default(params.get("sourceShape"), 1)),
        source_radius_mm=_num_or_default(params.get("sourceRadius"), -1),
        source_curv=int(_num_or_default(params.get("sourceCurv"), 0)),
        source_auto_angle_deg=float(eval_param(params.get("a0"), 0.0, 15.5)),
        interface_offset_mm=float(interface_offsets[0] if interface_offsets else 0.0),
        interfaces=interfaces,
        enclosure=enclosure_obj,
        infinite_baffle=(mode == "infinite-baffle"),
    )
    scale_to_metres = _bool(
        mesh, names=("scale_to_metres", "scaleToMetres"), default=True
    )
    mesh_path, info = build_mesh_with_info(
        geometry, density, output_path, scale_to_metres=scale_to_metres
    )
    mesh_report = _mesh_report(info.physical_groups, info.edge_stats_mm)
    return BuildResult(
        mesh_path=mesh_path,
        formula=formula,
        mode=mode,
        n_vertices=info.n_vertices,
        n_triangles=info.n_triangles,
        units=info.units,
        physical_groups=info.physical_groups,
        quadrants=quadrants,
        native_symmetry_plane=native_plane,
        native_check_open_edges=_native_check_open_edges_for_mode(mode),
        mesh_report=mesh_report,
        solve_cost=cost.estimate_solve_cost(info.n_triangles).to_dict(),
        metadata={**info.metadata, **geometry_sampling_metadata},
    )


__all__ = [
    "BuildResult",
    "MeridianBuildResult",
    "build_from_config",
    "build_geometry_params",
    "build_meridian",
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
