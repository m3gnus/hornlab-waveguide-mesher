from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:  # pragma: no cover - exercised only on Python 3.10
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from .geometry import HornEnclosure, MeshDensity, PointGridHornGeometry
from .mesher import build_mesh, load_mesh
from .profiles import build_point_grid


class ConfigError(ValueError):
    pass


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


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".toml", ".tml"}:
        return tomllib.loads(text)
    if suffix in {".cfg", ".txt"}:
        return parse_ath_config(text)
    raise ConfigError(f"unsupported config extension {suffix!r}; use .toml, .json, .cfg, or .txt")


def _maybe_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if not text:
        return value
    try:
        out = float(text)
    except ValueError:
        return text
    if not np.isfinite(out):
        return text
    return int(out) if out.is_integer() else out


def _parse_ath_blocks(content: str) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    blocks: dict[str, dict[str, str]] = {}
    flat: dict[str, str] = {}
    current: str | None = None

    for raw_line in content.splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if not line:
            continue
        start = line.split("=", 1)
        if len(start) == 2 and start[1].strip() == "{":
            current = start[0].strip()
            blocks.setdefault(current, {})
            continue
        if line == "}":
            current = None
            continue
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if current:
            blocks.setdefault(current, {})[key] = value
        else:
            flat[key] = value
    return blocks, flat


def parse_ath_config(content: str) -> dict[str, Any]:
    """Parse the ATH/MWG `.cfg` text shape used by WG validation fixtures."""
    blocks, flat = _parse_ath_blocks(content)
    formula = None
    profile_items: Mapping[str, str] = {}
    if "R-OSSE" in blocks:
        formula = "R-OSSE"
        profile_items = blocks["R-OSSE"]
    elif "ROSSE" in blocks:
        formula = "R-OSSE"
        profile_items = blocks["ROSSE"]
    elif "OSSE" in blocks:
        formula = "OSSE"
        profile_items = blocks["OSSE"]
    elif any(key in flat for key in ("Coverage.Angle", "Length", "Term.n")):
        formula = "OSSE"
        profile_items = flat
    if formula is None:
        raise ConfigError("ATH config must contain an OSSE or R-OSSE block")

    def mapped(items: Mapping[str, str], pairs: tuple[tuple[str, str], ...]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for src, dst in pairs:
            if src in items:
                out[dst] = _maybe_number(items[src])
        return out

    common_profile = mapped(
        profile_items,
        (
            ("r0", "r0"),
            ("Throat.Diameter", "throat_diameter"),
            ("a", "a"),
            ("Coverage.Angle", "a"),
            ("a0", "a0"),
            ("Throat.Angle", "a0"),
            ("k", "k"),
            ("OS.k", "k"),
            ("q", "q"),
            ("Term.q", "q"),
        ),
    )
    if "throat_diameter" in common_profile and "r0" not in common_profile:
        try:
            common_profile["r0"] = float(common_profile.pop("throat_diameter")) / 2.0
        except (TypeError, ValueError):
            common_profile.pop("throat_diameter", None)

    if formula == "OSSE":
        profile = {
            **common_profile,
            **mapped(
                profile_items,
                (
                    ("L", "L"),
                    ("Length", "L"),
                    ("n", "n"),
                    ("Term.n", "n"),
                    ("s", "s"),
                    ("Term.s", "s"),
                    ("h", "h"),
                    ("OS.h", "h"),
                    ("Throat.Ext.Length", "throatExtLength"),
                    ("Throat.Ext.Angle", "throatExtAngle"),
                    ("Slot.Length", "slotLength"),
                    ("Rot", "rot"),
                ),
            ),
        }
    else:
        profile = {
            **common_profile,
            **mapped(
                profile_items,
                (
                    ("R", "R"),
                    ("m", "m"),
                    ("b", "b"),
                    ("r", "r"),
                    ("tmax", "tmax"),
                ),
            ),
        }

    mesh_items = blocks.get("Mesh", {})
    mesh = mapped(
        mesh_items,
        (
            ("AngularSegments", "angularSegments"),
            ("LengthSegments", "lengthSegments"),
            ("WallThickness", "wallThickness"),
            ("Quadrants", "quadrants"),
            ("ThroatResolution", "throatResolution"),
            ("MouthResolution", "mouthResolution"),
            ("RearResolution", "rearResolution"),
        ),
    )

    morph_items = blocks.get("MORPH", blocks.get("Morph", {}))
    morph = mapped(
        morph_items,
        (
            ("TargetShape", "morphTarget"),
            ("TargetWidth", "morphWidth"),
            ("Width", "morphWidth"),
            ("TargetHeight", "morphHeight"),
            ("Height", "morphHeight"),
            ("CornerRadius", "morphCorner"),
            ("Rate", "morphRate"),
            ("FixedPart", "morphFixed"),
            ("AllowShrinkage", "morphAllowShrinkage"),
        ),
    )

    enc_items = blocks.get("Mesh.Enclosure", {})
    enclosure: dict[str, Any] = {}
    if enc_items:
        enclosure = mapped(
            enc_items,
            (
                ("Depth", "depth_mm"),
                ("EdgeRadius", "edge_mm"),
                ("EdgeType", "edge_type"),
                ("FrontResolution", "enc_front_resolution"),
                ("BackResolution", "enc_back_resolution"),
            ),
        )
        spacing = enc_items.get("Spacing")
        if spacing:
            parts = [part.strip() for part in spacing.split(",")]
            if len(parts) >= 4:
                enclosure.update(
                    {
                        "space_l_mm": _maybe_number(parts[0]),
                        "space_t_mm": _maybe_number(parts[1]),
                        "space_r_mm": _maybe_number(parts[2]),
                        "space_b_mm": _maybe_number(parts[3]),
                    }
                )

    source_items = blocks.get("Source", {})
    source = mapped(
        source_items,
        (
            ("Shape", "sourceShape"),
            ("Radius", "sourceRadius"),
            ("Curv", "sourceCurv"),
            ("VelocityProfile", "sourceVelocityProfile"),
        ),
    )

    config: dict[str, Any] = {"formula": formula, "profile": profile, "mesh": mesh}
    if morph:
        config["morph"] = morph
    if enclosure:
        config["enclosure"] = enclosure
    if source:
        config["source"] = source
    return config


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
    )


def build_geometry_params(config: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
    profile = _section(config, "profile", "parameters")
    mesh = _section(config, "mesh")
    enclosure = _section(config, "enclosure")
    cross = _section(config, "cross_section", "crossSection")
    morph = _section(config, "morph", "MORPH")
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

    common: dict[str, Any] = {
        "type": formula,
        "r0": _scalar_or_expr(profile, config, names=("r0_mm", "r0"), default=12.7),
        "a": _scalar_or_expr(profile, config, names=("a_deg", "a"), default=60.0),
        "a0": _scalar_or_expr(profile, config, names=("a0_deg", "a0"), default=15.5),
        "k": _scalar_or_expr(profile, config, names=("k",), default=1.0),
        "q": _scalar_or_expr(profile, config, names=("q",), default=1.0 if formula == "R-OSSE" else 0.995),
        "angularSegments": _int(mesh, config, names=("angular_segments", "angularSegments"), default=64),
        "lengthSegments": _int(mesh, config, names=("length_segments", "lengthSegments"), default=32),
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
        "quadrants": str(_pick(mesh, config, names=("quadrants",), default="1234")),
        "throatResolution": _float(mesh, config, names=("throat_res_mm", "throatResolution"), default=4.0),
        "mouthResolution": _float(mesh, config, names=("mouth_res_mm", "mouthResolution"), default=26.0),
        "rearResolution": _float(mesh, config, names=("rear_res_mm", "rearResolution"), default=25.0),
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

    geometry = PointGridHornGeometry(
        inner_points=inner_points,
        outer_points=outer_points,
        wall_thickness_mm=float(params["wallThickness"] or 0.0),
        preserve_grid=_bool(mesh, names=("preserve_grid", "preserveGrid"), default=False),
        closed=bool(grid.get("full_circle", True)),
        enclosure=enclosure_obj,
    )
    density = MeshDensity(
        throat_res_mm=_float(mesh, names=("throat_res_mm", "throat_res", "throatResolution"), default=4.0),
        mouth_res_mm=_float(mesh, names=("mouth_res_mm", "mouth_res", "mouthResolution"), default=26.0),
        rear_res_mm=_float(mesh, names=("rear_res_mm", "rear_res", "rearResolution"), default=25.0),
        enc_front_res_mm=_pick(mesh, enclosure, names=("enc_front_res_mm", "enc_front_resolution", "encFrontResolution"), default=None),
        enc_back_res_mm=_pick(mesh, enclosure, names=("enc_back_res_mm", "enc_back_resolution", "encBackResolution"), default=None),
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hornlab-waveguide",
        description="Build an OSSE or R-OSSE waveguide mesh from a TOML/JSON config.",
    )
    parser.add_argument("config", help="Input .toml or .json config file")
    parser.add_argument("-o", "--output", help="Output .msh path; overrides output.path in config")
    parser.add_argument("--summary", help="Optional JSON summary output path")
    parser.add_argument("--print-summary", action="store_true", help="Print build summary as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        output = args.output or _pick(_section(config, "output"), config, names=("path", "output_path"), default=None)
        if not output:
            raise ConfigError("output path required; set output.path or pass -o/--output")
        result = build_from_config(config, output)
        summary = result.as_dict()
        if args.summary:
            Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
            Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if args.print_summary:
            print(json.dumps(summary, indent=2))
        else:
            print(
                f"Wrote {result.mesh_path} "
                f"({result.formula}, {result.mode}, {result.n_vertices} vertices, "
                f"{result.n_triangles} triangles, units={result.units})"
            )
        return 0
    except Exception as exc:
        print(f"hornlab-waveguide: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
