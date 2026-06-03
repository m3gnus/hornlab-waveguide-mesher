from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:  # pragma: no cover - exercised only on Python 3.10
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


class ConfigError(ValueError):
    pass

def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".toml", ".tml"}:
        return tomllib.loads(text)
    if suffix in {".cfg", ".txt"}:
        return parse_text_config(text)
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


def parse_text_config(content: str) -> dict[str, Any]:
    """Parse the text `.cfg` shape used by imported waveguide configs."""
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
        raise ConfigError("text config must contain an OSSE or R-OSSE block")

    def mapped(items: Mapping[str, str], pairs: tuple[tuple[str, str], ...]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for src, dst in pairs:
            if src in items:
                out[dst] = _maybe_number(items[src])
        return out

    def prefixed(prefix: str) -> dict[str, str]:
        return {
            key[len(prefix) :]: value
            for key, value in flat.items()
            if key.startswith(prefix) and len(key) > len(prefix)
        }

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

    mesh_items = {**prefixed("Mesh."), **blocks.get("Mesh", {})}
    mesh = mapped(
        mesh_items,
        (
            ("AngularSegments", "angularSegments"),
            ("CornerSegments", "cornerSegments"),
            ("LengthSegments", "lengthSegments"),
            ("WallThickness", "wallThickness"),
            ("Quadrants", "quadrants"),
            ("ThroatResolution", "throatResolution"),
            ("MouthResolution", "mouthResolution"),
            ("RearResolution", "rearResolution"),
            ("SubdomainSlices", "subdomainSlices"),
            ("InterfaceOffset", "interfaceOffset"),
            ("InterfaceResolution", "interfaceResolution"),
            ("SamplingMode", "samplingMode"),
        ),
    )
    zmap_points = mesh_items.get("ZMapPoints", mesh_items.get("ZMap"))
    if zmap_points is not None:
        mesh["zMapPoints"] = zmap_points
    mesh.setdefault("samplingMode", "zmap" if zmap_points is not None else "ath-default-zmap")
    morph_items = {
        **prefixed("Morph."),
        **prefixed("MORPH."),
        **blocks.get("Morph", {}),
        **blocks.get("MORPH", {}),
    }
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

    gcurve_items = {
        **prefixed("GCurve."),
        **prefixed("GCURVE."),
        **blocks.get("GCurve", {}),
        **blocks.get("GCURVE", {}),
    }
    gcurve = mapped(
        gcurve_items,
        (
            ("Type", "gcurveType"),
            ("Width", "gcurveWidth"),
            ("AspectRatio", "gcurveAspectRatio"),
            ("Dist", "gcurveDist"),
            ("Distance", "gcurveDist"),
            ("Rot", "gcurveRot"),
            ("SF", "gcurveSF"),
            ("SF.a", "gcurveSfA"),
            ("SF.b", "gcurveSfB"),
            ("SF.m1", "gcurveSfM1"),
            ("SF.m2", "gcurveSfM2"),
            ("SF.n1", "gcurveSfN1"),
            ("SF.n2", "gcurveSfN2"),
            ("SF.n3", "gcurveSfN3"),
            ("SE.n", "gcurveSeN"),
        ),
    )

    enc_items = {**prefixed("Mesh.Enclosure."), **blocks.get("Mesh.Enclosure", {})}
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

    source_items = {**prefixed("Source."), **blocks.get("Source", {})}
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
    if gcurve:
        config["gcurve"] = gcurve
    if enclosure:
        config["enclosure"] = enclosure
    if source:
        config["source"] = source
    return config


parse_legacy_config = parse_text_config
parse_ath_config = parse_text_config
