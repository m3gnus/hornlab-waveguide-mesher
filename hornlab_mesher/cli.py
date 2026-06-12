from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config_builder import (
    BuildResult,
    _bool,
    _enclosure_from_config,
    _first_number,
    _float,
    _int,
    _interfaces_from_params,
    _normalise_formula,
    _normalise_mode,
    _number_list,
    _pick,
    _reshape_grid,
    _scalar_or_expr,
    _section,
    build_from_config,
    build_geometry_params,
)
from .config_parser import ConfigError, load_config, parse_ath_config, parse_legacy_config, parse_text_config

__all__ = [
    "BuildResult",
    "ConfigError",
    "build_from_config",
    "build_geometry_params",
    "load_config",
    "main",
    "parse_ath_config",
    "parse_legacy_config",
    "parse_text_config",
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hornlab-waveguide",
        description="Build an OSSE or R-OSSE waveguide mesh from a TOML/JSON config or imported ATH-style .cfg/.txt.",
    )
    parser.add_argument("config", help="Input .toml, .json, .cfg, or .txt config file")
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
