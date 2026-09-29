from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Iterator

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
from .config_parser import (
    ConfigError,
    load_config,
    parse_ath_config,
    parse_legacy_config,
    parse_text_config,
)
from .mesher import MesherError

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
    parser.add_argument(
        "-o", "--output", help="Output .msh path; overrides output.path in config"
    )
    parser.add_argument("--summary", help="Optional JSON summary output path")
    parser.add_argument(
        "--print-summary", action="store_true", help="Print build summary as JSON"
    )
    parser.add_argument(
        "--allow-large-mesh",
        action="store_true",
        help="Explicitly allow the generated mesh to exceed mesh.max_triangles",
    )
    parser.add_argument(
        "--step",
        metavar="PATH",
        help=(
            "Write the CAD model to a .step/.stp file for Fusion 360, Onshape, "
            "etc. Skips the mesh build unless -o/--output is also given."
        ),
    )
    parser.add_argument(
        "--step-keep-throat",
        action="store_true",
        help="Keep the driver membrane in the STEP body instead of opening the bore",
    )
    return parser


#: Failures that describe the input or the environment rather than a defect in
#: the package: reported as one line with exit status 2. ``ConfigError`` is a
#: ``ValueError``, and the geometry validators raise plain ``ValueError`` too;
#: ``MesherError`` is a build the mesher refused or could not complete.
_USER_ERRORS: tuple[type[BaseException], ...] = (ValueError, OSError, MesherError)


@contextlib.contextmanager
def _stdout_to_stderr() -> Iterator[None]:
    """Send everything written to stdout -- Python or native -- to stderr.

    OpenCASCADE prints its STEP transfer statistics straight to file
    descriptor 1, whatever ``sys.stdout`` is, so a Python-level redirect alone
    cannot keep them out of a ``--print-summary`` stream that must parse as
    JSON. Descriptor 1 is pointed at descriptor 2 for the duration.
    """

    sys.stdout.flush()
    sys.stderr.flush()
    try:
        saved = os.dup(1)
    except OSError:
        # No descriptor 1 (a windowed host): only Python writes can move.
        with contextlib.redirect_stdout(sys.stderr):
            yield
        return
    try:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 1)
        os.close(saved)


def main(argv: list[str] | None = None) -> int:
    """Run the ``hornlab-waveguide`` command line.

    Exit status 0 on success, 2 for an invalid config, a refused build or an
    unreadable/unwritable file (one ``error:`` line on stderr), and 1 with a
    traceback for anything else, which is a defect rather than bad input.
    With ``--print-summary`` stdout carries only the JSON summary; progress
    lines and native library output go to stderr.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)
    quiet_stdout = (
        _stdout_to_stderr if args.print_summary else contextlib.nullcontext
    )
    try:
        with quiet_stdout():
            summary = _run(args)
        if summary is not None and args.print_summary:
            print(json.dumps(summary, indent=2))
        return 0
    except _USER_ERRORS as exc:
        print(f"hornlab-waveguide: error: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        print(
            "hornlab-waveguide: internal error (traceback above); "
            "this is a bug, not a problem with the config",
            file=sys.stderr,
        )
        return 1


def _run(args: argparse.Namespace) -> dict | None:
    """Do the work; returns the build summary (the STEP summary for a STEP-only run)."""

    config = load_config(args.config)
    if args.step:
        from .cad import write_step_from_config

        step_path, cad = write_step_from_config(
            config, args.step, open_throat=not args.step_keep_throat
        )
        volume = (
            f", {cad.volume_mm3:,.0f} mm^3"
            if cad.volume_mm3 is not None
            else ""
        )
        print(
            f"Wrote {step_path} ({cad.body}, {cad.n_faces} faces{volume}, "
            f"units={cad.units})"
        )
        if not args.output:
            step_summary = {
                "step_path": str(step_path),
                "body": cad.body,
                "n_faces": cad.n_faces,
                "volume_mm3": cad.volume_mm3,
                "bounding_box_mm": [list(corner) for corner in cad.bounding_box_mm],
                "throat_opened": cad.throat_opened,
                "units": cad.units,
            }
            if args.summary:
                Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
                Path(args.summary).write_text(
                    json.dumps(step_summary, indent=2) + "\n", encoding="utf-8"
                )
            return step_summary
    output = args.output or _pick(
        _section(config, "output"),
        config,
        names=("path", "output_path"),
        default=None,
    )
    if not output:
        raise ConfigError(
            "output path required; set output.path or pass -o/--output"
        )
    result = build_from_config(
        config,
        output,
        allow_large_mesh=True if args.allow_large_mesh else None,
    )
    summary = result.as_dict()
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    if not args.print_summary:
        print(
            f"Wrote {result.mesh_path} "
            f"({result.formula}, {result.mode}, {result.n_vertices} vertices, "
            f"{result.n_triangles} triangles, units={result.units})"
        )
    return summary


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
