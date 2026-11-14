"""Dropping B-spline poles during STEP export must not change a single STEP byte.

``cad._write_step_once`` builds inside ``drop_bspline_poles()`` so the final
prune does not have to delete every pole vertex at once (quadratic in gmsh).
Each case exports the same design in two fresh processes, one with the poles
dropped and one with them kept (a second export in one process renumbers OCC's
assembly counters), and compares the STEP text with only the timestamp masked.
Measured 14 Nov, the point-grid, OSSE, R-OSSE (enclosure and curled) and
native-expression cases drop poles; the other native, FREEFORM and ATH cases
drop none and are controls that the context leaves them untouched.
Both runs must use serial OCC: WG keeps Geometry.OCCParallel off until a
cross-platform qualification exists.
"""

from __future__ import annotations

import contextlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

gmsh = pytest.importorskip("gmsh")

from hornlab_mesher import load_config  # noqa: E402
from hornlab_mesher.builders import _occ  # noqa: E402
from hornlab_mesher.cad import write_step, write_step_from_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _rosse_curled():
    # A WG-imported free-standing R-OSSE with a curled lip (tmax 1.2), 6 mm
    # wall, as WG hands it to the solid STEP export, at a 12 mm CAD grid.
    return {"formula": "R-OSSE", "mode": "freestanding",
            "profile": {"formula": "R-OSSE", "R": "150", "r0": "12.7", "a": "60", "a0": "15.5",
                        "k": "1", "q": "1", "r": "0.4", "b": "0.2", "m": "0.85", "tmax": "1.2",
                        "_athLengthMode": "total"},
            "mesh": {"angularSegments": "64", "lengthSegments": "32", "quadrants": 1234,
                     "wallThickness": "6", "athRearShape": 1, "surfaceFit": "interpolate",
                     "cornerSegments": 12, "throatResolution": 12.0, "mouthResolution": 12.0,
                     "rearResolution": 12.0},
            "cross_section": {"exponent": 2.0, "aspectRatio": 1.0}, "morph": {}, "gcurve": {},
            "source": {}, "scale": 1.0, "_textImportVersion": "ath-2026-09a-v3",
            "_athAxialSamplingRule": "ath-v2025-12+", "_athNacPresent": False}


def _native(module, *args):
    import importlib
    return lambda: importlib.import_module(f"tests.{module}").config(*args)


def _expressions():
    from tests.test_native_expressions import recipe
    config = recipe("OSSE-AXIAL")
    config["profile"]["a_deg"] = "40 + cos(2*p)"
    return config


def _freeform_coarse():
    from tests.test_freeform_curves import coarse_config, reviewer
    return coarse_config(reviewer())


def _point_grid_freestanding():
    from tests.test_cad_step import _freestanding
    return _freestanding()


CASES = {
    "point-grid-freestanding": _point_grid_freestanding,
    "osse-freestanding": lambda: load_config(ROOT / "examples/osse-freestanding.toml"),
    "rosse-enclosure": lambda: load_config(ROOT / "examples/rosse-enclosure.toml"),
    "rosse-curled-freestanding": _rosse_curled,
    "terminating-arc": _native("test_native_terminating_arc", 100.0),
    "mouth-roundover": _native("test_native_mouth_roundover"),
    "throat-adapter": _native("test_native_throat_adapter"),
    "adapter-controls": _native("test_native_adapter_controls"),
    "native-expressions": _expressions,
    "standalone-source-body": _native("test_standalone_source_body"),
    "freeform-bare": lambda: load_config(ROOT / "examples/freeform-bare.toml"),
    "freeform-coarse": _freeform_coarse,
    "ath-import": _native("test_ath_cad", "p1"),
}


def _export(name: str, target: Path, keep_poles: bool) -> float:
    """Worker body: export one case, return the OCCParallel value it wrote with."""
    if keep_poles:
        _occ.drop_bspline_poles = contextlib.nullcontext
    design = CASES[name]()
    parallel = []
    write = gmsh.write

    def recording_write(path, *args, **kwargs):
        if str(path).endswith(".step"):
            parallel.append(gmsh.option.getNumber("Geometry.OCCParallel"))
        return write(path, *args, **kwargs)

    gmsh.write = recording_write
    if isinstance(design, dict):
        write_step_from_config(design, target)
    else:
        write_step(design, target)
    assert len(parallel) == 1
    return parallel[0]


def _masked(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return re.sub(r"'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d'", "'TIME'", text)


def _run(name: str, target: Path, keep_poles: bool) -> float:
    script = ("import sys; from tests.test_step_pole_identity import _export; "
              "print(_export(sys.argv[1], __import__('pathlib').Path(sys.argv[2]), sys.argv[3] == '1'))")
    done = subprocess.run([sys.executable, "-c", script, name, str(target), "1" if keep_poles else "0"],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    return float(done.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("name", sorted(CASES))
def test_dropping_poles_does_not_change_the_step(tmp_path, name):
    assert _run(name, tmp_path / "dropped.step", keep_poles=False) == 0
    assert _run(name, tmp_path / "kept.step", keep_poles=True) == 0
    assert _masked(tmp_path / "dropped.step") == _masked(tmp_path / "kept.step")
