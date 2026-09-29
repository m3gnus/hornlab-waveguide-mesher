"""Legacy spline-span walls whose fills swallow a grid step must refuse, not crash.

On this infinite-baffle OSSE design (phi-expression coverage, slot and ``n``,
morph 1) the legacy wall fills stray ~5 mm from their boundary curves, which
widens the shared vertex tolerances past the 4.7 mm axial grid step. The next
curve loop then merges two grid points, and gmsh 4.15.2 segfaulted in
``addSurfaceFilling`` on the collapsed edge. The build runs in a subprocess so
a regression fails this test instead of killing pytest. Either outcome that
keeps the process alive passes: today's refusal, or a clean build should a
later gmsh/OCC fit these faces better.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

M2_LIKE_CONFIG = {
    "formula": "OSSE",
    "mode": "infinite-baffle",
    "profile": {
        "formula": "OSSE",
        "r0": 18.0,
        "a": "62 - 10*sin(p)**2 - 10*sin(2*(p+pi/4))**4",
        "a0": 0.0,
        "k": "0.9",
        "q": "0.996",
        "slotLength": "45 - 42*sin(2*p)**4",
        "_athLengthMode": "total",
        "L": "150",
        "n": "3 + 5*sin(2*p)**2",
        "s": "0.9",
        "throatProfile": "1",
    },
    "mesh": {
        "angularSegments": "100",
        "lengthSegments": "32",
        "cornerSegments": "4",
        "quadrants": 1234,
        "wallThickness": 0.0,
        "throatResolution": "3",
        "mouthResolution": "10",
        "surfaceFit": "interpolate",
        "scaleToMetres": True,
    },
    "cross_section": {"exponent": 2.0, "aspectRatio": 1.0},
    "morph": {"morphTarget": "1", "morphCorner": "8"},
    "gcurve": {},
    "source": {},
}

_CHILD = """
import json, sys
import hornlab_mesher
print("IMPORTED", hornlab_mesher.__file__, flush=True)
from hornlab_mesher.config_builder import build_from_config
from hornlab_mesher.mesher import MesherError
config, out = json.loads(sys.argv[1]), sys.argv[2]
try:
    build_from_config(config, out)
except MesherError as exc:
    print("REFUSED", exc)
else:
    print("BUILT")
"""


def _build_in_subprocess(topology: str, tmp_path: Path) -> subprocess.CompletedProcess:
    config = json.loads(json.dumps(M2_LIKE_CONFIG))
    config["mesh"]["topology"] = topology
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), *filter(None, [env.get("PYTHONPATH")])]
    )
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(config), str(tmp_path / "m.msh")],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
        cwd=str(REPO_ROOT),
    )
    # An editable install's .pth can beat PYTHONPATH; prove this tree was built.
    imported = [line for line in result.stdout.splitlines() if line.startswith("IMPORTED ")]
    assert imported, result.stdout[-2000:] + result.stderr[-2000:]
    assert Path(imported[0].split(" ", 1)[1]).resolve().parent == REPO_ROOT / "hornlab_mesher"
    return result


def test_legacy_topology_refuses_or_builds_but_never_segfaults(tmp_path):
    result = _build_in_subprocess("legacy", tmp_path)

    assert result.returncode == 0, (
        f"build process died with return code {result.returncode} "
        f"(-11 / 139 is the gmsh segfault)\n{result.stderr[-2000:]}"
    )
    if "BUILT" in result.stdout:
        assert (tmp_path / "m.msh").stat().st_size > 0
        return
    assert "REFUSED" in result.stdout, result.stdout[-2000:]
    assert "surface construction refused" in result.stdout
    assert "legacy topology's filled spline-span wall" in result.stdout


def test_acoustic_topology_still_builds_the_same_design(tmp_path):
    result = _build_in_subprocess("acoustic", tmp_path)

    assert result.returncode == 0, result.stderr[-2000:]
    assert "BUILT" in result.stdout, result.stdout[-2000:]
    assert (tmp_path / "m.msh").stat().st_size > 0
