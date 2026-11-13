"""Atomic publication of a native artifact built in a fresh worker process.

The worker runs as ``python -m <module> <stage>`` so it never shares the
caller's Gmsh state. It reads ``request.json`` from the stage and writes the
artifact there; only a complete stage is renamed onto the destination.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from .source_contour import canonical


def claim_destination(destination, kind):
    """The absolute destination, refused if it exists; its parent is created."""
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(f"{kind} publication requires a new destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def publish(request, destination, *, kind, module):
    """Run ``module``'s worker on ``request`` and publish its stage atomically."""
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{kind}-"
    ) as temporary:
        stage = Path(temporary)
        (stage / "request.json").write_bytes(canonical(request))
        run = subprocess.run(
            [sys.executable, "-m", module, str(stage)],
            capture_output=True,
            text=True,
            check=False,
        )
        if run.returncode:
            raise ValueError(f"native {kind} export failed: " + run.stderr)
        (stage / "request.json").unlink()
        stage.rename(destination)
    return json.loads((destination / "source.json").read_text(encoding="utf-8"))
