"""Shared Git archives, isolated artifact writer and STEP identity comparison."""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def archive_tree(tmp_path_factory, base):
    archive = subprocess.run(
        ['git', 'archive', base], cwd=ROOT, check=True, capture_output=True,
    ).stdout
    tree = tmp_path_factory.mktemp('wall-identity-base')
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(tree, filter='data')
    return tree


WORKER = '''
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch
import gmsh
import numpy as np
from hornlab_mesher import build_from_config, write_step_from_config
root, source, destination = map(Path, sys.argv[1:4])
import hornlab_mesher
assert Path(hornlab_mesher.__file__).resolve().is_relative_to(root.resolve())
config = json.loads(source.read_text())

def preview_fields(value):
    if isinstance(value, np.ndarray):
        return {"sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
                "dtype": value.dtype.str, "shape": list(value.shape)}
    if dataclasses.is_dataclass(value):
        return {field.name: preview_fields(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {key: preview_fields(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [preview_fields(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value

def write_json(name, value):
    (destination / name).write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )

gmsh.initialize(interruptible=False)
gmsh.option.setString('Geometry.OCCSTEPTimeStamp', '2000-01-01T00:00:00')
try:
    if len(sys.argv) > 4 and sys.argv[4] == "native":
        from hornlab_mesher.config_builder import build_geometry_params, resolve_geometry
        from hornlab_mesher.preview.api import build_preview_geometry
        write_json("params.json", build_geometry_params(config))
        # ResolvedGeometry calls its metadata field sampling_metadata in v0.2.6.
        write_json("metadata.json", resolve_geometry(config).sampling_metadata)
        # Keep the timing fields too, but pin their input clock so they are
        # deterministic. Default preview options have no refinement deadline.
        with patch("time.perf_counter", return_value=0.0):
            write_json("preview.json", preview_fields(build_preview_geometry(config)))
    build_from_config(config, destination / 'out.msh')
    write_step_from_config(config, destination / 'out.step')
finally:
    gmsh.finalize()
'''


def run_worker(tree, source, target, *, native=False):
    """Write all artifacts from one fresh interpreter rooted in the given tree."""
    target.mkdir()
    command = [sys.executable, '-c', WORKER, str(tree), str(source), str(target)]
    if native:
        command.append('native')
    with (target / 'build.log').open('w') as log:
        subprocess.run(
            command, cwd=tree, env={**os.environ, 'PYTHONPATH': str(tree)},
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=50,
        )


def normalize_step(value):
    """Ignore only OCC's documented per-process translator product counter."""
    return re.sub(
        rb'Open CASCADE STEP translator 7\.8 \d+',
        b'Open CASCADE STEP translator 7.8 N', value,
    )


def assert_mesh_and_step_bytes(base_tree, tmp_path, name, config, *, native=False):
    source = tmp_path / 'config.json'
    source.write_text(json.dumps(config))
    artifacts = []
    for label, tree in (('base', base_tree), ('current', ROOT)):
        target = tmp_path / label
        run_worker(tree, source, target, native=native)
        artifacts.append(target)
    base, current = artifacts
    assert (base / 'out.msh').read_bytes() == (current / 'out.msh').read_bytes(), name
    assert normalize_step((base / 'out.step').read_bytes()) == normalize_step(
        (current / 'out.step').read_bytes()
    ), name
    if native:
        for artifact in ('params.json', 'metadata.json', 'preview.json'):
            assert (base / artifact).read_bytes() == (current / artifact).read_bytes(), (
                name, artifact,
            )
