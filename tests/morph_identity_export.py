"""Export one identity case in a fresh process, with an asserted source root."""
from __future__ import annotations

import hashlib
import copy
import json
import re
import sys
from pathlib import Path


def export(source_root, config, destination):
    source_root = Path(source_root).resolve()
    sys.path.insert(0, str(source_root))
    import hornlab_mesher
    assert Path(hornlab_mesher.__file__).resolve().parent.parent == source_root
    from hornlab_mesher.config_builder import resolve_geometry
    from hornlab_mesher.mesher import build_mesh_with_info
    from hornlab_mesher.cad import write_step

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    resolved = resolve_geometry(config)
    mesh, _ = build_mesh_with_info(resolved.geometry, resolved.density,
                                   destination / 'mouth.msh', scale_to_metres=False)
    step = destination / 'mouth.step'
    refusal = None
    if config['mesh']['quadrants'] != '1234':
        # The public exporter requires a full model; keep that contract explicit.
        from hornlab_mesher.mesher import MesherError
        try:
            write_step(resolved.geometry, step)
        except MesherError as exc:
            refusal = str(exc)
            assert 'STEP export needs the full model' in refusal
        else:
            raise AssertionError('sector STEP must retain its public refusal')
        full = copy.deepcopy(config)
        full['mesh']['quadrants'] = '1234'
        resolved = resolve_geometry(full)
    write_step(resolved.geometry, step)
    # Only OCC's FILE_NAME timestamp is variable; retain all geometry bytes.
    data = re.sub(rb"'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d'", b"'TIME'", step.read_bytes())
    return {'mesh_sha256': hashlib.sha256(Path(mesh).read_bytes()).hexdigest(),
            'step_sha256': hashlib.sha256(data).hexdigest(),
            'sector_step_refusal': refusal}


if __name__ == '__main__':
    result = export(sys.argv[1], json.loads(sys.argv[2]), sys.argv[3])
    (Path(sys.argv[3]) / 'hashes.json').write_text(json.dumps(result) + '\n')
