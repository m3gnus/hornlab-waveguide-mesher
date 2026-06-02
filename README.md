# hornlab-waveguide-mesher

Canonical OSSE/R-OSSE waveguide geometry and Gmsh mesher extracted from
HornLab. This package builds standalone waveguide meshes for acoustic BEM
workflows and can replace ATH-style waveguide mesh generation wherever that
format is currently used.

This repository is intentionally limited to OSSE and R-OSSE waveguide meshes.
It does not ship standalone cabinet, slot, port, driver, rectangular horn, or
lookup-table mesh builders.

The Python package imports as `hornlab_mesher`; the distribution and repository
name are `hornlab-waveguide-mesher`.

## Status

Implemented:

- OSSE waveguide point-grid generation.
- R-OSSE point-grid generation.
- Freestanding wall-shell and enclosure-capable point-grid meshing.
- Orientation validation and ABEC-compatible physical tags.
- Non-OSSE geometry requests are rejected by the Python CLI/config parser.

## Install For Development

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

## Build A Mesh

```bash
hornlab-waveguide examples/osse-freestanding.toml -o runs/scratch/osse.msh
hornlab-waveguide examples/rosse-enclosure.toml -o runs/scratch/rosse.msh
```

## Python API

```python
from hornlab_mesher import build_from_config

build_from_config(
    {
        "formula": "OSSE",
        "profile": {"L_mm": 120, "r0_mm": 12.7, "a_deg": 60, "a0_deg": 15.5},
        "mesh": {"angular_segments": 64, "length_segments": 32},
    },
    "waveguide.msh",
)
```

## Integration Target

Applications should call this package before solving:

1. Convert waveguide parameters or imported ATH-style config into
   an OSSE or R-OSSE config.
2. Build a canonical `.msh` with ABEC-compatible physical groups.
3. Pass that mesh into `hornlab-metal-bem` or another compatible solver.

Recommended command/backend shape:

```bash
hornlab-waveguide config.toml -o waveguide.msh
```

## Scope

The package deliberately does not include non-waveguide geometry families,
optimization code, standalone cabinet generation, or a JavaScript runtime.
