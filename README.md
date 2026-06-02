# hornlab-waveguide-mesher

Canonical OSSE/R-OSSE waveguide geometry and Gmsh mesher extracted from
HornLab. This is the package intended to replace ATH-style waveguide mesh
generation in Boundary Lab.

This repository is intentionally limited to OSSE and R-OSSE waveguide meshes.
It does not ship standalone cabinet, slot, port, driver, rectangular horn, or
lookup-table mesh builders.

The repository contains two cooperating packages:

- `hornlab_mesher`: Python Gmsh mesher, physical tags, mesh density, orientation
  validation, and `.msh` output.
- `hornlab-geometry`: JavaScript geometry evaluator used by the Python mesher
  through the bundled NDJSON CLI.

The Python package currently imports as `hornlab_mesher`; the distribution and
repository name are `hornlab-waveguide-mesher`.

## Status

Implemented:

- OSSE waveguide point-grid generation.
- R-OSSE point-grid generation through the canonical JS evaluator.
- Freestanding wall-shell and enclosure-capable point-grid meshing.
- Hidden ATH parity sampling mode for ASRO2-style validation:
  - `athParitySampling: true`
  - `samplingMode: "ath-parity"`
- Orientation validation and ABEC-compatible physical tags.
- Non-OSSE geometry requests are rejected at the Python CLI and bundled
  geometry CLI boundaries.

Known remaining ATH-replacement work:

- Full ATH tessellation/source-cap parity is not complete yet.
- Current hidden parity mode matches ATH angular and axial sampling, but WG/HornLab
  still produces a denser source cap and different Gmsh tessellation than ATH.
- Keep ATH parity flags hidden/internal; do not expose them in public UI.

## Install For Development

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
npm --prefix hornlab-geometry test
python -m pytest tests/test_point_grid_contract.py -q
```

## Build A Mesh

```bash
hornlab-waveguide examples/osse-freestanding.toml -o runs/scratch/osse.msh
hornlab-waveguide examples/rosse-enclosure.toml -o runs/scratch/rosse.msh
```

## Python API

```python
from hornlab_mesher.cli import build_from_config

build_from_config(
    {
        "formula": "OSSE",
        "profile": {"L_mm": 120, "r0_mm": 12.7, "a_deg": 60, "a0_deg": 15.5},
        "mesh": {"angular_segments": 64, "length_segments": 32},
    },
    "waveguide.msh",
)
```

## Boundary Lab Integration Target

Boundary Lab should call this package before solving:

1. Convert Boundary Lab waveguide parameters or imported ATH-style config into
   the JS geometry payload.
2. Build a canonical `.msh` with ABEC-compatible physical groups.
3. Pass that mesh into `hornlab-metal-bem` or Boundary Lab's existing solver.

Recommended command/backend shape:

```bash
hornlab-waveguide config.toml -o waveguide.msh
```

## Roadmap

1. Keep this extraction buildable with the current `hornlab_mesher` import.
2. Finish ATH source-cap and rear-return tessellation parity.
3. Add a Boundary Lab mesh-generation adapter.
