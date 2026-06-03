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

## At A Glance

The main path through the project is:

```text
config file or dict
  -> hornlab_mesher.config_builder.build_geometry_params
  -> hornlab_mesher.profiles.build_point_grid
  -> hornlab_mesher.geometry.PointGridHornGeometry
  -> hornlab_mesher.mesher.build_mesh
  -> Gmsh .msh with ABEC-compatible physical groups
```

For the full developer map, see [docs/architecture.md](docs/architecture.md).
For config keys, builder contracts, change workflow, and public API boundaries,
see:

- [docs/config-schema.md](docs/config-schema.md)
- [docs/builder-invariants.md](docs/builder-invariants.md)
- [docs/change-guide.md](docs/change-guide.md)
- [docs/public-api.md](docs/public-api.md)

For geometry rules and compatibility boundaries, see
[docs/geometry-contract.md](docs/geometry-contract.md).

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

Useful checks while changing geometry code:

```bash
python -m pytest tests/test_cli.py tests/test_point_grid_contract.py -q
python -m pytest tests/test_orientation_validation.py -q
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

## Repository Map

- `hornlab_mesher/cli.py`: command-line entry point and compatibility re-exports.
- `hornlab_mesher/config_builder.py`: config normalization and config-driven
  build orchestration.
- `hornlab_mesher/config_parser.py`: TOML/JSON plus imported ATH-style text
  config parsing.
- `hornlab_mesher/geometry.py`: dataclasses that describe buildable geometry,
  mesh density, and loaded mesh metadata.
- `hornlab_mesher/profiles.py`: profile/grid facade over the formula,
  morphing, and sampling modules.
- `hornlab_mesher/builders/`: Gmsh/OCC topology builders.
- `hornlab_mesher/density.py`: mesh-size fields and per-surface density rules.
- `hornlab_mesher/mesher.py`: build orchestration, physical groups, postprocess,
  orientation repair/validation, and final `.msh` write.
- `hornlab_mesher/tags.py`: physical tag numbers and ABEC-compatible names.
- `docs/config-schema.md`: accepted config sections, aliases, defaults, and ATH
  text import boundary.
- `docs/builder-invariants.md`: point-grid, builder, density, tag, and
  postprocess handoff contracts.
- `docs/change-guide.md`: checklist for safely changing profiles, topology,
  density, tags, import behavior, and API.
- `docs/public-api.md`: stable integration surface vs internal/test helpers.
- `docs/geometry-contract.md`: mathematical and topology contract.
- `examples/`: minimal buildable configs.
- `tests/`: contract tests for config import, profile parity, topology, density,
  and orientation behavior.

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
