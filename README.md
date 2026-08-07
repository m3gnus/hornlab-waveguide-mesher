# hornlab-waveguide-mesher

Canonical OSSE/R-OSSE/ICW waveguide geometry and Gmsh mesher extracted from
HornLab. This package builds standalone waveguide meshes for acoustic BEM
workflows and can replace ATH-style waveguide mesh generation wherever that
format is currently used.

This repository is intentionally limited to waveguide meshes. It does not ship
supported standalone slot, port, driver, rectangular horn, or cabinet mesh
builders. (An experimental cabinet compatibility bridge lives under
`hornlab_mesher.experimental` and is not part of the supported API.)

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
- ICW (Intrinsic-Curvature Waveguide) point-grid generation through TOML/JSON
  or dict configs.
- Freestanding wall-shell, coupled-aperture infinite-baffle
  (ABEC.SimType = 1), and enclosure-capable point-grid meshing.
- ATH text-config import with ATH defaults, global Scale/VerticalOffset,
  and azimuth-dependent profile expressions.
- Orientation validation and ABEC-compatible physical tags.
- Config-driven OSSE, R-OSSE, and ICW requests are supported. Experimental
  LOOKUP profiles are accepted for compatibility paths, but are not a stable
  public mesh-builder API.

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

## Export CAD (STEP)

The same OpenCASCADE model the solver meshes can be written out for CAD, so a
STEP file and a solve describe the same waveguide by construction:

```bash
hornlab-waveguide examples/osse-freestanding.toml --step runs/scratch/osse.step
```

Designs with wall thickness or a rear enclosure export as a **closed B-rep
solid** in millimetres, ready to import into Fusion 360, Onshape, or FreeCAD
with no thicken or cap step. The driver membrane is not material, so it is cut
away and the bore runs through. A design with no wall thickness has nothing to
enclose and exports as a surface body instead.

Two things differ from the mesh path on purpose: the acoustic level-of-detail
pass is skipped, so a fillet too small for the mesh to resolve still rounds the
part; and a symmetry-reduced design is reopened to all four quadrants, because a
part cannot be a quarter of itself.

## Python API

```python
from hornlab_mesher import build_from_config, write_step_from_config

config = {
    "formula": "OSSE",
    "profile": {"L_mm": 120, "r0_mm": 12.7, "a_deg": 60, "a0_deg": 15.5},
    "mesh": {"angular_segments": 64, "length_segments": 32, "wall_thickness_mm": 6.0},
}

build_from_config(config, "waveguide.msh")
path, info = write_step_from_config(config, "waveguide.step")
print(info.body, info.volume_mm3)  # -> solid 973743.4...
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
- `hornlab_mesher/cad.py`: STEP export of the same OCC model, sewn into a solid
  with the driver membrane cut away.
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
optimization code, or a JavaScript runtime. Cabinet generation is provided only
as an experimental compatibility bridge under `hornlab_mesher.experimental`,
not as a supported standalone builder.
