# Architecture

This repository is intentionally small: it owns OSSE/R-OSSE/ICW waveguide profile
evaluation, point-grid topology, Gmsh mesh generation, and ABEC-compatible mesh
tags. It does not own solver setup, optimization, cabinet generation, or a
JavaScript runtime.

## Design Goals

- Keep mathematical profile behavior separate from mesh topology behavior.
- Keep ATH compatibility isolated in config/profile adapters and parity tests.
- Make buildable geometry explicit through dataclasses instead of loose dicts.
- Treat physical group tags as a public contract.
- Validate generated meshes before writing the final artifact.

## Main Build Path

```text
input config
  -> load_config / parse_text_config
  -> config_builder.build_geometry_params
  -> profiles.build_point_grid
  -> PointGridHornGeometry + MeshDensity
  -> mesher.build_mesh
  -> builders.build_point_grid
  -> density.configure_density
  -> physical groups
  -> mesh postprocess and validation
  -> final .msh
```

The direct Python API can skip the config adapter and pass an `OsseHornGeometry`
or `PointGridHornGeometry` directly to `build_mesh`.

Related reference docs:

- [config-schema.md](config-schema.md): accepted external config shape.
- [builder-invariants.md](builder-invariants.md): builder handoff contracts.
- [change-guide.md](change-guide.md): safe edit workflow by change type.
- [public-api.md](public-api.md): stable API and internal helper boundary.
- [geometry-contract.md](geometry-contract.md): mathematical and topology rules.

## Module Responsibilities

`hornlab_mesher.cli`

- Owns command-line parsing, output-path selection, summary writing, and
  user-facing error reporting.
- Re-exports config-build helpers for backwards-compatible imports.

`hornlab_mesher.config_builder`

- Owns the `build_from_config` convenience API.
- Converts external config names into the internal profile parameter shape.
- Builds `PointGridHornGeometry` and `MeshDensity` objects from config input.

`hornlab_mesher.config_parser`

- Reads `.toml`, `.json`, `.cfg`, and `.txt`.
- Maps imported ATH-style text blocks into the same config sections used by
  TOML/JSON.
- Does not build geometry or call Gmsh.

`hornlab_mesher.geometry`

- Defines dataclasses for public inputs and build outputs.
- `PointGridHornGeometry` is the canonical free-form horn surface input.
- `HornEnclosure` describes the WG enclosure payload family.
- `BuiltGeometry` is an internal OCC surface-group handoff from builders to the
  mesher.

`hornlab_mesher.profiles`

- Facade over formula, morphing, and sampling helpers.
- Produces inner and optional outer point grids from profile parameters.
- Contains compatibility-facing helpers that tests use to pin legacy behavior.

`hornlab_mesher.builders`

- Owns Gmsh/OCC topology creation.
- `point_grid_dispatch.py` is the dispatcher for point-grid build modes.
- `point_grid_enclosure.py` is a compatibility wrapper for the former
  dispatcher module path.
- `point_grid_freestanding.py`, `point_grid_sources.py`,
  `point_grid_interfaces.py`, and `point_grid_surfaces.py` split repeated
  topology jobs by surface family.
- `enclosure.py` builds cabinet/enclosure surfaces around the mouth.
- `_occ.py` contains lower-level OCC helpers shared by builders.

`hornlab_mesher.density`

- Applies mesh-size fields to the surfaces returned by builders.
- Keeps density policy separate from geometry construction.

`hornlab_mesher.mesher`

- Serializes Gmsh access with a process-level lock.
- Dispatches geometry dataclasses to builders.
- Adds physical groups, generates the mesh, removes duplicate nodes, repairs
  orientation when needed, validates the result, scales to metres by default,
  and writes the final file.

`hornlab_mesher.normals`

- Removes degenerate triangles.
- Validates signed volume, shared-edge consistency, and source normal direction.
- Repairs generated triangle winding before the final validation boundary.

`hornlab_mesher.tags`

- Defines the physical-group contract:
  - `1`: rigid wall, `SD1G0`
  - `2`: primary source, `SD1D1001`
  - `3`: enclosure wall, `SD2G0`
  - `4`: interface, `I1-2`
  - `12`: infinite-baffle mouth aperture, `mouth_aperture`

## Geometry Modes

The config-level `mode` controls the topology produced from the evaluated point
grid:

- `bare`: inner horn surface plus source surfaces only.
- `infinite-baffle`: inner horn surface plus source cap and a planar
  `mouth_aperture` cap across the z=0 mouth, with the cavity in z <= 0.
- `freestanding`: horn wall shell with outer wall, mouth rim, rear cap, and
  source surfaces.
- `enclosure`: horn surface plus rear enclosure geometry around the mouth.

If enclosure depth is positive, config normalization treats the build as
`enclosure` even when the mode is omitted.

## Compatibility Boundary

ATH/WG compatibility is important, but it should stay explicit:

- Use ATH names in config parsing and parity tests.
- Use rule names in production geometry helpers.
- Update `docs/geometry-contract.md` when a change alters mathematical,
  topology, density, or physical-tag behavior.
- Add or update a contract test before changing a behavior that downstream BEM
  workflows depend on.

## Test Orientation

- `tests/test_cli.py`: external config and command/API behavior.
- `tests/test_point_grid_contract.py`: point-grid topology, physical tags, and
  enclosure/freestanding build contracts.
- `tests/test_ath_reference_parity.py`: optional reference archive parity,
  gated by `ATH_REFERENCE_ROOT`.
- `tests/test_profile_morph_gcurve.py`: morphing and guiding-curve rules.
- `tests/test_density_contract.py`: mesh-density formulas.
- `tests/test_orientation_validation.py`: postprocess and normal validation.
- `tests/test_osse_waveguide.py` and `tests/test_rosse_waveguide.py`: formula
  smoke and endpoint checks.

Run the full suite before changing shared builders or postprocessing:

```bash
python3 -m pytest tests -q
```
