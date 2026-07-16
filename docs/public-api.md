# Public API

The package imports as `hornlab_mesher`. The distribution and repository name
are `hornlab-waveguide-mesher`.

## Stable Entry Points

Prefer these APIs for external integrations.

### Config-Driven Builds

```python
from hornlab_mesher import build_from_config, load_config

config = load_config("examples/osse-freestanding.toml")
result = build_from_config(config, "waveguide.msh")
```

`build_from_config(config, output_path)` accepts a mapping and writes a tagged
Gmsh `.msh` file. It returns `BuildResult` from `hornlab_mesher.cli`, with:

- `mesh_path`
- `formula`
- `mode`
- `n_vertices`
- `n_triangles`
- `units`
- `physical_groups`
- `quadrants`
- `native_symmetry_plane`
- `native_check_open_edges`
- `mesh_report`
- `solve_cost`

Use this path when accepting TOML, JSON, or imported ATH/WG-style text configs.
`BuildResult.as_dict()` serializes the same fields with `mesh_path` and
physical-group keys converted to strings.
Experimental LOOKUP profiles are accepted as TOML/JSON compatibility input,
not as stable public API.

Config-driven builds enforce `mesh.max_triangles` as a full-domain-equivalent
ceiling (18,000 by default). Set `mesh.allow_large_mesh=true`, pass
`allow_large_mesh=True` to `build_from_config`, or use the CLI's
`--allow-large-mesh` flag only after reviewing the expected dense-BEM cost.

### Direct Mesh Builds

```python
from hornlab_mesher import MeshDensity, OsseHornGeometry, build_mesh

path = build_mesh(
    OsseHornGeometry(L_mm=120.0, r0_mm=12.7),
    MeshDensity(throat_res_mm=4.0, mouth_res_mm=26.0),
    "waveguide.msh",
)
```

`build_mesh(geometry, density=None, output_path=None, scale_to_metres=True)`
writes a tagged, validated Gmsh `.msh` file and returns its path.

`MeshDensity.max_triangles` applies the same full-domain-equivalent guard to
direct builds; `MeshDensity.allow_large_mesh=True` is the explicit override.

`build_mesh_with_info(...)` takes the same arguments and returns
`(path, MeshInfo)`; the info is collected at write time so the file is not
read back. Prefer it over `build_mesh` + `load_mesh` when the inspection
info is needed for a mesh built in the same call.

Supported buildable geometry at the public boundary:

- `OsseHornGeometry`
- `PointGridHornGeometry` via `hornlab_mesher.geometry`

`RosseHornGeometry` is exported as a 2D profile-parameter dataclass, but it is
not part of the current `HornGeometry` build union. Use the config path or
point-grid path for R-OSSE mesh generation.

### Mesh Inspection

```python
from hornlab_mesher import load_mesh

info = load_mesh("waveguide.msh")
```

`load_mesh(path)` reads a mesh and returns `MeshInfo` with counts, used physical
groups, bounding box, and inferred units.

## Public Dataclasses

Dataclasses exported from the package root:

- `CrossSection`
- `Enclosure`
- `HornEnclosure`
- `MeshDensity`
- `MeshInfo`
- `OsseHornGeometry`
- `RosseHornGeometry`

Additional dataclasses available from `hornlab_mesher.geometry`:

- `HornInterface`
- `PointGridHornGeometry`
- `PointGridBuildMode`
- `BuiltGeometry`

`BuiltGeometry` is an internal handoff from builders to `mesher.py`. It is
documented for maintainers, not recommended for application code.

## Formula And Builder Helpers

The package root also exports:

- `build_geometry_params`
- `build_osse_waveguide`
- `compute_osse_inner_points`
- `compute_osse_profile_points`
- `compute_rosse_profile_points`

Use these for tests, diagnostics, and integration tooling that needs profile
points without writing a final mesh. Treat `build_mesh` and `build_from_config`
as the stable application-facing build APIs.

Builder modules under `hornlab_mesher.builders` are lower-level Gmsh/OCC
implementation details. They may change surface splits, helper names, or kernel
strategy while preserving final physical tags and mesh behavior.

## CLI

The stable command shape is:

```bash
hornlab-waveguide config.toml -o waveguide.msh
```

Useful options:

- `-o`, `--output`: output `.msh` path.
- `--summary`: write JSON build summary.
- `--print-summary`: print JSON build summary.
- `--allow-large-mesh`: explicitly permit output above `mesh.max_triangles`.

The CLI accepts `.toml`, `.tml`, `.json`, `.cfg`, and `.txt` configs.

## Physical Tags

Physical tags are public solver-facing API:

| Tag | Name | Meaning |
| --- | --- | --- |
| `1` | `SD1G0` | Rigid waveguide wall. |
| `2` | `SD1D1001` | Primary source surface. |
| `3` | `SD2G0` | Enclosure wall. |
| `4` | `I1-2` | Acoustic interface surface. |
| `12` | `mouth_aperture` | Infinite-baffle Rayleigh aperture cap. |

Downstream solver integrations should consume physical tags and names rather
than relying on Gmsh surface counts.

## Internal And Test Helpers

These are not stable application APIs:

- Underscored helpers exported from `hornlab_mesher.profiles`.
- Modules under `hornlab_mesher.profile_*`.
- Modules under `hornlab_mesher.builders.*` other than documented diagnostic
  exports.
- `hornlab_mesher.normals` internals.
- `hornlab_mesher.density` internals.
- `hornlab_mesher.builders._occ`.

Some internals are intentionally imported by tests to pin ATH/WG compatibility.
That does not make them general integration surfaces.

## Compatibility Promise

Stable behavior means:

- Supported configs continue to build or fail explicitly.
- Final meshes keep documented physical tag meanings.
- Default units remain metres for written meshes unless `scale_to_metres=False`.
- Meshes are postprocessed and orientation-validated before delivery.

It does not mean exact Gmsh surface counts, internal helper names, or private
surface construction strategies are frozen.
