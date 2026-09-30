# Public API

The package imports as `hornlab_mesher`. The distribution and repository name
are `hornlab-waveguide-mesher`.

Consumers pin this package by commit SHA, not by version: the `version` in
`pyproject.toml` is not bumped per change and cannot identify a build. What a
consumer can rely on is therefore what this document lists and what the tests
named below pin. The two known consumers are Waveguide Generator (WG) and the
HornLab Fusion add-in (WGLink).

This document has three tiers:

- **Stable** -- the application-facing API. Changing it needs a deprecation
  path or a coordinated consumer change.
- **Integration** -- lower-level functions WG or the add-in call in
  production. They are real contracts: change one only together with its
  consumer, in the same landing batch, and keep the contract tests green.
- **Internal** -- everything else, including names consumers currently reach
  into from their tests. Those uses are listed so a refactor knows what it
  breaks; they are not promises.

## Stable

### Config-driven builds

```python
from hornlab_mesher import build_from_config, load_config

config = load_config("examples/osse-freestanding.toml")
result = build_from_config(config, "waveguide.msh")
```

`build_from_config(config, output_path, *, allow_large_mesh=None)` accepts a
mapping and writes a tagged Gmsh `.msh` file. It returns `BuildResult`
(`hornlab_mesher.config_builder.BuildResult`, also re-exported from
`hornlab_mesher.cli`) with `mesh_path`, `formula`, `mode`, `n_vertices`,
`n_triangles`, `units`, `physical_groups`, `quadrants`,
`native_symmetry_plane`, `native_check_open_edges`, `mesh_report` and
`solve_cost`. `BuildResult.as_dict()` serializes the same fields with
`mesh_path` and physical-group keys converted to strings.

`load_config(path)` reads `.toml`, `.tml`, `.json`, `.cfg` and `.txt`
(ATH-style) configs. The config schema, including every accepted alias, is in
[`config-schema.md`](config-schema.md). The formula may be given at the top
level or in the profile section (`profile` or `parameters`), as `formula` or
`type`; every consumer of a config -- builds, CAD export, preview -- resolves
it through `build_geometry_params`, so the aliases behave identically
everywhere.

Config-driven builds enforce `mesh.max_triangles` as a realized
full-domain-equivalent ceiling (18,000 by default). Before Gmsh runs, an
approximate fast-fail check rejects estimates above twice the effective
limit; the post-generation triangle count is the authoritative budget check.
Set `mesh.allow_large_mesh=true`, pass `allow_large_mesh=True`, or use the
CLI's `--allow-large-mesh` flag only after reviewing the expected dense-BEM
cost. Experimental LOOKUP profiles are accepted as TOML/JSON compatibility
input, not as stable API.

### Errors

- `ConfigError` (`hornlab_mesher.config_parser`, a `ValueError`) -- an
  invalid config. Geometry validators may also raise plain `ValueError`.
- `MesherError` (`hornlab_mesher.mesher`, root export) -- a build that was
  refused or failed. `build_mesh` wraps other failures in it; inspect
  `__cause__` for the original.
- WG classifies the triangle-budget refusal by message text: it matches
  `"effective limit"` and `"pre-mesh safety margin"`. Keep those phrases in
  the refusal messages until WG catches a typed exception instead.

### CLI

```bash
hornlab-waveguide config.toml -o waveguide.msh
```

- `-o`, `--output`: output `.msh` path (overrides `output.path`).
- `--summary PATH`: write the JSON build summary to a file.
- `--print-summary`: print the JSON build summary. stdout then carries only
  the JSON; progress lines and native (OpenCASCADE) output go to stderr.
- `--allow-large-mesh`: explicitly permit output above `mesh.max_triangles`.
- `--step PATH`: write the CAD model as STEP; skips the mesh unless `-o` is
  also given. A STEP-only run's `--print-summary`/`--summary` JSON describes
  the STEP file (`step_path`, `body`, `n_faces`, `volume_mm3`,
  `bounding_box_mm`, `throat_opened`, `units`). `--step-keep-throat` keeps the
  driver membrane in the body.

Exit status: `0` success; `2` invalid config, refused build, or an unreadable
or unwritable file (one `hornlab-waveguide: error:` line on stderr); `1` any
other failure, with its traceback -- that is a bug, not bad input.

### Physical tags

Physical tags are solver-facing API. Consume tags and names, never Gmsh
surface counts. Source: `hornlab_mesher.tags.PhysicalGroup` /
`PHYSICAL_NAMES`.

| Tag | Name | Meaning |
| --- | --- | --- |
| `1` | `SD1G0` | Rigid waveguide wall. |
| `2` | `SD1D1001` | Primary source surface. |
| `3` | `SD2G0` | Enclosure wall. |
| `4` | `I1-2` | Acoustic interface surface. |
| `8` | `mid_chamber` | Reserved: mid-driver chamber. |
| `9` | `mid_port_interior` | Reserved: port interior. |
| `10` | `mid_port_exit_left` | Reserved: left port exit (a source tag). |
| `11` | `mid_port_exit_right` | Reserved: right port exit (a source tag). |
| `12` | `mouth_aperture` | Infinite-baffle Rayleigh aperture cap. |

Tags 8-11 are not emitted by the builds in this package; WG's multi-source
(cardioid) path assigns them, so their numbers and names must not be reused.

### Live preview (`hornlab.preview/1`)

```python
from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry

preview = build_preview_geometry(config, PreviewOptionsV1(lod="fine"))
```

`build_preview_geometry(config, options)` returns render geometry, never a
solve mesh: `PreviewGeometryV1(surfaces=[PreviewSurfaceV1, ...], metadata)`.
Every surface carries positions, counter-clockwise indices, analytic normals
and optional curvature for one role (`horn.inner`, `horn.outer`, `mouth_rim`,
`source_cap`, `wall.*`, `enclosure.*`). `metadata["api_version"]` is
`"hornlab.preview/1"`; `metadata_version` moves with additive metadata
changes. Fidelity per role is reported as requested vs achieved; the achieved
chord is measured against the triangles actually emitted.

The preview degrades rather than refuses in three cases, each reported in
`metadata["warnings"]`: a folded outer wall is drawn only when no part of the
outer surface intersects the inner surface. The warning names the design type
and the folded facets' axial z range in the mesher coordinate frame (mm). Its
reversed triangles are wound to their normals and counted in the surface's
`foldedTriangles` metadata. `foldedTriangleIndices` is a sorted list of zero-based
triangle ordinals in that surface's emitted `indices.reshape(-1, 3)` buffer;
it is present only when folds were rewound, and its length equals
`foldedTriangles`. These IDs also appear in geometry
`metadata["surface_metadata"][role]`. Rewinding makes the folds look healthy in
a back-face Normals view: consumers should highlight these IDs explicitly.
An enclosure on a reduced-quadrant model is
not drawn (only the full 1234 model has one); and a caller-supplied
`max_normal_step_deg` below 1 degree with no `max_vertices` is bounded at
200,000 vertices per surface. All explicit sub-degree normal requests, including
those with a vertex cap, also have a 5 s monotonic deadline from build start,
shared across master levels and source-cap refinement. At measurement/refinement
checkpoints after expiry, the current valid grid is returned with a warning,
`refinement_budget.exhausted=true`, and
`refinement_time_limited=true` in affected fidelity records. When horn measurement
was interrupted, chord error is null and `measurement_complete=false`. Source-cap
measurements remain available when only its refinement stopped. The requested
accuracy is not claimed.
`refinement_budget.seconds` reports the budget even when it was sufficient.
This is a cooperative refinement guard, not a hard request timeout: canonical
sampling, surface assembly and a measurement already in progress may add time.
LOD requests without an explicit normal override do not use this guard.

The same names are importable from `hornlab_mesher.preview.api`, which is the
path WG uses; both paths are supported. `preview/api.py` is the orchestrator;
the implementation lives beside it in `contract.py` (dataclasses, metadata
validation, orientation proof), `primitives.py`, `source_cap.py`,
`enclosure.py`, `horn.py` and `fidelity.py`, all internal. `preview.api`
keeps importing `_lod_config` and `_guiding_curve_warnings` for WG.

### CAD and WGLink

- `write_step(geometry, path, ...)`, `write_step_from_config(config, path, *,
  open_throat=True)` and `CadInfo` (`hornlab_mesher.cad`, also root exports):
  STEP export of the same resolved geometry a build meshes.
- `write_wglink`, `read_wglink`, `WgLinkIdentity`, `WgLinkSourceInterface`
  (root exports): the WGLink bundle WG writes for the Fusion add-in and reads
  back. `write_wglink` refuses an output path that already exists (a
  `MesherError`, the earlier bundle left intact): write each bundle to a new
  path and replace it yourself. The add-in mirrors `hornlab_mesher.cad._SOURCE_INTERFACE_KEYS` and
  `_SOURCE_PATCH_POLICIES` in `wglink_bundle.py`; change both together.

### Pre-mesh sizing (`hornlab_mesher.mesh_sizing`)

`Region`, `estimate_mesh_cost`, `estimate_triangle_count`, `graded_size_mm`,
`role_size_mm`, `valid_f_max_hz`, `feasibility_from_ram_gb`,
`matrix_ram_bytes`, `solve_seconds_per_freq` and the `ROLE_*` constants.
**This module must stay pure Python** (standard library only): the Fusion
add-in imports it inside Fusion's embedded interpreter. `estimate_solve_cost`
(`hornlab_mesher.cost`) prices a measured triangle count.

## Integration API

Each row is used in production by the consumer named. Contract tests live in
`tests/test_wg_consumer_contracts.py` unless another file is given.

| Name | Consumer | Used for |
| --- | --- | --- |
| `config_builder.build_geometry_params` | WG | Resolve formula/mode/params (sizing, symmetry, export). Returns `(params, formula, mode)`. |
| `config_builder.build_point_grid` | WG | Point grid for sizing and symmetry; reads `grid_n_phi`, `grid_n_length`, `inner_points`, `outer_points` (flat lists, phi-major). |
| `config_builder.resolve_geometry` | WG | Export identity hash of `.geometry`; CAD plan from `.sampling_metadata["geometrySampleAngularSegments"/"geometrySampleLengthSegments"]`. |
| `viewport.build_viewport_geometry_from_config` | WG | Inner grid for export; `grid["inner_points"]` reshapes to `(grid_n_phi, grid_n_length + 1, 3)`. |
| `profile_sampling.ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY` | WG | Densify morph corner arcs for an export measurement reference; adds samples only. |
| `normals.validate_orientation` | WG | Mesh integrity report (`tests/test_orientation_validation.py`). |
| `tags.PhysicalGroup` | WG | Tag numbers above. |
| `geometry.PointGridHornGeometry` | WG | STEP export of imported fixtures. |
| `step_import.postprocess_mesh`, `run_occ_healing_fallbacks`, `OCC_HEALING_FALLBACKS`, `StepFaceGroup`, `StepLabelSelector`, `detect_symmetry_planes` | WG | Imported-CAD meshing (`tests/test_step_import.py`). |
| `step_import.parse_named_shell_faces`, `parse_styled_face_groups`, `advanced_face_order` | WG | STEP face labels and face order (`tests/test_step_text.py`). |
| `step_import.gmsh_surface_tags`, `gmsh_surface_geometries`, `anchor_surface_order` | WG | Recover STEP face order after OCC healing. |
| `step_import.mesh_frequency_validation` | WG | Per-source valid band; WG's solver reads `per_source[...]["effective_max_valid_frequency_hz"]`. |
| `step_import.REDUCED_ORIENTATION_MIRRORED_PARENT` | WG | Feature flag, read with `getattr`; its absence selects WG's fallback. |
| root `auto_cut_occ_geometry`, `OccAutoCutResult`, `OccSurfaceGroup`, `OccSurfaceRole`, `OccSurfaceSelector` | WG | Symmetry auto-cut of imported CAD (`tests/test_step_prepare.py`). |
| root `Region`, `estimate_mesh_cost`, `estimate_solve_cost` | WG | Imported-mesh cost estimate. |
| `grid_resample.normalized_arc_positions`, `resample_point_grid` | add-in | WGLink bundle resampling script (`tests/test_grid_resample.py`). |
| `mesh_sizing.*` | add-in | Sizing inside Fusion (see above; `tests/test_mesh_sizing.py`). |
| `cli.load_config`, `config_builder.resolve_geometry` | add-in | Development spike scripts only. |

## Internal names consumers reach into

Not promises, but a rename breaks these:

| Name | Where | Why |
| --- | --- | --- |
| `preview.api._lod_config` | WG `server/preview/core.py` (comment) | WG documents that the preview overwrites `mesh.angular_segments`/`length_segments` and drops their camel-case aliases. Pinned in `tests/test_wg_consumer_contracts.py`. |
| `preview.api._guiding_curve_warnings` | WG test (`hasattr` skip guard) | Detects a mesher with the guiding-curve saturation warning. |
| `profile_morph._guiding_curve_target_radius` | WG test | Expected mouth radius in a translation test. |
| `builders._occ.SURFACE_FIT_MODES` | WG test | Valid `surface_fit` values. |
| `hornlab_mesher.preview.api`, `.viewport`, `.normals`, `.tags`, `.config_builder` module paths | WG `server/mesh/prewarm.py` | Imported by name at startup to warm the first interaction. |

## Root exports with no consumer

`hornlab_mesher.__all__` has 75 names. WG imports 12 of them from the root.
Another 15 are used by a consumer through their defining module rather than
the root: `CadInfo`, `write_step`, `write_step_from_config`, `PhysicalGroup`,
`build_from_config`, `build_geometry_params`, `normalized_arc_positions`,
`resample_point_grid`, and `ROLE_RADIATING`, `ROLE_SHADOW`, `ROLE_THROAT`,
`estimate_triangle_count`, `graded_size_mm`, `role_size_mm`,
`valid_f_max_hz` (the add-in's tests, from `hornlab_mesher.mesh_sizing`).
The remaining 48 have no user in WG (production or tests) or the add-in:

- Quality: `ChordDeviationReport`, `ElementShapeReport`, `QualityGateResult`,
  `FAIL_CHORD_DEVIATION_MM`, `FAIL_P1_ANGLE_DEG`, `SLIVER_ANGLE_DEG`,
  `WARN_CHORD_DEVIATION_MM`, `WARN_P1_ANGLE_DEG`, `chord_deviation_report`,
  `element_shape_report`, `evaluate_quality_gate`, `gmsh_sicn`,
  `mesh_quality_report`
- Direct builds and geometry: `build_mesh`, `build_mesh_with_info`,
  `load_mesh`, `MeshInfo`, `MeshDensity`, `OsseHornGeometry`,
  `RosseHornGeometry`, `CrossSection`, `Enclosure`, `HornEnclosure`,
  `build_osse_waveguide`, `compute_osse_inner_points`,
  `compute_osse_profile_points`, `compute_rosse_profile_points`,
  `derive_datums`
- CAD / auto-cut helpers: `WgLinkInfo`, `SOURCE_INTERFACE_FEATURE`,
  `PlaneSymmetryVerdict`, `DEFAULT_AUTO_CUT_GRID`,
  `DEFAULT_AUTO_CUT_TOLERANCE_REL`, `DEFAULT_SYMMETRY_SNAP_BAND_MM`,
  `evaluate_occ_plane_symmetry`, `millimetres_to_step_units`,
  `remap_surface_tags`, `sample_occ_surface_points`,
  `snap_symmetry_plane_vertices`, `resample_grid_onto_existing`
- Sizing: `MeshCostEstimate`, `SolveCostEstimate`, `ROLE_NEAR_FIELD`,
  `ROLE_SOURCE`, `SPEED_OF_SOUND_M_S`, `VALIDATION_EPW`
- `MesherError`, `load_config` (used in this repository's own tests and
  docs only)

Recommendation, not yet done: keep `build_mesh`/`build_mesh_with_info`,
`load_mesh`, `MeshDensity`, `OsseHornGeometry`, `MesherError` and
`load_config` as documented direct-build API; move the quality report types,
the auto-cut helpers other than `auto_cut_occ_geometry`, and
`build_osse_waveguide`/`compute_osse_inner_points` (unreferenced even by this
repository's tests) out of `__all__` behind their defining modules, with one
release of deprecation warnings on root access. Removing any of them is a
breaking change for an unknown third-party caller, so it belongs in a
versioned release, not a pin move.

### Direct mesh builds

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
`build_mesh_with_info(...)` returns `(path, MeshInfo)`, collected at write
time. `MeshDensity.max_triangles` applies the same full-domain-equivalent
guard as config builds; `MeshDensity.allow_large_mesh=True` overrides it.
Buildable geometry: `OsseHornGeometry` and
`hornlab_mesher.geometry.PointGridHornGeometry`. `RosseHornGeometry` is a
profile-parameter dataclass, not a buildable geometry; use the config path.
`load_mesh(path)` reads a mesh back as `MeshInfo`.

## Internal

Everything not listed above, in particular:

- modules under `hornlab_mesher.profile_*` and `hornlab_mesher.builders.*`
  (builders may change surface splits, helper names and kernel strategy while
  preserving physical tags and mesh behaviour);
- `hornlab_mesher.density`, `hornlab_mesher.normals` internals,
  `hornlab_mesher.preview.fidelity`;
- underscored names anywhere, including the underscored `config_builder`
  helpers `hornlab_mesher.cli` re-exports for this repository's own tests.

## Compatibility promise

- Supported configs continue to build or fail explicitly.
- Final meshes keep the documented physical tag meanings.
- Written meshes are in metres unless `scale_to_metres=False`.
- Meshes are postprocessed and orientation-validated before delivery.
- Integration API rows change only together with their consumer.

It does not freeze Gmsh surface counts, internal helper names or private
surface construction strategies.

### Design dimensions

Preview `geometry.metadata.dimensions_mm` describes the **resolved design size**:
`mouth_opening: [W, H]` (the terminating design aperture, even when the curve rolls back),
`horn_overall: [W, H, D]`, and `enclosure_overall: [W, H, D]` only when an
enclosure exists. All values are millimetres; W/H/D are x/y/z bounding extents
(maximum minus minimum). The horn includes its modelled offset wall, rim and
rear return/plate; the enclosure is reported separately. An enclosed horn has
no separate offset shell in the canonical model. Source-cap geometry describes
an acoustic boundary and is excluded from the material dimensions.

These are full-object extents, independent of origin, symmetry reduction,
visibility, render LOD. R-OSSE depth uses the largest
axial excursion anywhere on the curve, including the wall and rear plane,
not the terminating station's coordinate. The enclosure uses the same outer
box bounds, depth clamp and whole-mm rounding as the mesh/CAD builder.

`dimensions_sampling` states `method: "resolved-design-geometry"` and
`lod_independent: true`. Values bound the point grids from `resolve_geometry`,
including repaired outer-wall controls. They are design sizes, not measurements
of fitted CAD surfaces, exported STEP solids or meshes. Exported surfaces on
morphed mouths can differ from the design by the surface-fitting tolerance.
See [surface fitting](config-schema.md#meshsurface_fit) for the fitting modes,
measured errors and limitations. The acoustic sampling criteria bound angular
chords to twice the local mesh target, smooth angular sagitta to 5% of it, and
axial chords to half of it; these are fitting criteria, not a universal bound
on exported dimensional error. Interpolation between controls can overshoot,
and approximate fits can shrink. The readouts promise no STEP measurement
accuracy. Global Scale acts on the inner geometry
before the unscaled wall thickness is applied.

When a morph target exists, `dimensions_requested_mm.mouth_opening: [W, H]`
reports its requested target before implicit sizing and the no-shrink floor,
with global Scale applied. Width and height expressions are evaluated at the
horizontal and vertical cardinal azimuths. Zero means an implicit target.
The effective design aperture remains in `dimensions_mm.mouth_opening`.
For example, an unscaled OSSE with a raw round mouth of 348.579 mm and a
320 × 240 mm rectangle request without shrinkage reports requested
`[320, 240]` and effective `[348.579…, 348.579…]`. Consumers should show both
when they differ, rather than presenting the request as the resolved size.

Only fine and inspection frames compute the canonical measurement. A bounded
process cache stores readouts and failures by an owned snapshot of the full
supplied design config; concurrent settled frames share one resolution.
When snapshot ownership succeeds, rendering, the content key and measurement
use the same owned design for the entire preview call. If snapshotting fails,
the renderer retains its existing config handling and dimensions are unavailable.
Repeated settled frames reuse the cached outcomes. Coarse frames never resolve geometry: they
look up this exact design and otherwise return `dimensions_mm: null` with
`dimensions_status: "pending"`. A cached measurement carries status `"current"`.
A canonical resolution failure carries status `"unavailable"`, null dimensions,
and `dimensions_error`. Pending has no error. Visibility and LOD options do not
participate in the measurement identity. Full-object measurement resolves all
quadrants at the origin; it does not reflect reduced preview samples.

Ordinary exceptions during config snapshotting, identity encoding, cache lookup,
result copying, canonical resolution, requested-target parsing, metadata validation,
cache publication and eviction produce unavailable dimensions while preserving
preview surfaces and pre-existing metadata. Publication/eviction failures give
the owner and waiting requests the same unavailable outcome, remove the in-flight
entry and permit a later retry. Renderer exceptions retain their existing behavior.
Process-control exceptions (`KeyboardInterrupt`, `SystemExit`) propagate. The
measurement does not repeat the preview's outer-wall or enclosure-clamp warnings. Consumers should
hide the readouts for older meshers with absent dimension keys; pending frames
may retain the same document's last canonical values, labelled as updating.
