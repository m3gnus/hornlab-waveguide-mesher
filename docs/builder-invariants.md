# Builder Invariants

This document records the handoff contracts between profile sampling, Gmsh/OCC
builders, mesh density, physical groups, and postprocessing.

## Point-Grid Shape

`PointGridHornGeometry.inner_points` is the canonical point-grid input.

- Shape is `(n_phi, n_length + 1, 3)`.
- Coordinates are finite `float64` values in millimetres.
- Axis order is `(x, y, z)`.
- Ring index `0` is the throat/source side.
- Ring index `n_length` is the mouth side.
- `n_phi >= 2` and `n_length + 1 >= 2`.

`outer_points`, when present, must use the same shape and units as
`inner_points`.

The CLI path receives flattened point-grid arrays from `profiles.build_point_grid`
and reshapes them with the expected `n_phi * (n_length + 1) * 3` size. A size
mismatch is a config/build error.

## Closed And Open Domains

`PointGridHornGeometry.closed` controls angular wrapping:

- `closed = True`: full 360 degree domain. Angular spans wrap from
  `n_phi - 1` to `0`.
- `closed = False`: open partial domain. Angular spans stop at `n_phi - 1`
  and do not wrap.

Open domains are snapped on symmetry planes before topology is built in the
WG-style freestanding and enclosure paths:

- First angular row is snapped to `y = 0`.
- Last angular row is snapped to `x = 0`.
- Final mesh vertices may also be snapped on `x` and `y` symmetry axes during
  postprocess for freestanding open-domain meshes.

Supported quadrant sampling is defined by the profile layer:

- `1`: first quadrant.
- `12`: upper half.
- `14`: right half.
- `1234`: full circle.

## Geometry Modes

The point-grid dispatcher selects topology from `PointGridHornGeometry.build_mode`:

- `bare`: `enclosure is None`, `outer_points is None`, not infinite baffle.
- `infinite-baffle`: `infinite_baffle` is set and `enclosure is None`.
- `freestanding`: `enclosure is None` and `outer_points is not None`.
- `enclosure`: `enclosure is not None`.

Bare mode builds the inner horn surface and a source cap. Infinite-baffle
mode (ABEC.SimType = 1, the ATH default for imported text configs) builds the
coupled interior-BEM/Rayleigh-aperture mesh: source cap plus inner wall plus a
planar aperture cap across the mouth. The mouth rim lies exactly on z=0, the
cavity lies in z <= 0, and the aperture cap is physical tag `12` with normals
toward +z. It has no planar `I1-2` mouth interface, outer wall, baffle skin,
wall thickening, rear cap, enclosure box, or geometry in front of the baffle
plane. Freestanding mode builds an inner wall, outer wall, mouth rim, rear cap,
and source cap.
Enclosure mode builds the inner horn, source cap, optional interfaces, and
enclosure surfaces around the mouth.

Supported point-grid source shapes are explicit: `source_shape = 0` builds a
flat throat disc/sector, and `source_shape = 1` builds a rounded throat cap.
Unsupported source shapes must raise instead of silently omitting physical tag
`2`.

## Surface Group Handoff

Builders return `BuiltGeometry`, not a mesh file. The important fields are:

- `surface_groups`: physical tag to Gmsh surface tags.
- `mesh_surface_groups`: density-role name to Gmsh surface tags.
- `axial_bounds_mm`: source-to-mouth axial range used by density formulas.
- `source_axis`: axis used by source normal validation.
- `enclosure_bounds`: enclosure bounding data for front/back interpolation.
- `symmetry_snap_axes`: axes that postprocess may snap to zero.
- `mesh_algorithm`: optional Gmsh 2D mesh algorithm override.

Builders must not configure density, add physical groups, generate the final
mesh, scale units, repair triangle orientation, or write the output file.
`hornlab_mesher.mesher` owns those steps.

## Physical Surface Groups

Physical tags are a public compatibility contract:

| Tag | Name | Meaning |
| --- | --- | --- |
| `1` | `SD1G0` | Rigid waveguide wall. |
| `2` | `SD1D1001` | Primary source surface. |
| `3` | `SD2G0` | Enclosure wall. |
| `4` | `I1-2` | Acoustic interface surface. |
| `12` | `mouth_aperture` | Infinite-baffle Rayleigh aperture cap. |

Every valid final mesh must contain at least tags `1` and `2`. Enclosure and
interface tags appear only when those surfaces are built. Tag `12` appears only
for coupled infinite-baffle meshes.

Surface splits are allowed to change inside a physical group when topology
needs more stable splines. Tag numbers and names must not drift casually.

## Density Ownership

`hornlab_mesher.density.configure_density` owns mesh-size fields. Builders only
name surfaces by density role.

Recognized role names include:

- `inner`: axial interpolation from throat to mouth resolution.
- `mouth`: axial interpolation from throat to mouth resolution.
- `mouth_aperture`: axial interpolation from throat to mouth resolution.
- `outer`: rear resolution for freestanding outer walls, otherwise axial
  interpolation.
- `throat_disc`: throat resolution.
- `rear`: rear resolution.
- `interface`: interface resolution, falling back to mouth resolution.
- `enclosure`: front/back z interpolation when enclosure bounds exist.
- `enclosure_edges_front`: front panel bilinear quadrant interpolation.
- `enclosure_edges_back`: back panel bilinear quadrant interpolation.

Density values are millimetres. Final mesh coordinates are scaled to metres by
default after meshing; density is configured before that scaling.

Global Gmsh size bounds come from the configured sizes unless explicitly set on
`MeshDensity`: minimum defaults to half the smallest positive size, maximum to
1.5 times the largest positive size.

## Enclosure Bounds

Enclosure builders should return bounds when enclosure density interpolation is
needed:

- `bx0`, `bx1`: x bounds for panel interpolation.
- `by0`, `by1`: y bounds for panel interpolation.
- `z_front`, `z_back`: front/back interpolation planes.

Without bounds, enclosure-side density falls back to mouth resolution for the
legacy enclosure role names.

## Orientation And Postprocess Boundary

Gmsh/OCC topology is not the final contract. After raw mesh generation,
`hornlab_mesher.mesher`:

1. Reads the raw Gmsh mesh with meshio.
2. Snaps requested symmetry planes.
3. Removes degenerate triangles.
4. Repairs global triangle winding.
5. Validates positive volume and source normal direction.
6. Scales coordinates to metres when requested.
7. Writes Gmsh 2.2 triangles with physical and geometrical tags.
8. Reloads the final file and validates required physical tags.

Orientation validation currently does not require watertightness or shared-edge
consistency for all meshes. If a mesh is watertight, inconsistent shared edges
are treated as an error, except coupled infinite-baffle meshes may carry the
explicit +z aperture normal by breaking winding consistency only on shared
wall-aperture tag `1`/`12` rim edges.

## Failure Policy

Unsupported modes, plan types, edge types, source shapes, sampling modes, or
geometry families should fail explicitly. Do not silently approximate a missing
topology, because downstream BEM setup depends on physical tags, normals, and
closed/open-domain semantics.
