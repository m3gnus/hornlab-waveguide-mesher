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
- `freestanding`: `enclosure is None`, `outer_points is not None`, and the
  requested wall thickness is positive. Config-driven freestanding requests
  with zero/negative thickness fail instead of silently becoming `bare`.
- `enclosure`: `enclosure is not None`.

These topology selectors are mutually exclusive. Direct API calls that combine
`infinite_baffle` with `enclosure` or `outer_points`, or combine an enclosure
with a freestanding `outer_points` shell, fail at geometry construction instead
of silently selecting one mode by precedence.

Bare mode builds the inner horn surface and a source cap. Infinite-baffle
mode (ABEC.SimType = 1, the ATH default for imported text configs) builds the
coupled interior-BEM/Rayleigh-aperture mesh: source cap plus inner wall plus a
planar aperture cap across the mouth. The mouth rim lies exactly on z=0, the
cavity lies in z <= 0, and the aperture cap is physical tag `12` with normals
toward -z into the cavity. Source normals point +z and the whole closed surface
uses one consistent negative-volume interior-domain winding. It has no planar
`I1-2` mouth interface, outer wall, baffle skin,
wall thickening, rear cap, enclosure box, or geometry in front of the baffle
plane. Freestanding mode builds an inner wall, outer wall, mouth rim, rear cap,
and source cap.
Enclosure mode builds the inner horn, source cap, optional interfaces, and
enclosure surfaces around the mouth.

Bare-shell winding is an acoustic-boundary contract, not a closed-solid
contract: source normals point from the throat into the horn, and inner-wall
normals point out of the wall material into the acoustic bore. Because the
mouth is open, its signed-volume indicator does not define either side.

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
- `mesh_algorithm`: the Gmsh 2D mesh algorithm the mode selects (see below).

Builders must not configure density, add physical groups, generate the final
mesh, scale units, repair triangle orientation, or write the output file.
`hornlab_mesher.mesher` owns those steps.

## Mesh Algorithm Selection

`mesh_algorithm` is not a user-facing override. Each point-grid build mode picks
the Gmsh 2D algorithm measured to suit its own surfaces, and `mesher` applies it
verbatim. `None` means "leave Gmsh on its own default", which is MeshAdapt (`1`).

| Build mode | `mesh_algorithm` | Gmsh algorithm |
| --- | --- | --- |
| `freestanding` | `6` | Frontal-Delaunay |
| `enclosure` | `5` | Delaunay |
| `bare` | `None` | MeshAdapt (Gmsh default) |
| `infinite-baffle` | `None` | MeshAdapt (Gmsh default) |

The legacy `wg_topology` freestanding path keeps `2` (Automatic), which it has
always used.

MeshAdapt is the slowest algorithm on the trimmed B-spline patches this mesher
builds, because its cost per triangle grows with surface count instead of
staying flat. Moving `freestanding` and `enclosure` off it is worth 2.94x and
1.46x at solver mesh densities.

`bare` deliberately stays on MeshAdapt. Its wall patches come from a fitted
spline surface whose parameterisation neither Delaunay front respects, so both
fronts trade triangle quality away for no speedup at all. On the FREEFORM bare
example at a quarter of its stock mesh resolutions the worst-case triangle angle
is 22.5 deg under MeshAdapt, 5.8 deg under Delaunay, and 4.7 deg under
Frontal-Delaunay, while the build time does not improve. `infinite-baffle` is
unmeasured and is therefore also left unchanged.

A builder that changes its algorithm must update
`tests/test_point_grid_contract.py::test_build_modes_select_their_measured_gmsh_algorithm`,
which pins the table above.

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
- `mouth_aperture`: aperture-cap interior resolution. The shared wall rim is
  still controlled by the `inner`/mouth boundary field; the cap interior uses
  `mouth_res_mm * aperture_res_scale` so coupled infinite-baffle solves do not
  over-mesh the smooth Rayleigh aperture.
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

Acoustic enclosure geometry suppresses cosmetic fillets and chamfers whose
across-feature length is smaller than the finest adjacent user target. A
quarter-round uses `pi * clamped_edge / 2`; a chamfer uses its face width. The
manufacturing request is retained in metadata, while the acoustic builder uses
the sharp enclosure path. Source, aperture, interface, physical-group, and
symmetry geometry is never suppressed.

Retained enclosure edge strips mesh at the finest touching front/back user
resolution. No `edge / N`, curvature, wavelength, or other geometry-derived
size is introduced. Distance-threshold grading remains at enclosure and mouth
seams, but its endpoints are strictly the two adjacent user mm targets. This
prevents needle-fan tears without refining below the user's finest value.

Global Gmsh size bounds come from the configured sizes. `MeshSizeMin` is a
defensive field floor at one quarter of the smallest target; it does not request
that size and cannot prevent shorter geometry-conformity edges. `MeshSizeMax`
is the largest user target. Point, curvature, and boundary-extension sizing are
disabled.

All builds use a full-domain-equivalent `max_triangles` guard (18,000 by
default). A fast-fail estimate rejects requests above twice the effective
limit before Gmsh; it is deliberately approximate and is not the budget
guarantee. The realized triangle count is checked authoritatively after
generation. The guard never rewrites mesh sizes. `allow_large_mesh=true` is
the explicit override for both checks.

Ordinary `acoustic` topology uses profile samples to fit a small number of OCC
patches; samples do not become mandatory BEM seams or vertices. `legacy`
topology remains available for ATH parity/export, and `preserve_grid` is valid
only in that explicit mode.

Because OCC B-spline patches approximate their control points, the config
builder raises internal angular/axial geometry sampling until chord and sagitta
errors are comfortably below the requested local mm element size. This affects
geometry fitting only, not final mesh density or triangle count. Very thin
freestanding shells below the stable acoustic feature floor are rejected with a
clear instruction to use bare mode or a resolvable wall thickness.

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
4. Repairs shared-edge consistency.
5. Orients closed surfaces by signed volume, but orients bare open walls from
   sparse normal references supplied by the builder's own axial/azimuthal
   parameterisation. The source cap remains anchored to `source_axis`.
6. Validates the applicable volume, edge, and source-normal contracts, and
   re-derives the bare-shell contract from the emitted triangles alone.
7. Scales coordinates to metres when requested.
8. Writes Gmsh 2.2 triangles with physical and geometrical tags.
9. Reloads the final file and validates required physical tags.

All generated meshes require consistent shared-edge winding. Bare source and
wall sheets may remain separate edge-connected components when Gmsh gives
their coincident throat rims different 1D discretisations; their semantic
orientations remain deterministic in either case.

A surface that has to weld to a fitted patch must be authored from that
patch's own curve, not merely from the same sample points. Gmsh welds these
seams by coincident mesh nodes, so a rim re-authored as a plain pole B-spline
welds to the wall only while the wall's edge *is* that curve -- true under
`surface_fit = "approximate"`, false under `"interpolate"`, whose wall edge
carries the interpolating poles. A closed wall avoids the question by filling
its caps on its own OCC boundary curves; an open sector cap cannot, because its
loop is closed by radial curves that start from the cap builder's own points,
so it reproduces the wall's throat isocurve pole-for-pole instead
(`builders/_occ.throat_boundary_curve`).

Bare shells are also checked independently of the repair that produced them,
because that repair depends on builder-supplied references and a lost or wrong
reference would otherwise ship silently. The check is the near-throat wall
area fraction whose normal faces the bore, measured about the source cap's own
centroid and net area vector; the build fails below `0.9`. Note that signed
volume cannot serve here even about a fixed material point: rollback profiles
(R-OSSE, ICW) hold the wall contract with a *positive* signed volume because
their wall curls back past the mouth plane, and symmetry-reduced bare shells
change its sign with the point it is taken about. The measured alignment is
`1.0` on every correctly wound bare shell and `0.0` on an inverted one, so the
threshold is not a tuned quantity. Coupled infinite-baffle
meshes have a stricter runtime contract: full domains must be watertight,
reduced-domain open edges must lie only on declared cut planes, tag `1`/`12`
must share a welded rim, and the entire surface must keep a consistent
negative-volume winding.

## Failure Policy

Unsupported modes, plan types, edge types, source shapes, sampling modes, or
geometry families should fail explicitly. Do not silently approximate a missing
topology, because downstream BEM setup depends on physical tags, normals, and
closed/open-domain semantics.
