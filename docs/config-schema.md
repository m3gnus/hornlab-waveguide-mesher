# Config Schema

This package accepts TOML, JSON, and imported ATH-style text configs. TOML and
JSON use the same section names. Text configs are parsed by
`hornlab_mesher.config_parser` into the same internal shape, then normalized by
`hornlab_mesher.config_builder.build_geometry_params`.

## File Types

- `.toml` and `.tml`: parsed as TOML.
- `.json`: parsed as JSON.
- `.cfg` and `.txt`: parsed as imported ATH/WG-style text.

Unsupported extensions fail before any geometry is built.

## Top-Level Keys

| Key | Aliases | Default | Notes |
| --- | --- | --- | --- |
| `formula` | `profile.formula`, `profile.type` | `OSSE` | Accepted values are `OSSE`, `R-OSSE`, `ROSSE`, `ICW`, `FREEFORM`, and experimental `LOOKUP`. `ROSSE` normalizes to `R-OSSE`. |
| `mode` | `mesh.mode` | `freestanding` | Accepted values are `freestanding`, `free-standing`, `free`, `bare`, `inner`, `open`, `infinite-baffle`, `ib`, `baffle`, `enclosure`, and `enclosed`. |
| `simType` | imported `ABEC.SimType` | none | When `mode` is omitted: `1` selects `infinite-baffle`, `2` selects `freestanding`. Text imports default it to `1` (`2` when an enclosure is present), matching ATH. |
| `scale` | imported `Scale` | `1.0` | Multiplies every linear geometry dimension after profile evaluation; resolutions stay in raw millimetres. |
| `output.path` | top-level `path`, `output_path`, CLI `-o` | none | Required by the CLI unless `-o/--output` is passed. |

If enclosure depth is positive, mode becomes `enclosure` even when `mode` is
omitted. `enclosure` mode requires `enclosure.depth_mm > 0`, and explicit
`freestanding` mode requires positive wall thickness (use `bare` for an
inner-only open horn). `infinite-baffle`
mode builds the coupled interior-BEM/Rayleigh-aperture surface: inner wall plus
source cap plus a planar mouth aperture cap tagged `mouth_aperture`. The mouth
rim lies exactly on z=0, the cavity lies in z <= 0, and all triangles use one
consistent interior-domain winding: source normals point +z while the aperture
normals point -z into the cavity. It has no `I1-2` mouth interface, outer wall,
baffle skin,
wall thickening, rear cap, enclosure box, or geometry in front of the baffle
plane. The aperture cap reuses the wall rim curves and defaults to mouth density
via `aperture_res_scale = 1.0`; larger values coarsen its interior.

## Sections

Accepted TOML/JSON sections:

- `profile` or `parameters`
- `mesh`
- `enclosure`
- `cross_section` or `crossSection`
- `morph` or `MORPH`
- `gcurve`, `GCurve`, or `GCURVE`
- `source` or `Source`
- `output`

Keys may live in their natural section or, for many legacy aliases, at the top
level. Section values win only by lookup order in `config_builder.py`; avoid
duplicate aliases with conflicting values.

## Profile Keys

Shared OSSE/R-OSSE keys:

| Canonical TOML/JSON key | Aliases | Default |
| --- | --- | --- |
| `r0_mm` | `r0` | `12.7` |
| `a_deg` | `a` | `60.0` |
| `a0_deg` | `a0` | `15.5` (`0` for text imports, the ATH default) |
| `k` | imported `OS.k`, `Term.k` | `1.0` |
| `q` | none | `0.995` for OSSE, `1.0` for R-OSSE |
| `s1` | imported block `s1` | `0.0` (mm/degree, finite, 0 through 10) |
| `s2` | imported block `s2` | `0.0` (1/mm, finite, 0 through 10) |
| `throat_ext_length_mm` | `throatExtLength` | `0.0` |
| `throat_ext_angle_deg` | `throatExtAngle` | `0.0` |
| `slot_length_mm` | `slotLength` | `0.0` |
| `driver_throat_diameter_mm` | `driverThroatDiameter`, `driverThroatDiameterMm` | none |
| `driver_throat_diameter_in` | `driverThroatDiameterIn` | none |
| `waveguide_throat_diameter_mm` | `waveguideThroatDiameter`, `waveguideThroatDiameterMm` | none |
| `waveguide_throat_diameter_in` | `waveguideThroatDiameterIn` | none |

`s1` and `s2` stretch the main axial coordinate by
`s1 * degrees(atan(s2 * x))`, leaving its radius unchanged. Either zero disables
stretch exactly; inactive normalization omits both keys and retains the legacy
geometry serialization, equality, hashes and cache identity. Active pairs
include both keys. Both must be plain finite numbers in the inclusive range
`[0, 10]`. Native JSON strings, expression strings and per-azimuth forms are
refused with `ConfigError` saying “per-azimuth throat stretch is not supported
yet”, including expressions that evaluate to zero and a pair with a zero
companion. ATH text numeric tokens are accepted as numbers. The bound limits
added displacement to 900 mm and keeps the map finite for finite coordinates.
They belong inside `R-OSSE`/`OSSE` ATH blocks and are refused on other profile
families (an ICW seed may contain numeric coefficients). Total-length helpers
and `L` remain unstretched sampling parameters; physical dimensions come from
the point grid.

One composition refusal rule applies to native JSON/dict configs and ATH text:
active OSSE slots, rotation with a prefix, R-OSSE rotation or explicit `Length`,
and guiding curves with a prefix or rotation are unsupported. With active
stretch, Slot.Length, Rot, throat extension length/angle and every guiding-curve
input must be plain finite numbers. Expressions, including zero identities,
raise `ConfigError`: “throat stretch does not support per-azimuth <name> yet”.
Composition activity uses exact numeric comparisons with zero. In-block and
top-level OSSE rotation obey the same rule. Discontinuous joins, floating-point
loss of axial ordering and nonadjacent meridian intersections or contacts,
including source/driver and mouth endpoints, also raise `ConfigError` before a
consumer runs. Resolve, both preview levels, solve mesh and STEP share that
error contract. With active stretch, solve and STEP use matching quadrant
patches and axial knot weighting; stretch-off retains the previous fitting path.

See [the geometry contract](geometry-contract.md#r-osse-s--os-se-s-throat-stretching)
for the measured ATH evidence and finite meridian-validation tolerance. With
stretch inactive, the only existing import change is top-level `Rot` beside an
OSSE block (honored if no in-block `Rot`). Numeric and expression-valued slots
and rotations otherwise retain the base import mapping, without classifying
expressions as zero. The existing inactive OSSE-block slot discrepancy against
ATH V2025-12 is recorded in the geometry contract. Supplied s1/s2 entries,
including null, are scanned at top level and in every native mapping section
before precedence; FREEFORM and ICW refuse every such entry. An ICW seed may
contain numeric OSSE/R-OSSE stretch coefficients.

Driver adapter keys are convenience inputs for OSSE/R-OSSE. When both driver
and waveguide throat diameters are provided, `r0` anchors the main waveguide
throat radius and the extension tapers backward to the driver throat. The
extension is derived from either `throatExtAngle` or `throatExtLength`. If both
extension length and angle are provided, they must reach the requested
waveguide throat diameter.

OSSE-only keys:

| Canonical TOML/JSON key | Aliases | Default |
| --- | --- | --- |
| `L_mm` | `L` | `120.0` (mandatory for text imports) |
| `n` | none | `4.0` |
| `s` | none | `0.0` (`0.7` for text imports, the ATH default) |
| `rot_deg` | `rot` | `0.0` |
| `h` | imported `OS.h` | `0.0` |

`h` adds a half-sine bulge `h * sin(pi * t)` in millimetres over the whole
axial length, extension and slot included; it vanishes at the throat and the
mouth. `OS.h` is not an ATH key (ath.exe ignores it); the text importer honours
it because Waveguide Generator writes it. With an active guiding curve the
coverage angle is solved so that the bulged wall meets the curve.

OSSE refuses `L <= 0`, `n <= 0` or `q <= 0` while the termination term is on (`s != 0`), and, in total length mode,
`Slot.Length >= Length`; each of these used to switch part of the profile off
silently.

R-OSSE-only keys:

| Canonical TOML/JSON key | Aliases | Default |
| --- | --- | --- |
| `R_mm` | `R` | `150.0` |
| `tmax` | none | `1.0` |
| `m` | none | formula default when omitted |
| `r` | none | formula default when omitted |
| `b` | none | formula default when omitted |

`tmax` truncates the main R-OSSE curve at `t = tmax`, and must be positive.
A throat extension or slot is prepended to the truncated curve without changing
it: the main curve and its mouth are the same with and without the prefix, as in
ATH.

ICW keys:

| Canonical TOML/JSON key | Aliases | Default |
| --- | --- | --- |
| `r0_mm` | `r0` | `12.7` |
| `a0_deg` | `a0` | `15.5` |
| `termination` | none | `flat_baffle` |
| `L_mm` | `L` | `120.0` for `flat_baffle`; omitted for rollback unless configured |
| `R_mm` | `R` | `150.0` for `flat_baffle`; omitted for rollback unless configured |
| `r_aperture` | none | none |
| `x_aperture` | none | none |
| `depth` | none | none |
| `x_setback` | none | none |
| `coverage_angle` | `coverage_angle_deg` | none |
| `hold_start` | none | none |
| `hold_end` | none | none |
| `kappa0` | none | none |
| `n_coeff` | none | ICW kernel default |
| `theta1` | `theta1_deg` | none |
| `kappa_abs_max` | none | none |
| `dkappa_ds_abs_max` | none | none |
| `theta_max_deg` | none | none |
| `pin_mouth_radius` | none | false |
| `icw_seed` | none | none |
| `icw_coeffs` | none | none |
| `icw_S` | none | none |

ICW is not available through ATH text import. Configure it through TOML, JSON,
or direct dict input. ICW always samples uniformly in normalized arc length and
rejects `sampling_mode = "zmap"` / `z_map_points`.

### FREEFORM

Select this family with `formula = "FREEFORM"`. FREEFORM is available through
TOML, JSON, and direct dict input, not ATH-style text import. Its H and V
meridians and cross-section schedule live inside `[profile]`:

```toml
formula = "FREEFORM"

[profile]
profileH = { points = [[0.0, 12.7], [60.0, 80.0], [120.0, 160.0]], throatAngleDeg = 15.5, mouthAngleDeg = 70.0 }
profileV = { points = [[0.0, 12.7], [60.0, 60.0], [120.0, 110.0]], throatAngleDeg = 15.5, mouthAngleDeg = 60.0 }
crossSections = [
  { t = 0.0, shape = "circle" },
  { t = 0.4, shape = "rounded_rectangle", cornerRadiusMm = 5.9 },
  { t = 1.0, shape = "rounded_rectangle", cornerRadiusMm = 5.9 },
]
```

`profileH` and `profileV` are both required. Each profile accepts these keys:

| Key | Default | Validation and meaning |
| --- | --- | --- |
| `points` | required | List of 2-64 anchors. A row is `[z, r]` or `[z, r, angleDeg]`; `z` and `r` are millimetres. Values must be finite, radii must be positive, and z values must be strictly increasing. A four-element row (the removed per-anchor `strength`) is refused. |
| `throatAngleDeg` | top-level/profile `a0_deg` or `a0`, otherwise `15.5` | Endpoint tangent angle in degrees from the +z axis, in `[-90, 90]`. |
| `mouthAngleDeg` | direction of the last anchor chord | Endpoint tangent angle in degrees from the +z axis, in `[-90, 90]`. |

Any other profile key is refused as unknown; that includes the removed
`throatTangentScale` and `mouthTangentScale`. Tangent speeds are solved
automatically.

An anchor-row `angleDeg` overrides the corresponding automatically derived
tangent. Interior anchor angles must be strictly inside `(-90, 90)`; endpoint
angles may equal either limit. An endpoint row with an explicit angle overrides
its block-level angle.

The two planes must have the same first and last z values (within `1e-9` mm)
and equal first radii (within `1e-6` mm). The first anchor is the throat; use
`z = 0` mm for the throat coordinate. The last anchor is the planar mouth, and
the H and V mouth radii may differ.

Every spline segment is rejected when its radius leaves the range of its two
anchor radii (beyond a small tolerance); positive-radius and forward-z guards
also apply. The removed `overshootPolicy` key is refused, whether it is given in
`[profile]` or at the top level. The one FREEFORM profile-level policy key is:

| Key | Default | Accepted values | Meaning |
| --- | --- | --- | --- |
| `inflectionPolicy` | `warn` | `warn`, `reject` | Reports significant reverse-curvature spans, or rejects the first such span. The removed value `allow` is not accepted. |

`crossSections` is a list of 2-32 axial shape stations. If omitted, it defaults
to a circle at `t = 0` and an ellipse at `t = 1`. Station `t` is normalized
axial position in `[0, 1]`; values must be strictly increasing, the first must
be `0`, and the last must be `1`. The first shape must be `circle` or
`ellipse`; `circle` is accepted at any station.

| `shape` value | Per-shape keys | Validation and meaning |
| --- | --- | --- |
| `circle` | none | Another spelling of the exponent-2 outline (`ellipse`); it is circular only where the local H/V radii are equal. |
| `ellipse` | none | Exponent-2 outline using the local H/V radii as semi-axes. |
| `superellipse` | `exponent` (default `2.0`) | Exponent must be in `[2, 16]`. |
| `rounded_rectangle` | required `cornerRadiusMm` | Absolute corner radius in millimetres. `cornerRatio` was removed and is rejected. |

For a rounded-rectangle station, `cornerRadiusMm` must be positive and must be
between 2% and 100% of `min(r_H, r_V)` at that station. Validation also covers
the station's adjacent blend spans: its smootherstep contribution multiplied
by the requested corner radius may not exceed the local minimum semi-axis. A
hold has full contribution throughout its span, so its radius must remain valid
over that entire span. The rejection identifies the binding `t`/z location and
the maximum feasible station radius.

Adjacent stations with the same complete descriptor hold that outline between
their two positions; for example, two `rounded_rectangle` stations with the
same `cornerRadiusMm` hold one absolute corner radius. Their `t` values are
still different because duplicate station positions are invalid. Different
descriptors blend with C2 smootherstep,
`6u^5 - 15u^4 + 10u^3`, over their station span.

FREEFORM uses the shared mesh sampling keys, with these additional rules:

- `angular_segments` (default `64`) sets the base azimuth budget;
  `corner_segments` (default `0`) adds rounded-corner sampling pressure.
- `length_segments` (default `32`) sets the base axial sampling. Every H/V
  anchor z and cross-section station is merged into the axial grid exactly.
- `sampling_mode` may be uniform (`uniform`, `linear`, `canonical`, or
  `default`) or a custom z-map (`zmap`, `z-map`, `custom`, `custom-zmap`, or
  `custom-z-map`). `z_map_points` selects the custom map. ATH parity sampling
  and ATH sampling modes are rejected for FREEFORM.
- Acoustic fitting refines those geometry samples against chord and sagitta
  limits derived from `throat_res_mm` and `mouth_res_mm`; it does not rewrite
  the requested mesh element sizes.

Ingest rejects non-convex station blends. Freestanding builds with positive
`wall_thickness_mm` also run a full-surface principal-curvature guard and then
check the generated outer offset for normal flips and self-intersections. A
positive rounded-cap `source.source_radius_mm` must be at least the FREEFORM
throat radius. Active morph targets and guiding curves are rejected because
`crossSections` owns the outline; FREEFORM likewise rejects non-default legacy
cross-section exponent/aspect, rotation, `h`, throat extensions, and slot
length.

Formula-specific keys are rejected when used with the other formula. For
example, `R_mm`, `m`, `r`, `b`, and `tmax` are invalid with `OSSE`, while
OSSE-only `n`, `s`, and `rot_deg` are invalid with `R-OSSE`. ICW rejects the
OSSE/R-OSSE shape keys at top level; put a seed formula inside `icw_seed` when
you want ICW to fit an existing OSSE/R-OSSE meridian.

Numeric profile keys may be numbers or expression strings. Expression strings
are evaluated later by the profile layer where supported.

## Cross Section

Use `[cross_section]` or `[crossSection]`.

| Key | Aliases | Default | Notes |
| --- | --- | --- | --- |
| `exponent` | `profile.exponent`, top-level `cross_section_exponent` | `2.0` | Superellipse exponent. |
| `aspect_ratio` | `aspectRatio` | `1.0` | Width-to-height scaling. |

## Mesh Keys

| Canonical TOML/JSON key | Aliases | Default | Notes |
| --- | --- | --- | --- |
| `angular_segments` | `angularSegments` | `64` | Geometry sampling for the fitted acoustic surface; it does not directly set BEM element count. |
| `corner_segments` | `cornerSegments` | `0` | Grows the angular point budget for rounded-rectangle morphs; the corner arc itself always carries four profiles per quadrant. |
| `length_segments` | `lengthSegments` | `32` | Geometry sampling for the fitted acoustic surface; it does not directly set BEM element count. |
| `sampling_mode` | `samplingMode` | `uniform` or `zmap` | Defaults to `zmap` when `z_map_points` is set. Text imports default to `ath-default-zmap`. |
| `vertical_offset_mm` | `verticalOffset` | `0.0` | Rigid +y translation applied after `scale`. |
| `ath_parity_sampling` | `athParitySampling` | `false` | Forces `ath-default-zmap`. |
| `z_map_points` | `zMapPoints`, `zmapPoints`, `ZMapPoints` | none | Full sample map or x,y control pairs in `[0, 1]`. |
| `wall_thickness_mm` | `wall_thickness`, `wallThickness` | `6.0` freestanding (`5.0` for text imports), `0.0` otherwise | Forced to `0.0` for `bare`, `enclosure`, and `infinite-baffle`. |
| `quadrants` | none | `1234` | `1`, `12`, `14`, and `1234` are supported by the sampler. |
| `throat_res_mm` | `throat_res`, `throatResolution` | `4.0` (`5.0` for text imports) | Mesh density, not grid shape. |
| `mouth_res_mm` | `mouth_res`, `mouthResolution` | `26.0` (`8.0` for text imports) | Mesh density, not grid shape. |
| `rear_res_mm` | `rear_res`, `rearResolution` | `15.0` | Mesh density, not grid shape. |
| `aperture_res_scale` | `apertureResolutionScale`, `aperture_cap_coarsening`, `apertureCapCoarsening` | `1.0` | Infinite-baffle aperture-cap interior size multiplier relative to `mouth_res_mm`; the welded rim keeps mouth density. |
| `subdomain_slices` | `subdomainSlices` | empty | Comma/list of requested point-grid ring indices for interfaces. If the acoustic fit changes the axial grid density, indices are relocated to preserve their normalized axial positions. Imported ATH `Mesh.SubdomainSlices` are shifted by one (ATH slice `k` is grid ring `k + 1`; the last slice is the mouth). |
| `interface_offset_mm` | `interfaceOffset` | `0.0` | Comma/list of interface protrusion depths. A single offset without slices places the interface at the mouth ring. Imported ATH configs that set slices but omit the offset use ATH's 5 mm default. |
| `interface_res_mm` | `interface_res`, `interfaceResolution` | falls back to `mouth_res_mm` | Mesh density for interface surfaces; ATH treats `Mesh.InterfaceResolution` as obsolete. |
| `topology` | `topology_mode`, `topologyMode` | `acoustic` | `acoustic` separates geometry samples from BEM topology. `legacy` retains ATH/parity patch and grid semantics. |
| `surface_fit` | `surfaceFit` | `auto` | B-spline fitting mode for the acoustic wall patches. `auto` is `interpolate` everywhere it is meshable and `approximate` on FREEFORM. `approximate` hands the sampled grid to OCC as control points; `interpolate` solves for poles whose surface passes through the grid. An explicit `interpolate` is refused on FREEFORM profiles. See below. |
| `preserve_grid` | `preserveGrid` | `false` | Legacy faceted point-grid topology. Requires `topology = "legacy"`; rejected in ordinary acoustic mode. |
| `scale_to_metres` | `scaleToMetres` | `true` | Final `.msh` units are metres when true. |
| `max_triangles` | `maxTriangles` | `18000` | Full-domain-equivalent triangle ceiling. Estimated gross overruns fail before meshing and realized overruns fail before return. Sizes are never rewritten. |
| `allow_large_mesh` | `allowLargeMesh` | `false` | Explicitly bypasses both triangle-limit checks without changing any mm target. The CLI equivalent is `--allow-large-mesh`. |

`BuildResult` reports `quadrants`, the matching `native_symmetry_plane` solver
flag for reduced grids (`1` -> `yz+xz`, `12` -> `xz`, `14` -> `yz`; full
grids -> `None`), and a `mesh_report` with realized per-group edge statistics.
The mesher does not derive or report frequency validity. Infinite-baffle
supports `1`, `12`, `14`, and `1234`; its reduced-domain
open edges are expected to lie only on the matching native cut plane(s), so
`native_check_open_edges` remains true.
Removed frequency/EPW and curvature-sizing keys raise `ConfigError` with a
migration hint; they are never silently ignored.

Sampling modes accepted by the profile layer:

- `uniform`, `linear`, `canonical`, `default`
- `ath`, `ath-parity`, `ath-zmap`, `ath-default`, `ath-default-zmap`,
  `default-zmap`

### `mesh.surface_fit`

Three values are accepted; anything else raises `ConfigError`.

- `auto` (default) resolves to `interpolate` on every profile that can be meshed
  with it, and to `approximate` on FREEFORM. Naming either mode explicitly pins
  it.
- `approximate` hands the sampled profile grid to `occ.addBSplineSurface` as
  control points. OCC treats them as poles, not as points the surface passes
  through, so the meshed wall hangs systematically *inside* the sampled one. The
  bias does not shrink with mesh refinement.
- `interpolate` instead solves for the poles whose surface passes through the
  sampled grid. The solve is separable and exact to machine precision, adds no
  control points, and therefore leaves the triangle count essentially unchanged.

`auto` became the default on the evidence below. Measured against a dense
analytic surface over twelve ATH reference configs, `interpolate` moves the
median `rms x triangles` -- the equal-cost accuracy comparison, since deviation
scales as one over triangle count -- from 0.93x of ATH's to 1.24x, improving
eleven of the twelve. Triangle count moves by at most 4%. Two configs regress and
are recorded rather than hidden: `Tritonia-V`'s maximum deviation rises from
0.806 mm to 1.057 mm, and `test4_morph_only_noshrink` loses 9% of per-triangle
efficiency at a `Morph.FixedPart` junction while its maximum improves 1.252 mm to
0.784 mm.

Measured on `examples/osse-freestanding.toml`, inner acoustic wall against a
4000-segment analytic meridian:

| `surface_fit` | triangles | rms | p95 | p99 |
| --- | --- | --- | --- | --- |
| `approximate` | 2800 | 0.280 mm | 0.378 mm | 0.432 mm |
| `interpolate` | 2872 | 0.161 mm | 0.200 mm | 0.261 mm |

Two limits apply. (A third, that every symmetry-reduced domain tore open at
the source, was a defect rather than a limit and is fixed -- see
`docs/builder-invariants.md`.)

- `interpolate` is refused on FREEFORM profiles. Their deliberate creases make
  the interpolating patch fit unmeshable — Gmsh grinds past ten minutes inside
  `mesh.generate` rather than failing — so the geometry constructor raises
  instead of hanging.
- Only the acoustic wall changes fit. The outer shell always keeps
  `approximate`, because `outer_topology` splices the rear rim on as a
  deliberate sharp corner that a cubic interpolant overshoots.

The default stays `approximate` because `interpolate` moves the nodes of every
acoustic mesh.

## Experimental LOOKUP Profiles

`formula = "LOOKUP"` accepts `lookupProfile` or `lookup_profile` in TOML/JSON
as an ordered list of `[z_mm, r_mm]` samples. This is a compatibility input for
archived/generated configs, not a stable public mesh-builder API.
- `zmap`, `z-map`, `custom`, `custom-zmap`, `custom-z-map`

## Enclosure Keys

Use `[enclosure]`. A positive `depth_mm` enables enclosure topology.

| Canonical TOML/JSON key | Aliases | Default |
| --- | --- | --- |
| `depth_mm` | `depth`, `encDepth` | `0.0` |
| `space_l_mm` | `space_l`, `left_margin_mm` | `25.0` |
| `space_t_mm` | `space_t`, `top_margin_mm` | `25.0` |
| `space_r_mm` | `space_r`, `right_margin_mm` | `25.0` |
| `space_b_mm` | `space_b`, `bottom_margin_mm` | `25.0` |
| `edge_mm` | `edge`, `encEdge` | `18.0` |
| `edge_type` | `edgeType`, `encEdgeType` | `1` |
| `plan_type` | `planType`, `encPlanType` | `1` |
| `plan_n` | `planN`, `encPlanN` | `2.0` |
| `depth_margin_mm` | `depth_margin`, `encDepthMargin` | `1.0` |
| `front_mesh_size_mm` | `frontMeshSize`, `enc_front_resolution`, `encFrontResolution` | none |
| `back_mesh_size_mm` | `backMeshSize`, `enc_back_resolution`, `encBackResolution` | none |

`plan_type`: `1` rounded rectangle, `2` ellipse, `3` superellipse.
`edge_type`: `1` rounded fillet, `2` chamfer.

Enclosure front/back mesh sizes may be scalar values or comma-separated
quadrant lists. Missing or invalid quadrant entries fall back to
`mouth_res_mm`.

## Morph Keys

Use `[morph]` or `[MORPH]`.

| Canonical TOML/JSON key | Aliases | Default |
| --- | --- | --- |
| `morph_target` | `morphTarget` | `0` |
| `morph_width_mm` | `morphWidth` | `0` |
| `morph_height_mm` | `morphHeight` | `0` |
| `morph_corner_mm` | `morphCorner` | `0` |
| `morph_rate` | `morphRate` | `3.0` |
| `morph_fixed` | `morphFixed` | `0` |
| `morph_allow_shrinkage` | `morphAllowShrinkage` | `0` |

`morph_width_mm` and `morph_height_mm` are full widths (the target
half-dimensions are half of them). ATH text imports spell the two dimensions
`Morph.TargetWidth` and `Morph.TargetHeight`. A negative `morph_rate` is
refused; rates from 0 to 1 are accepted. A throat extension and a slot are never
morphed. ATH text imports follow ath.exe instead for two morph details: an
absent `Morph.FixedPart` means 0.2 (ATH's effective default), and the slot is
morphed like the rest of the horn. See `docs/geometry-contract.md` for
target-shape semantics.

## Guiding Curve Keys

Use `[gcurve]`, `[GCurve]`, or `[GCURVE]`.

Guiding curves are supported for OSSE only. R-OSSE configs with an active
guiding curve are rejected instead of silently ignoring the keys.

| Canonical TOML/JSON key | Aliases | Default |
| --- | --- | --- |
| `gcurve_type` | `gcurveType` | `0` |
| `gcurve_width_mm` | `gcurveWidth` | `0` |
| `gcurve_aspect_ratio` | `gcurveAspectRatio` | `1` |
| `gcurve_dist` | `gcurveDist` | `0` |
| `gcurve_rot_deg` | `gcurveRot` | `0` |
| `gcurve_sf` | `gcurveSf`, `gcurveSF` | empty string |
| `gcurve_se_n` | `gcurveSeN` | `3` |
| `gcurve_sf_a` | `gcurveSfA` | `1` |
| `gcurve_sf_b` | `gcurveSfB` | `1` |
| `gcurve_sf_m1` | `gcurveSfM1` | `4` |
| `gcurve_sf_m2` | `gcurveSfM2` | none |
| `gcurve_sf_n1` | `gcurveSfN1` | `2` |
| `gcurve_sf_n2` | `gcurveSfN2` | `2` |
| `gcurve_sf_n3` | `gcurveSfN3` | `2` |

## Source Keys

Use `[source]` or `[Source]`.

| Canonical TOML/JSON key | Aliases | Default | Notes |
| --- | --- | --- | --- |
| `source_shape` | `sourceShape` | `1` | `0` builds a flat disc/sector source; `1` builds a rounded cap source. |
| `source_radius_mm` | `sourceRadius` | `-1` | Positive values override the automatic cap radius. |
| `source_curv` | `sourceCurv` | `0` | `-1` flips rounded source cap curvature direction. |

`sourceVelocityProfile` is imported from text configs but is not currently used
by the mesh builder.

`sourceVelocity` (ATH `Source.Velocity`) selects the velocity direction of the
driving elements rather than any geometry: `1` is normal to the element surface
and `2` is axial (pistonic motion along z). The cap or disc is meshed
identically either way, so the exported `.msh` cannot carry the distinction and
a solver reading only the mesh would drive an axial model normally. Only `1`,
ATH's own default, is accepted; `2` is refused with an explicit error. The
accepted value is recorded on the parsed config for consumers that do model
velocity direction.

## ATH Text Import Boundary

Text import supports OSSE and R-OSSE blocks plus selected flat ATH keys. ICW is
not part of the ATH text format and is rejected there. The
parser strips semicolon comments and accepts block syntax such as:

```text
OSSE = {
  Length = 120
  Coverage.Angle = 60
}
```

Imported text mappings include:

- Topology: `ABEC.SimType` (1 = infinite baffle, the ATH default when the key
  is omitted; 2 = free standing; an enclosure implies 2; explicit 1 plus an
  enclosure is rejected).
- Geometry transforms: flat `Scale` (multiplies all linear geometry after
  profile evaluation) and `Mesh.VerticalOffset` (+y translation after scale).
- Profile: `Coverage.Angle`, `Throat.Angle`, `Throat.Diameter`,
  `Length`, `Term.n`, `Term.s`, `Term.q`, `Term.k`, `OS.k`,
  `Throat.Ext.Length`, `Throat.Ext.Angle`, `Slot.Length`, `Rot`, `OS.h`
  (a Waveguide Generator extension; ATH ignores it), and R-OSSE
  `R`, `m`, `b`, `r`, `tmax`. `Length` is mandatory for OSSE imports.
- Mesh: `Mesh.AngularSegments`, `Mesh.CornerSegments`,
  `Mesh.LengthSegments`, `Mesh.WallThickness`, `Mesh.VerticalOffset`,
  `Mesh.Quadrants`, `Mesh.ThroatResolution`, `Mesh.MouthResolution`,
  `Mesh.RearResolution`, `Mesh.SubdomainSlices` (shifted by one onto grid
  rings), `Mesh.InterfaceOffset`, `Mesh.InterfaceResolution`,
  `Mesh.SamplingMode`, and `Mesh.ZMapPoints`.
- Morph: `Morph.*`, `MORPH.*`, `[Morph]`, and `[MORPH]` for
  `Morph.TargetShape`, `Morph.TargetWidth`, `Morph.TargetHeight`,
  `Morph.CornerRadius`, `Morph.Rate`, `Morph.FixedPart`, and
  `Morph.AllowShrinkage` (which accepts ATH boolean literals).
- Guiding curve: `GCurve.*`, `GCURVE.*`, `[GCurve]`, and `[GCURVE]`.

Names outside those two namespaces' ATH vocabularies — `Morph.Width`,
`Morph.Height`, `GCurve.Distance` — are ignored with a warning naming the value
that went unused and the canonical key, because ATH ignores them too and
meshes the config anyway. Rename them (`Morph.Width` -> `Morph.TargetWidth`,
`Morph.Height` -> `Morph.TargetHeight`, `GCurve.Distance` -> `GCurve.Dist`) to
get the geometry they were written for; leaving them alone keeps ATH's
geometry. Unrecognised `Mesh.*` keys are refused instead, since a mesh key that
silently falls back to a default changes the mesh with nothing to show for it.
- Enclosure: `Mesh.Enclosure.Depth`, `EdgeRadius`, `EdgeType`,
  `FrontResolution`, `BackResolution`, and `Spacing` as four comma-separated
  margins.
- Source: `Source.Shape` (translated from the ATH enum, where 1 = cap and
  2 = flat disc, to the internal 1 = cap / 0 = flat disc convention),
  `Source.Radius`, `Source.Curv`, `Source.Velocity` (direction enum; only 1,
  normal to the element surface, is supported), and `Source.VelocityProfile`.
  Multi-source models are refused: `Source.Contours`, `LFSource*`, and the
  indexed `Source.Velocity.<n>` form.

Text imports inject ATH's own defaults for omitted keys instead of the native
TOML/JSON defaults: `Throat.Angle` 0, `Term.s` 0.7, `Mesh.WallThickness` 5,
`Mesh.ThroatResolution` 4, `Mesh.MouthResolution` 8, `Mesh.RearResolution`
15, and `Morph.CornerRadius` 35 when a morph target is set. The default
sampling mode is `ath-default-zmap` (for OSSE a cubic bezier with control
points `(0.5, 0.1)` and `(0.5, 0.95)` fitted against ATH reference grids).

Deliberate deviation: `Mesh.Quadrants` keeps the full-circle default (`1234`)
instead of ATH's quarter default (`1`). Quarter meshes are fully supported
and `hornlab-metal-bem` solves them via its explicit
`native_symmetry_plane="yz+xz"` solve flag, but the `.msh` format carries no
symmetry marker and the solver loaders do not auto-detect reduced meshes — a
quarter mesh solved without the flag produces silently wrong free-space
results. Until the mesh format and solver loaders share an explicit symmetry
contract, full meshes stay the safe import default. Set `Mesh.Quadrants = 1`
explicitly for quarter grids and pass the matching symmetry flag to the
solver.

Solver-only and output keys are intentionally ignored: `ABEC.MeshFrequency`,
`ABEC.NumFrequencies`, `ABEC.f1`, `ABEC.f2`, `ABEC.Polars:*`, `ABEC.Abscissa`,
`Report`, `GridExport:*`, and `Output.*` (the CLI owns output paths).

Unsupported geometry keys fail explicitly instead of being approximated:
`Throat.Profile` other than 1 (OS-SE), `Rollback.*`, `Mesh.RearShape` other
than 1, and `Mesh.ThroatSegments`.

The text parser is an import adapter only. It does not build geometry, infer
unsupported ATH objects, or preserve unknown sections for later use. Unsupported
geometry families should fail explicitly rather than being approximated.
