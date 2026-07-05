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
| `formula` | `profile.formula`, `profile.type` | `OSSE` | Accepted values are `OSSE`, `R-OSSE`, `ROSSE`, `ICW`, and experimental `LOOKUP`. `ROSSE` normalizes to `R-OSSE`. |
| `mode` | `mesh.mode` | `freestanding` | Accepted values are `freestanding`, `free-standing`, `free`, `bare`, `inner`, `open`, `infinite-baffle`, `ib`, `baffle`, `enclosure`, and `enclosed`. |
| `simType` | imported `ABEC.SimType` | none | When `mode` is omitted: `1` selects `infinite-baffle`, `2` selects `freestanding`. Text imports default it to `1` (`2` when an enclosure is present), matching ATH. |
| `scale` | imported `Scale` | `1.0` | Multiplies every linear geometry dimension after profile evaluation; resolutions stay in raw millimetres. |
| `output.path` | top-level `path`, `output_path`, CLI `-o` | none | Required by the CLI unless `-o/--output` is passed. |

If enclosure depth is positive, mode becomes `enclosure` even when `mode` is
omitted. `enclosure` mode requires `enclosure.depth_mm > 0`. `infinite-baffle`
mode builds the inner surface, source, and a planar mouth-aperture interface
(`I1-2`) with no outer wall or rear cap.

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
| `throat_ext_length_mm` | `throatExtLength` | `0.0` |
| `throat_ext_angle_deg` | `throatExtAngle` | `0.0` |
| `slot_length_mm` | `slotLength` | `0.0` |
| `driver_throat_diameter_mm` | `driverThroatDiameter`, `driverThroatDiameterMm` | none |
| `driver_throat_diameter_in` | `driverThroatDiameterIn` | none |
| `waveguide_throat_diameter_mm` | `waveguideThroatDiameter`, `waveguideThroatDiameterMm` | none |
| `waveguide_throat_diameter_in` | `waveguideThroatDiameterIn` | none |

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

R-OSSE-only keys:

| Canonical TOML/JSON key | Aliases | Default |
| --- | --- | --- |
| `R_mm` | `R` | `150.0` |
| `tmax` | none | `1.0` |
| `m` | none | formula default when omitted |
| `r` | none | formula default when omitted |
| `b` | none | formula default when omitted |

For R-OSSE with throat extension enabled, `tmax` samples the normalized total
profile including the extension, slot, and main R-OSSE curve. Values below
`1.0` therefore truncate before the final mouth point.

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
| `angular_segments` | `angularSegments` | `64` | Normalized to an ATH-compatible multiple by the profile sampler. |
| `corner_segments` | `cornerSegments` | `0` | Grows the angular point budget for rounded-rectangle morphs; the corner arc itself always carries four profiles per quadrant. |
| `length_segments` | `lengthSegments` | `32` | Produces `length_segments + 1` axial rings. |
| `sampling_mode` | `samplingMode` | `uniform` or `zmap` | Defaults to `zmap` when `z_map_points` is set. Text imports default to `ath-default-zmap`. |
| `vertical_offset_mm` | `verticalOffset` | `0.0` | Rigid +y translation applied after `scale`. |
| `ath_parity_sampling` | `athParitySampling` | `false` | Forces `ath-default-zmap`. |
| `z_map_points` | `zMapPoints`, `zmapPoints`, `ZMapPoints` | none | Full sample map or x,y control pairs in `[0, 1]`. |
| `wall_thickness_mm` | `wall_thickness`, `wallThickness` | `6.0` freestanding (`5.0` for text imports), `0.0` otherwise | Forced to `0.0` for `bare`, `enclosure`, and `infinite-baffle`. |
| `quadrants` | none | `1234` | `1`, `12`, `14`, and `1234` are supported by the sampler. |
| `throat_res_mm` | `throat_res`, `throatResolution` | `4.0` (`5.0` for text imports) | Mesh density, not grid shape. |
| `mouth_res_mm` | `mouth_res`, `mouthResolution` | `26.0` (`8.0` for text imports) | Mesh density, not grid shape. |
| `rear_res_mm` | `rear_res`, `rearResolution` | `25.0` (`15.0` for text imports) | Mesh density, not grid shape. |
| `subdomain_slices` | `subdomainSlices` | empty | Comma/list of point-grid ring indices for interfaces. Imported ATH `Mesh.SubdomainSlices` are shifted by one (ATH slice `k` is grid ring `k + 1`; the last slice is the mouth). |
| `interface_offset_mm` | `interfaceOffset` | `0.0` | Comma/list of interface protrusion depths. A single offset without slices places the interface at the mouth ring. Imported ATH configs that set slices but omit the offset use ATH's 5 mm default. |
| `interface_res_mm` | `interface_res`, `interfaceResolution` | falls back to `mouth_res_mm` | Mesh density for interface surfaces; ATH treats `Mesh.InterfaceResolution` as obsolete. |
| `preserve_grid` | `preserveGrid` | `false` | Forces faceted point-grid wall surfaces instead of grouped OCC patches. Leave false for fast grouped-wall enclosure topology unless a config explicitly needs point-grid wall faces. |
| `scale_to_metres` | `scaleToMetres` | `true` | Final `.msh` units are metres when true. |
| `max_frequency_hz` | `maxFrequencyHz`, `maxFrequency`, `f_max_hz` | none | Frequency-aware sizing: clamps each resolution to `c / (epw_role * f)` so the band stays resolved. The mm knobs still apply where finer. |
| `elements_per_wavelength` | `elementsPerWavelength` | `6.0` | Generic target used for groups without a role override and for the `mesh_report` validity figures. |
| `throat_epw` | `throatEpw` | `8.0` | Elements-per-wavelength at the throat (strongest, most detailed field). |
| `mouth_epw` | `mouthEpw` | `6.0` | Elements-per-wavelength at the mouth; the inner wall grades from the throat value to this. |
| `rear_epw` | `rearEpw` | `2.5` | Elements-per-wavelength on shadowed rear/outer surfaces, which contribute little to the radiated field. |
| `interface_epw` | `interfaceEpw` | `6.0` | Elements-per-wavelength on subdomain interfaces. |
| `speed_of_sound_m_s` | `speedOfSound` | `343.0` | Speed of sound for frequency-aware sizing. |
| `curvature_segments` | `curvatureSegments` | `0` | Gmsh `MeshSizeFromCurvature` segments per full circle; `0` disables curvature-adaptive refinement. |

`BuildResult` reports `quadrants`, the matching `native_symmetry_plane` solver
flag for reduced grids (`1` -> `yz+xz`, `12` -> `xz`, `14` -> `yz`), and a
`mesh_report` with per-group edge statistics plus `valid_f_max_hz` -- the
highest frequency each surface group resolves at the configured
elements-per-wavelength. Downstream solves should clamp or warn from it.
Note the report applies the generic target to every group: on freestanding
horns the rigid-wall tag mixes the inner wall with the deliberately coarser
rear/outer surfaces, so its strict figure is conservative.

Sampling modes accepted by the profile layer:

- `uniform`, `linear`, `canonical`, `default`
- `ath`, `ath-parity`, `ath-zmap`, `ath-default`, `ath-default-zmap`,
  `default-zmap`

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

See `docs/geometry-contract.md` for target-shape semantics.

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
  `Throat.Ext.Length`, `Throat.Ext.Angle`, `Slot.Length`, `Rot`, and R-OSSE
  `R`, `m`, `b`, `r`, `tmax`. `Length` is mandatory for OSSE imports.
- Mesh: `Mesh.AngularSegments`, `Mesh.CornerSegments`,
  `Mesh.LengthSegments`, `Mesh.WallThickness`, `Mesh.VerticalOffset`,
  `Mesh.Quadrants`, `Mesh.ThroatResolution`, `Mesh.MouthResolution`,
  `Mesh.RearResolution`, `Mesh.SubdomainSlices` (shifted by one onto grid
  rings), `Mesh.InterfaceOffset`, `Mesh.InterfaceResolution`,
  `Mesh.SamplingMode`, and `Mesh.ZMapPoints`.
- Morph: `Morph.*`, `MORPH.*`, `[Morph]`, and `[MORPH]` for target shape,
  width, height, corner radius, rate, fixed part, and shrinkage
  (`AllowShrinkage` accepts ATH boolean literals).
- Guiding curve: `GCurve.*`, `GCURVE.*`, `[GCurve]`, and `[GCURVE]`.
- Enclosure: `Mesh.Enclosure.Depth`, `EdgeRadius`, `EdgeType`,
  `FrontResolution`, `BackResolution`, and `Spacing` as four comma-separated
  margins.
- Source: `Source.Shape` (translated from the ATH enum, where 1 = cap and
  2 = flat disc, to the internal 1 = cap / 0 = flat disc convention),
  `Source.Radius`, `Source.Curv`, and `Source.VelocityProfile`.

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
