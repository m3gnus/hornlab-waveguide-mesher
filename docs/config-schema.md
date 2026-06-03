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
| `formula` | `profile.formula`, `profile.type` | `OSSE` | Accepted values are `OSSE`, `R-OSSE`, and `ROSSE`. `ROSSE` normalizes to `R-OSSE`. |
| `mode` | `mesh.mode` | `freestanding` | Accepted values are `freestanding`, `free-standing`, `free`, `bare`, `inner`, `open`, `enclosure`, and `enclosed`. |
| `output.path` | top-level `path`, `output_path`, CLI `-o` | none | Required by the CLI unless `-o/--output` is passed. |

If enclosure depth is positive, mode becomes `enclosure` even when `mode` is
omitted. `enclosure` mode requires `enclosure.depth_mm > 0`.

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

Common keys for OSSE and R-OSSE:

| Canonical TOML/JSON key | Aliases | Default |
| --- | --- | --- |
| `r0_mm` | `r0` | `12.7` |
| `a_deg` | `a` | `60.0` |
| `a0_deg` | `a0` | `15.5` |
| `k` | none | `1.0` |
| `q` | none | `0.995` for OSSE, `1.0` for R-OSSE |
| `throat_ext_length_mm` | `throatExtLength` | `0.0` |
| `throat_ext_angle_deg` | `throatExtAngle` | `0.0` |
| `slot_length_mm` | `slotLength` | `0.0` |
| `driver_throat_diameter_mm` | `driverThroatDiameter`, `driverThroatDiameterMm` | none |
| `driver_throat_diameter_in` | `driverThroatDiameterIn` | none |
| `waveguide_throat_diameter_mm` | `waveguideThroatDiameter`, `waveguideThroatDiameterMm` | none |
| `waveguide_throat_diameter_in` | `waveguideThroatDiameterIn` | none |

Driver adapter keys are convenience inputs. When both driver and waveguide
throat diameters are provided, `r0` becomes the driver throat radius and the
throat extension is derived from either `throatExtAngle` or `throatExtLength`.
If both extension length and angle are provided, they must reach the requested
waveguide throat diameter.

OSSE-only keys:

| Canonical TOML/JSON key | Aliases | Default |
| --- | --- | --- |
| `L_mm` | `L` | `120.0` |
| `n` | none | `4.0` |
| `s` | none | `0.0` |
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
| `corner_segments` | `cornerSegments` | `0` | Adds rounded-rectangle morph corner samples. |
| `length_segments` | `lengthSegments` | `32` | Produces `length_segments + 1` axial rings. |
| `sampling_mode` | `samplingMode` | `uniform` or `zmap` | Defaults to `zmap` when `z_map_points` is set. |
| `ath_parity_sampling` | `athParitySampling` | `false` | Forces `ath-default-zmap`. |
| `z_map_points` | `zMapPoints`, `zmapPoints`, `ZMapPoints` | none | Full sample map or x,y control pairs in `[0, 1]`. |
| `wall_thickness_mm` | `wall_thickness`, `wallThickness` | `6.0` freestanding, `0.0` otherwise | Forced to `0.0` for `bare` and `enclosure`. |
| `quadrants` | none | `1234` | `1`, `12`, `14`, and `1234` are supported by the sampler. |
| `throat_res_mm` | `throat_res`, `throatResolution` | `4.0` | Mesh density, not grid shape. |
| `mouth_res_mm` | `mouth_res`, `mouthResolution` | `26.0` | Mesh density, not grid shape. |
| `rear_res_mm` | `rear_res`, `rearResolution` | `25.0` | Mesh density, not grid shape. |
| `subdomain_slices` | `subdomainSlices` | empty | Comma/list of point-grid slice indices for interfaces. |
| `interface_offset_mm` | `interfaceOffset` | `0.0` | Comma/list of interface protrusion depths. |
| `interface_res_mm` | `interface_res`, `interfaceResolution` | `12.0` | Mesh density for interface surfaces. |
| `preserve_grid` | `preserveGrid` | `false` | Used by the bare inner-surface builder. |
| `scale_to_metres` | `scaleToMetres` | `true` | Final `.msh` units are metres when true. |

Sampling modes accepted by the profile layer:

- `uniform`, `linear`, `canonical`, `default`
- `ath`, `ath-parity`, `ath-zmap`, `ath-default`, `ath-default-zmap`,
  `default-zmap`
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

Text import supports OSSE and R-OSSE blocks plus selected flat ATH keys. The
parser strips semicolon comments and accepts block syntax such as:

```text
OSSE = {
  Length = 120
  Coverage.Angle = 60
}
```

Imported text mappings include:

- Profile: `Coverage.Angle`, `Throat.Angle`, `Throat.Diameter`,
  `Length`, `Term.n`, `Term.s`, `Term.q`, `OS.k`, `Throat.Ext.Length`,
  `Throat.Ext.Angle`, `Slot.Length`, `Rot`, and R-OSSE `R`, `m`, `b`, `r`,
  `tmax`.
- Mesh: `Mesh.AngularSegments`, `Mesh.CornerSegments`,
  `Mesh.LengthSegments`, `Mesh.WallThickness`, `Mesh.Quadrants`,
  `Mesh.ThroatResolution`, `Mesh.MouthResolution`, `Mesh.RearResolution`,
  `Mesh.SubdomainSlices`, `Mesh.InterfaceOffset`,
  `Mesh.InterfaceResolution`, `Mesh.SamplingMode`, and `Mesh.ZMapPoints`.
- Morph: `Morph.*`, `MORPH.*`, `[Morph]`, and `[MORPH]` for target shape,
  width, height, corner radius, rate, fixed part, and shrinkage.
- Guiding curve: `GCurve.*`, `GCURVE.*`, `[GCurve]`, and `[GCURVE]`.
- Enclosure: `Mesh.Enclosure.Depth`, `EdgeRadius`, `EdgeType`,
  `FrontResolution`, `BackResolution`, and `Spacing` as four comma-separated
  margins.
- Source: `Source.Shape`, `Source.Radius`, `Source.Curv`, and
  `Source.VelocityProfile`.

The text parser is an import adapter only. It does not build geometry, infer
unsupported ATH objects, or preserve unknown sections for later use. Unsupported
geometry families should fail explicitly rather than being approximated.
