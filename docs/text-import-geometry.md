# Text import geometry

`parse_text_config` retains the versioned `_textImportVersion` field. Its
current value is `ath-2026-09a-v2`. JSON round trips and callers that copy the
parsed config must preserve it: geometry normalization retains it in the
profile parameter dictionary. An unknown version is refused rather than
reinterpreted. The geometry stamp identifies this import contract; the ATH
version echo selects only the sampling rules documented below. It does not claim full
feature, station allocation, topology, or acoustic parity.

Native configurations without the field keep their established dimensions
and no-shrink behavior. Adding the field deliberately selects the import
contract and its bounded refusals.

The direct CFG/CLI reader honors saved application comment stamps before
discarding comments: `; Waveguide Generator geometry-interpretation: native-v1`
selects native interpretation and either `ath-2026-08c-v1` or `ath-2026-09a-v2` selects imported geometry.
Unknown or conflicting stamps are refused. Unstamped external CFGs select the
import contract; historical `; Parameter config` or `; MWG config` headers
select native interpretation, consistently with the application reader. An
explicit stamp takes precedence over the historical header. Native saved CFGs
retain native profile/morph defaults and additive slot length; explicit
`Length.Mode` retains its authored override.
Saved stamped files and historical native files can combine a profile block
with the writer's flat formula aliases. Block fields take precedence across
aliases; missing block controls are read from the flat controls. Unstamped
external ATH retains its strict populated-block interpretation and refusals.

Stamps are read from top-level semicolon comments, including an inline comment
on a top-level control or block opener. Comments inside blocks, including a
block closer's inline comment, do not select interpretation. Historical dialect
headers retain the application's existing file-wide sniffing rule.

Saved stamped CFGs and historical native CFGs default to freestanding topology
when the simulation type is omitted. Unstamped external ATH retains its
infinite-baffle default. Explicit `Simulation.SimType` is accepted alongside
`ABEC.SimType`; when both occur, the ABEC key takes precedence. Constant numeric
`Throat.Diameter` expressions such as the application writer's `2*(4)` are
accepted; azimuth-varying diameter expressions remain unsupported.

Sparse v2 imported dictionary transport uses the same known defaults as the text
reader: implicit sampling uses the ATH axial map, mouth mesh resolution is 8 mm,
and a supplied morph target defaults to corner radius 35 mm, fixed part 0.2,
and slot morphing enabled. Explicit controls remain authoritative. These defaults
also reach the mesh density used to fit actual surfaces. Native omitted mouth
resolution retains the native 26 mm default. These defaults apply to the
supported OSSE/R-OSSE text profile families. Under v2, OSSE defaults to `a=45`,
`a0=0`, `s=0.5`; R-OSSE defaults to `R=200`, `a=40`, `a0=0`, `r=0.25`,
`m=0.9`, `b=0.4`, `q=3.5`, measured against ATH V2026-09a output.
Native authored geometry types
retain their own defaults even when provenance is carried alongside them.

Expressions use the scalar C99 function whitelist in
[the config schema](config-schema.md#parameter-expressions). ATH V2026-09a
probes establish the Gielis `sf(m,n1,n2,n3,a,b,p)` formula, natural `log`,
and half-away-from-zero `round`. The two-value functions `frexp` and `modf`
are refused during parsing and evaluation.

Current (v2, V2025-12+ rule) imported angular sampling rounds `AngularSegments`
up to a multiple of eight, for plain profiles and morphs alike, and
`CornerSegments` does not enter the count: 100 gives 104, and 64 stays 64 with
`CornerSegments` 4 or 8 (V2026-09a GridExport). The presence of integer `NAC`
keeps `AngularSegments` as written; its value does not affect the grid and is
ignored with a named warning. A rounded-rectangle morph corner gets
`max(3, CornerSegments - 1)` arc intervals, equal in arc angle, and five when
`CornerSegments` is absent; a wall share ending in .5 rounds up. Earlier mesher
versions added `CornerSegments` to the count and always used three arc
intervals.

The current default OSSE axial map uses a cubic Bézier from `(0,0)` to
`(1,1)` with controls `(0.5,0.1)`, `(0.5,0.92)`. For each equally spaced
abscissa it selects the first curve parameter on a 0.001 grid reaching that
abscissa. This replaces the approximate historical map in current imported OSSE
geometry. Explicit uniform/custom maps, R-OSSE's reference map and native
geometry keep their sampling rules. Current imported morphs reserve at least the
first segment even for `FixedPart=0`; the first positive station has zero
blend. Morph progress uses the axial map before throat stretch, so stretch
changes z while preserving the morphed x/y coordinates.

An ATH echo line `; Ath version Vyyyy-mm[x]` in the leading comment header
selects the axial interpretation at import: V2025-06 and older use the previous
OSSE station map and morph start (zero fixed part can start at the throat).
Their implicit rectangular morph dimensions use the raw ring before the corner
budget. Unstamped text and V2025-12 or newer use the current rule above. Stamps strictly
between V2025-06 and V2025-12 are unverified and use the current rule. The selected
rule is retained as a plain dictionary key and logged; native TOML/JSON keep
their existing interpretation. Evidence: V2025-12 and V2026-09a produce byte-identical
CORE and Solana grids, while all 23 archived configs echo V2025-06; the previous
sampler reproduces the four affected archive parity cases.

| Control | Imported text interpretation (v2) | Native authoring |
| --- | --- | --- |
| Omitted profile coefficients | OSSE `a=45`; R-OSSE `R=200`, `a=40`, `a0=0`, `r=0.25`, `m=0.9`, `b=0.4`, `q=3.5`. | Existing profile defaults. |
| RearShape 1 without a lip | A materially warped rear uses the measured throat normal-offset ring and a fan, preserving the analytic outer shell. Planar rears retain their construction. | Existing planar rear rule. |
| OSSE `Slot.Length` | A conical prefix at slope `tan(a0)`. `r0` is its starting radius; the main profile starts at `r0 + slot*tan(a0)`. The slot is included in `Length`; the main termination uses `Length - slot`. | A cylindrical prefix at `r0`, added to the body length unless an explicit total-length mode is selected. |
| R-OSSE `Slot.Length` | An axial translation of the main meridian, including signed lengths; no cylindrical slot rows and no body-parameter reallocation. With an extension, its ten pre-body rows stay in place and the body starts at `extension + slot`. | Existing cylindrical prefix and additive parameter allocation. |
| R-OSSE `Throat.Ext.Length` | Retains ten uniformly spaced extension segments before the complete authored body map. The extension tapers back from the main throat radius. Existing OSSE sampling and rotation rules are unchanged. | Existing native sampling and rotation behavior. |
| OSSE `Throat.Ext.Length` with `Rot` | The extension stays unrotated at its own angle; only the body rotates about its endpoint. Current rotated imports retain ten extension segments before the body map. | Existing native rotation behavior. |
| OSSE-block `Rot` | Ignored with a warning naming `OSSE.Rot`; only top-level `Rot` applies. | Existing authored precedence. |
| OSSE-block `r0` expression | Refused by name; ATH reads only a leading numeric token. Literal signed decimal/exponent numbers remain accepted. | Existing expression evaluation. |
| R-OSSE morph | Each meridian blends additively in radius with `((u - u0)/(1 - u0))^Rate`, `u = z/z_mouth` from the end of any extension. `FixedPart` is compared with `z/z_max` and starts at the first station at or past it, never the throat; a Circle target ignores it. Along the rollback `u > 1`, so the factor exceeds one; a start beyond the mouth's z gives negative factors (a fractional `Rate` there is refused by name, as ATH writes NaN). ATH builds no Superellipse morph on R-OSSE, and builds a Rectangle unmorphed when the depth varies around the mouth (a size-dependent tolerance, not pinned); a Circle it always morphs. For those two cases WG uses its own R-OSSE morph, whenever the depth varies at all: ATH's construction there balloons the wall far past the mouth. | Existing morph behavior. |
| Morph clamps | `Rate` below 1.01 is used as 1.01, `FixedPart` above 0.99 as 0.99, `TargetWidth`/`TargetHeight` above 1000 mm as 1000. An implicit rectangle half-size is `floor(mouth) + 1`. On OSSE a start on the mouth station moves the mouth alone onto the target. | `Rate` in `[0, 1)` and the other values as written. |
| `Mesh.Enclosure.Spacing` | Four margins in +x,+y,-x,-y order. | Left (-x), top (+y), right (+x), bottom (-y). |
| Circular `Morph.TargetShape=2` | Uses the largest sampled raw mouth radius, ignoring explicit target width/height and without rounding to a whole millimetre. | Honors explicit dimensions and retains the implicit dimension rule. |
| Rectangular morph with `Morph.AllowShrinkage=1` | Uses the requested target dimensions. | Uses the requested target dimensions. |
| Rectangular morph with default/false shrinkage and smaller target dimensions | V2 OSSE/R-OSSE uses ATH's sampled per-axis boundary expansion and retains the original meridians when rebuilding directions; see below. Saved v1 retains its earlier refusal. | Floors each target half-dimension at the corresponding raw extent. |

Imported OSSE slot construction has bounded physical-curve witnesses covering
different slot lengths, positive/zero/negative throat slopes, disabled and
enabled superellipse termination, and extension/rotation controls. Scalar and
vector evaluators use the same slot interpretation. Imported slots with an
active guiding curve are explicitly refused: that composition has a different
reference interpretation and remains unqualified. Existing stretch and adapter
composition restrictions remain in force. Native guiding-curve/slot behavior
is retained.

Physical-curve comparisons evaluate radius at exported axial coordinates;
equal station indices do not establish equal wall geometry. The committed
slot regression table is a small physical-profile witness, not a complete
reference archive or a full mesh/STEP parity qualification.

ATH OSSE imports now support `Mesh.Roundover = { Radius; Segments }` with
Segments default 5, and `Mesh.RearShape = 3` with `FrontOffset` default 40 mm
and `Step` default 25 mm. RearShape 1 with a lip follows ATH's selected normal
offset rings. These controls are retained as import-only mesh data across JSON
round trips; the native C6 roundover is unchanged. The complete authored grid,
including the nonmonotonic lip, supplies both legacy and acoustic mesh modes.
See [ATH import lip](mouth-roundover-contract.md#ath-import-lip) and
[Opus rear](ath-rear-import-contract.md#opus-rear-and-rearshape-1-with-an-ath-lip)
for rules, fixture evidence and named refusals. Preview tessellates the same
authored mesh patches at every LOD; it never recomputes the lip on a refined
body grid. Viewport exports retain the authored inner rows, independent rear
row count and apex, and optionally carry that tessellation. STEP retains a
named refusal because GEO patches have no faithful OCC transport yet. Existing imports without
an active lip or Opus rear retain their previous consumers. Planar rears keep
the existing plane bit for bit. V2 warped rears use the measured RearShape 1
throat (-5 mm shift, then the wall offset along the authored meridional
normal), joined to the analytic outer shell and closed by a fan. The m6_var_wall
throat differs from ATH by at most 0.000688 mm; simply extruding the analytic
outer throat misses by up to 0.398502 mm. Both mesh routes, preview and STEP
share the selected rear boundary.


The saved v1 contract retains the geometry preceding v2: the fitted OSSE axial
map, the previous morph start, and the corner budget only for rectangular
morphs. It also retains the earlier coefficient defaults (OSSE `a=60`;
R-OSSE `R=150`, `a=60`, `a0=15.5`, `r=0.4`, `m=0.85`, `b=0.2`, `q=1`),
Spacing in left/top/right/bottom order, OSSE-block rotation precedence,
rotation of the entire throat extension about the original throat, and
normal expression evaluation for OSSE-block `r0`. R-OSSE morph remains
accepted under its previous construction. Materially warped RearShape 1
walls retain their previous named refusal; such walls could not be built
through the v1 resolver. The OSSE text defaults `a0=0` and `s=0.5` already
precede these v2 changes; sparse v1 dictionary defaults remain unchanged.
An ATH version echo and NAC do not upgrade a v1 design. Import-only
lip/Opus controls require v2. Saved v1 CFGs retain their stamp when parsed;
fresh unstamped external CFGs receive v2. Unknown versions/rules are refused.
Native and v1 arrays are pinned to the preceding contract in
`tests/fixtures/rear-native.npz` and `rear-v1.npz`; floating coordinates are
compared with absolute tolerance 1e-9 mm and zero relative tolerance, while
triangle indices are exact. `tests/fixtures/ath-mismatches-v1.npz` pins the earlier coefficient, Spacing,
rotation, extension, morph and radius-expression grids against a515334
(with OSSE `s=0.5` explicit for the separately qualified default).
`rear-imported.npz` independently pins the current
imported inner/outer grids and exercised preview arrays.

Consumers constructing their own mesher dictionaries must retain these keys:

| Key | Location | Meaning |
| --- | --- | --- |
| `_textImportVersion` | Top level; carried into geometry params | `ath-2026-08c-v1` preserves saved geometry; `ath-2026-09a-v2` opts into the current import rules. |
| `_athAxialSamplingRule` | Top level; carried into geometry params | v2 accepts `ath-v2025-06` or `ath-v2025-12+` (default). This also selects the morph start and plain-profile corner budget. v1 keeps its historical rules regardless of this key. |
| `_athNacPresent` | Top level or `mesh`; carried into geometry params | Boolean NAC presence. With the current v2 sampling rule, true suppresses the corner addition; NAC's integer value is irrelevant to point-grid geometry. |

V2025-06's legacy angular rule keeps the corner addition for rectangular morphs
and excludes it for plain profiles. The archive parity gate retains its existing
checks. The initial UTF-8 BOM is stripped before reading the version echo.
For `ldexp`/`scalbn`, finite floating exponents are truncated toward zero when
converted to the C integer parameter. `logb` of either infinity returns positive
infinity; `round` preserves signed infinity.

For active import lips or Opus rears, `Scale` other than 1, active
`Throat.Ext.Length`/`Throat.Ext.Angle`, nonzero `Rot`, nonzero
`Mesh.VerticalOffset`, and `Source.Shape = 2` are refused by name pending ATH
probes. The boundary applies to parsed CFGs and dictionary transport alike.


The lip/Opus preview seeds structured ring lofts and preserves duplicated seam
vertices with one-sided patch normals. Authored patch boundary curves are
reported in `creaseFeatures`; no smoothness target is imposed across those
features. Refinement uses shared curve midpoints, balanced triangle cuts and
cached parametric measurements. It stops when the targets are met, no nodes
are added, edges reach a geometric size floor, or a resource bound is reached.
`vertex_cap_limited`, `iteration_bound_limited` and `refinement_stalled` are
separate flags; `refinement_limit` names the reason. The preview tests check
all emitted normal steps within patches, zero folds, closure and explicit
LOD vertex/time budgets independently of those flags.

## Remaining import blockers corrected in v2

Native Windows ATH V2026-09a captures establish these corrections. The
executable SHA-256 is
`4795b08e9b03f83489ec390cbba8abe0fa900aff2d87862f8fca5f014d003d8d`.
Exact inputs, full-ring profile exports and normalized-file hashes are in
[the blocker fixtures](../tests/fixtures/ath_import/ath-v2026-09a/blockers/README.md).
Licensed binaries, the full reference archive, and raw process logs remain
outside this repository.

R-OSSE slots translate the physical main curve, including the rollback. They
leave its radii, parameter stations and row count unchanged. A 12 mm slot on
the reference example starts the body at z=12 mm; adding a cylindrical prefix
and reparameterizing the body was the drift. An R-OSSE extension instead contributes
ten pre-body segments, followed by all LengthSegments body segments. With
slot and extension together, ATH leaves the extension's pre-body rows alone
and moves the body junction by the slot; the joining segment spans that gap.
This is the measured construction, rather than a new cylindrical slot.

For a rectangular no-shrink morph, let r_i be the initially sampled raw mouth
radius, p_i its direction, and T_i the requested rounded-rectangle ray radius.
ATH computes each largest positive Cartesian deficit:

- dx = max(0, max_i((r_i - T_i) * abs(cos(p_i))))
- dy = max(0, max_i((r_i - T_i) * abs(sin(p_i))))

It adds 2*dx and 2*dy to the **half-dimensions**, rebuilds the angular layout,
and repeats the boundary check until no positive deficit remains. It retains
the initial raw meridians by profile index, including their azimuth-dependent
coefficients; it does not recalculate them at the new directions. The ordinary
morph blend then uses the corrected target and those retained meridians.
For the circular R=150 mm example with a 240 x 360 mm request and 20 mm
corners, 32 angular segments give a 360 x 388.3281573 mm boundary. At 64
angular segments its height is 390.7419516 mm. Even the originally larger
axis can expand. A literal dimension floor, absolute radial displacement, or
re-evaluating OSSE a(p) at the final directions misses the observed wall.
Circle behavior and other unqualified target constructions are unchanged.

The shared sampler retains the effective half-dimensions and a final-direction
to original-meridian correspondence. Adaptive solve grids, mouth fitting and
preview LODs reuse that construction. Between authored directions, refinement
interpolates the source-direction correspondence; it does not derive a new
ATH design from a different angular budget. `AllowShrinkage=1`, native
authoring and saved v1 retain their preceding behavior. Current v2 R-OSSE slot/extension and inward rectangular no-shrink outputs
intentionally change. OSSE extension allocation and non-expanding morph
previews retain their prior output.
Existing coefficient, morph, archive, native/v1, mesh and STEP assertions
retain their tolerances.

### Consumers that replace sampling controls

Call `hornlab_mesher.config_builder.retain_ath_import_construction(config)`
on the **authored** mesher config before replacing angular, corner or axial
counts. Carry its returned transient construction fields through subsequent
normalization and sampling. For already-normalized parameters and an authored
grid, `profile_sampling.retain_ath_morph_construction(params, grid)` retains
the boundary and meridian correspondence. These fields are runtime data;
saved files continue to use the existing geometry stamp.

Waveguide Generator consumer work remains separate. At consumer dev
`49c63ed26aadb66781fc1554db2a61195fb668d8`, the failing boundaries are
`server/exports/core.py::_prepared_design` (overwrites authored grid and corner
counts before translation), its solid-STEP `mesh_cfg` construction,
`server/exports/sizing.py::_point_grid` (resamples parameters), and
`server/preview/core.py::_cache_relevant_config` (removes authored angular/axial
counts). Freeze the authored construction before export resampling, retain it
in sizing/planner calls, and retain the geometry-relevant authored controls or
frozen construction in the preview cache key. Changing only CornerSegments
from 6 to 12 on the example above changes its height from 388.3281573 to
360 mm without that handoff. No WG pins or consumer code move in this change.
