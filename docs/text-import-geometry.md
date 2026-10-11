# Text import geometry

`parse_text_config` retains the versioned `_textImportVersion` field. Its
current value is `ath-2026-08c-v2`. JSON round trips and callers that copy the
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
selects native interpretation and either `ath-2026-08c-v1` or `ath-2026-08c-v2` selects imported geometry.
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

Sparse imported dictionary transport uses the same known defaults as the text
reader: implicit sampling uses the ATH axial map, mouth mesh resolution is 8 mm,
and a supplied morph target defaults to corner radius 35 mm, fixed part 0.2,
and slot morphing enabled. Explicit controls remain authoritative. These defaults
also reach the mesh density used to fit actual surfaces. Native omitted mouth
resolution retains the native 26 mm default. These defaults apply to the
supported OSSE/R-OSSE text profile families; native authored geometry types
retain their own defaults even when provenance is carried alongside them.

Expressions use the scalar C99 function whitelist in
[the config schema](config-schema.md#parameter-expressions). ATH V2026-08c
probes establish the Gielis `sf(m,n1,n2,n3,a,b,p)` formula, natural `log`,
and half-away-from-zero `round`. The two-value functions `frexp` and `modf`
are refused during parsing and evaluation.

Current (v2, V2025-12+ rule) imported angular sampling adds the corner budget and rounds up to four
segments even for a plain profile: `AngularSegments=100`, `CornerSegments=4`
gives 104. The presence of integer `NAC` disables this addition; the value
does not affect the grid and is ignored with a named warning. Previously only
rectangular morphs received the corner-budget addition.

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
their existing interpretation. Evidence: V2025-12, V2026-08c and V2026-09a produce byte-identical
CORE and Solana grids, while all 23 archived configs echo V2025-06; the previous
sampler reproduces the four affected archive parity cases.

| Control | Imported text interpretation | Native authoring |
| --- | --- | --- |
| OSSE `Slot.Length` | A conical prefix at slope `tan(a0)`. `r0` is its starting radius; the main profile starts at `r0 + slot*tan(a0)`. The slot is included in `Length`; the main termination uses `Length - slot`. | A cylindrical prefix at `r0`, added to the body length unless an explicit total-length mode is selected. |
| `Throat.Ext.Length` | Adds a tapered extension behind the nominal throat; it does not change the slot or main body length. | Existing native extension behavior. |
| Circular `Morph.TargetShape=2` | Uses the largest sampled raw mouth radius, ignoring explicit target width/height and without rounding to a whole millimetre. | Honors explicit dimensions and retains the implicit dimension rule. |
| Rectangular morph with `Morph.AllowShrinkage=1` | Uses the requested target dimensions. | Uses the requested target dimensions. |
| Rectangular morph with default/false shrinkage and smaller target dimensions | Refused with a named diagnostic. This reference behavior is sample dependent and has no qualified construction. Choose explicit `AllowShrinkage=1`, or author a native configuration for a literal dimension floor. | Floors each target half-dimension at the corresponding raw extent. |

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
an active lip or Opus rear retain their previous consumers and rear-plane rules.


The saved v1 contract retains the geometry preceding v2: the fitted OSSE axial
map, the previous morph start, and the corner budget only for rectangular
morphs. An ATH version echo and NAC do not upgrade a v1 design. Import-only
lip/Opus controls require v2. Saved v1 CFGs retain their stamp when parsed;
fresh unstamped external CFGs receive v2. Unknown versions/rules are refused.
Native and v1 arrays are pinned to the preceding contract in
`tests/fixtures/rear-native.npz` and `rear-v1.npz`; floating coordinates are
compared with absolute tolerance 1e-9 mm and zero relative tolerance, while
triangle indices are exact. `rear-imported.npz` independently pins the current
imported inner/outer grids and exercised preview arrays.

Consumers constructing their own mesher dictionaries must retain these keys:

| Key | Location | Meaning |
| --- | --- | --- |
| `_textImportVersion` | Top level; carried into geometry params | `ath-2026-08c-v1` preserves saved geometry; `ath-2026-08c-v2` opts into the current import rules. |
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
