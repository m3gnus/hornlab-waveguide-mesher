# Text import geometry

`parse_text_config` retains the versioned `_textImportVersion` field. Its
current value is `ath-2026-08c-v1`. JSON round trips and callers that copy the
parsed config must preserve it: geometry normalization retains it in the
profile parameter dictionary. An unknown version is refused rather than
reinterpreted. The version identifies this import contract; the importer does
not detect which application build authored a file. It does not claim full
feature, station allocation, topology, or acoustic parity.

Native configurations without the field keep their established dimensions
and no-shrink behavior. Adding the field deliberately selects the import
contract and its bounded refusals.

The direct CFG/CLI reader honors saved application comment stamps before
discarding comments: `; Waveguide Generator geometry-interpretation: native-v1`
selects native interpretation and `ath-2026-08c-v1` selects imported geometry.
Unknown or conflicting stamps are refused. Unstamped external CFGs select the
import contract; historical `; Parameter config` or `; MWG config` headers
select native interpretation, consistently with the application reader. An
explicit stamp takes precedence over the historical header. Native saved CFGs
retain native profile/morph defaults and additive slot length; explicit
`Length.Mode` retains its authored override.

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
