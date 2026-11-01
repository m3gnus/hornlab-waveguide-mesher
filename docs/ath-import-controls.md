# ATH import transforms and extension sampling

These additions belong to the `ath-2026-09a-v3` import contract. Native
authoring and saved v1/v2 imports retain their preceding interpretation. The
captured reference is ATH V2026-09a; this is a bounded geometry comparison,
not a claim of complete ATH support or identical triangulation.

The existing `_textImportVersion` field and saved CFG comment stamp select
this behavior. Fresh external imports receive v3. Explicit v2 files and
persisted v2 dictionaries remain v2 through loading, copying and JSON
round trips; there is no automatic upgrade. Selecting v3 for a saved design
is an explicit geometry change, so reimport a copy only when that change is
intended. In particular, saved v2 keeps its old R-OSSE table, unrotated OSSE
extension stations, nominal source curvature and unit `Scale.Z` no-op.

## Independent axial scale

Top-level `Scale.Z` accepts a positive finite constant or constant expression.
After profile evaluation, rotation and morphing, `Scale` multiplies x and y;
an explicit `Scale.Z` multiplies z instead of `Scale`. Without `Scale.Z`,
global `Scale` still multiplies all three coordinates. Thus `Scale = 3` and
`Scale.Z = 1` leave depth unchanged while tripling transverse dimensions.
Wall thickness and explicit source radius retain their millimetre controls.
Azimuth-dependent `Scale.Z`, imported roundover and Opus-rear combinations
remain refused pending measurements.

## R-OSSE rotation

Top-level `Rot` accepts finite degrees or expressions using azimuth `p` in
radians. Rotation is around the nominal main throat radius at the end of
the extension. The straight extension stays unrotated. A slot translates
the body before rotation; with an extension, its last authored segment joins
the rotated, translated body throat. Scalar and array evaluation share the
same rotation helper used by OSSE. In-block R-OSSE rotation remains ignored
with a warning. Active throat stretch plus R-OSSE rotation remains refused.

## Extension and body intervals

`Mesh.ThroatExtSegments` selects 1–4096 uniformly spaced extension intervals;
the default is ten. It can be flat or a member of `Mesh`. The count adds to
`Mesh.LengthSegments`, which continues to sample the complete body. No
extension means no added rows. Current OSSE sampling uses this split with
or without rotation. Acoustic refinement subdivides the authored segments,
including the slot join, without replacing the physical join by a different
curve. The older OSSE sampling rule does not accept this new control.

Zero, negative, nonfinite and fractional counts are refused. ATH itself
omits the driver-end rows for zero, crashes for a negative count, and truncates
fractions; these are deliberately not imported as valid extension controls.
`Mesh.ThroatSegments` remains refused: V2026-09a ignores that older spelling.

V3 imported R-OSSE default body stations use the cubic map with controls
(.26,.15), (.46,.49), sampled at parameter increments of .001 before
abscissa inversion. This replaces interpolation of a 20-interval table;
native and saved v1/v2 sampling retain that table. Explicit custom maps retain
their existing behavior.

## Source and terminal validation

With independent axial scaling, explicit extension counts, or active R-OSSE
rotation, automatic source curvature uses the mean angle of the first
authored wall chords after transformation. Density and preview LOD must not
change that source. Explicit `Source.Radius` overrides the automatic radius;
`Source.Curv = -1` retains its curvature direction. The common source-angle
resolver serves mesh/CAD resolution and preview.

One reference defect is retained as a limitation: with a straight extension,
ATH's generated source pole is displaced by the extension length and does
not lie on its declared sphere. HornLab matches ATH's separately reported
wavefront radius and keeps a consistent spherical cap. The two extension
source witnesses therefore do **not** establish identical source geometry.
Other source measurements match the captured pole, centre and radius within
.002 mm, accounting for the reference script's .001 mm coordinate rounding.

`tests/test_ath_import_controls.py` checks 52 full grids plus reduced-domain
subsets against native exports at .0002 mm, source measurements, rotation
scalar/array agreement, authored slot interpolation, preview LOD, invalid
input boundaries, actual tagged meshes and reopened STEP surfaces. Fixture
hashes and input pairs live in
`tests/fixtures/ath_import/ath-v2026-09a/controls/`.
