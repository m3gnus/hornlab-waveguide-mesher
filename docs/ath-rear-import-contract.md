# Bounded ATH rear-return import

The first section covers RearShape 1 without a lip. The separate Opus/lip
section below preserves non-planar imported rears and supersedes its planar
restriction for those paths.

The versioned text import contracts `ath-2026-08c-v1` and `ath-2026-08c-v2` preserve ATH's
freestanding rear-return anchor for planar and bounded near-planar outer
throat rings:

`rear_z(phi) = outer_throat_z(phi) - wall_thickness_mm`

Wall thickness remains unscaled, as in the existing ATH import contract.
The source, bore, main throat and outer shell are not translated. Native
configurations without the version marker retain their original rear plane:
`mean(inner_throat_z) - wall_thickness_mm`.

One shared ring helper supplies all three freestanding builders and preview.
STEP exports the same OCC model as the acoustic builder. Imported geometry
uses internal immutable subclasses only when a freestanding outer wall exists;
base/native dataclass fields and state remain unchanged. Replacement and pickle
retain imported behavior, including active-stretch and mouth-fitted variants.
Bare, enclosure, baffle and authored adapter designs retain their existing types.

The existing RearShape 1 path without a lip uses horizontal rear caps. A near-planar imported ring is
represented at the midrange of its pointwise target rear Z values. This is the
horizontal plane that minimizes the largest vertical boundary displacement.
Each boundary point must move by no more than **0.0001 mm (0.1 micrometre)**
and **0.01% of unscaled wall thickness**, whichever is smaller. This explicit
representation error budget preserves the previously supported imported
near-planar rings without translating the bore or changing the outer shell.
It is a new bounded geometric allowance, not a claim that deviations are
floating-point noise or a waiver based on CAD tolerances. XY remains exact.
The actual computed displacement is checked before projection; nonfinite
coordinates and nonfinite/nonpositive wall thickness are refused.

Larger deviations use a warped fan with the measured ATH V2026-08c RearShape 1
throat construction: authored inner row 0 shifted -5 mm, then offset by wall
thickness along its meridional chord normal. The analytic outer shell remains
unchanged, joined to this ring by the rear return. Fixture `m6_var_wall` has a
0.258 mm rear span. The measured rear ring differs from ATH by at most
0.000688 mm, versus 0.398502 mm for an extrusion of the analytic outer throat, compared at matching azimuths.
Planar rings retain their previous construction bit for bit.
For newly supported rings (target Z spread greater than the former 1e-7 mm
planarity cutoff), resolution checks the opposite acoustic/legacy sampling
route as well. If both rings individually fit the displacement budget, their
planes must agree within the smaller of 1e-9 mm and the displacement budget;
otherwise a named cross-route refusal prevents
silently accepting different rear depths. This consistency tolerance is not an
additional projection allowance. A route whose own ring is materially warped
supplies no competing planar authority. This does not
promise legacy support for every acoustic-supported import.

Production geometry for the selected configuration owns the certified rear Z.
Preview resolves that same geometry even when its rear cap is hidden; its own
visual resampling supplies only XY and the outer-shell display. Every LOD uses
the production-certified plane, without interpreting independently sampled
viewport normals as a new rear-depth authority. This preserves the visual
outer-return adjacency while all actual production rear points retain their
bounded displacement. All three builder paths and STEP use their selected
production ring. Native rear interpretation remains exact and unchanged, with
no additional sampling or dataclass state.

Explicit imported enclosure depth <= 0 requests geometry that ATH retains as
an enclosure sheet. The supported sealed enclosure builder cannot reproduce
that sheet, so config normalization raises a named unsupported diagnostic
before preview, mesh or STEP can silently substitute a freestanding horn.
Omitted enclosure depths are unaffected. Native zero-depth configurations
still disable enclosure mode.

Bounded reference evidence includes straight extension half-angles 0, 5.25 and
15 degrees, the R-OSSE extension in the reported case, and a variable-angle ring.
The measured warped fan supports the imported RearShape 1 throat rule;
this contract does not establish zero-thickness
enclosure sheet support, full ATH mesh parity or a Fusion import qualification.

## Opus rear and RearShape 1 with an ATH lip

`Mesh.RearShape = 3` selects the import-only Opus rear under `ath-2026-08c-v2`. It amends the planar
rule above: its throat ring is non-planar and is preserved pointwise, with a
fan to an axial apex. Native rear geometry and RearShape 1 without a lip retain
their existing rules. Enclosure, infinite baffle and CircSym combinations are
refused by name. RearShape 2/5, RearCubic, OpusRear, MouthArc and RoundedEdge
remain unsupported.

Measured against ATH V2026-08c output, Opus clamps wall thickness to
0.1–1000 mm (including an authored zero), defaults `FrontOffset` to 40 mm
and `Step` to 25 mm, and selects rings from the full inner grid, lip included:

```text
zf = z_last(p=0) - FrontOffset
e = last index <= n-2 whose z(p=0) <= zf
prev = 0
select i in 1..e if i == e or z_i < prev or z_i - prev > Step
update prev only on selection
throat = G[0] shifted by -10 mm in z, then offset by w*n[0]
middle = selected G[i] + w*n[i]
front = G[last] shifted by -FrontOffset in z, without a normal offset
apex = (0, 0, mean z of the two exported throat-ring endpoints)
```

Normals are meridional profile-plane chord normals (forward at the throat,
backward at the original mouth); a spatial cross-product normal changes the
rear coordinates. Lip rows use the explicit normals in the
[ATH import lip](mouth-roundover-contract.md#ath-import-lip) section. Selection
uses changes in z from the last selected row, starting at z=0, rather than arc
length. Surfaces are a throat fan, consecutive ring lofts, and a front-to-lip
band. The final lip ring determines that band's attachment, not the maximum-z
body ring. Both mesh topology modes retain the same authored sampling authority;
legacy preserve-grid uses straight angular edges, other modes use shared ring
splines. The point sizes are RearResolution on the throat, selected middle rings and
apex, and 1.2 times MouthResolution on the Opus front ring. Mesher density fields
own final element sizing: the mouth field resolves the lip and the rear field
covers the outer loft and fan.

RearShape 1 with a lip uses the same normals/selection mechanism, a throat
shift of -5 mm, fixed Step of 5 mm, selection through the last lip row, and no
separate front ring. The offset lip is concentric with radius `R-w` on a circular
mouth. Neither rear is flattened to a horizontal plane.

Preview uses the mesh builder's authored ring splines, lofts, warped fan and
front band at coarse, fine and inspection LOD. Shared boundary curves own
shared tessellation nodes. The first outer loft keeps `wall.rear_return`, the
fan keeps `wall.rear_cap`, and the remaining lofts keep `horn.outer`. Triangle
mid-edges, centroid and interior quarter stations measure chord error against
the actual parametric patches. Parametric normals drive refinement and winding;
curvature differentiates those patches rather than the triangles. Achieved
fidelity and refinement limits are reported through the existing preview
metadata. Narrow folded patches retain the existing explicit folded-triangle
orientation metadata. Symmetry cuts close after reflection, without a rigid
sheet on the cut.

The viewport's authored `inner_points`/`grid_n_length` export contract stays
intact. Its selected `outer_points` has an independent
`outer_grid_n_length` (also `rear_grid_n_length`) and `rear_apex`; renderers must
use these rather than assuming equal inner/outer row counts. Passing `lod` to
`build_viewport_geometry_from_config` adds the complete preview surfaces.
A bare inner-export request retains lip rows when the caller explicitly removes
Opus rear controls. Bare mode with RearShape 3 is refused before wall clamping,
without a clamp warning.

Active lips and Opus rears also refuse `Scale != 1`, active `Throat.Ext.*`,
nonzero `Rot` or `Mesh.VerticalOffset`, and `Source.Shape = 2` pending ATH probes.

STEP retains a named refusal: the mesh uses GEO spline/Coons patches, whereas
the STEP writer exports OCC geometry. Faithful support requires matching OCC
ring patches, a warped fan, shared seam curves and qualified solid sewing;
substituting the existing planar fill or an unconstrained OCC loft would
change the authored surfaces. Existing planar imports retain STEP support.
Excessive FrontOffset with no eligible rear row, **FrontOffset < 0**, nonpositive/
nonfinite Step, and nonfinite FrontOffset are named refusals. Negative offsets
have no qualified ATH ring fit.

Evidence is the extracted wall/rear point sets and profile grids in
`tests/fixtures/ath_import/ath-v2026-08c`: `of20` pins non-default Step/FrontOffset,
`onr` omits the lip, `odef` omits both defaults, `ow0` pins the wall clamp,
`ofull` covers a full ring, `rs1` pins the concentric outer lip, and `coreabec`
pins morph/stretch and the warped throat. CORE selects `[4,7,9,11,12]`, with
apex z approximately -10.1977 mm. Tests compare all selected wall rings, apex
and Opus front ring in both directions at 2e-3 mm, and verify closed mesh
edges after symmetry reflection. Geometry parity does not establish identical
ATH topology or triangulation.

GEO ring patches publish a twelve-point tensor Gauss area forecast to density
sizing. It integrates each patch's parametric area rather than summing a full
transverse bounding box at every short ring loft. Existing OCC/faceted/native
area forecasts remain unchanged. CORE builds with the default triangle budget;
its quadrant mesh contains approximately 4,238 triangles and closes after
reflection across both declared symmetry planes.
