# Bounded ATH rear-return import

The versioned text import contract `ath-2026-08c-v1` preserves ATH's
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

Current rear caps are horizontal planes. A near-planar imported ring is
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

Larger deviations receive the named nonplanar-rear refusal through resolution
and preview. Materially warped ATH rear rings are not flattened to a mean plane.
For newly supported rings (target Z spread greater than the former 1e-7 mm
planarity cutoff), resolution checks the opposite acoustic/legacy sampling
route as well. If both rings individually fit the displacement budget, their
planes must agree within the smaller of 1e-9 mm and the displacement budget;
otherwise a named cross-route refusal prevents
silently accepting different rear depths. This consistency tolerance is not an
additional projection allowance. A route whose own ring is materially warped
remains unsupported and supplies no competing qualified plane. This does not
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
This contract does not establish arbitrary warped rear topology, zero-thickness
enclosure sheet support, full ATH mesh parity or a Fusion import qualification.
