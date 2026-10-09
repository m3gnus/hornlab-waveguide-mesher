# Bounded ATH rear-return import

The versioned text import contract `ath-2026-08c-v1` preserves ATH's
freestanding rear-return anchor for planar outer throat rings:

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

Current rear caps are planar. Imports whose authored outer throat z varies by
more than 1e-7 mm receive a named nonplanar-rear refusal through resolution and
preview; preview validates the full ring even when its rear cap is hidden.
Warped ATH rear rings are not flattened to a mean plane.

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
