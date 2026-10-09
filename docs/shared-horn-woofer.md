# Shared horn and woofer assembly

`SourceAssembly` attaches two canonical circular source contours to one finite
rectangular enclosure. A straight circular conical horn joins its throat source
to a front opening; the woofer joins an independent front aperture, with a rigid
collar where needed. Placement, cone/cap/surround dimensions and box dimensions
remain independent. Both axes point+Z; the common observation origin is the
front plane's centre. Arbitrary axes and phase plugs need separate contracts.

`export_assembly(model, [horn_drive, woofer_drive], new_directory)` publishes
geometry.step, preview.msh and source.json atomically in an isolated process.
One sewn shell contains every scattering face. Sources and successive segments
share actual OCC edges. STEP is a zero-volume surface representation. Output
budgets remain250000 triangles, checked before and after meshing.

The required features are native-source-contour-v1 and
native-shared-horn-woofer-v1. Source-qualified patch IDs percent-encode the source
and local patch identities separated by /. They survive density changes and
avoid collisions between presets with the same patch names. One channel covers
each physical diaphragm; rigid patches have no drive. Literal relative weights
and normal/axial motion remain separate from channel gain, delay and polarity.
Geometry, excitation and density have separate hashes; temporary STEP selectors
bind only to exact member bytes. Clearance is greater than0.1mm at box edges,
between aperture disks, at the back and between the source and horn mouth.

The consumer counterpart is Waveguide Generator's native imported adapter and
source editor assembly ingestion API. It verifies the canonical recipe against
reopened STEP, shared joins, complete roles and every moving/rigid mesh facet
within0.15mm. It uses one full-domain mesh for both channel bases, with inactive
sources held at zero velocity and one common frame. Existing consumer resource
ceilings remain in force. This geometry/dispatch feature does not qualify a
native acoustic runtime or establish acoustic convergence.
