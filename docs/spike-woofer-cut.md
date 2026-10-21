# Spike 1d: piston woofers in the design enclosure

2026-10-21. Branch `spike/assembly-woofer-cut-261021`, based on local dev
`d740c28` (FREEFORM corner topology/density/build-cost fix).

## Decision

**Go for the shared design-enclosure architecture.** No second horn or box builder
is needed. All six requested horn families produced watertight, outward-oriented
full-domain meshes with two separately tagged flat pistons, both flush and 2 mm
recessed. **FREEFORM corner anchors are a conditional go:** the tested rounded
station mouth works with sharp enclosure edges; rounded enclosure edges fail in
existing corner-wall sewing even without woofers. That combination needs a fix or
an explicit production refusal. This is geometry feasibility, not a qualified
WG ingestion or acoustic solve, and not ATH parity evidence.

## Measured cases

`tests/test_spike_woofer_cut.py` creates the exact configurations. OSSE and R-OSSE
start from the examples; FREEFORM cases reuse the owner fixtures in
`test_freeform_builds.py` and `test_freeform_corner_wall.py`; LOOKUP uses the
existing four-point profile `[[0,12.7],[30,28],[80,85],[120,150]]`; OSSE morph uses
a 300 x 200 mm rounded rectangle, corner radius 30 mm.

| Horn | Go/no-go | Flush triangles | 2 mm recess triangles | Baffle Z, mm |
|---|---|---:|---:|---:|
| OSSE round | Go | 14,762 | 15,198 | 120 |
| R-OSSE | Go | 10,920 | 11,386 | 34.8820 |
| FREEFORM rounded_rectangle station mouth | Go | 10,754 | 11,214 | 120 |
| LOOKUP | Go | 11,156 | 11,620 | 120 |
| OSSE rectangle morph | Go | 14,518 | 14,972 | 120 |
| FREEFORM corner anchors + rounded_rectangle mouth | Conditional go, edge=0 | 5,182 | 5,408 | 120 |

Every row, at both recesses: build succeeds; boundary edges=0, nonmanifold
edges=0, inconsistent edges=0; signed volume is positive; every piston triangle
has positive Z normal. Physical tags are exactly `{1,2,101,102}`; source maps are
`horn-throat:2`, `woofer-1:101`, `woofer-2:102`. Mesh physical names for the
woofers are `woofer-1` and `woofer-2`. The original throat tag/name remains 2 /
`SD1D1001`.

The two circles have radii 35 and 25 mm. One is centered on x=0 below the mouth
and crosses the enclosure's sector seam; the other is at x=25 above it. Their
centres are 65 mm beyond the respective sampled mouth Y extrema. Box depth is
240 mm, margins L/R=80, top/bottom=150, rounded edges=18 mm except the corner
case. Circular rim maximum midpoint chord errors are 0.0513 and 0.0367 mm;
piston areas are 3841.17 and 1959.66 mm², within 0.2% of the analytic circles.
Full measurements, including volumes and environment versions, are in
`docs/spike-woofer-cut-evidence.json`.

The test suite additionally exercises N=0,1,3; metre output; +13 mm vertical
placement; accepted 0.01 mm recess; invalid dimensions, NaN, overlapping discs,
mouth collision, discs outside the flat baffle, excessive recess, and all three
reduced domains. Removing an emitted piston demonstrably triggers the existing
production closed-shell guard.

The rounded-edge corner failure is retained as
`test_corner_rounded_enclosure_failure_precedes_woofer_hook`: remove the woofer
list, set edge=18, and the ordinary builder raises
`FREEFORM corner sewing lost face ancestry` in `_sew_roles`. The hook is never
reached. The same configuration with edge=0 works. No broad claim about every
corner profile or every roundover is made.

## Builder change and proposed real contract

The ordinary configuration resolves to `PointGridHornGeometry`. The OSSE,
R-OSSE, station FREEFORM, LOOKUP and morph paths use the shared point-grid
builder and `builders/enclosure.py::build_enclosure_box`; the corner path uses
`corner_wall.py::build_corner_wall` and the same enclosure builder. The spike
hooks in `mesher.py` immediately after `_dispatch_builder`, including the corner
path, before density configuration, physical tagging and mesh generation.

`spike_woofer_cut.py` identifies the existing planar front faces by plane type
and centre Z (OCC bounding boxes include sizeable tolerances). OCC `fragment`
partitions those actual faces with circular discs. Its ancestry maps identify
the rigid baffle and each source, including discs crossing sector seams. The
complete disc must be covered by original front faces with the full analytic
area, otherwise the layout is refused. For a recess, extrusion supplies a lower
flat cap and rigid cylindrical collar; internal walls on sector seams are
removed. Both physical and density groups are updated from the map. All rigid
surfaces are merged into physical tag 1 only on the opt-in assembly path;
enclosure density roles remain separate. Existing density, duplicate-node
handling, orientation repair and closure validation are reused.

The temporary JSON/TOML entry is intentionally named `enclosure.spike_woofers`:

```json
{"enclosure":{"depth_mm":240,"space_b_mm":150,"spike_woofers":[
  {"x_mm":0,"y_mm":-275,"radius_mm":35,"recess_mm":2}
]}}
```

For 3a, use typed immutable `enclosure.woofers[]`, with an explicit adapter for
`Mesh.Enclosure.Woofers = [...]` if ATH-style config text is desired. Each item
should carry stable `radiator_id` and `segment_id`, resolved `x_mm`, `y_mm`,
`radius_mm`, `recess_mm`, and optional local resolution. Resolve saved placement
relations in WG before this boundary; the mesher consumes concrete geometry.
Patch map identities become `radiator_id/segment_id`. Do not use list order or
labels as long-term identity; `woofer-k` is only the spike's deterministic name.

ATH `Mesh.Enclosure.LFSource.A` maps to an above/centred radiator, `.B` to a
below/centred radiator, with Radius mapping to radius_mm. ATH Spacing becomes
the saved gap relation, resolved against the horn mouth using the 1a corpus's
measured reference edge; that reference is still open in the coverage ledger.
Do not silently assume a centre distance or a bounding-box distance. DrivingWeight
and SID are excitation/provenance data, not geometric tags or radiator identity.
ATH parsing is unchanged and still refuses LFSource today.

The emitted metadata has the WG imported map **shape**:

```json
{"rigid_tag":1,"source_tags":{"horn-throat":2,"woofer-1":101},
 "tag_map":{
   "1":{"source_id":null,"instance_id":null,"role":"rigid"},
   "2":{"source_id":"horn-throat","instance_id":null,"role":"primary"},
   "101":{"source_id":"woofer-1","instance_id":null,"role":"secondary"}}}
```

This is deliberately namespace `spike-design-assembly-v1`. WG's
`server/mesh/imported.py::allocate_imported_tags` uses `wg-import-v1`, allocates
all sources from 101, and its import path uses encoded physical names. These
meshes are **not ready-made WG imported records**. Slice 1c/3a must explicitly
translate/admit retained throat tag 2 and publish the complete compiled record,
or remap the mesh on ingestion. No consumer was modified in the read-only WG
checkout. Numeric tags alone do not establish imported-source identity.

## Symmetry, frames and normals

For a symmetric horn and symmetric L/R box, a single woofer below the horn at
x=0 preserves reflection in **x=0** (YZ plane). It breaks reflection in y=0
(XZ plane). Therefore `quadrants=14` (right half, x reflection) can represent
that whole assembly, but `12` (upper half, y reflection) and `1` (quarter,
both reflections) cannot. Asymmetric box margins, horn features or placement can
remove even that remaining symmetry. The main two-woofer fixture has x=25 on
one source and therefore requires the full domain.

Production should refuse an explicitly requested incompatible reduction rather
than silently mirror the horn-only domain and invent sources. Default to full.
An off-plane source can participate in a geometrically mirror-paired layout,
but equal geometry alone does not grant symmetry to independent excitations:
reflection must preserve the driven patch/weights of each basis solve. Separate
upper/lower or left/right radiator IDs generally break per-excitation symmetry,
even when a tied in-phase pair permits a grouped symmetric solve. Quarter-domain
individual-source bases need explicit symmetry-sector reconstruction, beyond 3a.
The spike conservatively refuses **every** reduced domain with woofers, including
the potentially valid right half; it does not implement clipping piston discs.

In the design builder's native mm frame, +Z points from throat toward mouth;
the tested flat HF diaphragm is at z=0. `enclosure_box_bounds` sets front Z to
`max(mouth[:,2])`; back Z is front minus the clamped enclosure depth (at least
horn length plus depth margin). The baffle therefore coincides with the planar
mouth in these tests. It is not generally z=0 or a constant configured horn
length: the R-OSSE example's sampled mouth is at z=34.8820 mm. A recessed piston
is at z_front-recess; the collar runs from that plane to z_front.

X/Y woofer values are in the resolved horn's pre-placement mm frame. Existing
Mesh.VerticalOffset shifts the finished horn, box and woofers together in Y;
metre conversion follows. These operations are tested. No new automatic scaling
rule for woofer dimensions under top-level design scale was introduced; the real
adapter must resolve dimensions and apply placement exactly once.

For normal design observations, WG resolves a selected mouth origin to the horn
mouth centre (here `(0,0,z_front)` before placement), or throat to the source
centre. However `SolverContext.from_imported_request` currently **forces throat origin**:
it has no authoritative imported mouth frame. Importing this geometry without
an explicit compiled-record frame would therefore move the observation origin
back from the baffle by the horn length. 1c/3c must publish both horn mouth and
HF-source frames and one common assembly observation origin, not an enclosing
box centre or a per-woofer origin. A flat HF source is used here; a curved HF
source needs its own geometric reference rather than an assumed z=0 centroid.

Outward means outward from the solid into the acoustic exterior, including the
horn bore: horn wall normals face the bore, front pistons face +Z, back faces
face -Z, and recessed collar normals face into the recess. Existing
`repair_orientation` plus edge-consistency/positive-volume checks determine the
connected shell winding; tests separately require every woofer facet to face
+Z. Generic `validate_orientation` source-normal diagnostics currently select
only physical tag 2, so multi-source per-patch normal checks must become a real
admission invariant, rather than relying on that existing primary-only check.

## Boundary completeness and remaining limits

Available now: `_require_baffle_keeps_mouth_curves` detects dropped horn-mouth
curves; the enclosure rejects a horn wall bisecting its baffle outside the
opening; the postprocessor rejects nonmanifold/inconsistently wound edges;
`_validate_closed_shell_contract` rejects free edges away from declared symmetry
cuts, with a location diagnostic. Full-domain generated enclosures are held to
closure even though generic orientation validation uses require_watertight=False
for reduced/bare modes. Interface tag 4 has an explicit exemption; this spike
refuses subdomain interfaces instead of treating them as an exterior boundary.

Still needed for production: exact once-only classification of all active OCC
faces and all emitted facets; positive nonzero area and geometric/normal checks
for **every** declared patch; compare source tags to the compiled identity map;
check per connected component, including disjoint future bodies; check minimum
mouth/outer-edge clearance, collar vs horn-wall intersections and self
intersections. Watertightness alone cannot detect overlapping closed surfaces
or a wrongly tagged rigid patch. Current spike containment is a full-disc area
check, not a comprehensive finite-clearance collision system. It checks 0.1 mm
between woofer circles, but does not impose the same clearance against the
mouth or enclosure perimeter. Deep recesses, unusual rollbacks and near-tangent
layouts remain outside the proof. Existing unsupported enclosure plans are not
newly enabled; this proves the current rectangular enclosure path.

Review found that the existing 0.005 mm approximate weld can collapse the rim of
an accepted 0.001 mm recess onto the baffle, leaving a non-flat piston. Fixed by
refusing `0 < recess < 0.01 mm`, with a refusal regression and an accepted
0.01 mm placed/metre-output case. A production conformal topology could instead
avoid approximate welding and certify the lower-plane geometry explicitly.

## Validation, estimate and parity risk

Python 3.13 / Gmsh 4.15.0 on this Mac; single-thread targeted tests only.

- `python3 -m pytest tests/test_spike_woofer_cut.py -q`: **27 passed** (about 20 s).
- Seven existing targeted tests passed (6.77 s): the three FREEFORM closed modes;
  canonical enclosure tags; the two closed mouth/baffle rim cases; quarter
  enclosure endpoint attachment.
- No full suite and no `run-ath-parity.sh --full` were run, per the spike brief.
  No acoustic solver, installed-artifact, Windows or WG end-to-end qualification
  is claimed. The 0.1 mm rim measurements are not ATH-reference parity.
- Two independent reviews (geometry and contract) found no blocking defect for
  the stated spike scope. Shallow-recess finding fixed; namespace translation
  limitation retained explicitly above. Duplication check: reused the shared
  enclosure, formula dispatch, density, tagging, orientation and closure helpers;
  no second source_assembly/front_baffle box implementation was introduced.

Real slice 3a remains **L, approximately 4–6 producer days**, including review:
about 1 day typed config/identity and group plumbing, 1–2 days robust clearances
and shared topology/density, 1 day compiled-map/symmetry/frame admission, and
1–2 days corpus tests and qualification. Repairing rounded-edge corner sewing
may add 1–2 days if that combination is required rather than explicitly refused.
WG solve/UI work and building the 1a reference corpus are separate dependencies.

The real slice needs the full ATH parity gate at its checkpoint via the compute
broker (lane compute, priority 4 for study work, requester specified by the brief),
plus LFSource corpus aperture position/radius and enclosure comparisons. Keep the
no-woofer path/tag namespace byte-compatible; current existing tests exercise it
but do not establish full parity. Main risks: OCC fragmentation re-authoring mouth
curves and face ancestry, new density groups or circle settings changing horn
meshes, changed tag assumptions in design/imported consumers, arbitrary-placement
symmetry errors, ATH Spacing/roundover semantics, and the existing corner sewing
failure. Scope the new logic behind nonempty woofer data and run horn-only parity
as well as the new corpus. Any failure must be listed by exact case, not waived
as a generic known failure.

No merge or push is authorized for this spike; dev remains frozen for 0.3.6.
The deliverable hand-back records the exact branch commit SHA separately.
