# Native OSSE composition

This contract defines the release-line path toward composable native OSSE.
Slice 0 added this contract and the legacy identity gate. Slice 1 replaces
the one-feature lock with one rule table in `native_boundary.py` and enables
Arc × Axial. Slice 2 adds Adapter × Arc (× Axial) on one shared composite
meridian, slice 3 adds Roundover × Axial, and slice 4 adds Adapter ×
Roundover (× Axial) as a closed solid. An unimplemented pair must refuse
with a clear message.

## Slots

Each design has at most one feature in each slot:

| Slot | Choices |
| --- | --- |
| Throat | none; authored adapter (`OSSE-ADAPTER`, `throat_adapter`); controls adapter (`OSSE-ADAPTER-CONTROLS`, `throat_adapter` with controls mode) |
| Mouth | none; arc (`terminating_arc`); roundover (`mesh.mouth_roundover_radius_mm`) |
| Axial | none; affine axial override (`axial_scale`) |

`SOURCE-DISK` is a standalone source body, alone, and never composes.

## Application order

Construct the intrinsic conic, apply the throat feature, then apply the axial
affine map: Z is multiplied by `axial_scale`, radius by `scale`. Construct
the mouth feature in finished, post-map millimetres. Apply vertical placement
offset last. Without an axial override, existing uniform-scale behavior holds.

The mouth feature is constructed against the mapped body. In particular, the
terminating arc's circle osculates the **mapped conic** at the join, matching
its point, tangent and curvature. The circle itself is never mapped: an
anisotropic map would turn it into an ellipse.

## Markers and canonicalization

The new composite marker is `OSSE-NATIVE`, used only when two or more features
are active and their feature set has no legacy marker. Features retain their
existing payload keys: root `throat_adapter`, `terminating_arc` and
`axial_scale`; roundover retains its existing mesh control. Mesher readers
at or below 0.2.7 must fail closed on `OSSE-NATIVE`; the composition tests
run the released 0.2.6 and 0.2.7 readers in fresh processes.

A feature set with a legacy marker is always written and read with that
marker. Single features keep `OSSE-ADAPTER`, `OSSE-ADAPTER-CONTROLS`,
`OSSE-ARC`, `OSSE-ROUNDOVER` or `OSSE-AXIAL`. Authored adapter plus axial
keeps `OSSE-ADAPTER-AXIAL`. Standalone sources keep `SOURCE-DISK`.

`OSSE-NATIVE` naming a feature set with a legacy marker is refused: there is
one spelling per design. A legacy marker carrying an additional feature key
outside its declared feature set is refused with a message pointing to
`OSSE-NATIVE`. The adapter-plus-axial legacy marker retains both its keys.
Every existing design and its outputs must stay byte-identical.

## Pair status in this release line

| Combination | Status |
| --- | --- |
| Arc × Axial | Implemented as `OSSE-NATIVE`; see [terminating arc](native-terminating-arc.md#arc-with-axial-scale) |
| Adapter × Arc (× Axial) | Implemented as `OSSE-NATIVE`, authored adapter only; see [terminating arc](native-terminating-arc.md#arc-with-a-throat-adapter) |
| Roundover × Axial | Implemented as `OSSE-NATIVE`; see [mouth roundover](native-mouth-roundover.md#roundover-with-axial-scale) |
| Adapter × Roundover (× Axial) | Implemented as `OSSE-NATIVE`, authored adapter only; see [mouth roundover](native-mouth-roundover.md#roundover-with-a-throat-adapter) |
| Roundover × Arc | Same mouth slot conflict; always refused |
| Adapter × Controls | Same throat slot conflict; always refused |
| Anything with SOURCE-DISK | Always refused |

Adapter here includes either throat choice where applicable; this table does
not authorize combinations beyond their implemented, validated domains.
Authored Adapter × Axial is already supported as `OSSE-ADAPTER-AXIAL`.

Refusals name the rule they hit: a slot `conflict`, `write <marker>` when a
legacy marker exists, `write OSSE-NATIVE` for a legacy marker carrying an
extra feature, `not supported yet` for an unimplemented composite, and
`SOURCE-DISK must stay alone`. The authored and controls adapters share the
`throat_adapter` key, so they cannot both be supplied.
[The composition tests](../tests/test_native_composition.py) run this rule
matrix over every pair and triple in both spellings.

## Shared construction

Every arc design, legacy or composite, is one `CompositeMeridian`
(`composite_meridian.py`): an optional mapped adapter cubic, the exact
retained conic as a rational quadratic, and the osculating circle. One
builder (`builders/terminating_arc.py`) emits it for MSH and STEP, and one
preview owns its render geometry. The arc and lip mesh certificates share
one facet-to-surface bound over a circle descriptor
(`builders/circle_certificate.py`). For the lip it bounds each triangle
with its local reach, so the reported `mouthRoundoverMeshChordBoundMm` can be
smaller than v0.2.6 reported for the identical mesh; the arc bound is
unchanged. The shared certificate adds a size-scaled roundoff guard of
`1e-10 * max(1, radius, abs(center_z), center_r, rim_reach)` to both the
vertex residual and the local reach (which is also rounded upward). A
worst triangle at the rim can therefore report a slightly higher lip
bound than v0.2.6. A composite's STEP file carries one
`native-composite` header recipe; the legacy markers keep their existing
emitters and bytes. Roundover × Axial keeps the freestanding lip
construction (`mouth_roundover.py`), whose body coefficients come from the
mapped conic; its STEP file carries the same header recipe. Adapter ×
Roundover (× Axial) extends that construction (`AdapterRoundover`): the
throat is the same `PhysicalAdapter` cubic as Adapter × Arc, the retained
conic is the mapped body from the adapter join translated into composite
coordinates, and the roundover builder adds the cubic and its certified
wall offset ahead of the conic. The lip certificate is unchanged.

## Identity gate

[The single-feature identity harness](../tests/test_native_single_feature_identity.py)
compares each legacy native family, from frozen literal fixtures, against tag `v0.2.6` in fresh processes:
canonical JSON of the full geometry-parameter tuple and resolved metadata
(including construction fingerprints), MSH bytes, STEP bytes with a pinned
timestamp and only the translator counter normalized, and deterministic
preview serialization with its timing clock pinned. Preview arrays retain
dtype and shape plus a SHA-256 digest of their raw bytes. Build-result
metadata is compared too; only `mouthRoundoverMeshChordBoundMm` may differ.
The harness requires `new <= old + 1e-8` mm, an explicit absolute tolerance
covering the numerical guard while allowing the accepted tighter bound.
All other build metadata remains exactly equal. The existing
adapter-plus-axial marker is included.

## Owner decisions (2026-10-23)

1. The arc's `join_t` stays a fraction of the authored body length `L`,
   including when a throat adapter is present.
2. Adapter × Arc with an authored `axial_scale == scale` keeps a separate
   identity from Adapter × Arc without `axial_scale`, even though their
   geometry is equal. Waveguide Generator will omit `axial_scale` when it
   equals `scale`.
3. The tighter `mouthRoundoverMeshChordBoundMm` metadata from the shared
   certificate is accepted, with the numerical guard allowance described
   above. Mesh and STEP bytes remain subject to the identity gate.
4. Adapter checks keep the conservative full-body domain through `L`,
   even when the terminating arc replaces the body beyond its join.
