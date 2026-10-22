# Native OSSE composition

This contract defines the release-line path toward composable native OSSE.
Slice 0 added this contract and the legacy identity gate. Slice 1 replaces
the one-feature lock with one rule table in `native_boundary.py` and enables
Arc × Axial. An unimplemented pair must refuse with a clear message.

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
at or below 0.2.6 must fail closed on `OSSE-NATIVE`.

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
| Adapter × Arc (× Axial) | Later; refused until implemented |
| Roundover × Axial | Later; refused until implemented |
| Adapter × Roundover (× Axial) | Later; refused until implemented |
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

## Identity gate

[The single-feature identity harness](../tests/test_native_single_feature_identity.py)
compares each legacy native family, from frozen literal fixtures, against tag `v0.2.6` in fresh processes:
canonical JSON of the full geometry-parameter tuple and resolved metadata
(including construction fingerprints), MSH bytes, STEP bytes with a pinned
timestamp and only the translator counter normalized, and deterministic
preview serialization with its timing clock pinned. Preview arrays retain
dtype and shape plus a SHA-256 digest of their raw bytes. The existing
adapter-plus-axial marker is included.
