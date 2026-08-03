# Batch N — preview surface orientation: audit, contract, fix (branch `preview-api`)

Owner report (Magnus): the rendered preview "looks wrong" — suspected flipped normals/winding. The v2 renderer currently masks orientation with DoubleSide materials, so latent orientation errors surface as wrong lighting (the scene renders suspiciously dark). Make orientation a TESTED CONTRACT.

**Path discipline: modify ONLY `hornlab_mesher/preview/**` and `tests/test_preview_*.py`. Additive discipline continues — no existing-module edits. Full suite must stay green (current: 581 passed, 23 skipped).**

## The contract to implement and test

Define and enforce per-role orientation (document in the module docstring AND in metadata per surface as `orientation: "air-side" | "exterior"`):

- `horn.inner`: normals point INTO the acoustic air domain (away from the wall, toward the horn cavity/listening half-space). This is the surface the user looks at from the front — its lit side is the concave side.
- `horn.outer`, `enclosure.*`, `wall.rear_cap`: normals point to the solid's EXTERIOR (away from the enclosed volume).
- `source_cap`: into the air domain (same side as horn.inner faces).
- `mouth_rim`: exterior/front-facing.
- Winding must AGREE with the normal (counter-clockwise when viewed from the normal side) for every triangle — renderers using FrontSide + the shipped normal must light correctly with zero DoubleSide crutches.

## Verification approach (do not hand-wave this)

**Signed volume is NOT a valid oracle for open shells** (hard-won workspace lesson). Instead:
1. Analytic side-checks per role: for sampled triangles, verify `normal · (p_test − centroid) > 0` where `p_test` is a point known to lie on the required side (horn axis points for horn.inner near the throat; a point far outside the enclosure bounding box for exterior surfaces; etc.). Build these reference points from the config analytically, not from the mesh.
2. Winding↔normal agreement: for every triangle, `cross(b−a, c−a) · n_avg > 0` (with the shipped per-vertex normals).
3. Run across all four families × freestanding + enclosure × all LODs in the invariant tests.

Fix whatever the audit finds (winding flips, normal sign flips, per-surface inconsistencies — including any surface where different regions disagree). Determinism must hold.

## Also (same batch, small)

Emit `orientation` + `windingChecked: true` in per-surface metadata so FRAME-SPEC consumers can rely on it, and bump the preview API metadata version note.

## Rules
- Self-verify: new orientation tests + full suite green.
- Final message: per-role audit findings (which surfaces were wrong and how), the fix summary, test counts, and an explicit statement the v2 renderer may now use FrontSide.
