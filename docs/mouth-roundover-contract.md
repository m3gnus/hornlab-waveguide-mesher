# Mouth roundover: geometry and import contract

**Status: proposed contract for Program C6, not yet reviewed by the owner.**
This document adds no executable behaviour. The ATH text importer refuses
`Mesh.Roundover` until an implementation satisfies this contract. The open
decisions are listed under "Owner questions" at the end.

A mouth roundover is an extra radius added at the mouth of a free-standing
horn: the inner wall continues past the mouth as a circular arc that turns back
toward the rear. It is not a round enclosure and it is not an R-OSSE rollback.
Today the wall ends on a ruled face between the inner and outer mouth rings
([geometry contract](geometry-contract.md), "Freestanding Mouth Closure"), and
that section already says a rounded closure needs an explicit geometry option
implemented by every preview, CAD and mesh consumer. This is that option.

## Evidence and limits

- ATH thread, posts #18,915 and #18,916 (2025-12-05): the author adds "a
  roundover to a general free-standing OSSE horn (an additional radius added to
  the mouth)", with the syntax

  ```text
  Mesh.Roundover = {
    Radius = 25     ; [mm]
    Segments = 5
  }
  ```

- 34 probes run on ATH build V2026-08c for this document. ATH's `GridExport`
  stops with an out-of-range error when a roundover is active, so the geometry
  was read from the `bem_mesh.geo` point list ATH writes for its mesher
  (`Output.ABECProject = 1`). Those coordinates carry three decimals, so every
  fit below is limited to about 0.001 mm.

The probes establish what ATH builds for the settings tested. They do not make
ATH's construction the right one to copy: it is grid-dependent and leaves a
corner at the mouth (see "What ATH builds"). The contract below therefore
defines the mesher's own construction and states how it differs.

## Coordinates

Meridian coordinates `(z, r)` in mm, z increasing toward the mouth, as in the
geometry contract. `M = (z_M, r_M)` is the inner mouth point of one meridian.
`beta` is the inner wall's tangent angle at M, measured from the +z axis, from
the analytic profile derivative. `a = 90 deg - beta` is how far the wall is
from perpendicular to the axis at the mouth. `R` is the roundover radius and
`w` the wall thickness.

## What ATH builds (V2026-08c, measured)

Base case: `OSSE = { L = 120, r0 = 12.7, a0 = 10, a = 45, s = 0.7, k = 1,
n = 4, q = 0.995 }`, free standing, `Mesh.WallThickness = 5`,
`Mesh.LengthSegments = 20`, linear z-map. Its mouth point is
`M = (120, 175.611)`.

1. **Inner arc.** ATH appends `Segments` points after M. For every probe they
   lie, within 0.0008 mm, on a circle of radius `R` centred at
   `(z_M - R, r_M)`, directly behind the mouth point, at angles

   ```text
   theta_k = -a_hat + k * (90 deg - a_hat) / Segments,   k = 1 .. Segments
   ```

   measured from the +z direction at the centre. M itself sits at `theta = 0`.
   `a_hat` is the value ATH logs as "using mouth angle". For `R = 25`,
   `Segments = 5` the points are (119.416, 180.985), (116.763, 187.915),
   (112.193, 193.760), (106.108, 198.007), (99.045, 200.281).
2. **A corner at the mouth.** Because the circle is centred directly behind M,
   the arc leaves M radially, at 90 degrees to the axis, whatever the wall
   angle. The wall arrives at `beta`. The two differ by `a`: 4.08 degrees in
   the base case, 8.64 degrees at `s = 0.3`.
3. **The arc ends early.** The last point is at `90 deg - 2 a_hat`, so the lip
   does not turn fully back. For an unterminated horn (`s = 0`, wall at 45
   degrees) the arc has no sweep at all: the five points collapse onto M.
4. **`a_hat` depends on the sampling grid.** It is an estimate from grid
   points, not the analytic angle (4.076 degrees in the base case):

   | `Mesh.LengthSegments` | 10 | 20 | 40 | 80 | 160 | 320 |
   | --- | --- | --- | --- | --- | --- | --- |
   | logged mouth angle, degrees | 7.143 | 4.656 | 3.998 | 3.913 | 4.045 | 4.109 |

   It also changes with the z-map (4.656 linear, 3.726 with
   `0.5,0.2,0.5,0.8`, 3.487 with the default map). The rule ATH uses was not
   identified; chord, three-point, spline and extrapolation estimates from the
   exported grid all fail to reproduce it. The arc end therefore moves with the
   grid: (99.045, 200.281) at 20 segments, (98.478, 200.368) at 40.
5. **Outer wall.** The outer surface gets `Segments` points on a concentric
   circle of radius `R - w` (20.000 mm for `R = 25`, `w = 5`; 15.000 for
   `w = 10`), and the lip is closed between the two arc ends.
6. **Radius is not scaled.** With `Scale = 2` the horn doubles and the arc
   keeps the offsets of `R = 25`.
7. **Segments** changes only the number of points on the same arc (1, 3, 5,
   10 tested; end point identical). `Segments = 0` adds nothing. Omitting
   `Segments` gives 5.
8. **Quadrants** 1, 14 and 1234 give the same meridian.
9. **No validation.** ATH builds a roundover for every case below without a
   diagnostic, and the result is not a usable lip:

   | Setting | What ATH produces |
   | --- | --- |
   | `R <= w` (R = 5 or 4, w = 5; R = 25, w = 30) | outer arc of radius 0, or an inverted outer arc of radius `w - R` |
   | `Radius = -10` | an arc curling forward and inward, in front of the mouth |
   | `Radius = 0` | a displaced inner point list |
   | R-OSSE | logged mouth angle -84.248; points scattered behind the rolled-back mouth |
   | `ABEC.SimType = 1` (infinite baffle) | the same arc, behind the baffle plane |
   | `Morph.TargetShape = 1` | the arc is added before the morph; off the principal meridian its radial reach is scaled with the mouth radius (24.577 mm at 0 degrees, 23.465 mm at 90 degrees), so it is no longer a circle of radius R |
   | `Mesh.Enclosure` | the enclosure front starts at the arc end (z = 99.045), behind the mouth plane |
   | `ABEC.SimProfile = 0` (CircSym) | the roundover is logged but the node list ends at the mouth radius, followed by a small stray loop |
   | `Mesh.Roundover = 25` (scalar form) | ATH exits with an access violation |

## Mesher construction (proposed)

The roundover is a **geometry control** of the free-standing wall, in the
meridian plane of each profile:

```text
t  = (cos beta, sin beta)                  wall tangent at M
n  = (-sin beta, cos beta)                 unit normal, toward the rear and outward
C  = M + R n                               arc centre
arc(phi) = C + R (sin(beta + phi), -cos(beta + phi)),   0 <= phi <= 180 deg - beta
```

- The arc starts at M with the wall's own tangent: position and tangent are
  continuous at the mouth (G1). Curvature is not; a jump from the wall's
  curvature to `1/R` is permitted.
- The arc ends where its tangent points straight back along -z. The lip end is
  `E = (C_z, C_r + R)`, the largest radius of the device.
- `beta` comes from the analytic profile derivative at the mouth, never from
  grid points. The construction is independent of `length_segments`,
  `angular_segments`, the z-map and the preview level of detail.
- The mouth plane `z = z_M` stays the front-most plane. Overall depth is
  unchanged; overall width grows to `2 (C_r + R)`.

Worked example, base case with `R = 25`: analytic `beta = 85.924143`,
`a = 4.075857`.

```text
M = (120.000000, 175.610710)
C = ( 95.063229, 177.387639)
sweep = 94.075857 deg
E = ( 95.063229, 202.387639)        device width 404.775 mm
```

ATH's polyline for the same config ends at (99.045, 200.281) and its device
width is 400.56 mm. At `s = 0` (wall at 44.851 degrees) this construction
gives `C = (102.368370, 140.600498)`, a sweep of 135.149 degrees and
`E = (102.368370, 165.600498)`, where ATH produces no arc.

### Outer wall and closure

The outer wall is the existing offset of the finished inner surface by `w`
along the outward material normal. Over the arc that offset is the concentric
arc of radius `R - w` about C, from the outer mouth point to
`E_outer = (C_z, C_r + R - w)`. The lip is closed by the existing ruled mouth
face, moved from the mouth ring to the arc end: a flat annulus at `z = C_z`
between `E_outer` and `E`, facing the rear. No other closure is added.

This requires `R > w`. At `R = w` the outer arc degenerates to a point and
beyond it the offset inverts, which is what ATH emits.

### Scale

`R` is in finished millimetres and is **not** multiplied by `Scale`, the same
convention the mesher already applies to wall thickness and the one ATH was
measured to use.

### Discretisation

`Segments` is not a geometry input. The lip is built as an exact circular arc
of revolution. The mouth ring and the lip-end ring are mandatory stations in
every consumer, so no fit smooths across either. Solve-mesh element size on the
lip follows `mouth_res_mm`, capped so that the arc is resolved by at least six
elements over its sweep. The preview stays error-bounded against the exact arc
and independent of the design's segment counts.

### Supported combinations

Initial support is the case the forum request describes. Everything else is
refused with a stated reason.

| Combination | Status | Reason when refused |
| --- | --- | --- |
| OSSE, free-standing, `wall_thickness_mm > 0`, circular mouth (no morph, no guiding curve, no azimuth-dependent parameters), planar mouth | supported | |
| Quadrants 1, 12, 14, 1234 | supported | the arc lies in the meridian plane, so symmetry planes cut it cleanly |
| `R <= w` | refused | `Mouth roundover refused: radius must exceed the wall thickness.` |
| `R` not finite or `R <= 0` | refused | `Mouth roundover refused: radius must be finite and positive.` |
| Wall tangent at the mouth pointing backward or undefined | refused | `Mouth roundover refused: mouth tangent is backward or undefined.` |
| Lip or its offset meets the outer wall, the rear closure or itself | refused | `Mouth roundover refused: the lip folds or intersects the wall.` |
| Non-circular mouth: morph, guiding curve, azimuth-dependent parameters | refused | `Mouth roundover refused: a non-circular mouth has no qualified lip construction.` |
| R-OSSE | refused | `Mouth roundover refused: R-OSSE terminates by its own rollback.` |
| ICW, FREEFORM | refused | `Mouth roundover refused: this profile family has no qualified lip construction.` |
| Infinite baffle | refused | `Mouth roundover refused: an infinite-baffle model has no free mouth edge.` |
| Enclosure | refused | `Mouth roundover refused: an enclosure replaces the free mouth edge.` |
| Bare inner surface (`wall_thickness_mm = 0`) | refused | `Mouth roundover refused: the lip needs a wall thickness.` |
| Throat extension, slot, throat stretch, throat adapter | supported | they do not change the mouth construction |

A non-circular mouth needs the arc in the plane normal to the mouth outline,
not in the meridian plane, and a construction for the corners. ATH's behaviour
there (arc added before the morph, then scaled) is not a circle of radius R and
is not adopted. That case stays refused until it has its own contract.

### Config and import

- Native key: `mesh.mouth_roundover_radius_mm`, default `0` (off). Off and
  absent are the same geometry and take the existing code path unchanged.
- ATH text import: `Mesh.Roundover = { Radius = R  Segments = N }`. `Radius`
  maps to the native key for a supported combination. `Segments` is accepted
  and unused. Any other member, the scalar form `Mesh.Roundover = R`, and any
  unsupported combination are refused.
- The radius enters geometry identity, preview and solve cache keys and the
  CAD bundle identity. Changing it while off changes nothing.
- The preview dimensions metadata reports the widened overall size.

### What an imported roundover does not reproduce

An imported roundover is built with the construction above, not with ATH's.
For the same `R` the differences are:

- no corner at the mouth (ATH: a corner of `a` degrees);
- the lip turns fully back (ATH: stops `2 a` degrees short), so the device is
  wider: 404.78 mm against 400.56 mm in the base case;
- the result does not depend on the sampling grid (ATH: the arc end moves by
  about 0.6 mm between 20 and 40 length segments).

Both arcs have radius R and start at the same mouth point; their centres are
`R a` apart to first order (1.8 mm in the base case).

## Validity

All checks are made on the analytic construction, before meshing:

| Rule | Check |
| --- | --- |
| Finite, positive radius larger than the wall | scalar input validation |
| Forward, defined wall tangent at the mouth | analytic derivative, `0 < beta < 180 deg` and nonzero norm |
| Positive radii | every arc point has `r >= r_M` when `beta <= 90 deg`; evaluate the arc's minimum radius otherwise |
| Lip clear of the outer wall and rear closure | the lip end plane `z = C_z` must lie in front of the rear closure; the outer offset must pass the existing fold and intersection checks on the finished surface |
| Planar, circular mouth | refuse otherwise, per the table above |

## Compatibility

- With the roundover off, existing designs give identical geometry, mesh and
  STEP output; the new path is not entered.
- Scalar and vectorised evaluators agree on M, `beta`, C and E to 1e-9 mm.
- Preview, solve mesh and STEP describe the same lip: each within 0.01 mm of
  the exact arc, pairwise within 0.02 mm. STEP reopens symmetry sectors as
  today and the lip is part of the closed solid.
- Physical groups: the lip and its closure belong to the rigid wall group, with
  normals continuous with the inner wall across the mouth ring.

## Implementation test plan mapped to C6 Accept

| C6 Accept | Assertions | Mutation that must be caught |
| --- | --- | --- |
| Unchanged when disabled | Off and absent give the pre-C6 grid, mesh hash and STEP for free-standing, enclosure and infinite-baffle fixtures | New path entered while off; inactive radius changes identity |
| Watertight, correct normals | Closed surface, no open edges beyond symmetry planes, outward normals on lip, offset arc and closure; refusals for `R <= w`, folded lips and unsupported combinations | Lip left open at the arc end; outer arc built with radius `R + w`; closure facing forward |
| Preview agrees with export | Preview, solve mesh and STEP against the exact arc at coarse and fine levels; mouth and lip-end rings present in all three; dimensions metadata includes the lip | Lip only in the solve mesh; arc splined across the mouth ring; radius scaled by `Scale` |
| Mesh refinement | Lip element count and chordal error against the exact arc at two mesh sizes; result independent of `length_segments` and the z-map | `beta` taken from grid points; lip resolved by a fixed segment count |
| ATH reference | The ATH V2026-08c probes as fixtures: every ATH inner point lies on the circle of radius R centred at `(z_M - R, r_M)`, and the documented differences from this contract hold to 0.001 mm | Claiming ATH parity for the lip; changing the construction without updating the stated differences |

Fixtures to archive when implementation begins (config, ATH log and the
`bem_mesh.geo` point list for each): the base case at `R` 10, 25 and 40;
`Segments` 1, 3 and 10; terminations `s` 0, 0.3 and 1.2 and `q = 0.98`; wall
thickness 10; `Mesh.LengthSegments = 40`; `Scale = 2`; quadrants 1 and 14; and
the refused cases (`R <= w`, negative radius, R-OSSE, infinite baffle, morph,
enclosure, CircSym).

## Owner questions

1. **Tangent-continuous lip, or ATH's circle?** Recommended: the construction
   above. ATH's circle centred behind the mouth point leaves a corner at the
   mouth, gives no roundover at all on an unterminated horn, and ends at a
   grid-dependent angle. The alternative is to copy ATH's circle with the
   analytic angle, which keeps the corner.
2. **Where does the lip end?** Recommended: where the tangent points straight
   back (a sweep of `180 deg - beta`), so the closure is a flat rear-facing
   annulus. The alternative is ATH's shorter sweep of `beta`.
3. **Should ATH `Mesh.Roundover` be imported at all, given the result differs
   from ATH's by the amounts stated?** Recommended: yes, for the supported
   combinations, with the differences documented and shown in the import
   report. The alternative is a native-only option with the ATH key refused.
4. **Radius not multiplied by `Scale`?** Recommended: yes, matching wall
   thickness and the measured ATH behaviour.
5. **Refuse `R <= w` rather than thin the wall locally?** Recommended: refuse.
   A lip thinner than the wall needs a variable-thickness wall, which is a
   separate feature.
6. **Non-circular mouths (morph, guiding curve) stay refused in the first
   release?** Recommended: yes; they need a lip in the plane normal to the
   mouth outline and a corner rule, under their own contract.
