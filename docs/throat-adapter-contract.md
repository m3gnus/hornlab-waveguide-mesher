# Curved throat adapter: geometry and import contract

**Status: contract for Program C5, reviewed by the owner on 2026-09-30 (decisions below).**
This document adds no executable behavior. Existing imports must not be described
as supporting `Throat.Ext.Ctrl` merely because they retain its text. The future
implementation must distinguish `off`, `ath` and `authored` modes and satisfy all
three groups in C5's Accept table. C4 throat stretching remains a separate feature.

## Evidence and limits

The public references are:

- [ATH thread](https://www.diyaudio.com/community/threads/acoustic-horn-design-the-easy-way-ath4.338806/),
  posts #15814 (2024-10-12), #16930 (2025-01-22), #20005 and #20006
  (2026-02-25). #15814 defines the six coordinates and the join at x = 0;
  #16930 describes removal of a throat length; #20005 says the Gen2 scripts
  split at x0 = 41 mm; #20006 gives the 36-STD-2 controls.
- Stored degree-mode states of [ph2ckhdurh](https://www.desmos.com/calculator/ph2ckhdurh),
  [ct3lhumv1r](https://www.desmos.com/calculator/ct3lhumv1r) and
  [kkw0aexxgb](https://www.desmos.com/calculator/kkw0aexxgb).
  The last explicitly labels base-profile dimensions in mm and angles in degrees.
- ATH 4.8.2 User Guide, §4.1.1: `Throat.Diameter`, `Throat.Angle`,
  `Throat.Ext.Angle`, `Throat.Ext.Length`, `Slot.Length`, `Length`.
  It documents the conical extension, not the later cubic-control import.
  Appendix A's mathematical function `trunc` is not a definition of the
  geometry parameter `trunc`.
- R-OSSE Waveguide rev7, and the existing [geometry contract](geometry-contract.md).

The stored states and thread posts establish a geometric construction. They do
**not** establish every behavior of the ATH executable or the Gen2 scripts.
Sections below mark those gaps and give binding refusals and experiments; no
inference about executable precedence is an import rule. Smooth connection is
the imported design's responsibility. Its controls must never be adjusted to
make a smoother curve.

Source inspection: mesher base `5c8ea4dc`, WG read-only snapshot `d77515be`.
At these snapshots `calculate_osse` / `calculate_osse_curve` and
`calculate_rosse` / `calculate_rosse_curve` implement extension → slot → body.
The OSSE branch `throat_profile == 3` replaces the **whole** body with a circular
arc; it is not this adapter and remains outside C5. ATH text import already
refuses `Throat.Profile` other than 1. `Throat.Ext.Ctrl` has no geometry mapping
in either repository. WG can retain it as an extra key; the mesher's fixed
profile mapping drops it without producing its geometry. Both need explicit recognition and
refusal before a supported adapter import is enabled.

## Coordinates and common curve

Use `(z,r)` for meridian geometry in mm, with increasing z toward the mouth.
The Desmos/ATH drawing calls these `(x,y)`; drawing x is the mesher's axial z,
not the 3D horizontal coordinate. At azimuth phi in radians the surface is

```text
S(phi,u) = (r(phi,u) cos(phi), r(phi,u) sin(phi), z(phi,u)).
B(u) = (1-u)^3 P0 + 3u(1-u)^2 P1 + 3u^2(1-u) P2 + u^3 P3,  0 <= u <= 1
B'(0) = 3(P1-P0);  B'(1) = 3(P3-P2).
```

Controls and handle lengths are physical mm before global `Scale`. Half-angles
are degrees relative to the positive axial direction, converted once with
`theta_rad = pi theta_deg / 180`. Bézier u is dimensionless; it is neither
axial distance nor arc-length fraction. The final geometry uses the existing
positive global scale and rigid placement conventions exactly once.

## Imported ATH adapter

### Proven coordinate mapping

Given `Throat.Ext.Ctrl = c1,v1,c2,v2,c3,v3`, in the **join-local** frame:

```text
P0 = (c1,v1)           driver-side endpoint
P1 = (c2,v2)           driver-side handle endpoint
P2 = (c3,v3)           body-side handle endpoint
P3 = (0,rJ)           automatic fourth point
rJ = r_base(tJ);       zJ = z_base(tJ)
body_local(t) = (z_base(t)-zJ, r_base(t)), tJ <= t <= t_end.
```

The six entries are three absolute axial/radius pairs, in order from the driver
end toward the body. The radius values are distances from the axis, not offsets
from r0 or diameters. Negative c values place the adapter behind the join. For
example `c1 = -38` gives 38 mm of axial adapter reach. There is no explicit fourth
pair and no normalized handle-scale interpretation. The graphs define
`v3 = rJ + c3 * r_base'(tJ)/z_base'(tJ)` to *author* a tangent handle. An import
must use the supplied v3, even when it does not equal that expression.

A mesher driver-plane datum may add the single rigid axial translation `-c1`:
`B_z(0)=0`, `B_z(1)=-c1`, and the retained body becomes
`z_base(t)-zJ-c1`. This is a canonical internal placement, not a claim about
ATH's exported origin. Recover ATH placement from measured export metadata;
unverified placement cannot be advertised as exact CAD/source parity.

Preserve the six values, chosen base-profile split, fourth radius, base-profile
parameters, retained interval, all meridian positions and tangent directions,
and any corner at the join. The imported base must also match its qualified
formula/domain policy: today's `_calculate_rosse_main` mouth-term clamp is
not proof that a later ATH script or a literal Desmos equation clips that term.
Do not project P2 onto the body tangent, replace v1
with `Throat.Diameter/2`, clamp radii, rescale handles, refit the body, or fall
back to a straight extension. Exact means the analytic construction is retained;
mesh/STEP approximations have the declared fitting tolerance below.

### Worked imported example

The stored Gen2 template has `R=200, r0=18, a=40, a0=8, k=2, r=0.3,
m=0.8, b=0, q=4, tJ=0.12`. Applying the R-OSSE equations in the existing
geometry contract, with all trigonometry in degrees, gives:

```text
L = 249.150229694052 mm
zJ = 27.696587829709 mm;  rJ = 29.223655005882 mm
dr/dz at tJ = 0.607947752891
P0 = (-38,17.78); P1 = (-27.4,18.6)
P2 = (-12.8,21.441923768883); P3 = (0,29.223655005882)
B(0.5) = (-19.825,20.891178289066) mm
exit half-angle = atan((18.6-17.78)/10.6) = 4.423505142588 deg.
```

Its endpoint derivative is `(38.4,23.345193710996)` mm per unit u, parallel to
the body tangent. This smoothness follows from this state's derived v3, not
from an import-side correction. After translating to the driver plane the
midpoint is `(18.175,20.891178289066)` and the join is at z = 38 mm.

The published 36-STD-2 string is
`-41.855,17.78,-31.238,18.896,-18.709,20.4`. Its driver diameter is
35.56 mm, axial reach is 41.855 mm, and its initial derivative is
`(31.851,3.348)`, giving 6.000578248573 degrees. Its fourth radius is **not** in
that post. Do not pair these controls with the template's rJ and call the result
36-STD-2. Likewise the template's calculated zJ = 27.697 mm is not the x0 =
41 mm split of the distributed Gen2 scripts. Those are different base designs.

### `trunc`, lengths and acceptance boundary

Established: `trunc` removes a throat part of a specified length, while the
Desmos construction cuts at a specified parameter tJ and translates the
retained body to join-local zero. The evidence does not establish whether ATH
uses axial length, arc length, a script-derived parameter or a root-selection
rule on a folded curve; it also does not establish which frame or C4 stage it
measures. Never equate `trunc` to `tmax` (a mouth-side parameter limit).
The two length readings would solve different equations:

```text
axial candidate: z_base(tJ)-z_base(0) = trunc
arc candidate: integral(0..tJ, sqrt(z_base'(t)^2+r_base'(t)^2) dt) = trunc.
```

For the stored Gen2 template at tJ=0.12 they give 27.696587829709 mm
and 30.080408924402 mm respectively (numerically integrated arc length).
This difference is why a length cannot be converted into a split without A1.
Neither candidate is claimed as ATH's rule.


For reference, the existing **non-cubic** implementation has these rules:

```text
driver radius = r0 - E tan(Aext),  E = Throat.Ext.Length
OSSE ATH total mode: body L = Length - Slot.Length
                    total = Length + E
OSSE native profile mode: total = L + Slot.Length + E
R-OSSE: derived L and main curve unchanged by E or slot;
        parameter layout span = E + Slot.Length + tmax*L.
```

Example: OSSE ATH `Length=120`, `Slot.Length=10`, `E=20`, `Aext=5`,
`Throat.Diameter=36` gives body length 110 mm, total 140 mm and driver radius
16.25022673 mm. These established straight-extension rules **cannot** be
transferred to a cubic import without measurement. `Throat.Ext.Length` may be
ignored, overridden, combined or used for sampling with `Ctrl`; the evidence
settles none of those alternatives. `Throat.Ext.Angle` must not override the
angle implied by P1-P0 without proof. Slot placement, Length budgeting and
Throat.Diameter precedence with Ctrl are also unestablished.

The first implementation may enable only an ATH executable/script version and
combination with retained reference output proving the split, precedence,
placement and sampling interpretation. A normalized `ath` payload carries that
verified interpretation identifier plus explicit base split tJ; an arbitrary
user-supplied tJ is not evidence of ATH parity. Until then raw Ctrl imports are
refused, including a zero or absent `trunc`. Retaining a refused file's original
text is allowed; preview, solve and export are blocked consistently.

The following exact messages are part of the proposed import contract. Use the
first applicable specific reason, then include key/location context separately.

| Unknown or unsupported behavior | Refusal message | Experiment |
|---|---|---|
| How split is selected, including absent/zero `trunc`, and where `trunc` is declared | `ATH adapter import refused: base-profile split semantics are not verified for this ATH version.` | A1 |
| Units, measurement frame, root selection and valid range of nonzero `trunc` | `ATH adapter import refused: trunc length and split selection are not verified.` | A1, A2 |
| Ctrl with nonzero Ext.Length or Ext.Angle; precedence over diameter or angle | `ATH adapter import refused: control-point and straight-extension precedence is not verified.` | A3 |
| Ctrl with a nonzero Slot.Length or ambiguous Length budget | `ATH adapter import refused: slot placement and Length budgeting are not verified.` | A4 |
| Automatic source location/shape and exported axial origin | `ATH adapter import refused: driver-plane placement and source semantics are not verified.` | A5 |
| Scale application to controls/trunc/wall | `ATH adapter import refused: adapter scaling semantics are not verified.` | A5 |
| Control expressions, azimuth-dependent controls/split, different H/V bodies, or interpolation | `ATH adapter import refused: azimuth-dependent adapter semantics are not verified.` | A6 |
| Morph, non-circular cross-section, profile rotation, bulge h or guiding curve changing the adapter/join | `ATH adapter import refused: profile-transform order is not verified.` | A7 |
| Nonzero C4 s1/s2 with Ctrl or trunc | `ATH adapter import refused: stretch and adapter ordering is not verified.` | A8 |
| Unexpected arity, non-finite controls, or a control expression without verified grammar | `ATH adapter import refused: expected six finite scalar coordinates in mm.` | A6 |
| Base formula clipping/domain policy differs from the literal imported curve | `ATH adapter import refused: base-profile clipping semantics are not verified.` | A9 |
| Wall construction and tessellation not qualified for the selected imported case | `ATH adapter import refused: wall and surface fitting semantics are not verified.` | A7, A9 |
| Unsupported base formula, hidden arc, rollback/arcterm or script-defined geometry | `ATH adapter import refused: this base-profile construction is not supported.` | A9 |

Finite scalar controls are the initial proposed grammar, not a claim that ATH
accepts only scalars. Scale must be positive and uniform; its Ctrl behavior is
also gated by A5. Each newly proven combination narrows the refusal, never the
exact-preservation requirement. No source in this review settles the later
ATH wall or tessellation behavior; A5/A7/A9 must cover those before claiming
complete imported artifact parity.

### Experiments to close the gaps

Use the actual adapter-capable ATH release and its complete runnable config;
record executable/version, config bytes, diagnostics and unrounded GridExport
coordinates. The user guide alone cannot supply a runnable Gen2 script. Each
experiment below specifies changes to that baseline. Unsupported syntax or
ignored keys is evidence of refusal, not a zero-valued result. Capture meridians
at 0, 45 and 90 degrees, source boundaries, and mouth bounds. Increase sampling
until fit uncertainty is below 0.001 mm; fit the cubic in parameter u, not in z.

Baseline profile and controls for the experiment matrix:

```text
R-OSSE = {
  R = 200
  r0 = 18
  a = 40
  a0 = 8
  k = 2
  r = 0.3
  m = 0.8
  b = 0
  q = 4
  tmax = 1
}
Throat.Ext.Ctrl = -38,17.78,-27.4,18.6,-12.8,21.441923768883
Throat.Ext.Length = 0
Throat.Ext.Angle = 0
Slot.Length = 0
Scale = 1
```

This is the **profile/parameter fragment**, not a substitute for the release's
required mesh/output/source boilerplate. Enable that release's GridExport and
export its standard single source; keep all other settings fixed. Also run the
same complete config with Ctrl omitted to recover the base curve independently.

| ID | Config variants | What to measure / decision |
|---|---|---|
| A1 | In the release's actual profile script/block, test `trunc` omitted, 0, 10, 27.696587829709, 41; separately test a top-level declaration and inspect recognition diagnostics. Use b=0 and b=0.25. | Identify the accepted key scope; map the join to base t by both coordinates. Compare removed axial distance, integrated arc length and t. Record body translation, unchanged retained coordinates and how P3 is chosen. |
| A2 | Use r=0.35,m=0.8,b=0.25,tmax=1; test trunc at an axial value having two base t roots, and at negative, beyond-depth and beyond-arc-length values. | Which root, or refusal? Is the mouth limit retained independently? Distinguish clipping, invalid input and ignored syntax. Repeat with tmax=0.8. |
| A3 | Ctrl fixed; Ext.Length=0/10/20, Ext.Angle=0/5/15 in a cross-product; Throat.Diameter=36/40 where accepted, then v1=16/17.78 while r0=18 stays fixed. | P0..P3, driver radius, extent and retained body. Determine precedence and whether a straight segment survives; compare Throat.Angle=0/8/15 separately from the controls. A deliberate v3 perturbation +2 mm must remain a tangent mismatch if preservation is the true behavior. |
| A4 | OSSE baseline `Length=120, Throat.Diameter=36, Throat.Angle=8, Term.s=0, Coverage.Angle=40`; same Ctrl; Slot.Length=0/10/20 and Length=120/140. Repeat R-OSSE with Length omitted/120/140. | Locate slot relative to cubic/body; compare split radius, mouth radius, total extent and retained body length. Establish whether changing Length changes split or only termination; do not reuse the cone length rules. |
| A5 | Scale=1/2, Mesh.VerticalOffset=0/7; automatic Source.Shape=-1 versus the release's explicit flat source. | All control positions, source rim/plane/normal, export origin, units and whether scale reaches Ctrl/trunc/wall thickness exactly once. Compare GridExport to mesh and STEP/STL boundaries, including any source recess. |
| A6 | Replace accepted a with `40+5*sin(p)^2`; test a control value `17.78+sin(p)`, then a per-p trunc if grammar allows. | Parser acceptance, evaluation units of p, P3 and split at all azimuths, control interpolation and source circularity. Repeat discontinuous/non-finite expressions to establish failure behavior. |
| A7 | Morph.TargetShape=0/1/2 with Morph.FixedPart=0/0.3, explicit target width/height 420/360; test Rot=5, h=2 (OSSE), then the release's guiding-curve/non-circular settings individually. | Does the cubic change? Does the endpoint meet the transformed body, and with what tangent? Track morph progress/snap relative to split and Ctrl length; repeat wall thickness 0/2/6 and identify folds or repairs. Perturb v3 by +2 mm and measure the imported corner's outer-wall closure separately. |
| A8 | C4-enabled release, profile s1=0/0.45 and s2=0/0.2, at trunc=0 and 41; Ctrl unchanged; Ext/slot separately zero and nonzero after A3/A4. | Compare stretching the full composite versus the base only, and splits measured before/after stretch. Test radius preservation, controls, fourth point, tangent and mouth placement. |
| A9 | Ctrl with OSSE/R-OSSE separately; then Throat.Profile=3, rollback/arcterm and the actual Gen2 script. Sweep Mesh.LengthSegments=32/128/512 and Mesh.ThroatSegments omitted/8 where supported. | Record accepted combinations, mathematical body and tessellation; use a domain-edge profile that activates the current R-OSSE mouth-term clamp to establish whether ATH clips that term. Confirm that automatic P3 and fitting do not depend on sampling density. Unsupported combinations keep their refusal. |

Archive these as new parity fixtures before enabling their interpretation.
Executable experiments are pending; none were run for this documentation task.

## WG-authored adapter

### Controls, join and derivation

Proposed initial support: OSSE profile 1 and R-OSSE, including smooth
azimuth-dependent base parameters and supported morphs. Other formula families,
rotation/bulge/guiding-curve combinations without a qualified derivative path,
and script-defined body modifications refuse with
`Curved adapter refused: this body-transform combination has no qualified join tangent.`
No hidden circular-arc control is exposed.

The six user controls are finite scalar values:

| Control | Meaning |
|---|---|
| Driver exit diameter D | Circular driver opening, mm; separate from body r0 |
| Exit half-angle alpha | Wall direction from +z, degrees; -90 < alpha < 90 |
| Adapter axial length A | Driver plane to common join plane, mm; A > 0 |
| Join location tJ | Original **base curve parameter**; R-OSSE t, OSSE normalized z/L |
| Driver handle h0 | Euclidean distance P0→P1 in mm |
| Body handle h1 | Euclidean distance P2→P3 in mm |

The join must satisfy `0 <= tJ < t_end`; R-OSSE `t_end=tmax`, OSSE `t_end=1`.
A zero join is legal if its forward tangent is regular. A join at a folded
R-OSSE branch with a backward axial tangent is refused for the initial adapter;
a retained body may fold later. No axial inverse or nearest ring selects tJ.

Resolve the final body meridian `C_phi(t)=(Z_phi(t),R_phi(t))` after C4 stretch
and supported body shaping/morph, before wall offset, scale or placement. Use
the actual derivative with respect to the original t. Let

```text
J_phi = C_phi(tJ)
T_phi = C_phi'(tJ) / ||C_phi'(tJ)||
C_retained,phi(t) = (A + Z_phi(t)-Z_phi(tJ), R_phi(t)), tJ <= t <= t_end
P0_phi = (0,D/2)
P1_phi = P0_phi + h0 (cos(alpha),sin(alpha))
P3_phi = (A,R_phi(tJ))
P2_phi = P3_phi - h1 T_phi.
```

A and both handle lengths are axial/radial plane values, not 3D chord lengths
between azimuths. Only h0/h1 set derivative magnitudes. The construction gives
`B(1)=C_retained(tJ)` and `B'(1)=3h1 T_phi`, a positive multiple of the body
derivative: position and oriented tangent continuity (G1). It does not guarantee
matching speed (C1) or curvature (G2/C2). For diagnosis only,
`kappa=(z' r''-r' z'')/(z'^2+r'^2)^(3/2)`; a curvature jump is permitted.

The adapter replaces the discarded base throat. Existing Ext.Length,
Ext.Angle, Slot.Length and diameter-derived **straight** adapter aliases must
be zero/absent in authored mode; otherwise:
`Curved adapter refused: clear the straight extension and slot before enabling authored mode.`
Length remains the uncut OSSE body's nominal input, not the final assembled
depth. For OSSE without stretch, `depth=A+L(1-tJ)`; R-OSSE dimensions use the
full retained curve's bounds, never its final axial coordinate. The driver exit
D controls the source rim, and alpha controls the automatic source opening
angle; r0 continues to define the original base profile. An explicit source incompatible with that rim refuses
with `Curved adapter refused: source boundary does not match the driver exit.`

### Worked authored example and differing meridians

Use the first stored R-OSSE state (`a=35,a0=5,r0=18,k=2,R=200,r=0.35,
m=0.8,b=0.25,q=3.8`) and tJ=0.12. With D=25.4 mm, alpha=10 degrees,
A=45 mm, h0=10 mm, h1=15 mm, no stretch or morph:

```text
J = (32.095016032750,28.531905989029) mm
T = (1,0.525253518401) / sqrt(1+0.525253518401^2)
P0 = (0,12.7)
P1 = (9.848077530122,14.436481776669)
P3 = (45,28.531905989029)
P2 = P3 - 15*T  [approximately (31.720417295162,21.556758450419)]
B'(0) = 30*(cos(10 deg),sin(10 deg))
B'(1) = 45*T; body_retained(tJ) = P3.
```

The body moves axially by `45-32.095016032750=12.904983967250` mm;
its radii and parameter derivatives stay unchanged. The old throat interval
0..0.12 is removed, not compressed into the adapter.

Apply this construction at **every** phi, not just H/V and not by interpolating
two finished handle sets. For example, if H has the J and slope above while V
has `J_V=(28,24)` mm and slope 0.4 at the same tJ, then V has
`P3_V=(45,24)`, `T_V=(1,0.4)/sqrt(1.16)` and
`P2_V=(31.072849637,18.429139855)` for h1=15. H and V share the circular
P0/P1 and common axial join plane but have different radii and body handles.
Both join correctly. Intermediate azimuths use their own resolved C_phi.

This common-plane policy translates each retained meridian by its own
`A-Z_phi(tJ)`. It preserves each meridian's shape and tangent but can change the
body's cross-azimuth axial layout and mouth planarity. It is an explicit WG
authoring policy, **not** imported ATH behavior. The complete loft must pass
surface checks. A topology requiring a planar mouth refuses a resulting warped
mouth with `Curved adapter refused: the rebased body mouth is not planar for this build mode.`
Do not move tJ per azimuth or flatten the mouth to hide it. This policy is an
owner review question below.

### Morph, C4 and wall order

Proposed authored order is:

```text
original base curve and original t
→ C4 stretch of the base axial coordinate
→ supported cross-section/body morph (with qualified derivative)
→ resolve J_phi/T_phi and cut the body at tJ
→ derive the cubic and rebase retained meridians
→ wall construction → uniform Scale → rigid placement.
```

C4 transforms the body, **before** adapter construction:

```text
Z_S(t)=Z(t)+s1*(180/pi)*atan(s2*Z(t))
dZ_S/dt=[1+(180/pi)*s1*s2/(1+(s2*Z(t))^2)] dZ/dt.
```

s1 has units mm/degree in this convention; s2 has units 1/mm. R-OSSE uses its
full Z(t), including b; OSSE uses its base axial coordinate. At the example
join above, s1=0.45,s2=0.2 multiplies the axial tangent by about 1.12218436 and
reduces the join slope to about 0.46806348; the adapter's final handle must use
that slope. Its requested A and driver angle stay fixed. Stretching the finished
cubic instead would change A and alpha, defeating their meaning. This authored
policy does not settle ATH's order: nonzero imported stretch remains refused
until A8. Coordinate the shared derivative/layout API with C4 on
`feature/c4-throat-stretch`; do not copy a competing stretch implementation.

For authored mode define morph on the original body's normalized parametric
progress, independent of preview rings or adapter length; use its existing
body target-radius rule, with a continuous onset of zero value and derivative.
For R-OSSE progress is t/t_end, for OSSE it is t. Do not reapply morph to B.
For body progress p=t/t_end (R-OSSE) or p=t (OSSE), let f0 be the
continuous configured Morph.FixedPart in [0,1]. If f0=1 the morph is dormant;
otherwise the active authored schedule is

```text
f(p)=0                              for p <= f0
f(p)=((p-f0)/(1-f0))^rate            for f0 < p <= 1
rate > 1 when the morph is active, so f and its first derivative meet at f0.
```

A rate at or below 1 with an active authored morph refuses with
`Curved adapter refused: active morph must have a continuous onset tangent (rate greater than 1).`
Do not alter that rate silently. The off path retains today's allowed rates.
For `R_effective(t)=R_raw(t)+DeltaR*f(t)`, use
`R_effective'=R_raw'+DeltaR*f'` in T_phi. A morph active at tJ is consequently
included in P3/P2. Example: body `(Z',R')=(100,40)`, DeltaR=20, and
`f'(tJ)=0.5` gives slope `(40+10)/100=0.5`, not 0.4. If morph starts at tJ,
require f=f'=0 there; a kink or unqualified corner refuses via the qualified
join-tangent message. Mouth target bounds are resolved before the adapter cut,
and body shaping must not scale the circular driver radius. The disabled path
keeps today's station-snapped schedule unchanged.

Wall offset is computed from the **finished 3D inner surface**, including
azimuth derivatives, never from separate H/V offsets. Locally it is
`S_outer=S_inner+w*n_out`, with outward material normal n_out. A regular offset
requires a nonzero surface Jacobian and no principal-curvature factor
`1-w*kappa_i` crossing zero for the chosen normal convention. Example: a
concave offset with `kappa=0.2/mm` becomes singular at w=5 mm; w=6 mm cannot be
accepted just because the inner radius is positive. Validate the actual outer
surface globally as well. Initial C5 curved adapters refuse folds with
`Curved adapter refused: wall offset folds or intersects; reduce thickness or open the local curve.`
At a non-G1 imported join there is no unique surface normal. With positive
wall thickness initially refuse:
`ATH adapter import refused: a corner at the adapter join has no qualified wall construction.`
Zero-wall imported corners remain admissible if the inner surface is valid.
A future miter/fillet or other corner closure requires an explicit construction
and ATH parity fixture; do not average normals to invent that construction.
The existing OSSE envelope repair may be supported later only after preview,
solve and CAD qualify the same envelope. Do not silently repair the imported
inner cubic, or retain today's warning-only R-OSSE wall behavior for a new
adapter that fails the contract.

## Validity and certification

These rules apply to both modes; invalid imports are refused without modifying
the input. A geometrically valid imported tangent mismatch is allowed and
reported as a corner. Authored mode requires G1. Use double precision and report
phi, u/t interval and offending control in separate diagnostic context.

| Rule | Check | Refusal message |
|---|---|---|
| Finite coordinates, positive D/A and nonzero handles | Validate scalar inputs, units and formula domain before evaluation | `Curved adapter refused: dimensions and handles must be finite and positive.` |
| Positive radius everywhere | Solve cubic r'(u)=0 (quadratic), evaluate endpoints and every real root in (0,1); interval-bound retained body and full azimuth range | `Curved adapter refused: radius reaches zero or becomes negative.` |
| Forward adapter axis | Find minimum of quadratic z'(u) at endpoints and interior vertex; require z'>0 on [0,1] with certified numerical margin | `Curved adapter refused: adapter axial coordinate is not strictly increasing.` |
| Valid authored handles | Require 0<h0*cos(alpha)<A and 0<h1*T_z<A and h0*cos(alpha)<A-h1*T_z; require positive P1/P2 radii | `Curved adapter refused: handles cross, leave the adapter span or reach the axis.` |
| Regular forward body tangent at join | Analytic or certified derivative, nonzero norm and T_z>0; differentiate full resolved body, not a sampled chord | `Curved adapter refused: join tangent is zero, backward or undefined.` |
| No curve loops or overlap | Strict axial adapter rule proves its own injectivity. Check adapter against all retained body intervals and non-neighbor body intervals using bounded subdivision | `Curved adapter refused: adapter or retained body loops or intersects.` |
| Valid full inner loft | Certify continuous azimuth input, nonzero S_phi cross S_u and oriented cells; BVH plus nonadjacent intersection tests on error-bounded surface patches | `Curved adapter refused: inner surface folds or self-intersects.` |
| Valid outer wall and closures | Local offset Jacobian, outer/outer and inner/outer intersections, rim/cap/enclosure clearance and watertight topology | `Curved adapter refused: wall offset folds or intersects; reduce thickness or open the local curve.` |
| Unresolved numeric certification | Refine intervals/patches until a sign or separation is established, with bounded work; never treat a dense sample as proof | `Curved adapter refused: validity could not be certified at the requested tolerance.` |

The imported handle rule requires distinct P0/P1 and P2/P3, and strict forward
z throughout the cubic; it does not impose authored control ordering if the
analytic derivative proves validity. For authored mode the stronger ordered
control polygon is a conservative user limit. The example h0=10,h1=15,A=45
has axial handle projections 9.848 and about 13.279, whose sum is below 45.
Setting h0=40,h1=30 instead gives a sum above 45 and is refused. For radius
checking, a cubic with endpoint radii 1, handles -10,-10, and endpoint radius 1
has `r(0.5)=-7.25`: endpoints alone cannot certify positivity.

Only the adapter and join must be forward in z. Retained R-OSSE may fold back:
keep its original ordered t interval and certify no parametric self-intersection
or contact with the adapter. Never invert z or reject a legitimate rollback
solely for negative z' farther along the body. Positive radii plus increasing z
suffice for an individual adapter meridian; they do not prove validity of a
loft with varying z_phi, its offset or enclosure. Arbitrary expressions that
cannot be bounded over phi are refused with the certification message rather
than checked at H/V alone. Shared endpoint/edge contact is excluded from
intersection tests; other tangencies count as intersections.

Tolerance proposal: analytic endpoints/derivatives and scalar/vectorized
agreement within 1e-9 mm absolute (tangent direction within 1e-9 radians);
independent ATH GridExport agreement within 0.001 mm after documented datum,
units and export precision; preview/solve/STEP surfaces each within 0.01 mm
of the canonical inner adapter, with pairwise distance at most 0.02 mm. Small
features require tighter local limits. Offset surfaces require the same stated
positional bound. Interval certification includes floating-point error and
surface-fit error; if their bounds overlap a singularity or contact, refuse.
Tolerance never licenses a control-point correction.

## Compatibility, persistence and geometry identity

- Missing adapter or `mode=off` dispatches the exact existing geometry path:
  same parameter arithmetic, station maps, extension/slot rules, morph,
  wall behavior and source policy. Do not normalize/recompute existing designs
  through the new composite evaluator. Existing straight/conical extensions
  and diameter-based straight adapter derivation remain unchanged.
- Proposed canonical schema is a discriminated `throat_adapter` object with
  `mode=off|ath|authored` and `contract_revision=1`. Authored payload stores
  `driver_exit_diameter_mm`, `exit_half_angle_deg`, `length_mm`, `join_t`,
  `driver_handle_mm`, `body_handle_mm`. ATH payload stores the six raw scalar
  controls in order, verified `join_t`, original trunc/key scope if supplied,
  and the verified ATH interpretation identifier. Derived P3 is recomputed
  from the same base revision, never stored as an independently editable input.
- Missing and off are the same active geometry. Saved inactive controls may
  remain in editor state but must not change geometry identity. Switching to
  an active mode changes identity. Every active control, split, interpretation,
  source rim rule, C4 value and effective body transform enters canonical
  serialization, mesher translation and preview/solve/export cache inputs.
- Native save/load preserves full precision and mode. ATH round-trip preserves
  control order, original numeric text when unchanged, split semantics and
  raw `trunc` text; it must not convert an imported mismatch to authored G1.
  Native authored format uses its own explicit block/object, never emits a
  guessed `trunc` as if it were ATH-compatible. If ATH output cannot express
  the split/order, refuse: `ATH export refused: this authored adapter has no verified ATH representation.`
- Recognized geometry keys must leave WG `extra_keys` and the mesher ignored
  namespace. A text file carrying an unsupported adapter must fail before
  ordinary preview/solve/export. Old consumers must reject a format requiring
  active adapter support, rather than drop the block and build an old horn.
- WG currently hashes resolved geometry for CAD freshness in
  `server/exports/geometry_identity.py::geometry_hash_for_design`, preview's
  translated config in `server/preview/core.py`, and the normalized solve config
  in `server/mesh/builder.py::_solver_mesh_cache_key`. Preserve those chains;
  verify active parameters survive all projections. A semantic construction
  fingerprint (including adapter revision/mode/active controls) accompanies
  the resolved geometry identity, so different active constructions cannot
  alias merely because a coarse control grid misses their difference. Retain
  existing off-mode hash inputs. CAD bundle generator identity must carry
  this same construction fingerprint.
- Preview = solve mesh = STEP/CAD means one continuous composite definition,
  same split, source rim and wall rule; sampling may differ within the stated
  tolerance. Join and driver endpoint are mandatory semantic stations.
  STEP reopens symmetry sectors and removes the source cap as today; neither
  operation may change the adapter. An imported G1 mismatch must become an
  explicit shared edge, not a smoothing B-spline across the corner.

Example identity mutation: changing authored h1 from 15 to 16 mm changes P2
by `-T_phi` and the cubic interior by `-3u^2(1-u)T_phi`, giving a 0.375 mm
midpoint displacement. Native round-trip must retain it, all active request
keys/fingerprints must change, and cached preview/mesh/CAD output must miss.
Changing an inactive h1 must leave the off-mode geometry and identity unchanged.

## Verification against stored Desmos states

A throwaway scratchpad script reads the stored JSON (including the nested
`state` wrappers), asserts degree mode, parses the saved equation rows and
slider values, differentiates the stored body expressions for v3, and evaluates
the literal Bernstein equations. It compares an independent scalar R-OSSE
implementation and de Casteljau evaluation at 10,001 uniform parameters each
for both body t=0..1 and adapter u=0..1 per state. The body comparison includes
the displayed translation by x0; retained geometry uses only t>=tJ. Endpoints
are included even where the saved drawing excludes u=0. The script and results
stay outside the repository.

| Saved state | tJ | zJ mm | rJ mm | Maximum body deviation mm | Maximum adapter deviation mm |
|---|---:|---:|---:|---:|---:|
| ph2ckhdurh | 0.12 | 32.095016032750 | 28.531905989029 | 7.2462e-14 | 6.7504e-14 |
| ct3lhumv1r | 0.108 | 28.977448259642 | 26.943942596307 | 7.2462e-14 | 2.1611e-14 |
| kkw0aexxgb | 0.12 | 27.696587829709 | 29.223655005882 | 9.0994e-14 | 1.5889e-14 |

Maximum Euclidean deviation over **60,006 points: 9.0994e-14 mm**, below the
1e-9 mm numerical-reading tolerance. This is an independent numerical check of
the stored equations, **not** a comparison against ath.exe or a guarantee of
unmeasured import behavior. All 18 mutated readings (six per state) are rejected
by displacement above 0.001 mm or failure of the numerical domain: wrong fourth
radius, swapped axial/radius coordinates, radius treated as offset, reversed
handles, omitted body translation and radians in place of degrees. The
mutation sentinels fail as expected. Input SHA-256 digests identify the precise
snapshots:

```text
ph2ckhdurh: 5bfd377c01e262cd5f04f04fd6008fd13b6e7d25833036274d34e46cb1a2f9bf
ct3lhumv1r: 46384d4fa0f8f1f6be1422457b49bf4e52470b0504a6796f0f85dc50a9734bc2
kkw0aexxgb: 643ce49e90f6f2b6f10b92bd44271ed2629bd30e8f1b809a6472e69775f5e34e
```

## Implementation test plan mapped to C5 Accept

These are future acceptance tests; only the numerical-reading check above has
been executed for this contract. Tests must fail under the listed mutations.

| C5 Accept group | Required fixtures and assertions | Mutation that must be caught |
|---|---|---|
| Geometry: reference agreement | All three saved curves at endpoints, interior and split; A1–A9 parity fixtures for each enabled ATH combination, tolerances above; scalar/vectorized differential tests | Radius/diameter confusion, swapped pairs, dropped t^3*rJ, radians, wrong origin, clamp or hidden control smoothing |
| Geometry: endpoints/tangents | Authored exit diameter/alpha and G1 at every azimuth; H != V, nonzero b, tmax<1, exact non-ring tJ, and join before R-OSSE fold; imported intentional mismatch retained | P2 made from base a0 or H tangent alone; inversion of z; derivative from a coarse chord |
| Geometry: radii/handles/loops | Negative interior radius with positive endpoints, zero/huge/crossing handles, backward/zero join tangent, derivative touching zero, later R-OSSE fold valid, adapter/body contact invalid; uncertifiable phi expression refused | Endpoint-only radius test; H/V-only or sampled-only certification; rejection of every folded body |
| Compatibility: disabled | Exact off/absent grid and source equality against pre-C5 fixtures, including cone, slot, ATH length modes, morph and wall cases | New composite normalization entered while off; changing inactive controls invalidates old geometry |
| Compatibility: persistence/import | Native and supported ATH text round-trip; malformed/ambiguous imports refuse in both parsers; one-at-a-time mutation of all active fields reaches translated config, construction identity and all cache keys | Adapter left in extra_keys, missing join_t/interpretation fingerprint, fallback to straight extension |
| Integration: terminals | Error-bounded coarse/fine preview, acoustic fit and STEP surface distance against canonical cubic/body; mandatory endpoints/join, source rim, source removal, full/reduced domain and scaling/placement | Preview-only adapter, join skipped by z-map, source still at old r0 or auto cap still using a0 when alpha changes, scale twice, fitted spline smoothing an imported corner |
| Integration: morph/walls | Morph active at join and onset at join; varying H/V and intermediate azimuths; inner/outer intersection and offset singularity; imported non-G1 join with wall=0 accepted and wall>0 refused; planar-mouth mode refusal; C4 combined derivative test | Morph applied twice or to adapter, unstretched handle tangent, separate H/V wall offsets, warning-only folded outer wall |

Run targeted mesher formula/import/point-grid/preview/STEP tests first, then the
full mesher suite with required ATH parity through the compute broker. After
the lander-owned mesher pin move, WG needs schema/textcfg/registry/translation,
identity/cache and integration tests plus full Python and Node 20 frontend
checks. No pin change belongs in this contract-only branch.

## Implementation outline in dependency order

Names below were rechecked at the source snapshots above; new helpers are
explicitly identified as proposed. Do not begin this work before review.

1. **Evidence gate and input model.** Complete A1–A9 for the combinations to
   enable. Mesher `config_parser.py::parse_text_config` and
   `_reject_unsupported_ath_keys` recognize Ctrl/trunc and enforce gates;
   `config_builder.py::build_geometry_params`, `_validate_formula_specific_keys`
   and `_apply_driver_adapter` normalize the discriminated mode and reject
   mixed straight/curved settings. Keep straight adapter derivation intact.
   Add a proposed shared adapter payload/resolver rather than infer controls
   from existing diameter aliases.
2. **C4-compatible base and tangent API.** In `profile_formulas.py`, factor
   body point/derivative access from `_calculate_rosse_main`,
   `_rosse_main_curve`, `_rosse_main_coefficients`, `_osse_radius` /
   `_osse_radius_curve` (the latter in `profile_common.py`). Coordinate with
   C4's stretch helpers. Add proposed shared `resolve_throat_adapter`,
   `evaluate_throat_adapter` and derivative/validity helpers. Update
   `calculate_osse`, `calculate_osse_curve`, `calculate_rosse`,
   `calculate_rosse_curve` to dispatch off versus composite paths.
   `osse_length_config`, `rosse_axial_layout`, `osse_total_length` and
   `rosse_total_length` must distinguish layout span from actual bounds;
   never reinterpret authored join_t as today's composite-prefix parameter.
3. **Continuous body shaping and composite grid.** Extend
   `profile_sampling.py::_raw_radial_grid`, `_ThroatPrefix`, `_morph_schedule`,
   `_pin_axial_stations`, `build_point_grid_arrays` / `build_point_grid` and
   `profile_morph.py` derivative access. Adapter mode uses the specified
   continuous body-morph schedule; off retains its existing arithmetic.
   Publish a shared adapter-u/body-t station map, exact join and construction
   fingerprint. Keep all phi rows on one topological station schedule even
   with different P2/P3, and certify inner geometry before offset.
4. **Walls, source and downstream fitting.** Qualify
   `profile_sampling.py::_outer_offset_shell` and
   `freeform.py::validate_outer_offset_grid` on the
   finished composite surface, with strict C5 refusals. Update
   `config_builder.py::_source_auto_angle_deg`, `_requested_axial_layout`,
   `_build_acoustic_sampling_grid` and `resolve_geometry` so the source and
   fit use the new driver rim/angle and pinned join. Preserve corners in OCC
   builders rather than interpolate across them. Verify mode/closure guards.
5. **Preview and CAD.** Update
   `preview/horn.py::_semantic_t_stations`, `preview/api.py::_sample_master_level`
   / `build_preview_geometry`, and `viewport.py::build_viewport_geometry_from_config`
   to consume shared stations and certified walls. `cad.py::write_step_from_config`
   / `write_step` continue resolving/building the same geometry;
   `write_wglink` / `_identity_sections` carry its fingerprint and driver datum.
   Retain units, reopening and source-cap removal policies.
6. **WG model, persistence and adapters after dependency qualification.**
   `server/design/schema.py::DesignCommon` adds the discriminated payload with
   formula gates. `server/design/textcfg.py::_build_payload`, `parse`,
   `_serialize_canonical`, `serialize` consume/emit supported controls and
   refuse unsupported imports. `server/preview/translate.py::_profile` /
   `design_to_mesher_config` map the same payload for preview, solve and CAD
   with one unit/Scale conversion. Update model serialization and generated
   OpenAPI types; preserve old missing/off defaults and reject future required
   adapter formats. WG's shared clones remain read-only in this task.
7. **WG controls and identity.** Add mode-specific controls to
   `frontend/src/design/parameterRegistry.ts` including trace-key lists;
   update `frontend/src/api/designIo.ts` and `frontend/src/stores/design.ts`
   save/load and family conversions. Do not silently carry an active adapter
   into an unsupported family. Regenerate the parameter catalog through its
   existing `scripts/gen_parameter_catalog.ts` generator. Verify `server/preview/core.py::_cache_relevant_config`,
   `server/mesh/builder.py::_solver_mesher_config` / `_solver_mesh_cache_key`,
   `server/exports/geometry_identity.py::geometry_hash_for_design` /
   `geometry_hash`, and CAD freshness/bundle identity include the active
   construction and never discard it as mesh sampling input.
8. **Acceptance and delivery.** Execute the table above, required full suites,
   independent diff review and exact-pin qualification through the lander.
   Document the final enabled ATH interpretation matrix and restrictions in
   both repositories; all unmeasured cases keep their explicit refusal.

## Owner decisions (2026-09-30)

- Build the WG-authored adapter first. Imported ATH `Throat.Ext.Ctrl` stays refused
  per the table above until the A1–A9 experiments are run; they will be run against
  ATH V2025-12, which recognizes `Throat.Ext.Ctrl` and `trunc`.
- Question 1 (common join plane): accepted as recommended.
- Question 3 (imported tangent mismatch): accepted as recommended. Keep and report the
  corner; refuse a positive wall thickness at it.
- Question 5 (C4 order): stretch the body **before** deriving the adapter.
- Question 6 (morph and offset), including the continuous-onset rule for an active
  morph: accepted.
- Questions 2, 4, 7 and 8 proceed on their recommendations unless changed later.

## Owner questions and recommended answers

1. **Common join plane versus preserving cross-azimuth body placement?**
   Recommend the common driver/join planes defined here, with per-meridian
   rebasing and an explicit refusal when a required planar mouth is lost.
   A single global translation would preserve body placement but make adapter
   length vary around phi; that would need a different length-control meaning.
2. **How much ATH support should the first release claim?** Recommend enable
   only measured executable/script combinations, initially scalar controls,
   no Ext/slot conflict or unverified transforms. Keep every other import
   refused even if it produces a plausible drawing.
3. **Should imported tangent mismatch be invalid?** Recommend preserve and
   report a corner if the geometry passes validity; refuse a positive wall at
   that corner until its closure construction is qualified. Smoothness remains the
   imported designer's responsibility; authored mode always enforces G1.
4. **Allow an adapter whose axial coordinate folds?** Recommend strict forward
   adapter z and forward tangent at the join initially, while allowing a valid
   later R-OSSE body fold. An arbitrary folded adapter needs separate topology
   and source-clearance qualification.
5. **C4 order?** Recommend stretch the authored body before deriving handles,
   leaving D, alpha and A as physical user controls. Require A8 rather than
   imposing that policy on imported ATH geometry.
6. **Morph and offset policy?** Recommend resolve the transformed body tangent
   before deriving the adapter and never morph the adapter a second time.
   Refuse invalid C5 offsets initially; expand envelope repair only with
   terminal agreement tests. Curvature continuity is not a release promise.
7. **Tolerances and input breadth?** Recommend the numerical/ATH/artifact bounds
   above and scalar authored controls first. Smooth per-phi body variation is
   supported only with certifiable derivatives and surface validity. Add
   per-phi driver/handle controls only after defining the source shape.
8. **Format and old-reader protection?** Recommend a tagged native adapter
   object/block with a required format capability for active designs, a
   construction fingerprint, and a hard refusal on unsupported ATH export.
   Do not serialize a guessed ATH trunc or rely on retained unknown keys.
