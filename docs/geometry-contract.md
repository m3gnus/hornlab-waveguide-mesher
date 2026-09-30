# Geometry Contract

This document defines the geometry rules that `hornlab-waveguide-mesher`
implements for OS-SE/OSSE, R-OSSE, ICW, and FREEFORM waveguides. It also separates
canonical mathematical behavior from ATH compatibility behavior so code
changes do not hide reference-tool quirks inside generic helper names.

## Sources

- `Ath-4.8.2-UserGuide.pdf`
- `R-OSSE Waveguide rev7.pdf`
- `OS-SE Waveguide.pdf`

The PDFs are reference material, not generated artifacts in this repository.
This contract records the implementation rules derived from them.

## Coordinate System

Waveguide profiles are evaluated as axial/radial curves and then revolved or
sampled around the z axis.

- `z` is axial distance from the throat plane.
- `phi` is the angular coordinate around the waveguide axis.
- `r(z, phi)` is the radial distance from the z axis.
- A 3D grid point is `(r cos(phi), r sin(phi), z)`.

Partial domains use ATH quadrant semantics:

- `1`: x >= 0 and y >= 0
- `12`: y >= 0
- `14`: x >= 0
- `1234`: full domain

## OS-SE / OSSE Profile

The canonical OS-SE profile is the generalized oblate spheroidal base plus a
superellipse-like termination term:

```text
r_osse(z) =
  sqrt((k r0)^2 + 2 k r0 z tan(a0) + z^2 tan(a)^2)
  + r0 (1 - k)
  + s L / q * (1 - (1 - (q z / L)^n)^(1 / n))
```

Implementation rules:

- `r0` is throat radius.
- `a` is nominal coverage angle, as a half angle in degrees.
- `a0` is throat opening angle, as a half angle in degrees.
- `k` is throat expansion factor.
- `s`, `q`, and `n` control the smooth termination term.
- `q z / L > 1` clamps the termination term to `s L / q`.
- Throat extension and slot length are explicit axial sections before the main
  OS-SE profile. `r0` anchors the main waveguide throat; a throat extension
  tapers backward to the driver-end radius and does not enlarge the main curve
  or mouth.
- `Rot` transforms the unstretched 2D profile around `(0, r0)`, before
  the optional axial throat stretch.

For a freestanding OSSE wall, a regular normal offset is preserved exactly.
When neighboring normals cross inside a concave throat or guiding-curve groove,
the exterior remains the constant-distance envelope: the internal loops are
not part of the material boundary. The mesher computes that envelope from the
sampled acoustic surface's symmetric quad-center triangle fans, offset faces,
edge cylinders and vertex spheres, then samples its farthest radial intersection
in the original angular order. This applies to folds at any axial station,
including non-leading ones.
The acoustic grid and requested wall thickness are unchanged. The envelope is
exact for the sampled triangulation; its approximation to the analytic surface
therefore remains limited by the grid resolution.

If the original outer axial stations roll back, they are redistributed in
proportion to the input axial progress between the same throat and mouth
stations. The repair requires monotone input axial stations and must pass the
existing normal-flip and self-intersection checks. A rotated profile that fails
those preconditions retains its explicit fold diagnostic. FREEFORM's rejection
contract and the other profile families' offset behavior are unchanged.

Adaptive previews repair a folded offset after selecting their render grid,
including deferred rounded-rectangle morph walls. Healthy offsets keep their
existing path. Acoustic control-grid fitting likewise repairs only the accepted
grid, after all inner chord and curvature checks, not discarded probe grids.
Exact geometric bounds discard interior offset features, and
rotational reuse requires the entire sampled surface and resolved station
schedule to agree within floating-point noise; equivalent rows share both the
evaluated radius and station. These shortcuts retain the requested preview
detail. Rigid vertical placement is applied after intrinsic offset evaluation.

## R-OSSE Profile

R-OSSE is parametric. It is not a single-valued radius function of `z`; this is
what allows rollback/free-space termination shapes.

For `0 <= t <= 1`:

```text
c1 = (k r0)^2
c2 = 2 k r0 tan(a0)
c3 = tan(a)^2
L = (sqrt(c2^2 - 4 c3 (c1 - (R + r0 (k - 1))^2)) - c2) / (2 c3)

x(t) =
  L (sqrt(r^2 + m^2) - sqrt(r^2 + (t - m)^2))
  + b L (sqrt(r^2 + (1 - m)^2) - sqrt(r^2 + m^2)) t^2

y(t) =
  (1 - t^q) (sqrt(c1 + c2 L t + c3 (L t)^2) + r0 (1 - k))
  + t^q (R + L (1 - sqrt(1 + c3 (t - 1)^2)))
```

Implementation rules:

- `R` is waveguide outer radius.
- `a`, `a0`, `r0`, and `k` have the same angle/radius meaning as OS-SE.
- `r`, `m`, `b`, and `q` shape apex radius, apex shift, bending, and throat
  transition.
- `L` is derived from the requested mouth radius and profile parameters.
- Throat extension and slot length are explicit axial sections before the main
  R-OSSE curve. As with OS-SE, `r0` anchors the main waveguide throat; the
  extension tapers backward to the driver-end radius. The main R-OSSE curve,
  derived length, and mouth radius are unchanged by the extension, also when
  `tmax < 1`: the composite parameter shares the axis as
  `ext + slot + tmax * L`, so the main curve always ends at `t = tmax`
  (ath.exe: identical mouth with and without a 30 mm extension at tmax 0.8).

Compatibility note for OS-SE and R-OSSE throat extensions: the taper-back
implementation reproduces ATH's profile radius and total length exactly, but it
deliberately does not reproduce ATH's recessed driven cap behind the straight
extension duct. That recess is a transmission-line detail expected to matter
acoustically only for horns with a long throat extension; revisit it only if a
real device shows a response discrepancy.

## R-OSSE-S / OS-SE-S throat stretching

The reference is the degree-mode Desmos graphs `sljsrsipbq`, `5jogzxvqvb`
and `k0akvpmsnn`, published in the ATH thread, posts #18211–18214 and #18259.
The axial map is:

```text
S(x) = x + s1 * (180 / pi) * atan(s2 * x)
```

- Radius is unchanged. R-OSSE uses the complete main `x(t)`, including its
  `b` term; OS-SE uses the main axial parameter in millimetres.
- `s1` has units mm/degree and `s2` has units 1/mm. Both default to zero and
  must resolve to finite values in `[0, 10000]` at every evaluated azimuth.
  The explicit upper bound limits added displacement to 900,000 mm. This is
  far below half an ulp at the largest finite double, so the axial map stays
  finite for any finite input coordinate (an overflowing atan argument uses
  its finite limiting angle). Coefficients above the bound are refused during
  validation, including expressions, before acoustic resolution or meshing.
  They accept the same per-azimuth expressions as the other coefficients
  (`p` and expression trigonometry use radians; only this axial atan uses degrees).
- Either coefficient zero bypasses the map, preserving the existing floating
  point operations and geometry exactly.
- `S'(x) = 1 + (180/pi)*s1*s2/(1+(s2*x)^2) >= 1`. The map is strictly
  increasing in **x**. OS-SE remains axially monotone before `Rot`; R-OSSE
  preserves its intentional foldback in **t**, rather than straightening it.
- The native composition is extension, slot, stretched main profile:
  `x = ext + slot + S(x_main)`. The straight extension and slot keep their
  lengths and radii. For OS-SE without a prefix, ATH applies `Rot` first and stretches the
  resulting axial coordinate; the rotated radius is unchanged by stretch.
  The combination of stretch, `Rot` and a prefix is unmeasured and refused
  in both ATH text import and native evaluators, before mesh generation.
  This includes a slot alone, extension alone, and combined prefixes; an
  evaluated nonzero rotation with active stretch cannot use an unmeasured
  join convention. The scalar and vector evaluators check the actual radial
  and axial main endpoint against the prefix endpoint. A discontinuous
  composite meridian is refused. The map is analytically increasing; the
  vector evaluator additionally checks that distinct unstretched axial
  coordinates retain their ordering after stretch and prefix translation,
  refusing floating-point collapse. R-OSSE foldback is also checked against
  the unchanged straight prefix, on a 1,025-station full-main probe and on
  the requested vector stations; intersections are refused. Validation is
  cached by every resolved main coefficient, prefix input and truncation.
  Existing R-OSSE foldback that remains clear of the prefix is preserved.
- `L`/`Length` remains an **unstretched sampling/profile parameter**, not a
  requested final depth. Native main-length mode retains `main_L = L`;
  ATH total-length mode retains `main_L = Length - Slot.Length` and adds
  `Throat.Ext.Length` on top. The OS-SE mouth is at
  `ext + slot + S(main_L)` before rotation. The public total-length helpers
  continue to return the unstretched parameter span used by samplers.
  R-OSSE likewise retains its derived L and composite t/tmax allocation.
- Guiding-curve inversion, radial bulge and morph progress retain their
  unstretched parameter coordinates. Wall offsets are computed from the
  stretched surface. Preview, solve and CAD use the same profile evaluator.
  With stretch, a guiding-curve distance is therefore a parameter location,
  not a final physical axial distance.
- With either coefficient absent or a literal zero, config normalization and
  builder mappings omit both coefficients. Inactive geometry dataclass
  instances expose the legacy field schema to `asdict`/`astuple`, equality,
  hash and repr, exactly as at base `5c8ea4dc`. Dormant values do not contribute.
  Active instances include both coefficients in serialization and identity.
  Potentially active expressions are retained without deciding identity at
  a single azimuth. There is no OSSE/R-OSSE geometry memo. ICW seed memo keys
  canonicalize dormant stretch pairs and retain both active coefficients.
  Caller-owned cache keys should use these canonical geometry mappings.

**Measured ATH evidence (V2025-12):** all 36 paired probe exports carry
`; Ath version V2025-12` in their generated `config.txt`. The run logs have no
version banner. The portable points and generating configs are in
`tests/fixtures/throat_stretch/ath-v2025-12/`. All 60,096 exported profile
points (16 meridians per case) were compared in ordinal order. OSSE stations
come from the paired zero axial coordinate, undoing `Rot` where needed;
R-OSSE stations come from the published radial equation, preserving station
order through axial foldback. The 28 accepted cases agree in both evaluators
within 2.634e-5 mm, below the 2e-4 mm GridExport tolerance. Eight OSSE slot
cases are refused rather than claimed as parity; see below.

- Degree convention: OSSE `s1=0.5,s2=0.2,L=160` ends at 204.105045 mm,
  an addition of 44.105045 mm. Its log prints
  `s1=28.647890,s2=0.200000` (the input s1 multiplied by 180/pi).
  R-OSSE `s1=1,s2=0.05` has maximum depth 150.832199 -> 233.278973 mm,
  an addition of 82.446774 mm. R-OSSE logs contain no stretch diagnostic;
  the exported points establish the same degree convention. All paired
  main axial differences follow the degree map within 2.628e-6 mm, and
  all paired exported transverse coordinates are identical.
- `s1=0` with nonzero `s2` disables stretch in both families. The exported
  zero profiles match the unstretched equations. Both zero-coefficient paths
  retain identical evaluator arithmetic; this does not excuse the preexisting
  importer errors described below.
- Extensions stay straight and unstretched. A 12 mm extension adds exactly
  12 mm to OSSE and R-OSSE depth and preserves the main radii. R-OSSE's
  8 mm slot adds 8 mm outside the map, alone or with the extension; its
  main curve, derived L and mouth stay unchanged.
- **OSSE block Slot.Length is unsupported on text import**, including at
  zero stretch. ATH retains axial L=160 and stretched depth 204.105045 mm
  with Slot.Length=8, but changes the mouth radius 205.931011 -> 195.623160 mm
  and the intervening radial transition. This differs from the native explicit
  tube prefix even with stretch off (maximum deviation 7.699858 mm; the
  Scale=0.48 cases differ by up to 3.788557 mm). The 36 probes do not establish
  that transition's construction. A dense block/flat slot/no-slot comparison
  must settle it before the importer can support it. Active stretch with a
  slot in flat OSSE text is also unmeasured and refused. Native JSON explicit
  slot geometry and its zero-feature behavior remain unchanged.
- OSSE block L takes precedence over top-level Length: adding Length=180
  beside L=160 produces byte-identical profile CSVs to the plain case,
  both at zero and active stretch. Likewise Tritonia L=135 with Length=180.
  The flat ATH total-length budget described above remains the native
  unstretched import rule; these block probes do not qualify flat stretch
  with Slot.Length. Top-level Length with R-OSSE stretch is unmeasured and refused on
  text import; it must not be interpreted as a target stretched depth.
- OSSE Rot=10 is honoured at top level beside a block. At the terminating
  Desmos station, rotated zero x=123.546177 mm becomes 167.387410 mm while
  radius stays 230.738088 mm. This is S(x_rot); rotating S(x) would change
  that radius. The scalar and vector evaluators now stretch after rotation.
  In-block Rot retains its legacy import behavior with stretch off (including
  precedence over top-level Rot). With active stretch it is unmeasured and
  refused. Active stretch with R-OSSE Rot is refused on text import; OSSE Rot
  plus an extension/slot is refused in both import and native evaluation.
  The only existing import changes against `5c8ea4dc` with stretch absent are:
  top-level Rot beside an OSSE block is now imported when the block has no
  Rot, and an OSSE block with nonzero Slot.Length is now refused.
- Guiding curves resolve against the unstretched profile. With Width=250,
  Dist=0.5 and SE.n=2 the Desmos coverage is 57.392970 degrees, identical
  in the paired profiles, and its mouth radius stays 320.434282 mm.
  The guide station is parameter z=80 mm, which stretches to 123.211833 mm;
  it is not a requirement to pass through radius 125 mm at physical z=80.
  Active stretch plus a guiding curve and a prefix or Rot is unmeasured
  and refused on text import.
- Scale is applied **after** stretch. Tritonia-S uses L=135, s1=0.8,
  s2=0.06 and Scale=0.48: depth 64.800000 -> 96.657431 mm equals
  `0.48*S(135)`. At Scale=0.48, extension 12 contributes 5.76 mm.
  A consumer which pre-scales dimensions, such as WG's preview translator,
  must also send `s1_scaled=Scale*s1` and `s2_scaled=s2/Scale`, preserving
  expressions. This satisfies `S_scaled(Scale*x)=Scale*S(x)`, including
  rotation and guiding curves. Applying unchanged s2 to pre-scaled x changes
  the measured throat shape.

Text import treats expressions as potentially active when deciding whether
an unmeasured combination must be refused. No extrapolated combination is
claimed as ATH parity. These measurements qualify the inner profile, not
ATH wall meshing, guiding curves combined with other transforms, or C5 adapter
composition. Existing archive parity still runs as a separate required gate.

Point reference tolerance is 1e-10 mm (double precision); scalar/array agreement
uses 64 machine eps times the profile scale. An ATH GridExport comparison must
use its printed precision (normally 2e-4 mm), with no fitted degree/radian scale.
This smooth map is separate from the C5 throat adapter and changes no adapter
control points or extension construction.

## ICW Profile

ICW (Intrinsic-Curvature Waveguide) is a native mesher profile rather than an
ATH text-format feature. It defines the meridian as an intrinsic curvature
curve and samples by normalized arc length.

Implementation rules:

- ICW can be configured through TOML/JSON/dict config with `formula = "ICW"`.
- `r0` and `a0` set the throat radius and opening angle.
- `termination = "flat_baffle"` uses axial/mouth targets (`L`, `R`).
- `termination = "rollback"` uses aperture/setback/depth targets.
- `icw_seed` fits an ICW curve to an OSSE/R-OSSE seed profile; direct mode uses
  `icw_coeffs` plus `icw_S`.
- ICW does not use ATH z-map sampling. It rejects `samplingMode = "zmap"` and
  `zMapPoints` because the natural grid coordinate is normalized arc length.
- OSSE/R-OSSE shape keys (`m`, `r`, `b`, `tmax`, OSSE `n/s/rot`) are rejected at
  top level for ICW unless nested inside an `icw_seed`.

## FREEFORM Profile and Loft

FREEFORM defines two independent meridians: H at azimuth 0 and V at azimuth
90 degrees. Each is a vector-valued parametric cubic Hermite curve
`P(u) = (z(u), r(u))` through all supplied anchors. Anchors are parameterized
by normalized cumulative chord length. PCHIP supplies the automatic interior
z and radius derivatives; explicit anchor tangents and the endpoint
angle/scale controls replace those derivatives where configured. Anchor
positions therefore remain exactly on the analytic curve.

The implementation validates each polynomial segment analytically and on a
dense radius sample. Axial motion must remain forward: `z'(u)` cannot be
negative and may be zero only at a curve endpoint. Radius must stay positive.
Radius may not leave the range of the adjacent anchor radii (the former
`overshootPolicy` key is refused). H and V
share the same start/end z span and throat radius, so their local radii

```text
a(t) = r_H(t)
b(t) = r_V(t)
```

are the semi-axes of every cross-section and the mouth is planar.

The cross-section station schedule lofts `circle`, `ellipse`, `superellipse`,
and `rounded_rectangle` outlines using those local semi-axes. Within the span
from station `k` to `k+1`, local progress `u` uses the C2 smootherstep

```text
w(u) = 6u^5 - 15u^4 + 10u^3
rho(phi, t) = (1 - w) rho_k(phi; a(t), b(t))
            + w rho_k+1(phi; a(t), b(t))
```

Every outline hits `a(t)` at phi=0 and `b(t)` at phi=90 degrees, so the loft
honors both meridians throughout the blend. Consecutive stations with the same
complete descriptor produce a hold: the outline descriptor stays constant
while the H/V semi-axes continue to follow their curves.

Rounded-rectangle corner radii are absolute millimetre values. At a station,
the value must lie between 2% and 100% of the local minimum semi-axis. Across
the adjacent active spans, validation uses the station's actual smootherstep
weight and requires `weight * cornerRadiusMm <= min(a(t), b(t))`. Equal
descriptors form a hold, for which either endpoint has full weight across the
whole span. This weight-aware active window prevents an absolute corner from
binding an unrelated region where its contribution is negligible while still
rejecting an impossible local corner.

### FREEFORM sampling

The base axial map is uniform or a user-supplied custom z-map. The sampler
merges every H/V anchor and cross-section station into that map, collapsing
only positions within `1e-7` normalized-`t` floating-point noise. A base
sample is snapped to the exact semantic feature; two distinct features within
that tolerance are rejected rather than merged. Rounded-rectangle tangency
azimuths depend on the local semi-axes and corner radius, so they can move
along z. FREEFORM therefore constructs a separate azimuth grid for every
axial ring. The point-grid conversion consumes `phi_grid[i, j]`, preserving
the moving corners instead of projecting every ring onto one mouth-derived
angle list. Cardinal axes remain pinned for symmetry and H/V exactness.

Acoustic fitting may refine the axial, ordinary angular, and rounded-corner
arc sampling to meet chord and sagitta limits. `angular_segments`,
`corner_segments`, and `length_segments` are geometry sampling inputs; the
mesh resolution fields independently control requested BEM element size.

### FREEFORM diagnostics and guards

Cross-section blends are sampled for polygon convexity at ingest. A failure
identifies the station span and offending normalized position and, for a
rounded-rectangle blend, reports an estimated feasible corner-radius hint.

For a freestanding shell, the inner loft is checked over all azimuths with a
finite-difference first/second fundamental-form calculation. The analytic
normal points toward the acoustic axis and the wall is offset in the opposite
direction, so the signed local risk is
`max(0, -wallThickness * kappa_i)`. The build is rejected when that risk
reaches the `0.4` safety margin; convex curvature that expands under the
outward offset is not rejected. After the outer wall is generated, its grid is
checked for normal flips, meridian self-intersections, and ring
self-intersections. The corner-radius active-span guard described above runs
before both surface checks. Enclosure and infinite-baffle modes do not create
this freestanding outer offset.

An inflection span is a contiguous portion of an H or V Hermite meridian with
negative signed curvature and more than a 1 degree drop in tangent angle.
Spans are sampled on the spline's 4001-point inversion grid and reported by
default. `inflectionPolicy = "reject"` rejects the first span; `"warn"` is the
default and retains it as a diagnostic. There is no `"allow"` policy.

`FreeformGeometry.report()` returns:

- `maxNormalDeviationMm`: per-plane maximum normal distance from the Hermite
  curve to each anchor chord.
- `curveSamples`: 192 exact Hermite `[z, r]` samples per plane for an
  authoritative display curve.
- `throatRadiusMm`: the shared H/V throat radius.
- `tangentAnglesDeg`: resolved throat and mouth angles for H and V.
- `anchorTangents`: every anchor as `z`, `r`, and its explicit `angleDeg` and
  `strength` (the latter two are null when automatic).
- `inflectionSpans`: per-plane `zStartMm`, `zEndMm`, and `tangentDropDeg`.

The report is carried into build metadata as `freeformReport`. Acoustic OCC
builds additionally report `freeformProfileDeviationMm`, the fitted wall's
maximum deviation from the analytic H/V axis samples.

## Guiding Curve

ATH distinguishes explicit profile definitions from implicit coverage
definition by guiding curve. In guiding-curve mode, the coverage angle for each
profile is solved so the profile passes through a virtual closed curve at
`GCurve.Dist`.

Canonical rule:

- Compute the target guiding-curve radius `r_g(phi)`.
- Interpret `GCurve.Dist` in `(0, 1]` as a fraction of the main horn length;
  values greater than `1` are absolute millimetres. Absent, the curve sits at
  the mouth.
- `GCurve.Distance` is **not** an ATH key and does not move the curve. ATH
  ignores it; so does this importer, with a warning naming `GCurve.Dist`.
  Reading it as an alias placed a real archive config's curve at mid-length
  and missed ATH's own mesh by 89 mm.
- Invert OS-SE coverage angle `a` so `r_osse(target_z, phi) == r_g(phi)`.
  A non-zero `h` bulge is included: the inversion targets
  `r_g - h * sin(pi * t_g)`, so the bulged wall meets the curve.
- `Rot` is applied after the inversion, so a rotated profile misses the curve.
  That is ATH's geometry too (ath.exe probe, `Rot = 10`), and is kept.
- A non-circular `cross_section` (exponent != 2 or aspect ratio != 1) is
  refused with an active guiding curve: its scale is applied after the
  inversion, so the wall would miss the curve.

Supported guiding curve targets:

- `GCurve.Type = 1`: superellipse. `GCurve.SE.n` below 2 is clamped to 2,
  as ath.exe does.
- `GCurve.Type = 2`: superformula. `GCurve.AspectRatio` scales the
  superformula point `(r cos p, r sin p)` in x/y and assigns its length to the
  original azimuth `p`. That is not the polar radius of the scaled curve, but it
  is what ath.exe V2025-06 builds (mouth radii pinned in
  `tests/test_geometry_review_fixes.py`), so it is kept. A `GCurve.SF` list must
  have six values; a single value (WG writes `GCurve.SF = 0`) means "not given" and the
  `GCurve.SF.*` fields apply.

Guiding curves are an OS-SE/OSSE feature in this implementation. R-OSSE with
an active guiding curve must fail explicitly.

Unsupported guiding curve types must fail explicitly.

## Morphing

Morphing is a universal target-mouth rule, not an ATH case fix. The OS-SE paper
defines a target mouth radius `rM(phi)` and transforms the raw radius toward
that target after a fixed axial portion:

```text
for z < zf:
  rm(z, phi) = r(z, phi)

for z >= zf:
  rm(z, phi) =
    r(z, phi) + ((z - zf) / (L - zf))^gamma * (rM(phi) - r(L, phi))
```

Implementation rules:

- The blend runs over the horn after the throat extension: with `e` the
  extension's share of the normalised axial parameter, progress is
  `u = (t - e) / (1 - e)`. Extension stations have `u <= 0` and are never
  morphed, whatever the axial map (verified against ath.exe V2025-06, which also
  measures `Morph.FixedPart` from the end of the extension).
- `Morph.FixedPart` maps to `zf / L` in `u` and is snapped onto the first axial
  station at or past it. A slot is reserved by position: the blend starts no
  earlier than the first station at or past the slot's end. Earlier versions
  reserved `ceil(n * (ext + slot) / L)` rings, which is a position only on a
  uniform map; on the acoustic fit's throat-clustered map it morphed the last
  rings of a straight extension. ATH text imports keep ATH's rule instead,
  which morphs the slot, and default an absent `Morph.FixedPart` to ATH's
  effective 0.2 (the m2-clone reference relies on both).
- The acoustic fit samples on its own axial map but reuses the requested
  grid's snapped start and pins a ring there, so the solve surface is the
  surface the requested (preview) grid shows.
- ath.exe snaps an explicit `Morph.FixedPart = 0` to its first slice after the
  throat, not the throat itself; this mesher starts at the throat, as the
  formula above says. Measured differences: about 0.4-0.6 mm mid-horn at
  rate 2 without a slot, up to about 1.6 mm at the end of a 12 mm slot. Not
  reproduced.
- The blend progress `(z - zf) / (L - zf)` uses the global normalized axial
  position and is identical for every azimuth — the per-azimuth slot length
  does not shift it (verified against the ATH m2-clone grid).
- `Morph.Rate` maps to `gamma`. ATH documents a minimum of `1`; rates in
  `[0, 1)` are accepted (Waveguide Generator uses them) and negative rates are
  refused.
- `Morph.TargetShape = 0` leaves the raw mouth outline unchanged.
- `Morph.TargetShape = 1` targets a rounded rectangle.
- `Morph.TargetShape = 2` targets a circle.
- `Morph.TargetWidth` and `Morph.TargetHeight` are full target widths; the
  target half-dimensions are half of them (ATH parity: 325 mm -> 162.5 mm).
  `Morph.Width` and `Morph.Height` are **not** ATH keys and set nothing. ATH
  ignores them; so does this importer, with a warning naming the canonical
  keys. Reading them as aliases morphed archive configs onto a mouth ATH never
  builds — 54 mm out on one, 124 mm on another. Rename them to
  `Morph.TargetWidth` / `Morph.TargetHeight` to restore the written intent.
- `TargetWidth = 0` or `TargetHeight = 0` derives that half-dimension
  implicitly by rounding the raw mouth extent up to whole millimetres
  (ATH m2-clone: raw 228.414/203.515 -> targets 229/204), which is what ATH
  does with a config that gives no target dimensions at all.
- If shrinkage is disabled, the target half-dimensions are floored at the raw
  mouth extents; the mouth still becomes the exact (enlarged) target curve
  rather than a per-azimuth max of target and raw.
- For rounded-rectangle targets the azimuth list places four profiles per
  quadrant on the corner arc (both wall tangency points plus two interior
  points at 30/60 degrees of arc parameter) regardless of
  `Mesh.CornerSegments`, which grows the total angular point budget:
  `AngularSegments + CornerSegments` is rounded up to a whole number per
  quadrant (m2-clone: 100 + 4 -> 104; solana: 36 + 1 -> 40). Wall spans split
  the remaining segments proportionally to their angular extents.

## Freestanding Mouth Closure

A freestanding wall is defined by the acoustic inner surface and its
`wall_thickness_mm` offset outer surface. At the mouth, the material ends on a
ruled face: each inner mouth-ring point connects by a straight span to the
corresponding terminal point of the outer wall. Lofting all spans forms the mouth surface.

For an axisymmetric grid, both terminal rings remain circular between sampled
azimuths. This angular invariant also applies when `surface_fit = "approximate"`:
that option may retain the approximating axial profile, but it must not give a
nominally circular ring an azimuth-dependent radius.

Wall thickness does not also imply a rounded lip. No freestanding lip-radius
parameter exists in the public config, so a solver-specific semicircle would
describe a different body. A future rounded closure therefore requires an
explicit geometry option implemented by every preview, CAD, and mesh consumer.

## Geometry Grid vs Mesh Density

ATH separates the geometry grid from the final BEM mesh density:

- `Mesh.AngularSegments` is the number of calculated profiles around the
  waveguide and must be adjusted to a multiple of four.
- `Mesh.LengthSegments` is the number of axial slices.
- `Mesh.ZMapPoints` controls axial spacing of grid slices.
- `Mesh.ThroatResolution`, `Mesh.MouthResolution`, and related resolution
  values control the final BEM mesh size, not the geometry grid shape.

Canonical rule:

- Sampling mode must be explicit in code.
- Uniform sampling is a valid canonical policy.
- ATH-compatible z mapping is a compatibility policy unless it is documented as
  the default input semantics for imported ATH configs.
- Custom z maps are normalized to `[0, 1]`, monotonic, finite, and include both
  endpoints after normalization/filling.

## Surface Topology

The mesher may split smooth geometry into several Gmsh surfaces. These splits
are not formula behavior; they are meshing topology.

Universal splitting rules:

- Split at open-domain symmetry boundaries.
- For full domains, split at cardinal/quadrant boundaries when that improves
  stable spline construction or preserves expected physical patch grouping.
- Keep spline spans below a stable control-point count.
- Keep wall, source, rear, mouth, interface, and enclosure surfaces in
  separately named mesh groups when they have different physical tags or mesh
  density rules.

ATH parity tests may assert exact surface counts, but production helper names
should describe the topology rule rather than ATH.

## Interfaces

ATH subdomain interfaces are virtual boundaries between acoustic subdomains.
They are configured by:

- `Mesh.SubdomainSlices`: requested-grid slice indices where interfaces are
  placed. When acoustic sampling refines or trims the axial grid, each slice is
  relocated to preserve its normalized axial position.
- `Mesh.InterfaceOffset`: forward protrusion per interface.

Canonical rule:

- Interfaces are optional.
- Multiple interfaces are representable.
- If an imported ATH config omits `Mesh.SubdomainSlices`, the compatibility
  default is the last slice before the mouth.
- If an imported ATH config sets `Mesh.SubdomainSlices` but omits
  `Mesh.InterfaceOffset`, the compatibility default is ATH's 5 mm protrusion.
- `Mesh.InterfaceDraw` is not implemented by the mesher today; the generated
  interface is the offset surface, not a drawn-depth ATH interface body.
- Interface surfaces get their own physical group and mesh-density rule.

## Enclosures

Enclosures are rear/side/front baffle geometry around the waveguide mouth.

Canonical rule:

- Closed-domain rounded rectangle, ellipse, and superellipse plans are valid
  when supported by the builder.
- Open-domain enclosures are sector versions of the same plan geometry, not a
  separate ATH-only concept.
- Edge treatment is either rounded fillet or chamfer.
- Enclosure mesh resolution comes from mesh-density/config values.

Unsupported plan/edge/domain combinations must raise `NotImplementedError`
rather than silently generating an approximate shape.

## Compatibility Boundary

The codebase should use explicit names for compatibility behavior:

- `sampling_mode = "ath-default-zmap"` or similar for ATH axial sampling.
- `topology_mode = "legacy"` only when ATH/parity surface grouping or faceted
  point-grid topology is intentionally requested. Ordinary solve meshes use
  `topology_mode = "acoustic"` so geometry sampling does not dictate BEM
  topology.
- `surface_fit = "auto"` is the default, and resolves to `interpolate` wherever
  it is meshable. `approximate` was the default until it was measured: it is
  what every existing mesh had been built with, not the more faithful fit. OCC
  reads the sampled grid as control points, so the meshed wall sits inside the
  designed one by roughly `R * dtheta^2 / 6` for a cubic pole fit, and refining
  the mesh converges onto that biased surface rather than onto the sampled one.
  It remains available by name for callers that need the legacy pole fit.
  Freestanding axisymmetric shells are the narrow exception: their angular
  direction is interpolated even in `approximate` mode so a circular ring does
  not acquire an azimuth-dependent radius; the axial direction retains the
  requested pole fit.

`surface_fit = "interpolate"` removes that bias by solving for the poles whose
surface passes through the sampled grid, at the same control-point and triangle
count. Two rules constrain it:

- Patches cut from one wall must share a v-parameterisation. A clamped
  interpolating spline reproduces its end data, so neighbouring patches already
  share the poles of their common seam curve, but they trace the same curve only
  if they also share its knot vector. Deriving v per patch tears the shell open
  along every seam.
- It is refused on FREEFORM profiles, whose deliberate creases make the
  interpolating patch fit unmeshable.

See `docs/config-schema.md` for the measured fit error and the config key.

Names like `ath_*` are acceptable in tests and compatibility adapters. Generic
geometry helpers should instead name the rule they implement.
