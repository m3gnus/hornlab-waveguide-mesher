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
- A final `Rot` transforms the computed 2D profile around `(0, r0)`.

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
  derived length, and mouth radius are unchanged by the extension.

Compatibility note for OS-SE and R-OSSE throat extensions: the taper-back
implementation reproduces ATH's profile radius and total length exactly, but it
deliberately does not reproduce ATH's recessed driven cap behind the straight
extension duct. That recess is a transmission-line detail expected to matter
acoustically only for horns with a long throat extension; revisit it only if a
real device shows a response discrepancy.

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
By default, radius may not leave the range of the adjacent anchor radii; the
explicit `overshootPolicy = "allow"` relaxes only that range check. H and V
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
only positions within floating-point noise. Rounded-rectangle tangency
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
finite-difference first/second fundamental-form calculation. The build is
rejected when either principal curvature violates
`|wallThickness * kappa_i| < 0.4`. After the outer wall is generated, its grid
is checked for normal flips, meridian self-intersections, and ring
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
  values greater than `1` are absolute millimetres.
- Invert OS-SE coverage angle `a` so `r_osse(target_z, phi) == r_g(phi)`.

Supported guiding curve targets:

- `GCurve.Type = 1`: superellipse.
- `GCurve.Type = 2`: superformula.

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

- `Morph.FixedPart` maps to `zf / L` and is snapped onto the axial grid. When
  the profile has a throat extension or slot, the fixed region additionally
  reserves `ceil(n * (ext + slot_max) / L)` axial slices (the longest fixed
  prefix over all azimuths); the blend then starts at that grid slice.
- The blend progress `(z - zf) / (L - zf)` uses the global normalized axial
  position and is identical for every azimuth — the per-azimuth slot length
  does not shift it (verified against the ATH m2-clone grid).
- `Morph.Rate` maps to `gamma` and must be at least `1` for canonical use.
- `Morph.TargetShape = 0` leaves the raw mouth outline unchanged.
- `Morph.TargetShape = 1` targets a rounded rectangle.
- `Morph.TargetShape = 2` targets a circle.
- `TargetWidth = 0` or `TargetHeight = 0` derives that half-dimension
  implicitly by rounding the raw mouth extent up to whole millimetres
  (ATH m2-clone: raw 228.414/203.515 -> targets 229/204).
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

Names like `ath_*` are acceptable in tests and compatibility adapters. Generic
geometry helpers should instead name the rule they implement.
