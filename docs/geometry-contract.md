# Geometry Contract

This document defines the geometry rules that `hornlab-waveguide-mesher`
implements for OS-SE/OSSE, R-OSSE, and ICW waveguides. It also separates
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
  `Mesh.CornerSegments`, which only grows the total angular point budget:
  the grid carries `AngularSegments + CornerSegments` profiles rounded up to
  a whole number per quadrant (m2-clone: 100 + 4 -> 104; solana: 36 + 1 ->
  40). Wall spans split the remaining segments proportionally to their
  angular extents.

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

- `Mesh.SubdomainSlices`: grid slice indices where interfaces are placed.
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
- `topology_mode = "ath-compatible"` only when exact ATH surface grouping is
  intentionally requested.

Names like `ath_*` are acceptable in tests and compatibility adapters. Generic
geometry helpers should instead name the rule they implement.
