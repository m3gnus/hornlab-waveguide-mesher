# Mesh quality: what is measured, what the thresholds rest on, and what they do not claim

`hornlab_mesher.quality` measures two independent things and gates on both. This
document is the evidence for the thresholds, and — more importantly — the record
of a hypothesis that **did not survive being tested**. Read the null result
before quoting the gate for anything.

## Why two measures and not one

**Element shape.** Minimum interior angle and the radius ratio `2*r_in/r_circ`.
The radius ratio rather than an aspect ratio because an aspect ratio cannot tell
a needle from a cap.

**Chord deviation.** How far the faceting stands off the surface it stands in
for, estimated from the dihedral turn across each interior edge and reported as a
length in millimetres.

Neither measure can see the other's defect. A facet that swallows a 3.6 mm
roundover radius in one 26 mm element can be exactly equilateral — every angle,
aspect-ratio and SICN statistic passes it while the surface sits millimetres out
of position. A sliver, conversely, can lie exactly on the true surface. This is
not a hypothetical pair: both defects are present in the meshes measured below.

## The null result: element shape does not predict solver trouble

This module was commissioned on the hypothesis that sliver triangles are what
makes some meshes hard for an iterative solver. **Measured, that hypothesis does
not hold, and the gate must not be described as fixing it.**

Every full (non-quarter) export in the ATH reference archive was welded at 1 nm,
orientation-repaired, scaled to metres, and solved with the exterior Metal BEM
engine's GMRES path at 500 / 1000 / 2000 / 4000 / 6000 Hz, unit normal velocity
on the driver-interface tag. `X` is a non-convergence returned as `info=-999`;
the numbers are iteration counts. Quality is from this module.

| case | tris | min angle | 1st pct | tris <10° | min gamma | max chord dev | GMRES 0.5/1/2/4/6 kHz |
|---|---:|---:|---:|---:|---:|---:|---|
| `250728solana` | 16,394 | 16.27 | 24.07 | 0 | 0.1843 | 3.74 mm | 15 **X** 59 **X** **X** |
| `250917asro68` | 6,640 | 17.59 | 21.54 | 0 | 0.2399 | 13.45 mm | 19 21 27 30 29 |
| `260308Tritonia-M` | 5,172 | 13.38 | 22.08 | 0 | 0.3622 | 4.73 mm | 12 16 42 120 93 |
| `260308muh6` | 7,792 | 6.74 | 12.00 | 34 | 0.0304 | 5.05 mm | 11 15 35 98 108 |
| `260308tritonia` | 6,304 | 15.36 | 22.30 | 0 | 0.4527 | 2.79 mm | 15 17 19 20 21 |
| `260330saw` | 5,114 | 16.13 | 22.99 | 0 | 0.3517 | 3.04 mm | 16 17 20 22 21 |
| `260330solana` | 8,384 | **2.05** | 19.70 | 2 | **0.0028** | 9.39 mm | 19 21 27 31 32 |
| `asro2` | 2,275 | 17.59 | 21.95 | 0 | 0.2255 | 4.22 mm | 10 12 14 15 16 |
| `test1_gcurve_only` | 4,816 | 14.89 | 21.41 | 0 | 0.3955 | 3.19 mm | 15 17 22 22 22 |
| `test3_morph_only_shrink` | 10,210 | 15.12 | 26.86 | 0 | 0.4502 | 2.94 mm | 17 19 21 22 21 |
| `test6_small_gcurve_with_covangle` | 3,944 | 6.89 | **7.35** | **128** | 0.2023 | 4.23 mm | 15 **X** **X** 30 37 |
| `tritonia-v` | 6,442 | 13.38 | 22.20 | 0 | 0.2909 | 4.90 mm | 12 18 49 **X** 103 |

`test2`, `test4` and `test5` are omitted: they are geometric duplicates of
`test1`, `test3` and `test1` and behave identically.

Three things follow, and the first is the one that matters:

1. **The extremes rank the population backwards.** `260330solana` holds the
   archive's worst single triangle by both measures — 2.05° and a radius ratio of
   0.0028, an order of magnitude worse than anything else here — and converges in
   19 to 32 iterations at every frequency tried. `250728solana` has no triangle
   under 16.27° and stagnates at three frequencies of five. A gate on the worst
   triangle would reject the healthy mesh and pass the failing one.
2. **The percentile does not separate either.** It ranks `test6` worst (7.35°),
   which is a mesh that stagnates — but `250728solana` sits at 24.07°, second
   best in the archive, and stagnates more often. One agreement out of two is not
   a predictor.
3. **The failures are not knife-edge.** Both stagnations survive a ±20 Hz sweep
   about the failing frequency, so on these meshes this is a band, not the
   isolated single-hertz fragility seen elsewhere. `test6` at 2 kHz recovers by
   +20 Hz; nothing else does.

### The controlled experiment

Correlation over twelve meshes is weak evidence either way, so the hypothesis was
tested directly: **remove the slivers, hold the geometry, re-solve.**

Tangential Laplacian smoothing — each vertex's displacement projected out of its
own normal, so the surface is preserved to second order and no vertex can cross a
thin wall; the physical-tag boundary pinned — was applied to three meshes.

| mesh | 1st pct angle before → after | tris <10° | wall/element ratio, 1st pct | GMRES before → after |
|---|---|---:|---|---|
| `test6` (stagnates) | 7.35° → **23.06°** | 128 → **0** | 0.297 → 0.302 | 1 kHz X → **X**; 2 kHz X → **ok(19)** |
| `250728solana` (stagnates) | 24.07° → 36.40° | 0 → 0 | open shell, no wall | 15/X/59/X/X → **15/X/59/X/X**, unchanged |
| `260330solana` (healthy) | 19.70° → 34.78° | 2 → 0 | 0.364 → 0.336 | 19/21/27/31/32 → 19/21/29/32/31 |

`test6`'s sliver rim was removed outright — the 1st percentile went from the
archive's worst to better than its median, and no triangle was left under 10° —
while the thin rim that defines the geometry survived (the 1st-percentile ratio of
opposed-wall separation to element size moved 0.297 to 0.302). Of its two
stagnating frequencies, **one was cured and one was not.**

The two controls are what make that readable. On `250728solana` the same
treatment improved every shape statistic by as much or more and changed the
solver outcome in no way at all, iteration for iteration. On the healthy
`260330solana` it moved iteration counts by at most two. So the smoothing is not
a general solvent, and the one cure on `test6` is attributable to its slivers
rather than to perturbation.

**Honest summary: of six stagnating (mesh, frequency) pairs in this archive,
removing slivers fixed one.** Element shape is a contributor to solver difficulty
on one mesh and is not the mechanism behind the rest.

### What is therefore *not* claimed

- This gate is **not** a predictor of GMRES convergence, and no code or comment
  may describe it as one.
- It is **not** a fix for `info=-999`. The premise that `info=-999` is a sliver
  problem has been corrected twice; the table above is the third refutation.
- Whether the *other* engine's residual floor above 1e-6 on these exports is
  sliver-driven was **not tested here**: the wrapper on this machine exposes no
  solver selection, its GMRES lives on unmerged branches, and building it exceeds
  the solve budget. That claim remains open, neither supported nor refuted.

### What *is* claimed

The mesh is the artifact the user asked for. A mesh a fifth of which is slivers,
or one whose faceting sits 10 mm off the surface, is not that artifact whatever
the solver then does with it. Both measures are gated on that basis alone.

## Where the thresholds come from

### Element shape: 1st-percentile minimum angle, gate at 10°

The population has a gap there, in both libraries measured, and the gate sits on
the plateau rather than on a slope.

- **ATH reference archive**, 15 full exports: 1st percentile is 7.35, then 12.00,
  then 19.70, 21.41, 21.54, 21.95, 22.08, 22.20, 22.30, 22.99, 24.07, 26.86.
- **The application's own mesh library**, 30 archived and imported `.msh` files:
  7.67 (seven meshes), then 20.71, 21.05, 26.54, 29.02, 30.15 … 46.33.

Nothing in either population lies between 8° and 12°, so it does not matter where
in that gap the number lands. `FAIL_P1_ANGLE_DEG = 10.0`.

`WARN_P1_ANGLE_DEG = 15.0` is the advisory band above it. There is only one gap
in the data and the gate is on it, so **this second number is a judgement, not a
measurement**, and is recorded as such.

### Chord deviation: gate at 5 mm

Also a measured gap. Over the application's mesh library the maximum chord
deviation runs 0.29, 0.32, 0.33, 1.16, 1.24, 1.27, 1.87, 1.87, 2.13, 2.78,
3.41 (×4), 3.80 (×3), 4.23 — and then jumps to 8.38, 8.85 (×6), 10.78 (×4).
Nothing lies between 4.23 and 8.38.

The gap is corroborated by the one case where ground truth is known. A stock
R-OSSE (R 150, r0 12.7, a 60, a0 15.5, 3 mm wall) built at ATH's default
resolutions is the mesh whose acoustic surface chords its own mouth rollback and
passes out through the shell behind it:

| build | tris | max chord deviation |
|---|---:|---:|
| ATH defaults (mouth 26 mm, rear 15 mm) | 1,638 | **10.74 mm** |
| rear resolution 7 mm | 4,414 | 10.64 mm |
| mouth resolution 8 mm | 4,896 | 4.34 mm |
| mouth resolution 4 mm | 12,854 | **1.28 mm** |

The measure tracks the defect and its remedy, and is correctly insensitive to
rear resolution — which is the independent finding that rules the rear guard out
as the fix. The gap brackets the defect (10.74) and its resolved form (1.28).
`FAIL_CHORD_DEVIATION_MM = 5.0`; `WARN_CHORD_DEVIATION_MM = 3.0` is again a
judgement rather than a gap.

### The known blind spot

A genuine sharp crease has no arc to deviate from, so the measure cannot in
principle tell an under-resolved curve from a real corner. In the measured
library it does not misfire — the 90° enclosure corners report 0.29 to 1.24 mm
because their facets are small — but a large flat-faced body meshed coarsely
would trip it. The gate quotes the location of the worst edge for exactly this
reason: read where it is before acting on the number.

### Why the default is report-only

Seven of the thirty library meshes cross the element-shape gate and twelve cross
the chord gate. A gate enabled by default would reject a large part of the
existing library on the day it landed, which is why `quality_gate="report"` is
the default and `"strict"` is opt-in — the same reasoning that shipped the
self-intersection guard as a warning.

## Retired negative: gmsh's own optimisation is not worth enabling

gmsh is never asked to optimise or report element quality in this pipeline, so
the obvious question is whether it should be. Measured on two geometries, one
process, machine load average 26 (timings are therefore indicative; the quality
figures are deterministic and load-independent):

| geometry | setting | seconds | min angle | 1st pct | 5th pct | min gamma | min SICN |
|---|---|---:|---:|---:|---:|---:|---:|
| R-OSSE | default (`Mesh.Smoothing=1`) | 0.74 | 10.58 | 22.38 | 33.84 | 0.1178 | 0.2785 |
| R-OSSE | `Mesh.Smoothing=5` | 0.92 | 10.58 | 22.63 | 35.99 | 0.2363 | 0.3124 |
| R-OSSE | `Mesh.Smoothing=20` | 1.67 | 10.58 | 22.63 | 35.99 | 0.2363 | 0.3124 |
| R-OSSE | `+ Laplace2D` | 1.33 | 10.58 | 22.63 | 35.09 | 0.2363 | 0.3124 |
| OSSE | default | 2.95 | 10.80 | 24.55 | 37.57 | 0.3383 | 0.3187 |
| OSSE | `Mesh.Smoothing=5` | 4.14 | 10.80 | 24.55 | 40.73 | 0.3383 | 0.3187 |
| OSSE | `Mesh.Smoothing=20` | 8.54 | 10.80 | 24.55 | 40.90 | 0.3383 | 0.3187 |

`Mesh.Smoothing=5` doubles the R-OSSE's worst radius ratio and moves the OSSE's
extremes not at all; `Mesh.Smoothing=20` buys nothing over 5 on either and costs
2.3–2.9x the mesh time; `Laplace2D` adds nothing beyond `Smoothing=5`. Only the
5th percentile moves reliably, by 2–3°. **Not enabled**, and recorded here beside
`General.NumThreads=8` so it is not re-attempted.

SICN is available through `quality.gmsh_sicn` while a gmsh model is live. It is
kept separate from the two measures above because an archived `.msh`, an imported
CAD mesh and a solver-side triage all have vertices and triangles and none of
them has a gmsh handle.

## Provenance

All distributions above were measured on `main` (`ea19a73`). The R-OSSE
calibration was additionally run against the open mouth-rim clearance branch and
produced **bit-identical output** — 1,638 triangles and 10.74 mm at ATH defaults
on both trees — so that branch's meridian bound did not engage for this
configuration. That is worth reconciling against the branch's own reported
1,630 → 2,644 triangle change on what appears to be the same stock R-OSSE, but it
is noted rather than chased here.
