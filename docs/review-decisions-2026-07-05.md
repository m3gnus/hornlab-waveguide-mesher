# Review Decisions 2026-07-05

This log captures the D1-D13 decisions from the Boundary Lab / ATH parity review.
It separates current policy from deferred work so future changes do not rediscover
the same tradeoffs.

## Current Policy

- D1 y-cut plus `Mesh.VerticalOffset`: keep ATH parity for now. ATH emits a
  translated y-cut mesh while declaring symmetry about `y=0`; the mesher
  reproduces that behavior deliberately because Boundary Lab/WG parity tests
  depend on it. Alternatives remain: reject that combination, or keep the mesh
  unshifted and carry placement metadata separately.
- D2 free-standing ABEC two-subdomain parity: keep the mesher single-domain by
  design. ATH can emit `I1-2` plus `SD1`/`SD2` free-standing split surfaces; the
  mesher currently rejects unsupported free-standing subdomain-interface requests
  so Boundary Lab can route them to Ath instead of silently building a different
  model.
- D4 naming: keep `ICW` for now, expanded as "Intrinsic-Curvature Waveguide"
  in user-facing contexts when needed. Avoid `ICP`; possible alternatives such
  as `kappa-spline`, `KSW`, or `Cesaro` are less clear.
- D7 downstream pinning: Waveguide Generator now pins the local reviewed mesher
  commit. Public installs require the mesher commit to be pushed before or with
  the WG pin bump, because the pin resolves through GitHub.
- D12 snap tolerance: keep the existing 1.0 mm snap tolerance. The snap/weld
  ordering fix addressed the observed duplicate-vertex interlock without
  changing the tolerance policy.

## Deferred Decisions

- D3 throat-extension source recess: ATH recesses the driven cap behind the
  straight throat-extension duct. The current mesher taper-back fix matches the
  profile radius/length, but does not add that recessed driven-surface detail.
  Decide whether this is required for acoustic fidelity or should be documented
  as a deliberate simplification.
- D5 `Scale` x wall thickness: ATH does not scale wall thickness; the mesher
  currently scales it with other linear geometry. Matching ATH would change
  existing scaled outputs, including optimizer-era runs, so this remains a user
  decision.
- D6 Boundary Lab runner protocol: consider replacing stdout/stderr sniffing with
  a structured one-line build-result JSON contract. Also decide whether duplicate
  `Mesh.Quadrants`, block-form `Mesh = {}`, and case-sensitivity divergences
  should be normalized further.
- D8 expression evaluation security: replace the current expression evaluator
  with an AST allowlist before treating arbitrary third-party ATH config files
  as trusted input. This is strongly recommended but was outside the parity-fix
  batch.
- D9 parity sweep backlog: run future ATH/Wine sweeps for morph extremes,
  custom z-map points, guiding-curve type 2 variants, asymmetric enclosure
  spacing, rotations, and interface placement.
- D10 enclosure roundover topology: review sector versus through-section
  roundover geometry paths and any `n_phi`-dependent topology differences.
- D11 R-OSSE free-standing rear shell: an audit noted a rear-shell z delta
  against ATH that was not resolved in this batch.
- D13 vectorization backlog: M10/M20/M28 were handled, but full weld bucketing,
  per-phi OSSE vectorization, broader `np.interp` z-map use, and morph-target
  hoists remain optional performance work.
