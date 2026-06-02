/**
 * Monotonic cubic Hermite (PCHIP) interpolation helpers.
 *
 * Used by the cross-section timeline (Phase 4 of the profile-system
 * redesign) so that interpolating `exponent` / `aspectRatio` between
 * keyframes does not overshoot — a plain Catmull-Rom or natural cubic
 * spline can produce exponents < 2 (non-physical for superellipse) when
 * keyframes differ sharply.  PCHIP (Fritsch-Carlson) preserves
 * monotonicity between any two adjacent samples.
 *
 * Evaluate-at-t style: the helper computes the slope it needs per query
 * from the keyframe neighborhood.  No precomputed spline cache is kept,
 * which keeps the timeline free of hidden per-vertex state.
 */

/**
 * Evaluate a PCHIP interpolation of `ys` sampled at `xs` at query x.
 *
 * Contract:
 *   - xs must be strictly increasing, length >= 2.
 *   - ys.length === xs.length.
 *   - For x <= xs[0]: returns ys[0].  For x >= xs[last]: returns ys[last].
 *   - For length 2: falls back to linear (PCHIP degenerates anyway).
 *
 * Fritsch-Carlson monotonic cubic Hermite:
 *   1. Compute per-segment secant slopes d[i] = (ys[i+1] - ys[i]) / (xs[i+1] - xs[i]).
 *   2. Compute per-node tangent slopes m[i] such that the interpolant stays
 *      monotonic.  Endpoints get a one-sided approximation; interior nodes
 *      use the harmonic mean when adjacent secants have the same sign, 0
 *      when signs differ (inflection point) or either secant is zero.
 *   3. Cubic Hermite basis on the segment bracketing x.
 */
export function pchipEval(xs, ys, x) {
  const n = xs.length;
  if (n === 0) return 0;
  if (n === 1) return ys[0];
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];

  // 2-keyframe fast path: PCHIP degenerates to linear.
  if (n === 2) {
    const dx = xs[1] - xs[0];
    if (dx <= 0) return ys[0];
    const t = (x - xs[0]) / dx;
    return ys[0] + (ys[1] - ys[0]) * t;
  }

  // Find bracket: xs[i] <= x < xs[i+1]
  let i = 0;
  for (let k = 1; k < n; k++) {
    if (x < xs[k]) { i = k - 1; break; }
    i = k - 1;
  }

  // Compute tangent slopes at the two bracket nodes (and neighbors as needed)
  // using the Fritsch-Carlson rule.  We only need m[i] and m[i+1].
  const m0 = pchipNodeSlope(xs, ys, i);
  const m1 = pchipNodeSlope(xs, ys, i + 1);

  const h = xs[i + 1] - xs[i];
  const t = (x - xs[i]) / h;
  // Hermite basis
  const t2 = t * t;
  const t3 = t2 * t;
  const h00 = 2 * t3 - 3 * t2 + 1;
  const h10 = t3 - 2 * t2 + t;
  const h01 = -2 * t3 + 3 * t2;
  const h11 = t3 - t2;
  return h00 * ys[i] + h10 * h * m0 + h01 * ys[i + 1] + h11 * h * m1;
}

function pchipNodeSlope(xs, ys, i) {
  const n = xs.length;
  if (n < 2) return 0;
  if (i <= 0) {
    // One-sided endpoint slope (three-point formula for interior consistency,
    // clamped to preserve monotonicity).
    const h0 = xs[1] - xs[0];
    const d0 = (ys[1] - ys[0]) / h0;
    if (n === 2) return d0;
    const h1 = xs[2] - xs[1];
    const d1 = (ys[2] - ys[1]) / h1;
    // Three-point estimate
    const m = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    // Monotonicity clamp
    if (m * d0 <= 0) return 0;
    if (d0 * d1 <= 0 && Math.abs(m) > 3 * Math.abs(d0)) return 3 * d0;
    return m;
  }
  if (i >= n - 1) {
    const h0 = xs[n - 1] - xs[n - 2];
    const d0 = (ys[n - 1] - ys[n - 2]) / h0;
    if (n === 2) return d0;
    const h1 = xs[n - 2] - xs[n - 3];
    const d1 = (ys[n - 2] - ys[n - 3]) / h1;
    const m = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if (m * d0 <= 0) return 0;
    if (d0 * d1 <= 0 && Math.abs(m) > 3 * Math.abs(d0)) return 3 * d0;
    return m;
  }
  // Interior node
  const hPrev = xs[i] - xs[i - 1];
  const hNext = xs[i + 1] - xs[i];
  const dPrev = (ys[i] - ys[i - 1]) / hPrev;
  const dNext = (ys[i + 1] - ys[i]) / hNext;
  // If signs differ or either is zero, local extremum → slope = 0.
  if (dPrev * dNext <= 0) return 0;
  // Weighted harmonic mean (Fritsch-Carlson)
  const w1 = 2 * hNext + hPrev;
  const w2 = hNext + 2 * hPrev;
  return (w1 + w2) / (w1 / dPrev + w2 / dNext);
}
