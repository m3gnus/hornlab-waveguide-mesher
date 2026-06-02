/**
 * Lookup-table horn profile using PCHIP (monotone cubic Hermite) interpolation.
 *
 * A caller supplies a set of (z, r) control points; this profile interpolates
 * between them to produce a smooth contour. Because the profile is defined by
 * raw data rather than an analytical formula the result is axially symmetric
 * (no phi dependence) - the same r(z) is returned for every azimuthal angle.
 *
 * INTERNAL API: This profile type is intentionally hidden from the UI.  It
 * exists as a data-driven profile surface for internal automation and imports.
 *
 * Storage format: params.lookupPoints = [[z0,r0], [z1,r1], ...]  (JSON array)
 *   - z values must be monotonically increasing
 *   - At least 2 points required
 *   - First point should have z=0 (throat)
 */

// ---------------------------------------------------------------------------
// PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
// Fritsch–Carlson monotone method — preserves monotonicity of the data.
// ---------------------------------------------------------------------------

function pchipSlopes(xs, ys) {
  const n = xs.length;
  const d = new Float64Array(n);

  // Secants
  const delta = new Float64Array(n - 1);
  for (let i = 0; i < n - 1; i++) {
    delta[i] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);
  }

  // Interior slopes — Fritsch–Carlson
  for (let i = 1; i < n - 1; i++) {
    if (delta[i - 1] * delta[i] <= 0) {
      d[i] = 0;
    } else {
      const w1 = 2 * (xs[i + 1] - xs[i]) + (xs[i] - xs[i - 1]);
      const w2 = (xs[i + 1] - xs[i]) + 2 * (xs[i] - xs[i - 1]);
      d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
    }
  }

  // Endpoint slopes — one-sided shape-preserving
  d[0] = _endpointSlope(xs[0], xs[1], xs[2], delta[0], delta[1]);
  d[n - 1] = _endpointSlope(
    xs[n - 1], xs[n - 2], xs[n - 3],
    delta[n - 2], delta[n - 3]
  );

  return d;
}

function _endpointSlope(x0, x1, x2, d0, d1) {
  const h0 = x1 - x0;
  const h1 = x2 - x1;
  let slope = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
  if (Math.sign(slope) !== Math.sign(d0)) {
    slope = 0;
  } else if (Math.sign(d0) !== Math.sign(d1) && Math.abs(slope) > 3 * Math.abs(d0)) {
    slope = 3 * d0;
  }
  return slope;
}

function linearEval(xs, ys, z) {
  if (z <= xs[0]) return ys[0];
  if (z >= xs[1]) return ys[1];

  const span = xs[1] - xs[0];
  if (span === 0) return ys[0];

  return ys[0] + ((z - xs[0]) / span) * (ys[1] - ys[0]);
}

function pchipEval(xs, ys, slopes, z) {
  // Clamp to data range
  if (z <= xs[0]) return ys[0];
  if (z >= xs[xs.length - 1]) return ys[xs.length - 1];

  // Binary search for interval
  let lo = 0;
  let hi = xs.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= z) lo = mid;
    else hi = mid;
  }

  const h = xs[hi] - xs[lo];
  const t = (z - xs[lo]) / h;
  const t2 = t * t;
  const t3 = t2 * t;

  // Hermite basis
  const h00 = 2 * t3 - 3 * t2 + 1;
  const h10 = t3 - 2 * t2 + t;
  const h01 = -2 * t3 + 3 * t2;
  const h11 = t3 - t2;

  return h00 * ys[lo] + h10 * h * slopes[lo] + h01 * ys[hi] + h11 * h * slopes[hi];
}

// ---------------------------------------------------------------------------
// Cache: pre-compute PCHIP slopes once per param set
// ---------------------------------------------------------------------------

let _cachedSignature = null;
let _cachedXs = null;
let _cachedYs = null;
let _cachedSlopes = null;
let _cachedL = 0;
let _cachedSlopesValid = false; // false → use linear fallback

// Quantize control points before hashing so sub-nanometre float drift (from
// parameter round-trips) does not evict the cache.  1e-9 is well below any
// physical tolerance for horn geometry and comfortably above Float64 ULP.
const _CACHE_QUANTUM = 1e-9;
function _quantize(v) {
  return Math.round(v / _CACHE_QUANTUM) * _CACHE_QUANTUM;
}

function _pointsSignature(points) {
  const parts = new Array(points.length);
  for (let i = 0; i < points.length; i++) {
    parts[i] = `${_quantize(points[i][0])},${_quantize(points[i][1])}`;
  }
  return parts.join(';');
}

function _isSlopeArrayFinite(slopes) {
  for (let i = 0; i < slopes.length; i++) {
    if (!Number.isFinite(slopes[i])) return false;
  }
  return true;
}

function _ensureCache(points) {
  const signature = _pointsSignature(points);
  if (_cachedSignature === signature) return;

  const sorted = [...points].sort((a, b) => a[0] - b[0]);
  const n = sorted.length;

  _cachedXs = new Float64Array(n);
  _cachedYs = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    _cachedXs[i] = sorted[i][0];
    _cachedYs[i] = sorted[i][1];
  }

  if (n >= 3) {
    const slopes = pchipSlopes(_cachedXs, _cachedYs);
    // Guard: duplicate or near-duplicate x-values produce divide-by-zero in
    // the secants, which poisons the slopes with NaN.  Fall back to linear
    // interpolation between the sorted points in that case.
    if (_isSlopeArrayFinite(slopes)) {
      _cachedSlopes = slopes;
      _cachedSlopesValid = true;
    } else {
      _cachedSlopes = new Float64Array(n);
      _cachedSlopesValid = false;
    }
  } else {
    _cachedSlopes = new Float64Array(n); // linear fallback for 2 points
    _cachedSlopesValid = false;
  }

  _cachedL = _cachedXs[n - 1] - _cachedXs[0];
  _cachedSignature = signature;
}

/**
 * Piecewise-linear evaluator across the full sorted point array (used as a
 * fallback when PCHIP slopes are non-finite).
 */
function _linearEvalAll(xs, ys, z) {
  if (z <= xs[0]) return ys[0];
  const n = xs.length;
  if (z >= xs[n - 1]) return ys[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= z) lo = mid;
    else hi = mid;
  }
  const span = xs[hi] - xs[lo];
  if (span <= 0) return ys[lo];
  return ys[lo] + ((z - xs[lo]) / span) * (ys[hi] - ys[lo]);
}

// ---------------------------------------------------------------------------
// Public entry — matches calculateOSSE / calculateROSSE contract
// ---------------------------------------------------------------------------

/**
 * @param {number} z  Axial position (in the same units as the control points)
 * @param {number} _p Azimuthal angle (unused — lookup profiles are axisymmetric).
 *                    Kept in the signature to match calculateOSSE / calculateROSSE.
 * @param {object} params  Must contain `lookupPoints: [[z,r], ...]`
 * @returns {{ x: number, y: number }}
 */
export function calculateLookup(z, _p, params) {
  const points = params.lookupPoints;
  if (!points || points.length < 2) {
    return { x: z, y: 0 };
  }

  _ensureCache(points);

  const n = _cachedXs.length;
  let r;
  if (n === 2) {
    r = linearEval(_cachedXs, _cachedYs, z);
  } else if (!_cachedSlopesValid) {
    r = _linearEvalAll(_cachedXs, _cachedYs, z);
  } else {
    r = pchipEval(_cachedXs, _cachedYs, _cachedSlopes, z);
  }

  return { x: z, y: r };
}

/**
 * Return the total axial length of the lookup profile.
 */
export function getLookupLength(params) {
  const points = params.lookupPoints;
  if (!points || points.length < 2) return 0;
  _ensureCache(points);
  return _cachedL;
}
