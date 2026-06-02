import { evalParam } from '../../common.js';
import { evaluateMultiAxisProfile, resolveProfileSystem } from '../profileSystem.js';

// Minimum resolution delta between throat and mouth that is worth a
// non-uniform slice map.  Below this (1 % difference in resolution) the
// generated map is numerically indistinguishable from a uniform t = j / N
// distribution — callers treat `null` as "fall through to the segment-count
// heuristic or use uniform spacing".  Returning null in that case avoids
// emitting a redundant array.
const RESOLUTION_DELTA_EPSILON = 0.01;

// Curvature-adaptive slice-map: number of parameter samples along t used to
// estimate κ and ds in the (axial, radial) plane.
const ADAPTIVE_T_SAMPLES = 400;
// Phi angles at which we measure curvature; max across phi is used so that
// multi-axis profiles (H/V asymmetry) get slices placed where ANY axis has
// high curvature.
const ADAPTIVE_PHI_SAMPLES = [0, Math.PI / 4, Math.PI / 2, (3 * Math.PI) / 4];

// ATH V2025-06 ASRO2 GridExport throat slice placement for Mesh.LengthSegments=20.
// Kept internal: this is a diagnostic/parity sampling mode, not a UI feature.
const ATH_PARITY_T_20 = Object.freeze([
  0.0,
  0.031652775,
  0.069285650,
  0.111291038,
  0.158158738,
  0.208217141,
  0.261010634,
  0.315152186,
  0.371049458,
  0.427239696,
  0.483180970,
  0.538366332,
  0.593546216,
  0.647147114,
  0.701376236,
  0.753382922,
  0.804185680,
  0.854976845,
  0.904174233,
  0.953060714,
  1.0
]);

function isAthParitySampling(params, options = {}) {
  const mode = options.mode ?? params?.samplingMode ?? params?.meshSamplingMode;
  return params?.athParitySampling === true || mode === 'ath-parity';
}

function buildAthParitySliceMap(lengthSteps) {
  const steps = Math.max(1, Math.round(lengthSteps));
  if (steps === ATH_PARITY_T_20.length - 1) {
    return [...ATH_PARITY_T_20];
  }

  const out = new Array(steps + 1);
  out[0] = 0.0;
  out[steps] = 1.0;
  const refSteps = ATH_PARITY_T_20.length - 1;
  for (let j = 1; j < steps; j += 1) {
    const pos = (j / steps) * refSteps;
    const lo = Math.floor(pos);
    const hi = Math.min(refSteps, lo + 1);
    const frac = pos - lo;
    out[j] = ATH_PARITY_T_20[lo] + (ATH_PARITY_T_20[hi] - ATH_PARITY_T_20[lo]) * frac;
  }
  return out;
}

function buildResolutionMap(lengthSteps, resT, resM) {
  if (!Number.isFinite(resT) || !Number.isFinite(resM) || resT <= 0 || resM <= 0) return null;
  if (Math.abs(resT - resM) <= RESOLUTION_DELTA_EPSILON) return null;

  const avgRes = 0.5 * (resT + resM);
  return Array.from({ length: lengthSteps + 1 }, (_, j) => {
    const t = j / lengthSteps;
    return (resT * t + 0.5 * (resM - resT) * t * t) / avgRes;
  });
}

function buildThroatSegmentMap(lengthSteps, throatSegments, extLen, slotLen, L) {
  if (throatSegments <= 0 || throatSegments >= lengthSteps) return null;

  const totalLength = L + extLen + slotLen;
  const extFraction = (extLen + slotLen) / totalLength;

  if (extFraction <= 0 || extFraction >= 1) return null;

  return Array.from({ length: lengthSteps + 1 }, (_, j) => {
    if (j <= throatSegments) {
      return extFraction * (j / throatSegments);
    }
    const t = (j - throatSegments) / (lengthSteps - throatSegments);
    return extFraction + (1 - extFraction) * t;
  });
}

export function buildSliceMap(params, lengthSteps, options = {}) {
  if (isAthParitySampling(params, options)) {
    return buildAthParitySliceMap(lengthSteps);
  }

  if (options.mode === 'adaptive') {
    const adaptive = buildAdaptiveSliceMap(params, lengthSteps, {
      resolvedSystem: options.resolvedSystem
    });
    if (adaptive) return adaptive;
    // Fall through to the default heuristic if adaptive produced nothing
    // meaningful (e.g. degenerate profile).
  }

  // If an explicit slice density override is provided, use it directly.
  // This decouples viewport axial distribution from the BEM element-size params.
  const density = parseFloat(params.throatSliceDensity);
  if (Number.isFinite(density) && density > 0 && density < 1) {
    return buildResolutionMap(lengthSteps, density, 1.0 - density);
  }

  const resT = Number(params.throatResolution);
  const resM = Number(params.mouthResolution);
  const resolutionMap = buildResolutionMap(lengthSteps, resT, resM);
  if (resolutionMap) return resolutionMap;

  const throatSegments = Number(params.throatSegments || 0);
  const extLen = Math.max(0, evalParam(params.throatExtLength || 0, 0));
  const slotLen = Math.max(0, evalParam(params.slotLength || 0, 0));
  const L = Math.max(0, evalParam(params.L || 0, 0));

  return buildThroatSegmentMap(lengthSteps, throatSegments, extLen, slotLen, L);
}

/**
 * Curvature-adaptive slice map: distributes t ∈ [0, 1] so that slices cluster
 * where the profile has high axial-radial curvature.
 *
 * Uses the sagitta-optimal metric ∫(√κ + ε)·ds — the same weighting that
 * `buildSuperellipseQuadrantAngles` uses for phi — where κ and ds are
 * measured in the (axial, radial) plane.  Curvature is maxed across a set of
 * phi angles so multi-axis profiles don't starve the axis with the tighter
 * mouth roundover.
 *
 * Returns an array of length `lengthSteps + 1` with strictly monotonic
 * values from 0 to 1, or `null` if the profile is too flat to benefit.
 */
export function buildAdaptiveSliceMap(params, lengthSteps, options = {}) {
  const N = Math.max(16, options.sampleCount || ADAPTIVE_T_SAMPLES);
  const phis = options.phiSamples || ADAPTIVE_PHI_SAMPLES;
  const steps = Math.max(1, Math.round(lengthSteps));
  if (!Array.isArray(phis) || phis.length === 0) return null;

  let system;
  try {
    system = options.resolvedSystem || resolveProfileSystem(params);
  } catch (_err) {
    return null;
  }
  if (!system) return null;

  // Pass 1: sample the profile at every (t_i, phi_k).
  const xByPhi = [];
  const yByPhi = [];
  for (let k = 0; k < phis.length; k += 1) {
    xByPhi.push(new Float64Array(N + 1));
    yByPhi.push(new Float64Array(N + 1));
  }
  for (let i = 0; i <= N; i += 1) {
    const t = i / N;
    for (let k = 0; k < phis.length; k += 1) {
      let profile;
      try {
        profile = evaluateMultiAxisProfile(t, phis[k], system);
      } catch (_err) {
        return null;
      }
      const px = Number(profile?.x);
      const py = Number(profile?.y);
      if (!Number.isFinite(px) || !Number.isFinite(py)) return null;
      xByPhi[k][i] = px;
      yByPhi[k][i] = py;
    }
  }

  // Pass 2: per-sample √κ (max over phi) and per-segment ds (max over phi).
  const sqrtK = new Float64Array(N + 1);
  const ds = new Float64Array(N);

  for (let k = 0; k < phis.length; k += 1) {
    const xk = xByPhi[k];
    const yk = yByPhi[k];
    for (let i = 0; i <= N; i += 1) {
      const iPrev = Math.max(0, i - 1);
      const iNext = Math.min(N, i + 1);
      const h = (iNext - iPrev) / N;
      if (h <= 0) continue;
      const dx = (xk[iNext] - xk[iPrev]) / h;
      const dy = (yk[iNext] - yk[iPrev]) / h;
      let d2x = 0;
      let d2y = 0;
      if (i > 0 && i < N) {
        const h0 = 1 / N;
        d2x = (xk[iNext] - 2 * xk[i] + xk[iPrev]) / (h0 * h0);
        d2y = (yk[iNext] - 2 * yk[i] + yk[iPrev]) / (h0 * h0);
      }
      const speed2 = dx * dx + dy * dy;
      const num = Math.abs(dx * d2y - dy * d2x);
      const den = Math.pow(speed2, 1.5);
      const kappa = den > 1e-15 ? num / den : 0;
      const s = Math.sqrt(kappa);
      if (s > sqrtK[i]) sqrtK[i] = s;
    }
    for (let i = 0; i < N; i += 1) {
      const d = Math.hypot(xk[i + 1] - xk[i], yk[i + 1] - yk[i]);
      if (d > ds[i]) ds[i] = d;
    }
  }

  let sumSqrtKDs = 0;
  let sumDs = 0;
  for (let i = 0; i < N; i += 1) {
    const midSqrtK = 0.5 * (sqrtK[i] + sqrtK[i + 1]);
    sumSqrtKDs += midSqrtK * ds[i];
    sumDs += ds[i];
  }
  if (sumDs < 1e-12) return null;

  // ε = 0.2 × mean(√κ) on the profile arc — same floor the superellipse
  // angle builder uses.  Prevents flat regions from collapsing to
  // near-zero weight.
  const meanSqrtK = sumSqrtKDs / sumDs;
  const eps = 0.2 * meanSqrtK + 1e-12;

  // Pass 3: cumulative weight along t, sampled at t_i = i/N.
  const cumW = new Float64Array(N + 1);
  for (let i = 0; i < N; i += 1) {
    const midSqrtK = 0.5 * (sqrtK[i] + sqrtK[i + 1]);
    cumW[i + 1] = cumW[i] + (midSqrtK + eps) * ds[i];
  }
  const totalW = cumW[N];
  if (totalW < 1e-12) return null;

  // If the adaptive weight is nearly uniform with respect to t, the map is
  // indistinguishable from j/N — skip to avoid re-meshing for nothing.
  let maxDeviation = 0;
  for (let i = 1; i < N; i += 1) {
    const uniform = (i / N) * totalW;
    const dev = Math.abs(cumW[i] - uniform) / totalW;
    if (dev > maxDeviation) maxDeviation = dev;
  }
  if (maxDeviation < 0.02) return null;

  // Pass 4: invert — for each slice j, find t such that cumW(t) = j/steps · totalW.
  const sliceMap = new Array(steps + 1);
  sliceMap[0] = 0;
  sliceMap[steps] = 1;
  let cursor = 0;
  for (let j = 1; j < steps; j += 1) {
    const target = (j / steps) * totalW;
    while (cursor < N && cumW[cursor + 1] < target) cursor += 1;
    if (cursor >= N) {
      sliceMap[j] = 1;
      continue;
    }
    const lo = cumW[cursor];
    const hi = cumW[cursor + 1];
    const span = hi - lo;
    const u = span > 1e-15 ? (target - lo) / span : 0;
    sliceMap[j] = (cursor + u) / N;
  }

  // Enforce strict monotonicity (guards against pathological profiles).
  for (let j = 1; j <= steps; j += 1) {
    if (sliceMap[j] <= sliceMap[j - 1]) {
      sliceMap[j] = Math.min(1, sliceMap[j - 1] + 1e-9);
    }
  }
  sliceMap[steps] = 1;
  return sliceMap;
}
