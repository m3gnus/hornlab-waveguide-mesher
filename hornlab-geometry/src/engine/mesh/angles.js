import { evalParam } from '../../common.js';
import { DEFAULTS } from '../constants.js';
import { resolveProfileSystem, evaluateMultiAxisProfile } from '../profileSystem.js';


function buildUniformAngles(segmentCount) {
  return Array.from({ length: segmentCount }, (_, i) => (i / segmentCount) * Math.PI * 2);
}

function isAthParitySampling(params, options = {}) {
  const mode = options.mode ?? params?.samplingMode ?? params?.meshSamplingMode;
  return params?.athParitySampling === true || mode === 'ath-parity';
}

function normalizeAngularSegments(rawCount) {
  const count = Math.max(4, Math.round(Number(rawCount) || 0));
  if (count % 4 === 0) return count;
  // ATH-compatible fallback: snap up to a full 8-way symmetric ring.
  return Math.max(8, Math.ceil(count / 8) * 8);
}

function buildQuadrantAngles(pointsPerQuadrant, halfW, halfH, cornerR, cornerSegments) {
  if (!Number.isFinite(halfW) || !Number.isFinite(halfH) || halfW <= 0 || halfH <= 0) {
    return null;
  }

  const maxCorner = Math.max(0, Math.min(halfW, halfH) - 1e-6);
  const clampedCorner = Math.min(cornerR, maxCorner);

  if (clampedCorner <= 0 || cornerSegments <= 0) {
    return Array.from({ length: pointsPerQuadrant + 1 }, (_, i) => (Math.PI / 2) * (i / pointsPerQuadrant));
  }

  const theta1 = Math.atan2(halfH - clampedCorner, halfW);
  const theta2 = Math.atan2(halfH, halfW - clampedCorner);

  // cornerSegments is the number of internal points for the corner arc.
  // We need pointsPerQuadrant + 1 total points in the array.
  const remainingSegments = Math.max(1, pointsPerQuadrant - cornerSegments);

  // When halfW ≈ halfH ≈ clampedCorner the corner fills the whole quadrant,
  // so theta1 → 0 and theta2 → π/2, leaving no side segments.  Split the
  // remainder 50/50 in that degenerate case rather than dividing by zero.
  const sideSpan1 = theta1;
  const sideSpan2 = Math.max(0, (Math.PI / 2) - theta2);
  const sideSum = sideSpan1 + sideSpan2;
  const side1Frac = sideSum > 1e-12 ? sideSpan1 / sideSum : 0.5;
  const side1Seg = Math.max(1, Math.min(
    remainingSegments - 1,
    Math.round(remainingSegments * side1Frac)
  ));
  const side2Seg = Math.max(1, remainingSegments - side1Seg);

  const angles = [];
  // Loop 1: Start point to corner start
  for (let i = 0; i <= side1Seg; i += 1) {
    angles.push(theta1 * (i / side1Seg));
  }

  // Loop 2: Corner internal points
  const cx = halfW - clampedCorner;
  const cy = halfH - clampedCorner;
  if (cornerSegments > 0) {
    for (let i = 1; i <= cornerSegments; i += 1) {
      const phi = (i / (cornerSegments + 1)) * (Math.PI / 2);
      angles.push(Math.atan2(cy + clampedCorner * Math.sin(phi), cx + clampedCorner * Math.cos(phi)));
    }
  }

  // Loop 3: Corner end to 90 degrees
  for (let i = 1; i <= side2Seg; i += 1) {
    angles.push(theta2 + ((Math.PI / 2) - theta2) * (i / side2Seg));
  }

  // Final length should be (side1Seg + 1) + cornerSegments + side2Seg
  // = remainingSegments + 1 + cornerSegments = pointsPerQuadrant + 1.
  return angles;
}

function mirrorQuadrantAngles(quadrantAngles) {
  const full = [...quadrantAngles];

  for (let i = quadrantAngles.length - 2; i >= 0; i -= 1) {
    full.push(Math.PI - quadrantAngles[i]);
  }

  for (let i = 1; i < quadrantAngles.length; i += 1) {
    full.push(Math.PI + quadrantAngles[i]);
  }

  for (let i = quadrantAngles.length - 2; i > 0; i -= 1) {
    full.push((Math.PI * 2) - quadrantAngles[i]);
  }

  return full;
}

/**
 * Build quadrant angles for a superellipse |x/a|^n + |y/b|^n = 1 using
 * curvature-weighted arc-length distribution.
 *
 * For n > 2 the sides (near 0° and 90°) are nearly flat while the corners
 * (near 45°) concentrate all the bending.  Uniform-angle or even plain
 * arc-length sampling puts too many vertices on the flat sides and too few
 * at the corners, making the tessellation visibly round off the edges.
 *
 * We distribute vertices uniformly in ∫(√κ + ε) ds — the sagitta-optimal
 * metric (piecewise-linear chord error ∝ κ·s²) with a floor ε that
 * prevents degenerate long segments on the flat sides.
 *
 * Returns pointsPerQuadrant + 1 angles from 0 to π/2.
 */
export function buildSuperellipseQuadrantAngles(pointsPerQuadrant, a, b, n) {
  const NSAMPLE = 1000;
  const h = (Math.PI / 2) / NSAMPLE;

  function seRadius(phi) {
    const ca = Math.abs(Math.cos(phi));
    const sa = Math.abs(Math.sin(phi));
    if (ca < 1e-12) return b;
    if (sa < 1e-12) return a;
    return Math.pow(Math.pow(ca / a, n) + Math.pow(sa / b, n), -1 / n);
  }

  // --- Pass 1: cumulative arc-length and √κ·ds ---
  const phis = new Float64Array(NSAMPLE + 1);
  const cumArc = new Float64Array(NSAMPLE + 1);
  const cumSqrtK = new Float64Array(NSAMPLE + 1); // ∫ √κ ds
  let prevX = a;
  let prevY = 0;

  for (let i = 1; i <= NSAMPLE; i += 1) {
    const phi = i * h;
    phis[i] = phi;

    const r = seRadius(phi);
    const x = r * Math.cos(phi);
    const y = r * Math.sin(phi);
    const ds = Math.hypot(x - prevX, y - prevY);
    cumArc[i] = cumArc[i - 1] + ds;

    // Polar curvature: κ = |r² + 2r'² − r·r''| / (r² + r'²)^(3/2)
    const rm = seRadius(Math.max(0, phi - h));
    const rp = seRadius(Math.min(Math.PI / 2, phi + h));
    const dr = (rp - rm) / (2 * h);
    const d2r = (rp - 2 * r + rm) / (h * h);
    const num = Math.abs(r * r + 2 * dr * dr - r * d2r);
    const den = Math.pow(r * r + dr * dr, 1.5);
    const kappa = den > 1e-15 ? num / den : 0;

    cumSqrtK[i] = cumSqrtK[i - 1] + Math.sqrt(kappa) * ds;
    prevX = x;
    prevY = y;
  }

  const totalArc = cumArc[NSAMPLE];
  const totalSqrtK = cumSqrtK[NSAMPLE];

  if (totalArc < 1e-15) {
    return Array.from({ length: pointsPerQuadrant + 1 }, (_, i) => (Math.PI / 2) * (i / pointsPerQuadrant));
  }

  // --- Pass 2: build blended cumulative weight ---
  // ε = 0.2 × mean(√κ) prevents degenerate long segments on flat sides.
  const meanSqrtK = totalSqrtK / totalArc;
  const eps = 0.2 * meanSqrtK;

  const cumW = new Float64Array(NSAMPLE + 1);
  prevX = a;
  prevY = 0;
  for (let i = 1; i <= NSAMPLE; i += 1) {
    const phi = i * h;
    const r = seRadius(phi);
    const x = r * Math.cos(phi);
    const y = r * Math.sin(phi);
    const ds = Math.hypot(x - prevX, y - prevY);

    // Local √κ from the cumulative integral (avoids recomputing)
    const localSqrtK = (cumSqrtK[i] - cumSqrtK[i - 1]) / (ds || 1e-15);

    cumW[i] = cumW[i - 1] + (localSqrtK + eps) * ds;
    prevX = x;
    prevY = y;
  }

  const totalW = cumW[NSAMPLE];
  if (totalW < 1e-15) {
    return Array.from({ length: pointsPerQuadrant + 1 }, (_, i) => (Math.PI / 2) * (i / pointsPerQuadrant));
  }

  // --- Distribute points uniformly in weighted metric ---
  const result = [0];
  let si = 0;
  for (let k = 1; k < pointsPerQuadrant; k += 1) {
    const target = (k / pointsPerQuadrant) * totalW;
    while (si < NSAMPLE && cumW[si + 1] < target) si += 1;
    const denom = cumW[si + 1] - cumW[si];
    const frac = denom > 1e-15 ? (target - cumW[si]) / denom : 0;
    result.push(phis[si] + frac * h);
  }
  result.push(Math.PI / 2);

  return result;
}

/**
 * Build curvature-weighted quadrant angles by numerically sampling the
 * actual mouth curve r(phi) at t=1. Works for any profile system shape —
 * multi-axis blends, superellipse cross-sections, arbitrary contours.
 *
 * Reuses the same ∫(√κ + ε) ds metric as buildSuperellipseQuadrantAngles
 * but derives r(phi) from the profile system evaluation instead of an
 * analytical superellipse.
 */
function buildCurvatureWeightedQuadrantAngles(pointsPerQuadrant, profileSystem) {
  const NSAMPLE = 1000;
  const h = (Math.PI / 2) / NSAMPLE;

  function rAt(phi) {
    const result = evaluateMultiAxisProfile(1, phi, profileSystem);
    return result.y;
  }

  // --- Pass 1: cumulative arc-length and √κ·ds ---
  const phis = new Float64Array(NSAMPLE + 1);
  const cumArc = new Float64Array(NSAMPLE + 1);
  const cumSqrtK = new Float64Array(NSAMPLE + 1);

  const r0 = rAt(0);
  let prevX = r0;
  let prevY = 0;
  const rValues = new Float64Array(NSAMPLE + 1);
  rValues[0] = r0;

  for (let i = 1; i <= NSAMPLE; i += 1) {
    const phi = i * h;
    phis[i] = phi;

    const r = rAt(phi);
    rValues[i] = r;
    const x = r * Math.cos(phi);
    const y = r * Math.sin(phi);
    const ds = Math.hypot(x - prevX, y - prevY);
    cumArc[i] = cumArc[i - 1] + ds;

    // Polar curvature: κ = |r² + 2r'² − r·r''| / (r² + r'²)^(3/2)
    const rm = rValues[i - 1];
    const rp = rAt(Math.min(Math.PI / 2, phi + h));
    const dr = (rp - rm) / (2 * h);
    const d2r = (rp - 2 * r + rm) / (h * h);
    const num = Math.abs(r * r + 2 * dr * dr - r * d2r);
    const den = Math.pow(r * r + dr * dr, 1.5);
    const kappa = den > 1e-15 ? num / den : 0;

    cumSqrtK[i] = cumSqrtK[i - 1] + Math.sqrt(kappa) * ds;
    prevX = x;
    prevY = y;
  }

  const totalArc = cumArc[NSAMPLE];
  const totalSqrtK = cumSqrtK[NSAMPLE];

  if (totalArc < 1e-15) {
    return Array.from({ length: pointsPerQuadrant + 1 }, (_, i) => (Math.PI / 2) * (i / pointsPerQuadrant));
  }

  // --- Pass 2: build blended cumulative weight ---
  const meanSqrtK = totalSqrtK / totalArc;
  const eps = 0.2 * meanSqrtK;

  const cumW = new Float64Array(NSAMPLE + 1);
  prevX = rValues[0];
  prevY = 0;
  for (let i = 1; i <= NSAMPLE; i += 1) {
    const phi = i * h;
    const r = rValues[i];
    const x = r * Math.cos(phi);
    const y = r * Math.sin(phi);
    const ds = Math.hypot(x - prevX, y - prevY);

    const localSqrtK = (cumSqrtK[i] - cumSqrtK[i - 1]) / (ds || 1e-15);
    cumW[i] = cumW[i - 1] + (localSqrtK + eps) * ds;
    prevX = x;
    prevY = y;
  }

  const totalW = cumW[NSAMPLE];
  if (totalW < 1e-15) {
    return Array.from({ length: pointsPerQuadrant + 1 }, (_, i) => (Math.PI / 2) * (i / pointsPerQuadrant));
  }

  // --- Distribute points uniformly in weighted metric ---
  const result = [0];
  let si = 0;
  for (let k = 1; k < pointsPerQuadrant; k += 1) {
    const target = (k / pointsPerQuadrant) * totalW;
    while (si < NSAMPLE && cumW[si + 1] < target) si += 1;
    const denom = cumW[si + 1] - cumW[si];
    const frac = denom > 1e-15 ? (target - cumW[si]) / denom : 0;
    result.push(phis[si] + frac * h);
  }
  result.push(Math.PI / 2);

  return result;
}

export function buildAngleList(params, mouthExtents, options = {}) {
  const angularSegments = normalizeAngularSegments(
    Number(params.angularSegments || DEFAULTS.ANGULAR_SEGMENTS)
  );
  if (!Number.isFinite(angularSegments) || angularSegments < 4) {
    return { fullAngles: [0], pointsPerQuadrant: 0 };
  }

  const pointsPerQuadrant = angularSegments / 4;

  // Hidden diagnostic/parity mode for ATH ASRO-style point-grid comparisons:
  // ATH places the effective 56-sample full ring uniformly in phi, yielding
  // 15 quarter-domain samples including both symmetry axes.
  if (isAthParitySampling(params, options)) {
    return {
      fullAngles: buildUniformAngles(angularSegments),
      pointsPerQuadrant
    };
  }

  // Multi-axis profile system: use curvature-weighted sampling of the actual
  // mouth curve, which automatically adapts to any shape — blended profiles,
  // superellipse cross-sections, angular-sharpness zones.
  const profileSystem = resolveProfileSystem(params);
  if (profileSystem) {
    const quadrantAngles = buildCurvatureWeightedQuadrantAngles(pointsPerQuadrant, profileSystem);
    if (quadrantAngles && quadrantAngles.length === pointsPerQuadrant + 1) {
      return {
        fullAngles: mirrorQuadrantAngles(quadrantAngles),
        pointsPerQuadrant
      };
    }
    // Fallback to uniform if curvature sampling fails
    return { fullAngles: buildUniformAngles(angularSegments), pointsPerQuadrant: 0 };
  }

  const isSquareCrossSection = params.type === 'Classical'
    && Number(params.classicalCrossSection) === 1;

  let quadrantAngles;

  if (isSquareCrossSection) {
    const n = Math.max(2, Math.min(20, Number(params.classicalSqExponent ?? 6)));
    const hw = mouthExtents.halfW;
    const hh = mouthExtents.halfH;

    if (n > 2 + 1e-9 && hw > 0 && hh > 0) {
      // Distribute angles by superellipse arc length so vertices concentrate
      // at the high-curvature corners instead of being uniformly spaced.
      quadrantAngles = buildSuperellipseQuadrantAngles(pointsPerQuadrant, hw, hh, n);
    } else {
      quadrantAngles = buildQuadrantAngles(pointsPerQuadrant, hw, hh, 0, 0);
    }
  } else {
    const cornerR = Math.max(0, evalParam(params.morphCorner || 0, 0));
    const cornerSegs = Math.max(0, Math.round(params.cornerSegments || 4) - 1);
    quadrantAngles = buildQuadrantAngles(
      pointsPerQuadrant, mouthExtents.halfW, mouthExtents.halfH, cornerR, cornerSegs
    );
  }

  if (!quadrantAngles || quadrantAngles.length !== pointsPerQuadrant + 1) {
    return { fullAngles: buildUniformAngles(angularSegments), pointsPerQuadrant: 0 };
  }

  return {
    fullAngles: mirrorQuadrantAngles(quadrantAngles),
    pointsPerQuadrant
  };
}
