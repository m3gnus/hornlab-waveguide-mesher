import { clamp, evalParam, toRad } from '../../common.js';
import { DEFAULTS, HORN_PROFILES } from '../constants.js';
import { getGuidingCurveRadius } from './guidingCurve.js';
import { validateParameters } from './validation.js';

function computeOsseBaseRadius(z, r0, k, a0, a) {
  const term1 = (k * r0) ** 2;
  const term2 = 2 * k * r0 * z * Math.tan(a0);
  const term3 = (z ** 2) * (Math.tan(a) ** 2);
  return Math.sqrt(term1 + term2 + term3) + r0 * (1 - k);
}

function computeOsseTermRadius(z, L, s, n, q) {
  if (z <= 0 || n <= 0 || q <= 0 || L <= 0) return 0;

  const zNorm = q * z / L;
  if (zNorm > 1.0) return (s * L / q);

  return (s * L / q) * (1 - Math.pow(1 - Math.pow(zNorm, n), 1 / n));
}

export function computeOsseRadius(z, p, params, overrides = {}) {
  const L = overrides.L ?? evalParam(params.L, p);
  const a = toRad(overrides.aDeg ?? evalParam(params.a, p));
  const a0 = toRad(overrides.a0Deg ?? evalParam(params.a0, p));
  const r0 = overrides.r0 ?? evalParam(params.r0, p);

  const s = params.s !== undefined ? evalParam(params.s, p) : 0;
  const k = params.k === undefined ? DEFAULTS.K : evalParam(params.k, p);
  const n = params.n === undefined ? DEFAULTS.N : evalParam(params.n, p);
  const q = params.q === undefined ? DEFAULTS.Q : evalParam(params.q, p);

  return computeOsseBaseRadius(z, r0, k, a0, a) + computeOsseTermRadius(z, L, s, n, q);
}

function calculateArcCenterFromRadius(p1, p2, arcRadius, preferUpper = true) {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const d = Math.hypot(dx, dy);

  if (d <= 0 || arcRadius < d / 2) return null;

  const midX = (p1.x + p2.x) / 2;
  const midY = (p1.y + p2.y) / 2;
  const h = Math.sqrt(Math.max(0, arcRadius ** 2 - (d / 2) ** 2));

  const nx = -dy / d;
  const ny = dx / d;

  const c1 = { x: midX + nx * h, y: midY + ny * h };
  const c2 = { x: midX - nx * h, y: midY - ny * h };

  return preferUpper ? (c1.y >= c2.y ? c1 : c2) : (c1.y < c2.y ? c1 : c2);
}

function calculateArcCenterFromTangent(p1, p2, tangentAngle) {
  const t = { x: Math.cos(tangentAngle), y: Math.sin(tangentAngle) };
  const n = { x: -t.y, y: t.x };

  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const dDotN = dx * n.x + dy * n.y;

  if (Math.abs(dDotN) <= 1e-6) return null;

  const arcRadius = -((dx ** 2 + dy ** 2) / (2 * dDotN));
  return {
    x: p2.x + n.x * arcRadius,
    y: p2.y + n.y * arcRadius,
    radius: arcRadius
  };
}

function evaluateCircularArc(zMain, r0Main, mouthR, params, p, L) {
  const explicitRadius = evalParam(params.circArcRadius || 0, p);
  const p1 = { x: 0, y: r0Main };
  const p2 = { x: L, y: mouthR };

  let center = null;
  let arcRadius = explicitRadius;

  if (Number.isFinite(explicitRadius) && explicitRadius > 0) {
    center = calculateArcCenterFromRadius(p1, p2, explicitRadius, mouthR > r0Main);
  }

  if (!center) {
    const termAngle = toRad(evalParam(params.circArcTermAngle || 1, p));
    const tangent = calculateArcCenterFromTangent(p1, p2, termAngle);
    if (tangent) {
      center = { x: tangent.x, y: tangent.y };
      arcRadius = tangent.radius;
    }
  }

  if (!center || !Number.isFinite(arcRadius) || arcRadius === 0) {
    return mouthR;
  }

  const dx = zMain - center.x;
  const under = arcRadius ** 2 - dx ** 2;
  if (under < 0) return mouthR;

  const sign = Math.sign(mouthR - center.y) || 1;
  return center.y + sign * Math.sqrt(under);
}

/**
 * Binary search for coverage angle (degrees) that produces targetR at zMain.
 * Used by guiding curve inversion and morph shrinkage.
 *
 * When the search saturates at one of the clamp boundaries (0.5° or 89°),
 * a console.warn is emitted so callers / developers are alerted.
 */
export function invertOsseCoverageAngle(targetR, zMain, p, params, osseConfig) {
  const LOW_BOUND = 0.5;
  const HIGH_BOUND = 89;
  const SATURATE_EPS = 1e-4;

  let low = LOW_BOUND;
  let high = HIGH_BOUND;
  for (let i = 0; i < 24; i += 1) {
    const mid = (low + high) / 2;
    const rMid = computeOsseRadius(zMain, p, params, {
      L: osseConfig.L,
      aDeg: mid,
      a0Deg: osseConfig.a0Deg,
      r0: osseConfig.r0Main
    });
    if (!Number.isFinite(rMid)) break;
    if (rMid < targetR) {
      low = mid;
    } else {
      high = mid;
    }
  }
  const result = clamp((low + high) / 2, LOW_BOUND, HIGH_BOUND);

  if (Math.abs(result - LOW_BOUND) < SATURATE_EPS) {
    console.warn(
      `invertOsseCoverageAngle: saturated at lower bound (${LOW_BOUND}°). ` +
      `targetR=${targetR}, zMain=${zMain}, p=${p}`
    );
  } else if (Math.abs(result - HIGH_BOUND) < SATURATE_EPS) {
    console.warn(
      `invertOsseCoverageAngle: saturated at upper bound (${HIGH_BOUND}°). ` +
      `targetR=${targetR}, zMain=${zMain}, p=${p}`
    );
  }

  return result;
}

function buildCoverageCacheKey(p, params, config) {
  // Include all fields that affect the coverage-angle computation:
  // config geometry (L, a0Deg, r0Base, extLen, extAngleRad) +
  // OSSE shape params (k, n, q, s) +
  // guiding-curve params (gcurveType, gcurveWidth, gcurveAspectRatio, gcurveRot, gcurveSeN) +
  // the azimuthal position p.
  const k = params.k === undefined ? DEFAULTS.K : evalParam(params.k, p);
  const n = params.n === undefined ? DEFAULTS.N : evalParam(params.n, p);
  const q = params.q === undefined ? DEFAULTS.Q : evalParam(params.q, p);
  const s = params.s !== undefined ? evalParam(params.s, p) : 0;
  const gcType = Number(params.gcurveType || 0);
  const gcW = evalParam(params.gcurveWidth || 0, p);
  const gcAR = evalParam(params.gcurveAspectRatio || 1, p);
  const gcRot = evalParam(params.gcurveRot || 0, p);
  const gcSeN = evalParam(params.gcurveSeN || 3, p);

  return [
    p.toFixed(6),
    config.L,
    config.a0Deg,
    config.r0Base,
    config.extLen,
    config.extAngleRad,
    k, n, q, s,
    gcType, gcW, gcAR, gcRot, gcSeN,
  ].join('|');
}

function computeCoverageAngleFromGuidingCurve(p, params, config, coverageCache = null) {
  if (coverageCache instanceof Map) {
    const key = buildCoverageCacheKey(p, params, config);
    if (coverageCache.has(key)) return coverageCache.get(key);

    const computed = computeCoverageAngleFromGuidingCurve(p, params, config, null);
    coverageCache.set(key, computed);
    return computed;
  }

  const {
    extLen,
    slotLen,
    r0Base,
    extAngleRad,
    a0Deg,
    L
  } = config;

  const targetR = getGuidingCurveRadius(p, params);
  if (!Number.isFinite(targetR)) return evalParam(params.a, p);

  // GCurve semantics (ATH parity): the guiding curve defines the MOUTH
  // cross-section — the OS-SE profile is inverted so it hits the gcurve
  // radius at z = L (the end of the main body), not at z = L·gcurveDist.
  //
  // The user guide describes the guiding curve as "a closed loop of wire
  // that the horn surface goes through at some defined distance from the
  // throat" (§2.1.3 of the Ath user guide), but the shipping ATH binary
  // that produced our reference meshes treats it as the mouth boundary —
  // test1 (Dist=0.5) and test5 (Dist=0.8) both produce identical r(z)
  // profiles terminating at r = GCurve.Width/2 at z = L, regardless of
  // Dist.  Matching the reference binary is the parity contract.
  //
  // gcurveDist is kept on the config for future use / UI display but is
  // not consulted here.  Passing it through invertOsseCoverageAngle at
  // z = L·Dist instead of z = L caused mouth radii up to +260 % over the
  // ATH reference for configs where GCurve.Width ≠ Morph.TargetWidth.
  const r0Main = r0Base + extLen * Math.tan(extAngleRad);
  const zMain = Math.max(0, L);
  if (!Number.isFinite(zMain) || zMain <= 0) return evalParam(params.a, p);

  return invertOsseCoverageAngle(targetR, zMain, p, params, { L, a0Deg, r0Main });
}

export function calculateOSSE(z, p, params, options = {}) {
  const validation = validateParameters(params, 'OSSE');
  if (!validation.valid) {
    console.error('Validation failed:', validation.errors);
    return { x: NaN, y: NaN };
  }

  const L = options.L ?? evalParam(params.L, p);
  const extLen = Math.max(0, evalParam(params.throatExtLength || 0, p));
  const slotLen = Math.max(0, evalParam(params.slotLength || 0, p));
  const totalLength = L + extLen + slotLen;
  const extAngleRad = toRad(evalParam(params.throatExtAngle || 0, p));

  const r0Base = evalParam(params.r0, p);
  const a0Deg = evalParam(params.a0, p);
  const r0Main = r0Base + extLen * Math.tan(extAngleRad);

  const config = { totalLength, extLen, slotLen, r0Base, extAngleRad, a0Deg, L };

  const throatProfile = Number(params.throatProfile || HORN_PROFILES.STANDARD);
  const gcurveType = Number(params.gcurveType || 0);
  const coverageAngle = options.coverageAngle
    ?? (gcurveType === 0
      ? evalParam(params.a, p)
      : computeCoverageAngleFromGuidingCurve(p, params, config, options.gcurveCache));

  let radius;
  if (z <= extLen) {
    radius = r0Base + z * Math.tan(extAngleRad);
  } else if (z <= extLen + slotLen) {
    radius = r0Main;
  } else {
    const zMain = z - extLen - slotLen;

    if (throatProfile === HORN_PROFILES.CIRCULAR_ARC) {
      const aRad = toRad(coverageAngle);
      const mouthR = r0Main + L * Math.tan(aRad);
      radius = evaluateCircularArc(zMain, r0Main, mouthR, params, p, L);
    } else {
      radius = computeOsseRadius(zMain, p, params, {
        L,
        aDeg: coverageAngle,
        a0Deg,
        r0: r0Main
      });
    }
  }

  let x = z;
  let y = radius;
  const rotDeg = evalParam(params.rot || 0, p);

  if (Number.isFinite(rotDeg) && rotDeg !== 0) {
    const rotRad = toRad(rotDeg);
    const dx = x;
    const dy = y - r0Base;
    x = dx * Math.cos(rotRad) - dy * Math.sin(rotRad);
    y = r0Base + dx * Math.sin(rotRad) + dy * Math.cos(rotRad);
  }

  return { x, y };
}
