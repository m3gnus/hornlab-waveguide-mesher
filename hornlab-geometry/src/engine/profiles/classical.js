import { evalParam, toRad } from '../../common.js';
import { CLASSICAL_SHAPES } from '../constants.js';
import {
  superellipseHalfDims,
  superellipseRadius,
  transitionAt
} from '../crossSection.js';
import { validateParameters } from './validation.js';

/**
 * Classical horn profile functions.
 *
 * All profiles compute r(z) where z is the axial distance from the throat
 * and r0 is the throat radius.  The returned {x, y} pair matches the
 * contract of OSSE/R-OSSE: x = axial position, y = radial radius.
 *
 * Square cross-section mode uses area-driven superellipse geometry:
 * the horn law controls area S(z), while aspect ratio and superellipse
 * exponent control the cross-section shape independently.
 *
 * options.normalized — when true the first argument is t in [0,1] instead
 * of z in mm (used by tractrix which derives its own length).
 */

// ---------------------------------------------------------------------------
// Null-safe param read: treats 0 as a valid value (unlike ||).
// ---------------------------------------------------------------------------
function paramVal(val, fallback) {
  return val != null && val !== '' ? val : fallback;
}

// superellipseHalfDims / superellipseRadius are imported from crossSection.js
// (the canonical source).  They used to be duplicated here from before the
// cross-section logic was extracted.

// ---------------------------------------------------------------------------
// Individual shape functions — each returns the radius at axial position z.
// ---------------------------------------------------------------------------

function conicalRadius(z, r0, halfAngleRad, L, flare2) {
  if (!flare2 || !flare2.enabled) {
    return r0 + z * Math.tan(halfAngleRad);
  }
  const zTransition = flare2.start * L;
  if (z <= zTransition) {
    return r0 + z * Math.tan(halfAngleRad);
  }

  const rAtTransition = r0 + zTransition * Math.tan(halfAngleRad);
  const dz = z - zTransition;

  if (!flare2.smooth) {
    return rAtTransition + dz * Math.tan(flare2.angle);
  }

  // Smooth blend: analytical integral of tanh-blended slope
  const blendWidth = Math.max(L * 0.05, 1);
  const sigma = dz / blendWidth;
  const slope1 = Math.tan(halfAngleRad);
  const slope2 = Math.tan(flare2.angle);
  const logCosh2Sigma = Math.log(Math.cosh(2 * sigma));
  const blendIntegral = blendWidth * (0.5 * sigma + 0.25 * logCosh2Sigma);

  return rAtTransition + slope1 * dz + (slope2 - slope1) * blendIntegral;
}

function exponentialRadius(z, r0, m) {
  return r0 * Math.exp(m * z / 2);
}

function hyperbolicRadius(z, r0, m, T) {
  // Area-law convention: radius = sqrt(area expansion factor)
  return r0 * Math.sqrt(Math.cosh(m * z) + T * Math.sinh(m * z));
}

function catenoidalRadius(z, r0, a) {
  const scale = a > 0 ? a : r0;
  return r0 * Math.cosh(z / scale);
}

function besselRadius(z, r0, x0, alpha) {
  if (x0 === 0) return r0;
  const base = 1 + z / x0;
  if (base <= 0 && alpha !== Math.floor(alpha)) return r0;
  return r0 * Math.pow(base, alpha);
}

function tractrixProfile(t, r0, R) {
  const a = R;
  if (r0 >= a || a <= 0 || r0 <= 0) return { x: 0, y: Math.max(r0, 0) };

  const ratio = a / r0;
  const tThroat = Math.log(ratio + Math.sqrt(ratio * ratio - 1));
  const xThroat = a * (tThroat - Math.tanh(tThroat));

  const tParam = tThroat * (1 - t);
  const coshT = Math.cosh(tParam);
  const xRaw = a * (tParam - Math.tanh(tParam));

  return { x: xThroat - xRaw, y: a / coshT };
}

function tractrixLength(r0, R) {
  const a = R;
  if (r0 >= a || a <= 0 || r0 <= 0) return 0;
  const ratio = a / r0;
  const tThroat = Math.log(ratio + Math.sqrt(ratio * ratio - 1));
  return a * (tThroat - Math.tanh(tThroat));
}

// ---------------------------------------------------------------------------
// Evaluate shape radius (circular mode)
// ---------------------------------------------------------------------------
function evaluateShapeRadius(shape, z, p, r0, params, L) {
  switch (shape) {
    case CLASSICAL_SHAPES.CONICAL: {
      const coverage = toRad(evalParam(paramVal(params.classicalCoverageAngle, 20), p));
      const halfAngle = coverage / 2;
      const flare2 = getFlare2Config(params, p);
      return conicalRadius(z, r0, halfAngle, L, flare2);
    }
    case CLASSICAL_SHAPES.EXPONENTIAL: {
      const m = evalParam(paramVal(params.classicalM, 0.02), p);
      return exponentialRadius(z, r0, m);
    }
    case CLASSICAL_SHAPES.HYPERBOLIC: {
      const m = evalParam(paramVal(params.classicalM, 0.02), p);
      const T = evalParam(paramVal(params.classicalT, 1), p);
      return hyperbolicRadius(z, r0, m, T);
    }
    case CLASSICAL_SHAPES.CATENOIDAL: {
      const catA = evalParam(params.classicalCatA != null ? params.classicalCatA : r0, p);
      return catenoidalRadius(z, r0, catA);
    }
    case CLASSICAL_SHAPES.BESSEL: {
      const x0 = evalParam(paramVal(params.classicalX0, 50), p);
      const alpha = evalParam(paramVal(params.classicalAlpha, 1), p);
      return besselRadius(z, r0, x0, alpha);
    }
    default:
      return conicalRadius(z, r0, toRad(10), L, null);
  }
}

function getFlare2Config(params, p) {
  if (!Number(params.classicalFlare2Enabled)) return null;
  const coverage = toRad(evalParam(paramVal(params.classicalFlare2Angle, 40), p));
  return {
    enabled: true,
    angle: coverage / 2, // flare2 angle is also full coverage angle, halved internally
    start: Math.max(0, Math.min(1, evalParam(paramVal(params.classicalFlare2Start, 0.5), p))),
    smooth: Number(params.classicalFlare2Blend) === 1
  };
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

export function calculateClassical(z_or_t, p, params, options = {}) {
  const validation = validateParameters(params, 'Classical');
  if (!validation.valid) {
    console.error('Validation failed:', validation.errors);
    return { x: NaN, y: NaN };
  }

  const r0 = evalParam(params.r0, p);
  const shape = Number(paramVal(params.classicalShape, CLASSICAL_SHAPES.CONICAL));
  const crossSection = Number(paramVal(params.classicalCrossSection, 0));

  // --- Tractrix: derives its own length, uses normalized t ---
  if (shape === CLASSICAL_SHAPES.TRACTRIX) {
    const t = options.normalized ? z_or_t : 0;
    const R = evalParam(paramVal(params.R, 140), p);
    const profile = tractrixProfile(t, r0, R);
    if (crossSection === 1) {
      return applySquareCrossSection(profile.x, profile.y, t, p, params);
    }
    return profile;
  }

  // --- Standard classical shapes ---
  const L = evalParam(params.L, p);
  const z = z_or_t;
  const rCirc = evaluateShapeRadius(shape, z, p, r0, params, L);

  if (crossSection === 1) {
    const t = L > 0 ? z / L : 0;
    return applySquareCrossSection(z, rCirc, t, p, params);
  }

  return { x: z, y: rCirc };
}

/**
 * Expose the derived horn length for adaptive phi and other consumers.
 */
export function getClassicalLength(p, params) {
  const r0 = evalParam(params.r0, p);
  const shape = Number(paramVal(params.classicalShape, CLASSICAL_SHAPES.CONICAL));
  if (shape === CLASSICAL_SHAPES.TRACTRIX) {
    const R = evalParam(paramVal(params.R, 140), p);
    return tractrixLength(r0, R);
  }
  return evalParam(params.L, p);
}

// ---------------------------------------------------------------------------
// Area-driven superellipse cross-section transition.
// The horn law defines the area S(z) = π * rCirc². The cross-section shape
// transitions from a circle (n=2, k=1) at the throat to a superellipse
// (user-specified n and k) at the mouth.
// ---------------------------------------------------------------------------

function applySquareCrossSection(x, rCirc, t, p, params) {
  const roundStart = Math.max(0, Math.min(1, evalParam(paramVal(params.classicalRoundStart, 0), p)));
  const tf = transitionAt(t, roundStart);

  if (tf <= 0) {
    return { x, y: rCirc };
  }

  // Target shape parameters (interpolated from circle toward target)
  const kTarget = Math.max(0.1, evalParam(paramVal(params.classicalAspectRatio, 1), p));
  const nTarget = Math.max(2, Math.min(20, evalParam(paramVal(params.classicalSqExponent, 6), p)));

  // Interpolate shape params: circle (k=1, n=2) → target
  const k = 1 + (kTarget - 1) * tf;
  const n = 2 + (nTarget - 2) * tf;

  // Area from circular profile (the horn law is area-preserving)
  const S = Math.PI * rCirc * rCirc;

  // Solve superellipse half-dimensions to match area
  const { a, b } = superellipseHalfDims(S, k, n);

  // Polar radius at azimuthal angle p
  const r = superellipseRadius(p, a, b, n);
  return { x, y: r };
}
