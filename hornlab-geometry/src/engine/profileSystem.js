/**
 * Multi-axis profile system — data model, family adapters, and evaluation.
 *
 * Allows each axis (H, V, or user-defined intermediate angles) to use a
 * different profile family (OSSE, R-OSSE, Classical) with independent
 * parameters.  Angular interpolation blends the evaluated radii (not the
 * raw parameters) between adjacent axes.
 *
 * Design decisions informed by critical review:
 *   - ProfileSystem is stored in params.profileSystem (not context)
 *   - Radii are blended at shared axial stations; profile.x values from
 *     individual families are NOT used in multi-axis mode
 *   - Extruded sweep deferred (r(phi,z) cannot represent true extrusion)
 *   - Python backend uses JS-computed point grids (no backend changes)
 *   - buildShrinkData disabled in multi-axis mode (OSSE-specific)
 */

import { evalParam, toRad } from '../common.js';
import { DEFAULTS } from './constants.js';
import {
  applyCrossSection,
  applyCrossSectionManualHV,
  applyCrossSectionRectFillet,
  sanitizeTimeline,
  ASPECT_RATIO_MODES,
} from './crossSection.js';
import { lerp } from './math.js';
import { prepareGeometryParams } from '../params.js';
import { calculateOSSE } from './profiles/osse.js';
import { calculateROSSE } from './profiles/rosse.js';
import { calculateClassical, getClassicalLength } from './profiles/classical.js';
import { CLASSICAL_SHAPES, MORPH_TARGETS } from './constants.js';
import { calculateLookup, getLookupLength } from './profiles/lookup.js';
import {
  migrateAxisToSections,
  evaluateAxisSections,
  getSectionsTotalLength,
  getSectionEffectiveLength,
  getParametricTail,
  resolveRoundoverSpec,
  sampleRoundoverMouth,
  SECTION_KINDS,
  MOUTH_TYPES,
} from './profileSections.js';

// ---------------------------------------------------------------------------
// Module-scoped caches — keyed by object identity, so JSON roundtrip
// (undo/redo, localStorage save/reload) yields a cache miss and forces
// recompilation.  This avoids stale data surviving serialization.
// ---------------------------------------------------------------------------
const _compiledParamsCache = new WeakMap();
const _profileLengthCache = new WeakMap();

const SECTION_LENGTH_PARAM_KEYS = new Set([
  'L',
  'R',
  'r0',
  'throatExtLength',
  'slotLength',
  'circArcRadius',
  'morphCorner',
  'morphWidth',
  'morphHeight',
  'gcurveWidth',
  'sourceRadius',
  'wallThickness',
  'verticalOffset',
  'radius',
  'throatRadius',
  'throatR',
  'fullLength',
]);

// ---------------------------------------------------------------------------
// Profile system modes
// ---------------------------------------------------------------------------
// `mode` is retained on the persisted shape for back-compat; only SINGLE is
// ever read.  Genuine multi-axis is detected via `axes.length > 1`, so the
// MULTI_AXIS enum value was never wired in.
export const PROFILE_MODES = Object.freeze({
  SINGLE: 'single'
});

export const INTERPOLATION_MODES = Object.freeze({
  SMOOTHSTEP: 'smoothstep',
  COSINE: 'cosine',
  LINEAR: 'linear'
});

// ---------------------------------------------------------------------------
// Smoothstep / interpolation helpers
// ---------------------------------------------------------------------------

function smoothstep(t) {
  return t * t * (3 - 2 * t);
}

// ---------------------------------------------------------------------------
// Per-axis superellipse exponent blending
// ---------------------------------------------------------------------------

/**
 * Blend two superellipse exponents in 1/n space.
 *
 * Why 1/n: perceptually, most of the "round → rectangle" transition happens
 * between n=2 and n≈6.  Linear interpolation of `n` collapses the visible
 * transition into the first 20 % of the slider range; linear interpolation
 * of `1/n` (standard superellipse-shape convention) is smoother.
 *
 * Why the internal clamps: a per-axis exponent that arrives as 0 via JSON
 * or expression input would make `1/n` infinite and silently produce a
 * zero-radius degenerate ring — not a thrown error.  The `Math.max(2, …)`
 * floor and `Math.min(20, …)` ceiling are load-bearing, not decorative.
 */
export function blendExponent(nA, nB, w) {
  const nAsafe = Math.max(2, Math.min(20, nA ?? 2));
  const nBsafe = Math.max(2, Math.min(20, nB ?? nAsafe));
  const invN = (1 - w) / nAsafe + w / nBsafe;
  return 1 / invN;
}

/**
 * Resolve a cross-section spec with the per-phi blended exponent baked in.
 *
 * Why the null-axisB branch: `findBracketingAxes` returns `axisB: null`
 * at single-axis configs and at the phi=0 / phi=π/2 boundaries of a
 * multi-axis config.  Dereferencing a null axis throws at every vertex
 * on those boundaries, so the guard is required — not a "nice to have".
 *
 * Axes without an explicit `exponent` inherit `cs.exponent` (which itself
 * falls back to 2), preserving bit-equal behavior for legacy configs.
 */
export function resolveAxisBlendedSpec(cs, axisA, axisB, weight) {
  const expA = axisA?.exponent ?? cs?.exponent ?? 2;
  if (!axisB) {
    return { ...cs, exponent: Math.max(2, Math.min(20, expA)) };
  }
  const expB = axisB.exponent ?? cs?.exponent ?? 2;
  return { ...cs, exponent: blendExponent(expA, expB, weight) };
}

// Session-scoped guard so the "timeline overrides per-axis exponent" notice
// fires once per load, not once per vertex.
let _timelinePerAxisWarned = false;

function cosineInterp(t) {
  return 0.5 * (1 - Math.cos(t * Math.PI));
}

function interpolate(t, mode) {
  if (mode === INTERPOLATION_MODES.LINEAR) return t;
  if (mode === INTERPOLATION_MODES.COSINE) return cosineInterp(t);
  return smoothstep(t);
}

// ---------------------------------------------------------------------------
// Fold angle to first quadrant [0, π/2]
// ---------------------------------------------------------------------------

function foldToFirstQuadrant(phi) {
  // NaN/Infinity → 0.  Without this guard, the modulo below produces NaN
  // and propagates silently into evaluateAxisRadius, returning NaN radii.
  if (!Number.isFinite(phi)) return 0;
  // Normalize to [0, 2π)
  let p = ((phi % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
  // Q1: [0, π/2]
  if (p <= Math.PI / 2) return p;
  // Q2: (π/2, π] → mirror around π/2
  if (p <= Math.PI) return Math.PI - p;
  // Q3: (π, 3π/2] → mirror around π
  if (p <= 3 * Math.PI / 2) return p - Math.PI;
  // Q4: (3π/2, 2π) → mirror around 3π/2
  return 2 * Math.PI - p;
}

// ---------------------------------------------------------------------------
// Family adapters — uniform interface for each profile family
// ---------------------------------------------------------------------------

/**
 * Get the horn length for a given family and its params.
 * All families evaluate at p=0 (H-axis) since each axis has its own params.
 * Accepts either raw or compiled params.
 */
export function getProfileLength(family, params) {
  switch (family) {
    case 'OSSE': {
      const L = evalParam(params.L, 0);
      const extLen = Math.max(0, evalParam(params.throatExtLength || 0, 0));
      const slotLen = Math.max(0, evalParam(params.slotLength || 0, 0));
      return L + extLen + slotLen;
    }
    case 'R-OSSE': {
      // R-OSSE is t-based; its length is implicit. Evaluate at t=1 to get
      // the mouth position, which gives us the axial length.
      const tmax = params.tmax === undefined ? DEFAULTS.TMAX : evalParam(params.tmax, 0);
      const profile = calculateROSSE(tmax, 0, params);
      return profile.x;
    }
    case 'Classical':
      return getClassicalLength(0, params);
    case 'LOOKUP':
      return getLookupLength(params);
    default:
      return evalParam(params.L || 200, 0);
  }
}

/**
 * Cached profile-length lookup.  The WeakMap is keyed by the compiled-params
 * object, so each per-axis parameter set computes its length at most once.
 * This eliminates the double-evaluation in the R-OSSE case (Bug 6) where
 * getProfileLength internally calls calculateROSSE and then evaluateFamilyRadius
 * calls it again.  Also benefits OSSE/Classical which call getProfileLength on
 * every vertex.
 */
function getCachedProfileLength(family, compiledParams) {
  let familyMap = _profileLengthCache.get(compiledParams);
  if (familyMap) {
    const cached = familyMap.get(family);
    if (cached !== undefined) return cached;
  } else {
    familyMap = new Map();
    _profileLengthCache.set(compiledParams, familyMap);
  }
  const len = getProfileLength(family, compiledParams);
  familyMap.set(family, len);
  return len;
}

/**
 * Prepare per-axis params by compiling expression strings into callable
 * functions.  Uses a module-scoped WeakMap keyed by `axis.params` so that:
 *   - Compiled results are never stored on the serializable state object
 *   - JSON roundtrip (undo/redo, save/reload) creates new param objects,
 *     causing a cache miss and automatic recompilation
 *   - No stale data survives serialization
 */
function getCompiledParams(axis) {
  if (_compiledParamsCache.has(axis.params)) {
    return _compiledParamsCache.get(axis.params);
  }
  const compiled = prepareGeometryParams(
    { ...axis.params, type: axis.family },
    { type: axis.family, applyVerticalOffset: false }
  );
  _compiledParamsCache.set(axis.params, compiled);
  return compiled;
}

function scaleLengthValue(value, scale) {
  if (value === undefined || value === null || value === '') return value;
  if (typeof value === 'function') {
    const scaled = (p) => scale * value(p);
    if (value._rawExpr !== undefined) scaled._rawExpr = `(${value._rawExpr}) * ${scale}`;
    return scaled;
  }
  if (typeof value === 'number' && Number.isFinite(value)) return value * scale;
  if (typeof value === 'string' && value.trim() !== '' && Number.isFinite(Number(value))) {
    return Number(value) * scale;
  }
  return value;
}

function scaleProfileSections(sections, scale) {
  if (!Array.isArray(sections) || scale === 1) return sections;
  return sections.map((section) => {
    const next = {
      ...section,
      length: scaleLengthValue(section.length, scale),
    };
    if (section.params && typeof section.params === 'object') {
      const params = { ...section.params };
      for (const key of SECTION_LENGTH_PARAM_KEYS) {
        if (params[key] !== undefined) params[key] = scaleLengthValue(params[key], scale);
      }
      next.params = params;
    }
    return next;
  });
}

/**
 * Evaluate a single axis's profile family at a given axial distance.
 * Returns the radius at that axial station.
 *
 * In multi-axis mode we only use the radius; the axial position is
 * determined by the master length, not by the individual family.
 *
 * NaN from a family evaluator is propagated rather than coerced to 0.
 * v2 uses this evaluator for legacy-migrated single-axis configs as well
 * as genuine multi-axis systems, and silently zeroing a malformed profile
 * would change geometry vs. the legacy path (which returned NaN and let it
 * flow through to the mesh).  Callers that need NaN-free output should
 * validate profile params upstream, not rely on this path to clean them.
 */
/**
 * Evaluate an axis's radius at axial position z.  Dispatches through the
 * sections chain when axis.sections is populated; otherwise falls back to
 * the direct family evaluator.  Factored out so Phase 3B can extend the
 * sections path without touching the hot family switch below.
 *
 * This is the primary radius seam used by evaluateMultiAxisProfile,
 * sampleAxisProfile, and computeCoverageAngle.
 */
function evaluateAxisRadius(axis, z, compiledParams, p = 0) {
  if (Array.isArray(axis.sections) && axis.sections.length > 0) {
    return evaluateAxisSections(axis, z, compiledParams, evaluateFamilyRadius, p);
  }
  return evaluateFamilyRadius(axis.family, z, compiledParams, p);
}

/**
 * Body exit tangent angle (radians from +z) for an axis, when its
 * sections chain contains a roundover mouth.  Returns `undefined` when
 * the axis has no roundover mouth, no body section, or the body is
 * degenerate — callers should treat that as "let resolveRoundoverSpec
 * fall back to π/2" (radial body).
 *
 * Used to feed getSectionsTotalLength / getParametricTail / coverage
 * computations so they measure the roundover's physical sweep, not the
 * radial-body approximation.
 */
function computeAxisBodyTangentAngle(axis, compiledParams, p = 0) {
  const sections = Array.isArray(axis?.sections) ? axis.sections : [];
  if (sections.length === 0) return undefined;
  const hasRoundoverMouth = sections.some(
    (s) => s.kind === SECTION_KINDS.MOUTH && s.type === MOUTH_TYPES.ROUNDOVER,
  );
  if (!hasRoundoverMouth) return undefined;
  const bodySection = sections.find((s) => s.kind === SECTION_KINDS.BODY) || null;
  if (!bodySection || !(bodySection.length > 0)) return undefined;
  // Walk pre-mouth sections to find the body's exit arc-length.  The body
  // usually sits directly after the throat; this keeps the tangent
  // correct even if the section order is unusual.
  let bodyExitArc = 0;
  for (const s of sections) {
    if (s.kind === SECTION_KINDS.MOUTH) break;
    bodyExitArc += getSectionEffectiveLength(s);
    if (s === bodySection) break;
  }
  const eps = Math.max(0.1, bodySection.length * 1e-3);
  const rEnd = evaluateAxisRadius(axis, bodyExitArc, compiledParams, p);
  const rBack = evaluateAxisRadius(
    axis, Math.max(0, bodyExitArc - eps), compiledParams, p,
  );
  if (!Number.isFinite(rEnd) || !Number.isFinite(rBack) || eps <= 0) return undefined;
  const slope = (rEnd - rBack) / eps;
  return Math.atan2(slope, 1);
}

/**
 * Parametric per-axis profile evaluator — returns { x, y } for an axis
 * at normalized arc-length parameter t in [0, 1].
 *
 * When the axis carries a mouth-roundover section, the tail of the
 * profile is a circular arc whose axial position (x) is NOT monotonic
 * in t — past the 90° peak, x decreases again while r continues to
 * rise, wrapping the mouth back toward the throat.  This evaluator
 * bridges the scalar section-chain (which dispatches radius lookups
 * at arc-length positions) into the parametric (x, y) representation
 * the mesh builder and 2D preview need to draw the wrap-back faithfully.
 *
 * For axes without a parametric tail (plain OSSE / Classical),
 * x = t · masterLength as before.  R-OSSE with a body-only chain
 * bypasses that linear approximation and returns its natural
 * parametric (x, y) via calculateROSSE — necessary when R-OSSE shares
 * a system with another axis that has a parametric tail (e.g. OSSE
 * with a mouth roundover), so its morning-glory wrap-back survives the
 * per-axis blend instead of collapsing to a monotonic x = t · L.
 */
export function evaluateAxisProfile(axis, t, compiledParams, p = 0) {
  // Resolve the tail without a tangent first — we only need the body
  // section + pre-mouth arc length to probe the body's exit slope.  Once
  // we have the real tangent we re-resolve the roundover spec below.
  const preTail = getParametricTail(axis);

  if (!preTail) {
    // R-OSSE natural parametric fast path.  calculateROSSE is a joint
    // (x, y)(t) with non-monotonic x past the rollover peak — the classic
    // R-OSSE signature.  When the axis has only a body section (no throat
    // extension, no explicit mouth), return that joint pair directly so
    // the wrap-back survives multi-axis blending with a parametric-tail
    // axis.  R-OSSE with a throat section keeps the generic section-aware
    // path below (which ramps the throat radius correctly at the cost of
    // losing the wrap-back — a known existing tradeoff for that edge case).
    const sections = Array.isArray(axis.sections) ? axis.sections : [];
    if (axis.family === 'R-OSSE' && sections.length <= 1) {
      const tmax = compiledParams.tmax === undefined
        ? DEFAULTS.TMAX
        : evalParam(compiledParams.tmax, p);
      const profile = calculateROSSE(t * tmax, p, compiledParams);
      return { x: profile.x, y: profile.y };
    }
    const total = sections.length > 0
      ? getSectionsTotalLength(axis.sections)
      : getCachedProfileLength(axis.family, compiledParams);
    const x = t * total;
    const y = evaluateAxisRadius(axis, x, compiledParams, p);
    return { x, y };
  }

  const { bodyArcLength, mouthSection } = preTail;

  // The roundover's physical sweep (and therefore its arc length) depends
  // on the body's exit tangent — we rotate that tangent into the user's
  // target end direction (π/2 + user_angle from +z).  Compute the body
  // tangent once here so the parametric t→s mapping uses the correct
  // total length, and so that the arc we sample in the tail agrees.
  // Use the same finite-difference eps as getBodyMouthInfo in
  // profileSections.js — 0.1 mm (or bodyLen * 1e-3) stays above float64
  // finite-difference noise at horn scales.
  const bodyExitR = evaluateAxisRadius(axis, bodyArcLength, compiledParams, p);
  const eps = Math.max(0.1, bodyArcLength * 1e-3);
  const bodyExitRBack = evaluateAxisRadius(
    axis, Math.max(0, bodyArcLength - eps), compiledParams, p,
  );
  const bodyMouthSlope = eps > 0 ? (bodyExitR - bodyExitRBack) / eps : 0;
  const bodyTangentAngle = Math.atan2(bodyMouthSlope, 1);
  const spec = resolveRoundoverSpec(mouthSection, bodyTangentAngle);
  const tailArcLength = spec ? spec.arcLength : 0;
  const total = bodyArcLength + tailArcLength;
  const s = t * total;

  if (s <= bodyArcLength + 1e-9) {
    const y = evaluateAxisRadius(axis, Math.min(s, bodyArcLength), compiledParams, p);
    return { x: s, y };
  }

  const arcS = s - bodyArcLength;
  const arcPoint = sampleRoundoverMouth(arcS, mouthSection, bodyExitR, bodyMouthSlope);
  return { x: bodyArcLength + arcPoint.x, y: arcPoint.r };
}

/**
 * Does this axis produce a naturally parametric (x, y) curve that
 * `evaluateAxisProfile` handles specially — i.e. has either an explicit
 * roundover mouth section OR is a body-only R-OSSE whose calculateROSSE
 * returns a non-monotonic x(t)?  Used by `evaluateMultiAxisProfile` to
 * decide whether the parametric-tail path should fire instead of the
 * shared-axial-station masterZ blend.
 */
function axisHasParametricProfile(axis) {
  if (!axis) return false;
  if (getParametricTail(axis)) return true;
  const sections = Array.isArray(axis.sections) ? axis.sections : [];
  if (axis.family === 'R-OSSE' && sections.length <= 1) return true;
  return false;
}

function evaluateFamilyRadius(family, z, compiledParams, p = 0) {
  switch (family) {
    case 'OSSE': {
      const L = getCachedProfileLength('OSSE', compiledParams);
      const clampedZ = Math.min(z, L);
      const profile = calculateOSSE(clampedZ, p, compiledParams, {});
      const h = compiledParams.h === undefined ? 0 : evalParam(compiledParams.h, p);
      let r = profile.y;
      if (h > 0 && L > 0) {
        r += h * Math.sin((clampedZ / L) * Math.PI);
      }
      return r;
    }
    case 'R-OSSE': {
      const L = getCachedProfileLength('R-OSSE', compiledParams);
      const t = L > 0 ? Math.min(1, z / L) : 0;
      const tmax = compiledParams.tmax === undefined ? DEFAULTS.TMAX : evalParam(compiledParams.tmax, p);
      const profile = calculateROSSE(t * tmax, p, compiledParams);
      return profile.y;
    }
    case 'Classical': {
      const L = getCachedProfileLength('Classical', compiledParams);
      const clampedZ = Math.min(z, L);
      // Tractrix's classical evaluator expects normalized t in [0, 1] rather
      // than raw z in mm (see calculateClassical's `options.normalized` path
      // and the tractrix shape branch).  Translate z → t here so tractrix
      // returns a real radius at the requested station instead of the throat
      // point (which made tractrix a constant-r0 cylinder in multi-axis mode
      // and caused computeCoverageAngle to return 0°).
      const rawShape = compiledParams.classicalShape;
      const shape = Number(
        rawShape != null && rawShape !== '' ? rawShape : CLASSICAL_SHAPES.CONICAL
      );
      if (shape === CLASSICAL_SHAPES.TRACTRIX) {
        const t = L > 0 ? clampedZ / L : 0;
        const profile = calculateClassical(t, p, compiledParams, { normalized: true });
        return profile.y;
      }
      const profile = calculateClassical(clampedZ, p, compiledParams);
      return profile.y;
    }
    case 'LOOKUP': {
      const L = getCachedProfileLength('LOOKUP', compiledParams);
      const clampedZ = Math.min(z, L);
      const profile = calculateLookup(clampedZ, p, compiledParams);
      return profile.y;
    }
    default:
      return 0;
  }
}

// ---------------------------------------------------------------------------
// Multi-axis angular weight computation
// ---------------------------------------------------------------------------

/**
 * Compute the interpolation weight between two axes for a given phi.
 *
 * angularSharpness controls how abruptly one axis hands off to the next in
 * the azimuthal direction.  It is NOT front-view corner rounding — that is
 * owned by crossSection.exponent (superellipse n) on the cross-section spec.
 * This parameter decides how much of the angular span between two axes each
 * one "claims" as a flat zone (weight pinned at 0 or 1) before blending:
 *   angularSharpness = 0: smooth interpolation across the full span
 *                         (axes blend evenly — minimal angular discontinuity)
 *   angularSharpness = 1: each axis claims its full half, no blending
 *                         (hard handoff at the midpoint between axes)
 *
 * Returns a weight in [0, 1] where 0 = fully axisA, 1 = fully axisB.
 */
function computeAxisWeight(phi, a1Rad, a2Rad, angularSharpness, interpMode) {
  const span = a2Rad - a1Rad;
  if (span <= 0) return 0;

  const t = (phi - a1Rad) / span;
  if (t <= 0) return 0;
  if (t >= 1) return 1;

  // angularSharpness controls the flat zone fraction
  const flatHalf = Math.max(0, Math.min(0.5, angularSharpness * 0.5));

  if (t <= flatHalf) return 0;           // fully axis A
  if (t >= 1 - flatHalf) return 1;       // fully axis B

  // Smooth transition in the remaining middle region
  const denom = 1 - 2 * flatHalf;
  if (denom < 1e-9) return 0.5;
  const blendT = (t - flatHalf) / denom;
  return interpolate(blendT, interpMode);
}

/**
 * Find the two bracketing axes for a given phi (already folded to Q1).
 * Returns { axisA, axisB, weight }.
 */
function findBracketingAxes(qPhi, axes, angularSharpness, interpMode) {
  if (axes.length === 1) {
    return { axisA: axes[0], axisB: null, weight: 0 };
  }

  // Before first axis: use first axis
  if (qPhi <= toRad(axes[0].angleDeg) + 1e-9) {
    return { axisA: axes[0], axisB: null, weight: 0 };
  }

  // After last axis: use last axis
  if (qPhi >= toRad(axes[axes.length - 1].angleDeg) - 1e-9) {
    return { axisA: axes[axes.length - 1], axisB: null, weight: 0 };
  }

  for (let i = 0; i < axes.length - 1; i++) {
    const a1 = toRad(axes[i].angleDeg);
    const a2 = toRad(axes[i + 1].angleDeg);
    if (qPhi >= a1 - 1e-9 && qPhi <= a2 + 1e-9) {
      const weight = computeAxisWeight(qPhi, a1, a2, angularSharpness, interpMode);
      return { axisA: axes[i], axisB: axes[i + 1], weight };
    }
  }

  // Should never reach here if axes are sorted
  return { axisA: axes[0], axisB: null, weight: 0 };
}

// ---------------------------------------------------------------------------
// Profile system construction and evaluation
// ---------------------------------------------------------------------------

/**
 * Create a default cross-section spec.
 *
 * Defaults produce geometry identical to a plain circular cross-section:
 *   exponent = 2          → superellipse reduces to a circle/ellipse
 *   aspectRatio = 1        → not an ellipse, a circle
 *   aspectRatioMode = 'natural' → don't apply a separate cross-section
 *                                 aspect ratio on top of the per-axis blend;
 *                                 when H and V axes have different coverage
 *                                 angles the blend naturally produces an
 *                                 ellipse without any manual stretch.
 *   roundStart = 0          → shape-shaping active from the throat (no-op at n=2)
 *
 * `aspectRatio` is only consulted when `aspectRatioMode === 'manual'`;
 * `applyCrossSection` treats the `natural` mode as aspectRatio = 1 in the
 * superellipse math so the per-axis radii do the stretching instead.
 */
export function defaultCrossSection() {
  return {
    aspectRatio: 1,
    aspectRatioMode: 'natural',
    exponent: 2,
    roundStart: 0
  };
}

// ---------------------------------------------------------------------------
// Legacy morph → cross-section timeline migration (Phase 4)
// ---------------------------------------------------------------------------

// Session-scoped notice guard so the migration banner logs once.
let _morphMigrationWarned = false;

const LEGACY_MORPH_KEYS = [
  'morphTarget', 'morphFixed', 'morphRate',
  'morphWidth', 'morphHeight', 'morphCorner',
  'morphAllowShrinkage'
];

/**
 * If top-level params describe a legacy morph (morphTarget != NONE and
 * morphRate > 0), produce a 2-keyframe cross-section timeline that
 * approximates the morph behavior and return it.  Otherwise return null.
 *
 * Keyframe 1 at t = morphFixed carries the base cross-section (the current
 * axis's profile shape — we interpolate the exponent channel only so the
 * radius is untouched before the morph begins).
 * Keyframe 2 at t = 1 carries the target-derived shape:
 *   RoundedRect → exponent ≈ 6, aspectRatio = morphWidth/morphHeight
 *   Circle      → exponent = 2, aspectRatio = 1
 *
 * Byte-identical parity with the old morph path is NOT required (per the
 * plan) — the timeline transition is PCHIP, not power-law, so the visual
 * result will differ slightly for morphRate != 1 configs.  A console notice
 * is emitted once per session to document the behavior change.
 */
export function migrateLegacyMorphToTimeline(params) {
  if (!params || typeof params !== 'object') return null;
  const target = Number(params.morphTarget || MORPH_TARGETS.NONE);
  if (target === MORPH_TARGETS.NONE) return null;
  const rate = Number(params.morphRate || 0);
  if (!Number.isFinite(rate) || rate <= 0) return null;

  const morphFixed = Math.max(0, Math.min(1, Number(params.morphFixed) || 0));
  const startT = Math.min(0.999, morphFixed);

  let targetExponent;
  let targetAspect = 1;
  if (target === MORPH_TARGETS.CIRCLE) {
    targetExponent = 2;
  } else {
    // RECTANGLE / rounded-rect
    targetExponent = 6;
    const w = Number(params.morphWidth) || 0;
    const h = Number(params.morphHeight) || 0;
    if (w > 0 && h > 0) {
      targetAspect = w / h;
    }
  }

  // Base keyframe carries the pre-morph state (circular cross-section,
  // aspectRatio = 1).  The target-end keyframe carries the desired shape.
  const timeline = [
    { t: startT, exponent: 2, aspectRatio: 1 },
    { t: 1, exponent: targetExponent, aspectRatio: targetAspect },
  ];

  if (!_morphMigrationWarned) {
    _morphMigrationWarned = true;
    try {
      console.info(
        '[ProfileSystem] Legacy Morph.TargetShape config migrated to a ' +
        '2-keyframe cross-section timeline. Visual result may differ slightly ' +
        'from the old power-law morph; adjust keyframes in the Shape Timeline ' +
        'panel to taste.'
      );
    } catch { /* noop if console is stubbed */ }
  }

  return timeline;
}

/**
 * Strip the legacy morph keys from a params object (non-destructive).
 * Used once the migration has produced a timeline, so the parameter state
 * downstream carries only the new shape.
 */
export function stripLegacyMorphKeys(params) {
  if (!params || typeof params !== 'object') return params;
  const out = { ...params };
  for (const k of LEGACY_MORPH_KEYS) delete out[k];
  return out;
}

/**
 * Migrate a legacy cross-section spec (pre-redesign, no aspectRatioMode) to
 * the current shape.  Rules:
 *   - If aspectRatioMode is already set, pass through untouched.
 *   - Otherwise infer from aspectRatio: exactly 1 → 'natural' (the pre-
 *     redesign default), any other value → 'manual' (the user deliberately
 *     set an aspect ratio).
 *
 * Exported so config-loading tests can exercise the migration directly.
 */
export function migrateCrossSection(cs) {
  if (!cs || typeof cs !== 'object') return defaultCrossSection();
  if (cs.aspectRatioMode === 'natural' || cs.aspectRatioMode === 'manual') {
    return cs;
  }
  const aspect = Number(cs.aspectRatio);
  const mode = (Number.isFinite(aspect) && Math.abs(aspect - 1) < 1e-9) ? 'natural' : 'manual';
  return { ...cs, aspectRatioMode: mode };
}

/**
 * Best-effort throat-circularity validator.
 *
 * Why best-effort: the throat is circular only when every axis's `r0`
 * agrees and no `r0` expression depends on phi.  Full symbolic dependency
 * analysis is out of scope for v1 — this samples `r0` at 8 equally-spaced
 * phi values per axis and flags disagreements.  Pathological expressions
 * (`r0 = '12.7 + sin(8*p)'` at multiples of π/4) evade detection; that's
 * documented in the plan as a known limitation.
 *
 * Returns an array of human-readable warning strings (empty array = OK).
 * UI surfaces warnings as soft "best-effort" advisories, not hard errors.
 */
export function validateThroatCircularity(system) {
  const warnings = [];
  if (!system || !Array.isArray(system.axes) || system.axes.length === 0) {
    return warnings;
  }
  const SAMPLES = 8;
  const axisR0s = system.axes.map((ax) => {
    const raw = ax?.params?.r0;
    const samples = [];
    for (let i = 0; i < SAMPLES; i++) {
      const phi = (i / SAMPLES) * Math.PI * 2;
      const v = Number(evalParam(raw, phi));
      samples.push(Number.isFinite(v) ? v : NaN);
    }
    const finite = samples.filter(Number.isFinite);
    const min = finite.length ? Math.min(...finite) : NaN;
    const max = finite.length ? Math.max(...finite) : NaN;
    return {
      axis: ax.id || `@${ax.angleDeg}°`,
      min,
      max,
      atZero: Number(evalParam(raw, 0)),
      phiDependent: Number.isFinite(min) && Number.isFinite(max) && (max - min) > 0.01,
    };
  });
  for (const a of axisR0s) {
    if (a.phiDependent) {
      warnings.push(
        `Axis ${a.axis}: r0 varies with phi (${a.min.toFixed(2)}..${a.max.toFixed(2)}mm) — throat will not be circular`
      );
    }
  }
  const zeros = axisR0s.map((a) => a.atZero).filter(Number.isFinite);
  if (zeros.length >= 2) {
    const r0Min = Math.min(...zeros);
    const r0Max = Math.max(...zeros);
    if (r0Max - r0Min > 0.1) {
      warnings.push(
        `Throat radius mismatch across axes: ${r0Min.toFixed(2)}..${r0Max.toFixed(2)}mm`
      );
    }
  }
  return warnings;
}

/**
 * Normalize a profile system from params.
 *
 * In v2 (Phase 1A of the profile-system redesign), this ALWAYS returns a
 * system — there is no legacy/null path.  Configs without a profileSystem
 * are migrated at resolve time to a one-axis system at angle 0° built from
 * the top-level profile params.
 *
 * A single axis at 0° with no cross-section shaping evaluates identically
 * to the pre-v2 legacy path, so existing configs keep their geometry.
 */
export function resolveProfileSystem(params) {
  let ps = params.profileSystem;
  const hasValidShape = ps
    && Array.isArray(ps.axes)
    && ps.axes.length >= 1
    && ps.axes.every(a => a && a.family && a.params && typeof a.angleDeg === 'number');

  // Top-level `scale` must be threaded into per-axis params so that length-
  // dimensioned per-axis fields (L, r0, morph*, etc.) receive the same
  // multiplier as the top-level.  `axis.params` is serializable UI state and
  // must not be mutated, so we rebuild each axis with a fresh params object
  // carrying the current scale below.
  const topLevelScaleRaw = params.scale ?? params.Scale ?? 1;
  const topLevelScaleNum = typeof topLevelScaleRaw === 'number'
    ? topLevelScaleRaw
    : Number(topLevelScaleRaw);
  const topLevelScale = Number.isFinite(topLevelScaleNum) ? topLevelScaleNum : 1;

  if (!hasValidShape) {
    // Migrate legacy params → one-axis system at 0°.  The top-level params
    // have usually already been run through prepareGeometryParams by the
    // caller (buildWaveguideMesh receives prepared params from the pipeline),
    // so we copy them as-is and force scale=1 on the axis to prevent
    // double-scaling when getCompiledParams re-runs prepareGeometryParams.
    const family = params.type || 'OSSE';
    const crossSection = defaultCrossSection();
    // Legacy morph keys (morphTarget, morphCorner, etc.) are preserved on
    // axis.params so applyMorphing / buildShrinkData keep running byte-
    // identically.  Auto-migrating them into a cross-section timeline is
    // lossy for rounded-rect corners (no cornerRadius channel on the
    // timeline yet), so the Shape Timeline is opt-in via the UI only.
    ps = {
      mode: PROFILE_MODES.SINGLE,
      axes: [
        {
          id: 'base',
          angleDeg: 0,
          family,
          params: { ...params, scale: 1 }
        }
      ],
      crossSection,
      angularSharpness: 0,
      interpolation: INTERPOLATION_MODES.SMOOTHSTEP
    };
  } else {
    // Back-compat migration: earlier versions called this field `straightness`.
    // Rename in place so downstream code only has to read `angularSharpness`.
    // The old key is dropped after the copy so it doesn't survive serialization.
    if (ps.angularSharpness === undefined && ps.straightness !== undefined) {
      ps = { ...ps, angularSharpness: ps.straightness };
      delete ps.straightness;
    } else if (ps.straightness !== undefined) {
      // Both present — prefer the new name, drop the legacy one.
      ps = { ...ps };
      delete ps.straightness;
    }
    // Cross-section redesign migration: pre-redesign configs have
    // `aspectRatio` but no `aspectRatioMode`.  Infer the mode from the
    // value (1 → natural, other → manual) so the new UI renders sensibly.
    if (ps.crossSection && ps.crossSection.aspectRatioMode === undefined) {
      ps = { ...ps, crossSection: migrateCrossSection(ps.crossSection) };
    }
    // Phase 4: sanitize cross-section timeline if present so the evaluator
    // can trust it (sorted, dedup'd, clamped to [0, 1], >= 2 keyframes).
    // If the timeline becomes a single valid keyframe, we keep it on the
    // spec — applyCrossSection falls through to the scalar spec in that
    // case, so there's no behavior change.
    if (ps.crossSection && ps.crossSection.timeline !== undefined) {
      const clean = sanitizeTimeline(ps.crossSection.timeline);
      if (clean) {
        ps = { ...ps, crossSection: { ...ps.crossSection, timeline: clean } };
      } else {
        // Invalid timeline array → drop the key entirely.
        const { timeline: _drop, ...csRest } = ps.crossSection;
        ps = { ...ps, crossSection: csRest };
      }
    }

    // Scale regression fix: genuine multi-axis configs (a profileSystem
    // serialized by the UI) do not carry a `scale` on each axis.params.  The
    // top-level scale only gets applied to the top-level `L`, `r0`, etc. in
    // prepareGeometryParams, which means axis.params stays at its raw
    // (un-scaled) values and both (a) the section migration below and (b)
    // the per-axis family evaluator end up producing geometry at the raw
    // scale, ignoring the user's Scale setting.
    //
    // Fix: rebuild each axis with its `.params` run through the SAME
    // prepareGeometryParams call that the top-level params went through,
    // with top-level `scale` threaded in.  That applies the scale multiplier
    // to per-axis L / r0 / morph* / gcurve* / etc. in a single canonical
    // place so downstream migrateAxisToSections (which reads numeric L/
    // throatExtLength/etc. directly off axis.params) also sees scaled
    // lengths.  The resulting params carry `scale: 1` so the downstream
    // getCompiledParams invocation does not re-apply the scale multiplier
    // (double-scaling).  The original user-facing axis.params is not
    // mutated — resolveProfileSystem returns a new system whose axes are
    // scaled clones, and the UI reads its own un-scaled params from state.
    //
    // Done inside the `else` branch only: the legacy single-axis migration
    // branch already builds axis.params from the already-prepared top-level
    // params with `scale: 1`, which preserves byte-identical geometry on
    // the legacy path.
    if (topLevelScale !== 1) {
      const scaledAxes = ps.axes.map((axis) => {
        const scaledParams = prepareGeometryParams(
          { ...axis.params, type: axis.family, scale: topLevelScale },
          { type: axis.family, applyVerticalOffset: false },
        );
        // Neutralize scale so getCompiledParams does not multiply again.
        scaledParams.scale = 1;
        return {
          ...axis,
          params: scaledParams,
          sections: scaleProfileSections(axis.sections, topLevelScale),
        };
      });
      ps = { ...ps, axes: scaledAxes };
    }
  }

  // Per-axis-cross-section migrations (idempotent, run on every resolve).
  //
  // Why the two migrations live here (post if/else, shared): the legacy
  // single-axis branch above synthesizes a fresh defaultCrossSection()
  // carrying `roundStart: 0`, and the multi-axis branch may receive a spec
  // with `roundStart` set by a pre-redesign save.  Running the migration
  // once here covers both branches without duplication.
  if (
    ps.crossSection
    && ps.crossSection.roundStart !== undefined
    && ps.crossSection.transitionStart === undefined
  ) {
    // Why `shape: 'linear'`: legacy `roundStart` was a linear ramp to t=1.
    // Smoothstep would silently change vertex coordinates for every existing
    // config.  The linear marker keeps migrated geometry byte-identical;
    // newly-authored specs default to 'smooth' in the cross-section panel.
    const { roundStart, ...csRest } = ps.crossSection;
    ps = {
      ...ps,
      crossSection: {
        ...csRest,
        transitionStart: roundStart,
        transitionEnd: 1,
        shape: 'linear',
      },
    };
  }
  // Copy `cs.exponent` onto any axis that lacks an explicit exponent.  The
  // `resolveAxisBlendedSpec` fallback already resolves to `cs.exponent` for
  // undefined axis exponents, so this migration is additive and bit-equal.
  // It exists so that when the Phase B UI drops the global exponent slider
  // and promotes per-axis sliders, single-axis legacy configs show their
  // old global value on the one axis instead of the default 2.
  if (ps.crossSection?.exponent !== undefined) {
    const globalExp = ps.crossSection.exponent;
    let changed = false;
    const updatedAxes = ps.axes.map((axis) => {
      if (axis.exponent !== undefined) return axis;
      changed = true;
      return { ...axis, exponent: globalExp };
    });
    if (changed) ps = { ...ps, axes: updatedAxes };
  }

  // Defensive sort + dedup by angleDeg.  The UI always sorts on add/edit,
  // but tests and programmatic config import can bypass the UI; without this
  // pass findBracketingAxes would silently fall through on an unsorted list
  // and a duplicate-angle axis would be unreachable from the bracketing scan.
  if (ps.axes.length > 1) {
    const sorted = [...ps.axes].sort((a, b) => a.angleDeg - b.angleDeg);
    const dedup = [];
    for (const ax of sorted) {
      const prev = dedup[dedup.length - 1];
      if (!prev || Math.abs(prev.angleDeg - ax.angleDeg) > 1e-6) dedup.push(ax);
    }
    if (dedup.length !== ps.axes.length || sorted.some((a, i) => a !== ps.axes[i])) {
      ps = { ...ps, axes: dedup };
    }
  }

  // Compile expression strings in per-axis params (cached per axis via WeakMap).
  // This is essential: raw params from the UI contain expression strings
  // like "45 - 5*cos(2*p)^5" that must be compiled into callable functions
  // before profile evaluators can use them.
  for (const axis of ps.axes) {
    getCompiledParams(axis);
  }

  // Phase 3A: eagerly migrate each axis's legacy flat throat/flare params
  // into a canonical sections array.  Byte-identical geometry is preserved
  // because:
  //   - migrateAxisToSections does not mutate axis.params (legacy keys stay
  //     on the body section's params and the family evaluator still reads
  //     them internally).
  //   - The sections path dispatches back to the same family evaluator at
  //     the same global z, so per-vertex radii are identical.
  // See src/geometry/engine/profileSections.js for the contract.
  // migrateAxisToSections reads axis.params directly (via evalOr and, for
  // R-OSSE, calculateROSSE's internal validator). Legacy configs that import
  // with expression strings on params like `k` / `R` / `a` would otherwise
  // produce a zero-length body — evalParam returns the raw string, Number.
  // isFinite rejects it, and the family length falls back to 0. Pass the
  // compiled view so those expressions resolve to callable functions here;
  // persisted axis.params is untouched because the returned axis carries its
  // original params below.
  const migratedAxes = ps.axes.map((axis) => {
    if (Array.isArray(axis.sections) && axis.sections.length > 0) return axis;
    const compiled = getCompiledParams(axis);
    const migrated = migrateAxisToSections({ ...axis, params: compiled });
    // Restore the raw axis.params on the outer axis so serialization / UI
    // reads (which walk state.params.profileSystem) still see the original
    // expression strings. The inner sections keep the compiled params they
    // were built with — sections are ephemeral per render and aren't saved.
    return { ...migrated, params: axis.params };
  });
  // Only rebuild the system wrapper when at least one axis changed — this
  // keeps the returned object stable for callers that rely on identity.
  const anyMigrated = migratedAxes.some((axis, i) => axis !== ps.axes[i]);
  if (anyMigrated) ps = { ...ps, axes: migratedAxes };

  return ps;
}

/**
 * Compute the master axial length for a profile system.
 * This is the maximum length across all axes, recomputed each time
 * (O(n_axes) -- typically 2-4 calls, so cheap enough to skip caching).
 */
function computeMasterLength(system) {
  let maxLen = 0;
  for (const axis of system.axes) {
    // Prefer the sections-based total when available — in Phase 3A it is
    // numerically equal to getProfileLength(family, params) by construction,
    // but downstream work (Phase 3B) may diverge once sections carry their
    // own lengths independent of the flat params.
    let len;
    if (Array.isArray(axis.sections) && axis.sections.length > 0) {
      const compiled = getCompiledParams(axis);
      const bodyTangentAngle = computeAxisBodyTangentAngle(axis, compiled);
      len = getSectionsTotalLength(axis.sections, bodyTangentAngle);
      // Guard: zero or missing length falls back to the family computation
      // so partially-constructed axes (e.g. in tests) still produce a mesh.
      if (!Number.isFinite(len) || len <= 0) {
        len = getProfileLength(axis.family, compiled);
      }
    } else {
      const compiled = getCompiledParams(axis);
      len = getProfileLength(axis.family, compiled);
    }
    if (Number.isFinite(len) && len > maxLen) maxLen = len;
  }
  return maxLen || 200; // fallback
}

/**
 * Evaluate the multi-axis profile system at normalized position t and angle p.
 *
 * Returns { x, y } to match the existing profile contract:
 *   x = axial position (mm)
 *   y = radius (mm)
 */
export function evaluateMultiAxisProfile(t, p, system) {
  // ──────────────────────────────────────────────────────────────────────
  // Single-axis R-OSSE fast path — forward-port of the pre-multi-axis R-OSSE
  // build contract.
  //
  // R-OSSE is an azimuthally-varying profile: `R(p)`, `a(p)`, `k(p)`, `r(p)`,
  // `m(p)`, `b(p)`, `q(p)` are all expected to vary with phi.  The classical
  // R-OSSE math couples axial position `x(t, p)` and radius `y(t, p)` as a
  // joint function of t, so each azimuth has its own axial curve — the mouth
  // "ring" is intentionally NOT flat in z.  ATH's reference .geo shows this
  // directly: slices past the throat have H-plane z ≠ V-plane z.
  //
  // Pre-multi-axis `computeRosseProfileAt` (mesh/horn.js in f09ae29) used the
  // joint `{x, y}` from `calculateROSSE(t*tmax, p, params)` directly.  The
  // generic multi-axis blender below loses that coupling by pinning axial
  // position to `masterZ = t * masterLength(p=0)`, which flattens the mouth
  // ring and produces the visibly-wrong R-OSSE geometry reported after the
  // multi-axis merge (commits 8b328e9, 4199184, debd79e).
  //
  // We bypass the blender for the degenerate "single-axis R-OSSE" case
  // (which is what every legacy R-OSSE config auto-migrates to in v2).  For
  // a genuine multi-axis config that uses R-OSSE on one axis and a different
  // family on another, the shared-axial-station blend below still applies —
  // that is an explicit user choice to mix profiles along phi.
  const axes = system.axes;
  const allRosseNoTail = axes.every(a =>
    a.family === 'R-OSSE'
    && (!Array.isArray(a.sections) || a.sections.length <= 1)
  );

  // ──────────────────────────────────────────────────────────────────────
  // manualHV top-level dispatch.
  //
  // In manualHV mode the per-axis radii are taken as LITERAL half-width and
  // half-height of the cross-section superellipse, with no area-preserving
  // rescale.  That makes the regular bracketing-and-blending pipeline
  // irrelevant: at every phi the H half-dim comes from axes[0] and the V
  // half-dim comes from axes[1], full stop.  We bypass bracketing entirely
  // so the cardinal directions (phi=0 and phi=π/2) — where bracketing
  // returns axisB=null — also get the literal-HV treatment instead of
  // falling through to the natural-mode area-preserving rescale.
  //
  // Requires at least two axes; degrades to the regular path otherwise.
  // For axes.length > 2 we still take axes[0] as H and axes[1] as V — the
  // manualHV semantics are explicitly two-axis (H and V), so users who add
  // intermediate axes opt into the regular blend by choice of mode.
  const csSpec = system.crossSection;
  if (csSpec && axes.length >= 2 && (
    csSpec.aspectRatioMode === ASPECT_RATIO_MODES.MANUAL_HV ||
    csSpec.aspectRatioMode === ASPECT_RATIO_MODES.RECT_FILLET
  )) {
    const masterLength = computeMasterLength(system);
    const masterZ = t * masterLength;
    const axisH = axes[0];
    const axisV = axes[1];
    const rH = evaluateAxisRadius(axisH, masterZ, getCompiledParams(axisH), p);
    const rV = evaluateAxisRadius(axisV, masterZ, getCompiledParams(axisV), p);
    const r = csSpec.aspectRatioMode === ASPECT_RATIO_MODES.RECT_FILLET
      ? applyCrossSectionRectFillet(rH, rV, t, p, csSpec)
      : applyCrossSectionManualHV(rH, rV, t, p, csSpec);
    return { x: masterZ, y: r };
  }

  // 1. Fold phi into first quadrant
  const qPhi = foldToFirstQuadrant(p);

  // 2. Find bracketing axes and compute weight
  const angularSharpness = system.angularSharpness || 0;
  const interpMode = system.interpolation || INTERPOLATION_MODES.SMOOTHSTEP;
  const { axisA, axisB, weight } = findBracketingAxes(
    qPhi, system.axes, angularSharpness, interpMode
  );

  // Per-axis exponent fast-path gate.  When no axis carries an explicit
  // exponent, and no timeline is active, we pass the raw cs spec straight
  // through — byte-identical to pre-per-axis-exponent output.  Timeline
  // precedence: a timeline with >= 2 keyframes is a global schedule and
  // overrides per-axis exponents entirely (warn once if they conflict).
  const cs = system.crossSection;
  const hasPerAxisExp = axes.some(ax => ax.exponent !== undefined);
  const timelineActive = Array.isArray(cs?.timeline) && cs.timeline.length >= 2;
  if (timelineActive && hasPerAxisExp && !_timelinePerAxisWarned) {
    _timelinePerAxisWarned = true;
    try {
      console.warn(
        '[ProfileSystem] crossSection.timeline is active; per-axis exponent ' +
        'values are ignored in favor of the timeline schedule.'
      );
    } catch { /* noop if console is stubbed */ }
  }

  // R-OSSE fast path — extends the single-axis fix in commit acc2d4d to
  // multi-axis configs.  R-OSSE's `calculateROSSE(t, p, params)` returns a
  // joint `{x, y}` whose axial position is non-monotonic in t (the mouth
  // wraps back over itself past the rollover peak — the morning-glory
  // shape).  The masterZ blender below pins `x = t · masterLength`, which
  // collapses that wrap-back into a forward-only flare and reads, visually,
  // as the mouth roundover bending the wrong way.
  //
  // For the legacy-equivalent shape (every axis is plain R-OSSE with no
  // user-attached roundover section), evaluate calculateROSSE per axis and
  // blend the natural (x, r) the way the parametric-tail path does — x by
  // linear interpolation and r area-proportionally (sqrt of lerp(r²)).
  // Genuine multi-axis configs that mix R-OSSE with another family, or
  // attach an explicit roundover section, fall through to the existing
  // parametric-tail / masterZ paths.
  if (allRosseNoTail) {
    const compA = getCompiledParams(axisA);
    const tmaxA = compA.tmax === undefined ? DEFAULTS.TMAX : evalParam(compA.tmax, p);
    const profA = calculateROSSE(t * tmaxA, p, compA);
    let xOut;
    let rRaw;
    if (axisB) {
      const compB = getCompiledParams(axisB);
      const tmaxB = compB.tmax === undefined ? DEFAULTS.TMAX : evalParam(compB.tmax, p);
      const profB = calculateROSSE(t * tmaxB, p, compB);
      xOut = lerp(profA.x, profB.x, weight);
      rRaw = Math.sqrt(lerp(profA.y * profA.y, profB.y * profB.y, weight));
    } else {
      xOut = profA.x;
      rRaw = profA.y;
    }
    let r = rRaw;
    if (cs) {
      const effectiveSpec = (hasPerAxisExp && !timelineActive)
        ? resolveAxisBlendedSpec(cs, axisA, axisB, weight)
        : cs;
      r = applyCrossSection(r, t, p, effectiveSpec);
    }
    return { x: xOut, y: r };
  }

  // Parametric fast path — forwards axes whose tail is a circular-arc
  // roundover (OSSE, Classical) through evaluateAxisProfile so the mesh
  // builder and 2D preview see the true (x, r) past the peak, not the
  // monotonic-z approximation the scalar section chain produces.
  //
  // This also fires for genuine multi-axis systems as long as at least one
  // of the bracketing axes has a parametric tail: each axis's own (x, y) is
  // evaluated parametrically, then both x and r² are angularly blended.
  // Non-tail axes fall through evaluateAxisProfile's non-tail branch
  // (x = t · axis-length), so the multi-axis case with a roundover on
  // only one axis still blends coherently.
  //
  // R-OSSE's natural mouth wrap-back (morning-glory) is also a parametric
  // profile even without an explicit roundover section — include body-only
  // R-OSSE axes in the trigger so mixed systems (e.g., R-OSSE on H +
  // OSSE+roundover on V) route R-OSSE through evaluateAxisProfile's
  // calculateROSSE fast path, preserving the wrap-back in the per-axis
  // blend AND at phi boundaries where bracketing reduces to a single axis
  // (axisB == null).  Without this trigger the fallback masterZ path would
  // clamp R-OSSE to t * masterLength and flatten its mouth curl.
  const parametricA = axisHasParametricProfile(axisA);
  const parametricB = axisB ? axisHasParametricProfile(axisB) : false;
  if (parametricA || parametricB) {
    const profA = evaluateAxisProfile(axisA, t, getCompiledParams(axisA), p);
    const profB = axisB
      ? evaluateAxisProfile(axisB, t, getCompiledParams(axisB), p)
      : profA;
    const x = axisB ? lerp(profA.x, profB.x, weight) : profA.x;
    let r = axisB
      ? Math.sqrt(lerp(profA.y * profA.y, profB.y * profB.y, weight))
      : profA.y;
    if (cs) {
      const effectiveSpec = (hasPerAxisExp && !timelineActive)
        ? resolveAxisBlendedSpec(cs, axisA, axisB, weight)
        : cs;
      r = applyCrossSection(r, t, p, effectiveSpec);
    }
    return { x, y: r };
  }

  const masterLength = computeMasterLength(system);
  const masterZ = t * masterLength;

  // 3. Evaluate each axis at the shared axial station.  evaluateAxisRadius
  // dispatches through the sections chain when axis.sections is populated.
  // Phi is threaded through so per-phi expressions (e.g., a = "45 - 5*cos(2*p)^5")
  // evaluate with the real azimuthal angle, not the hardcoded p=0 that the
  // pre-fix path used.
  const rA = evaluateAxisRadius(axisA, masterZ, getCompiledParams(axisA), p);
  const rB = axisB ? evaluateAxisRadius(axisB, masterZ, getCompiledParams(axisB), p) : rA;

  // 4. Blend radii (area-proportional: blend r² to conserve cross-sectional area)
  let r = Math.sqrt(lerp(rA * rA, rB * rB, weight));

  // 5. Apply cross-section shaping.  applyCrossSection short-circuits to rBase
  // when the spec reduces to a circle (exponent = 2, aspectRatio = 1), so no
  // mode flag is needed — the shape parameters are the single source of truth.
  if (cs) {
    const effectiveSpec = (hasPerAxisExp && !timelineActive)
      ? resolveAxisBlendedSpec(cs, axisA, axisB, weight)
      : cs;
    r = applyCrossSection(r, t, p, effectiveSpec);
  }

  return { x: masterZ, y: r };
}

/**
 * Create a profile system for the UI default state: a single symmetric axis
 * at 0° carrying the incoming family and params as-is.  This preserves the
 * user's authored coverage (e.g. `a` / `classicalCoverageAngle`, including
 * expression-valued `a` like `"48.5 - 5.6*cos(2*p)^5"`) verbatim rather than
 * silently stamping an asymmetric H=90°/V=60° default on top.  Users who
 * want asymmetric coverage click "+ Add Axis" and set the V-axis coverage
 * explicitly.
 *
 * `mode` is kept on the returned object for back-compat with any
 * consumers that still inspect it (morph/enclosure gates inspect
 * `axes.length` now, but the field remains in the shape).
 */
export function createDefaultProfileSystem(family, params) {
  return {
    mode: PROFILE_MODES.SINGLE,
    axes: [
      { id: 'base', angleDeg: 0, family, params: { ...params } },
    ],
    crossSection: defaultCrossSection(),
    angularSharpness: 0,
    interpolation: INTERPOLATION_MODES.SMOOTHSTEP
  };
}

/**
 * Sample the mouth curve r(phi) at t=1 for a profile system.
 * Returns an array of { phi, r } points.
 */
export function sampleMouthCurve(system, numSamples = 360) {
  const points = [];
  for (let i = 0; i < numSamples; i++) {
    const phi = (i / numSamples) * Math.PI * 2;
    const profile = evaluateMultiAxisProfile(1, phi, system);
    points.push({ phi, r: profile.y });
  }
  return points;
}

/**
 * Sample a single axis's profile curve for 2D visualization.
 * Returns an array of { z, r } points.
 *
 * Sampling is parametric in t ∈ [0, 1] so that profiles whose x(t) is
 * non-monotonic (R-OSSE with mouth curl-back, OSSE with a roundover
 * mouth that sweeps past 90°) render their wrap-back faithfully — the
 * returned (z, r) comes directly from the axis's own parametric (x, y),
 * not from a uniform-z sampler that collapses multi-valued profiles to
 * the monotonic-z branch.
 *
 * For monotonic axes (plain OSSE / Classical / R-OSSE with tmax ≤ 1)
 * the returned samples are numerically identical to the old uniform-z
 * path (z = t · L).
 */
export function sampleAxisProfile(axisNode, numSamples = 200) {
  const compiled = getCompiledParams(axisNode);

  // Build an ad-hoc single-axis system so we can route through the
  // parametric evaluator (R-OSSE fast path + parametric-tail path).
  // The cross-section is pinned to the default (circular) since the
  // 2D view plots the axis's own profile, not the blended mouth.
  const system = {
    mode: PROFILE_MODES.SINGLE,
    axes: [axisNode],
    crossSection: defaultCrossSection(),
    angularSharpness: 0,
    interpolation: INTERPOLATION_MODES.SMOOTHSTEP,
  };

  const points = [];
  for (let i = 0; i <= numSamples; i++) {
    const t = i / numSamples;
    const profile = evaluateMultiAxisProfile(t, 0, system);
    points.push({ z: profile.x, r: profile.y });
  }
  return points;
}

/**
 * Compute a nominal coverage angle (full included angle, degrees) for an
 * axis by finite-differencing the profile near the mouth.  Used by the
 * multi-axis panel to display a derived coverage angle for families that
 * don't natively take one as a parameter (Hyperbolic, Exponential,
 * Catenoidal, Bessel, Tractrix).
 *
 * The math: compute r at t = 0.99 and t = 1.0, take dr/dz, then
 * halfAngle = atan(dr/dz), coverage = 2 * halfAngle.
 *
 * `axis` should be a { family, params } shape (or a full axis node with
 * angleDeg / id — those are ignored).  Raw or compiled params both work;
 * raw params get compiled by the caller's usual path.
 *
 * Returns NaN when the profile can't be evaluated (zero length, invalid
 * params).  Callers should render "—" in that case.
 */
export function computeCoverageAngle(axis) {
  if (!axis || !axis.family || !axis.params) return NaN;

  // Reuse the compiled-params cache when available, otherwise compile here.
  // Using a temporary cache key (the params object) is fine — WeakMap keeps
  // it off the serializable state.
  const compiled = _compiledParamsCache.has(axis.params)
    ? _compiledParamsCache.get(axis.params)
    : prepareGeometryParams(
        { ...axis.params, type: axis.family },
        { type: axis.family, applyVerticalOffset: false }
      );

  // Use the sections-aware length when available so migrated axes measure
  // coverage at the actual mouth of the chain, not just the body's L.
  let L;
  if (Array.isArray(axis.sections) && axis.sections.length > 0) {
    const bodyTangentAngle = computeAxisBodyTangentAngle(axis, compiled);
    L = getSectionsTotalLength(axis.sections, bodyTangentAngle);
    if (!Number.isFinite(L) || L <= 0) L = getProfileLength(axis.family, compiled);
  } else {
    L = getProfileLength(axis.family, compiled);
  }
  if (!Number.isFinite(L) || L <= 0) return NaN;

  const z1 = 0.99 * L;
  const z2 = L;
  const r1 = evaluateAxisRadius(axis, z1, compiled);
  const r2 = evaluateAxisRadius(axis, z2, compiled);
  if (!Number.isFinite(r1) || !Number.isFinite(r2)) return NaN;

  const dz = z2 - z1;
  if (dz <= 0) return NaN;
  const dr = r2 - r1;
  const halfAngleRad = Math.atan2(dr, dz);
  return (halfAngleRad * 360) / Math.PI; // 2 * halfAngle in degrees
}
