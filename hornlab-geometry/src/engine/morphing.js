/**
 * Legacy morph stub (retired in Phase 4 of the profile-system redesign).
 *
 * The "morph to rounded rectangle at the mouth" feature has been superseded
 * by the cross-section timeline on `profileSystem.crossSection.timeline`.
 * `resolveProfileSystem` migrates legacy configs with `morphTarget` /
 * `morphFixed` / `morphRate` into a 2-keyframe timeline and strips those
 * keys from the axis params, so new builds never see a non-None morphTarget.
 *
 * This file is retained as a stub so that pre-Phase-4 call sites
 * (mesh/horn.js, mesh/enclosure.js) keep linking without behavior changes.
 * `applyMorphing` is now an identity function in the common case; the
 * rounded-rect radius helper is still exported because the OSSE-specific
 * coverage-angle shrinkage math (`buildShrinkData`) reads morphWidth /
 * morphHeight / morphCorner directly from legacy params to compute mouth
 * target extents — that path is unchanged.
 *
 * If this file is imported in application code, a one-shot deprecation
 * warning is emitted (unless NODE_ENV === 'test' to keep CI output clean).
 */

import { MORPH_TARGETS } from './constants.js';
import { lerp } from './math.js';

let _deprecationWarned = false;
function warnDeprecated() {
  if (_deprecationWarned) return;
  _deprecationWarned = true;
  if (typeof process !== 'undefined' && process.env && process.env.NODE_ENV === 'test') return;
  try {
    console.warn(
      '[morphing.js] applyMorphing is deprecated. Use crossSection.timeline on the profile system instead.'
    );
  } catch { /* noop */ }
}

export function getRoundedRectRadius(p, halfWidth, halfHeight, cornerRadius) {
  const absCos = Math.abs(Math.cos(p));
  const absSin = Math.abs(Math.sin(p));

  if (absCos < 1e-9) return halfHeight;
  if (absSin < 1e-9) return halfWidth;

  const r = Math.max(0, Math.min(cornerRadius, Math.min(halfWidth, halfHeight)));
  if (r <= 1e-9) {
    return Math.min(halfWidth / absCos, halfHeight / absSin);
  }

  const yAtX = (halfWidth * absSin) / absCos;
  if (yAtX <= halfHeight - r + 1e-9) return halfWidth / absCos;

  const xAtY = (halfHeight * absCos) / absSin;
  if (xAtY <= halfWidth - r + 1e-9) return halfHeight / absSin;

  const cx = halfWidth - r;
  const cy = halfHeight - r;
  const A = absCos ** 2 + absSin ** 2;
  const B = -2 * (absCos * cx + absSin * cy);
  const C = cx ** 2 + cy ** 2 - r ** 2;
  const disc = Math.max(0, B ** 2 - 4 * A * C);

  return (-B + Math.sqrt(disc)) / (2 * A);
}

function getMorphTargetRadius(p, targetShape, halfWidth, halfHeight, cornerRadius) {
  if (targetShape === MORPH_TARGETS.CIRCLE) {
    return Math.sqrt(Math.max(0, halfWidth * halfHeight));
  }
  return getRoundedRectRadius(p, halfWidth, halfHeight, cornerRadius);
}

/**
 * Legacy morph entry point — now a near-identity for any config that has
 * been through `resolveProfileSystem` (Phase 4 migration strips morphTarget).
 *
 * Kept as a thin shim because mesh/horn.js still calls it from multiple hot
 * loops.  In a raw legacy params object (no profile system), it still does
 * the old power-law morph so pre-migration call paths (scripts / tests that
 * bypass resolveProfileSystem) don't silently break.
 *
 * The `axes.length > 1` multi-axis disable gate from the pre-Phase-4
 * implementation has been REMOVED — multi-axis systems now express shape
 * transitions via the cross-section timeline, and the stub returns early
 * on morphTarget === NONE anyway.
 */
export function applyMorphing(currentR, t, p, params, morphTargetInfo = null) {
  const targetShape = Number(params.morphTarget || MORPH_TARGETS.NONE);
  if (targetShape === MORPH_TARGETS.NONE) return currentR;

  // Skip morph when Classical square cross-section is active to avoid
  // double-transforming the cross-section (square transition + morph).
  if (params.type === 'Classical' && Number(params.classicalCrossSection) === 1) return currentR;

  warnDeprecated();

  const morphStart = Number(params.morphFixed || 0);
  if (t <= morphStart) return currentR;

  const rate = Number(params.morphRate || 3);
  const morphFactor = Math.pow((t - morphStart) / Math.max(1e-9, 1 - morphStart), rate);

  const hasExplicit = (params.morphWidth > 0) || (params.morphHeight > 0);
  const halfWidth = params.morphWidth > 0 ? params.morphWidth / 2 : (morphTargetInfo?.halfW ?? currentR);
  const halfHeight = params.morphHeight > 0 ? params.morphHeight / 2 : (morphTargetInfo?.halfH ?? currentR);

  if (!hasExplicit && !morphTargetInfo) return currentR;

  const targetR = getMorphTargetRadius(p, targetShape, halfWidth, halfHeight, params.morphCorner || 0);
  const allowShrinkage = params.morphAllowShrinkage === 1 || params.morphAllowShrinkage === true;
  const safeTarget = allowShrinkage ? targetR : Math.max(currentR, targetR);

  return lerp(currentR, safeTarget, morphFactor);
}
