/**
 * Mouth-roundover arc helper — shared parametric geometry for any family
 * whose mouth wants to curl over/around in the (axial, radial) plane.
 *
 * Contract
 * --------
 * Given a body exit point P0 = (x0, r0), a body exit tangent direction
 * (tangentDx, tangentDr) that points "forward" along the horn, an arc
 * radius R, and a sweep angle θ (radians, 0..2π), build a circular arc of
 * radius R that:
 *
 *   - passes through P0,
 *   - has tangent equal to (tangentDx, tangentDr) at P0 (C1 continuous
 *     with the body), and
 *   - curls **toward the axis** (outside of the body, i.e. the center
 *     of curvature is on the mouth-outboard side of the body tangent).
 *
 * The arc is parameterized by arc length s in [0, R · θ].  The end
 * tangent is the start tangent rotated by θ (CCW in the (x, r) plane).
 *
 * User-facing angle mapping:
 *   user angle = physical arc sweep (no doubling).
 *     0°   → no roundover
 *     90°  → quarter-turn arc; for a horn whose body tangent is roughly
 *            radial at the mouth, the rim tangent lands pointing backward
 *            toward the throat (canonical roundover shape).
 *     180° → half-circle wrap.
 *
 * Conventions
 * -----------
 * Coordinates are (x = axial, r = radial).  The arc center is placed at
 * C = P0 + R · n, where n is the tangent rotated 90° CCW in the (x, r)
 * plane:
 *
 *     n = (-t_r, t_x)
 *
 * For a body that exits roughly forward (t_x > 0, t_r ≥ 0), n points
 * upward-and-backward, i.e. the arc curves the mouth up and back toward
 * the throat.
 */

import { clamp, toRad } from '../common.js';

export const MAX_SWEEP_DEG = 180;
export const DEFAULT_SWEEP_DEG = 90;

/**
 * Build a parametric arc descriptor.
 *
 * @param {object} spec
 * @param {number} spec.x0           — axial position of body exit
 * @param {number} spec.r0           — radial position of body exit
 * @param {number} spec.tangentDx    — axial component of body exit tangent
 * @param {number} spec.tangentDr    — radial component of body exit tangent
 * @param {number} spec.radius       — arc radius (mm)
 * @param {number} spec.sweepRad     — sweep angle (radians); clamped to [0, 2π]
 * @returns {{
 *   arcLength: number,
 *   axialExtent: { min: number, max: number },
 *   evaluate: (s: number) => { x: number, r: number },
 *   tangentAt: (s: number) => { dx: number, dr: number },
 * }}
 */
export function makeRoundoverArc({ x0, r0, tangentDx, tangentDr, radius, sweepRad }) {
  const sweep = clamp(sweepRad, 0, 2 * Math.PI);
  const R = Math.max(0, Number(radius) || 0);

  const tLen = Math.hypot(tangentDx, tangentDr);
  const tx = tLen > 1e-12 ? tangentDx / tLen : 1;
  const tr = tLen > 1e-12 ? tangentDr / tLen : 0;

  // Rotate tangent 90° CCW → outward-and-upward normal for a forward-facing
  // body tangent.  Placing the arc center at P0 + R·n curls the mouth up
  // and back.
  const nx = -tr;
  const nr = tx;
  const cx = x0 + R * nx;
  const cr = r0 + R * nr;

  // Angle of the vector C→P0; this is where the arc starts.
  const theta0 = Math.atan2(r0 - cr, x0 - cx);

  const arcLength = R * sweep;

  // Pre-compute axial extent: the arc's x(s) = cx + R·cos(theta0 + s/R).
  // Derivative dx/ds = -sin(theta0 + s/R).  Zero at s/R = π/2 - theta0
  // (mod π), giving a local extremum.  We sample the start, end, and any
  // interior critical points to find the min/max axial bounds.
  let xMin = Infinity;
  let xMax = -Infinity;
  const sample = (s) => {
    const a = theta0 + (R > 0 ? s / R : 0);
    const x = cx + R * Math.cos(a);
    if (x < xMin) xMin = x;
    if (x > xMax) xMax = x;
  };
  sample(0);
  sample(arcLength);
  // Critical points of x(a) = cx + R·cos(a) are at a = 0 (max x) and a = π
  // (min x).  Check whether those a values are hit by the sweep range.
  for (const critA of [0, Math.PI, -Math.PI, 2 * Math.PI, -2 * Math.PI]) {
    const s = R > 0 ? (critA - theta0) * R : -1;
    if (s >= 0 && s <= arcLength) sample(s);
  }

  return {
    arcLength,
    axialExtent: { min: xMin, max: xMax },
    evaluate(s) {
      const sClamped = clamp(s, 0, arcLength);
      const a = theta0 + (R > 0 ? sClamped / R : 0);
      return {
        x: cx + R * Math.cos(a),
        r: cr + R * Math.sin(a),
      };
    },
    tangentAt(s) {
      const sClamped = clamp(s, 0, arcLength);
      const a = theta0 + (R > 0 ? sClamped / R : 0);
      return { dx: -Math.sin(a), dr: Math.cos(a) };
    },
  };
}

/**
 * Parametric arc length of a roundover with the given (radius, angle°).
 * This is also the section's stored `length` field in the sections data
 * model: it lets the sections chain's scalar callers treat the roundover
 * as a standard section of that parametric length.
 *
 * User angle = physical arc sweep (no doubling):
 *   user 90°  → 90° arc  (canonical roundover: rim points backward for a
 *                         horn whose body tangent is roughly radial)
 *   user 180° → 180° arc (half-circle wrap)
 */
export function roundoverArcLength(radiusMm, angleDeg) {
  const R = Math.max(0, Number(radiusMm) || 0);
  const sweep = clamp(toRad(Number(angleDeg) || 0), 0, 2 * Math.PI);
  return R * sweep;
}
