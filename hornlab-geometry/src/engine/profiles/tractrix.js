import { evalParam } from '../../common.js';

/**
 * Tractrix horn profile.
 *
 * The tractrix is defined by the property that the tangent from any point
 * on the curve to the symmetry axis has constant length equal to the mouth
 * radius `a`. It maintains spherical wavefronts.
 *
 * Parametric form (from mouth toward axis):
 *   x(t) = a * (t - tanh(t))
 *   y(t) = a / cosh(t)
 *
 * where a = mouth radius, t = 0 at mouth (y = a), t -> inf at axis (y -> 0).
 *
 * The horn is truncated at the throat where y(t) = r0, giving a finite length.
 * The normalized parameter u in [0, 1] maps from throat (u=0) to mouth (u=1).
 */

/**
 * Find the tractrix parameter t where y(t) = targetR, i.e. a/cosh(t) = targetR.
 * Solves: cosh(t) = a / targetR  =>  t = acosh(a / targetR)
 */
function findTractrixParam(a, targetR) {
  if (targetR <= 0 || targetR > a) return 0;
  const ratio = a / targetR;
  // acosh(x) = ln(x + sqrt(x^2 - 1))
  return Math.log(ratio + Math.sqrt(Math.max(0, ratio * ratio - 1)));
}

/**
 * Calculate a point on the tractrix profile.
 *
 * @param {number} u - Normalized parameter [0, 1], 0 = throat, 1 = mouth
 * @param {number} p - Azimuthal angle (for expression evaluation)
 * @param {Object} params - Profile parameters: r0 (throat radius), R (mouth radius)
 * @returns {{ x: number, y: number }} - Axial position x and radial position y
 */
export function calculateTractrix(u, p, params) {
  const r0 = evalParam(params.r0, p);
  const R = evalParam(params.R, p);

  // Mouth radius a = R
  const a = R;

  // Clamp: if throat >= mouth, degenerate to cylinder
  if (r0 >= a) {
    return { x: 0, y: r0 };
  }

  // Find tractrix parameter at throat (y = r0) and mouth (y = a => t = 0)
  const tThroat = findTractrixParam(a, r0);
  // tMouth = 0 (at the mouth, y = a)

  // Map u in [0,1] from throat to mouth:
  // u=0 -> t=tThroat (throat), u=1 -> t=0 (mouth)
  const t = tThroat * (1 - u);

  // Tractrix parametric equations
  const coshT = Math.cosh(t);
  const tanhT = Math.tanh(t);

  const xRaw = a * (t - tanhT);
  const yRaw = a / coshT;

  // x at the throat (u=0, t=tThroat) gives the total horn length.
  // We want x=0 at the throat and x increasing toward the mouth.
  const xThroat = a * (tThroat - Math.tanh(tThroat));
  const x = xThroat - xRaw;

  return { x, y: yRaw };
}
