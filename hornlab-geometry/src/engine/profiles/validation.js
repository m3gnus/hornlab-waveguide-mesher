import { evalParam } from '../../common.js';

const VALIDATION_RULES = {
  a0: { min: 0, max: 90, message: 'a0 must be between 0 and 90 degrees' },
  r0: { min: 0, exclusive: true, message: 'r0 must be positive' },
  k: { min: 0, exclusive: true, message: 'k must be greater than 0' },
  tmax: { min: 0, max: 1, message: 'tmax must be between 0 and 1' },
  a: { min: 0, max: 90, exclusiveMin: true, exclusiveMax: true, message: 'Coverage angle a must be in (0, 90) degrees exclusive' },
  q: { min: 0, exclusive: true, message: 'OSSE exponent q must be greater than 0' },
  n: { min: 0, exclusive: true, message: 'Superellipse exponent n must be greater than 0' },
  m: { min: 0, max: 1, message: 'Blending parameter m must be between 0 and 1' },
  r: { min: 0, exclusive: true, message: 'Regularization parameter r must be greater than 0' },
  b: { min: 0, max: 1, message: 'Bias parameter b must be between 0 and 1' }
};

function validateRule(name, value, rule) {
  if (value === undefined) return null;
  if (!Number.isFinite(value)) {
    return `${name} must be a finite number`;
  }
  if (rule.exclusiveMin && value <= rule.min) return rule.message;
  else if (rule.min !== undefined && value < rule.min) return rule.message;
  if (rule.exclusiveMax && value >= rule.max) return rule.message;
  else if (rule.max !== undefined && value > rule.max) return rule.message;
  if (rule.exclusive && value <= 0) return rule.message;
  return null;
}

export function validateParameters(params, _modelType) {
  const sampleP = 0;
  const errors = [];

  for (const [name, rule] of Object.entries(VALIDATION_RULES)) {
    const value = params[name] !== undefined ? evalParam(params[name], sampleP) : undefined;
    const error = validateRule(name, value, rule);
    if (error) errors.push(error);
  }

  // Cross-parameter check: R must be > r0 for R-OSSE
  if (params.R !== undefined && params.r0 !== undefined) {
    const R = evalParam(params.R, sampleP);
    const r0 = evalParam(params.r0, sampleP);
    if (Number.isFinite(R) && Number.isFinite(r0) && R <= r0) {
      errors.push('Mouth radius R must be greater than throat radius r0');
    }
  }

  // Validate L(p) > 0 at multiple sample angles
  if (params.L !== undefined) {
    const sampleAngles = [0, Math.PI/4, Math.PI/2, 3*Math.PI/4, Math.PI, 5*Math.PI/4, 3*Math.PI/2, 7*Math.PI/4];
    for (const p of sampleAngles) {
      const Lval = evalParam(params.L, p);
      if (!Number.isFinite(Lval) || Lval <= 0) {
        errors.push('Length L must be positive at all angles');
        break;
      }
    }
  }

  return { valid: errors.length === 0, errors };
}
