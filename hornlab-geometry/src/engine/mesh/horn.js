import { evalParam, toRad } from '../../common.js';
import { DEFAULTS, MORPH_TARGETS } from '../constants.js';
import { applyMorphing, getRoundedRectRadius } from '../morphing.js';
import { resolveProfileSystem, evaluateMultiAxisProfile } from '../profileSystem.js';
import { getClassicalLength } from '../profiles/classical.js';
import { calculateOSSE, invertOsseCoverageAngle } from '../profiles/osse.js';

export function evaluateInnerProfileAt(t, p, params, context) {
  // In v2, resolveProfileSystem ALWAYS returns a system (a one-axis system
  // migrated from legacy params when profileSystem is absent), so there is
  // no legacy single-profile fallback.  The fast path uses the pre-resolved
  // system from context to avoid per-vertex re-resolution.
  const system = context?.resolvedSystem || resolveProfileSystem(params);
  return evaluateMultiAxisProfile(t, p, system);
}

export function computeMouthExtents(params, context) {
  const sampleCount = Math.max(360, Math.round((params.angularSegments || DEFAULTS.ANGULAR_SEGMENTS) * 4));
  const needsTarget = params.morphTarget !== undefined && Number(params.morphTarget) !== MORPH_TARGETS.NONE;
  const hasExplicit = (params.morphWidth > 0) || (params.morphHeight > 0);

  let rawMaxX = 0;
  let rawMaxZ = 0;

  const evaluateAt = (p) => evaluateInnerProfileAt(1, p, params, context);

  for (let i = 0; i < sampleCount; i += 1) {
    const p = (i / sampleCount) * Math.PI * 2;
    const profile = evaluateAt(p);
    const r = profile.y;
    rawMaxX = Math.max(rawMaxX, Math.abs(r * Math.cos(p)));
    rawMaxZ = Math.max(rawMaxZ, Math.abs(r * Math.sin(p)));
  }

  const morphTargetInfo = (needsTarget && !hasExplicit) ? { halfW: rawMaxX, halfH: rawMaxZ } : null;

  if (!needsTarget) {
    return { halfW: rawMaxX, halfH: rawMaxZ, morphTargetInfo };
  }

  // For shrinkage: the mouth radius at phi angles where the morph target
  // is smaller uses the target coverage angle profile, not the raw.
  const allowShrinkage = params.morphAllowShrinkage === 1 || params.morphAllowShrinkage === true;
  const halfW = params.morphWidth > 0 ? params.morphWidth / 2 : 0;
  const halfH = params.morphHeight > 0 ? params.morphHeight / 2 : 0;
  const cornerR = params.morphCorner || 0;

  let maxX = 0;
  let maxZ = 0;
  for (let i = 0; i < sampleCount; i += 1) {
    const p = (i / sampleCount) * Math.PI * 2;
    const profile = evaluateAt(p);

    let r;
    if (allowShrinkage && hasExplicit && params.type === 'OSSE') {
      const targetMouthR = getRoundedRectRadius(p, halfW, halfH, cornerR);
      if (targetMouthR < profile.y) {
        // Shrinkage: at the mouth (t=1, morphFactor=1), the radius equals
        // the shrink profile's mouth radius, which equals targetMouthR.
        r = targetMouthR;
      } else {
        r = applyMorphing(profile.y, 1, p, params, morphTargetInfo);
      }
    } else {
      r = applyMorphing(profile.y, 1, p, params, morphTargetInfo);
    }

    maxX = Math.max(maxX, Math.abs(r * Math.cos(p)));
    maxZ = Math.max(maxZ, Math.abs(r * Math.sin(p)));
  }

  return { halfW: maxX, halfH: maxZ, morphTargetInfo };
}

export function buildMorphTargets(params, lengthSteps, angleList, sliceMap, context) {
  const safeAngles = Array.isArray(angleList) && angleList.length > 0 ? angleList : [0, Math.PI / 2];

  return Array.from({ length: lengthSteps + 1 }, (_, j) => {
    const t = sliceMap ? sliceMap[j] : j / lengthSteps;
    let maxX = 0;
    let maxZ = 0;

    for (const p of safeAngles) {
      const profile = evaluateInnerProfileAt(t, p, params, context);
      const r = profile.y;
      maxX = Math.max(maxX, Math.abs(r * Math.cos(p)));
      maxZ = Math.max(maxZ, Math.abs(r * Math.sin(p)));
    }

    return { halfW: maxX, halfH: maxZ };
  });
}

/**
 * Pre-compute per-phi coverage angles that make the OSSE mouth radius hit
 * the morph target at every phi.  Returns a Map(phiIndex → coverageAngle).
 *
 * Unified path (always covers every phi in angleList): previously we only
 * populated entries where targetMouthR < rawMouthR and let growth-only phi
 * fall through to applyMorphing's power-law blend.  That split produced a
 * C0 discontinuity around the ring on configs where the guiding curve
 * made rawMouthR(phi) cross targetMouthR(phi) — the shrink path follows a
 * pure OSSE curve in t while the growth path follows a
 * lerp(rawOSSE(t,phi), targetR, morphFactor) curve, and the two disagree
 * at intermediate t even when they match at the mouth.  Saw/SF-driven
 * gcurves cross often, so the ring radius jumps around phi and the
 * viewport mesh shows visible wrinkles.  Evaluating every phi with the
 * same coverage-angle-inverted OSSE family keeps r(t,phi) continuous in
 * phi at the cost of ignoring morphFixed/morphRate shape control in
 * shrink-on mode (shape is dictated by the OSSE family, matching the
 * ATH binary's behaviour).
 */
export function buildShrinkData(params, angleList, context) {
  if (params.type !== 'OSSE') return null;

  const targetShape = Number(params.morphTarget || MORPH_TARGETS.NONE);
  if (targetShape === MORPH_TARGETS.NONE) return null;

  const allowShrinkage = params.morphAllowShrinkage === 1 || params.morphAllowShrinkage === true;
  if (!allowShrinkage) return null;

  const halfW = params.morphWidth > 0 ? params.morphWidth / 2 : 0;
  const halfH = params.morphHeight > 0 ? params.morphHeight / 2 : 0;
  if (halfW <= 0 && halfH <= 0) return null;

  const cornerR = params.morphCorner || 0;

  const shrinkMap = new Map();

  for (let i = 0; i < angleList.length; i += 1) {
    const p = angleList[i];

    // Extract OSSE config at this phi
    const L = evalParam(params.L, p);
    const extLen = Math.max(0, evalParam(params.throatExtLength || 0, p));
    const extAngleRad = toRad(evalParam(params.throatExtAngle || 0, p));
    const r0Main = evalParam(params.r0, p) + extLen * Math.tan(extAngleRad);
    const a0Deg = evalParam(params.a0, p);

    // Morph target at the mouth (handle both rectangle and circle)
    let targetMouthR;
    if (targetShape === MORPH_TARGETS.CIRCLE) {
      targetMouthR = Math.sqrt(Math.max(0, halfW * halfH));
    } else {
      targetMouthR = getRoundedRectRadius(p, halfW, halfH, cornerR);
    }

    // Invert OSSE to find coverage that produces targetMouthR at z=L.
    const shrinkCov = invertOsseCoverageAngle(targetMouthR, L, p, params, { L, a0Deg, r0Main });
    shrinkMap.set(i, shrinkCov);
  }

  return shrinkMap.size > 0 ? shrinkMap : null;
}

export function createRingVertices(params, sliceMap, angleList, morphTargets, ringCount, lengthSteps, context, shrinkData) {
  const vertices = [];

  // When coverage-angle shrinkage is active, the growth path must never push
  // inward — all narrowing is via the coverage path.  Allocate the overlay
  // once here so the hot per-vertex loop doesn't churn the GC.
  const growOnly = shrinkData ? Object.create(params) : null;
  if (growOnly) growOnly.morphAllowShrinkage = 0;
  const paramsForGrowth = shrinkData ? growOnly : params;

  for (let j = 0; j <= lengthSteps; j += 1) {
    const t = sliceMap ? sliceMap[j] : j / lengthSteps;

    for (let i = 0; i < ringCount; i += 1) {
      const p = angleList[i];
      const profile = evaluateInnerProfileAt(t, p, params, context);

      let r;
      if (shrinkData && shrinkData.has(i)) {
        // Coverage-angle replacement for shrinkage: compute the profile at
        // the reduced coverage angle directly.  The OSSE formula at the
        // target coverage naturally transitions from r0 at the throat to the
        // morph target at the mouth — no blending needed.
        const shrinkProfile = calculateOSSE(
          t * (evalParam(params.L, p) + Math.max(0, evalParam(params.throatExtLength || 0, p)) + Math.max(0, evalParam(params.slotLength || 0, p))),
          p, params, { coverageAngle: shrinkData.get(i) }
        );
        r = shrinkProfile.y;
      } else {
        const morphTargetInfo = morphTargets?.[j] || null;
        r = applyMorphing(profile.y, t, p, paramsForGrowth, morphTargetInfo);
      }

      vertices.push(
        r * Math.cos(p),
        profile.x,
        r * Math.sin(p)
      );
    }
  }

  return vertices;
}

// Winding convention: CCW when viewed from inside the horn (normals point
// inward toward the horn axis). This is the BEM convention — the interior
// acoustic surface is the boundary. All other mesh generators (enclosure
// stitchRing, freestandingWall, source disc) produce winding that is
// BFS-consistent with this convention across shared edges.
export function createHornIndices(ringCount, lengthSteps, fullCircle) {
  const indices = [];
  const radialSteps = fullCircle ? ringCount : Math.max(0, ringCount - 1);

  for (let j = 0; j < lengthSteps; j += 1) {
    for (let i = 0; i < radialSteps; i += 1) {
      const row1 = j * ringCount;
      const row2 = (j + 1) * ringCount;
      const i2 = fullCircle ? (i + 1) % ringCount : i + 1;

      indices.push(row1 + i, row1 + i2, row2 + i2);
      indices.push(row1 + i, row2 + i2, row2 + i);
    }
  }

  return indices;
}

// ---------------------------------------------------------------------------
// Adaptive phi-count mesh: variable phi samples per ring for near-isotropic
// triangles throughout the horn surface (throat → mouth).
// Only used for full-circle renders without enclosure/wall geometry.
// ---------------------------------------------------------------------------

function _sampleEffectiveRadius(params, t, context) {
  // Estimate the effective circumference radius at axial position t.
  // For circular profiles this equals R(t, 0).
  // For morphed/non-circular profiles we sample the perimeter with 16 chords.
  const NSAMPLE = 16;
  let circumference = 0;
  let prevX = null;
  let prevZ = null;

  for (let i = 0; i <= NSAMPLE; i += 1) {
    const p = (i / NSAMPLE) * Math.PI * 2;
    const profile = evaluateInnerProfileAt(t, p, params, context);

    const r = profile.y; // pre-morph radius (morph is minor near throat)
    const x = r * Math.cos(p);
    const z = r * Math.sin(p);

    if (prevX !== null) {
      circumference += Math.hypot(x - prevX, z - prevZ);
    }
    prevX = x;
    prevZ = z;
  }

  return circumference / (Math.PI * 2);
}

/**
 * Compute per-ring phi counts so that circumferential edge ≈ axialStep × targetAspect.
 * Returns an array of length (lengthSteps + 1), each value a multiple of 4, in
 * [12, userMax]. Values are monotonically non-decreasing (throat → mouth).
 */
export function computeAdaptivePhiCounts(params, lengthSteps, sliceMap, userMax, profileContext) {
  let totalLength;
  if (params.type === 'Classical') {
    totalLength = Math.max(0, getClassicalLength(0, params));
  } else {
    const L = Math.max(0, evalParam(params.L || 0, 0));
    const extLen = Math.max(0, evalParam(params.throatExtLength || 0, 0));
    const slotLen = Math.max(0, evalParam(params.slotLength || 0, 0));
    totalLength = L + extLen + slotLen;
  }
  const TARGET_ASPECT = 1.5;
  const MIN_PHI = 12;

  const counts = [];

  for (let j = 0; j <= lengthSteps; j += 1) {
    const t = sliceMap ? sliceMap[j] : j / lengthSteps;
    const rEff = _sampleEffectiveRadius(params, t, profileContext);

    // Axial step via central difference in t, converted to length units.
    const jPrev = Math.max(0, j - 1);
    const jNext = Math.min(lengthSteps, j + 1);
    const tPrev = sliceMap ? sliceMap[jPrev] : jPrev / lengthSteps;
    const tNext = sliceMap ? sliceMap[jNext] : jNext / lengthSteps;
    const dt = (tNext - tPrev) / 2;
    const axialStep = dt * totalLength;

    let n;
    if (axialStep > 0 && rEff > 0) {
      const targetEdge = axialStep * TARGET_ASPECT;
      n = Math.round((2 * Math.PI * rEff) / targetEdge);
    } else {
      n = userMax;
    }

    // Snap to multiple of 4 and clamp.
    const snapped = Math.max(MIN_PHI, Math.min(userMax, Math.round(n / 4) * 4));
    counts.push(snapped);
  }

  // Enforce monotonically non-decreasing (horn only expands toward mouth).
  for (let j = 1; j <= lengthSteps; j += 1) {
    if (counts[j] < counts[j - 1]) counts[j] = counts[j - 1];
  }

  return counts;
}

/**
 * Like createRingVertices but each ring uses its own phi count from phiCounts[].
 * Angles are uniformly distributed: phi_i = (i / N) * 2π.
 */
export function createAdaptiveRingVertices(params, sliceMap, morphTargets, phiCounts, lengthSteps, context, shrinkData) {
  const vertices = [];

  // Pre-hoist shrink-related scalars out of the per-vertex hot loop.
  const halfW = params.morphWidth > 0 ? params.morphWidth / 2 : 0;
  const halfH = params.morphHeight > 0 ? params.morphHeight / 2 : 0;
  const cornerR = params.morphCorner || 0;
  const targetShape = Number(params.morphTarget || MORPH_TARGETS.NONE);
  const circleTargetR = targetShape === MORPH_TARGETS.CIRCLE
    ? Math.sqrt(Math.max(0, halfW * halfH))
    : 0;

  // Cache shrink coverage angles by phi so adjacent rings that share a phi
  // count reuse the 24-iteration binary search.  Key quantization must be
  // tighter than any phi spacing we produce — 1e-6 rad is well below the
  // ~2π/N step for N up to ~10^6.
  const shrinkCovCache = shrinkData ? new Map() : null;
  const getShrinkCov = shrinkData
    ? (p, L, a0Deg, r0Main, targetMouthR) => {
        const key = p.toFixed(6);
        const cached = shrinkCovCache.get(key);
        if (cached !== undefined) return cached;
        const cov = invertOsseCoverageAngle(targetMouthR, L, p, params, { L, a0Deg, r0Main });
        shrinkCovCache.set(key, cov);
        return cov;
      }
    : null;

  for (let j = 0; j <= lengthSteps; j += 1) {
    const t = sliceMap ? sliceMap[j] : j / lengthSteps;
    const N = phiCounts[j];

    for (let i = 0; i < N; i += 1) {
      const p = (i / N) * Math.PI * 2;
      const profile = evaluateInnerProfileAt(t, p, params, context);

      let r;
      if (shrinkData) {
        // Unified coverage-angle path at EVERY phi when shrink is enabled.
        // See buildShrinkData docstring for why we no longer split into
        // shrink-vs-growth branches here.
        const targetMouthR = targetShape === MORPH_TARGETS.CIRCLE
          ? circleTargetR
          : getRoundedRectRadius(p, halfW, halfH, cornerR);
        const L = evalParam(params.L, p);
        const extLen = Math.max(0, evalParam(params.throatExtLength || 0, p));
        const extAngleRad = toRad(evalParam(params.throatExtAngle || 0, p));
        const r0Main = evalParam(params.r0, p) + extLen * Math.tan(extAngleRad);
        const a0Deg = evalParam(params.a0, p);
        const slotLen = Math.max(0, evalParam(params.slotLength || 0, p));
        const totalLength = L + extLen + slotLen;

        const shrinkCov = getShrinkCov(p, L, a0Deg, r0Main, targetMouthR);
        const shrinkProfile = calculateOSSE(t * totalLength, p, params, { coverageAngle: shrinkCov });
        r = shrinkProfile.y;
      } else {
        const morphTargetInfo = morphTargets?.[j] || null;
        r = applyMorphing(profile.y, t, p, params, morphTargetInfo);
      }

      vertices.push(
        r * Math.cos(p),
        profile.x,
        r * Math.sin(p)
      );
    }
  }

  return vertices;
}

/**
 * Generate indices for a variable-phi-count horn mesh (full circle only).
 *
 * Between ring j (N1 phi) and ring j+1 (N2 phi, N2 >= N1):
 *   - If N1 === N2: standard 2-triangle quads.
 *   - If N2 > N1: fan triangles from each ring-j vertex to the ring-j+1 vertices
 *     spanning the same phi sector. Total triangles per ring-pair = N1 + N2.
 *
 * Winding convention matches createHornIndices (CCW when viewed from inside horn).
 */
export function createAdaptiveFanIndices(phiCounts, lengthSteps) {
  const indices = [];

  // Pre-compute the vertex offset of the first vertex in each ring.
  const ringStart = [0];
  for (let j = 0; j < lengthSteps; j += 1) {
    ringStart.push(ringStart[j] + phiCounts[j]);
  }

  for (let j = 0; j < lengthSteps; j += 1) {
    const N1 = phiCounts[j];
    const N2 = phiCounts[j + 1];
    const base1 = ringStart[j];
    const base2 = ringStart[j + 1];

    for (let i = 0; i < N1; i += 1) {
      const a = base1 + i;
      const b = base1 + (i + 1) % N1;

      // Find the ring-j+1 sector boundaries for this phi interval.
      const kLo = Math.round((i * N2) / N1);
      const kHi = Math.max(kLo, Math.round(((i + 1) * N2) / N1));

      // Triangle 1: close the right boundary (matches existing winding for K=1).
      indices.push(a, b, base2 + (kHi % N2));

      // Fan triangles from a back to kLo (inclusive).
      // For K=1 this produces exactly the second standard triangle.
      for (let k = kHi - 1; k >= kLo; k -= 1) {
        indices.push(a, base2 + ((k + 1) % N2), base2 + (k % N2));
      }
    }
  }

  return indices;
}
