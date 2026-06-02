/**
 * Cross-section shape module — area-driven superellipse geometry.
 *
 * Extracted from classical.js so all profile families (OSSE, R-OSSE,
 * Classical) can apply squareness/roundness independently of the horn law.
 *
 * The horn law defines the area S(z) = π·r². This module converts that
 * area into a polar radius for the chosen cross-section shape at each
 * azimuthal angle, preserving the original area.
 *
 * The profile-system redesign added optional keyframe timelines: the
 * cross-section spec can carry `timeline: [{t, exponent,
 * aspectRatio}, ...]`.  When present with >= 2 keyframes, applyCrossSection
 * PCHIP-interpolates exponent and aspectRatio at the current axial t
 * instead of using the scalar spec.exponent / spec.aspectRatio values.
 * This replaces the retired single-axis `morphing.js` morph feature with a
 * multi-keyframe generalization that also works in multi-axis mode.
 */

import { pchipEval } from './interp.js';

// ---------------------------------------------------------------------------
// Gamma function (Lanczos approximation) for superellipse area formula.
// ---------------------------------------------------------------------------
const LANCZOS_C = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
];

function gammaFn(x) {
  if (x < 0.5) return Math.PI / (Math.sin(Math.PI * x) * gammaFn(1 - x));
  x -= 1;
  let a = LANCZOS_C[0];
  for (let i = 1; i < 9; i++) a += LANCZOS_C[i] / (x + i);
  const t = x + 7.5;
  return Math.sqrt(2 * Math.PI) * Math.pow(t, x + 0.5) * Math.exp(-t) * a;
}

// Superellipse area = G(n) * a * b, where G(n) = 4·Γ(1+1/n)²/Γ(1+2/n)
// G(2) = π (circle), G(∞) → 4 (rectangle)
const _gammaFactorCache = new Map();

export function gammaFactor(n) {
  const key = Math.round(n * 10000); // cache to 0.0001 precision
  if (_gammaFactorCache.has(key)) return _gammaFactorCache.get(key);
  const val = 4 * Math.pow(gammaFn(1 + 1 / n), 2) / gammaFn(1 + 2 / n);
  _gammaFactorCache.set(key, val);
  return val;
}

/**
 * Compute superellipse half-dimensions from area, aspect ratio, and exponent.
 * Returns { a: halfWidth, b: halfHeight }.
 *
 * `k` is clamped to >= 0.1 and `n` to >= 2 — the same floors applyCrossSection
 * applies before calling, repeated here so direct callers (tests, future
 * integrations) never produce NaN/Infinity from a degenerate spec.
 */
export function superellipseHalfDims(S, k, n) {
  const kSafe = Math.max(0.1, k);
  const nSafe = Math.max(2, n);
  const G = gammaFactor(nSafe);
  const b = Math.sqrt(Math.max(0, S) / (kSafe * G));
  return { a: kSafe * b, b };
}

/**
 * Polar radius of superellipse |x/a|^n + |y/b|^n = 1 at angle phi.
 *
 * `n` is clamped to >= 2 to keep the `Math.pow(..., -1/n)` term finite when
 * a caller forgets to sanitize.
 */
export function superellipseRadius(phi, a, b, n) {
  const nSafe = Math.max(2, n);
  const ca = Math.abs(Math.cos(phi));
  const sa = Math.abs(Math.sin(phi));
  if (ca < 1e-12) return b;
  if (sa < 1e-12) return a;
  return Math.pow(
    Math.pow(ca / a, nSafe) + Math.pow(sa / b, nSafe),
    -1 / nSafe
  );
}

/**
 * Transition factor: 0 at t <= transitionStart, 1 at t >= transitionEnd.
 *
 * Why two-arg legacy path: pre-redesign callers passed `transitionAt(t,
 * roundStart)` with an implicit end of 1 and a linear ramp.  The migration
 * in `resolveProfileSystem` rewrites `roundStart` to
 * `transitionStart`/`transitionEnd`/`shape: 'linear'`, so migrated configs
 * reach the new three-arg signature with identical shape.  We still accept
 * the two-arg form as defense in depth for any direct caller that hasn't
 * been updated — detected by `arguments.length === 2` and treated as the
 * linear, end=1 legacy ramp.
 *
 * `shape: 'linear'` reproduces today's ramp exactly; `'smooth'` uses
 * smoothstep so newly-authored mid-horn ramps have C¹ continuity at both
 * edges.
 */
export function transitionAt(t, transitionStart, transitionEnd, shape = 'smooth') {
  // Legacy two-arg call: transitionAt(t, roundStart) → linear ramp to 1.
  if (arguments.length === 2) {
    const rs = transitionStart;
    if (t <= rs) return 0;
    const denom = 1 - rs;
    if (denom < 1e-9) return 1;
    return Math.min(1, (t - rs) / denom);
  }
  if (t <= transitionStart) return 0;
  if (t >= transitionEnd) return 1;
  const span = transitionEnd - transitionStart;
  if (span < 1e-9) return 1;
  const raw = (t - transitionStart) / span;
  return shape === 'linear' ? raw : raw * raw * (3 - 2 * raw);
}

// ---------------------------------------------------------------------------
// Cross-section spec constants
// ---------------------------------------------------------------------------
export const CROSS_SECTION_MODES = Object.freeze({
  CIRCULAR: 'circular',
  SUPERELLIPSE: 'superellipse'
});

// aspectRatioMode values:
//   'natural' (default): per-axis radii blend area-proportionally; superellipse
//                        is applied with k=1 to the blended radius. This is
//                        area-preserving — the cardinal H/V dimensions are
//                        rescaled by sqrt(π/G(n)) (≈0.886 for n=8).
//   'manual':            superellipse aspect ratio is set by spec.aspectRatio
//                        (single scalar). Composes with the multi-axis blend.
//   'manualHV':          per-axis H and V radii are taken as the literal
//                        superellipse half-dimensions a and b. NO area-preserving
//                        rescale. The multi-axis blend is bypassed at the
//                        cross-section step. Requires exactly two axes whose
//                        per-axis radii are interpreted as half-width (axisA, H)
//                        and half-height (axisB, V). Use applyCrossSectionManualHV.
//   'rectFillet':        true rectangle + quarter-circle corner fillets. Walls
//                        are razor-flat between fillets. The fillet radius is
//                        controlled by spec.filletRadius (mm, scalar or
//                        timeline-keyframe). When filletRadius >= min(rH, rV)
//                        the cross-section degenerates to an ellipse (circle
//                        when rH=rV). Same H/V semantics as manualHV.
//                        Use applyCrossSectionRectFillet.
export const ASPECT_RATIO_MODES = Object.freeze({
  NATURAL: 'natural',
  MANUAL: 'manual',
  MANUAL_HV: 'manualHV',
  RECT_FILLET: 'rectFillet'
});

/**
 * Apply cross-section shaping to a circular-equivalent radius.
 *
 * The horn law produces rBase (the radius that gives the correct area for
 * a circular cross-section). This function converts it to the actual polar
 * radius for the chosen cross-section shape.
 *
 * The old `spec.mode` field is no longer consulted — shape is fully
 * determined by the numeric parameters.  When `exponent` is ≤ 2 + ε and
 * `aspectRatio` is 1, the spec describes a circle and the base radius is
 * returned unchanged (skips the superellipse math and gamma-factor lookup).
 *
 * @param {number} rBase  - Circular-equivalent radius from the horn law
 * @param {number} t      - Normalized axial position [0,1]
 * @param {number} phi    - Azimuthal angle [0, 2π]
 * @param {Object} spec   - Cross-section spec { aspectRatio, exponent,
 *                          transitionStart, transitionEnd, shape }
 *                          (legacy `roundStart` accepted as a fallback for
 *                          any caller that bypasses the migration).
 * @returns {number} The shaped polar radius
 */
export function applyCrossSection(rBase, t, phi, spec) {
  if (!spec) return rBase;

  // Phase 4: when the spec carries a timeline with >= 2 keyframes, resolve
  // exponent + aspectRatio (if manual) at the current axial t via PCHIP.
  // A single keyframe degrades to constant shape — treated the same as an
  // absent timeline so the legacy scalar-spec path still works.
  const timeline = sanitizeTimeline(spec.timeline);
  const hasTimeline = timeline && timeline.length >= 2;

  let expTarget;
  let effectiveAspect;
  if (hasTimeline) {
    // Timeline overrides both the scalar spec.exponent and roundStart —
    // keyframe `t` values control where transitions happen.
    const kf = evalTimelineAt(timeline, t);
    expTarget = kf.exponent ?? spec.exponent ?? 2;
    effectiveAspect = spec.aspectRatioMode === 'manual'
      ? (kf.aspectRatio ?? spec.aspectRatio ?? 1)
      : 1;
    // Skip shaping when the resolved keyframe reduces to a circle.
    if (expTarget <= 2 + 1e-9 && Math.abs(effectiveAspect - 1) < 1e-9) return rBase;

    const kTarget = Math.max(0.1, effectiveAspect || 1);
    const nTarget = Math.max(2, Math.min(20, expTarget));
    const S = Math.PI * rBase * rBase;
    const { a, b } = superellipseHalfDims(S, kTarget, nTarget);
    return superellipseRadius(phi, a, b, nTarget);
  }

  expTarget = spec.exponent ?? 2;
  // Effective aspect ratio: when mode is 'natural' (the post-redesign
  // default), the per-axis profile blend is the SOLE source of H/V
  // asymmetry — the cross-section spec should NOT apply an additional
  // stretch on top of that.  Treating the effective aspect as 1 here
  // lets the superellipse math describe corner rounding and squareness
  // without fighting the blend.
  effectiveAspect = spec.aspectRatioMode === 'manual'
    ? (spec.aspectRatio ?? 1)
    : 1;
  // Fast path: spec reduces to a circle → no shaping to apply.
  // (Also short-circuits legacy specs carrying mode === 'circular'.)
  if (expTarget <= 2 + 1e-9 && Math.abs(effectiveAspect - 1) < 1e-9) return rBase;
  if (spec.mode === CROSS_SECTION_MODES.CIRCULAR) return rBase;

  // Why the three-way read: the profileSystem migration rewrites legacy
  // `roundStart` into `transitionStart`/`transitionEnd`/`shape: 'linear'`
  // before this function ever sees a migrated spec.  The `roundStart`
  // fallback here is defense in depth for any caller that bypasses the
  // migration (direct tests, programmatic config imports).
  const start = Math.max(0, Math.min(1, spec.transitionStart ?? spec.roundStart ?? 0));
  const endRaw = spec.transitionEnd ?? 1;
  const end = Math.max(start, Math.min(1, endRaw));
  const shape = spec.shape ?? 'smooth';
  const tf = transitionAt(t, start, end, shape);
  if (tf <= 0) return rBase;

  // Aspect ratio is clamped to [0.1, ∞).  The floor prevents degenerate
  // tall-and-narrow cross-sections (a = k·b → 0 as k → 0) which collapse
  // the horn width to zero and break the ray-cast enclosure generators.
  // 0.1 keeps the width at least 10 % of the height — below that users
  // should switch to rotating the axes rather than stretching the shape.
  // In 'natural' mode we force k = 1 so the per-axis blend is the only
  // source of H/V asymmetry; in 'manual' mode we honor the user's value.
  const kTarget = Math.max(0.1, effectiveAspect || 1);
  const nTarget = Math.max(2, Math.min(20, spec.exponent || 6));

  // Interpolate shape params: circle (k=1, n=2) → target
  const k = 1 + (kTarget - 1) * tf;
  const n = 2 + (nTarget - 2) * tf;

  // Area from circular profile (the horn law is area-preserving)
  const S = Math.PI * rBase * rBase;

  // Solve superellipse half-dimensions to match area
  const { a, b } = superellipseHalfDims(S, k, n);

  // Polar radius at azimuthal angle phi
  return superellipseRadius(phi, a, b, n);
}

// ---------------------------------------------------------------------------
// Cross-section timeline helpers (Phase 4)
// ---------------------------------------------------------------------------

/**
 * Validate + normalize a timeline array.
 *
 * Rules (from the plan):
 *   - Array of {t, exponent?, aspectRatio?} objects.
 *   - Sorted by t ascending; duplicate-t keyframes are deduplicated
 *     (last-write-wins after sort).
 *   - First t >= 0, last t <= 1; individual entries clamped into [0, 1].
 *   - Returns null for non-arrays, empty arrays, or arrays where every
 *     entry is malformed.  A single valid keyframe returns an array of
 *     length 1 (callers then use the scalar-spec fallback).
 *
 * Pure function — does not mutate the input.
 */
export function sanitizeTimeline(timeline) {
  if (!Array.isArray(timeline) || timeline.length === 0) return null;
  const cleaned = [];
  for (const kf of timeline) {
    if (!kf || typeof kf !== 'object') continue;
    const tRaw = Number(kf.t);
    if (!Number.isFinite(tRaw)) continue;
    const t = Math.max(0, Math.min(1, tRaw));
    const entry = { t };
    if (Number.isFinite(Number(kf.exponent))) {
      entry.exponent = Math.max(2, Math.min(20, Number(kf.exponent)));
    }
    if (Number.isFinite(Number(kf.aspectRatio))) {
      entry.aspectRatio = Math.max(0.1, Number(kf.aspectRatio));
    }
    cleaned.push(entry);
  }
  if (cleaned.length === 0) return null;
  cleaned.sort((a, b) => a.t - b.t);
  // Dedup by t (keep last occurrence so user-authored later keyframes win).
  const dedup = [];
  for (let i = 0; i < cleaned.length; i++) {
    const cur = cleaned[i];
    const prev = dedup[dedup.length - 1];
    if (prev && Math.abs(prev.t - cur.t) < 1e-9) {
      dedup[dedup.length - 1] = { ...prev, ...cur };
    } else {
      dedup.push(cur);
    }
  }
  return dedup;
}

/**
 * Evaluate a sanitized timeline at axial position t, returning
 * { exponent, aspectRatio } interpolated via PCHIP across keyframes.
 *
 * Caller is expected to have pre-sanitized the timeline (sanitizeTimeline).
 * Keyframes that omit `exponent` or `aspectRatio` contribute nothing to
 * that channel — we interpolate only over keyframes that carry the field.
 * When no keyframe carries a field, the return value for that field is
 * undefined and callers fall back to the scalar spec.
 */
export function evalTimelineAt(timeline, t) {
  if (!Array.isArray(timeline) || timeline.length === 0) return {};
  if (timeline.length === 1) {
    return { exponent: timeline[0].exponent, aspectRatio: timeline[0].aspectRatio };
  }

  const out = {};

  // Exponent channel
  const expXs = [];
  const expYs = [];
  for (const kf of timeline) {
    if (kf.exponent !== undefined) { expXs.push(kf.t); expYs.push(kf.exponent); }
  }
  if (expXs.length >= 2) {
    out.exponent = pchipEval(expXs, expYs, t);
  } else if (expXs.length === 1) {
    out.exponent = expYs[0];
  }

  // Aspect ratio channel
  const arXs = [];
  const arYs = [];
  for (const kf of timeline) {
    if (kf.aspectRatio !== undefined) { arXs.push(kf.t); arYs.push(kf.aspectRatio); }
  }
  if (arXs.length >= 2) {
    out.aspectRatio = pchipEval(arXs, arYs, t);
  } else if (arXs.length === 1) {
    out.aspectRatio = arYs[0];
  }

  return out;
}

// ---------------------------------------------------------------------------
// Manual H/V cross-section path (literal half-dimension mode)
// ---------------------------------------------------------------------------

/**
 * Evaluate the cross-section as a superellipse whose half-dimensions are the
 * per-axis radii rH (half-width) and rV (half-height) — taken literally, with
 * no area-preserving rescale.
 *
 * Contrast with applyCrossSection's default ('natural'/'manual') path:
 *   - applyCrossSection treats its rBase as a circular-equivalent radius and
 *     solves superellipseHalfDims so that S = π·rBase² is preserved across
 *     the morph from circle to rounded-rect. That distorts the cardinal H/V
 *     dimensions by sqrt(π/G(n)) (≈0.886 at n=8, ≈0.893 at n=10).
 *   - This function bypasses the rescale: a = rH, b = rV are used directly
 *     as the superellipse semi-axes. The resulting shape passes exactly
 *     through (rH, 0) at phi=0 and (0, rV) at phi=π/2 for every n.
 *
 * Use this when the H and V profile axes describe literal wall coordinates —
 * e.g. a Synergy Calc V5-style rectangular conical horn where the spreadsheet's
 * tan(coverage_half_angle)·z geometry must hold.
 *
 * Exponent resolution: when the spec carries a ≥2-keyframe timeline, exponent
 * is PCHIP-interpolated at the current axial t. Otherwise spec.exponent
 * (default 2) is used. Result is clamped to [2, 20] to match
 * applyCrossSection.
 *
 * Transition window: when spec.transitionStart/transitionEnd are set, the
 * effective exponent ramps from 2 (circle) at transitionStart to the target
 * exponent at transitionEnd. This lets the shape morph from a circular
 * throat (n=2) to a rounded rectangle (n=target) over a controllable range.
 *
 * @param {number} rH   - Half-width (H-axis radius at this z)
 * @param {number} rV   - Half-height (V-axis radius at this z)
 * @param {number} t    - Normalized axial position [0,1]
 * @param {number} phi  - Azimuthal angle [0, 2π]
 * @param {Object} spec - Cross-section spec (timeline / exponent / transition)
 * @returns {number} Polar radius of the literal-rH/rV superellipse at phi
 */
export function applyCrossSectionManualHV(rH, rV, t, phi, spec) {
  const a = Math.max(0, rH);
  const b = Math.max(0, rV);
  if (!spec) {
    // No spec: behave as a plain ellipse (n=2) with these semi-axes.
    return superellipseRadius(phi, a, b, 2);
  }

  // Resolve target exponent from timeline (if active) or scalar spec.
  const timeline = sanitizeTimeline(spec.timeline);
  const hasTimeline = timeline && timeline.length >= 2;
  let nTargetRaw;
  if (hasTimeline) {
    const kf = evalTimelineAt(timeline, t);
    nTargetRaw = kf.exponent ?? spec.exponent ?? 2;
  } else {
    nTargetRaw = spec.exponent ?? 2;
  }
  const nTarget = Math.max(2, Math.min(20, nTargetRaw));

  // Optional transition window: ramp n from 2 → nTarget across
  // [transitionStart, transitionEnd]. Inactive when transitionStart is
  // missing/0 and transitionEnd is missing/1 (default: shape applied
  // throughout). Mirrors applyCrossSection's transition-window semantics.
  let nEffective = nTarget;
  if (!hasTimeline) {
    const start = Math.max(0, Math.min(1, spec.transitionStart ?? spec.roundStart ?? 0));
    const endRaw = spec.transitionEnd ?? 1;
    const end = Math.max(start, Math.min(1, endRaw));
    if (start > 0 || end < 1) {
      const shape = spec.shape ?? 'smooth';
      const tf = transitionAt(t, start, end, shape);
      nEffective = 2 + (nTarget - 2) * tf;
    }
  }

  return superellipseRadius(phi, a, b, nEffective);
}

// ---------------------------------------------------------------------------
// Rect + quarter-circle fillet cross-section
// ---------------------------------------------------------------------------

/**
 * Polar radius of a true rectangle with quarter-circle corner fillets.
 *
 * Cross-section (Q1 by symmetry):
 *   - Right wall: x = a, |y| ≤ b - rf
 *   - Top wall:   y = b, |x| ≤ a - rf
 *   - Corner:     quarter-circle of radius rf centered at (a-rf, b-rf)
 *
 * `rf` (filletRadius) is clamped to [0, min(a, b)]. At rf = 0 the corner is
 * razor-sharp; at rf = min(a, b) the cross-section degenerates to an
 * ellipse (or a circle when a = b) — useful for smoothly morphing the
 * throat from circular at z=0 to sharp-walled rectangular downstream.
 *
 * Polar radius derivation:
 *   At angle phi (folded to Q1 via |cos|, |sin|):
 *     tanLow  = (b - rf) / a       — angle below which the ray hits the right wall
 *     tanHigh = b / (a - rf)        — angle above which the ray hits the top wall
 *   - tan(phi) ≤ tanLow:  r = a / cos(phi)         (right wall)
 *   - tan(phi) ≥ tanHigh: r = b / sin(phi)         (top wall)
 *   - otherwise:          ray-circle intersection with the corner fillet,
 *                         taking the FAR root (outer surface).
 *
 * Ray-circle math (corner centered at (xF, yF) = (a-rf, b-rf), radius rf):
 *   r·(c, s) on circle ⇒ r² - 2r(c·xF + s·yF) + (xF² + yF² - rf²) = 0
 *   r = (c·xF + s·yF) + sqrt((c·xF + s·yF)² - (xF² + yF² - rf²))
 */
export function applyCrossSectionRectFillet(rH, rV, t, phi, spec) {
  const a = Math.max(0, rH);
  const b = Math.max(0, rV);

  // Resolve target fillet radius from timeline (if active) or scalar spec.
  let rfTarget;
  let nForDegenerate = 2; // exponent used when fillet covers everything
  if (spec) {
    const timeline = sanitizeTimeline(spec.timeline);
    const hasTimeline = timeline && timeline.length >= 2;
    if (hasTimeline) {
      const kf = evalTimelineAt(timeline, t);
      // filletRadius is currently NOT a timeline channel in sanitizeTimeline,
      // but exponent IS — we accept either to drive the morph. Most users will
      // schedule filletRadius via a custom spec.filletRadiusTimeline (below)
      // or via the scalar spec.filletRadius alongside an exponent timeline.
      rfTarget = spec.filletRadius;
      nForDegenerate = kf.exponent ?? spec.exponent ?? 2;
    } else {
      rfTarget = spec.filletRadius;
      nForDegenerate = spec.exponent ?? 2;
    }
    // Optional dedicated timeline for filletRadius — overrides spec.filletRadius
    // when present. Format: spec.filletRadiusTimeline = [{t, filletRadius}, ...].
    const frTimeline = Array.isArray(spec.filletRadiusTimeline)
      ? spec.filletRadiusTimeline
      : null;
    if (frTimeline && frTimeline.length >= 2) {
      rfTarget = pchipFromKeyframes(frTimeline, 'filletRadius', t, rfTarget ?? 0);
    }
  }
  if (!Number.isFinite(rfTarget) || rfTarget < 0) rfTarget = 0;

  const minDim = Math.min(a, b);
  const rf = Math.min(rfTarget, minDim);

  // Degenerate cases ──────────────────────────────────────────────────
  if (a === 0 || b === 0) return 0;
  if (rf >= minDim - 1e-12) {
    // Fillet fills the cross-section: it's an ellipse (or circle if a=b).
    // Use n=nForDegenerate so a timeline-controlled morph stays continuous as
    // the fillet shrinks; n=2 is the default and gives a true ellipse.
    return superellipseRadius(phi, a, b, Math.max(2, Math.min(20, nForDegenerate)));
  }

  // Q1 fold via abs values
  const c = Math.abs(Math.cos(phi));
  const s = Math.abs(Math.sin(phi));
  if (c < 1e-12) return b;            // phi = ±π/2
  if (s < 1e-12) return a;            // phi = 0 or π

  const xF = a - rf;
  const yF = b - rf;
  const tanPhi = s / c;
  const tanLow  = yF / a;             // tan at the start of the corner arc
  const tanHigh = b / xF;             // tan at the end of the corner arc

  if (tanPhi <= tanLow) {
    // Ray hits the right wall at x = a.
    return a / c;
  }
  if (tanPhi >= tanHigh) {
    // Ray hits the top wall at y = b.
    return b / s;
  }

  // Corner-fillet region: ray-circle intersection with the FAR root.
  const dot = c * xF + s * yF;
  const cxy = xF * xF + yF * yF - rf * rf;
  const disc = dot * dot - cxy;
  return dot + Math.sqrt(Math.max(0, disc));
}

// Local PCHIP helper — extracts {t, <key>} keyframes and evaluates at t,
// independent of sanitizeTimeline's fixed channel set. Falls back to
// `fallback` when fewer than 2 valid keyframes exist.
function pchipFromKeyframes(keyframes, key, t, fallback) {
  const xs = [];
  const ys = [];
  for (const kf of keyframes) {
    if (!kf || typeof kf !== 'object') continue;
    const tv = Number(kf.t);
    const yv = Number(kf[key]);
    if (Number.isFinite(tv) && Number.isFinite(yv)) {
      xs.push(Math.max(0, Math.min(1, tv)));
      ys.push(yv);
    }
  }
  if (xs.length < 2) return fallback;
  // Sort by t (in case the input wasn't sorted).
  const idx = xs.map((_, i) => i).sort((p, q) => xs[p] - xs[q]);
  const sx = idx.map(i => xs[i]);
  const sy = idx.map(i => ys[i]);
  return pchipEval(sx, sy, t);
}
