/**
 * ProfileSection abstraction — Phase 3A foundation + Phase 3B standalone
 * section evaluators.
 *
 * An axis's geometry is a chain of sections evaluated in series:
 *   [throat?][body][mouth?]
 *
 * The body is always present.  Throat and mouth are optional.
 *
 * Each section is a plain serializable object:
 *   { kind, type, length, params }
 *
 *   kind    — 'throat' | 'body' | 'mouth'
 *   type    — section-type string, scoped to the kind
 *   length  — axial length in mm (0 allowed, means "display-only" / no-op)
 *   params  — section-type-specific parameters (a plain object)
 *
 * Phase 3A shipped the data model + migration + dispatcher that forwarded
 * every section back to the family evaluator at global z.  Phase 3B makes
 * throat and mouth sections first-class geometry producers: the section
 * evaluators compute radii directly from their local z, and the body
 * evaluator receives a params clone with the legacy flat throat/slot/flare
 * keys zeroed.  Byte identity with pre-migration configs still holds — the
 * migration offsets and the new evaluator math were picked so `calculateOSSE`
 * and `calculateClassical` produce the same sequence of values when fed a
 * section-local z and a stripped params object as they did when fed a global
 * z and the original params.
 *
 * See `docs/archive/260523-shipped-plan-history.md` for the longer arc.
 */

import { clamp, evalParam, toRad } from '../common.js';
import { calculateROSSE } from './profiles/rosse.js';
import {
  DEFAULT_SWEEP_DEG,
  MAX_SWEEP_DEG,
  makeRoundoverArc,
  roundoverArcLength,
} from './roundover.js';

// ---------------------------------------------------------------------------
// Constants — exported enums that downstream UI / config code will use.
// ---------------------------------------------------------------------------

export const SECTION_KINDS = Object.freeze({
  THROAT: 'throat',
  BODY: 'body',
  MOUTH: 'mouth',
});

export const THROAT_TYPES = Object.freeze({
  NONE: 'none',
  STRAIGHT: 'straight',
  CONICAL: 'conical',
  OSSE: 'osse',
  QUADRATIC: 'quadratic',
});

export const BODY_TYPES = Object.freeze({
  OSSE: 'OSSE',
  R_OSSE: 'R-OSSE',
  CLASSICAL: 'Classical',
  LOOKUP: 'Lookup',
});

export const MOUTH_TYPES = Object.freeze({
  NONE: 'none',
  FLARE: 'flare',
  ROUNDOVER: 'roundover',
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function evalOr(value, fallback, p = 0) {
  if (value === undefined || value === null || value === '') return fallback;
  const v = evalParam(value, p);
  return Number.isFinite(v) ? v : fallback;
}

// Keys stripped from the body-params clone when the axis has sections.
// These are the legacy flat keys the family evaluators used to do their own
// internal slicing.  Phase 3B section evaluators own that math now, so the
// body must NOT see these keys — it should behave as a pure body.
const OSSE_BODY_STRIPPED_KEYS = Object.freeze([
  'throatExtLength',
  'throatExtAngle',
  'slotLength',
]);
const CLASSICAL_BODY_STRIPPED_KEYS = Object.freeze([
  'classicalFlare2Enabled',
  'classicalFlare2Angle',
  'classicalFlare2Start',
  'classicalFlare2Blend',
]);

function stripKeys(params, keys) {
  if (!params) return params;
  const out = { ...params };
  for (const key of keys) {
    if (key in out) delete out[key];
  }
  return out;
}

// ---------------------------------------------------------------------------
// Migration — legacy flat params → sections array
// ---------------------------------------------------------------------------

/**
 * Given an axis (with { family, params }), return a sections array that
 * canonicalises the axis's geometry chain.  The returned axis is shallow-
 * cloned; `axis.params` is not mutated.
 *
 * Rules (OSSE):
 *   throatExtLength > 0  → prepend conical throat section (length = extLen)
 *   slotLength > 0       → append straight throat section AFTER the conical
 *                          one (matches `calculateOSSE`'s branch order:
 *                          `z <= extLen` → conical, then slot, then body).
 *   Body length = L; body params have the legacy throat keys stripped and
 *   `r0` rewritten to r0Main (the body's throat radius after the throat
 *   extension adds r0Base · tan(extAngle)).
 *
 * Rules (Classical + flare2):
 *   classicalFlare2Enabled → split the family's L into a body section of
 *   length `start * L` and a flare mouth section of length `(1 - start) * L`.
 *   The body evaluator sees flare2 disabled; the mouth evaluator replicates
 *   the existing conicalRadius flare2 math standalone.  This preserves
 *   byte identity with the pre-sections evaluator when the legacy math is
 *   applied to a contiguous global z.
 */
// Families whose body formula already produces a natural mouth treatment
// (e.g. R-OSSE's roundover is baked into its r/R term).  The UI hides the
// mouth dropdown for these families; the engine mirrors that by stripping
// any stray mouth sections at migration time so a prior-authored
// Flare/Roundover on an R-OSSE axis is never evaluated.
const NATURAL_MOUTH_FAMILIES = new Set(['R-OSSE']);

/**
 * Bump a roundover mouth section from the legacy length-based shape
 * (quarter-cosine rolled-lip, params = { radius }, section.length =
 * user-authored axial length) into the new arc-based shape (params =
 * { angle, radius }, section.length derived from the arc).
 *
 * Rules:
 *   - If the section already has `angle` on params, leave it alone —
 *     it is already on the new shape.
 *   - Otherwise map to a 90° default (the shared user-facing default —
 *     the old length value does not carry enough information to recover
 *     the original arc sweep, so we accept a one-time visual bump).
 *   - Section.length is rebuilt as radius · angle_rad; downstream code
 *     uses getSectionEffectiveLength so out-of-sync stored length would
 *     be harmless, but keeping it right avoids round-trip confusion.
 */
/**
 * One-time convention flip for flare mouth sections.  Pre-Option-A
 * flares stored `angle` as the full cone apex angle (e.g. 40° → a
 * half-angle-20° cone).  Option A stores the tilt off the mouth plane
 * (e.g. 70°).  Legacy sections are flipped once; sections tagged with
 * angleConvention = 'mouth-plane-v1' are left alone.
 */
function migrateFlareMouth(section) {
  if (!section || section.kind !== SECTION_KINDS.MOUTH || section.type !== MOUTH_TYPES.FLARE) {
    return section;
  }
  const p = section.params || {};
  if (p.angleConvention === 'mouth-plane-v1') return section;
  const legacyAngle = Number(p.angle);
  if (!Number.isFinite(legacyAngle)) {
    return { ...section, params: { ...p, angleConvention: 'mouth-plane-v1' } };
  }
  const newAngle = clamp(90 - legacyAngle / 2, 1, 90);
  return {
    ...section,
    params: { ...p, angle: newAngle, angleConvention: 'mouth-plane-v1' },
  };
}

function migrateRoundoverMouth(section) {
  if (!section || section.kind !== SECTION_KINDS.MOUTH || section.type !== MOUTH_TYPES.ROUNDOVER) {
    return section;
  }
  const p = section.params || {};
  if (p.angle !== undefined && p.angle !== null && p.angle !== '') {
    // Already on the new shape — if length is stale, re-derive.
    const spec = resolveRoundoverSpec(section);
    if (spec && Math.abs(Number(section.length) - spec.arcLength) > 1e-6) {
      return { ...section, length: spec.arcLength };
    }
    return section;
  }
  const radius = Math.max(0, evalOr(p.radius, 8));
  const angle = DEFAULT_SWEEP_DEG;
  const length = roundoverArcLength(radius, angle);
  return {
    ...section,
    length,
    params: { ...p, angle, radius },
  };
}

export function migrateAxisToSections(axis) {
  if (!axis || !axis.family || !axis.params) return axis;
  if (Array.isArray(axis.sections) && axis.sections.length > 0) {
    let next = axis;
    // Natural-mouth families: strip any mouth section that may have been
    // stored from a pre-redesign config or from a family switch that
    // didn't clear sections.  Returns a new axis if anything was removed.
    if (NATURAL_MOUTH_FAMILIES.has(axis.family)) {
      const cleaned = axis.sections.filter((s) => s.kind !== SECTION_KINDS.MOUTH);
      if (cleaned.length !== axis.sections.length) {
        next = { ...axis, sections: cleaned };
      }
    }
    // Roundover mouth units migration: pre-redesign roundover sections
    // stored an explicit `length` (mm) without an `angle` (°).  Map them
    // to the new angle+radius contract by assigning the default 90° sweep
    // and recomputing length from (radius, 90°).  The geometric result
    // will differ from the old quarter-cosine shape — a one-time,
    // known-incompatible units bump; the UI slider and the geometry
    // now agree on a sharper arc rollover.
    const rebuilt = next.sections
      .map((s) => migrateRoundoverMouth(s))
      .map((s) => migrateFlareMouth(s));
    const anyChanged = rebuilt.some((s, i) => s !== next.sections[i]);
    if (anyChanged) next = { ...next, sections: rebuilt };
    return next;
  }

  const p = axis.params;
  const family = axis.family;

  const sections = [];

  if (family === 'OSSE') {
    const extLen = Math.max(0, evalOr(p.throatExtLength, 0));
    const extAngle = evalOr(p.throatExtAngle, 0);
    const slotLen = Math.max(0, evalOr(p.slotLength, 0));
    const r0Base = evalOr(p.r0, 0);
    const r0Main = r0Base + extLen * Math.tan(toRad(extAngle));

    if (extLen > 0) {
      sections.push({
        kind: SECTION_KINDS.THROAT,
        type: THROAT_TYPES.CONICAL,
        length: extLen,
        params: {
          angle: extAngle,
          // `matchTangent` is a forward-looking flag; legacy OSSE throat
          // extensions did NOT enforce tangent continuity — the junction
          // between extension and body is linear.  Default false preserves
          // byte identity; the user can opt in via the UI in 3B.
          matchTangent: false,
        },
      });
    }
    if (slotLen > 0) {
      sections.push({
        kind: SECTION_KINDS.THROAT,
        type: THROAT_TYPES.STRAIGHT,
        length: slotLen,
        params: {},
      });
    }

    // Body section.  Length = original L; params have the legacy throat keys
    // stripped and r0 rewritten to r0Main so the body evaluator behaves
    // identically whether called with global z or section-local z.
    const bodyLength = Math.max(0, evalOr(p.L, 0));
    const bodyParams = stripKeys(p, OSSE_BODY_STRIPPED_KEYS);
    bodyParams.r0 = r0Main;
    // Stash the original r0 on the body section for round-tripping back to
    // legacy flat params (save/load).  Phase 4+ work.  Not consulted by the
    // evaluator.
    sections.push({
      kind: SECTION_KINDS.BODY,
      type: family,
      length: bodyLength,
      params: bodyParams,
      legacyParams: { r0: r0Base, throatExtLength: extLen, throatExtAngle: extAngle, slotLength: slotLen },
    });
  } else if (family === 'Classical' && Number(p.classicalFlare2Enabled)) {
    // Classical with Flare2 enabled — carve the tail off the body into a
    // flare mouth section so the body evaluator stops doing the flare math.
    const L = evalOr(p.L, 200);
    const start = Math.max(0, Math.min(1, evalOr(p.classicalFlare2Start, 0.5)));
    const bodyLength = start * L;
    const mouthLength = L - bodyLength;

    const bodyParams = stripKeys(p, CLASSICAL_BODY_STRIPPED_KEYS);
    sections.push({
      kind: SECTION_KINDS.BODY,
      type: family,
      length: bodyLength,
      params: bodyParams,
      legacyParams: {
        classicalFlare2Enabled: Number(p.classicalFlare2Enabled),
        classicalFlare2Angle: evalOr(p.classicalFlare2Angle, 40),
        classicalFlare2Start: start,
        classicalFlare2Blend: Number(p.classicalFlare2Blend) || 0,
        L,
      },
    });

    // Legacy classicalFlare2Angle is the full cone apex angle.  Option A
    // wants the angle measured from the mouth plane.  Convert:
    //   new_angle = 90° − (old_angle / 2)
    // e.g. old 40° (half-angle 20°) → new 70°.
    const legacyFlareAngle = evalOr(p.classicalFlare2Angle, 40);
    const newFlareAngle = 90 - legacyFlareAngle / 2;
    sections.push({
      kind: SECTION_KINDS.MOUTH,
      type: MOUTH_TYPES.FLARE,
      length: mouthLength,
      params: {
        angle: newFlareAngle,
        blend: Number(p.classicalFlare2Blend) === 1 ? 'smooth' : 'sharp',
        start,
        // The smooth blend width in the legacy math is max(L*0.05, 1) using
        // the ORIGINAL total L, not the mouth's own length.  Stash it here
        // so the standalone mouth evaluator can reproduce it exactly.
        fullLength: L,
        // Flag this section so re-migrations don't flip the convention
        // again.  Pre-redesign flare sections (if any) lack this flag
        // and will get a one-time convention flip in migrateFlareMouth.
        angleConvention: 'mouth-plane-v1',
      },
    });
  } else {
    // No sections-specific migration for this family — just a single body
    // section spanning its full length.  Length falls back to params.L /
    // family defaults for the UI cursor; the evaluator dispatches to the
    // family's own length rules.
    sections.push({
      kind: SECTION_KINDS.BODY,
      type: family,
      length: computeBodyLength(family, p),
      params: p,
    });
  }

  return { ...axis, sections };
}

/**
 * Body length for families that don't get section-specific migration
 * (R-OSSE, Lookup, plain Classical without Flare2).  Mirrors the legacy
 * getProfileLength() up to the slice math handled elsewhere.
 *
 * R-OSSE does NOT carry an explicit L parameter — its length is derived
 * from (R, r0, k, a, a0, tmax).  Using `params.L || 200` drove masterLength
 * to 200 regardless of the real R-OSSE length (typically ~120 at defaults),
 * which made the radius clamp to the mouth for z > Lreal and produced a
 * cylindrical bowl instead of the characteristic flare.  Delegate to the
 * family-specific length function so the real axial extent is used.
 */
export function computeBodyLength(family, params) {
  switch (family) {
    case 'OSSE':
      return Math.max(0, evalOr(params.L, 0));
    case 'R-OSSE':
      return Math.max(0, computeRosseDerivedLength(params));
    case 'Classical':
      return Math.max(0, evalOr(params.L, 200));
    case 'LOOKUP':
      return Math.max(0, evalOr(params.L, 200));
    default:
      return Math.max(0, evalOr(params.L, 200));
  }
}

/**
 * R-OSSE's axial length at t = tmax.  R-OSSE has no explicit L parameter;
 * its length is derived from (R, r0, k, a, a0, tmax) through the R-OSSE
 * evaluator.  Matches getProfileLength('R-OSSE', params) byte-for-byte.
 */
function computeRosseDerivedLength(params) {
  const tmax = evalOr(params.tmax, 1.0);
  const profile = calculateROSSE(tmax, 0, params);
  return Number.isFinite(profile?.x) ? profile.x : 0;
}

// ---------------------------------------------------------------------------
// Evaluator — sections-aware radius lookup
// ---------------------------------------------------------------------------

/**
 * Evaluate an axis's sections at an axial station z (mm from throat).
 *
 * @param {object} axis        — the axis node (must have .sections)
 * @param {number} z           — axial station, 0 at throat
 * @param {object} compiledParams — compiled axis.params (expressions → fns)
 * @param {Function} familyEvaluator — callback (family, z, compiledParams) → radius
 * @returns {number} radius at z
 *
 * Phase 3B: throat and mouth section evaluators compute radii standalone
 * (not via the family evaluator).  The body evaluator still dispatches to
 * the family function, but with `localZ` and a stripped params clone — the
 * body sees a pure body, no legacy throat/flare slicing.
 */
export function evaluateAxisSections(axis, z, compiledParams, familyEvaluator, p = 0) {
  const sections = axis.sections;
  if (!Array.isArray(sections) || sections.length === 0) {
    return familyEvaluator(axis.family, z, compiledParams, p);
  }

  // Resolve the body's exit tangent once so mouth-roundover arc lengths
  // (both in total-length and in per-section cursor math) match the
  // physical sweep the evaluator will produce.  Without this the chain
  // undercounts whenever the body exits non-radially.  We cache the
  // resolved (r, slope) so evaluateMouthSection below can reuse it
  // instead of probing the body twice more.
  const bodySection = sections.find((s) => s.kind === SECTION_KINDS.BODY) || null;
  const hasRoundoverMouth = sections.some(
    (s) => s.kind === SECTION_KINDS.MOUTH && s.type === MOUTH_TYPES.ROUNDOVER,
  );
  let bodyTangentAngle; // undefined → resolveRoundoverSpec falls back to π/2
  let cachedBodyMouthInfo = null;
  if (hasRoundoverMouth && bodySection) {
    cachedBodyMouthInfo = getBodyMouthInfo(
      bodySection, axis, compiledParams, familyEvaluator, p,
    );
    bodyTangentAngle = Math.atan2(cachedBodyMouthInfo.slope, 1);
  }

  const totalLength = getSectionsTotalLength(sections, bodyTangentAngle);
  const clampedZ = Math.max(0, Math.min(z, totalLength));

  // Walk sections to find which contains clampedZ, tracking the body section
  // (needed by throat/mouth evaluators to pick up bodyThroatR / bodyMouthR /
  // body exit tangent).
  let activeSection = null;
  let activeStart = 0;
  let cursor = 0;

  for (const section of sections) {
    const end = cursor + getSectionEffectiveLength(section, bodyTangentAngle);
    if (clampedZ <= end + 1e-12) {
      activeSection = section;
      activeStart = cursor;
      break;
    }
    cursor = end;
  }
  if (!activeSection) {
    activeSection = sections[sections.length - 1];
    activeStart = totalLength - getSectionEffectiveLength(activeSection, bodyTangentAngle);
  }

  const localZ = clampedZ - activeStart;

  switch (activeSection.kind) {
    case SECTION_KINDS.THROAT:
      return evaluateThroatSection(
        activeSection,
        localZ,
        bodySection,
        axis,
        compiledParams,
        familyEvaluator,
        p,
      );
    case SECTION_KINDS.BODY:
      return evaluateBodySection(
        activeSection,
        localZ,
        axis,
        compiledParams,
        familyEvaluator,
        p,
      );
    case SECTION_KINDS.MOUTH:
      return evaluateMouthSection(
        activeSection,
        localZ,
        bodySection,
        axis,
        compiledParams,
        familyEvaluator,
        p,
        cachedBodyMouthInfo,
      );
    default:
      return familyEvaluator(axis.family, clampedZ, compiledParams, p);
  }
}

/**
 * Effective parametric length for a single section.  For most sections
 * this is `section.length`; for mouth roundovers it is derived from
 * (radius, angle) because those are the user-visible inputs and
 * `section.length` is redundant / can go stale.
 *
 * `bodyTangentAngle` (radians, measured from +z axis) is forwarded to
 * the roundover arc-length resolver so mouth roundovers get the correct
 * physical sweep for non-radial body exits.  When omitted, the resolver
 * falls back to π/2 (radial body) — an approximation used only when the
 * caller has no body context (e.g. writers/migrations sizing sections
 * before the body is evaluable).
 */
export function getSectionEffectiveLength(section, bodyTangentAngle) {
  if (!section) return 0;
  if (section.kind === SECTION_KINDS.MOUTH && section.type === MOUTH_TYPES.ROUNDOVER) {
    const derived = roundoverSectionArcLength(section, bodyTangentAngle);
    if (derived > 0) return derived;
  }
  const len = Number(section.length);
  return Number.isFinite(len) && len > 0 ? len : 0;
}

/**
 * Total axial/parametric length across all sections in an axis.
 *
 * Note: for a mouth roundover with sweep > 90° the parametric arc length
 * exceeds the axial distance the arc actually reaches — the arc doubles
 * back on itself past the peak.  The multi-axis evaluator handles that
 * by sampling parametrically; scalar callers (computeCoverageAngle,
 * multi-axis radius blend) see the arc as a regular section of the
 * full parametric length.  This is the right contract for the blend
 * because `t = 1` should still reach the arc endpoint, not the peak.
 */
export function getSectionsTotalLength(sections, bodyTangentAngle) {
  if (!Array.isArray(sections)) return 0;
  let total = 0;
  for (const section of sections) {
    total += getSectionEffectiveLength(section, bodyTangentAngle);
  }
  return total;
}

// ---------------------------------------------------------------------------
// Section-local radius seams
// ---------------------------------------------------------------------------

/**
 * Resolve the throat radius that a throat section should ramp up to — i.e.
 * the body's radius at z=0 (after params stripping r0 → r0Main).
 */
function getBodyThroatRadius(bodySection) {
  if (!bodySection || !bodySection.params) return 0;
  const r0 = bodySection.params.r0;
  return evalOr(r0, 0);
}

/**
 * Resolve the body's mouth radius and mouth tangent.  Used by mouth sections
 * that need to start where the body ends with the correct slope.
 */
function getBodyMouthInfo(bodySection, axis, compiledParams, familyEvaluator, p = 0) {
  if (!bodySection || bodySection.length <= 0) {
    return { r: getBodyThroatRadius(bodySection), slope: 0 };
  }
  const compiledBody = compileBodyParams(bodySection.params, compiledParams, axis.family);
  const bodyLen = bodySection.length;
  const rEnd = familyEvaluator(axis.family, bodyLen, compiledBody, p);
  // Finite-difference slope at the mouth.  A small offset (0.1 mm or 0.1 %
  // of body length, whichever is larger) keeps the derivative numerically
  // stable across wildly-different L values.
  const eps = Math.max(0.1, bodyLen * 1e-3);
  const rBack = familyEvaluator(axis.family, Math.max(0, bodyLen - eps), compiledBody, p);
  const slope = (rEnd - rBack) / eps;
  return { r: rEnd, slope };
}

/**
 * Resolve the body's slope at its throat (z = 0) via forward finite
 * difference.  Used by the quadratic throat evaluator so the adapter can
 * match the body's tangent at the body/adapter junction — giving the
 * adapter's body-end a slope equal to the body's natural "coverage at
 * throat" rather than a hardcoded value.
 */
function getBodyThroatTangent(bodySection, axis, compiledParams, familyEvaluator, p = 0) {
  if (!bodySection || bodySection.length <= 0) return 0;
  const compiledBody = compileBodyParams(bodySection.params, compiledParams, axis.family);
  const bodyLen = bodySection.length;
  const eps = Math.max(0.1, bodyLen * 1e-3);
  const r0 = familyEvaluator(axis.family, 0, compiledBody, p);
  const rEps = familyEvaluator(axis.family, eps, compiledBody, p);
  const slope = (rEps - r0) / eps;
  return Number.isFinite(slope) ? slope : 0;
}

// ---------------------------------------------------------------------------
// Body section dispatch — needs a compiled-params clone that reflects the
// stripped body params (so the family evaluator doesn't re-apply the legacy
// throat/slot/flare math).
// ---------------------------------------------------------------------------

// A tiny WeakMap cache so we don't rebuild the compiled body params on every
// radius call.  Keyed on the body section's raw params object (stable across
// re-resolves because migration produces a fresh params object per migrate).
const _bodyCompiledCache = new WeakMap();

function compileBodyParams(bodyParams, axisCompiledParams, family) {
  if (!bodyParams) return axisCompiledParams;
  // Fast path: if the body params share identity with the axis params, nothing
  // was stripped and we can reuse the compiled set.
  if (bodyParams === axisCompiledParams) return axisCompiledParams;
  const cached = _bodyCompiledCache.get(bodyParams);
  if (cached) return cached;
  // Inline compile: start from the axis's already-compiled params (which
  // carry the expression functions for `a`, `s`, etc.) and apply ONLY the
  // family-specific overrides the migration performs.  We must NOT blanket-
  // copy bodyParams on top, because bodyParams came from a `stripKeys(raw)`
  // clone — its values are raw strings/numbers and would clobber the
  // compiled functions in axisCompiledParams, producing NaN at evaluation.
  const merged = { ...axisCompiledParams };
  if (family === 'OSSE') {
    // OSSE body sees a zeroed throat extension / slot so the family
    // evaluator stops doing its own slicing, and its r0 is replaced with
    // r0Main (the radius at the body's throat after the extension).
    merged.throatExtLength = 0;
    merged.slotLength = 0;
    merged.throatExtAngle = 0;
    if (bodyParams.r0 !== undefined) merged.r0 = bodyParams.r0;
  } else if (family === 'Classical') {
    merged.classicalFlare2Enabled = 0;
  }
  _bodyCompiledCache.set(bodyParams, merged);
  return merged;
}

// ---------------------------------------------------------------------------
// Section evaluators
// ---------------------------------------------------------------------------

function evaluateThroatSection(section, localZ, bodySection, axis, compiledParams, familyEvaluator, p = 0) {
  const bodyThroatR = getBodyThroatRadius(bodySection);
  const type = section.type;

  switch (type) {
    case THROAT_TYPES.STRAIGHT:
      return evaluateStraightThroat(localZ, section, bodyThroatR);
    case THROAT_TYPES.CONICAL: {
      // Conical `matchTangent` uses the body's throat slope for the adapter
      // itself — compute it lazily only when the flag is set.
      const params = section.params || {};
      const bodyThroatTangent = params.matchTangent
        ? getBodyThroatTangent(bodySection, axis, compiledParams, familyEvaluator, p)
        : null;
      return evaluateConicalThroat(localZ, section, bodyThroatR, bodyThroatTangent);
    }
    case THROAT_TYPES.OSSE: {
      // OS throat uses the OS hyperbola from the OS-SE paper.  By default
      // it derives the nominal OS angle so the adapter is C1-continuous
      // with the body; an explicit `coverageAngle` can override that.
      const bodyThroatTangent = getBodyThroatTangent(
        bodySection, axis, compiledParams, familyEvaluator, p,
      );
      return evaluateOsseThroat(localZ, section, bodyThroatR, bodyThroatTangent);
    }
    case THROAT_TYPES.QUADRATIC: {
      // Quadratic throat matches the body's throat slope at the body-side
      // junction by construction (user specifies the driver-side angle;
      // the body-side is derived).
      const bodyThroatTangent = getBodyThroatTangent(
        bodySection, axis, compiledParams, familyEvaluator, p,
      );
      return evaluateQuadraticThroat(localZ, section, bodyThroatR, bodyThroatTangent);
    }
    default:
      // Unknown type → degenerate to a straight cylinder at bodyThroatR.
      return bodyThroatR;
  }
}

function evaluateBodySection(section, localZ, axis, compiledParams, familyEvaluator, p = 0) {
  const bodyCompiled = compileBodyParams(section.params, compiledParams, axis.family);
  return familyEvaluator(axis.family, localZ, bodyCompiled, p);
}

function evaluateMouthSection(
  section, localZ, bodySection, axis, compiledParams, familyEvaluator, p = 0,
  cachedBodyMouthInfo = null,
) {
  const { r: bodyMouthR, slope: bodyMouthSlope } = cachedBodyMouthInfo
    || getBodyMouthInfo(bodySection, axis, compiledParams, familyEvaluator, p);
  const type = section.type;
  switch (type) {
    case MOUTH_TYPES.FLARE:
      return evaluateFlareMouth(localZ, section, bodyMouthR, bodyMouthSlope);
    case MOUTH_TYPES.ROUNDOVER:
      return evaluateRoundoverMouth(localZ, section, bodyMouthR, bodyMouthSlope);
    default:
      return bodyMouthR;
  }
}

// ---------------------------------------------------------------------------
// Throat-type evaluators — each is a pure function of (localZ, section,
// bodyThroatR) and returns a radius.  Tested independently.
// ---------------------------------------------------------------------------

/**
 * Straight cylindrical throat.  Constant radius = bodyThroatR.
 */
export function evaluateStraightThroat(_localZ, _section, bodyThroatR) {
  return bodyThroatR;
}

/**
 * Conical throat adapter — linear ramp from a driver-side radius up to the
 * body's throat radius.  The half-angle `angle` (degrees) defines the slope;
 * the section's length defines the driver-side start radius:
 *
 *     startR = bodyThroatR − length · tan(angle)
 *     r(localZ) = startR + localZ · tan(angle)
 *
 * At localZ = length this equals bodyThroatR — the ramp meets the body.
 *
 * When `params.matchTangent` is true, the conical's exit slope is forced to
 * equal the body's throat slope.  The shipped legacy OSSE throat extension
 * does NOT do this (it transitions linearly into the body at a fixed angle
 * — the slope discontinuity is accepted for byte identity), so migrated
 * configs default to matchTangent = false.
 */
export function evaluateConicalThroat(localZ, section, bodyThroatR, bodyThroatTangent = null) {
  const length = section.length || 0;
  const p = section.params || {};
  const angleRad = toRad(evalOr(p.angle, 0));
  const slope = (p.matchTangent && Number.isFinite(bodyThroatTangent))
    ? bodyThroatTangent
    : Math.tan(angleRad);
  const startR = bodyThroatR - length * slope;
  return startR + localZ * slope;
}

function hasParamValue(params, key) {
  return params
    && params[key] !== undefined
    && params[key] !== null
    && params[key] !== '';
}

function solveTangentMatchedOsseStartRadius(length, bodyThroatR, driverSlope, bodySlope) {
  // Endpoint + tangent equations for:
  //   r(z)^2 = r0^2 + 2*r0*m0*z + ma^2*z^2
  //   r'(L) = m1
  // reduce to:
  //   r0^2 + L*m0*r0 + L*R*m1 - R^2 = 0
  const b = length * driverSlope;
  const c = length * bodyThroatR * bodySlope - bodyThroatR * bodyThroatR;
  const disc = b * b - 4 * c;
  if (disc < 0) return NaN;
  return (-b + Math.sqrt(disc)) / 2;
}

function solveCoverageOsseStartRadius(length, bodyThroatR, driverSlope, osSlope) {
  // Endpoint equation with explicit OS nominal slope ma:
  //   R^2 = r0^2 + 2*r0*m0*L + ma^2*L^2
  const b = 2 * length * driverSlope;
  const c = (osSlope * length) ** 2 - bodyThroatR * bodyThroatR;
  const disc = b * b - 4 * c;
  if (disc < 0) return NaN;
  return (-b + Math.sqrt(disc)) / 2;
}

/**
 * OS throat adapter — a finite segment of the generalized OS throat curve
 * from the OS-SE paper, used as a pre-body adapter:
 *
 *   r(z) = sqrt(r0^2 + 2*r0*z*tan(a0) + z^2*tan(a)^2)
 *
 * Parameters:
 *   angle         — driver-side tangent angle a0 in degrees. 0° gives a
 *                   flat start at the driver side.
 *   coverageAngle — optional OS nominal angle a in degrees. When omitted,
 *                   the nominal angle is derived so r'(length) matches the
 *                   body's throat tangent.
 *
 * The driver-side radius r0 is derived from the section length and the
 * body-side endpoint, matching the Quadratic throat's "area is an output"
 * convention. This keeps the adapter useful for scripted parameter sweeps where
 * throat area is intentionally varied.
 */
export function evaluateOsseThroat(localZ, section, bodyThroatR, bodyThroatTangent = 0) {
  const length = section.length || 0;
  if (length <= 0) return bodyThroatR;

  const p = section.params || {};
  const driverSlope = Math.tan(toRad(evalOr(p.angle, 0)));
  const z = Math.max(0, Math.min(localZ, length));
  let startR;
  let osSlopeSquared;

  if (hasParamValue(p, 'coverageAngle')) {
    const osSlope = Math.tan(toRad(evalOr(p.coverageAngle, 0)));
    startR = solveCoverageOsseStartRadius(length, bodyThroatR, driverSlope, osSlope);
    osSlopeSquared = osSlope * osSlope;
  } else {
    const bodySlope = Number.isFinite(bodyThroatTangent) ? bodyThroatTangent : 0;
    startR = solveTangentMatchedOsseStartRadius(
      length, bodyThroatR, driverSlope, bodySlope,
    );
    osSlopeSquared = (bodyThroatR * bodySlope - startR * driverSlope) / length;
  }

  if (!Number.isFinite(startR) || startR < 0 || !Number.isFinite(osSlopeSquared)) {
    return evaluateQuadraticThroat(localZ, section, bodyThroatR, bodyThroatTangent);
  }
  if (osSlopeSquared < 0) {
    // A real OS hyperbola cannot satisfy this endpoint/tangent combination.
    // Fall back to the quadratic adapter, which remains well-defined.
    return evaluateQuadraticThroat(localZ, section, bodyThroatR, bodyThroatTangent);
  }

  const under = startR * startR
    + 2 * startR * driverSlope * z
    + osSlopeSquared * z * z;
  return Math.sqrt(Math.max(0, under));
}

/**
 * Quadratic throat adapter — a true quadratic in r (so its slope is linear
 * in z) that interpolates between a user-specified driver-side tangent and
 * the body's own throat tangent.
 *
 * Parameters (on section.params):
 *   angle — driver-side tangent angle in degrees.  0° is a horizontal
 *     start (tangent to the axis at the driver); larger values tilt the
 *     driver entrance outward.  Default 0 — a Hughes-QT-like flat start.
 *
 * Derived:
 *   m0 = tan(angle)              — driver-side slope (user input)
 *   m1 = bodyThroatTangent       — body-side slope (auto-matched to body)
 *   r1 = bodyThroatR             — body-side radius (meets the body at z=L)
 *   r0 = r1 − L · (m0 + m1) / 2  — driver-side radius (DERIVED)
 *
 *   r(z) = r0 + m0 · z + (m1 − m0) · z² / (2L)
 *   r'(z) = m0 + (m1 − m0) · z / L  (slope is linear in z)
 *
 * Continuity: r(0) = r0, r(L) = r1, r'(0) = m0, r'(L) = m1.  Because the
 * body-side slope exactly matches the body's tangent at its throat, there
 * is no tangent discontinuity at the adapter-body junction — the curve
 * flows smoothly into the body's coverage angle.
 *
 * This replaces the old √(z/L) form, which bulged the wrong way — it had
 * near-vertical slope at the driver and near-zero slope at the body-side,
 * producing a "bulge" that opened rapidly into the driver then flattened
 * into the body.  The new form does the opposite: flat (or user-tilted)
 * at the driver, opening to the body's full coverage at the junction —
 * which is what the classic Hughes QT adapter actually looks like.
 *
 * Legacy configs that still set `throatRadius` / `throatR` are tolerated —
 * they are IGNORED in favour of the derived r0 (the old param made no
 * sense under the new formulation; the value is implicit in the length
 * and the two endpoint slopes).
 *
 * The `matchTangent` flag is no longer needed (body-side slope is always
 * matched by construction) and is ignored.
 */
export function evaluateQuadraticThroat(localZ, section, bodyThroatR, bodyThroatTangent = 0) {
  const length = section.length || 0;
  if (length <= 0) return bodyThroatR;
  const p = section.params || {};
  const m0 = Math.tan(toRad(evalOr(p.angle, 0)));
  const m1 = Number.isFinite(bodyThroatTangent) ? bodyThroatTangent : 0;
  const r0 = bodyThroatR - length * (m0 + m1) / 2;
  const clamped = Math.max(0, Math.min(localZ, length));
  return r0 + m0 * clamped + (m1 - m0) * clamped * clamped / (2 * length);
}

// ---------------------------------------------------------------------------
// Mouth-type evaluators
// ---------------------------------------------------------------------------

/**
 * Flare mouth — a conical extension of the horn mouth.
 *
 * Angle convention (Option A, matching the roundover):
 *   user  0° → flare wall lies in the mouth plane (flat disk; slope ∞).
 *   user 90° → flare wall lies along the horn axis (cylinder; slope 0).
 *   intermediate → a cone tilted by that many degrees off the mouth plane.
 *
 * Equivalently, the end-tangent angle from the +z axis is π/2 − user_angle,
 * and the wall's slope dr/dz = 1 / tan(user_angle).
 *
 * Sharp blend:
 *     r(localZ) = bodyMouthR + localZ · (1 / tan(user_angle))
 *
 * Smooth blend: analytic integral of a tanh-blended slope transitioning
 * from the body's exit slope (bodyMouthSlope) to the flare's slope over a
 * blend width.  Blend width = max(fullLength · 0.05, 1) where fullLength
 * is whatever the migration stashed (legacy: the pre-split horn length).
 *
 * Pure flat-disk (user_angle = 0°) is excluded because r(z) is then
 * multi-valued.  The minimum angle is clamped to 1°, which is nearly flat
 * in practice but keeps the scalar evaluator well-defined.
 */
const FLARE_ANGLE_MIN_DEG = 1;
const FLARE_ANGLE_MAX_DEG = 90;

function flareSlope(angleDeg) {
  const a = clamp(angleDeg, FLARE_ANGLE_MIN_DEG, FLARE_ANGLE_MAX_DEG);
  const t = Math.tan(toRad(a));
  return t > 0 ? 1 / t : 0;
}

export function evaluateFlareMouth(localZ, section, bodyMouthR, bodyMouthSlope = 0) {
  const p = section.params || {};
  const angleDeg = evalOr(p.angle, 30);
  const slope2 = flareSlope(angleDeg);
  const smooth = (p.blend === 'smooth') || Number(p.blend) === 1;

  if (!smooth) {
    return bodyMouthR + localZ * slope2;
  }

  const fullLength = evalOr(p.fullLength, section.length || 0);
  const blendWidth = Math.max(fullLength * 0.05, 1);
  const dz = localZ;
  const sigma = dz / blendWidth;
  const slope1 = bodyMouthSlope;
  const logCosh2Sigma = Math.log(Math.cosh(2 * sigma));
  const blendIntegral = blendWidth * (0.5 * sigma + 0.25 * logCosh2Sigma);
  return bodyMouthR + slope1 * dz + (slope2 - slope1) * blendIntegral;
}

/**
 * Roundover mouth — circular-arc rollover.
 *
 * Parameters (section.params):
 *   angle  — sweep angle in degrees, 0 ≤ angle ≤ 180.  Default 90°.
 *            Think of it as "how far around the arc the roundover goes":
 *              0°   = no roundover
 *              90°  = quarter circle, ends tangent vertical at max radius
 *              180° = half circle, wraps all the way back over itself
 *   radius — arc radius in mm.  Default 8.
 *
 * Section `length` is the arc length (radius · angleRad).  That lets the
 * scalar sections-chain callers treat the roundover as a regular section
 * whose parametric length is the arc length — though past the 90° sweep
 * the axial extent contracts again (wrap-back).  Callers that need a
 * faithful (x, r) representation past the peak should use the parametric
 * evaluator `sampleRoundoverMouth` instead — the scalar form here returns
 * radius-as-a-function-of-arc-length (i.e., the arc's r-component at
 * localZ = arc length from the body exit), not r(axial-z).  For the
 * monotonic-sweep case (angle ≤ 90°) these are indistinguishable.
 *
 * The arc is C1-continuous with the body at the junction: start tangent
 * = (bodyMouthSlope's direction).  Historically `bodyMouthSlope` was
 * ignored and the rollover started with zero slope (C1 only with a flat
 * body exit).  The arc form takes the slope into account so a flaring
 * body (conical, OSSE with wide coverage) joins cleanly.
 */
export function evaluateRoundoverMouth(localZ, section, bodyMouthR, bodyMouthSlope = 0) {
  const bodyTangentAngle = Math.atan2(bodyMouthSlope, 1);
  const spec = resolveRoundoverSpec(section, bodyTangentAngle);
  if (!spec) return bodyMouthR;
  const arc = makeRoundoverArc({
    x0: 0,
    r0: bodyMouthR,
    tangentDx: 1,
    tangentDr: bodyMouthSlope,
    radius: spec.radius,
    sweepRad: spec.sweepRad,
  });
  const s = clamp(localZ, 0, arc.arcLength);
  return arc.evaluate(s).r;
}

/**
 * Parametric sample of the roundover mouth at arc-length position s.
 * Returns { x, r } relative to the body exit.  Use this from the
 * multi-axis evaluator / 2D preview when the roundover sweep can exceed
 * 90° (and therefore produces non-monotonic x(s)).
 */
export function sampleRoundoverMouth(localS, section, bodyMouthR, bodyMouthSlope = 0) {
  const bodyTangentAngle = Math.atan2(bodyMouthSlope, 1);
  const spec = resolveRoundoverSpec(section, bodyTangentAngle);
  if (!spec) return { x: 0, r: bodyMouthR };
  const arc = makeRoundoverArc({
    x0: 0,
    r0: bodyMouthR,
    tangentDx: 1,
    tangentDr: bodyMouthSlope,
    radius: spec.radius,
    sweepRad: spec.sweepRad,
  });
  return arc.evaluate(clamp(localS, 0, arc.arcLength));
}

/**
 * Resolve the roundover mouth's arc spec from a section.  Returns null
 * when the section has degenerate parameters (radius ≤ 0 or angle ≤ 0),
 * in which case callers collapse the roundover to the body's exit point.
 */
export function resolveRoundoverSpec(section, bodyTangentAngle = Math.PI / 2) {
  const p = section.params || {};
  const radius = Math.max(0, evalOr(p.radius, 8));
  const angleDeg = clamp(evalOr(p.angle, DEFAULT_SWEEP_DEG), 0, MAX_SWEEP_DEG);
  // "Off" sentinels: zero radius OR zero angle collapses to no roundover.
  // Without this short-circuit, user angle = 0° with a non-radial body
  // tangent would produce a non-zero physical sweep (sweep = π/2 −
  // bodyTangentAngle), which is surprising — users expect 0° to mean
  // "identity".  Accept both zero-radius and zero-angle as opt-outs.
  if (radius <= 0 || angleDeg <= 0) return null;
  // User angle is measured from the mouth plane:
  //   user 0°  → end tangent lies in the mouth plane (purely radial,
  //              perpendicular to the horn axis).
  //   user 90° → end tangent lies along the horn axis (purely backward
  //              toward the throat; parallel to the enclosure sides).
  //
  // End-tangent angle from +z axis = π/2 + user_angle.  The physical
  // sweep is whatever is needed to rotate the body's exit tangent into
  // that end direction — it depends on the body tangent at the mouth
  // and is therefore not the same as the user angle.
  //
  // When no body context is supplied, we fall back to bodyTangentAngle
  // = π/2 (body exits radially), which makes sweep = user_angle — a
  // reasonable default for sizing callers that don't yet know the body.
  const endTangentAngle = Math.PI / 2 + toRad(angleDeg);
  const sweepRad = clamp(endTangentAngle - bodyTangentAngle, 0, 2 * Math.PI);
  if (radius <= 0 || sweepRad <= 0) return null;
  return { radius, angleDeg, sweepRad, arcLength: radius * sweepRad };
}

/**
 * Arc length that a roundover section occupies along the parametric
 * coordinate chain.  Exported for writers/migrations that need to set
 * section.length consistently with (radius, angle).
 *
 * `bodyTangentAngle` (radians from +z axis) matches what the evaluator
 * uses in evaluateRoundoverMouth (atan2(bodyMouthSlope, 1)).  When the
 * body exits non-radially the physical sweep — and therefore the arc
 * length — differs from the user-authored sweep angle.  When omitted
 * this function falls back to π/2 (body exits radially), matching the
 * resolveRoundoverSpec default.  This approximation is only correct
 * when the caller has no body context; prefer threading the real body
 * tangent from the body section's finite-difference slope.
 */
export function roundoverSectionArcLength(section, bodyTangentAngle) {
  const spec = resolveRoundoverSpec(section, bodyTangentAngle);
  return spec ? spec.arcLength : 0;
}

// ---------------------------------------------------------------------------
// Lightweight section-construction helpers — handy for tests and for the
// Phase 3B UI dropdowns.  They don't enforce validation; callers should
// pass sane parameters.
// ---------------------------------------------------------------------------

/**
 * If the axis has a mouth-roundover section with non-degenerate params,
 * return a parametric-tail descriptor the profile-system evaluator can
 * use to extend the profile past the body's monotonic-z region into the
 * arc.  `null` when the axis has no parametric tail (no roundover, or a
 * degenerate one).
 *
 * bodyArcLength — cumulative effective length of throat + body (the
 *                 scalar chain up to the mouth junction).
 * tailArcLength — length the roundover arc adds to the parametric total.
 * section       — the roundover section itself, forwarded for the
 *                 evaluator to call sampleRoundoverMouth.
 * bodySection   — the axis's body section, used to probe the body's
 *                 exit point and tangent at the junction.
 */
export function getParametricTail(axis, bodyTangentAngle) {
  const sections = Array.isArray(axis?.sections) ? axis.sections : [];
  if (sections.length === 0) return null;
  const mouth = sections.find((s) => s.kind === SECTION_KINDS.MOUTH);
  if (!mouth || mouth.type !== MOUTH_TYPES.ROUNDOVER) return null;
  const spec = resolveRoundoverSpec(mouth, bodyTangentAngle);
  if (!spec) return null;

  const bodySection = sections.find((s) => s.kind === SECTION_KINDS.BODY) || null;
  let bodyArcLength = 0;
  for (const s of sections) {
    if (s === mouth) break;
    // Pre-mouth sections (throat, body) don't use bodyTangentAngle, but we
    // forward it defensively in case a future section kind needs it.
    bodyArcLength += getSectionEffectiveLength(s, bodyTangentAngle);
  }

  return {
    mouthSection: mouth,
    bodySection,
    bodyArcLength,
    tailArcLength: spec.arcLength,
    spec,
  };
}

export function makeThroatSection(type, length, params = {}) {
  return { kind: SECTION_KINDS.THROAT, type, length, params };
}

export function makeBodySection(family, length, params = {}) {
  return { kind: SECTION_KINDS.BODY, type: family, length, params };
}

export function makeMouthSection(type, length, params = {}) {
  return { kind: SECTION_KINDS.MOUTH, type, length, params };
}
