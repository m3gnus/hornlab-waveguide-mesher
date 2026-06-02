export { DEFAULTS, HORN_PROFILES, GUIDING_CURVES, MORPH_TARGETS } from './constants.js';
export { applyMorphing, getRoundedRectRadius } from './morphing.js';
export { buildWaveguideMesh, buildHornMesh } from './buildWaveguideMesh.js';
export { validateParameters } from './profiles/validation.js';
export { calculateOSSE, computeOsseRadius } from './profiles/osse.js';
export { calculateROSSE } from './profiles/rosse.js';
export { getGuidingCurveRadius } from './profiles/guidingCurve.js';
export { addEnclosureGeometry } from './mesh/enclosure.js';
export { addFreestandingWallGeometry } from './mesh/freestandingWall.js';
export {
  PROFILE_MODES,
  INTERPOLATION_MODES,
  resolveProfileSystem,
  evaluateMultiAxisProfile,
  createDefaultProfileSystem,
  sampleMouthCurve,
  sampleAxisProfile,
  getProfileLength,
  defaultCrossSection
} from './profileSystem.js';
export {
  CROSS_SECTION_MODES,
  applyCrossSection,
  superellipseHalfDims,
  superellipseRadius,
  gammaFactor,
  transitionAt,
  sanitizeTimeline,
  evalTimelineAt
} from './crossSection.js';
export { pchipEval } from './interp.js';
