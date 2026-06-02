import { orientMeshConsistently } from '../meshIntegrity.js';
import { validateMeshQuality } from '../quality.js';
import { DEFAULTS, MORPH_TARGETS } from './constants.js';
import { parseQuadrants } from '../common.js';
import { resolveProfileSystem, stripLegacyMorphKeys } from './profileSystem.js';
import { buildAngleList } from './mesh/angles.js';
import { addEnclosureGeometry } from './mesh/enclosure.js';
import { addFreestandingWallGeometry } from './mesh/freestandingWall.js';
import {
  buildMorphTargets,
  buildShrinkData,
  computeAdaptivePhiCounts,
  computeMouthExtents,
  createAdaptiveFanIndices,
  createAdaptiveRingVertices,
  createHornIndices,
  createRingVertices
} from './mesh/horn.js';
import { buildSliceMap } from './mesh/sliceMap.js';
import { generateThroatSource } from './mesh/source.js';

function clampSegmentCount(value, fallback, min) {
  const n = Number(value);
  if (!Number.isFinite(n) || n < min) return fallback;
  return Math.round(n);
}

// Exported so regression tests can exercise the hard-error promotion directly
// without having to corrupt a builder mesh post-hoc.
export function enforceQualityIssues(quality) {
  // Warnings — surface every metric we compute, not just a subset (the previous
  // logger silently dropped sameDirectionSharedEdges, seam, and source connectivity).
  if (quality.degenerateTriangles > 0) {
    console.error(`[Geometry] Degenerate triangles detected: ${quality.degenerateTriangles}`);
  }
  if (quality.duplicateVertices > 0) {
    console.error(
      `[Geometry] Duplicate coincident vertices detected: ${quality.duplicateVertices} (phantom-boundary risk; seam stitching should reuse indices rather than duplicate positions)`
    );
  }
  // Hard failures — these indicate topologically broken meshes that produce
  // visibly wrong rendering and corrupt BEM payloads. The data was previously
  // computed but discarded, masking real bugs.
  const errors = [];
  if (quality.tJunctions !== null && quality.tJunctions > 0) {
    errors.push(
      `${quality.tJunctions} T-junction(s) (vertex lies on another triangle's edge — breaks BEM edge-continuity, causes visible cracks under subdivision)`
    );
  }
  if (quality.nonManifoldEdges > 0) {
    errors.push(`${quality.nonManifoldEdges} non-manifold edge(s) (shared by >2 triangles)`);
  }
  if (quality.sameDirectionSharedEdges > 0) {
    errors.push(`${quality.sameDirectionSharedEdges} shared edge(s) with same-direction winding (inverted-normal patches)`);
  }
  if (quality.seam && quality.seam.sameDirection > 0) {
    errors.push(`${quality.seam.sameDirection} horn↔enclosure seam edge(s) with same-direction winding`);
  }
  // For closed meshes, negative signed volume = every triangle is wound
  // inside-out relative to the solid. This used to be silently repaired by
  // orientMeshConsistently's preferOutward global flip; now we refuse to
  // ship it so the builder bug gets fixed upstream.
  if (quality.boundaryEdges === 0 && quality.signedVolume !== undefined && quality.signedVolume <= 0) {
    errors.push(
      `closed mesh has signed volume ${quality.signedVolume.toExponential(3)} (≤0); normals are inside-out`
    );
  }
  if (errors.length > 0) {
    throw new Error(`[Geometry] Mesh quality violation:\n  - ${errors.join('\n  - ')}`);
  }
}

function resolveOuterBuildMode(params, options = {}) {
  const encDepth = Number(params.encDepth || 0);
  const wallThickness = Number(params.wallThickness || 0);
  const enclosureRequested = encDepth > 0;
  const enclosureEnabled = options.includeEnclosure !== false;

  if (params.type === 'R-OSSE' && enclosureRequested) {
    throw new Error(
      'R-OSSE enclosure is not supported by the default geometry contract. Use OSSE for enclosures or set encDepth=0.'
    );
  }

  // Classical + enclosure was historically blocked because the OSSE-tuned
  // ray-cast enclosure had not been validated against Classical's slope-broken
  // (flare2) profiles. The enclosure builder consumes the actual mouth ring
  // vertices via ringCount + angleList — both of which are produced uniformly
  // for Classical, so the path is geometrically valid. Allowed through for
  // the rectangular-conical-horn use case (multi-axis Classical + flare2 +
  // manualHV); revisit if specific Classical shapes (e.g. tractrix wrap-back)
  // surface enclosure-builder bugs.
  //
  // Genuine multi-axis (> 1 axis) is similarly allowed: the enclosure
  // ray-cast walks the as-built mouth ring per phi rather than recomputing
  // the profile, so non-axisymmetric mouths compose correctly.

  // (R-OSSE remains blocked above because its morning-glory wrap-back
  // produces a non-monotonic mouth contour that the enclosure stitching
  // does not yet handle.)

  if (enclosureRequested && enclosureEnabled) return 'enclosure';
  if (encDepth <= 0 && wallThickness > 0) return 'freestandingWall';
  return 'bare';
}

function selectQuadrantAngles(fullAngles, quadrantInfo) {
  if (quadrantInfo?.fullCircle) {
    return fullAngles;
  }
  const start = Number(quadrantInfo?.startAngle);
  const end = Number(quadrantInfo?.endAngle);
  if (!Array.isArray(fullAngles) || !Number.isFinite(start) || !Number.isFinite(end)) {
    return fullAngles;
  }

  const eps = 1.0e-12;
  if (start < 0 && end > 0) {
    const negativeBranch = fullAngles
      .filter((angle) => angle >= (Math.PI * 2 + start) - eps)
      .map((angle) => angle - Math.PI * 2);
    const positiveBranch = fullAngles.filter((angle) => angle <= end + eps);
    const selected = [...negativeBranch, ...positiveBranch];
    return selected.length >= 2 ? selected : fullAngles;
  }

  const selected = fullAngles.filter((angle) => angle >= start - eps && angle <= end + eps);
  return selected.length >= 2 ? selected : fullAngles;
}

export function buildWaveguideMesh(params, options = {}) {
  const angularSegments = clampSegmentCount(params.angularSegments, DEFAULTS.ANGULAR_SEGMENTS, 4);
  const lengthSteps = clampSegmentCount(params.lengthSegments, DEFAULTS.LENGTH_SEGMENTS, 1);

  const includeEnclosure = options.includeEnclosure !== false;
  const groupInfo = options.groupInfo ?? (options.collectGroups ? {} : null);

  let meshParams = {
    ...params,
    angularSegments,
    lengthSegments: lengthSteps
  };

  // Resolve multi-axis profile system once for the entire build so that
  // evaluateInnerProfileAt can skip per-vertex re-resolution (Bug 5).
  const resolvedSystem = resolveProfileSystem(meshParams);

  // Phase 4: if resolveProfileSystem migrated a legacy morph into a
  // cross-section timeline, strip the morph keys from meshParams so the
  // legacy applyMorphing path (morphing.js stub) does not double-apply on
  // top of the timeline.  The shrinkData path in mesh/horn.js (OSSE-only,
  // allowShrinkage) is intentionally lost in this migration — it was an
  // OSSE-specific coverage-angle hack that no longer composes with the
  // timeline.  Users who relied on shrinkage can author the equivalent
  // mouth shape via the Shape Timeline directly.
  const csTimeline = resolvedSystem?.crossSection?.timeline;
  const hasTimeline = Array.isArray(csTimeline) && csTimeline.length >= 2;
  if (hasTimeline) {
    meshParams = stripLegacyMorphKeys(meshParams);
  }

  const profileContext = {
    coverageCache: new Map(),
    resolvedSystem
  };

  const samplingMode = options.samplingMode
    ?? meshParams.samplingMode
    ?? meshParams.meshSamplingMode
    ?? (meshParams.athParitySampling === true ? 'ath-parity' : null);
  const sliceMapMode = samplingMode === 'ath-parity' ? 'ath-parity' : options.sliceMapMode;
  const sliceMap = buildSliceMap(meshParams, lengthSteps, {
    mode: sliceMapMode,
    resolvedSystem
  });
  const mouthExtents = computeMouthExtents(meshParams, profileContext);

  const angleListData = buildAngleList(meshParams, mouthExtents, {
    mode: samplingMode
  });
  const quadrantInfo = parseQuadrants(meshParams.quadrants);
  const angleList = selectQuadrantAngles(angleListData.fullAngles, quadrantInfo);
  const ringCount = angleList.length;
  const fullCircle = Boolean(quadrantInfo.fullCircle);

  const morphTarget = Number(meshParams.morphTarget || MORPH_TARGETS.NONE);
  const needsMorphTargets = morphTarget !== MORPH_TARGETS.NONE
    && (!meshParams.morphWidth || !meshParams.morphHeight);
  const morphTargets = needsMorphTargets
    ? buildMorphTargets(meshParams, lengthSteps, angleList, sliceMap, profileContext)
    : null;

  // Coverage-angle-based morph shrinkage (OSSE only, when AllowShrinkage is on).
  // Disabled when there is more than one axis — buildShrinkData is OSSE-specific
  // and uses invertOsseCoverageAngle which assumes a single OSSE profile for
  // all angles.  A one-axis system keeps the legacy single-profile shrinkage
  // path active.
  const meshAxes = meshParams.profileSystem?.axes;
  const isMultiAxis = Array.isArray(meshAxes) && meshAxes.length > 1;
  // Warn once per build when shrinkage is silently disabled by multi-axis —
  // a user moving from single-axis OSSE+Morph to multi-axis loses the feature
  // otherwise without any feedback.
  if (isMultiAxis
      && Number(meshParams.morphAllowShrinkage || 0)
      && Number(meshParams.morphTarget || MORPH_TARGETS.NONE) !== MORPH_TARGETS.NONE) {
    console.warn(
      '[ProfileSystem] Morph.AllowShrinkage is set but the system has '
      + `${meshAxes.length} axes; shrinkage is OSSE single-profile only and is `
      + 'skipped.  Use a single-axis system or author the equivalent mouth '
      + 'shape via the Shape Timeline.'
    );
  }
  const shrinkData = isMultiAxis ? null : buildShrinkData(meshParams, angleList, profileContext);

  // Adaptive phi: only when the caller explicitly opts in AND the geometry is a plain
  // full-circle horn (no enclosure/wall). Enclosure/wall functions assume uniform ring
  // topology. ABEC/simulation exports rely on a consistent ringCount and must NOT opt in.
  const outerBuildMode = resolveOuterBuildMode(meshParams, { includeEnclosure });
  const hasEnclosure = outerBuildMode === 'enclosure';
  const hasWall = outerBuildMode === 'freestandingWall';
  if (hasEnclosure && !fullCircle) {
    throw new Error('Partial-domain quadrants are not supported with enclosure geometry yet.');
  }
  const useAdaptivePhi = (options.adaptivePhi === true)
    && fullCircle
    && !hasEnclosure
    && !hasWall;

  let vertices;
  let indices;
  let mouthRingCount; // phi count of the outermost (mouth) ring
  let throatRingCount; // phi count of the innermost (throat) ring
  let secondRingCount; // phi count of the second ring (used for "into horn" direction)

  if (useAdaptivePhi) {
    const phiCounts = computeAdaptivePhiCounts(
      meshParams, lengthSteps, sliceMap, angularSegments, profileContext
    );
    vertices = createAdaptiveRingVertices(
      meshParams, sliceMap, morphTargets, phiCounts, lengthSteps, profileContext, shrinkData
    );
    indices = createAdaptiveFanIndices(phiCounts, lengthSteps);
    mouthRingCount = phiCounts[lengthSteps];
    throatRingCount = phiCounts[0];
    secondRingCount = phiCounts[1] || throatRingCount;
  } else {
    vertices = createRingVertices(
      meshParams, sliceMap, angleList, morphTargets, ringCount, lengthSteps, profileContext, shrinkData
    );
    indices = createHornIndices(ringCount, lengthSteps, fullCircle);
    mouthRingCount = ringCount;
    throatRingCount = ringCount;
    secondRingCount = ringCount;
  }

  const hornEndTri = indices.length / 3;
  if (groupInfo) {
    groupInfo.horn = { start: 0, end: hornEndTri };
    groupInfo[hasEnclosure ? 'horn_wall' : 'inner_wall'] = { start: 0, end: hornEndTri };
  }

  if (hasEnclosure) {
    addEnclosureGeometry(
      vertices,
      indices,
      meshParams,
      0,
      null,
      groupInfo,
      mouthRingCount,
      angleList
    );
  } else if (hasWall) {
    addFreestandingWallGeometry(vertices, indices, meshParams, {
      ringCount: mouthRingCount,
      lengthSteps,
      fullCircle,
      groupInfo
    });
  }

  const sourceStartTri = indices.length / 3;
  generateThroatSource(vertices, indices, throatRingCount, fullCircle, {
    sourceShape: meshParams.sourceShape,
    sourceRadius: meshParams.sourceRadius,
    sourceCurv: meshParams.sourceCurv,
    nextRingStart: throatRingCount,
    nextRingSize: secondRingCount
  });
  const sourceEndTri = indices.length / 3;

  if (groupInfo && sourceEndTri > sourceStartTri) {
    groupInfo.source = { start: sourceStartTri, end: sourceEndTri };
    groupInfo.throat_disc = { start: sourceStartTri, end: sourceEndTri };
  }

  const vertexCount = vertices.length / 3;
  let maxIndex = -1;
  for (let i = 0; i < indices.length; i++) {
    if (indices[i] > maxIndex) maxIndex = indices[i];
  }
  if (maxIndex >= vertexCount) {
    console.error(`[Geometry] Invalid mesh generated: max index ${maxIndex} >= vertex count ${vertexCount}`);
  }

  // BFS orientation is now a defensive assertion, not a runtime fixer. Every
  // builder is expected to ship triangles wound consistently from
  // construction (see docs/modules/geometry.md § Winding Convention). If BFS
  // would have to flip anything, that means a builder regressed — fail loudly
  // instead of silently repairing it and masking the bug.
  //
  // options.skipOrient is a diagnostic-only escape hatch used by
  // scripts/diagnostics/audit-builder-orientation.js to measure raw builder
  // output. Production callers must not set this.
  if (!options.skipOrient) {
    const orientReport = orientMeshConsistently(vertices, indices, { preferOutward: false });
    if (orientReport.trianglesFlipped > 0) {
      throw new Error(
        `[Geometry] Builder winding regression: BFS had to flip ${orientReport.trianglesFlipped} triangle(s). ` +
        `Run scripts/diagnostics/audit-builder-orientation.js to identify the offending builder.`
      );
    }
  }

  // T-junction detection is O(V × E) — opt-in because the viewport pipeline's
  // per-edit rebuilds don't need it and the cost (~100ms at 6k tris) is
  // noticeable. Callers that care about BEM accuracy (canonical payload,
  // diagnostic scripts, regression tests) should set checkTJunctions: true.
  // When opted in, T-junctions are a HARD failure — the builder is expected
  // to ship stitches without vertices sitting on other triangles' edges.
  const quality = validateMeshQuality(vertices, indices, groupInfo, {
    checkTJunctions: options.checkTJunctions === true
  });
  if (!options.skipQualityEnforcement) {
    enforceQualityIssues(quality);
  }

  // Parametric normals disabled: the crease-vertex detector in the viewport
  // pipeline (detachCreaseVertices) combined with Three.js computeVertexNormals()
  // produces better results — preserving superellipse corner sharpness and
  // consistent shading between horn and enclosure surfaces.
  const normals = null;

  const result = {
    vertices,
    indices,
    normals,
    ringCount: mouthRingCount,
    fullCircle,
    // angleList is freshly built by buildAngleList and not retained elsewhere
    // in this module, so returning the reference is safe.  sliceMap in
    // contrast may come from the caller (cached across builds), so keep the
    // defensive copy there.
    angleList: useAdaptivePhi ? null : angleList,
    sliceMap: Array.isArray(sliceMap) ? [...sliceMap] : null
  };

  if (groupInfo) {
    result.groups = groupInfo;
  }

  return result;
}

export const buildHornMesh = buildWaveguideMesh;
