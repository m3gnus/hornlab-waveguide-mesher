function requireFiniteInteger(name, value, min = 0) {
  const numeric = Number(value);
  if (!Number.isInteger(numeric) || numeric < min) {
    throw new Error(`extractPointGrid requires integer "${name}" >= ${min}.`);
  }
  return numeric;
}

/**
 * Convert the uniform horn vertex layout into the Python point-grid layout.
 *
 * JS mesh layout is row-major by axial slice: (j, i) with coordinates [x, axial, z].
 * Python expects phi-major points: (i, j) with coordinates [x, y, axial].
 *
 * @param {ArrayLike<number>} vertices
 * @param {number} ringCount
 * @param {number} lengthSteps
 * @param {boolean} hasOuter
 * @returns {{
 *   innerPoints: Float64Array,
 *   outerPoints: Float64Array | null,
 *   nPhi: number,
 *   nLength: number
 * }}
 */
export function extractPointGrid(vertices, ringCount, lengthSteps, hasOuter) {
  const resolvedRingCount = requireFiniteInteger('ringCount', ringCount, 1);
  const resolvedLengthSteps = requireFiniteInteger('lengthSteps', lengthSteps, 0);
  const innerCount = (resolvedLengthSteps + 1) * resolvedRingCount;
  const expectedVertexFloats = innerCount * 3 * (hasOuter ? 2 : 1);

  if (!vertices || typeof vertices.length !== 'number' || vertices.length < expectedVertexFloats) {
    throw new Error(
      `extractPointGrid expected at least ${expectedVertexFloats} vertex floats for the requested grid layout.`
    );
  }

  const innerPoints = new Float64Array(innerCount * 3);

  for (let j = 0; j <= resolvedLengthSteps; j += 1) {
    for (let i = 0; i < resolvedRingCount; i += 1) {
      const srcIdx = (j * resolvedRingCount + i) * 3;
      const dstIdx = (i * (resolvedLengthSteps + 1) + j) * 3;
      innerPoints[dstIdx] = Number(vertices[srcIdx]);
      innerPoints[dstIdx + 1] = Number(vertices[srcIdx + 2]);
      innerPoints[dstIdx + 2] = Number(vertices[srcIdx + 1]);
    }
  }

  let outerPoints = null;
  if (hasOuter) {
    outerPoints = new Float64Array(innerCount * 3);
    const outerOffset = innerCount * 3;

    for (let j = 0; j <= resolvedLengthSteps; j += 1) {
      for (let i = 0; i < resolvedRingCount; i += 1) {
        const srcIdx = outerOffset + (j * resolvedRingCount + i) * 3;
        const dstIdx = (i * (resolvedLengthSteps + 1) + j) * 3;
        outerPoints[dstIdx] = Number(vertices[srcIdx]);
        outerPoints[dstIdx + 1] = Number(vertices[srcIdx + 2]);
        outerPoints[dstIdx + 2] = Number(vertices[srcIdx + 1]);
      }
    }
  }

  return {
    innerPoints,
    outerPoints,
    nPhi: resolvedRingCount,
    nLength: resolvedLengthSteps
  };
}

/**
 * Read the angular and axial parameter arrays captured on a mesh result.
 *
 * @param {object} meshResult
 * @returns {{ angleList: Float64Array, sliceMap: Float64Array | null }}
 */
export function extractPointGridAxes(meshResult) {
  if (!meshResult || typeof meshResult !== 'object') {
    throw new Error('extractPointGridAxes requires a mesh result object.');
  }

  if (!Array.isArray(meshResult.angleList) || meshResult.angleList.length === 0) {
    throw new Error('extractPointGridAxes requires meshResult.angleList from a uniform ring mesh.');
  }

  return {
    angleList: Float64Array.from(meshResult.angleList),
    sliceMap: Array.isArray(meshResult.sliceMap) ? Float64Array.from(meshResult.sliceMap) : null
  };
}
