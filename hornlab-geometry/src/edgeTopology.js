/**
 * Shared edge-topology utilities used by meshIntegrity.js and quality.js.
 *
 * buildEdgeTopology(indices, startTri?, endTri?) returns:
 *   {
 *     edgeCounts : Map<key, number>              — how many triangles touch each edge
 *     edgeUses   : Map<key, Array<{tri, sign}>>  — per-triangle adjacency + winding
 *   }
 *
 * `sign` is +1 when the triangle traverses the edge in the canonical direction
 * (u < v), -1 otherwise.  Both meshIntegrity and quality derive all of their
 * per-edge statistics from these two maps.
 */

export function edgeKey(a, b) {
  return a < b ? `${a},${b}` : `${b},${a}`;
}

export function triangleArea2(vertices, a, b, c) {
  const ax = vertices[a * 3];
  const ay = vertices[a * 3 + 1];
  const az = vertices[a * 3 + 2];
  const bx = vertices[b * 3];
  const by = vertices[b * 3 + 1];
  const bz = vertices[b * 3 + 2];
  const cx = vertices[c * 3];
  const cy = vertices[c * 3 + 1];
  const cz = vertices[c * 3 + 2];

  const abx = bx - ax;
  const aby = by - ay;
  const abz = bz - az;
  const acx = cx - ax;
  const acy = cy - ay;
  const acz = cz - az;

  const nx = aby * acz - abz * acy;
  const ny = abz * acx - abx * acz;
  const nz = abx * acy - aby * acx;
  return Math.hypot(nx, ny, nz);
}

/**
 * Build per-edge adjacency data for a range of triangles.
 *
 * @param {ArrayLike<number>} indices   - flat triangle index buffer
 * @param {number} [startTri=0]         - first triangle to include (inclusive)
 * @param {number|null} [endTri=null]   - last triangle to include (exclusive); null = all
 * @returns {{ edgeCounts: Map<string,number>, edgeUses: Map<string,Array<{tri:number,sign:number}>> }}
 */
export function buildEdgeTopology(indices, startTri = 0, endTri = null) {
  const triCount = indices.length / 3;
  const start = Math.max(0, startTri);
  const end = Math.min(endTri === null ? triCount : endTri, triCount);

  const edgeCounts = new Map();
  const edgeUses = new Map();

  for (let t = start; t < end; t += 1) {
    const off = t * 3;
    const tri = [indices[off], indices[off + 1], indices[off + 2]];
    const edges = [
      [tri[0], tri[1]],
      [tri[1], tri[2]],
      [tri[2], tri[0]]
    ];

    for (const [u, v] of edges) {
      if (u === v) continue;
      const key = edgeKey(u, v);
      const sign = u < v ? 1 : -1;

      edgeCounts.set(key, (edgeCounts.get(key) || 0) + 1);

      if (!edgeUses.has(key)) edgeUses.set(key, []);
      edgeUses.get(key).push({ tri: t, sign });
    }
  }

  return { edgeCounts, edgeUses };
}
