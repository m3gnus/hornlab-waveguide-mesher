import { computeSignedVolume } from './meshIntegrity.js';
import { edgeKey, triangleArea2, buildEdgeTopology } from './edgeTopology.js';

const DEFAULT_DUPLICATE_VERTEX_EPSILON = 1e-6;
// T-junction detection uses an absolute tolerance on perpendicular distance.
// The value balances catching near-duplicate vertices from stitch bugs
// (~1e-5 offsets from seamNudge × normal-spread) against ignoring natural
// mesh-topology near-misses (adjacent ring vertices sitting close to a
// diagonal edge in a long, fine ring).
const DEFAULT_TJUNCTION_EPSILON = 1e-4;

/**
 * Count vertices that share (to within `epsilon`) a position with a previous
 * vertex. These are "phantom-boundary" duplicates: BEM and solid-volume
 * consumers expect coincident points to be referenced by a single vertex
 * index reused across triangles. Vertex splits introduced intentionally by
 * the viewport's crease-detach pass are handled by validateViewportMesh,
 * not by validateMeshQuality — so this check is safe to run on the
 * canonical payload.
 */
function countDuplicateVertices(vertices, epsilon = DEFAULT_DUPLICATE_VERTEX_EPSILON) {
  const vCount = vertices.length / 3;
  if (vCount === 0) return 0;
  const seen = new Map();
  const invEps = 1 / epsilon;
  let duplicates = 0;
  for (let i = 0; i < vCount; i += 1) {
    const qx = Math.round(vertices[i * 3] * invEps);
    const qy = Math.round(vertices[i * 3 + 1] * invEps);
    const qz = Math.round(vertices[i * 3 + 2] * invEps);
    const key = `${qx}|${qy}|${qz}`;
    if (seen.has(key)) {
      duplicates += 1;
    } else {
      seen.set(key, i);
    }
  }
  return duplicates;
}

/**
 * Locate T-junctions — vertices that lie on the interior of another triangle's
 * edge (within `epsilon`) without being one of that edge's endpoints, AND
 * that are topologically connected to the edge (share a triangle with one of
 * its endpoints). T-junctions pass the manifold/non-manifold edge-count
 * check (each edge still has the right number of adjacent triangles), but
 * create visible cracks under displacement / subdivision and cause BEM
 * integration errors where the T-vertex violates the assumed edge-continuity
 * of adjacent triangles.
 *
 * The topological-connectivity check (w shares at least one triangle with u
 * or v) rejects purely coincidental cross-surface alignments where a vertex
 * on one surface (e.g. horn) geometrically lands on an edge of another
 * surface (e.g. rear-cap fan) without the two surfaces being stitched there.
 * Such coincidences don't break BEM continuity, because the two surfaces are
 * independent patches — the "T" is cosmetic, not topological.
 *
 * Returns an array of violations. Each entry records the edge's endpoints
 * (u, v), one triangle that owns the edge (`tri`), the stray vertex on the
 * edge interior (`w`), the parametric position along u→v (`t` ∈ (0, 1)),
 * and the perpendicular distance (`dist`) of w from the edge line.
 *
 * Naive O(E × V). Each edge is tested once (canonical direction); endpoints
 * are skipped. For a ~6k-triangle / ~3k-vertex mesh this is ~50M comparisons
 * — low hundreds of milliseconds.
 */
export function findTJunctions(vertices, indices, epsilon = DEFAULT_TJUNCTION_EPSILON) {
  const vCount = vertices.length / 3;
  const triCount = indices.length / 3;
  const out = [];
  if (vCount < 4 || triCount === 0) return out;

  // Build vertex-to-triangle adjacency once. vertexTris[w] = Set of triangle
  // indices that use vertex w.
  const vertexTris = new Array(vCount);
  for (let i = 0; i < vCount; i += 1) vertexTris[i] = null;
  for (let t = 0; t < triCount; t += 1) {
    for (let e = 0; e < 3; e += 1) {
      const idx = indices[t * 3 + e];
      if (!vertexTris[idx]) vertexTris[idx] = new Set();
      vertexTris[idx].add(t);
    }
  }
  function sharesTriangle(a, b) {
    const sa = vertexTris[a];
    const sb = vertexTris[b];
    if (!sa || !sb) return false;
    const [small, big] = sa.size <= sb.size ? [sa, sb] : [sb, sa];
    for (const t of small) if (big.has(t)) return true;
    return false;
  }

  // Build edge → owning triangles map so the sliver-triangle self-apex case
  // (w is the third vertex of a near-degenerate triangle using edge (u, v))
  // can be excluded — such a vertex is not a topological T-junction, just a
  // numerically collinear triangle.
  const edgeTris = new Map(); // edgeKey -> Set<triangle index>
  for (let t = 0; t < triCount; t += 1) {
    const a = indices[t * 3];
    const b = indices[t * 3 + 1];
    const c = indices[t * 3 + 2];
    for (const [p, q] of [[a, b], [b, c], [c, a]]) {
      if (p === q) continue;
      const key = edgeKey(p, q);
      let s = edgeTris.get(key);
      if (!s) { s = new Set(); edgeTris.set(key, s); }
      s.add(t);
    }
  }
  function isThirdVertexOfEdgeTriangle(w, edgeKey_) {
    const ts = edgeTris.get(edgeKey_);
    if (!ts) return false;
    for (const t of ts) {
      if (indices[t * 3] === w || indices[t * 3 + 1] === w || indices[t * 3 + 2] === w) return true;
    }
    return false;
  }

  const seenEdges = new Set();
  const eps2 = epsilon * epsilon;

  for (let t = 0; t < triCount; t += 1) {
    const off = t * 3;
    const tri = [indices[off], indices[off + 1], indices[off + 2]];
    for (let e = 0; e < 3; e += 1) {
      const u = tri[e];
      const v = tri[(e + 1) % 3];
      if (u === v) continue;
      const key = edgeKey(u, v);
      if (seenEdges.has(key)) continue;
      seenEdges.add(key);

      const ux = vertices[u * 3];
      const uy = vertices[u * 3 + 1];
      const uz = vertices[u * 3 + 2];
      const dx = vertices[v * 3] - ux;
      const dy = vertices[v * 3 + 1] - uy;
      const dz = vertices[v * 3 + 2] - uz;
      const lenSq = dx * dx + dy * dy + dz * dz;
      if (lenSq < 1e-20) continue;

      for (let w = 0; w < vCount; w += 1) {
        if (w === u || w === v) continue;
        const wx = vertices[w * 3];
        const wy = vertices[w * 3 + 1];
        const wz = vertices[w * 3 + 2];
        const pw = ((wx - ux) * dx + (wy - uy) * dy + (wz - uz) * dz) / lenSq;
        if (pw <= 1e-6 || pw >= 1 - 1e-6) continue;

        const px = ux + pw * dx;
        const py = uy + pw * dy;
        const pz = uz + pw * dz;
        const distSq = (wx - px) * (wx - px) + (wy - py) * (wy - py) + (wz - pz) * (wz - pz);
        if (distSq >= eps2) continue;

        // Exclude sliver self-apex: w is the third vertex of a near-collinear
        // triangle that already includes edge (u, v). Not a topological
        // T-junction — just a sliver.
        if (isThirdVertexOfEdgeTriangle(w, key)) continue;

        // Exclude cross-surface coincidences: w and (u, v) must be connected
        // via a shared triangle. A horn vertex that geometrically lands on
        // an independent enclosure edge is not a BEM continuity issue.
        if (!sharesTriangle(w, u) && !sharesTriangle(w, v)) continue;

        out.push({ tri: t, u, v, w, t: pw, dist: Math.sqrt(distSq) });
        break;
      }
    }
  }

  return out;
}

function countTJunctions(vertices, indices, epsilon = DEFAULT_TJUNCTION_EPSILON) {
  return findTJunctions(vertices, indices, epsilon).length;
}

function buildEdgeStats(indices, startTri = 0, endTri = null) {
  const { edgeCounts, edgeUses } = buildEdgeTopology(indices, startTri, endTri);

  // Convert edgeUses (Map<key, [{tri, sign}]>) to orientedEdges (Map<key, [sign]>)
  // to preserve the interface expected by countSharedEdges and validateMeshQuality.
  const orientedEdges = new Map();
  for (const [key, uses] of edgeUses.entries()) {
    orientedEdges.set(key, uses.map((u) => u.sign));
  }

  return { edgeCounts, orientedEdges };
}

function countConnectedComponents(indices) {
  const triCount = indices.length / 3;
  const edgeToTriangles = new Map();

  for (let t = 0; t < triCount; t += 1) {
    const off = t * 3;
    const tri = [indices[off], indices[off + 1], indices[off + 2]];
    for (const [u, v] of [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]]) {
      const key = edgeKey(u, v);
      if (!edgeToTriangles.has(key)) edgeToTriangles.set(key, []);
      edgeToTriangles.get(key).push(t);
    }
  }

  const adjacency = Array.from({ length: triCount }, () => []);
  for (const tris of edgeToTriangles.values()) {
    for (let i = 0; i < tris.length; i += 1) {
      for (let j = i + 1; j < tris.length; j += 1) {
        adjacency[tris[i]].push(tris[j]);
        adjacency[tris[j]].push(tris[i]);
      }
    }
  }

  const visited = new Uint8Array(triCount);
  let components = 0;

  for (let i = 0; i < triCount; i += 1) {
    if (visited[i]) continue;
    components += 1;
    const stack = [i];
    visited[i] = 1;

    while (stack.length > 0) {
      const t = stack.pop();
      for (const next of adjacency[t]) {
        if (!visited[next]) {
          visited[next] = 1;
          stack.push(next);
        }
      }
    }
  }

  return components;
}

function countSharedEdges(rangeA, rangeB, indices) {
  if (!rangeA || !rangeB) return { shared: 0, sameDirection: 0, oppositeDirection: 0 };
  const a = buildEdgeStats(indices, rangeA.start, rangeA.end).orientedEdges;
  const b = buildEdgeStats(indices, rangeB.start, rangeB.end).orientedEdges;

  let shared = 0;
  let sameDirection = 0;
  let oppositeDirection = 0;

  for (const [key, listA] of a.entries()) {
    const listB = b.get(key);
    if (!listB) continue;
    shared += 1;
    for (const oa of listA) {
      for (const ob of listB) {
        if (oa === ob) sameDirection += 1;
        else oppositeDirection += 1;
      }
    }
  }

  return { shared, sameDirection, oppositeDirection };
}

export function validateMeshQuality(vertices, indices, groups = null, options = {}) {
  const triCount = indices.length / 3;

  let degenerateTriangles = 0;
  for (let t = 0; t < triCount; t += 1) {
    const off = t * 3;
    if (triangleArea2(vertices, indices[off], indices[off + 1], indices[off + 2]) <= 1e-10) {
      degenerateTriangles += 1;
    }
  }

  const { edgeCounts, orientedEdges } = buildEdgeStats(indices);
  let boundaryEdges = 0;
  let nonManifoldEdges = 0;
  let sameDirectionSharedEdges = 0;

  for (const [key, count] of edgeCounts.entries()) {
    if (count === 1) boundaryEdges += 1;
    if (count > 2) nonManifoldEdges += 1;

    const orientations = orientedEdges.get(key) || [];
    if (orientations.length === 2 && orientations[0] === orientations[1]) {
      sameDirectionSharedEdges += 1;
    }
  }

  const components = countConnectedComponents(indices);

  const seamStats = countSharedEdges(groups?.horn, groups?.enclosure, indices);
  const sourceConnectivity = countSharedEdges(groups?.source, groups?.horn, indices);

  // For closed (watertight) meshes, signed volume indicates global orientation:
  // positive = outward normals, negative = inside-out. Only meaningful when
  // boundaryEdges === 0; we compute it unconditionally since it's cheap and
  // callers can gate on boundaryEdges themselves.
  const signedVolume = computeSignedVolume(vertices, indices);

  // Position-coincident duplicate vertices. Always cheap.
  const duplicateVertices = countDuplicateVertices(
    vertices,
    options.duplicateVertexEpsilon ?? DEFAULT_DUPLICATE_VERTEX_EPSILON
  );

  // T-junction detection is O(V × E); opt-in via checkTJunctions to avoid a
  // ~500ms cost on every build. Currently we turn it on for the canonical
  // payload assertion path but leave it off for the viewport pipeline.
  const tJunctions = options.checkTJunctions
    ? countTJunctions(
        vertices,
        indices,
        options.tJunctionEpsilon ?? DEFAULT_TJUNCTION_EPSILON
      )
    : null;

  return {
    triCount,
    degenerateTriangles,
    boundaryEdges,
    nonManifoldEdges,
    sameDirectionSharedEdges,
    components,
    signedVolume,
    duplicateVertices,
    tJunctions,
    seam: seamStats,
    sourceConnectivity: sourceConnectivity.shared
  };
}
