/**
 * Per-region enclosure seam validation.
 *
 * The enclosure builder stitches the horn mouth to a rectangular speaker box
 * through a sequence of discrete triangle bands: flat-front (mouth→ring0),
 * front roundover, sidewall, back roundover, back cap. Each boundary between
 * adjacent bands is a "seam" — a ring of vertices shared between the two
 * triangle ranges either side of it.
 *
 * These checks catch the hardest enclosure bugs (fan-stitch corner artifacts,
 * chamfer-boundary mis-snap, off-by-one at ring0 reuse) at build time with a
 * region-named error, instead of letting them surface as a visual glitch or a
 * downstream solver complaint. The global validateMeshQuality pass only sees
 * the aggregate; it reports "N same-direction edges" with no hint of which
 * band is broken. Per-seam checks point straight at the broken stitch.
 *
 * The "expected shared edges" for a seam is computed from the mesh itself:
 * each band emits some subset of the seam's ring-adjacent edges (some may be
 * legitimately dropped by degenerate-triangle filtering in chamfer/collapse
 * cases). If the two bands emit the SAME set of ring-adjacent edges with
 * opposite winding, the seam is intact. If they disagree — one band skipped
 * a ring edge the other included, or included a ring edge with same-direction
 * winding — the check fires a region-named error.
 *
 * See docs/modules/geometry.md § Enclosure Seam Contracts for the full seam
 * list, contracts, and the procedure for adding a new enclosure variant.
 */

import { buildEdgeTopology, edgeKey } from '../../edgeTopology.js';

function isEmptyRange(range) {
  return !range || range.end <= range.start;
}

/**
 * Collect the set of ring-adjacent edges emitted by triangles in `range`.
 * An edge (u,v) is "ring-adjacent" when both u and v are in the vertex window
 * [ringStart, ringStart+ringSize) AND their ring positions differ by 1
 * (cyclically when fullCircle). Returns a Set<string> of canonical edge keys.
 */
function collectRingEdges(indices, range, ringStart, ringSize, fullCircle) {
  const edges = new Set();
  const ringEnd = ringStart + ringSize;
  for (let t = range.start; t < range.end; t += 1) {
    const off = t * 3;
    const tri = [indices[off], indices[off + 1], indices[off + 2]];
    const triEdges = [
      [tri[0], tri[1]],
      [tri[1], tri[2]],
      [tri[2], tri[0]]
    ];
    for (const [u, v] of triEdges) {
      if (u < ringStart || u >= ringEnd) continue;
      if (v < ringStart || v >= ringEnd) continue;
      const ui = u - ringStart;
      const vi = v - ringStart;
      let adjacent;
      if (fullCircle) {
        const d = Math.abs(ui - vi);
        adjacent = d === 1 || d === ringSize - 1;
      } else {
        adjacent = Math.abs(ui - vi) === 1;
      }
      if (adjacent) edges.add(edgeKey(u, v));
    }
  }
  return edges;
}

/**
 * Compute shared-edge statistics between two non-overlapping triangle ranges.
 *
 * Returns { shared, sameDirection, oppositeDirection }:
 *   - shared: number of distinct edges that appear in both ranges
 *   - sameDirection: shared-edge pairs with matching winding (a bug — two
 *     triangles either side of the seam have incompatible orientations)
 *   - oppositeDirection: shared-edge pairs with opposite winding (the
 *     correct 2-manifold case)
 *
 * Uses buildEdgeTopology's range-scoped mode so each call is O(triCount) over
 * the range, not the whole mesh.
 */
export function computeSeamStats(indices, rangeA, rangeB) {
  if (isEmptyRange(rangeA) || isEmptyRange(rangeB)) {
    return { shared: 0, sameDirection: 0, oppositeDirection: 0 };
  }

  const topoA = buildEdgeTopology(indices, rangeA.start, rangeA.end);
  const topoB = buildEdgeTopology(indices, rangeB.start, rangeB.end);

  let shared = 0;
  let sameDirection = 0;
  let oppositeDirection = 0;

  for (const [key, usesA] of topoA.edgeUses.entries()) {
    const usesB = topoB.edgeUses.get(key);
    if (!usesB) continue;
    shared += 1;
    for (const a of usesA) {
      for (const b of usesB) {
        if (a.sign === b.sign) sameDirection += 1;
        else oppositeDirection += 1;
      }
    }
  }

  return { shared, sameDirection, oppositeDirection };
}

function setDiff(a, b) {
  const out = [];
  for (const k of a) if (!b.has(k)) out.push(k);
  return out;
}

/**
 * Validate an array of seam contracts. Each seam is:
 *   {
 *     name: string,                  — human-readable label, used in errors
 *     rangeA: { start, end },        — triangle range A (indices into `indices` buffer)
 *     rangeB: { start, end },        — triangle range B, on the other side of the seam
 *     ringStart: number,             — first vertex index of the seam's ring
 *     ringSize: number,              — vertex count of the seam's ring
 *     fullCircle: boolean,           — true when the ring is cyclic (no open ends)
 *     expectedAllOpposite?: boolean, — if true, sameDirection across the seam must be 0
 *     expectedMinShared?: number,    — if set, shared edge count must be ≥ this
 *   }
 *
 * Throws with a multi-line error listing every violated seam. Returns an
 * array of per-seam reports (one entry per seam, with stats + skipped flag).
 */
export function validateEnclosureSeams(indices, seams) {
  const errors = [];
  const reports = [];

  for (const seam of seams) {
    if (isEmptyRange(seam.rangeA) || isEmptyRange(seam.rangeB)) {
      reports.push({ name: seam.name, skipped: true });
      continue;
    }

    const stats = computeSeamStats(indices, seam.rangeA, seam.rangeB);

    const ringEdgesA = collectRingEdges(
      indices, seam.rangeA, seam.ringStart, seam.ringSize, seam.fullCircle
    );
    const ringEdgesB = collectRingEdges(
      indices, seam.rangeB, seam.ringStart, seam.ringSize, seam.fullCircle
    );

    const onlyInA = setDiff(ringEdgesA, ringEdgesB);
    const onlyInB = setDiff(ringEdgesB, ringEdgesA);

    reports.push({
      name: seam.name,
      stats,
      ringEdgesA: ringEdgesA.size,
      ringEdgesB: ringEdgesB.size,
      onlyInA: onlyInA.length,
      onlyInB: onlyInB.length
    });

    // Contract 1: both bands must emit the SAME set of ring-adjacent edges.
    // Off-by-one errors (missing triangle, extra triangle, wrong band size)
    // show up here as a mismatched set.
    if (onlyInA.length > 0 || onlyInB.length > 0) {
      errors.push(
        `${seam.name}, ringSize=${seam.ringSize}: ring-edge sets differ across seam ` +
        `(band A has ${ringEdgesA.size}, band B has ${ringEdgesB.size}, ` +
        `${onlyInA.length} only in A, ${onlyInB.length} only in B)`
      );
    }

    // Contract 2: all shared edges must be opposite-winding. A same-direction
    // shared edge means the two bands disagree on which side faces outward —
    // a corner or stitch winding bug.
    if (seam.expectedAllOpposite && stats.sameDirection > 0) {
      errors.push(
        `${seam.name}, ringSize=${seam.ringSize}: ${stats.sameDirection} edge(s) with ` +
        `same-direction winding across seam (${stats.shared} total shared)`
      );
    }

    // Contract 3 (optional): shared edge count must be at least N. Catches the
    // case where both bands agree but share nothing — e.g., a band built from
    // the wrong ring that happens to match its neighbor's emptiness.
    if (seam.expectedMinShared !== undefined && stats.shared < seam.expectedMinShared) {
      errors.push(
        `${seam.name}, ringSize=${seam.ringSize}: expected ≥${seam.expectedMinShared} ` +
        `shared edge(s), got ${stats.shared}`
      );
    }
  }

  if (errors.length > 0) {
    throw new Error(`[Enclosure] Seam validation failed:\n  - ${errors.join('\n  - ')}`);
  }

  return reports;
}
