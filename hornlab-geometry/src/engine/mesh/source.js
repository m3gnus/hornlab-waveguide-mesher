const CAP_INTERMEDIATE_RINGS = 4;

/**
 * Generate throat source geometry (flat disc or spherical cap) and push
 * vertices/indices directly into the provided arrays.
 *
 * @param {number[]} vertices  – flat xyz vertex array (mutated)
 * @param {number[]} indices   – flat triangle index array (mutated)
 * @param {number}   ringCount – number of vertices in the throat ring
 * @param {boolean}  fullCircle
 * @param {object}   [options]
 * @param {number}   [options.sourceShape=2]  – 1 = Spherical Cap, 2 = Flat Disc
 * @param {number}   [options.sourceRadius=-1] – sphere radius (mm); <=0 = auto
 * @param {number}   [options.sourceCurv=0]   – 0 = auto(convex), 1 = convex, -1 = concave
 * @param {number}   [options.nextRingStart]  – vertex index where the second horn ring begins
 * @param {number}   [options.nextRingSize]   – vertex count of the second horn ring
 */
export function generateThroatSource(vertices, indices, ringCount, fullCircle, options = {}) {
  if (!Number.isFinite(ringCount) || ringCount < 2) return;

  const sourceShape = Number(options.sourceShape || 2);
  const sourceRadius = Number(options.sourceRadius ?? -1);
  const sourceCurv = Number(options.sourceCurv ?? 0);

  // Centroid of throat ring
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < ringCount; i++) {
    cx += vertices[i * 3];
    cy += vertices[i * 3 + 1];
    cz += vertices[i * 3 + 2];
  }
  cx /= ringCount;
  cy /= ringCount;
  cz /= ringCount;
  if (!fullCircle) {
    cx = 0;
    cz = 0;
  }

  const segmentCount = fullCircle ? ringCount : Math.max(0, ringCount - 1);

  // ── Flat Disc (sourceShape = 2) ──────────────────────────────────────
  if (sourceShape !== 1) {
    const centerIdx = vertices.length / 3;
    vertices.push(cx, cy, cz);
    // Edge order [b, a] so that (centerIdx, b, a) normals point into the horn.
    for (let i = 0; i < segmentCount; i++) {
      const a = i;
      const b = fullCircle ? (i + 1) % ringCount : i + 1;
      indices.push(centerIdx, b, a);
    }
    return;
  }

  // ── Spherical Cap (sourceShape = 1) ──────────────────────────────────

  // Average throat radius
  let throatR = 0;
  for (let i = 0; i < ringCount; i++) {
    const dx = vertices[i * 3] - cx;
    const dy = vertices[i * 3 + 1] - cy;
    const dz = vertices[i * 3 + 2] - cz;
    throatR += Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  throatR /= ringCount;

  if (throatR < 1e-6) return;

  // Determine "into horn" direction from the second ring centroid.
  // This is robust regardless of ring winding order.
  const nextStart = options.nextRingStart ?? ringCount;
  const nextSize = options.nextRingSize ?? ringCount;
  let nx2 = 0, ny2 = 0, nz2 = 0;
  for (let i = 0; i < nextSize; i++) {
    nx2 += vertices[(nextStart + i) * 3];
    ny2 += vertices[(nextStart + i) * 3 + 1];
    nz2 += vertices[(nextStart + i) * 3 + 2];
  }
  nx2 /= nextSize;
  ny2 /= nextSize;
  nz2 /= nextSize;

  let nx = nx2 - cx, ny = ny2 - cy, nz = nz2 - cz;
  const nLen = Math.sqrt(nx * nx + ny * ny + nz * nz);
  if (nLen < 1e-12) return; // degenerate – skip source
  nx /= nLen;
  ny /= nLen;
  nz /= nLen;

  // Sphere radius R – auto = 2× throat radius for a gentle curve
  let R = sourceRadius > 0 ? sourceRadius : 2 * throatR;
  if (R < throatR * 1.001) R = throatR * 1.001; // R must exceed throatR

  // "Into horn" direction points from throat toward mouth (toward listener).
  // Convex = dome bulging into the horn (toward listener, along direction)
  // Concave = dish recessed away from horn (opposite direction)
  const curvSign = sourceCurv === -1 ? -1 : 1;

  const sqrtBase = Math.sqrt(Math.max(0, R * R - throatR * throatR));

  // Radial unit vectors per throat vertex
  const radX = new Float64Array(ringCount);
  const radY = new Float64Array(ringCount);
  const radZ = new Float64Array(ringCount);
  for (let i = 0; i < ringCount; i++) {
    const dx = vertices[i * 3] - cx;
    const dy = vertices[i * 3 + 1] - cy;
    const dz = vertices[i * 3 + 2] - cz;
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (len > 1e-12) {
      radX[i] = dx / len;
      radY[i] = dy / len;
      radZ[i] = dz / len;
    }
  }

  // Create intermediate ring vertices
  const ringBaseIdx = []; // first vertex index of each intermediate ring
  for (let j = 0; j < CAP_INTERMEDIATE_RINGS; j++) {
    const f = (j + 1) / (CAP_INTERMEDIATE_RINGS + 1);
    const rFrac = 1 - f; // 1 at throat edge, 0 at pole
    const rj = throatR * rFrac;
    const hj = Math.sqrt(R * R - rj * rj) - sqrtBase;

    ringBaseIdx.push(vertices.length / 3);
    for (let i = 0; i < ringCount; i++) {
      vertices.push(
        cx + rFrac * radX[i] * throatR + curvSign * hj * nx,
        cy + rFrac * radY[i] * throatR + curvSign * hj * ny,
        cz + rFrac * radZ[i] * throatR + curvSign * hj * nz
      );
    }
  }

  // Pole (center) vertex
  const capH = R - sqrtBase;
  const poleIdx = vertices.length / 3;
  vertices.push(
    cx + curvSign * capH * nx,
    cy + curvSign * capH * ny,
    cz + curvSign * capH * nz
  );

  // Helper: vertex index for ring j at angular position i
  //   j = -1 → throat ring (existing vertices 0..ringCount-1)
  //   j = 0..N-1 → intermediate rings
  const vtx = (j, i) => (j < 0 ? i : ringBaseIdx[j] + i);

  // Triangle strips between consecutive rings + fan to pole
  for (let band = 0; band <= CAP_INTERMEDIATE_RINGS; band++) {
    const outerJ = band - 1;

    for (let i = 0; i < segmentCount; i++) {
      const ni = fullCircle ? (i + 1) % ringCount : i + 1;

      if (band < CAP_INTERMEDIATE_RINGS) {
        // Quad strip → two triangles. Winding (outer, band, outer-next) and
        // (outer-next, band, band-next) matches the pole-fan convention
        // (center, next, current) so the cap ships BFS-consistent with the
        // horn at the throat ring and with itself across intermediate rings.
        indices.push(vtx(outerJ, i), vtx(band, i), vtx(outerJ, ni));
        indices.push(vtx(outerJ, ni), vtx(band, i), vtx(band, ni));
      } else {
        // Fan to pole – (center, next, current) winding.
        indices.push(poleIdx, vtx(outerJ, ni), vtx(outerJ, i));
      }
    }
  }
}
