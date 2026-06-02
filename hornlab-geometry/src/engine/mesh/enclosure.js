/* Enclosure mesh generation */
// ===========================================================================
// Enclosure Geometry (Rear Chamber for BEM Simulation)
// ===========================================================================

import { validateEnclosureSeams } from './enclosureSeams.js';

// Legacy plan-outline helpers were removed in Session 9 because the current
// enclosure pipeline only uses angular ray-casting against a rounded box.

// ---------------------------------------------------------------------------
// Angular-distribution-based enclosure point generation.
//
// The front roundover uses the mouth's angle list for a smooth baffle
// transition.  At the flat sidewall, a fan stitch transitions to a refined
// angle set that adds extra points at the enclosure's rounded corners.
// Corner subdivision density matches the Y-axis roundover arc step
// (edgeDepth × π/2 / edgeSlices), giving all four corners uniform curvature.
// ---------------------------------------------------------------------------

/**
 * Compute the intersection of a ray from (cx, cz) at angle `angle` with a
 * rounded rectangle defined by (boxLeft, boxRight, boxBot, boxTop) with
 * corner radius `cr` and optional chamfer (edgeType === 2).
 */
function intersectRayWithRoundedBox(
    angle, cx, cz,
    boxLeft, boxRight, boxBot, boxTop,
    cr, edgeType,
    params
) {
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);

    const scale = params.scale || 1;
    const EPS = 1e-12 * scale;
    let bestT = Infinity;
    let hitX = cx + cosA;
    let hitZ = cz + sinA;
    let hitNx = cosA;
    let hitNz = sinA;

    const trySegment = (x1, z1, x2, z2, nx, nz) => {
        const ex = x2 - x1;
        const ez = z2 - z1;
        const det = cosA * (-ez) - sinA * (-ex);
        if (Math.abs(det) <= EPS) return;

        const rhsX = x1 - cx;
        const rhsZ = z1 - cz;
        const t = (rhsX * (-ez) - rhsZ * (-ex)) / det;
        const u = (cosA * rhsZ - sinA * rhsX) / det;

        if (t > EPS && u >= -EPS && u <= 1 + EPS && t < bestT) {
            bestT = t;
            hitX = cx + cosA * t;
            hitZ = cz + sinA * t;
            hitNx = nx;
            hitNz = nz;
        }
    };

    const tryArc = (acx, acz, r, startAngle, endAngle) => {
        const ox = cx - acx;
        const oz = cz - acz;
        const A = 1;
        const B = 2 * (ox * cosA + oz * sinA);
        const C = ox * ox + oz * oz - r * r;
        const disc = B * B - 4 * A * C;
        if (disc < 0) return;

        const sqrtDisc = Math.sqrt(disc);
        for (const t of [(-B - sqrtDisc) / (2 * A), (-B + sqrtDisc) / (2 * A)]) {
            if (t <= EPS || t >= bestT) continue;
            const px = cx + cosA * t;
            const pz = cz + sinA * t;
            let pa = Math.atan2(pz - acz, px - acx);
            let swept = endAngle - startAngle;
            let relAngle = pa - startAngle;
            while (relAngle < -EPS) relAngle += Math.PI * 2;
            while (relAngle > Math.PI * 2 + EPS) relAngle -= Math.PI * 2;
            if (relAngle <= swept + EPS) {
                bestT = t;
                hitX = px;
                hitZ = pz;
                const dx = px - acx;
                const dz = pz - acz;
                const len = Math.hypot(dx, dz);
                hitNx = len > 0 ? dx / len : 0;
                hitNz = len > 0 ? dz / len : 0;
            }
        }
    };

    const tryChamfer = (acx, acz, r, startAngle, endAngle) => {
        const x1 = acx + r * Math.cos(startAngle);
        const z1 = acz + r * Math.sin(startAngle);
        const x2 = acx + r * Math.cos(endAngle);
        const z2 = acz + r * Math.sin(endAngle);
        const midA = (startAngle + endAngle) / 2;
        trySegment(x1, z1, x2, z2, Math.cos(midA), Math.sin(midA));
    };

    const halfW = (boxRight - boxLeft) / 2;
    const halfH = (boxTop - boxBot) / 2;
    const bCx = (boxRight + boxLeft) / 2;
    const bCz = (boxTop + boxBot) / 2;
    const r = Math.min(cr, halfW - 1e-4 * scale, halfH - 1e-4 * scale);
    const useCorners = r > 1e-3 * scale;

    trySegment(boxRight, bCz - halfH + (useCorners ? r : 0), boxRight, bCz + halfH - (useCorners ? r : 0), 1, 0);
    trySegment(bCx + halfW - (useCorners ? r : 0), boxTop, bCx - halfW + (useCorners ? r : 0), boxTop, 0, 1);
    trySegment(boxLeft, bCz + halfH - (useCorners ? r : 0), boxLeft, bCz - halfH + (useCorners ? r : 0), -1, 0);
    trySegment(bCx - halfW + (useCorners ? r : 0), boxBot, bCx + halfW - (useCorners ? r : 0), boxBot, 0, -1);

    if (useCorners) {
        const corners = [
            { cx: bCx + halfW - r, cz: bCz - halfH + r, start: -Math.PI / 2, end: 0 },
            { cx: bCx + halfW - r, cz: bCz + halfH - r, start: 0, end: Math.PI / 2 },
            { cx: bCx - halfW + r, cz: bCz + halfH - r, start: Math.PI / 2, end: Math.PI },
            { cx: bCx - halfW + r, cz: bCz - halfH + r, start: Math.PI, end: Math.PI * 1.5 }
        ];
        for (const c of corners) {
            if (edgeType === 2) tryChamfer(c.cx, c.cz, r, c.start, c.end);
            else tryArc(c.cx, c.cz, r, c.start, c.end);
        }
    }
    return { x: hitX, z: hitZ, nx: hitNx, nz: hitNz };
}

/**
 * Compute the intersection of a ray from (cx, cz) at `angle` with an
 * axis-aligned ellipse centred at ((boxLeft+boxRight)/2, (boxBot+boxTop)/2)
 * with half-widths halfW, halfH.
 *
 * Parametric ray: P = (cx,cz) + t*(cosA, sinA), t > 0
 * Ellipse: (x-ecx)^2/halfW^2 + (z-ecz)^2/halfH^2 = 1
 */
function intersectRayWithEllipse(
    angle, cx, cz,
    boxLeft, boxRight, boxBot, boxTop,
    params
) {
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);
    const scale = params.scale || 1;
    const EPS = 1e-12 * scale;

    const ecx = (boxRight + boxLeft) / 2;
    const ecz = (boxTop + boxBot) / 2;
    const a = (boxRight - boxLeft) / 2; // halfW
    const b = (boxTop - boxBot) / 2;   // halfH

    // Transform ray origin to ellipse-centered coordinates
    const ox = cx - ecx;
    const oz = cz - ecz;

    // Solve (ox + t*cosA)^2/a^2 + (oz + t*sinA)^2/b^2 = 1
    const A = (cosA * cosA) / (a * a) + (sinA * sinA) / (b * b);
    const B = 2 * ((ox * cosA) / (a * a) + (oz * sinA) / (b * b));
    const C = (ox * ox) / (a * a) + (oz * oz) / (b * b) - 1;

    const disc = B * B - 4 * A * C;
    if (disc < 0) {
        // Fallback: return point along ray direction on ellipse
        const fa = Math.atan2(sinA, cosA);
        const hx = ecx + a * Math.cos(fa);
        const hz = ecz + b * Math.sin(fa);
        const len = Math.hypot(hx - ecx, hz - ecz);
        return { x: hx, z: hz, nx: len > 0 ? (hx - ecx) / len : cosA, nz: len > 0 ? (hz - ecz) / len : sinA };
    }

    const sqrtDisc = Math.sqrt(disc);
    const t1 = (-B - sqrtDisc) / (2 * A);
    const t2 = (-B + sqrtDisc) / (2 * A);

    // Pick smallest positive t
    let t = (t1 > EPS) ? t1 : (t2 > EPS ? t2 : t1);
    if (t <= EPS) t = t2;

    const hx = cx + cosA * t;
    const hz = cz + sinA * t;

    // Normal to ellipse at (hx, hz): gradient of (x-ecx)^2/a^2 + (z-ecz)^2/b^2
    let nx = 2 * (hx - ecx) / (a * a);
    let nz = 2 * (hz - ecz) / (b * b);
    const nLen = Math.hypot(nx, nz);
    if (nLen > 0) { nx /= nLen; nz /= nLen; }

    return { x: hx, z: hz, nx, nz };
}

/**
 * Compute the intersection of a ray from (cx, cz) at `angle` with a
 * superellipse |x/a|^n + |z/b|^n = 1 centred at the box center.
 *
 * Uses Newton-Raphson on the parametric angle to find the intersection.
 */
function intersectRayWithSuperellipse(
    angle, cx, cz,
    boxLeft, boxRight, boxBot, boxTop,
    n, params
) {
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);
    const scale = params.scale || 1;

    const ecx = (boxRight + boxLeft) / 2;
    const ecz = (boxTop + boxBot) / 2;
    const a = (boxRight - boxLeft) / 2; // halfW
    const b = (boxTop - boxBot) / 2;   // halfH

    // Superellipse parametric form:
    //   x(theta) = ecx + a * sign(cos(theta)) * |cos(theta)|^(2/n)
    //   z(theta) = ecz + b * sign(sin(theta)) * |sin(theta)|^(2/n)
    //
    // We want the point on the superellipse that lies along the ray from (cx,cz)
    // at direction angle. Use Newton-Raphson on cross product = 0:
    //   f(theta) = (sx - cx) * sinA - (sz - cz) * cosA = 0

    const exp = 2 / n;
    const sgnPow = (val, e) => {
        const av = Math.abs(val);
        return (av < 1e-15 ? 0 : Math.sign(val) * Math.pow(av, e));
    };

    const superX = (theta) => ecx + a * sgnPow(Math.cos(theta), exp);
    const superZ = (theta) => ecz + b * sgnPow(Math.sin(theta), exp);

    // f(theta): cross product of (S-C) with ray direction should be zero
    const f = (theta) => {
        const sx = superX(theta) - cx;
        const sz = superZ(theta) - cz;
        return sx * sinA - sz * cosA;
    };

    // Also need: dot product > 0 (same direction as ray)
    const dot = (theta) => {
        const sx = superX(theta) - cx;
        const sz = superZ(theta) - cz;
        return sx * cosA + sz * sinA;
    };

    // Start Newton-Raphson from the ray angle (good initial guess)
    let theta = angle;
    const dTheta = 1e-6;
    for (let iter = 0; iter < 30; iter++) {
        const fv = f(theta);
        if (Math.abs(fv) < 1e-10 * scale) break;
        const fp = (f(theta + dTheta) - f(theta - dTheta)) / (2 * dTheta);
        if (Math.abs(fp) < 1e-20) break;
        theta -= fv / fp;
    }

    // Verify dot product is positive (ray goes forward). If not, flip to opposite side.
    if (dot(theta) < 0) {
        theta += Math.PI;
        for (let iter = 0; iter < 30; iter++) {
            const fv = f(theta);
            if (Math.abs(fv) < 1e-10 * scale) break;
            const fp = (f(theta + dTheta) - f(theta - dTheta)) / (2 * dTheta);
            if (Math.abs(fp) < 1e-20) break;
            theta -= fv / fp;
        }
    }

    const hx = superX(theta);
    const hz = superZ(theta);

    // Normal to superellipse: gradient of |x/a|^n + |z/b|^n
    // d/dx = n * sign(x-ecx) * |x-ecx|^(n-1) / a^n
    // d/dz = n * sign(z-ecz) * |z-ecz|^(n-1) / b^n
    const dx = hx - ecx;
    const dz = hz - ecz;
    let nx = sgnPow(dx / a, n - 1) / a;
    let nz = sgnPow(dz / b, n - 1) / b;
    const nLen = Math.hypot(nx, nz);
    if (nLen > 0) { nx /= nLen; nz /= nLen; }

    return { x: hx, z: hz, nx, nz };
}

function generateEnclosurePointsFromAngles(
    angleList, cx, cz,
    boxLeft, boxRight, boxBot, boxTop,
    edgeR, edgeType,
    params
) {
    const ringSize = angleList.length;
    const outerPts = [];
    const insetPts = [];
    const halfW = (boxRight - boxLeft) / 2;
    const halfH = (boxTop - boxBot) / 2;
    const boxCR = parseFloat(edgeR) || 0;
    const scale = params.scale || 1;
    const minHalf = Math.min(halfW, halfH);
    const clampedBoxCR = Math.min(boxCR, minHalf * 0.5, halfW - 1e-4 * scale, halfH - 1e-4 * scale);

    const planType = parseInt(params.encPlanType) || 1;
    const planN = parseFloat(params.encPlanN) || 2.0;

    for (let i = 0; i < ringSize; i++) {
        const angle = angleList[i];
        let hit;
        if (planType === 2) {
            // Ellipse
            hit = intersectRayWithEllipse(
                angle, cx, cz,
                boxLeft, boxRight, boxBot, boxTop,
                params
            );
        } else if (planType === 3) {
            // Superellipse
            hit = intersectRayWithSuperellipse(
                angle, cx, cz,
                boxLeft, boxRight, boxBot, boxTop,
                planN, params
            );
        } else {
            // Default: Rounded Rectangle
            hit = intersectRayWithRoundedBox(
                angle, cx, cz,
                boxLeft, boxRight, boxBot, boxTop,
                clampedBoxCR, edgeType,
                params
            );
        }
        outerPts.push({ x: hit.x, z: hit.z, nx: hit.nx, nz: hit.nz });
        insetPts.push({
            x: hit.x - hit.nx * clampedBoxCR,
            z: hit.z - hit.nz * clampedBoxCR,
            nx: hit.nx,
            nz: hit.nz
        });
    }

    // For chamfered corners on rounded-rectangle plan, snap inset points to the
    // arc center so the chamfer surface forms a proper triangle instead of a
    // small rectangle.  Rounded corners already converge naturally (radial
    // normals -> arc center), but chamfer normals are fixed (the midpoint angle
    // of the corner), so the normal-based offset produces a parallel line
    // instead of a point.
    if (planType === 1 && edgeType === 2 && clampedBoxCR > 1e-6) {
        const bCx = (boxRight + boxLeft) / 2;
        const bCz = (boxTop + boxBot) / 2;
        const r = clampedBoxCR;
        const arcCenters = [
            { x: bCx + halfW - r, z: bCz - halfH + r },
            { x: bCx + halfW - r, z: bCz + halfH - r },
            { x: bCx - halfW + r, z: bCz + halfH - r },
            { x: bCx - halfW + r, z: bCz - halfH + r },
        ];
        for (let i = 0; i < ringSize; i++) {
            const nx = outerPts[i].nx;
            const nz = outerPts[i].nz;
            // Chamfer hits have non-axis-aligned normals (both components > 0)
            if (Math.abs(nx) > 0.01 && Math.abs(nz) > 0.01) {
                let bestDist = Infinity;
                let best = arcCenters[0];
                for (const c of arcCenters) {
                    const d = Math.hypot(outerPts[i].x - c.x, outerPts[i].z - c.z);
                    if (d < bestDist) { bestDist = d; best = c; }
                }
                insetPts[i].x = best.x;
                insetPts[i].z = best.z;
            }
        }
    }

    return { outerPts, insetPts, clampedBoxCR };
}

/**
 * Refine mouth angles for enclosure corners.
 *
 * Flat edges keep the original mouth-angle spacing (for smooth baffle
 * transition).  Corner arcs — detected by a change in the outward normal
 * between consecutive outer points — are subdivided so that the XZ arc step
 * matches the Y-axis roundover arc step (edgeDepth × π/2 / edgeSlices).
 *
 * Returns { refined, mapping } where `refined` is the new angle array and
 * `mapping[i]` gives the index in `refined` that corresponds to mouth angle i.
 */
function refineAnglesForEnclosure(mouthAngles, outerPts, edgeSlices, edgeDepth, cx, cz,
    edgeType, boxLeft, boxRight, boxBot, boxTop, clampedBoxCR) {
    const n = mouthAngles.length;
    const roundoverArcStep = edgeSlices > 0 && edgeDepth > 0
        ? (edgeDepth * Math.PI / 2) / edgeSlices
        : Infinity;

    // For chamfer edges, precompute boundary points AND angles where flat meets chamfer
    let chamferBoundaries = null;
    if (edgeType === 2 && clampedBoxCR > 1e-6) {
        const halfW = (boxRight - boxLeft) / 2;
        const halfH = (boxTop - boxBot) / 2;
        const bCx = (boxRight + boxLeft) / 2;
        const bCz = (boxTop + boxBot) / 2;
        const r = clampedBoxCR;
        chamferBoundaries = [];
        const corners = [
            { acx: bCx + halfW - r, acz: bCz - halfH + r, start: -Math.PI / 2, end: 0 },
            { acx: bCx + halfW - r, acz: bCz + halfH - r, start: 0, end: Math.PI / 2 },
            { acx: bCx - halfW + r, acz: bCz + halfH - r, start: Math.PI / 2, end: Math.PI },
            { acx: bCx - halfW + r, acz: bCz - halfH + r, start: Math.PI, end: Math.PI * 1.5 }
        ];
        for (const c of corners) {
            const bp1 = { x: c.acx + r * Math.cos(c.start), z: c.acz + r * Math.sin(c.start) };
            const bp2 = { x: c.acx + r * Math.cos(c.end), z: c.acz + r * Math.sin(c.end) };
            chamferBoundaries.push(
                { x: bp1.x, z: bp1.z, angle: Math.atan2(bp1.z - cz, bp1.x - cx) },
                { x: bp2.x, z: bp2.z, angle: Math.atan2(bp2.z - cz, bp2.x - cx) }
            );
        }
    }

    const isChamferNormal = (pt) =>
        Math.abs(pt.nx) > 0.01 && Math.abs(pt.nz) > 0.01;

    const refined = [];
    const mapping = [0];

    for (let i = 0; i < n; i++) {
        refined.push(mouthAngles[i]);
        const j = (i + 1) % n;

        // Detect corner vs flat edge by checking normal change
        const ndx = Math.abs(outerPts[i].nx - outerPts[j].nx);
        const ndz = Math.abs(outerPts[i].nz - outerPts[j].nz);
        const isCorner = (ndx + ndz) > 0.01;
        const avgNx = (outerPts[i].nx + outerPts[j].nx) * 0.5;
        const avgNz = (outerPts[i].nz + outerPts[j].nz) * 0.5;
        const isTopOrBottom = Math.abs(avgNz) > 0.9 && Math.abs(avgNx) < 0.3;
        const isBottom = avgNz < -0.9 && Math.abs(avgNx) < 0.3;

        // For chamfer: when adjacent points cross the flat/chamfer boundary,
        // find the exact boundary angle and use it to split the span.
        let boundaryT = -1;
        let boundaryExactAngle = null;
        if (chamferBoundaries && isCorner &&
            isChamferNormal(outerPts[i]) !== isChamferNormal(outerPts[j])) {
            // Use angle-based detection: check which boundary angle falls
            // between mouthAngles[i] and mouthAngles[j].  This avoids the
            // chord-projection approach which fails when the boundary is near
            // a chord endpoint (t < 0.01 or t > 0.99).
            const twoPi = Math.PI * 2;
            const ai = ((mouthAngles[i] % twoPi) + twoPi) % twoPi;
            const aj = ((mouthAngles[j] % twoPi) + twoPi) % twoPi;
            // Angular span from i to j (CCW)
            const span = ((aj - ai) % twoPi + twoPi) % twoPi || twoPi;
            for (const cb of chamferBoundaries) {
                const ba = ((cb.angle % twoPi) + twoPi) % twoPi;
                const dist = ((ba - ai) % twoPi + twoPi) % twoPi;
                // Boundary falls strictly between i and j (with small margin
                // to avoid near-duplicate vertices)
                if (dist > 0.001 && dist < span - 0.001) {
                    boundaryT = dist / span;
                    boundaryExactAngle = cb.angle;
                    break;
                }
            }
        }

        if (roundoverArcStep < Infinity) {
            const dist = Math.hypot(outerPts[j].x - outerPts[i].x, outerPts[j].z - outerPts[i].z);
            // Corners keep the original fine step. Top/bottom spans get a slightly
            // coarser adaptive split so their front/back roundover reads smoother
            // without globally increasing enclosure density.
            const targetStep = isCorner
                ? roundoverArcStep
                : (isBottom
                    ? roundoverArcStep * 0.7
                    : (isTopOrBottom ? roundoverArcStep * 1.25 : Infinity));
            const subdivs = targetStep < Infinity
                ? Math.max(0, Math.ceil(dist / targetStep) - 1)
                : 0;

            if (boundaryT > 0 && subdivs > 0) {
                // Split subdivisions: some before the boundary, some after.
                // First emit the boundary point itself, then fill each side.
                const beforeCount = Math.max(0, Math.round(subdivs * boundaryT) - 1);
                const afterCount = Math.max(0, subdivs - beforeCount - 1);
                for (let k = 1; k <= beforeCount; k++) {
                    const t = boundaryT * k / (beforeCount + 1);
                    const px = outerPts[i].x + (outerPts[j].x - outerPts[i].x) * t;
                    const pz = outerPts[i].z + (outerPts[j].z - outerPts[i].z) * t;
                    refined.push(Math.atan2(pz - cz, px - cx));
                }
                // The boundary point itself — use the exact angle, not chord
                // interpolation, to avoid angle→ray-cast roundtrip error
                refined.push(boundaryExactAngle);
                for (let k = 1; k <= afterCount; k++) {
                    const t = boundaryT + (1 - boundaryT) * k / (afterCount + 1);
                    const px = outerPts[i].x + (outerPts[j].x - outerPts[i].x) * t;
                    const pz = outerPts[i].z + (outerPts[j].z - outerPts[i].z) * t;
                    refined.push(Math.atan2(pz - cz, px - cx));
                }
            } else if (boundaryT > 0) {
                // No subdivisions but we have a boundary — insert exact angle
                refined.push(boundaryExactAngle);
            } else {
                for (let k = 1; k <= subdivs; k++) {
                    const t = k / (subdivs + 1);
                    const px = outerPts[i].x + (outerPts[j].x - outerPts[i].x) * t;
                    const pz = outerPts[i].z + (outerPts[j].z - outerPts[i].z) * t;
                    refined.push(Math.atan2(pz - cz, px - cx));
                }
            }
        } else if (boundaryT > 0) {
            // No roundover subdivisions but still insert the exact boundary angle
            refined.push(boundaryExactAngle);
        }
        // Preserve direct index mapping from mouth-angle index -> refined index.
        if (i < n - 1) mapping.push(refined.length);
    }
    return { refined, mapping };
}

/**
 * Fan-stitch between a small ring (sSize points) and a larger ring (lSize
 * points).  `mapping[i]` gives the index in the large ring corresponding to
 * small-ring point i.  Between mapping[i] and mapping[i+1] there may be
 * extra large-ring vertices that fan out from small-ring vertex i.
 *
 * Mapping is expected to be monotonic non-decreasing across non-wrap
 * iterations. When mapping[i] === mapping[i+1] (empty span — two adjacent
 * small-ring vertices share the same large-ring target, e.g. because the
 * small ring is a dedup'd collapse of the large ring), only a bridge
 * triangle (sA, sB, large[mapping[i+1]]) is emitted. The true wrap-around
 * case (proper lE < lS at the closing iteration) still expands the fan
 * over the full lS → lSize-1 → 0 → lE arc.
 */
function fanStitchRings(indices, pushTri, sStart, sSize, lStart, lSize, mapping, wrap = true) {
    const limit = wrap ? sSize : sSize - 1;
    for (let i = 0; i < limit; i++) {
        const i2 = (i + 1) % sSize;
        const lS = mapping[i];
        const lE = mapping[i2];
        const sA = sStart + i;
        const sB = sStart + i2;
        const isWrapIter = wrap && i === sSize - 1;

        if (isWrapIter && lE < lS) {
            // True wrap-around: lS → lSize-1 → 0 → lE
            for (let k = lS; k < lSize - 1; k++) {
                pushTri(sA, lStart + k + 1, lStart + k);
            }
            pushTri(sA, lStart + 0, lStart + lSize - 1);
            for (let k = 0; k < lE; k++) {
                pushTri(sA, lStart + k + 1, lStart + k);
            }
            pushTri(sA, sB, lStart + lE);
        } else {
            // Normal span (possibly empty when lE === lS — only the bridge triangle is emitted).
            for (let k = lS; k < lE; k++) {
                pushTri(sA, lStart + k + 1, lStart + k);
            }
            pushTri(sA, sB, lStart + lE);
        }
    }
}

export function addEnclosureGeometry(vertices, indices, params, verticalOffset = 0, quadrantInfo = null, groupInfo = null, ringCount = null, angleList = null) {
    const ringSize = Number.isFinite(ringCount) && ringCount > 0
        ? ringCount
        : Math.max(2, Math.round(params.angularSegments || 0));

    const lastRowStart = params.lengthSegments * ringSize;
    const mouthY = vertices[lastRowStart * 3 + 1];

    const depth = parseFloat(params.encDepth) || 0;
    const edgeR = parseFloat(params.encEdge) || 0;
    const edgeType = parseInt(params.encEdgeType) || 1;
    const axialSegs = edgeR > 0 ? Math.max(4, parseInt(params.cornerSegments) || 4) : 1;
    const backY = mouthY - depth;

    let maxX = -Infinity, minX = Infinity, maxZ = -Infinity, minZ = Infinity;
    for (let i = 0; i < ringSize; i++) {
        const idx = lastRowStart + i;
        const mx = vertices[idx * 3];
        const mz = vertices[idx * 3 + 2];
        maxX = Math.max(maxX, mx);
        minX = Math.min(minX, mx);
        maxZ = Math.max(maxZ, mz);
        minZ = Math.min(minZ, mz);
    }

    const scale = params.scale || 1;
    if (params.useAthEnclosureRounding) {
        if (Number.isFinite(maxX)) maxX = Math.ceil(maxX / scale) * scale;
        if (Number.isFinite(minX)) minX = Math.floor(minX / scale) * scale;
        if (Number.isFinite(maxZ)) maxZ = Math.ceil(maxZ / scale) * scale;
        if (Number.isFinite(minZ)) minZ = Math.floor(minZ / scale) * scale;
    }

    // Compute enclosure box boundaries
    const sL = parseFloat(params.encSpaceL) || 25;
    const sT = parseFloat(params.encSpaceT) || 25;
    const sR = parseFloat(params.encSpaceR) || 25;
    const sB = parseFloat(params.encSpaceB) || 25;

    let boxRight = maxX + sR;
    let boxLeft = minX - sL;
    let boxTop = maxZ + sT + verticalOffset;
    let boxBot = minZ - sB + verticalOffset;

    // Centroid for ray-casting
    let mCx = 0, mCz = 0;
    for (let i = 0; i < ringSize; i++) {
        const idx = lastRowStart + i;
        mCx += vertices[idx * 3];
        mCz += vertices[idx * 3 + 2];
    }
    mCx /= ringSize;
    mCz /= ringSize;

    const cx = mCx;
    const cz = mCz;

    // --- Step 3: Generate enclosure points using mouth angles ---
    // Ring 0 always uses the mouth's angle list for a 1:1 baffle stitch.
    const mouthAngles = (Array.isArray(angleList) && angleList.length === ringSize)
        ? angleList
        : Array.from({ length: ringSize }, (_, i) => (i / ringSize) * Math.PI * 2);

    const mouthResult = generateEnclosurePointsFromAngles(
        mouthAngles, cx, cz,
        boxLeft, boxRight, boxBot, boxTop,
        edgeR, edgeType,
        params
    );
    const clampedEdgeR = mouthResult.clampedBoxCR;
    const edgeDepth = Math.min(clampedEdgeR || 0, Math.max(0, depth * 0.49));
    const edgeSlices = edgeDepth > 0 ? Math.max(1, axialSegs) : 0;

    // --- Step 3b: Refine angles at corners to match Y-axis roundover ---
    const { refined: refinedAngles, mapping: mouthToRefinedMap } =
        refineAnglesForEnclosure(mouthAngles, mouthResult.outerPts, edgeSlices, edgeDepth, cx, cz,
            edgeType, boxLeft, boxRight, boxBot, boxTop, clampedEdgeR);
    const refinedSize = refinedAngles.length;
    const addedPts = refinedSize - ringSize;

    // Generate enclosure points for the refined angle set (used for all body rings)
    let outerPts, insetPts;
    if (addedPts > 0) {
        const refinedResult = generateEnclosurePointsFromAngles(
            refinedAngles, cx, cz,
            boxLeft, boxRight, boxBot, boxTop,
            edgeR, edgeType,
            params
        );
        outerPts = refinedResult.outerPts;
        insetPts = refinedResult.insetPts;
    } else {
        outerPts = mouthResult.outerPts;
        insetPts = mouthResult.insetPts;
    }

    // For chamfer: snap boundary vertices to exact flat/chamfer transition
    // positions and set correct normals + insetPts.  The ray-cast places them
    // very close (exact angle is used) but floating-point roundtrip can still
    // leave sub-ULP drift.  More importantly, the boundary vertex may get a
    // chamfer normal from the ray-cast, which would produce an incorrect
    // insetPt (offset along the chamfer normal instead of converging to the
    // arc center).  We fix both position AND normal/insetPt here.
    if (edgeType === 2 && clampedEdgeR > 1e-6) {
        const halfW = (boxRight - boxLeft) / 2;
        const halfH = (boxTop - boxBot) / 2;
        const bCx = (boxRight + boxLeft) / 2;
        const bCz = (boxTop + boxBot) / 2;
        const r = clampedEdgeR;
        const corners = [
            { acx: bCx + halfW - r, acz: bCz - halfH + r, start: -Math.PI / 2, end: 0 },
            { acx: bCx + halfW - r, acz: bCz + halfH - r, start: 0, end: Math.PI / 2 },
            { acx: bCx - halfW + r, acz: bCz + halfH - r, start: Math.PI / 2, end: Math.PI },
            { acx: bCx - halfW + r, acz: bCz - halfH + r, start: Math.PI, end: Math.PI * 1.5 }
        ];
        // Build boundary info with the flat-face normal and arc center for each
        const boundaryInfo = [];
        for (const c of corners) {
            // Start boundary: where the previous flat face ends and chamfer begins.
            // The flat-face normal is tangent to the arc at the start angle.
            const startCos = Math.cos(c.start), startSin = Math.sin(c.start);
            // End boundary: where chamfer ends and the next flat face begins.
            const endCos = Math.cos(c.end), endSin = Math.sin(c.end);
            boundaryInfo.push(
                {
                    x: c.acx + r * startCos, z: c.acz + r * startSin,
                    // Flat-face normal at chamfer start (perpendicular to the
                    // flat face that ends here)
                    nx: Math.abs(startCos) > 0.5 ? Math.sign(startCos) : 0,
                    nz: Math.abs(startSin) > 0.5 ? Math.sign(startSin) : 0,
                    acx: c.acx, acz: c.acz
                },
                {
                    x: c.acx + r * endCos, z: c.acz + r * endSin,
                    nx: Math.abs(endCos) > 0.5 ? Math.sign(endCos) : 0,
                    nz: Math.abs(endSin) > 0.5 ? Math.sign(endSin) : 0,
                    acx: c.acx, acz: c.acz
                }
            );
        }
        const snapDist = r * 0.15;
        for (const bi of boundaryInfo) {
            let bestIdx = -1;
            let bestD = snapDist;
            for (let pi = 0; pi < outerPts.length; pi++) {
                const d = Math.hypot(outerPts[pi].x - bi.x, outerPts[pi].z - bi.z);
                if (d < bestD) { bestD = d; bestIdx = pi; }
            }
            if (bestIdx >= 0) {
                // Snap position to exact boundary
                outerPts[bestIdx].x = bi.x;
                outerPts[bestIdx].z = bi.z;
                // Set the flat-face normal so that downstream insetPt
                // computation (and roundover interpolation) stays on the
                // correct face plane
                outerPts[bestIdx].nx = bi.nx;
                outerPts[bestIdx].nz = bi.nz;
                // Set insetPt directly to the arc center — this is where all
                // chamfer roundover edges must converge
                insetPts[bestIdx].x = bi.acx;
                insetPts[bestIdx].z = bi.acz;
            }
        }
    }

    const bodySize = refinedSize;
    const mouthOuterPts = mouthResult.outerPts;
    const mouthInsetPts = mouthResult.insetPts;

    // --- Helpers (needed before any geometry) ---
    const triangleArea2 = (a, b, c) => {
        const ax = vertices[a * 3], ay = vertices[a * 3 + 1], az = vertices[a * 3 + 2];
        const bx = vertices[b * 3], by = vertices[b * 3 + 1], bz = vertices[b * 3 + 2];
        const ccx = vertices[c * 3], cy = vertices[c * 3 + 1], cz = vertices[c * 3 + 2];
        const abx = bx - ax, aby = by - ay, abz = bz - az;
        const acx = ccx - ax, acy = cy - ay, acz = cz - az;
        const nx = aby * acz - abz * acy;
        const ny = abz * acx - abx * acz;
        const nz = abx * acy - aby * acx;
        return Math.hypot(nx, ny, nz);
    };
    const pushTri = (a, b, c) => {
        if (a === b || b === c || c === a) return;
        if (triangleArea2(a, b, c) <= 1e-10) return;
        indices.push(a, b, c);
    };
    const fullCircle = !quadrantInfo || quadrantInfo.fullCircle;
    // Winding: reversed row order from createHornIndices so enclosure normals
    // face outward (away from horn). BFS-consistent with the horn across the
    // mouth-to-ring0 stitch boundary.
    const stitchRing = (r1Start, r2Start, size) => {
        const limit = fullCircle ? size : size - 1;
        for (let i = 0; i < limit; i++) {
            const i2 = (i + 1) % size;
            pushTri(r2Start + i, r1Start + i, r1Start + i2);
            pushTri(r2Start + i, r1Start + i2, r2Start + i2);
        }
    };

    // --- Step 4: Ring 0 — mouth-aligned inset ring for baffle stitch ---
    const mergeEps = 1e-6 * scale;
    let reuseMouthAsRing0 = true;
    for (let i = 0; i < ringSize; i++) {
        const ipt = mouthInsetPts[i];
        const mouthX = vertices[(lastRowStart + i) * 3];
        const mouthZ = vertices[(lastRowStart + i) * 3 + 2];
        if (Math.hypot(ipt.x - mouthX, ipt.z - mouthZ) > mergeEps) {
            reuseMouthAsRing0 = false;
            break;
        }
    }

    // Ring 0 may have fewer unique vertices than the refined bodySize when
    // rounded-corner arcs collapse multiple arc-point insets to the same
    // arc-center position (inset = hit - normal * r, and the arc's outward
    // normal multiplied by r exactly lands on the arc center for every arc
    // hit). Pushing one ring0 vertex per refined index in that case creates
    // near-coincident vertices — they pass the manifold check but sit on the
    // ring-adjacent edge of their neighbour, producing T-junctions that
    // cascade through every roundover pass above ring 0.
    //
    // Downstream rings (radialT > 0) interpolate toward distinct outer
    // points, so they do NOT collapse; only ring 0 needs deduplication.
    //
    // ring0Group[refinedIdx] = offset into the ring0 ring (0..ring0UniqueSize)
    // ring0FirstRefined[g]  = first refined index in unique group g
    let ring0Start;
    let ring0UniqueSize;
    let ring0Group;
    let ring0FirstRefined;
    if (reuseMouthAsRing0) {
        ring0Start = lastRowStart;
        ring0UniqueSize = ringSize;
        ring0Group = Array.from({ length: ringSize }, (_, i) => i);
        ring0FirstRefined = Array.from({ length: ringSize }, (_, i) => i);
    } else {
        const seamNudge = 1e-4 * scale;
        ring0Start = vertices.length / 3;
        const ring0Size = addedPts > 0 ? bodySize : ringSize;
        const ring0Pts = addedPts > 0 ? insetPts : mouthInsetPts;
        const insetMergeEps = 1e-6 * scale;
        ring0Group = new Array(ring0Size);
        ring0FirstRefined = [];
        const uniqueXZ = [];
        for (let i = 0; i < ring0Size; i++) {
            const ipt = ring0Pts[i];
            let g = -1;
            for (let k = 0; k < uniqueXZ.length; k++) {
                const p = uniqueXZ[k];
                if (Math.abs(p.x - ipt.x) < insetMergeEps && Math.abs(p.z - ipt.z) < insetMergeEps) {
                    g = k; break;
                }
            }
            if (g < 0) {
                g = uniqueXZ.length;
                uniqueXZ.push({ x: ipt.x, z: ipt.z });
                ring0FirstRefined.push(i);
                vertices.push(
                    ipt.x - (ipt.nx || 0) * seamNudge,
                    mouthY,
                    ipt.z - (ipt.nz || 0) * seamNudge
                );
            }
            ring0Group[i] = g;
        }
        ring0UniqueSize = uniqueXZ.length;
    }

    const enclosureStartTri = indices.length / 3;

    // Mouth to ring 0 stitch
    if (!reuseMouthAsRing0) {
        const ring0PreDedupSize = addedPts > 0 ? bodySize : ringSize;
        if (ring0UniqueSize < ring0PreDedupSize || addedPts > 0) {
            // Fan-stitch mouth (ringSize) → ring0 (ring0UniqueSize) on the
            // coplanar front baffle plane where fan triangles are invisible.
            // Mapping composes mouthToRefinedMap with ring0Group so that
            // collapsed-corner refined indices resolve to a single ring0 vertex.
            const mouthToRing0Group = mouthToRefinedMap.map((r) => ring0Group[r]);
            fanStitchRings(indices, pushTri, lastRowStart, ringSize, ring0Start, ring0UniqueSize, mouthToRing0Group, fullCircle);
        } else {
            const limit = fullCircle ? ringSize : ringSize - 1;
            for (let i = 0; i < limit; i++) {
                const i2 = (i + 1) % ringSize;
                pushTri(lastRowStart + i, lastRowStart + i2, ring0Start + i);
                pushTri(lastRowStart + i2, ring0Start + i2, ring0Start + i);
            }
        }
    }
    const flatFrontEndTri = indices.length / 3;

    // --- Step 5: Front roundover rings ---
    // When corner refinement is active (addedPts > 0), rings 1..edgeSlices use
    // the refined bodySize point set so the front roundover has the same
    // density as the sidewalls and back roundover. Ring 0 may have fewer
    // unique vertices (see dedup above), so the first pass uses a fan-stitch;
    // subsequent passes are 1:1 between same-size rings.
    let prevRing = ring0Start;
    let prevRingSize = ring0UniqueSize;
    const frontRingSize = addedPts > 0 ? bodySize : ringSize;
    const frontInsetPts = addedPts > 0 ? insetPts : mouthInsetPts;
    const frontOuterPts = addedPts > 0 ? outerPts : mouthOuterPts;
    const frontRoundoverPassRanges = [];
    for (let j = 1; j <= edgeSlices; j++) {
        const ringIdx = vertices.length / 3;
        const t = j / edgeSlices;
        let axialT = t, radialT = t;
        if (edgeType === 1) {
            const angle = t * (Math.PI / 2);
            axialT = 1 - Math.cos(angle);
            radialT = Math.sin(angle);
        }
        const y = mouthY - (axialT * edgeDepth);
        for (let i = 0; i < frontRingSize; i++) {
            const ipt = frontInsetPts[i];
            const opt = frontOuterPts[i];
            vertices.push(
                ipt.x + (opt.x - ipt.x) * radialT,
                y,
                ipt.z + (opt.z - ipt.z) * radialT
            );
        }
        const passStart = indices.length / 3;
        if (prevRingSize === frontRingSize) {
            stitchRing(prevRing, ringIdx, frontRingSize);
        } else {
            // First pass only — fan-stitch ring0 unique (< bodySize) → ring1 bodySize.
            fanStitchRings(indices, pushTri, prevRing, prevRingSize, ringIdx, frontRingSize, ring0FirstRefined, fullCircle);
        }
        frontRoundoverPassRanges.push({
            start: passStart,
            end: indices.length / 3,
            upperRingStart: ringIdx
        });
        prevRing = ringIdx;
        prevRingSize = frontRingSize;
    }
    const frontRoundoverEndTri = indices.length / 3;
    const frontRoundoverEnd = prevRing;

    // --- Step 6: Sidewall (1:1 stitch, uniform ring size) ---
    // Both the front roundover end and back ring now use bodySize points,
    // so a simple 1:1 stitch replaces the previous fan-stitch.  The fan-stitch
    // (when needed) has been moved to the coplanar front baffle where it is
    // invisible, eliminating corner artifacts on the curved sidewall surface.
    const outerBackY = edgeDepth > 0 ? backY + edgeDepth : backY;
    const backRingStart = vertices.length / 3;
    for (let i = 0; i < bodySize; i++) {
        const opt = outerPts[i];
        vertices.push(opt.x, outerBackY, opt.z);
    }
    stitchRing(frontRoundoverEnd, backRingStart, bodySize);
    const sideWallEndTri = indices.length / 3;

    // --- Step 7: Back roundover rings (refined ring size) ---
    let currentRingStart = backRingStart;
    const backRoundoverPassRanges = [];
    for (let j = 1; j <= edgeSlices; j++) {
        const t = j / edgeSlices;
        let axialT = t;
        let radialT = 1 - t;
        if (edgeType === 1) {
            const angle = t * (Math.PI / 2);
            axialT = Math.sin(angle);
            radialT = Math.cos(angle);
        }
        if (j === edgeSlices) {
            radialT = Math.max(radialT, 1e-3);
        }
        const y = backY + (1 - axialT) * edgeDepth;
        const ringStart = vertices.length / 3;
        for (let i = 0; i < bodySize; i++) {
            const ipt = insetPts[i];
            const opt = outerPts[i];
            vertices.push(
                ipt.x + (opt.x - ipt.x) * radialT,
                y,
                ipt.z + (opt.z - ipt.z) * radialT
            );
        }
        const passStart = indices.length / 3;
        stitchRing(currentRingStart, ringStart, bodySize);
        backRoundoverPassRanges.push({
            start: passStart,
            end: indices.length / 3,
            upperRingStart: ringStart
        });
        currentRingStart = ringStart;
    }
    const backRoundoverEndTri = indices.length / 3;

    // --- Step 8: Back Cap ---
    let avgX = 0, avgZ = 0;
    const capBoundary = Array.from({ length: bodySize }, (_, i) => ({
        x: vertices[(currentRingStart + i) * 3],
        z: vertices[(currentRingStart + i) * 3 + 2]
    }));
    for (let i = 0; i < bodySize; i++) {
        avgX += capBoundary[i].x;
        avgZ += capBoundary[i].z;
    }
    avgX /= bodySize;
    avgZ /= bodySize;

    let capRingStart = currentRingStart;
    const capSliceRanges = [];
    if (fullCircle) {
        const capSlices = 3;
        for (let s = 1; s < capSlices; s++) {
            const blend = s / capSlices;
            const ringStart = vertices.length / 3;
            for (let i = 0; i < bodySize; i++) {
                const cp = capBoundary[i];
                vertices.push(
                    cp.x + (avgX - cp.x) * blend,
                    backY,
                    cp.z + (avgZ - cp.z) * blend
                );
            }
            const sliceStart = indices.length / 3;
            stitchRing(capRingStart, ringStart, bodySize);
            capSliceRanges.push({
                start: sliceStart,
                end: indices.length / 3,
                upperRingStart: ringStart
            });
            capRingStart = ringStart;
        }
    }

    const capStart = vertices.length / 3;
    vertices.push(avgX, backY, avgZ);
    const apexFanStart = indices.length / 3;
    const capLimit = fullCircle ? bodySize : bodySize - 1;
    for (let i = 0; i < capLimit; i++) {
        const i2 = (i + 1) % bodySize;
        pushTri(capRingStart + i, capRingStart + i2, capStart);
    }
    const apexFanRange = { start: apexFanStart, end: indices.length / 3 };
    const backCapEndTri = indices.length / 3;

    const enclosureEndTri = indices.length / 3;
    if (groupInfo) {
        groupInfo.enclosure = { start: enclosureStartTri, end: enclosureEndTri };
        groupInfo.enc_front = { start: enclosureStartTri, end: flatFrontEndTri };
        groupInfo.enc_edge = [
            { start: flatFrontEndTri, end: frontRoundoverEndTri },
            { start: sideWallEndTri, end: backRoundoverEndTri }
        ];
        groupInfo.enc_side = { start: frontRoundoverEndTri, end: sideWallEndTri };
        groupInfo.enc_rear = { start: backRoundoverEndTri, end: backCapEndTri };
    }

    // ---- Per-region seam assertions ----
    //
    // Build a list of adjacent triangle-band pairs ("seams"). Each seam is
    // defined by its ring: two bands on either side of a ring must emit the
    // SAME set of ring-adjacent edges (catches off-by-one stitching bugs) and
    // those edges must be opposite-winding (catches corner/winding bugs).
    //
    // Expected shared count is derived from the mesh itself (ring edges each
    // band actually emits). This keeps the check strict for the common case
    // while tolerating legitimate chamfer-corner collapse where a few ring
    // vertices coincide and their adjacent edges degenerate symmetrically.
    //
    // The seams cover the hardest stitch bugs (fan-stitch corner artifacts,
    // chamfer-boundary mis-snap, off-by-one at ring0 reuse) and fire a
    // region-named error identifying the broken stitch instead of the
    // global validateMeshQuality pass reporting an aggregate count.

    // Actual ring0 location and size after the reuse/build decision above.
    // When reuseMouthAsRing0, ring0 IS the mouth (shared vertex window); the
    // flat-front band is empty and the first enclosure band is the first
    // front roundover pass. Otherwise ring0 was freshly pushed at ring0Start
    // with `ring0UniqueSize` vertices (fewer than `bodySize` when rounded
    // corners collapse multiple arc-point insets to the same arc center).
    const ring0VertexStart = reuseMouthAsRing0 ? lastRowStart : ring0Start;
    const ring0VertexSize = reuseMouthAsRing0 ? ringSize : ring0UniqueSize;

    // Assemble an ordered list of bands. `nextRingStart`/`nextRingSize`
    // describe the vertex window of the ring BETWEEN this band and the next.
    // Each vertex window was recorded at the moment the corresponding ring
    // was pushed (explicit `upperRingStart`), so validation does not rely on
    // index-inference heuristics.
    const bands = [];
    if (flatFrontEndTri > enclosureStartTri) {
        bands.push({
            name: 'flat-front (mouth→ring0)',
            range: { start: enclosureStartTri, end: flatFrontEndTri },
            nextRingStart: ring0VertexStart,
            nextRingSize: ring0VertexSize
        });
    }
    for (let j = 0; j < frontRoundoverPassRanges.length; j++) {
        const r = frontRoundoverPassRanges[j];
        bands.push({
            name: `front-roundover-pass-${j}`,
            range: { start: r.start, end: r.end },
            nextRingStart: r.upperRingStart,
            nextRingSize: frontRingSize
        });
    }
    if (sideWallEndTri > frontRoundoverEndTri) {
        bands.push({
            name: 'sidewall',
            range: { start: frontRoundoverEndTri, end: sideWallEndTri },
            nextRingStart: backRingStart,
            nextRingSize: bodySize
        });
    }
    for (let j = 0; j < backRoundoverPassRanges.length; j++) {
        const r = backRoundoverPassRanges[j];
        bands.push({
            name: `back-roundover-pass-${j}`,
            range: { start: r.start, end: r.end },
            nextRingStart: r.upperRingStart,
            nextRingSize: bodySize
        });
    }
    for (let k = 0; k < capSliceRanges.length; k++) {
        const r = capSliceRanges[k];
        bands.push({
            name: `cap-slice-${k}`,
            range: { start: r.start, end: r.end },
            nextRingStart: r.upperRingStart,
            nextRingSize: bodySize
        });
    }
    if (apexFanRange.end > apexFanRange.start) {
        bands.push({
            name: 'apex-fan',
            range: apexFanRange,
            nextRingStart: null,  // terminal — apex point, no outgoing seam
            nextRingSize: null
        });
    }

    const seams = [];
    // Seam 0: horn ↔ first enclosure band. The ring at this seam is the
    // horn mouth; mouth vertices begin at lastRowStart with `ringSize` points.
    if (bands.length > 0) {
        seams.push({
            name: `horn-mouth ↔ enclosure-${bands[0].name}`,
            rangeA: { start: 0, end: enclosureStartTri },
            rangeB: bands[0].range,
            ringStart: lastRowStart,
            ringSize,
            fullCircle,
            expectedAllOpposite: true,
            expectedMinShared: 1
        });
    }
    // Seams between consecutive enclosure bands. Skip terminal bands (no
    // outgoing ring) and any band where the ring offset is unresolved.
    for (let i = 0; i < bands.length - 1; i++) {
        const a = bands[i];
        const b = bands[i + 1];
        if (a.nextRingStart === null || a.nextRingSize === null) continue;
        seams.push({
            name: `${a.name} ↔ ${b.name}`,
            rangeA: a.range,
            rangeB: b.range,
            ringStart: a.nextRingStart,
            ringSize: a.nextRingSize,
            fullCircle,
            expectedAllOpposite: true,
            expectedMinShared: 1
        });
    }

    const seamReports = validateEnclosureSeams(indices, seams);
    if (groupInfo) {
        groupInfo.enclosureSeams = {
            specs: seams,
            reports: seamReports
        };
    }
}
