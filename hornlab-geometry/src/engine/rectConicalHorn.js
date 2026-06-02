/**
 * Rectangular Conical Horn — shared parameter builder.
 *
 * Single-driver rectangular conical horn with optional anti-waistbanding
 * second flare, an optional OSSE/Quadratic throat
 * adapter prepended to the body, and a rounded-rect cross-section that
 * lofts smoothly from a circular driver entry through the curved throat
 * to the rectangular body.
 *
 * Geometry:
 *   - Per-axis Classical conical body (H + V).  Each axis carries its own
 *     primary coverage angle (`primary_H_deg`, `primary_V_deg`) and an
 *     optional second flare (`flare2_H_deg`, `flare2_V_deg`) starting at
 *     `flare2_ratio = D23 / D34` of the locked mouth dimension.
 *   - Optional throat adapter: prepended to the body.  The driver-side
 *     half-angle α₀ is fixed by `throat_driver_deg` (default 15.5° per side
 *     = 31° total, matching a JBL 1″ CD exit).  A closed-form forward-solve
 *     computes per-axis body-throat radius (R_H_BODY, R_V_BODY) so each
 *     axis is tangent-continuous at the throat→body junction.
 *   - Cross-section uses `aspectRatioMode: 'rectFillet'` with a
 *     `filletRadiusTimeline` that lofts the corner radius from the driver
 *     circle at z=0 to the body's `body_fillet_mm` at the body junction.
 *   - Optional enclosure (cabinet) with rounded baffle edge.
 *
 * The body produces a clean rounded-rect prism — flat-plywood-buildable —
 * and the throat is the only 3D-printed (curved) section.
 *
 * Shared between:
 *   - `scripts/build_rect_horn_param.mjs` (CLI entry that writes .msh files)
 *   - The "Rectangular Conical Horn" preset in the Waveguide Generator UI
 *
 * The script must continue to produce byte-identical .meta.json + .msh
 * outputs for any given input params, so the helpers exported here are the
 * canonical implementation — neither caller forks the math locally.
 */

// ---------------------------------------------------------------------------
// Defaults
//
// Two sets, both serving the same parameter schema:
//
//   RECT_HORN_PRESET_DEFAULTS — values surfaced by the WG UI's "Rectangular
//     Conical Horn" preset.  These are the curated starting point a user
//     sees when they pick the preset (anti-waistbanding 2nd flare, 1″ OSSE
//     adapter, 8 mm body corners, rounded baffle).
//
//   RECT_HORN_BUILDER_DEFAULTS — values used by the CLI (build_rect_horn_param.mjs)
//     when a key is OMITTED from the input JSON.  These match the original
//     script defaults so the CLI's behaviour for bare-minimum input JSONs is
//     preserved across the refactor.
//
// Both run through the same `buildRectConicalHornParams` pipeline.  The
// caller picks the default bundle they want before merging user input.
// ---------------------------------------------------------------------------

// `enc_depth_mm` semantics:
//   - `null` (default): "auto" — enclosure cabinet extends 30 mm past the
//     deepest point of the horn (TOTAL_LENGTH + 30).  This matches the
//     original hardcoded behaviour, so existing CLI invocations that omit
//     the key produce byte-identical output.
//   - `0`: freestanding — no enclosure cabinet at all.  The OCC mesher
//     sees `enc_depth = 0`, `enc_edge = 0`, all `encSpace_*` margins = 0,
//     and `cornerSegments = 1`.  Combined with `wall_thickness_mm > 0`
//     this yields a closed, thick-walled tube with a mouth ring — suitable
//     for 3D-printing or a standalone (non-cabinet) mount.
//   - `> 0`: explicit margin in mm past the deepest point of the horn
//     (replaces the implicit 30 mm).
//
// `wall_thickness_mm` semantics:
//   - `0` (default): zero-thickness shell, matching the legacy CLI behaviour.
//   - `> 0`: the horn walls are thickened by this many mm in the OCC build.
//     12 mm is a typical 3D-print target.

export const RECT_HORN_PRESET_DEFAULTS = Object.freeze({
  primary_H_deg: 90,
  primary_V_deg: 60,
  flare2_ratio: 0.75,
  flare2_H_deg: 135,
  flare2_V_deg: 115,
  mouth_height_mm: 381,        // 15"
  throat_dia_mm: 25.4,         // 1"
  throat_length_mm: 35,
  throat_type: 'osse',         // 'osse' | 'quadratic'
  throat_driver_deg: 15.5,     // JBL 1" CD exit: 31° total → 15.5° per side
  body_fillet_mm: 8,
  enc_edge_mm: 18,
  enc_depth_mm: null,          // null = auto (TOTAL_LENGTH + 30); 0 = freestanding
  wall_thickness_mm: 0,        // > 0 = thickened walls (3D-print / standalone)
  // Mesh resolutions (mm) — used by the gmsh-OCC adaptive mesher.
  throat_res_mm: 6,
  mouth_res_mm: 20,
  rear_res_mm: 40,
  enc_front_res_mm: 20,
  enc_back_res_mm: 40,
});

// Original script defaults — preserved exactly for byte-identical CLI parity.
export const RECT_HORN_BUILDER_DEFAULTS = Object.freeze({
  primary_H_deg: 90,
  primary_V_deg: 50,
  flare2_ratio: 0.65,
  flare2_H_deg: 135,
  flare2_V_deg: 120,
  mouth_height_mm: 381,
  throat_dia_mm: 25.4,
  throat_length_mm: 0,
  throat_type: 'osse',
  throat_driver_deg: 15.5,
  body_fillet_mm: 5,
  enc_edge_mm: 0,
  enc_depth_mm: null,          // null = auto (TOTAL_LENGTH + 30); 0 = freestanding
  wall_thickness_mm: 0,        // > 0 = thickened walls (3D-print / standalone)
  throat_res_mm: 6,
  mouth_res_mm: 20,
  rear_res_mm: 40,
  enc_front_res_mm: 20,
  enc_back_res_mm: 40,
});

// Backwards-compatible alias: anything that imported `RECT_HORN_DEFAULTS`
// before the split gets the preset defaults (the user-facing one).
export const RECT_HORN_DEFAULTS = RECT_HORN_PRESET_DEFAULTS;

// ---------------------------------------------------------------------------
// Throat-adapter forward-solve helpers.
//
// Given a fixed driver-entry radius r0 and a target body coverage half-angle,
// solve for the body-throat radius R such that r'(L) matches tan(half-angle)
// at the body junction.
//
// OSSE:        r(z)² = r0² + 2·r0·z·tan(α₀) + tan(α_body)²·z²
//              R = (L·m1 + sqrt(L²·m1² + 4·r0² + 4·r0·m0·L)) / 2
//
// Quadratic:   r(z) = r0 + m0·z + (m1−m0)·z²/(2L)
//              R = r0 + L·(m0 + m1)/2          (slope linear in z)
//
// Where m0 = tan(α₀)  (driver-side half-angle, user input)
//       m1 = tan(α_body)  (body coverage half-angle)
// ---------------------------------------------------------------------------

export function bodyThroatRadiusOsse(L, halfAngleRad, m0, r0_start) {
  if (!(L > 0)) return r0_start;
  const m1 = Math.tan(halfAngleRad);
  const inner = L * L * m1 * m1 + 4 * r0_start * r0_start + 4 * r0_start * m0 * L;
  if (!(inner >= 0)) return r0_start;
  return (L * m1 + Math.sqrt(inner)) / 2;
}

export function bodyThroatRadiusQuadratic(L, halfAngleRad, m0, r0_start) {
  if (!(L > 0)) return r0_start;
  const m1 = Math.tan(halfAngleRad);
  return r0_start + L * (m0 + m1) / 2;
}

export function bodyThroatRadius(L, halfAngleRad, m0, r0_start, throatType) {
  if (throatType === 'quadratic') {
    return bodyThroatRadiusQuadratic(L, halfAngleRad, m0, r0_start);
  }
  return bodyThroatRadiusOsse(L, halfAngleRad, m0, r0_start);
}

// ---------------------------------------------------------------------------
// Top-level builder.
//
// Resolves user params (with defaults), runs the forward-solve, and returns
// a rich object the callers can consume:
//   - `params`: a complete top-level params object suitable to feed
//     `buildWaveguideMesh` (carries the profileSystem with axes/sections,
//     enclosure controls, and cross-section spec).
//   - `meta`: derived geometric values (mouth dimensions, body length, etc.)
//     that callers may surface to the user or write into a sidecar.
//   - `mesh`: the user-facing mesh resolutions (used by the OCC adapter
//     payload — not passed into buildWaveguideMesh).
// ---------------------------------------------------------------------------

const ANGULAR_SEGMENTS = 60;
const LENGTH_SEGMENTS  = 16;

function classicalAxisParams(coverageDeg, flare2Deg, r0_axis, bodyLength, flare2StartNorm) {
  return {
    L: bodyLength,
    r0: r0_axis,
    classicalShape: 1,
    classicalCoverageAngle: coverageDeg,
    classicalCrossSection: 0,
    classicalFlare2Enabled: 1,
    classicalFlare2Angle: flare2Deg,
    classicalFlare2Start: flare2StartNorm,
    classicalFlare2Blend: 0,
    rot: 0,
    scale: 1,
  };
}

function buildAxisSections({
  hasThroat,
  throatType,
  throatLengthMm,
  throatDriverDeg,
  coverageDeg,
  flare2Deg,
  r0_axis,
  bodyLength,
  L23,
  L34,
  flare2StartNorm,
}) {
  if (!hasThroat) return undefined;
  const bodyParams = {
    L: bodyLength,
    r0: r0_axis,
    classicalShape: 1,
    classicalCoverageAngle: coverageDeg,
    classicalCrossSection: 0,
    classicalFlare2Enabled: 0,
    rot: 0,
    scale: 1,
  };
  return [
    {
      kind: 'throat',
      type: throatType,
      length: throatLengthMm,
      params: { angle: throatDriverDeg },
    },
    {
      kind: 'body',
      type: 'Classical',
      length: L23,
      params: bodyParams,
      legacyParams: {
        classicalFlare2Enabled: 1,
        classicalFlare2Angle: flare2Deg,
        classicalFlare2Start: flare2StartNorm,
        classicalFlare2Blend: 0,
        L: bodyLength,
      },
    },
    {
      kind: 'mouth',
      type: 'flare',
      length: L34,
      params: {
        angle: 90 - flare2Deg / 2,
        blend: 'sharp',
        start: flare2StartNorm,
        fullLength: bodyLength,
        angleConvention: 'mouth-plane-v1',
      },
    },
  ];
}

/**
 * Build the full Rectangular Conical Horn params bundle from user input.
 *
 * @param {Object} userInput - User-provided params (keys override defaults).
 * @param {Object} [options]
 * @param {Object} [options.defaults] - Default bundle to merge under userInput.
 *   Defaults to RECT_HORN_PRESET_DEFAULTS (the UI preset's curated values).
 *   Pass RECT_HORN_BUILDER_DEFAULTS to preserve the CLI's original defaults
 *   when a key is omitted from the input JSON.
 *
 * Returns:
 *   {
 *     params,     // top-level params object for buildWaveguideMesh
 *     meta,       // derived geometric values
 *     mesh,       // user-facing mesh resolutions (mm)
 *     warnings,   // array of advisory strings (non-fatal)
 *   }
 *
 * Throws on degenerate inputs (negative or zero body lengths, unknown
 * throat type, etc).
 */
export function buildRectConicalHornParams(userInput = {}, options = {}) {
  const defaults = options.defaults ?? RECT_HORN_PRESET_DEFAULTS;
  const u = { ...defaults, ...userInput };

  const THETA_W_DEG  = +u.primary_H_deg;
  const THETA_H_DEG  = +u.primary_V_deg;
  const THETA_W2_DEG = +u.flare2_H_deg;
  const THETA_H2_DEG = +u.flare2_V_deg;
  const RATW         = +u.flare2_ratio;
  // Freestanding sentinel: enc_depth_mm === 0 forces enc_edge to 0 too,
  // regardless of any user-supplied enc_edge_mm (the cabinet doesn't exist,
  // so its baffle roundover is meaningless).
  const ENC_DEPTH_RAW = u.enc_depth_mm;
  const FREESTANDING = ENC_DEPTH_RAW === 0;
  const ENC_EDGE_MM  = FREESTANDING ? 0 : +u.enc_edge_mm;
  const WALL_THICKNESS_MM = +u.wall_thickness_mm || 0;
  const hasUserMouthWidth = Object.hasOwn(userInput, 'mouth_width_mm')
    && userInput.mouth_width_mm !== null
    && userInput.mouth_width_mm !== undefined;
  const hasUserMouthHeight = Object.hasOwn(userInput, 'mouth_height_mm')
    && userInput.mouth_height_mm !== null
    && userInput.mouth_height_mm !== undefined;
  if (hasUserMouthWidth && hasUserMouthHeight) {
    throw new Error('Specify only one mouth lock dimension: mouth_width_mm (H) or mouth_height_mm (V)');
  }
  const MOUTH_LOCK_AXIS = hasUserMouthWidth ? 'H' : 'V';
  const MOUTH_LOCK_MM = hasUserMouthWidth ? +userInput.mouth_width_mm : +u.mouth_height_mm;
  const THROAT_DIA_MM   = +u.throat_dia_mm;
  const BODY_FILLET_MM  = +u.body_fillet_mm;
  const THROAT_LENGTH_MM = +u.throat_length_mm;
  const THROAT_DRIVER_DEG = +u.throat_driver_deg;
  const THROAT_TYPE = String(u.throat_type ?? 'osse').toLowerCase();

  if (THROAT_LENGTH_MM > 0 && THROAT_TYPE !== 'osse' && THROAT_TYPE !== 'quadratic') {
    throw new Error(`Unknown throat_type "${THROAT_TYPE}" (expected 'osse' or 'quadratic')`);
  }

  const THROAT_RES = +u.throat_res_mm;
  const MOUTH_RES  = +u.mouth_res_mm;
  const REAR_RES   = +u.rear_res_mm;
  const ENC_FRONT_RES = +u.enc_front_res_mm;
  const ENC_BACK_RES  = +u.enc_back_res_mm;

  const R0 = THROAT_DIA_MM / 2;

  const halfV  = (THETA_H_DEG  * Math.PI / 180) / 2;
  const halfV2 = (THETA_H2_DEG * Math.PI / 180) / 2;
  const halfH  = (THETA_W_DEG  * Math.PI / 180) / 2;
  const halfH2 = (THETA_W2_DEG * Math.PI / 180) / 2;

  const HAS_THROAT = THROAT_LENGTH_MM > 0;
  const M0_DRIVER = HAS_THROAT
    ? Math.tan(THROAT_DRIVER_DEG * Math.PI / 180)
    : 0;
  const R_H_BODY = HAS_THROAT
    ? bodyThroatRadius(THROAT_LENGTH_MM, halfH, M0_DRIVER, R0, THROAT_TYPE)
    : R0;
  const R_V_BODY = HAS_THROAT
    ? bodyThroatRadius(THROAT_LENGTH_MM, halfV, M0_DRIVER, R0, THROAT_TYPE)
    : R0;

  if (!(MOUTH_LOCK_MM > 0) || !isFinite(MOUTH_LOCK_MM)) {
    throw new Error(`Invalid mouth_${MOUTH_LOCK_AXIS === 'H' ? 'width' : 'height'}_mm ${MOUTH_LOCK_MM}`);
  }

  const D34_LOCK_HALF = MOUTH_LOCK_MM / 2;
  const D23_LOCK_HALF = RATW * D34_LOCK_HALF;
  const lockPrimaryHalfAngle = MOUTH_LOCK_AXIS === 'H' ? halfH : halfV;
  const lockSecondHalfAngle = MOUTH_LOCK_AXIS === 'H' ? halfH2 : halfV2;
  const lockBodyRadius = MOUTH_LOCK_AXIS === 'H' ? R_H_BODY : R_V_BODY;
  const L23 = (D23_LOCK_HALF - lockBodyRadius) / Math.tan(lockPrimaryHalfAngle);
  const L34 = (D34_LOCK_HALF - D23_LOCK_HALF) / Math.tan(lockSecondHalfAngle);
  const BODY_LENGTH = L23 + L34;
  const TOTAL_LENGTH = THROAT_LENGTH_MM + BODY_LENGTH;
  const FLARE2_START_NORM_BODY = BODY_LENGTH > 0 ? L23 / BODY_LENGTH : 0.5;
  const FLARE2_START_NORM_TOTAL = TOTAL_LENGTH > 0
    ? (THROAT_LENGTH_MM + L23) / TOTAL_LENGTH
    : 0.5;
  const D34H_HALF = MOUTH_LOCK_AXIS === 'H'
    ? D34_LOCK_HALF
    : R_H_BODY + L23 * Math.tan(halfH) + L34 * Math.tan(halfH2);
  const D34V_HALF = MOUTH_LOCK_AXIS === 'V'
    ? D34_LOCK_HALF
    : R_V_BODY + L23 * Math.tan(halfV) + L34 * Math.tan(halfV2);

  if (!isFinite(TOTAL_LENGTH) || TOTAL_LENGTH <= 0 || L23 <= 0 || L34 <= 0) {
    throw new Error(
      `Degenerate geometry: L_throat=${THROAT_LENGTH_MM} L23=${L23} L34=${L34} body=${BODY_LENGTH} total=${TOTAL_LENGTH}`,
    );
  }

  const hAxisParams = classicalAxisParams(
    THETA_W_DEG, THETA_W2_DEG, R_H_BODY, BODY_LENGTH, FLARE2_START_NORM_BODY,
  );
  const vAxisParams = classicalAxisParams(
    THETA_H_DEG, THETA_H2_DEG, R_V_BODY, BODY_LENGTH, FLARE2_START_NORM_BODY,
  );
  const hAxisSections = buildAxisSections({
    hasThroat: HAS_THROAT,
    throatType: THROAT_TYPE,
    throatLengthMm: THROAT_LENGTH_MM,
    throatDriverDeg: THROAT_DRIVER_DEG,
    coverageDeg: THETA_W_DEG,
    flare2Deg: THETA_W2_DEG,
    r0_axis: R_H_BODY,
    bodyLength: BODY_LENGTH,
    L23,
    L34,
    flare2StartNorm: FLARE2_START_NORM_BODY,
  });
  const vAxisSections = buildAxisSections({
    hasThroat: HAS_THROAT,
    throatType: THROAT_TYPE,
    throatLengthMm: THROAT_LENGTH_MM,
    throatDriverDeg: THROAT_DRIVER_DEG,
    coverageDeg: THETA_H_DEG,
    flare2Deg: THETA_H2_DEG,
    r0_axis: R_V_BODY,
    bodyLength: BODY_LENGTH,
    L23,
    L34,
    flare2StartNorm: FLARE2_START_NORM_BODY,
  });

  // Cross-section timeline: lofts circle → rounded-rect across the throat.
  // After the body junction the rounded-rect fillet stays constant — the
  // body reads as a clean rectangular prism with rounded corners.
  const tBodyJunction = HAS_THROAT ? THROAT_LENGTH_MM / TOTAL_LENGTH : 0;
  const tThroatMid = HAS_THROAT ? tBodyJunction * 0.5 : 0.025;
  const filletAtMid = (R0 + BODY_FILLET_MM) / 2;
  const FILLET_TIMELINE = HAS_THROAT
    ? [
        { t: 0,             filletRadius: R0 },
        { t: tThroatMid,    filletRadius: filletAtMid },
        { t: tBodyJunction, filletRadius: BODY_FILLET_MM },
      ]
    : [
        { t: 0,    filletRadius: R0 },
        { t: 0.05, filletRadius: BODY_FILLET_MM * 1.5 },
        { t: 0.15, filletRadius: BODY_FILLET_MM },
      ];

  // Resolve encDepth:
  //   - null / undefined → auto (TOTAL_LENGTH + 30), original behaviour
  //   - 0 → freestanding (no enclosure box)
  //   - > 0 → explicit margin past the deepest point of the horn
  let ENC_DEPTH;
  if (ENC_DEPTH_RAW === null || ENC_DEPTH_RAW === undefined) {
    ENC_DEPTH = TOTAL_LENGTH + 30;
  } else {
    const v = +ENC_DEPTH_RAW;
    ENC_DEPTH = v > 0 ? TOTAL_LENGTH + v : 0;
  }
  const ENC_SPACE = FREESTANDING ? 0 : 25;

  const params = {
    type: 'Classical',
    ...classicalAxisParams(
      THETA_W_DEG, THETA_W2_DEG, R_H_BODY, BODY_LENGTH, FLARE2_START_NORM_BODY,
    ),
    angularSegments: ANGULAR_SEGMENTS,
    lengthSegments: LENGTH_SEGMENTS,
    encDepth: ENC_DEPTH,
    encEdge: ENC_EDGE_MM,
    encEdgeType: 1,                  // rounded baffle edge
    encSpaceL: ENC_SPACE,
    encSpaceR: ENC_SPACE,
    encSpaceT: ENC_SPACE,
    encSpaceB: ENC_SPACE,
    cornerSegments: (ENC_EDGE_MM > 0 && !FREESTANDING) ? 6 : 1,
    wallThickness: WALL_THICKNESS_MM,
    profileSystem: {
      axes: [
        { id: 'h', angleDeg: 0,  family: 'Classical',
          params: hAxisParams,
          ...(hAxisSections ? { sections: hAxisSections } : {}) },
        { id: 'v', angleDeg: 90, family: 'Classical',
          params: vAxisParams,
          ...(vAxisSections ? { sections: vAxisSections } : {}) },
      ],
      crossSection: {
        aspectRatioMode: 'rectFillet',
        filletRadius: BODY_FILLET_MM,
        filletRadiusTimeline: FILLET_TIMELINE,
      },
      angularSharpness: 0,
      interpolation: 'smoothstep',
    },
  };

  const meta = {
    coverage: { H_deg: THETA_W_DEG, V_deg: THETA_H_DEG },
    secondFlare: { H_deg: THETA_W2_DEG, V_deg: THETA_H2_DEG },
    flare2_ratio: RATW,
    mouth_lock: { axis: MOUTH_LOCK_AXIS, dimension_mm: MOUTH_LOCK_MM },
    enc_edge_mm: ENC_EDGE_MM,
    wall_thickness_mm: WALL_THICKNESS_MM,
    mouth_mm: { width: 2 * D34H_HALF, height: 2 * D34V_HALF },
    throat_mm: { diameter: THROAT_DIA_MM, r0: R0 },
    throatAdapter: HAS_THROAT
      ? {
          type: THROAT_TYPE,
          length_mm: THROAT_LENGTH_MM,
          driver_half_angle_deg: THROAT_DRIVER_DEG,
          body_throat_H_mm: R_H_BODY,
          body_throat_V_mm: R_V_BODY,
        }
      : null,
    bodyLengthMm: BODY_LENGTH,
    axialLengthMm: TOTAL_LENGTH,
    L23_mm: L23,
    L34_mm: L34,
    flare2StartNormBody: FLARE2_START_NORM_BODY,
    flare2StartNormTotal: FLARE2_START_NORM_TOTAL,
    bodyFilletRadiusMm: BODY_FILLET_MM,
  };

  const mesh = {
    throat_res_mm: THROAT_RES,
    mouth_res_mm: MOUTH_RES,
    rear_res_mm: REAR_RES,
    enc_front_res_mm: ENC_FRONT_RES,
    enc_back_res_mm: ENC_BACK_RES,
    angularSegments: ANGULAR_SEGMENTS,
    lengthSegments: LENGTH_SEGMENTS,
  };

  return { params, meta, mesh, warnings: [] };
}
