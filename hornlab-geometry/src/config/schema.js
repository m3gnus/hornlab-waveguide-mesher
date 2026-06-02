export const FORMULA_FIELD_ALLOWLIST = Object.freeze({
  "R-OSSE": ["R", "a", "a0", "r0", "k", "m", "b", "r", "q", "tmax"],
  OSSE: ["L", "a", "a0", "r0", "k", "s", "n", "q", "h"],
  Classical: [
    "L", "R", "r0",
    "classicalCoverageAngle", "classicalM", "classicalT",
    "classicalCatA", "classicalX0", "classicalAlpha",
    "classicalAspectRatio", "classicalRoundStart",
    "classicalFlare2Angle", "classicalFlare2Start",
  ],
  LOOKUP: ["lookupPoints"],
  MORPH: [
    "morphWidth",
    "morphHeight",
    "morphCorner",
    "morphRate",
    "morphFixed",
  ],
  GEOMETRY: [
    "throatExtAngle",
    "throatExtLength",
    "slotLength",
    "rot",
    "gcurveDist",
    "gcurveWidth",
    "gcurveAspectRatio",
    "gcurveSeN",
    "gcurveSf",
    "gcurveSfA",
    "gcurveSfB",
    "gcurveSfM1",
    "gcurveSfM2",
    "gcurveSfN1",
    "gcurveSfN2",
    "gcurveSfN3",
    "gcurveRot",
    "circArcTermAngle",
    "circArcRadius",
  ],
});

export const PARAM_SCHEMA = {
  "R-OSSE": {
    scale: {
      type: "range",
      label: "Scale",
      min: 0.1,
      max: 2,
      step: 0.001,
      default: 1.0,
      tooltip:
        "Scaling factor for waveguide geometry only. Values < 1 shrink the waveguide, > 1 enlarge it. Does not affect enclosure dimensions.",
    },
    R: {
      type: "expression",
      label: "Mouth Radius (R)",
      unit: "mm",
      default: 140,
      tooltip:
        "Mouth radius as a function of azimuthal angle p. Can be constant or an expression.",
    },
    a: {
      type: "expression",
      label: "Wall Angle",
      unit: "°",
      default: 25,
      tooltip:
        "Wall off-axis angle (half-angle) at the mouth, as a function of p. Full coverage equals 2·a.",
    },
    a0: {
      type: "expression",
      label: "Throat Wall Angle",
      unit: "°",
      default: 15.5,
      tooltip:
        'Initial throat wall off-axis angle (half-angle) in degrees. Full throat coverage equals 2·a0. Can be constant or an expression such as "15 + 2*sin(p)".',
    },
    r0: {
      type: "expression",
      label: "Throat Radius (r0)",
      unit: "mm",
      default: 12.7,
      tooltip:
        'Initial throat radius. Can be constant or an expression such as "12.7 + sin(p)".',
    },
    k: {
      type: "number",
      label: "Throat Rounding (k)",
      min: 0.1,
      max: 10,
      step: 0.1,
      default: 2.0,
      tooltip: "Controls the throat rounding and smoothness.",
    },
    m: {
      type: "number",
      label: "Apex Shift (m)",
      min: 0,
      max: 1,
      step: 0.01,
      default: 0.85,
      tooltip: "Shifts the apex position along the horn axis.",
    },
    b: {
      type: "expression",
      label: "Bending (b)",
      default: "0.2",
      tooltip: "Controls profile curvature.",
    },
    r: {
      type: "number",
      label: "Apex Radius (r)",
      min: 0.01,
      max: 2,
      step: 0.01,
      default: 0.4,
      tooltip: "Radius of the apex region.",
    },
    q: {
      type: "number",
      label: "Shape Factor (q)",
      min: 0.5,
      max: 10,
      step: 0.1,
      default: 3.4,
      tooltip: "Controls the overall horn shape profile.",
    },
    tmax: {
      type: "number",
      label: "Truncation Limit (tmax)",
      min: 0.5,
      max: 1.5,
      step: 0.01,
      default: 1.0,
      tooltip: "Truncates the horn at a fraction of the computed length.",
    },
  },
  OSSE: {
    scale: {
      type: "range",
      label: "Scale",
      min: 0.1,
      max: 2,
      step: 0.001,
      default: 1.0,
      tooltip:
        "Scaling factor for waveguide geometry only. Values < 1 shrink the waveguide, > 1 enlarge it. Does not affect enclosure dimensions.",
    },
    L: {
      type: "expression",
      label: "Horn Length (L)",
      unit: "mm",
      default: 130,
      tooltip: "Axial horn length. Can be constant or an expression.",
    },
    a: {
      type: "expression",
      label: "Wall Angle",
      unit: "°",
      default: "45 - 5*cos(2*p)^5 - 2*sin(p)^12",
      tooltip: "Wall off-axis angle (half-angle) at the mouth as a function of p. Full coverage equals 2·a.",
    },
    a0: {
      type: "expression",
      label: "Throat Wall Angle",
      unit: "°",
      default: 10,
      tooltip:
        "Initial throat wall off-axis angle (half-angle) in degrees. Full throat coverage equals 2·a0. Can be constant or an expression.",
    },
    r0: {
      type: "expression",
      label: "Throat Radius (r0)",
      unit: "mm",
      default: 12.7,
      tooltip:
        'Initial throat radius. Can be constant or an expression such as "12.7 + sin(p)".',
    },
    k: {
      type: "number",
      label: "Flare Constant (k)",
      min: 0.1,
      max: 15,
      step: 0.1,
      default: 7.0,
      tooltip: "Expansion rate of the horn profile.",
    },
    s: {
      type: "expression",
      label: "Termination Shape (s)",
      default: "0.85 + 0.3*cos(p)^2",
      tooltip: "Shape factor for the termination flare.",
    },
    n: {
      type: "number",
      label: "Termination Curvature (n)",
      min: 1,
      max: 10,
      step: 0.001,
      default: 4,
      tooltip: "Curvature control exponent for the termination.",
    },
    q: {
      type: "number",
      label: "Termination Smoothness (q)",
      min: 0.1,
      max: 2,
      step: 0.001,
      default: 0.991,
      tooltip: "Transition smoothness at the termination.",
    },
    h: {
      type: "number",
      label: "Shape Factor (h)",
      min: 0,
      max: 10,
      step: 0.1,
      default: 0.0,
      tooltip: "Additional shape control parameter.",
    },
  },
  Classical: {
    scale: {
      type: "range",
      label: "Scale",
      min: 0.1,
      max: 2,
      step: 0.001,
      default: 1.0,
      tooltip:
        "Scaling factor for waveguide geometry only. Values < 1 shrink the waveguide, > 1 enlarge it. Does not affect enclosure dimensions.",
    },
    L: {
      type: "expression",
      label: "Horn Length (L)",
      unit: "mm",
      default: 200,
      tooltip: "Axial horn length. Can be constant or an expression.",
      showWhen: { key: "classicalShape", value: [1, 2, 3, 4, 5] },
    },
    R: {
      type: "expression",
      label: "Mouth Radius (R)",
      unit: "mm",
      default: 140,
      tooltip:
        "Mouth radius for tractrix profile. Can be constant or an expression.",
      showWhen: { key: "classicalShape", value: [6] },
    },
    r0: {
      type: "expression",
      label: "Throat Radius (r0)",
      unit: "mm",
      default: 12.7,
      tooltip:
        'Initial throat radius. Can be constant or an expression such as "12.7 + sin(p)".',
    },
    classicalShape: {
      type: "select",
      label: "Horn Shape",
      options: [
        { value: 1, label: "Conical" },
        // Exponential and Catenoidal are special cases of Hyperbolic
        // (T = 1 and T = 0 respectively). They stay in the data model for
        // back-compat but are hidden from the family picker; users reach
        // them via Hyperbolic + preset buttons.  Loading a legacy config
        // still renders them via a Legacy: disabled option in the select.
        { value: 2, label: "Exponential", hidden: true },
        { value: 3, label: "Hyperbolic" },
        { value: 4, label: "Catenoidal", hidden: true },
        { value: 5, label: "Bessel" },
        { value: 6, label: "Tractrix" },
      ],
      default: 1,
      tooltip: "Classical horn profile shape.",
    },
    classicalCrossSection: {
      type: "select",
      label: "Cross Section",
      options: [
        { value: 0, label: "Circular" },
        { value: 1, label: "Square" },
      ],
      default: 0,
      tooltip: "Circular produces a round horn. Square transitions from circular throat to rectangular mouth.",
    },
    classicalCoverageAngle: {
      type: "expression",
      label: "Coverage Angle",
      unit: "deg",
      default: 20,
      tooltip: "Full included coverage angle of the conical horn in degrees (wall half-angle = coverage/2).",
      showWhen: { key: "classicalShape", value: [1] },
    },
    classicalM: {
      type: "expression",
      label: "Flare Constant (m)",
      default: 0.02,
      tooltip:
        "Flare constant for exponential and hyperbolic horns. Can be derived from cutoff frequency: m = 4*pi*fc/c.",
      showWhen: { key: "classicalShape", value: [2, 3] },
    },
    classicalT: {
      type: "expression",
      label: "Family Parameter (T)",
      default: 1,
      tooltip:
        "Hyperbolic horn family parameter (0 to 1). T=0 gives a catenoidal-like horn, T=1 gives an exponential-like horn. Uses area-law convention (radius = sqrt of area expansion).",
      showWhen: { key: "classicalShape", value: [3] },
    },
    classicalCatA: {
      type: "expression",
      label: "Catenary Scale (a)",
      unit: "mm",
      default: 50,
      tooltip:
        "Scale parameter for the catenoidal horn. Larger values produce a gentler flare.",
      showWhen: { key: "classicalShape", value: [4] },
    },
    classicalX0: {
      type: "expression",
      label: "Reference Length (x0)",
      unit: "mm",
      default: 50,
      tooltip: "Reference length for the Bessel horn profile.",
      showWhen: { key: "classicalShape", value: [5] },
    },
    classicalAlpha: {
      type: "expression",
      label: "Flare Exponent (alpha)",
      default: 1,
      tooltip:
        "Flare exponent for the Bessel horn. alpha=1 gives conical, alpha>1 gives accelerating flare.",
      showWhen: { key: "classicalShape", value: [5] },
    },
    // --- 2nd Flare (conical only) ---
    classicalFlare2Enabled: {
      type: "select",
      label: "2nd Flare",
      options: [
        { value: 0, label: "Off" },
        { value: 1, label: "On" },
      ],
      default: 0,
      tooltip: "Enable a secondary flare section near the mouth with a wider angle.",
      showWhen: { key: "classicalShape", value: [1] },
    },
    classicalFlare2Angle: {
      type: "expression",
      label: "2nd Flare Angle",
      unit: "deg",
      default: 20,
      tooltip: "Full coverage angle for the secondary flare section in degrees. Should be wider than the primary coverage angle.",
      showWhen: [{ key: "classicalShape", value: [1] }, { key: "classicalFlare2Enabled", value: [1] }],
    },
    classicalFlare2Start: {
      type: "expression",
      label: "2nd Flare Start",
      default: 0.5,
      tooltip: "Position where the 2nd flare begins, as a fraction of horn length (0 = throat, 1 = mouth).",
      showWhen: [{ key: "classicalShape", value: [1] }, { key: "classicalFlare2Enabled", value: [1] }],
    },
    classicalFlare2Blend: {
      type: "select",
      label: "2nd Flare Blend",
      options: [
        { value: 0, label: "Sharp" },
        { value: 1, label: "Smooth" },
      ],
      default: 0,
      tooltip: "Sharp creates a compound cone. Smooth uses a tangent-continuous blend between angles.",
      showWhen: [{ key: "classicalShape", value: [1] }, { key: "classicalFlare2Enabled", value: [1] }],
    },
    // --- Square cross-section shape controls ---
    classicalAspectRatio: {
      type: "expression",
      label: "Aspect Ratio (W:H)",
      default: 1,
      tooltip: "Width-to-height ratio at the mouth. 1 = square, >1 = wider than tall. The area expansion follows the horn law; this controls only the shape.",
      showWhen: { key: "classicalCrossSection", value: [1] },
    },
    classicalSqExponent: {
      type: "number",
      label: "Rectangularity",
      min: 2,
      max: 20,
      step: 0.1,
      default: 6,
      tooltip: "Superellipse exponent at the mouth. 2 = circle/ellipse, 4-8 = rounded rectangle, 20 = nearly sharp rectangle.",
      showWhen: { key: "classicalCrossSection", value: [1] },
    },
    classicalRoundStart: {
      type: "number",
      label: "Shape Transition Start",
      min: 0,
      max: 1,
      step: 0.01,
      default: 0,
      tooltip: "Fraction of horn length where the circular-to-rectangular transition begins. 0 = transitions from the throat, 1 = stays circular until the mouth.",
      showWhen: { key: "classicalCrossSection", value: [1] },
    },
  },
  GEOMETRY: {
    throatProfile: {
      type: "select",
      label: "Throat Profile",
      options: [
        { value: 1, label: "OS-SE (Profile 1)" },
        { value: 3, label: "Circular Arc (Profile 3)" },
      ],
      default: 1,
      tooltip: "Profile type: OS-SE or circular arc.",
    },
    throatExtAngle: {
      type: "expression",
      label: "Throat Extension Angle",
      unit: "deg",
      default: "0",
      tooltip: "Half-angle of the optional conical throat extension.",
    },
    throatExtLength: {
      type: "expression",
      label: "Throat Extension Length",
      unit: "mm",
      default: "0",
      tooltip: "Axial length of the optional conical throat extension.",
    },
    slotLength: {
      type: "expression",
      label: "Straight Slot Length",
      unit: "mm",
      default: "0",
      tooltip: "Axial length of an initial straight waveguide segment.",
    },
    rot: {
      type: "expression",
      label: "Profile Rotation",
      unit: "deg",
      default: "0",
      tooltip: "Rotate the computed profile around point [0, r0].",
    },
    gcurveType: {
      type: "select",
      label: "Guiding Curve Mode",
      options: [
        { value: 0, label: "Explicit Coverage" },
        { value: 1, label: "Superellipse" },
        { value: 2, label: "Superformula" },
      ],
      default: 0,
      tooltip: "Use guiding curve to infer coverage angle.",
    },
    gcurveDist: {
      type: "expression",
      label: "Guiding Curve Distance",
      default: "0.5",
      tooltip:
        "Guiding-curve distance from the throat, expressed as a fraction or in millimetres.",
      showWhen: { key: "gcurveType", value: [1, 2] },
    },
    gcurveWidth: {
      type: "expression",
      label: "Guiding Curve Width",
      unit: "mm",
      default: "0",
      tooltip: "Guiding-curve width along X.",
      showWhen: { key: "gcurveType", value: [1, 2] },
    },
    gcurveAspectRatio: {
      type: "expression",
      label: "Guiding Curve Aspect Ratio",
      default: "1",
      tooltip: "Height-to-width ratio for the guiding curve.",
      showWhen: { key: "gcurveType", value: [1, 2] },
    },
    gcurveSeN: {
      type: "expression",
      label: "Guiding Superellipse Exponent",
      default: "3",
      tooltip:
        "Exponent used when the guiding curve runs in superellipse mode.",
      showWhen: { key: "gcurveType", value: [1] },
    },
    gcurveSf: {
      type: "expression",
      label: "Superformula Tuple",
      default: "",
      tooltip:
        "Comma-separated superformula parameters in the order a,b,m,n1,n2,n3.",
      showWhen: { key: "gcurveType", value: [2] },
    },
    gcurveSfA: {
      type: "expression",
      label: "Superformula a",
      default: "",
      tooltip: "Superformula a parameter.",
      showWhen: { key: "gcurveType", value: [2] },
    },
    gcurveSfB: {
      type: "expression",
      label: "Superformula b",
      default: "",
      tooltip: "Superformula b parameter.",
      showWhen: { key: "gcurveType", value: [2] },
    },
    gcurveSfM1: {
      type: "expression",
      label: "Superformula m1",
      default: "",
      tooltip: "Superformula m1 parameter.",
      showWhen: { key: "gcurveType", value: [2] },
    },
    gcurveSfM2: {
      type: "expression",
      label: "Superformula m2",
      default: "",
      tooltip: "Superformula m2 parameter.",
      showWhen: { key: "gcurveType", value: [2] },
    },
    gcurveSfN1: {
      type: "expression",
      label: "Superformula n1",
      default: "",
      tooltip: "Superformula n1 parameter.",
      showWhen: { key: "gcurveType", value: [2] },
    },
    gcurveSfN2: {
      type: "expression",
      label: "Superformula n2",
      default: "",
      tooltip: "Superformula n2 parameter.",
      showWhen: { key: "gcurveType", value: [2] },
    },
    gcurveSfN3: {
      type: "expression",
      label: "Superformula n3",
      default: "",
      tooltip: "Superformula n3 parameter.",
      showWhen: { key: "gcurveType", value: [2] },
    },
    gcurveRot: {
      type: "expression",
      label: "Guiding Curve Rotation",
      unit: "deg",
      default: "0",
      tooltip: "Rotate the guiding curve anticlockwise.",
      showWhen: { key: "gcurveType", value: [1, 2] },
    },
    circArcTermAngle: {
      type: "expression",
      label: "Circular Arc Terminal Angle",
      unit: "deg",
      default: "1",
      tooltip: "Mouth terminal angle for the circular-arc throat profile.",
    },
    circArcRadius: {
      type: "expression",
      label: "Circular Arc Radius Override",
      unit: "mm",
      default: "0",
      tooltip: "Explicit radius override for the circular-arc throat profile.",
    },
  },
  MORPH: {
    morphTarget: {
      type: "select",
      label: "Target Shape",
      options: [
        { value: 0, label: "None" },
        { value: 1, label: "Rectangle" },
        { value: 2, label: "Circle" },
      ],
      default: 0,
      tooltip:
        "Post-profile shaping that blends the mouth toward a target silhouette. None = no morph (use the Cross Section & Shape controls above instead). Rectangle morphs to a width×height box with optional corner radius. Circle re-rounds an otherwise squared mouth.",
    },
    morphWidth: {
      type: "number",
      label: "Target Width",
      unit: "mm",
      default: 0,
      tooltip:
        "Half-width of the morph target at the mouth. Only used when Target Shape = Rectangle. Typical starting value: the outer radius the horn would otherwise reach (≥ mouth radius).",
    },
    morphHeight: {
      type: "number",
      label: "Target Height",
      unit: "mm",
      default: 0,
      tooltip:
        "Half-height of the morph target at the mouth. Only used when Target Shape = Rectangle. Combine with Target Width to set the mouth aspect ratio.",
    },
    morphCorner: {
      type: "range",
      label: "Corner Radius",
      unit: "mm",
      min: 0,
      max: 100,
      step: 1,
      default: 0,
      tooltip:
        "Corner fillet radius on the rectangular morph target. 0 = sharp corners; larger values round the rim. Range 0–100 mm; start around 10–20 mm.",
    },
    morphRate: {
      type: "number",
      label: "Morph Rate",
      step: 0.1,
      default: 3.0,
      tooltip:
        "How quickly the profile blends from circular at the throat to the morph target at the mouth. Higher values concentrate the morph near the mouth; lower values spread it over more of the length. Typical range 1–6; default 3.",
    },
    morphFixed: {
      type: "range",
      label: "Unmorphed Throat Fraction",
      min: 0,
      max: 1,
      step: 0.01,
      default: 0.0,
      tooltip:
        "Fraction of horn length (from the throat) that stays unmorphed. 0 = morph begins at the throat; 0.5 = morph is confined to the mouth half.",
    },
    morphAllowShrinkage: {
      type: "select",
      label: "Allow Inward Morph",
      options: [
        { value: 0, label: "No" },
        { value: 1, label: "Yes" },
      ],
      default: 0,
      tooltip:
        "When No (default), the morph target can only push the wall outward — no inward dips. Yes allows the target to pull the wall inward where the target silhouette is smaller than the natural profile; use only if you deliberately want a pinched rim.",
    },
  },
  MESH: {
    angularSegments: {
      type: "number",
      label: "Preview Angular Segments",
      default: 40,
      tooltip:
        "Three.js viewport tessellation around the horn circumference. Does not change backend OCC solve/export mesh element sizes.",
    },
    lengthSegments: {
      type: "number",
      label: "Preview Length Segments",
      default: 20,
      tooltip:
        "Three.js viewport tessellation along the horn length. Does not change backend OCC solve/export mesh element sizes.",
    },
    cornerSegments: {
      type: "number",
      label: "Preview Corner Segments",
      default: 4,
      tooltip:
        "Three.js viewport tessellation for rounded corners and morph edges only.",
    },
    throatSegments: {
      type: "number",
      label: "Preview Throat Segments",
      default: 0,
      tooltip:
        "Extra Three.js viewport tessellation near the throat. Does not change backend OCC solve/export mesh element sizes.",
    },
    throatResolution: {
      type: "number",
      label: "Throat Mesh Resolution",
      unit: "mm",
      default: 4.0,
      tooltip:
        "Backend OCC solve/export mesh element size near the throat. Also influences viewport slice spacing unless Throat Slice Density overrides it. Controlled by Mesh Quality preset in Settings.",
    },
    mouthResolution: {
      type: "number",
      label: "Mouth Mesh Resolution",
      unit: "mm",
      default: 16.0,
      tooltip:
        "Backend OCC solve/export mesh element size near the mouth. Also influences viewport slice spacing unless Throat Slice Density overrides it. Controlled by Mesh Quality preset in Settings.",
    },
    throatSliceDensity: {
      type: "number",
      label: "Preview Slice Bias",
      default: null,
      tooltip:
        "Viewport slice clustering (0.5 = uniform, lower = tighter near the throat). When set, it overrides the throat-to-mouth resolution ratio for viewport slice distribution only.",
    },
    verticalOffset: {
      type: "number",
      label: "Export Vertical Offset",
      unit: "mm",
      default: 0.0,
      tooltip:
        "Vertical offset for the simulation and export coordinate system. Does not affect the 3D viewer.",
    },
    wallThickness: {
      type: "number",
      label: "Wall Thickness",
      unit: "mm",
      // Canonical default — kept in sync with `DEFAULT_WALL_THICKNESS` in
      // `src/modules/design/index.js`.  Previously `0`, which silently
      // diverged from the simulation/export default (`5`/`6` depending on
      // path); the same state then produced two different wall thicknesses
      // on parallel code paths.  See backlog T2.6.
      default: 6,
      tooltip:
        "Applies only to freestanding horns (Enclosure Depth = 0). Builds a normal-offset wall shell one wall-thickness from the horn surface and a rear disc behind the throat.",
    },
    rearResolution: {
      type: "number",
      label: "Rear Mesh Resolution",
      unit: "mm",
      default: 24.0,
      tooltip:
        "Backend OCC solve/export mesh element size for the rear wall on freestanding thickened horns. Controlled by Mesh Quality preset in Settings.",
    },
    quadrants: {
      type: "select",
      label: "Quadrants",
      default: "1234",
      // hidden: active OCC solve/export always builds full-domain meshes (1234).
      // This field is kept for import/export round-trip compatibility only and
      // must not be shown in the active parameter panel.
      hidden: true,
      tooltip:
        "Import-compatibility only. Active OCC solve/export always uses full-domain (1234). Legacy values are accepted but not applied.",
      options: [
        { value: "1234", label: "Full (1234)" },
        { value: "12", label: "Half Y≥0 (12)" },
        { value: "14", label: "Half X≥0 (14)" },
        { value: "1", label: "Quarter Q1 (1)" },
      ],
    },
  },
  ENCLOSURE: {
    encDepth: {
      type: "number",
      label: "Enclosure Depth",
      unit: "mm",
      default: 0,
      tooltip:
        "Axial depth of the enclosure box behind the horn mouth. 0 = freestanding horn (no enclosure). Positive values add a front/back baffle pair with the horn embedded; also enables the enclosure margin and edge controls.",
    },
    encDepthMargin: {
      type: "number",
      label: "Back-Wall Margin",
      unit: "mm",
      default: 1,
      step: 0.1,
      min: 0,
      showWhen: { key: "encDepth", value: { gt: 0 } },
      tooltip:
        "Minimum clearance from the deepest horn point to the cabinet back wall. Defaults to 1 mm (legacy minimum to avoid intersection). When you want the cabinet depth to track the horn length on every candidate, set Enclosure Depth to a low positive sentinel and raise this margin (typical 20 mm).",
    },
    vScaleTarget: {
      type: "number",
      label: "V-Mouth Target",
      unit: "mm",
      default: 0,
      step: 1,
      min: 0,
      tooltip:
        "0 = off. >0: post-process all Y coordinates so the V-mouth lands at exactly this full diameter (mm). Decouples mouth height from per-axis OSSE coverage angles. Y-only — X and Z are untouched. Z-tapered: equals 1.0 at the throat, ramps to the target at the mouth (or starts at the cross-section transition_start when a cross-section reshape is active).",
    },
    encPlanType: {
      type: "select",
      label: "Ground Plan",
      options: [
        { value: 1, label: "Rounded Rectangle" },
        { value: 2, label: "Ellipse" },
        { value: 3, label: "Superellipse" },
      ],
      default: 1,
      tooltip:
        "Enclosure ground-plan shape. Rounded Rectangle is the classic box with corner radii. Ellipse and Superellipse use the same curve math as guiding curves.",
    },
    encPlanN: {
      type: "number",
      label: "Superellipse Exponent",
      default: 2.0,
      step: 0.1,
      min: 0.5,
      max: 20,
      showWhen: { key: "encPlanType", value: 3 },
      tooltip:
        "Exponent n for the superellipse |x/a|^n + |y/b|^n = 1. n=2 is an ellipse, n>2 approaches a rectangle, n<2 pinches inward.",
    },
    encEdge: {
      type: "number",
      label: "Edge Radius",
      unit: "mm",
      default: 18,
      tooltip:
        "Corner / edge radius on the enclosure front baffle (Rounded Rectangle only). Also used as the chamfer size when Edge Finish = Chamfered. Typical 10–30 mm.",
    },
    encEdgeType: {
      type: "select",
      label: "Edge Finish",
      options: [
        { value: 1, label: "Rounded" },
        { value: 2, label: "Chamfered" },
      ],
      default: 1,
      tooltip:
        "Rounded = smooth radius (softens diffraction). Chamfered = 45° bevel at Edge Radius. Rounded is the usual choice; Chamfered is easier to fabricate from flat stock.",
    },
    encSpaceL: {
      type: "number",
      label: "Left Margin",
      unit: "mm",
      default: 25,
      tooltip: "Horizontal space from the horn mouth to the left enclosure edge.",
    },
    encSpaceT: {
      type: "number",
      label: "Top Margin",
      unit: "mm",
      default: 25,
      tooltip: "Vertical space from the horn mouth to the top enclosure edge.",
    },
    encSpaceR: {
      type: "number",
      label: "Right Margin",
      unit: "mm",
      default: 25,
      tooltip: "Horizontal space from the horn mouth to the right enclosure edge.",
    },
    encSpaceB: {
      type: "number",
      label: "Bottom Margin",
      unit: "mm",
      default: 25,
      tooltip: "Vertical space from the horn mouth to the bottom enclosure edge.",
    },
    encFrontResolution: {
      type: "expression",
      label: "Front Baffle Mesh Resolution",
      unit: "mm",
      default: "25,25,25,25",
      tooltip:
        "Backend OCC solve/export mesh element sizes for enclosure front-baffle quadrants (q1..q4).",
    },
    encBackResolution: {
      type: "expression",
      label: "Rear Baffle Mesh Resolution",
      unit: "mm",
      default: "40,40,40,40",
      tooltip:
        "Backend OCC solve/export mesh element sizes for enclosure back-baffle quadrants (q1..q4).",
    },
  },
  SOURCE: {
    sourceShape: {
      type: "select",
      label: "Source Surface",
      options: [
        { value: 1, label: "Spherical Cap" },
        { value: 2, label: "Flat Disc" },
      ],
      default: 1,
      tooltip:
        "Shape of the radiating boundary at the throat used by the BEM solver. Spherical Cap matches a compression-driver diaphragm; Flat Disc matches a planar piston. Choose based on the driver being modelled.",
    },
    sourceRadius: {
      type: "number",
      label: "Source Radius",
      unit: "mm",
      default: -1,
      tooltip:
        "Radius of the radiating surface at the throat. -1 auto-selects the throat radius (r0). Override with a positive value only when the driver exit differs from the horn throat.",
    },
    sourceCurv: {
      type: "select",
      label: "Source Curvature",
      options: [
        { value: 0, label: "Auto" },
        { value: 1, label: "Convex" },
        { value: -1, label: "Concave" },
      ],
      default: 0,
      tooltip:
        "Curvature sign of the spherical cap. Auto (default) infers direction from the throat geometry. Convex = dome bulges into the horn; Concave = dome bulges back into the driver chamber.",
    },
    sourceVelocityProfile: {
      type: "select",
      label: "Velocity Profile",
      options: [
        { value: "piston", label: "Piston (Uniform)" },
        { value: "dome", label: "Dome (Cosine)" },
        { value: "ring", label: "Ring (r > 0.7)" },
      ],
      default: "piston",
      tooltip: "Radial velocity weighting applied to the throat source elements.",
    },
  },
  SIMULATION: {
    frequencyStart: {
      type: "number",
      label: "Sweep Start",
      unit: "Hz",
      default: 400,
      min: 20,
      max: 20000,
      step: 10,
      controlId: "freq-start",
      tooltip: "Lowest frequency in the backend BEM sweep.",
    },
    frequencyEnd: {
      type: "number",
      label: "Sweep End",
      unit: "Hz",
      default: 16000,
      min: 20,
      max: 20000,
      step: 10,
      controlId: "freq-end",
      tooltip: "Highest frequency in the backend BEM sweep.",
    },
    numFrequencies: {
      type: "number",
      label: "Frequency Samples",
      default: 32,
      min: 10,
      max: 200,
      step: 1,
      controlId: "freq-steps",
      tooltip: "Number of solved frequencies between the start and end values.",
    },
  },
  // Output actions are handled via export buttons in the UI.
};

for (const [group, keys] of Object.entries(FORMULA_FIELD_ALLOWLIST)) {
  const schemaGroup = PARAM_SCHEMA[group];
  if (!schemaGroup) continue;
  for (const key of keys) {
    if (schemaGroup[key]) {
      schemaGroup[key].supportsFormula = true;
    }
  }
}
