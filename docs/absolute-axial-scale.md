# Absolute axial scale

Native `OSSE-AXIAL` models use root `scale` for X and Y and root `axial_scale`
for Z. The axial factor is absolute: `scale: 2, axial_scale: 1` doubles the
opening diameters while retaining the authored depth. Finished
`mesh.vertical_offset_mm` is applied afterwards as a rigid Y translation.

```json
{
  "formula": "OSSE-AXIAL",
  "scale": 2,
  "axial_scale": 1.2,
  "profile": {
    "L_mm": 120, "r0_mm": 18,
    "a_deg": 40, "a0_deg": 8, "k": 1, "s": 0
  },
  "mesh": {
    "wall_thickness_mm": 0,
    "vertical_offset_mm": 7,
    "throat_res_mm": 4, "mouth_res_mm": 12
  },
  "source": {"source_shape": 0}
}
```

This example has a 144 mm depth and a 36 mm throat radius. The source tangent
angle becomes `atan((scale / axial_scale) * tan(a0))`; the authored angle is
not retained after anisotropic scaling.

The initial scope is a full circular, scalar, unterminated (`s: 0`) OSSE body,
bare with zero wall thickness and a flat native source. Wall shells, prefixes,
adapters, lips, enclosures, baffles, symmetry reduction, noncircular profiles,
expressions, imported controls and approximate surface fits are refused.
Omitting `mode`, wall thickness or source shape selects `bare`, zero and flat
respectively for this formula. A supplied `mode` must be exactly `bare`.

Use the formula marker together with `axial_scale`. Older readers reject the
unknown formula rather than silently discarding the axis control. Supplying
`axial_scale` with an ordinary formula is an error even when its value equals
`scale`. Ordinary configurations without the axis control retain the existing
construction. Formula/type aliases must all agree; supplied coverage must be
exactly `1234`. Root `vertical_offset_mm` is also accepted, but duplicate root
and mesh placement controls must agree. The CLI routing controls `output.path`,
root `path` and root `output_path` remain supported; they do not affect the
geometry or its fingerprint. The output object also accepts `output_path`,
using the existing CLI precedence. Path values must be strings or null.
Other aliases and unknown geometry controls are refused in the native scope.

The qualified numerical domain is:

- `scale` and `axial_scale` each in `[0.01, 10]`, with their axial-to-uniform
  ratio in `[0.1, 10]`.
- Physical depth `[1, 2000]` mm, physical throat radius `[1, 200]` mm,
  `k` in `[0.5, 10]`, and physical mouth radius at most 10000 mm.
- Authored angles `5 <= a <= 75` and `0 <= a0 < a`, in degrees.
- Finished Y placement within ±10000 mm.

Numeric controls require finite scalar numbers. Supplied segment counts and
`max_triangles` require positive integer values at most 10000000. Mesh
resolutions must lie in `[0.01, 10000]` mm. Unit conversion and large-mesh
opt-in require actual booleans; an API override accepts a boolean or `None`.
Unsupported requests fail before terminal geometry construction.
Angular and length sampling counts are accepted as validated compatibility
hints; certified analytic stations determine this model's shape. Use the
physical mesh resolutions to change terminal tessellation.

Preview, mesh and STEP share one analytic physical meridian. The OCC body uses
cubic Hermite spans with exact endpoint positions and tangents; a bound on the
fourth derivative certifies the entire span within 0.0001 mm of that meridian.
This is a CAD surface fitting bound. Mesh triangle chord error remains governed
by the requested physical mesh resolution and is separate from the CAD bound.
The ordinary 18000-triangle budget still applies; dense output requires
`allow_large_mesh: true` or an explicitly sufficient budget.
Fine physical resolutions can produce hundreds of thousands of triangles
and take several minutes; large-mesh opt-in does not impose a runtime limit.

Preview defaults use physical chord targets 0.15 mm for `coarse`, 0.05 mm for
`fine`, and 0.025 mm for `inspection`. Explicit chord and normal-step requests
are certified over the meridian and angular intervals, including a conservative
binary32 placement allowance. The preview refuses a request that cannot fit
its station, angular or vertex budget. These physical tolerances apply after
the anisotropic transform. Sampling counts and mesh density do not change the
analytic construction fingerprint.
When curvature is requested, mean and principal curvature follow the shipped
inward wall normals. Principal curvature retains the sign of the principal
value with the largest magnitude; equal magnitudes choose the positive value.

Full continuous bounds, transformed source angle and fingerprint are returned
in `absoluteAxialScale` metadata. STEP persists the canonical controls and
identity in a `hornlab-absolute-axial-scale` comment and returns corresponding
CAD metadata. STEP is a zero-thickness surface body in this initial bare scope.
Authoritative throat and mouth datums remain at Z=0 and the transformed depth;
Y placement shifts both together. Body-only grid/formula helpers and `.wglink`
transport refuse this model until their contracts can carry the analytic
construction. Use `resolve_geometry` and the complete preview API.
