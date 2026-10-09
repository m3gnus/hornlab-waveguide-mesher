# Authored adapter with absolute axial scaling

`OSSE-ADAPTER-AXIAL` composes a native circular cubic throat adapter and its
retained OSSE body with independent transverse and axial scales:

```json
{
  "formula": "OSSE-ADAPTER-AXIAL",
  "mode": "bare",
  "scale": 2,
  "axial_scale": 1,
  "vertical_offset_mm": 7,
  "profile": {"L_mm": 120, "r0_mm": 12.7, "a_deg": 40, "a0_deg": 15.5, "k": 1},
  "throat_adapter": {
    "mode": "authored", "contract_revision": 1,
    "driver_exit_diameter_mm": 25.4, "exit_half_angle_deg": 10,
    "length_mm": 40, "join_t": 0.137,
    "driver_handle_mm": 10, "body_handle_mm": 15
  },
  "source": {"source_shape": 0},
  "mesh": {"throat_res_mm": 5, "mouth_res_mm": 15, "rear_res_mm": 10}
}
```

All authored distances and handle lengths are intrinsic millimetres. The
finished construction is `(scale*x, scale*y + vertical_offset_mm,
axial_scale*z)`: the axial factor replaces the global scale on Z. The example
has a 50.8 mm driver diameter, 143.56 mm assembled depth, a join at Z=40 mm,
and its source center at `(0,7,0)` mm. Its actual driver tangent angle is
19.4254 degrees. Generally that angle is
`atan2(scale*sin(authored_angle), axial_scale*cos(authored_angle))`.

Exact cubic and rational-conic poles undergo the same affine transformation;
rational weights are preserved. Mesh and STEP share this authority, including
the source rim and adapter join. The flat source points in +Z. The source,
join and mouth datums retain their exact physical planes. Preview dimensions
include interior cubic radius extrema, so the body envelope can exceed the
mouth diameter. Mesh resolutions remain finished-mm requests; output unit
conversion happens after construction and placement.

The formula requires root `axial_scale` and an active root `throat_adapter`.
Supplied formula/type selectors must agree. Older readers refuse the new
formula. Selecting an older formula with a supplied axial factor also refuses,
even when the factors are equal. The six adapter controls retain their
[authored meanings](native-throat-adapter.md).

This initial domain supports full-circle bare acoustic models with zero wall,
flat native source, and exact interpolating construction. Both scales are in
`[0.01,10]`, their axial/transverse ratio is in `[0.1,10]`, and finished Y is
within 10000 mm of the origin. Assembled depth is in `[1,2000]` mm, driver radius
in `[1,200]` mm, `k` in `[0.5,10]`, and authored body angles satisfy
`5 <= a <= 75` and `0 <= a0 < a` degrees. Safe negative and zero authored driver
angles remain supported. The existing intrinsic adapter validity checks apply,
with additional physical control clearances greater than 0.0001 mm, radial
control bounds at most 10000 mm and a conservative coordinate envelope at most
100000 mm. Unresolved short spans, degeneracy and endpoint precision refuse.

Use canonical profile keys `L_mm`, `r0_mm`, `a_deg`, `a0_deg`, `k`, and optional
zero `s`. Root/profile duplicates, hidden aliases and unsupported supplied
controls refuse before rewriting. Enclosures, baffles, wall shells, roundovers,
termination, morph, guiding curves, prefixes, noncircular profiles and curved
sources are outside this domain. JSON and TOML support the same controls.

The complete resolve, preview, mesh and STEP APIs support this construction.
Body-only profile/grid/viewport helpers and linked bundles refuse because they
cannot transport this complete analytic authority. STEP is a surface body;
it retains a neutral construction recipe, both scales, placement and a stable
fingerprint. STEP does not transport acoustic physical tags. Shape identity
excludes density, sampling hints, output paths and preview LOD.

Preview uses exact physical derivatives and signed principal curvatures.
The adapter/body join is G1: its curvature can jump. The shared join row reports
the retained-body one-sided curvature, identified in surface metadata.
Continuous chord bounds combine a meridian second-derivative bound, circular
angular sagitta and absolute-position binary32 rounding. They bound geometric
surface distance, not errors at identical parameter coordinates. Continuous
normal bounds combine cubic derivative-control slope ranges or monotone conic
tangent ranges with the azimuth range. Requests exceeding sampling, precision
or vertex limits refuse. The default chord target is 0.008 mm and the default
normal target is 3 degrees for all three LODs; custom tolerances remain effective.
