# Exact native adapter controls

`OSSE-ADAPTER-CONTROLS` constructs a literal cubic adapter followed by the
complete circular OSSE body. Each supplied point remains independent; a
tangent mismatch at the join is a deliberate corner.

```toml
formula = "OSSE-ADAPTER-CONTROLS"
mode = "bare"
scale = 0.25

[profile]
L_mm = 137
r0_mm = 17
a_deg = 37
a0_deg = 6
k = 1.25
s = 0

[throat_adapter]
mode = "controls"
contract_revision = 1
join_t = 0
control_points_mm = [[-31.7, 16.2], [-22.1, 16.9], [-9.3, 20.4]]

[source]
source_shape = 0

[mesh]
wall_thickness_mm = 0
throat_res_mm = 1
mouth_res_mm = 1.5
rear_res_mm = 2
max_triangles = 250000
allow_large_mesh = true
```

The root `throat_adapter` object requires exactly `mode`, `contract_revision`,
`join_t`, and `control_points_mm`. Revision is integer 1, join is numeric zero,
and the controls are exactly three finite numeric pairs. The new formula
marker lets older readers refuse unsupported designs. Existing `authored`
adapters retain their own controls and smooth-join behavior.

Pairs are absolute join-local `(Z, radius)` positions before uniform scale:
P0=(c1,v1), P1=(c2,v2), P2=(c3,v3). The body supplies P3=(0,r0). Set A=-c1;
the physical driver-frame cubic poles are `scale*(Pi+(A,0))`. Its source is
at Z=0 with radius `scale*v1`, its body join at `scale*A`, and its mouth at
`scale*(A+L)`. The body length L is preserved in full. Finished Y translation
acts after scale. The signed driver exit angle is `atan2(v2-v1,c2-c1)`;
it does not curve the flat source disk or replace any control.

The signed join tangent jump is body tangent minus cubic tangent in forward
source-to-mouth order. Preview duplicates only the join rendering vertices,
with exactly coincident positions and separate one-sided normals and signed
principal curvatures. Branch-local triangles never cross the corner. Mesh
and STEP retain one shared exact join edge.

The first domain is scalar circular OSSE with zero termination, bare mode,
zero wall, full-circle coverage and a flat native source. Canonical profile
keys are `L_mm`, `r0_mm`, `a_deg`, `a0_deg`, `k`, and optional zero `s`.
Canonical source accepts only zero `source_shape`. Mesh accepts the three
resolutions, positive integer sampling counts and triangle budget, actual
boolean `allow_large_mesh`/`scale_to_metres`, zero wall, acoustic topology,
auto/interpolate surface fit, full-circle `quadrants` and finished
`vertical_offset_mm`. Root supports formula/type, profile, source, mesh,
adapter, bare mode, scale, quadrants, finished placement, and output routing
via path/output_path. Duplicate formula/type and placement values must agree.
Unknown fields and supplied aliases outside these spellings refuse.

Body L and r0 are positive and at most 10000 mm, k positive and at most 100,
`0 <= a0_deg < a_deg < 80`, `a_deg >= 1`, and angular separation at least
0.000001 degree. Uniform scale is in [0.001,100], controls have absolute
magnitude at most 1000000 mm, and the complete physical envelope including
Y is bounded by 100000 mm. Physical adapter/body spans, full cubic radius
and full cubic Z derivative must exceed 0.0001 mm. Coupled precision guards
can refuse unresolved inputs within these outer limits.

Validity uses bounded dyadic Bernstein subdivision with Decimal arithmetic
and outward margins over the entire polynomial. It admits unordered axial
poles and nonpositive interior radial poles when the actual curve is strictly
forward and positive. It never projects P2 onto the body tangent, clips a
radius, repairs a handle, or substitutes sampled minima for a certificate.
Positive Z derivative separates cubic and body interiors and makes each
revolved branch regular. The full body is an exact rational quadratic conic
with positive weights and a continuous implicit-residual certificate.

Dimensions use the exact cubic radial extrema and analytic body mouth,
independently of density and preview LOD. Driver, join and mouth planes are
distinct datums. Overall diameter can exceed the mouth diameter when the
cubic has an interior radial maximum. Construction identity includes all six
full-precision controls, body parameters, scale and finished Y; signed zero
normalizes to zero. Density, LOD, routing and output units do not alter it.
Immutable byte-backed pole arrays and detached recipe data own geometry;
the compatibility grid provides no fitting or normal authority.

All preview LODs default to 0.008 mm chord and 3 degrees normal step within
each smooth branch. Whole-interval chart Hessian bounds, meridian-angle bounds
and binary32 placement rounding certify the result; the reported join crease
is excluded from smooth normal continuity. Explicit requests retain their
budgets or refuse at the bounded station/angular work and one-million vertex
ceiling. Source normal is +Z and its signed curvature is zero.

Mesh revolves the exact supplied cubic and conic, preserving quadrant,
source and join ancestry. Physical groups are rigid wall 1 and source 2;
only the true mouth loop is open. A minimum sixteen source-circle intervals
preserves small-source disk area at coarse requests; finer throat density
remains effective. Approximate fitting, retry fitting and distance welding
are disabled on this route. Local derivative sizing uses a fixed separating
projection of the outward derivative hull, including steep radial motion.
This also bounds finite meridian chord displacement within each interval.
Sizing remains a heuristic and its
triangle cost is an estimate. Existing user budgets and explicit large-mesh
opt-in remain effective.

Before publication, every actual cubic and conic wall facet must satisfy a
continuous finite-branch chart bound of at most 0.01 mm. The bound includes
full vertex-to-projection residual, finite-domain inversion and outward
numeric margins. It bounds facets to their intended smooth surface in one
direction; it does not alone prove reverse coverage or continuity across the
crease. Postprocessed wall normals must also face their exact branch bore.
Refusals preserve existing output files.

STEP exports one sewn shell-based surface representation with shared edge
identities, eight wall faces and an optional ninth source face. It contains
no volume or solid. Default `open_throat=true` removes the source disk;
actual boolean false retains it. CAD coordinates remain millimetres, as on
the existing STEP API; mesh conversion to metres acts once after placement.
Linked bundles, legacy text conversion, body-only formula/grid helpers,
other profile families and combined native features refuse until they can
transport this complete authority. Direct density, preview flags, geometry
and CAD options have strict types and cannot silently bypass native intent.
Direct normalized model parameters must retain the supported neutral source,
wall, profile and feature controls before geometric identity is reduced.
