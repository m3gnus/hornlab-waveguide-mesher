# Native circular terminating arc

Use `formula = "OSSE-ARC"` to retain a circular OSSE prefix and continue it
with an exact circle matched to its tangent and curvature. The explicit join
fraction determines geometry independently of sampling and mesh density.
The formula marker makes older readers refuse an unsupported design.

```toml
formula = "OSSE-ARC"
mode = "bare"
scale = 1.0

[profile]
L_mm = 120
r0_mm = 12.7
a_deg = 40
a0_deg = 15.5
k = 1
s = 0

[terminating_arc]
contract_revision = 1
join_t = 0.05
end_tangent_deg = 100

[mesh]
wall_thickness_mm = 0
throat_res_mm = 5
mouth_res_mm = 15
allow_large_mesh = true

[source]
source_shape = 0
```

The root `terminating_arc` object requires exactly the three shown controls.
`contract_revision` is integer 1. `join_t` is the fraction of the original
body length retained, with `0 < join_t <= 1`. `end_tangent_deg` is the absolute
oriented tangent angle measured from forward +Z toward increasing radius.
It must exceed the tangent at the join and be at most 180 degrees. At 90
degrees the suffix points radially outward; above 90 it turns backward in Z.
There is no inactive object mode: remove the object and select ordinary OSSE
to disable this feature. A supplied object under an ordinary formula refuses,
including null and empty objects. Only the exact root spelling is accepted.

The first domain is scalar circular OSSE with `s = 0`, bare mode, zero wall
thickness, full-circle coverage and a flat native source. Only canonical
profile controls in the example are accepted. Authored throat adapters,
independent axial scaling, mouth shells, source bodies, guides, morphs,
extensions, slots, stretch, rotation, enclosures, baffles, sectors, curved
sources, lookup profiles and imported text designs are refused. Every
supplied formula/type selector must agree. Duplicate placement values must
agree. Density counts must be positive integer numbers, and mesh flags and
the optional `allow_large_mesh` API override require actual booleans.
Native construction requires acoustic topology and `surface_fit` auto or
interpolate; export always uses the exact analytic construction.

Uniform scale in [0.01,10] multiplies all intrinsic distances once. Finished
body length is in [1,2000] mm, throat radius in [1,200] mm, `k` in [0.5,10],
`5 <= a_deg <= 75` and `0 <= a0_deg < a_deg`. The retained body and circular
arc must each span at least 0.1 mm, circle radius is in [0.1,10000] mm, actual
opening Z is at least 0.1 mm and radial reach is at most 10000 mm. The complete
numeric coordinate envelope is bounded by 100000 mm. Finished Y placement
is limited to 10000 mm in either direction and acts last. Output conversion
to metres acts after construction and placement. Precision certificates can
refuse unresolved inputs within these outer limits. The limits are coupled:
180 degrees is inside the angle envelope but cannot satisfy the source-plane
clearance for these OSSE bodies. Controls are never silently shortened.

At the join J=(zJ,rJ), let beta be the body tangent angle and kappa its positive
meridian curvature. Set R=1/kappa and C=J+R*(-sin(beta),cos(beta)). The suffix
is `C + R*(sin(phi),-cos(phi))`, from beta to the authored end tangent angle.
The prefix is an exact positive rational quadratic conic. The circle shares
its exact join point, tangent and curvature, giving a G2 geometric join.
Radius increases along both branches, so their open interiors cannot
intersect even when the suffix turns backward.

The actual endpoint defines the opening and `WG_MOUTH_PLANE`.
`WG_ARC_JOIN_PLANE` marks the join. `WG_BODY_MOUTH_REFERENCE_PLANE` marks the
original body length and is explicitly nominal/reference; it can be outside
the emitted geometry. When the circle passes 90 degrees, full forward extent
is `C.z + R`, rather than endpoint Z. Overall width is twice endpoint radius.
These complete dimensions and datums are independent of density and preview
LOD. STEP returns datums and exact bounds alongside its construction recipe
and fingerprint.

Preview defaults are 0.008 mm chord error and 3 degrees normal step on all
three LODs. The continuous geometric triangle bound combines the whole-span
meridian chord bound, angular radial shrink and binary32 coordinate rounding.
This bounds geometric distance to the analytic surface; it does not assert
identical-parameter correspondence. Monotone body tangent angles and exact
circle angle ranges certify the normal bound. Custom tolerances remain
effective and requests beyond precision, sampling or vertex limits refuse.
Signed curvature follows the inward surface normal. Circle meridian
curvature is -1/R and angular curvature is cos(phi)/r, including its negative
sign after 90 degrees. Dominant principal curvature retains its sign.

Mesh and STEP revolve the exact conic and circle with shared quadrant,
join and source edges. Physical groups remain rigid wall 1 and source 2;
the source points +Z and the actual mouth stays open. STEP is a surface body.
Native mesh sampling keeps at least 16 source-rim intervals to preserve cap
area for small sources at coarse throat resolution; finer throat requests
remain effective.
Exact duplicate-node removal preserves dense edges; approximate distance
welding is skipped for this native construction.
STEP transports one sewn open shell with shared edge identities, including
the source edge when `open_throat = false`. The direct native STEP flag must
be a boolean. Its published representation contains no solid or volume.

Arc sizing uses a local radius target, capped by requested mouth resolution
and six intervals along the suffix. The target is a sizing heuristic; every
actual circle triangle separately receives a continuous Hessian certificate
of facet-to-analytic-surface error at most 0.01 mm. This certificate covers
the circular suffix only; it does not claim parameter coverage or bound the
separately tessellated retained conic. A failed certificate refuses export
with a density remedy. Large arcs can require hundreds of thousands of
triangles, so the example explicitly opts into that cost. Pre-mesh estimates
include the circle area and its varying size field; the ordinary triangle
budget applies unless explicitly overridden. Existing output files survive
refused or failed exports.
For each triangle, the angular Hessian bound uses the maximum analytic radius
over that triangle's finite tangent-angle interval. The recovered angles are
clamped to the authored suffix, and full vertex-to-projection distance plus
an outward numeric margin contributes to the bound. The meridian and mixed
Hessian terms retain the full circle radius. Normal validation selects the
source-adjacent retained-body faces by exact topology, preserving its usual
inversion threshold when a returning suffix re-enters the throat's axial band.

Direct native mesh density accepts only the three canonical resolutions,
positive integer triangle budget and actual boolean large-mesh override;
unsupported active density controls refuse. Direct preview inclusion flags
also require booleans, and its hard vertex ceiling is one million.

Construction identity includes original body and arc controls, uniform scale
and finished placement. Density, preview LOD, routing and output units do not
change it. The immutable meridian owns detached recipe data and immutable
arrays. Body-only formula/grid helpers and linked bundles refuse this feature
until they can transport the complete analytic authority.
