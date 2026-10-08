# Native circular mouth roundover

Use `formula = "OSSE-ROUNDOVER"` with `mesh.mouth_roundover_radius_mm` to
continue a circular wall smoothly around its mouth. The formula marker makes
older readers refuse the design instead of silently discarding its lip.

```toml
formula = "OSSE-ROUNDOVER"
mode = "freestanding"
scale = 1.0

[profile]
L_mm = 120
r0_mm = 12.7
a_deg = 45
a0_deg = 10
k = 1
s = 0

[mesh]
wall_thickness_mm = 5
mouth_roundover_radius_mm = 25
mouth_res_mm = 8
allow_large_mesh = true

[source]
source_shape = 0
```

The first implementation accepts scalar, unterminated (`s = 0`) circular OSSE
profiles, full-circle coverage, freestanding walls, and a flat source (the
default for this formula). Both the lip radius and wall thickness use finished
millimetres; global scale multiplies the body only. Body length must finish
between 1 and 2000 mm, throat radius between 1 and 200 mm, `k` between 0.5 and
10, and angles satisfy `0 <= a0_deg < a_deg <= 75`, with `a_deg >= 5`. Scale
is restricted to 0.01 through 10 and vertical placement to 10000 mm in either
direction. The wall is at least 0.1 mm and the radius exceeds it by at least
0.1 mm, with a maximum radius of 200 mm. A conservative curvature guard refuses
an outer offset approaching a fold, and the complete lip must clear the rear.

Termination, non-circular sections, symmetry sectors, guiding curves, morphs,
rollback, throat extensions, slots, stretch, adapters, enclosures, baffles,
bare surfaces and curved sources are explicitly refused. Only canonical keys
shown in this document and the supported mesh density controls are accepted.
The text import option remains refused. This is a bounded native geometry
feature, with broader combinations reserved for separate qualification.

All supplied `formula` and `type` fields must agree with `OSSE-ROUNDOVER`.
Coverage must be exactly `1234` (integer or string), and duplicate root/mesh
vertical placement controls must agree. Supplied `length_segments` and
`angular_segments` must be positive integer numbers; valid counts do not
change the analytic construction. `allow_large_mesh` and `scale_to_metres`
accept only booleans, including the optional `allow_large_mesh` API override
(which also accepts `None` to retain the configuration setting).

The inner circle starts at the original body mouth `M = (zM,rM)` with the
analytic body tangent angle `beta`. Its centre is
`C = M + R*(-sin(beta), cos(beta))` and its sweep is `pi-beta`. The outer
circle is concentric with radius `R-wall`. A rear-facing flat annulus joins
their ends. The original mouth datum remains at `zM`; the full forward bound
is `zM + R*(1-sin(beta))`. Overall width is `2*(Cr+R)` and overall depth also
includes the rear plate. The preview reports both the original opening and the
complete envelope, independently of sampling counts.

Preview curvature is signed relative to each emitted surface normal. Mean
curvature averages the two signed principal curvatures, and the principal
readout retains the sign of the curvature with the greatest magnitude. Equal
opposite magnitudes select the positive principal curvature.

Preview, mesh and STEP use this same canonical meridian. OCC keeps the lip
arcs exact. The body and its analytic normal offset use cubic Hermite spans
with matching endpoint tangents and a whole-interval fourth-derivative error
bound of 0.0001 mm. The default preview certifies its combined meridian,
angular and binary32 rendering chord error below 0.008 mm, refusing requests
that exceed its sampling or vertex budget. Solve density on the lip is capped
by its curvature and requested mouth resolution, with at least six intervals
over the sweep. Meshes retain the rigid wall and source physical groups.
These small chord tolerances can require several hundred thousand triangles;
the example opts into that cost explicitly. The ordinary triangle budget still
applies when `allow_large_mesh` is false, and a denser lip can exceed it.
The pre-mesh cost estimate includes the actual lip area and its capped size,
so a clearly exceeded budget refuses before tessellation. Each generated
curved lip triangle also receives a continuous torus interpolation bound;
an error above 0.01 mm refuses the export instead of relying on the requested
element size alone.

Changing radius changes the construction fingerprint carried by preview,
mesh metadata and STEP. STEP includes the construction recipe and fingerprint;
its CAD result carries the original mouth datum and exact full envelope.
Linked CAD bundles are refused until their analytic recipe transport is
qualified. Body-only formula/grid entry points refuse active lips, so they
cannot return a truncated design. With an ordinary formula, a zero or absent
radius keeps the existing geometry and export paths exactly.
