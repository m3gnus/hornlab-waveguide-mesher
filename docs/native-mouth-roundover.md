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
bare surfaces and curved sources are explicitly refused. An absent or validated
`off` throat adapter preserves the lip identity; an active adapter requires
`OSSE-NATIVE` (see below), and duplicate adapter locations are refused. Only canonical keys
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
element size alone. All native OCC interfaces share topology, followed by
exact Gmsh duplicate-node removal. The finished native mesh skips approximate
fixed-distance welding, which would collapse valid dense edges and invalidate
the construction. Legacy mesh paths retain their existing welding behavior.

Changing radius changes the construction fingerprint carried by preview,
mesh metadata and STEP. STEP includes the construction recipe and fingerprint;
its CAD result carries the original mouth datum and exact full envelope.
Linked CAD bundles are refused until their analytic recipe transport is
qualified. Body-only formula/grid entry points refuse active lips, so they
cannot return a truncated design. With an ordinary formula, a zero or absent
radius keeps the existing geometry and export paths exactly.

## Roundover with axial scale

`formula = "OSSE-NATIVE"` with an active `mesh.mouth_roundover_radius_mm`
and root `axial_scale` composes the lip with the absolute axial override, see
the [composition contract](native-osse-composition.md). The body is mapped
first: Z by `axial_scale`, radius by `scale`, with both factors and their
ratio in the [absolute axial scale](absolute-axial-scale.md) domain. The
outer wall offset, the lip radius, the rim and the rear closure are then
built in finished millimetres against the mapped body, so the lip stays a
true circle tangent to the mapped mouth. The mapped body is the axial
feature's own model, with its domain and 10000 mm mouth limit. Every other
control, domain check, fit bound and certificate is the single-feature lip's. `OSSE-ROUNDOVER` with
`axial_scale`, or `OSSE-AXIAL` with an active radius, is refused with a
pointer to `OSSE-NATIVE`.

The construction is `circular-osse-axial-lip-v1`. Its fingerprint is the
SHA-256 of the identity the STEP header carries, which records the physical
length, throat radius and slopes, authored angles and `k`, wall, radius,
placement and both factors, so an axial factor equal to `scale` builds the
same horn under a different identity. Resolved formula, preview metadata and
`mouthRoundover` metadata report `OSSE-NATIVE`, `xyScale` and `axialScale`,
and mesh metadata carries `construction_fingerprint`. STEP writes the
composite `native-composite` header recipe instead of the
`hornlab-mouth-roundover` trailer. Mesher readers 0.2.6 and 0.2.7 refuse
this design.

## Roundover with a throat adapter

`formula = "OSSE-NATIVE"` with an active root `throat_adapter` (authored
mode) and an active `mesh.mouth_roundover_radius_mm`, with or without root
`axial_scale`, gives a compression-driver throat and a rolled mouth on one
closed solid. See the [composition contract](native-osse-composition.md).
Construction order: the intrinsic conic, the authored adapter cubic, the
axial map (`axial_scale`, or `scale` when absent), then the offset wall,
lip, rim and rear closure in finished millimetres, then placement. The
throat is the same mapped cubic, with the same checks and refusal prefix, as
`OSSE-ADAPTER-AXIAL`; the retained conic starts on its exact end point, so
the inner wall is G1 at the join. The lip is built at the body mouth exactly
as above.

The outer wall offsets the cubic by the wall thickness along its normal.
Mesh and STEP use a cubic Hermite fit of that offset with exact endpoint
positions and tangents, certified by a whole-interval bound on its fourth
derivative (Faa di Bruno over Bernstein hulls of the cubic's speed) to the
same 0.0001 mm as the conic's fit. Inner and outer walls each share one
point at the join, and both fits spend the roundoff gap there within that
bound. The inner cubic is the exact Bezier. These bound the CAD and mesh
geometry against the analytic walls; as for Adapter × Arc, the mesh
facet-to-surface certificate covers the lip only. The source disk sits on
the driver rim, and the rear return starts at the outer wall's throat point
(in front of the throat plane for a converging exit angle).

Extra refusals guard the offset and the closed solid:

- the adapter exit angle must be within 75 degrees of the axis;
- the wall tangent, over the cubic's derivative hull and the retained conic
  up to the mouth, must turn by less than 85 degrees. With a narrower range
  than 90 degrees, every outer-wall point lies above the inner wall at its
  own z, so the walls cannot cross;
- `wall * curvature` must stay below 0.4 over the whole cubic, certified on
  Bernstein hulls (the conic's existing fold guard is unchanged);
- the cubic and its offset must clear the lip by 0.1 mm, either axially (the
  adapter ends before the lip centre) or radially (the throat stays inside
  the mouth radius), and must stay within the lip's outer reach, so the
  roundover bounds stay the full envelope.

The construction is `circular-cubic-conic-lip-v1`. Its identity records
the adapter payload, the authored body, the feature set and every physical
control, so an authored `axial_scale` equal to `scale` builds the same horn
under a different identity. The fingerprint is the SHA-256 of that identity,
which the STEP `native-composite` header recipe carries. `mouthRoundover`
metadata adds `adapterJoinMm`, `sourceRadiusMm`, `sourceAngleDeg` and
`throatAdapter`; datums add `WG_ADAPTER_JOIN_PLANE`. The mesh is a closed,
consistently oriented body with the rigid-wall and source groups, as for
every lip; STEP is the opened solid. Controls adapters, terminating arcs and
standalone sources stay refused, and mesher readers 0.2.6 and 0.2.7 refuse
the design.
