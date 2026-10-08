# Native circular throat adapter

Native JSON and TOML configurations can replace the beginning of a scalar
OSSE body with a cubic adapter. The driver opening and the retained body meet
with continuous oriented tangent direction. Curvature may change at the join.

```json
{
  "formula": "OSSE-ADAPTER",
  "mode": "bare",
  "profile": {"L": 120, "r0": 12.7, "a": 40, "a0": 15.5, "k": 1, "s": 0},
  "source": {"source_shape": 0},
  "throat_adapter": {
    "mode": "authored",
    "contract_revision": 1,
    "driver_exit_diameter_mm": 25.4,
    "exit_half_angle_deg": 10,
    "length_mm": 40,
    "join_t": 0.137,
    "driver_handle_mm": 10,
    "body_handle_mm": 15
  }
}
```

The required `OSSE-ADAPTER` formula discriminator makes earlier readers reject
the design instead of silently dropping the adapter. An active payload with
ordinary `OSSE`, or the new discriminator without an active payload, refuses.
To disable it, use ordinary `OSSE` and remove the object or set `mode: "off"`.
Inactive controls do not affect geometry or construction identity. Unknown
fields, modes and revisions refuse. The object may instead be inside `profile`,
but cannot appear in both locations. Native save/load retains its six controls
at full numeric precision.

`join_t` identifies a point on the original body, at axial distance `L*join_t`.
The cubic starts at the driver plane z=0, radius diameter/2, and ends at z equal
to `length_mm`, with the original body's radius and analytic tangent there.
Each handle is a distance in the meridian plane. The retained body is translated
without changing its shape. Assembled depth is `length_mm + L*(1-join_t)` before
global scale. Scale acts once; vertical placement follows it.

Sampling uses a composite parameter from 0 to 1: the cubic occupies 0 to .5,
and the retained original body occupies .5 to 1. The exact driver, join and
mouth are semantic stations. Preview samples this construction; mesh and STEP
use its exact cubic and rational conic as surfaces of revolution. A construction
fingerprint records the active controls, base parameters, scale, placement and
source rule independently of mesh density. Preview and mesh metadata carry this
identity; STEP includes the native recipe in its header and returned CAD metadata.
The solid-only `.wglink` bundle format remains unavailable for this bare surface.
Default coarse and fine previews use a chord-error budget of .008 mm, tightened
for small adapters. An explicitly requested preview tolerance remains effective.

The initial scope is circular scalar OSSE profile 1 with zero termination
strength, bare acoustic topology and an explicit flat source (`source_shape=0`).
Straight adapter aliases, extensions, slots, stretch, bulge, rotation, morph,
guiding curves, lookup profiles, noncircular sections, symmetry sectors, walls,
enclosures, baffles, subdomain interfaces and approximate surface fitting
refuse. Other profile families and control imports remain unsupported.

Certification conservatively requires positive radii and ordered forward
cubic controls, positive body radius, and disjoint open axial spans joined at
one endpoint. The circular revolution then has a nonzero Jacobian and cannot
self-intersect. Near-contact handles and unresolved numeric conditions refuse.
The conic domain requires `0 <= a0 < a < 80` degrees, `a >= 1` degree, positive
`L` and `r0` no larger than 10,000 mm, positive `k <= 100`, and scale between
.001 and 100. Built coordinates must remain within 100,000 mm and certified
feature clearances exceed .0001 mm. These bounds are validity limits, not
silent corrections of user controls. Mesh triangles still approximate the
exact surfaces according to the requested mesh resolution.
The native mesh uses shared OCC edges and Gmsh duplicate-node removal at the
source and adapter join. It skips the legacy approximate distance weld, which
would collapse certified small source disks or densely sampled source edges.
