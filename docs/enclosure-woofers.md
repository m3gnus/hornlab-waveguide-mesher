# Resolved enclosure woofers

The config-driven enclosure path accepts concrete geometry in
`enclosure.woofers`. Placement relations and excitation processing belong to the
caller. The experimental `spike_woofers` field is refused.

```json
{"enclosure":{"depth_mm":240,"space_b_mm":150,"woofers":[
  {"radiator_id":"lf-b","x_mm":0,"y_mm":-275,"radius_mm":35,
   "shape":{"kind":"piston","recess_mm":2},"active":true,
   "resolution_mm":10}
]}}
```

`radiator_id` is a persistent, unique nonempty identity, not a display label or
array index. It cannot contain `/`, `|`, quotes, backslashes or control characters.
The geometry boundary uses frozen `EnclosureWoofer`, `PistonShape` and
`AthConeShape` values; the enclosure stores a tuple. Coordinates and dimensions
are finite millimetres in the horn throat frame, before final vertical placement
and output-unit conversion. Radius and optional resolution must be positive.
`active` defaults to true and must be a boolean. Recess defaults to zero and must
be zero or at least 0.01 mm, matching the shared vertex-weld tolerance.

`shape: {"kind":"ath_cone"}` produces a revolved conical annulus and spherical
cap. At aperture radius r, the rim is flush, the join is at radius 0.35r and
depth 0.4r, and the sphere radius is 0.525r. Its centre is at
`-(0.4 + sqrt(0.525² - 0.35²))r`; its apex is about 0.266312r below the baffle.
The cone and cap share one radiator identity and axial +Z motion. Mesh density
defaults to 10 mm per radiator; the cap also has a curvature-based chord sizing
cap. Resolution is a characteristic length, not a maximum-edge guarantee.

Every active segment has a distinct physical group from 101 and a patch ID
`radiator_id/segment_id` (`piston`, `cone`, or `cap`). Numeric tags are allocated
per build; use patch IDs for persistent identity. The primary horn retains tag 2
and source identity `horn-throat`; rigid surfaces use tag 1. Physical names use
`wg-assembly-v1|tag=101|source_id=lf-b/cone|instance_id=null|role=secondary`.
The primary name uses the same namespace and `role=primary`; rigid is named
`wg-assembly-v1|rigid`.

Result metadata publishes `tag_namespace`, `rigid_tag`, `source_tags` (patch ID
to tag), and `tag_map` (string tag to `source_id`, nullable `instance_id`, and
`role`). `assembly.patches` records every woofer segment, including inactive
ones, with radiator/segment identity, physical tag, OCC ancestry, area and
resolved geometry. An inactive woofer is meshed with rigid tag 1, recorded with
`role=rigid`, and omitted from `source_tags`. Its identity survives in the patch
record rather than as an active source physical name.

Patch centres and areas use the explicitly recorded
`assembly.coordinate_frame = horn-throat-pre-placement-mm`;
`assembly.vertical_offset_mm` records the final Y translation. Apply that
translation once and then convert millimetres to the mesh output units when
comparing the record to emitted vertices. This is mesher metadata for the caller's
compiled record, not a replacement for assembly frames, drives or solve records.

Admission requires at least 0.1 mm clearance to the actual mouth/flat-baffle
boundary and between apertures, full baffle containment, and no diaphragm/collar
intersection with the horn. Every OCC face and emitted facet has one physical
owner; every declared segment must retain positive area, analytic geometry and
exterior-facing normals after mesh cleanup. Existing closure and orientation
checks remain active, including validation of disconnected components.

The default domain is full. An explicit x=0 half (`quadrants=14`) is supported
only for centred woofers, equal left/right enclosure margins and a horn symmetric
about x=0. Its diaphragms and collars are clipped to x>=0 without a fictitious
closing face on the symmetry plane. Other reductions and subdomain interfaces
are refused. The supported construction is the existing point-grid design
enclosure, including rounded FREEFORM corners.

With no woofers (field absent or empty), the assembly hook and namespace changes
are inactive. The regression compares mesh and STEP bytes against the corner-fix
base in the same native runtime for six horn families.
