# Standalone circular source body

`SOURCE-DISK` builds a closed exterior cylinder with one normal-driven circular disk on its front. A rigid coplanar annulus surrounds the disk; the cylindrical side and rear are rigid. The front is at z=0 and the rear at negative z. The source normal is +Z. The rear is not driven.

This construction has its own immutable geometry and evaluates no horn profile. It has no mouth, throat, aperture or subdomain interface.

```toml
formula = "SOURCE-DISK"
mode = "standalone-source"

[source_body]
radius_mm = 18
outer_radius_mm = 22
depth_mm = 8

[mesh]
size_mm = 3
vertical_offset_mm = 7
scale_to_metres = true
max_triangles = 18000
allow_large_mesh = false
```

Save this as `source.toml`, then use the existing CLI:

```sh
hornlab-waveguide source.toml -o source.msh --print-summary
hornlab-waveguide source.toml --step source.step --print-summary
```

JSON accepts the same object structure. Python callers can use `build_from_config`, `write_step_from_config`, or the dedicated geometry:

```python
from hornlab_mesher import StandaloneSourceGeometry, build_mesh, write_step

body = StandaloneSourceGeometry(radius_mm=18, outer_radius_mm=22, depth_mm=8)
build_mesh(body, output_path="source.msh")
write_step(body, "source.step")
```

All shape dimensions are finished millimetres. Source radius is 0.1–200 mm, outer radius is at least 0.1 mm greater and at most 500 mm, depth is 0.1–500 mm, and the magnitude of final Y placement is at most 10000 mm. Numeric controls require finite scalar numbers; booleans, expressions and numeric strings are refused. There is no additional scale.

`mesh.size_mm` is a uniform target in 0.1–50 mm, default 3 mm. The effective target is at most radius/8, and each circular boundary has at least 32 segments. Mesh density changes preserve physical shape and construction identity. Direct Python builds default to uniform 3 mm. An explicit `MeshDensity` must give equal `throat_res_mm`, `mouth_res_mm` and `rear_res_mm`; these three values jointly specify the uniform target for this type. Optional/interface/enclosure controls must retain their defaults. The source does not interpret them as horn regions.

The conservative allocation estimate is `3 * exterior_area / effective_size**2`. It is an estimate, not a triangle upper bound. Builds above its hard 200000 envelope are refused before meshing. Actual output is also limited to 200000 triangles. The default user limit is 18000; an estimate above twice that limit is refused early, and the actual count is checked before publishing. `allow_large_mesh=true` bypasses the user limit, while the hard envelope remains enforced. Some combinations inside the physical ranges therefore require a coarser mesh or are refused. A refused build preserves an existing output file.

The mesh contains only rigid tag 1 (`SD1G0`) and moving source tag 2 (`SD1D1001`). Source and annulus share their actual boundary vertices. Before publication the mesher checks every edge has incidence two, the shell is connected, source/rim geometry and outward normals, source area within 1%, and positive volume within 1% of the exact cylinder. Mesh size is a target; no global facet chord tolerance is implied. Metadata records the physical recipe, density-independent fingerprint, full bounds, requested/effective size, budget estimate and realized certificate. Mesh coordinates are metres by default, or millimetres with `scale_to_metres=false`; geometric metadata and edge statistics remain millimetres.

The preview emits four exterior roles: `source_body.disk`, `source_body.annulus`, `source_body.side`, and `source_body.rear`. Existing preview switches select the source disk (`include_source_cap`), annulus and side (`include_outer`), and rear (`include_rear_cap`). Inner/enclosure switches do not create additional surfaces. Circular chord, cylinder normal-angle and binary32 placement bounds are checked before allocation. Requested vertex/tolerance budgets that cannot be satisfied are refused. Preview geometry depends on physical controls and preview options, independently of mesh density. Planar curvature is zero; outward cylinder mean curvature is -1/(2R), with signed principal curvature -1/R.

Dimensions report `source_diameter` and `body_overall`. Datums describe `SOURCE_AXIS`, `SOURCE_PLANE`, `SOURCE_OUTLINE`, `BODY_REAR_PLANE` and `BODY_BOUNDS`. Bounds include final Y placement. No horn mouth or throat datum is fabricated.

STEP exports one four-face closed solid in millimetres. The source disk stays present with the default `open_throat=True`, and `throat_opened` reports false. The strict JSON recipe and fingerprint are embedded in a STEP comment and returned in CAD metadata. STEP preserves face geometry and the shared source/annulus edge; it does not transport solver physical groups. A downstream STEP importer must select and map the source face. Source-only `.wglink` bundles are refused until that schema can represent this construction.

The root discriminator and exact root `source_body` object are required. Every supplied unsupported key is refused, including zero, null and empty values. Horn profile coefficients, scale, prefix, morph, adapters, lips, enclosures, curved/weighted sources, arrays, interfaces, imported geometry and symmetry-reduced coverage are outside this slice. Only full `quadrants=1234` is accepted. Existing readers refuse the unknown formula instead of interpreting the construction as an ordinary horn. GUI integration, installed artifacts and acoustic convergence are separate consumer qualifications.
