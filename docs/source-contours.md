# Native circular source contours

`hornlab_mesher.source_contour` provides immutable `SourceContour`,
`ContourPoint`, `ContourSegment` and separate `ContourDrive` models. `flat`,
`dome` and `cone` presets expand to explicit points and line/circular-arc
segments; cone includes an independently dimensioned spherical dust cap.
Surround width/depth and rigid land width are independent dimensions.

```python
from hornlab_mesher.source_contour import dome, ContourDrive
from hornlab_mesher.contour_artifact import export_contour

shape = dome(18, 6, surround_width_mm=4,
             surround_depth_mm=-1, land_width_mm=2)
drive = ContourDrive("motor", (("dome", 1), ("surround", 0.4)), "axial")
export_contour(shape, drive, "source-artifact",
               mouth_radius_mm=48, length_mm=70, mesh_size_mm=2)
```

The initial attachment is a native circular conical horn cavity inside a rigid
circular housing, in finished mm with its throat at the origin and forward
axis +Z. Housing radius and backing depth are editable. The source closes the
horn's actual circular throat: joins share OCC edges and mesh nodes. The full
acoustic surface is closed; STEP carries one shared shell with no solid volume.
This is a bounded Python API; arbitrary existing horn configurations, drawing
UI, rotated axes, folded contours and manufacturing solids are separate work.

The ordered meridian has one pole, strictly increasing radius and an exact
z=0 attachment. Lines and arcs retain deliberate corners. Arcs explicitly name
their center and clockwise/counterclockwise branch. Circular segments must be
single-valued in radius and no longer than a semicircle. Presets and saved
point models share the version-1 serialization. At most 256 patches are allowed.

Each segment has a stable patch ID and explicit `moving`/`rigid` role. Several
patches share one physical-source ID and one channel. The drive covers exactly
moving patches with finite literal real weights; zero, non-unit and negative
values are retained. A zero moving weight does not make a patch rigid.
Normal and aligned axial prescribed motion are distinct from geometry.

`source.json` binds `geometry.step` and `preview.msh` to exact SHA256 digests.
The geometry recipe, excitation and density have separate identities. STEP
ADVANCED_FACE selectors address only those exact bytes and are regenerated
after every export; they are never durable patch IDs. Publication requires a
new directory and is atomic; failures preserve existing destinations.

`server.cadlink.native_source.ingest_native_source` in Waveguide Generator uses
the general imported STEP mesher, verifies the canonical patch geometry on
reopened faces, and publishes its own native ingestion record. It preserves
complete rigid coverage and exact weights/motion. Both the manifest and solve
request require `native-source-contour-v1`; unknown features and contradictory
requests fail before dispatch. This is separate from the planar linked-throat
and managed CAD-return contracts. First consumer support is explicitly Metal;
weighted prescribed channels cannot carry a lumped DriverSpec.

The producer tests check finite line/arc distances for every facet with a
1-Lipschitz covering bound of 0.15 mm, shared joins, closed oriented topology,
normals, roles, identities and reopened STEP. Consumer tests exercise actual
ingestion and dispatch, then evaluate production per-face BC multipliers on
CPU against an independent normal-dot-axis oracle. Those tests intercept the
acoustic solve; they make no acoustic accuracy claim. Native contour support
requires the coordinated producer and consumer revisions; dependency pin and
installed qualification belong to integration.

The imported consumer preserves local curvature refinement and certifies its
actual moving and rigid facets against the same 0.15 mm tolerance before
publication. Requested and effective mesh sizes are recorded separately.
