# Native front-baffle woofer

The canonical `SourceContour` model can close a circular aperture on a finite
rectangular baffle. Cone depth, spherical dust-cap radius/height, surround
width/depth and rigid land remain independently editable. Baffle aperture,
width, height, enclosure depth and source XYZ placement are separate inputs.

```python
from hornlab_mesher.source_contour import cone, ContourDrive
from hornlab_mesher.front_baffle import FrontBaffle
from hornlab_mesher.woofer_artifact import export_woofer

shape = cone(10, 4, cap_radius_mm=3, cap_height_mm=2,
             surround_width_mm=2, surround_depth_mm=.6, land_width_mm=1)
baffle = FrontBaffle(44, 54, 12, 14, (3, -4, 7))
drive = ContourDrive("motor", (("dust-cap", 1), ("cone", -.5), ("surround", 0)))
export_woofer(shape, drive, baffle, "woofer-artifact", mesh_size_mm=2)
```

Coordinates are finished mm. The rectangular enclosure is centred on XY=0;
its front lies at the source's Z coordinate, and its back is `depth_mm` behind
that plane. The source uses an aligned +Z axis and its local rim lies at z=0.
`baffle.preview(shape)` translates the same canonical preview points.
`FrontBaffle.to_dict/from_dict` and `SourceContour.to_dict/from_dict` preserve
saved models; presets expand to this explicit representation.

The aperture covers the entire contour footprint, including its rigid land.
A larger aperture is closed by an explicit rigid mounting collar. A collar
and the clearance to each outer baffle edge must exceed 0.1 mm. The recessed
source also clears the back by more than 0.1 mm. Invalid dimensions are refused
without clamping. The skin contains source patches, front, four sides, back and
an optional collar, sewn with actual shared CAD edges. STEP has one closed
surface shell and no solid volume; preview MSH shares nodes across these joins.

The version-1 artifact requires both `native-source-contour-v1` and
`native-front-baffle-woofer-v1`. Geometry hashes include placement and aperture;
excitation and mesh density retain separate identities. Stable moving/rigid
patch IDs are independent of STEP selectors. Publication requires a new
directory, uses an isolated worker and preserves caller Gmsh state.

Waveguide Generator's `server.cadlink.native_source.ingest_native_source`
verifies canonical finite source and baffle faces before imported meshing.
Its actual moving and rigid facets must certify within 0.15 mm, and its mesh
must be closed with consistent winding. Existing triangle/memory limits apply.
The native LF patches form one physical source/channel; literal signed/zero
weights and normal/aligned axial motion use the existing prescribed-drive
contract. Tests inspect CPU boundary multipliers without acoustic solves.

This slice supports one circular woofer on a rectangular enclosure through
the Python API. Drawing UI, arbitrary source axes, general horn/woofer assembly,
multiple sources and manufacturing solids are separate features. Coordinated
module/consumer installation and dependency pins require integration.
