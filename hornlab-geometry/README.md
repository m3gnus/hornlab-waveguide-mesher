# @hornlab/geometry

Canonical parametric horn geometry for HornLab. Owns the formulas
(OSSE/ROSSE/LOOKUP profiles, cross-sections, axial sweeps), the mesh
authoring (B-spline horn walls, enclosures, source patches), and the
canonical mesh payload contract (surface tags, integrity validation).

This package was extracted from Waveguide Generator geometry work so
standalone consumers (`hornlab-mesher` via the geometry-cli subprocess and
future tools) can import it cleanly without depending on an entire UI repo.

## Consumers

- **hornlab-mesher** (Python): spawns the geometry-cli subprocess to
  evaluate profiles and grids before lofting OCC surfaces.
- **Boundary Lab or other tools**: can use the CLI/API to generate point
  grids before meshing or solving.

## Layout

```
src/
  index.js              top-level re-exports
  expression.js         ATH expression parser
  params.js             prepareGeometryParams + normalization
  pipeline.js           buildGeometry* / buildCanonicalMeshPayload*
  tags.js               canonical surface tags (1=wall, 2=source, ...)
  common.js             evalParam, toRad, parseQuadrants, ...
  meshIntegrity.js      assertBemMeshIntegrity
  edgeTopology.js
  quality.js
  engine/
    profiles/{osse,rosse,tractrix,classical,guidingCurve,validation}.js
    profileSystem.js / profileSections.js
    crossSection.js / morphing.js / roundover.js / interp.js
    rectConicalHorn.js
    buildWaveguideMesh.js / constants.js / math.js / index.js
    mesh/{horn,enclosure,enclosureSeams,freestandingWall,source,angles,sliceMap}.js
```

## Versioning

`0.1.0`. Pinned via `file:../hornlab-geometry` in consumer `package.json`
files until the workspace grows enough to justify a real version pin.

Geometry semantics changes that affect the point-grid contract should bump the
minor version.
