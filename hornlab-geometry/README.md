# @hornlab/geometry

Canonical parametric horn geometry for HornLab. Owns the formulas
(OSSE/ROSSE/LOOKUP profiles, cross-sections, axial sweeps), the mesh
authoring (B-spline horn walls, enclosures, source patches), and the
canonical mesh payload contract (surface tags, integrity validation).

This package was extracted from `Waveguide-Generator/src/geometry/` so
non-WG consumers (`hornlab-mesher` via the geometry-cli subprocess,
Optimizer-Dashboard, future tools) can import it cleanly without
depending on the entire WG repo.

## Consumers

- **Waveguide-Generator**: imports for the browser UI live view + the
  Python mesh-build pipeline.
- **Waveguide-Generator/bin/geometry-cli.js**: NDJSON REPL bridging the
  geometry to non-JS consumers (Python).
- **hornlab-mesher** (Python): spawns the geometry-cli subprocess to
  evaluate profiles and grids before lofting OCC surfaces.
- **Optimizer-Dashboard** (potential): could import directly for a
  parametric live preview in the SetupPanel.

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

Geometry semantics changes that affect the WG point-grid contract should
bump the minor version and be flagged in WG release notes.
