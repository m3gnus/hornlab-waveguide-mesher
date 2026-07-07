# Change Guide

Use this guide when changing behavior that affects generated profiles, topology,
mesh density, physical tags, or compatibility with imported ATH/WG configs.

## Before Changing Behavior

1. Identify the contract being changed: profile math, point-grid shape,
   topology, density, tags, orientation, config import, or public API.
2. Read the matching docs:
   - `docs/config-schema.md`
   - `docs/geometry-contract.md`
   - `docs/builder-invariants.md`
   - `docs/public-api.md`
3. Add or update a focused contract test before changing behavior that a solver
   or imported reference config can observe.
4. Keep config import compatibility separate from canonical geometry rules.

Run targeted tests while iterating:

```bash
python3 -m pytest tests/test_cli.py tests/test_point_grid_contract.py -q
python3 -m pytest tests/test_density_contract.py tests/test_orientation_validation.py -q
```

Run the full suite before finishing shared builder or postprocess changes:

```bash
python3 -m pytest tests -q
```

## Profile Parameter Changes

Profile parameters enter through `config_builder.build_geometry_params`, then
flow into `profiles.build_point_grid`.

When adding or changing a parameter:

- Add TOML/JSON aliases in `config_builder.py` only where external configs need
  them.
- Add ATH text mappings in `config_parser.py` only for imported ATH/WG names.
- Keep units explicit in key names: use `_mm` and `_deg` for new canonical
  config keys.
- Decide whether expression strings are supported. Use `_scalar_or_expr` only
  when the profile layer can evaluate expressions safely.
- Update `docs/config-schema.md` and, if math changes, `docs/geometry-contract.md`.
- Add tests in `tests/test_cli.py` for config normalization and in the relevant
  profile test for mathematical behavior.

Do not hide compatibility quirks behind generic helper names. If a rule exists
only to match ATH, name it as compatibility behavior.

## Source Behavior

Source surfaces are built from the throat ring and assigned physical tag `2`
(`SD1D1001`).

Current behavior:

- `source_shape = 0` builds a flat throat disc/sector source.
- `source_shape = 1` builds a rounded throat cap source.
- Other source shapes fail explicitly before final mesh validation.
- Positive `source_radius_mm` overrides automatic rounded-cap radius.
- `source_curv = -1` flips rounded-cap curvature direction.
- `source_auto_angle_deg` comes from the normalized throat angle.

When changing source behavior:

- Preserve the requirement that final meshes contain physical tag `2`.
- Add source-shape tests for both closed and open domains.
- Verify source normals with `tests/test_orientation_validation.py`.
- Update `docs/config-schema.md` and `docs/builder-invariants.md`.

## Enclosure Plan And Edge Types

`HornEnclosure` records the WG enclosure payload family. Current documented
meanings are:

- `plan_type = 1`: rounded rectangle.
- `plan_type = 2`: ellipse.
- `plan_type = 3`: superellipse.
- `edge_type = 1`: rounded fillet.
- `edge_type = 2`: chamfer.

When adding or changing enclosure topology:

- Keep all enclosure surfaces under physical tag `3` (`SD2G0`).
- Return density role groups for `enclosure`, `enclosure_edges_front`, and
  `enclosure_edges_back` when front/back interpolation applies.
- Return `enclosure_bounds` with `bx0`, `bx1`, `by0`, `by1`, `z_front`, and
  `z_back` when density needs spatial interpolation.
- Test full-domain and supported open-domain cases separately.
- Raise `NotImplementedError` or `ConfigError` for unsupported combinations
  instead of producing approximate geometry.

## Density Rules

Density policy belongs in `density.py`, not builders.

When changing density:

- Keep all density values in millimetres.
- Add or update role names in `docs/builder-invariants.md`.
- Make sure builders return any new role names consistently.
- Update `tests/test_density_contract.py` for formulas, defaults, and fallback
  behavior.
- Check that final unit scaling still happens only after meshing.

Avoid encoding density directly in topology builders unless it is a Gmsh-kernel
stability hint that does not replace the central density policy.

## Physical Tags

Physical tags are solver-facing API:

- `1` / `SD1G0`: rigid wall.
- `2` / `SD1D1001`: primary source.
- `3` / `SD2G0`: enclosure wall.
- `4` / `I1-2`: interface.
- `12` / `mouth_aperture`: infinite-baffle Rayleigh aperture cap.

When changing tags:

- Treat tag number/name changes as breaking changes.
- Update `tags.py`, docs, tests, and downstream consumers together.
- Ensure `load_mesh` reports only used 2D physical groups.
- Ensure `build_mesh` final validation still catches missing rigid wall and
  source tags.

Prefer adding a new explicit tag over reusing an existing tag for a different
acoustic boundary.

## Interfaces And Topology

Interfaces are built from configured point-grid slices and offset forward along
the z axis. They receive physical tag `4` (`I1-2`) and density role
`interface`.

When changing topology:

- Preserve the point-grid shape contract.
- Preserve closed vs open angular wrapping semantics.
- Keep surface splits separate from physical groups. More Gmsh surfaces may be
  fine if the physical groups and density roles remain correct.
- Update `tests/test_point_grid_contract.py` for observable topology behavior.
- Update `docs/geometry-contract.md` for mathematical/topological rules and
  `docs/builder-invariants.md` for handoff details.

## Config Import

The ATH text parser is an adapter. It should map supported ATH/WG names into
the same config sections used by TOML/JSON, then stop.

When changing import:

- Keep unknown text sections from implying unsupported geometry.
- Add parser tests for aliases and defaults.
- Keep imported ATH defaults explicit, especially sampling defaults.
- Do not make production geometry helpers depend on raw ATH key names.

## Public API

The public API is the package root plus documented CLI behavior. Before adding
new exports:

- Decide whether the symbol is stable enough for external callers.
- Prefer dataclasses and high-level build functions over builder internals.
- Document the export in `docs/public-api.md`.
- Add a small import/API smoke test when appropriate.

Underscored helpers exported from `profiles.py` are compatibility/test helpers,
not an invitation to build new integrations on them.
