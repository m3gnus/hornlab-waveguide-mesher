These numeric fixtures transcribe the published degree-mode Desmos expressions:

- https://www.desmos.com/calculator/sljsrsipbq
- https://www.desmos.com/calculator/5jogzxvqvb
- https://www.desmos.com/calculator/k0akvpmsnn

Each CSV contains 401 uniformly spaced stations, including both endpoints:
R-OSSE t=0..1 and OS-SE t=0..160 mm. The first column is the unstretched
parameter, followed by stretched x and unchanged radius, in millimetres.
`references.json` records the exact graph parameters. In particular the second
R-OSSE graph uses r0=8.5; the forum's executable example uses r0=8.25. They are
separate references, not rounded versions of one input.

Values were computed with an independent transcription of the graph equations,
using double precision and atan * 180/pi. Neither mesher evaluator was used to
produce them. The point comparison tolerance is 1e-10 mm. Independent review
re-evaluated the equations and measured a maximum discrepancy of 5.69e-14 mm.

`disabled-corpus.json` contains seven configurations spanning both families,
OSSE prefixes/morph, freestanding walls, bare horns, infinite baffle, and
an enclosure. Tests compare omitted/zero coefficients' raw MSH bytes and
inner/outer grids within one run, so Gmsh/platform differences do not turn
historical golden hashes into a cross-platform requirement. Pre-implementation
and post-implementation hashes at base 5c8ea4dc are retained as local producer
evidence and listed in the handoff report.

These Desmos fixtures are mathematical references. Executable ATH V2025-12
exports and their provenance are stored separately in `ath-v2025-12/`;
existing archive parity remains a required final gate.

`base-compatibility.json` was captured by executing base `5c8ea4dc` from a
read-only Git archive extraction, with Python 3.13 (64-bit). It records default
and prefixed dataclass serialization, repr, hashes and builder mappings;
native and text-imported normalization; full resolved asdict values (array byte
hashes), fields, instance keys and repr for OSSE, R-OSSE, FREEFORM and ICW;
and 23 archive imports plus 36
block/flat rotation/slot variants, eight zero-expression imports and 70 OSSE
block composition imports covering numeric slots, expression slots/rotations,
finite-sample aliases, narrow spikes and scaled zero identities. Inactive
imports equal base except top-level Rot beside a block without its own Rot. Archive inputs are identified by content
SHA256, so no local paths or archive comments are embedded. The required
parity gate also runs this entire corpus against the live archive.

To recapture: extract `git archive 5c8ea4dc` into a scratch directory, mark
that extraction read-only, then run `scripts/capture_throat_stretch_base.py`
with arguments `<extracted-base>` and `<output-json>`, using the test venv
with `PYTHONDONTWRITEBYTECODE=1` and `ATH_REFERENCE_ROOT` configured. The script
asserts that imports resolve into the extracted base. Do not generate these
expected values with the feature implementation.

`base-resolved-arrays.npz` preserves the numeric arrays from the verified base
extraction used by `base-compatibility.json`. Each array matches that capture's
shape, dtype and SHA-256 exactly. Captured-base comparisons use `rtol=1e-12`
for floats, with `atol=1e-12` only for expected zeros (in the field's unit).
Structure and non-float fields remain exact; same-platform inactive geometry
and identity comparisons remain byte-exact.
