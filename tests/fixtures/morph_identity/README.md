# Same-runtime main export identity

The regression test extracts pinned main
`a7888d701f0c4394b555e69236f844fe2560dc6b` with `git archive` once per
test session. For every config it exports main and the working tree in
separate fresh subprocesses, using the same Python executable and installed
dependencies. Each child asserts its package root; `-I` isolates inherited
Python path settings. Complete solve bytes and timestamp-normalized STEP
bytes must match. No geometry tolerance or runtime-fingerprint exemption
is used.

The 32 original coarse configs come from `cases()` in
`scripts/capture_morph_identity.py`. The test also exports both revisions
for 16 explicit `morphTarget=0` configs: 48 solve and 48 STEP comparisons.
Reduced-sector STEP refusal is checked before rebuilding the full STEP.

CI uses `actions/checkout` with `fetch-depth: 0` on all four matrix legs
and sets `HORNLAB_MORPH_IDENTITY=required`, so missing Git or the pinned
commit fails the job. Outside CI, missing Git/history explicitly skips the
guard. No fetch or Git metadata write occurs in the test; offline users
need the pinned object already present to execute it. The required-ATH
full suite also collects and executes this guard whenever Git/history
are available. Run it directly with:

```sh
python -m pytest tests/test_morph_mouth_fidelity.py -k preserve_main_output_bytes -q
```

## Historical capture

`baseline.json` stores SHA-256 hashes of complete solve files (mm) and STEP
files from main `a7888d701f0c4394b555e69236f844fe2560dc6b`. Only the OCC
`FILE_NAME` timestamp is normalized. The manifest records every config and
the capture runtime. These recorded hashes are historical round-4 evidence,
not cross-platform test expectations: native exporter version/platform
changes may alter them. The regression never reads these recorded hashes.

Coverage: OSSE, R-OSSE, FREEFORM and ICW, each with full `1234`, half `12`
and `14`, and quarter `1`, both unmorphed and circular-target mouths.

The public STEP exporter refuses reduced sectors. For each half/quarter
config the capture verifies and records that refusal, then rebuilds with
`quadrants=1234` as instructed by the exporter and hashes the full STEP.
Thus all 32 solve files are covered; STEP covers the corresponding full
designs and the sector refusal contract, not unsupported sector STEP files.

Regenerate from the repository with its own dev venv:

```sh
python scripts/capture_morph_identity.py --revision a7888d701f0c4394b555e69236f844fe2560dc6b
```

The script refuses branch names and every other revision, verifies the
fixed commit is an ancestor of main, and uses a clean `git archive`
extraction. Every export runs in a fresh process whose package import path
is asserted inside that extraction, resetting OCC product-name counters.
It never exports from the working branch. Hashes keep fixtures small while
guarding all emitted bytes. Capture is normally under two minutes; submit
through the compute broker if it exceeds that estimate on your machine.
