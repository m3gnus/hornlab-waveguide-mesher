# Frozen main export identity

`baseline.json` stores SHA-256 hashes of complete solve files (mm) and STEP
files from main `a7888d701f0c4394b555e69236f844fe2560dc6b`. Only the OCC
`FILE_NAME` timestamp is normalized. The manifest records every config and
the capture runtime. Native exporter version/platform changes may alter
these bytes; investigate a mismatch before considering a new baseline.

Coverage: OSSE, R-OSSE, FREEFORM and ICW, each with full `1234`, half `12`
and `14`, and quarter `1`, both unmorphed and circular-target mouths. Tests
also compare explicit `morphTarget=0` against the unmorphed baseline.
The baseline is independent of the implementation under test.

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
