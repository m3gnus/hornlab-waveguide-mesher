"""Freeze export hashes from the exact accepted main baseline, never a branch."""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from importlib.metadata import version
from pathlib import Path

BASE = 'a7888d701f0c4394b555e69236f844fe2560dc6b'
ROOT = Path(__file__).resolve().parents[1]


class BaselineUnavailable(RuntimeError):
    """Git or the pinned baseline object is unavailable in this checkout."""


def extract_baseline(source):
    """Extract the literal accepted main commit without modifying Git metadata.

    CI checks out full history. Offline/shallow developer checkouts may lack
    the object; callers must report that rather than compare HEAD with itself.
    """
    command = ['git', '-C', str(ROOT)]
    try:
        subprocess.check_output(command + ['cat-file', '-e', BASE + '^{commit}'],
                                stderr=subprocess.PIPE, timeout=30)
    except FileNotFoundError as exc:
        raise BaselineUnavailable('Git is unavailable; cannot extract pinned main ' + BASE) from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise BaselineUnavailable('Pinned main ' + BASE + ' is unavailable; '
                                  'use a checkout with full history (offline checkouts may skip)') from exc
    archive = subprocess.check_output(command + ['archive', BASE], timeout=30)
    source = Path(source)
    source.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        # Python 3.10/3.11 also support these tests. Older tarfile versions
        # lack extraction filters; this archive is from the literal Git SHA.
        options = {'filter': 'data'} if hasattr(tarfile, 'data_filter') else {}
        tar.extractall(source, **options)
    assert not (source / '.git').exists()
    assert (source / 'hornlab_mesher/__init__.py').is_file()
    return source


def cases():
    profiles = {
        'OSSE': {'L': 120, 'r0': 12.7, 'a0': 15.5, 'a': 55, 'k': 1, 'q': .995},
        'R-OSSE': {'R': 110, 'r0': 12.7, 'a0': 12.7, 'a': 40, 'k': 1,
                   'r': .37, 'm': 1.2, 'b': 0, 'q': .995, 'tmax': 1},
        'FREEFORM': {'profileH': {'points': [[0, 12.7], [40, 40], [80, 75]],
                                 'throatAngleDeg': 15.5, 'mouthAngleDeg': 45},
                     'profileV': {'points': [[0, 12.7], [40, 35], [80, 60]],
                                 'throatAngleDeg': 15.5, 'mouthAngleDeg': 35}},
        'ICW': {'L': 120, 'r0': 12.7, 'a0': 18, 'R': 110, 'termination': 'flat_baffle'},
    }
    for formula, profile in profiles.items():
        for quadrants in ['1234', '12', '14', '1']:
            for mouth in ['unmorphed', 'round']:
                config = {'formula': formula, 'mode': 'bare', 'profile': profile,
                          'source': {'source_shape': 0},
                          'mesh': {'quadrants': quadrants, 'allow_large_mesh': True}}
                if mouth == 'round':
                    config['morph'] = {'morphTarget': 2, 'morphWidth': 360,
                                       'morphHeight': 360, 'morphAllowShrinkage': 1}
                yield {'id': f'{formula}-{quadrants}-{mouth}', 'config': config}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True,
                        help='Full literal baseline SHA; branch names are refused')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'tests/fixtures/morph_identity/baseline.json')
    args = parser.parse_args()
    if args.revision != BASE:
        parser.error(f'refusing regeneration from a branch or another revision; use {BASE}')
    # Verify the fixed commit belongs to main, then archive only that commit.
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', BASE, 'main'], check=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix='morph-main-capture-') as scratch:
        source = extract_baseline(Path(scratch) / 'source')
        for case in cases():
            dest = Path(scratch) / case['id']
            subprocess.run([sys.executable, str(ROOT / 'tests/morph_identity_export.py'),
                            str(source), json.dumps(case['config']), str(dest)],
                           cwd=source, check=True)
            rows.append({**case, **json.loads((dest / 'hashes.json').read_text())})
            print(case['id'], 'captured', flush=True)
    manifest = {'baseline_commit': BASE, 'solve_units': 'mm',
                'step_domain': 'full model rebuilt for reduced solve configs; sector refusal captured',
                'step_normalization': 'OCC FILE_NAME timestamp only',
                'runtime': {'python': '.'.join(map(str, sys.version_info[:3])),
                            **{p: version(p) for p in ['gmsh', 'meshio', 'numpy', 'scipy']}},
                'cases': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{len(rows)} baseline solve/STEP pairs captured', flush=True)


if __name__ == '__main__':
    main()
