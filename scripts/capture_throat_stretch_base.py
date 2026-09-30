"""Capture C4 compatibility expectations from a read-only base extraction.

Usage: python scripts/capture_throat_stretch_base.py <base-dir> <output-json>
Requires ATH_REFERENCE_ROOT and PYTHONDONTWRITEBYTECODE=1.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
from dataclasses import asdict, fields

base_dir = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(base_dir))
from hornlab_mesher.geometry import OsseHornGeometry, RosseHornGeometry
from hornlab_mesher.config_builder import build_geometry_params, resolve_geometry
from hornlab_mesher.config_parser import parse_text_config
from hornlab_mesher.builders.osse_waveguide import _osse_params
from hornlab_mesher.builders.rosse_waveguide import _rosse_params
from hornlab_mesher.profile_formulas import _icw_cache_key

assert base_dir in Path(sys.modules['hornlab_mesher'].__file__).parents
assert 's1' not in OsseHornGeometry.__dataclass_fields__, 'use the unstretched base'
assert sys.dont_write_bytecode, 'use PYTHONDONTWRITEBYTECODE=1'
out = {'base': '5c8ea4dc', 'identities': {}, 'imports': []}
for family, cls, adapter in [
    ('OSSE', OsseHornGeometry, _osse_params),
    ('R-OSSE', RosseHornGeometry, _rosse_params),
]:
    for kwargs in [{}, {'k': 1.38, 'throat_ext_length_mm': 20, 'throat_ext_angle_deg': -20}]:
        geometry = cls(**kwargs)
        name = family + ('-default' if not kwargs else '-prefix')
        out['identities'][name] = {
            'family': family, 'kwargs': kwargs, 'hash': hash(geometry),
            'asdict': asdict(geometry), 'repr': repr(geometry), 'adapter': adapter(geometry),
        }
    native = {'profile': {'formula': family, 'k': 1.38}}
    dimension = 'L = 160' if family == 'OSSE' else 'R = 200'
    text = f'{family} = {{\n{dimension}\nk = 1.38\n}}'
    out['identities'][family + '-config'] = {
        'native': native, 'text': text,
        'native_params': build_geometry_params(native)[0],
        'text_params': build_geometry_params(parse_text_config(text))[0],
    }
for i, path in enumerate(sorted(Path(os.environ['ATH_REFERENCE_ROOT']).glob('*/config.txt'))):
    text = path.read_text()
    config = parse_text_config(text)
    out['imports'].append({
        'id': f'archive-{i + 1:02}',
        'sha256': hashlib.sha256(text.encode()).hexdigest(),
        'parsed': config, 'params': build_geometry_params(config)[0],
    })
assert len(out['imports']) == 23, 'required archive corpus changed'
for family in ['OSSE', 'R-OSSE']:
    for block in [False, True]:
        for rot in ['', '\nRot = 10']:
            for slot in ['', '\nSlot.Length = 0', '\nSlot.Length = 8']:
                for in_rot in [False, True]:
                    if in_rot and not block:
                        continue
                    if block:
                        dimension = 'L = 160' if family == 'OSSE' else 'R = 200'
                        block_rot = '\nRot = 5' if in_rot else ''
                        text = f'{family} = {{\n{dimension}{block_rot}\n}}'
                    elif family == 'OSSE':
                        text = 'Length = 160\nCoverage.Angle = 40'
                    else:
                        text = 'R-OSSE = {\nR = 200\n}'
                    text += rot + slot
                    try:
                        config = parse_text_config(text)
                        result = {'parsed': config, 'params': build_geometry_params(config)[0]}
                    except Exception as exc:
                        result = {'error': type(exc).__name__, 'message': str(exc)}
                    out['imports'].append({
                        'id': f'variant-{len(out["imports"]) - 22:02}',
                        'text': text, **result,
                    })
out['icw_disabled_key'] = _icw_cache_key({'icw_seed': {'type': 'OSSE', 'L': 160}})
# Full resolved schema/value oracles for every C4 family, including native
# families that cannot receive stretch. Hash arrays without lossy rounding.
import numpy as np


def resolved_value(value):
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {key: resolved_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [resolved_value(item) for item in value]
    return value


out['zero_imports'] = []
for expression in ['sin(p-p)', '0/(1+p^2)', '(p+1)-(1+p)', 'cos(p)^2+sin(p)^2-1']:
    for scale in ['', '\nScale = .48']:
        text = 'OSSE = {\nL = 160\n}\nSlot.Length = ' + expression + scale
        out['zero_imports'].append({'text': text, 'parsed': parse_text_config(text)})
# Preserve the inactive importer, even for slots/rotations whose expressions
# alias finite azimuth samples or have floating-point identity residuals.
out['composition_imports'] = []
for slot in ['0', '8', '1e-13', '0*p', 'sin(720*p)', 'sin(720*p)^2',
             '20*sin(720*p)^2', '20 if abs(p-pi/64)<0.001 else 0',
             '10000*(cos(p)^2+sin(p)^2-1)', '1/0']:
    for rot, placement in [('0', 'block'), ('10', 'block'), ('0*p', 'block'),
                           ('20*sin(720*p)^2', 'block'),
                           ('10000*(cos(p)^2+sin(p)^2-1)', 'block'),
                           ('10', 'top'), ('20*sin(720*p)^2', 'top')]:
        text = 'OSSE = {\nL = 160'
        if placement == 'block':
            text += '\nRot = ' + rot
        text += '\n}\nSlot.Length = ' + slot
        if placement == 'top':
            text += '\nRot = ' + rot
        case = {'text': text, 'parsed': parse_text_config(text)}
        if placement == 'top':
            case['top_level_rot'] = float(rot) if rot == '10' else rot
        out['composition_imports'].append(case)
out['resolved'] = {}
for family, profile in {
    'OSSE': {'L': 80},
    'R-OSSE': {'R': 80},
    'FREEFORM': {
        'profileH': {'points': [[0, 12.7], [40, 40], [80, 75]],
                     'throatAngleDeg': 15.5, 'mouthAngleDeg': 45},
        'profileV': {'points': [[0, 12.7], [40, 35], [80, 60]],
                     'throatAngleDeg': 15.5, 'mouthAngleDeg': 35}},
    'ICW': {'r0': 12.7, 'a0': 15.5, 'icw_coeffs': [0, 0, 0, 0, 0, 0], 'icw_S': 80},
}.items():
    config = {'mode': 'bare', 'profile': {'formula': family, **profile},
              'mesh': {'angularSegments': 16, 'lengthSegments': 12,
                       'surface_fit': 'approximate' if family == 'FREEFORM' else 'interpolate'}}
    resolved = resolve_geometry(config)
    out['resolved'][family] = {
        'config': config, 'asdict': resolved_value(asdict(resolved)),
        'geometry_fields': [f.name for f in fields(resolved.geometry)],
        'geometry_state_keys': sorted(vars(resolved.geometry)),
        'repr': repr(resolved),
    }
Path(sys.argv[2]).write_text(json.dumps(out, indent=2) + '\n')
