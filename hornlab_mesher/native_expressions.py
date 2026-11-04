"""Per-azimuth transport of the existing native OSSE meridian constructions.

Only previously refused expression bodies enter here. Literal recipes retain
their analytic revolution builders, fingerprints and sampling byte for byte.
The surface uses the same interpolating point-grid owner as ordinary OSSE;
each construction span is kept separate so joins and lip rims cannot disappear.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .config_parser import ConfigError, _eval_constant_param
from .geometry import PointGridHornGeometry
from .native_boundary import NATIVE_MARKERS
from .profile_common import eval_param

BODY_FIELDS = frozenset({"L_mm", "r0_mm", "a_deg", "a0_deg", "k", "L", "r0", "a", "a0"})


def _fail(message):
    raise ConfigError("Native OSSE expression refused: " + message)


def expression_body(config):
    return any(isinstance(section.get(key), str) for section in _body_sections(config) for key in BODY_FIELDS)


def _body_sections(config):
    return [config]+[config[key] for key in ("profile", "parameters") if isinstance(config.get(key), Mapping)]


def scalar_config(config, phi):
    """Evaluate only shared body dimensions, through ordinary OSSE's evaluator."""
    clean = copy.deepcopy(dict(config))
    for section in _body_sections(clean):
        for key in BODY_FIELDS.intersection(section):
            value = section[key]
            if isinstance(value, str):
                try:
                    section[key] = eval_param(value, phi)
                except (ValueError, TypeError, ArithmeticError) as exc:
                    _fail(f"{key} failed at azimuth {math.degrees(phi):g} degrees ({exc}). "
                          "Edit the body expression so it is finite and valid at every azimuth.")
    return clean


def _sampling(config):
    mesh = config.get("mesh", {})
    counts = []
    for key, default, minimum in (("angular_segments", 64, 16), ("length_segments", 32, 8)):
        value = mesh.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not float(value).is_integer():
            _fail(f"{key} must be a positive whole number. Enter a whole number in the mesh settings.")
        if not 0 < value <= 1024:
            _fail(f"{key} must be from 1 to 1024 for expression bodies. Reduce the surface sample count.")
        counts.append(max(minimum, int(value)))
    # Keep cardinal meridians in the shared grid and all samples periodic.
    counts[0] = 4*math.ceil(counts[0]/4)
    if counts[0]*(counts[1]+1) > 150_000:
        _fail("the surface sample budget was exceeded. Reduce angular or length samples.")
    return counts


def scalar_params(config, phi, resolver):
    try:
        return resolver(scalar_config(config, phi))
    except ValueError as exc:
        _fail(f"the body is invalid at azimuth {math.degrees(phi):g} degrees ({exc}). "
              "Edit the body expression or feature controls to stay inside the native construction domain at every azimuth.")


def configuration(config, resolver):
    from .native_boundary import validate_native_boundary
    marker = validate_native_boundary(config)
    if marker not in NATIVE_MARKERS-{"SOURCE-DISK"} or not expression_body(config):
        return None
    # A legacy alias can carry ignored text behind a supplied canonical value.
    # Never change an already buildable recipe or evaluate its dormant text.
    try:
        return resolver(config)
    except ValueError:
        pass
    # Constant expressions share the literal construction. This is the same
    # dependency check and evaluator used by ATH's constant axial controls.
    config = copy.deepcopy(dict(config))
    for section in _body_sections(config):
        for key in BODY_FIELDS.intersection(section):
            if isinstance(section[key], str):
                try:
                    section[key] = _eval_constant_param(section[key])
                except (ValueError, SyntaxError, ArithmeticError):
                    pass
    if not expression_body(config):
        return resolver(config)
    nphi, _ = _sampling(config)
    mode = None
    for phi in np.arange(nphi)*2*math.pi/nphi:
        _, _, mode = scalar_params(config, float(phi), resolver)
    return {"nativeExpressionConfig": copy.deepcopy(dict(config)), "type": marker}, marker, mode


def meridian_branches(params, intervals, *, mesh_caps=None):
    """Sample owned curves; this adapter contains no profile equations."""
    unit = np.linspace(0, 1, intervals+1)
    branches = []
    if "controlsAdapter" in params:
        from .adapter_controls import ControlsMeridian
        model = ControlsMeridian.from_params(params)
        return [("horn.inner", model.cubic(unit)[0]), ("horn.inner", model.body(unit)[0])]
    if "terminatingArc" in params:
        from .composite_meridian import CompositeMeridian
        model = CompositeMeridian.from_params(params)
        if model.throat is not None:
            branches.append(("horn.inner", model.throat.cubic_values(unit)[0]))
        branches.extend([
            ("horn.inner", model.body(np.linspace(model.conic_start_z, model.join_z, intervals+1))[0]),
            ("horn.inner", model.arc(np.linspace(model.beta, model.end, intervals+1))),
        ])
        return branches
    if "mouthRoundoverRadiusMm" in params:
        from .builders.mouth_roundover import roundover_mesh_size
        from .mouth_roundover import Roundover
        model = Roundover.from_params(params)
        if mesh_caps is not None:
            mesh_caps.append(float(roundover_mesh_size(model)))
        for outer in (False, True):
            role = "horn.outer" if outer else "horn.inner"
            if model.throat is not None:
                branches.append((role, model.cubic(unit, outer)[0]))
            branches.append((role, model.body(np.linspace(model.start, model.length, intervals+1), outer)[0]))
            branches.append((role, model.arc(unit*model.sweep, outer)))
        inner, outer = model.arc(model.sweep), model.arc(model.sweep, True)
        branches.append(("mouth_rim", np.array([inner, outer])))
        rear = model.rear_start
        branches.append(("wall.rear_return", np.array([[-model.wall, rear[1]], rear])))
        return branches
    if "adapterAxialScale" in params or params.get("throat_adapter"):
        if "adapterAxialScale" in params:
            from .adapter_axial import PhysicalAdapter
            model = PhysicalAdapter.from_params(params)
            return [("horn.inner", model.cubic_values(unit)[0]),
                    ("horn.inner", model.body_values(model.payload["join_t"]+unit*(1-model.payload["join_t"]))[0])]
        from .throat_adapter import resolve_adapter
        model = resolve_adapter(params)
        scale = float(params["scale"])
        for stations in (unit/2, .5+unit/2):
            z, r = model.evaluate(stations)
            branches.append(("horn.inner", np.column_stack((z, r))*scale))
        return branches
    from .axial_scale import AxialModel
    model = AxialModel.from_params(params)
    return [("horn.inner", model.body(unit*model.length)[0])]


@dataclass(frozen=True)
class ExpressionSurface:
    config: dict
    branches: tuple
    lip_size_mm: float | None = None

    @classmethod
    def from_config(cls, config, *, sampling=None):
        from .config_builder import build_geometry_params
        nphi, intervals = _sampling(config) if sampling is None else sampling
        if nphi*(intervals+1) > 150_000:
            _fail("the surface sample budget was exceeded. Reduce angular or length samples.")
        grids = None
        roles = None
        mesh_caps = []
        for i, phi in enumerate(np.arange(nphi)*2*math.pi/nphi):
            params, _, _ = scalar_params(config, float(phi), build_geometry_params)
            spans = meridian_branches(params, intervals, mesh_caps=mesh_caps)
            if grids is None:
                roles = [role for role, _ in spans]
                grids = [np.empty((nphi, len(zr), 3)) for _, zr in spans]
            for grid, (_, zr) in zip(grids, spans):
                grid[i] = np.column_stack((zr[:, 1]*math.cos(phi), zr[:, 1]*math.sin(phi), zr[:, 0]))
        branches = tuple((role, np.frombuffer(grid.tobytes(), dtype=np.float64).reshape(grid.shape))
                         for role, grid in zip(roles, grids))
        return cls(copy.deepcopy(dict(config)), branches, min(mesh_caps) if mesh_caps else None)

    def grid(self, role):
        parts = [grid for name, grid in self.branches if name == role]
        return np.concatenate([parts[0]]+[part[:, 1:] for part in parts[1:]], axis=1) if parts else None

    @property
    def offset(self):
        return self.config.get("mesh", {}).get("vertical_offset_mm", self.config.get("vertical_offset_mm", 0))

    def dimensions(self):
        points = np.concatenate([grid.reshape(-1, 3) for _, grid in self.branches])
        inner = self.grid("horn.inner")
        return {"source_diameter": np.ptp(inner[:, 0, :2], axis=0).tolist(),
                "mouth_opening": np.ptp(inner[:, -1, :2], axis=0).tolist(),
                "horn_overall": np.ptp(points, axis=0).tolist()}

    @property
    def identity(self):
        config = {key: value for key, value in self.config.items() if key not in {"output", "path", "output_path"}}
        return {"contractRevision": 1, "construction": "native-osse-meridian-grid-v1", "config": config}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ExpressionGeometry(PointGridHornGeometry):
    expression_surface: ExpressionSurface | None = None


def resolve(config, params, allow_large_mesh=None):
    from .config_builder import ResolvedGeometry, _mesh_density_from_config
    surface = ExpressionSurface.from_config(params["nativeExpressionConfig"])
    mesh = config.get("mesh", {})
    geometry = ExpressionGeometry(inner_points=surface.grid("horn.inner"), outer_points=surface.grid("horn.outer"),
        wall_thickness_mm=mesh.get("wall_thickness_mm", 0), source_shape=0,
        source_radius_mm=-1, source_curv=0, closed=True, symmetry_planes=(),
        vertical_offset_mm=surface.offset, expression_surface=surface)
    return ResolvedGeometry(geometry, _mesh_density_from_config(config, allow_large_mesh=allow_large_mesh),
        params["type"], "freestanding" if geometry.outer_points is not None else "bare", "1234", None,
        mesh.get("scale_to_metres", True), {"nativeExpressionBody": True})
