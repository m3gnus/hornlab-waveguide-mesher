"""Import-only lip and rear rules measured against ATH V2026-09a output.

The authored grid is the authority: its chord angles, not analytic derivatives
or a refined sampling grid, determine both the lip and the rear rings.
"""
from dataclasses import dataclass
import math
import numpy as np

from .geometry import PointGridHornGeometry


def validate_combinations(params):
    """Refuse compositions awaiting measured ATH lip/rear witnesses."""
    from .text_import import TEXT_IMPORT_VERSION_KEY, TEXT_IMPORT_VERSION
    if params.get(TEXT_IMPORT_VERSION_KEY) != TEXT_IMPORT_VERSION:
        raise ValueError("Mesh.Roundover / Mesh.RearShape = 3 require ath-2026-09a-v2")
    for key, name, inactive in (("scale", "Scale", 1),
                               ("throatExtLength", "Throat.Ext.Length", 0),
                               ("throatExtAngle", "Throat.Ext.Angle", 0),
                               ("rot", "Rot", 0),
                               ("verticalOffset", "Mesh.VerticalOffset", 0)):
        value = params.get(key, inactive)
        if not isinstance(value, (int, float)) or value != inactive:
            raise ValueError(f"{name} with Mesh.Roundover / Mesh.RearShape = 3 is unsupported: awaiting ATH probes")
    if params.get("sourceShape", 1) == 2:
        raise ValueError("Source.Shape = 2 with Mesh.Roundover / Mesh.RearShape = 3 is unsupported: awaiting ATH probes")


def meridional_normals(points):
    """Profile-plane chord normals, with one-sided body endpoint chords."""
    grid = np.asarray(points, dtype=float)
    tangent = np.empty_like(grid)
    tangent[:, 1:-1] = grid[:, 2:] - grid[:, :-2]
    tangent[:, 0] = grid[:, 1] - grid[:, 0]
    tangent[:, -1] = grid[:, -1] - grid[:, -2]
    radius = np.linalg.norm(grid[..., :2], axis=-1)
    if np.any(radius <= 0):
        raise ValueError("ATH import meridional normals require positive profile radii")
    radial_unit = grid[..., :2] / radius[..., None]
    dr = np.sum(tangent[..., :2] * radial_unit, axis=-1)
    dz = tangent[..., 2]
    length = np.hypot(dr, dz)
    if np.any(length <= 0):
        raise ValueError("ATH import meridional normals require nonzero profile chords")
    return np.concatenate((radial_unit * (dz / length)[..., None],
                           (-dr / length)[..., None]), axis=-1)


def append_lip(body, radius, segments, *, profile_zero=None):
    """Append uniformly scaled copies of the original, post-stretch mouth."""
    grid = np.asarray(body, dtype=float)
    if grid.shape[1] < 4:
        raise ValueError("Mesh.Roundover requires Mesh.LengthSegments >= 3 for the mouth chord angle")
    p0 = grid[0] if profile_zero is None else np.asarray(profile_zero)
    g1 = math.atan2(p0[-1, 2] - p0[-3, 2], p0[-1, 0] - p0[-3, 0])
    g2 = math.atan2(p0[-2, 2] - p0[-4, 2], p0[-2, 0] - p0[-4, 0])
    angle = 2 * g1 - g2
    theta = np.arange(1, segments + 1) / segments * (math.pi / 2 - angle) - angle
    scale = 1 + radius * np.sin(theta) / p0[-1, 0]
    lip = np.repeat(grid[:, -1:, :], segments, axis=1)
    lip[..., :2] *= scale[None, :, None]
    lip[..., 2] -= radius * (1 - np.cos(theta))[None, :]
    phi = np.arctan2(lip[..., 1], lip[..., 0])
    normals = np.stack((-np.sin(theta)[None, :] * np.cos(phi),
                        -np.sin(theta)[None, :] * np.sin(phi),
                        np.broadcast_to(-np.cos(theta), phi.shape)), axis=-1)
    return np.concatenate((grid, lip), axis=1), np.concatenate(
        (meridional_normals(grid), normals), axis=1), angle


def rear_rings(grid, normals, shape, wall, front_offset=40, step=25, *, profile_zero_z=None, closed=False):
    """Throat, selected offset rings and (for Opus) unoffset front ring."""
    z = grid[0, :, 2] if profile_zero_z is None else profile_zero_z
    if shape == 3:
        if front_offset < 0:
            raise ValueError("Mesh.RearShape.FrontOffset < 0 is unsupported: negative offsets have no qualified ATH ring fit")
        candidates = np.flatnonzero(z[:-1] <= z[-1] - front_offset)
        if not len(candidates):
            raise ValueError("Mesh.RearShape.FrontOffset places the Opus front behind every body row; no rear loft can be built")
        end = int(candidates[-1])
        shift = -10
    else:
        end, step, shift = len(z) - 1, 5, -5
    selected, previous = [], 0.0
    for i in range(1, end + 1):
        if i == end or z[i] < previous or z[i] - previous > step:
            selected.append(i)
            previous = z[i]
    throat = grid[:, 0].copy()
    throat[:, 2] += shift
    rings = [throat + wall * normals[:, 0]]
    rings += [grid[:, i] + wall * normals[:, i] for i in selected]
    if shape == 3:
        front = grid[:, -1].copy()
        front[:, 2] -= front_offset
        rings.append(front)
    rings = np.stack(rings, axis=1)
    # ATH includes the repeated first endpoint when exporting a full ring.
    last_endpoint = rings[0, 0, 2] if closed else rings[-1, 0, 2]
    apex = np.array([0, 0, .5 * (rings[0, 0, 2] + last_endpoint)])
    return rings, apex, selected


@dataclass(frozen=True)
class AthImportedSurfaceGeometry(PointGridHornGeometry):
    """An authored inner grid with an independently selected ATH rear grid."""
    ath_rear_points: object = None
    ath_rear_apex: object = None
    ath_normals: object = None
    ath_body_rows: int = 0
    ath_lip_radius_mm: float = 0
    ath_rear_shape: int = 1
    ath_rear_resolution_mm: float = 15
    ath_mouth_resolution_mm: float = 8


def resolve(config, params, formula, mode, allow_large_mesh):
    from .config_builder import (ConfigError, ResolvedGeometry, _mesh_density_from_config,
                                 _mesh_topology_mode, _mesh_surface_fit, _section, _bool,
                                 _native_symmetry_plane_for_quadrants)
    from .profile_sampling import build_point_grid_arrays
    from .profiles import azimuthal_mean
    if mode not in {"freestanding", "bare"}:
        raise ConfigError("Mesh.Roundover/RearShape = 3 require free-standing or bare import topology; enclosure/infinite baffle are unsupported")
    if params.get("athRoundover") and formula != "OSSE":
        raise ConfigError("Mesh.Roundover with R-OSSE is unsupported: ATH folds the lip back")
    if params.get("athRearShape") == 3 and mode != "freestanding":
        raise ConfigError("Mesh.RearShape = 3 requires a free-standing wall")
    if params.get("subdomainSlices") or params.get("interfaceOffset"):
        raise ConfigError("Mesh.Roundover/RearShape = 3 with Mesh.SubdomainSlices/Mesh.InterfaceOffset is unsupported: free-standing subdomain interfaces are not implemented")
    wall = float(params["wallThickness"])
    lip = params.get("athRoundover")
    if lip and mode == "freestanding" and params.get("athRearShape", 1) == 1 and wall >= lip["radius"]:
        raise ConfigError("Mesh.Roundover with RearShape = 1 requires Radius > Mesh.WallThickness: the outer lip would collapse or fold")
    try:
        data = build_point_grid_arrays(params)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    inner, normals = data["inner_grid"], data["ath_normals"]
    outer, apex, selected = None, None, []
    if mode == "freestanding" and wall > 0:
        try:
            outer, apex, selected = rear_rings(inner, normals, params.get("athRearShape", 1), wall,
                params.get("athRearFrontOffset", 40), params.get("athRearStep", 25),
                profile_zero_z=data.get("ath_profile_zero_z"), closed=data["full_circle"])
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
    mesh = _section(config, "mesh")
    if _mesh_surface_fit(mesh) == "approximate":
        raise ConfigError("Mesh.SurfaceFit = approximate is unsupported for Mesh.Roundover / Mesh.RearShape = 3 imports: the authored ATH rings require interpolation")
    density = _mesh_density_from_config(config, allow_large_mesh=allow_large_mesh)
    geometry = AthImportedSurfaceGeometry(
        inner_points=inner, outer_points=data["outer_grid"],
        ath_rear_points=outer, ath_rear_apex=apex, ath_normals=normals,
        ath_body_rows=data["ath_body_rows"], ath_lip_radius_mm=0 if lip is None else lip["radius"],
        ath_rear_shape=params.get("athRearShape", 1),
        ath_rear_resolution_mm=density.rear_res_mm, ath_mouth_resolution_mm=density.mouth_res_mm,
        topology_mode=_mesh_topology_mode(mesh), surface_fit="interpolate", wall_thickness_mm=wall,
        closed=data["full_circle"], symmetry_planes=tuple(data["symmetry_planes"]),
        vertical_offset_mm=data["vertical_offset_mm"],
        source_shape=int(params.get("sourceShape", 1)), source_radius_mm=float(params.get("sourceRadius", -1)),
        source_curv=int(params.get("sourceCurv", 0)), source_auto_angle_deg=azimuthal_mean(params.get("a0"), 0),
        preserve_grid=_bool(mesh, names=("preserve_grid", "preserveGrid"), default=False),
    )
    return ResolvedGeometry(geometry, density,
        formula, mode, data["quadrants"], _native_symmetry_plane_for_quadrants(data["quadrants"]),
        _bool(mesh, names=("scale_to_metres", "scaleToMetres"), default=True),
        {"athImportSurfaces": {"bodyRows": data["ath_body_rows"], "rearSelectedRows": selected,
                               "mouthAngleDeg": math.degrees(data.get("ath_mouth_angle", 0)),
                               "samplingAuthority": "authored-grid"}})


def normalized_controls(mesh, wall):
    """Validate imported dictionary transport as strictly as the CFG reader."""
    controls = {key: mesh[key] for key in ("athRoundover", "athRearShape", "athRearFrontOffset", "athRearStep")
                if key in mesh}
    shape = controls.get("athRearShape", 1)
    if shape not in {1, 3}:
        raise ValueError("Mesh.RearShape must be 1 or 3")
    lip = controls.get("athRoundover")
    if lip is not None:
        if not isinstance(lip, dict) or set(lip) - {"radius", "segments"}:
            raise ValueError("Mesh.Roundover transport requires only Radius/Segments members")
        try:
            radius = float(lip.get("radius", float("nan")))
            segments = float(lip.get("segments", 5))
        except (ValueError, TypeError) as exc:
            raise ValueError("Mesh.Roundover.Radius/Segments must be finite numbers") from exc
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("Mesh.Roundover.Radius must be finite and > 0")
        if not math.isfinite(segments) or segments < 1 or segments != int(segments):
            raise ValueError("Mesh.Roundover.Segments must be a finite integer >= 1")
        controls["athRoundover"] = {"radius": radius, "segments": int(segments)}
    if shape != 3 and any(key in controls for key in ("athRearFrontOffset", "athRearStep")):
        raise ValueError("Mesh.RearShape.FrontOffset/Step require Mesh.RearShape = 3")
    if shape == 3:
        if not math.isfinite(wall):
            raise ValueError("Mesh.WallThickness for Mesh.RearShape = 3 must be finite")
        wall = min(1000, max(.1, wall))
        for key, name, default in (("athRearFrontOffset", "FrontOffset", 40), ("athRearStep", "Step", 25)):
            try:
                value = float(controls.get(key, default))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Mesh.RearShape.{name} must be finite") from exc
            if not math.isfinite(value) or (name == "Step" and value <= 0):
                raise ValueError(f"Mesh.RearShape.{name} must be finite" + (" and > 0" if name == "Step" else ""))
            if name == "FrontOffset" and value < 0:
                raise ValueError("Mesh.RearShape.FrontOffset < 0 is unsupported: negative offsets have no qualified ATH ring fit")
            controls[key] = value
    return controls, wall
