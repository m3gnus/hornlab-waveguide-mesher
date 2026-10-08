"""Preview of the canonical circular lip, with continuous chord bounds."""
from __future__ import annotations

import math
import numpy as np

from ..mouth_roundover import Roundover
from .contract import PreviewGeometryV1, PreviewSurfaceV1
from .primitives import _grid_indices, _orient_indices_to_normals, _flat_cap, _flat_strip


def _signed_curvatures(meridian_curvature, radial_normal, radius, *, inward):
    """Signed fundamental-form curvatures for the emitted surface normal."""
    meridian = np.asarray(meridian_curvature)*(-1 if inward else 1)
    angular = -np.asarray(radial_normal)/np.asarray(radius)
    principal = np.where(np.abs(meridian) > np.abs(angular), meridian,
        np.where(np.abs(angular) > np.abs(meridian), angular, np.maximum(meridian,angular)))
    return (meridian+angular)/2, principal


def build(params, options):
    model = Roundover.from_params(params)
    if options.lod not in {"coarse", "fine", "inspection"}:
        raise ValueError("lod must be 'coarse', 'fine', or 'inspection'")
    target = 0.008 if options.max_chord_error_mm is None else float(options.max_chord_error_mm)
    normal = 3 if options.max_normal_step_deg is None else float(options.max_normal_step_deg)
    if not math.isfinite(target) or target <= 0 or not math.isfinite(normal) or not 0 < normal <= 180:
        raise ValueError("preview chord and normal tolerances must be finite and positive")
    render_roundoff = math.sqrt(3)*max(abs(v) for row in model.bounds for v in row)*2**-24
    budget = (target-render_roundoff)/2
    if budget <= 0:
        raise ValueError("preview tolerance is below the binary32 placement precision")
    reach = float(model.center[1]+model.radius)
    nphi = max(16, math.ceil(2*math.pi/math.sqrt(8*budget/reach)), math.ceil(720/normal),
               int(options.min_silhouette_segments or 0))
    nphi = 4*math.ceil(nphi/4)
    if nphi > 4096:
        raise ValueError("preview angular sampling exceeds its certified budget")
    az = np.arange(nphi)*2*math.pi/nphi
    surfaces = []
    max_meridian_bound = 0.0
    vertex_count = 0
    vertex_limit = options.max_vertices if options.max_vertices is not None else 1_000_000
    if isinstance(vertex_limit,bool) or not isinstance(vertex_limit,int) or vertex_limit < 8:
        raise ValueError("max_vertices must be an integer of at least 8")

    def reserve(count):
        nonlocal vertex_count
        if vertex_count+count > vertex_limit:
            raise ValueError("preview cannot certify the requested tolerance within the vertex budget")
        vertex_count += count

    def surface(role, meridian, derivative, inward=False, curvature=None):
        reserve(len(meridian)*nphi)
        radial = meridian[:, 1, None]
        points = np.stack((radial*np.cos(az), radial*np.sin(az)+model.offset,
                           np.broadcast_to(meridian[:,0,None], (len(meridian), nphi))), axis=-1)
        unit = derivative/np.linalg.norm(derivative, axis=-1)[:,None]
        nr = -unit[:,0] if inward else unit[:,0]
        nz = unit[:,1] if inward else -unit[:,1]
        normals = np.stack((nr[:,None]*np.cos(az), nr[:,None]*np.sin(az),
                            np.broadcast_to(nz[:,None], (len(meridian),nphi))), axis=-1)
        positions = points.reshape(-1,3)
        flat_normals = normals.reshape(-1,3)
        indices = _grid_indices(len(meridian), nphi, closed_phi=True)
        oriented = _orient_indices_to_normals(role, positions, indices, flat_normals)
        mean = principal = None
        if options.include_curvature and curvature is not None:
            signed_mean, signed_principal = _signed_curvatures(
                curvature[:,None], nr[:,None], radial, inward=inward)
            mean = np.broadcast_to(signed_mean, (len(meridian),nphi)).reshape(-1)
            principal = np.broadcast_to(signed_principal, (len(meridian),nphi)).reshape(-1)
        surfaces.append(PreviewSurfaceV1(role, positions, oriented.indices, flat_normals,
                         "smooth", "analytic-parametric", True, mean, principal))
        return points

    ends = {}
    for outer in (False, True):
        enabled = options.include_outer if outer else options.include_inner
        z = model.stations(4*budget, outer)
        refined = [z[0]]
        def refine_normal(lo,hi):
            _, tangent = model.body(np.array([lo,hi]), outer)
            angle = np.arctan2(tangent[:,1],tangent[:,0])
            if math.degrees(abs(angle[1]-angle[0])) <= normal/2:
                refined.append(hi)
            else:
                midpoint = (lo+hi)/2
                refine_normal(lo,midpoint)
                refine_normal(midpoint,hi)
        for lo,hi in zip(z[:-1],z[1:]):
            refine_normal(lo,hi)
        z = np.asarray(refined)
        meridian, derivative = model.body(z, outer)
        A, B, C = model.coefficients
        v = np.sqrt(A+2*B*z+C*z*z)
        d = (B+C*z)/v
        curvature = (C*A-B*B)/v**3/(1+d*d)**1.5
        if outer:
            curvature = curvature/(1-model.wall*curvature)
        bound = max(model.second_bound(lo,hi,outer)*(hi-lo)**2/8 for lo,hi in zip(z[:-1],z[1:]))
        max_meridian_bound = max(max_meridian_bound,bound)
        if enabled:
            body_points = surface("horn.outer" if outer else "horn.inner", meridian, derivative, not outer, curvature)
            ends[(outer,"throat")] = body_points[0]
        rad = model.radius-model.wall if outer else model.radius
        intervals = max(6, math.ceil(model.sweep/math.sqrt(8*budget/rad)), math.ceil(2*math.degrees(model.sweep)/normal))
        phi = np.unique(np.r_[np.linspace(0,model.sweep,intervals+1), math.pi/2-model.beta])
        lip = model.arc(phi,outer)
        deriv = np.stack((np.cos(model.beta+phi), np.sin(model.beta+phi)),axis=-1)
        max_meridian_bound = max(max_meridian_bound, rad*np.max(np.diff(phi))**2/8)
        if enabled:
            lip_points = surface("horn.outer" if outer else "horn.inner", lip, deriv, not outer,
                                  np.full(len(phi),1/rad))
            ends[(outer,"lip")] = lip_points[-1]
    if options.include_inner and options.include_outer:
        reserve(2*nphi)
        surfaces.append(_flat_strip("mouth_rim",ends[(False,"lip")],ends[(True,"lip")],(0,0,-1),
                                    closed_phi=True, include_curvature=options.include_curvature))
    # The rear disk is the existing closed acoustic back plate. CAD cuts the
    # source membrane through it when opening the material bore.
    if options.include_outer:
        throat = model.body(0,True)[0]
        return_meridian = np.array([[-model.wall,throat[1]],throat])
        surface("wall.rear_return",return_meridian,np.tile([1.,0.],(2,1)),False,np.zeros(2))
    if options.include_rear_cap:
        reserve(nphi+1)
        r = float(model.body(0,True)[0][1])
        ring = np.column_stack((r*np.cos(az),r*np.sin(az)+model.offset,np.full(nphi,-model.wall)))
        surfaces.append(_flat_cap("wall.rear_cap",ring,(0,0,-1),closed_phi=True,
                                  include_curvature=options.include_curvature))
    if options.include_source_cap:
        reserve(nphi+1)
        r = model.r0
        ring = np.column_stack((r*np.cos(az),r*np.sin(az)+model.offset,np.zeros(nphi)))
        surfaces.append(_flat_cap("source_cap",ring,(0,0,1),closed_phi=True,
                                  include_curvature=options.include_curvature))
    combined = []
    for role in dict.fromkeys(s.role for s in surfaces):
        parts = [s for s in surfaces if s.role == role]
        if len(parts) == 1:
            combined.append(parts[0])
            continue
        offsets = np.cumsum([0]+[len(s.positions) for s in parts[:-1]])
        combined.append(PreviewSurfaceV1(role, np.concatenate([s.positions for s in parts]),
            np.concatenate([s.indices+offset for s,offset in zip(parts,offsets)]),
            np.concatenate([s.normals for s in parts]),"smooth","analytic-parametric",True,
            None if parts[0].curvature_mean is None else np.concatenate([s.curvature_mean for s in parts]),
            None if parts[0].curvature_principal is None else np.concatenate([s.curvature_principal for s in parts])))
    surfaces = combined
    vertex_count = sum(len(s.positions) for s in surfaces)
    if vertex_count > (options.max_vertices if options.max_vertices is not None else 1_000_000):
        raise ValueError("preview cannot certify the requested tolerance within the vertex budget")
    bound = max_meridian_bound+reach*(2*math.pi/nphi)**2/8+render_roundoff
    if bound > target*(1+1e-12):
        raise ValueError("preview could not certify the requested chord tolerance")
    box = np.asarray(model.bounds)
    fidelity = {"max_chord_error_mm_requested":target, "max_chord_error_mm_achieved":bound,
                    "max_chord_error_mm":bound, "measurement_complete":True, "vertex_cap_limited":False,
                    "method":"continuous-meridian-and-angular-bound", "binary32RoundoffBoundMm":render_roundoff,
                    "max_normal_step_deg_requested":normal, "max_normal_step_deg_achieved":normal,
                    "max_normal_step_deg":normal, "normal_bound_method":"sum-of-principal-direction-angle-ranges"}
    metadata = {**model.metadata(), "api_version":"hornlab.preview/1", "metadata_version":"hornlab.preview/1.4",
        "formula":"OSSE-ROUNDOVER", "mode":"freestanding", "units":"mm", "boundsMm":model.bounds,
        "dimensions_status":"current", "lod":options.lod, "coordinate_frame":"mesher-xyz",
        "dimensions_sampling":{"method":"analytic-full-envelope", "lod_independent":True},
        "dimensions_mm":{"mouth_opening":[2*float(model.body(model.length)[0][1])]*2,
                        "horn_overall":(box[1]-box[0]).tolist()},
        "fidelity":{s.role:dict(fidelity) for s in surfaces}, "mouthRoundoverChordBoundMm":bound,
        "source_shape":0,"source_radius_mm":model.r0,"source_cap_height_mm":0.0,
        "source_throat_radius_mm":model.r0, "warnings":[],
        "surface_metadata":{s.role:dict(s.metadata) for s in surfaces}}
    return PreviewGeometryV1(surfaces,metadata)
