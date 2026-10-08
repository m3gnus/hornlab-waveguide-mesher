"""Continuous physical chord and normal bounds for the absolute-axis body."""
import math

import numpy as np

from ..axial_scale import AxialModel
from .contract import PreviewGeometryV1, PreviewSurfaceV1
from .primitives import _grid_indices, _orient_indices_to_normals, _flat_cap


def build(params, options):
    model = AxialModel.from_params(params)
    presets = {"coarse":(0.15,8), "fine":(0.05,3), "inspection":(0.025,2)}
    if options.lod not in presets:
        raise ValueError("lod must be 'coarse', 'fine', or 'inspection'")
    default_chord, default_normal = presets[options.lod]
    chord = default_chord if options.max_chord_error_mm is None else options.max_chord_error_mm
    normal = default_normal if options.max_normal_step_deg is None else options.max_normal_step_deg
    if type(chord) not in (int,float) or not math.isfinite(chord) or chord <= 0:
        raise ValueError("max_chord_error_mm must be finite and positive")
    if type(normal) not in (int,float) or not math.isfinite(normal) or not 0 < normal <= 180:
        raise ValueError("max_normal_step_deg must be in (0,180]")
    limit = 1_000_000 if options.max_vertices is None else options.max_vertices
    if type(limit) is not int or limit < 8:
        raise ValueError("max_vertices must be an integer of at least 8")
    silhouette = 0 if options.min_silhouette_segments is None else options.min_silhouette_segments
    if type(silhouette) is not int or silhouette < 0:
        raise ValueError("min_silhouette_segments must be a nonnegative integer")
    roundoff = math.sqrt(3)*max(abs(v) for row in model.bounds for v in row)*2**-24
    budget = (chord-roundoff)/2
    if budget <= 0:
        raise ValueError("preview tolerance is below the binary32 placement precision")
    reach = float(model.body(model.length)[0][1])
    nphi = 4*math.ceil(max(16, math.ceil(2*math.pi/math.sqrt(8*budget/reach)), math.ceil(720/normal), silhouette)/4)
    if nphi > 4096:
        raise ValueError("preview angular sampling exceeds its certified budget")
    z = model.stations(budget)
    refined = [0.0]
    def refine(lo,hi):
        tangent = model.body(np.array([lo,hi]))[1]
        angle = np.arctan2(tangent[:,1],tangent[:,0])
        if math.degrees(abs(angle[1]-angle[0])) <= normal/2:
            refined.append(hi)
            if len(refined) > 8192:
                raise ValueError("preview axial sampling exceeds its certified budget")
        else:
            mid = (lo+hi)/2
            refine(lo,mid)
            refine(mid,hi)
    for lo,hi in zip(z[:-1],z[1:]):
        refine(lo,hi)
    z = np.asarray(refined)
    vertex_count = (len(z)*nphi if options.include_inner else 0)+(nphi+1 if options.include_source_cap else 0)
    if vertex_count > limit:
        raise ValueError("preview cannot certify the requested tolerance within the vertex budget")
    az = np.arange(nphi)*2*math.pi/nphi
    meridian, derivative = model.body(z)
    surfaces = []
    if options.include_inner:
        radius = meridian[:,1,None]
        points = np.stack((radius*np.cos(az), radius*np.sin(az)+model.offset,
                           np.broadcast_to(z[:,None], (len(z),nphi))),axis=-1)
        unit = derivative/np.linalg.norm(derivative,axis=-1)[:,None]
        normals = np.stack((-unit[:,0,None]*np.cos(az), -unit[:,0,None]*np.sin(az),
                            np.broadcast_to(unit[:,1,None],(len(z),nphi))),axis=-1)
        positions, ns = points.reshape(-1,3), normals.reshape(-1,3)
        indices = _grid_indices(len(z),nphi,closed_phi=True)
        oriented = _orient_indices_to_normals("horn.inner",positions,indices,ns)
        mean = principal = None
        if options.include_curvature:
            A,B,C = model.coefficients
            d = derivative[:,1]
            meridian_k = -model.curvature_numerator/(A+2*B*z+C*z*z)**1.5/(1+d*d)**1.5
            angular_k = unit[:,0]/meridian[:,1]
            mean = np.broadcast_to(((meridian_k+angular_k)/2)[:,None],(len(z),nphi)).reshape(-1)
            largest = np.where(abs(meridian_k)>abs(angular_k),meridian_k,angular_k)
            principal = np.broadcast_to(largest[:,None],(len(z),nphi)).reshape(-1)
        surfaces.append(PreviewSurfaceV1("horn.inner",positions,oriented.indices,ns,"smooth","analytic-parametric",True,mean,principal))
    if options.include_source_cap:
        ring = np.column_stack((meridian[0,1]*np.cos(az),meridian[0,1]*np.sin(az)+model.offset,np.zeros(nphi)))
        surfaces.append(_flat_cap("source_cap",ring,(0,0,1),closed_phi=True,include_curvature=options.include_curvature))
    meridian_bound = max(model.second_bound(lo,hi)*(hi-lo)**2/8 for lo,hi in zip(z[:-1],z[1:]))
    bound = meridian_bound+reach*(2*math.pi/nphi)**2/8+roundoff
    if bound > chord*(1+1e-12):
        raise ValueError("preview could not certify the requested chord tolerance")
    fidelity = {"max_chord_error_mm_requested":chord,"max_chord_error_mm_achieved":bound,
        "max_chord_error_mm":bound,"measurement_complete":True,"vertex_cap_limited":False,
        "method":"continuous-meridian-and-angular-bound","binary32RoundoffBoundMm":roundoff,
        "max_normal_step_deg_requested":normal,"max_normal_step_deg_achieved":normal,
        "max_normal_step_deg":normal,"normal_bound_method":"sum-of-principal-direction-angle-ranges"}
    metadata = {**model.metadata(),"api_version":"hornlab.preview/1","metadata_version":"hornlab.preview/1.4",
        "formula":"OSSE-AXIAL","mode":"bare","units":"mm","boundsMm":model.bounds,
        "dimensions_status":"current","lod":options.lod,"coordinate_frame":"mesher-xyz",
        "dimensions_sampling":{"method":"analytic-full-envelope","lod_independent":True},
        "dimensions_mm":{"mouth_opening":[2*reach]*2,"horn_overall":[2*reach,2*reach,model.length]},
        "fidelity":{s.role:dict(fidelity) for s in surfaces},"source_shape":0,"source_radius_mm":model.r0,
        "source_cap_height_mm":0.0,"source_throat_radius_mm":model.r0,"warnings":[],
        "surface_metadata":{s.role:dict(s.metadata) for s in surfaces}}
    return PreviewGeometryV1(surfaces,metadata)
