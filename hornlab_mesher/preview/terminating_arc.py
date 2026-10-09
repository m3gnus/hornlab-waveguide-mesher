"""Continuous geometric triangle bounds for the exact conic/circle meridian."""
import math

import numpy as np

from ..terminating_arc import ArcMeridian,FORMULA
from .contract import PreviewGeometryV1,PreviewSurfaceV1
from .primitives import _grid_indices,_orient_indices_to_normals,_flat_cap


def _finite(value):
    try:
        return type(value) in (int,float) and math.isfinite(float(value))
    except (OverflowError,ValueError,TypeError):
        return False


def build(params,options):
    for name in ("include_inner","include_outer","include_enclosure","include_source_cap","include_rear_cap","include_curvature"):
        if type(getattr(options,name)) is not bool:
            raise ValueError(f"{name} must be a boolean")
    model=ArcMeridian.from_params(params)
    if options.lod not in {"coarse","fine","inspection"}:
        raise ValueError("lod must be 'coarse', 'fine', or 'inspection'")
    chord=.008 if options.max_chord_error_mm is None else options.max_chord_error_mm
    normal=3. if options.max_normal_step_deg is None else options.max_normal_step_deg
    if not _finite(chord) or chord <= 0:
        raise ValueError("max_chord_error_mm must be finite and positive")
    if not _finite(normal) or not 0 < normal <= 180:
        raise ValueError("max_normal_step_deg must be in (0,180]")
    if normal < 720/4096:
        raise ValueError("preview angular sampling exceeds its certified budget")
    limit=1_000_000 if options.max_vertices is None else options.max_vertices
    silhouette=0 if options.min_silhouette_segments is None else options.min_silhouette_segments
    if type(limit) is not int or not 8 <= limit <= 1_000_000:
        raise ValueError("max_vertices must be an integer in [8,1000000]")
    if type(silhouette) is not int or not 0 <= silhouette <= 4096:
        raise ValueError("min_silhouette_segments must be an integer in [0,4096]")
    magnitude=max(1.,*(abs(v) for row in model.bounds for v in row))
    roundoff=math.sqrt(3)*magnitude*2**-24+1e-10*magnitude
    budget=(float(chord)-roundoff)/2
    if budget <= 0:
        raise ValueError("preview tolerance is below binary32 placement precision")
    # Test the angular ceiling without first dividing a tiny tolerance.
    if budget < model.reach*(2*math.pi/4096)**2/8:
        raise ValueError("preview angular sampling exceeds its certified budget")
    nphi=4*math.ceil(max(16,math.ceil(2*math.pi/math.sqrt(8*budget/model.reach)),
        math.ceil(720/normal),silhouette)/4)
    if nphi > 4096:
        raise ValueError("preview angular sampling exceeds its certified budget")
    z=[0.]
    def split(lo,hi,depth=0):
        derivative=model.body(np.array([lo,hi]))[1][:,1]
        angle=math.degrees(math.atan(derivative[1])-math.atan(derivative[0]))
        error=model.second_bound(lo,hi)*(hi-lo)**2/8
        if error <= budget and angle <= normal/2:
            z.append(hi)
            if len(z) > 8192:
                raise ValueError("preview meridian sampling exceeds its certified budget")
        else:
            mid=(lo+hi)/2
            if depth >= 32 or mid in (lo,hi):
                raise ValueError("preview meridian tolerance could not be resolved")
            split(lo,mid,depth+1);split(mid,hi,depth+1)
    split(0.,model.join_z)
    sweep=model.end-model.beta
    narc=max(1,math.ceil(sweep/math.sqrt(8*budget/model.radius)),math.ceil(math.degrees(sweep)/(normal/2)))
    if narc > 8192:
        raise ValueError("preview arc sampling exceeds its certified budget")
    count=len(z)+narc
    vertices=(count*nphi if options.include_inner else 0)+(nphi+1 if options.include_source_cap else 0)
    if vertices > limit:
        raise ValueError("preview cannot certify the requested tolerance within its vertex budget")
    z=np.asarray(z)
    phi=np.linspace(model.beta,model.end,narc+1)
    bp,bd=model.body(z)
    meridian=np.vstack((bp,model.arc(phi)[1:]))
    tangent=np.concatenate((np.arctan(bd[:,1]),phi[1:]))
    az=np.arange(nphi)*2*math.pi/nphi
    cs,sn=np.cos(az),np.sin(az)
    surfaces=[]
    if options.include_inner:
        positions=np.stack((meridian[:,1,None]*cs,meridian[:,1,None]*sn+model.offset,
            np.broadcast_to(meridian[:,0,None],(count,nphi))),axis=-1).reshape(-1,3)
        normals=np.stack((-np.cos(tangent)[:,None]*cs,-np.cos(tangent)[:,None]*sn,
            np.broadcast_to(np.sin(tangent)[:,None],(count,nphi))),axis=-1).reshape(-1,3)
        triangles=_grid_indices(count,nphi,closed_phi=True)
        oriented=_orient_indices_to_normals("horn.inner",positions,triangles,normals)
        mean=principal=None
        if options.include_curvature:
            meridian_k=np.concatenate((-model.second_bound(z,z)/(1+bd[:,1]**2)**1.5,
                np.full(narc,-1/model.radius)))
            angular_k=np.cos(tangent)/meridian[:,1]
            mean=np.repeat((meridian_k+angular_k)/2,nphi)
            principal=np.repeat(np.where(abs(meridian_k)>abs(angular_k),meridian_k,angular_k),nphi)
        surfaces.append(PreviewSurfaceV1("horn.inner",positions,oriented.indices,normals,"smooth",
            "analytic-parametric",True,mean,principal))
    if options.include_source_cap:
        ring=np.column_stack((model.r0*cs,model.r0*sn+model.offset,np.zeros(nphi)))
        surfaces.append(_flat_cap("source_cap",ring,(0,0,1),closed_phi=True,include_curvature=options.include_curvature))
    body_bound=max(model.second_bound(lo,hi)*(hi-lo)**2/8 for lo,hi in zip(z[:-1],z[1:]))
    arc_bound=model.radius*(sweep/narc)**2/8
    angular_bound=model.reach*(2*math.pi/nphi)**2/8
    bound=max(body_bound,arc_bound)+angular_bound+roundoff
    normal_bound=max(float(np.max(np.diff(np.arctan(bd[:,1])),initial=0)),sweep/narc)*180/math.pi+360/nphi
    if bound > chord*(1+1e-12) or normal_bound > normal*(1+1e-12):
        raise ValueError("preview could not certify the requested continuous fidelity")
    fidelity={"max_chord_error_mm_requested":chord,"max_chord_error_mm_achieved":bound,
        "max_chord_error_mm":bound,"measurement_complete":True,"vertex_cap_limited":False,
        "method":"continuous-meridian-and-angular-geometric-triangle-bound",
        "binary32RoundoffBoundMm":roundoff,"max_normal_step_deg_requested":normal,
        "max_normal_step_deg_achieved":normal_bound,"max_normal_step_deg":normal_bound,
        "normal_bound_method":"sum-of-principal-direction-angle-ranges"}
    box=np.asarray(model.bounds)
    metadata={**model.metadata(),"api_version":"hornlab.preview/1","metadata_version":"hornlab.preview/1.4",
        "formula":FORMULA,"mode":"bare","units":"mm","boundsMm":model.bounds,"lod":options.lod,
        "dimensions_status":"current","coordinate_frame":"mesher-xyz",
        "dimensions_sampling":{"method":"analytic-full-envelope","lod_independent":True},
        "dimensions_mm":{"mouth_opening":[2*model.reach]*2,"horn_overall":(box[1]-box[0]).tolist()},
        "fidelity":{s.role:dict(fidelity) for s in surfaces},"source_shape":0,"source_radius_mm":model.r0,
        "source_cap_height_mm":0.,"source_throat_radius_mm":model.r0,"warnings":[],
        "surface_metadata":{s.role:dict(s.metadata) for s in surfaces}}
    return PreviewGeometryV1(surfaces,metadata)
