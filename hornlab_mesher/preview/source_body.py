"""Analytically bounded previews of the source disk and closed exterior."""
import math

import numpy as np

from ..source_body import FORMULA, MODE, StandaloneSourceGeometry
from .contract import PreviewGeometryV1, PreviewSurfaceV1, _orient_indices_to_normals


def build(params, options):
    model=StandaloneSourceGeometry(**params["sourceBody"])
    presets={"coarse":(.15,8.),"fine":(.05,3.),"inspection":(.025,2.)}
    if options.lod not in presets:
        raise ValueError("lod must be coarse, fine or inspection")
    for key in ("include_inner","include_outer","include_enclosure","include_source_cap","include_rear_cap","include_curvature"):
        if type(getattr(options,key)) is not bool:
            raise ValueError(f"{key} must be a boolean")
    chord,normal=presets[options.lod]
    chord=chord if options.max_chord_error_mm is None else options.max_chord_error_mm
    normal=normal if options.max_normal_step_deg is None else options.max_normal_step_deg
    if type(chord) not in (int,float) or not math.isfinite(chord) or chord<=0:
        raise ValueError("max_chord_error_mm must be finite and positive")
    if type(normal) not in (int,float) or not math.isfinite(normal) or not 0<normal<=180:
        raise ValueError("max_normal_step_deg must be in (0,180]")
    limit=100000 if options.max_vertices is None else options.max_vertices
    silhouette=16 if options.min_silhouette_segments is None else options.min_silhouette_segments
    if type(limit) is not int or not 8<=limit<=1000000:
        raise ValueError("max_vertices must be an integer in [8,1000000]")
    if type(silhouette) is not int or not 3<=silhouette<=4096:
        raise ValueError("min_silhouette_segments must be an integer in [3,4096]")
    r,R,d,y=model.radius_mm,model.outer_radius_mm,model.depth_mm,model.vertical_offset_mm
    selected=options.include_outer or options.include_source_cap or options.include_rear_cap
    radius=R if options.include_outer or options.include_rear_cap else r
    roundoff=math.sqrt(3)*max(abs(v) for row in model.bounds for v in row)*2**-24
    budget=chord-roundoff
    if selected and budget<=0:
        raise ValueError("preview tolerance is below binary32 placement precision")
    n=0 if not selected else 4*math.ceil(max(silhouette,16,2*math.pi/math.sqrt(8*budget/radius),360/normal)/4)
    if n>4096:
        raise ValueError("standalone source preview angular budget exceeds4096")
    count=(4*n if options.include_outer else 0)+(n+1 if options.include_source_cap else 0)+(n+1 if options.include_rear_cap else 0)
    if count>limit:
        raise ValueError("standalone source preview exceeds its vertex budget")
    surfaces=[]
    if selected:
        phi=np.arange(n)*2*math.pi/n
        def ring(radius,z):
            return np.column_stack((radius*np.cos(phi),radius*np.sin(phi)+y,np.full(n,z)))
        def surface(role,p,t,normals,planar):
            p=np.asarray(p,dtype=float);t=np.asarray(t,dtype=np.uint32).reshape(-1)
            normals=np.asarray(normals,dtype=float)
            t=_orient_indices_to_normals(role,p,t,normals).indices
            mean=principal=None
            if options.include_curvature:
                mean=np.full(len(p),0. if planar else -1/(2*R))
                principal=np.full(len(p),0. if planar else -1/R)
            surfaces.append(PreviewSurfaceV1(role,p,t,normals,"flat" if planar else "smooth",
                "exact-planar" if planar else "analytic-parametric",True,mean,principal))
        def cap(role,radius,z,sign):
            p=np.vstack(([0.,y,z],ring(radius,z)))
            t=[[0,i+1,(i+1)%n+1] for i in range(n)]
            surface(role,p,t,np.tile([0.,0.,sign],(n+1,1)),True)
        if options.include_source_cap:
            cap("source_body.disk",r,0.,1.)
        if options.include_outer:
            inner,outer=ring(r,0.),ring(R,0.)
            t=[]
            for i in range(n):
                j=(i+1)%n;t.extend([[i,j,n+j],[i,n+j,n+i]])
            surface("source_body.annulus",np.vstack((inner,outer)),t,np.tile([0.,0.,1.],(2*n,1)),True)
            front,rear=ring(R,0.),ring(R,-d)
            radial=np.column_stack((np.cos(phi),np.sin(phi),np.zeros(n)))
            surface("source_body.side",np.vstack((front,rear)),t,np.vstack((radial,radial)),False)
        if options.include_rear_cap:
            cap("source_body.rear",R,-d,-1.)
    achieved=0. if not selected else radius*(2*math.pi/n)**2/8+roundoff
    if selected and achieved>chord*(1+1e-12):
        raise ValueError("standalone source preview could not certify chord tolerance")
    fidelity={"max_chord_error_mm_requested":chord,"max_chord_error_mm_achieved":achieved,
        "max_chord_error_mm":achieved,"measurement_complete":True,"vertex_cap_limited":False,
        "method":"continuous-circular-sagitta-bound","binary32RoundoffBoundMm":roundoff,
        "max_normal_step_deg_requested":normal,"max_normal_step_deg_achieved":0. if not selected else 360/n,
        "max_normal_step_deg":0. if not selected else 360/n,"normal_bound_method":"exact-cylinder-angle"}
    return PreviewGeometryV1(surfaces,{**model.metadata(),"api_version":"hornlab.preview/1","metadata_version":"hornlab.preview/1.4",
        "formula":FORMULA,"mode":MODE,"units":"mm","boundsMm":model.bounds,"lod":options.lod,
        "coordinate_frame":"mesher-xyz","dimensions_status":"current",
        "dimensions_sampling":{"method":"analytic-full-envelope","lod_independent":True},
        "dimensions_mm":{"source_diameter":[2*r,2*r],"body_overall":[2*R,2*R,d]},
        "datums":model.datums(),"warnings":[],"fidelity":{s.role:dict(fidelity) for s in surfaces},
        "surface_metadata":{s.role:dict(s.metadata) for s in surfaces}})
