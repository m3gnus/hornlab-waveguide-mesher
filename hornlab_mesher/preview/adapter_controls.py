"""Two analytic branch grids preserve both sides of a native join crease."""
import math

import numpy as np

from ..adapter_controls import ControlsMeridian,number
from .contract import PreviewGeometryV1,PreviewSurfaceV1
from .primitives import _grid_indices,_orient_indices_to_normals,_flat_cap


def build(params,options):
    m=ControlsMeridian.from_params(params)
    if options.lod not in {"coarse","fine","inspection"}:raise ValueError("invalid native preview LOD")
    for name in ("include_inner","include_outer","include_enclosure","include_source_cap","include_rear_cap","include_curvature"):
        if type(getattr(options,name)) is not bool:raise ValueError(f"{name} must be a boolean")
    chord=.008 if options.max_chord_error_mm is None else number(options.max_chord_error_mm,"max_chord_error_mm")
    normal=3. if options.max_normal_step_deg is None else number(options.max_normal_step_deg,"max_normal_step_deg")
    if chord<=0 or not 0<normal<=180:raise ValueError("native preview tolerances must be positive and normal step at most 180 degrees")
    limit=1_000_000 if options.max_vertices is None else options.max_vertices
    if type(limit) is not int or not 8<=limit<=1_000_000:raise ValueError("max_vertices must be an integer in [8,1000000]")
    silhouette=0 if options.min_silhouette_segments is None else options.min_silhouette_segments
    if type(silhouette) is not int or not 0<=silhouette<=4096:raise ValueError("min_silhouette_segments must be an integer in [0,4096]")
    size=max(1.,*(abs(v) for row in m.bounds for v in row))
    rounding=math.sqrt(3)*np.finfo(np.float32).eps*size+1e-10*size
    budget=chord-rounding
    if budget<=0 or normal<360/4096:raise ValueError("native preview tolerance exceeds the precision or angular work budget")
    # Test representability/work before ceil or allocation, including tiny
    # positive subnormal targets that would otherwise overflow a ratio.
    if budget<m.reach*(2*math.pi/4096)**2/2:raise ValueError("native preview exceeds its angular sampling budget")
    nphi=4*math.ceil(max(16,2*math.pi*math.sqrt(m.reach/(2*budget)),1080/normal,silhouette)/4)
    if nphi>4096:raise ValueError("native preview exceeds its angular sampling budget")
    da=2*math.pi/nphi;normal_radians=math.radians(normal)
    arrays=[];normal_arrays=[];mean_arrays=[];principal_arrays=[];indices=[];branches=[]
    worst=0.;normal_worst=0.;vertex_start=0
    for branch in ("cubic","body"):
        stations=[0.];pending=[(0.,1.,0)]
        while pending:
            lo,hi,depth=pending.pop();du=hi-lo
            Muu,Mr,rmax,_,zmin=m.interval_bounds(branch,lo,hi)
            bound=float((Muu*du*du+2*Mr*du*da+rmax*da*da)/8)
            normal_bound=math.inf if zmin<=0 else float(Muu/zmin*du+da)
            if bound<=budget and normal_bound<=normal_radians:
                stations.append(hi);worst=max(worst,bound);normal_worst=max(normal_worst,normal_bound)
            else:
                if depth>=32 or len(stations)+len(pending)>8192:raise ValueError("native preview cannot certify within its branch sampling budget")
                mid=(lo+hi)/2;pending.extend(((mid,hi,depth+1),(lo,mid,depth+1)))
        u=np.asarray(stations);count=len(u)*nphi
        required=(vertex_start+count if options.include_inner else 0)+(nphi+1 if options.include_source_cap else 0)
        if required>limit:raise ValueError("native preview cannot certify within its vertex budget")
        if options.include_inner:
            p,d,e=m.branch(branch,u);speed=np.linalg.norm(d,axis=-1);az=np.arange(nphi)*da
            points=np.stack((p[:,1,None]*np.cos(az),p[:,1,None]*np.sin(az)+m.offset,np.broadcast_to(p[:,0,None],(len(u),nphi))),axis=-1).reshape(-1,3)
            normals=np.stack((-d[:,0,None]*np.cos(az)/speed[:,None],-d[:,0,None]*np.sin(az)/speed[:,None],np.broadcast_to(d[:,1,None]/speed[:,None],(len(u),nphi))),axis=-1).reshape(-1,3)
            km=-(d[:,0]*e[:,1]-d[:,1]*e[:,0])/speed**3;ka=d[:,0]/(p[:,1]*speed)
            mean=np.broadcast_to(((km+ka)/2)[:,None],(len(u),nphi)).reshape(-1)
            dominant=np.where(abs(km)>=abs(ka),km,ka);principal=np.broadcast_to(dominant[:,None],(len(u),nphi)).reshape(-1)
            arrays.append(points);normal_arrays.append(normals);mean_arrays.append(mean);principal_arrays.append(principal)
            indices.append(_grid_indices(len(u),nphi,closed_phi=True)+vertex_start)
        branches.append({"branch":branch,"vertexStart":vertex_start,"vertexEnd":vertex_start+count,
            "parameters":stations,"stationCount":len(u)})
        vertex_start+=count
    surfaces=[]
    if options.include_inner:
        points=np.vstack(arrays);normals=np.vstack(normal_arrays);idx=np.concatenate(indices)
        oriented=_orient_indices_to_normals("horn.inner",points,idx,normals)
        surfaces.append(PreviewSurfaceV1("horn.inner",points,oriented.indices,normals,"smooth","analytic-parametric",True,
            np.concatenate(mean_arrays) if options.include_curvature else None,
            np.concatenate(principal_arrays) if options.include_curvature else None,
            {"branchRanges":branches,"angularSegments":nphi,"joinTangentJumpDeg":m.join_jump_deg,
             "normalBoundScope":"each smooth branch; reported join crease excluded"}))
    if options.include_source_cap:
        az=np.arange(nphi)*da;ring=np.column_stack((m.source_radius*np.cos(az),m.source_radius*np.sin(az)+m.offset,np.zeros(nphi)))
        surfaces.append(_flat_cap("source_cap",ring,(0,0,1),closed_phi=True,include_curvature=options.include_curvature))
    bound=worst+rounding
    if bound>chord or math.degrees(normal_worst)>normal*(1+1e-12):raise ValueError("native preview failed its final continuous certificate")
    fidelity={"method":"continuous-branch-chart-Hessian-bound","measurement_complete":True,"vertex_cap_limited":False,
        "max_chord_error_mm_requested":chord,"max_chord_error_mm":bound,"max_chord_error_mm_achieved":bound,
        "binary32RoundoffBoundMm":rounding,"max_normal_step_deg_requested":normal,
        "max_normal_step_deg":math.degrees(normal_worst),"max_normal_step_deg_achieved":math.degrees(normal_worst),
        "normal_bound_method":"whole-interval-meridian-angle-plus-azimuth","normal_bound_scope":"smooth branches; join crease separately reported"}
    metadata={**m.metadata(),"api_version":"hornlab.preview/1","metadata_version":"hornlab.preview/1.4",
        "formula":"OSSE-ADAPTER-CONTROLS","mode":"bare","units":"mm","lod":options.lod,"coordinate_frame":"mesher-xyz",
        "boundsMm":m.bounds,"dimensions_status":"current","dimensions_sampling":{"method":"exact-cubic-extrema-and-conic-mouth","lod_independent":True},
        "dimensions_mm":{"source_diameter":[2*m.source_radius]*2,"join_diameter":[2*m.r0]*2,
            "mouth_opening":[2*float(m.body(1)[0][1])]*2,"horn_overall":[2*m.reach,2*m.reach,m.depth]},
        "fidelity":{s.role:dict(fidelity) for s in surfaces},"branchRanges":branches,
        "source_shape":0,"source_radius_mm":m.source_radius,"source_throat_radius_mm":m.source_radius,
        "source_cap_radius_mm":m.source_radius,"source_cap_height_mm":0.,"source_auto_angle_deg":m.source_angle_deg,
        "warnings":[],"surface_metadata":{s.role:dict(s.metadata) for s in surfaces}}
    return PreviewGeometryV1(surfaces,metadata)
