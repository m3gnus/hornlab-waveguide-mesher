"""Certified physical preview of a circular adapter and retained conic."""
from __future__ import annotations

import math
import numpy as np

from ..adapter_axial import PhysicalAdapter, FORMULA
from .contract import PreviewGeometryV1, PreviewSurfaceV1
from .primitives import _grid_indices, _orient_indices_to_normals, _flat_cap


def _split(controls, value):
    rows = [np.asarray(controls,dtype=float)]
    while len(rows[-1]) > 1:
        rows.append((1-value)*rows[-1][:-1]+value*rows[-1][1:])
    return np.array([r[0] for r in rows]),np.array([r[-1] for r in rows[::-1]])


def _cubic_bounds(model, lo, hi):
    # Restrict derivative controls to this interval by de Casteljau. Positive
    # axial coefficients make every tangent slope a positive weighted average
    # of the control slopes, including unobserved interior extrema.
    controls = 3*np.diff(model.cubic,axis=0)
    if lo:
        controls = _split(controls,lo)[1]
    if hi < 1:
        controls = _split(controls,(hi-lo)/(1-lo))[0]
    angles = np.arctan2(controls[:,1],controls[:,0])
    second = model.cubic_values(np.array([lo,hi]))[2]
    return float(np.max(np.linalg.norm(second,axis=1)))*(hi-lo)**2/8,float(np.ptp(angles))


def _finite_number(value):
    try:
        return type(value) in (int,float) and math.isfinite(value)
    except OverflowError:
        return False


def build(params, options):
    model = PhysicalAdapter.from_params(params)
    if options.lod not in ("coarse","fine","inspection"):
        raise ValueError("lod must be 'coarse', 'fine', or 'inspection'")
    for name in ("include_inner","include_outer","include_enclosure","include_source_cap","include_rear_cap","include_curvature"):
        if type(getattr(options,name)) is not bool:
            raise ValueError(f"{name} must be a boolean")
    target = .008 if options.max_chord_error_mm is None else options.max_chord_error_mm
    normal = 3. if options.max_normal_step_deg is None else options.max_normal_step_deg
    if not _finite_number(target) or target <= 0:
        raise ValueError("max_chord_error_mm must be finite and positive")
    if not _finite_number(normal) or not 0 < normal <= 180:
        raise ValueError("max_normal_step_deg must be in (0,180]")
    if normal < 720/4096:
        raise ValueError("preview angular sampling exceeds its certified budget")
    limit = 1_000_000 if options.max_vertices is None else options.max_vertices
    if type(limit) is not int or not 8 <= limit <= 1_000_000:
        raise ValueError("max_vertices must be an integer in [8,1000000]")
    silhouette = 16 if options.min_silhouette_segments is None else options.min_silhouette_segments
    if type(silhouette) is not int or not 3 <= silhouette <= 4096:
        raise ValueError("min_silhouette_segments must be an integer in [3,4096]")
    roundoff = math.sqrt(3)*max(1.,*(abs(v) for row in model.bounds for v in row))*float(np.finfo(np.float32).eps)
    budget = (target-roundoff)/2
    if budget <= 0:
        raise ValueError("preview tolerance is below the binary32 placement precision")
    reach = model.reach
    nphi = 4*math.ceil(max(16,math.ceil(2*math.pi/math.sqrt(8*budget/reach)),math.ceil(720/normal),silhouette)/4)
    if nphi > 4096:
        raise ValueError("preview angular sampling exceeds its certified budget")
    angular_error = reach*(2*math.pi/nphi)**2/8
    angular_normal = 2*math.pi/nphi
    max_meridian_error = max_meridian_normal = 0.
    branches = []
    for cubic in (True,False):
        lo = 0. if cubic else model.payload["join_t"]
        values = [lo]

        def refine(left,right):
            nonlocal max_meridian_error,max_meridian_normal
            if cubic:
                error,angle = _cubic_bounds(model,left,right)
            else:
                _,d,dd = model.body_values(np.array([left,right]))
                error = float(np.linalg.norm(dd[0]))*(right-left)**2/8
                angle = float(np.ptp(np.arctan2(d[:,1],d[:,0])))
            if error <= budget and angle <= math.radians(normal)/2:
                values.append(right)
                max_meridian_error = max(max_meridian_error,error)
                max_meridian_normal = max(max_meridian_normal,angle)
                if len(values) > 8192:
                    raise ValueError("preview meridian sampling exceeds its certified budget")
            else:
                mid = (left+right)/2
                if mid == left or mid == right:
                    raise ValueError("preview cannot resolve a meridian interval")
                refine(left,mid)
                refine(mid,right)

        refine(lo,1.)
        branches.append(np.array(values))
    # One shared join ring. Curvature there is the retained-body one-sided
    # value; the two branches are G1 and need not have equal curvature.
    cp,cd,cdd = model.cubic_values(branches[0][:-1])
    bp,bd,bdd = model.body_values(branches[1])
    bp[0] = model.cubic[-1]
    meridian,derivative,second = (np.vstack(parts) for parts in ((cp,bp),(cd,bd),(cdd,bdd)))
    count = (len(meridian)*nphi if options.include_inner else 0)+(nphi+1 if options.include_source_cap else 0)
    if count > limit:
        raise ValueError("preview cannot certify the requested tolerance within the vertex budget")
    az = np.arange(nphi)*2*math.pi/nphi
    surfaces = []
    if options.include_inner:
        radius = meridian[:,1,None]
        points = np.stack((radius*np.cos(az),radius*np.sin(az)+model.offset,np.broadcast_to(meridian[:,0,None],(len(meridian),nphi))),axis=-1)
        speed = np.linalg.norm(derivative,axis=1)
        unit = derivative/speed[:,None]
        normals = np.stack((-unit[:,0,None]*np.cos(az),-unit[:,0,None]*np.sin(az),np.broadcast_to(unit[:,1,None],(len(meridian),nphi))),axis=-1)
        positions,ns = points.reshape(-1,3),normals.reshape(-1,3)
        oriented = _orient_indices_to_normals("horn.inner",positions,_grid_indices(len(meridian),nphi,closed_phi=True),ns)
        mean = principal = None
        if options.include_curvature:
            meridian_k = -(derivative[:,0]*second[:,1]-derivative[:,1]*second[:,0])/speed**3
            angular_k = unit[:,0]/meridian[:,1]
            largest = np.where(abs(meridian_k)>abs(angular_k),meridian_k,np.where(abs(angular_k)>abs(meridian_k),angular_k,np.maximum(meridian_k,angular_k)))
            mean = np.broadcast_to(((meridian_k+angular_k)/2)[:,None],(len(meridian),nphi)).reshape(-1)
            principal = np.broadcast_to(largest[:,None],(len(meridian),nphi)).reshape(-1)
        surfaces.append(PreviewSurfaceV1("horn.inner",positions,oriented.indices,ns,"smooth","analytic-parametric",True,mean,principal,
            {"join_curvature":"retained-body-one-sided","join_station":len(cp)}))
    if options.include_source_cap:
        r = float(model.cubic[0,1])
        ring = np.column_stack((r*np.cos(az),r*np.sin(az)+model.offset,np.zeros(nphi)))
        surfaces.append(_flat_cap("source_cap",ring,(0,0,1),closed_phi=True,include_curvature=options.include_curvature))
    # This is geometric distance, not same-parameter interpolation error.
    # For any triangle barycentrics, the weighted (z,r) lies on the meridian
    # chord. Its XY vector is a positive weighted average of the two bounding
    # azimuths; radial shrink is <= r*(1-cos(deltaPhi/2)). The meridian chord
    # differs from the curve by <= max||C''||*deltaU**2/8. Conversely, at each
    # interpolated meridian parameter the two triangles cover every azimuth
    # in the wedge, so the same sum bounds surface-to-triangle distance.
    # A radial/azimuth mixed term is unnecessary for this geometric bound.
    bound = max_meridian_error+angular_error+roundoff
    normal_bound = math.degrees(max_meridian_normal+angular_normal)
    if bound > target*(1+1e-12) or normal_bound > normal*(1+1e-12):
        raise ValueError("preview could not certify the requested tolerances")
    fidelity = {"max_chord_error_mm_requested":target,"max_chord_error_mm_achieved":bound,"max_chord_error_mm":bound,
        "measurement_complete":True,"vertex_cap_limited":False,"method":"continuous-physical-meridian-and-angular-bound",
        "binary32RoundoffBoundMm":roundoff,"max_normal_step_deg_requested":normal,
        "max_normal_step_deg_achieved":normal_bound,"max_normal_step_deg":normal_bound,
        "normal_bound_method":"derivative-control-slope-hull-and-angular-range"}
    metadata = {**model.metadata(),"api_version":"hornlab.preview/1","metadata_version":"hornlab.preview/1.4",
        "formula":FORMULA,"mode":"bare","units":"mm","coordinate_frame":"mesher-xyz","lod":options.lod,
        "boundsMm":model.bounds,"dimensions_status":"current","dimensions_mm":model.dimensions,
        "dimensions_sampling":{"method":"analytic-full-envelope","lod_independent":True},
        "actual_segment_counts":{"horn_phi":nphi,"horn_axial":len(meridian)-1,"source_cap_radial":int(options.include_source_cap)},
        "semantic_stations":{"inserted_first":["driver","adapter join","mouth"],"unavailable_additively":[]},
        "source_shape":0,"source_radius_mm":float(model.cubic[0,1]),"source_cap_height_mm":0.,
        "source_throat_radius_mm":float(model.cubic[0,1]),"source_auto_angle_deg":model.source_angle,
        "warnings":[],"fidelity":{s.role:dict(fidelity) for s in surfaces},
        "surface_metadata":{s.role:dict(s.metadata) for s in surfaces}}
    return PreviewGeometryV1(surfaces,metadata)
