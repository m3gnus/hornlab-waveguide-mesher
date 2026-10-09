"""Literal native cubic controls followed by a full exact circular OSSE body."""
from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from decimal import Decimal, localcontext
import hashlib
import json
import math
from typing import Mapping

import numpy as np

from .config_parser import ConfigError
from .throat_adapter import _conic_residual_bound

FORMULA = "OSSE-ADAPTER-CONTROLS"


def fail(message):
    raise ConfigError("Exact control adapter refused: " + message)


def number(value, name):
    try:
        valid = type(value) in (int, float) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        fail(f"{name} must be a finite scalar number.")
    value = float(value)
    return 0.0 if value == 0 else value


def normalize_payload(value):
    if not isinstance(value,Mapping) or set(value)!={"mode","contract_revision","join_t","control_points_mm"}:
        fail("the exact root throat_adapter requires mode, contract_revision, join_t and control_points_mm.")
    if value["mode"]!="controls" or type(value["contract_revision"]) is not int or value["contract_revision"]!=1:
        fail("mode must be controls and contract_revision must be integer 1.")
    if number(value["join_t"],"join_t")!=0:fail("join_t must be numeric zero.")
    pairs=value["control_points_mm"]
    if not isinstance(pairs,(list,tuple)) or len(pairs)!=3 or any(not isinstance(row,(list,tuple)) or len(row)!=2 for row in pairs):
        fail("control_points_mm requires exactly three pairs.")
    controls=[[number(v,"control_points_mm") for v in row] for row in pairs]
    if any(abs(v)>1e6 for row in controls for v in row):fail("control coordinates exceed 1000000 mm.")
    return {"mode":"controls","contract_revision":1,"join_t":0.0,"control_points_mm":controls}


def configuration(config, resolver):
    """Inspect all original supplied intent before delegating the ordinary body."""
    pending, seen, intent = [config], set(), False
    while pending:
        section = pending.pop()
        if not isinstance(section, (Mapping,list,tuple)) or id(section) in seen:
            continue
        seen.add(id(section))
        if isinstance(section,(list,tuple)):
            pending.extend(section)
            continue
        for key, value in section.items():
            normalized = str(key).replace("_", "").lower()
            if normalized == "controlpointsmm" or (normalized == "mode" and isinstance(value,str) and value.strip().lower() == "controls"):
                intent = True
            if normalized in {"formula", "type"} and isinstance(value, str) and value.strip().upper() == FORMULA:
                intent = True
            if isinstance(value, (Mapping,list,tuple)):
                pending.append(value)
    if not intent:
        return None
    for key in ("profile", "mesh", "source"):
        if key in config and not isinstance(config[key], Mapping):
            fail(f"{key} must be an object.")
    profile, mesh, source = (config.get(key, {}) for key in ("profile", "mesh", "source"))
    names = [s[k] for s in (config, profile) for k in ("formula", "type") if k in s]
    if not names or any(not isinstance(n, str) or n.strip().upper() != FORMULA for n in names):
        fail(f"every supplied formula and type must be {FORMULA}.")
    allowed = {
        "root": {"formula", "type", "profile", "mesh", "source", "throat_adapter", "mode", "scale", "quadrants", "vertical_offset_mm", "output", "path", "output_path"},
        "profile": {"formula", "type", "L_mm", "r0_mm", "a_deg", "a0_deg", "k", "s"},
        "mesh": {"wall_thickness_mm", "angular_segments", "length_segments", "throat_res_mm", "mouth_res_mm", "rear_res_mm", "max_triangles", "allow_large_mesh", "scale_to_metres", "topology_mode", "surface_fit", "quadrants", "vertical_offset_mm"},
        "source": {"source_shape"},
    }
    for name, section in (("root", config), ("profile", profile), ("mesh", mesh), ("source", source)):
        extra = set(section)-allowed[name]
        if extra:
            fail(f"unsupported {name} fields: {', '.join(sorted(map(str, extra)))}.")
    if "scale" in config:
        number(config["scale"],"scale")
    payload = normalize_payload(config.get("throat_adapter"))
    for section in (config, mesh):
        if "quadrants" in section and not ((type(section["quadrants"]) is int and section["quadrants"] == 1234) or (type(section["quadrants"]) is str and section["quadrants"] == "1234")):
            fail("quadrants must be exactly 1234.")
        if "vertical_offset_mm" in section:
            number(section["vertical_offset_mm"], "vertical_offset_mm")
    if "vertical_offset_mm" in config and "vertical_offset_mm" in mesh and config["vertical_offset_mm"] != mesh["vertical_offset_mm"]:
        fail("duplicate placement controls must agree.")
    for key in ("angular_segments", "length_segments", "max_triangles"):
        if key in mesh and (type(mesh[key]) is not int or not 0 < mesh[key] <= 10_000_000):
            fail(f"{key} must be a positive integer no larger than 10000000.")
    for key in ("allow_large_mesh", "scale_to_metres"):
        if key in mesh and type(mesh[key]) is not bool:
            fail(f"{key} must be a boolean.")
    for key in ("throat_res_mm", "mouth_res_mm", "rear_res_mm"):
        if key in mesh and not .01 <= number(mesh[key], key) <= 10000:
            fail(f"{key} must be in [0.01,10000] mm.")
    for key, value in profile.items():
        if key not in {"formula", "type"}:
            number(value, key)
    if config.get("mode", "bare") != "bare" or number(mesh.get("wall_thickness_mm", 0), "wall_thickness_mm") != 0:
        fail("only bare, zero-wall geometry is supported.")
    if mesh.get("topology_mode", "acoustic") != "acoustic" or mesh.get("surface_fit", "auto") not in {"auto", "interpolate"}:
        fail("acoustic topology with an interpolating fit is required.")
    if number(profile.get("s", 0), "s") != 0 or number(source.get("source_shape", 0), "source_shape") != 0:
        fail("the full unterminated body and a flat native source are required.")
    output = config.get("output", {})
    if not isinstance(output, Mapping) or set(output)-{"path", "output_path"}:
        fail("output supports only path and output_path.")
    for section in (config, output):
        for key in ("path", "output_path"):
            if key in section and section[key] is not None and not isinstance(section[key], str):
                fail(f"{key} must be a string or null.")
    clean = copy.deepcopy(dict(config))
    clean.pop("throat_adapter")
    clean["formula"] = "OSSE"; clean.pop("type", None)
    clean["mode"] = "bare"
    clean.setdefault("profile", {}).pop("formula", None); clean["profile"].pop("type", None)
    clean.setdefault("source", {}).setdefault("source_shape", 0)
    clean.setdefault("mesh", {}).setdefault("wall_thickness_mm", 0)
    params, _, mode = resolver(clean)
    params["controlsAdapter"] = payload
    params["type"] = FORMULA
    model = ControlsMeridian.from_params(params)
    params["controlsFingerprint"] = model.fingerprint
    return params, FORMULA, mode


def _positive(coefficients, name):
    """Whole-polynomial lower bound using bounded exact dyadic subdivision."""
    with localcontext() as context:
        context.prec = 80
        coefficients = [v if isinstance(v, Decimal) else Decimal.from_float(float(v)) for v in coefficients]
        allowance = Decimal("1e-60")*max(Decimal(1), *(abs(v) for v in coefficients))
        threshold = Decimal("0.0001")
        pending, lower, nodes = [(coefficients, 0)], Decimal("Infinity"), 0
        while pending:
            row, depth = pending.pop(); nodes += 1
            low, high = min(row)-allowance, max(row)+allowance
            if low > threshold:
                lower = min(lower, low)
                continue
            if high <= threshold or depth >= 32 or nodes >= 32768:
                fail(f"whole-curve {name} positivity could not be certified.")
            levels = [row]
            while len(levels[-1]) > 1:
                levels.append([(a+b)/2 for a,b in zip(levels[-1][:-1], levels[-1][1:])])
            left = [level[0] for level in levels]
            right = [level[-1] for level in reversed(levels)]
            pending.extend(((right,depth+1), (left,depth+1)))
        return math.nextafter(float(lower), -math.inf), nodes


def _validate_model_params(p):
    """Refuse supplied normalized intent before reducing the physical identity."""
    if not isinstance(p,Mapping):fail("native model parameters must be an object.")
    neutral_numbers={"q":.995,"throatExtLength":0,"throatExtAngle":0,"slotLength":0,
        "cornerSegments":0,"wallThickness":0,"encDepth":0,"morphTarget":0,
        "morphWidth":0,"morphHeight":0,"morphCorner":0,"morphExponent":2,
        "morphRate":3,"morphFixed":0,"morphAllowShrinkage":0,"gcurveType":0,
        "gcurveWidth":0,"gcurveAspectRatio":1,"gcurveDist":0,"gcurveRot":0,
        "gcurveSeN":3,"gcurveSfA":1,"gcurveSfB":1,"gcurveSfM1":4,
        "gcurveSfN1":2,"gcurveSfN2":2,"gcurveSfN3":2,"sourceShape":0,
        "sourceRadius":-1,"sourceCurv":0,"n":4,"s":0,"h":0,"rot":0}
    neutral_values={"lookupProfile":None,"samplingMode":"uniform","athParitySampling":False,
        "zMapPoints":None,"zMapKind":None,"gcurveSF":"","gcurveSf":"",
        "gcurveSfM2":None,"subdomainSlices":"","interfaceOffset":None,"interfaceResolution":None}
    geometric={"L","r0","k","a","a0","scale","verticalOffset","controlsAdapter"}
    diagnostics={"angularSegments","lengthSegments","throatResolution","mouthResolution","rearResolution"}
    extra=set(p)-geometric-diagnostics-set(neutral_numbers)-set(neutral_values)-{"type","formula","quadrants","profileSystem","controlsFingerprint"}
    if extra:fail("unsupported normalized model fields: "+", ".join(sorted(map(str,extra)))+".")
    for key in ("type","formula"):
        if key in p and (type(p[key]) is not str or p[key]!=FORMULA):fail(f"normalized {key} must be {FORMULA}.")
    for key,expected in neutral_numbers.items():
        if key in p and number(p[key],key)!=expected:fail(f"unsupported active normalized model control {key}.")
    for key,expected in neutral_values.items():
        if key in p and (type(p[key]) is not type(expected) or p[key]!=expected):
            fail(f"unsupported active normalized model control {key}.")
    for key in ("angularSegments","lengthSegments"):
        if key in p and (type(p[key]) is not int or not 0<p[key]<=10_000_000):
            fail(f"{key} must be a positive integer no larger than 10000000.")
    for key in ("throatResolution","mouthResolution","rearResolution"):
        if key in p and not .01<=number(p[key],key)<=10000:fail(f"{key} must be in [0.01,10000] mm.")
    if "quadrants" in p and not ((type(p["quadrants"]) is str and p["quadrants"]=="1234") or (type(p["quadrants"]) is int and p["quadrants"]==1234)):
        fail("normalized quadrants must be exactly 1234.")
    if "profileSystem" in p:
        system=p["profileSystem"]
        if not isinstance(system,Mapping) or set(system)!={"crossSection"}:fail("only the circular normalized profile system is supported.")
        section=system["crossSection"]
        if (not isinstance(section,Mapping) or set(section)!={"exponent","aspectRatio"}
            or number(section["exponent"],"crossSection exponent")!=2
            or number(section["aspectRatio"],"crossSection aspectRatio")!=1):
            fail("only the circular normalized profile system is supported.")


@dataclass(frozen=True)
class ControlsMeridian:
    """Physical cubic/conic authority; byte-backed arrays and detached identity."""
    length: float
    r0: float
    k: float
    tan_a: float
    tan_a0: float
    scale: float
    offset: float
    adapter_length: float
    minimum_z_derivative: float
    minimum_radius: float
    conic_residual_mm: float
    _cubic_bytes: bytes
    _body_bytes: bytes
    _weight_bytes: bytes
    _identity_json: str
    _authority_seal: str

    @staticmethod
    def _digest(values):
        return hashlib.sha256(repr(tuple(values)).encode()).hexdigest()

    def __post_init__(self):
        values=[getattr(self,f.name) for f in fields(self) if f.name!="_authority_seal"]
        if self._authority_seal!=self._digest(values):
            fail("immutable controls authority does not match its certified construction.")
        try:
            identity=json.loads(self._identity_json)
            params={**identity["body"],"scale":identity["scale"],"verticalOffset":identity["verticalOffsetMm"],
                    "controlsAdapter":identity["controls"]}
            expected=type(self).from_params(params,_values_only=True)
            if tuple(values)!=expected:
                fail("immutable controls authority does not match its certified construction.")
        except (KeyError,TypeError,ValueError,OverflowError) as exc:
            fail("immutable controls authority does not match its certified construction.")
        if max(abs(v) for row in self.bounds for v in row)>100000:
            fail("physical full envelope exceeds 100000 mm.")

    @classmethod
    def from_params(cls, p, *, _values_only=False):
        _validate_model_params(p)
        values = {key:number(p.get(key, default), key) for key,default in
                  (("L",120),("r0",12.7),("k",1),("a",60),("a0",15.5),("scale",1),("verticalOffset",0))}
        L,r0,k,a,a0,scale,Y = (values[k] for k in ("L","r0","k","a","a0","scale","verticalOffset"))
        if not (0<L<=10000 and 0<r0<=10000 and 0<k<=100 and .001<=scale<=100 and 0<=a0<a<80 and a>=1 and a-a0>=1e-6):
            fail("body, angles or uniform scale lie outside the certified domain.")
        payload = normalize_payload(p.get("controlsAdapter"))
        controls = np.asarray(payload["control_points_mm"],dtype=np.float64)
        A = -float(controls[0,0])
        if min(A*scale,L*scale,r0*scale) <= 1e-4:
            fail("adapter/body span or join radius is unresolved.")
        cubic = np.vstack((controls, [0,r0])); cubic[:,0] += A; cubic *= scale
        if not np.all(np.isfinite(cubic)) or max(abs(Y),float(np.max(np.abs(cubic))))>100000:
            fail("physical controls or placement exceed 100000 mm.")
        # Compute derivative coefficients in Decimal before any cancellation.
        with localcontext() as context:
            context.prec=80
            D=lambda v:Decimal.from_float(float(v))
            derivative=[3*(D(cubic[i+1,0])-D(cubic[i,0])) for i in range(3)]
        min_z,_=_positive(derivative,"axial derivative")
        min_r,_=_positive(cubic[:,1],"radius")
        ta,t0=math.tan(math.radians(a)),math.tan(math.radians(a0))
        K=(k*r0)**2*(1-(t0/ta)**2)
        if not K>1e-9*(k*r0)**2:
            fail("conic degeneracy could not be resolved.")
        root=math.sqrt((k*r0)**2+2*k*r0*L*t0+L*L*ta*ta)
        rM=r0+(2*k*r0*L*t0+L*L*ta*ta)/(root+k*r0)
        center=-k*r0*t0/(ta*ta)
        e0=math.asinh(ta*(-center)/math.sqrt(K));e1=math.asinh(ta*(L-center)/math.sqrt(K))
        w=math.cosh((e1-e0)/2);f=math.tanh((e1-e0)/2)**2
        poles=np.asarray([[A,r0],[A+.5*L*(1-f)+center*f,.5*(r0+rM)*(1-f)+r0*(1-k)*f],[A+L,rM]])
        weights=np.asarray([1.,w,1.])
        if not np.all(np.isfinite(poles)) or not 1<=w<1e6 or min(np.diff(poles[:,0]))*scale<=1e-4 or min(poles[:,1])*scale<=1e-4:
            fail("conic pole precision could not be certified.")
        size=max(1.,float(np.max(np.abs(poles))))
        for vec,angle in ((poles[1]-poles[0],math.atan(t0)),(poles[2]-poles[1],math.atan((k*r0*t0+L*ta*ta)/root))):
            if abs(math.atan2(vec[1],vec[0])-angle)+64*np.finfo(float).eps*size/np.linalg.norm(vec)>5e-10:
                fail("conic endpoint tangent precision could not be certified.")
        residual=_conic_residual_bound(poles,weights,length=L,join_z=0,adapter_length=A,r0=r0,k=k,tan_a=ta,tan_a0=t0)*scale
        if residual+1e-10*size*scale>1e-4:
            fail("continuous exact conic residual could not be certified.")
        identity={"contractRevision":1,"formula":FORMULA,"controls":payload,
                  "body":{key:values[key] for key in ("L","r0","k","a","a0")},
                  "scale":scale,"verticalOffsetMm":Y,"construction":"literal-cubic-full-conic-v1","source":"flat-driver-rim-v1"}
        authority=(L*scale,r0*scale,k,ta,t0,scale,Y,A*scale,min_z,min_r,residual,
                  cubic.tobytes(),(poles*scale).tobytes(),weights.tobytes(),json.dumps(identity,sort_keys=True,separators=(",",":"),allow_nan=False))
        if "controlsFingerprint" in p and (type(p["controlsFingerprint"]) is not str or p["controlsFingerprint"]!=hashlib.sha256(authority[-1].encode()).hexdigest()):
            fail("supplied controls fingerprint does not match the exact model.")
        if _values_only:return authority
        model=cls(*authority,cls._digest(authority))
        if max(abs(v) for row in model.bounds for v in row)>100000:
            fail("physical full envelope exceeds 100000 mm.")
        return model

    @property
    def cubic_poles(self):return np.frombuffer(self._cubic_bytes,dtype=np.float64).reshape(4,2)
    @property
    def body_poles(self):return np.frombuffer(self._body_bytes,dtype=np.float64).reshape(3,2)
    @property
    def body_weights(self):return np.frombuffer(self._weight_bytes,dtype=np.float64)
    @property
    def identity(self):return json.loads(self._identity_json)
    @property
    def payload(self):return self.identity["controls"]
    @property
    def fingerprint(self):return hashlib.sha256(self._identity_json.encode()).hexdigest()
    @property
    def source_radius(self):return float(self.cubic_poles[0,1])
    @property
    def source_angle_deg(self):
        d=self.cubic_poles[1]-self.cubic_poles[0]
        return math.degrees(math.atan2(d[1],d[0]))
    @property
    def depth(self):return self.adapter_length+self.length
    @property
    def join_jump_deg(self):
        d=self.cubic(1)[1]
        return math.degrees(math.atan(self.tan_a0)-math.atan2(d[1],d[0]))

    def cubic(self,u):
        u=self._parameter(u);v=1-u;P=self.cubic_poles
        point=v[...,None]**3*P[0]+3*v[...,None]**2*u[...,None]*P[1]+3*v[...,None]*u[...,None]**2*P[2]+u[...,None]**3*P[3]
        D=3*np.diff(P,axis=0);E=2*np.diff(D,axis=0)
        first=v[...,None]**2*D[0]+2*v[...,None]*u[...,None]*D[1]+u[...,None]**2*D[2]
        second=v[...,None]*E[0]+u[...,None]*E[1]
        return point,first,second

    def body(self,t):
        t=self._parameter(t);z=t*self.length
        A=(self.k*self.r0)**2;B=self.k*self.r0*self.tan_a0;C=self.tan_a**2
        root=np.sqrt(A+2*B*z+C*z*z)
        radius=self.r0+(2*B*z+C*z*z)/(root+self.k*self.r0)
        d=(B+C*z)/root;dd=(A*C-B*B)/root**3
        return np.stack((self.adapter_length+z,radius),axis=-1),np.stack((np.full_like(t,self.length),self.length*d),axis=-1),np.stack((np.zeros_like(t),self.length**2*dd),axis=-1)

    def branch(self,name,u):
        if name not in {"cubic","body"}:fail("unknown native branch.")
        return self.cubic(u) if name=="cubic" else self.body(u)

    @staticmethod
    def _parameter(value):
        try:
            raw=np.asarray(value)
            if raw.dtype.kind not in "fiu":fail("branch parameter must be numeric in [0,1].")
            result=raw.astype(np.float64,copy=False)
            if not np.all(np.isfinite(result)) or np.any((result<0)|(result>1)):
                fail("branch parameter must be finite in [0,1].")
            return result
        except (OverflowError,TypeError,ValueError) as exc:
            fail("branch parameter must be finite in [0,1].")

    def interval_bounds(self,name,lo,hi):
        """Outward whole-chart derivative/radius hulls on finite intervals."""
        lo,hi=np.asarray(lo),np.asarray(hi);h=hi-lo
        p0,d0,e0=self.branch(name,lo);p1,d1,e1=self.branch(name,hi)
        guard=1e-10*max(1.,self.depth,float(np.max(np.abs(self.cubic_poles))),self.r0)
        if name=="cubic":
            P=np.stack((p0,p0+h[...,None]*d0/3,p1-h[...,None]*d1/3,p1),axis=-2)
            D=np.stack((d0,d0+h[...,None]*e0/2,d1),axis=-2)
            Muu=np.maximum(np.linalg.norm(e0,axis=-1),np.linalg.norm(e1,axis=-1))+guard
            Mr=np.max(np.abs(D[...,1]),axis=-1)+guard
            rmax=np.max(P[...,1],axis=-1)+guard
            rmin=np.min(P[...,1],axis=-1)-guard
            zmin=np.min(D[...,0],axis=-1)-guard
        else:
            Muu=np.linalg.norm(e0,axis=-1)+guard
            Mr=np.abs(d1[...,1])+guard
            rmax=p1[...,1]+guard;rmin=p0[...,1]-guard;zmin=self.length-guard
        return Muu,Mr,rmax,rmin,zmin

    def sizing_speed_bound(self,name,lo,hi):
        """Fixed derivative-hull projection bound for heuristic mesh sizing."""
        lo,hi=np.asarray(lo),np.asarray(hi);h=hi-lo
        _,d0,e0=self.branch(name,lo);_,d1,_=self.branch(name,hi)
        guard=1e-10*max(1.,self.depth,float(np.max(np.abs(self.cubic_poles))),self.r0)
        if name=="cubic":
            D=np.stack((d0,d0+h[...,None]*e0/2,d1),axis=-2)
            low=np.min(D,axis=-2)-guard;high=np.max(D,axis=-2)+guard
        else:
            low=np.minimum(d0,d1)-guard;high=np.maximum(d0,d1)+guard
        # A same-sign component interval has a positive distance from zero.
        # The signed nearest-to-zero box vector gives a fixed separating
        # direction: every hull derivative projects at least its norm along
        # that direction. Integration therefore also bounds finite meridian
        # chord displacement, including steep non-axial intervals. A box
        # containing the origin gives no positive bound and is subdivided.
        component=np.maximum(0.,np.maximum(low,-high))
        return np.maximum(0.,np.linalg.norm(component,axis=-1)-guard)

    @property
    def radial_extrema(self):
        """Decimal quadratic derivative roots bound the exact cubic envelope."""
        with localcontext() as context:
            context.prec=80
            P=[Decimal.from_float(float(v)) for v in self.cubic_poles[:,1]]
            a=3*(-P[0]+3*P[1]-3*P[2]+P[3]);b=6*(P[0]-2*P[1]+P[2]);c=3*(P[1]-P[0])
            roots=[]
            if a==0:
                if b!=0:roots=[-c/b]
            elif b*b-4*a*c>=0:
                root=(b*b-4*a*c).sqrt();roots=[(-b-root)/(2*a),(-b+root)/(2*a)]
            values=[P[0],P[3]]
            for u in roots:
                if 0<u<1:
                    v=1-u;values.append(v**3*P[0]+3*v*v*u*P[1]+3*v*u*u*P[2]+u**3*P[3])
            return float(min(values)),math.nextafter(float(max(values)),math.inf)

    @property
    def reach(self):return max(self.radial_extrema[1],float(self.body(1)[0][1]))
    @property
    def bounds(self):
        r=self.reach
        return [[-r,self.offset-r,0.],[r,self.offset+r,self.depth]]
    def metadata(self):
        return {"construction_fingerprint":self.fingerprint,"controlsAdapter":{"contractRevision":1,
            "controls":self.payload,"sourceRadiusMm":self.source_radius,"sourceAngleDeg":self.source_angle_deg,
            "joinPlaneMm":self.adapter_length,"bodyLengthMm":self.length,"depthMm":self.depth,
            "joinTangentJumpDeg":self.join_jump_deg,"joinAbsoluteTangentJumpDeg":abs(self.join_jump_deg),
            "minimumRadiusBoundMm":self.minimum_radius,"minimumAxialDerivativeBoundMm":self.minimum_z_derivative,
            "conicResidualBoundMm":self.conic_residual_mm,"boundsMm":self.bounds,"fingerprint":self.fingerprint}}


def validate_density(density):
    from .geometry import MeshDensity
    if density is None:return MeshDensity()
    if type(density) is not MeshDensity:fail("density must be MeshDensity or None.")
    for key in ("throat_res_mm","mouth_res_mm","rear_res_mm"):
        if not .01<=number(getattr(density,key),key)<=10000:fail(f"{key} must be in [0.01,10000] mm.")
    if type(density.max_triangles) is not int or not 0<density.max_triangles<=10_000_000:
        fail("max_triangles must be a positive integer no larger than 10000000.")
    if type(density.allow_large_mesh) is not bool:fail("allow_large_mesh must be a boolean.")
    default=MeshDensity()
    for field in fields(default):
        if field.name in {"throat_res_mm","mouth_res_mm","rear_res_mm","max_triangles","allow_large_mesh"}:continue
        value,expected=getattr(density,field.name),getattr(default,field.name)
        if type(value) is not type(expected) or value!=expected:fail(f"unsupported active density control {field.name}.")
    return density


def validate_geometry(g):
    if type(g.controls_meridian) is not ControlsMeridian:fail("native geometry requires its immutable ControlsMeridian.")
    m=g.controls_meridian
    m.__post_init__()
    for key in ("wall_thickness_mm","source_shape","source_radius_mm","source_curv","source_auto_angle_deg","interface_offset_mm","vertical_offset_mm"):
        number(getattr(g,key),key)
    if any(getattr(g,key,None) is not None for key in ("adapter_meridian","roundover","axial_model","arc_meridian")):
        fail("combined active native geometry features are not supported.")
    if (g.topology_mode!="acoustic" or g.surface_fit not in {"auto","interpolate"} or g.preserve_grid is not False
        or g.closed is not True or g.outer_points is not None or g.wall_thickness_mm!=0
        or g.source_shape!=0 or g.source_radius_mm!=-1 or g.source_curv!=0
        or g.source_auto_angle_deg!=m.source_angle_deg or g.interface_offset_mm!=0 or g.interfaces
        or g.wg_topology is not True or g.enclosure is not None or g.infinite_baffle is not False
        or g.symmetry_planes or g.vertical_offset_mm!=m.offset or g.freeform_axis_samples_mm is not None or g.freeform_report is not None):
        fail("direct native geometry must retain canonical placement, source, topology and exact authority.")


def resolve(config,params,allow_large_mesh=None):
    from .config_builder import ResolvedGeometry,_mesh_density_from_config
    from .geometry import _ControlsPointGridHornGeometry
    if allow_large_mesh is not None and type(allow_large_mesh) is not bool:fail("allow_large_mesh override must be a boolean or None.")
    m=ControlsMeridian.from_params(params)
    meridian=np.vstack((m.cubic(np.linspace(0,1,17))[0],m.body(np.linspace(0,1,33)[1:])[0]))
    az=np.arange(64)*2*math.pi/64
    grid=np.stack((np.cos(az)[:,None]*meridian[:,1],np.sin(az)[:,None]*meridian[:,1],np.broadcast_to(meridian[:,0],(64,len(meridian)))),axis=-1)
    grid=np.frombuffer(grid.tobytes(),dtype=np.float64).reshape(grid.shape)
    g=_ControlsPointGridHornGeometry(inner_points=grid,wall_thickness_mm=0,source_shape=0,source_radius_mm=-1,
        source_curv=0,source_auto_angle_deg=m.source_angle_deg,closed=True,symmetry_planes=(),vertical_offset_mm=m.offset,controls_meridian=m)
    density=validate_density(_mesh_density_from_config(config,allow_large_mesh=allow_large_mesh))
    return ResolvedGeometry(g,density,FORMULA,"bare","1234",None,config.get("mesh",{}).get("scale_to_metres",True),m.metadata())
