"""One exact native meridian: optional throat cubic, retained conic, mouth circle."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from .config_parser import ConfigError
from .throat_adapter import _conic_residual_bound


def _fail(message):
    raise ConfigError("Terminating arc refused: " + message)


def _number(value, name):
    try:
        valid = type(value) in (int, float) and math.isfinite(float(value))
    except (ValueError, TypeError, OverflowError):
        valid = False
    if not valid:
        _fail(f"{name} must be a finite scalar number.")
    return float(value)


def mapped_throat(params):
    """The mapped body, the composite's throat cubic and its composite-z shift.

    The body is AxialModel's, mapped by ``absoluteAxialScale`` (else the
    uniform ``scale``). The throat is the OSSE-ADAPTER-AXIAL PhysicalAdapter
    with the same map, or None; its bare cubic ignores any wall. The retained
    conic starts at body z0 = length*join_t, which sits at composite z
    z0+shift, the cubic's exact end. Returns (body, throat, z0, shift).
    """
    from .axial_scale import AxialModel
    axial = params.get("absoluteAxialScale", params.get("scale", 1))
    body = AxialModel.from_params(dict(params, absoluteAxialScale=axial))
    if not params.get("throat_adapter"):
        return body, None, 0., 0.
    from .adapter_axial import KEY, PhysicalAdapter
    throat = PhysicalAdapter.from_params(dict(params, type="OSSE", wallThickness=0, **{KEY: axial}))
    z0 = body.length*throat.payload["join_t"]
    return body, throat, z0, float(throat.cubic[-1, 0])-z0


@dataclass(frozen=True)
class CircleArc:
    """Meridian circle (z,r) = center+radius*(sin phi,-cos phi), phi in [start,end].

    Radius grows with phi on [start,end] within [0,pi]; both the terminating
    arc and the mouth lip satisfy this, and the local reach below relies on it.
    """
    center_z: float
    center_r: float
    radius: float
    start: float
    end: float

    def points(self, phi):
        phi = np.asarray(phi, dtype=np.float64)
        return np.stack((self.center_z+self.radius*np.sin(phi), self.center_r-self.radius*np.cos(phi)), axis=-1)


@dataclass(frozen=True)
class CompositeMeridian:
    """Immutable physical meridian; arrays are backed by immutable bytes.

    Build order follows the composition contract: the intrinsic conic, the
    throat feature, the axial map, then the mouth circle in finished mm. The
    optional throat is the mapped adapter cubic on z in [0,poles[0,0]]. The
    retained conic uses physical body coordinate z-shift; without a throat
    shift is zero and the conic starts at z=0. The circle uses the absolute
    oriented tangent angle phi in [beta,end]. All lengths are mm.
    """
    length: float
    r0: float
    k: float
    slope: float
    throat_slope: float
    join_t: float
    end: float
    offset: float
    radius: float
    beta: float
    center_z: float
    center_r: float
    _poles: bytes
    _weights: bytes
    _identity: str
    shift: float = 0.0
    throat: object | None = None

    @classmethod
    def from_params(cls, params):
        # The same mapped cubic and checks as OSSE-ADAPTER-AXIAL; its conic
        # is replaced here by the span ending at the circle join.
        body, throat, z0, shift = mapped_throat(params)
        payload = params["terminatingArc"]
        t = _number(payload["join_t"], "join_t")
        end = math.radians(_number(payload["end_tangent_deg"], "end_tangent_deg"))
        if not 0 < t <= 1 or body.length*t < .1:
            _fail("join_t requires 0 < join_t <= 1 and a retained body of at least 0.1 mm.")
        z = body.length*t
        if throat is None:
            start, start_slope = np.array([0, body.r0]), body.throat_slope
        else:
            start = np.array(throat.cubic[-1])
            start_slope = float(body.body(z0)[1][1])
            if z-z0 < .1:
                _fail("join_t must exceed the adapter join_t and retain at least 0.1 mm of conic.")
        J, tangent = body.body(z)
        J = J+np.array([shift, 0.])
        d = float(tangent[1])
        dd = body.second_bound(z, z)
        if not math.isfinite(dd) or dd <= 0:
            _fail("positive body curvature could not be resolved.")
        beta = math.atan(d)
        radius = math.hypot(1, d)**3/dd
        if not math.isfinite(radius) or not .1 <= radius <= 10000:
            _fail("curvature-matched radius must be in [0.1, 10000] mm.")
        if not beta < end <= math.pi or radius*(end-beta) < .1:
            _fail("end_tangent_deg must exceed the join tangent, be at most 180, and retain 0.1 mm of arc.")
        cz, cr = float(J[0])-radius*math.sin(beta), float(J[1])+radius*math.cos(beta)
        end_z = cz+radius*math.sin(end)
        end_r = cr-radius*math.cos(end)
        if end_z < .1 or end_r > 10000:
            _fail("the complete arc must clear the source by 0.1 mm and its radius must not exceed 10000 mm.")
        # Exact rational hyperbola segment, from its hyperbolic midpoint.
        ta, t0, kr = body.slope, body.throat_slope, body.k*body.r0
        K = kr*kr*(1-(t0/ta)**2)
        if K <= 1e-9*kr*kr:
            _fail("the conic is too close to degeneracy to certify.")
        center = -kr*t0/(ta*ta)
        eta0 = math.asinh(ta*(z0-center)/math.sqrt(K))
        eta1 = math.asinh(ta*(z-center)/math.sqrt(K))
        weight = math.cosh((eta1-eta0)/2)
        fraction = math.tanh((eta1-eta0)/2)**2
        poles = np.array([start, [.5*(z0+z)*(1-fraction)+center*fraction+shift,
            .5*(float(start[1])+float(J[1]))*(1-fraction)+body.r0*(1-body.k)*fraction], J])
        weights = np.array([1., weight, 1.])
        size = max(1., body.length, radius, abs(cz), cr, end_r, abs(body.offset)+end_r)
        if throat is not None:
            size = max(size, float(np.max(np.abs(throat.cubic))), abs(body.offset)+throat.cubic_reach)
            # A returning circle below the adapter join has phi past the
            # crossing of that plane, so its radius is at least the radius
            # there; every cubic point is within its reach.
            if end_z < float(start[0]):
                crossing = math.pi-math.asin(min(1., (float(start[0])-cz)/radius))
                if cr-radius*math.cos(crossing)-throat.cubic_reach < .1:
                    _fail("the returning arc must clear the throat adapter by 0.1 mm.")
        margin=1e-10*size
        if min(z-z0,radius*(end-beta),end_z) <= .1+margin:
            _fail("the retained span, arc or source-plane clearance could not be certified above 0.1 mm.")
        if size > 100000 or np.min(np.diff(poles[:,0])) <= 1e-4 or np.min(poles[:,1]) <= 1e-4:
            _fail("exact conic controls or complete coordinates exceed the qualified precision range.")
        error = _conic_residual_bound(poles, weights, length=body.length, join_z=0,
            adapter_length=shift, r0=body.r0, k=body.k, tan_a=ta, tan_a0=t0)
        if error+1e-10*size > 1e-4:
            _fail("continuous rational-conic position precision could not be certified.")
        vectors = (poles[1]-poles[0], poles[2]-poles[1])
        for vector, angle in zip(vectors, (math.atan(start_slope), beta)):
            margin = 64*np.finfo(float).eps*size/np.linalg.norm(vector)
            if abs(math.atan2(vector[1],vector[0])-angle)+margin > 5e-10:
                _fail("conic endpoint tangent precision could not be certified.")
        identity = {"contractRevision":1,"construction":"circular-osse-osculating-arc-v1",
            "formula":"OSSE-ARC","authoredBody":{key:float(params[key]) for key in ("L","r0","a","a0","k","scale")},
            "body":{"lengthMm":body.length,"throatRadiusMm":body.r0,"k":body.k,
                    "slope":ta,"throatSlope":t0},
            "arc":{"joinT":t,"endTangentDeg":float(payload["end_tangent_deg"])},
            "verticalOffsetMm":body.offset,"source":"flat-throat-rim-v1"}
        if params.get("type") == "OSSE-NATIVE":
            identity["formula"] = "OSSE-NATIVE"
            # Only an authored axial feature names its factor; otherwise the
            # map is the uniform scale already recorded with the body.
            if "absoluteAxialScale" in params:
                identity["authoredBody"]["axial_scale"] = body.axial_scale
        if throat is not None:
            identity["construction"] = "circular-cubic-conic-osculating-arc-v1"
            identity["adapter"] = dict(throat.payload)
            identity["source"] = "flat-driver-rim-v1"
        return cls(body.length,body.r0,body.k,ta,t0,t,end,body.offset,radius,beta,cz,cr,
            poles.tobytes(),weights.tobytes(),json.dumps(identity,sort_keys=True,separators=(",",":"),allow_nan=False),
            shift,throat)

    @property
    def poles(self):
        return np.frombuffer(self._poles,dtype=np.float64).reshape(3,2)

    @property
    def weights(self):
        return np.frombuffer(self._weights,dtype=np.float64)

    @property
    def identity(self):
        return json.loads(self._identity)

    @property
    def fingerprint(self):
        return hashlib.sha256(self._identity.encode()).hexdigest()

    @property
    def join_z(self):
        return self.length*self.join_t+self.shift

    @property
    def conic_start_z(self):
        """Exact first pole: the throat plane, or the adapter join."""
        return float(self.poles[0,0])

    @property
    def body_mouth_z(self):
        return self.length+self.shift

    @property
    def source_radius(self):
        return self.r0 if self.throat is None else self.throat.driver_radius

    @property
    def source_angle_deg(self):
        return math.degrees(math.atan(self.throat_slope)) if self.throat is None else self.throat.source_angle

    @property
    def coefficients(self):
        return (self.k*self.r0)**2, self.k*self.r0*self.throat_slope, self.slope**2

    def body(self,z):
        """Retained conic at composite z; the throat cubic is separate."""
        z=np.asarray(z,dtype=np.float64)
        u=z-self.shift
        A,B,C=self.coefficients
        root=np.sqrt(A+2*B*u+C*u*u)
        p=np.stack((z,self.r0+(2*B*u+C*u*u)/(root+self.k*self.r0)),axis=-1)
        tangent=np.stack((np.ones_like(z),(B+C*u)/root),axis=-1)
        return p,tangent

    def second_bound(self,lo,hi):
        A,B,C=self.coefficients
        lo=lo-self.shift
        return (A*C-B*B)/(A+2*B*lo+C*lo*lo)**1.5

    @property
    def circle(self):
        return CircleArc(self.center_z,self.center_r,self.radius,self.beta,self.end)

    def arc(self,phi):
        return self.circle.points(phi)

    @property
    def opening(self):
        return self.arc(self.end)

    @property
    def reach(self):
        return float(self.opening[1])

    @property
    def envelope_radius(self):
        """Largest radius anywhere; the circle opening unless a throat bulges."""
        return self.reach if self.throat is None else max(self.reach,self.throat.cubic_reach)

    @property
    def bounds(self):
        zmax=self.center_z+self.radius if self.beta <= math.pi/2 <= self.end else float(self.opening[0])
        r=self.envelope_radius
        return [[-r,self.offset-r,0.],[r,self.offset+r,zmax]]

    def metadata(self):
        arc={**self.identity,"fingerprint":self.fingerprint,"joinMm":self.body(self.join_z)[0].tolist(),
            "openingMm":self.opening.tolist(),"nominalBodyMouthZMm":self.body_mouth_z,
            "radiusMm":self.radius,"joinTangentDeg":math.degrees(self.beta),"boundsMm":self.bounds,
            "conicPositionBoundMm":0.0001}
        if self.throat is not None:
            arc.update(adapterJoinMm=self.poles[0].tolist(),sourceRadiusMm=self.source_radius,
                sourceAngleDeg=self.source_angle_deg)
        return {"construction_fingerprint":self.fingerprint,"terminatingArc":arc}
