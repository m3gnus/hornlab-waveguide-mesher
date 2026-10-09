"""Canonical rotational wall fitted by the existing resolved-grid axial authority.

The exact circular revolution intentionally removes the old tensor patch's
angular circle approximation. It never replaces the nonlinear meridian by a cone.
"""

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline
from scipy.spatial import cKDTree

FEATURE = "native-general-horn-attachment-v1"
CHORD_TOLERANCE_MM = 1e-7


@dataclass(frozen=True)
class GeneralHornWall:
    degree: int
    knots: tuple[float, ...]
    poles_mm: tuple[tuple[float, float], ...]  # radius, axial distance from throat
    formula: str
    circle_correction_bound_mm: float = 0.0

    def __post_init__(self):
        if (type(self.circle_correction_bound_mm) not in (int, float)
                or not math.isfinite(self.circle_correction_bound_mm)
                or not 0 <= self.circle_correction_bound_mm <= .15):
            raise ValueError("original angular circle fit correction exceeds 0.15 mm; refine the design angular sampling before attachment; for acoustic fits use finer throat/mouth resolutions in millimetres (AngularSegments may be overridden)")
        if type(self.degree) is not int or not 1 <= self.degree <= 3:
            raise ValueError("general horn spline degree must be 1, 2 or 3")
        if (not isinstance(self.knots, (tuple, list)) or not isinstance(self.poles_mm, (tuple, list))
                or len(self.knots) > 4100
                or any(type(x) not in (int, float) for x in self.knots)
                or any(not isinstance(p, (tuple, list)) or len(p) != 2
                       or any(type(x) not in (int, float) for x in p) for p in self.poles_mm)):
            raise ValueError("general horn spline coordinates must be finite numeric arrays")
        knots, poles = np.asarray(self.knots), np.asarray(self.poles_mm)
        if (knots.ndim != 1 or poles.ndim != 2 or poles.shape[1] != 2
                or not 2 <= len(poles) <= 4096
                or len(knots) != len(poles) + self.degree + 1
                or not np.isfinite(knots).all() or not np.isfinite(poles).all()
                or np.any(np.diff(knots) < 0)
                or tuple(knots[:self.degree + 1]) != (0,) * (self.degree + 1)
                or tuple(knots[-self.degree - 1:]) != (1,) * (self.degree + 1)
                or np.any(poles[:, 0] <= 0) or abs(poles[0, 1]) > 1e-9):
            raise ValueError("invalid canonical general horn spline")
        if self.formula not in ("OSSE", "ROSSE", "R-OSSE", "ICW", "FREEFORM"):
            raise ValueError("unsupported general horn family")
        object.__setattr__(self, "knots", tuple(float(x) for x in knots))
        object.__setattr__(self, "poles_mm", tuple(tuple(float(x) for x in p) for p in poles))
        _, multiplicities = np.unique(knots[self.degree + 1:-self.degree - 1], return_counts=True)
        if np.any(multiplicities > self.degree):
            raise ValueError("general horn interior knots must preserve meridian continuity")
        # A monotone meridian coordinate excludes self intersections while
        # allowing ordinary R-OSSE axial rollback with increasing radius.
        forward_z = all(np.min(np.diff(b[:, 1])) >= -1e-10 and b[-1, 1] > b[0, 1]
                        for b in self.beziers)
        forward_r = all(np.min(np.diff(b[:, 0])) >= -1e-10 and b[-1, 0] > b[0, 0]
                        for b in self.beziers)
        if not (forward_z or forward_r):
            # A rotated monotone coordinate proves a simple meridian even when
            # the existing interpolating R-OSSE lip reverses both R and Z near
            # its end. This certifies all Bezier tangent convex hulls, not samples.
            from scipy.optimize import linprog
            tangents = np.concatenate([np.diff(b, axis=0) for b in self.beziers])
            inequalities = np.column_stack((-tangents, np.ones(len(tangents))))
            result = linprog([0, 0, -1], A_ub=inequalities, b_ub=np.zeros(len(tangents)),
                             bounds=[(-1, 1), (-1, 1), (0, None)], method="highs")
            if (not result.success or result.x[2] <= 1e-12
                    or float(np.min(tangents @ result.x[:2])) <= 1e-12):
                raise ValueError("general horn attachment requires a provably simple meridian; this folded curve has no monotone projection")

    @classmethod
    def from_config(cls, config):
        from .config_builder import resolve_geometry
        from .geometry import PointGridHornGeometry
        from .builders._occ import grid_v_parameters

        resolved = resolve_geometry(config)
        g = resolved.geometry
        if (not isinstance(g, PointGridHornGeometry) or not g.closed
                or any(getattr(g, name, None) is not None for name in ("adapter_meridian", "axial_model", "roundover", "terminating_arc"))
                or g.outer_points is not None or g.enclosure is not None
                or g.infinite_baffle or g.interfaces or g.vertical_offset_mm
                or g.topology_mode != "acoustic"):
            raise ValueError("general horn attachment requires a full bare resolved horn; existing shells, interfaces, placement, symmetry and authored terminals must be composed explicitly")
        xyz = np.asarray(g.inner_points)
        r, z = np.linalg.norm(xyz[..., :2], axis=-1), xyz[..., 2]
        if np.max(np.ptp(r, axis=0)) > 1e-8 or np.max(np.ptp(z, axis=0)) > 1e-8:
            raise ValueError("general horn attachment requires a rotational wall and circular coaxial rims; noncircular or azimuth-varying geometry is unsupported")
        # Also bind the axis and angular rays; equal radii alone cannot prove this.
        directions = xyz[..., :2] / r[..., None]
        if (np.max(np.ptp(directions, axis=1)) > 1e-9
                or np.linalg.norm(xyz[:, 0, :2].mean(axis=0)) > 1e-8):
            raise ValueError("general horn attachment requires coaxial circular rings")
        meridian = np.column_stack((r[0], z[0] - z[0, 0]))
        degree = min(3, len(meridian) - 1)
        if g.surface_fit == "interpolate":
            # Identical v parameters and axial interpolation as add_bspline_patch.
            curve = make_interp_spline(grid_v_parameters(xyz), meridian, k=degree)
        else:
            # OCC's default clamped uniform approximating fit; do not interpolate
            # FREEFORM creases or change its established fitting policy.
            distinct = np.linspace(0, 1, len(meridian) - degree + 1)
            knots = np.r_[np.zeros(degree), distinct, np.ones(degree)]
            curve = BSpline(knots, meridian, degree)
        # Parameter normalization preserves the geometric spline and its fit.
        normalized_knots = curve.t / curve.t[-1]
        correction = _angular_correction_bound(xyz, g.surface_fit, float(curve.c[:, 0].max()))
        return cls(degree, tuple(float(x) for x in normalized_knots),
                   tuple(tuple(float(x) for x in p) for p in curve.c), resolved.formula, correction)

    def to_dict(self):
        return {"version": 1, "degree": self.degree, "knots": self.knots,
                "poles_mm": self.poles_mm, "formula": self.formula,
                "circle_correction_bound_mm": self.circle_correction_bound_mm}

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        version = value.pop("version", None)
        if type(version) is not int or version != 1:
            raise ValueError("unsupported general horn wall version")
        # Validate the version independently rather than accepting future payloads.
        return cls(**value)

    @cached_property
    def spline(self):
        return BSpline(self.knots, np.asarray(self.poles_mm), self.degree)

    @cached_property
    def beziers(self):
        # Exact knot insertion yields each span's own control polygon, including
        # C0 joins where the left and right endpoint derivatives differ.
        curve = self.spline
        values, counts = np.unique(self.knots, return_counts=True)
        for value, count in zip(values[1:-1], counts[1:-1]):
            if count < self.degree:
                curve = curve.insert_knot(float(value), self.degree - int(count))
        return tuple(np.asarray(curve.c[i * self.degree:i * self.degree + self.degree + 1])
                     for i in range(len(values) - 1))

    @property
    def length_mm(self):
        return self.poles_mm[-1][1]

    @property
    def throat_radius_mm(self):
        return self.poles_mm[0][0]

    @property
    def mouth_radius_mm(self):
        return self.poles_mm[-1][0]

    def evaluate(self, t):
        return self.spline(t)

    @cached_property
    def chords(self):
        """Every curve point is within a fixed bound of these finite chords.

        Bezier convex hull plus de Casteljau subdivision proves the bound;
        it is not inferred from sampled midpoint residuals.
        """
        result = []
        pending = list(self.beziers)
        while pending:
            controls = pending.pop()
            a, b = controls[0], controls[-1]
            direction = b - a
            t = np.clip((controls - a) @ direction / (direction @ direction), 0, 1)
            bound = np.linalg.norm(controls - a - t[:, None] * direction, axis=1).max()
            if bound <= CHORD_TOLERANCE_MM:
                result.append((a, b))
                continue
            layers, left, right = controls.copy(), [a], [b]
            while len(layers) > 1:
                layers = (layers[:-1] + layers[1:]) / 2
                left.append(layers[0])
                right.append(layers[-1])
            pending.extend((np.asarray(left), np.asarray(right[::-1])))
            if len(pending) + len(result) > 200000:
                raise ValueError("general horn certificate subdivision budget exceeded")
        return np.asarray(result)

    @cached_property
    def chord_tree(self):
        return cKDTree(self.chords.mean(axis=1))

    def distance(self, xyz, origin):
        """Conservative finite-surface distance upper bound, at most 0.1 micron fit slack."""
        local = np.asarray(xyz) - np.asarray(origin)
        rz = np.stack((np.linalg.norm(local[..., :2], axis=-1), local[..., 2]), axis=-1)
        flat = rz.reshape(-1, 2)
        _, candidates = self.chord_tree.query(flat, k=min(8, len(self.chords)))
        candidates = np.asarray(candidates).reshape(len(flat), -1)
        selected = self.chords[candidates]
        a, direction = selected[..., 0, :], selected[..., 1, :] - selected[..., 0, :]
        t = np.clip(np.sum((flat[:, None] - a) * direction, axis=-1)
                    / np.sum(direction * direction, axis=-1), 0, 1)
        distance = np.linalg.norm(flat[:, None] - a - t[..., None] * direction, axis=-1).min(axis=1)
        return (distance + CHORD_TOLERANCE_MM).reshape(rz.shape[:-1])

    @cached_property
    def area_mm2(self):
        from scipy.integrate import quad
        return sum(quad(lambda t: 2 * math.pi * self.spline(t)[0]
                        * np.linalg.norm(self.spline(t, 1)), a, b,
                        epsabs=1e-7, epsrel=1e-10)[0]
                   for a, b in zip(np.unique(self.knots), np.unique(self.knots)[1:]))

    def minimum_body_clearance(self, body):
        """Lower bound to every finite body edge, including inlet/outlet faces."""
        zmid = (body.z0_mm + body.z1_mm) / 2
        if body.radius(zmid) >= self.radius_at_z(zmid):
            raise ValueError("passive body lies outside the general horn")
        a, b = self.chords[:, 0], self.chords[:, 1]
        direction = b - a
        best = math.inf
        for c, d in body.edges.values():
            c, d = np.asarray(c), np.asarray(d)
            edge = d - c
            for point in (c, d):
                u = np.clip(np.sum((point - a) * direction, axis=-1) / np.sum(direction**2, axis=-1), 0, 1)
                best = min(best, float(np.linalg.norm(point - a - u[:, None] * direction, axis=1).min()))
            for point in (a, b):
                u = np.clip((point - c) @ edge / (edge @ edge), 0, 1)
                best = min(best, float(np.linalg.norm(point - c - u[:, None] * edge, axis=1).min()))
            # Intersections make endpoint distances insufficient in 2D.
            cross = lambda x, y: x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
            denominator = cross(direction, edge)
            safe = abs(denominator) > 1e-20
            u = np.divide(cross(c - a, edge), denominator, out=np.full(len(a), -1.), where=safe)
            v = np.divide(cross(c - a, direction), denominator, out=np.full(len(a), -1.), where=safe)
            if np.any(safe & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)):
                return 0.
        return best - CHORD_TOLERANCE_MM

    def radii_at_z(self, z):
        from scipy.interpolate import PPoly
        polynomial = PPoly.from_spline(BSpline(self.knots, np.asarray(self.poles_mm)[:, 1], self.degree))
        polynomial.c[-1] -= z
        roots = polynomial.roots(extrapolate=False)
        roots = roots[np.isfinite(roots) & (roots >= 0) & (roots <= 1)]
        if not len(roots):
            raise ValueError("passage plane does not intersect the general horn")
        # Rolled mouths may cross a plane twice. The inner branch bounds the
        # open passage; all branches enter minimum_body_clearance independently.
        return self.spline(roots)[:, 0]

    def radius_at_z(self, z):
        return float(self.radii_at_z(z).min())

    def add_occ_face(self, occ, origin):
        from .builders._occ import _knots_and_multiplicities
        cx, cy, cz = origin
        points = [occ.addPoint(cx + r, cy, cz + z) for r, z in self.poles_mm]
        knots, multiplicities = _knots_and_multiplicities(np.asarray(self.knots))
        edge = occ.addBSpline(points, degree=self.degree, knots=knots, multiplicities=multiplicities)
        return next(tag for dim, tag in occ.revolve([(1, edge)], cx, cy, cz, 0, 0, 1, 2 * math.pi) if dim == 2)


def wall_face_area(gmsh, face, wall):
    """Integrate reopened rotational CAD derivatives on every axial knot span.

    OCC getMass uses a coarse default integration for spline revolutions. Keep
    canonical area admission at its existing tolerance by measuring the actual
    CAD surface with converged quadrature instead of widening that tolerance.
    """
    lo, hi = map(np.asarray, gmsh.model.getParametrizationBounds(2, face))
    if abs((hi[0] - lo[0]) - 2 * math.pi) > 1e-8 or abs(hi[1] - lo[1] - 1) > 1e-8:
        raise ValueError("general horn CAD surface has incompatible revolution parameters")
    previous = None
    for count in (16, 32, 64, 128):
        nodes, weights = np.polynomial.legendre.leggauss(count)
        total = 0.
        for a, b in zip(np.unique(wall.knots), np.unique(wall.knots)[1:]):
            v = (a + b) / 2 + nodes * (b - a) / 2
            uv = np.column_stack((np.full(len(v), (lo[0] + hi[0]) / 2), lo[1] + v))
            derivatives = np.asarray(gmsh.model.getDerivative(2, face, uv.reshape(-1))).reshape(-1, 6)
            jacobian = np.linalg.norm(np.cross(derivatives[:, :3], derivatives[:, 3:]), axis=1)
            total += float(weights @ jacobian) * (b - a) / 2 * (hi[0] - lo[0])
        if previous is not None and abs(total - previous) <= 1e-10 * max(1, total):
            return float(total)
        previous = total
    raise ValueError("general horn CAD area quadrature did not converge")


def _angular_correction_bound(xyz, surface_fit, maximum_radius):
    """Convex-hull bound for replacing the old tensor angular fit by a circle."""
    from .builders.point_grid_surfaces import _bspline_patch_phi_groups
    bound = 0.
    for indices in _bspline_patch_phi_groups(len(xyz), closed=True):
        ring = xyz[indices, 0, :2]
        ring = ring / np.linalg.norm(ring, axis=1)[:, None]
        degree = min(3, len(ring) - 1)
        if surface_fit == "interpolate":
            lengths = np.linalg.norm(np.diff(ring, axis=0), axis=1)
            params = np.r_[0, np.cumsum(lengths)] / lengths.sum()
            curve = make_interp_spline(params, ring, k=degree)
        else:
            knots = np.r_[np.zeros(degree), np.linspace(0, 1, len(ring) - degree + 1), np.ones(degree)]
            curve = BSpline(knots, ring, degree)
        pending = []
        for a, b in zip(np.unique(curve.t), np.unique(curve.t)[1:]):
            h, first, last = b - a, curve(a), curve(b)
            controls = ([first, last] if degree == 1 else
                        [first, first + h / 2 * curve(a, 1), last] if degree == 2 else
                        [first, first + h / 3 * curve(a, 1), last - h / 3 * curve(b, 1), last])
            pending.append(np.asarray(controls))
        work = 0
        while pending:
            controls = pending.pop()
            work += 1
            if work > 200000:
                raise ValueError("general horn angular correction certificate budget exceeded")
            a, b = controls[0], controls[-1]
            if max(abs(np.linalg.norm(a) - 1), abs(np.linalg.norm(b) - 1)) * maximum_radius > .15:
                raise ValueError("original angular circle fit correction exceeds 0.15 mm; refine the design angular sampling before attachment; for acoustic fits use finer throat/mouth resolutions in millimetres (AngularSegments may be overridden)")
            direction = b - a
            t = np.clip((controls - a) @ direction / (direction @ direction), 0, 1)
            error = np.linalg.norm(controls - a - t[:, None] * direction, axis=1).max()
            q = np.clip(-a @ direction / (direction @ direction), 0, 1)
            low = np.linalg.norm(a + q * direction) - error
            high = max(np.linalg.norm(a), np.linalg.norm(b)) + error
            interval_bound = max(abs(low - 1), abs(high - 1)) * maximum_radius
            # Refine until the radial interval has an independently small slack.
            if (error + abs(np.linalg.norm(a) - np.linalg.norm(b))) * maximum_radius <= 1e-4:
                bound = max(bound, interval_bound)
                continue
            layers, left, right = controls.copy(), [a], [b]
            while len(layers) > 1:
                layers = (layers[:-1] + layers[1:]) / 2
                left.append(layers[0]); right.append(layers[-1])
            pending.extend((np.asarray(left), np.asarray(right[::-1])))
    return float(bound)
