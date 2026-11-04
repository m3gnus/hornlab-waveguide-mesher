"""OCC transport of the existing authored ATH patches; no profile construction.

Ring bands are piecewise bicubics (linear across the band), copied with their
original parameter breaks. Triangular GEO fillings and spherical source caps
are sampled in their valid parameter domain and refined against the original
patch. The solve/preview builder remains the authority for every surface.
"""

import math
from dataclasses import replace
from functools import lru_cache

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import lu_factor, lu_solve

from ..errors import UserRefusal
from .point_grid_surfaces import _GeoSurfaceBuilder, _safe_surface_from_curves

# Half of WG's existing 0.020 mm STEP surface tolerance. Sewing still uses
# exact ring/line boundaries; this allowance is for the filling's interior.
FIT_TOL_MM = 0.010


@lru_cache(maxsize=32)
def _basis(intervals, degree):
    breaks = np.linspace(0, 1, intervals + 1)
    multiplicities = [degree + 1, *([min(2, degree)] * (intervals - 1)), degree + 1]
    knots = np.repeat(breaks, multiplicities)
    parameters = np.array(
        [np.mean(knots[i + 1 : i + degree + 1]) for i in range(len(knots) - degree - 1)]
    )
    matrix = BSpline.design_matrix(parameters, knots, degree).toarray()
    return breaks, multiplicities, knots, parameters, lu_factor(matrix)


class _PatchTransport(_GeoSurfaceBuilder):
    def __init__(self):
        super().__init__()
        self.intervals = {}
        self.patches = {}
        self.occ_curves = {}

    def _curve(self, curve):
        tag = abs(curve)
        if tag not in self.occ_curves:
            occ = self.gmsh.model.occ
            n = self.intervals.get(tag)
            if n is None:
                ends = np.asarray(self.gmsh.model.getValue(1, tag, [0, 1])).reshape(
                    2, 3
                )
                points = [occ.addPoint(*point) for point in ends]
                copied = occ.addLine(*points)
            else:
                breaks, mult, _, params, lu = _basis(n, 3)
                values = np.asarray(self.gmsh.model.getValue(1, tag, params)).reshape(
                    -1, 3
                )
                poles = lu_solve(lu, values)
                points = [occ.addPoint(*point) for point in poles]
                copied = occ.addBSpline(
                    points, degree=3, knots=breaks, multiplicities=mult
                )
            occ.remove([(0, point) for point in points], recursive=False)
            self.occ_curves[tag] = int(copied)
        return self.occ_curves[tag] if curve > 0 else -self.occ_curves[tag]

    def spline(self, points):
        tag = super().spline(points)
        self.intervals[abs(tag)] = len(points) - 1
        return tag

    def surface(self, curves, *, sphere_center_tag=-1):
        dim, tag = super().surface(curves, sphere_center_tag=sphere_center_tag)
        self.patches[tag] = (curves, sphere_center_tag >= 0)
        return dim, tag

    def _copy(self, tag):
        curves, spherical = self.patches[tag]
        spans = [self.intervals.get(abs(c), 1) for c in curves]
        triangular = len(curves) == 3
        fan = triangular and abs(curves[1]) in self.intervals

        def raw(u, v):
            uu, vv = np.meshgrid(u, v, indexing="ij")
            if triangular:
                # GEO triangles use 0 <= v <= u <= 1. Collapse the square
                # at the fan apex, keeping the ring on an entire square edge.
                uu, vv = (vv, vv * uu) if fan else (uu + (1 - uu) * vv, vv)
            uv = np.stack((uu, vv), axis=-1)
            return np.asarray(self.gmsh.model.getValue(2, tag, uv.reshape(-1))).reshape(
                uv.shape[0], uv.shape[1], 3
            )

        def edge(curve, t):
            low, high = self.gmsh.model.getParametrizationBounds(1, abs(curve))
            parameter = np.asarray(t) if curve > 0 else 1 - np.asarray(t)
            return np.asarray(
                self.gmsh.model.getValue(
                    1, abs(curve), low[0] + (high[0] - low[0]) * parameter
                )
            ).reshape(-1, 3)

        def query(u, v):
            values = raw(u, v)
            if not spherical:
                return values
            # GEO projects a spherical filling onto its sphere even at the
            # spline rim. The acoustic mesh shares the *curve*, not that
            # projected rim. Honour those same curves when sewing the cap.
            bottom = edge(curves[0], u) - raw(u, [0])[:, 0]
            left = edge(-curves[-1], v) - raw([0], v)[0]
            right = edge(curves[1], v) - raw([1], v)[0]
            top = (
                np.repeat(edge(curves[1], [1]), len(u), axis=0)
                if triangular
                else edge(-curves[2], u)
            ) - raw(u, [1])[:, 0]
            uu, vv = np.asarray(u)[:, None, None], np.asarray(v)[None, :, None]
            corners = (
                (1 - uu) * (1 - vv) * bottom[0]
                + uu * (1 - vv) * bottom[-1]
                + (1 - uu) * vv * top[0]
                + uu * vv * top[-1]
            )
            return (
                values
                + (1 - vv) * bottom[:, None]
                + vv * top[:, None]
                + (1 - uu) * left[None]
                + uu * right[None]
                - corners
            )

        if not spherical:
            # Coons fillings of coplanar boundaries are planar. Retain the
            # exact boundary curves and use the existing planar-fill owner.
            boundary = np.concatenate(
                [edge(c, np.linspace(0, 1, 3 * n + 1)) for c, n in zip(curves, spans)]
            )
            center = boundary.mean(axis=0)
            _, singular, axes = np.linalg.svd(boundary - center, full_matrices=False)
            if (
                singular[1] > 1e-8
                and np.max(np.abs((boundary - center) @ axes[-1])) < 1e-9
            ):
                return _safe_surface_from_curves([self._curve(c) for c in curves])[1]

        if triangular:
            nu, nv, dv = max(spans), max(spans), 3
        else:
            nu, nv = math.lcm(spans[0], spans[2]), math.lcm(spans[1], spans[3])
            dv = 3 if spherical or nv > 1 else 1
        error = math.inf
        for _ in range(6):
            if max(nu, nv) > 512 or (2 * nu + 2) * (2 * nv + 2) > 100_000:
                break
            ku, mu, tu, u, lu = _basis(nu, 3)
            kv, mv, tv, v, lv = _basis(nv, dv)
            values = query(u, v)
            poles = lu_solve(lu, values.reshape(len(u), -1)).reshape(values.shape)
            poles = (
                lu_solve(lv, poles.transpose(1, 0, 2).reshape(len(v), -1))
                .reshape(len(v), len(u), 3)
                .transpose(1, 0, 2)
            )

            # Independent interior and boundary probes in every knot span.
            def probes(breaks):
                return np.unique(
                    np.r_[
                        breaks,
                        (
                            breaks[:-1, None]
                            + np.diff(breaks)[:, None] * np.array([0.2113, 0.5, 0.7887])
                        ).ravel(),
                    ]
                )

            pu, pv = probes(ku), probes(kv)
            bu = BSpline.design_matrix(pu, tu, 3).toarray()
            bv = BSpline.design_matrix(pv, tv, dv).toarray()
            fitted = np.einsum("ui,ijc,vj->uvc", bu, poles, bv, optimize=True)
            error = float(np.max(np.linalg.norm(fitted - query(pu, pv), axis=-1)))
            if error <= FIT_TOL_MM:
                if (
                    spherical
                    and np.max(np.linalg.norm(fitted - raw(pu, pv), axis=-1)) > 0.020
                ):
                    raise UserRefusal(
                        "STEP export",
                        "The imported spherical source cap does not meet the wall within the CAD tolerance.",
                        "Export with an open throat, or export STL to retain the acoustic cap.",
                    )
                occ = self.gmsh.model.occ
                points = [
                    occ.addPoint(*point)
                    for point in poles.transpose(1, 0, 2).reshape(-1, 3)
                ]
                copied = int(
                    occ.addBSplineSurface(
                        points,
                        len(u),
                        degreeU=3,
                        degreeV=dv,
                        knotsU=ku,
                        knotsV=kv,
                        multiplicitiesU=mu,
                        multiplicitiesV=mv,
                    )
                )
                # Poles are construction points, not BRep vertices. Remove
                # them per patch instead of leaving a quadratic-sized cleanup
                # of all accumulated poles to the CAD owner's final pruning.
                occ.remove([(0, point) for point in points], recursive=False)
                return copied
            nu, nv = 2 * nu, 2 * nv
        raise UserRefusal(
            "STEP export",
            "The imported ATH surface could not be copied within the CAD tolerance.",
            "Export STL to retain this shape, or adjust the ATH lip or rear settings and import again.",
            diagnostic=f"ATH patch {tag}, spans {spans}: residual {error:g} mm at {nu} x {nv} intervals",
        )

    def convert(self, built):
        gmsh, occ = self.gmsh, self.gmsh.model.occ
        original = gmsh.model.getEntities()
        # Both kernels live in one model. Disjoint tags keep getValue queries
        # on GEO until extraction finishes and the built-in entities are removed.
        for dim in (0, 1, 2):
            occ.setMaxTag(dim, max((t for d, t in original if d == dim), default=0))
        tags = {tag: self._copy(tag) for tag in self.patches}
        gmsh.model.geo.remove(original, recursive=True)
        gmsh.model.geo.synchronize()
        occ.synchronize()
        return replace(
            built,
            surface_groups={
                key: [tags[t] for t in group]
                for key, group in built.surface_groups.items()
            },
            mesh_surface_groups={
                key: [tags[t] for t in group]
                for key, group in built.mesh_surface_groups.items()
            },
        )


def build_ath_occ(geometry, *, open_throat=True):
    from .point_grid_freestanding import _build_ath_import_surfaces

    if open_throat:
        # The membrane is removed by the CAD owner's bore cut. A flat closure
        # on the same throat curves gives the identical open part, without
        # transporting a non-material spherical diaphragm only to discard it.
        geometry = replace(geometry, source_shape=0)
    transport = _PatchTransport()
    built = _build_ath_import_surfaces(geometry, surface_builder=transport)
    # The preserve-grid variant is already authored with shared OCC facets.
    return transport.convert(built) if transport.patches else built
