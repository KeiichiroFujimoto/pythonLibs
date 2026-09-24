"""Conservative transfer of a surface flux (e.g. heat flux) between non-matching surface meshes.

The source flux q (cell values, or point values interpolated linearly on the
source facets) is carried to the target facets through the common
refinement of the two surfaces:

1. both surfaces are split into flat triangles (quadratic facets through
   their mid-side nodes, polygons by fans);
2. every source triangle is projected onto each nearby, similarly oriented
   target triangle and clipped by it: the pieces of the common refinement;
3. per field, each source triangle is split along the zero line of q, so the
   incoming (q > 0) and outgoing (q < 0) parts are handled separately;
4. the heat of every source triangle, Q_s = integral of q over its true 3-D
   area, is distributed to the target facets in proportion to the integrals
   of q over its pieces (projected geometry).

Hence, exactly (to round-off),

    sum over target facets of Q+  =  source incoming heat
    sum over target facets of Q-  =  source outgoing heat

and no facet receives heat of the wrong sign. Source triangles without
overlap (gaps along boundaries or curvature mismatch) go to the nearest
target facet within ``maxDistance``; heat farther away is reported as
dropped (the target covers only part of the source).

Outputs (``FluxMapResult``):

    cellFlux      piecewise-constant target flux Q_t / A_t (curved facet areas)
    nodalLoads    consistent nodal loads f_i = sum_t q_t int_t N_i dA, sum f = total heat
    pointFlux     optional nodal flux values: lumped L2 projection corrected by
                  the conservative projection (incoming and outgoing parts
                  separately, sign bounds, per-group heat constraints)

The geometric part (pairs and pieces) is built once; ``map`` is cheap and can
be called for every time step of a transient flux.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pythonLibs.fieldMapping.Geometry import (boxPairs, clipHalfPlane, clipPolygons, closestPointOnTriangles,
                                              triangleFrames)
from pythonLibs.fieldMapping.mesh.Mesh import UnstructuredMesh
from pythonLibs.regressionHandler.constraints.Projection import projectAffine
from pythonLibs.regressionHandler.numerics.Sparse import SparseMatrix


def _linearIntegral(poly, count, vals):
    """(area, integral of the linear field) of polygons (P, V, 2) with vertex values (P, V) (fan from vertex 0)."""
    p0 = poly[:, :1]
    a, b = poly[:, 1:-1] - p0, poly[:, 2:] - p0
    cr = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    valid = np.arange(poly.shape[1] - 2)[None, :] < (count[:, None] - 2)
    cr = np.where(valid, 0.5 * cr, 0.0)
    mean = (vals[:, :1] + vals[:, 1:-1] + vals[:, 2:]) / 3.0
    return np.abs(cr.sum(axis=1)), np.abs(np.sum(cr * mean, axis=1))


def _signedParts(poly, count, vals):
    """Integrals of max(q, 0) and min(q, 0) over polygons with linear q (vertex values)."""
    empty = np.zeros(poly.shape[:2] + (0,))
    pp, pn, _, pv = clipHalfPlane(poly, count, empty, vals)
    _, plus = _linearIntegral(pp, pn, pv)
    mp, mn, _, mv = clipHalfPlane(poly, count, empty, -vals)
    _, minus = _linearIntegral(mp, mn, mv)
    return plus, -minus


@dataclass
class FluxMapResult:
    """Mapped flux on the target surface and conservation diagnostics."""
    cellFlux: np.ndarray
    cellHeatPlus: np.ndarray
    cellHeatMinus: np.ndarray
    nodalLoads: np.ndarray
    pointFlux: Optional[np.ndarray]
    diagnostics: dict = field(default_factory=dict)

    def attach(self, mesh: UnstructuredMesh, name: str = "heatFlux") -> UnstructuredMesh:
        """Add the results as cell / point data of the target surface mesh."""
        mesh.cellData[name] = self.cellFlux
        mesh.cellData[name + "HeatIn"] = self.cellHeatPlus
        mesh.cellData[name + "HeatOut"] = self.cellHeatMinus
        mesh.pointData[name + "NodalLoad"] = self.nodalLoads
        if self.pointFlux is not None:
            mesh.pointData[name] = self.pointFlux
        return mesh


class SurfaceFluxMapper:
    """Conservative, sign-preserving surface flux transfer (see the module docstring).

    Args:
        source:       mesh whose 2-D cells carry the flux
        target:       surface mesh, or a volume mesh (its boundary surface is used;
                      nodal loads are then indexed by the volume mesh's points)
        targetCells:  optional subset of the target surface cells (ids or mask)
        maxDistance:  largest normal gap between the surfaces (default: half the
                      larger median edge length)
        maxAngle:     largest angle between facet normals of matched pieces (degrees)
        orientation:  "auto" (flip source normals if they point the other way),
                      "same" or "opposite"
        unmatched:    "nearest" (gaps go to the nearest target facet within
                      maxDistance) or "drop"
        order:        quadrature order of the target facets (nodal loads, areas)
    """

    def __init__(self, source: UnstructuredMesh, target: UnstructuredMesh, targetCells=None,
                 maxDistance: Optional[float] = None, maxAngle: float = 60.0, orientation: str = "auto",
                 unmatched: str = "nearest", order: int = 4, chunk: int = 200_000) -> None:
        if unmatched not in ("nearest", "drop"):
            raise ValueError("unmatched must be 'nearest' or 'drop'")
        if orientation not in ("auto", "same", "opposite"):
            raise ValueError("orientation must be auto, same or opposite")
        self.source = source
        self._unmatchedMode = unmatched
        surf = target if np.all(target.cellDimensions() <= 2) else target.boundarySurface()
        if targetCells is not None:
            sel = np.asarray(targetCells)
            sel = np.flatnonzero(sel) if sel.dtype == bool else sel
            mask = np.zeros(surf.nCells, dtype=bool)
            mask[sel] = True
        else:
            mask = surf.cellDimensions() == 2
        self.target = surf
        self._targetMask = mask
        self._order = order
        # flat triangles
        sTri, sPar = source.triangulate()
        tTri, tPar = surf.triangulate()
        keepT = mask[tPar]
        tTri, tPar = tTri[keepT], tPar[keepT]
        self._sTri, self._sPar = sTri, sPar
        self._tTri, self._tPar = tTri, tPar
        sx, tx = source.points[sTri], surf.points[tTri]
        _, _, _, sN, sA = triangleFrames(sx)
        to, te1, te2, tN, tA = triangleFrames(tx)
        self._sArea = sA
        edge = lambda x: np.median(np.linalg.norm(x[:, 1] - x[:, 0], axis=1)) if x.size else 0.0   # noqa: E731
        self.maxDistance = float(maxDistance) if maxDistance is not None else 0.5 * max(edge(sx), edge(tx))
        tol = self.maxDistance
        i, j = boxPairs(sx.min(axis=1) - tol, sx.max(axis=1) + tol, tx.min(axis=1) - tol, tx.max(axis=1) + tol)
        # distance of the source triangle to the target triangle (centroid based)
        _, dist, _ = closestPointOnTriangles(sx[i].mean(axis=1), tx[j]) if i.size else (None, np.zeros(0), None)
        near = dist <= tol
        i, j = i[near], j[near]
        dots = np.einsum("pd,pd->p", sN[i], tN[j])
        if orientation == "auto":
            w = sA[i] * np.abs(dots)
            sign = -1.0 if np.sum(w * np.sign(dots)) < 0 else 1.0
        else:
            sign = 1.0 if orientation == "same" else -1.0
        self.orientationSign = sign
        ok = sign * dots >= np.cos(np.radians(maxAngle))
        i, j = i[ok], j[ok]
        # clip in the target triangle frame
        pieces = []
        for s in range(0, i.size, chunk):
            ii, jj = i[s:s + chunk], j[s:s + chunk]
            rel = sx[ii] - to[jj][:, None, :]
            sub = np.stack([np.einsum("pvd,pd->pv", rel, te1[jj]), np.einsum("pvd,pd->pv", rel, te2[jj])], axis=2)
            relT = tx[jj] - to[jj][:, None, :]
            clip = np.stack([np.einsum("pvd,pd->pv", relT, te1[jj]), np.einsum("pvd,pd->pv", relT, te2[jj])], axis=2)
            bary = np.tile(np.eye(3), (ii.size, 1, 1))
            poly, cnt, att = clipPolygons(sub, np.full(ii.size, 3), bary, clip)
            area, _ = _linearIntegral(poly, cnt, np.ones(poly.shape[:2]))
            keep = (cnt >= 3) & (area > 1e-14 * np.maximum(sA[ii], tA[jj]))
            pieces.append((ii[keep], tPar[jj[keep]], poly[keep], cnt[keep], att[keep], area[keep]))
        cat = lambda k, shape: np.concatenate([p[k] for p in pieces]) if pieces else np.zeros(shape)   # noqa: E731
        self._pSrc = cat(0, (0,)).astype(np.int64)
        self._pTgt = cat(1, (0,)).astype(np.int64)
        self._pPoly = cat(2, (0, 6, 2))
        self._pCnt = cat(3, (0,)).astype(np.int64)
        self._pBary = cat(4, (0, 6, 3))
        self._pArea = cat(5, (0,))
        # nearest target facet for every source triangle (fallback for gaps)
        self._nearest = self._nearestTarget(sx, tx, tPar, tol)
        # target quadrature (areas, loads)
        tq = surf.quadrature(order, np.flatnonzero(mask))
        self._tq = tq
        self.targetAreas = np.bincount(tq.cell, weights=tq.weight, minlength=surf.nCells)
        # projected overlap area / source triangle area (about 1 where the target covers the source)
        self.coverage = np.bincount(self._pSrc, weights=self._pArea, minlength=sTri.shape[0]) / np.maximum(sA, 1e-300)

    def _nearestTarget(self, sx, tx, tPar, tol):
        cen = sx.mean(axis=1)
        out = np.full(sx.shape[0], -1, dtype=np.int64)
        best = np.full(sx.shape[0], np.inf)
        i, j = boxPairs(cen - tol, cen + tol, tx.min(axis=1), tx.max(axis=1))
        if i.size:
            _, d, _ = closestPointOnTriangles(cen[i], tx[j])
            order = np.lexsort((d, i))
            i, j, d = i[order], j[order], d[order]
            first = np.concatenate([[True], i[1:] != i[:-1]])
            ok = first & (d <= tol)
            out[i[ok]] = tPar[j[ok]]
            best[i[ok]] = d[ok]
        return out

    # ------------------------------------------------------------------ source representation
    def _vertexValues(self, values, location: str, reconstruction: str) -> np.ndarray:
        values = np.asarray(values, dtype=float).ravel()
        if location == "point":
            if values.size != self.source.nPoints:
                raise ValueError(f"point flux needs {self.source.nPoints} values")
            return values[self._sTri]
        if location != "cell":
            raise ValueError("location must be 'cell' or 'point'")
        if values.size != self.source.nCells:
            raise ValueError(f"cell flux needs {self.source.nCells} values")
        qc = values[self._sPar]
        if reconstruction == "constant":
            return np.repeat(qc[:, None], 3, axis=1)
        if reconstruction != "linear":
            raise ValueError("reconstruction must be 'constant' or 'linear'")
        return self._linearReconstruction(values)

    def _linearReconstruction(self, qc: np.ndarray) -> np.ndarray:
        """Mean-, bound- and sign-preserving linear reconstruction of cell values on each source facet."""
        src = self.source
        sx = src.points[self._sTri]
        area = self._sArea
        nC = src.nCells
        cellArea = np.bincount(self._sPar, weights=area, minlength=nC)
        cen = np.zeros((nC, 3))
        np.add.at(cen, self._sPar, area[:, None] * sx.mean(axis=1))
        cen /= np.maximum(cellArea, 1e-300)[:, None]
        normal = np.zeros((nC, 3))
        np.add.at(normal, self._sPar, np.cross(sx[:, 1] - sx[:, 0], sx[:, 2] - sx[:, 0]))
        normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-300)[:, None]
        # face neighbours through shared points
        sizes = src.cellSizes()
        cellOf = np.repeat(np.arange(nC), sizes)
        pts = src.connectivity
        order = np.argsort(pts, kind="stable")
        pSorted, cSorted = pts[order], cellOf[order]
        starts = np.searchsorted(pSorted, np.arange(src.nPoints))
        ends = np.searchsorted(pSorted, np.arange(src.nPoints), side="right")
        pairs_a, pairs_b = [], []
        cnt = ends - starts
        for k in range(int(cnt.max()) if cnt.size else 0):
            for l in range(int(cnt.max())):
                if k == l:
                    continue
                ok = (cnt > max(k, l))
                pairs_a.append(cSorted[starts[ok] + k])
                pairs_b.append(cSorted[starts[ok] + l])
        a = np.concatenate(pairs_a) if pairs_a else np.zeros(0, np.int64)
        b = np.concatenate(pairs_b) if pairs_b else np.zeros(0, np.int64)
        key = np.unique(a * nC + b)
        a, b = key // nC, key % nC
        d = cen[b] - cen[a]
        d = d - np.einsum("pd,pd->p", d, normal[a])[:, None] * normal[a]     # tangent plane
        dq = qc[b] - qc[a]
        # normal equations per cell (3x3 with a tangent regularization)
        m = np.zeros((nC, 3, 3))
        r = np.zeros((nC, 3))
        np.add.at(m, a, d[:, :, None] * d[:, None, :])
        np.add.at(r, a, d * dq[:, None])
        m += (np.trace(m, axis1=1, axis2=2)[:, None, None] * 1e-12 + 1e-300) * np.eye(3) + \
            normal[:, :, None] * normal[:, None, :] * (np.trace(m, axis1=1, axis2=2)[:, None, None] + 1.0)
        g = np.linalg.solve(m, r[:, :, None])[:, :, 0]
        # Barth-Jespersen limiter on the facet vertices, plus sign preservation
        qmin = qc.copy()
        qmax = qc.copy()
        np.minimum.at(qmin, a, qc[b])
        np.maximum.at(qmax, a, qc[b])
        vx = src.points[pts]
        dv = np.einsum("pd,pd->p", vx - cen[cellOf], g[cellOf])
        with np.errstate(divide="ignore", invalid="ignore"):
            up = np.where(dv > 0, (qmax[cellOf] - qc[cellOf]) / dv, np.inf)
            lo = np.where(dv < 0, (qmin[cellOf] - qc[cellOf]) / dv, np.inf)
            sgn = np.where(qc[cellOf] * dv < 0, np.abs(qc[cellOf]) / np.abs(dv), np.inf)
        phi = np.ones(nC)
        np.minimum.at(phi, cellOf, np.minimum(np.minimum(up, lo), sgn))
        phi = np.clip(phi, 0.0, 1.0)
        g = g * phi[:, None]
        par = self._sPar
        return qc[par][:, None] + np.einsum("svd,sd->sv", sx - cen[par][:, None, :], g[par])

    # ------------------------------------------------------------------ mapping
    def map(self, values, location: str = "cell", reconstruction: str = "constant", pointFlux: bool = False,
            groups=None) -> FluxMapResult:
        """Map one flux field; see the module docstring.

        Args:
            values:          source flux per cell or per point
            location:        "cell" or "point"
            reconstruction:  for cell values: "constant" or "linear" (limited, mean
                             and sign preserving; sharper coarse -> fine transfer)
            pointFlux:       also compute conservative nodal flux values
            groups:          optional label per target surface cell: the nodal flux
                             conserves the heat of every group separately
        """
        qv = self._vertexValues(values, location, reconstruction)            # (nS, 3)
        ref = np.tile(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), (qv.shape[0], 1, 1))
        fp, fm = _signedParts(ref, np.full(qv.shape[0], 3), qv)
        qPlus, qMinus = fp * 2.0 * self._sArea, fm * 2.0 * self._sArea       # heat of each source triangle
        # pieces: q at the polygon vertices from the source barycentric coordinates
        pv = np.einsum("pvk,pk->pv", self._pBary, qv[self._pSrc])
        iPlus, iMinus = _signedParts(self._pPoly, self._pCnt, pv)
        nS = qv.shape[0]
        sPlus = np.bincount(self._pSrc, weights=iPlus, minlength=nS)
        sMinus = np.bincount(self._pSrc, weights=iMinus, minlength=nS)
        nT = self.target.nCells
        with np.errstate(invalid="ignore", divide="ignore"):
            wPlus = np.where(sPlus[self._pSrc] > 0, iPlus / sPlus[self._pSrc], 0.0)
            wMinus = np.where(sMinus[self._pSrc] < 0, iMinus / sMinus[self._pSrc], 0.0)
        heatPlus = np.bincount(self._pTgt, weights=qPlus[self._pSrc] * wPlus, minlength=nT)
        heatMinus = np.bincount(self._pTgt, weights=qMinus[self._pSrc] * wMinus, minlength=nT)
        # source triangles whose part found no overlap
        orphanP = (qPlus > 0) & (sPlus <= 0)
        orphanM = (qMinus < 0) & (sMinus >= 0)
        near = self._nearest
        assign = near >= 0
        droppedPlus = droppedMinus = 0.0
        for orphan, q, heat, sgn in ((orphanP, qPlus, heatPlus, 1), (orphanM, qMinus, heatMinus, -1)):
            if self._unmatched_nearest():
                ok = orphan & assign
                np.add.at(heat, near[ok], q[ok])
                lost = float(q[orphan & ~assign].sum())
            else:
                lost = float(q[orphan].sum())
            if sgn > 0:
                droppedPlus = lost
            else:
                droppedMinus = lost
        area = self.targetAreas
        cellFlux = np.where(area > 0, (heatPlus + heatMinus) / np.where(area > 0, area, 1.0), 0.0)
        tq = self._tq
        loads = tq.loadVector(cellFlux[tq.cell])
        pf = self._pointFlux(heatPlus, heatMinus, groups) if pointFlux else None
        srcPlus, srcMinus = float(qPlus.sum()), float(qMinus.sum())
        tgtPlus, tgtMinus = float(heatPlus.sum()), float(heatMinus.sum())
        srcPeak = float(np.max(np.abs(qv))) if qv.size else 0.0
        diag = {"sourceHeatIn": srcPlus, "sourceHeatOut": srcMinus, "targetHeatIn": tgtPlus, "targetHeatOut": tgtMinus,
                "droppedHeatIn": droppedPlus, "droppedHeatOut": droppedMinus,
                "relativeErrorIn": (tgtPlus + droppedPlus - srcPlus) / max(abs(srcPlus), 1e-300),
                "relativeErrorOut": (tgtMinus + droppedMinus - srcMinus) / max(abs(srcMinus), 1e-300),
                "nodalLoadTotal": float(loads.sum()),
                "orphanTriangles": int((orphanP | orphanM).sum()),
                "sourcePeakFlux": srcPeak,
                "targetPeakFlux": float(np.max(np.abs(cellFlux[self._targetMask]))) if self._targetMask.any() else 0.0,
                "coverageMin": float(self.coverage.min()) if self.coverage.size else 1.0,
                "nPieces": int(self._pSrc.size)}
        if pf is not None:
            diag["pointFluxHeat"] = float((self._massRows(np.zeros(self.target.nCells, int)) @ pf)[0])
        return FluxMapResult(cellFlux, heatPlus, heatMinus, loads, pf, diag)

    def _unmatched_nearest(self) -> bool:
        return getattr(self, "_unmatchedMode", "nearest") == "nearest"

    # ------------------------------------------------------------------ nodal flux
    def _massRows(self, groups: np.ndarray) -> SparseMatrix:
        """C[g, i] = integral of N_i over the target cells of group g."""
        tq = self._tq
        g = np.asarray(groups)[tq.cell]
        rows = g[tq.interp.row]
        return SparseMatrix(rows, tq.interp.col, tq.interp.data * tq.weight[tq.interp.row],
                            (int(g.max()) + 1 if g.size else 1, self.target.nPoints))

    def _pointFlux(self, heatPlus, heatMinus, groups) -> np.ndarray:
        tq = self._tq
        nP = self.target.nPoints
        # HRZ lumping: diagonal of the element mass matrices scaled to the element area
        diagE = tq.interp.data ** 2 * tq.weight[tq.interp.row]
        cellOfEntry = tq.cell[tq.interp.row]
        sumDiag = np.bincount(cellOfEntry, weights=diagE, minlength=self.target.nCells)
        scale = np.where(sumDiag > 0, self.targetAreas / np.where(sumDiag > 0, sumDiag, 1.0), 0.0)
        lumpE = diagE * scale[cellOfEntry]
        lumped = np.bincount(tq.interp.col, weights=lumpE, minlength=nP)
        used = lumped > 0
        if groups is None:
            gl = np.zeros(self.target.nCells, dtype=np.int64)
        else:
            _, gl = np.unique(np.asarray(groups), return_inverse=True)
            gl = gl.ravel()
        c = self._massRows(gl)
        area = np.where(self.targetAreas > 0, self.targetAreas, 1.0)
        out = np.zeros(nP)
        for heat, lb, ub in ((heatPlus, 0.0, np.inf), (heatMinus, -np.inf, 0.0)):
            if not np.any(heat != 0):
                continue
            flux = heat / area
            x0 = np.bincount(tq.interp.col, weights=lumpE * flux[cellOfEntry], minlength=nP)
            x0 = np.where(used, x0 / np.where(used, lumped, 1.0), 0.0)
            d = np.bincount(gl[self._targetMask], weights=heat[self._targetMask], minlength=c.shape[0])
            idx = np.flatnonzero(used)
            sub = SparseMatrix(c.row, np.searchsorted(idx, c.col), c.data, (c.shape[0], idx.size))
            res = projectAffine(x0[idx], lumped[idx], sub, d, lb, ub)
            out[idx] += res.x
        return out
