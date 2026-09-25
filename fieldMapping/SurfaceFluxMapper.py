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

Face areas. The heat of a source face is q_f A_f with the face area A_f of
the chosen convention (``sourceArea``): "vector" (magnitude of the face-area
vector, the finite-volume convention), "fan" (sum of the centroid-fan
triangle areas), "fe" (isoparametric area) or the solver's own face areas
(a cell array); "auto" uses "vector" for linear faces and polygons and
"fe" for quadratic faces. Inside a face, the heat is split over its
flat sub-triangles in proportion to their integrals of q. The target flux
is Q_t / A_t with ``targetArea`` (default "fe", consistent with the nodal
loads). Warped quadrilaterals are split around their centroid so no
diagonal is preferred.

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
    pointFlux     optional nodal flux values: face-area weighted average of the
                  adjacent face fluxes (tributary area A_f / n_f per node, default)
                  or HRZ-lumped L2 projection, corrected by the conservative
                  projection so that sum_i a_i q_i equals the heat (incoming and
                  outgoing parts separately, sign bounds, per-group constraints)

The geometric part (pairs and pieces) is built once; ``map`` is cheap and can
be called for every time step of a transient flux.

Reverse direction (two-way coupling). ``mapBack`` carries an *intensive* field
(e.g. the wall temperature the structure computed) from the target back to the
source through the same pieces: every source cell gets the overlap-area weighted
mean of the target values, so the area integral over the overlap is conserved
(sum_s A_s^cov T_s = sum_pieces A_p T_t) and no new extrema appear. Source cells
or nodes the target does not cover get ``fill`` (NaN by default).
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


def _diagonal(m: SparseMatrix) -> np.ndarray:
    d = np.zeros(m.shape[0])
    on = m.row == m.col
    np.add.at(d, m.row[on], m.data[on])
    return d


def _spmm(a: SparseMatrix, b: SparseMatrix) -> SparseMatrix:
    """Sparse product a @ b (coordinate join; duplicates summed by SparseMatrix)."""
    order = np.argsort(b.row, kind="stable")
    br, bc, bd = b.row[order], b.col[order], b.data[order]
    start = np.searchsorted(br, np.arange(b.shape[0] + 1))
    rep = np.diff(start)[a.col]
    ia = np.repeat(np.arange(a.data.size), rep)
    jb = start[a.col][ia] + (np.arange(ia.size) - np.repeat(np.cumsum(rep) - rep, rep))
    return SparseMatrix(a.row[ia], bc[jb], a.data[ia] * bd[jb], (a.shape[0], b.shape[1]))


def _pcg(m: SparseMatrix, b: np.ndarray, diag: np.ndarray, tol: float = 1e-13, maxIter: int = 2000):
    """Jacobi-preconditioned conjugate gradients for an SPD sparse matrix (mass matrices)."""
    x = b / np.where(diag > 0, diag, 1.0)
    r = b - m @ x
    z = r / np.where(diag > 0, diag, 1.0)
    p = z.copy()
    rz = float(r @ z)
    norm = max(float(np.linalg.norm(b)), 1e-300)
    for k in range(maxIter):
        if np.linalg.norm(r) <= tol * norm:
            return x, k, float(np.linalg.norm(r) / norm)
        mp = m @ p
        alpha = rz / float(p @ mp)
        x += alpha * p
        r -= alpha * mp
        z = r / np.where(diag > 0, diag, 1.0)
        rzNew = float(r @ z)
        p = z + (rzNew / rz) * p
        rz = rzNew
    return x, maxIter, float(np.linalg.norm(r) / norm)


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


AREA_CONVENTIONS = ("auto", "vector", "fan", "fe")
_QUADRATIC_FACES = (22, 23, 28)


def facetSubTriangles(mesh: UnstructuredMesh, cells=None):
    """Flat sub-triangles of 2-D cells whose vertices are combinations of mesh points.

    Triangles are kept, quads and polygons are split around their vertex
    centroid (no preferred diagonal), quadratic faces through their mid-side
    nodes. Returns (coords (m, 3, 3), parent cell (m,), V) with V a
    SparseMatrix (3 m, nPoints) so that sub-triangle vertex values = V @ nodal values.
    """
    idx = np.flatnonzero(mesh.cellDimensions() == 2) if cells is None else np.asarray(cells, dtype=np.int64)
    parents, rows, cols, vals = [], [], [], []
    count = 0

    def add(ids, verts):
        """verts: list of 3 entries, each (node columns (k, j), weights (j,))."""
        nonlocal count
        k = ids.size
        tri = count + np.arange(k)
        for v, (nodes, w) in enumerate(verts):
            j = nodes.shape[1]
            rows.append(np.repeat(3 * tri + v, j))
            cols.append(nodes.ravel())
            vals.append(np.tile(w, k))
        parents.append(ids)
        count += k

    for t, (ids, conn) in mesh.blocks(idx).items():
        one = lambda c: (conn[:, [c]], np.ones(1))                        # noqa: E731
        if t == 5:
            add(ids, [one(0), one(1), one(2)])
        elif t == 9:
            cen = (conn, np.full(4, 0.25))
            for a in range(4):
                add(ids, [cen, one(a), one((a + 1) % 4)])
        elif t == 22:
            for pc in ([0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]):
                add(ids, [one(pc[0]), one(pc[1]), one(pc[2])])
        elif t in (23, 28):
            for pc in ([0, 4, 7], [4, 1, 5], [5, 2, 6], [6, 3, 7], [4, 5, 7], [5, 6, 7]):
                add(ids, [one(pc[0]), one(pc[1]), one(pc[2])])
        else:
            raise ValueError(f"cannot split cell type {t} into triangles")
    for c in idx[mesh.cellTypes[idx] == 7]:
        v = mesh.cell(c)
        n = v.size
        ids = np.full(n, c)
        cen = (np.tile(v, (n, 1)), np.full(n, 1.0 / n))
        add(ids, [cen, (v[:, None], np.ones(1)), (np.roll(v, -1)[:, None], np.ones(1))])
    if count == 0:
        return np.zeros((0, 3, 3)), np.zeros(0, np.int64), SparseMatrix([], [], [], (0, mesh.nPoints))
    V = SparseMatrix.raw(np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), (3 * count, mesh.nPoints))
    coords = (V @ mesh.points).reshape(count, 3, 3)
    return coords, np.concatenate(parents).astype(np.int64), V


def faceAreas(mesh: UnstructuredMesh, convention="auto", cells=None, order: int = 4) -> np.ndarray:
    """Area of every cell (zero outside ``cells``) in the given convention.

    convention: "vector" (|sum of sub-triangle area vectors|, finite-volume
    faces), "fan" (sum of the sub-triangle areas), "fe" (isoparametric
    quadrature), "auto" (vector for linear faces / polygons, fe for quadratic
    faces), the name of a cell array, or an array of areas.
    """
    idx = np.flatnonzero(mesh.cellDimensions() == 2) if cells is None else np.asarray(cells, dtype=np.int64)
    if not isinstance(convention, str) or convention not in AREA_CONVENTIONS:
        arr = np.asarray(mesh.cellData[convention] if isinstance(convention, str) else convention, dtype=float)
        if arr.shape != (mesh.nCells,):
            raise ValueError("face area array needs one value per cell")
        out = np.zeros(mesh.nCells)
        out[idx] = arr[idx]
        return out
    out = np.zeros(mesh.nCells)
    if convention in ("fe", "auto"):
        q = mesh.quadrature(order, idx)
        fe = np.bincount(q.cell, weights=q.weight, minlength=mesh.nCells)
        if convention == "fe":
            return fe
    x, par, _ = facetSubTriangles(mesh, idx)
    cr = 0.5 * np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    if convention == "fan":
        return np.bincount(par, weights=np.linalg.norm(cr, axis=1), minlength=mesh.nCells)
    vec = np.zeros((mesh.nCells, 3))
    np.add.at(vec, par, cr)
    out = np.linalg.norm(vec, axis=1)
    if convention == "auto":
        quad = np.isin(mesh.cellTypes, _QUADRATIC_FACES)
        out = np.where(quad, fe, out)
    return out


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


@dataclass
class FieldBackResult:
    """Intensive target field carried back onto the source surface (``SurfaceFluxMapper.mapBack``)."""
    cellValues: np.ndarray            # per source cell (fill where uncovered)
    pointValues: np.ndarray           # per source point (fill where no adjacent covered cell)
    cellCoverage: np.ndarray          # covered fraction of each source cell's area (0..1)
    diagnostics: dict = field(default_factory=dict)
    pointCoverage: Optional[np.ndarray] = None   # covered share of each source point's support (0..1):
                                                 # below 1 the point value comes from part of its support only


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
        sourceArea:   face-area convention of the source ("auto", "vector", "fan", "fe",
                      or a cell array / array with the solver's face areas)
        targetArea:   face-area convention of the target flux Q_t / A_t (default "fe")
        nodalWeighting: nodal flux weights: "area" (tributary face area A_f / n_f) or
                      "consistent" (HRZ-lumped finite-element mass)
    """

    def __init__(self, source: UnstructuredMesh, target: UnstructuredMesh, targetCells=None,
                 maxDistance: Optional[float] = None, maxAngle: float = 60.0, orientation: str = "auto",
                 unmatched: str = "nearest", order: int = 4, chunk: int = 200_000, sourceArea="auto",
                 targetArea="fe", nodalWeighting: str = "area") -> None:
        if nodalWeighting not in ("area", "consistent"):
            raise ValueError("nodalWeighting must be 'area' or 'consistent'")
        self._nodalWeighting = nodalWeighting
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
        # flat sub-triangles (centroid fans for quads / polygons)
        sx, sPar, sV = facetSubTriangles(source)
        tx, tPar, tV = facetSubTriangles(surf, np.flatnonzero(mask))
        self._tx, self._tV = tx, tV
        self._sx, self._sPar, self._sV = sx, sPar, sV
        self._tPar = tPar
        _, _, _, sN, sA = triangleFrames(sx)
        to, te1, te2, tN, tA = triangleFrames(tx)
        self._sArea = sA
        # face areas: the heat of a source face is q_f A_f in the chosen convention
        srcCells = np.unique(sPar)
        self.sourceFaceAreas = faceAreas(source, sourceArea, srcCells, order)
        subArea = np.bincount(sPar, weights=sA, minlength=source.nCells)
        self._sScale = np.where(subArea > 0, self.sourceFaceAreas / np.where(subArea > 0, subArea, 1.0), 0.0)
        self.targetFaceAreas = faceAreas(surf, targetArea, np.flatnonzero(mask), order)
        edge = lambda x: np.median(np.linalg.norm(x[:, 1] - x[:, 0], axis=1)) if x.size else 0.0   # noqa: E731
        self.maxDistance = float(maxDistance) if maxDistance is not None else 0.5 * max(edge(sx), edge(tx))
        tol = self.maxDistance
        i, j = boxPairs(sx.min(axis=1) - tol, sx.max(axis=1) + tol, tx.min(axis=1) - tol, tx.max(axis=1) + tol)
        # normal gap between the surfaces: distance of the source vertices to the target plane (the
        # box test already keeps the pair close in the plane)
        gap = np.abs(np.einsum("pvd,pd->pv", sx[i] - to[j][:, None, :], tN[j])).min(axis=1) if i.size else np.zeros(0)
        near = gap <= tol
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
            pieces.append((ii[keep], tPar[jj[keep]], poly[keep], cnt[keep], att[keep], area[keep], jj[keep]))
        cat = lambda k, shape: np.concatenate([p[k] for p in pieces]) if pieces else np.zeros(shape)   # noqa: E731
        self._pSrc = cat(0, (0,)).astype(np.int64)
        self._pTgt = cat(1, (0,)).astype(np.int64)
        self._pPoly = cat(2, (0, 6, 2))
        self._pCnt = cat(3, (0,)).astype(np.int64)
        self._pBary = cat(4, (0, 6, 3))
        self._pArea = cat(5, (0,))
        self._pTsub = cat(6, (0,)).astype(np.int64)
        # nearest target facet for every source triangle (fallback for gaps)
        self._nearest = self._nearestTarget(sx, tx, tPar, tol)
        # target quadrature (areas, loads)
        tq = surf.quadrature(order, np.flatnonzero(mask))
        self._tq = tq
        self.targetAreas = np.bincount(tq.cell, weights=tq.weight, minlength=surf.nCells)      # FE areas
        # projected overlap area / source triangle area (about 1 where the target covers the source)
        self.coverage = np.bincount(self._pSrc, weights=self._pArea, minlength=sx.shape[0]) / np.maximum(sA, 1e-300)

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
            return (self._sV @ values).reshape(-1, 3)
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
        sx = self._sx
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
            groups=None, sourceCoverage: str = "renormalize") -> FluxMapResult:
        """Map one flux field; see the module docstring.

        Args:
            values:          source flux per cell or per point
            location:        "cell" or "point"
            reconstruction:  for cell values: "constant" or "linear" (limited, mean
                             and sign preserving; sharper coarse -> fine transfer)
            pointFlux:       also compute conservative nodal flux values
            groups:          optional label per target surface cell: the nodal flux
                             conserves the heat of every group separately
            sourceCoverage:  "renormalize" (default): a source triangle's whole heat goes to
                             the target pieces it overlaps — right when the target covers the
                             source and only thin gaps (curvature, boundaries) are missing.
                             "clip": only the part of its heat lying over the target moves;
                             the rest is reported as dropped — right when the target is a
                             patch of a larger source (a panel of a vehicle surface), where
                             renormalizing would pile a straddling triangle's heat onto the
                             patch edge. Identical to "renormalize" where coverage >= 1.
        """
        if sourceCoverage not in ("renormalize", "clip"):
            raise ValueError("sourceCoverage must be 'renormalize' or 'clip'")
        qv = self._vertexValues(values, location, reconstruction)            # (nS, 3)
        ref = np.tile(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), (qv.shape[0], 1, 1))
        fp, fm = _signedParts(ref, np.full(qv.shape[0], 3), qv)
        # heat of each source sub-triangle, scaled so that every face carries q_f A_f (face-area convention)
        scale = 2.0 * self._sArea * self._sScale[self._sPar]
        qPlus, qMinus = fp * scale, fm * scale
        # pieces: q at the polygon vertices from the source barycentric coordinates
        pv = np.einsum("pvk,pk->pv", self._pBary, qv[self._pSrc])
        iPlus, iMinus = _signedParts(self._pPoly, self._pCnt, pv)
        nS = qv.shape[0]
        sPlus = np.bincount(self._pSrc, weights=iPlus, minlength=nS)
        sMinus = np.bincount(self._pSrc, weights=iMinus, minlength=nS)
        nT = self.target.nCells
        overlapPlus, overlapMinus = sPlus, sMinus          # before any clip scaling (orphan test below)
        clippedPlus = clippedMinus = 0.0
        if sourceCoverage == "clip":
            # the physical integral of each source sub-triangle's positive / negative part: a
            # piece takes iPlus / max(overlap total, whole triangle), so a partly covered
            # triangle moves only its covered fraction (the remainder is dropped)
            fullPlus, fullMinus = fp * 2.0 * self._sArea, fm * 2.0 * self._sArea
            sPlus, sMinus = np.maximum(sPlus, fullPlus), np.minimum(sMinus, fullMinus)
            with np.errstate(invalid="ignore", divide="ignore"):
                fracPlus = np.where(sPlus > 0, np.bincount(self._pSrc, weights=iPlus, minlength=nS) / sPlus, 0.0)
                fracMinus = np.where(sMinus < 0, np.bincount(self._pSrc, weights=iMinus, minlength=nS) / sMinus, 0.0)
            covered = np.bincount(self._pSrc, minlength=nS) > 0
            clippedPlus = float((qPlus * (1.0 - fracPlus))[covered].sum())
            clippedMinus = float((qMinus * (1.0 - fracMinus))[covered].sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            wPlus = np.where(sPlus[self._pSrc] > 0, iPlus / sPlus[self._pSrc], 0.0)
            wMinus = np.where(sMinus[self._pSrc] < 0, iMinus / sMinus[self._pSrc], 0.0)
        heatPlus = np.bincount(self._pTgt, weights=qPlus[self._pSrc] * wPlus, minlength=nT)
        heatMinus = np.bincount(self._pTgt, weights=qMinus[self._pSrc] * wMinus, minlength=nT)
        # source triangles whose part found no overlap
        orphanP = (qPlus > 0) & (overlapPlus <= 0)
        orphanM = (qMinus < 0) & (overlapMinus >= 0)
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
        droppedPlus += clippedPlus
        droppedMinus += clippedMinus
        area = self.targetFaceAreas
        heat = heatPlus + heatMinus
        cellFlux = np.where(area > 0, heat / np.where(area > 0, area, 1.0), 0.0)
        # nodal loads: the heat of every target face spread with its shape functions (sum = heat exactly)
        tq = self._tq
        fe = np.where(self.targetAreas > 0, self.targetAreas, 1.0)
        loads = tq.loadVector((heat / fe)[tq.cell])
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
                "sourceArea": float(self.sourceFaceAreas.sum()),
                "targetArea": float(area[self._targetMask].sum()),
                "matchedTargetArea": float(area[(heatPlus != 0) | (heatMinus != 0)].sum()),
                "nPieces": int(self._pSrc.size)}
        if pf is not None:
            diag["pointFluxHeat"] = float((self._nodalRows(np.zeros(self.target.nCells, int)) @ pf)[0])
        return FluxMapResult(cellFlux, heatPlus, heatMinus, loads, pf, diag)

    def mapBack(self, values, location: str = "cell", fill: float = np.nan, nodal: str = "average",
                minSupport: float = 0.05) -> FieldBackResult:
        """Carry an intensive target field (temperature, pressure, ...) back to the source surface.

        Args:
            values:   per target surface cell ("cell") or per target point ("point"; reduced to
                      facet means with the target quadrature)
            location: "cell" or "point"
            fill:     value for source cells / points the target does not cover
            nodal:    source point values: "average" (covered-area weighted mean of the adjacent
                      cells: bounded, first order - smooths peaks) or "projection" (L2 projection
                      onto the source's linear elements over the covered area: reproduces linear
                      fields exactly, second order; point input is integrated exactly on the pieces)
            minSupport: "projection" only: nodes whose covered share of their element support is
                      below this fraction (ill-conditioned at patch edges) keep the "average" value

        Every source cell gets the overlap-area weighted mean of the target facet values over its
        pieces of the common refinement; every source point the covered-area weighted mean of its
        adjacent cells (weights A_cov / n per cell and node). The integral over the overlap is
        conserved to round-off and the result stays within the target's range. ``pointCoverage`` is the
        covered share of each point's support: a point on the edge of (or outside) a target patch gets
        the value of the part that is covered, i.e. of a nearby location, not its own.
        """
        if nodal not in ("average", "projection"):
            raise ValueError("nodal must be 'average' or 'projection'")
        v = np.asarray(values, dtype=float).ravel()
        pointInput = v.copy() if location == "point" else None
        if location == "point":
            if v.size != self.target.nPoints:
                raise ValueError(f"point values need {self.target.nPoints} entries")
            tq = self._tq
            fe = np.where(self.targetAreas > 0, self.targetAreas, 1.0)
            v = np.bincount(tq.cell, weights=tq.weight * (tq.interp @ v), minlength=self.target.nCells) / fe
        elif location == "cell":
            if v.size != self.target.nCells:
                raise ValueError(f"cell values need {self.target.nCells} entries")
        else:
            raise ValueError("location must be 'cell' or 'point'")
        src = self.source
        nC = src.nCells
        cellOfPiece = self._sPar[self._pSrc]
        covArea = np.bincount(cellOfPiece, weights=self._pArea, minlength=nC)
        integral = np.bincount(cellOfPiece, weights=self._pArea * v[self._pTgt], minlength=nC)
        subArea = np.bincount(self._sPar, weights=self._sArea, minlength=nC)
        covered = covArea > 0
        cellValues = np.full(nC, float(fill))
        cellValues[covered] = integral[covered] / covArea[covered]
        coverage = np.where(subArea > 0, np.minimum(covArea / np.where(subArea > 0, subArea, 1.0), 1.0), 0.0)
        # nodes: covered-area weighted mean of the adjacent covered cells
        sizes = src.offsets[1:] - src.offsets[:-1]
        cellOfEntry = np.repeat(np.arange(nC), sizes)
        node = src.connectivity                                          # offsets[0] = 0: entries in cell order
        w = (covArea / np.maximum(sizes, 1))[cellOfEntry]
        num = np.bincount(node, weights=w * np.where(covered, cellValues, 0.0)[cellOfEntry], minlength=src.nPoints)
        den = np.bincount(node, weights=w, minlength=src.nPoints)
        pointValues = np.full(src.nPoints, float(fill))
        pointValues[den > 0] = num[den > 0] / den[den > 0]
        support = np.bincount(node, weights=(subArea / np.maximum(sizes, 1))[cellOfEntry], minlength=src.nPoints)
        pointCoverage = np.where(support > 0, np.minimum(den / np.where(support > 0, support, 1.0), 1.0), 0.0)
        pieceIntegral = float(np.sum(self._pArea * v[self._pTgt]))
        sourceIntegral = float(np.sum(covArea[covered] * cellValues[covered]))
        diag = {"coveredSourceCells": int(covered.sum()), "coveredSourcePoints": int((den > 0).sum()),
                "overlapArea": float(covArea.sum()), "overlapIntegral": pieceIntegral,
                "relativeIntegralError": (sourceIntegral - pieceIntegral) / max(abs(pieceIntegral), 1e-300),
                "targetRange": [float(v[self._targetMask].min()), float(v[self._targetMask].max())]
                if self._targetMask.any() else [float("nan")] * 2}
        if nodal == "projection":
            projected, info = self._projectBack(v, pointInput, fill, minSupport)
            thin = np.isfinite(pointValues) & ~np.isfinite(projected)
            pointValues = np.where(thin, pointValues, projected)       # thin support: the bounded average
            diag.update(info)
        return FieldBackResult(cellValues, pointValues, coverage, diag, pointCoverage)

    def _pieceFans(self):
        """Fan triangles of the pieces: (piece id, vertex slots (F, 3), areas (F,))."""
        P, V = self._pPoly.shape[:2]
        k = np.arange(V - 2)
        valid = k[None, :] < (self._pCnt[:, None] - 2)
        piece, kk = np.nonzero(valid)
        slots = np.column_stack([np.zeros_like(kk), kk + 1, kk + 2])
        pts = self._pPoly[piece[:, None], slots]                              # (F, 3, 2)
        a, b = pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]
        return piece, slots, 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

    def _targetAtPieceVertices(self, pointInput):
        """Linear target field at every piece polygon vertex (from the target sub-triangle)."""
        tV = (self._tV @ pointInput).reshape(-1, 3)[self._pTsub]              # (P, 3)
        to, te1, te2, _, _ = triangleFrames(self._tx[self._pTsub])
        rel = self._tx[self._pTsub] - to[:, None, :]
        tri = np.stack([np.einsum("pvd,pd->pv", rel, te1), np.einsum("pvd,pd->pv", rel, te2)], axis=2)   # (P, 3, 2)
        A, B, C = tri[:, 0], tri[:, 1], tri[:, 2]
        det = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
        d = self._pPoly - A[:, None, :]
        l1 = (d[..., 0] * (C[:, 1] - A[:, 1])[:, None] - d[..., 1] * (C[:, 0] - A[:, 0])[:, None]) / det[:, None]
        l2 = ((B[:, 0] - A[:, 0])[:, None] * d[..., 1] - (B[:, 1] - A[:, 1])[:, None] * d[..., 0]) / det[:, None]
        return tV[:, :1] * (1.0 - l1 - l2) + tV[:, 1:2] * l1 + tV[:, 2:3] * l2          # (P, V)

    def _projectBack(self, cellValues, pointInput, fill, minSupport):
        """L2 projection of the target field onto the source's linear (sub-triangle) elements."""
        piece, slots, fanArea = self._pieceFans()
        lam = self._pBary[piece[:, None], slots]                               # (F, 3 vertices, 3 lambdas)
        if pointInput is not None:
            T = self._targetAtPieceVertices(pointInput)[piece[:, None], slots]  # (F, 3)
        else:
            T = np.repeat(cellValues[self._pTgt[piece]][:, None], 3, axis=1)
        # exact quadratic integrals over a triangle: int f g = A / 12 (sum f_i g_i + sum f sum g)
        quad = lambda f, g: fanArea / 12.0 * (np.sum(f * g, axis=1) + f.sum(axis=1) * g.sum(axis=1))   # noqa: E731
        sub = self._pSrc[piece]
        nS = self._sx.shape[0]
        rows, cols, vals, bRows, bVals = [], [], [], [], []
        for a in range(3):
            bRows.append(3 * sub + a)
            bVals.append(quad(lam[:, :, a], T))
            for b in range(3):
                rows.append(3 * sub + a)
                cols.append(3 * sub + b)
                vals.append(quad(lam[:, :, a], lam[:, :, b]))
        Msub = SparseMatrix(np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), (3 * nS, 3 * nS))
        bsub = np.bincount(np.concatenate(bRows), weights=np.concatenate(bVals), minlength=3 * nS)
        V = self._sV
        # M = V^T Msub V, b = V^T bsub (sub-triangle vertices are combinations of source points)
        nP = self.source.nPoints
        M = _spmm(V.T, _spmm(Msub, V))
        b = V.rmatvec(bsub)
        lumped = np.bincount(M.row, weights=M.data, minlength=nP)              # covered support of each node
        full = V.rmatvec(np.repeat(self._sArea / 3.0, 3))                       # whole support
        # solve on every node with any covered support (the restricted projection stays exact for
        # linear fields); thin-support nodes are poorly determined and reported as ``fill`` afterwards
        live = lumped > 0.0
        idx = np.flatnonzero(live)
        keep = live[M.row] & live[M.col]
        Mr = SparseMatrix(np.searchsorted(idx, M.row[keep]), np.searchsorted(idx, M.col[keep]), M.data[keep],
                          (idx.size, idx.size))
        x, iters, res = _pcg(Mr, b[idx], _diagonal(Mr))
        out = np.full(nP, float(fill))
        out[idx] = x
        thin = live & (lumped < minSupport * np.maximum(full, 1e-300))
        out[thin] = float(fill)
        return out, {"projectionNodes": int((live & ~thin).sum()), "projectionIterations": iters,
                     "projectionResidual": res, "projectionDroppedNodes": int(thin.sum())}

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

    def _tributary(self):
        """Per (face, node) entries of the tributary areas A_f / n_f of the target faces."""
        surf = self.target
        cells = np.flatnonzero(self._targetMask)
        sizes = surf.offsets[cells + 1] - surf.offsets[cells]
        face = np.repeat(cells, sizes)
        pos = np.repeat(surf.offsets[cells], sizes) + (np.arange(int(sizes.sum())) - np.repeat(np.cumsum(sizes) - sizes,
                                                                                                    sizes))
        node = surf.connectivity[pos]
        return face, node, self.targetFaceAreas[face] / np.repeat(sizes, sizes)

    def _nodalRows(self, groups: np.ndarray) -> SparseMatrix:
        """Heat of the nodal flux per group: C[g, i] = a_i^g (tributary area or int N_i)."""
        if self._nodalWeighting == "consistent":
            return self._massRows(groups)
        face, node, a = self._tributary()
        g = np.asarray(groups)[face]
        return SparseMatrix(g, node, a, (int(g.max()) + 1 if g.size else 1, self.target.nPoints))

    def _pointFlux(self, heatPlus, heatMinus, groups) -> np.ndarray:
        if self._nodalWeighting == "area":
            return self._pointFluxArea(heatPlus, heatMinus, groups)
        return self._pointFluxConsistent(heatPlus, heatMinus, groups)

    def _groupLabels(self, groups) -> np.ndarray:
        if groups is None:
            return np.zeros(self.target.nCells, dtype=np.int64)
        _, gl = np.unique(np.asarray(groups), return_inverse=True)
        return gl.ravel()

    def _pointFluxArea(self, heatPlus, heatMinus, groups) -> np.ndarray:
        """Face-area weighted nodal flux: q_i = sum_f a_fi q_f / sum_f a_fi, corrected to conserve the heat."""
        nP = self.target.nPoints
        face, node, a = self._tributary()
        trib = np.bincount(node, weights=a, minlength=nP)
        used = trib > 0
        gl = self._groupLabels(groups)
        c = self._nodalRows(gl)
        area = np.where(self.targetFaceAreas > 0, self.targetFaceAreas, 1.0)
        out = np.zeros(nP)
        idx = np.flatnonzero(used)
        sub = SparseMatrix(c.row, np.searchsorted(idx, c.col), c.data, (c.shape[0], idx.size))
        for heat, lb, ub in ((heatPlus, 0.0, np.inf), (heatMinus, -np.inf, 0.0)):
            if not np.any(heat != 0):
                continue
            x0 = np.bincount(node, weights=a * (heat / area)[face], minlength=nP)[idx] / trib[idx]
            d = np.bincount(gl[self._targetMask], weights=heat[self._targetMask], minlength=c.shape[0])
            out[idx] += projectAffine(x0, trib[idx], sub, d, lb, ub).x
        return out

    def _pointFluxConsistent(self, heatPlus, heatMinus, groups) -> np.ndarray:
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
        gl = self._groupLabels(groups)
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
