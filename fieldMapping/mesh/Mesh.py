"""Unstructured mesh with finite-element integration over any VTK cell, including polygons and polyhedra.

Storage (numpy, VTK conventions):

    points          (n, 3)
    cellTypes       (c,) VTK type ids
    connectivity    flat node ids, cell i uses connectivity[offsets[i]:offsets[i+1]]
    offsets         (c + 1,) with offsets[0] = 0
    polyFaces       {cell: [face node arrays]} for polyhedra (VTK type 42)
    pointData / cellData   {name: (n,) or (n, k) arrays}

Integration (``quadrature``) returns quadrature points, physical weights and a
sparse interpolation operator from nodal values to the quadrature points:

    integral of f  =  weights @ (interp @ fNodal)

Standard cells use their isoparametric shape functions (curved quadratic cells
are integrated exactly on their curved geometry). A polygon is split into
triangles (centroid, v_i, v_i+1) and a polyhedron into tetrahedra
(cell centroid, face centroid, v_i, v_i+1); the field is linear on each piece,
with the centroid values taken as vertex averages (the standard polyhedral
interpolation). This reproduces any linear field exactly and gives the exact
measure of any polyhedron with planar faces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from pythonLibs.fieldMapping.mesh.Elements import (POLYGON, POLYHEDRON, RENUMBER, TRIANGLE, TETRA, cellDimension,
                                                   element)
from pythonLibs.regressionHandler.numerics.Sparse import SparseMatrix


def orientPolyhedron(points: np.ndarray, faces: list) -> list:
    """Faces of a closed polyhedron oriented consistently and outward.

    Neighbouring faces must traverse their shared edge in opposite directions;
    a breadth-first pass flips inconsistent faces, and the whole set is flipped
    if the signed volume (divergence theorem) is negative. Works for
    non-convex cells.
    """
    faces = [np.asarray(f, dtype=np.int64) for f in faces]
    edgeFaces: dict = {}
    for i, f in enumerate(faces):
        for a, b in zip(f, np.roll(f, -1)):
            edgeFaces.setdefault((min(a, b), max(a, b)), []).append(i)
    done = [False] * len(faces)
    for start in range(len(faces)):
        if done[start]:
            continue
        done[start] = True
        queue = [start]
        while queue:
            i = queue.pop()
            f = faces[i]
            directed = set(zip(f.tolist(), np.roll(f, -1).tolist()))
            for a, b in directed:
                for j in edgeFaces[(min(a, b), max(a, b))]:
                    if j == i or done[j]:
                        continue
                    g = faces[j]
                    if (a, b) in set(zip(g.tolist(), np.roll(g, -1).tolist())):
                        faces[j] = g[::-1]
                    done[j] = True
                    queue.append(j)
    vol = 0.0
    for f in faces:
        p = points[f]
        vol += float(np.sum(np.einsum("ij,ij->i", np.repeat(p[:1], len(f) - 2, axis=0),
                                      np.cross(p[1:-1], p[2:]))))
    return [f[::-1] for f in faces] if vol < 0 else faces


def rowGroups(key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(group id per row, rows per group) of identical rows of an integer array (lexsort based)."""
    if key.shape[0] == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    order = np.lexsort(key.T[::-1])
    k = key[order]
    new = np.concatenate([[True], np.any(k[1:] != k[:-1], axis=1)])
    gid = np.cumsum(new) - 1
    inv = np.empty(key.shape[0], dtype=np.int64)
    inv[order] = gid
    return inv, np.bincount(gid)


@dataclass
class Quadrature:
    """Quadrature points of a set of cells.

    cell:    (Q,) owning cell of each point
    weight:  (Q,) physical weights (volume / area / length elements)
    x:       (Q, 3) positions
    interp:  SparseMatrix (Q, nPoints): values at the points = interp @ nodal values
    """
    cell: np.ndarray
    weight: np.ndarray
    x: np.ndarray
    interp: SparseMatrix

    def integrate(self, nodal) -> float:
        return float(self.weight @ (self.interp @ np.asarray(nodal, dtype=float)))

    def cellIntegrals(self, values, nCells: int) -> np.ndarray:
        """Per-cell integrals of quadrature-point values (Q,)."""
        return np.bincount(self.cell, weights=self.weight * values, minlength=nCells)

    def loadVector(self, qpValues) -> np.ndarray:
        """Consistent nodal loads int N_i f: interp^T (weight * f)."""
        return self.interp.rmatvec(self.weight * np.asarray(qpValues, dtype=float))


class UnstructuredMesh:
    """Points, cells (any VTK type incl. polyhedra) and attached point / cell data."""

    def __init__(self, points, cellTypes, connectivity, offsets, polyFaces: Optional[dict] = None,
                 pointData: Optional[dict] = None, cellData: Optional[dict] = None) -> None:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] not in (2, 3):
            raise ValueError("points must be (n, 2) or (n, 3)")
        if pts.shape[1] == 2:
            pts = np.hstack([pts, np.zeros((pts.shape[0], 1))])
        types = np.asarray(cellTypes, dtype=np.uint8).copy()
        conn = np.asarray(connectivity, dtype=np.int64).copy()
        offs = np.asarray(offsets, dtype=np.int64)
        if offs.size == types.size:
            offs = np.concatenate([[0], offs])
        if offs.size != types.size + 1 or offs[-1] != conn.size:
            raise ValueError("offsets must have nCells + 1 entries ending at len(connectivity)")
        # pixel / voxel -> quad / hexahedron numbering
        for old, (new, perm) in RENUMBER.items():
            for c in np.flatnonzero(types == old):
                seg = conn[offs[c]:offs[c + 1]]
                conn[offs[c]:offs[c + 1]] = seg[perm]
                types[c] = new
        if conn.size and (conn.min() < 0 or conn.max() >= pts.shape[0]):
            raise ValueError("connectivity refers to missing points")
        self.points = pts
        self.cellTypes = types
        self.connectivity = conn
        self.offsets = offs
        self.polyFaces = {int(k): orientPolyhedron(pts, [np.asarray(f, dtype=np.int64) for f in v])
                          for k, v in (polyFaces or {}).items()}
        missing = set(np.flatnonzero(types == POLYHEDRON).tolist()) - set(self.polyFaces)
        if missing:
            raise ValueError(f"polyhedral cells without faces: {sorted(missing)[:5]}")
        self.pointData = dict(pointData or {})
        self.cellData = dict(cellData or {})

    # ------------------------------------------------------------------ construction helpers
    @classmethod
    def fromBlocks(cls, points, blocks: Sequence[tuple[int, np.ndarray]], polyhedra: Optional[list] = None
                   ) -> "UnstructuredMesh":
        """From (vtkType, (k, nNodes) connectivity) blocks and optional polyhedra given as face lists."""
        types, conn, offs = [], [], [0]
        for t, c in blocks:
            c = np.atleast_2d(np.asarray(c, dtype=np.int64))
            types.extend([t] * c.shape[0])
            conn.append(c.ravel())
            offs.extend((offs[-1] + np.arange(1, c.shape[0] + 1) * c.shape[1]).tolist())
        polyFaces = {}
        for faces in polyhedra or []:
            nodes = np.unique(np.concatenate([np.asarray(f) for f in faces]))
            polyFaces[len(types)] = faces
            types.append(POLYHEDRON)
            conn.append(nodes)
            offs.append(offs[-1] + nodes.size)
        return cls(points, types, np.concatenate(conn) if conn else np.zeros(0, np.int64), offs, polyFaces)

    @classmethod
    def merge(cls, meshes: Sequence["UnstructuredMesh"]) -> "UnstructuredMesh":
        pts, types, conn, offs, poly = [], [], [], [0], {}
        pShift = cShift = 0
        for m in meshes:
            pts.append(m.points)
            types.append(m.cellTypes)
            conn.append(m.connectivity + pShift)
            offs.extend((m.offsets[1:] + offs[-1]).tolist())
            poly.update({c + cShift: [f + pShift for f in fl] for c, fl in m.polyFaces.items()})
            pShift += m.nPoints
            cShift += m.nCells
        out = cls(np.vstack(pts), np.concatenate(types), np.concatenate(conn), offs, poly)
        for attr in ("pointData", "cellData"):
            names = set.intersection(*[set(getattr(m, attr)) for m in meshes])
            for n in names:
                getattr(out, attr)[n] = np.concatenate([np.asarray(getattr(m, attr)[n]) for m in meshes])
        return out

    # ------------------------------------------------------------------ basic access
    @property
    def nPoints(self) -> int:
        return self.points.shape[0]

    @property
    def nCells(self) -> int:
        return self.cellTypes.size

    def cell(self, c: int) -> np.ndarray:
        return self.connectivity[self.offsets[c]:self.offsets[c + 1]]

    def cellSizes(self) -> np.ndarray:
        return np.diff(self.offsets)

    def cellDimensions(self) -> np.ndarray:
        dims = {int(t): cellDimension(int(t)) for t in np.unique(self.cellTypes)}
        return np.array([dims[int(t)] for t in self.cellTypes], dtype=int)

    def cellCentroids(self) -> np.ndarray:
        """Vertex averages of the cells."""
        cid = np.repeat(np.arange(self.nCells), self.cellSizes())
        s = np.zeros((self.nCells, 3))
        np.add.at(s, cid, self.points[self.connectivity])
        return s / np.maximum(self.cellSizes(), 1)[:, None]

    def blocks(self, cells: Optional[np.ndarray] = None) -> dict:
        """{vtkType: (cellIds, (k, nNodes) connectivity)} for standard cells (polygons / polyhedra excluded)."""
        idx = np.arange(self.nCells) if cells is None else np.asarray(cells, dtype=np.int64)
        out = {}
        for t in np.unique(self.cellTypes[idx]):
            if t in (POLYGON, POLYHEDRON):
                continue
            ids = idx[self.cellTypes[idx] == t]
            n = element(int(t)).nNodes
            sizes = self.offsets[ids + 1] - self.offsets[ids]
            if np.any(sizes != n):
                raise ValueError(f"cells of type {int(t)} must have {n} nodes")
            conn = self.connectivity[self.offsets[ids][:, None] + np.arange(n)]
            out[int(t)] = (ids, conn)
        return out

    def subset(self, cells, compact: bool = True) -> tuple["UnstructuredMesh", np.ndarray]:
        """Mesh of the selected cells; returns (mesh, old point ids of its points)."""
        cells = np.asarray(cells)
        if cells.dtype == bool:
            cells = np.flatnonzero(cells)
        sizes = self.offsets[cells + 1] - self.offsets[cells]
        seg = np.concatenate([self.connectivity[self.offsets[c]:self.offsets[c + 1]] for c in cells]) \
            if cells.size else np.zeros(0, np.int64)
        keep = np.unique(seg) if compact else np.arange(self.nPoints)
        remap = np.full(self.nPoints, -1, dtype=np.int64)
        remap[keep] = np.arange(keep.size)
        poly = {i: [remap[f] for f in self.polyFaces[int(c)]] for i, c in enumerate(cells) if int(c) in self.polyFaces}
        out = UnstructuredMesh(self.points[keep], self.cellTypes[cells], remap[seg],
                               np.concatenate([[0], np.cumsum(sizes)]), poly)
        out.pointData = {k: np.asarray(v)[keep] for k, v in self.pointData.items()}
        out.cellData = {k: np.asarray(v)[cells] for k, v in self.cellData.items()}
        return out, keep

    # ------------------------------------------------------------------ integration
    def quadrature(self, order: int = 2, cells: Optional[np.ndarray] = None) -> Quadrature:
        """Quadrature of all (or the selected) cells of the highest dimension present among them."""
        idx = np.arange(self.nCells) if cells is None else np.asarray(cells, dtype=np.int64)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        parts = []
        for t, (ids, conn) in self.blocks(idx).items():
            parts.append(self._standardQuadrature(element(t), ids, conn, order))
        poly = idx[self.cellTypes[idx] == POLYGON]
        if poly.size:
            parts.append(self._polygonQuadrature(poly, order))
        polyh = idx[self.cellTypes[idx] == POLYHEDRON]
        if polyh.size:
            parts.append(self._polyhedronQuadrature(polyh, order))
        if not parts:
            return Quadrature(np.zeros(0, np.int64), np.zeros(0), np.zeros((0, 3)),
                              SparseMatrix([], [], [], (0, self.nPoints)))
        cell = np.concatenate([p[0] for p in parts])
        weight = np.concatenate([p[1] for p in parts])
        x = np.vstack([p[2] for p in parts])
        offs = np.cumsum([0] + [p[0].size for p in parts])
        rows = np.concatenate([p[3][0] + o for p, o in zip(parts, offs)])
        cols = np.concatenate([p[3][1] for p in parts])
        vals = np.concatenate([p[3][2] for p in parts])
        return Quadrature(cell, weight, x, SparseMatrix.raw(rows, cols, vals, (cell.size, self.nPoints)))

    @staticmethod
    def _measure(jac: np.ndarray) -> np.ndarray:
        """|det J| (3-D), |J1 x J2| (2-D) or |J1| (1-D) for jac (..., 3, dim)."""
        dim = jac.shape[-1]
        if dim == 3:
            return np.abs(np.linalg.det(jac))
        if dim == 2:
            return np.linalg.norm(np.cross(jac[..., 0], jac[..., 1]), axis=-1)
        return np.linalg.norm(jac[..., 0], axis=-1)

    def _standardQuadrature(self, el, ids, conn, order):
        p, w = el.quadrature(order)
        n = el.shape(p)                                      # (q, a)
        dn = el.dshape(p)                                    # (q, a, dim)
        xc = self.points[conn]                               # (k, a, 3)
        x = np.einsum("qa,kad->kqd", n, xc)
        jac = np.einsum("qae,kad->kqde", dn, xc)             # (k, q, 3, dim)
        meas = self._measure(jac) * w[None, :]
        k, q = conn.shape[0], p.shape[0]
        rows = np.repeat(np.arange(k * q), el.nNodes)
        cols = np.repeat(conn, q, axis=0).ravel()
        vals = np.tile(n.ravel(), k)
        return np.repeat(ids, q), meas.ravel(), x.reshape(-1, 3), (rows, cols, vals)

    def _fanPieces(self, faceCell, faceNodes, faceOffs, withCell: bool):
        """Simplices of a centroid fan: per edge (v_i, v_i+1) of every face -> (cell, face, a, b)."""
        sizes = np.diff(faceOffs)
        f = np.repeat(np.arange(sizes.size), sizes)
        pos = np.arange(faceNodes.size) - np.repeat(faceOffs[:-1], sizes)
        nxt = np.where(pos + 1 < sizes[f], np.arange(faceNodes.size) + 1, faceOffs[:-1][f])
        return faceCell[f], f, faceNodes, faceNodes[nxt]

    def _polygonQuadrature(self, cells, order):
        sizes = self.offsets[cells + 1] - self.offsets[cells]
        nodes = np.concatenate([self.cell(c) for c in cells])
        faceOffs = np.concatenate([[0], np.cumsum(sizes)])
        cellOf, f, a, b = self._fanPieces(cells, nodes, faceOffs, False)
        centre = np.zeros((cells.size, 3))
        np.add.at(centre, np.repeat(np.arange(cells.size), sizes), self.points[nodes])
        centre /= sizes[:, None]
        tri = element(TRIANGLE)
        p, w = tri.quadrature(order)
        lam = tri.shape(p)                                   # (q, 3): centre, a, b
        verts = np.stack([centre[f], self.points[a], self.points[b]], axis=1)   # (m, 3, 3)
        x = np.einsum("qv,mvd->mqd", lam, verts)
        area = 0.5 * np.linalg.norm(np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0]), axis=1)
        weight = (2.0 * area[:, None] * w[None, :]).ravel()
        m, q = f.size, p.shape[0]
        qpId = np.arange(m * q).reshape(m, q)
        # interpolation: centre value = average of the polygon's vertices
        rowsA, colsA, valsA = qpId.ravel(), np.repeat(a, q), np.tile(lam[:, 1], m)
        rowsB, colsB, valsB = qpId.ravel(), np.repeat(b, q), np.tile(lam[:, 2], m)
        # centre part: every qp of piece j gets lam0 / size for each vertex of its polygon
        cnt = sizes[f]
        rowsC = np.repeat(qpId.ravel(), np.repeat(cnt, q))
        vstart = faceOffs[:-1][f]
        colIdx = np.concatenate([np.tile(np.arange(s0, s0 + c), q) for s0, c in zip(vstart, cnt)]) \
            if m else np.zeros(0, np.int64)
        colsC = nodes[colIdx]
        valsC = np.repeat((lam[None, :, 0] / cnt[:, None]).ravel(), np.repeat(cnt, q))
        return (np.repeat(cellOf, q), weight, x.reshape(-1, 3),
                (np.concatenate([rowsA, rowsB, rowsC]), np.concatenate([colsA, colsB, colsC]),
                 np.concatenate([valsA, valsB, valsC])))

    def _polyhedronFaces(self, cells):
        faceCell, faceNodes, sizes = [], [], []
        for c in cells:
            for f in self.polyFaces[int(c)]:
                faceCell.append(c)
                faceNodes.append(f)
                sizes.append(len(f))
        return (np.array(faceCell, dtype=np.int64), np.concatenate(faceNodes).astype(np.int64),
                np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64))

    def _polyhedronQuadrature(self, cells, order):
        faceCell, faceNodes, faceOffs = self._polyhedronFaces(cells)
        fsize = np.diff(faceOffs)
        cellOf, f, a, b = self._fanPieces(faceCell, faceNodes, faceOffs, True)
        # cell centre from the unique vertices, face centres from the face vertices
        local = {int(c): i for i, c in enumerate(cells)}
        li = np.array([local[int(c)] for c in cellOf], dtype=np.int64)
        csize = self.offsets[cells + 1] - self.offsets[cells]
        cc = np.zeros((cells.size, 3))
        np.add.at(cc, np.repeat(np.arange(cells.size), csize), self.points[np.concatenate([self.cell(c) for c in cells])])
        cc /= csize[:, None]
        fc = np.zeros((fsize.size, 3))
        np.add.at(fc, np.repeat(np.arange(fsize.size), fsize), self.points[faceNodes])
        fc /= fsize[:, None]
        tet = element(TETRA)
        p, w = tet.quadrature(order)
        lam = tet.shape(p)                                   # (q, 4): cell centre, face centre, a, b
        verts = np.stack([cc[li], fc[f], self.points[a], self.points[b]], axis=1)
        x = np.einsum("qv,mvd->mqd", lam, verts)
        # signed volumes with outward faces: exact for any closed polyhedron (also non-convex)
        vol = np.einsum("md,md->m", np.cross(verts[:, 2] - verts[:, 1], verts[:, 3] - verts[:, 1]),
                        verts[:, 1] - verts[:, 0]) / 6.0
        weight = (6.0 * vol[:, None] * w[None, :]).ravel()
        m, q = f.size, p.shape[0]
        qp = np.arange(m * q)
        rows = [qp, qp]
        cols = [np.repeat(a, q), np.repeat(b, q)]
        vals = [np.tile(lam[:, 2], m), np.tile(lam[:, 3], m)]
        # face-centre part
        cnt = fsize[f]
        rows.append(np.repeat(qp, np.repeat(cnt, q)))
        cols.append(np.concatenate([np.tile(faceNodes[faceOffs[j]:faceOffs[j + 1]], q) for j in f])
                    if m else np.zeros(0, np.int64))
        vals.append(np.repeat((lam[None, :, 1] / cnt[:, None]).ravel(), np.repeat(cnt, q)))
        # cell-centre part
        ccnt = csize[li]
        rows.append(np.repeat(qp, np.repeat(ccnt, q)))
        cols.append(np.concatenate([np.tile(self.cell(c), q) for c in cellOf]) if m else np.zeros(0, np.int64))
        vals.append(np.repeat((lam[None, :, 0] / ccnt[:, None]).ravel(), np.repeat(ccnt, q)))
        return (np.repeat(cellOf, q), weight, x.reshape(-1, 3),
                (np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)))

    def cellMeasures(self, order: int = 2) -> np.ndarray:
        """Volume / area / length of every cell."""
        q = self.quadrature(order)
        return np.bincount(q.cell, weights=q.weight, minlength=self.nCells)

    # ------------------------------------------------------------------ faces and surfaces
    def faces(self, cells: Optional[np.ndarray] = None) -> dict:
        """All faces of the 3-D cells: owner cell, local index, VTK face type and node lists (CSR).

        Faces are oriented outward from their owner cell (standard faces by the
        element definitions, reversed for inverted cells; polyhedral faces as
        stored, which ``orientPolyhedron`` makes consistent and outward).
        Returns {"cell", "local", "type", "nodes", "offsets", "key"} where key
        rows identify the same geometric face from both sides.
        """
        idx = np.arange(self.nCells) if cells is None else np.asarray(cells, dtype=np.int64)
        idx = idx[self.cellDimensions()[idx] == 3]
        blocks = []                                   # (cell, local, type, nodes (k, m), corners (k, c))
        for t, (ids, conn) in self.blocks(idx).items():
            el = element(t)
            # inverted cells (negative Jacobian) get their faces reversed so they still point outward
            centre = el.ref.mean(axis=0, keepdims=True)
            jac = np.einsum("ae,kad->kde", el.dshape(centre)[0], self.points[conn])
            inverted = np.linalg.det(jac) * el.orientation < 0
            for j, (fl, ft) in enumerate(zip(el.faces, el.faceTypes)):
                nodes = conn[:, fl]
                k = 3 if ft in (5, 22) else 4
                if inverted.any():
                    rev = [0] + list(range(k - 1, 0, -1))
                    if len(fl) > k:
                        rev += [k + i for i in range(k - 1, -1, -1)]
                    nodes = np.where(inverted[:, None], nodes[:, rev], nodes)
                blocks.append((ids, np.full(ids.size, j), np.full(ids.size, ft), nodes, nodes[:, :k]))
        polyh = idx[self.cellTypes[idx] == POLYHEDRON]
        for c in polyh:                                # stored outward-oriented
            for j, f in enumerate(self.polyFaces[int(c)]):
                f = np.asarray(f)
                ft = POLYGON if f.size > 4 else (5 if f.size == 3 else 9)
                blocks.append((np.array([c]), np.array([j]), np.array([ft]), f[None, :], f[None, :]))
        if not blocks:
            return {"cell": np.zeros(0, np.int64), "local": np.zeros(0, np.int64), "type": np.zeros(0, np.uint8),
                    "nodes": np.zeros(0, np.int64), "offsets": np.zeros(1, np.int64), "key": np.zeros((0, 3), np.int64)}
        width = max(b[4].shape[1] for b in blocks)
        keys = []
        for b in blocks:
            kk = np.full((b[4].shape[0], width), -1, dtype=np.int64)
            kk[:, :b[4].shape[1]] = np.sort(b[4], axis=1)
            keys.append(kk)
        key = np.vstack(keys)
        key.sort(axis=1)
        sizes = np.concatenate([np.full(b[3].shape[0], b[3].shape[1]) for b in blocks])
        return {"cell": np.concatenate([b[0] for b in blocks]), "local": np.concatenate([b[1] for b in blocks]),
                "type": np.concatenate([b[2] for b in blocks]).astype(np.uint8),
                "nodes": np.concatenate([b[3].ravel() for b in blocks]).astype(np.int64),
                "offsets": np.concatenate([[0], np.cumsum(sizes)]), "key": key}

    def _faceMesh(self, f: dict, select: np.ndarray, extra: Optional[dict] = None) -> "UnstructuredMesh":
        sel = np.flatnonzero(select)
        sizes = f["offsets"][sel + 1] - f["offsets"][sel]
        pos = np.repeat(f["offsets"][sel], sizes) + (np.arange(int(sizes.sum())) - np.repeat(np.cumsum(sizes) - sizes, sizes))
        nodes = f["nodes"][pos]
        out = UnstructuredMesh(self.points, f["type"][sel], nodes, np.concatenate([[0], np.cumsum(sizes)]))
        out.cellData["ownerCell"] = f["cell"][sel]
        out.cellData["ownerFace"] = f["local"][sel]
        for k, v in (extra or {}).items():
            out.cellData[k] = v[sel]
        for k, v in self.cellData.items():
            out.cellData.setdefault(k, np.asarray(v)[f["cell"][sel]])
        return out

    def boundarySurface(self, cells: Optional[np.ndarray] = None) -> "UnstructuredMesh":
        """Boundary faces of the 3-D cells as a surface mesh on the same points (outward oriented).

        cellData: ownerCell, ownerFace and the owner's cell data.
        """
        f = self.faces(cells)
        inv, counts = rowGroups(f["key"])
        return self._faceMesh(f, counts[inv] == 1)

    def interfaceSurface(self, labels, cells: Optional[np.ndarray] = None) -> "UnstructuredMesh":
        """Internal faces between cells with different ``labels`` (one copy, oriented out of the lower label).

        cellData: ownerCell (lower label side), neighbourCell, labelBelow / labelAbove.
        """
        labels = np.asarray(labels)
        f = self.faces(cells)
        inv, counts = rowGroups(f["key"])
        shared = np.flatnonzero(counts[inv] == 2)
        order = shared[np.argsort(inv[shared], kind="stable")]
        a, b = order[0::2], order[1::2]
        la, lb = labels[f["cell"][a]], labels[f["cell"][b]]
        diff = la != lb
        a, b, la, lb = a[diff], b[diff], la[diff], lb[diff]
        swap = la > lb
        own = np.where(swap, b, a)
        other = np.where(swap, a, b)
        sel = np.zeros(f["cell"].size, dtype=bool)
        sel[own] = True
        extraN = np.zeros(f["cell"].size, dtype=np.int64)
        extraN[own] = f["cell"][other]
        lab = np.zeros(f["cell"].size, dtype=labels.dtype)
        labB = np.zeros(f["cell"].size, dtype=labels.dtype)
        lab[own] = labels[f["cell"][own]]
        labB[own] = labels[f["cell"][other]]
        return self._faceMesh(f, sel, {"neighbourCell": extraN, "labelBelow": lab, "labelAbove": labB})

    def triangulate(self) -> tuple[np.ndarray, np.ndarray]:
        """Flat triangles of the 2-D cells: ((m, 3) point-id triangles, (m,) parent cell).

        Quads split into 2 triangles, quadratic triangles into 4 and quadratic
        quads into 6 using their mid-side nodes, polygons by a fan from their
        first vertex (orientation kept).
        """
        tris, parent = [], []
        for t, (ids, conn) in self.blocks(np.flatnonzero(self.cellDimensions() == 2)).items():
            if t == 5:
                pieces = [[0, 1, 2]]
            elif t == 9:
                pieces = [[0, 1, 2], [0, 2, 3]]
            elif t == 22:
                pieces = [[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]]
            elif t in (23, 28):
                pieces = [[0, 4, 7], [4, 1, 5], [5, 2, 6], [6, 3, 7], [4, 5, 7], [5, 6, 7]]
            else:
                raise ValueError(f"cannot triangulate cell type {t}")
            for pc in pieces:
                tris.append(conn[:, pc])
                parent.append(ids)
        for c in np.flatnonzero(self.cellTypes == POLYGON):
            v = self.cell(c)
            tris.append(np.column_stack([np.full(v.size - 2, v[0]), v[1:-1], v[2:]]))
            parent.append(np.full(v.size - 2, c))
        if not tris:
            return np.zeros((0, 3), np.int64), np.zeros(0, np.int64)
        return np.vstack(tris), np.concatenate(parent)

    def __repr__(self) -> str:
        from pythonLibs.fieldMapping.mesh.Elements import TYPE_NAMES
        kinds = {TYPE_NAMES.get(int(t), str(t)): int(n) for t, n in zip(*np.unique(self.cellTypes, return_counts=True))}
        return f"<UnstructuredMesh {self.nPoints} points, {self.nCells} cells {kinds}>"
