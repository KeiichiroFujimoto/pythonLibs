"""Simple structured meshes for examples, verification and convergence studies.

    structuredBox       hexahedra / tetrahedra / wedges / quadratic hexahedra / polyhedra on a box
    planeSurface        quads / triangles / quadratic facets / polygons on a rectangle (optional jitter)
    sphereSurface       cubed-sphere quads / triangles on a sphere
    sphericalShell      layered cubed-sphere shell of (quadratic) hexahedra with a "layer" cell array
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from pythonLibs.fieldMapping.mesh.Mesh import UnstructuredMesh

_HEX_TETS = [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]]   # Kuhn: conforming
_HEX_WEDGES = [[0, 1, 3, 4, 5, 7], [1, 2, 3, 5, 6, 7]]
_HEX_FACES = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
_HEX20_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def _gridIndex(n):
    nx, ny, nz = n
    return lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k


def _hexCells(n):
    nx, ny, nz = n
    idx = _gridIndex(n)
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    return np.column_stack([idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k),
                            idx(i, j, k + 1), idx(i + 1, j, k + 1), idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1)])


def _addMidNodes(points, cells, edges):
    """Quadratic cells: mid-edge nodes shared between neighbours."""
    e = np.concatenate([np.sort(cells[:, list(p)], axis=1) for p in edges])
    uniq, inv = np.unique(e, axis=0, return_inverse=True)
    mids = 0.5 * (points[uniq[:, 0]] + points[uniq[:, 1]])
    ids = points.shape[0] + inv.ravel().reshape(len(edges), -1).T
    return np.vstack([points, mids]), np.hstack([cells, ids])


def structuredBox(lower=(0, 0, 0), upper=(1, 1, 1), n=(4, 4, 4), cellType: str = "hexahedron",
                  mapping=None) -> UnstructuredMesh:
    """Box mesh; cellType: hexahedron, tetra, wedge, quadraticHexahedron or polyhedron.

    ``mapping`` (optional) maps the (m, 3) points to new coordinates (curvilinear meshes).
    """
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    axes = [np.linspace(lo[d], hi[d], int(n[d]) + 1) for d in range(3)]
    g = np.meshgrid(*axes, indexing="ij")
    pts = np.column_stack([a.ravel() for a in g])
    hexes = _hexCells(tuple(int(v) for v in n))
    if cellType == "hexahedron":
        blocks, poly = [(12, hexes)], None
    elif cellType == "tetra":
        blocks, poly = [(10, np.vstack([hexes[:, t] for t in _HEX_TETS]))], None
    elif cellType == "wedge":
        blocks, poly = [(13, np.vstack([hexes[:, w] for w in _HEX_WEDGES]))], None
    elif cellType == "quadraticHexahedron":
        pts, h20 = _addMidNodes(pts, hexes, _HEX20_EDGES)
        blocks, poly = [(25, h20)], None
    elif cellType == "polyhedron":
        blocks, poly = [], [[list(h[f]) for f in _HEX_FACES] for h in hexes]
    else:
        raise ValueError("cellType must be hexahedron, tetra, wedge, quadraticHexahedron or polyhedron")
    if mapping is not None:
        pts = np.asarray(mapping(pts), dtype=float)
    return UnstructuredMesh.fromBlocks(pts, blocks, poly)


def planeSurface(lower=(0, 0), upper=(1, 1), n=(4, 4), cellType: str = "quad", jitter: float = 0.0, seed: int = 0,
                 z: float = 0.0) -> UnstructuredMesh:
    """Rectangle in the plane z = const; cellType quad, triangle, quadraticQuad, quadraticTriangle or polygon."""
    nx, ny = int(n[0]), int(n[1])
    x, y = np.meshgrid(np.linspace(lower[0], upper[0], nx + 1), np.linspace(lower[1], upper[1], ny + 1),
                       indexing="ij")
    pts = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, float(z))])
    if jitter:
        rng = np.random.default_rng(seed)
        inner = (x.ravel() > lower[0]) & (x.ravel() < upper[0]) & (y.ravel() > lower[1]) & (y.ravel() < upper[1])
        h = np.array([(upper[0] - lower[0]) / nx, (upper[1] - lower[1]) / ny])
        pts[inner, :2] += rng.uniform(-jitter, jitter, (int(inner.sum()), 2)) * h
    idx = lambda i, j: i * (ny + 1) + j                                     # noqa: E731
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    i, j = i.ravel(), j.ravel()
    quads = np.column_stack([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])
    if cellType == "quad":
        return UnstructuredMesh.fromBlocks(pts, [(9, quads)])
    if cellType == "polygon":
        return UnstructuredMesh.fromBlocks(pts, [(7, quads)])
    tris = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    if cellType == "triangle":
        return UnstructuredMesh.fromBlocks(pts, [(5, tris)])
    if cellType == "quadraticQuad":
        p2, q8 = _addMidNodes(pts, quads, [(0, 1), (1, 2), (2, 3), (3, 0)])
        return UnstructuredMesh.fromBlocks(p2, [(23, q8)])
    if cellType == "quadraticTriangle":
        p2, t6 = _addMidNodes(pts, tris, [(0, 1), (1, 2), (2, 0)])
        return UnstructuredMesh.fromBlocks(p2, [(22, t6)])
    raise ValueError("cellType must be quad, triangle, quadraticQuad, quadraticTriangle or polygon")


def _cubeFaceGrid(n):
    """Points and quads of the 6 faces of the cube [-1, 1]^3 (duplicate edge points merged)."""
    t = np.linspace(-1, 1, n + 1)
    a, b = np.meshgrid(t, t, indexing="ij")
    a, b = a.ravel(), b.ravel()
    faces = []
    for axis in range(3):
        for s in (-1.0, 1.0):
            p = np.zeros((a.size, 3))
            others = [d for d in range(3) if d != axis]
            p[:, axis] = s
            p[:, others[0]], p[:, others[1]] = (a, b) if s > 0 else (b, a)
            faces.append(p)
    pts = np.vstack(faces)
    key = np.round(pts * 1e9).astype(np.int64)
    uniq, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    idx = lambda f, i, j: inv[f * (n + 1) ** 2 + i * (n + 1) + j]          # noqa: E731
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    i, j = i.ravel(), j.ravel()
    quads = np.vstack([np.column_stack([idx(f, i, j), idx(f, i + 1, j), idx(f, i + 1, j + 1), idx(f, i, j + 1)])
                       for f in range(6)])
    return pts[first], quads


def sphereSurface(radius: float = 1.0, n: int = 8, cellType: str = "quad") -> UnstructuredMesh:
    """Cubed-sphere surface (outward facets); cellType quad or triangle."""
    pts, quads = _cubeFaceGrid(n)
    pts = radius * pts / np.linalg.norm(pts, axis=1)[:, None]
    # orient outward
    c = pts[quads].mean(axis=1)
    nrm = np.cross(pts[quads[:, 1]] - pts[quads[:, 0]], pts[quads[:, 3]] - pts[quads[:, 0]])
    flip = np.einsum("ij,ij->i", nrm, c) < 0
    quads[flip] = quads[flip][:, ::-1]
    if cellType == "quad":
        return UnstructuredMesh.fromBlocks(pts, [(9, quads)])
    if cellType == "triangle":
        return UnstructuredMesh.fromBlocks(pts, [(5, np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]))])
    raise ValueError("cellType must be quad or triangle")


def sphericalShell(radii: Sequence[float] = (0.9, 1.0), n: int = 6, layers: Sequence[int] = (2,),
                   quadratic: bool = False) -> UnstructuredMesh:
    """Cubed-sphere shell between radii[0] < ... < radii[-1] with ``layers[i]`` hex layers in shell i.

    Cell data "layer" numbers the shells from the outside (0 = outermost).
    Quadratic cells put their mid-side nodes on the exact spherical geometry.
    """
    radii = np.asarray(radii, dtype=float)
    if radii.size != len(layers) + 1:
        raise ValueError("give one layer count per shell (len(radii) - 1)")
    pts2, quads = _cubeFaceGrid(n)
    dirs = pts2 / np.linalg.norm(pts2, axis=1)[:, None]
    rs, shell = [radii[0]], []
    for s, k in enumerate(layers):
        rs.extend(np.linspace(radii[s], radii[s + 1], k + 1)[1:].tolist())
        shell.extend([s] * k)
    rs = np.array(rs)
    nd = dirs.shape[0]
    pts = np.vstack([r * dirs for r in rs])
    cells = []
    c = pts2[quads].mean(axis=1)
    nrm = np.cross(pts2[quads[:, 1]] - pts2[quads[:, 0]], pts2[quads[:, 3]] - pts2[quads[:, 0]])
    q = quads.copy()
    flip = np.einsum("ij,ij->i", nrm, c) < 0
    q[flip] = q[flip][:, ::-1]                    # bottom quad counter-clockwise seen from outside: positive Jacobian
    for L in range(rs.size - 1):
        cells.append(np.hstack([q + L * nd, q + (L + 1) * nd]))
    hexes = np.vstack(cells)
    shellIdx = np.concatenate([np.full(q.shape[0], shell[L]) for L in range(rs.size - 1)])
    if quadratic:
        pts, h20 = _addMidNodes(pts, hexes, _HEX20_EDGES)
        # project mid-side nodes onto their sphere (radius = mean of the edge end radii)
        e = np.concatenate([np.sort(hexes[:, list(p)], axis=1) for p in _HEX20_EDGES])
        uniq = np.unique(e, axis=0)
        r = 0.5 * (np.linalg.norm(pts[uniq[:, 0]], axis=1) + np.linalg.norm(pts[uniq[:, 1]], axis=1))
        pm = pts[rs.size * nd:]
        pts[rs.size * nd:] = pm / np.linalg.norm(pm, axis=1)[:, None] * r[:, None]
        mesh = UnstructuredMesh.fromBlocks(pts, [(25, h20)])
    else:
        mesh = UnstructuredMesh.fromBlocks(pts, [(12, hexes)])
    nShell = radii.size - 1
    mesh.cellData["layer"] = (nShell - 1 - shellIdx).astype(np.int32)
    return mesh
