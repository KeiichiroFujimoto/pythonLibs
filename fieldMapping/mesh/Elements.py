"""Finite-element library in VTK node ordering: shape functions, faces and quadrature.

Every standard VTK cell type used for field transfer is described by

    dim          topological dimension (1, 2, 3)
    ref          reference coordinates of its nodes
    shape(xi)    shape functions N (q, nNodes) at reference points xi (q, dim)
    dshape(xi)   derivatives dN/dxi (q, nNodes, dim)
    faces        local node lists of the boundary faces (3-D cells; corner nodes
                 first, then mid-side nodes for quadratic cells) and their VTK types
    quadrature(order)   reference points and weights exact for polynomials of
                        total degree ``order`` (collapsed Gauss rules on simplices)

Linear: vertex, line, triangle, quad (pixel), tetra, hexahedron (voxel), wedge,
pyramid. Quadratic: quadraticEdge, quadraticTriangle, quadraticQuad
(serendipity), biquadraticQuad, quadraticTetra, quadraticHexahedron
(serendipity), quadraticWedge. Polygons and polyhedra are handled by
decomposition in ``mesh.Mesh``. Pixel / voxel are renumbered to quad / hex.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

import numpy as np

# ---------------------------------------------------------------- VTK type ids
VERTEX, LINE, TRIANGLE, POLYGON, PIXEL, QUAD = 1, 3, 5, 7, 8, 9
TETRA, VOXEL, HEXAHEDRON, WEDGE, PYRAMID = 10, 11, 12, 13, 14
QUADRATIC_EDGE, QUADRATIC_TRIANGLE, QUADRATIC_QUAD, QUADRATIC_TETRA = 21, 22, 23, 24
QUADRATIC_HEXAHEDRON, QUADRATIC_WEDGE, BIQUADRATIC_QUAD, POLYHEDRON = 25, 26, 28, 42

TYPE_NAMES = {VERTEX: "vertex", LINE: "line", TRIANGLE: "triangle", POLYGON: "polygon", PIXEL: "pixel",
              QUAD: "quad", TETRA: "tetra", VOXEL: "voxel", HEXAHEDRON: "hexahedron", WEDGE: "wedge",
              PYRAMID: "pyramid", QUADRATIC_EDGE: "quadraticEdge", QUADRATIC_TRIANGLE: "quadraticTriangle",
              QUADRATIC_QUAD: "quadraticQuad", QUADRATIC_TETRA: "quadraticTetra",
              QUADRATIC_HEXAHEDRON: "quadraticHexahedron", QUADRATIC_WEDGE: "quadraticWedge",
              BIQUADRATIC_QUAD: "biquadraticQuad", POLYHEDRON: "polyhedron"}

# pixel / voxel -> quad / hexahedron node order
RENUMBER = {PIXEL: (QUAD, [0, 1, 3, 2]), VOXEL: (HEXAHEDRON, [0, 1, 3, 2, 4, 5, 7, 6])}


def gaussLegendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes / weights on [-1, 1]."""
    k = np.arange(1, n)
    beta = k / np.sqrt(4.0 * k * k - 1.0)
    x, v = np.linalg.eigh(np.diag(beta, 1) + np.diag(beta, -1))
    return x, 2.0 * v[0] ** 2


def gaussJacobi(n: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes / weights on [-1, 1] for the weight (1 - x)^alpha (Golub-Welsch)."""
    a, b = float(alpha), 0.0
    k = np.arange(n, dtype=float)
    ab = a + b
    diag = np.where(2 * k + ab == 0, (b - a) / (ab + 2), (b * b - a * a) / ((2 * k + ab) * (2 * k + ab + 2)))
    kk = np.arange(1, n, dtype=float)
    off = np.sqrt(4 * kk * (kk + a) * (kk + b) * (kk + ab) / ((2 * kk + ab) ** 2 * (2 * kk + ab + 1) * (2 * kk + ab - 1)))
    x, v = np.linalg.eigh(np.diag(diag) + np.diag(off, 1) + np.diag(off, -1))
    mu0 = 2.0 ** (ab + 1) / (a + 1)                    # integral of (1-x)^a on [-1, 1] (b = 0)
    return x, mu0 * v[0] ** 2


# ---------------------------------------------------------------- reference quadrature
@lru_cache(maxsize=None)
def lineRule(order: int):
    n = order // 2 + 1
    x, w = gaussLegendre(n)
    return x[:, None], w


@lru_cache(maxsize=None)
def squareRule(order: int):
    x, w = gaussLegendre(order // 2 + 1)
    a, b = np.meshgrid(x, x, indexing="ij")
    wa, wb = np.meshgrid(w, w, indexing="ij")
    return np.column_stack([a.ravel(), b.ravel()]), (wa * wb).ravel()


@lru_cache(maxsize=None)
def cubeRule(order: int):
    x, w = gaussLegendre(order // 2 + 1)
    g = np.meshgrid(x, x, x, indexing="ij")
    gw = np.meshgrid(w, w, w, indexing="ij")
    return np.column_stack([a.ravel() for a in g]), np.prod([a.ravel() for a in gw], axis=0)


@lru_cache(maxsize=None)
def triangleRule(order: int):
    """Collapsed (Duffy) Gauss-Jacobi rule on {r, s >= 0, r + s <= 1}, exact for total degree ``order``."""
    n = order // 2 + 1
    xa, wa = gaussLegendre(n)
    xb, wb = gaussJacobi(n, 1.0)
    u = 0.5 * (1 + xa)                                   # [0, 1]
    v = 0.5 * (1 + xb)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    ww = np.outer(wa, wb) / 8.0                           # (1/2)(1/2)(1/2 for the Jacobi weight scaling)
    r = uu * (1 - vv)
    s = vv
    return np.column_stack([r.ravel(), s.ravel()]), ww.ravel()


@lru_cache(maxsize=None)
def tetraRule(order: int):
    """Collapsed Gauss-Jacobi rule on the unit tetrahedron, exact for total degree ``order``."""
    n = order // 2 + 1
    xa, wa = gaussLegendre(n)
    xb, wb = gaussJacobi(n, 1.0)
    xc, wc = gaussJacobi(n, 2.0)
    a, b, c = 0.5 * (1 + xa), 0.5 * (1 + xb), 0.5 * (1 + xc)
    A, B, C = np.meshgrid(a, b, c, indexing="ij")
    W = (wa[:, None, None] * wb[None, :, None] * wc[None, None, :]) / 64.0
    t = C
    s = B * (1 - C)
    r = A * (1 - B) * (1 - C)
    return np.column_stack([r.ravel(), s.ravel(), t.ravel()]), W.ravel()


@lru_cache(maxsize=None)
def wedgeRule(order: int):
    p, w = triangleRule(order)
    z, wz = gaussLegendre(order // 2 + 1)
    pts = np.column_stack([np.repeat(p, z.size, axis=0), np.tile(z, p.shape[0])])
    return pts, np.repeat(w, z.size) * np.tile(wz, p.shape[0])


@lru_cache(maxsize=None)
def pyramidRule(order: int):
    """Collapsed rule on the pyramid |x|, |y| <= 1 - z, 0 <= z <= 1 (Gauss-Jacobi in z)."""
    n = order // 2 + 2
    x, wx = gaussLegendre(n)
    zj, wz = gaussJacobi(n, 2.0)
    z = 0.5 * (1 + zj)
    X, Y, Z = np.meshgrid(x, x, z, indexing="ij")
    W = wx[:, None, None] * wx[None, :, None] * (wz / 8.0)[None, None, :]
    return np.column_stack([(X * (1 - Z)).ravel(), (Y * (1 - Z)).ravel(), Z.ravel()]), W.ravel()


# ---------------------------------------------------------------- element definitions
class Element:
    """One cell type: reference nodes, shape functions, faces and quadrature."""

    def __init__(self, vtkType: int, dim: int, ref, shape: Callable, dshape: Callable, rule: Callable,
                 faces: Optional[list] = None, faceTypes: Optional[list] = None, corners: Optional[int] = None,
                 linearType: Optional[int] = None, orientation: int = 1) -> None:
        self.vtkType = vtkType
        self.name = TYPE_NAMES[vtkType]
        self.dim = dim
        self.ref = np.asarray(ref, dtype=float)
        self.nNodes = self.ref.shape[0]
        self._shape, self._dshape, self._rule = shape, dshape, rule
        self.faces = faces or []
        self.faceTypes = faceTypes or []
        self.corners = corners or self.nNodes
        self.linearType = linearType or vtkType
        self.quadratic = linearType is not None and linearType != vtkType
        # sign of det(dx/dxi) for a valid VTK cell (the VTK wedge base (0, 1, 2) faces away from (3, 4, 5))
        self.orientation = int(orientation)

    def shape(self, xi) -> np.ndarray:
        return self._shape(np.atleast_2d(np.asarray(xi, dtype=float)))

    def dshape(self, xi) -> np.ndarray:
        return self._dshape(np.atleast_2d(np.asarray(xi, dtype=float)))

    def quadrature(self, order: int):
        return self._rule(max(int(order), 1))

    def __repr__(self) -> str:
        return f"<Element {self.name} ({self.nNodes} nodes)>"


# lines
def _line2(x):
    t = x[:, 0]
    return np.column_stack([0.5 * (1 - t), 0.5 * (1 + t)])


def _dline2(x):
    return np.broadcast_to(np.array([[-0.5], [0.5]]), (x.shape[0], 2, 1)).copy()


def _line3(x):
    t = x[:, 0]
    return np.column_stack([0.5 * t * (t - 1), 0.5 * t * (t + 1), 1 - t * t])


def _dline3(x):
    t = x[:, 0]
    return np.stack([t - 0.5, t + 0.5, -2 * t], axis=1)[:, :, None]


# triangles
def _tri3(x):
    r, s = x[:, 0], x[:, 1]
    return np.column_stack([1 - r - s, r, s])


def _dtri3(x):
    return np.broadcast_to(np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]), (x.shape[0], 3, 2)).copy()


def _tri6(x):
    r, s = x[:, 0], x[:, 1]
    l0, l1, l2 = 1 - r - s, r, s
    return np.column_stack([l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), 4 * l0 * l1, 4 * l1 * l2,
                            4 * l2 * l0])


def _dtri6(x):
    r, s = x[:, 0], x[:, 1]
    l0 = 1 - r - s
    dr = np.column_stack([-(4 * l0 - 1), 4 * r - 1, 0 * r, 4 * (l0 - r), 4 * s, -4 * s])
    ds = np.column_stack([-(4 * l0 - 1), 0 * r, 4 * s - 1, -4 * r, 4 * r, 4 * (l0 - s)])
    return np.stack([dr, ds], axis=2)


# quads on [-1, 1]^2
_Q4 = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)


def _quad4(x):
    return 0.25 * (1 + x[:, None, 0] * _Q4[:, 0]) * (1 + x[:, None, 1] * _Q4[:, 1])


def _dquad4(x):
    a = 0.25 * _Q4[:, 0] * (1 + x[:, None, 1] * _Q4[:, 1])
    b = 0.25 * _Q4[:, 1] * (1 + x[:, None, 0] * _Q4[:, 0])
    return np.stack([a, b], axis=2)


_Q8 = np.vstack([_Q4, [[0, -1], [1, 0], [0, 1], [-1, 0]]])


def _quad8(x):
    xi, eta = x[:, None, 0], x[:, None, 1]
    xc, ec = _Q8[:4, 0], _Q8[:4, 1]
    corner = 0.25 * (1 + xi * xc) * (1 + eta * ec) * (xi * xc + eta * ec - 1)
    xm, em = _Q8[4:, 0], _Q8[4:, 1]
    mid = np.where(xm == 0, 0.5 * (1 - xi * xi) * (1 + eta * em), 0.5 * (1 + xi * xm) * (1 - eta * eta))
    return np.hstack([corner, mid])


def _dquad8(x):
    xi, eta = x[:, None, 0], x[:, None, 1]
    xc, ec = _Q8[:4, 0], _Q8[:4, 1]
    dcx = 0.25 * xc * (1 + eta * ec) * (2 * xi * xc + eta * ec)
    dce = 0.25 * ec * (1 + xi * xc) * (xi * xc + 2 * eta * ec)
    xm, em = _Q8[4:, 0], _Q8[4:, 1]
    dmx = np.where(xm == 0, -xi * (1 + eta * em), 0.5 * xm * (1 - eta * eta))
    dme = np.where(xm == 0, 0.5 * em * (1 - xi * xi), -eta * (1 + xi * xm))
    return np.stack([np.hstack([dcx, dmx]), np.hstack([dce, dme])], axis=2)


_Q9 = np.vstack([_Q8, [[0, 0]]])


def _quad9(x):
    def l(t, a):
        return np.where(a == 0, 1 - t * t, 0.5 * t * (t + a))
    return l(x[:, None, 0], _Q9[:, 0]) * l(x[:, None, 1], _Q9[:, 1])


def _dquad9(x):
    def l(t, a):
        return np.where(a == 0, 1 - t * t, 0.5 * t * (t + a))

    def dl(t, a):
        return np.where(a == 0, -2 * t, t + 0.5 * a)
    xi, eta = x[:, None, 0], x[:, None, 1]
    return np.stack([dl(xi, _Q9[:, 0]) * l(eta, _Q9[:, 1]), l(xi, _Q9[:, 0]) * dl(eta, _Q9[:, 1])], axis=2)


# tetrahedra
def _tet4(x):
    r, s, t = x[:, 0], x[:, 1], x[:, 2]
    return np.column_stack([1 - r - s - t, r, s, t])


def _dtet4(x):
    return np.broadcast_to(np.array([[-1.0, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]), (x.shape[0], 4, 3)).copy()


_TET10_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]


def _tet10(x):
    lam = _tet4(x)
    corner = lam * (2 * lam - 1)
    mid = np.column_stack([4 * lam[:, i] * lam[:, j] for i, j in _TET10_EDGES])
    return np.hstack([corner, mid])


def _dtet10(x):
    lam = _tet4(x)
    dl = _dtet4(x)                                        # (q, 4, 3)
    corner = (4 * lam - 1)[:, :, None] * dl
    mid = np.stack([4 * (lam[:, i, None] * dl[:, j] + lam[:, j, None] * dl[:, i]) for i, j in _TET10_EDGES], axis=1)
    return np.concatenate([corner, mid], axis=1)


# hexahedra on [-1, 1]^3
_H8 = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)


def _hex8(x):
    return 0.125 * np.prod(1 + x[:, None, :] * _H8[None], axis=2)


def _dhex8(x):
    f = 1 + x[:, None, :] * _H8[None]                    # (q, 8, 3)
    out = np.empty(f.shape)
    for k in range(3):
        others = [j for j in range(3) if j != k]
        out[:, :, k] = 0.125 * _H8[None, :, k] * f[:, :, others[0]] * f[:, :, others[1]]
    return out


_H20 = np.vstack([_H8, [[0, -1, -1], [1, 0, -1], [0, 1, -1], [-1, 0, -1],
                        [0, -1, 1], [1, 0, 1], [0, 1, 1], [-1, 0, 1],
                        [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]])


def _hex20(x):
    xi = x[:, None, :]
    c = _H20[None, :8]
    corner = 0.125 * np.prod(1 + xi * c, axis=2) * (np.sum(xi * c, axis=2) - 2)
    m = _H20[None, 8:]
    zero = (m == 0)
    fac = np.where(zero, 1 - xi * xi, 1 + xi * m)
    mid = 0.25 * np.prod(fac, axis=2)
    return np.hstack([corner, mid])


def _dhex20(x):
    xi = x[:, None, :]
    c = _H20[None, :8]
    f = 1 + xi * c
    s = np.sum(xi * c, axis=2) - 2
    dc = np.empty((x.shape[0], 8, 3))
    for k in range(3):
        o = [j for j in range(3) if j != k]
        dc[:, :, k] = 0.125 * f[:, :, o[0]] * f[:, :, o[1]] * c[:, :, k] * (s + f[:, :, k])
    m = _H20[None, 8:]
    zero = (m == 0)
    fac = np.where(zero, 1 - xi * xi, 1 + xi * m)
    dfac = np.where(zero, -2 * xi, m)
    dm = np.empty((x.shape[0], 12, 3))
    for k in range(3):
        o = [j for j in range(3) if j != k]
        dm[:, :, k] = 0.25 * dfac[:, :, k] * fac[:, :, o[0]] * fac[:, :, o[1]]
    return np.concatenate([dc, dm], axis=1)


# wedges: triangle (r, s) x zeta in [-1, 1]; nodes 0-2 at zeta=-1, 3-5 at zeta=+1
def _wedge6(x):
    tri = _tri3(x[:, :2])
    z = x[:, 2:3]
    return np.hstack([tri * 0.5 * (1 - z), tri * 0.5 * (1 + z)])


def _dwedge6(x):
    tri, dtri = _tri3(x[:, :2]), _dtri3(x[:, :2])
    z = x[:, 2][:, None, None]
    lo = np.concatenate([dtri * 0.5 * (1 - z), (-0.5 * tri)[:, :, None]], axis=2)
    hi = np.concatenate([dtri * 0.5 * (1 + z), (0.5 * tri)[:, :, None]], axis=2)
    return np.concatenate([lo, hi], axis=1)


_W15_TRI_EDGES = [(0, 1), (1, 2), (2, 0)]


def _wedge15(x):
    lam = _tri3(x[:, :2])
    z = x[:, 2:3]
    out = []
    for zi in (-1.0, 1.0):
        out.append(0.5 * lam * ((2 * lam - 1) * (1 + zi * z) - (1 - z * z)))
    for zi in (-1.0, 1.0):
        out.append(np.column_stack([2 * lam[:, i] * lam[:, j] for i, j in _W15_TRI_EDGES]) * (1 + zi * z))
    out.append(lam * (1 - z * z))
    return np.hstack(out)


def _dwedge15(x):
    lam = _tri3(x[:, :2])
    dl = _dtri3(x[:, :2])[:, :, :]                        # (q, 3, 2)
    z = x[:, 2][:, None]
    parts = []
    for zi in (-1.0, 1.0):
        a = (2 * lam - 1) * (1 + zi * z) - (1 - z * z)
        da = (4 * lam - 1) * (1 + zi * z) - (1 - z * z)    # d/dlam of lam * a  = a + lam * 2 (1 + zi z)
        dr = 0.5 * da[:, :, None] * dl
        dz = 0.5 * lam * ((2 * lam - 1) * zi + 2 * z)
        parts.append(np.concatenate([dr, dz[:, :, None]], axis=2))
    for zi in (-1.0, 1.0):
        e = [(2 * (lam[:, i, None] * dl[:, j] + lam[:, j, None] * dl[:, i]) * (1 + zi * z)[:, :1],
              2 * lam[:, i] * lam[:, j] * zi) for i, j in _W15_TRI_EDGES]
        dr = np.stack([a for a, _ in e], axis=1)
        dz = np.stack([b for _, b in e], axis=1)
        parts.append(np.concatenate([dr, dz[:, :, None]], axis=2))
    dr = dl * (1 - z * z)[:, :, None][:, :1]
    dz = lam * (-2 * z)
    parts.append(np.concatenate([dr, dz[:, :, None]], axis=2))
    return np.concatenate(parts, axis=1)


# pyramid: base (+-1, +-1, 0), apex (0, 0, 1)
def _pyr5(x):
    xi, eta, z = x[:, 0], x[:, 1], x[:, 2]
    om = np.where(np.abs(1 - z) < 1e-12, 1e-12, 1 - z)
    base = []
    for bx, by in _Q4:
        base.append(0.25 * ((1 - z + bx * xi) * (1 - z + by * eta) / om))
    return np.column_stack(base + [z])


def _dpyr5(x):
    xi, eta, z = x[:, 0], x[:, 1], x[:, 2]
    om = np.where(np.abs(1 - z) < 1e-12, 1e-12, 1 - z)
    out = np.zeros((x.shape[0], 5, 3))
    for i, (bx, by) in enumerate(_Q4):
        a, b = 1 - z + bx * xi, 1 - z + by * eta
        out[:, i, 0] = 0.25 * bx * b / om
        out[:, i, 1] = 0.25 * by * a / om
        out[:, i, 2] = 0.25 * (-(a + b) / om + a * b / (om * om))
    out[:, 4, 2] = 1.0
    return out


_ELEMENTS: dict = {}


def _register(e: Element) -> None:
    _ELEMENTS[e.vtkType] = e


_register(Element(VERTEX, 0, [[0.0]], lambda x: np.ones((x.shape[0], 1)), lambda x: np.zeros((x.shape[0], 1, 1)),
                  lambda o: (np.zeros((1, 1)), np.ones(1))))
_register(Element(LINE, 1, [[-1.0], [1.0]], _line2, _dline2, lineRule))
_register(Element(QUADRATIC_EDGE, 1, [[-1.0], [1.0], [0.0]], _line3, _dline3, lineRule, corners=2, linearType=LINE))
_register(Element(TRIANGLE, 2, [[0, 0], [1, 0], [0, 1]], _tri3, _dtri3, triangleRule))
_register(Element(QUADRATIC_TRIANGLE, 2, [[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]], _tri6, _dtri6,
                  triangleRule, corners=3, linearType=TRIANGLE))
_register(Element(QUAD, 2, _Q4, _quad4, _dquad4, squareRule))
_register(Element(QUADRATIC_QUAD, 2, _Q8, _quad8, _dquad8, squareRule, corners=4, linearType=QUAD))
_register(Element(BIQUADRATIC_QUAD, 2, _Q9, _quad9, _dquad9, squareRule, corners=4, linearType=QUAD))

_TET_FACES = [[0, 1, 3], [1, 2, 3], [2, 0, 3], [0, 2, 1]]
_register(Element(TETRA, 3, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], _tet4, _dtet4, tetraRule,
                  faces=_TET_FACES, faceTypes=[TRIANGLE] * 4))
_HEX_FACES = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
_register(Element(HEXAHEDRON, 3, _H8, _hex8, _dhex8, cubeRule, faces=_HEX_FACES, faceTypes=[QUAD] * 6))
_WEDGE_FACES = [[0, 1, 2], [3, 5, 4], [0, 3, 4, 1], [1, 4, 5, 2], [2, 5, 3, 0]]
_register(Element(WEDGE, 3, [[0, 0, -1], [1, 0, -1], [0, 1, -1], [0, 0, 1], [1, 0, 1], [0, 1, 1]], _wedge6, _dwedge6,
                  wedgeRule, faces=_WEDGE_FACES, faceTypes=[TRIANGLE, TRIANGLE, QUAD, QUAD, QUAD], orientation=-1))
_PYR_FACES = [[0, 3, 2, 1], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
_register(Element(PYRAMID, 3, [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1]], _pyr5, _dpyr5,
                  pyramidRule, faces=_PYR_FACES, faceTypes=[QUAD, TRIANGLE, TRIANGLE, TRIANGLE, TRIANGLE]))


def _quadraticFaces(cornerFaces, edges):
    """Face node lists (corners then mid-side nodes) from corner faces and the edge -> node table."""
    lookup = {frozenset(e): n for n, e in edges.items()}
    out = []
    for f in cornerFaces:
        mids = [lookup[frozenset((f[i], f[(i + 1) % len(f)]))] for i in range(len(f))]
        out.append(list(f) + mids)
    return out


_TET10_EDGE_NODES = {4 + k: e for k, e in enumerate(_TET10_EDGES)}
_register(Element(QUADRATIC_TETRA, 3, np.vstack([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                                 [[0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0, 0.5],
                                                  [0, 0.5, 0.5]]]),
                  _tet10, _dtet10, tetraRule, faces=_quadraticFaces(_TET_FACES, _TET10_EDGE_NODES),
                  faceTypes=[QUADRATIC_TRIANGLE] * 4, corners=4, linearType=TETRA))
_HEX20_EDGE_NODES = {8: (0, 1), 9: (1, 2), 10: (2, 3), 11: (3, 0), 12: (4, 5), 13: (5, 6), 14: (6, 7), 15: (7, 4),
                     16: (0, 4), 17: (1, 5), 18: (2, 6), 19: (3, 7)}
_register(Element(QUADRATIC_HEXAHEDRON, 3, _H20, _hex20, _dhex20, cubeRule,
                  faces=_quadraticFaces(_HEX_FACES, _HEX20_EDGE_NODES), faceTypes=[QUADRATIC_QUAD] * 6, corners=8,
                  linearType=HEXAHEDRON))
_W15_EDGE_NODES = {6: (0, 1), 7: (1, 2), 8: (2, 0), 9: (3, 4), 10: (4, 5), 11: (5, 3), 12: (0, 3), 13: (1, 4),
                   14: (2, 5)}
_W15_REF = np.array([[0, 0, -1], [1, 0, -1], [0, 1, -1], [0, 0, 1], [1, 0, 1], [0, 1, 1],
                     [0.5, 0, -1], [0.5, 0.5, -1], [0, 0.5, -1], [0.5, 0, 1], [0.5, 0.5, 1], [0, 0.5, 1],
                     [0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
_register(Element(QUADRATIC_WEDGE, 3, _W15_REF, _wedge15, _dwedge15, wedgeRule,
                  faces=_quadraticFaces(_WEDGE_FACES, _W15_EDGE_NODES),
                  faceTypes=[QUADRATIC_TRIANGLE, QUADRATIC_TRIANGLE, QUADRATIC_QUAD, QUADRATIC_QUAD, QUADRATIC_QUAD],
                  corners=6, linearType=WEDGE, orientation=-1))


def element(vtkType: int) -> Element:
    """Element definition of a VTK cell type (polygon / polyhedron are handled by decomposition)."""
    t = RENUMBER.get(int(vtkType), (int(vtkType), None))[0]
    try:
        return _ELEMENTS[t]
    except KeyError:
        raise ValueError(f"cell type {vtkType} ({TYPE_NAMES.get(int(vtkType), 'unknown')}) is not supported "
                         "as a finite element") from None


def supportedTypes() -> list[int]:
    return sorted(_ELEMENTS) + [POLYGON, POLYHEDRON, PIXEL, VOXEL]


def cellDimension(vtkType: int) -> int:
    t = int(vtkType)
    if t == POLYGON:
        return 2
    if t == POLYHEDRON:
        return 3
    return element(t).dim
