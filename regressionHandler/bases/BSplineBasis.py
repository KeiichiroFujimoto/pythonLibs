"""Uniform B-spline bases in 1D and tensor-product form (nx <= 3 by default).

Combined with a ``RidgeSolver(penalty="smoothness")`` this gives P-splines
(Eilers & Marx 1996): many equally spaced knots plus a difference penalty on
neighbouring coefficients, with the smoothing weight chosen by GCV. In 2D/3D
the same construction yields regularized tensor-product splines.

Evaluation uses de Boor's triangular scheme on the (degree + 1) nonzero
functions of each knot span, so cost is O(n * degree^2) per dimension.
Inputs outside the training range are clamped to it (constant extrapolation).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.bases.BasisBase import BasisBase
from pythonLibs.regressionHandler.core.Registry import registry


def _deBoorStep(prev: np.ndarray, u: np.ndarray, k: int) -> np.ndarray:
    """Raise nonzero values from degree k-1 to degree k (uniform knots, spacing 1).

    prev[:, j] holds N_{i-(k-1)+j, k-1}(u) for j = 0..k-1 where i is the span.
    Returns out[:, j] = N_{i-k+j, k}(u) for j = 0..k.
    """
    n = u.size
    out = np.zeros((n, prev.shape[1]))
    for j in range(k + 1):
        acc = np.zeros(n)
        if j >= 1:
            # N_{m,k} gets (x - t_m)/(k h) * N_{m,k-1} with m = i-k+j; x - t_m = u + (k - j)
            acc += (u + (k - j)) / k * prev[:, j - 1]
        if j <= k - 1:
            # and (t_{m+k+1} - x)/(k h) * N_{m+1,k-1}; t_{m+k+1} - x = (j + 1) - u
            acc += ((j + 1) - u) / k * prev[:, j]
        out[:, j] = acc
    return out


def uniformBSplineMatrix(x: np.ndarray, lo: float, hi: float, nSegments: int, degree: int,
                         derivative: bool = False) -> np.ndarray:
    """Design matrix (n, nSegments + degree) of uniform B-splines on [lo, hi]."""
    h = (hi - lo) / nSegments
    xc = np.clip(x, lo, hi)
    s = (xc - lo) / h
    span = np.minimum(np.floor(s).astype(int), nSegments - 1)
    u = s - span
    vals = np.zeros((x.size, degree + 1))
    vals[:, 0] = 1.0
    target = degree - 1 if derivative else degree
    for k in range(1, target + 1):
        vals = _deBoorStep(vals, u, k)
    if derivative:
        if degree == 0:
            return np.zeros((x.size, nSegments + degree))
        # d/dx N_{m,p} = (N_{m,p-1} - N_{m+1,p-1}) / h for uniform knots.
        lower = vals[:, :degree]
        d = np.zeros((x.size, degree + 1))
        d[:, 1:] += lower
        d[:, :-1] -= lower
        vals = d / h
        outside = (x < lo) | (x > hi)
        vals[outside] = 0.0
    out = np.zeros((x.size, nSegments + degree))
    rows = np.arange(x.size)[:, None]
    cols = span[:, None] + np.arange(degree + 1)[None, :]
    out[rows, cols] = vals
    return out


def differenceMatrix(p: int, order: int) -> np.ndarray:
    d = np.eye(p)
    for _ in range(order):
        d = d[1:] - d[:-1]
    return d


@registry("basis").register("bspline")
class BSplineBasis(BasisBase):

    def _declareOptions(self, declare) -> None:
        declare("nSegments", 20, types=(int, list), desc="Knot intervals per input (int or list per input)")
        declare("degree", 3, types=int, lower=0, upper=5, desc="Spline degree (3 = cubic)")
        declare("penaltyOrder", 2, types=int, lower=0, upper=4, desc="Difference order of the roughness penalty")
        declare("maxInputs", 3, types=int, lower=1, desc="Refuse more inputs than this (tensor size grows as p^nx)")
        declare("rangeExtension", 0.0, types=(int, float), lower=0.0,
                desc="Fraction of the data range added below the minimum and above the maximum")

    def _fit(self, x: np.ndarray) -> None:
        if self.nx > self.options["maxInputs"]:
            raise ValueError(f"tensor B-splines with {self.nx} inputs exceed maxInputs={self.options['maxInputs']}")
        seg = self.options["nSegments"]
        self._nSeg = [int(s) for s in seg] if isinstance(seg, list) else [int(seg)] * self.nx
        if len(self._nSeg) != self.nx:
            raise ValueError("nSegments list length must equal the number of inputs")
        lo, hi = x.min(axis=0), x.max(axis=0)
        ext = float(self.options["rangeExtension"]) * (hi - lo)
        self._lo = lo - ext
        hi = hi + ext
        self._hi = np.where(hi > self._lo, hi, self._lo + 1.0)

    @property
    def _sizes(self) -> list[int]:
        return [s + self.options["degree"] for s in self._nSeg]

    @property
    def nTerms(self) -> int:
        return int(np.prod(self._sizes))

    def _factor(self, x: np.ndarray, j: int, derivative: bool = False) -> np.ndarray:
        return uniformBSplineMatrix(x[:, j], self._lo[j], self._hi[j], self._nSeg[j],
                                    self.options["degree"], derivative)

    @staticmethod
    def _rowKron(mats: list[np.ndarray]) -> np.ndarray:
        out = mats[0]
        for m in mats[1:]:
            out = (out[:, :, None] * m[:, None, :]).reshape(out.shape[0], -1)
        return out

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self._rowKron([self._factor(x, j) for j in range(self.nx)])

    def derivative(self, x: np.ndarray, kx: int) -> np.ndarray:
        return self._rowKron([self._factor(x, j, derivative=(j == kx)) for j in range(self.nx)])

    def penaltyMatrix(self) -> Optional[np.ndarray]:
        order = self.options["penaltyOrder"]
        sizes = self._sizes
        total = np.zeros((self.nTerms, self.nTerms))
        for j in range(self.nx):
            d = differenceMatrix(sizes[j], min(order, sizes[j] - 1))
            block = d.T @ d
            mats = [np.eye(s) for s in sizes]
            mats[j] = block
            k = mats[0]
            for m in mats[1:]:
                k = np.kron(k, m)
            total += k
        return total

    def termNames(self, featureNames: Optional[list[str]] = None) -> list[str]:
        idx = np.indices(self._sizes).reshape(self.nx, -1).T
        return ["B(" + ",".join(str(i) for i in row) + ")" for row in idx]

    def _stateToDict(self) -> dict:
        if self.nx is None:
            return {}
        return {"nx": self.nx, "nSeg": self._nSeg, "lo": self._lo.tolist(), "hi": self._hi.tolist()}

    def _stateFromDict(self, state: dict) -> None:
        self.nx = state["nx"]
        self._nSeg = list(state["nSeg"])
        self._lo = np.array(state["lo"], dtype=float)
        self._hi = np.array(state["hi"], dtype=float)
