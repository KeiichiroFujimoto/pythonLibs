"""Shape-constrained regression: isotonic regression and monotone / convex penalized splines (1-D).

IsotonicModel
    Weighted least squares under a monotone constraint, solved exactly by
    the pool-adjacent-violators algorithm; predictions interpolate linearly
    between the fitted block values (constant beyond the data).

ShapeSplineModel
    P-spline (cubic B-splines + difference penalty) whose coefficients obey a
    shape: increasing, decreasing, convex, concave or combinations. For
    B-splines, monotone / convex coefficients give a monotone / convex curve,
    so the shape becomes linear inequalities A c >= 0 on coefficient
    differences; the penalized least-squares problem under them is solved
    exactly (least-distance programming via non-negative least squares).
    The smoothing weight is chosen by GCV of the unconstrained fit or fixed.

References:
    Barlow, Bartholomew, Bremner & Brunk (1972) *Statistical Inference under Order Restrictions*.
    Eilers & Marx (1996) Statistical Science 11(2); Bollaerts, Eilers & van Mechelen (2006) Stat. Modelling 6.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.bases.BSplineBasis import BSplineBasis, differenceMatrix
from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase
from pythonLibs.regressionHandler.numerics.Optimizers import minimizeScalar


def pava(y: np.ndarray, w: Optional[np.ndarray] = None, increasing: bool = True) -> np.ndarray:
    """Weighted isotonic regression of the sequence y (pool adjacent violators), O(n)."""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    if not increasing:
        return -pava(-y, w, True)
    vals, wts, sizes = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi)
        wts.append(wi)
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, s2 = vals.pop(), wts.pop(), sizes.pop()
            v1, w1, s1 = vals.pop(), wts.pop(), sizes.pop()
            wt = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / wt)
            wts.append(wt)
            sizes.append(s1 + s2)
    return np.repeat(vals, sizes)


def nnls(a: np.ndarray, b: np.ndarray, maxIter: Optional[int] = None) -> np.ndarray:
    """Non-negative least squares min |A x - b|, x >= 0 (Lawson-Hanson active set)."""
    m, n = a.shape
    x = np.zeros(n)
    passive = np.zeros(n, dtype=bool)
    maxIter = maxIter or 3 * n + 50
    wv = a.T @ (b - a @ x)
    tol = 10.0 * np.finfo(float).eps * np.linalg.norm(a, 1) * max(m, n)
    for _ in range(maxIter):
        if passive.all() or np.max(np.where(passive, -np.inf, wv)) <= tol:
            break
        j = int(np.argmax(np.where(passive, -np.inf, wv)))
        passive[j] = True
        while True:
            z = np.zeros(n)
            z[passive] = np.linalg.lstsq(a[:, passive], b, rcond=None)[0]
            if np.all(z[passive] > tol):
                x = z
                break
            neg = passive & (z <= tol)
            alpha = np.min(x[neg] / np.maximum(x[neg] - z[neg], 1e-300))
            x = x + alpha * (z - x)
            passive &= x > tol
            x[~passive] = 0.0
        wv = a.T @ (b - a @ x)
    return x


def constrainedLeastSquares(m: np.ndarray, r: np.ndarray, a: np.ndarray) -> np.ndarray:
    """min |M c - r|  subject to  A c >= 0, exactly (LSI -> LDP -> NNLS; Lawson & Hanson 1974, ch. 23).

    M must have full column rank.
    """
    if a.shape[0] == 0:
        return np.linalg.lstsq(m, r, rcond=None)[0]
    q, rr = np.linalg.qr(m)
    qtr = q.T @ r
    rInv = np.linalg.inv(rr)
    # c = R^-1 (u + Q^T r):  min |u|  s.t.  G u >= h,  G = A R^-1,  h = -G Q^T r
    g = a @ rInv
    h = -g @ qtr
    if np.all(h <= 0):                                   # unconstrained solution already feasible
        return rInv @ qtr
    e = np.vstack([g.T, h[None, :]])
    f = np.zeros(e.shape[0])
    f[-1] = 1.0
    v = nnls(e, f)
    rho = e @ v - f
    if abs(rho[-1]) < 1e-14:
        raise ValueError("shape constraints are infeasible")
    u = -rho[:-1] / rho[-1]
    return rInv @ (u + qtr)


@registry("model").register("isotonic")
class IsotonicModel(SurrogateModelBase):
    """1-D isotonic (monotone) regression by pool-adjacent-violators."""

    def _initialize(self) -> None:
        self.options.declare("increasing", True, values=(True, False, "auto"),
                             desc="Direction (auto: sign of the correlation)")
        self.supports.update(multiOutput=False)

    def _train(self) -> None:
        if self.nx != 1:
            raise ValueError("isotonic regression needs exactly one input")
        x, y, w = self.xt[:, 0], self.yt[:, 0], self.wt
        inc = self.options["increasing"]
        if inc == "auto":
            inc = bool(np.corrcoef(x, y)[0, 1] >= 0) if np.std(x) > 0 and np.std(y) > 0 else True
        self._increasing = bool(inc)
        order = np.lexsort((y, x))
        xs, ys, ws = x[order], y[order], w[order]
        # tied x values share one fitted value: pool them first
        ux, inv = np.unique(xs, return_inverse=True)
        wsum = np.bincount(inv, weights=ws)
        ymean = np.bincount(inv, weights=ws * ys) / wsum
        self._x = ux
        self._fit = pava(ymean, wsum, self._increasing)

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x[:, 0], self._x, self._fit)[:, None]

    def _effectiveParams(self):
        return float(np.unique(self._fit).size)

    def _stateToDict(self) -> dict:
        return {"x": self._x.tolist(), "fit": self._fit.tolist(), "increasing": self._increasing}

    def _stateFromDict(self, state: dict) -> None:
        self._x = np.array(state["x"], dtype=float)
        self._fit = np.array(state["fit"], dtype=float)
        self._increasing = bool(state["increasing"])


_SHAPES = ("increasing", "decreasing", "convex", "concave", "increasingConvex", "increasingConcave",
           "decreasingConvex", "decreasingConcave", "none")


@registry("model").register("shapeSpline")
class ShapeSplineModel(SurrogateModelBase):
    """1-D P-spline with exact monotone and / or convex shape constraints."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("shape", "increasing", values=_SHAPES, desc="Shape constraint")
        d("nSegments", 20, types=int, lower=2, desc="B-spline knot intervals")
        d("penaltyOrder", 2, types=int, lower=1, upper=3, desc="Difference order of the roughness penalty")
        d("alpha", "gcv", values=("gcv",), types=(int, float), lower=0.0, desc="Smoothing weight or 'gcv'")
        self.supports.update(multiOutput=False, derivatives=True)

    def _constraints(self, p: int) -> np.ndarray:
        """Rows A with A c >= 0 expressing the shape through coefficient differences."""
        shape = self.options["shape"]
        rows = []
        if shape.startswith("increasing"):
            rows.append(differenceMatrix(p, 1))
        if shape.startswith("decreasing"):
            rows.append(-differenceMatrix(p, 1))
        if shape.lower().endswith("convex"):
            rows.append(differenceMatrix(p, 2))
        if shape.lower().endswith("concave"):
            rows.append(-differenceMatrix(p, 2))
        return np.vstack(rows) if rows else np.zeros((0, p))

    def _train(self) -> None:
        if self.nx != 1:
            raise ValueError("shapeSpline needs exactly one input (use a GAM for several)")
        x, y, w = self.xt, self.yt[:, 0], self.wt
        self._basis = BSplineBasis(nSegments=self.options["nSegments"], degree=3).fit(x)
        phi = self._basis.transform(x)
        p = phi.shape[1]
        d = differenceMatrix(p, self.options["penaltyOrder"])
        pen = d.T @ d
        alpha = self.options["alpha"]
        if alpha == "gcv":
            xtwx, xtwy = phi.T @ (phi * w[:, None]), phi.T @ (w * y)

            def gcv(logA):
                c = np.linalg.solve(xtwx + 10.0 ** logA * pen, xtwy)
                edf = np.trace(np.linalg.solve(xtwx + 10.0 ** logA * pen, xtwx))
                rss = float(np.sum(w * (y - phi @ c) ** 2))
                return y.size * rss / max(y.size - edf, 1e-8) ** 2
            grid = np.linspace(-8, 6, 29)
            vals = [gcv(g) for g in grid]
            k = int(np.argmin(vals))
            best = minimizeScalar(gcv, (grid[max(k - 1, 0)], grid[min(k + 1, 28)]), xatol=1e-3)
            alpha = float(10.0 ** (best.x[0] if best.fun <= vals[k] else grid[k]))
        self._alpha = float(alpha)
        sw = np.sqrt(w)
        m = np.vstack([phi * sw[:, None], np.sqrt(self._alpha) * d, 1e-8 * np.eye(p)])
        r = np.concatenate([y * sw, np.zeros(m.shape[0] - y.size)])
        self._coef = constrainedLeastSquares(m, r, self._constraints(p))

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        return (self._basis.transform(x) @ self._coef)[:, None]

    def _predictDerivatives(self, x: np.ndarray, kx: int) -> np.ndarray:
        return (self._basis.derivative(x, kx) @ self._coef)[:, None]

    def _effectiveParams(self):
        return float(self._coef.size)

    def _stateToDict(self) -> dict:
        return {"basis": self._basis.toDict(), "coef": self._coef.tolist(), "alpha": self._alpha}

    def _stateFromDict(self, state: dict) -> None:
        from pythonLibs.regressionHandler.core.Registry import buildComponent
        self._basis = buildComponent("basis", state["basis"])
        self._coef = np.array(state["coef"], dtype=float)
        self._alpha = float(state["alpha"])
