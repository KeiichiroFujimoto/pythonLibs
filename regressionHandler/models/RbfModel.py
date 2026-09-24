"""Radial basis function interpolation / smoothing.

    s(x) = sum_i c_i phi(|x - x_i|) + p(x),   (K + smoothing W^-1) c + P d = y,  P^T c = 0

- options ``kernel``, ``epsilon``, ``smoothing`` and ``degree``
  (``degree=None`` picks the minimum polynomial degree that makes the
  kernel conditionally positive definite)
- ``smoothing="loo"`` picks the smoothing weight minimizing the exact
  leave-one-out error (Rippa 1999: e_i = c_i / (A^-1)_ii, no refits)
- ``neighbors=k`` solves a small local system per prediction point, which
  scales to large datasets
- inputs are standardized, so one epsilon suits all dimensions

References:
    Fasshauer (2007) *Meshfree Approximation Methods with MATLAB*, ch. 6-8.
    Rippa (1999) "An algorithm for selecting a good value for the parameter c
    in radial basis function interpolation", Adv. Comput. Math. 11.
"""
from __future__ import annotations

import numpy as np

from pythonLibs.regressionHandler.bases.BasisBase import polynomialExponents
from pythonLibs.regressionHandler.bases.PolynomialBasis import PolynomialBasis
from pythonLibs.regressionHandler.bases.RadialBasis import (RBF_KERNELS, pairwiseDistances, rbfDerivativeOverR,
                                                            rbfValue)
from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase
from pythonLibs.regressionHandler.numerics.NeighborSearch import NeighborSearch
from pythonLibs.regressionHandler.numerics.Optimizers import minimizeScalar

_MIN_DEGREE = {"gaussian": -1, "inverseMultiquadric": -1, "inverseQuadratic": -1, "multiquadric": 0,
               "linear": 0, "thinPlateSpline": 1, "cubic": 1, "quintic": 2}


@registry("model").register("rbf")
class RbfModel(SurrogateModelBase):
    """Radial basis function interpolation / smoothing with polynomial tail."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("kernel", "thinPlateSpline", values=RBF_KERNELS, desc="Radial function")
        d("epsilon", None, types=(int, float), lower=0.0,
          desc="Shape parameter for gaussian / (inverse) multiquadric kernels (None: 1 / mean spacing)")
        d("smoothing", 0.0, values=("loo", "gcv"), types=(int, float),
          desc="0 interpolates; > 0 smooths; 'loo' / 'gcv' choose it by leave-one-out error / GCV")
        d("normalize", "std", values=("std", "range", "none"),
          desc="Input scaling: standardize, map to [0, 1] or none")
        d("degree", None, types=int, lower=-1, upper=3, desc="Polynomial tail degree (None: kernel minimum)")
        d("neighbors", None, types=int, lower=2, desc="Use only this many nearest points per prediction")
        d("smoothingRange", [-12.0, 2.0], types=list, desc="log10 search interval for smoothing='loo' / 'gcv'")
        self.supports.update(multiOutput=True, variances=False, derivatives=True, parameterInference=False)

    # ------------------------------------------------------------------ training
    def _train(self) -> None:
        x, y, w = self.xt, self.yt, self.wt
        mode = self.options["normalize"]
        if mode == "std":
            self._mean = x.mean(axis=0)
            sd = x.std(axis=0)
        elif mode == "range":
            self._mean = x.min(axis=0)
            sd = np.ptp(x, axis=0)
        else:
            self._mean, sd = np.zeros(x.shape[1]), np.ones(x.shape[1])
        self._std = np.where(sd > 0, sd, 1.0)
        self._xs = (x - self._mean) / self._std
        self._y = y
        self._w = w / w.mean()
        kernel = self.options["kernel"]
        deg = self.options["degree"]
        deg = _MIN_DEGREE[kernel] if deg is None else deg
        if deg < _MIN_DEGREE[kernel]:
            raise ValueError(f"kernel {kernel} needs degree >= {_MIN_DEGREE[kernel]}")
        self._deg = deg
        self._poly = PolynomialBasis(degree=deg).fit(self._xs) if deg >= 0 else None
        n = x.shape[0]
        q = polynomialExponents(self.nx, deg).shape[0] if deg >= 0 else 0
        if q > n:
            raise ValueError(f"degree {deg} polynomial needs at least {q} points")
        self._eps = self._chooseEpsilon()
        self._search = None
        self._neighbors = self.options["neighbors"]
        if self._neighbors is not None and self._neighbors >= n:
            self._neighbors = None
        self.supports["derivatives"] = self._neighbors is None
        s = self.options["smoothing"]
        if self._neighbors is None:
            self._setTrainingSystem(np.arange(n))
            self._smoothing = self._chooseSmoothing(s) if isinstance(s, str) else float(s)
        else:
            # Local mode never forms the n x n system; 'loo' is tuned on a subsample.
            if isinstance(s, str):
                sub = np.sort(np.random.default_rng(0).choice(n, min(n, 1500), replace=False))
                self._setTrainingSystem(sub)
                self._smoothing = self._chooseSmoothing(s)
            else:
                self._smoothing = float(s)
            self._kTrain = self._pTrain = self._sysY = self._sysW = None
        if self._smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        if self._neighbors is None:
            self._solveGlobal(self._smoothing)
        else:
            self._edf = float(n)

    def _setTrainingSystem(self, idx: np.ndarray) -> None:
        xs = self._xs[idx]
        self._kTrain = rbfValue(self.options["kernel"], pairwiseDistances(xs, xs), self._eps)
        self._pTrain = self._poly.transform(xs) if self._poly is not None else np.zeros((idx.size, 0))
        self._sysY = self._y[idx]
        self._sysW = self._w[idx]

    def _chooseEpsilon(self) -> float:
        if self.options["epsilon"] is not None:
            return float(self.options["epsilon"])
        xs = self._xs
        sample = xs if xs.shape[0] <= 2000 else xs[np.random.default_rng(0).choice(xs.shape[0], 2000, False)]
        d = pairwiseDistances(sample, sample)
        np.fill_diagonal(d, np.inf)
        spacing = float(np.mean(np.min(d, axis=1)))
        return 1.0 / spacing if spacing > 0 else 1.0

    def _system(self, smoothing: float) -> np.ndarray:
        n, q = self._pTrain.shape
        a = np.zeros((n + q, n + q))
        a[:n, :n] = self._kTrain + np.diag(smoothing / self._sysW)
        a[:n, n:] = self._pTrain
        a[n:, :n] = self._pTrain.T
        return a

    def _solveGlobal(self, smoothing: float) -> None:
        n, q = self._pTrain.shape
        a = self._system(smoothing)
        rhs = np.vstack([self._sysY, np.zeros((q, self._sysY.shape[1]))])
        sol = np.linalg.solve(a, rhs)
        self._coef, self._polyCoef = sol[:n], sol[n:]
        if smoothing > 0:
            aInvDiag = np.diag(np.linalg.inv(a))[:n]
            self._edf = float(n - smoothing * np.sum(aInvDiag / self._sysW))
        else:
            self._edf = float(n)

    def _chooseSmoothing(self, rule: str = "loo") -> float:
        return self._chooseSmoothingGcv() if rule == "gcv" else self._chooseSmoothingLoo()

    def _chooseSmoothingLoo(self) -> float:
        n, q = self._pTrain.shape
        rhs = np.vstack([self._sysY, np.zeros((q, self._sysY.shape[1]))])

        def looScore(logS: float) -> float:
            try:
                aInv = np.linalg.inv(self._system(10.0 ** logS))
            except np.linalg.LinAlgError:
                return np.inf
            c = (aInv @ rhs)[:n]
            e = c / np.diag(aInv)[:n, None]
            return float(np.mean(e * e))

        lo, hi = map(float, self.options["smoothingRange"])
        grid = np.linspace(lo, hi, 29)
        vals = np.array([looScore(g) for g in grid])
        k = int(np.argmin(vals))
        left, right = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
        best = minimizeScalar(looScore, (left, right), xatol=1e-3)
        self._looRmse = float(np.sqrt(min(best.fun, vals[k])))
        return float(10.0 ** (best.x[0] if best.fun <= vals[k] else grid[k]))

    def _chooseSmoothingGcv(self) -> float:
        """GCV smoothing via a generalized eigen-decomposition in the null space of P^T.

        With Q2 spanning {c : P^T c = 0}, M = Q2^T K Q2, N = Q2^T W^-1 Q2 and
        M V = N V S (V^T N V = I), the residual is lambda W^-1 Q2 V (S + lambda)^-1 V^T Q2^T y
        and tr(I - A) = lambda sum 1 / (s_k + lambda): every trial lambda costs O(n^2).
        """
        n, q = self._pTrain.shape
        if q:
            qFull, _ = np.linalg.qr(self._pTrain, mode="complete")
            q2 = qFull[:, q:]
        else:
            q2 = np.eye(n)
        winv = 1.0 / self._sysW
        m = q2.T @ self._kTrain @ q2
        nMat = (q2 * winv[:, None]).T @ q2
        lower = np.linalg.cholesky(nMat)
        li = np.linalg.inv(lower)
        s, u = np.linalg.eigh(li @ m @ li.T)
        v = li.T @ u
        proj = v.T @ q2.T @ self._sysY                      # (n - q, ny)
        back = (q2 * winv[:, None]) @ v                     # W^-1 Q2 V
        wts = self._sysW

        def gcv(logL: float) -> float:
            lam = 10.0 ** logL
            resid = lam * back @ (proj / (s + lam)[:, None])
            trIminusA = lam * float(np.sum(1.0 / (s + lam)))
            rss = float(np.sum(wts[:, None] * resid * resid)) / (n * resid.shape[1])
            return rss / max(trIminusA / n, 1e-12) ** 2

        lo, hi = map(float, self.options["smoothingRange"])
        grid = np.linspace(lo, hi, 57)
        vals = np.array([gcv(g) for g in grid])
        k = int(np.argmin(vals))
        left, right = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
        best = minimizeScalar(gcv, (left, right), xatol=1e-4)
        self._gcv = float(min(best.fun, vals[k]))
        return float(10.0 ** (best.x[0] if best.fun <= vals[k] else grid[k]))

    # ------------------------------------------------------------------ prediction
    def _scaled(self, x):
        return (x - self._mean) / self._std

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        xs = self._scaled(x)
        if self._neighbors is not None:
            return self._predictLocal(xs)
        out = rbfValue(self.options["kernel"], pairwiseDistances(xs, self._xs), self._eps) @ self._coef
        if self._poly is not None:
            out = out + self._poly.transform(xs) @ self._polyCoef
        return out

    def _predictLocal(self, xs: np.ndarray, chunk: int = 1024) -> np.ndarray:
        k = self._neighbors
        kernel = self.options["kernel"]
        if getattr(self, "_search", None) is None:
            self._search = NeighborSearch(self._xs)
        dist, allIdx = self._search.query(xs, k)
        out = np.empty((xs.shape[0], self._y.shape[1]))
        q = self._poly.nTerms if self._poly is not None else 0
        for start in range(0, xs.shape[0], chunk):
            xq = xs[start:start + chunk]
            idx = allIdx[start:start + chunk]
            pts = self._xs[idx]                                      # (m, k, nx)
            diff = pts[:, :, None, :] - pts[:, None, :, :]
            kk = rbfValue(kernel, np.sqrt(np.sum(diff * diff, axis=-1)), self._eps)
            kk = kk + np.eye(k)[None] * (self._smoothing / self._w[idx])[:, :, None]
            m = xq.shape[0]
            a = np.zeros((m, k + q, k + q))
            a[:, :k, :k] = kk
            if q:
                pl = self._poly.transform(pts.reshape(-1, self.nx)).reshape(m, k, q)
                a[:, :k, k:] = pl
                a[:, k:, :k] = np.transpose(pl, (0, 2, 1))
            rhs = np.zeros((m, k + q, self._y.shape[1]))
            rhs[:, :k] = self._y[idx]
            sol = np.linalg.solve(a, rhs)
            phiQ = rbfValue(kernel, dist[start:start + chunk], self._eps)
            val = np.einsum("mk,mkj->mj", phiQ, sol[:, :k])
            if q:
                val = val + np.einsum("mq,mqj->mj", self._poly.transform(xq), sol[:, k:])
            out[start:start + chunk] = val
        return out

    def _predictDerivatives(self, x: np.ndarray, kx: int) -> np.ndarray:
        xs = self._scaled(x)
        r = pairwiseDistances(xs, self._xs)
        g = rbfDerivativeOverR(self.options["kernel"], r, self._eps) * (xs[:, [kx]] - self._xs[:, kx][None, :])
        out = g @ self._coef
        if self._poly is not None:
            out = out + self._poly.derivative(xs, kx) @ self._polyCoef
        return out / self._std[kx]

    def _effectiveParams(self):
        return self._edf

    @property
    def smoothingValue(self) -> float:
        return self._smoothing

    # ------------------------------------------------------------------ persistence
    def _stateToDict(self) -> dict:
        return {"mean": self._mean.tolist(), "std": self._std.tolist(), "xs": self._xs.tolist(),
                "y": self._y.tolist(), "w": self._w.tolist(), "eps": self._eps, "smoothing": self._smoothing,
                "deg": self._deg}

    def _stateFromDict(self, state: dict) -> None:
        self._mean, self._std = np.array(state["mean"]), np.array(state["std"])
        self._xs = np.array(state["xs"], dtype=float).reshape(-1, self.nx)
        self._y = np.array(state["y"], dtype=float).reshape(self._xs.shape[0], -1)
        self._w = np.array(state["w"], dtype=float)
        self._eps, self._smoothing, self._deg = float(state["eps"]), float(state["smoothing"]), int(state["deg"])
        self._poly = PolynomialBasis(degree=self._deg).fit(self._xs) if self._deg >= 0 else None
        n = self._xs.shape[0]
        self._search = None
        self._neighbors = self.options["neighbors"] if self.options["neighbors"] and self.options["neighbors"] < n \
            else None
        self.supports["derivatives"] = self._neighbors is None
        if self._neighbors is None:
            self._setTrainingSystem(np.arange(n))
        if self._neighbors is None:
            self._solveGlobal(self._smoothing)
        else:
            self._edf = float(n)
