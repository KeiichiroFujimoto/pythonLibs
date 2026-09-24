"""LOESS: locally weighted polynomial regression in any number of inputs.

For each prediction point the ``span`` fraction of nearest training points
is weighted by the tricube kernel of the (standardized) distance and a
local polynomial of ``degree`` 0, 1 or 2 is fitted by weighted least
squares; its value at the point is the prediction. Optional robustness
iterations reweight points with Tukey's bisquare of the residuals
(Cleveland 1979). The fit is a linear smoother y_hat = L y, so the
equivalent kernel gives standard errors and the residual variance uses
delta1 = tr((I - L)^T (I - L)) (Cleveland & Grosse 1991).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.bases.BasisBase import polynomialExponents
from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase
from pythonLibs.regressionHandler.numerics.NeighborSearch import NeighborSearch


@registry("model").register("loess")
class LocalRegressionModel(SurrogateModelBase):
    """LOESS local polynomial regression with robustness iterations and standard errors."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("span", 0.3, types=(int, float), lower=1e-6, upper=1.0, desc="Fraction of points in each local fit")
        d("degree", 1, types=int, lower=0, upper=2, desc="Local polynomial degree")
        d("iterations", 0, types=int, lower=0, upper=10, desc="Robustness (bisquare) iterations")
        self.supports.update(multiOutput=False, variances=True, derivatives=False, parameterInference=False)

    def _train(self) -> None:
        x, y, w = self.xt, self.yt[:, 0], self.wt
        n = x.shape[0]
        self._mean = x.mean(axis=0)
        sd = x.std(axis=0)
        self._std = np.where(sd > 0, sd, 1.0)
        self._xs = (x - self._mean) / self._std
        self._y = y
        self._w = w / w.mean()
        self._exps = polynomialExponents(self.nx, self.options["degree"])
        q = self._exps.shape[0]
        self._k = int(min(n, max(q + 1, np.ceil(self.options["span"] * n))))
        self._search = NeighborSearch(self._xs)
        self._robust = np.ones(n)
        for _ in range(self.options["iterations"]):
            fitted, _, _ = self._localFit(self._xs)
            r = y - fitted
            s = np.median(np.abs(r))
            if s <= 0:
                break
            u = r / (6.0 * s)
            self._robust = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
        fitted, selfWeight, lNorm2 = self._localFit(self._xs, trainingIndex=True)
        trL = float(np.sum(selfWeight))
        delta1 = float(n - 2.0 * trL + np.sum(lNorm2))
        self._edf = trL
        self._delta1 = max(delta1, 1e-8)
        resid = y - fitted
        self._sigma2 = float(np.sum(self._w * self._robust * resid * resid) / self._delta1)

    def _localFit(self, xq: np.ndarray, trainingIndex: bool = False, chunk: int = 2048):
        """Local predictions, the self weight l_ii (training only) and |l / sqrt(w)|^2 for each query."""
        k = self._k
        exps = self._exps
        q = exps.shape[0]
        dist, idx = self._search.query(xq, k)
        out = np.empty(xq.shape[0])
        selfW = np.zeros(xq.shape[0])
        norm2 = np.empty(xq.shape[0])
        for s in range(0, xq.shape[0], chunk):
            d = dist[s:s + chunk]
            ii = idx[s:s + chunk]
            h = np.maximum(d[:, -1], 1e-12) * (1.0 + 1e-10)
            tri = np.clip(1.0 - (d / h[:, None]) ** 3, 0.0, None) ** 3
            wt = tri * self._w[ii] * self._robust[ii]
            local = self._xs[ii] - xq[s:s + chunk][:, None, :]              # (m, k, nx)
            design = np.ones(local.shape[:2] + (q,))
            for t, e in enumerate(exps):
                for j in np.flatnonzero(e):
                    design[:, :, t] *= local[:, :, j] ** e[j]
            xtw = np.transpose(design, (0, 2, 1)) * wt[:, None, :]          # (m, q, k)
            gram = xtw @ design
            ridge = 1e-10 * np.trace(gram, axis1=1, axis2=2)[:, None, None] * np.eye(q)[None]
            # Equivalent kernel l = e0^T (X^T W X)^-1 X^T W (the intercept row).
            e0 = np.zeros((xq[s:s + chunk].shape[0], q, 1))
            e0[:, 0, 0] = 1.0
            coefRow = np.linalg.solve(gram + ridge, e0)[:, :, 0]            # (m, q)
            ell = np.einsum("mq,mqk->mk", coefRow, xtw)                      # (m, k)
            out[s:s + chunk] = np.einsum("mk,mk->m", ell, self._y[ii])
            norm2[s:s + chunk] = np.sum(ell * ell / self._w[ii], axis=1)
            if trainingIndex:
                rows = np.arange(s, s + ell.shape[0])
                hit = ii == rows[:, None]
                selfW[s:s + chunk] = np.sum(np.where(hit, ell, 0.0), axis=1)
        return out, selfW, norm2

    def _scaled(self, x):
        return (x - self._mean) / self._std

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        return self._localFit(self._scaled(x))[0][:, None]

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        _, _, norm2 = self._localFit(self._scaled(x))
        var = self._sigma2 * norm2
        if kind == "prediction":
            var = var + self._sigma2
        return var[:, None]

    def _effectiveParams(self):
        return self._edf

    def _intervalDof(self) -> Optional[float]:
        return self._delta1 if self._delta1 > 1 else None

    def _stateToDict(self) -> dict:
        return {"mean": self._mean.tolist(), "std": self._std.tolist(), "xs": self._xs.tolist(),
                "y": self._y.tolist(), "w": self._w.tolist(), "robust": self._robust.tolist(), "k": self._k,
                "edf": self._edf, "delta1": self._delta1, "sigma2": self._sigma2}

    def _stateFromDict(self, state: dict) -> None:
        self._mean, self._std = np.array(state["mean"]), np.array(state["std"])
        self._xs = np.array(state["xs"], dtype=float).reshape(-1, self.nx)
        self._y, self._w = np.array(state["y"]), np.array(state["w"])
        self._robust = np.array(state["robust"])
        self._k, self._edf = int(state["k"]), float(state["edf"])
        self._delta1, self._sigma2 = float(state["delta1"]), float(state["sigma2"])
        self._exps = polynomialExponents(self.nx, self.options["degree"])
        self._search = NeighborSearch(self._xs)
