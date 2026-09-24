"""Inverse distance weighting (Shepard interpolation).

    s(x) = sum_i w_i(x) y_i / sum_i w_i(x),   w_i = sampleWeight_i / |x - x_i|^p

Exact at the data points, cheap, and free of hyperparameter fitting; useful
as a robust baseline and for very large or scattered datasets. Inputs are
standardized; ``neighbors`` restricts the sum to the k nearest points.
"""
from __future__ import annotations

import numpy as np

from pythonLibs.regressionHandler.bases.RadialBasis import pairwiseDistances
from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase
from pythonLibs.regressionHandler.numerics.NeighborSearch import NeighborSearch


@registry("model").register("idw")
class IdwModel(SurrogateModelBase):
    """Inverse distance weighting interpolation."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("p", 2.5, types=(int, float), lower=0.0, desc="Distance exponent")
        d("neighbors", None, types=int, lower=1, desc="Use only the k nearest points")
        self.supports.update(multiOutput=True, variances=False, derivatives=False, parameterInference=False)

    def _train(self) -> None:
        self._mean = self.xt.mean(axis=0)
        sd = self.xt.std(axis=0)
        self._std = np.where(sd > 0, sd, 1.0)
        self._xs = (self.xt - self._mean) / self._std
        self._y = self.yt.copy()
        self._w = self.wt / self.wt.mean()
        self._search = None

    def _predictValues(self, x: np.ndarray, chunk: int = 2048) -> np.ndarray:
        xs = (x - self._mean) / self._std
        k = self.options["neighbors"]
        p = self.options["p"]
        if k is not None and k < self._xs.shape[0]:
            if getattr(self, "_search", None) is None:
                self._search = NeighborSearch(self._xs)
            d, idx = self._search.query(xs, k)
            return self._combine(d, self._w[idx], self._y[idx], p)
        out = np.empty((xs.shape[0], self._y.shape[1]))
        for start in range(0, xs.shape[0], chunk):
            d = pairwiseDistances(xs[start:start + chunk], self._xs)
            out[start:start + chunk] = self._combine(d, self._w[None, :], None, p)
        return out

    def _combine(self, d, sw, yLocal, p):
        exact = d <= 1e-14
        with np.errstate(divide="ignore"):
            wts = sw / np.where(exact, 1.0, d) ** p
        hit = exact.any(axis=1)
        wts[hit] = np.where(exact[hit], 1.0, 0.0)
        wts = wts / wts.sum(axis=1, keepdims=True)
        return wts @ self._y if yLocal is None else np.einsum("mk,mkj->mj", wts, yLocal)

    def _effectiveParams(self):
        return float(self.xt.shape[0]) if self.xt is not None else float(self.metrics[0].nSamples)

    def _stateToDict(self) -> dict:
        return {"mean": self._mean.tolist(), "std": self._std.tolist(), "xs": self._xs.tolist(),
                "y": self._y.tolist(), "w": self._w.tolist()}

    def _stateFromDict(self, state: dict) -> None:
        self._mean, self._std = np.array(state["mean"]), np.array(state["std"])
        self._xs = np.array(state["xs"], dtype=float).reshape(-1, self.nx)
        self._y = np.array(state["y"], dtype=float).reshape(self._xs.shape[0], -1)
        self._w = np.array(state["w"], dtype=float)
        self._search = None
