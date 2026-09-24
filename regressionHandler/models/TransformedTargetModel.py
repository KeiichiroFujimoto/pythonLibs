"""Any model on a transformed response (Box-Cox, Yeo-Johnson, log), with back-transformation.

    z = T_lambda(y),   z = inner model(x) + error

``lambda="auto"`` maximizes the profile log-likelihood

    l(lambda) = -n/2 log(RSS_lambda / n) + log |J_lambda|,   J = prod dT/dy

(Box & Cox 1964) with the inner model refitted for every trial lambda (Brent).
Predictions are back-transformed as the median T^-1(zhat) or, with
``retransform="mean"``, the smearing estimate mean_j T^-1(zhat + e_j)
(Duan 1983). Intervals map the inner-model interval through T^-1 (monotone),
so they are asymmetric on the original scale.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import PredictionInterval, SurrogateModelBase
from pythonLibs.regressionHandler.numerics.Optimizers import minimizeScalar


# ---------------------------------------------------------------- transforms
def boxCox(y, lam):
    y = np.asarray(y, dtype=float)
    return np.log(y) if abs(lam) < 1e-12 else (y ** lam - 1.0) / lam


def boxCoxInverse(z, lam):
    z = np.asarray(z, dtype=float)
    if abs(lam) < 1e-12:
        return np.exp(z)
    return np.maximum(lam * z + 1.0, 1e-300) ** (1.0 / lam)


def yeoJohnson(y, lam):
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    pos = y >= 0
    out[pos] = np.log1p(y[pos]) if abs(lam) < 1e-12 else ((y[pos] + 1.0) ** lam - 1.0) / lam
    l2 = 2.0 - lam
    out[~pos] = -np.log1p(-y[~pos]) if abs(l2) < 1e-12 else -((1.0 - y[~pos]) ** l2 - 1.0) / l2
    return out


def yeoJohnsonInverse(z, lam):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = np.expm1(z[pos]) if abs(lam) < 1e-12 else np.maximum(lam * z[pos] + 1.0, 1e-300) ** (1.0 / lam) - 1.0
    l2 = 2.0 - lam
    out[~pos] = -np.expm1(-z[~pos]) if abs(l2) < 1e-12 else 1.0 - np.maximum(1.0 - l2 * z[~pos], 1e-300) ** (1.0 / l2)
    return out


def _logJacobian(y, transform: str, lam: float) -> float:
    if transform == "boxcox":
        return float((lam - 1.0) * np.sum(np.log(y)))
    if transform == "yeoJohnson":
        return float((lam - 1.0) * np.sum(np.sign(y) * np.log1p(np.abs(y))))
    if transform == "log":
        return float(-np.sum(np.log(y)))
    return 0.0


@registry("model").register("transformedTarget")
class TransformedTargetModel(SurrogateModelBase):
    """Wrap any model: fit on T(y) (Box-Cox / Yeo-Johnson / log), predict on the original scale."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("model", "linear", types=(str, dict, object), desc="Inner model spec (shorthand, dict or instance)")
        d("transform", "boxcox", values=("boxcox", "yeoJohnson", "log", "none"), desc="Response transform")
        d("lambda", "auto", values=("auto",), types=(int, float), desc="Transform parameter or 'auto' (profile ML)")
        d("lambdaBounds", [-2.0, 3.0], types=list, desc="Search interval for lambda='auto'")
        d("retransform", "median", values=("median", "mean"), desc="Back-transformation of predictions")
        self.supports.update(variances=True, derivatives=True, weights=False)

    def _inner(self):
        from pythonLibs.regressionHandler.models.ModelFactory import createModel
        spec = self.options["model"]
        return createModel(copy.deepcopy(spec) if isinstance(spec, (dict, str)) else spec)

    def _fwd(self, y, lam):
        t = self.options["transform"]
        if t == "boxcox":
            return boxCox(y, lam)
        if t == "yeoJohnson":
            return yeoJohnson(y, lam)
        if t == "log":
            return np.log(y)
        return np.asarray(y, dtype=float)

    def _inv(self, z, lam):
        t = self.options["transform"]
        if t == "boxcox":
            return boxCoxInverse(z, lam)
        if t == "yeoJohnson":
            return yeoJohnsonInverse(z, lam)
        if t == "log":
            return np.exp(z)
        return np.asarray(z, dtype=float)

    def _dInv(self, z, lam):
        h = 1e-6 * np.maximum(np.abs(z), 1.0)
        return (self._inv(z + h, lam) - self._inv(z - h, lam)) / (2.0 * h)

    def _profile(self, lam: float) -> float:
        x, y = self.xt, self.yt[:, 0]
        z = self._fwd(y, lam)
        m = self._inner().fit(x, z)
        rss = float(np.sum((z - m.predictValues(x)[:, 0]) ** 2))
        n = y.size
        return -0.5 * n * np.log(max(rss / n, 1e-300)) + _logJacobian(y, self.options["transform"], lam)

    def _train(self) -> None:
        y = self.yt[:, 0]
        t = self.options["transform"]
        if t in ("boxcox", "log") and np.any(y <= 0):
            raise ValueError(f"{t} needs a positive response (use transform='yeoJohnson')")
        lam = self.options["lambda"]
        if t not in ("boxcox", "yeoJohnson"):
            lam = 0.0
        elif lam == "auto":
            lo, hi = map(float, self.options["lambdaBounds"])
            grid = np.linspace(lo, hi, 11)
            vals = np.array([-self._profile(g) for g in grid])
            k = int(np.argmin(vals))
            best = minimizeScalar(lambda v: -self._profile(v), (grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]),
                                  xatol=1e-7)
            lam = float(best.x[0]) if best.fun <= vals[k] else float(grid[k])
        self._lambda = float(lam)
        z = self._fwd(y, self._lambda)
        self._model = self._inner().fit(self.xt, z)
        self._resid = z - self._model.predictValues(self.xt)[:, 0]
        self._profileLogLik = self._profile(self._lambda) if t != "none" else float("nan")
        self.supports["variances"] = bool(self._model.supports.get("variances"))

    # ------------------------------------------------------------------ prediction
    def _backMean(self, z):
        if self.options["retransform"] == "median":
            return self._inv(z, self._lambda)
        out = np.empty_like(z)
        for s in range(0, z.size, 2048):
            zz = z[s:s + 2048, None] + self._resid[None, :]
            out[s:s + 2048] = np.mean(self._inv(zz.ravel(), self._lambda).reshape(zz.shape), axis=1)
        return out

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        return self._backMean(self._model.predictValues(x)[:, 0])[:, None]

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        z = self._model.predictValues(x)[:, 0]
        vz = self._model.predictVariances(x, kind)[:, 0]
        return (self._dInv(z, self._lambda) ** 2 * vz)[:, None]

    def predictInterval(self, x, level: float = 0.95, kind: str = "prediction") -> PredictionInterval:
        """Inner-model interval mapped through T^-1 (asymmetric on the original scale)."""
        self._checkTrained()
        inner = self._model.predictInterval(self._validX(x), level, kind)
        lo = self._inv(inner.lower[:, 0], self._lambda)
        hi = self._inv(inner.upper[:, 0], self._lambda)
        mean = self._predictValues(self._validX(x))[:, 0]
        std = np.sqrt(np.maximum(self._predictVariances(self._validX(x), kind)[:, 0], 0.0))
        return PredictionInterval(mean[:, None], np.minimum(lo, hi)[:, None], np.maximum(lo, hi)[:, None],
                                  std[:, None], float(level), kind)

    def _predictDerivatives(self, x: np.ndarray, kx: int) -> np.ndarray:
        z = self._model.predictValues(x)[:, 0]
        return (self._dInv(z, self._lambda) * self._model.predictDerivatives(x, kx)[:, 0])[:, None]

    def _effectiveParams(self):
        return float(np.atleast_1d(self._model.nEffectiveParams)[0]) + (1.0 if self.options["lambda"] == "auto" else 0.0)

    @property
    def transformParameter(self) -> float:
        self._checkTrained()
        return self._lambda

    @property
    def innerModel(self):
        return self._model

    def _stateToDict(self) -> dict:
        return {"lambda": self._lambda, "model": self._model.toDict(), "resid": self._resid.tolist(),
                "profileLogLik": self._profileLogLik}

    def _stateFromDict(self, state: dict) -> None:
        self._lambda = float(state["lambda"])
        self._model = SurrogateModelBase.fromDict(state["model"])
        self._resid = np.array(state["resid"], dtype=float)
        self._profileLogLik = float(state["profileLogLik"])
