"""Gradient-enhanced Kriging (GEK): values and partial derivatives observed jointly.

Outputs are the columns [y, dy/dx_0, ..., dy/dx_(nx-1)]; any entry may be NaN
(unobserved), so gradients can be given at some points only. For a stationary
kernel k(r2), r2 = sum_m c_m d_m^2, d = x - x', the joint covariance is

    Cov(y, y')            = s2 f
    Cov(dy/dx_k, y')      = s2 2 c_k d_k f'
    Cov(y, dy'/dx'_l)     = -s2 2 c_l d_l f'
    Cov(dy/dx_k, dy'/dx'_l) = -s2 (4 c_k c_l d_k d_l f'' + 2 c_k delta_kl f')

(f', f'' derivatives in r2; the kernel must be twice differentiable:
squaredExponential, matern52 or matern with nu > 1). The trend is profiled
by generalized least squares (derivative rows use the trend derivatives),
the process variance in closed form, and the lengthscales maximize the
(restricted) likelihood. Predictions return values and gradients with
their variances.

References:
    Morris, Mitchell & Ylvisaker (1993) Technometrics 35(3).
    Rasmussen & Williams (2006) *Gaussian Processes for Machine Learning*, sec. 9.4.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.bases.PolynomialBasis import PolynomialBasis
from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase
from pythonLibs.regressionHandler.kernels.Kernels import StationaryKernel, buildKernel
from pythonLibs.regressionHandler.numerics.LinearAlgebra import solveTriangular
from pythonLibs.regressionHandler.numerics.Optimizers import minimize, multiStart
from pythonLibs.regressionHandler.sampling.Sampling import latinHypercube

_TREND_DEGREE = {"none": -1, "constant": 0, "linear": 1, "quadratic": 2}


@registry("model").register("gek")
class GradientKrigingModel(SurrogateModelBase):
    """Gradient-enhanced Kriging; outputs [y, dy/dx_0, ...], NaN = not observed."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("corr", "squaredExponential", types=(str, dict, object),
          desc="Twice differentiable stationary kernel: squaredExponential, matern52, matern (nu > 1)")
        d("poly", "constant", values=tuple(_TREND_DEGREE), desc="Polynomial trend")
        d("nugget", 1e-10, types=(int, float), lower=0.0, desc="Value noise / process variance ratio")
        d("gradientNugget", 1e-10, types=(int, float), lower=0.0, desc="Gradient noise / process variance ratio")
        d("likelihood", "reml", values=("reml", "ml"), desc="Estimation criterion")
        d("normalize", True, types=bool, desc="Standardize inputs and values (gradients rescaled accordingly)")
        d("nStart", 5, types=int, lower=1, desc="Optimizer starts")
        d("maxIter", 200, types=int, lower=1, desc="Optimizer iterations per start")
        d("seed", 0, types=int, desc="Seed of the multi-start design")
        d("hyperparameters", None, types=list, desc="Fixed log kernel parameters")
        self.supports.update(multiOutput=True, missingOutputs=True, variances=True, weights=False)

    def _validateOptions(self) -> None:
        k = buildKernel(self.options["corr"])
        if not isinstance(k, StationaryKernel) or not hasattr(k, "fPrime2"):
            raise ValueError("GEK needs a twice differentiable stationary kernel "
                             "(squaredExponential, matern52, matern with nu > 1)")
        if k.options.get("distance", "scaled") != "scaled":
            raise ValueError("GEK supports the scaled (ARD / isotropic) distance only")

    # ------------------------------------------------------------------ setup
    def _prepare(self) -> None:
        x, y = self.xt, self.yt
        nx = self.nx
        if y.shape[1] != nx + 1:
            raise ValueError(f"outputs must be [y, dy/dx_0 .. dy/dx_{nx - 1}] ({nx + 1} columns)")
        if self.options["normalize"]:
            self._xMean, sx = x.mean(axis=0), x.std(axis=0)
            self._xStd = np.where(sx > 0, sx, 1.0)
            v = y[:, 0][np.isfinite(y[:, 0])]
            self._yMean = float(v.mean()) if v.size else 0.0
            sy = float(v.std()) if v.size > 1 else 1.0
            self._yStd = sy if sy > 0 else 1.0
        else:
            self._xMean, self._xStd, self._yMean, self._yStd = np.zeros(nx), np.ones(nx), 0.0, 1.0
        self._xs = (x - self._xMean) / self._xStd
        scale = np.concatenate([[1.0 / self._yStd], self._xStd / self._yStd])
        ys = y.copy()
        ys[:, 0] -= self._yMean
        ys *= scale[None, :]
        obs = np.isfinite(ys)
        self._pi, self._type = np.nonzero(obs)
        self._yObs = ys[self._pi, self._type]
        self._kernel = buildKernel(self.options["corr"]).setup(nx)
        deg = _TREND_DEGREE[self.options["poly"]]
        self._poly = PolynomialBasis(degree=deg).fit(self._xs) if deg >= 0 else None
        self._F = self._trendRows(self._xs[self._pi], self._type)

    def _trendRows(self, xs, types) -> np.ndarray:
        if self._poly is None:
            return np.zeros((xs.shape[0], 0))
        out = np.empty((xs.shape[0], self._poly.nTerms))
        val = types == 0
        if np.any(val):
            out[val] = self._poly.transform(xs[val])
        for k in range(self.nx):
            sel = types == k + 1
            if np.any(sel):
                out[sel] = self._poly.derivative(xs[sel], k)
        return out

    # ------------------------------------------------------------------ covariance
    def _cov(self, xa, ta, xb, tb, kp) -> np.ndarray:
        """Correlation between observations of types ta at xa and tb at xb (types: 0 value, k+1 d/dx_k)."""
        k = self._kernel
        pd, ps = k._split(kp)
        c = k.invL2(pd)
        d = xa[:, None, :] - xb[None, :, :]
        r2 = np.tensordot(d * d, c, axes=([2], [0]))
        f = k._f(r2, ps)
        f1 = k._fPrime(r2, ps)
        f2 = k.fPrime2(r2, ps)
        ka = np.maximum(ta - 1, 0)
        lb = np.maximum(tb - 1, 0)
        ia, ib = np.arange(xa.shape[0])[:, None], np.arange(xb.shape[0])[None, :]
        da = d[ia, ib, ka[:, None]]
        db = d[ia, ib, lb[None, :]]
        ca, cb = c[ka][:, None], c[lb][None, :]
        va, vb = (ta == 0)[:, None], (tb == 0)[None, :]
        same = (ka[:, None] == lb[None, :])
        out = np.where(va & vb, f, 0.0)
        out = np.where(~va & vb, 2.0 * ca * da * f1, out)
        out = np.where(va & ~vb, -2.0 * cb * db * f1, out)
        out = np.where(~va & ~vb, -(4.0 * ca * cb * da * db * f2 + 2.0 * ca * same * f1), out)
        return out

    def _nuggetDiag(self) -> np.ndarray:
        return np.where(self._type == 0, float(self.options["nugget"]), float(self.options["gradientNugget"]))

    # ------------------------------------------------------------------ likelihood
    def _profile(self, kp):
        x = self._xs[self._pi]
        c = self._cov(x, self._type, x, self._type, kp) + np.diag(self._nuggetDiag())
        n = c.shape[0]
        jitter = 0.0
        while True:
            try:
                chol = np.linalg.cholesky(c + jitter * np.eye(n))
                break
            except np.linalg.LinAlgError:
                jitter = 1e-10 if jitter == 0.0 else jitter * 10.0
                if jitter > 1e-3:
                    raise
        f, y = self._F, self._yObs
        q = f.shape[1]
        lf = solveTriangular(chol, f, lower=True) if q else np.zeros((n, 0))
        ly = solveTriangular(chol, y, lower=True)
        if q:
            qf, rf = np.linalg.qr(lf)
            beta = np.linalg.solve(rf, qf.T @ ly)
            aInv = np.linalg.inv(rf.T @ rf)
            logDetA = 2.0 * float(np.sum(np.log(np.abs(np.diag(rf)))))
        else:
            beta, aInv, logDetA = np.zeros(0), np.zeros((0, 0)), 0.0
        r = ly - lf @ beta
        ssr = float(r @ r)
        reml = self.options["likelihood"] == "reml"
        dof = n - q if reml else n
        sigma2 = max(ssr / dof, 1e-300)
        logDet = 2.0 * float(np.sum(np.log(np.diag(chol))))
        nll = 0.5 * (dof * np.log(sigma2) + logDet + (logDetA if reml else 0.0))
        return nll, {"chol": chol, "lf": lf, "beta": beta, "aInv": aInv, "sigma2": sigma2, "r": r}

    def _nll(self, kp) -> float:
        try:
            return float(self._profile(np.asarray(kp, dtype=float))[0])
        except np.linalg.LinAlgError:
            return 1e20

    def _train(self) -> None:
        self._prepare()
        fixed = self.options["hyperparameters"]
        if fixed is not None:
            p = np.asarray(fixed, dtype=float)
            self._optResult = None
        else:
            bounds = self._kernel.bounds()
            p0 = np.clip(self._kernel.initialParams(), bounds[:, 0], bounds[:, 1])
            starts = [p0]
            if self.options["nStart"] > 1:
                mid = bounds.mean(axis=1)
                half = 0.3 * (bounds[:, 1] - bounds[:, 0])
                box = np.column_stack([np.maximum(bounds[:, 0], mid - half), np.minimum(bounds[:, 1], mid + half)])
                starts.extend(latinHypercube(self.options["nStart"] - 1, box, criterion="maximin",
                                             seed=self.options["seed"]))
            self._optResult = multiStart(
                lambda s: minimize(self._nll, s, bounds=[tuple(b) for b in bounds], maxIter=self.options["maxIter"],
                                   tol=1e-7), np.array(starts))
            p = self._optResult.x
        self._params = p
        nll, self._fit = self._profile(p)
        self._logLik = -nll

    # ------------------------------------------------------------------ prediction
    def _parts(self, x, output: int, full: bool = False):
        xs = (x - self._xMean) / self._xStd
        t = np.full(x.shape[0], output)
        fit = self._fit
        cx = self._cov(xs, t, self._xs[self._pi], self._type, self._params)
        v = solveTriangular(fit["chol"], cx.T, lower=True)
        r = fit["r"]
        mean = v.T @ r
        if self._F.shape[1]:
            fx = self._trendRows(xs, t)
            mean = mean + fx @ fit["beta"]
        if full:
            prior = self._cov(xs, t, xs, t, self._params)
            cov = prior - v.T @ v
        else:
            prior = np.diag(self._cov(xs[:1], t[:1], xs[:1], t[:1], self._params))[0]
            cov = prior - np.sum(v * v, axis=0)
        if self._F.shape[1]:
            u = fit["lf"].T @ v - fx.T
            cov = cov + (u.T @ fit["aInv"] @ u if full else np.sum(u * (fit["aInv"] @ u), axis=0))
        return mean, cov * fit["sigma2"]

    def _unscale(self, output: int):
        if output == 0:
            return self._yMean, self._yStd
        return 0.0, self._yStd / self._xStd[output - 1]

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        out = np.empty((x.shape[0], self.nx + 1))
        for b in range(self.nx + 1):
            mean, _ = self._parts(x, b)
            shift, scale = self._unscale(b)
            out[:, b] = shift + scale * mean
        return out

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        out = np.empty((x.shape[0], self.nx + 1))
        nug = [float(self.options["nugget"])] + [float(self.options["gradientNugget"])] * self.nx
        for b in range(self.nx + 1):
            _, var = self._parts(x, b)
            var = np.maximum(var, 0.0)
            if kind == "prediction":
                var = var + nug[b] * self._fit["sigma2"]
            out[:, b] = var * self._unscale(b)[1] ** 2
        return out

    def _effectiveParams(self):
        return np.full(self.nx + 1, float(self._F.shape[1]))

    @property
    def hyperparameters(self) -> dict:
        self._checkTrained()
        return {"logParams": dict(zip(self._kernel.paramNames(), self._params.tolist())),
                "processVariance": self._fit["sigma2"] * self._yStd ** 2, "logLikelihood": self._logLik,
                "trendCoefficients": self._fit["beta"].tolist()}

    # ------------------------------------------------------------------ persistence
    def _stateToDict(self) -> dict:
        y = self.yt
        return {"params": self._params.tolist(), "x": self.xt.tolist(),
                "y": [[None if not np.isfinite(v) else float(v) for v in row] for row in y]}

    def _stateFromDict(self, state: dict) -> None:
        self.xt = np.array(state["x"], dtype=float)
        self.yt = np.array([[np.nan if v is None else v for v in row] for row in state["y"]], dtype=float)
        self._prepare()
        self._optResult = None
        self._params = np.array(state["params"], dtype=float)
        nll, self._fit = self._profile(self._params)
        self._logLik = -nll
