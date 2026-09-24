"""Universal Kriging / Gaussian-process regression (with KPLS dimension reduction).

Model (on standardized inputs and outputs)::

    y(x) = f(x)^T beta + Z(x) + eps,   Cov[Z(x), Z(x')] = sigma2 * k(x, x'; theta),
    Var[eps_i] = sigma2 * nugget / w_i

- trend f: constant / linear / quadratic polynomial (``poly``)
- k: any kernel from ``kernels`` (ARD by default, composable with + and *)
- beta and sigma2 are profiled out in closed form (generalized least squares);
  theta and the nugget maximize the REML (default) or ML profile likelihood
  with its analytic gradient, by multi-start projected L-BFGS
- ``nugget="auto"`` estimates observation noise (regression Kriging);
  a small fixed nugget gives an interpolating model
- ``plsComponents`` switches to KPLS: the kernel lengthscales are driven by
  a few PLS directions, so the number of hyperparameters stays small for
  many inputs

Predictions return the universal-Kriging variance (including trend
uncertainty) and analytic gradients.

References:
    Sacks, Welch, Mitchell & Wynn (1989) Statistical Science 4(4).
    Rasmussen & Williams (2006) *Gaussian Processes for Machine Learning*, ch. 2, 5.
    Bouhlel et al. (2016) "Improving Kriging surrogates of high-dimensional
    design models by Partial Least Squares dimension reduction", SMO 53.
"""
from __future__ import annotations

import numpy as np

from pythonLibs.regressionHandler.bases.PolynomialBasis import PolynomialBasis
from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase
from pythonLibs.regressionHandler.kernels.Kernels import buildKernel
from pythonLibs.regressionHandler.numerics.LinearAlgebra import solveTriangular
from pythonLibs.regressionHandler.numerics.Optimizers import minimize, multiStart
from pythonLibs.regressionHandler.numerics.PLS import plsRotations
from pythonLibs.regressionHandler.sampling.Sampling import latinHypercube

_TREND_DEGREE = {"none": -1, "constant": 0, "linear": 1, "quadratic": 2}


@registry("model").register("kriging")
class KrigingModel(SurrogateModelBase):
    """Universal Kriging / GP regression with estimated noise (KPLS with plsComponents)."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("corr", "squaredExponential", types=(str, dict, object),
          desc="Kernel spec: squaredExponential, matern52, matern32, absoluteExponential, powerExponential, "
               "rationalQuadratic, periodic, or sum/product dicts")
        d("poly", "constant", values=tuple(_TREND_DEGREE), desc="Regression trend")
        d("nugget", "auto", values=("auto",), types=(int, float),
          desc="Noise-to-signal variance ratio; 'auto' estimates it (regression), a tiny value interpolates")
        d("nuggetBounds", [1e-10, 10.0], types=list, desc="Bounds of the estimated nugget")
        d("likelihood", "reml", values=("reml", "ml"), desc="Restricted or plain maximum likelihood")
        d("nStart", 5, types=int, lower=1,
          desc="Optimizer starts (initial guess + LHS points; a tied-lengthscale start is added for ARD kernels)")
        d("maxIter", 200, types=int, lower=1, desc="L-BFGS iterations per start")
        d("seed", 0, types=int, desc="Seed of the multi-start design")
        d("plsComponents", None, types=int, lower=1, desc="Use KPLS with this many PLS components")
        d("hyperparameters", None, types=list,
          desc="Fixed log-hyperparameters (kernel params + log nugget if auto); skips optimization")
        self.supports.update(multiOutput=False, variances=True, derivatives=True, parameterInference=False)

    def _validateOptions(self) -> None:
        buildKernel(self.options["corr"])

    # ------------------------------------------------------------------ training
    def _train(self) -> None:
        x, y, w = self.xt, self.yt[:, 0], self.wt
        n = x.shape[0]
        self._xMean = x.mean(axis=0)
        sx = x.std(axis=0)
        self._xStd = np.where(sx > 0, sx, 1.0)
        self._yMean = float(y.mean())
        sy = float(y.std())
        self._yStd = sy if sy > 0 else 1.0
        self._xs = (x - self._xMean) / self._xStd
        self._ys = (y - self._yMean) / self._yStd
        self._wn = w / w.mean()
        self._pls = None
        if self.options["plsComponents"]:
            self._pls = plsRotations(self._xs, self._ys, self.options["plsComponents"])
        self._kernel = buildKernel(self.options["corr"]).setup(self.nx, self._pls)
        deg = _TREND_DEGREE[self.options["poly"]]
        self._trend = PolynomialBasis(degree=deg).fit(self._xs) if deg >= 0 else None
        self._f = self._trend.transform(self._xs) if self._trend is not None else np.zeros((n, 0))
        if self._f.shape[1] >= n:
            raise ValueError(f"trend '{self.options['poly']}' needs more than {self._f.shape[1]} points")

        self._cache = self._kernel.trainingCache(self._xs)
        nk = self._kernel.nParams()
        fixed = self.options["hyperparameters"]
        if fixed is not None:
            p = np.asarray(fixed, dtype=float)
            if p.size != nk + self._autoNugget:
                raise ValueError(f"expected {nk + self._autoNugget} hyperparameters, got {p.size}")
            self._optResult = None
        else:
            bounds = self._bounds()
            p0 = np.concatenate([self._kernel.initialParams(), [np.log(1e-2)] if self._autoNugget else []])
            p0 = np.clip(p0, bounds[:, 0], bounds[:, 1])
            starts = [p0]
            if nk > 1:
                tied = self._tiedStart(p0, bounds)
                if tied is not None:
                    starts.append(tied)
            if self.options["nStart"] > 1:
                # Restrict random starts to the central part of the box (in log space).
                mid = bounds.mean(axis=1)
                half = 0.35 * (bounds[:, 1] - bounds[:, 0])
                box = np.column_stack([np.maximum(bounds[:, 0], mid - half), np.minimum(bounds[:, 1], mid + half)])
                box[box[:, 0] >= box[:, 1], 1] = box[box[:, 0] >= box[:, 1], 0] + 1e-9
                starts.extend(latinHypercube(self.options["nStart"] - 1, box, criterion="maximin",
                                             seed=self.options["seed"]))
            boundPairs = [tuple(b) for b in bounds]
            self._optResult = multiStart(
                lambda s: minimize(self._negLogLikelihood, s, jac=True, bounds=boundPairs,
                                   maxIter=self.options["maxIter"], tol=1e-6),
                np.array(starts))
            p = self._optResult.x
        self._factorize(p)
        self._cache = None

    def _tiedStart(self, p0, bounds):
        """Start point from a cheap fit with all kernel parameters tied to one value.

        With many ARD lengthscales the likelihood is often multimodal; the
        tied (isotropic-like) optimum lands in the basin of the global optimum
        far more reliably than random starts do.
        """
        nk = self._kernel.nParams()

        def tiedObjective(q):
            p = np.concatenate([np.full(nk, q[0]), q[1:]])
            f, g = self._negLogLikelihood(p)
            return f, np.concatenate([[g[:nk].sum()], g[nk:]])

        lo, hi = float(np.max(bounds[:nk, 0])), float(np.min(bounds[:nk, 1]))
        if lo >= hi:
            return None
        tiedBounds = [(lo, hi)] + [tuple(b) for b in bounds[nk:]]
        best = None
        for level in np.linspace(lo + 0.25 * (hi - lo), hi - 0.4 * (hi - lo), 3):
            q0 = np.concatenate([[level], p0[nk:]])
            try:
                res = minimize(tiedObjective, q0, jac=True, bounds=tiedBounds, maxIter=self.options["maxIter"], tol=1e-6)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if best is None or res.fun < best.fun:
                best = res
        if best is None:
            return None
        return np.concatenate([np.full(nk, best.x[0]), best.x[1:]])

    @property
    def _autoNugget(self) -> int:
        return 1 if self.options["nugget"] == "auto" else 0

    def _bounds(self) -> np.ndarray:
        b = self._kernel.bounds()
        if self._autoNugget:
            lo, hi = self.options["nuggetBounds"]
            b = np.vstack([b, [np.log(lo), np.log(hi)]])
        return b

    def _split(self, p):
        nk = self._kernel.nParams()
        eta = float(np.exp(p[nk])) if self._autoNugget else float(self.options["nugget"])
        return p[:nk], max(eta, 1e-12)

    def _negLogLikelihood(self, p):
        """Negative profile log-likelihood (REML or ML) and its gradient."""
        p = np.asarray(p, dtype=float)
        kp, eta = self._split(p)
        x, y, f, wn = self._xs, self._ys, self._f, self._wn
        n, q = f.shape
        k, kGrads = self._kernel.matrixAndGradients(x, kp, self._cache)
        r = k + np.diag(eta / wn)
        try:
            chol = np.linalg.cholesky(r)
        except np.linalg.LinAlgError:
            return 1e20, np.zeros_like(p)
        lInv = np.linalg.inv(chol)
        rInv = lInv.T @ lInv
        logDetR = 2.0 * np.sum(np.log(np.diag(chol)))
        reml = self.options["likelihood"] == "reml"
        if q:
            riF = rInv @ f
            a = f.T @ riF
            try:
                aChol = np.linalg.cholesky(a)
            except np.linalg.LinAlgError:
                return 1e20, np.zeros_like(p)
            aInv = np.linalg.inv(a)
            beta = aInv @ (riF.T @ y)
            resid = y - f @ beta
            logDetA = 2.0 * np.sum(np.log(np.diag(aChol)))
            proj = rInv - riF @ aInv @ riF.T
        else:
            resid, logDetA, proj = y, 0.0, rInv
        alpha = rInv @ resid
        ssr = float(resid @ alpha)
        dof = n - q if reml else n
        sigma2 = max(ssr / dof, 1e-300)
        nll = 0.5 * (dof * np.log(sigma2) + logDetR + (logDetA if reml else 0.0))
        pm = proj if reml else rInv
        grads = [0.5 * (np.sum(pm * g) - (alpha @ g @ alpha) / sigma2) for g in kGrads]
        if self._autoNugget:
            dn = eta / wn
            grads.append(0.5 * (np.sum(np.diag(pm) * dn) - np.sum(alpha * alpha * dn) / sigma2))
        return float(nll), np.array(grads)

    def _factorize(self, p) -> None:
        p = np.asarray(p, dtype=float)
        self._params = p
        kp, eta = self._split(p)
        self._kp, self._eta = kp, eta
        x, y, f, wn = self._xs, self._ys, self._f, self._wn
        n, q = f.shape
        r = self._kernel.matrix(x, x, kp) + np.diag(eta / wn)
        jitter = 0.0
        while True:
            try:
                self._chol = np.linalg.cholesky(r + jitter * np.eye(n))
                break
            except np.linalg.LinAlgError:
                jitter = 1e-10 if jitter == 0.0 else jitter * 10.0
                if jitter > 1e-2:
                    raise
        self._jitter = jitter
        lf = solveTriangular(self._chol, f, lower=True) if q else np.zeros((n, 0))
        ly = solveTriangular(self._chol, y, lower=True)
        if q:
            qf, rf = np.linalg.qr(lf)
            self._beta = np.linalg.solve(rf, qf.T @ ly)
            self._aInv = np.linalg.inv(rf.T @ rf)
        else:
            self._beta = np.zeros(0)
            self._aInv = np.zeros((0, 0))
        self._lf = lf
        resid = y - f @ self._beta
        self._alpha = solveTriangular(self._chol, solveTriangular(self._chol, resid, lower=True), lower=True, trans=True)
        dof = n - q if self.options["likelihood"] == "reml" else n
        self._sigma2 = max(float(resid @ self._alpha) / dof, 1e-300)
        # Effective dof: trace of the smoother S = I - eta W^-1 P.
        lInv = solveTriangular(self._chol, np.eye(n), lower=True)
        rInv = lInv.T @ lInv
        diagP = np.diag(rInv) - (np.sum((rInv @ f) @ self._aInv * (rInv @ f), axis=1) if q else 0.0)
        self._edf = float(n - np.sum(eta / wn * diagP))
        self._logLik = -float(self._negLogLikelihood(p)[0])

    # ------------------------------------------------------------------ prediction
    def _scaled(self, x):
        return (x - self._xMean) / self._xStd

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        xs = self._scaled(x)
        mean = self._kernel.matrix(xs, self._xs, self._kp) @ self._alpha
        if self._trend is not None:
            mean = mean + self._trend.transform(xs) @ self._beta
        return (self._yMean + self._yStd * mean)[:, None]

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        xs = self._scaled(x)
        rx = self._kernel.matrix(xs, self._xs, self._kp)
        v = solveTriangular(self._chol, rx.T, lower=True)
        kxx = np.ones(x.shape[0])
        var = kxx - np.sum(v * v, axis=0)
        if self._trend is not None:
            u = self._lf.T @ v - self._trend.transform(xs).T
            var = var + np.sum(u * (self._aInv @ u), axis=0)
        var = np.maximum(var, 0.0) * self._sigma2
        if kind == "prediction":
            var = var + self._sigma2 * self._eta
        return (var * self._yStd ** 2)[:, None]

    def _predictDerivatives(self, x: np.ndarray, kx: int) -> np.ndarray:
        xs = self._scaled(x)
        d = self._kernel.dx(xs, self._xs, self._kp, kx) @ self._alpha
        if self._trend is not None:
            d = d + self._trend.derivative(xs, kx) @ self._beta
        return (d * self._yStd / self._xStd[kx])[:, None]

    def _effectiveParams(self):
        return self._edf

    # ------------------------------------------------------------------ reporting
    @property
    def hyperparameters(self) -> dict:
        """Fitted hyperparameters in physical units."""
        self._checkTrained()
        if self._subModels is not None:
            return {"outputs": [m.hyperparameters for m in self._subModels]}
        names = self._kernel.paramNames()
        out = {"logParams": dict(zip(names + (["log_nugget"] if self._autoNugget else []), self._params.tolist())),
               "processVariance": self._sigma2 * self._yStd ** 2,
               "nugget": self._eta,
               "noiseVariance": self._sigma2 * self._eta * self._yStd ** 2,
               "logLikelihood": self._logLik,
               "trendCoefficients": self._beta.tolist()}
        if self._pls is None and getattr(self._kernel, "invL2", None) is not None and "ard" in self._kernel.options:
            inv = self._kernel.invL2(self._kp)
            out["lengthscales"] = (self._xStd / np.sqrt(inv)).tolist()
        if self._optResult is not None:
            out["optimizer"] = {"success": self._optResult.success, "message": self._optResult.message,
                                "nStarts": self._optResult.extra.get("nStarts"),
                                "nFailedStarts": self._optResult.extra.get("nFailedStarts")}
        return out

    def describe(self) -> str:
        base = super().describe()
        return base.replace("kriging(", "kpls(", 1) if self.options["plsComponents"] else base

    # ------------------------------------------------------------------ persistence
    def _stateToDict(self) -> dict:
        return {"params": self._params.tolist(), "xMean": self._xMean.tolist(), "xStd": self._xStd.tolist(),
                "yMean": self._yMean, "yStd": self._yStd, "xs": self._xs.tolist(), "ys": self._ys.tolist(),
                "wn": self._wn.tolist(), "pls": None if self._pls is None else self._pls.tolist()}

    def _stateFromDict(self, state: dict) -> None:
        self._xMean = np.array(state["xMean"])
        self._xStd = np.array(state["xStd"])
        self._yMean, self._yStd = float(state["yMean"]), float(state["yStd"])
        self._xs = np.array(state["xs"], dtype=float).reshape(-1, self.nx)
        self._ys = np.array(state["ys"], dtype=float)
        self._wn = np.array(state["wn"], dtype=float)
        self._pls = None if state["pls"] is None else np.array(state["pls"], dtype=float)
        self._kernel = buildKernel(self.options["corr"]).setup(self.nx, self._pls)
        deg = _TREND_DEGREE[self.options["poly"]]
        self._trend = PolynomialBasis(degree=deg).fit(self._xs) if deg >= 0 else None
        self._f = self._trend.transform(self._xs) if self._trend is not None else np.zeros((self._xs.shape[0], 0))
        self._optResult = None
        self._cache = None
        self._factorize(np.array(state["params"], dtype=float))


@registry("model").register("kpls")
class KplsModel(KrigingModel):
    """Kriging with PLS-reduced hyperparameters for many inputs (KPLS)."""

    def _initialize(self) -> None:
        super()._initialize()
        self.options["plsComponents"] = 2
