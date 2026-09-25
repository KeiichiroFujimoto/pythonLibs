"""Multi-output Kriging with a linear model of coregionalization (cokriging).

Model for ny outputs (each standardized unless ``normalize=False``)::

    y_a(x) = f(x)^T beta_a + sum_q u_qa(x) + eps_a
    Cov[u_qa(x), u_qb(x')] = B_q[a, b] k_q(x, x'),   B_q = L_q L_q^T (rank ``rank``)
    Var[eps_a] = tau2_a

One latent kernel (``corr`` a single spec) is the intrinsic coregionalization
model (ICM); a list of specs gives the linear model of coregionalization
(LMC). Outputs may be observed at different inputs: mark unobserved values
with NaN (heterotopic / collocated data).

All covariance parameters (kernel parameters, the entries of L_q and the
log noise variances) maximize the REML (default) or ML likelihood with its
analytic gradient; the trend coefficients are profiled out by generalized
least squares.

References:
    Goovaerts (1997) *Geostatistics for Natural Resources Evaluation*, ch. 6.
    Wackernagel (2003) *Multivariate Geostatistics*.
    Alvarez, Rosasco & Lawrence (2012) Foundations and Trends in Machine Learning 4(3).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.bases.PolynomialBasis import PolynomialBasis
from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase
from pythonLibs.regressionHandler.kernels.Kernels import buildKernel
from pythonLibs.regressionHandler.models.KrigingModel import _jitteredCholesky
from pythonLibs.regressionHandler.numerics.LinearAlgebra import solveTriangular
from pythonLibs.regressionHandler.numerics.Optimizers import minimize, multiStart
from pythonLibs.regressionHandler.sampling.Sampling import latinHypercube

_TREND_DEGREE = {"none": -1, "constant": 0, "linear": 1, "quadratic": 2}


@registry("model").register("cokriging")
class CokrigingModel(SurrogateModelBase):
    """Cokriging: jointly correlated outputs (ICM / LMC), heterotopic data allowed (NaN = unobserved)."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("corr", "matern52", types=(str, dict, list, object),
          desc="Kernel spec of the latent process, or a list of specs (one per latent process: LMC)")
        d("rank", None, types=int, lower=1, desc="Rank of each coregionalization matrix (default: ny)")
        d("poly", "constant", values=tuple(_TREND_DEGREE), desc="Polynomial trend of every output")
        d("normalize", True, types=bool, desc="Standardize inputs and every output internally")
        d("nugget", "auto", values=("auto",), types=(int, float),
          desc="Noise variance of every output (standardized units); 'auto' estimates one per output")
        d("nuggetBounds", [1e-8, 10.0], types=list, desc="Bounds of the estimated noise variances")
        d("likelihood", "reml", values=("reml", "ml"), desc="Estimation criterion")
        d("nStart", 5, types=int, lower=1, desc="Optimizer starts")
        d("maxIter", 300, types=int, lower=1, desc="L-BFGS iterations per start")
        d("seed", 0, types=int, desc="Seed of the multi-start design")
        d("hyperparameters", None, types=list, desc="Fixed parameter vector (skips optimization)")
        self.supports.update(multiOutput=True, missingOutputs=True, variances=True, derivatives=True,
                             covariance=True, weights=False)

    def _validateOptions(self) -> None:
        for spec in self._kernelSpecs():
            buildKernel(spec)

    def _kernelSpecs(self) -> list:
        c = self.options["corr"]
        return list(c) if isinstance(c, list) else [c]

    # ------------------------------------------------------------------ setup
    def _prepare(self) -> None:
        x, y = self.xt, self.yt
        ny = y.shape[1]
        if self.options["normalize"]:
            self._xMean = x.mean(axis=0)
            sx = x.std(axis=0)
            self._xStd = np.where(sx > 0, sx, 1.0)
            self._yMean = np.array([np.nanmean(y[:, a]) for a in range(ny)])
            sy = np.array([np.nanstd(y[:, a]) for a in range(ny)])
            self._yStd = np.where(sy > 0, sy, 1.0)
        else:
            self._xMean, self._xStd = np.zeros(self.nx), np.ones(self.nx)
            self._yMean, self._yStd = np.zeros(ny), np.ones(ny)
        self._xs = (x - self._xMean) / self._xStd
        ys = (y - self._yMean) / self._yStd
        obs = np.isfinite(ys)
        self._pi, self._oa = np.nonzero(obs)                  # point index, output index of each observation
        self._yObs = ys[self._pi, self._oa]
        self._kernels = [buildKernel(spec).setup(self.nx) for spec in self._kernelSpecs()]
        self._rank = min(self.options["rank"] or ny, ny)
        deg = _TREND_DEGREE[self.options["poly"]]
        self._poly = PolynomialBasis(degree=deg).fit(self._xs) if deg >= 0 else None
        self._F = self._trendRows(self._xs[self._pi], self._oa)
        if self._F.shape[1] >= self._yObs.size:
            raise ValueError("too few observations for the trend")

    def _trendRows(self, xs, outputs) -> np.ndarray:
        if self._poly is None:
            return np.zeros((xs.shape[0], 0))
        phi = self._poly.transform(xs)
        p = phi.shape[1]
        out = np.zeros((xs.shape[0], p * self.ny))
        for a in range(self.ny):
            rows = outputs == a
            out[rows, a * p:(a + 1) * p] = phi[rows]
        return out

    # ------------------------------------------------------------------ parameters
    def _layout(self):
        """Slices of the parameter vector: per latent process (kernel params, L entries), then noise."""
        out, start = [], 0
        nl = self.ny * self._rank
        for k in self._kernels:
            ks = slice(start, start + k.nParams())
            ls = slice(ks.stop, ks.stop + nl)
            out.append((ks, ls))
            start = ls.stop
        noise = slice(start, start + (self.ny if self.options["nugget"] == "auto" else 0))
        return out, noise

    def _nParams(self) -> int:
        layout, noise = self._layout()
        return noise.stop

    def _coreg(self, p, q) -> np.ndarray:
        layout, _ = self._layout()
        return np.asarray(p[layout[q][1]], dtype=float).reshape(self.ny, self._rank)

    def _noise(self, p) -> np.ndarray:
        _, noise = self._layout()
        if self.options["nugget"] == "auto":
            return np.exp(np.asarray(p[noise], dtype=float))
        return np.full(self.ny, float(self.options["nugget"]))

    def _initialParams(self) -> np.ndarray:
        ys = (self.yt - self._yMean) / self._yStd
        corr = np.eye(self.ny)
        for a in range(self.ny):
            for b in range(a):
                both = np.isfinite(ys[:, a]) & np.isfinite(ys[:, b])
                if both.sum() > 2:
                    c = np.corrcoef(ys[both, a], ys[both, b])[0, 1]
                    corr[a, b] = corr[b, a] = 0.9 * c if np.isfinite(c) else 0.0
        w, v = np.linalg.eigh(corr)
        l = (v * np.sqrt(np.maximum(w, 1e-3)))[:, ::-1][:, : self._rank]
        nq = len(self._kernels)
        parts = []
        for k in self._kernels:
            parts.extend([k.initialParams(), (l / np.sqrt(nq)).ravel()])
        if self.options["nugget"] == "auto":
            parts.append(np.full(self.ny, np.log(1e-2)))
        return np.concatenate(parts)

    def _bounds(self) -> np.ndarray:
        parts = []
        for k in self._kernels:
            parts.extend([k.bounds(), np.tile([-10.0, 10.0], (self.ny * self._rank, 1))])
        if self.options["nugget"] == "auto":
            lo, hi = np.log(self.options["nuggetBounds"])
            parts.append(np.tile([lo, hi], (self.ny, 1)))
        return np.vstack(parts)

    def paramNames(self) -> list[str]:
        names = []
        for q, k in enumerate(self._kernels):
            names += [f"q{q}.{n}" for n in k.paramNames()]
            names += [f"q{q}.L{a}{r}" for a in range(self.ny) for r in range(self._rank)]
        if self.options["nugget"] == "auto":
            names += [f"log_noise{a}" for a in range(self.ny)]
        return names

    # ------------------------------------------------------------------ covariance
    def _covariance(self, p, withGrads: bool):
        layout, noiseSlice = self._layout()
        x = self._xs
        pi, oa = self._pi, self._oa
        nObs = pi.size
        c = np.zeros((nObs, nObs))
        grads = []
        for q, k in enumerate(self._kernels):
            ks, _ = layout[q]
            lq = self._coreg(p, q)
            bq = lq @ lq.T
            if withGrads:
                kMat, kGrads = k.matrixAndGradients(x, p[ks])
            else:
                kMat, kGrads = k.matrix(x, x, p[ks]), []
            kObs = kMat[np.ix_(pi, pi)]
            bObs = bq[np.ix_(oa, oa)]
            c += bObs * kObs
            if withGrads:
                grads.extend(bObs * g[np.ix_(pi, pi)] for g in kGrads)
                for a in range(self.ny):
                    for r in range(self._rank):
                        db = np.zeros((self.ny, self.ny))
                        db[a, :] += lq[:, r]
                        db[:, a] += lq[:, r]
                        grads.append(db[np.ix_(oa, oa)] * kObs)
        tau2 = self._noise(p)
        c[np.diag_indices(nObs)] += tau2[oa]
        if withGrads and self.options["nugget"] == "auto":
            for a in range(self.ny):
                grads.append(np.diag(np.where(oa == a, tau2[a], 0.0)))
        return c, grads

    def _negLogLikelihood(self, p):
        p = np.asarray(p, dtype=float)
        c, grads = self._covariance(p, True)
        y, f = self._yObs, self._F
        n, q = f.shape
        c[np.diag_indices(n)] += getattr(self, "_likelihoodJitter", 0.0)
        try:
            chol = np.linalg.cholesky(c)
        except np.linalg.LinAlgError:
            return 1e20, np.zeros_like(p)
        lInv = solveTriangular(chol, np.eye(n), lower=True)
        cInv = lInv.T @ lInv
        logDet = 2.0 * float(np.sum(np.log(np.diag(chol))))
        reml = self.options["likelihood"] == "reml"
        if q:
            cf = cInv @ f
            a = f.T @ cf
            try:
                aChol = np.linalg.cholesky(a)
            except np.linalg.LinAlgError:
                return 1e20, np.zeros_like(p)
            aInv = np.linalg.inv(a)
            proj = cInv - cf @ aInv @ cf.T
            logDetA = 2.0 * float(np.sum(np.log(np.diag(aChol))))
        else:
            proj, logDetA = cInv, 0.0
        alpha = proj @ y
        quad = float(y @ alpha)
        dof = n - q if reml else n
        nll = 0.5 * (quad + logDet + (logDetA if reml else 0.0) + dof * np.log(2.0 * np.pi))
        w = proj if reml else cInv
        g = np.array([0.5 * (np.sum(w * dc) - alpha @ dc @ alpha) for dc in grads])
        return nll, g

    # ------------------------------------------------------------------ training
    def _train(self) -> None:
        self._prepare()
        self._likelihoodJitter = 0.0
        fixed = self.options["hyperparameters"]
        if fixed is not None:
            p = np.asarray(fixed, dtype=float)
            if p.size != self._nParams():
                raise ValueError(f"expected {self._nParams()} hyperparameters, got {p.size}")
            self._optResult = None
        else:
            bounds = self._bounds()
            p0 = np.clip(self._initialParams(), bounds[:, 0], bounds[:, 1])
            starts = [p0]
            if self.options["nStart"] > 1:
                rng = np.random.default_rng(self.options["seed"])
                layout, _ = self._layout()
                box = np.column_stack([np.maximum(bounds[:, 0], p0 - 1.5), np.minimum(bounds[:, 1], p0 + 1.5)])
                design = latinHypercube(self.options["nStart"] - 1, box, criterion="maximin", seed=self.options["seed"])
                for row in design:
                    s = row.copy()
                    for (_, ls) in layout:           # keep the coregionalization start, perturb it mildly
                        s[ls] = p0[ls] * (1.0 + 0.3 * rng.standard_normal(ls.stop - ls.start))
                    starts.append(s)
            def optimize():
                return multiStart(
                    lambda s: minimize(self._negLogLikelihood, s, jac=True, bounds=[tuple(b) for b in bounds],
                                       maxIter=self.options["maxIter"], tol=1e-7),
                    np.array(starts))

            self._optResult = optimize()
            if self._optResult.fun >= 1e20:
                # C is singular at every start (e.g. duplicated inputs with a zero nugget), so no likelihood
                # was ever evaluated: optimize that of the jitter-regularized C the factorization uses anyway.
                _, self._likelihoodJitter = _jitteredCholesky(self._covariance(p0, False)[0])
                self._optResult = optimize()
            p = self._optResult.x
        self._factorize(p)

    def _factorize(self, p) -> None:
        self._params = np.asarray(p, dtype=float)
        c, _ = self._covariance(self._params, False)
        n = c.shape[0]
        self._chol, jitter = _jitteredCholesky(c)
        # the reported likelihood refers to the matrix actually factorized
        self._likelihoodJitter = jitter
        f, y = self._F, self._yObs
        q = f.shape[1]
        self._lf = solveTriangular(self._chol, f, lower=True) if q else np.zeros((n, 0))
        ly = solveTriangular(self._chol, y, lower=True)
        if q:
            qf, rf = np.linalg.qr(self._lf)
            self._beta = np.linalg.solve(rf, qf.T @ ly)
            self._aInv = np.linalg.inv(rf.T @ rf)
        else:
            self._beta, self._aInv = np.zeros(0), np.zeros((0, 0))
        resid = y - f @ self._beta
        self._alpha = solveTriangular(self._chol, solveTriangular(self._chol, resid, lower=True), lower=True,
                                      trans=True)
        # The likelihood is evaluated on the standardized outputs; the Jacobian of y_a -> y_a / yStd_a
        # puts it on the raw output scale (REML: n_a - p contrasts per output, as for KrigingModel).
        nObs = np.bincount(self._oa, minlength=self.ny)
        if self.options["likelihood"] == "reml":
            nObs = nObs - (self._poly.nTerms if self._poly is not None else 0)
        self._logLik = -float(self._negLogLikelihood(self._params)[0]) - float(np.sum(nObs * np.log(self._yStd)))

    # ------------------------------------------------------------------ prediction
    def _scaled(self, x):
        return (x - self._xMean) / self._xStd

    def _crossCov(self, xs, b: int, kx: Optional[int] = None) -> np.ndarray:
        """Cov(u_b(x), y_obs) (m, nObs), or its derivative in input kx."""
        layout, _ = self._layout()
        out = np.zeros((xs.shape[0], self._pi.size))
        for q, k in enumerate(self._kernels):
            ks, _ = layout[q]
            lq = self._coreg(self._params, q)
            bq = lq @ lq.T
            km = k.matrix(xs, self._xs, self._params[ks]) if kx is None else k.dx(xs, self._xs, self._params[ks], kx)
            out += bq[b, self._oa][None, :] * km[:, self._pi]
        return out

    def _priorVar(self, xs, b: int) -> np.ndarray:
        layout, _ = self._layout()
        v = np.zeros(xs.shape[0])
        for q, k in enumerate(self._kernels):
            ks, _ = layout[q]
            lq = self._coreg(self._params, q)
            v += float(lq[b] @ lq[b]) * np.diag(k.matrix(xs[:1], xs[:1], self._params[ks]))[0]
        return v

    def _trendAt(self, xs, b: int) -> np.ndarray:
        return self._trendRows(xs, np.full(xs.shape[0], b))

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        xs = self._scaled(x)
        out = np.empty((x.shape[0], self.ny))
        for b in range(self.ny):
            mean = self._crossCov(xs, b) @ self._alpha
            if self._F.shape[1]:
                mean = mean + self._trendAt(xs, b) @ self._beta
            out[:, b] = self._yMean[b] + self._yStd[b] * mean
        return out

    def _latentCov(self, xs, b: int, full: bool):
        cx = self._crossCov(xs, b)
        v = solveTriangular(self._chol, cx.T, lower=True)
        if full:
            layout, _ = self._layout()
            prior = np.zeros((xs.shape[0], xs.shape[0]))
            for q, k in enumerate(self._kernels):
                ks, _ = layout[q]
                lq = self._coreg(self._params, q)
                prior += float(lq[b] @ lq[b]) * k.matrix(xs, xs, self._params[ks])
            cov = prior - v.T @ v
        else:
            cov = self._priorVar(xs, b) - np.sum(v * v, axis=0)
        if self._F.shape[1]:
            u = self._lf.T @ v - self._trendAt(xs, b).T
            cov = cov + (u.T @ self._aInv @ u if full else np.sum(u * (self._aInv @ u), axis=0))
        return cov

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        xs = self._scaled(x)
        tau2 = self._noise(self._params)
        out = np.empty((x.shape[0], self.ny))
        for b in range(self.ny):
            var = np.maximum(self._latentCov(xs, b, False), 0.0)
            if kind == "prediction":
                var = var + tau2[b]
            out[:, b] = var * self._yStd[b] ** 2
        return out

    def _predictCovariance(self, x: np.ndarray, kind: str) -> np.ndarray:
        xs = self._scaled(x)
        tau2 = self._noise(self._params)
        out = []
        for b in range(self.ny):
            cov = self._latentCov(xs, b, True)
            cov = 0.5 * (cov + cov.T)
            if kind == "prediction":
                cov = cov + tau2[b] * np.eye(x.shape[0])
            out.append(cov * self._yStd[b] ** 2)
        return np.stack(out)

    def _predictDerivatives(self, x: np.ndarray, kx: int) -> np.ndarray:
        xs = self._scaled(x)
        out = np.empty((x.shape[0], self.ny))
        for b in range(self.ny):
            d = self._crossCov(xs, b, kx) @ self._alpha
            if self._poly is not None:
                p = self._poly.nTerms
                d = d + self._poly.derivative(xs, kx) @ self._beta[b * p:(b + 1) * p]
            out[:, b] = d * self._yStd[b] / self._xStd[kx]
        return out

    def _effectiveParams(self):
        return np.full(self.ny, float(self._poly.nTerms if self._poly is not None else 0))

    # ------------------------------------------------------------------ reporting
    @property
    def coregionalization(self) -> list[np.ndarray]:
        """B_q in raw output units (ny, ny) per latent process."""
        self._checkTrained()
        s = self._yStd
        return [(lambda l: (l @ l.T) * np.outer(s, s))(self._coreg(self._params, q)) for q in range(len(self._kernels))]

    @property
    def hyperparameters(self) -> dict:
        self._checkTrained()
        bs = self.coregionalization
        total = sum(bs)
        sd = np.sqrt(np.diag(total))
        out = {"logParams": dict(zip(self.paramNames(), self._params.tolist())),
               "coregionalization": [b.tolist() for b in bs],
               "crossCorrelation": (total / np.outer(sd, sd)).tolist(),
               "noiseVariance": (self._noise(self._params) * self._yStd ** 2).tolist(),
               "logLikelihood": self._logLik,
               "trendCoefficients": self._beta.tolist()}
        layout, _ = self._layout()
        out["kernels"] = [dict(zip(k.paramNames(), self._params[layout[q][0]].tolist()))
                          for q, k in enumerate(self._kernels)]
        if self._optResult is not None:
            out["optimizer"] = {"success": self._optResult.success, "message": self._optResult.message}
        return out

    # ------------------------------------------------------------------ persistence
    def _stateToDict(self) -> dict:
        y = np.where(np.isfinite(self.yt), self.yt, np.nan) if self.yt is not None else None
        return {"params": self._params.tolist(), "x": self.xt.tolist(),
                "y": [[None if not np.isfinite(v) else float(v) for v in row] for row in y]}

    def _stateFromDict(self, state: dict) -> None:
        self.xt = np.array(state["x"], dtype=float)
        self.yt = np.array([[np.nan if v is None else v for v in row] for row in state["y"]], dtype=float)
        self._prepare()
        self._optResult = None
        self._factorize(np.array(state["params"], dtype=float))
