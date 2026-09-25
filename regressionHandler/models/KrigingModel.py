"""Universal Kriging / Gaussian-process regression (with KPLS dimension reduction).

Model (on standardized inputs and outputs unless ``normalize=False``)::

    y(x) = f(x)^T beta + Z(s) + eps,   Cov[Z(s), Z(s')] = sigma2 * k(s, s'; theta),
    Var[eps_i] = sigma2 * lambda / w_i        (lambda = nugget = tau^2 / sigma2)

- s are the spatial inputs (``spatialColumns``, default all inputs); the
  remaining columns act only as linear covariates in the trend
- trend f: polynomial of degree ``poly`` in the spatial inputs plus linear
  covariate terms, or any basis spec given as ``trend``
- k: any kernel from ``kernels`` (general-smoothness Matern, Wendland,
  anisotropic and great-circle distances, sums / products ...)
- beta and sigma2 are profiled out in closed form (generalized least squares);
  theta and lambda maximize the REML (default) or ML profile likelihood with
  its analytic gradient by multi-start projected L-BFGS; ``likelihood="gcv"``
  then re-chooses lambda by GCV for the REML kernel
- ``nugget="auto"`` estimates lambda; a small fixed value gives an
  interpolating model
- ``plsComponents`` switches to KPLS
- spatial-statistics reporting: lambda, tau, sigma2, range,
  effective dof, GCV, logLikelihood, with Hessian or profile-likelihood intervals;
  ``simulate`` draws conditional realizations

References:
    Sacks, Welch, Mitchell & Wynn (1989) Statistical Science 4(4).
    Rasmussen & Williams (2006) *Gaussian Processes for Machine Learning*, ch. 2, 5.
    Stein (1999) *Interpolation of Spatial Data: Some Theory for Kriging*.
    Wendland (1995) Advances in Computational Mathematics 4.
    Bouhlel et al. (2016) Structural and Multidisciplinary Optimization 53.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.bases.PolynomialBasis import PolynomialBasis
from pythonLibs.regressionHandler.core.Registry import buildComponent, registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase
from pythonLibs.regressionHandler.kernels.Kernels import buildKernel
from pythonLibs.regressionHandler.numerics.LinearAlgebra import solveTriangular
from pythonLibs.regressionHandler.numerics.Optimizers import minimize, minimizeScalar, multiStart
from pythonLibs.regressionHandler.numerics.PLS import plsRotations
from pythonLibs.regressionHandler.sampling.Sampling import latinHypercube

_TREND_DEGREE = {"none": -1, "constant": 0, "linear": 1, "quadratic": 2}


def fullLogLikelihood(nll: float, dof: int, yStd: float) -> float:
    """Maximized (restricted) log-likelihood on the output scale from a profiled objective.

    The profiled objective 0.5 (dof log sigma2 + log|R| [+ log|F^T R^-1 F|]) of the
    standardized output omits -dof/2 (1 + log 2 pi) and the Jacobian dof log(yStd).
    """
    return float(-nll - 0.5 * dof * (1.0 + np.log(2.0 * np.pi)) - dof * np.log(yStd))


def _independentColumns(f: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """Indices of a maximal set of linearly independent columns (greedy Gram-Schmidt, first kept)."""
    if f.shape[1] == 0:
        return np.zeros(0, dtype=int)
    scale = max(float(np.max(np.linalg.norm(f, axis=0))), 1e-300)
    basis, keep = [], []
    for j in range(f.shape[1]):
        v = f[:, j].astype(float).copy()
        for _ in range(2):                                  # re-orthogonalize once
            for b in basis:
                v -= (b @ v) * b
        nv = float(np.linalg.norm(v))
        if nv > rtol * max(scale, float(np.linalg.norm(f[:, j]))) and nv > rtol * scale:
            basis.append(v / nv)
            keep.append(j)
    return np.array(keep, dtype=int)


def _jitteredCholesky(r: np.ndarray) -> tuple[np.ndarray, float]:
    """Cholesky factor of r + jitter I with the smallest jitter in 0, 1e-10, 1e-9, ... 1e-2 that succeeds."""
    n = r.shape[0]
    jitter = 0.0
    while True:
        try:
            return np.linalg.cholesky(r + jitter * np.eye(n)), jitter
        except np.linalg.LinAlgError:
            jitter = 1e-10 if jitter == 0.0 else jitter * 10.0
            if jitter > 1e-2:
                raise


@registry("model").register("kriging")
class KrigingModel(SurrogateModelBase):
    """Universal Kriging / GP regression with estimated noise (KPLS with plsComponents)."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("corr", "squaredExponential", types=(str, dict, object),
          desc="Kernel spec: squaredExponential, matern (any nu), matern52, matern32, absoluteExponential, "
               "wendland, powerExponential, rationalQuadratic, periodic, or sum/product dicts")
        d("poly", "constant", values=tuple(_TREND_DEGREE), desc="Polynomial trend in the spatial inputs")
        d("trend", None, types=(str, dict, object), desc="Basis spec replacing the polynomial trend (all inputs)")
        d("spatialColumns", None, types=list,
          desc="Input columns used by the kernel (default all); the others enter the trend linearly")
        d("normalize", True, types=bool,
          desc="Standardize inputs and output internally (False: work in raw units)")
        d("nugget", "auto", values=("auto",), types=(int, float),
          desc="lambda = noise / process variance ratio; 'auto' estimates it, a tiny value interpolates")
        d("nuggetBounds", [1e-10, 10.0], types=list, desc="Bounds of the estimated nugget")
        d("likelihood", "reml", values=("reml", "ml", "restrictedProfile", "gcv"),
          desc="Estimation criterion: restricted / full likelihood, profile REML "
               "(logLikelihood + log|Omega| / 2, sigma2 = quadratic form / n), "
               "or 'gcv' (kernel by REML, then lambda by GCV)")
        d("nStart", 5, types=int, lower=1,
          desc="Optimizer starts (initial guess + LHS points; a tied-lengthscale start is added for ARD kernels)")
        d("maxIter", 200, types=int, lower=1, desc="L-BFGS iterations per start")
        d("seed", 0, types=int, desc="Seed of the multi-start design")
        d("plsComponents", None, types=int, lower=1, desc="Use KPLS with this many PLS components")
        d("hyperparameters", None, types=list,
          desc="Fixed log-hyperparameters (kernel params + log nugget if auto); skips optimization")
        self.supports.update(multiOutput=False, variances=True, derivatives=True, parameterInference=False,
                             covariance=True)

    def _validateOptions(self) -> None:
        buildKernel(self.options["corr"])
        if self.options["trend"] is not None:
            buildComponent("basis", copy.deepcopy(self.options["trend"]))

    # ------------------------------------------------------------------ data preparation
    def _prepare(self) -> None:
        """Scalings, spatial columns, kernel and trend from xt / yt / wt."""
        x, y, w = self.xt, self.yt[:, 0], self.wt
        sc = self.options["spatialColumns"]
        self._sc = np.arange(self.nx) if sc is None else np.asarray(sc, dtype=int)
        if self._sc.size == 0 or self._sc.min() < 0 or self._sc.max() >= self.nx:
            raise ValueError("spatialColumns must list valid input columns")
        kernel = buildKernel(self.options["corr"])
        greatCircle = kernel.options.get("distance") == "greatCircle" if "distance" in kernel.options else False
        # longitude / latitude columns of great-circle kernels (also inside composites) stay in degrees
        gcCols = self._sc[kernel.greatCircleColumns()] if kernel.greatCircleColumns() else np.zeros(0, dtype=int)
        if self.options["normalize"]:
            self._xMean = x.mean(axis=0)
            sx = x.std(axis=0)
            self._xStd = np.where(sx > 0, sx, 1.0)
            self._yMean = float(y.mean())
            sy = float(y.std())
            self._yStd = sy if sy > 0 else 1.0
            self._xMean[gcCols] = 0.0
            self._xStd[gcCols] = 1.0
        else:
            self._xMean, self._xStd = np.zeros(self.nx), np.ones(self.nx)
            self._yMean, self._yStd = 0.0, 1.0
        self._xs = (x - self._xMean) / self._xStd
        self._ys = (y - self._yMean) / self._yStd
        self._wn = w / w.mean()
        self._unitScale = self._typicalScale(kernel, greatCircle)
        self._pls = None
        if self.options["plsComponents"]:
            self._pls = plsRotations(self._xs[:, self._sc], self._ys, self.options["plsComponents"])
        self._kernel = self._setupKernel(kernel)
        self._buildTrend()

    def _setupKernel(self, kernel=None):
        """Kernel adapted to the scaled training inputs and set up (also used when loading)."""
        kernel = buildKernel(self.options["corr"]) if kernel is None else kernel
        kernel.adapt(self._xs[:, self._sc])
        return kernel.setup(self._sc.size, self._pls, self._unitScale)

    def _typicalScale(self, kernel, greatCircle: bool) -> float:
        """Unit for lengthscale starts / bounds: 1 when standardized, else the data spread."""
        s = self._xs[:, self._sc]
        if greatCircle:
            sub = s[np.random.default_rng(0).choice(s.shape[0], min(s.shape[0], 300), replace=False)]
            d = kernel.setup(2).greatCircleDistance(sub, sub)
            pos = d[d > 0]
            return float(np.median(pos)) if pos.size else 1.0
        if self.options["normalize"]:
            return 1.0
        spread = s.std(axis=0)
        spread = spread[spread > 0]
        return float(np.exp(np.mean(np.log(spread)))) if spread.size else 1.0

    def _buildTrend(self) -> None:
        spec = self.options["trend"]
        n = self._xs.shape[0]
        self._covCols = np.setdiff1d(np.arange(self.nx), self._sc)
        if spec is not None:
            self._trend = copy.deepcopy(buildComponent("basis", spec)).fit(self._xs)
            self._polyTrend = None
        else:
            deg = _TREND_DEGREE[self.options["poly"]]
            self._trend = None
            self._polyTrend = PolynomialBasis(degree=deg).fit(self._xs[:, self._sc]) if deg >= 0 else None
        self._trendKeep = None
        full = self._trendMatrix(self._xs)
        self._trendKeep = _independentColumns(full)
        if self._trendKeep.size == full.shape[1]:
            self._trendKeep = None
        self._f = self._trendMatrix(self._xs)
        if self._f.shape[1] >= n:
            raise ValueError(f"trend has {self._f.shape[1]} terms but only {n} points")

    def _trendMatrix(self, xs) -> np.ndarray:
        f = self._trendMatrixFull(xs)
        keep = getattr(self, "_trendKeep", None)
        return f if keep is None else f[:, keep]

    def _trendDerivative(self, xs, kx: int) -> np.ndarray:
        d = self._trendDerivativeFull(xs, kx)
        keep = getattr(self, "_trendKeep", None)
        return d if keep is None else d[:, keep]

    def _trendMatrixFull(self, xs) -> np.ndarray:
        if self._trend is not None:
            return self._trend.transform(xs)
        cols = []
        if self._polyTrend is not None:
            cols.append(self._polyTrend.transform(xs[:, self._sc]))
        if self._covCols.size:
            cols.append(xs[:, self._covCols])
        return np.hstack(cols) if cols else np.zeros((xs.shape[0], 0))

    def _trendDerivativeFull(self, xs, kx: int) -> np.ndarray:
        if self._trend is not None:
            if self._trend.hasDerivative:
                return self._trend.derivative(xs, kx)
            h = 1e-6 * max(1.0, float(np.max(np.abs(xs[:, kx]))))
            xp, xm = xs.copy(), xs.copy()
            xp[:, kx] += h
            xm[:, kx] -= h
            return (self._trend.transform(xp) - self._trend.transform(xm)) / (2.0 * h)
        cols = []
        if self._polyTrend is not None:
            where = np.flatnonzero(self._sc == kx)
            cols.append(self._polyTrend.derivative(xs[:, self._sc], int(where[0])) if where.size
                        else np.zeros((xs.shape[0], self._polyTrend.nTerms)))
        if self._covCols.size:
            cols.append((self._covCols == kx)[None, :].astype(float).repeat(xs.shape[0], axis=0))
        return np.hstack(cols) if cols else np.zeros((xs.shape[0], 0))

    # ------------------------------------------------------------------ training
    def _train(self) -> None:
        self._prepare()
        self._cache = self._kernel.trainingCache(self._xs[:, self._sc])
        self._likelihoodJitter = 0.0
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
            if len(self._kernel.lengthscaleIndices()) > 1:
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
            objective, jac = self._objective()

            def optimize():
                return multiStart(
                    lambda s: minimize(objective, s, jac=jac, bounds=boundPairs, maxIter=self.options["maxIter"],
                                       tol=1e-6),
                    np.array(starts))

            self._optResult = optimize()
            if self._optResult.fun >= 1e20:
                # R is singular at every start (e.g. duplicated inputs with a zero nugget), so no likelihood
                # was ever evaluated: optimize that of the jitter-regularized R the factorization uses anyway.
                kp0, eta0 = self._split(p0)
                x0 = self._xs[:, self._sc]
                _, self._likelihoodJitter = _jitteredCholesky(self._kernel.matrix(x0, x0, kp0) + np.diag(eta0 / self._wn))
                self._optResult = optimize()
            p = self._optResult.x
            if self.options["likelihood"] == "gcv" and self._autoNugget:
                p = self._gcvNugget(p)
        self._factorize(p)
        self._cache = None

    def _objective(self):
        if self.options["likelihood"] == "gcv":
            # Kernel parameters by REML; the nugget is then re-chosen by GCV.
            # A joint GCV search over all covariance parameters is ill-posed: it drifts to near-singular
            # kernels with lambda -> 0.
            return (lambda q: self._negLogLikelihood(q, "reml")), True
        return self._negLogLikelihood, True

    def _gcvNugget(self, p) -> np.ndarray:
        """Minimize GCV over log(lambda) within nuggetBounds for fixed kernel parameters (grid + Brent)."""
        nk = self._kernel.nParams()
        spectrum = self._smootherSpectrum(p[:nk])

        def score(logEta: float) -> float:
            return self._gcvFromSpectrum(spectrum, float(np.exp(logEta)))[0]

        lo, hi = np.log(self.options["nuggetBounds"])
        grid = np.linspace(lo, hi, 81)
        vals = np.array([score(g) for g in grid])
        k = int(np.argmin(vals))
        best = minimizeScalar(score, (grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]), xatol=1e-6)
        out = np.array(p, dtype=float)
        out[nk] = best.x[0] if best.fun <= vals[k] else grid[k]
        return out

    def _tiedStart(self, p0, bounds):
        """Start point from a cheap fit with all lengthscales tied to one value.

        With many ARD lengthscales the likelihood is often multimodal; the
        tied (isotropic-like) optimum lands in the basin of the global optimum
        far more reliably than random starts do.
        """
        tie = np.array(self._kernel.lengthscaleIndices(), dtype=int)
        free = np.setdiff1d(np.arange(p0.size), tie)
        objective, jac = self._objective()

        def expand(q):
            p = p0.copy()
            p[tie] = q[0]
            p[free] = q[1:]
            return p

        def tiedObjective(q):
            if jac is True:
                f, g = objective(expand(q))
                return f, np.concatenate([[g[tie].sum()], g[free]])
            return objective(expand(q))

        lo, hi = float(np.max(bounds[tie, 0])), float(np.min(bounds[tie, 1]))
        if lo >= hi:
            return None
        tiedBounds = [(lo, hi)] + [tuple(b) for b in bounds[free]]
        best = None
        for level in np.linspace(lo + 0.25 * (hi - lo), hi - 0.4 * (hi - lo), 3):
            q0 = np.concatenate([[level], p0[free]])
            try:
                res = minimize(tiedObjective, q0, jac=jac, bounds=tiedBounds, maxIter=self.options["maxIter"],
                               tol=1e-6)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if best is None or res.fun < best.fun:
                best = res
        return None if best is None else expand(best.x)

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
        if self._autoNugget:
            return p[:nk], max(float(np.exp(p[nk])), 1e-12)
        # A fixed nugget is used as given (the Cholesky jitter still guards singular systems).
        return p[:nk], float(self.options["nugget"])

    def _negLogLikelihood(self, p, criterion: Optional[str] = None):
        """Negative profile log-likelihood (REML or ML) and its gradient."""
        p = np.asarray(p, dtype=float)
        kp, eta = self._split(p)
        x, y, f, wn = self._xs[:, self._sc], self._ys, self._f, self._wn
        n, q = f.shape
        k, kGrads = self._kernel.matrixAndGradients(x, kp, self._cache)
        r = k + np.diag(eta / wn + getattr(self, "_likelihoodJitter", 0.0))
        try:
            chol = np.linalg.cholesky(r)
        except np.linalg.LinAlgError:
            return 1e20, np.zeros_like(p)
        lInv = np.linalg.inv(chol)
        rInv = lInv.T @ lInv
        logDetR = 2.0 * np.sum(np.log(np.diag(chol)))
        criterion = criterion or self.options["likelihood"]
        reml = criterion in ("reml", "restrictedProfile")
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
        dof = n - q if criterion == "reml" else n        # restrictedProfile: sigma2 = quadratic form / n
        sigma2 = max(ssr / dof, 1e-300)
        nll = 0.5 * (dof * np.log(sigma2) + logDetR + (logDetA if reml else 0.0))
        pm = proj if reml else rInv
        grads = [0.5 * (np.sum(pm * g) - (alpha @ g @ alpha) / sigma2) for g in kGrads]
        if self._autoNugget:
            dn = eta / wn
            grads.append(0.5 * (np.sum(np.diag(pm) * dn) - np.sum(alpha * alpha * dn) / sigma2))
        return float(nll), np.array(grads)

    def _smootherSpectrum(self, kp):
        """Generalized eigen-decomposition of the kernel in the null space of F^T.

        With Q2 spanning {c : F^T c = 0}, M = Q2^T K Q2, N = Q2^T W^-1 Q2 and
        M V = N V S (V^T N V = I): (I - S_lambda) y = lambda W^-1 Q2 V (S + lambda)^-1 V^T Q2^T y
        and tr(I - S_lambda) = lambda sum 1 / (s_k + lambda). Unlike (K + lambda W^-1)^-1 this
        stays accurate as lambda -> 0, and each trial lambda costs O(n^2).
        """
        x, y, f, wn = self._xs[:, self._sc], self._ys, self._f, self._wn
        n, q = f.shape
        if q:
            qFull, _ = np.linalg.qr(f, mode="complete")
            q2 = qFull[:, q:]
        else:
            q2 = np.eye(n)
        m = q2.T @ self._kernel.matrix(x, x, kp) @ q2
        lower = np.linalg.cholesky((q2 / wn[:, None]).T @ q2)
        li = solveTriangular(lower, np.eye(lower.shape[0]), lower=True)
        s, u = np.linalg.eigh(li @ m @ li.T)
        v = li.T @ u
        return np.maximum(s, 0.0), v.T @ (q2.T @ y), (q2 / wn[:, None]) @ v

    def _gcvFromSpectrum(self, spectrum, eta: float) -> tuple[float, float, np.ndarray]:
        """(GCV on the standardized scale, effective dof, residual y - S y)."""
        s, proj, back = spectrum
        n = back.shape[0]
        resid = eta * (back @ (proj / (s + eta)))
        trIminusS = eta * float(np.sum(1.0 / (s + eta)))
        rss = float(np.sum(self._wn * resid * resid)) / n
        return rss / max(trIminusS / n, 1e-12) ** 2, n - trIminusS, resid

    def _factorize(self, p) -> None:
        p = np.asarray(p, dtype=float)
        self._params = p
        kp, eta = self._split(p)
        self._kp, self._eta = kp, eta
        x, y, f, wn = self._xs[:, self._sc], self._ys, self._f, self._wn
        n, q = f.shape
        r = self._kernel.matrix(x, x, kp) + np.diag(eta / wn)
        self._chol, jitter = _jitteredCholesky(r)
        self._jitter = jitter
        # the reported likelihood (and parameterIntervals) refer to the matrix actually factorized
        self._likelihoodJitter = jitter
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
        ssr = float(resid @ self._alpha)
        dof = n - q if self.options["likelihood"] == "reml" else n
        self._sigma2 = max(ssr / dof, 1e-300)
        # Effective dof (trace of the smoother) and GCV from the stable spectral form. A Cholesky jitter
        # acts as extra nugget (exactly so for unit weights); without it a zero nugget on a singular
        # kernel matrix (duplicated inputs) gives 0 / 0 here.
        gcv, self._edf, _ = self._gcvFromSpectrum(self._smootherSpectrum(kp), eta + jitter)
        # Spatial-statistics summary on the raw output scale.
        logDetR = 2.0 * float(np.sum(np.log(np.diag(self._chol))))
        sigma2Ml = max(ssr / n, 1e-300) * self._yStd ** 2
        self._logLikelihoodProfile = float(-0.5 * n * (1.0 + np.log(2.0 * np.pi)) - 0.5 * n * np.log(sigma2Ml) - 0.5 * logDetR)
        logDetOmega = float(np.linalg.slogdet(self._aInv)[1]) if q else 0.0
        self._logRestrictedProfile = self._logLikelihoodProfile + 0.5 * logDetOmega
        self._gcv = float(gcv * self._yStd ** 2)
        self._sigma2Ml = sigma2Ml
        criterion = self.options["likelihood"]
        crit = "reml" if criterion == "gcv" else criterion
        self._logLik = fullLogLikelihood(float(self._negLogLikelihood(p, crit)[0]), n - q if crit == "reml" else n,
                                         self._yStd)

    # ------------------------------------------------------------------ prediction
    def _scaled(self, x):
        return (x - self._xMean) / self._xStd

    def _crossCov(self, xs) -> np.ndarray:
        return self._kernel.matrix(xs[:, self._sc], self._xs[:, self._sc], self._kp)

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        xs = self._scaled(x)
        mean = self._crossCov(xs) @ self._alpha
        if self._f.shape[1]:
            mean = mean + self._trendMatrix(xs) @ self._beta
        return (self._yMean + self._yStd * mean)[:, None]

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        xs = self._scaled(x)
        rx = self._crossCov(xs)
        v = solveTriangular(self._chol, rx.T, lower=True)
        kxx = np.diag(self._kernel.matrix(xs[:1, self._sc], xs[:1, self._sc], self._kp))[0] * np.ones(x.shape[0])
        var = kxx - np.sum(v * v, axis=0)
        if self._f.shape[1]:
            u = self._lf.T @ v - self._trendMatrix(xs).T
            var = var + np.sum(u * (self._aInv @ u), axis=0)
        var = np.maximum(var, 0.0) * self._sigma2
        if kind == "prediction":
            var = var + self._sigma2 * self._eta
        return (var * self._yStd ** 2)[:, None]

    def _predictCovariance(self, x: np.ndarray, kind: str) -> np.ndarray:
        """Joint posterior covariance (m, m) of the latent mean (or of new observations)."""
        xs = self._scaled(x)
        xsc = xs[:, self._sc]
        v = solveTriangular(self._chol, self._crossCov(xs).T, lower=True)
        cov = self._kernel.matrix(xsc, xsc, self._kp) - v.T @ v
        if self._f.shape[1]:
            u = self._lf.T @ v - self._trendMatrix(xs).T
            cov = cov + u.T @ self._aInv @ u
        cov = 0.5 * (cov + cov.T) * self._sigma2
        if kind == "prediction":
            cov = cov + np.eye(x.shape[0]) * self._sigma2 * self._eta
        return cov * self._yStd ** 2

    def _predictDerivatives(self, x: np.ndarray, kx: int) -> np.ndarray:
        xs = self._scaled(x)
        where = np.flatnonzero(self._sc == kx)
        d = np.zeros(x.shape[0])
        if where.size:
            d = self._kernel.dx(xs[:, self._sc], self._xs[:, self._sc], self._kp, int(where[0])) @ self._alpha
        if self._f.shape[1]:
            d = d + self._trendDerivative(xs, kx) @ self._beta
        return (d * self._yStd / self._xStd[kx])[:, None]

    def _effectiveParams(self):
        return self._edf

    # ------------------------------------------------------------------ inference on hyperparameters
    def _paramReport(self, logP: np.ndarray) -> dict:
        """Physical values of hyperparameters from a log-parameter vector (lengthscales, nu, lambda)."""
        names = self._kernel.paramNames() + (["log_nugget"] if self._autoNugget else [])
        out = {}
        nk = self._kernel.nParams()
        lsCols = dict(zip(self._kernel.lengthscaleIndices(), self._kernel.lengthscaleColumnSets()))
        for i, name in enumerate(names):
            if i in lsCols and self._pls is None:
                # a lengthscale shared by columns of different spread is reported per column
                vals = np.exp(logP[i]) * self._xStd[self._sc[lsCols[i]]]
                out[name.replace("log_", "")] = float(vals[0]) if np.allclose(vals, vals[0]) else vals.tolist()
            elif name == "log_nu":
                out["nu"] = float(np.exp(logP[i]))
            elif name == "log_nugget":
                out["lambda"] = float(np.exp(logP[i]))
            elif i < nk and "log_" in name:
                # other log parameters (PLS or periodic lengthscales, periods, nested nu, L_ii ...) are
                # reported as values in the kernel's input units, never as raw logs under a physical name
                out[name.replace("log_", "", 1)] = float(np.exp(logP[i]))
            elif i < nk:
                out[name] = float(logP[i])
        return out

    def parameterIntervals(self, level: float = 0.95, method: str = "hessian", nGrid: int = 21) -> dict:
        """Confidence intervals of the hyperparameters (physical units).

        ``hessian``: Wald interval exp(log p +/- z se) from the numerical Hessian of
        the negative log-likelihood in log parameters.
        ``profile``: likelihood-ratio interval, re-optimizing the other parameters
        on a grid of each parameter (slower, asymmetric, more accurate).
        """
        self._checkTrained()
        if self._subModels is not None:
            return {"outputs": [m.parameterIntervals(level, method, nGrid) for m in self._subModels]}
        if self.options["likelihood"] == "gcv":
            raise ValueError("intervals need a likelihood criterion (ml or reml)")
        from pythonLibs.regressionHandler.numerics.Distributions import ChiSquared, Normal
        self._cache = self._kernel.trainingCache(self._xs[:, self._sc])
        p = self._params.copy()
        bounds = self._bounds()
        est = self._paramReport(p)
        names = list(est)
        try:
            if method == "hessian":
                h = np.empty((p.size, p.size))
                for i in range(p.size):
                    step = 1e-4 * max(1.0, abs(p[i]))
                    up, dn = p.copy(), p.copy()
                    up[i] += step
                    dn[i] -= step
                    h[:, i] = (self._negLogLikelihood(up)[1] - self._negLogLikelihood(dn)[1]) / (2.0 * step)
                h = 0.5 * (h + h.T)
                cov = np.linalg.pinv(h)
                z = float(Normal().ppf(0.5 + level / 2.0))
                se = np.sqrt(np.maximum(np.diag(cov), 0.0))
                lo = self._paramReport(p - z * se)
                hi = self._paramReport(p + z * se)
                return {n: {"estimate": est[n], "lower": min(lo[n], hi[n]), "upper": max(lo[n], hi[n])}
                        for n in names}
            if method != "profile":
                raise ValueError("method must be 'hessian' or 'profile'")
            drop = 0.5 * float(ChiSquared(1).ppf(level))
            best = -self._negLogLikelihood(p)[0]
            out = {}
            for i, name in enumerate(names):
                span = np.linspace(-3.0, 3.0, nGrid)
                values, ll = [], []
                others = [j for j in range(p.size) if j != i]
                for delta in span:
                    fixedVal = float(np.clip(p[i] + delta, bounds[i, 0], bounds[i, 1]))

                    def sub(q, fv=fixedVal):
                        full = p.copy()
                        full[i] = fv
                        full[others] = q
                        f, g = self._negLogLikelihood(full)
                        return f, g[others]

                    if others:
                        res = minimize(sub, p[others], jac=True, bounds=[tuple(bounds[j]) for j in others],
                                       maxIter=self.options["maxIter"], tol=1e-7)
                        ll.append(-res.fun)
                    else:
                        full = p.copy()
                        full[i] = fixedVal
                        ll.append(-self._negLogLikelihood(full)[0])
                    values.append(fixedVal)
                values, ll = np.array(values), np.array(ll)
                inside = values[best - ll <= drop]
                lo = self._paramReport(np.where(np.arange(p.size) == i, inside.min(), p))[name]
                hi = self._paramReport(np.where(np.arange(p.size) == i, inside.max(), p))[name]
                out[name] = {"estimate": est[name], "lower": lo, "upper": hi,
                             "profile": {"logValues": values.tolist(), "logLikelihood": ll.tolist()}}
            return out
        finally:
            self._cache = None

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
        report = self._paramReport(self._params)
        lengths = [v for k, v in report.items() if "engthscale" in k]
        if lengths:
            out["lengthscales"] = lengths
        if "nu" in report:
            out["nu"] = report["nu"]
        if self._optResult is not None:
            out["optimizer"] = {"success": self._optResult.success, "message": self._optResult.message,
                                "nStarts": self._optResult.extra.get("nStarts"),
                                "nFailedStarts": self._optResult.extra.get("nFailedStarts")}
        out["spatial"] = self.spatialSummary()
        return out

    def spatialSummary(self) -> dict:
        """Spatial-statistics summary of the covariance model (raw output scale).

        lambda = tau^2 / sigma2; sigma2 and tau use the ML variance estimate
        (quadratic form / n); effective dof is the exact smoother trace.
        """
        self._checkTrained()
        if self._subModels is not None:
            return {"outputs": [m.spatialSummary() for m in self._subModels]}
        out = {"lambda": self._eta, "sigma2": self._sigma2Ml, "tau": float(np.sqrt(self._eta * self._sigma2Ml)),
               "effectiveDof": self._edf, "gcv": self._gcv, "logLikelihood": self._logLikelihoodProfile,
               "logRestrictedProfile": self._logRestrictedProfile, "n": int(self._xs.shape[0])}
        report = self._paramReport(self._params)
        ranges = getattr(self._kernel, "rangeParameters", None)
        if callable(ranges) and self._kernel.options.get("distance", "scaled") == "scaled" and self._pls is None:
            r = ranges(self._kp)
            sets = self._kernel.lengthscaleColumnSets()
            if r.size == 1 and len(sets) == 1:
                # isotropic: the range in each column's units (one value if the spreads agree)
                ar = r[0] * self._xStd[self._sc[sets[0]]]
                out["range"] = float(ar[0]) if np.allclose(ar, ar[0]) else ar.tolist()
            else:
                ar = r * self._xStd[self._sc][: r.size]
                out["range"] = ar.tolist() if ar.size > 1 else float(ar[0])
        else:
            lengths = [v for k, v in report.items() if "engthscale" in k]
            if lengths:
                out["lengthscales"] = lengths if len(lengths) > 1 else lengths[0]
        if "nu" in report:
            out["nu"] = report["nu"]
        rep = self.replicates()
        if rep["nGroups"] < self._xs.shape[0]:
            out["pureErrorVariance"] = rep["pureErrorVariance"]
        return out

    # ------------------------------------------------------------------ leave-one-out
    def _looScaled(self) -> tuple[np.ndarray, np.ndarray]:
        """LOO residuals and prediction variances on the standardized scale (Dubrule 1983).

        With covariance parameters held fixed and the trend re-estimated
        without point i: y_i - yhat_(-i) = (P y)_i / P_ii and the prediction
        variance of y_i is sigma2 / P_ii, where
        P = C^-1 - C^-1 F (F^T C^-1 F)^-1 F^T C^-1.
        """
        n = self._xs.shape[0]
        lInv = solveTriangular(self._chol, np.eye(n), lower=True)
        cInv = lInv.T @ lInv
        if self._f.shape[1]:
            cf = cInv @ self._f
            proj = cInv - cf @ self._aInv @ cf.T
        else:
            proj = cInv
        d = np.diag(proj)
        return (proj @ self._ys) / d, self._sigma2 / d

    def looResiduals(self) -> np.ndarray:
        """Exact leave-one-out residuals y_i - yhat_(-i), shape (n, ny) (covariance parameters held fixed)."""
        self._checkTrained()
        if self._subModels is not None:
            return np.hstack([m.looResiduals() for m in self._subModels])
        r, _ = self._looScaled()
        return (r * self._yStd)[:, None]

    def looDiagnostics(self) -> dict:
        """Exact LOO predictions, residuals, prediction variances and standardized residuals.

        For a well-specified covariance model the standardized residuals have
        mean ~0 and variance ~1 (``meanSquaredStandardized``).
        """
        self._checkTrained()
        if self._subModels is not None:
            return {"outputs": [m.looDiagnostics() for m in self._subModels]}
        r, v = self._looScaled()
        resid = r * self._yStd
        var = v * self._yStd ** 2
        z = resid / np.sqrt(var)
        y = self._ys * self._yStd + self._yMean
        return {"predictions": y - resid, "residuals": resid, "variances": var, "standardized": z,
                "rmse": float(np.sqrt(np.mean(resid ** 2))), "meanStandardized": float(np.mean(z)),
                "meanSquaredStandardized": float(np.mean(z * z))}

    def replicates(self) -> dict:
        """Replicated input locations and the pure-error variance they imply."""
        self._checkTrained()
        if self._subModels is not None:
            return {"outputs": [m.replicates() for m in self._subModels]}
        x = self.xt if self.xt is not None else self._xs * self._xStd + self._xMean
        y = self.yt[:, 0] if self.yt is not None else self._ys * self._yStd + self._yMean
        _, inverse, counts = np.unique(x, axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.ravel()
        groups = counts.size
        if groups == x.shape[0]:
            return {"nGroups": groups, "nReplicated": 0, "pureErrorVariance": None}
        means = np.bincount(inverse, weights=y) / counts
        ss = float(np.sum((y - means[inverse]) ** 2))
        return {"nGroups": groups, "nReplicated": int(np.sum(counts > 1)),
                "pureErrorVariance": ss / (x.shape[0] - groups)}

    def summary(self) -> str:
        text = super().summary()
        if self._subModels is not None:
            return text
        fs = self.spatialSummary()
        rows = [f"  {k:>16}: {v}" if not isinstance(v, float) else f"  {k:>16}: {v:.6g}" for k, v in fs.items()]
        return text + "\n\nCovariance model (" + self.options["likelihood"].upper() + "):\n" + "\n".join(rows)

    def describe(self) -> str:
        base = super().describe()
        return base.replace("kriging(", "kpls(", 1) if self.options["plsComponents"] else base

    # ------------------------------------------------------------------ persistence
    def _stateToDict(self) -> dict:
        return {"params": self._params.tolist(), "xMean": self._xMean.tolist(), "xStd": self._xStd.tolist(),
                "yMean": self._yMean, "yStd": self._yStd, "xs": self._xs.tolist(), "ys": self._ys.tolist(),
                "wn": self._wn.tolist(), "pls": None if self._pls is None else self._pls.tolist(),
                "unitScale": self._unitScale}

    def _stateFromDict(self, state: dict) -> None:
        self._xMean = np.array(state["xMean"])
        self._xStd = np.array(state["xStd"])
        self._yMean, self._yStd = float(state["yMean"]), float(state["yStd"])
        self._xs = np.array(state["xs"], dtype=float).reshape(-1, self.nx)
        self._ys = np.array(state["ys"], dtype=float)
        self._wn = np.array(state["wn"], dtype=float)
        self._pls = None if state["pls"] is None else np.array(state["pls"], dtype=float)
        sc = self.options["spatialColumns"]
        self._sc = np.arange(self.nx) if sc is None else np.asarray(sc, dtype=int)
        self._unitScale = float(state.get("unitScale", 1.0))
        self._kernel = self._setupKernel()
        self._buildTrend()
        self._optResult = None
        self._cache = None
        self._factorize(np.array(state["params"], dtype=float))


@registry("model").register("kpls")
class KplsModel(KrigingModel):
    """Kriging with PLS-reduced hyperparameters for many inputs (KPLS)."""

    def _initialize(self) -> None:
        super()._initialize()
        self.options["plsComponents"] = 2
