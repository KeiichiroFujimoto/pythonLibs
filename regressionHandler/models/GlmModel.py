"""Generalized linear models: any basis x any exponential family x any link.

    g(E[y | x]) = Phi(x) beta + offset,   Var[y | x] = phi V(mu) / w

::

    GlmModel(family="poisson", basis={"type": "polynomial", "degree": 2})     # log-linear counts
    GlmModel(family="binomial", basis="linear")                                # logistic regression (0/1 or proportions,
                                                                               #   trials as sample weights)
    GlmModel(family={"type": "gamma", "link": "log"})
    GlmModel(family={"type": "negativeBinomial"})                              # theta estimated
    GlmModel(family="binomial", basis={"type": "bspline", "nSegments": 20},
             penalty="smoothness", alpha="auto")                               # penalized (P-spline) GLM, UBRE

Estimation is (penalized) IRLS; inference uses the Fisher information
(z tests for known-scale families, t tests with the Pearson dispersion
otherwise). Predictions are on the response scale; ``predictLink`` gives
the linear predictor and ``predictInterval(kind="confidence")`` maps the
link-scale interval through the inverse link (it stays inside the valid
range, e.g. [0, 1] for probabilities).

With several outputs one GLM is fitted per output; the GLM-specific methods
then return {outputName: result} (or stacked columns for array results).
The log-likelihood (AIC, BIC) follows the usual conventions for weighted
fits: gaussian scale D / n with sum(log w) / 2, gamma and inverse Gaussian
scale D / sum(w) with the density weighted by w.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.core.FitResult import FitResult
from pythonLibs.regressionHandler.core.Registry import buildComponent, registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import PredictionInterval, SurrogateModelBase
from pythonLibs.regressionHandler.glm.Families import buildFamily
from pythonLibs.regressionHandler.glm.Pirls import PirlsResult, pirls, selectSmoothing
from pythonLibs.regressionHandler.numerics.Distributions import Normal, StudentT

import pythonLibs.regressionHandler.bases  # noqa: F401  (registers bases)


@registry("model").register("glm")
class GlmModel(SurrogateModelBase):
    """Generalized linear model (binomial, poisson, gamma, inverse Gaussian, negative binomial, tweedie ...)."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("basis", {"type": "polynomial", "degree": 1}, types=(str, dict, object), desc="Basis spec of the linear predictor")
        d("family", "gaussian", types=(str, dict, object), desc="Family name or {'type', 'link', ...}")
        d("link", None, types=str, desc="Link override (default: the family's canonical link)")
        d("penalty", None, values=("ridge", "smoothness"), desc="Quadratic penalty on the coefficients")
        d("alpha", "auto", values=("auto", "gcv", "ubre"), types=(int, float), lower=0.0,
          desc="Penalty weight, or chosen by GCV / UBRE ('auto': UBRE for known scale, else GCV)")
        d("offsetColumn", None, types=int, desc="Input column added to the linear predictor (not in the basis)")
        d("dispersion", "pearson", values=("pearson", "deviance"), types=(int, float), lower=0.0,
          desc="Scale estimate for unknown-scale families, or a fixed value")
        d("maxIter", 100, types=int, lower=1, desc="IRLS iterations")
        d("tol", 1e-10, types=float, lower=0.0, desc="Relative change of the penalized deviance")
        self.supports.update(variances=True, derivatives=True, parameterInference=True)

    def _validateOptions(self) -> None:
        self._makeFamily()
        buildComponent("basis", copy.deepcopy(self.options["basis"]))

    def _makeFamily(self):
        spec = self.options["family"]
        spec = {"type": spec} if isinstance(spec, str) else (dict(spec) if isinstance(spec, dict) else spec)
        if isinstance(spec, dict) and self.options["link"] is not None:
            spec["link"] = self.options["link"]
        return buildFamily(spec)

    # ------------------------------------------------------------------ design
    def _checkOffsetColumn(self) -> None:
        # negative indices count from the end; anything outside [-nx, nx) would silently wrap around
        oc = self.options["offsetColumn"]
        if oc is not None and not -self.nx <= oc < self.nx:
            raise ValueError(f"offsetColumn={oc} is out of range for {self.nx} inputs")

    def _split(self, x):
        oc = self.options["offsetColumn"]
        if oc is None:
            return x, np.zeros(x.shape[0])
        oc = oc % x.shape[1]
        keep = [j for j in range(x.shape[1]) if j != oc]
        return x[:, keep], x[:, oc]

    def _design(self, x):
        xb, off = self._split(x)
        return self._basis.transform(xb), off

    # ------------------------------------------------------------------ training
    def _train(self) -> None:
        y, w = self.yt[:, 0], self.wt
        self._checkOffsetColumn()
        self._family = self._makeFamily()
        self._family.checkResponse(y)
        xb, off = self._split(self.xt)
        self._basis = copy.deepcopy(buildComponent("basis", self.options["basis"])).fit(xb)
        phi = self._basis.transform(xb)
        pen = None
        if self.options["penalty"] == "ridge":
            pen = np.diag((~self._basis.biasMask).astype(float))
        elif self.options["penalty"] == "smoothness":
            pen = self._basis.penaltyMatrix()
            if pen is None:
                raise ValueError("this basis has no smoothness penalty; use penalty='ridge'")
        self._alpha = 0.0
        self._criterion = None
        for _ in range(50):                                 # outer loop only iterates for estimated theta
            res = self._fitOnce(phi, y, w, off, pen)
            before = getattr(self._family, "theta", None)
            if getattr(self._family, "estimateTheta", False):
                self._family.updateTheta(y, res.mu, w)
            if before is None or abs(self._family.theta - before) <= 1e-7 * max(before, 1.0):
                break
        self._res = res
        n = y.size
        fam = self._family
        if fam.scaleKnown:
            self._phi = 1.0
        elif not isinstance(self.options["dispersion"], str):
            self._phi = float(self.options["dispersion"])
        elif self.options["dispersion"] == "pearson":
            self._phi = float(np.sum(w * (y - res.mu) ** 2 / fam.variance(res.mu)) / max(n - res.edf, 1.0))
        else:
            self._phi = res.deviance / max(n - res.edf, 1.0)
        self._nullDeviance = self._nullFit(y, w, off)
        self._logLik = self._logLikelihood(y, w)
        self.result = self._buildResult()

    def _fitOnce(self, phi, y, w, off, pen):
        a = self.options["alpha"]
        if pen is None:
            return pirls(phi, y, w, self._family, None, off, None, self.options["maxIter"], self.options["tol"])
        if isinstance(a, str):
            lam, res, score = selectSmoothing(phi, y, w, self._family, [pen], a, off, maxIter=self.options["maxIter"],
                                              scale=self._fixedScale())
            self._alpha, self._criterion = float(lam[0]), float(score)
            return res
        self._alpha = float(a)
        return pirls(phi, y, w, self._family, a * pen, off, None, self.options["maxIter"], self.options["tol"])

    def _fixedScale(self) -> Optional[float]:
        """Known scale for UBRE: 1 for known-scale families, a numeric ``dispersion``, else None."""
        if self._family.scaleKnown:
            return 1.0
        d = self.options["dispersion"]
        return None if isinstance(d, str) else float(d)

    def _nullFit(self, y, w, off) -> float:
        ones = np.ones((y.size, 1))
        try:
            return pirls(ones, y, w, self._family, None, off, None, self.options["maxIter"]).deviance
        except np.linalg.LinAlgError:
            return float("nan")

    def _logLikelihood(self, y, w) -> float:
        fam, mu = self._family, self._res.mu
        if fam.scaleKnown:
            return fam.logLikelihood(y, mu, w, 1.0)
        # ML-type scale used for AIC: D / n for the gaussian (weights enter as precisions),
        # D / sum(w) where the weights scale the density (gamma, inverse Gaussian)
        phiMl = self._res.deviance / (y.size if fam.name == "gaussian" else float(np.sum(w)))
        return fam.logLikelihood(y, mu, w, phiMl)

    def _buildResult(self) -> FitResult:
        dof = float("inf") if self._family.scaleKnown else max(self.nTrain - self._res.edf, 1.0)
        return FitResult(parameterNames=self.termNames, params=self._res.coef[:, None],
                         covariance=(self._phi * self._res.covUnscaled)[None], dofResid=dof,
                         sigma2=np.array([self._phi]), outputNames=self.outputNames)

    # ------------------------------------------------------------------ prediction
    def predictLink(self, x) -> np.ndarray:
        """Linear predictor eta = Phi(x) beta + offset, shape (m,) (one column per output if several)."""
        self._checkTrained()
        if self._subModels is not None:
            return np.column_stack([m.predictLink(x) for m in self._subModels])
        phi, off = self._design(self._validX(x))
        return phi @ self._res.coef + off

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        phi, off = self._design(x)
        return self._family.link.inverse(phi @ self._res.coef + off)[:, None]

    def _linkVariance(self, phi) -> np.ndarray:
        return np.einsum("ij,jk,ik->i", phi, self._res.covUnscaled, phi) * self._phi

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        phi, off = self._design(x)
        eta = phi @ self._res.coef + off
        var = self._family.link.dmuDeta(eta) ** 2 * self._linkVariance(phi)
        if kind == "prediction":
            var = var + self._phi * self._family.variance(self._family.link.inverse(eta))
        return var[:, None]

    def predictInterval(self, x, level: float = 0.95, kind: str = "confidence") -> PredictionInterval:
        """``confidence``: link-scale interval mapped through g^-1 (asymmetric, inside the valid range).
        ``prediction``: normal approximation on the response scale."""
        if kind != "confidence":
            return super().predictInterval(x, level, kind)
        self._checkTrained()
        if self._subModels is not None:
            parts = [m.predictInterval(x, level, kind) for m in self._subModels]
            return PredictionInterval(*(np.hstack([getattr(p, a) for p in parts])
                                        for a in ("mean", "lower", "upper", "std")), float(level), kind)
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        phi, off = self._design(self._validX(x))
        eta = phi @ self._res.coef + off
        se = np.sqrt(np.maximum(self._linkVariance(phi), 0.0))
        dof = self._intervalDof()
        q = float((StudentT(dof) if dof is not None else Normal()).ppf(0.5 + level / 2.0))
        inv = self._family.link.inverse
        a, b = inv(eta - q * se), inv(eta + q * se)
        mean = inv(eta)
        std = np.abs(self._family.link.dmuDeta(eta)) * se
        return PredictionInterval(mean[:, None], np.minimum(a, b)[:, None], np.maximum(a, b)[:, None],
                                  std[:, None], float(level), kind)

    def _predictDerivatives(self, x: np.ndarray, kx: int) -> np.ndarray:
        phi, off = self._design(x)
        eta = phi @ self._res.coef + off
        oc = self.options["offsetColumn"]
        if oc is not None and kx == oc % self.nx:
            deta = np.ones(x.shape[0])
        else:
            xb, _ = self._split(x)
            local = kx if oc is None or kx < oc % self.nx else kx - 1
            deta = self._basis.derivative(xb, local) @ self._res.coef
        return (self._family.link.dmuDeta(eta) * deta)[:, None]

    def _effectiveParams(self):
        return self._res.edf

    def _intervalDof(self) -> Optional[float]:
        return None if self._family.scaleKnown else max(self.nTrain - self._res.edf, 1.0)

    # ------------------------------------------------------------------ reporting
    @property
    def termNames(self) -> list[str]:
        names = self.featureNames
        oc = self.options["offsetColumn"]
        if oc is not None:
            names = [n for j, n in enumerate(names) if j != oc % self.nx]
        return self._basis.termNames(names)

    @property
    def coefficients(self) -> np.ndarray:
        """(p,) coefficients, or {outputName: (p,)} with several outputs."""
        self._checkTrained()
        if self._subModels is not None:
            return {n: m.coefficients for n, m in zip(self.outputNames, self._subModels)}
        return self._res.coef.copy()

    @property
    def family(self):
        if self._subModels is not None:
            return self._subModels[0].family
        return self._family

    def residuals(self, kind: str = "deviance") -> np.ndarray:
        """Training residuals: response, pearson, deviance or working (one column per output if several)."""
        self._checkTrained()
        if self._subModels is not None:
            return np.column_stack([m.residuals(kind) for m in self._subModels])
        self._requireTrainingData()
        y, w = self.yt[:, 0], self.wt
        eta = self.predictLink(self.xt)
        mu = self._family.link.inverse(eta)
        fam = self._family
        if kind == "response":
            return y - mu
        if kind == "pearson":
            return (y - mu) * np.sqrt(w / fam.variance(mu))
        if kind == "deviance":
            return np.sign(y - mu) * np.sqrt(np.maximum(w * fam.unitDeviance(y, mu), 0.0))
        if kind == "working":
            return (y - mu) / fam.link.dmuDeta(eta)
        raise ValueError("kind must be response, pearson, deviance or working")

    def glmSummary(self) -> dict:
        """Deviance table: deviance, null deviance, dispersion, edf, log-likelihood, AIC, BIC."""
        self._checkTrained()
        if self._subModels is not None:
            return self._eachOutput("glmSummary")
        n = self.nTrain
        k = self._res.edf + self._family.extraParams
        ll = self._logLik
        return {"family": self._family.describe(), "deviance": self._res.deviance,
                "nullDeviance": self._nullDeviance, "dispersion": self._phi, "edf": self._res.edf,
                "residualDof": n - self._res.edf, "logLikelihood": ll, "aic": -2.0 * ll + 2.0 * k,
                "bic": -2.0 * ll + np.log(n) * k, "iterations": self._res.iterations,
                "converged": self._res.converged, "penaltyWeight": self._alpha,
                "theta": getattr(self._family, "theta", None) if self._family.name == "negativeBinomial" else None}

    def summary(self) -> str:
        self._checkTrained()
        if self._subModels is not None:
            return f"Model: {self.describe()}\n\n" + "\n\n".join(
                f"[{name}]\n{m.summary()}" for name, m in zip(self.outputNames, self._subModels))
        text = super().summary()
        rows = [f"  {k:>14}: {v:.6g}" if isinstance(v, float) else f"  {k:>14}: {v}"
                for k, v in self.glmSummary().items() if v is not None]
        return text + "\n\nGLM:\n" + "\n".join(rows)

    # ------------------------------------------------------------------ persistence
    def _stateToDict(self) -> dict:
        r = self._res
        return {"basis": self._basis.toDict() if getattr(self, "_basis", None) is not None else None,
                "family": self._family.toDict(), "coef": r.coef.tolist(),
                "cov": r.covUnscaled.tolist(), "phi": self._phi, "edf": r.edf, "deviance": r.deviance,
                "nullDeviance": self._nullDeviance, "logLik": self._logLik, "alpha": self._alpha,
                "iterations": r.iterations, "converged": r.converged}

    def _stateFromDict(self, state: dict) -> None:
        self._basis = buildComponent("basis", state["basis"])
        fam = dict(state["family"])
        est = fam.pop("estimateTheta", None)
        self._family = buildFamily(fam)
        if est is not None:
            self._family.estimateTheta = bool(est)
        coef = np.array(state["coef"], dtype=float)
        self._res = PirlsResult(coef=coef, eta=np.zeros(0), mu=np.zeros(0), weights=np.zeros(0),
                                deviance=float(state["deviance"]), penalty=0.0,
                                covUnscaled=np.array(state["cov"], dtype=float), hatDiag=np.zeros(0),
                                edf=float(state["edf"]), edfTerms=np.zeros(0), iterations=int(state["iterations"]),
                                converged=bool(state["converged"]), rank=coef.size)
        self._phi = float(state["phi"])
        self._nullDeviance = float(state["nullDeviance"])
        self._logLik = float(state["logLik"])
        self._alpha = float(state["alpha"])
