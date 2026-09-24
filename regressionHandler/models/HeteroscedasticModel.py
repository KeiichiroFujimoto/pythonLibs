"""Heteroscedastic linear regression: a mean basis and a log-linear variance function, fitted jointly.

    y = Phi(x) beta + eps,   eps ~ N(0, sigma^2(x)),   log sigma^2(x) = Psi(x) gamma

Maximum likelihood by alternating (i) weighted least squares for beta with
weights 1 / sigma^2(x) and (ii) a gamma GLM with log link for the squared
residuals (r^2 / sigma^2 ~ chi^2_1 = Gamma(1/2): dispersion 2), which is the
exact ML step for gamma given beta. Iterations stop when the Gaussian
log-likelihood no longer increases.

Prediction intervals use the fitted sigma^2(x): wider where the data are
noisier. ``predictStd`` returns sigma(x).

References:
    Harvey (1976) Econometrica 44(3).
    Carroll & Ruppert (1988) *Transformation and Weighting in Regression*.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.core.FitResult import FitResult
from pythonLibs.regressionHandler.core.Registry import buildComponent, registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase
from pythonLibs.regressionHandler.glm.Families import Gamma
from pythonLibs.regressionHandler.glm.Pirls import pirls
from pythonLibs.regressionHandler.numerics.LinearAlgebra import leastSquaresSvd

import pythonLibs.regressionHandler.bases  # noqa: F401  (registers bases)


@registry("model").register("heteroscedastic")
class HeteroscedasticModel(SurrogateModelBase):
    """Linear mean + log-linear variance function, joint maximum likelihood."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("basis", {"type": "polynomial", "degree": 1}, types=(str, dict, object), desc="Mean basis")
        d("varianceBasis", {"type": "polynomial", "degree": 1}, types=(str, dict, object),
          desc="Basis of log sigma^2(x)")
        d("maxIter", 100, types=int, lower=1, desc="Alternating iterations")
        d("tol", 1e-10, types=float, lower=0.0, desc="Relative log-likelihood change")
        self.supports.update(variances=True, derivatives=True, parameterInference=True, weights=False)

    def _validateOptions(self) -> None:
        buildComponent("basis", copy.deepcopy(self.options["basis"]))
        buildComponent("basis", copy.deepcopy(self.options["varianceBasis"]))

    def _train(self) -> None:
        x, y = self.xt, self.yt[:, 0]
        n = y.size
        self._mb = copy.deepcopy(buildComponent("basis", self.options["basis"])).fit(x)
        self._vb = copy.deepcopy(buildComponent("basis", self.options["varianceBasis"])).fit(x)
        phi, psi = self._mb.transform(x), self._vb.transform(x)
        fam = Gamma(link="log")
        s2 = np.full(n, float(np.var(y)) or 1.0)
        gamma = None
        ll = -np.inf
        it = 0
        for it in range(1, self.options["maxIter"] + 1):
            sw = 1.0 / np.sqrt(s2)
            beta = leastSquaresSvd(phi * sw[:, None], y * sw).coef
            r2 = np.maximum((y - phi @ beta) ** 2, 1e-12 * float(np.mean((y - phi @ beta) ** 2)) + 1e-300)
            res = pirls(psi, r2, np.ones(n), fam, start=gamma)
            gamma = res.coef
            s2 = np.exp(psi @ gamma)
            llNew = float(-0.5 * np.sum(np.log(2.0 * np.pi * s2) + (y - phi @ beta) ** 2 / s2))
            if abs(llNew - ll) <= self.options["tol"] * (1.0 + abs(llNew)):
                ll = llNew
                break
            ll = llNew
        sw = 1.0 / np.sqrt(s2)
        sol = leastSquaresSvd(phi * sw[:, None], y * sw)
        self._beta, self._gamma = sol.coef, gamma
        self._covBeta = sol.inverseGram()
        self._covGamma = 2.0 * res.covUnscaled                       # gamma GLM dispersion of chi^2_1 / 1
        self._logLik, self._iterations = ll, it
        self.result = FitResult(parameterNames=self._mb.termNames(self.featureNames), params=self._beta[:, None],
                                covariance=self._covBeta[None], dofResid=float("inf"), sigma2=np.array([1.0]),
                                outputNames=self.outputNames)

    # ------------------------------------------------------------------ prediction
    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        return (self._mb.transform(x) @ self._beta)[:, None]

    def predictStd(self, x) -> np.ndarray:
        """Noise standard deviation sigma(x), shape (m,)."""
        self._checkTrained()
        return np.sqrt(np.exp(self._vb.transform(self._validX(x)) @ self._gamma))

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        phi = self._mb.transform(x)
        var = np.einsum("ij,jk,ik->i", phi, self._covBeta, phi)
        if kind == "prediction":
            var = var + np.exp(self._vb.transform(x) @ self._gamma)
        return var[:, None]

    def _predictDerivatives(self, x: np.ndarray, kx: int) -> np.ndarray:
        return (self._mb.derivative(x, kx) @ self._beta)[:, None]

    def _effectiveParams(self):
        return float(self._beta.size)

    @property
    def varianceParameters(self) -> dict:
        """log-variance coefficients with standard errors."""
        self._checkTrained()
        return {"names": self._vb.termNames(self.featureNames), "coef": self._gamma.tolist(),
                "stdErrors": np.sqrt(np.diag(self._covGamma)).tolist(), "logLikelihood": self._logLik,
                "iterations": self._iterations}

    def _stateToDict(self) -> dict:
        return {"mb": self._mb.toDict(), "vb": self._vb.toDict(), "beta": self._beta.tolist(),
                "gamma": self._gamma.tolist(), "covBeta": self._covBeta.tolist(), "covGamma": self._covGamma.tolist(),
                "logLik": self._logLik, "iterations": self._iterations}

    def _stateFromDict(self, state: dict) -> None:
        self._mb = buildComponent("basis", state["mb"])
        self._vb = buildComponent("basis", state["vb"])
        self._beta = np.array(state["beta"], dtype=float)
        self._gamma = np.array(state["gamma"], dtype=float)
        self._covBeta = np.array(state["covBeta"], dtype=float)
        self._covGamma = np.array(state["covGamma"], dtype=float)
        self._logLik = float(state["logLik"])
        self._iterations = int(state["iterations"])
