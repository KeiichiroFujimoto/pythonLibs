"""Bayesian linear regression with evidence (type-II ML) hyperparameters, incl. ARD / relevance vectors.

    y = Phi(x) w + eps,   eps ~ N(0, 1 / beta),   w_j ~ N(0, 1 / alpha_j)

prior="ridge": one alpha for all non-intercept coefficients (Bayesian ridge);
prior="ard":   one alpha per coefficient (automatic relevance determination);
coefficients whose alpha diverges are pruned (sparse "relevance vector" fits).

alpha and beta maximize the marginal likelihood by MacKay's fixed point

    gamma_j = 1 - alpha_j Sigma_jj,   alpha_j = gamma_j / m_j^2,   beta = (n - sum gamma) / |y - Phi m|^2

with the posterior N(m, Sigma), Sigma = (A + beta Phi^T Phi)^-1. Predictive
variance: phi^T Sigma phi (+ 1 / beta for new observations). The intercept
gets a vague fixed prior.

References:
    MacKay (1992) Neural Computation 4(3).
    Tipping (2001) JMLR 1 (relevance vector machine).
    Bishop (2006) *Pattern Recognition and Machine Learning*, sec. 3.5, 7.2.
"""
from __future__ import annotations

import copy

import numpy as np

from pythonLibs.regressionHandler.core.FitResult import FitResult
from pythonLibs.regressionHandler.core.Registry import buildComponent, registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase

import pythonLibs.regressionHandler.bases  # noqa: F401  (registers bases)


@registry("model").register("bayesLinear")
class BayesianLinearModel(SurrogateModelBase):
    """Bayesian ridge / ARD linear regression on any basis, evidence-maximized hyperparameters."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("basis", {"type": "polynomial", "degree": 1}, types=(str, dict, object), desc="Basis spec")
        d("prior", "ridge", values=("ridge", "ard"), desc="Single precision (ridge) or one per coefficient (ARD)")
        d("maxIter", 1000, types=int, lower=1, desc="Fixed-point iterations")
        d("tol", 1e-10, types=float, lower=0.0, desc="Relative change of the log evidence")
        d("pruneAlpha", 1e6, types=float, lower=0.0,
          desc="ARD: remove a term when alpha_j / |phi_j|^2 exceeds this (prior outweighs the data)")
        self.supports.update(variances=True, derivatives=True, parameterInference=True)

    def _validateOptions(self) -> None:
        buildComponent("basis", copy.deepcopy(self.options["basis"]))

    def _train(self) -> None:
        y, w = self.yt[:, 0], self.wt
        self._basis = copy.deepcopy(buildComponent("basis", self.options["basis"])).fit(self.xt)
        phi = self._basis.transform(self.xt)
        n, p = phi.shape
        sw = np.sqrt(w)
        a, b = phi * sw[:, None], y * sw
        bias = self._basis.biasMask
        # priors act on the original coefficients (the ridge prior is isotropic in them);
        # colNorm2 only scales the ARD pruning test
        colNorm2 = np.maximum(np.sum(a * a, axis=0), 1e-300)
        ata, atb = a.T @ a, a.T @ b
        alpha = np.where(bias, 1e-10, 1.0 / max(float(np.mean(atb * atb) / np.mean(np.diag(ata)) ** 2), 1e-12))
        beta = 1.0 / max(float(np.var(b)) * 0.1, 1e-12)
        active = np.ones(p, dtype=bool)
        ard = self.options["prior"] == "ard"
        logEv = -np.inf
        for it in range(1, self.options["maxIter"] + 1):
            idx = np.flatnonzero(active)
            prec = np.diag(alpha[idx]) + beta * ata[np.ix_(idx, idx)]
            ch = np.linalg.cholesky(prec)
            sigma = np.linalg.inv(ch).T @ np.linalg.inv(ch)
            m = beta * sigma @ atb[idx]
            resid = b - a[:, idx] @ m
            rss = float(resid @ resid)
            gamma = 1.0 - alpha[idx] * np.diag(sigma)
            logDetPrec = 2.0 * float(np.sum(np.log(np.diag(ch))))
            logEvNew = 0.5 * (float(np.sum(np.log(alpha[idx][~bias[idx]]))) + n * np.log(beta) - beta * rss
                              - float(np.sum(alpha[idx] * m * m)) - logDetPrec - n * np.log(2 * np.pi))
            free = ~bias[idx]
            if ard:
                alpha[idx[free]] = gamma[free] / np.maximum(m[free] ** 2, 1e-300)
            elif free.any():
                alpha[idx[free]] = float(np.sum(gamma[free])) / max(float(np.sum(m[free] ** 2)), 1e-300)
            beta = max(n - float(np.sum(gamma)), 1e-12) / max(rss, 1e-300)
            if ard:
                active &= ~((alpha / colNorm2 > self.options["pruneAlpha"]) & ~bias)
            if abs(logEvNew - logEv) <= self.options["tol"] * (1.0 + abs(logEvNew)):
                logEv = logEvNew
                break
            logEv = logEvNew
        idx = np.flatnonzero(active)
        prec = np.diag(alpha[idx]) + beta * ata[np.ix_(idx, idx)]
        sigma = np.linalg.inv(prec)
        m = beta * sigma @ atb[idx]
        self._mean = np.zeros(p)
        self._cov = np.zeros((p, p))
        self._mean[idx] = m
        self._cov[np.ix_(idx, idx)] = sigma
        self._alpha = alpha.copy()
        self._alpha[~active] = np.inf
        self._beta, self._logEvidence, self._iterations = beta, logEv, it
        self._active = active
        self.result = FitResult(parameterNames=self.termNames, params=self._mean[:, None], covariance=self._cov[None],
                                dofResid=float("inf"), sigma2=np.array([1.0 / beta]), outputNames=self.outputNames)

    # ------------------------------------------------------------------ prediction
    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        return (self._basis.transform(x) @ self._mean)[:, None]

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        phi = self._basis.transform(x)
        var = np.einsum("ij,jk,ik->i", phi, self._cov, phi)
        if kind == "prediction":
            var = var + 1.0 / self._beta
        return var[:, None]

    def _predictDerivatives(self, x: np.ndarray, kx: int) -> np.ndarray:
        return (self._basis.derivative(x, kx) @ self._mean)[:, None]

    def _effectiveParams(self):
        return float(np.sum(self._active))

    @property
    def termNames(self) -> list[str]:
        return self._basis.termNames(self.featureNames)

    @property
    def evidence(self) -> dict:
        """Log marginal likelihood, noise precision, coefficient precisions and the retained terms."""
        self._checkTrained()
        return {"logEvidence": self._logEvidence, "noiseVariance": 1.0 / self._beta,
                "alpha": [None if not np.isfinite(a) else float(a) for a in self._alpha],
                "activeTerms": [n for n, a in zip(self.termNames, self._active) if a], "iterations": self._iterations}

    def _stateToDict(self) -> dict:
        return {"basis": self._basis.toDict(), "mean": self._mean.tolist(), "cov": self._cov.tolist(),
                "beta": self._beta, "alpha": [None if not np.isfinite(a) else float(a) for a in self._alpha],
                "active": self._active.tolist(), "logEvidence": self._logEvidence, "iterations": self._iterations}

    def _stateFromDict(self, state: dict) -> None:
        self._basis = buildComponent("basis", state["basis"])
        self._mean = np.array(state["mean"], dtype=float)
        self._cov = np.array(state["cov"], dtype=float)
        self._beta = float(state["beta"])
        self._alpha = np.array([np.inf if a is None else a for a in state["alpha"]], dtype=float)
        self._active = np.array(state["active"], dtype=bool)
        self._logEvidence, self._iterations = float(state["logEvidence"]), int(state["iterations"])
