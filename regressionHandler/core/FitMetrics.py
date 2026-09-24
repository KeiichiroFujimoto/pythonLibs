"""Goodness-of-fit statistics and information criteria.

Information criteria assume Gaussian errors with variance estimated by
maximum likelihood (``sigma2 = SSE_w / n``). ``nParams`` is the *effective*
number of parameters: the coefficient count for parametric models and the
trace of the hat matrix for linear smoothers (ridge, splines, kriging), so
parametric and non-parametric candidates are ranked on the same scale. The
noise variance counts as one extra parameter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FitMetrics:
    """Training-data statistics of one output.

    Attributes:
        nSamples:      number of observations
        nParams:       effective number of mean-function parameters
        sse:           weighted sum of squared residuals
        rmse:          unweighted root-mean-square residual
        mae:           unweighted mean absolute residual
        maxAbsError:   largest absolute residual
        rSquared:      weighted coefficient of determination
        adjRSquared:   R^2 adjusted for nParams (nan when dof <= 0)
        logLikelihood: Gaussian log-likelihood at the MLE variance
        aic, aicc, bic: information criteria (lower is better)
    """
    nSamples: int
    nParams: float
    sse: float
    rmse: float
    mae: float
    maxAbsError: float
    rSquared: float
    adjRSquared: float
    logLikelihood: float
    aic: float
    aicc: float
    bic: float

    @classmethod
    def compute(cls, y, yHat, nParams: float, weights=None) -> "FitMetrics":
        y = np.asarray(y, dtype=float)
        yHat = np.asarray(yHat, dtype=float)
        n = y.size
        if n == 0:                     # an output without observations (heterotopic multi-output data)
            nan = float("nan")
            return cls(nSamples=0, nParams=float(nParams), sse=nan, rmse=nan, mae=nan, maxAbsError=nan,
                       rSquared=nan, adjRSquared=nan, logLikelihood=nan, aic=nan, aicc=nan, bic=nan)
        w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
        r = y - yHat
        sse = float(np.sum(w * r * r))
        yBar = float(np.sum(w * y) / np.sum(w))
        sst = float(np.sum(w * (y - yBar) ** 2))
        if sst > 0.0:
            rSquared = 1.0 - sse / sst
        else:
            rSquared = 1.0 if sse == 0.0 else 0.0
        dofResid = n - nParams
        adjRSquared = 1.0 - (1.0 - rSquared) * (n - 1) / dofResid if dofResid > 0 else float("nan")
        sigma2 = max(sse / n, 1e-300)
        logLikelihood = float(0.5 * np.sum(np.log(w)) - 0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0))
        k = float(nParams) + 1.0
        aic = 2.0 * k - 2.0 * logLikelihood
        aicc = aic + 2.0 * k * (k + 1.0) / (n - k - 1.0) if n - k - 1.0 > 0 else float("inf")
        bic = k * np.log(n) - 2.0 * logLikelihood
        return cls(nSamples=int(n), nParams=float(nParams), sse=sse,
                   rmse=float(np.sqrt(np.mean(r * r))), mae=float(np.mean(np.abs(r))),
                   maxAbsError=float(np.max(np.abs(r))), rSquared=float(rSquared),
                   adjRSquared=float(adjRSquared), logLikelihood=logLikelihood,
                   aic=float(aic), aicc=float(aicc), bic=float(bic))

    def toDict(self) -> dict:
        return asdict(self)

    @classmethod
    def fromDict(cls, d: dict) -> "FitMetrics":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})
