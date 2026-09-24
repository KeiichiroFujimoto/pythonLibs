"""Bootstrap uncertainty for any model (including ones without analytic intervals).

Methods:
    residual  y* = y_hat + resampled, dof-inflated centered residuals (fixed design)
    wild      y* = y_hat + r_i v_i with Rademacher v_i (robust to heteroscedasticity)
    pairs     resample (x_i, y_i) rows (random design, least model dependent)

Returns percentile intervals for predictions at ``xNew`` and, when the model
exposes parameter estimates, for the parameters. Replicates run in threads.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.core.InputValidation import asFeatureMatrix, asOutputMatrix, asWeights, toJsonable


@dataclass
class BootstrapResult:
    """Percentile bootstrap summary.

    Attributes:
        mean, std, lower, upper: (m, ny) statistics of the replicated predictions at xNew
        samples:     (nOk, m, ny) replicated predictions
        paramNames:  parameter names (or None)
        paramSamples, paramLower, paramUpper, paramStd: parameter statistics (or None)
        level, method, nBoot, nFailed
    """
    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    samples: np.ndarray
    paramNames: Optional[list]
    paramSamples: Optional[np.ndarray]
    paramLower: Optional[np.ndarray]
    paramUpper: Optional[np.ndarray]
    paramStd: Optional[np.ndarray]
    level: float
    method: str
    nBoot: int
    nFailed: int

    def toDict(self, includeSamples: bool = False) -> dict:
        d = {"mean": self.mean, "std": self.std, "lower": self.lower, "upper": self.upper, "level": self.level,
             "method": self.method, "nBoot": self.nBoot, "nFailed": self.nFailed}
        if self.paramSamples is not None:
            d["parameters"] = [{"name": n, "std": s, "lower": lo, "upper": hi} for n, s, lo, hi in
                               zip(self.paramNames, self.paramStd, self.paramLower, self.paramUpper)]
        if includeSamples:
            d["samples"] = self.samples
        return toJsonable(d)


def _parameterVector(model):
    res = getattr(model, "result", None)
    if res is None:
        return None, None
    names = [f"{o}.{p}" if res.params.shape[1] > 1 else p for o in res.outputNames for p in res.parameterNames]
    return names, res.params.T.ravel()


def bootstrap(model, x, y, xNew=None, weights=None, nBoot: int = 200, method: str = "residual",
              level: float = 0.95, seed: int = 0, nJobs: int = 1, includeNoise: bool = False) -> BootstrapResult:
    """Bootstrap a model configuration on (x, y).

    ``includeNoise=True`` adds a resampled residual to every replicated
    prediction, turning the band into a prediction (not confidence) band.
    """
    if method not in ("residual", "wild", "pairs"):
        raise ValueError("method must be residual, wild or pairs")
    xa = asFeatureMatrix(x)
    ya = asOutputMatrix(y, xa.shape[0])
    wa = None if weights is None else asWeights(weights, xa.shape[0])
    xq = xa if xNew is None else asFeatureMatrix(xNew, xa.shape[1])
    n = xa.shape[0]
    base = model.clone().fit(xa, ya, wa)
    fitted = base.predictValues(xa)
    edf = np.broadcast_to(np.asarray(base.nEffectiveParams, dtype=float), (ya.shape[1],))
    inflate = np.sqrt(n / np.maximum(n - edf, 1.0))
    resid = (ya - fitted) * inflate
    resid = resid - resid.mean(axis=0)
    names, _ = _parameterVector(base)
    rootSeq = np.random.SeedSequence(seed)
    childSeeds = rootSeq.spawn(nBoot)

    def replicate(b):
        rng = np.random.default_rng(childSeeds[b])
        try:
            if method == "pairs":
                idx = rng.integers(0, n, n)
                if np.unique(xa[idx], axis=0).shape[0] < min(n, 3):
                    return None
                m = model.clone().fit(xa[idx], ya[idx], None if wa is None else wa[idx])
            else:
                if method == "residual":
                    noise = resid[rng.integers(0, n, n)]
                else:
                    noise = resid * rng.choice([-1.0, 1.0], size=(n, 1))
                m = model.clone().fit(xa, fitted + noise, wa)
            pred = m.predictValues(xq)
            if includeNoise:
                pred = pred + resid[rng.integers(0, n, xq.shape[0])]
            _, params = _parameterVector(m)
            return pred, params
        except Exception:
            return None

    if nJobs > 1:
        with ThreadPoolExecutor(max_workers=nJobs) as pool:
            reps = list(pool.map(replicate, range(nBoot)))
    else:
        reps = [replicate(b) for b in range(nBoot)]
    ok = [r for r in reps if r is not None]
    if len(ok) < 2:
        raise RuntimeError("fewer than two bootstrap replicates succeeded")
    samples = np.stack([r[0] for r in ok])
    a = 0.5 * (1.0 - level)
    lower, upper = np.quantile(samples, [a, 1.0 - a], axis=0)
    paramSamples = paramLower = paramUpper = paramStd = None
    if names is not None and all(r[1] is not None and r[1].size == len(names) for r in ok):
        paramSamples = np.stack([r[1] for r in ok])
        paramLower, paramUpper = np.quantile(paramSamples, [a, 1.0 - a], axis=0)
        paramStd = paramSamples.std(axis=0, ddof=1)
    return BootstrapResult(mean=samples.mean(axis=0), std=samples.std(axis=0, ddof=1), lower=lower, upper=upper,
                           samples=samples, paramNames=names if paramSamples is not None else None,
                           paramSamples=paramSamples, paramLower=paramLower, paramUpper=paramUpper,
                           paramStd=paramStd, level=level, method=method, nBoot=nBoot, nFailed=nBoot - len(ok))
