"""Out-of-sample error estimation: k-fold and leave-one-out cross-validation.

``crossValidate`` refits a clone of the model on each training fold and
collects out-of-fold predictions. Folds can run in threads (numpy releases
the GIL inside BLAS/LAPACK, so large folds parallelize well). For linear
models that report leverages, ``method="analytic"`` returns exact LOO
residuals e_i / (1 - h_ii) from a single fit (PRESS statistic).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np

from pythonLibs.regressionHandler.core.InputValidation import asFeatureMatrix, asOutputMatrix, asWeights


def regressionMetrics(yTrue, yPred) -> dict:
    """Plain error metrics per output: rmse, mae, maxAbsError, rSquared, nrmse."""
    yt = np.asarray(yTrue, dtype=float)
    yp = np.asarray(yPred, dtype=float)
    if yt.ndim == 1:
        yt, yp = yt[:, None], yp.reshape(-1, 1)
    r = yt - yp
    sst = np.sum((yt - yt.mean(axis=0)) ** 2, axis=0)
    sse = np.sum(r * r, axis=0)
    span = np.ptp(yt, axis=0)
    rmse = np.sqrt(np.mean(r * r, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "rmse": rmse, "mae": np.mean(np.abs(r), axis=0), "maxAbsError": np.max(np.abs(r), axis=0),
            "rSquared": np.where(sst > 0, 1.0 - sse / sst, np.where(sse == 0, 1.0, 0.0)),
            "nrmse": np.where(span > 0, rmse / span, np.nan),
        }


def kFoldIndices(n: int, nFolds: int = 5, shuffle: bool = True, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train, test) index pairs; ``nFolds = n`` gives leave-one-out."""
    if not 2 <= nFolds <= n:
        raise ValueError(f"nFolds must be in [2, {n}]")
    order = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
    folds = np.array_split(order, nFolds)
    return [(np.sort(np.concatenate(folds[:k] + folds[k + 1:])), np.sort(folds[k])) for k in range(nFolds)]


@dataclass
class CvResult:
    """Cross-validation outcome.

    Attributes:
        predictions: (n, ny) out-of-fold predictions
        residuals:   (n, ny) out-of-fold residuals
        foldRmse:    (nFolds, ny) RMSE of each fold
        rmse, mae, maxAbsError, q2: (ny,) pooled statistics (q2 = predictive R^2)
        nFolds:      number of folds (n for LOO)
        method:      "refit" or "analytic"
        failedFolds: folds whose fit raised (their points are NaN)
    """
    predictions: np.ndarray
    residuals: np.ndarray
    foldRmse: np.ndarray
    rmse: np.ndarray
    mae: np.ndarray
    maxAbsError: np.ndarray
    q2: np.ndarray
    nFolds: int
    method: str
    failedFolds: list

    @property
    def rmseStdError(self) -> np.ndarray:
        """Standard error of the mean fold RMSE (for the one-standard-error rule)."""
        f = self.foldRmse[np.all(np.isfinite(self.foldRmse), axis=1)]
        if f.shape[0] < 2:
            return np.full(self.rmse.shape, np.nan)
        return f.std(axis=0, ddof=1) / np.sqrt(f.shape[0])

    def toDict(self) -> dict:
        """Pooled statistics and fold RMSEs as JSON-ready values (predictions and residuals are left out)."""
        return {"rmse": self.rmse.tolist(), "mae": self.mae.tolist(), "maxAbsError": self.maxAbsError.tolist(),
                "q2": self.q2.tolist(), "foldRmse": self.foldRmse.tolist(),
                "rmseStdError": self.rmseStdError.tolist(), "nFolds": self.nFolds, "method": self.method,
                "failedFolds": self.failedFolds}


def crossValidate(model, x, y, weights=None, nFolds: int = 5, shuffle: bool = True, seed: int = 0,
                  method: str = "refit", nJobs: int = 1) -> CvResult:
    """Cross-validate an (untrained or trained) model; the model itself is not modified.

    Args:
        model:   any SurrogateModelBase (a clone is fitted per fold)
        nFolds:  number of folds; ``nFolds >= n`` means leave-one-out
        method:  "refit" (always valid) or "analytic" (exact LOO for linear models with leverages)
        nJobs:   worker threads for the folds
    """
    xArr = asFeatureMatrix(x)
    yArr = asOutputMatrix(y, xArr.shape[0])
    n = xArr.shape[0]
    wArr = None if weights is None else asWeights(weights, n)
    if method == "analytic":
        return _analyticLoo(model, xArr, yArr, wArr)
    if method != "refit":
        raise ValueError("method must be 'refit' or 'analytic'")
    splits = kFoldIndices(n, min(nFolds, n), shuffle, seed)

    def runFold(split):
        train, test = split
        m = model.clone()
        m.fit(xArr[train], yArr[train], None if wArr is None else wArr[train])
        return m.predictValues(xArr[test])

    preds = np.full(yArr.shape, np.nan)
    failed = []
    if nJobs > 1:
        with ThreadPoolExecutor(max_workers=nJobs) as pool:
            outcomes = list(pool.map(lambda s: _safe(runFold, s), splits))
    else:
        outcomes = [_safe(runFold, s) for s in splits]
    foldRmse = np.full((len(splits), yArr.shape[1]), np.nan)
    for k, ((_, test), (value, error)) in enumerate(zip(splits, outcomes)):
        if error is not None:
            failed.append({"fold": k, "error": error})
            continue
        preds[test] = value
        foldRmse[k] = np.sqrt(np.mean((yArr[test] - value) ** 2, axis=0))
    return _summarize(yArr, preds, foldRmse, len(splits), "refit", failed)


def _safe(fn, arg):
    try:
        return fn(arg), None
    except Exception as exc:  # a failing fold must not abort the whole CV
        return None, f"{type(exc).__name__}: {exc}"


def _analyticLoo(model, x, y, w) -> CvResult:
    # a trained model is reused only if it was fitted to exactly this data (inputs, outputs and weights)
    sameData = getattr(model, "isTrained", False) and model.xt is not None and \
        model.xt.shape == x.shape and np.array_equal(model.xt, x) and \
        model.yt.shape == y.shape and np.array_equal(model.yt, y) and \
        np.array_equal(model.wt, np.ones(x.shape[0]) if w is None else w)
    m = model if sameData else model.clone().fit(x, y, w)
    loo = getattr(m, "looResiduals", None)
    if loo is None:
        raise NotImplementedError(f"{type(m).__name__} has no analytic LOO; use method='refit'")
    r = loo()
    preds = y - r
    return _summarize(y, preds, np.abs(r), y.shape[0], "analytic", [])


def _summarize(y, preds, foldRmse, nFolds, method, failed) -> CvResult:
    ok = np.all(np.isfinite(preds), axis=1)
    yy, pp = y[ok], preds[ok]
    r = y - preds
    stats = regressionMetrics(yy, pp) if ok.any() else None
    sst = np.sum((yy - yy.mean(axis=0)) ** 2, axis=0)
    press = np.sum((yy - pp) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        q2 = np.where(sst > 0, 1.0 - press / sst, np.nan)
    nan = np.full(y.shape[1], np.nan)
    return CvResult(predictions=preds, residuals=r, foldRmse=foldRmse,
                    rmse=stats["rmse"] if stats else nan, mae=stats["mae"] if stats else nan,
                    maxAbsError=stats["maxAbsError"] if stats else nan, q2=q2, nFolds=nFolds,
                    method=method, failedFolds=failed)
