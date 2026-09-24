"""Multi-fidelity Kriging (recursive autoregressive co-Kriging).

Fidelity levels 0 (cheapest) ... L-1 (target) are linked by

    y_0(x) = Z_0(x)
    y_l(x) = rho_l(x) y_(l-1)(x) + delta_l(x),   l >= 1

where every delta_l is a universal-Kriging process and rho_l is a constant or
a linear function of x. Following the recursive formulation, level l is a
``KrigingModel`` whose trend contains the level-(l-1) prediction mean as a
covariate, so rho_l is estimated jointly with the trend by generalized least
squares and the designs do not have to be nested. Predictions propagate
recursively::

    mean_l(x) = rho_l(x) mean_(l-1)(x) + delta-mean_l(x)
    var_l(x)  = rho_l(x)^2 var_(l-1)(x) + delta-var_l(x)

Inputs carry the fidelity level in ``levelColumn`` (default: last column);
prediction evaluates the level given in that column (e.g. the top level).
``fitLevels([x0, x1, ...], [y0, y1, ...])`` builds that layout.

References:
    Kennedy & O'Hagan (2000) Biometrika 87(1).
    Le Gratiet & Garnier (2014) Int. J. Uncertainty Quantification 4(5).
"""
from __future__ import annotations

import copy

import numpy as np

from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase
from pythonLibs.regressionHandler.models.KrigingModel import KrigingModel


@registry("model").register("multiFidelity")
class MultiFidelityKrigingModel(SurrogateModelBase):
    """Recursive multi-fidelity Kriging; the fidelity level is an input column."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("corr", "matern52", types=(str, dict, object), desc="Kernel of every level (or see 'levelOptions')")
        d("poly", "constant", values=("none", "constant", "linear", "quadratic"), desc="Trend of every level")
        d("rho", "constant", values=("constant", "linear"), desc="Scaling rho_l(x) between levels")
        d("levelColumn", -1, types=int, desc="Input column holding the integer fidelity level")
        d("kriging", {}, types=dict, desc="Extra KrigingModel options for every level (nugget, likelihood ...)")
        d("levelOptions", None, types=list, desc="Optional per-level KrigingModel option dicts")
        self.supports.update(variances=True, weights=False)

    # ------------------------------------------------------------------ layout helpers
    def _lc(self) -> int:
        return self.options["levelColumn"] % self.nx

    def _space(self, x):
        lc = self._lc()
        return x[:, [j for j in range(self.nx) if j != lc]], x[:, lc]

    def fitLevels(self, xs, ys) -> "MultiFidelityKrigingModel":
        """Fit from per-level data lists (lowest fidelity first)."""
        rows, vals = [], []
        for level, (xl, yl) in enumerate(zip(xs, ys)):
            xl = np.asarray(xl, dtype=float)
            xl = xl[:, None] if xl.ndim == 1 else xl
            rows.append(np.column_stack([xl, np.full(xl.shape[0], float(level))]))
            vals.append(np.asarray(yl, dtype=float).ravel())
        self.options["levelColumn"] = -1
        return self.fit(np.vstack(rows), np.concatenate(vals))

    def _levelModel(self, level: int, nSpace: int) -> KrigingModel:
        opts = {"corr": copy.deepcopy(self.options["corr"]), "poly": self.options["poly"]}
        opts.update(copy.deepcopy(self.options["kriging"]))
        per = self.options["levelOptions"]
        if per is not None and level < len(per) and per[level]:
            opts.update(copy.deepcopy(per[level]))
        if level > 0:
            opts["spatialColumns"] = list(range(nSpace))
        return KrigingModel(**opts)

    def _augment(self, xs, prevMean) -> np.ndarray:
        cols = [xs, prevMean[:, None]]
        if self.options["rho"] == "linear":
            cols.append(xs * prevMean[:, None])
        return np.hstack(cols)

    # ------------------------------------------------------------------ training
    def _train(self) -> None:
        xs, lv = self._space(self.xt)
        y = self.yt[:, 0]
        levels = np.unique(lv)
        if not np.allclose(levels, np.arange(levels.size)):
            raise ValueError("fidelity levels must be the integers 0, 1, ..., L-1")
        self._nSpace = xs.shape[1]
        self._models = []
        for level in range(levels.size):
            sel = lv == level
            m = self._levelModel(level, self._nSpace)
            if level == 0:
                m.fit(xs[sel], y[sel])
            else:
                prev, _ = self._recursive(xs[sel], level - 1, kind="confidence")
                m.fit(self._augment(xs[sel], prev), y[sel])
            self._models.append(m)

    def _recursive(self, xs, level: int, kind: str):
        """(mean, variance) of level ``level`` at spatial inputs xs."""
        m0 = self._models[0]
        mean = m0.predictValues(xs)[:, 0]
        var = m0.predictVariances(xs, "confidence" if level > 0 else kind)[:, 0]
        for l in range(1, level + 1):
            m = self._models[l]
            aug = self._augment(xs, mean)
            rho = self._rho(m, aug)
            newMean = m.predictValues(aug)[:, 0]
            newVar = m.predictVariances(aug, kind if l == level else "confidence")[:, 0] + rho ** 2 * var
            mean, var = newMean, newVar
        return mean, var

    def _rho(self, m: KrigingModel, aug) -> np.ndarray:
        """d mean_l / d mean_(l-1): the trend coefficient(s) of the previous-level covariate."""
        g = m.predictDerivatives(aug, self._nSpace)[:, 0]
        if self.options["rho"] == "linear":
            # aug = [x, prev, x*prev]: d/dprev = beta_prev + sum_j beta_j x_j
            for j in range(self._nSpace):
                g = g + m.predictDerivatives(aug, self._nSpace + 1 + j)[:, 0] * aug[:, j]
        return g

    # ------------------------------------------------------------------ prediction
    def _levels(self, x) -> tuple[np.ndarray, np.ndarray]:
        xs, lv = self._space(x)
        lv = np.rint(lv).astype(int)
        if np.any(lv < 0) or np.any(lv >= len(self._models)):
            raise ValueError(f"fidelity level must be in 0..{len(self._models) - 1}")
        return xs, lv

    def _evaluate(self, x, kind):
        xs, lv = self._levels(x)
        mean, var = np.empty(x.shape[0]), np.empty(x.shape[0])
        for level in np.unique(lv):
            sel = lv == level
            mean[sel], var[sel] = self._recursive(xs[sel], int(level), kind)
        return mean, var

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        return self._evaluate(x, "confidence")[0][:, None]

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        return np.maximum(self._evaluate(x, kind)[1], 0.0)[:, None]

    def predictTop(self, xSpace, kind: str = "confidence") -> tuple[np.ndarray, np.ndarray]:
        """(mean, variance) of the highest fidelity at spatial inputs (no level column)."""
        self._checkTrained()
        xs = np.asarray(xSpace, dtype=float)
        xs = xs[:, None] if xs.ndim == 1 else xs
        return self._recursive(xs, len(self._models) - 1, kind)

    def _effectiveParams(self):
        return float(sum(m.nEffectiveParams for m in self._models))

    @property
    def hyperparameters(self) -> dict:
        self._checkTrained()
        out = []
        for level, m in enumerate(self._models):
            h = m.hyperparameters
            entry = {"level": level, "logParams": h["logParams"], "processVariance": h["processVariance"],
                     "noiseVariance": h["noiseVariance"]}
            if level > 0:
                entry["trendCoefficients"] = h["trendCoefficients"]
            out.append(entry)
        return {"levels": out}

    # ------------------------------------------------------------------ persistence
    def _stateToDict(self) -> dict:
        return {"models": [m.toDict() for m in self._models], "nSpace": self._nSpace}

    def _stateFromDict(self, state: dict) -> None:
        self._models = [SurrogateModelBase.fromDict(d) for d in state["models"]]
        self._nSpace = int(state["nSpace"])
