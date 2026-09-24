"""Base class of every regression / surrogate model.

Template-method API with declared ``options``, capability flags in
``supports`` and ``xt (nt, nx)`` / ``yt (nt, ny)`` training arrays, plus
sample weights, confidence/prediction intervals, parameter inference
(``result``), fit metrics, JSON persistence and ``clone``.

Subclass contract (all arrays already validated, 2D)::

    _initialize()                      declare options, set supports flags
    _train()                           fit on self.xt, self.yt, self.wt
    _predictValues(x)          -> (n, ny)
    _predictVariances(x, kind) -> (n, ny)   if supports["variances"]
    _predictDerivatives(x, kx) -> (n, ny)   if supports["derivatives"]
    _effectiveParams()         -> float or (ny,)
    _intervalDof()             -> float or None (None: normal quantiles)
    _stateToDict() / _stateFromDict(state)

Models that do not support several outputs natively (``supports
["multiOutput"] = False``) are trained transparently as one sub-model per
output column, so every model accepts ``yt`` of shape (nt, ny).
"""
from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np

from pythonLibs.regressionHandler.core.FitMetrics import FitMetrics
from pythonLibs.regressionHandler.core.FitResult import FitResult
from pythonLibs.regressionHandler.core.InputValidation import asFeatureMatrix, asOutputMatrix, asWeights
from pythonLibs.regressionHandler.core.OptionsDictionary import OptionsDictionary
from pythonLibs.regressionHandler.core.Registry import _optionsFromDict, _optionsToDict, registry
from pythonLibs.regressionHandler.numerics.Distributions import Normal, StudentT

FORMAT_VERSION = 1
INTERVAL_KINDS = ("confidence", "prediction")


@dataclass(frozen=True)
class PredictionInterval:
    """Mean prediction with a symmetric uncertainty band, arrays of shape (n, ny).

    ``confidence`` bounds the mean response; ``prediction`` also includes the
    observation noise of a new sample with unit weight.
    """
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    std: np.ndarray
    level: float
    kind: str

    def toDict(self) -> dict:
        return {"mean": self.mean.tolist(), "lower": self.lower.tolist(), "upper": self.upper.tolist(),
                "std": self.std.tolist(), "level": self.level, "kind": self.kind}


class SurrogateModelBase(ABC):

    registryName: ClassVar[str] = ""

    def __init__(self, **options) -> None:
        self.options = OptionsDictionary()
        self.supports = {
            "multiOutput": False,       # trains all outputs jointly
            "weights": True,            # accepts sample weights
            "variances": False,         # analytic predictive variances / intervals
            "derivatives": False,       # analytic dy/dx (else finite differences)
            "parameterInference": False,  # FitResult with standard errors
            "covariance": False,        # joint posterior covariance (conditional simulation)
        }
        self._initialize()
        self.options.update(options)
        self._validateOptions()
        self.xt: Optional[np.ndarray] = None
        self.yt: Optional[np.ndarray] = None
        self.wt: Optional[np.ndarray] = None
        self.nx: Optional[int] = None
        self.ny: Optional[int] = None
        self.featureNames: list[str] = []
        self.outputNames: list[str] = []
        self.metrics: list[FitMetrics] = []
        self.result: Optional[FitResult] = None
        self._subModels: Optional[list["SurrogateModelBase"]] = None
        self._trained = False
        self._xRange: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ hooks
    @abstractmethod
    def _initialize(self) -> None:
        """Declare options with ``self.options.declare`` and set ``self.supports``."""

    @abstractmethod
    def _train(self) -> None:
        """Fit on ``self.xt`` (nt, nx), ``self.yt`` (nt, ny), ``self.wt`` (nt,)."""

    @abstractmethod
    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        """Mean prediction (n, ny)."""

    def _validateOptions(self) -> None:
        """Cross-check options after construction (raise early on bad component specs)."""

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} does not provide predictive variances")

    def _predictDerivatives(self, x: np.ndarray, kx: int) -> np.ndarray:
        raise NotImplementedError

    def _effectiveParams(self):
        """Effective number of mean-function parameters (float or per-output array)."""
        raise NotImplementedError

    def _intervalDof(self) -> Optional[float]:
        return None

    def _stateToDict(self) -> dict:
        raise NotImplementedError(f"{type(self).__name__} cannot be serialized")

    def _stateFromDict(self, state: dict) -> None:
        raise NotImplementedError(f"{type(self).__name__} cannot be deserialized")

    # ------------------------------------------------------------------ training
    def setTrainingValues(self, xt, yt, weights=None, featureNames=None, outputNames=None) -> None:
        """Store training data: xt (nt, nx) or (nt,), yt (nt, ny) or (nt,)."""
        x = asFeatureMatrix(xt, name="xt")
        y = asOutputMatrix(yt, x.shape[0], name="yt")
        w = asWeights(weights, x.shape[0])
        if weights is not None and not self.supports["weights"]:
            raise ValueError(f"{type(self).__name__} does not support sample weights")
        self.xt, self.yt, self.wt = x, y, w
        self.nx, self.ny = x.shape[1], y.shape[1]
        self.featureNames = list(featureNames) if featureNames else [f"x{i}" for i in range(self.nx)]
        self.outputNames = list(outputNames) if outputNames else [f"y{j}" for j in range(self.ny)]
        if len(self.featureNames) != self.nx or len(self.outputNames) != self.ny:
            raise ValueError("featureNames / outputNames do not match the data dimensions")
        self._trained = False

    def train(self) -> "SurrogateModelBase":
        """Fit the model to the stored training values. Returns self."""
        if self.xt is None:
            raise RuntimeError("call setTrainingValues() before train()")
        self._xRange = np.ptp(self.xt, axis=0)
        self.result = None
        if self.ny > 1 and not self.supports["multiOutput"]:
            self._subModels = []
            for j in range(self.ny):
                sub = self._spawn()
                sub.setTrainingValues(self.xt, self.yt[:, j], None if np.all(self.wt == 1.0) else self.wt,
                                      self.featureNames, [self.outputNames[j]])
                sub.train()
                self._subModels.append(sub)
            self.result = self._combineSubResults()
        else:
            self._subModels = None
            self._train()
        self._trained = True
        yHat = self._predictAll(self.xt)
        nEff = np.broadcast_to(np.asarray(self.nEffectiveParams, dtype=float), (self.ny,))
        self.metrics = [FitMetrics.compute(self.yt[:, j], yHat[:, j], nEff[j], self.wt) for j in range(self.ny)]
        return self

    def fit(self, xt, yt, weights=None, featureNames=None, outputNames=None) -> "SurrogateModelBase":
        """Shortcut for ``setTrainingValues`` + ``train``."""
        self.setTrainingValues(xt, yt, weights, featureNames, outputNames)
        return self.train()

    def _spawn(self) -> "SurrogateModelBase":
        return type(self)(**copy.deepcopy(self.options.toDict()))

    def _combineSubResults(self) -> Optional[FitResult]:
        results = [m.result for m in self._subModels]
        if any(r is None for r in results):
            return None
        names = results[0].parameterNames
        if any(r.parameterNames != names for r in results):
            return None
        return FitResult(parameterNames=names, params=np.hstack([r.params for r in results]),
                         covariance=np.concatenate([r.covariance for r in results]),
                         dofResid=min(r.dofResid for r in results),
                         sigma2=np.concatenate([r.sigma2 for r in results]), outputNames=self.outputNames)

    # ------------------------------------------------------------------ prediction
    @property
    def isTrained(self) -> bool:
        return self._trained

    def _checkTrained(self) -> None:
        if not self._trained:
            raise RuntimeError(f"{type(self).__name__} is not trained")

    def _validX(self, x) -> np.ndarray:
        return asFeatureMatrix(x, self.nx)

    def _predictAll(self, x: np.ndarray) -> np.ndarray:
        if self._subModels is not None:
            return np.hstack([m._predictAll(x) for m in self._subModels])
        return self._predictValues(x)

    def predictValues(self, x) -> np.ndarray:
        """Mean prediction, shape (n, ny)."""
        self._checkTrained()
        return self._predictAll(self._validX(x))

    def predict(self, x) -> np.ndarray:
        """Like ``predictValues`` but returns (n,) for single-output models."""
        y = self.predictValues(x)
        return y[:, 0] if self.ny == 1 else y

    def predictVariances(self, x, kind: str = "confidence") -> np.ndarray:
        """Predictive variance (n, ny) of the mean (``confidence``) or of a new observation."""
        self._checkTrained()
        if kind not in INTERVAL_KINDS:
            raise ValueError(f"kind must be one of {INTERVAL_KINDS}")
        if not self.supports["variances"]:
            raise NotImplementedError(f"{type(self).__name__} does not provide predictive variances; "
                                      "use evaluation.Bootstrap for resampling intervals")
        xv = self._validX(x)
        if self._subModels is not None:
            return np.hstack([m._predictVariances(xv, kind) for m in self._subModels])
        return self._predictVariances(xv, kind)

    def predictDerivatives(self, x, kx: int) -> np.ndarray:
        """dy/dx_kx (n, ny); analytic when supported, else central differences."""
        self._checkTrained()
        if not 0 <= kx < self.nx:
            raise ValueError(f"kx must be in [0, {self.nx})")
        xv = self._validX(x)
        if self.supports["derivatives"]:
            if self._subModels is not None:
                return np.hstack([m._predictDerivatives(xv, kx) for m in self._subModels])
            return self._predictDerivatives(xv, kx)
        span = self._xRange[kx] if self._xRange is not None and self._xRange[kx] > 0 else 1.0
        h = 1e-5 * max(span, 1e-12)
        xp, xm = xv.copy(), xv.copy()
        xp[:, kx] += h
        xm[:, kx] -= h
        return (self._predictAll(xp) - self._predictAll(xm)) / (2.0 * h)

    def _predictCovariance(self, x: np.ndarray, kind: str) -> np.ndarray:
        raise NotImplementedError

    def predictCovariance(self, x, kind: str = "confidence") -> np.ndarray:
        """Joint posterior covariance of the predictions, shape (ny, m, m)."""
        self._checkTrained()
        if kind not in INTERVAL_KINDS:
            raise ValueError(f"kind must be one of {INTERVAL_KINDS}")
        if not self.supports.get("covariance", False):
            raise NotImplementedError(f"{type(self).__name__} does not provide a joint predictive covariance")
        xv = self._validX(x)
        if self._subModels is not None:
            return np.concatenate([m.predictCovariance(xv, kind) for m in self._subModels])
        cov = self._predictCovariance(xv, kind)
        return cov[None] if cov.ndim == 2 else cov

    def simulate(self, x, nSamples: int = 1, seed: Optional[int] = 0, kind: str = "confidence") -> np.ndarray:
        """Conditional simulation: draws (nSamples, m, ny) from the joint posterior at x.

        ``confidence`` draws the latent mean surface, ``prediction`` adds
        independent observation noise.
        """
        mean = self.predictValues(x)
        cov = self.predictCovariance(x, kind)
        rng = np.random.default_rng(seed)
        out = np.empty((nSamples,) + mean.shape)
        for j in range(mean.shape[1]):
            c = cov[j]
            scale = max(float(np.max(np.diag(c))), 1e-300)
            jitter = 0.0
            for _ in range(12):
                try:
                    lower = np.linalg.cholesky(c + jitter * np.eye(c.shape[0]))
                    break
                except np.linalg.LinAlgError:
                    jitter = scale * 1e-12 if jitter == 0.0 else jitter * 10.0
            else:
                w, v = np.linalg.eigh(c)
                lower = v * np.sqrt(np.maximum(w, 0.0))
            out[:, :, j] = mean[:, j] + rng.standard_normal((nSamples, c.shape[0])) @ lower.T
        return out

    def predictGradient(self, x) -> np.ndarray:
        """(n, nx, ny) gradient of every output."""
        return np.stack([self.predictDerivatives(x, k) for k in range(self.nx)], axis=1)

    def predictInterval(self, x, level: float = 0.95, kind: str = "prediction") -> PredictionInterval:
        """Mean with a two-sided ``level`` band (t quantiles when a residual dof exists)."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        mean = self.predictValues(x)
        std = np.sqrt(np.maximum(self.predictVariances(x, kind), 0.0))
        dof = self.intervalDof
        dist = StudentT(dof) if dof is not None and np.isfinite(dof) and dof > 0 else Normal()
        q = float(dist.ppf(0.5 + level / 2.0))
        return PredictionInterval(mean, mean - q * std, mean + q * std, std, float(level), kind)

    @property
    def intervalDof(self) -> Optional[float]:
        if self._subModels is not None:
            dofs = [m.intervalDof for m in self._subModels]
            return None if any(d is None for d in dofs) else min(dofs)
        return self._intervalDof()

    @property
    def nEffectiveParams(self):
        if self._subModels is not None:
            return np.array([float(np.ravel(m.nEffectiveParams)[0]) for m in self._subModels])
        return self._effectiveParams()

    # ------------------------------------------------------------------ utilities
    def clone(self) -> "SurrogateModelBase":
        """Untrained copy with identical options."""
        return self._spawn()

    def describe(self) -> str:
        opts = ", ".join(f"{k}={_short(v)}" for k, v in self.options.nonDefault().items())
        return f"{self.registryName or type(self).__name__}({opts})"

    def summary(self) -> str:
        self._checkTrained()
        lines = [f"Model: {self.describe()}", f"Training points: {self.xt.shape[0]}, inputs: {self.nx}, "
                 f"outputs: {self.ny}"]
        for name, m in zip(self.outputNames, self.metrics):
            lines.append(f"  {name}: R2={m.rSquared:.6g}  adjR2={m.adjRSquared:.6g}  RMSE={m.rmse:.6g}  "
                         f"nEff={m.nParams:.4g}  AICc={m.aicc:.6g}  BIC={m.bic:.6g}")
        if self.result is not None:
            lines.append("")
            lines.append(self.result.summary())
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<{self.describe()} ({'trained' if self._trained else 'untrained'})>"

    # ------------------------------------------------------------------ persistence
    def toDict(self, includeTrainingData: bool = True) -> dict:
        d = {"format": "regressionHandler.model", "version": FORMAT_VERSION, "type": self.registryName,
             "options": _optionsToDict(self.options.toDict())}
        if self._trained:
            state = {"nx": self.nx, "ny": self.ny, "featureNames": self.featureNames,
                     "outputNames": self.outputNames, "metrics": [m.toDict() for m in self.metrics],
                     "xRange": self._xRange.tolist(),
                     "result": self.result.toDict() if self.result is not None else None}
            if self._subModels is not None:
                state["subModels"] = [m.toDict(includeTrainingData) for m in self._subModels]
            else:
                state["model"] = self._stateToDict()
            if includeTrainingData:
                state["training"] = {"xt": self.xt.tolist(), "yt": self.yt.tolist(), "wt": self.wt.tolist()}
            d["state"] = state
        return d

    @classmethod
    def fromDict(cls, d: dict) -> "SurrogateModelBase":
        modelCls = registry("model").get(d["type"]) if cls is SurrogateModelBase else cls
        model = modelCls(**_optionsFromDict(d.get("options", {})))
        state = d.get("state")
        if state is not None:
            model.nx, model.ny = int(state["nx"]), int(state["ny"])
            model.featureNames, model.outputNames = list(state["featureNames"]), list(state["outputNames"])
            model.metrics = [FitMetrics.fromDict(m) for m in state["metrics"]]
            model._xRange = np.array(state["xRange"], dtype=float)
            model.result = FitResult.fromDict(state["result"]) if state.get("result") else None
            training = state.get("training")
            if training:
                model.xt = np.array(training["xt"], dtype=float)
                model.yt = np.array(training["yt"], dtype=float)
                model.wt = np.array(training["wt"], dtype=float)
            if "subModels" in state:
                model._subModels = [SurrogateModelBase.fromDict(s) for s in state["subModels"]]
            else:
                model._stateFromDict(state["model"])
            model._trained = True
        return model

    def save(self, filePath: str, includeTrainingData: bool = True) -> None:
        with open(filePath, "w", encoding="utf-8") as f:
            json.dump(self.toDict(includeTrainingData), f)

    @staticmethod
    def load(filePath: str) -> "SurrogateModelBase":
        with open(filePath, "r", encoding="utf-8") as f:
            return SurrogateModelBase.fromDict(json.load(f))


def _short(v) -> str:
    text = repr(v)
    return text if len(text) <= 40 else text[:37] + "..."
