"""Model comparison, hyperparameter search and stepwise term selection.

- ``ModelSelector``: fits every candidate spec, scores it by k-fold CV
  RMSE (default), AICc or BIC, ranks them and keeps the refitted winner.
  Several outputs are combined through RMSE normalized by each output's
  standard deviation. ``oneStandardError=True`` picks the simplest model
  whose CV error is within one standard error of the best (Breiman's rule).
- ``defaultCandidates``: a sensible candidate list for the data size and
  dimension (the handler's ``autoFit`` uses it).
- ``tuneHyperparameters``: grid search over (dotted) spec options by CV.
- ``stepwiseSelect``: forward / backward / bidirectional selection of the
  terms of any basis by AICc, BIC or exact LOO error.
"""
from __future__ import annotations

import copy
import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from pythonLibs.regressionHandler.bases.BasisBase import BasisBase
from pythonLibs.regressionHandler.core.InputValidation import asFeatureMatrix, asOutputMatrix, asWeights
from pythonLibs.regressionHandler.core.Registry import buildComponent, registry
from pythonLibs.regressionHandler.evaluation.CrossValidation import crossValidate
from pythonLibs.regressionHandler.models.ModelFactory import createModel
from pythonLibs.regressionHandler.models.LinearBasisModel import LinearBasisModel

CRITERIA = ("cv", "aicc", "bic")


def _specName(spec: Any) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return spec.get("name") or repr({k: v for k, v in spec.items() if k != "name"})
    return repr(spec)


@dataclass
class ModelScore:
    """Score of one candidate (per-output arrays have length ny)."""
    name: str
    spec: Any
    model: Any = None
    score: float = np.inf
    cvRmse: Optional[np.ndarray] = None
    cvRmseStdError: Optional[np.ndarray] = None
    cvNrmse: float = np.nan
    q2: Optional[np.ndarray] = None
    aicc: float = np.nan
    bic: float = np.nan
    rSquared: Optional[np.ndarray] = None
    nParams: float = np.nan
    seconds: float = np.nan
    error: Optional[str] = None

    def toDict(self) -> dict:
        def arr(v):
            return None if v is None else np.asarray(v).tolist()
        return {"name": self.name, "score": self.score, "cvRmse": arr(self.cvRmse),
                "cvRmseStdError": arr(self.cvRmseStdError), "cvNrmse": self.cvNrmse, "q2": arr(self.q2),
                "aicc": self.aicc, "bic": self.bic, "rSquared": arr(self.rSquared), "nParams": self.nParams,
                "seconds": self.seconds, "error": self.error,
                "model": None if self.model is None else self.model.describe()}


@dataclass
class SelectionResult:
    scores: list[ModelScore]
    criterion: str
    bestIndex: int
    oneStandardError: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> ModelScore:
        return self.scores[self.bestIndex]

    @property
    def bestModel(self):
        return self.best.model

    def table(self) -> list[dict]:
        return [s.toDict() for s in self.scores]

    def toDict(self) -> dict:
        return {"criterion": self.criterion, "best": self.best.name, "oneStandardError": self.oneStandardError,
                "ranking": self.table(), "notes": self.notes}

    def summary(self) -> str:
        lines = [f"Model comparison by {self.criterion} (best: {self.best.name})",
                 f"{'rank':>4} {'model':<28} {'score':>12} {'cvNRMSE':>9} {'AICc':>11} {'nEff':>7} {'s':>7}"]
        for i, s in enumerate(self.scores):
            mark = "*" if i == self.bestIndex else " "
            if s.error:
                lines.append(f"{i + 1:>4}{mark}{s.name[:28]:<28} failed: {s.error[:60]}")
                continue
            lines.append(f"{i + 1:>4}{mark}{s.name[:28]:<28} {s.score:>12.5g} {s.cvNrmse:>9.4g} {s.aicc:>11.5g} "
                         f"{s.nParams:>7.3g} {s.seconds:>7.2f}")
        return "\n".join(lines + self.notes)


class ModelSelector:

    def __init__(self, candidates, criterion: str = "cv", nFolds: int = 5, seed: int = 0,
                 oneStandardError: bool = False, nJobs: int = 1) -> None:
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}")
        if isinstance(candidates, dict):
            self.candidates = list(candidates.items())
        else:
            self.candidates = [(_specName(c), c) for c in candidates]
        if not self.candidates:
            raise ValueError("no candidate models given")
        self.criterion = criterion
        self.nFolds = nFolds
        self.seed = seed
        self.oneStandardError = oneStandardError
        self.nJobs = nJobs

    def run(self, x, y, weights=None, featureNames=None, outputNames=None) -> SelectionResult:
        xa = asFeatureMatrix(x)
        ya = asOutputMatrix(y, xa.shape[0])
        wa = None if weights is None else asWeights(weights, xa.shape[0])
        ySd = ya.std(axis=0)
        ySd[ySd == 0] = 1.0

        def evaluate(item) -> ModelScore:
            name, spec = item
            s = ModelScore(name=name, spec=spec)
            t0 = time.perf_counter()
            try:
                if isinstance(spec, dict):
                    spec = {k: v for k, v in copy.deepcopy(spec).items() if k != "name"}
                model = createModel(spec if hasattr(spec, "clone") else copy.deepcopy(spec))
                model.fit(xa, ya, wa, featureNames, outputNames)
                s.model = model
                s.rSquared = np.array([m.rSquared for m in model.metrics])
                s.aicc = float(sum(m.aicc for m in model.metrics))
                s.bic = float(sum(m.bic for m in model.metrics))
                s.nParams = float(np.mean(np.ravel(model.nEffectiveParams)))
                # CV is computed for every criterion so each ranking reports out-of-sample error.
                cv = crossValidate(model, xa, ya, wa, nFolds=min(self.nFolds, xa.shape[0]), seed=self.seed)
                if cv.failedFolds:
                    raise RuntimeError(f"{len(cv.failedFolds)} CV folds failed: {cv.failedFolds[0]['error']}")
                s.cvRmse, s.cvRmseStdError, s.q2 = cv.rmse, cv.rmseStdError, cv.q2
                s.cvNrmse = float(np.mean(cv.rmse / ySd))
                s.score = {"cv": s.cvNrmse, "aicc": s.aicc, "bic": s.bic}[self.criterion]
                if not np.isfinite(s.score):
                    s.score = np.inf
            except Exception as exc:  # a failing candidate must not stop the comparison
                s.error = f"{type(exc).__name__}: {exc}"
                s.score = np.inf
            s.seconds = time.perf_counter() - t0
            return s

        if self.nJobs > 1:
            with ThreadPoolExecutor(max_workers=self.nJobs) as pool:
                scores = list(pool.map(evaluate, self.candidates))
        else:
            scores = [evaluate(c) for c in self.candidates]
        scores.sort(key=lambda s: (s.score, s.nParams if np.isfinite(s.nParams) else np.inf))
        if not np.isfinite(scores[0].score):
            raise RuntimeError("every candidate failed: " + "; ".join(f"{s.name}: {s.error}" for s in scores[:3]))
        best = 0
        notes = []
        if self.oneStandardError and self.criterion == "cv":
            ref = scores[0]
            se = float(np.mean(ref.cvRmseStdError / ySd)) if ref.cvRmseStdError is not None else 0.0
            limit = ref.score + (se if np.isfinite(se) else 0.0)
            eligible = [i for i, s in enumerate(scores) if np.isfinite(s.score) and s.score <= limit]
            best = min(eligible, key=lambda i: scores[i].nParams)
            if best != 0:
                notes.append(f"one-standard-error rule chose {scores[best].name} over {ref.name}")
        return SelectionResult(scores=scores, criterion=self.criterion, bestIndex=best,
                               oneStandardError=self.oneStandardError, notes=notes)


def defaultCandidates(x, y) -> list:
    """Candidate specs suited to the data size and number of inputs."""
    xa = asFeatureMatrix(x)
    ya = asOutputMatrix(y, xa.shape[0])
    n, nx = xa.shape
    from math import comb
    cands: list = ["linear"]
    if n > 2 * comb(nx + 2, 2):
        cands.append("quadratic")
    if nx <= 4 and n > 2 * comb(nx + 3, 3):
        cands.append("poly3")
    if nx <= 6 and n > 2 * comb(nx + 2, 2):
        cands.append("ridge-poly3")
    elif nx >= 2 and n >= 10:
        cands.append("ridge-poly2")
    if n >= 8 and n <= 1500:
        cands.extend(["gp", "rbf-smooth"])
    if nx >= 6 and 8 <= n <= 1500:
        cands.append("kpls")
    if n > 1500:
        cands.append({"type": "rbf", "neighbors": 30, "name": "rbf-local"})
    if nx <= 2 and n >= 25:
        cands.append("pspline")
    if n >= 30 and nx <= 4:
        cands.append({"type": "loess", "span": 0.3, "degree": 1, "name": "loess"})
    if nx == 1 and ya.shape[1] == 1:
        xv, yv = xa[:, 0], ya[:, 0]
        cands.append("logistic")
        if np.all(xv > 0):
            cands.extend(["powerLaw", "logarithmic"])
        if np.all(yv > 0):
            cands.append("exponential")
        if np.all(xv != 0) and np.all(yv > 0):
            cands.append("inverseExponential")
    return cands


# ---------------------------------------------------------------- hyperparameter search
def _setDotted(spec: dict, path: str, value) -> None:
    keys = path.split(".")
    node = spec
    for k in keys[:-1]:
        child = node.get(k)
        if isinstance(child, str):
            child = {"type": child}
        elif child is None:
            child = {}
        node[k] = child
        node = child
    node[keys[-1]] = value


def tuneHyperparameters(spec, grid: dict, x, y, weights=None, nFolds: int = 5, seed: int = 0,
                        nJobs: int = 1) -> SelectionResult:
    """Grid search: ``grid`` maps (dotted) option paths to value lists, e.g.
    ``{"basis.degree": [2, 3, 4], "solver.alpha": [1e-4, 1e-2]}``."""
    from pythonLibs.regressionHandler.models.ModelFactory import specFromName
    # names resolve like createModel: shorthand, ModelLibrary form or registered type
    base = specFromName(spec) if isinstance(spec, str) else copy.deepcopy(spec)
    keys = list(grid)
    cands = []
    for values in itertools.product(*[grid[k] for k in keys]):
        s = copy.deepcopy(base)
        for k, v in zip(keys, values):
            _setDotted(s, k, v)
        label = ", ".join(f"{k}={v}" for k, v in zip(keys, values))
        cands.append((label, s))
    return ModelSelector(dict(cands), "cv", nFolds, seed, nJobs=nJobs).run(x, y, weights)


# ---------------------------------------------------------------- stepwise selection
@registry("basis").register("subset")
class SubsetBasis(BasisBase):
    """Selected columns of another basis (result of stepwise selection)."""

    def _declareOptions(self, declare) -> None:
        declare("basis", {"type": "polynomial", "degree": 2}, types=(str, dict, object), desc="Parent basis spec")
        declare("indices", [], types=list, desc="Kept column indices of the parent basis")

    def _parent(self) -> BasisBase:
        if not hasattr(self, "_p"):
            self._p = copy.deepcopy(buildComponent("basis", self.options["basis"]))
        return self._p

    def _fit(self, x):
        self._parent().fit(x)

    @property
    def _idx(self):
        return np.asarray(self.options["indices"], dtype=int)

    @property
    def nTerms(self) -> int:
        return int(self._idx.size)

    def transform(self, x):
        return self._parent().transform(x)[:, self._idx]

    @property
    def hasDerivative(self) -> bool:
        return self._parent().hasDerivative

    def derivative(self, x, kx):
        return self._parent().derivative(x, kx)[:, self._idx]

    @property
    def biasMask(self):
        return self._parent().biasMask[self._idx]

    def termNames(self, featureNames=None):
        names = self._parent().termNames(featureNames)
        return [names[i] for i in self._idx]

    def _stateToDict(self) -> dict:
        return {"parent": self._parent().toDict()} if self._parent().nx is not None else {}

    def _stateFromDict(self, state: dict) -> None:
        if "parent" in state:
            self._p = buildComponent("basis", state["parent"])
            self.nx = self._p.nx


@dataclass
class StepwiseResult:
    model: LinearBasisModel
    selected: list[int]
    termNames: list[str]
    history: list[dict]
    criterion: str

    def toDict(self) -> dict:
        return {"criterion": self.criterion, "selectedTerms": self.termNames, "selectedIndices": self.selected,
                "history": self.history, "model": self.model.describe()}


def _subsetScore(phi, y, w, cols, criterion) -> float:
    a = phi[:, cols] * np.sqrt(w)[:, None]
    yw = y * np.sqrt(w)[:, None]
    q, r = np.linalg.qr(a)
    if np.min(np.abs(np.diag(r))) < 1e-12 * max(1.0, np.max(np.abs(np.diag(r)))):
        return np.inf
    fitted = q @ (q.T @ yw)
    resid = yw - fitted
    n, p = a.shape
    if criterion == "loo":
        h = np.sum(q * q, axis=1)
        return float(np.mean((resid / np.maximum(1 - h, 1e-12)[:, None]) ** 2))
    sse = np.sum(resid * resid, axis=0)
    total = 0.0
    for s in sse:
        ll = -0.5 * n * (np.log(2 * np.pi * max(s / n, 1e-300)) + 1.0) + 0.5 * np.sum(np.log(w))
        k = p + 1.0
        if criterion == "bic":
            total += k * np.log(n) - 2 * ll
        else:
            total += 2 * k - 2 * ll + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf)
    return float(total)


def stepwiseSelect(x, y, basis=None, weights=None, criterion: str = "aicc", direction: str = "both",
                   maxTerms: Optional[int] = None, forceIntercept: bool = True, featureNames=None,
                   outputNames=None) -> StepwiseResult:
    """Select basis terms greedily; returns an OLS LinearBasisModel on the chosen subset."""
    if criterion not in ("aicc", "bic", "loo"):
        raise ValueError("criterion must be aicc, bic or loo")
    if direction not in ("forward", "backward", "both"):
        raise ValueError("direction must be forward, backward or both")
    xa = asFeatureMatrix(x)
    ya = asOutputMatrix(y, xa.shape[0])
    wa = asWeights(weights, xa.shape[0])
    spec = basis if basis is not None else {"type": "polynomial", "degree": 2}
    b = copy.deepcopy(buildComponent("basis", spec)).fit(xa)
    phi = b.transform(xa)
    p = phi.shape[1]
    names = b.termNames(list(featureNames) if featureNames else None)
    forced = [int(i) for i in np.flatnonzero(b.biasMask)] if forceIntercept else []
    maxTerms = min(maxTerms or p, xa.shape[0] - 2, p)
    current = list(range(p)) if direction == "backward" else list(forced)
    score = _subsetScore(phi, ya, wa, current, criterion) if current else np.inf
    history = [{"step": 0, "action": "start", "terms": [names[i] for i in current], "score": score}]
    improved = True
    step = 0
    while improved:
        improved = False
        best = (score, None, None)
        if direction in ("forward", "both") and len(current) < maxTerms:
            for j in range(p):
                if j not in current:
                    s = _subsetScore(phi, ya, wa, sorted(current + [j]), criterion)
                    if s < best[0] - 1e-10:
                        best = (s, "add", j)
        if direction in ("backward", "both"):
            for j in current:
                if j in forced or len(current) <= 1:
                    continue
                s = _subsetScore(phi, ya, wa, [c for c in current if c != j], criterion)
                if s < best[0] - 1e-10:
                    best = (s, "remove", j)
        if best[1] is not None:
            step += 1
            score, action, j = best
            current = sorted(current + [j]) if action == "add" else [c for c in current if c != j]
            history.append({"step": step, "action": action, "term": names[j], "score": score})
            improved = True
    model = LinearBasisModel(basis={"type": "subset", "basis": spec, "indices": current}, solver="ols")
    model.fit(xa, ya, weights, featureNames, outputNames)
    return StepwiseResult(model=model, selected=current, termNames=[names[i] for i in current],
                          history=history, criterion=criterion)
