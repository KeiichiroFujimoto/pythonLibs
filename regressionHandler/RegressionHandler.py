"""RegressionHandler: toolBase service exposing the regression toolkit.

The handler keeps named datasets and named models in memory and exposes
their operations as ``@secure_expose`` commands, so the same workflow runs
from Python, ``invoke(alias, **kwargs)`` pipelines, the ServiceREPL CLI,
REST wrappers or LLM tool calls::

    rh = RegressionHandler()
    rh.invoke("setData", dataName="d", x=x.tolist(), y=y.tolist(), featureNames=["x"])
    rh.invoke("fitModel", modelName="m", model="poly3", dataName="d")
    rh.invoke("predict", modelName="m", x=[[0.5]], level=0.95)

All command results are JSON-compatible dicts. The numerical work lives in
the plain-Python model classes (``models``, ``bases``, ``solvers``, ...);
the handler only manages state and conversion. Python callers can reach the
model objects directly with ``getModel`` / ``getDataset``.

Note: annotations are evaluated at definition time on purpose (no
``from __future__ import annotations``) because ``buildCatalog`` derives
parameter types from them.
"""
import copy
import json
from typing import Any, Optional

import numpy as np

from pythonLibs.tool.decorators import secure_expose
from pythonLibs.tool.toolBaseSecured import toolBaseSecured

from pythonLibs.regressionHandler.core.InputValidation import asFeatureMatrix, asOutputMatrix, asWeights, toJsonable
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase
from pythonLibs.regressionHandler.evaluation.Bootstrap import bootstrap
from pythonLibs.regressionHandler.evaluation.CrossValidation import crossValidate
from pythonLibs.regressionHandler.evaluation.Diagnostics import diagnose
from pythonLibs.regressionHandler.evaluation.ModelSelection import (ModelSelector, defaultCandidates, stepwiseSelect,
                                                                    tuneHyperparameters)
from pythonLibs.regressionHandler.evaluation.Report import modelReport
from pythonLibs.regressionHandler.evaluation.Sensitivity import sobolIndices
from pythonLibs.regressionHandler.evaluation.Variogram import empiricalVariogram, fitVariogram
from pythonLibs.regressionHandler.models.ModelFactory import availableModels, createModel
from pythonLibs.regressionHandler.sampling.Problems import PROBLEMS, getProblem
from pythonLibs.regressionHandler.sampling.Sampling import fullFactorial, latinHypercube, randomSampling, sobolLike

_CATEGORY = "regression"
_MISSING = object()


def _parseJson(value):
    """Structured parameters (dicts / lists) may arrive as JSON text from the CLI or REST."""
    if isinstance(value, str) and value.lstrip()[:1] in ("{", "["):
        return json.loads(value)
    return value


def _finiteRows(*arrays) -> np.ndarray:
    """Rows where every given array (n,) or (n, k) is finite."""
    ok = np.ones(arrays[0].shape[0], dtype=bool)
    for a in arrays:
        a = np.asarray(a, dtype=float)
        ok &= np.all(np.isfinite(a.reshape(a.shape[0], -1)), axis=1)
    return ok


class RegressionDataset:
    """Training data held by the handler."""

    def __init__(self, x, y, weights=None, featureNames=None, outputNames=None):
        self.x = asFeatureMatrix(x)
        # None / NaN mark unobserved outputs (cokriging, gradient-enhanced Kriging); models that need
        # complete data reject them when fitting
        self.y = asOutputMatrix(np.array(y, dtype=float), self.x.shape[0], allowMissing=True)
        self.weights = None if weights is None else asWeights(weights, self.x.shape[0])
        self.featureNames = list(featureNames) if featureNames else [f"x{i}" for i in range(self.x.shape[1])]
        self.outputNames = list(outputNames) if outputNames else [f"y{j}" for j in range(self.y.shape[1])]
        if len(self.featureNames) != self.x.shape[1] or len(self.outputNames) != self.y.shape[1]:
            raise ValueError("featureNames / outputNames do not match the data dimensions")

    def summary(self) -> dict:
        def stats(a, names):
            return [{"name": n, "min": float(np.nanmin(a[:, i])), "max": float(np.nanmax(a[:, i])),
                     "mean": float(np.nanmean(a[:, i])), "std": float(np.nanstd(a[:, i])),
                     "missing": int(np.sum(np.isnan(a[:, i])))} for i, n in enumerate(names)]
        return {"nSamples": int(self.x.shape[0]), "nInputs": int(self.x.shape[1]), "nOutputs": int(self.y.shape[1]),
                "weighted": self.weights is not None, "inputs": stats(self.x, self.featureNames),
                "outputs": stats(self.y, self.outputNames)}


class RegressionHandler(toolBaseSecured):
    """Regression / surrogate-model service built on toolBaseSecured."""

    LAB_AUTOREGISTER = True
    LAB_NAME = "regression"
    service_name = "regression"

    parameterMap = {
        "dataName": "Name of a dataset registered with setData / loadDataFromFile",
        "modelName": "Name under which a fitted model is stored",
        "x": "Inputs: list of rows [[x0, x1, ...], ...] or a flat list for one input",
        "y": "Outputs: list of values, or list of rows for several outputs",
        "weights": "Optional positive sample weights (1 / variance of each observation)",
        "featureNames": "Optional input names",
        "outputNames": "Optional output names",
        "model": "Model spec: shorthand ('linear', 'quadratic', 'poly3', 'pspline', 'gp', ...), "
                 "registered type ('kriging', 'rbf', 'spline', 'loess', ...), library form ('powerLaw', "
                 "'logistic', ...) or dict {'type': ..., **options}",
        "options": "Extra model options merged into the spec",
        "level": "Coverage probability of the interval (e.g. 0.95); omit for point predictions",
        "kind": "'prediction' (new observation) or 'confidence' (mean response)",
        "nFolds": "Number of cross-validation folds (n for leave-one-out)",
        "method": "Cross-validation method: 'refit' or 'analytic' (exact LOO for linear models)",
        "filePath": "Path of the file to read or write",
        "inputColumns": "Column names used as inputs",
        "outputColumns": "Column names used as outputs",
        "weightColumn": "Optional column with sample weights",
        "sheetName": "Excel sheet name (default: first sheet)",
        "kx": "Index of the input to differentiate with respect to",
        "includeTrainingData": "Store the training data in the saved model file",
        "candidates": "Model specs to compare (default: automatic list for the data size / dimension)",
        "criterion": "Selection criterion: 'cv' (k-fold RMSE), 'aicc' or 'bic' ('loo' for stepwise)",
        "oneStandardError": "Pick the simplest model within one CV standard error of the best",
        "nJobs": "Worker threads",
        "grid": "Hyperparameter grid {dotted.option: [values]}, e.g. {'basis.degree': [2, 3, 4]}",
        "basis": "Basis spec whose terms stepwise selection chooses from",
        "direction": "Stepwise direction: 'forward', 'backward' or 'both'",
        "nBoot": "Number of bootstrap replicates",
        "includeNoise": "Bootstrap prediction (not confidence) bands",
        "nSamples": "Number of sample points",
        "xlimits": "Input box [[lo, hi], ...] (sensitivity analysis: box of the uniform inputs, default training range)",
        "problemName": "Benchmark problem: " + ", ".join(sorted(PROBLEMS)),
        "seed": "Random seed",
        "aspect": "Model detail to return (see 'aspects' in getModelSummary), e.g. glmSummary, termTable, "
                  "spatialSummary, looDiagnostics, parameterIntervals, varianceComponents, featureImportance",
        "aspectOptions": "Keyword arguments of the aspect, e.g. {'level': 0.9, 'method': 'profile'}",
        "blocks": "Regions: point lists [[x...], ...] or boxes {'lower': [...], 'upper': [...], 'n': k}",
        "blockWeights": "Optional averaging weights per block (list of lists or None)",
        "nPerDim": "Grid cells per input used to discretize box blocks",
        "output": "Index of the output to analyse",
        "variogramName": "Name under which an empirical variogram is stored",
        "columns": "Input columns used as coordinates (default: all)",
        "residualsOf": "Model whose residuals are analysed instead of the raw output (removes a trend)",
        "nBins": "Number of distance bins",
        "maxDistance": "Largest lag distance",
        "binEdges": "Explicit bin edges (overrides nBins / maxDistance)",
        "estimator": "Variogram estimator: 'classical' or 'robust'",
        "tolerance": "Angular tolerance in degrees of a directional variogram",
        "distance": "'euclidean' or 'greatCircle' (longitude, latitude in degrees)",
        "variogramModel": "exponential, gaussian, spherical, cubic, matern or wendland",
        "nu": "Matern smoothness",
        "fitWeights": "Variogram fitting weights: 'cressie', 'counts' or 'equal'",
        "taus": "Quantile levels, e.g. [0.1, 0.5, 0.9]",
        "dataNames": "Datasets of increasing fidelity (lowest first), all with the same inputs",
    }

    def __init__(self, variablesDict=None, caseName=None, *, secure_enabled: bool = False,
                 auth_handler=None, token=None, exposure_mode: str = "expose_only",
                 workdir_root=None) -> None:
        # State must exist before toolBaseSecured wraps (and introspects) methods.
        self._datasets: dict = {}
        self._models: dict = {}
        self._modelData: dict = {}
        self._selections: dict = {}
        self._variograms: dict = {}
        super().__init__(instance_subcls=None, variablesDict=variablesDict, caseName=caseName,
                         secure_enabled=secure_enabled, auth_handler=auth_handler, token=token,
                         exposure_mode=exposure_mode, workdir_root=workdir_root)

    # ------------------------------------------------------------------ Python accessors
    def getDataset(self, dataName: str = "default") -> RegressionDataset:
        try:
            return self._datasets[dataName]
        except KeyError:
            raise KeyError(f"unknown dataset {dataName!r}; available: {sorted(self._datasets)}") from None

    def getModel(self, modelName: str) -> SurrogateModelBase:
        try:
            return self._models[modelName]
        except KeyError:
            raise KeyError(f"unknown model {modelName!r}; available: {sorted(self._models)}") from None

    def addModel(self, modelName: str, model: SurrogateModelBase, dataName: Optional[str] = None) -> None:
        """Register an already trained model object (Python API)."""
        if not model.isTrained:
            raise ValueError("model must be trained")
        self._models[modelName] = model
        self._modelData[modelName] = dataName
        self._selections.pop(modelName, None)

    def _register(self, modelName: str, model: SurrogateModelBase, dataName: Optional[str]) -> dict:
        """Store a fitted model and return its report; nothing changes if the report fails."""
        previous = [(store, store.get(modelName, _MISSING))
                    for store in (self._models, self._modelData, self._selections)]
        self._models[modelName] = model
        self._modelData[modelName] = dataName
        self._selections.pop(modelName, None)
        try:
            return self._modelReport(modelName)
        except Exception:
            for store, value in previous:
                if value is _MISSING:
                    store.pop(modelName, None)
                else:
                    store[modelName] = value
            raise

    def _columnIndices(self, ds: RegressionDataset, columns) -> Optional[list]:
        """Input columns given by index or by feature name -> indices."""
        if columns is None:
            return None
        out = []
        for c in _parseJson(columns):
            if isinstance(c, str) and not c.lstrip("-").isdigit():
                if c not in ds.featureNames:
                    raise KeyError(f"unknown input {c!r}; available: {ds.featureNames}")
                out.append(ds.featureNames.index(c))
            else:
                j = int(c)
                if not -ds.x.shape[1] <= j < ds.x.shape[1]:
                    raise IndexError(f"input column {j} out of range")
                out.append(j % ds.x.shape[1])
        return out

    @staticmethod
    def _outputIndex(ds: RegressionDataset, output) -> int:
        if isinstance(output, str) and not output.lstrip("-").isdigit():
            if output not in ds.outputNames:
                raise KeyError(f"unknown output {output!r}; available: {ds.outputNames}")
            return ds.outputNames.index(output)
        j = int(output)
        if not -ds.y.shape[1] <= j < ds.y.shape[1]:
            raise IndexError(f"output {j} out of range")
        return j % ds.y.shape[1]

    # ------------------------------------------------------------------ data
    @secure_expose(alias="setData", category=_CATEGORY)
    def setData(self, x: list, y: list, dataName: str = "default", weights: Optional[list] = None,
                featureNames: Optional[list] = None, outputNames: Optional[list] = None) -> dict:
        """Register a dataset from in-memory arrays."""
        x, y, weights = _parseJson(x), _parseJson(y), _parseJson(weights)
        ds = RegressionDataset(x, y, weights, featureNames, outputNames)
        self._datasets[dataName] = ds
        return toJsonable({"dataName": dataName, **ds.summary()})

    @secure_expose(alias="loadDataFromFile", category=_CATEGORY)
    def loadDataFromFile(self, filePath: str, inputColumns: list, outputColumns: list,
                         dataName: str = "default", weightColumn: Optional[str] = None,
                         sheetName: Optional[str] = None) -> dict:
        """Register a dataset from a CSV / Excel file (columns selected by name)."""
        import pandas as pd
        lower = filePath.lower()
        if lower.endswith((".xlsx", ".xlsm", ".xls")):
            df = pd.read_excel(filePath, sheet_name=sheetName or 0)
        else:
            df = pd.read_csv(filePath, sep=None, engine="python")
        missing = [c for c in list(inputColumns) + list(outputColumns) + ([weightColumn] if weightColumn else [])
                   if c not in df.columns]
        if missing:
            raise KeyError(f"columns not found in {filePath}: {missing}; available: {list(df.columns)}")
        cols = list(inputColumns) + list(outputColumns) + ([weightColumn] if weightColumn else [])
        df = df[cols].apply(pd.to_numeric, errors="coerce")
        dropped = int(df.isna().any(axis=1).sum())
        df = df.dropna()
        result = self.setData(x=df[list(inputColumns)].to_numpy(), y=df[list(outputColumns)].to_numpy(),
                              dataName=dataName,
                              weights=df[weightColumn].to_numpy() if weightColumn else None,
                              featureNames=list(inputColumns), outputNames=list(outputColumns))
        self.addExecutionInputFiles(file_paths=[filePath])
        result["droppedRows"] = dropped
        return result

    @secure_expose(alias="listData", category=_CATEGORY)
    def listData(self) -> dict:
        """List registered datasets."""
        return {"datasets": [{"dataName": k, "nSamples": int(v.x.shape[0]), "nInputs": int(v.x.shape[1]),
                              "nOutputs": int(v.y.shape[1])} for k, v in sorted(self._datasets.items())]}

    @secure_expose(alias="getDataSummary", category=_CATEGORY)
    def getDataSummary(self, dataName: str = "default") -> dict:
        """Ranges and moments of every input and output of a dataset."""
        return toJsonable({"dataName": dataName, **self.getDataset(dataName).summary()})

    @secure_expose(alias="deleteData", category=_CATEGORY)
    def deleteData(self, dataName: str) -> dict:
        """Remove a dataset."""
        self.getDataset(dataName)
        del self._datasets[dataName]
        return {"deleted": dataName}

    # ------------------------------------------------------------------ models
    @secure_expose(alias="listModelTypes", category=_CATEGORY)
    def listModelTypes(self) -> dict:
        """Available model types, their options and shorthand specs."""
        return toJsonable({"models": availableModels()})

    @secure_expose(alias="listLibraryForms", category=_CATEGORY)
    def listLibraryForms(self) -> dict:
        """Nonlinear model forms usable as model='<name>' (nonlinear least squares)."""
        from pythonLibs.regressionHandler.models.ModelLibrary import libraryCatalog
        return toJsonable({"forms": libraryCatalog()})

    @secure_expose(alias="fitModel", category=_CATEGORY)
    def fitModel(self, modelName: str, model: Any = "linear", dataName: str = "default",
                 options: Optional[dict] = None) -> dict:
        """Fit a model to a dataset and store it under modelName."""
        ds = self.getDataset(dataName)
        m = createModel(copy.deepcopy(_parseJson(model)), **(_parseJson(options) or {}))
        m.fit(ds.x, ds.y, ds.weights, ds.featureNames, ds.outputNames)
        return self._register(modelName, m, dataName)

    @secure_expose(alias="listModels", category=_CATEGORY)
    def listModels(self) -> dict:
        """List fitted models with their headline metrics."""
        rows = []
        for name, m in sorted(self._models.items()):
            rows.append({"modelName": name, "model": m.describe(), "dataName": self._modelData.get(name),
                         "rSquared": [mm.rSquared for mm in m.metrics], "rmse": [mm.rmse for mm in m.metrics]})
        return toJsonable({"models": rows})

    @secure_expose(alias="getModelSummary", category=_CATEGORY)
    def getModelSummary(self, modelName: str) -> dict:
        """Metrics, parameter table and text summary of a fitted model."""
        return self._modelReport(modelName)

    @secure_expose(alias="predict", category=_CATEGORY)
    def predict(self, modelName: str, x: list, level: Optional[float] = None, kind: str = "prediction") -> dict:
        """Predict outputs (optionally with a confidence / prediction interval)."""
        x = _parseJson(x)
        m = self.getModel(modelName)
        if level is None:
            return toJsonable({"modelName": modelName, "outputNames": m.outputNames,
                               "mean": m.predictValues(x)})
        pi = m.predictInterval(x, level=level, kind=kind)
        return toJsonable({"modelName": modelName, "outputNames": m.outputNames, **pi.toDict()})

    @secure_expose(alias="predictDerivatives", category=_CATEGORY)
    def predictDerivatives(self, modelName: str, x: list, kx: Optional[int] = None) -> dict:
        """Derivatives dy/dx_kx, or the full gradient (n, nx, ny) when kx is omitted."""
        x = _parseJson(x)
        m = self.getModel(modelName)
        if kx is None:
            return toJsonable({"modelName": modelName, "featureNames": m.featureNames,
                               "outputNames": m.outputNames, "gradient": m.predictGradient(x)})
        return toJsonable({"modelName": modelName, "kx": kx, "derivative": m.predictDerivatives(x, kx)})

    @secure_expose(alias="crossValidateModel", category=_CATEGORY)
    def crossValidateModel(self, modelName: str, nFolds: int = 5, method: str = "refit", seed: int = 0) -> dict:
        """k-fold / leave-one-out cross-validation of a fitted model's configuration."""
        m = self.getModel(modelName)
        x, y, w = self._trainingData(modelName)
        cv = crossValidate(m, x, y, w, nFolds=nFolds, method=method, seed=seed)
        return toJsonable({"modelName": modelName, "outputNames": m.outputNames, **cv.toDict()})

    @secure_expose(alias="saveModel", category=_CATEGORY)
    def saveModel(self, modelName: str, filePath: str, includeTrainingData: bool = True) -> dict:
        """Save a fitted model as JSON."""
        self.getModel(modelName).save(filePath, includeTrainingData)
        self.addExecutionOutputFiles(file_paths=[filePath])
        return {"modelName": modelName, "filePath": filePath}

    @secure_expose(alias="loadModel", category=_CATEGORY)
    def loadModel(self, filePath: str, modelName: str) -> dict:
        """Load a JSON model saved by saveModel."""
        report = self._register(modelName, SurrogateModelBase.load(filePath), None)
        self.addExecutionInputFiles(file_paths=[filePath])
        return report

    @secure_expose(alias="deleteModel", category=_CATEGORY)
    def deleteModel(self, modelName: str) -> dict:
        """Remove a fitted model."""
        self.getModel(modelName)
        del self._models[modelName]
        self._modelData.pop(modelName, None)
        self._selections.pop(modelName, None)
        return {"deleted": modelName}

    # ------------------------------------------------------------------ selection
    def _trainingData(self, modelName: str):
        """The model's own training data (a dataset may have been replaced since), else its dataset."""
        m = self.getModel(modelName)
        if m.xt is not None:
            return m.xt, m.yt, m.wt
        dataName = self._modelData.get(modelName)
        if dataName in self._datasets:
            ds = self._datasets[dataName]
            if ds.x.shape[1] == m.nx and ds.y.shape[1] == m.ny:
                return ds.x, ds.y, ds.weights
        raise ValueError("no training data available for this model")

    def _storeSelection(self, modelName, dataName, selection) -> dict:
        self._register(modelName, selection.bestModel, dataName)
        self._selections[modelName] = selection
        return toJsonable({"modelName": modelName, **selection.toDict(), "summary": selection.summary()})

    @secure_expose(alias="compareModels", category=_CATEGORY)
    def compareModels(self, dataName: str = "default", candidates: Optional[list] = None, criterion: str = "cv",
                      nFolds: int = 5, oneStandardError: bool = False, nJobs: int = 1,
                      modelName: Optional[str] = None, seed: int = 0) -> dict:
        """Rank candidate models by CV / AICc / BIC; optionally store the best as modelName."""
        ds = self.getDataset(dataName)
        cands = _parseJson(candidates) or defaultCandidates(ds.x, ds.y)
        cands = [_parseJson(c) for c in cands]
        sel = ModelSelector(cands, criterion, nFolds, seed, oneStandardError, nJobs).run(
            ds.x, ds.y, ds.weights, ds.featureNames, ds.outputNames)
        if modelName:
            return self._storeSelection(modelName, dataName, sel)
        return toJsonable({**sel.toDict(), "summary": sel.summary()})

    @secure_expose(alias="autoFit", category=_CATEGORY)
    def autoFit(self, modelName: str, dataName: str = "default", criterion: str = "cv", nFolds: int = 5,
                oneStandardError: bool = True, nJobs: int = 1) -> dict:
        """Try a data-appropriate set of models and keep the best (one-SE rule by default)."""
        return self.compareModels(dataName=dataName, candidates=None, criterion=criterion, nFolds=nFolds,
                                  oneStandardError=oneStandardError, nJobs=nJobs, modelName=modelName)

    @secure_expose(alias="tuneModel", category=_CATEGORY)
    def tuneModel(self, modelName: str, model: Any, grid: dict, dataName: str = "default", nFolds: int = 5,
                  nJobs: int = 1) -> dict:
        """Grid-search model options by cross-validation and keep the best."""
        ds = self.getDataset(dataName)
        sel = tuneHyperparameters(_parseJson(model), _parseJson(grid), ds.x, ds.y, ds.weights, nFolds=nFolds, nJobs=nJobs)
        return self._storeSelection(modelName, dataName, sel)

    @secure_expose(alias="stepwiseSelect", category=_CATEGORY)
    def stepwiseSelect(self, modelName: str, dataName: str = "default", basis: Optional[dict] = None,
                       criterion: str = "aicc", direction: str = "both") -> dict:
        """Stepwise selection of basis terms (default: full quadratic) by AICc / BIC / LOO."""
        ds = self.getDataset(dataName)
        res = stepwiseSelect(ds.x, ds.y, basis=_parseJson(basis), weights=ds.weights, criterion=criterion, direction=direction,
                             featureNames=ds.featureNames, outputNames=ds.outputNames)
        report = self._register(modelName, res.model, dataName)
        report["stepwise"] = toJsonable(res.toDict())
        return report

    # ------------------------------------------------------------------ uncertainty & diagnostics
    @secure_expose(alias="diagnoseModel", category=_CATEGORY)
    def diagnoseModel(self, modelName: str) -> dict:
        """Residual diagnostics: outliers, influence, normality, heteroscedasticity, lack of fit, VIF."""
        m = self.getModel(modelName)
        x, y, w = self._trainingData(modelName)
        rep = diagnose(m) if m.xt is not None else diagnose(m, x, y, w)
        return {"modelName": modelName, **rep.toDict(), "summary": rep.summary()}

    @secure_expose(alias="bootstrapModel", category=_CATEGORY)
    def bootstrapModel(self, modelName: str, x: Optional[list] = None, nBoot: int = 200, method: str = "residual",
                       level: float = 0.95, includeNoise: bool = False, nJobs: int = 1, seed: int = 0) -> dict:
        """Bootstrap percentile intervals for predictions (at x) and parameters."""
        x = _parseJson(x)
        m = self.getModel(modelName)
        xt, yt, wt = self._trainingData(modelName)
        res = bootstrap(m, xt, yt, xNew=x, weights=wt, nBoot=nBoot, method=method, level=level, seed=seed,
                        nJobs=nJobs, includeNoise=includeNoise)
        return {"modelName": modelName, "outputNames": m.outputNames, **res.toDict()}

    # ------------------------------------------------------------------ design of experiments
    @secure_expose(alias="generateSamples", category=_CATEGORY)
    def generateSamples(self, nSamples: int, xlimits: list, method: str = "lhs", criterion: str = "maximin",
                        seed: int = 0) -> dict:
        """Design of experiments: 'lhs', 'fullFactorial' (nSamples = levels per input), 'random', 'halton'."""
        xlimits = _parseJson(xlimits)
        if method == "lhs":
            pts = latinHypercube(nSamples, xlimits, criterion=criterion, seed=seed)
        elif method == "fullFactorial":
            pts = fullFactorial(nSamples, xlimits)
        elif method == "random":
            pts = randomSampling(nSamples, xlimits, seed=seed)
        elif method == "halton":
            pts = sobolLike(nSamples, xlimits, seed=seed)
        else:
            raise ValueError("method must be lhs, fullFactorial, random or halton")
        return toJsonable({"method": method, "x": pts})

    @secure_expose(alias="sampleBenchmark", category=_CATEGORY)
    def sampleBenchmark(self, problemName: str, nSamples: int, dataName: Optional[str] = None,
                        seed: int = 0) -> dict:
        """Sample a benchmark function (LHS) and register it as a dataset."""
        prob = getProblem(problemName)
        x, y = prob.sample(nSamples, seed=seed)
        return self.setData(x=x, y=y, dataName=dataName or problemName,
                            featureNames=[f"x{i}" for i in range(prob.nx)], outputNames=[problemName])

    # ------------------------------------------------------------------ model details and joint predictions
    _ASPECTS = ("hyperparameters", "spatialSummary", "looDiagnostics", "looResiduals", "replicates",
                "parameterIntervals", "glmSummary", "gamSummary", "termTable", "residuals", "varianceComponents",
                "randomEffects", "evidence", "featureImportance", "outOfBag", "trainingHistory", "coregionalization",
                "varianceParameters", "transformParameter", "inputCorrections", "pseudoR2", "objective",
                "coefficients", "leverage", "parameters", "constraintReport", "groups")
    _ASPECT_KWARGS = {"parameterIntervals": {"level", "method", "nGrid"}, "residuals": {"kind"}}

    @classmethod
    def _aspectsOf(cls, m) -> list:
        return [a for a in cls._ASPECTS if hasattr(type(m), a)]

    @secure_expose(alias="inspectModel", category=_CATEGORY)
    def inspectModel(self, modelName: str, aspect: str, aspectOptions: Optional[dict] = None) -> dict:
        """Model-specific details (GLM / GAM tables, spatial summary, LOO, intervals, importance, ...)."""
        m = self.getModel(modelName)
        if aspect not in self._aspectsOf(m):
            raise ValueError(f"aspect {aspect!r} is not available for {m.registryName}; "
                             f"available: {self._aspectsOf(m)}")
        attr = getattr(type(m), aspect)
        opts = dict(_parseJson(aspectOptions) or {})
        if isinstance(attr, property):
            if opts:
                raise ValueError(f"{aspect} takes no options")
        else:
            unknown = set(opts) - self._ASPECT_KWARGS.get(aspect, set())
            if unknown:
                raise ValueError(f"unsupported options for {aspect}: {sorted(unknown)}")

        def get(obj):
            return getattr(obj, aspect) if isinstance(attr, property) else getattr(obj, aspect)(**opts)

        # models fitted as one sub-model per output: the aspect of each output
        subs = getattr(m, "_subModels", None)
        value = get(m) if subs is None else {"outputs": {n: get(s) for n, s in zip(m.outputNames, subs)}}
        return toJsonable({"modelName": modelName, "aspect": aspect, "value": value})

    @secure_expose(alias="predictCovariance", category=_CATEGORY)
    def predictCovariance(self, modelName: str, x: list, kind: str = "confidence") -> dict:
        """Joint posterior covariance of the predictions at x, one (m, m) matrix per output."""
        x = _parseJson(x)
        m = self.getModel(modelName)
        return toJsonable({"modelName": modelName, "outputNames": m.outputNames, "kind": kind,
                           "mean": m.predictValues(x), "covariance": m.predictCovariance(x, kind)})

    @secure_expose(alias="simulate", category=_CATEGORY)
    def simulate(self, modelName: str, x: list, nSamples: int = 1, seed: int = 0, kind: str = "confidence") -> dict:
        """Conditional simulation: nSamples joint draws (nSamples, m, ny) from the posterior at x."""
        x = _parseJson(x)
        m = self.getModel(modelName)
        return toJsonable({"modelName": modelName, "outputNames": m.outputNames,
                           "samples": m.simulate(x, nSamples, seed, kind)})

    @secure_expose(alias="predictBlock", category=_CATEGORY)
    def predictBlock(self, modelName: str, blocks: list, blockWeights: Optional[list] = None,
                     nPerDim: int = 8) -> dict:
        """Mean and variance of block averages (block Kriging for models with a joint covariance)."""
        m = self.getModel(modelName)
        res = m.predictBlock(_parseJson(blocks), _parseJson(blockWeights), nPerDim)
        return toJsonable({"modelName": modelName, "outputNames": m.outputNames, "mean": res["mean"],
                           "variance": res["variance"], "nPoints": [len(p) for p in res["points"]]})

    @secure_expose(alias="predictTerms", category=_CATEGORY)
    def predictTerms(self, modelName: str, x: list) -> dict:
        """Centred partial effects (and standard errors) of every term of an additive model."""
        x = _parseJson(x)
        m = self.getModel(modelName)
        if not hasattr(m, "predictTerms"):
            raise ValueError(f"{m.registryName} has no additive terms")
        return toJsonable({"modelName": modelName, "terms": m.predictTerms(x)})

    # ------------------------------------------------------------------ analyses
    @secure_expose(alias="sensitivityAnalysis", category=_CATEGORY)
    def sensitivityAnalysis(self, modelName: str, method: str = "auto", xlimits: Optional[list] = None,
                            nSamples: int = 4096, seed: int = 0, output: int = 0, nBoot: int = 200) -> dict:
        """Sobol indices (exact for polynomial chaos models, Monte Carlo otherwise) of one output."""
        m = self.getModel(modelName)
        res = sobolIndices(m, xlimits=_parseJson(xlimits), method=method, nSamples=nSamples, seed=seed, output=output,
                           nBoot=nBoot)
        return toJsonable({"modelName": modelName, "output": m.outputNames[output], **res.toDict(),
                           "summary": res.summary()})

    @secure_expose(alias="empiricalVariogram", category=_CATEGORY)
    def empiricalVariogram(self, dataName: str = "default", output: int = 0, columns: Optional[list] = None,
                           residualsOf: Optional[str] = None, nBins: int = 15, maxDistance: Optional[float] = None,
                           binEdges: Optional[list] = None, estimator: str = "classical",
                           direction: Optional[Any] = None, tolerance: float = 22.5, distance: str = "euclidean",
                           variogramName: Optional[str] = None) -> dict:
        """Binned empirical semivariogram of an output (or of a model's residuals)."""
        ds = self.getDataset(dataName)
        cols = self._columnIndices(ds, columns)
        out = self._outputIndex(ds, output)
        coords = ds.x if cols is None else ds.x[:, cols]
        z = ds.y[:, out]
        if residualsOf:
            z = z - self.getModel(residualsOf).predictValues(ds.x)[:, out]
        ok = np.isfinite(z)
        ev = empiricalVariogram(coords[ok], z[ok], nBins=nBins, maxDistance=maxDistance,
                                binEdges=_parseJson(binEdges), estimator=estimator, direction=_parseJson(direction),
                                tolerance=tolerance, distance=distance)
        if variogramName:
            self._variograms[variogramName] = {"variogram": ev, "dataName": dataName, "columns": cols,
                                               "output": out, "residualsOf": residualsOf}
        return toJsonable({"variogramName": variogramName, **ev.toDict()})

    @secure_expose(alias="fitVariogram", category=_CATEGORY)
    def fitVariogram(self, variogramName: str, variogramModel: str = "exponential", nu: float = 0.5,
                     fitWeights: str = "cressie", nugget: Optional[float] = None, fitNu: bool = False,
                     modelName: Optional[str] = None) -> dict:
        """Fit a variogram model; with modelName also fit a Kriging model with the fitted covariance."""
        try:
            stored = self._variograms[variogramName]
        except KeyError:
            raise KeyError(f"unknown variogram {variogramName!r}; available: {sorted(self._variograms)}") from None
        fit = fitVariogram(stored["variogram"], variogramModel, nu=nu, weights=fitWeights, nugget=nugget, fitNu=fitNu)
        out = {"variogramName": variogramName, **fit.toDict(), "practicalRange": fit.practicalRange()}
        if modelName:
            opts = fit.krigingOptions()
            cols = stored["columns"]
            if cols is not None:
                opts["spatialColumns"] = list(cols)
            # the Kriging model is fitted to the variogram's own output on its observed rows
            ds = self.getDataset(stored["dataName"])
            j = stored["output"]
            ok = _finiteRows(ds.y[:, j])
            m = createModel({"type": "kriging", "poly": "constant", **opts})
            m.fit(ds.x[ok], ds.y[ok][:, [j]], None if ds.weights is None else ds.weights[ok], ds.featureNames,
                  [ds.outputNames[j]])
            out["kriging"] = self._register(modelName, m, stored["dataName"])
        else:
            out["krigingOptions"] = fit.krigingOptions() if variogramModel != "cubic" else None
        return toJsonable(out)

    @secure_expose(alias="quantileProcess", category=_CATEGORY)
    def quantileProcess(self, taus: list, dataName: str = "default", basis: Optional[dict] = None,
                        output: int = 0) -> dict:
        """Quantile-regression coefficients beta(tau) for several levels (exact solutions)."""
        from pythonLibs.regressionHandler.models.QuantileModel import quantileProcess
        ds = self.getDataset(dataName)
        j = self._outputIndex(ds, output)
        ok = _finiteRows(ds.y[:, j])
        res = quantileProcess(ds.x[ok], ds.y[ok, j], _parseJson(taus), basis=_parseJson(basis),
                              weights=None if ds.weights is None else ds.weights[ok])
        return toJsonable({"dataName": dataName, "output": ds.outputNames[j], "nUsed": int(ok.sum()), **res})

    @secure_expose(alias="fitMultiFidelity", category=_CATEGORY)
    def fitMultiFidelity(self, modelName: str, dataNames: list, options: Optional[dict] = None,
                         output: int = 0) -> dict:
        """Recursive multi-fidelity Kriging from datasets of increasing fidelity (lowest first)."""
        from pythonLibs.regressionHandler.models.MultiFidelityKrigingModel import MultiFidelityKrigingModel
        dataNames = list(_parseJson(dataNames))
        sets = [self.getDataset(d) for d in dataNames]
        if len({s.x.shape[1] for s in sets}) != 1:
            raise ValueError("all fidelity levels need the same inputs")
        xs, ys = [], []
        for s in sets:
            j = self._outputIndex(s, output)
            ok = _finiteRows(s.y[:, j])
            xs.append(s.x[ok])
            ys.append(s.y[ok, j])
        m = MultiFidelityKrigingModel(**(_parseJson(options) or {}))
        m.fitLevels(xs, ys)
        report = self._register(modelName, m, None)
        report["levels"] = list(dataNames)
        return report

    # ------------------------------------------------------------------ reporting
    @secure_expose(alias="exportReport", category=_CATEGORY)
    def exportReport(self, modelName: str, filePath: Optional[str] = None, crossValidation: bool = True,
                     diagnostics: bool = True, nFolds: int = 5) -> dict:
        """Markdown report (fit, parameters, CV, diagnostics, comparison); written to filePath if given."""
        m = self.getModel(modelName)
        dataName = self._modelData.get(modelName)
        ds = self._datasets.get(dataName)
        cv = diag = None
        if crossValidation or diagnostics:
            x, y, w = self._trainingData(modelName)
            if crossValidation:
                cv = crossValidate(m, x, y, w, nFolds=min(nFolds, x.shape[0]))
            if diagnostics:
                diag = diagnose(m) if m.xt is not None else diagnose(m, x, y, w)
        text = modelReport(m, title=f"Regression report: {modelName}", cv=cv, diagnostics=diag,
                           selection=self._selections.get(modelName),
                           dataSummary=ds.summary() if ds is not None else None)
        if filePath:
            with open(filePath, "w", encoding="utf-8") as f:
                f.write(text)
            self.addExecutionOutputFiles(file_paths=[filePath])
        self.addDocumentResponseTitle(titleDocument=f"Regression report: {modelName}")
        self.addDocumentResponseContent(contentType="markdown", contentParams=[text])
        return {"modelName": modelName, "filePath": filePath, "markdown": text}

    # ------------------------------------------------------------------ helpers
    def _modelReport(self, modelName: str) -> dict:
        m = self.getModel(modelName)
        report = {
            "modelName": modelName, "model": m.describe(), "type": m.registryName,
            "dataName": self._modelData.get(modelName), "nInputs": m.nx, "nOutputs": m.ny,
            "featureNames": m.featureNames, "outputNames": m.outputNames,
            "supports": dict(m.supports),
            "aspects": self._aspectsOf(m),
            "metrics": {name: mm.toDict() for name, mm in zip(m.outputNames, m.metrics)},
            "summary": m.summary(),
        }
        if m.result is not None:
            report["parameters"] = {name: m.result.table(j) for j, name in enumerate(m.outputNames)}
        hyper = getattr(type(m), "hyperparameters", None)
        if isinstance(hyper, property):
            report["hyperparameters"] = m.hyperparameters
        equation = getattr(m, "equation", None)
        if callable(equation):
            try:
                report["equations"] = [equation(j) for j in range(m.ny)]
            except Exception:
                pass
        return toJsonable(report)
