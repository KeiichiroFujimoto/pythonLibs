# regressionHandler

Regression and surrogate modelling, 1D to N-D, with multi-output support.
All algorithms (special functions, distributions, optimizers, B-splines,
kernels, solvers) are implemented here on top of numpy only.

Design principles:

- one template-method base class (`SurrogateModelBase`) with declared,
  validated options and `supports` capability flags
- gradients (`predictDerivatives`) and variances (`predictVariances`) as
  first-class outputs next to the mean prediction
- parameter inference (estimates, standard errors, t, p, confidence
  intervals, `summary()`) and cross-validation built in
- JSON persistence (no pickle), silent by default

## Layout

```text
regressionHandler/
  RegressionHandler.py   toolBaseSecured service (@secure_expose commands)
  core/                  SurrogateModelBase, OptionsDictionary, Registry, FitResult, FitMetrics, SafeExpression
  numerics/              SpecialFunctions, Distributions, LinearAlgebra, Optimizers, PLS, NeighborSearch
  bases/                 polynomial, orthogonalPolynomial, radial, bspline, expression, combined
  solvers/               ols, ridge (GCV/LOO), elasticNet, robust (IRLS)
  kernels/               squaredExponential, matern32/52, absoluteExponential, powerExponential,
                         rationalQuadratic, periodic, sum/product (analytic gradients, ARD / PLS)
  models/                LinearBasisModel, KrigingModel (KPLS), RbfModel, IdwModel, NonlinearModel,
                         ModelLibrary, SplineModel, LocalRegressionModel (LOESS), ModelFactory
  evaluation/            crossValidate (k-fold, threaded, analytic LOO), ModelSelector / defaultCandidates,
                         tuneHyperparameters, stepwiseSelect, diagnose, bootstrap, modelReport
  sampling/              latinHypercube (maximin / ESE), fullFactorial, randomSampling, sobolLike,
                         benchmark problems (branin, rosenbrock, sphere, ackley, hartmann3/6, ishigami, friedman)
  tests/
```

## Python API

```python
import numpy as np
from pythonLibs.regressionHandler import createModel, crossValidate, loadModel

model = createModel("quadratic").fit(x, y)                 # x: (n, nx), y: (n,) or (n, ny)
model.predict(xNew)                                         # (m,) or (m, ny)
band = model.predictInterval(xNew, level=0.95, kind="prediction")
grad = model.predictGradient(xNew)                          # (m, nx, ny)
print(model.summary())                                      # R2, AICc, BIC + coefficient table
cv = crossValidate(model, x, y, nFolds=10)                  # or method="analytic" for exact LOO
model.save("model.json"); same = loadModel("model.json")
```

Building blocks compose freely:

```python
from pythonLibs.regressionHandler import LinearBasisModel

LinearBasisModel(basis={"type": "orthogonalPolynomial", "degree": 8}, solver="ridge")          # stable RSM
LinearBasisModel(basis={"type": "bspline", "nSegments": 25},
                 solver={"type": "ridge", "penalty": "smoothness"})                              # P-spline (1-3D)
LinearBasisModel(basis={"type": "expression", "terms": ["x", "1/x**2", "log(x)"], "variables": ["x"]})
LinearBasisModel(basis={"type": "polynomial", "degree": 2}, solver={"type": "robust", "loss": "bisquare"})
```

Shorthands: `linear`, `quadratic`, `poly<N>`, `ortho<N>`, `ridge-poly<N>`,
`robust-poly<N>`, `lasso-poly<N>`, `pspline`, `rbf-ridge`, `gp`, `rbf-smooth`,
plus the registered types `linearBasis`, `kriging`, `kpls`, `rbf`, `idw`.

## Models

| Type | Method | Inputs | Variances | Notes |
|---|---|---|---|---|
| `linearBasis` | linear-in-parameter least squares | N-D | yes (OLS/ridge/robust) | any basis x solver; exact LOO |
| `kriging` | universal Kriging / GP | N-D | yes (universal Kriging) | REML/ML, analytic likelihood gradient, estimated nugget, composable kernels |
| `kpls` | Kriging + PLS | high-D | yes | PLS-reduced lengthscales |
| `rbf` | radial basis functions | N-D | no | LOO (Rippa) smoothing, local `neighbors` mode |
| `idw` | inverse distance weighting | N-D | no | exact interpolation baseline |
| `nonlinear` | nonlinear least squares | N-D | yes (delta method) | safe expressions or callables, robust losses, bounds, multi-start |
| `spline` | penalized B-splines | 1-3 D | yes | P-spline, GCV / LOO smoothing |
| `loess` | local polynomial regression | N-D | yes (equivalent kernel) | local polynomial degree 0-2, robustness iterations |

### Model form library

`createModel("<form>")` builds a `NonlinearModel` with automatic initial
guesses: `linear`, `powerLaw`, `powerLawOffset`, `exponential`,
`exponentialDecay`, `exponentialRise`, `inverseExponential`, `logistic`,
`saturation`, `powerSaturation`, `gaussianPeak`, `weibullCdf`, `sinusoid`,
`rationalPower`, `linearPlusPower`, `logarithmic`, `inverse`. Add your own with
`registerForm(name, expression, params, guess)`.

```python
from pythonLibs.regressionHandler import NonlinearModel, createModel
m = createModel("exponentialDecay").fit(x, y)        # a, tau, c with std errors / CI in m.result
NonlinearModel(expression="c*u**a*v**b", params=["c", "a", "b"], variables=["u", "v"],
               p0=[1.0, 1.0, 1.0], loss="soft_l1", nStart=5).fit(X, y)
```

The k-nearest-neighbour search used by local models is a grid index,
exact and 30-60x faster than brute force on large sets.

```python
from pythonLibs.regressionHandler import KrigingModel
gp = KrigingModel(corr="matern52", poly="linear").fit(x, y)        # nugget="auto" estimates noise
gp.hyperparameters                                                 # lengthscales, noise variance, ...
gp.predictVariances(xNew)                                          # universal-Kriging variance
KrigingModel(corr={"type": "product", "kernels": ["matern52", "periodic"]})
```

## Selection, diagnostics and uncertainty

```python
from pythonLibs.regressionHandler import (ModelSelector, defaultCandidates, tuneHyperparameters, stepwiseSelect,
                                          diagnose, bootstrap, modelReport)

sel = ModelSelector(defaultCandidates(x, y), criterion="cv", oneStandardError=True, nJobs=4).run(x, y)
print(sel.summary()); best = sel.bestModel
tuneHyperparameters("linear", {"basis.degree": [1, 2, 3]}, x, y).best.name
stepwiseSelect(x, y, basis={"type": "polynomial", "degree": 2}, criterion="bic").termNames
print(diagnose(best).summary())            # outliers, Cook's D, normality, Breusch-Pagan, runs, VIF
bootstrap(best, x, y, xNew, method="wild", nBoot=500, nJobs=4)   # percentile bands for any model
open("report.md", "w").write(modelReport(best, selection=sel, diagnostics=diagnose(best)))
```

numpy is the only runtime dependency.

## toolBase service

```python
from pythonLibs.regressionHandler import RegressionHandler

rh = RegressionHandler()                      # secure_enabled=False, exposure_mode="expose_only"
rh.invoke("loadDataFromFile", filePath="data.csv", inputColumns=["x1", "x2"], outputColumns=["y"])
rh.invoke("fitModel", modelName="m", model="quadratic")
rh.invoke("predict", modelName="m", x=[[0.5, 1.0]], level=0.95)
rh.buildCatalog()                             # command catalog (REPL / REST / LLM tools)
rh.toCLI().run()                              # interactive ServiceREPL
```

Commands (category `regression`):

| Area | Commands |
|---|---|
| Data | `setData`, `loadDataFromFile`, `listData`, `getDataSummary`, `deleteData`, `generateSamples`, `sampleBenchmark` |
| Models | `listModelTypes`, `listLibraryForms`, `fitModel`, `listModels`, `getModelSummary`, `deleteModel`, `saveModel`, `loadModel` |
| Prediction | `predict` (with `level` for intervals), `predictDerivatives` |
| Selection | `autoFit`, `compareModels`, `tuneModel`, `stepwiseSelect`, `crossValidateModel` |
| Quality | `diagnoseModel`, `bootstrapModel`, `exportReport` (Markdown + toolBase documentResponse) | The class auto-registers in `LabRegistry` as
`regression`.

## Tests

```bash
pip install -e ".[all]" pytest
python -m pytest regressionHandler/tests -q
```

- `tests/analytic/` compares every component with closed-form or
  theoretical results: special-case identities of the incomplete gamma /
  beta functions and distributions, exact solutions of the optimizers,
  OLS / WLS / ridge formulas, Lasso soft-thresholding, the P-spline penalty
  null space, the PRESS identity, the Gaussian-process posterior written
  out explicitly, the RBF augmented system, the Cox-de Boor recursion, LOESS
  polynomial reproduction, influence measures from the hat matrix, the
  nominal size of the normality tests and the known optima of the
  benchmark functions.
- the other files test workflows, persistence, model selection and the
  toolBase service.

The tests use numpy and pytest only.
