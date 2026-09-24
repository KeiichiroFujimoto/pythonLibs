"""Persistence, summaries and shorthands of the statistical / machine-learning models."""
import json

import numpy as np
import pytest

from pythonLibs.regressionHandler import SurrogateModelBase, createModel


def _data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, 2))
    return x, 1 + 2 * x[:, 0] + np.sin(4 * x[:, 1]) + 0.1 * rng.standard_normal(n), rng


SPECS = [
    {"type": "glm", "family": "gaussian"},
    {"type": "glm", "family": {"type": "gamma", "link": "log"}},
    {"type": "gam"},
    {"type": "quantile", "tau": 0.3},
    {"type": "odr", "expression": "a + b*x0 + c*x1", "params": ["a", "b", "c"], "xSigma": [0.01, 0.01]},
    {"type": "heteroscedastic"},
    {"type": "transformedTarget", "model": "quadratic"},
    {"type": "randomForest", "nTrees": 20},
    {"type": "gradientBoosting", "nEstimators": 50},
    {"type": "neuralNetwork", "hidden": [8], "maxIter": 200, "nNetworks": 2},
    {"type": "bayesLinear", "prior": "ard"},
    "pce4",
]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s if isinstance(s, str) else s["type"])
def test_roundTripAndSummary(spec):
    x, y, rng = _data()
    if isinstance(spec, dict) and spec["type"] == "glm" and spec["family"] != "gaussian":
        y = np.exp(y / 3)
    m = createModel(spec).fit(x, y)
    t = rng.uniform(0.1, 0.9, (7, 2))
    loaded = SurrogateModelBase.fromDict(json.loads(json.dumps(m.toDict())))
    np.testing.assert_allclose(loaded.predict(t), m.predict(t), rtol=1e-9, atol=1e-12)
    if m.supports.get("variances"):
        kind = "confidence" if isinstance(spec, dict) and spec["type"] == "quantile" else "prediction"
        np.testing.assert_allclose(loaded.predictVariances(t, kind), m.predictVariances(t, kind), rtol=1e-8)
    assert isinstance(m.summary(), str)
    assert m.predictGradient(t).shape == (7, 2, 1)


def test_oneDimensionalAndGroupedModels():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 80)
    y = np.log1p(3 * x) + 0.05 * rng.standard_normal(80)
    for spec in ({"type": "isotonic"}, {"type": "shapeSpline", "shape": "increasingConcave"}):
        m = createModel(spec).fit(x, y)
        loaded = SurrogateModelBase.fromDict(json.loads(json.dumps(m.toDict())))
        np.testing.assert_allclose(loaded.predict(x), m.predict(x), rtol=1e-12)
    g = rng.integers(0, 6, 80).astype(float)
    mixed = createModel({"type": "mixed", "groupColumn": 1}).fit(np.column_stack([x, g]), y + 0.3 * g)
    loaded = SurrogateModelBase.fromDict(json.loads(json.dumps(mixed.toDict())))
    xt = np.column_stack([x[:5], np.array([0, 1, 2, 3, 99.0])])
    np.testing.assert_allclose(loaded.predict(xt), mixed.predict(xt), rtol=1e-12)
    assert mixed.predict(xt[4:])[0] == pytest.approx(mixed.predictPopulation(xt[4:])[0])
    assert 0 < mixed.varianceComponents()["icc"] < 1


def test_multiFidelityAndGradientKrigingRoundTrip():
    from pythonLibs.regressionHandler import GradientKrigingModel, MultiFidelityKrigingModel
    rng = np.random.default_rng(2)
    xl, xh = rng.uniform(0, 1, 15), rng.uniform(0, 1, 5)
    f = lambda z: np.sin(6 * z)
    mf = MultiFidelityKrigingModel().fitLevels([xl, xh], [0.8 * f(xl) + 0.2, f(xh)])
    t = np.column_stack([np.linspace(0, 1, 6), np.ones(6)])
    loaded = SurrogateModelBase.fromDict(json.loads(json.dumps(mf.toDict())))
    np.testing.assert_allclose(loaded.predict(t), mf.predict(t), rtol=1e-10)
    x = rng.uniform(0, 1, (8, 1))
    gek = GradientKrigingModel().fit(x, np.column_stack([f(x[:, 0]), 6 * np.cos(6 * x[:, 0])]))
    loaded = SurrogateModelBase.fromDict(json.loads(json.dumps(gek.toDict())))
    np.testing.assert_allclose(loaded.predict(t[:, :1]), gek.predict(t[:, :1]), rtol=1e-10)


def test_shorthands():
    x, y, rng = _data()
    assert createModel("logistic-regression").fit(x, (y > 2).astype(float)).glmSummary()["family"] == "binomial(logit)"
    assert createModel("median").fit(x, y).options["tau"] == 0.5
    assert createModel("poisson-regression").fit(x, rng.poisson(np.exp(y / 3)).astype(float)).glmSummary()["converged"]
