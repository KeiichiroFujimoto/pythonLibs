"""Regression tests for defects found by the audit of RBF / IDW / trees / evaluation / sampling / core / numerics."""
import importlib

import numpy as np
import pytest

from pythonLibs.regressionHandler import (RandomForestModel, RbfModel, SplineModel, createModel, crossValidate,
                                          diagnose, tuneHyperparameters)
from pythonLibs.regressionHandler.models.KrigingModel import KrigingModel

Optimizers = importlib.import_module("pythonLibs.regressionHandler.numerics.Optimizers")
NeighborSearch = importlib.import_module("pythonLibs.regressionHandler.numerics.NeighborSearch")
Sampling = importlib.import_module("pythonLibs.regressionHandler.sampling.Sampling")


@pytest.mark.parametrize("shift", [-5.0, 0.0, 5.0, 100.0])
def test_minimizeScalarTakesParabolicStepsOnAnyBracket(shift):
    # Brent's method minimizes a parabola in a handful of evaluations (scipy's fminbound: 6) whatever the
    # sign of the bracket; the acceptance test used to reject parabolic steps when a >= 0 (golden section only).
    res = Optimizers.minimizeScalar(lambda x: (x - shift - 0.3) ** 2, (shift - 2.0, shift + 3.0), xatol=1e-8)
    assert res.x[0] == pytest.approx(shift + 0.3, abs=1e-7)
    assert res.nfev <= 10


def test_analyticLooUsesTheGivenData():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (60, 2))
    y1 = np.sin(3 * x[:, 0]) + x[:, 1]
    y2 = np.cos(5 * x[:, 1])
    w = rng.uniform(0.5, 2.0, 60)
    trained = createModel("quadratic").fit(x, y1)
    # exact LOO (refit n times) is the external truth for the analytic PRESS residuals
    for y, weights in ((y2, None), (y1, w)):
        analytic = crossValidate(trained, x, y, weights, method="analytic")
        refit = crossValidate(trained, x, y, weights, nFolds=60)
        np.testing.assert_allclose(analytic.predictions, refit.predictions, rtol=1e-9, atol=1e-12)


def test_createModelInstanceOverridesAreRevalidated():
    x = np.linspace(0.0, 1.0, 60)
    y = np.sin(8 * x)
    m = createModel(SplineModel(nSegments=5), nSegments=30).fit(x, y)
    assert m.coefficients.shape == SplineModel(nSegments=30).fit(x, y).coefficients.shape
    with pytest.raises(KeyError):
        createModel(KrigingModel(), corr="notAKernel")


def test_optionsAcceptNumpyScalars():
    m = RandomForestModel(nTrees=np.int64(3), maxFeatures=np.float64(1.0))
    assert type(m.options["nTrees"]) is int
    assert createModel("rbf", kernel=np.str_("cubic")).options["kernel"] == "cubic"
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 40)
    res = tuneHyperparameters("poly2", {"basis.degree": np.arange(1, 4)}, x, np.sin(3 * x), nFolds=4)
    assert all(s.error is None for s in res.scores)


def test_tuneHyperparametersAcceptsLibraryForms():
    rng = np.random.default_rng(0)
    x = rng.uniform(1, 5, 40)
    y = 2.0 * x ** 1.5 * (1 + 0.01 * rng.normal(size=40))
    res = tuneHyperparameters("powerLaw", {"nStart": [1, 2]}, x, y, nFolds=4)
    assert all(s.error is None for s in res.scores)
    assert res.bestModel.registryName == "nonlinear"


def test_eseLatinHypercubeReturnsBestDesignVisited(monkeypatch):
    original = Sampling._minDistance
    for seed in range(60):
        seen = []

        def recording(points):
            seen.append(original(points))
            return seen[-1]

        monkeypatch.setattr(Sampling, "_minDistance", recording)
        design = Sampling.latinHypercube(12, np.tile([0.0, 1.0], (3, 1)), "ese", seed=seed, iterations=10)
        monkeypatch.setattr(Sampling, "_minDistance", original)
        assert original(design) == pytest.approx(max(seen), rel=1e-12)


def test_rbfCallSinglePointMultiOutputShape():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, (30, 2))
    y = np.column_stack([np.sin(3 * x[:, 0]), x[:, 1] ** 2])
    for opts in ({}, {"neighbors": 8}):
        m = RbfModel(**opts).fit(x, y)
        out = m(x[:1] + 0.01)
        assert out.shape == (1, 2)
        np.testing.assert_allclose(out, m.predict(x[:1] + 0.01), rtol=1e-9)


def test_bruteForceNeighborDistancesAreExact():
    rng = np.random.default_rng(0)
    points = 1000.0 + rng.uniform(0, 1, (300, 3))
    queries = np.vstack([points[:40], 1000.0 + rng.uniform(0, 1, (50, 3))])
    dist, idx = NeighborSearch.NeighborSearch(points).query(queries, 5)
    # a query at a data point is at distance 0; the others match the coordinate differences
    assert np.all(dist[:40, 0] == 0.0) and np.array_equal(idx[:40, 0], np.arange(40))
    direct = np.sqrt(np.sum((points[idx] - queries[:, None, :]) ** 2, axis=-1))
    np.testing.assert_allclose(dist, direct, rtol=1e-12, atol=1e-15)
    # a local RBF interpolates its data (the Gram distances used to give errors of ~5e-4 here)
    y = np.sin(3 * (points - 1000.0).sum(axis=1))
    m = RbfModel(kernel="gaussian", neighbors=20, normalize="none", epsilon=5.0).fit(points, y)
    assert np.max(np.abs(m.predict(points) - y)) < 1e-9


def test_diagnoseInterpolantHasNoResidualTestWarnings():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (60, 2))
    y = 1 + 2 * x[:, 0] + np.sin(4 * x[:, 1])
    rep = diagnose(RbfModel().fit(x, y))
    assert np.isnan(rep.outputs[0]["normalityP"])
    assert not any("not normal" in w or "Breusch" in w or "cluster" in w for w in rep.warnings)
