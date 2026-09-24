"""Kernels, Kriging / KPLS, RBF, IDW, sampling and neighbour search."""
import json

import numpy as np
import pytest

from pythonLibs.regressionHandler import (IdwModel, KplsModel, KrigingModel, RbfModel, SurrogateModelBase,
                                          createModel)
from pythonLibs.regressionHandler.kernels import (AbsoluteExponential, Matern32, Matern52, Periodic,
                                                  PowerExponential, RationalQuadratic, SquaredExponential,
                                                  buildKernel)
from pythonLibs.regressionHandler.numerics.NeighborSearch import NeighborSearch, _bruteForce
from pythonLibs.regressionHandler.sampling import fullFactorial, latinHypercube, sobolLike


def branin(x):
    a, b, c, r, s, t = 1, 5.1 / (4 * np.pi ** 2), 5 / np.pi, 6, 10, 1 / (8 * np.pi)
    return a * (x[:, 1] - b * x[:, 0] ** 2 + c * x[:, 0] - r) ** 2 + s * (1 - t) * np.cos(x[:, 0]) + s


BRANIN_LIMITS = np.array([[-5.0, 10.0], [0.0, 15.0]])

KERNELS = [SquaredExponential(), Matern32(), Matern52(), AbsoluteExponential(), PowerExponential(power=1.5),
           RationalQuadratic(alpha=2.0), SquaredExponential(ard=False), Periodic(),
           Matern52() * Periodic(), SquaredExponential() + Matern32()]


@pytest.mark.parametrize("kernel", KERNELS, ids=lambda k: repr(k)[:40])
def test_kernelGradients(kernel):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 3))
    xb = rng.normal(size=(5, 3))
    kernel.setup(3)
    p = rng.uniform(-0.5, 0.5, kernel.nParams())
    k, grads = kernel.matrixAndGradients(x, p, kernel.trainingCache(x))
    np.testing.assert_allclose(k, kernel.matrix(x, x, p), atol=1e-14)
    np.testing.assert_allclose(np.diag(k), 1.0)
    assert np.linalg.eigvalsh(k).min() > -1e-10
    for i in range(p.size):
        dp = np.zeros_like(p)
        dp[i] = 1e-6
        fd = (kernel.matrix(x, x, p + dp) - kernel.matrix(x, x, p - dp)) / 2e-6
        np.testing.assert_allclose(grads[i], fd, atol=1e-7)
    for kx in range(3):
        xp, xm = xb.copy(), xb.copy()
        xp[:, kx] += 1e-6
        xm[:, kx] -= 1e-6
        fd = (kernel.matrix(xp, x, p) - kernel.matrix(xm, x, p)) / 2e-6
        np.testing.assert_allclose(kernel.dx(xb, x, p, kx), fd, atol=1e-7)


def test_kernelAliasesAndPls():
    assert type(buildKernel("rbf")) is SquaredExponential
    assert type(buildKernel({"type": "absoluteExponential"})) is AbsoluteExponential
    k = SquaredExponential().setup(5, np.random.default_rng(0).normal(size=(5, 2)))
    assert k.nParams() == 2


@pytest.mark.parametrize("likelihood", ["reml", "ml"])
@pytest.mark.parametrize("poly", ["none", "constant", "linear"])
def test_likelihoodGradient(likelihood, poly):
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, (25, 2))
    y = np.sin(x[:, 0]) + 0.1 * x[:, 1] ** 2 + rng.normal(0, 0.1, 25)
    m = KrigingModel(likelihood=likelihood, poly=poly, nStart=1, maxIter=1).fit(x, y)
    m._cache = m._kernel.trainingCache(m._xs)
    p = np.array([0.1, -0.2, np.log(0.05)])
    _, g = m._negLogLikelihood(p)
    fd = [(m._negLogLikelihood(p + d)[0] - m._negLogLikelihood(p - d)[0]) / 2e-6 for d in np.eye(3) * 1e-6]
    np.testing.assert_allclose(g, fd, atol=1e-6)


def test_krigingInterpolatesBranin():
    x = latinHypercube(30, BRANIN_LIMITS, seed=1)
    t = latinHypercube(400, BRANIN_LIMITS, seed=2)
    m = KrigingModel(nugget=1e-8).fit(x, branin(x))
    err = m.predict(t) - branin(t)
    assert np.sqrt(np.mean(err ** 2)) / np.std(branin(t)) < 0.05
    np.testing.assert_allclose(m.predict(x), branin(x), atol=1e-3 * np.ptp(branin(x)))
    h = 1e-3
    xp, xm = t[:10].copy(), t[:10].copy()
    xp[:, 0] += h
    xm[:, 0] -= h
    np.testing.assert_allclose(m.predictDerivatives(t[:10], 0)[:, 0], (m.predict(xp) - m.predict(xm)) / (2 * h),
                               atol=1e-4 * np.abs(branin(t)).max())


def test_krigingEstimatesNoiseAndCalibratesIntervals():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 10, 80))
    y = np.sin(x) + rng.normal(0, 0.2, 80)
    m = createModel("gp").fit(x, y)
    assert np.sqrt(m.hyperparameters["noiseVariance"]) == pytest.approx(0.2, rel=0.25)
    xn = rng.uniform(0, 10, 3000)
    yn = np.sin(xn) + rng.normal(0, 0.2, 3000)
    pi = m.predictInterval(xn, 0.95, "prediction")
    coverage = np.mean((pi.lower[:, 0] <= yn) & (yn <= pi.upper[:, 0]))
    assert 0.92 < coverage < 0.98
    assert m.metrics[0].nParams < 40


def test_kplsFewHyperparameters():
    rng = np.random.default_rng(5)
    a = rng.uniform(0.2, 1, 10)
    x = latinHypercube(80, np.tile([-1.0, 1.0], (10, 1)), seed=3)
    t = latinHypercube(300, np.tile([-1.0, 1.0], (10, 1)), seed=4)
    f = lambda v: v @ a + 0.5 * np.sin(2 * v @ a)
    m = KplsModel(plsComponents=1).fit(x, f(x))
    assert len(m.hyperparameters["logParams"]) == 2
    assert np.sqrt(np.mean((m.predict(t) - f(t)) ** 2)) / np.std(f(t)) < 0.5
    assert m.describe().startswith("kpls")


def test_rbfLooSmoothingAndDerivatives():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (80, 2))
    truth = lambda v: np.sin(3 * v[:, 0]) * np.cos(2 * v[:, 1])
    y = truth(x) + rng.normal(0, 0.1, 80)
    t = rng.uniform(-1, 1, (300, 2))
    smooth = RbfModel(smoothing="loo").fit(x, y)
    interp = RbfModel().fit(x, y)
    assert smooth.smoothingValue > 1e-6
    assert np.mean((smooth.predict(t) - truth(t)) ** 2) < np.mean((interp.predict(t) - truth(t)) ** 2)
    xp, xm = t[:5].copy(), t[:5].copy()
    xp[:, 1] += 1e-6
    xm[:, 1] -= 1e-6
    np.testing.assert_allclose(interp.predictDerivatives(t[:5], 1)[:, 0],
                               (interp.predict(xp) - interp.predict(xm)) / 2e-6, atol=1e-5)


def test_idw():
    x = np.random.default_rng(0).uniform(size=(50, 3))
    y = x.sum(axis=1)
    for m in (IdwModel(), IdwModel(neighbors=8)):
        m.fit(x, y)
        np.testing.assert_allclose(m.predict(x), y, atol=1e-12)
        mid = m.predict(np.full((1, 3), 0.5))[0]
        assert y.min() <= mid <= y.max()


@pytest.mark.parametrize("spec", ["gp", "kpls", "rbf-smooth", "idw", {"type": "rbf", "neighbors": 10}])
def test_roundTrip(spec):
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(40, 2))
    y = np.column_stack([np.sin(4 * x[:, 0]) + x[:, 1], x[:, 0] * x[:, 1]])
    m = createModel(spec).fit(x, y)
    loaded = SurrogateModelBase.fromDict(json.loads(json.dumps(m.toDict())))
    t = rng.uniform(size=(20, 2))
    np.testing.assert_allclose(loaded.predictValues(t), m.predictValues(t), atol=1e-12)
    if m.supports["variances"]:
        np.testing.assert_allclose(loaded.predictVariances(t), m.predictVariances(t), atol=1e-12)


def test_samplingDesigns():
    lim = np.array([[0.0, 1.0], [10.0, 20.0], [-1.0, 1.0]])
    for crit in ("random", "center", "maximin", "ese"):
        s = latinHypercube(12, lim, criterion=crit, seed=0, iterations=5)
        assert s.shape == (12, 3)
        strata = np.floor((s - lim[:, 0]) / (lim[:, 1] - lim[:, 0]) * 12).astype(int)
        for k in range(3):
            assert sorted(strata[:, k]) == list(range(12))
    assert fullFactorial([3, 2, 1], lim).shape == (6, 3)
    assert sobolLike(64, lim).shape == (64, 3)


def test_neighborSearchIsExact():
    rng = np.random.default_rng(0)
    for n, d, k in [(20000, 2, 15), (6000, 3, 10), (500, 7, 5)]:
        pts = rng.uniform(size=(n, d))
        q = np.vstack([pts[:3000], rng.uniform(-0.5, 1.5, (200, d))])
        dist, _ = NeighborSearch(pts).query(q, k)
        ref, _ = _bruteForce(pts, q, k)
        np.testing.assert_allclose(dist, ref, atol=1e-7)


def test_rbfSolvesInterpolationSystem():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (40, 2))
    y = np.sin(3 * x[:, 0]) * np.cos(2 * x[:, 1])
    m = RbfModel(kernel="thinPlateSpline").fit(x, y)
    np.testing.assert_allclose(m.predict(x), y, atol=1e-10)
    # polynomial reproduction: TPS with a linear tail interpolates planes exactly everywhere
    plane = 1 + 2 * x[:, 0] - x[:, 1]
    t = rng.uniform(-1, 1, (50, 2))
    np.testing.assert_allclose(RbfModel().fit(x, plane).predict(t), 1 + 2 * t[:, 0] - t[:, 1], atol=1e-9)
    local = RbfModel(neighbors=15).fit(x, plane)
    np.testing.assert_allclose(local.predict(t), 1 + 2 * t[:, 0] - t[:, 1], atol=1e-9)


def test_krigingOptimumIsStationary():
    x = latinHypercube(40, np.tile([-1.0, 1.0], (3, 1)), seed=3)
    y = np.sin(x @ np.array([0.5, 0.9, 0.3]) * 2) + np.random.default_rng(5).normal(0, 0.1, 40)
    m = KrigingModel(poly="none", likelihood="ml", nStart=3).fit(x, y)
    m._cache = m._kernel.trainingCache(m._xs)
    bounds = m._bounds()
    _, grad = m._negLogLikelihood(m._params)
    interior = (m._params > bounds[:, 0] + 1e-6) & (m._params < bounds[:, 1] - 1e-6)
    assert np.max(np.abs(grad[interior])) < 1e-3
