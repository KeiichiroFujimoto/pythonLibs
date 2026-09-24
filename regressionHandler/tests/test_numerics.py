"""Special functions, distributions and optimizers against reference values."""
import numpy as np
import pytest

from pythonLibs.regressionHandler.numerics import SpecialFunctions as sf
from pythonLibs.regressionHandler.numerics.Distributions import ChiSquared, FDistribution, Normal, StudentT
from pythonLibs.regressionHandler.numerics.LinearAlgebra import (choSolve, choleskyWithJitter, leastSquaresSvd,
                                                                 solveTriangular)
from pythonLibs.regressionHandler.numerics.Optimizers import leastSquares, minimize, minimizeScalar, multiStart

# Reference values (independent high-precision evaluations).
REF = [
    (lambda: sf.gammaln(5.0), np.log(24.0)),
    (lambda: sf.gammaln(0.1), 2.252712651734206),
    (lambda: sf.betainc(2.0, 3.0, 0.4), 0.5248),
    (lambda: sf.gammainc(2.5, 1.7), 0.36143007689620493),
    (lambda: sf.gammaincc(2.5, 30.0), 1.2154569777183007e-11),
    (lambda: sf.erf(0.5), 0.5204998778130465),
    (lambda: sf.ndtri(0.975), 1.959963984540054),
    (lambda: sf.ndtri(1e-10), -6.361340902404056),
    (lambda: StudentT(12).ppf(0.975), 2.178812829667228),
    (lambda: StudentT(3).sf(4.5), 0.0102452061722267),
    (lambda: StudentT(1).cdf(-2.0), 0.14758361765043326),
    (lambda: ChiSquared(3).ppf(0.95), 7.814727903251178),
    (lambda: ChiSquared(10).sf(25.0), 0.005345505487134069),
    (lambda: FDistribution(3, 40).ppf(0.95), 2.838745398020641),
    (lambda: FDistribution(5, 12).sf(3.1), 0.05027370073306381),
    (lambda: Normal(2.0, 3.0).ppf(0.1), -1.844654696633801),
]


@pytest.mark.parametrize("fn,expected", REF)
def test_referenceValues(fn, expected):
    assert fn() == pytest.approx(expected, rel=1e-11, abs=1e-14)


def test_quantileRoundTrip():
    q = np.array([1e-9, 1e-4, 0.025, 0.3, 0.5, 0.8, 0.975, 1 - 1e-7])
    for dist in (StudentT(0.8), StudentT(7.5), StudentT(400), ChiSquared(0.6), ChiSquared(35), FDistribution(2, 9)):
        np.testing.assert_allclose(dist.cdf(dist.ppf(q)), q, rtol=1e-9)


def test_vectorAndScalarPathsAgree():
    a = np.array([0.3, 2.0, 40.0])
    b = np.array([5.0, 0.7, 60.0])
    x = np.array([0.2, 0.9, 0.45])
    vec = sf.betainc(a, b, x)
    np.testing.assert_allclose(vec, [sf.betainc(*v) for v in zip(a, b, x)], rtol=1e-13)


def test_triangularAndCholesky():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(300, 300))
    k = a @ a.T + 300 * np.eye(300)
    lower, jitter = choleskyWithJitter(k)
    assert jitter == 0.0
    b = rng.normal(size=(300, 4))
    np.testing.assert_allclose(k @ choSolve(lower, b), b, atol=1e-9)
    np.testing.assert_allclose(lower.T @ solveTriangular(lower, b[:, 0], trans=True), b[:, 0], atol=1e-9)


def test_leastSquaresRankDeficient():
    x = np.linspace(0, 1, 20)
    a = np.column_stack([np.ones(20), x, 2 * x])
    sol = leastSquaresSvd(a, 1 + 3 * x)
    assert sol.rank == 2
    np.testing.assert_allclose(a @ sol.coef, 1 + 3 * x, atol=1e-12)


def test_levenbergMarquardtRosenbrock():
    res = leastSquares(lambda v: np.array([10 * (v[1] - v[0] ** 2), 1 - v[0]]), [-1.2, 1.0])
    assert res.success
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-8)


def test_leastSquaresBoundsAndRobustLoss():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 5, 60)
    y = 3 * np.exp(-1.3 * t) + 0.5 + rng.normal(0, 0.02, t.size)
    res = leastSquares(lambda p: p[0] * np.exp(-p[1] * t) + p[2] - y, [1, 1, 0], bounds=([0, 0, 0.6], [10, 10, 10]))
    assert res.x[2] == pytest.approx(0.6)
    y[[10, 30, 50]] += 2.0
    lin = leastSquares(lambda p: p[0] * np.exp(-p[1] * t) + p[2] - y, [1, 1, 0])
    rob = leastSquares(lambda p: p[0] * np.exp(-p[1] * t) + p[2] - y, [1, 1, 0], loss="soft_l1", fScale=0.05)
    truth = np.array([3.0, 1.3, 0.5])
    assert np.linalg.norm(rob.x - truth) < 0.2 * np.linalg.norm(lin.x - truth)


def test_minimizeMethods():
    f = lambda v: (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2
    g = lambda v: np.array([-2 * (1 - v[0]) - 400 * v[0] * (v[1] - v[0] ** 2), 200 * (v[1] - v[0] ** 2)])
    np.testing.assert_allclose(minimize(f, [-1.2, 1], jac=g).x, [1, 1], atol=1e-5)
    np.testing.assert_allclose(minimize(f, [-1.2, 1], method="Nelder-Mead", tol=1e-10).x, [1, 1], atol=1e-6)
    bounded = minimize(f, [-1.2, 1], jac=g, bounds=[(-2, 0.5), (None, None)])
    np.testing.assert_allclose(bounded.x, [0.5, 0.25], atol=1e-6)
    assert minimizeScalar(lambda v: (v - 2.3) ** 2, (0, 10)).x[0] == pytest.approx(2.3, abs=1e-7)
    best = multiStart(lambda s: minimize(lambda v: np.sin(3 * v[0]) + 0.1 * v[0] ** 2, s), np.array([[-3.0], [0.5]]))
    assert best.fun < -0.9


def test_quantilesAgreeWithSimulation():
    rng = np.random.default_rng(0)
    for dist, draws in ((StudentT(5.0), rng.standard_t(5.0, 400_000)), (ChiSquared(3.0), rng.chisquare(3.0, 400_000))):
        q = np.array([0.05, 0.5, 0.95])
        np.testing.assert_allclose(dist.ppf(q), np.quantile(draws, q), rtol=0.02, atol=0.01)
