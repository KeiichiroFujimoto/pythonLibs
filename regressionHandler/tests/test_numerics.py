"""Numerical kernels: scalar/vector consistency, bounded and robust least squares, multi-start."""
import numpy as np
import pytest

from pythonLibs.regressionHandler.numerics import SpecialFunctions as sf
from pythonLibs.regressionHandler.numerics.Optimizers import leastSquares, minimize, minimizeScalar, multiStart


def test_vectorAndScalarPathsAgree():
    a = np.array([0.3, 2.0, 40.0])
    b = np.array([5.0, 0.7, 60.0])
    x = np.array([0.2, 0.9, 0.45])
    vec = sf.betainc(a, b, x)
    np.testing.assert_allclose(vec, [sf.betainc(*v) for v in zip(a, b, x)], rtol=1e-13)


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
