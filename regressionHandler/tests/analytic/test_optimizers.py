"""Optimizers and linear algebra on problems with known exact solutions."""
import numpy as np
import pytest

from pythonLibs.regressionHandler.numerics.LinearAlgebra import (choSolve, choleskyWithJitter, leastSquaresSvd,
                                                                 solveTriangular)
from pythonLibs.regressionHandler.numerics.Optimizers import leastSquares, minimize, minimizeScalar


def test_quadraticMinimumIsLinearSolve():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, 40))
    h = a @ a.T + 40 * np.eye(40)
    b = rng.normal(size=40)
    res = minimize(lambda v: (0.5 * v @ h @ v - b @ v, h @ v - b), np.zeros(40), jac=True, tol=1e-10)
    np.testing.assert_allclose(res.x, np.linalg.solve(h, b), atol=1e-7)


def test_boundedQuadraticIsProjection():
    # min |x - c|^2 on a box is the projection of c onto the box
    c = np.array([2.0, -3.0, 0.5, 7.0])
    lb, ub = np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 5.0])
    res = minimize(lambda v: (np.sum((v - c) ** 2), 2 * (v - c)), np.zeros(4), jac=True,
                   bounds=list(zip(lb, ub)), tol=1e-12)
    np.testing.assert_allclose(res.x, np.clip(c, lb, ub), atol=1e-9)


def test_rosenbrockMinimum():
    f = lambda v: (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2
    g = lambda v: np.array([-2 * (1 - v[0]) - 400 * v[0] * (v[1] - v[0] ** 2), 200 * (v[1] - v[0] ** 2)])
    np.testing.assert_allclose(minimize(f, [-1.2, 1], jac=g).x, [1, 1], atol=1e-5)
    np.testing.assert_allclose(minimize(f, [-1.2, 1], method="Nelder-Mead", tol=1e-10).x, [1, 1], atol=1e-6)
    lm = leastSquares(lambda v: np.array([10 * (v[1] - v[0] ** 2), 1 - v[0]]), [-1.2, 1.0])
    np.testing.assert_allclose(lm.x, [1, 1], atol=1e-10)


def test_brentOnParabolaAndBoundary():
    assert minimizeScalar(lambda v: (v - 2.3) ** 2 + 1, (0, 10)).x[0] == pytest.approx(2.3, abs=1e-7)
    assert minimizeScalar(lambda v: v, (1.0, 4.0)).x[0] == pytest.approx(1.0, abs=1e-6)   # minimum at the bound


def test_nonlinearLeastSquaresExactData():
    t = np.linspace(0, 5, 50)
    y = 3 * np.exp(-1.3 * t) + 0.5
    for loss in ("linear", "soft_l1", "huber", "cauchy", "arctan"):
        res = leastSquares(lambda p: p[0] * np.exp(-p[1] * t) + p[2] - y, [1, 1, 0], loss=loss)
        np.testing.assert_allclose(res.x, [3, 1.3, 0.5], rtol=1e-8)


def test_linearAlgebraIdentities():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(300, 300))
    k = a @ a.T + 300 * np.eye(300)
    lower, jitter = choleskyWithJitter(k)
    assert jitter == 0.0
    np.testing.assert_allclose(lower @ lower.T, k, atol=1e-9)
    b = rng.normal(size=(300, 3))
    np.testing.assert_allclose(k @ choSolve(lower, b), b, atol=1e-9)
    np.testing.assert_allclose(lower @ solveTriangular(lower, b), b, atol=1e-9)
    np.testing.assert_allclose(lower.T @ solveTriangular(lower, b, trans=True), b, atol=1e-9)
    # rank-deficient system: an exact solution, minimum-norm in equilibrated coordinates
    x = np.linspace(0, 1, 20)
    design = np.column_stack([np.ones(20), x, 2 * x])
    sol = leastSquaresSvd(design, 1 + 3 * x)
    assert sol.rank == 2
    np.testing.assert_allclose(design @ sol.coef, 1 + 3 * x, atol=1e-12)
    scaled = design / sol.colScale
    np.testing.assert_allclose(sol.coef * sol.colScale, np.linalg.pinv(scaled) @ (1 + 3 * x), atol=1e-12)
    # full-rank: equals the normal-equation solution
    full = np.column_stack([np.ones(20), x, x ** 2])
    np.testing.assert_allclose(leastSquaresSvd(full, np.exp(x)).coef,
                               np.linalg.solve(full.T @ full, full.T @ np.exp(x)), rtol=1e-10)
