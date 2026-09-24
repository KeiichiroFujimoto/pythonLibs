"""Linear-in-parameter models against their closed-form solutions."""
import numpy as np
import pytest

from pythonLibs.regressionHandler import LinearBasisModel, SplineModel, createModel, crossValidate
from pythonLibs.regressionHandler.numerics.Distributions import StudentT


def _design(x):
    return np.vander(x, 4, increasing=True)


def test_noiseFreePolynomialIsRecoveredExactly():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (60, 3))
    coef = {(0, 0, 0): 1.5, (1, 0, 0): -2.0, (0, 1, 1): 0.7, (0, 0, 2): 3.0, (2, 1, 0): -0.4}
    y = sum(c * np.prod(x ** np.array(e), axis=1) for e, c in coef.items())
    m = createModel("poly3").fit(x, y)
    exps = [tuple(e) for e in m.basis.exponents]
    est = dict(zip(exps, m.coefficients[:, 0]))
    for e in exps:
        assert est[e] == pytest.approx(coef.get(e, 0.0), abs=1e-10)
    assert m.metrics[0].rSquared == pytest.approx(1.0)


def test_olsInferenceFormulas():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 40)
    y = 1 + 2 * x - 3 * x ** 2 + 0.5 * x ** 3 + rng.normal(0, 0.05, 40)
    m = createModel("poly3").fit(x, y)
    a = _design(x)
    beta = np.linalg.solve(a.T @ a, a.T @ y)                     # (X'X)^-1 X'y
    resid = y - a @ beta
    s2 = resid @ resid / (40 - 4)
    cov = s2 * np.linalg.inv(a.T @ a)
    np.testing.assert_allclose(m.coefficients[:, 0], beta, rtol=1e-10)
    np.testing.assert_allclose(m.result.stdErrors[:, 0], np.sqrt(np.diag(cov)), rtol=1e-10)
    t = beta / np.sqrt(np.diag(cov))
    np.testing.assert_allclose(m.result.pValues[:, 0], 2 * StudentT(36).sf(np.abs(t)), rtol=1e-10)
    lo, hi = m.result.confInt(0.9)
    q = StudentT(36).ppf(0.95)
    np.testing.assert_allclose(hi[:, 0] - lo[:, 0], 2 * q * np.sqrt(np.diag(cov)), rtol=1e-10)
    x0 = np.array([1, 0.5, 0.25, 0.125])
    for kind, var in (("confidence", x0 @ cov @ x0), ("prediction", s2 + x0 @ cov @ x0)):
        band = m.predictInterval([0.5], 0.95, kind)
        assert band.upper[0, 0] - band.mean[0, 0] == pytest.approx(StudentT(36).ppf(0.975) * np.sqrt(var), rel=1e-10)
    # overall F statistic identity: R^2 = 1 - SSE / SST
    assert m.metrics[0].rSquared == pytest.approx(1 - resid @ resid / np.sum((y - y.mean()) ** 2), rel=1e-12)


def test_weightedLeastSquaresFormula():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, 30)
    y = 1 + 2 * x + rng.normal(0, 0.1, 30)
    w = rng.uniform(0.2, 5, 30)
    a = np.column_stack([np.ones(30), x])
    beta = np.linalg.solve(a.T @ (w[:, None] * a), a.T @ (w * y))   # (X'WX)^-1 X'Wy
    np.testing.assert_allclose(createModel("linear").fit(x, y, weights=w).coefficients[:, 0], beta, rtol=1e-11)
    wi = np.array([1, 2, 1, 3, 1])
    xi, yi = np.arange(5.0), np.array([0.1, 1.2, 1.9, 3.2, 3.9])
    np.testing.assert_allclose(createModel("linear").fit(xi, yi, weights=wi).coefficients,
                               createModel("linear").fit(np.repeat(xi, wi), np.repeat(yi, wi)).coefficients, rtol=1e-12)


def test_ridgeClosedForm():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(50, 3))
    y = x @ np.array([1.0, -2.0, 0.5]) + 3 + rng.normal(0, 0.3, 50)
    alpha = 0.7
    m = LinearBasisModel(basis={"type": "polynomial", "degree": 1}, solver={"type": "ridge", "alpha": alpha}).fit(x, y)
    a = np.column_stack([np.ones(50), x])
    pen = np.diag(np.r_[0.0, x.var(axis=0) * 50])                   # scale-free ridge, intercept free
    beta = np.linalg.solve(a.T @ a + alpha * pen, a.T @ y)
    np.testing.assert_allclose(m.coefficients[:, 0], beta, rtol=1e-9)
    hat = a @ np.linalg.solve(a.T @ a + alpha * pen, a.T)
    assert m.nEffectiveParams[0] == pytest.approx(np.trace(hat), rel=1e-9)


def test_lassoOnOrthonormalDesignIsSoftThresholding():
    # With orthonormal standardized columns the Lasso solution is S(z, alpha) exactly.
    n = 64
    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.normal(size=(n, 4)) - 0)
    q = q - q.mean(axis=0)
    q, _ = np.linalg.qr(q)
    x = q * np.sqrt(n)                                               # columns: mean 0, variance 1, orthogonal
    x = x - x.mean(axis=0)
    x = x / x.std(axis=0)
    x, _ = np.linalg.qr(x)
    x = x * np.sqrt(n)
    y = x @ np.array([2.0, -0.05, 0.8, 0.0]) + 5.0
    alpha = 0.1
    m = LinearBasisModel(basis={"type": "polynomial", "degree": 1},
                         solver={"type": "elasticNet", "alpha": alpha, "tol": 1e-14}).fit(x, y)
    z = x.T @ (y - y.mean()) / n / x.std(axis=0)
    expected = np.sign(z) * np.maximum(np.abs(z) - alpha, 0.0) / x.std(axis=0)
    np.testing.assert_allclose(m.coefficients[1:, 0], expected, atol=1e-10)


def test_psplineReproducesPenaltyNullSpace():
    # A difference penalty of order k does not act on polynomials of degree < k,
    # so they are reproduced exactly for any smoothing weight.
    x = np.linspace(0, 10, 120)
    for alpha in (1e-3, 1.0, 1e4):
        line = SplineModel(nSegments=15, alpha=alpha, penaltyOrder=2).fit(x, 2 - 0.3 * x)
        np.testing.assert_allclose(line.predict(x), 2 - 0.3 * x, atol=1e-9)
        quad = SplineModel(nSegments=15, alpha=alpha, penaltyOrder=3).fit(x, 1 + x - 0.1 * x ** 2)
        np.testing.assert_allclose(quad.predict(x), 1 + x - 0.1 * x ** 2, atol=1e-8)


def test_looShortcutEqualsRefit():
    rng = np.random.default_rng(5)
    x = np.linspace(0, 1, 40)
    y = np.sin(3 * x) + rng.normal(0, 0.05, 40)
    # PRESS identity e_(i) = e_i / (1 - h_ii) holds exactly for ordinary least squares
    for spec in ("poly3", "quadratic", {"type": "linearBasis", "basis": {"type": "orthogonalPolynomial", "degree": 5},
                                        "solver": "ols"}):
        m = createModel(spec).fit(x, y)
        np.testing.assert_allclose(crossValidate(m, x, y, method="analytic").predictions,
                                   crossValidate(m, x, y, nFolds=40).predictions, rtol=1e-9)


def test_robustRegressionRejectsGrossOutliersExactly():
    x = np.linspace(0, 1, 50)
    y = 1 + 2 * x
    y[[3, 17, 40]] += np.array([5.0, -4.0, 6.0])
    m = LinearBasisModel(basis={"type": "polynomial", "degree": 1}, solver={"type": "robust", "loss": "bisquare"})
    np.testing.assert_allclose(m.fit(x, y).coefficients[:, 0], [1, 2], atol=1e-8)


def test_analyticGradients():
    rng = np.random.default_rng(6)
    x = rng.uniform(-1, 1, (30, 2))
    y = 1 + x[:, 0] ** 2 * x[:, 1] + np.sin(0 * x[:, 0])
    m = createModel("poly3").fit(x, y)
    np.testing.assert_allclose(m.predictDerivatives(x, 0)[:, 0], 2 * x[:, 0] * x[:, 1], atol=1e-10)
    np.testing.assert_allclose(m.predictDerivatives(x, 1)[:, 0], x[:, 0] ** 2, atol=1e-10)
