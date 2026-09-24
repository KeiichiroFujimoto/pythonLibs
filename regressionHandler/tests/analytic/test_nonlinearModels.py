"""Nonlinear least squares: exact recovery on noise-free data and the asymptotic covariance formula."""
import numpy as np
import pytest

from pythonLibs.regressionHandler import NonlinearModel, createModel
from pythonLibs.regressionHandler.models.ModelLibrary import LIBRARY

TRUTH = {
    "linear": ([1, 2], (0, 5)), "powerLaw": ([2, 1.5], (0.5, 5)), "powerLawOffset": ([2, 1.5, 3], (0.5, 5)),
    "exponential": ([2, -0.7], (0, 5)), "exponentialDecay": ([5, 1.2, 1], (0, 6)),
    "exponentialRise": ([4, 0.8, 1], (0, 5)), "inverseExponential": ([2e-3, 1800], (280, 900)),
    "logistic": ([3, 2, 1, 0.5], (-3, 5)), "saturation": ([5, 2], (0.1, 10)),
    "powerSaturation": ([5, 2, 2.5], (0.1, 6)), "gaussianPeak": ([4, 2, 0.5, 1], (0, 4)),
    "weibullCdf": ([3, 2], (0.2, 8)), "sinusoid": ([2, 0.7, 0.4, 1], (0, 10)),
    "rationalPower": ([3.0, 1.5, 4.0], (0.5, 40)), "linearPlusPower": ([2.0, 0.05, 3.0], (0.1, 10)),
    "logarithmic": ([1, 2], (0.5, 10)), "inverse": ([1, 2], (0.5, 10)),
}


def _truth(name, params, x):
    m = NonlinearModel(library=name, p0=list(params)).fit(x, np.ones_like(x))
    m._p = np.array(params, dtype=float)
    return m.predict(x)


def test_everyLibraryFormHasAnAnalyticCase():
    assert set(TRUTH) == set(LIBRARY)


@pytest.mark.parametrize("name", sorted(TRUTH))
def test_exactRecoveryFromAutomaticInitialGuess(name):
    params, (lo, hi) = TRUTH[name]
    x = np.linspace(lo, hi, 60 if name == "sinusoid" else 40)
    m = NonlinearModel(library=name).fit(x, _truth(name, params, x))
    np.testing.assert_allclose(list(m.parameters.values()), params, rtol=1e-5)


def test_multivariateExpressionExactRecovery():
    x = np.random.default_rng(1).uniform(1, 3, (50, 2))
    y = 2 * x[:, 0] ** 1.3 * x[:, 1] ** -0.5
    m = NonlinearModel(expression="c*u**a*v**b", params=["c", "a", "b"], variables=["u", "v"], p0=[1, 1, 0]).fit(x, y)
    np.testing.assert_allclose(list(m.parameters.values()), [2, 1.3, -0.5], rtol=1e-8)


def test_normalEquationsAndCovariance():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 6, 40)
    y = 5 * np.exp(-x / 1.2) + 1 + rng.normal(0, 0.05, 40)
    m = createModel("exponentialDecay").fit(x, y)
    a, tau, c = m.parameters.values()
    fitted = a * np.exp(-x / tau) + c
    jac = np.column_stack([np.exp(-x / tau), a * x / tau ** 2 * np.exp(-x / tau), np.ones_like(x)])
    np.testing.assert_allclose(jac.T @ (y - fitted), 0.0, atol=1e-6)          # first-order optimality
    s2 = np.sum((y - fitted) ** 2) / (40 - 3)
    cov = s2 * np.linalg.inv(jac.T @ jac)                                    # s^2 (J'J)^-1
    np.testing.assert_allclose(m.result.stdErrors[:, 0], np.sqrt(np.diag(cov)), rtol=1e-4)
    g = jac[[10]]
    band = m.predictInterval(x[[10]], 0.95, "confidence")
    assert band.std[0, 0] == pytest.approx(np.sqrt(g @ cov @ g.T)[0, 0], rel=1e-4)  # delta method


def test_linearModelAsNonlinearMatchesOls():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 3, 30)
    y = 1 + 0.5 * x + rng.normal(0, 0.1, 30)
    nl = createModel("linear")
    lin = NonlinearModel(library="linear").fit(x, y)
    ols = nl.fit(x, y)
    np.testing.assert_allclose(list(lin.parameters.values()), ols.coefficients[:, 0], rtol=1e-8)
    np.testing.assert_allclose(lin.result.stdErrors, ols.result.stdErrors, rtol=1e-5)
