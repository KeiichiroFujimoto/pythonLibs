"""Residual diagnostics against their defining formulas and the nominal size of the tests."""
import numpy as np
import pytest

from pythonLibs.regressionHandler import createModel, diagnose
from pythonLibs.regressionHandler.evaluation.Diagnostics import (breuschPagan, durbinWatson, jarqueBera, normalTest,
                                                                 runsTest)
from pythonLibs.regressionHandler.numerics.Distributions import ChiSquared, Normal


def test_influenceMeasuresFromHatMatrix():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 60)
    y = 1 + 2 * x - 3 * x ** 2 + rng.normal(0, 0.05, 60)
    y[10] += 0.6
    d = diagnose(createModel("quadratic").fit(x, y)).outputs[0]
    a = np.column_stack([np.ones_like(x), x, x ** 2])
    hat = a @ np.linalg.solve(a.T @ a, a.T)
    h, e = np.diag(hat), y - hat @ y
    n, p = a.shape
    s2 = e @ e / (n - p)
    internal = e / np.sqrt(s2 * (1 - h))
    np.testing.assert_allclose(d["leverage"], h, atol=1e-12)
    np.testing.assert_allclose(d["cooksDistance"], internal ** 2 * h / (p * (1 - h)), atol=1e-12)
    np.testing.assert_allclose(d["studentized"], internal * np.sqrt((n - p - 1) / (n - p - internal ** 2)), atol=1e-10)
    assert d["outliers"] == [10]
    assert d["durbinWatson"] == pytest.approx(np.sum(np.diff(e) ** 2) / np.sum(e * e), rel=1e-12)


def test_statisticsFromDefinitions():
    r = np.random.default_rng(1).standard_t(5, size=150)
    d = r - r.mean()
    s = np.mean(d ** 3) / np.mean(d ** 2) ** 1.5
    k = np.mean(d ** 4) / np.mean(d ** 2) ** 2
    jb = len(r) / 6 * (s * s + (k - 3) ** 2 / 4)
    assert jarqueBera(r) == pytest.approx((jb, ChiSquared(2).sf(jb)), rel=1e-12)
    x = np.random.default_rng(2).uniform(size=(150, 2))
    e2 = r * r
    a = np.column_stack([np.ones(150), x])
    coef = np.linalg.lstsq(a, e2, rcond=None)[0]
    lm = 150 * (1 - np.sum((e2 - a @ coef) ** 2) / np.sum((e2 - e2.mean()) ** 2))
    assert breuschPagan(r, x) == pytest.approx((lm, ChiSquared(2).sf(lm)), rel=1e-10)
    assert durbinWatson(np.array([1.0, -1.0, 1.0, -1.0])) == pytest.approx(12.0 / 4.0)


def test_runsTestMoments():
    signs = np.array([1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1], dtype=float)
    n1, n2 = 6, 6
    runs = 1 + np.sum(signs[1:] != signs[:-1])
    mu = 2 * n1 * n2 / 12 + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - 12) / (12 ** 2 * 11)
    z, p = runsTest(signs, np.arange(12))
    assert z == pytest.approx((runs - mu) / np.sqrt(var), rel=1e-12)
    assert p == pytest.approx(2 * Normal().sf(abs(z)), rel=1e-12)


def test_normalityTestsHaveNominalSize():
    """Under normal data the tests reject at (approximately) their nominal 5 % level."""
    rng = np.random.default_rng(3)
    samples = rng.normal(size=(2000, 200))
    k2 = np.array([normalTest(s)[1] for s in samples])
    assert 0.035 < np.mean(k2 < 0.05) < 0.065
    heavy = rng.standard_t(3, size=(200, 200))
    assert np.mean([normalTest(s)[1] < 0.05 for s in heavy]) > 0.8   # power against heavy tails
