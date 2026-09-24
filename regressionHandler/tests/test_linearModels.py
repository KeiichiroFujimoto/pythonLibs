"""LinearBasisModel with every solver: estimates, inference, intervals, persistence."""
import json

import numpy as np
import pytest

from pythonLibs.regressionHandler import LinearBasisModel, SurrogateModelBase, createModel, crossValidate
from pythonLibs.regressionHandler.numerics.Distributions import StudentT


@pytest.fixture
def cubicData():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 40)
    y = 1 + 2 * x - 3 * x ** 2 + 0.5 * x ** 3 + rng.normal(0, 0.05, x.size)
    return x, y


def test_olsMatchesNormalEquations(cubicData):
    x, y = cubicData
    m = createModel("poly3").fit(x, y)
    a = np.vander(x, 4, increasing=True)
    coef = np.linalg.solve(a.T @ a, a.T @ y)
    resid = y - a @ coef
    s2 = resid @ resid / (40 - 4)
    cov = s2 * np.linalg.inv(a.T @ a)
    np.testing.assert_allclose(m.coefficients[:, 0], coef, rtol=1e-10)
    np.testing.assert_allclose(m.result.stdErrors[:, 0], np.sqrt(np.diag(cov)), rtol=1e-10)
    t = coef / np.sqrt(np.diag(cov))
    np.testing.assert_allclose(m.result.pValues[:, 0], 2 * StudentT(36).sf(np.abs(t)), rtol=1e-9)
    # prediction interval at x = 0.5
    xv = np.array([1, 0.5, 0.25, 0.125])
    half = StudentT(36).ppf(0.975) * np.sqrt(s2 + xv @ cov @ xv)
    pi = m.predictInterval([0.5], 0.95, "prediction")
    assert pi.upper[0, 0] == pytest.approx(xv @ coef + half, rel=1e-10)
    assert m.metrics[0].nParams == 4
    assert m.intervalDof == 36


def test_weightedFitEqualsDuplicatedRows():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.1, 1.2, 1.9, 3.2, 3.9])
    w = np.array([1.0, 2.0, 1.0, 3.0, 1.0])
    mw = createModel("linear").fit(x, y, weights=w)
    md = createModel("linear").fit(np.repeat(x, w.astype(int)), np.repeat(y, w.astype(int)))
    np.testing.assert_allclose(mw.coefficients, md.coefficients, rtol=1e-12)


def test_multiOutputAndDerivatives():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, (150, 3))
    y = np.column_stack([1 + x[:, 0] * x[:, 1] + x[:, 2] ** 2, 2 - x[:, 0]])
    m = createModel("quadratic").fit(x, y)
    assert m.predictValues(x[:4]).shape == (4, 2)
    np.testing.assert_allclose(m.predictValues(x), y, atol=1e-10)
    np.testing.assert_allclose(m.predictDerivatives(x[:5], 0)[:, 0], x[:5, 1], atol=1e-10)
    np.testing.assert_allclose(m.predictDerivatives(x[:5], 0)[:, 1], -1.0, atol=1e-10)
    assert m.predictGradient(x[:5]).shape == (5, 3, 2)


def test_ridgeGcvPSplineSmoothsNoise():
    rng = np.random.default_rng(2)
    x = np.sort(rng.uniform(0, 10, 300))
    y = np.sin(x) + rng.normal(0, 0.2, x.size)
    m = createModel("pspline").fit(x, y)
    assert np.sqrt(np.mean((m.predict(x) - np.sin(x)) ** 2)) < 0.06
    assert 5 < m.nEffectiveParams[0] < 20
    band = m.predictInterval(x, 0.95, "confidence")
    coverage = np.mean((band.lower[:, 0] <= np.sin(x)) & (np.sin(x) <= band.upper[:, 0]))
    assert coverage > 0.85


def test_robustResistsOutliers(cubicData):
    x, y = cubicData
    clean = createModel("poly3").fit(x, y).coefficients
    yo = y.copy()
    yo[[5, 20, 33]] += 3.0
    ols = createModel("poly3").fit(x, yo).coefficients
    rob = LinearBasisModel(basis={"type": "polynomial", "degree": 3},
                           solver={"type": "robust", "loss": "bisquare"}).fit(x, yo)
    assert np.abs(rob.coefficients - clean).max() < 0.1 * np.abs(ols - clean).max()
    assert set(np.flatnonzero(rob.solution.info["outlierMask"][:, 0])) == {5, 20, 33}


def test_lassoSelectsActiveTerms():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(120, 8))
    y = 3 * x[:, 0] - 2 * x[:, 3] + rng.normal(0, 0.1, 120)
    m = LinearBasisModel(basis={"type": "polynomial", "degree": 1},
                         solver={"type": "elasticNet", "alpha": 0.05}).fit(x, y)
    active = np.flatnonzero(np.abs(m.coefficients[1:, 0]) > 1e-8)
    assert set(active) == {0, 3}
    assert m.result is None and not m.supports["variances"]


def test_expressionBasisPhysicalCoefficients():
    x = np.linspace(1, 5, 30)
    y = 28 + 4 * x - 12 / x ** 2
    m = LinearBasisModel(basis={"type": "expression", "terms": ["u", "1/u**2"], "variables": ["u"]}).fit(x, y)
    np.testing.assert_allclose(m.coefficients[:, 0], [28, 4, -12], rtol=1e-8)
    assert m.equation().startswith("y0 = 28")


def test_highDegreeOrthogonalIsStable():
    x = np.linspace(300, 1300, 60)
    y = np.log(x) * np.sin(x / 150.0)
    m = createModel("ortho14").fit(x, y)
    assert m.metrics[0].rmse < 1e-3


def test_analyticLooEqualsRefitLoo(cubicData):
    x, y = cubicData
    m = createModel("poly3").fit(x, y)
    analytic = crossValidate(m, x, y, method="analytic")
    refit = crossValidate(m, x, y, nFolds=len(x))
    np.testing.assert_allclose(analytic.predictions, refit.predictions, rtol=1e-10)
    assert crossValidate(m, x, y, nFolds=5, nJobs=3).rmse[0] > 0


def test_jsonRoundTrip(tmp_path, cubicData):
    x, y = cubicData
    for spec in ("poly3", "pspline", "robust-poly2", "rbf-ridge", "lasso-poly2"):
        m = createModel(spec).fit(x, y)
        path = tmp_path / f"{spec}.json"
        m.save(str(path))
        loaded = SurrogateModelBase.load(str(path))
        np.testing.assert_allclose(loaded.predictValues(x), m.predictValues(x))
        if m.supports["variances"]:
            np.testing.assert_allclose(loaded.predictInterval(x).upper, m.predictInterval(x).upper)
        assert json.loads(path.read_text())["type"] == "linearBasis"


def test_errorsAreExplicit():
    with pytest.raises(ValueError):
        createModel("poly5").fit(np.arange(4.0), np.arange(4.0))
    with pytest.raises(RuntimeError):
        createModel("linear").predict([1.0])
    with pytest.raises(KeyError):
        createModel("noSuchModel")
    with pytest.raises(ValueError):
        createModel({"type": "linearBasis", "solver": {"type": "ridge", "alpha": "aic"}})
    m = createModel("linear").fit([0.0, 1.0, 2.0], [0.0, 1.0, 2.1])
    with pytest.raises(ValueError):
        m.predict(np.zeros((2, 3)))
