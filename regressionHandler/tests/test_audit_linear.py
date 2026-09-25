"""Regression tests for defects found by the audit of the linear / GLM / statistical models.

Every expected value comes from an external reference (closed form, finite
differences of the model's own predictions, statsmodels, mpmath) rather than
from constants of this package.
"""
import numpy as np
import pytest

from pythonLibs.regressionHandler import SurrogateModelBase
from pythonLibs.regressionHandler.glm.Families import _digammaTrigammaDifferences
from pythonLibs.regressionHandler.models.ConstrainedModel import ConstrainedModel
from pythonLibs.regressionHandler.models.DimensionlessModel import DimensionlessModel
from pythonLibs.regressionHandler.models.GamModel import GamModel
from pythonLibs.regressionHandler.models.GlmModel import GlmModel
from pythonLibs.regressionHandler.models.MixedModel import MixedModel
from pythonLibs.regressionHandler.models.QuantileModel import QuantileModel
from pythonLibs.regressionHandler.models.TransformedTargetModel import TransformedTargetModel


# ---------------------------------------------------------------- negative binomial theta
def test_negativeBinomialDigammaDifferencesMatchMpmath():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    y = np.array([0.0, 1.0, 3.0, 17.0, 250.0, 1e4])
    for t in (5.0, 1e4, 3e5, 1e8):
        dg, tg = _digammaTrigammaDifferences(y, t)
        refDg = np.array([float(mp.digamma(mp.mpf(v) + t) - mp.digamma(t)) for v in y])
        refTg = np.array([float(mp.polygamma(1, mp.mpf(v) + t) - mp.polygamma(1, t)) for v in y])
        nz = refDg != 0
        np.testing.assert_allclose(dg[nz], refDg[nz], rtol=1e-12)
        np.testing.assert_allclose(tg[nz], refTg[nz], rtol=1e-11)


def _countThetaUpdates(monkeypatch):
    from pythonLibs.regressionHandler.glm.Families import NegativeBinomial
    calls = []
    original = NegativeBinomial.updateTheta

    def counting(self, *args, **kwargs):
        original(self, *args, **kwargs)
        calls.append(self.theta)
    monkeypatch.setattr(NegativeBinomial, "updateTheta", counting)
    return calls


def _poissonData():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, (300, 1))
    return x, rng.poisson(np.exp(1 + x[:, 0])).astype(float)


def test_negativeBinomialThetaSettlesOnEquidispersedData(monkeypatch):
    # Poisson data: the NB likelihood increases monotonically in theta. Round-off in the old
    # score made theta oscillate (2.6e7 <-> 1e8) through all 50 outer iterations.
    calls = _countThetaUpdates(monkeypatch)
    x, y = _poissonData()
    m = GlmModel(family="negativeBinomial").fit(x, y)
    assert len(calls) < 10
    assert m.family.theta > 1e6                                   # the Poisson limit
    sm = pytest.importorskip("statsmodels.api")
    ref = sm.GLM(y, np.column_stack([np.ones(y.size), x]), family=sm.families.Poisson()).fit(tol=1e-12)
    np.testing.assert_allclose(m.coefficients, ref.params, rtol=1e-5)


def test_negativeBinomialGamThetaSettlesOnEquidispersedData(monkeypatch):
    calls = _countThetaUpdates(monkeypatch)
    x, y = _poissonData()
    GamModel(family="negativeBinomial").fit(x, y)
    assert len(calls) < 10


# ---------------------------------------------------------------- transformed target
@pytest.mark.parametrize("transform", ["boxcox", "yeoJohnson", "log"])
def test_transformedTargetMeanRetransformDerivativeMatchesPredictions(transform):
    rng = np.random.default_rng(10)
    x = rng.uniform(0, 2, (100, 1))
    y = np.exp(0.5 + 0.8 * x[:, 0] + rng.normal(0, 0.3, 100))
    m = TransformedTargetModel(transform=transform, retransform="mean").fit(x, y)
    h = 1e-6
    fd = (m.predict(x[:5] + h) - m.predict(x[:5] - h)) / (2 * h)
    np.testing.assert_allclose(m.predictDerivatives(x[:5], 0)[:, 0], fd, rtol=1e-6)


def test_transformedTargetLoadKeepsVarianceSupportOfInnerModel(tmp_path):
    rng = np.random.default_rng(10)
    x = rng.uniform(0, 2, (50, 1))
    y = np.exp(0.5 + 0.8 * x[:, 0] + rng.normal(0, 0.3, 50))
    m = TransformedTargetModel(transform="log", model={"type": "linearBasis", "solver": "elasticNet"}).fit(x, y)
    assert not m.supports["variances"]
    m.save(str(tmp_path / "m.json"))
    assert not SurrogateModelBase.load(str(tmp_path / "m.json")).supports["variances"]


# ---------------------------------------------------------------- quantile regression
def test_quantileNidStandardErrorsAreUndefinedNotZeroAtExtremeTau():
    # n = 100, tau = 0.005: the fits at tau +/- h coincide, so every local density estimate is 0
    # (R's summary.rq refuses this case); the old code returned standard errors of exactly 0.
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 3, (100, 1))
    y = 1 + x[:, 0] + rng.normal(0, 1, 100)
    m = QuantileModel(tau=0.005, se="nid").fit(x, y)
    assert np.all(np.isnan(m.result.stdErrors))
    assert "seWarning" in m._info
    # an interior quantile keeps finite, positive standard errors
    m = QuantileModel(tau=0.5, se="nid").fit(x, y)
    assert np.all(np.isfinite(m.result.stdErrors)) and np.all(m.result.stdErrors > 0)


# ---------------------------------------------------------------- constrained model persistence
def test_constrainedModelWithArrayConstraintPointsSaves(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, (40, 1))
    y = np.sin(3 * x[:, 0]) + rng.normal(0, 0.1, 40)
    grid = np.linspace(0, 1, 11)[:, None]                          # the documented usage: a numpy grid
    m = ConstrainedModel(constraints=[{"type": "bound", "points": grid, "lower": np.float64(0.2)}]).fit(x, y)
    m.save(str(tmp_path / "c.json"))
    m2 = SurrogateModelBase.load(str(tmp_path / "c.json"))
    np.testing.assert_allclose(m2.predict(grid), m.predict(grid), rtol=0, atol=1e-12)
    assert m.predict(grid).min() >= 0.2 - 1e-9


# ---------------------------------------------------------------- dimensionless model
def _pendulum(n, seed=14):
    rng = np.random.default_rng(seed)
    length, g = rng.uniform(0.5, 2, n), rng.uniform(9, 10, n)
    period = 2 * np.pi * np.sqrt(length / g) * (1 + rng.normal(0, 0.01, n))
    return np.column_stack([length, g]), period


def test_dimensionlessConstantModelVarianceAfterLoadWithoutTrainingData(tmp_path):
    x, t = _pendulum(20)
    m = DimensionlessModel(inputDimensions=[{"L": 1}, {"L": 1, "T": -2}], outputDimension={"T": 1}).fit(x, t)
    m.save(str(tmp_path / "d.json"), includeTrainingData=False)
    m2 = SurrogateModelBase.load(str(tmp_path / "d.json"))
    np.testing.assert_allclose(m2.predictInterval(x[:3]).lower, m.predictInterval(x[:3]).lower, rtol=1e-12)


def test_dimensionlessIntervalsUseBaseModelStudentT():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(14)
    n = 8
    rho, v, d, mu = rng.uniform(0.5, 2, n), rng.uniform(1, 10, n), rng.uniform(0.1, 1, n), rng.uniform(1e-3, 1e-2, n)
    force = rho * v ** 2 * d ** 2 * 0.4 * (rho * v * d / mu) ** -0.2 * np.exp(rng.normal(0, 0.05, n))
    x = np.column_stack([rho, v, d, mu])
    m = DimensionlessModel(inputDimensions=[{"M": 1, "L": -3}, {"L": 1, "T": -1}, {"L": 1},
                                            {"M": 1, "L": -1, "T": -1}],
                           outputDimension={"M": 1, "L": 1, "T": -2}).fit(x, force)
    groups = m.groups()
    scale = np.exp(np.log(x) @ np.array(groups["outputScale"]))
    design = np.column_stack([np.ones(n), np.log(x) @ np.array(groups["groups"]).T])
    ref = sm.OLS(force / scale, design).fit().get_prediction(design).summary_frame(alpha=0.05)
    np.testing.assert_allclose(m.predictInterval(x, 0.95, "prediction").lower[:, 0], scale * ref.obs_ci_lower,
                               rtol=1e-10)


# ---------------------------------------------------------------- column indices
def test_outOfRangeColumnIndicesRaise():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, (40, 2))
    y = rng.poisson(3, 40).astype(float)
    with pytest.raises(ValueError, match="out of range"):
        GlmModel(family="poisson", offsetColumn=2).fit(x, y)          # used to wrap to column 0
    with pytest.raises(ValueError, match="out of range"):
        GamModel(family="poisson", offsetColumn=2).fit(x, y)
    g = np.repeat(np.arange(4.0), 10)
    with pytest.raises(ValueError, match="out of range"):
        MixedModel(groupColumn=2).fit(np.column_stack([x[:, 0], g]), y)
    # negative indices keep their Python meaning
    m = GlmModel(family="poisson", offsetColumn=-1).fit(np.column_stack([x[:, 0], np.log(x[:, 1] + 1)]), y)
    assert m.termNames == ["1", "x0"]
