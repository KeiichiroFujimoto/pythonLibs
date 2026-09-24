"""Closed-form checks of likelihood conventions, persisted inference and multi-output behaviour."""
import numpy as np
import pytest

from pythonLibs.regressionHandler import GamModel, GlmModel, QuantileModel
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase
from pythonLibs.regressionHandler.solvers import larsOrder


def _data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    w = rng.uniform(0.5, 3.0, n)
    return rng, x, w


# ---------------------------------------------------------------- weighted log-likelihoods
def test_weightedGaussianLogLikelihood():
    rng, x, w = _data()
    y = 1 + 2 * x + rng.normal(0, 1, x.size) / np.sqrt(w)
    m = GlmModel(family="gaussian").fit(x[:, None], y, weights=w)
    n = x.size
    dev = float(np.sum(w * (y - m.predict(x[:, None]).ravel()) ** 2))
    # y_i ~ N(mu_i, s^2 / w_i) at the ML scale s^2 = D / n
    ll = -0.5 * n * (np.log(2 * np.pi * dev / n) + 1) + 0.5 * np.sum(np.log(w))
    g = m.glmSummary()
    assert g["logLikelihood"] == pytest.approx(ll, rel=1e-10)
    assert g["aic"] == pytest.approx(-2 * ll + 2 * 3, rel=1e-10)


def test_weightedGammaAndInverseGaussianLogLikelihood():
    from pythonLibs.regressionHandler.numerics.SpecialFunctions import gammaln
    rng, x, w = _data(seed=1)
    mu = np.exp(0.5 + x)
    y = rng.gamma(4.0, mu / 4.0)
    m = GlmModel(family={"type": "gamma", "link": "log"}).fit(x[:, None], y, weights=w)
    muHat = m.predict(x[:, None]).ravel()
    dev = m.glmSummary()["deviance"]
    disp = dev / np.sum(w)
    shape = 1 / disp
    # weighted gamma density at scale D / sum(w)
    dens = shape * np.log(shape * y / muHat) - shape * y / muHat - np.log(y) - gammaln(shape)
    assert m.glmSummary()["logLikelihood"] == pytest.approx(float(np.sum(w * dens)), rel=1e-10)

    y2 = mu + 0.05 * rng.standard_normal(x.size) * mu ** 1.5
    y2 = np.abs(y2)
    m2 = GlmModel(family={"type": "inverseGaussian", "link": "log"}).fit(x[:, None], y2, weights=w)
    dev2 = m2.glmSummary()["deviance"]
    sw = np.sum(w)
    ll2 = -0.5 * (sw * (1 + np.log(2 * np.pi * dev2 / sw)) + 3 * np.sum(w * np.log(y2)))
    assert m2.glmSummary()["logLikelihood"] == pytest.approx(ll2, rel=1e-10)


def test_negativeBinomialKeepsThetaParameterAfterLoad():
    rng, x, _ = _data(seed=2)
    y = rng.negative_binomial(3.0, 3.0 / (3.0 + np.exp(1 + x)))
    m = GlmModel(family="negativeBinomial").fit(x[:, None], y.astype(float))
    m2 = SurrogateModelBase.fromDict(m.toDict())
    assert m2.glmSummary()["aic"] == pytest.approx(m.glmSummary()["aic"], rel=1e-12)
    assert m2.family.extraParams == 1


# ---------------------------------------------------------------- models saved without training data
def test_inferenceSurvivesSavingWithoutTrainingData():
    rng, x, _ = _data(seed=3)
    y = np.sin(4 * x) + 0.1 * rng.standard_normal(x.size)
    xs = np.linspace(0.1, 0.9, 5)[:, None]
    for m in (GlmModel(family="gaussian", basis={"type": "polynomial", "degree": 3}).fit(x[:, None], y),
              GamModel().fit(x[:, None], y), QuantileModel(tau=0.3).fit(x[:, None], y)):
        m2 = SurrogateModelBase.fromDict(m.toDict(includeTrainingData=False))
        assert m2.nTrain == x.size
        a, b = m.predictInterval(xs, 0.9, "confidence"), m2.predictInterval(xs, 0.9, "confidence")
        np.testing.assert_allclose(b.upper, a.upper, rtol=1e-12)
        assert isinstance(m2.summary(), str)
        if isinstance(m, GamModel):
            assert m2.termTable() == m.termTable()
        if isinstance(m, GlmModel):
            assert m2.glmSummary()["aic"] == pytest.approx(m.glmSummary()["aic"])
            with pytest.raises(ValueError, match="training data"):
                m2.residuals()


# ---------------------------------------------------------------- several outputs
def test_multiOutputGlmAndGamDelegateToEachOutput():
    rng, x, _ = _data(seed=4)
    y = np.column_stack([np.sin(3 * x), np.cos(3 * x)]) + 0.1 * rng.standard_normal((x.size, 2))
    xs = np.linspace(0.1, 0.9, 4)[:, None]
    for m in (GlmModel(basis={"type": "polynomial", "degree": 3}), GamModel()):
        m.fit(x[:, None], y, outputNames=["a", "b"])
        singles = [type(m)(**m.options.toDict()).fit(x[:, None], y[:, j]) for j in range(2)]
        link = m.predictLink(xs)
        assert link.shape == (4, 2)
        for j, s in enumerate(singles):
            np.testing.assert_allclose(link[:, j], s.predictLink(xs), rtol=1e-10)
        pi = m.predictInterval(xs, 0.9, "confidence")
        assert pi.lower.shape == (4, 2)
        np.testing.assert_allclose(pi.lower[:, 1], singles[1].predictInterval(xs, 0.9, "confidence").lower[:, 0])
        assert set(m.glmSummary()) == {"a", "b"}
        assert m.residuals().shape == (x.size, 2)
        assert "[a]" in m.summary() and "[b]" in m.summary()
        if isinstance(m, GamModel):
            assert set(m.termTable()) == {"a", "b"} and set(m.predictTerms(xs)) == {"a", "b"}


# ---------------------------------------------------------------- UBRE scale
def test_ubreNeedsAKnownScale():
    rng, x, _ = _data(seed=5)
    y = np.sin(5 * x) + 0.2 * rng.standard_normal(x.size)
    with pytest.raises(ValueError, match="UBRE"):
        GamModel(method="ubre").fit(x[:, None], y)
    # with the true scale, UBRE = D / n - phi + 2 phi edf / n is minimized at the reported lambda
    phi = 0.04
    m = GamModel(method="ubre", dispersion=phi).fit(x[:, None], y)
    g = m.gamSummary()
    assert g["score"] == pytest.approx(g["deviance"] / x.size - phi + 2 * phi * g["edf"] / x.size, rel=1e-10)
    for f in (0.8, 1.25):
        other = GamModel(lambdas=[m._lambdas[0] * f], dispersion=phi).fit(x[:, None], y).gamSummary()
        assert other["deviance"] / x.size + 2 * phi * other["edf"] / x.size >= \
            g["deviance"] / x.size + 2 * phi * g["edf"] / x.size - 1e-12


# ---------------------------------------------------------------- term tests
def test_gamTermWaldTestUsesTermValues():
    rng, x, _ = _data(n=200, seed=6)
    x2 = rng.uniform(0, 1, x.size)
    y = np.sin(4 * x) + 0.1 * rng.standard_normal(x.size)
    m = GamModel().fit(np.column_stack([x, x2]), y)
    xd, _ = m._design(m.xt)
    for row, sl in zip(m.termTable(), m._blockSlices()):
        xj = xd[:, sl]
        f = xj @ m._res.coef[sl]
        vf = xj @ (m._res.covUnscaled[sl, sl] * m._phi) @ xj.T
        r = row["refDf"]
        wv, vv = np.linalg.eigh(vf)
        top = np.argsort(wv)[::-1][:r]
        stat = float(np.sum((vv[:, top].T @ f) ** 2 / wv[top]))
        assert row["F"] * r == pytest.approx(stat, rel=1e-6)
    rows = m.termTable()
    assert rows[0]["pValue"] < 1e-6 and rows[1]["pValue"] > 1e-3


# ---------------------------------------------------------------- LARS path length / quantile degenerate case
def test_larsOrderHasRequestedLength():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((30, 8))
    x -= x.mean(axis=0)
    x /= np.linalg.norm(x, axis=0)
    y = rng.standard_normal(30)
    y -= y.mean()
    assert [len(larsOrder(x, y, k)) for k in (0, 1, 3, 8, 20)] == [0, 1, 3, 8, 8]


@pytest.mark.parametrize("se", ["nid", "iid", "ker"])
def test_quantileConstantResponseHasUndefinedStandardErrors(se):
    x = np.linspace(0, 1, 30)[:, None]
    m = QuantileModel(se=se).fit(x, np.full(30, 2.0))
    np.testing.assert_allclose(m.coefficients, [2.0, 0.0], atol=1e-9)
    assert np.all(np.isnan(m.result.stdErrors))
    assert "undefined" in m._info["seWarning"]
