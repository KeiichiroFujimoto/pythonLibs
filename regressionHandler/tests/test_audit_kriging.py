"""Regression tests for defects found in the Kriging-family audit (closed-form / explicit-formula oracles)."""
import numpy as np
import pytest

from pythonLibs.regressionHandler import (CokrigingModel, GradientKrigingModel, KrigingModel, ScalableKrigingModel)


def _gaussianLogPdf(y, mean, cov):
    lower = np.linalg.cholesky(cov)
    z = np.linalg.solve(lower, y - mean)
    return float(-0.5 * (z @ z) - np.sum(np.log(np.diag(lower))) - 0.5 * y.size * np.log(2.0 * np.pi))


# ---------------------------------------------------------------- GEK log-likelihood on the raw scale
@pytest.mark.parametrize("likelihood", ["reml", "ml"])
@pytest.mark.parametrize("poly", ["constant", "linear"])
def test_gekLogLikelihoodWithoutGradientsEqualsKriging(likelihood, poly):
    rng = np.random.default_rng(11)
    x = rng.uniform(-1, 2, (12, 2)) * np.array([1.0, 3.0])
    y = np.sin(2 * x[:, 0]) + 0.5 * x[:, 1] ** 2
    obs = np.column_stack([y, np.full((12, 2), np.nan)])
    gek = GradientKrigingModel(poly=poly, likelihood=likelihood, nugget=1e-6, hyperparameters=[0.3, -0.2]).fit(x, obs)
    kri = KrigingModel(poly=poly, likelihood=likelihood, nugget=1e-6, hyperparameters=[0.3, -0.2]).fit(x, y)
    assert gek.hyperparameters["logLikelihood"] == pytest.approx(kri.hyperparameters["logLikelihood"], rel=1e-9)


@pytest.mark.parametrize("likelihood", ["reml", "ml"])
def test_gekLogLikelihoodDoesNotDependOnNormalization(likelihood):
    """Same raw model with and without internal scaling -> same raw-data likelihood (gradients observed)."""
    rng = np.random.default_rng(12)
    x = rng.uniform(0, 4, 9)
    y = np.column_stack([10 * np.sin(x) + 3, 10 * np.cos(x)])
    ell, lam, gRaw = 1.3, 1e-4, 1e-4
    raw = GradientKrigingModel(normalize=False, likelihood=likelihood, hyperparameters=[np.log(ell)], nugget=lam,
                               gradientNugget=gRaw).fit(x, y)
    sx = x.std()
    # standardized gradients are dy/dx * xStd / yStd: the same raw noise ratio is gRaw * xStd^2
    std = GradientKrigingModel(normalize=True, likelihood=likelihood, hyperparameters=[np.log(ell / sx)], nugget=lam,
                               gradientNugget=gRaw * sx ** 2).fit(x, y)
    t = np.linspace(0.2, 3.8, 5)
    np.testing.assert_allclose(std.predict(t), raw.predict(t), rtol=1e-7, atol=1e-9)
    assert std.hyperparameters["logLikelihood"] == pytest.approx(raw.hyperparameters["logLikelihood"], rel=1e-8)


# ---------------------------------------------------------------- cokriging log-likelihood on the raw scale
def _cokData():
    rng = np.random.default_rng(13)
    x = rng.uniform(0, 3, (20, 2))
    y = np.column_stack([10 * np.sin(x[:, 0]) + 5, np.sin(x[:, 0]) + 0.5 * x[:, 1]]) + 0.05 * rng.standard_normal((20, 2))
    y[14:, 0] = np.nan
    y[:3, 1] = np.nan
    return x, y


def test_cokrigingMlLogLikelihoodIsTheRawGaussianDensity():
    x, y = _cokData()
    m = CokrigingModel(likelihood="ml", nStart=1, maxIter=20).fit(x, y)
    pi, oa = m._pi, m._oa
    c, _ = m._covariance(m._params, False)
    scale = m._yStd[oa]
    mean = m._yMean[oa] + scale * (m._F @ m._beta)
    ll = _gaussianLogPdf(y[pi, oa], mean, c * np.outer(scale, scale))
    assert m.hyperparameters["logLikelihood"] == pytest.approx(ll, rel=1e-7)


def test_cokrigingRemlSingleOutputEqualsKriging():
    rng = np.random.default_rng(14)
    x = rng.uniform(0, 3, (25, 2))
    y = 20 * np.sin(x[:, 0]) + 3 * x[:, 1] + 0.3 * rng.standard_normal(25)
    kri = KrigingModel(corr="matern52", poly="linear", likelihood="reml", nugget=1e-2,
                       hyperparameters=[0.2, 0.5]).fit(x, y)
    s2 = kri._sigma2                                      # REML process variance (standardized output)
    cok = CokrigingModel(corr="matern52", poly="linear", likelihood="reml", nugget=1e-2 * s2,
                         hyperparameters=[0.2, 0.5, np.sqrt(s2)]).fit(x, y)
    assert cok.hyperparameters["logLikelihood"] == pytest.approx(kri.hyperparameters["logLikelihood"], rel=1e-9)


# ---------------------------------------------------------------- multi-output reporting
def test_multiOutputSpatialReports():
    rng = np.random.default_rng(15)
    x = rng.uniform(0, 1, (40, 2))
    x[-4:] = x[:4]
    y = np.column_stack([np.sin(3 * x[:, 0]), x[:, 1] ** 2]) + 0.01 * rng.standard_normal((40, 2))
    m = KrigingModel(nStart=1).fit(x, y)
    single = KrigingModel(nStart=1).fit(x, y[:, 1])
    assert m.spatialSummary()["outputs"][1]["lambda"] == pytest.approx(single.spatialSummary()["lambda"])
    assert m.replicates()["outputs"][1]["pureErrorVariance"] == pytest.approx(single.replicates()["pureErrorVariance"])
    s = ScalableKrigingModel(nStart=1).fit(x, y)
    sSingle = ScalableKrigingModel(nStart=1).fit(x, y[:, 0])
    assert s.hyperparameters["outputs"][0]["logLikelihood"] == pytest.approx(sSingle.hyperparameters["logLikelihood"])
    assert s.spatialSummary()["outputs"][0]["lambda"] == pytest.approx(sSingle.spatialSummary()["lambda"])
    assert "Model:" in s.summary()


# ---------------------------------------------------------------- hyperparameter report
@pytest.mark.parametrize("opts", [{"plsComponents": 2}, {"corr": "periodic"}])
def test_reportedLengthscalesAreNotLogValues(opts):
    rng = np.random.default_rng(16)
    x = rng.uniform(0, 1, (30, 3))
    y = np.sin(3 * x[:, 0]) + x[:, 1]
    m = KrigingModel(nStart=1, **opts).fit(x, y)
    h = m.hyperparameters
    logLs = [v for k, v in h["logParams"].items() if "engthscale" in k]
    np.testing.assert_allclose(h["lengthscales"], np.exp(logLs), rtol=1e-12)


# ---------------------------------------------------------------- singular covariance at every start
def test_zeroNuggetWithDuplicatedInputsStillOptimizes():
    rng = np.random.default_rng(17)
    x = rng.uniform(0, 1, (15, 2))
    x = np.vstack([x, x[:4]])
    y = np.sin(3 * x[:, 0]) + x[:, 1]
    zero = KrigingModel(nugget=0.0).fit(x, y)
    # R is singular for every parameter value; the factorization regularizes it by the 1e-10 jitter,
    # so the fit must equal the one with that value as the nugget (and not stay at the initial guess)
    ref = KrigingModel(nugget=1e-10).fit(x, y)
    assert zero._jitter == 1e-10
    np.testing.assert_allclose(zero._params, ref._params, rtol=1e-8)
    assert zero.hyperparameters["logLikelihood"] == pytest.approx(ref.hyperparameters["logLikelihood"], rel=1e-10)
    fixed = KrigingModel(nugget=0.0, hyperparameters=list(ref._params)).fit(x, y)
    assert fixed.hyperparameters["logLikelihood"] == pytest.approx(ref.hyperparameters["logLikelihood"], rel=1e-10)


def test_cokrigingZeroNuggetWithDuplicatedInputsStillOptimizes():
    rng = np.random.default_rng(18)
    x = rng.uniform(0, 1, (15, 2))
    x = np.vstack([x, x[:4]])
    y = np.column_stack([np.sin(3 * x[:, 0]), x[:, 1]])
    m = CokrigingModel(nugget=0.0, nStart=1).fit(x, y)
    assert np.isfinite(m.hyperparameters["logLikelihood"]) and m.hyperparameters["logLikelihood"] > -1e10
    assert not np.allclose(m._params[:2], m._kernels[0].initialParams())
    np.testing.assert_allclose(m.predictValues(x), y, atol=1e-5)


def test_zeroNuggetWithDuplicatedInputsHasFiniteEffectiveDof():
    rng = np.random.default_rng(19)
    x = rng.uniform(0, 1, (15, 2))
    x = np.vstack([x, x[:4]])
    y = np.sin(3 * x[:, 0]) + x[:, 1]
    zero = KrigingModel(nugget=0.0).fit(x, y)
    ref = KrigingModel(nugget=1e-10, hyperparameters=list(zero._params)).fit(x, y)
    fs, fr = zero.spatialSummary(), ref.spatialSummary()
    assert fs["effectiveDof"] == pytest.approx(fr["effectiveDof"], rel=1e-8)
    assert fs["gcv"] == pytest.approx(fr["gcv"], rel=1e-6)
    assert np.isfinite(zero.metrics[0].aicc)
