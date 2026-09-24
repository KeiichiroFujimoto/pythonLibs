"""GLM, GAM, quantile regression, multi-fidelity / gradient Kriging, LARS and Sobol indices vs closed forms."""
import itertools

import numpy as np
import pytest

from pythonLibs.regressionHandler import (GamModel, GlmModel, GradientKrigingModel, KrigingModel, LinearBasisModel,
                                          MultiFidelityKrigingModel, QuantileModel, getProblem)
from pythonLibs.regressionHandler.evaluation import ishigamiSobol, sobolIndices
from pythonLibs.regressionHandler.models.QuantileModel import checkLoss, quantileFit
from pythonLibs.regressionHandler.numerics.SpecialFunctions import digamma, gammaln, trigamma
from pythonLibs.regressionHandler.solvers import larsOrder

EULER = 0.5772156649015329


def _x(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, (n, 2)), rng


def _design(x):
    return np.column_stack([np.ones(len(x)), x])


# ---------------------------------------------------------------- polygamma
def test_polygammaIdentities():
    assert digamma(1.0) == pytest.approx(-EULER, rel=1e-14)
    assert digamma(0.5) == pytest.approx(-EULER - 2 * np.log(2), rel=1e-14)
    assert trigamma(1.0) == pytest.approx(np.pi ** 2 / 6, rel=1e-13)
    assert trigamma(0.5) == pytest.approx(np.pi ** 2 / 2, rel=1e-13)
    x = np.array([0.03, 0.7, 3.3, 11.0, 150.0])
    np.testing.assert_allclose(digamma(x + 1), digamma(x) + 1 / x, rtol=1e-13)
    np.testing.assert_allclose(trigamma(x + 1), trigamma(x) - 1 / x ** 2, rtol=1e-12)
    h = 1e-5
    np.testing.assert_allclose(digamma(x), (gammaln(x + h) - gammaln(x - h)) / (2 * h), rtol=1e-7)


# ---------------------------------------------------------------- GLM
def test_gaussianGlmEqualsOls():
    x, rng = _x()
    y = 1 + 2 * x[:, 0] - x[:, 1] + 0.3 * rng.standard_normal(200)
    m = GlmModel().fit(x, y)
    ols = LinearBasisModel().fit(x, y)
    np.testing.assert_allclose(m.coefficients, ols.coefficients.ravel(), rtol=1e-11)
    np.testing.assert_allclose(m.result.stdErrors, ols.result.stdErrors, rtol=1e-9)
    s = m.glmSummary()
    rss = float(np.sum((y - ols.predict(x).ravel()) ** 2))
    assert s["deviance"] == pytest.approx(rss, rel=1e-10)
    assert s["aic"] == pytest.approx(200 * np.log(2 * np.pi * rss / 200) + 200 + 2 * 4, rel=1e-10)


@pytest.mark.parametrize("family, sampler", [
    ("poisson", lambda eta, rng: rng.poisson(np.exp(eta)).astype(float)),
    ("binomial", lambda eta, rng: rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)),
])
def test_canonicalGlmScoreEquationsAndInformation(family, sampler):
    x, rng = _x()
    eta = 0.3 + 0.8 * x[:, 0] - 0.5 * x[:, 1]
    y = sampler(eta, rng)
    m = GlmModel(family=family).fit(x, y)
    X = _design(x)
    mu = m.predict(x).ravel()
    np.testing.assert_allclose(X.T @ (y - mu), 0.0, atol=1e-8)          # canonical-link score equations
    var = mu if family == "poisson" else mu * (1 - mu)
    np.testing.assert_allclose(m.result.covariance[0], np.linalg.inv(X.T @ (X * var[:, None])), rtol=1e-8)
    assert m.result.dofResid == np.inf                                    # z tests for known scale
    if family == "poisson":
        ll = np.sum(y * np.log(mu) - mu - gammaln(y + 1))
        assert m.glmSummary()["aic"] == pytest.approx(-2 * ll + 6, rel=1e-10)
        dev = 2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))
        assert m.glmSummary()["deviance"] == pytest.approx(dev, rel=1e-10)


def test_noncanonicalGlmScoreEquations():
    x, rng = _x(300, 1)
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.2 + x[:, 0])))).astype(float)
    m = GlmModel(family={"type": "binomial", "link": "probit"}).fit(x, y)
    from pythonLibs.regressionHandler.numerics.SpecialFunctions import ndtr
    eta = m.predictLink(x)
    mu = ndtr(eta)
    dmu = np.exp(-0.5 * eta ** 2) / np.sqrt(2 * np.pi)
    np.testing.assert_allclose(_design(x).T @ ((y - mu) * dmu / (mu * (1 - mu))), 0.0, atol=1e-8)


def test_gammaGlmDispersionAndLinkInterval():
    x, rng = _x(300, 2)
    mu = np.exp(0.5 + 0.3 * x[:, 0])
    y = rng.gamma(3.0, mu / 3.0)
    m = GlmModel(family={"type": "gamma", "link": "log"}).fit(x, y)
    fit = m.predict(x).ravel()
    phi = np.sum((y - fit) ** 2 / fit ** 2) / (300 - 3)
    assert m.glmSummary()["dispersion"] == pytest.approx(phi, rel=1e-10)
    band = m.predictInterval(x[:5], 0.9)
    eta, se = m.predictLink(x[:5]), np.sqrt(np.einsum("ij,jk,ik->i", _design(x[:5]), m.result.covariance[0],
                                                       _design(x[:5])))
    from pythonLibs.regressionHandler.numerics.Distributions import StudentT
    q = StudentT(297).ppf(0.95)
    np.testing.assert_allclose(band.lower.ravel(), np.exp(eta - q * se), rtol=1e-10)
    np.testing.assert_allclose(band.upper.ravel(), np.exp(eta + q * se), rtol=1e-10)


def test_logisticIntervalStaysInUnitInterval():
    x, rng = _x(60, 3)
    y = (x[:, 0] + 0.3 * rng.standard_normal(60) > 0).astype(float)
    band = GlmModel(family="binomial").fit(x, y).predictInterval(np.array([[3.0, 0.0], [-3.0, 0.0]]), 0.99)
    assert np.all(band.lower >= 0) and np.all(band.upper <= 1) and np.all(band.lower <= band.mean)


def test_poissonOffsetAndPenalizedStationarity():
    x, rng = _x(250, 4)
    exposure = rng.uniform(0.5, 3.0, 250)
    y = rng.poisson(exposure * np.exp(0.2 + 0.7 * x[:, 0])).astype(float)
    xo = np.column_stack([x, np.log(exposure)])
    m = GlmModel(family="poisson", offsetColumn=2).fit(xo, y)
    mu = m.predict(xo).ravel()
    np.testing.assert_allclose(_design(x).T @ (y - mu), 0.0, atol=1e-8)
    assert m.coefficients.size == 3
    pen = GlmModel(family="poisson", penalty="ridge", alpha=5.0).fit(x, y)
    mu = pen.predict(x).ravel()
    beta = pen.coefficients
    np.testing.assert_allclose(_design(x).T @ (y - mu), 5.0 * np.r_[0.0, beta[1:]], atol=1e-7)


def test_negativeBinomialThetaIsMaximumLikelihood():
    x, rng = _x(400, 5)
    mu = np.exp(1.0 + 0.5 * x[:, 0])
    y = rng.negative_binomial(3.0, 3.0 / (3.0 + mu)).astype(float)
    m = GlmModel(family="negativeBinomial").fit(x, y)
    fam = m.family
    theta = fam.theta
    fit = m.predict(x).ravel()
    w = np.ones_like(y)

    def ll(t):
        fam.theta = t
        return fam.logLikelihood(y, fit, w, 1.0)
    h = 1e-4 * theta
    assert abs(ll(theta + h) - ll(theta - h)) / (2 * h) < 1e-4
    fam.theta = theta


# ---------------------------------------------------------------- GAM
def _gamData(n=250, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, 2))
    return x, np.sin(2 * np.pi * x[:, 0]) + (x[:, 1] - 0.5) ** 2 * 3 + 0.2 * rng.standard_normal(n), rng


def test_gamWithFixedLambdasIsPenalizedLeastSquares():
    x, y, _ = _gamData()
    lam = [3.0, 40.0]
    m = GamModel(lambdas=lam).fit(x, y)
    design, _ = m._design(x)
    p = design.shape[1]
    s = np.zeros((p, p))
    for t, sl, l in zip(m._terms, m._blockSlices(), lam):
        s[sl, sl] = l * t.penalties()[0]
    beta = np.linalg.solve(design.T @ design + s, design.T @ y)
    np.testing.assert_allclose(m.coefficients, beta, rtol=1e-8, atol=1e-10)
    infl = design @ np.linalg.solve(design.T @ design + s, design.T)
    assert m.glmSummary()["edf"] == pytest.approx(np.trace(infl), rel=1e-9)
    rss = float(np.sum((y - infl @ y) ** 2))
    assert m.gamSummary()["score"] == pytest.approx(250 * rss / (250 - np.trace(infl)) ** 2, rel=1e-9)


def test_gamTermsAreCentredAndGcvIsMinimal():
    x, y, _ = _gamData(seed=1)
    m = GamModel().fit(x, y)
    for term in m.predictTerms(x).values():
        assert abs(term["fit"].sum()) < 1e-8
    lam = m.gamSummary()["lambdas"]
    best = m.gamSummary()["score"]
    for j in range(2):
        for fac in (0.5, 2.0):
            other = list(lam)
            other[j] *= fac
            assert GamModel(lambdas=other).fit(x, y).gamSummary()["score"] >= best - 1e-10
    assert m.predict(x).ravel() == pytest.approx(m.predictLink(x), rel=1e-12)


def test_gamFactorAndLinearTermsEqualOls():
    rng = np.random.default_rng(2)
    g = rng.integers(0, 3, 90).astype(float)
    z = rng.uniform(0, 1, 90)
    y = np.array([0.0, 1.0, -0.5])[g.astype(int)] + 2 * z + 0.1 * rng.standard_normal(90)
    x = np.column_stack([g, z])
    m = GamModel(terms=[{"type": "factor", "column": 0}, {"type": "linear", "columns": [1]}]).fit(x, y)
    X = np.column_stack([np.ones(90), g == 1, g == 2, z]).astype(float)
    np.testing.assert_allclose(m.coefficients, np.linalg.lstsq(X, y, rcond=None)[0], rtol=1e-9)


# ---------------------------------------------------------------- quantile regression
def test_quantileExactAgainstBruteForce():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, (14, 1))
    y = 1 + 2 * x[:, 0] + 0.4 * rng.standard_t(3, 14)
    X = _design(x)
    for tau in (0.2, 0.5, 0.9):
        best = min(np.sum(checkLoss(y - X @ np.linalg.solve(X[list(s)], y[list(s)]), tau))
                   for s in itertools.combinations(range(14), 2))
        beta, _ = quantileFit(X, y, tau)
        assert np.sum(checkLoss(y - X @ beta, tau)) == pytest.approx(best, abs=1e-9)


def test_quantileEquivarianceAndMedian():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 1, (101, 1))
    y = x[:, 0] + rng.standard_normal(101)
    X = _design(x)
    b = quantileFit(X, y, 0.3)[0]
    np.testing.assert_allclose(quantileFit(X, y + X @ [1.0, -2.0], 0.3)[0], b + [1.0, -2.0], atol=1e-8)
    np.testing.assert_allclose(quantileFit(X, 3.0 * y, 0.3)[0], 3.0 * b, atol=1e-8)
    np.testing.assert_allclose(quantileFit(X, -y, 0.7)[0], -b, atol=1e-8)
    med = QuantileModel(basis={"type": "polynomial", "degree": 0}, se="none").fit(x, y)
    assert med.coefficients[0] == pytest.approx(np.median(y), abs=1e-9)
    r = y - X @ b
    assert np.sum(r < -1e-9) <= 0.3 * 101 <= np.sum(r <= 1e-9)


# ---------------------------------------------------------------- multi-fidelity / gradient Kriging
def test_multiFidelitySingleLevelEqualsKriging():
    x, rng = _x(25, 6)
    y = np.sin(3 * x[:, 0]) + x[:, 1]
    mf = MultiFidelityKrigingModel(kriging={"nugget": 1e-6}).fitLevels([x], [y])
    k = KrigingModel(corr="matern52", nugget=1e-6).fit(x, y)
    t = rng.uniform(-1, 1, (10, 2))
    mean, var = mf.predictTop(t)
    np.testing.assert_allclose(mean, k.predict(t).ravel(), rtol=1e-10)
    np.testing.assert_allclose(var, k.predictVariances(t).ravel(), rtol=1e-10)


def test_multiFidelityRecoversScaledLevel():
    xl = np.linspace(0, 1, 15)[:, None]
    lf = lambda x: np.sin(8 * x[:, 0])
    xh = xl[::3]
    mf = MultiFidelityKrigingModel(poly="constant", kriging={"nugget": 1e-8}).fitLevels([xl, xh],
                                                                                    [lf(xl), 2.0 * lf(xh) + 1.0])
    t = np.linspace(0.05, 0.95, 20)[:, None]
    mean, _ = mf.predictTop(t)
    low, _ = mf._recursive(t, 0, "confidence")
    np.testing.assert_allclose(mean, 2.0 * low + 1.0, atol=5e-5)


def _gekData(seed=7):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 2, (10, 2))
    f = lambda z: np.sin(2 * z[:, 0]) * np.cos(z[:, 1])
    g = lambda z: np.column_stack([2 * np.cos(2 * z[:, 0]) * np.cos(z[:, 1]), -np.sin(2 * z[:, 0]) * np.sin(z[:, 1])])
    return x, f, g, rng


@pytest.mark.parametrize("corr", ["squaredExponential", "matern52", {"type": "matern", "nu": 3.5}])
def test_gekCovarianceBlocksAreKernelDerivatives(corr):
    x, f, g, rng = _gekData()
    m = GradientKrigingModel(corr=corr, hyperparameters=[0.1, -0.3] if corr != "x" else None).fit(
        x, np.column_stack([f(x), g(x)]))
    k, p = m._kernel, m._params
    xa, xb, h = rng.uniform(0, 1, (3, 2)), rng.uniform(0, 1, (4, 2)), 1e-5

    def da(a, b, kk):
        if kk == 0:
            return k.matrix(a, b, p)
        e = np.zeros(2)
        e[kk - 1] = h
        return (k.matrix(a + e, b, p) - k.matrix(a - e, b, p)) / (2 * h)
    for ta in range(3):
        for tb in range(3):
            got = m._cov(xa, np.full(3, ta), xb, np.full(4, tb), p)
            if tb == 0:
                ref = da(xa, xb, ta)
            else:
                e = np.zeros(2)
                e[tb - 1] = h
                ref = (da(xa, xb + e, ta) - da(xa, xb - e, ta)) / (2 * h)
            np.testing.assert_allclose(got, ref, atol=1e-5)   # finite-difference error of the reference


def test_gekInterpolatesGradientsAndReducesToKriging():
    x, f, g, rng = _gekData()
    m = GradientKrigingModel(hyperparameters=[0.2, 0.1]).fit(x, np.column_stack([f(x), g(x)]))
    np.testing.assert_allclose(m.predict(x)[:, 1:], g(x), atol=1e-4)
    t = rng.uniform(0.2, 1.8, (5, 2))
    p = m.predict(t)
    h = 1e-5
    for k in range(2):
        e = np.zeros(2)
        e[k] = h
        np.testing.assert_allclose(p[:, 1 + k], (m.predict(t + e)[:, 0] - m.predict(t - e)[:, 0]) / (2 * h), atol=1e-6)
    y = np.column_stack([f(x), np.full((10, 2), np.nan)])
    gek = GradientKrigingModel(hyperparameters=[0.2, 0.1], nugget=1e-8).fit(x, y)
    kri = KrigingModel(corr="squaredExponential", nugget=1e-8, hyperparameters=[0.2, 0.1]).fit(x, f(x))
    np.testing.assert_allclose(gek.predict(t)[:, 0], kri.predict(t).ravel(), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(gek.predictVariances(t)[:, 0], kri.predictVariances(t).ravel(), rtol=1e-6)


# ---------------------------------------------------------------- LARS / sparse PCE / Sobol
def test_larsOrthogonalDesignOrdersByCorrelation():
    rng = np.random.default_rng(8)
    q, _ = np.linalg.qr(rng.standard_normal((40, 6)))
    q = q - q.mean(axis=0)
    q /= np.linalg.norm(q, axis=0)
    y = q @ np.array([0.5, -3.0, 0.0, 2.0, 1.0, -0.2])
    y = y - y.mean()
    assert larsOrder(q, y, 6)[:4] == list(np.argsort(-np.abs(q.T @ y))[:4])


def test_sparsePceRecoversExactPolynomialAndSobol():
    rng = np.random.default_rng(9)
    x = rng.uniform(-1, 1, (60, 3))
    x[0], x[1] = -1, 1
    y = 1.0 + x[:, 0] + x[:, 1] * x[:, 2]
    m = LinearBasisModel(basis={"type": "orthogonalPolynomial", "degree": 4}, solver="lars").fit(x, y)
    t = rng.uniform(-1, 1, (20, 3))
    np.testing.assert_allclose(m.predict(t).ravel(), 1 + t[:, 0] + t[:, 1] * t[:, 2], atol=1e-10)
    assert len(m.solution.info["selected"][0]) == 3
    s = sobolIndices(m)
    # Var(x0) = 1/3, Var(x1 x2) = 1/9 for independent U(-1, 1)
    np.testing.assert_allclose(s.first, [0.75, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(s.total, [0.75, 0.25, 0.25], atol=1e-10)
    assert s.second[1, 2] == pytest.approx(0.25, abs=1e-10)
    assert s.variance == pytest.approx(1 / 3 + 1 / 9, rel=1e-10)


def test_ishigamiSobolReferenceAndMonteCarlo():
    ref = ishigamiSobol()
    np.testing.assert_allclose(ref["first"], [0.3139, 0.4424, 0.0], atol=1e-4)
    np.testing.assert_allclose(ref["total"], [0.5576, 0.4424, 0.2437], atol=1e-4)
    pr = getProblem("ishigami")
    mc = sobolIndices(pr, xlimits=pr.xlimits, nSamples=2 ** 13, nBoot=100)
    assert np.all(mc.firstInterval[:, 0] - 0.01 <= ref["first"]) and np.all(ref["first"] <= mc.firstInterval[:, 1] + 0.01)
    np.testing.assert_allclose(mc.total, ref["total"], atol=0.03)
