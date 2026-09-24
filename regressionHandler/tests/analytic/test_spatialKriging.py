"""Spatial-statistics features (general Matern, Wendland, distances, GCV, simulation) against closed forms."""
import numpy as np
import pytest

from pythonLibs.regressionHandler import KrigingModel, RbfModel, createModel
from pythonLibs.regressionHandler.kernels import (AbsoluteExponential, Matern, Matern32, Matern52,
                                                  SquaredExponential, Wendland)
from pythonLibs.regressionHandler.numerics.SpecialFunctions import besselK, gamma


# ---------------------------------------------------------------- special functions
def test_besselKHalfIntegerClosedForms():
    x = np.array([1e-3, 0.1, 0.9, 1.99, 2.0, 2.01, 5.0, 30.0, 300.0])
    base = np.sqrt(np.pi / (2 * x)) * np.exp(-x)
    np.testing.assert_allclose(besselK(0.5, x), base, rtol=1e-13)
    np.testing.assert_allclose(besselK(1.5, x), base * (1 + 1 / x), rtol=1e-13)
    np.testing.assert_allclose(besselK(2.5, x), base * (1 + 3 / x + 3 / x ** 2), rtol=1e-13)


@pytest.mark.parametrize("nu", [0.0, 0.3, 1.0, 1.7, 3.2])
def test_besselKRecurrenceAndSymmetry(nu):
    x = np.array([0.05, 0.5, 1.5, 2.5, 8.0, 40.0])
    # K_{nu+1}(x) = K_{nu-1}(x) + (2 nu / x) K_nu(x),  K_{-nu} = K_nu
    lhs = besselK(nu + 1, x)
    rhs = besselK(abs(nu - 1), x) + 2 * nu / x * besselK(nu, x)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


def test_besselKSmallArgumentAsymptote():
    # K_nu(x) ~ Gamma(nu)/2 (2/x)^nu as x -> 0 (nu > 0)
    for nu in (0.7, 1.5, 2.3):
        x = 1e-7
        assert besselK(nu, np.array([x]))[0] == pytest.approx(0.5 * gamma(nu) * (2 / x) ** nu, rel=1e-5)


def test_gammaValues():
    assert gamma(5.0) == pytest.approx(24.0, rel=1e-14)
    assert gamma(0.5) == pytest.approx(np.sqrt(np.pi), rel=1e-14)
    assert gamma(2.5) == pytest.approx(0.75 * np.sqrt(np.pi), rel=1e-14)


# ---------------------------------------------------------------- kernels
def _kernelValue(kernel, d, p, nx=1):
    kernel.setup(nx)
    a = np.zeros((1, nx))
    b = np.zeros((1, nx))
    b[0, 0] = d
    return kernel.matrix(a, b, np.asarray(p, dtype=float))[0, 0]


@pytest.mark.parametrize("nu, closed", [(0.5, AbsoluteExponential), (1.5, Matern32), (2.5, Matern52)])
def test_maternGeneralNuEqualsHalfIntegerKernels(nu, closed):
    for d in (0.0, 0.2, 1.1, 4.0):
        for p in ([0.0], [np.log(0.4)]):
            assert _kernelValue(Matern(nu=nu), d, p) == pytest.approx(_kernelValue(closed(), d, p), rel=1e-12, abs=1e-300)


def test_maternLargeNuTendsToSquaredExponential():
    for d in (0.3, 1.0, 2.0):
        assert _kernelValue(Matern(nu=200.0), d, [0.0]) == pytest.approx(_kernelValue(SquaredExponential(), d, [0.0]),
                                                                         rel=5e-3)


def test_maternRangeParameterization():
    # range form: k = 2^(1-nu)/Gamma(nu) (d/a)^nu K_nu(d/a); nu = 1/2 gives exp(-d/a), nu = 3/2 gives (1+d/a) exp(-d/a)
    a = 0.7
    for d in (0.1, 0.5, 2.0):
        assert _kernelValue(Matern(nu=0.5, parameterization="range"), d, [np.log(a)]) == \
            pytest.approx(np.exp(-d / a), rel=1e-13)
        assert _kernelValue(Matern(nu=1.5, parameterization="range"), d, [np.log(a)]) == \
            pytest.approx((1 + d / a) * np.exp(-d / a), rel=1e-13)
    k = Matern(nu=1.5, parameterization="range").setup(1)
    assert k.rangeParameters(np.array([np.log(a)]))[0] == pytest.approx(a)
    k = Matern(nu=1.5).setup(1)
    assert k.rangeParameters(np.array([np.log(a)]))[0] == pytest.approx(a / np.sqrt(3.0))


def _wendlandClosed(r, l, k):
    t = max(1 - r, 0.0)
    if k == 0:
        return t ** l
    if k == 1:
        return t ** (l + 1) * ((l + 1) * r + 1)
    return t ** (l + 2) * ((l * l + 4 * l + 3) * r * r + (3 * l + 6) * r + 3) / 3


@pytest.mark.parametrize("k", [0, 1, 2])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_wendlandClosedFormAndCompactSupport(k, dim):
    l = dim // 2 + k + 1
    ell = 1.3
    for d in (0.0, 0.2, 0.9, 1.29, 1.3, 2.0):
        val = _kernelValue(Wendland(k=k, dimension=dim), d, [np.log(ell)], nx=dim)
        assert val == pytest.approx(_wendlandClosed(d / ell, l, k), rel=1e-13, abs=1e-15)
        if d >= ell:
            assert val == 0.0


def test_wendlandPositiveDefinite():
    x = np.random.default_rng(0).uniform(0, 1, (60, 2))
    k = Wendland(k=2).setup(2)
    assert np.linalg.eigvalsh(k.matrix(x, x, np.array([np.log(0.5)]))).min() > 0


def _checkGradients(kernel, x, p, h=1e-6):
    grads = kernel.gradients(x, p)
    assert len(grads) == p.size
    for i in range(p.size):
        e = np.zeros(p.size)
        e[i] = h
        num = (kernel.matrix(x, x, p + e) - kernel.matrix(x, x, p - e)) / (2 * h)
        np.testing.assert_allclose(grads[i], num, atol=1e-7)


@pytest.mark.parametrize("kernel, p", [
    (Matern(nu=0.8, ard=True), [0.1, -0.3]),
    (Matern(nu="estimate", ard=False), [0.2, np.log(1.3)]),
    (Matern(nu=2.2, parameterization="range", ard=False), [-0.1]),
    (Wendland(k=1, ard=True), [0.4, 0.8]),
    (Wendland(k=2, ard=False), [0.5]),
    (SquaredExponential(distance="anisotropic"), [0.1, -0.2, 0.3]),
])
def test_kernelGradientsFiniteDifference(kernel, p):
    x = np.random.default_rng(3).uniform(0, 1.5, (12, 2))
    kernel.setup(2)
    _checkGradients(kernel, x, np.asarray(p, dtype=float))


def test_fixedAnisotropyMatrixEqualsArdLengthscales():
    x = np.random.default_rng(1).uniform(-1, 1, (9, 2))
    iso = SquaredExponential(ard=False, V=[[2.0, 0.0], [0.0, 3.0]]).setup(2)
    ard = SquaredExponential(ard=True).setup(2)
    np.testing.assert_allclose(iso.matrix(x, x, np.array([0.0])), ard.matrix(x, x, np.log([2.0, 3.0])), rtol=1e-13)
    ident = SquaredExponential(ard=False, V=[[1.0, 0.0], [0.0, 1.0]]).setup(2)
    plain = SquaredExponential(ard=False).setup(2)
    np.testing.assert_allclose(ident.matrix(x, x, np.array([0.3])), plain.matrix(x, x, np.array([0.3])), rtol=1e-14)


def test_anisotropicCholeskyDistance():
    # r2 = d^T L L^T d with L = [[e^a, 0], [c, e^b]]
    a, c, b = 0.2, 0.5, -0.4
    kern = SquaredExponential(distance="anisotropic").setup(2)
    xa, xb = np.array([[0.3, -0.2]]), np.array([[1.0, 0.4]])
    lmat = np.array([[np.exp(a), 0.0], [c, np.exp(b)]])
    d = xa - xb
    r2 = (d @ lmat @ lmat.T @ d.T).item()
    assert kern.matrix(xa, xb, np.array([a, c, b]))[0, 0] == pytest.approx(np.exp(-0.5 * r2), rel=1e-12)


def test_greatCircleKnownDistances():
    k = SquaredExponential(distance="greatCircle").setup(2)
    r = 6378.388
    pts = np.array([[0.0, 0.0], [90.0, 0.0], [0.0, 90.0], [180.0, 0.0], [-45.0, 0.0]])
    d = k.greatCircleDistance(pts, pts)
    assert d[0, 1] == pytest.approx(r * np.pi / 2, rel=1e-13)
    assert d[0, 2] == pytest.approx(r * np.pi / 2, rel=1e-13)
    assert d[0, 3] == pytest.approx(r * np.pi, rel=1e-13)
    assert d[1, 4] == pytest.approx(r * 3 * np.pi / 4, rel=1e-13)
    np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-9)
    np.testing.assert_allclose(d, d.T, rtol=1e-13)
    miles = SquaredExponential(distance="greatCircle", radiusUnit="miles").setup(2)
    assert miles.greatCircleDistance(pts[:1], pts[1:2])[0, 0] == pytest.approx(3963.34 * np.pi / 2, rel=1e-13)


# ---------------------------------------------------------------- Kriging statistics
def _spatialData(n=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 3, (n, 2))
    y = np.sin(x[:, 0]) * np.cos(x[:, 1]) + 0.3 * x[:, 0] + 0.1 * rng.standard_normal(n)
    return x, y


def _explicit(x, y, ell, lam, nu=1.5):
    """Universal Kriging with a linear trend written out in raw units (range parameterization)."""
    k = Matern(nu=nu, parameterization="range", ard=False).setup(2)
    kxx = k.matrix(x, x, np.array([np.log(ell)]))
    n = x.shape[0]
    m = kxx + lam * np.eye(n)
    f = np.column_stack([np.ones(n), x])
    mi = np.linalg.inv(m)
    omega = np.linalg.inv(f.T @ mi @ f)
    p = mi - mi @ f @ omega @ f.T @ mi
    s = np.eye(n) - lam * p
    sigma2 = float(y @ p @ y) / n
    return dict(s=s, sigma2=sigma2, logdetM=np.linalg.slogdet(m)[1], logdetOmega=np.linalg.slogdet(omega)[1],
                beta=omega @ f.T @ mi @ y, mi=mi, kernel=k, f=f, omega=omega)


def _fixedModel(x, y, ell, lam, **kw):
    return KrigingModel(corr={"type": "matern", "nu": 1.5, "parameterization": "range", "ard": False},
                        poly="linear", normalize=False, nugget=lam, likelihood="ml",
                        hyperparameters=[np.log(ell)], **kw).fit(x, y)


def test_krigingSpatialStatisticsClosedForm():
    x, y = _spatialData()
    ell, lam = 0.8, 0.05
    ref = _explicit(x, y, ell, lam)
    m = _fixedModel(x, y, ell, lam)
    fs = m.spatialSummary()
    n = x.shape[0]
    fitted = ref["s"] @ y
    np.testing.assert_allclose(m.predict(x).ravel(), fitted, rtol=1e-9, atol=1e-10)
    trS = float(np.trace(ref["s"]))
    assert fs["effectiveDof"] == pytest.approx(trS, rel=1e-9)
    assert fs["gcv"] == pytest.approx(np.mean((y - fitted) ** 2) / (1 - trS / n) ** 2, rel=1e-8)
    assert fs["sigma2"] == pytest.approx(ref["sigma2"], rel=1e-9)
    assert fs["tau"] == pytest.approx(np.sqrt(lam * ref["sigma2"]), rel=1e-9)
    assert fs["lambda"] == pytest.approx(lam)
    assert fs["range"] == pytest.approx(ell, rel=1e-12)
    like = -n / 2 - n / 2 * np.log(2 * np.pi) - n / 2 * np.log(ref["sigma2"]) - 0.5 * ref["logdetM"]
    assert fs["logLikelihood"] == pytest.approx(like, rel=1e-10)
    assert fs["logRestrictedProfile"] == pytest.approx(like + 0.5 * ref["logdetOmega"], rel=1e-10)
    np.testing.assert_allclose(m.hyperparameters["trendCoefficients"], ref["beta"], rtol=1e-8, atol=1e-10)


def test_krigingNormalizeDoesNotChangeFixedModel():
    """The fitted surface is equivariant to standardization when the lengthscale is rescaled accordingly."""
    x, y = _spatialData()
    raw = _fixedModel(x, y, 0.8, 0.05)
    t = np.random.default_rng(5).uniform(0, 3, (20, 2))
    # the ML estimate of lambda / range at fixed hyperparameters is independent of the output scale
    scaled = _fixedModel(x, 10.0 * y + 3.0, 0.8, 0.05)
    np.testing.assert_allclose(scaled.predict(t), 10.0 * raw.predict(t) + 3.0, rtol=1e-9)
    np.testing.assert_allclose(scaled.predictVariances(t), 100.0 * raw.predictVariances(t), rtol=1e-8)


def test_predictCovarianceMatchesExplicitAndVariances():
    x, y = _spatialData()
    ell, lam = 0.8, 0.05
    ref = _explicit(x, y, ell, lam)
    m = _fixedModel(x, y, ell, lam)
    t = np.random.default_rng(2).uniform(0, 3, (7, 2))
    k, p = ref["kernel"], np.array([np.log(ell)])
    kt = k.matrix(t, x, p)
    ft = np.column_stack([np.ones(7), t])
    u = ft - kt @ ref["mi"] @ ref["f"]
    cov = (k.matrix(t, t, p) - kt @ ref["mi"] @ kt.T + u @ ref["omega"] @ u.T) * ref["sigma2"]
    got = m.predictCovariance(t)[0]
    np.testing.assert_allclose(got, cov, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(np.diag(got), m.predictVariances(t).ravel(), rtol=1e-10)
    pred = m.predictCovariance(t, kind="prediction")[0]
    np.testing.assert_allclose(pred - got, np.eye(7) * lam * ref["sigma2"], atol=1e-12)


def test_conditionalSimulationMoments():
    x, y = _spatialData(n=25)
    m = _fixedModel(x, y, 0.8, 0.05)
    t = np.array([[0.5, 0.5], [1.5, 2.0], [2.8, 0.1]])
    draws = m.simulate(t, nSamples=20000, seed=4)
    assert draws.shape == (20000, 3, 1)
    mean, cov = m.predict(t).ravel(), m.predictCovariance(t)[0]
    se = np.sqrt(np.diag(cov) / 20000)
    assert np.all(np.abs(draws[:, :, 0].mean(0) - mean) < 5 * se)
    np.testing.assert_allclose(np.cov(draws[:, :, 0].T), cov, rtol=0.06, atol=0.06 * np.max(np.diag(cov)))
    # different seeds give different draws, the same seed reproduces them
    np.testing.assert_array_equal(m.simulate(t, 3, seed=1), m.simulate(t, 3, seed=1))
    assert not np.allclose(m.simulate(t, 3, seed=1), m.simulate(t, 3, seed=2))


def test_replicatesPureError():
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 3, (12, 2))
    x = np.vstack([base, base[:5], base[:5], base[:2]])
    y = np.sin(x[:, 0]) + 0.1 * rng.standard_normal(x.shape[0])
    m = KrigingModel(corr="matern52", poly="linear").fit(x, y)
    rep = m.replicates()
    assert rep["nGroups"] == 12 and rep["nReplicated"] == 5
    ss = 0.0
    for i in range(12):
        g = np.all(x == base[i], axis=1)
        ss += np.sum((y[g] - y[g].mean()) ** 2)
    assert rep["pureErrorVariance"] == pytest.approx(ss / (x.shape[0] - 12), rel=1e-12)
    assert m.spatialSummary()["pureErrorVariance"] == pytest.approx(rep["pureErrorVariance"])


def test_spatialColumnsCovariateEntersTrendLinearly():
    """A covariate column outside the kernel is recovered exactly as a linear trend term."""
    rng = np.random.default_rng(3)
    s = rng.uniform(0, 3, (50, 2))
    z = rng.uniform(-1, 1, 50)
    field = np.sin(s[:, 0]) + np.cos(s[:, 1])
    x = np.column_stack([s, z])
    m0 = KrigingModel(corr="matern52", poly="linear", spatialColumns=[0, 1], normalize=False).fit(x, field + 2.5 * z)
    m1 = KrigingModel(corr="matern52", poly="linear", spatialColumns=[0, 1], normalize=False,
                      hyperparameters=list(m0.hyperparameters["logParams"].values())).fit(np.column_stack([s, z]), field)
    t = np.column_stack([rng.uniform(0, 3, (10, 2)), rng.uniform(-1, 1, 10)])
    np.testing.assert_allclose(m0.predict(t).ravel() - 2.5 * t[:, 2], m1.predict(t).ravel(), rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(m0.predictGradient(t)[:, 2, 0], m1.predictGradient(t)[:, 2, 0] + 2.5, rtol=1e-7)


# ---------------------------------------------------------------- thin-plate spline GCV
def _tpsSmoother(x, lam):
    n = x.shape[0]
    cols = [RbfModel(kernel="thinPlateSpline", degree=1, smoothing=lam, normalize="range").fit(x, e).predict(x).ravel()
            for e in np.eye(n)]
    return np.column_stack(cols)


def test_tpsGcvMatchesExplicitSmoother():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (30, 2))
    y = np.sin(3 * x[:, 0]) + x[:, 1] ** 2 + 0.1 * rng.standard_normal(30)
    m = createModel("tps").fit(x, y)
    lam = m.smoothingValue
    n = x.shape[0]

    def gcv(l):
        s = _tpsSmoother(x, l)
        res = y - s @ y
        return float(np.mean(res ** 2) / (1 - np.trace(s) / n) ** 2)

    g0 = gcv(lam)
    assert m._gcv == pytest.approx(g0, rel=1e-7)
    np.testing.assert_allclose(m.predict(x).ravel(), _tpsSmoother(x, lam) @ y, rtol=1e-8, atol=1e-10)
    assert g0 <= gcv(lam * 1.5) + 1e-12 and g0 <= gcv(lam / 1.5) + 1e-12


def test_gcvLambdaMinimizesExplicitGcv():
    x, y = _spatialData(50, seed=4)
    m = KrigingModel(corr={"type": "matern", "nu": 1.5, "parameterization": "range", "ard": False},
                     poly="linear", normalize=False, likelihood="gcv").fit(x, y)
    fs = m.spatialSummary()
    ell, lam = fs["range"], fs["lambda"]
    n = x.shape[0]

    def gcv(l):
        s = _explicit(x, y, ell, l)["s"]
        res = y - s @ y
        return float(np.mean(res ** 2) / (1 - np.trace(s) / n) ** 2)

    g0 = gcv(lam)
    assert fs["gcv"] == pytest.approx(g0, rel=1e-7)
    assert g0 <= gcv(lam * 1.3) and g0 <= gcv(lam / 1.3)


@pytest.mark.parametrize("criterion", ["ml", "reml", "restrictedProfile"])
def test_likelihoodGradientsFiniteDifference(criterion):
    x, y = _spatialData(30)
    m = KrigingModel(corr={"type": "matern", "nu": "estimate", "ard": True}, poly="linear",
                     likelihood=criterion).fit(x, y)
    m._cache = m._kernel.trainingCache(m._xs[:, m._sc])
    p = m._params + np.array([0.1, -0.2, 0.05, 0.3])
    val, grad = m._negLogLikelihood(p)
    h = 1e-6
    num = [(m._negLogLikelihood(p + h * e)[0] - m._negLogLikelihood(p - h * e)[0]) / (2 * h) for e in np.eye(p.size)]
    np.testing.assert_allclose(grad, num, rtol=1e-5, atol=1e-6)


def test_restrictedProfileMaximizesLogRestrictedProfile():
    """likelihood='restrictedProfile' maximizes logRestrictedProfile = logLikelihood + log|Omega| / 2."""
    x, y = _spatialData(50, seed=2)
    corr = {"type": "matern", "nu": 1.0, "parameterization": "range", "ard": False}
    m = KrigingModel(corr=corr, poly="linear", normalize=False, likelihood="restrictedProfile").fit(x, y)
    fs = m.spatialSummary()
    for da, dl in ((1.1, 1.0), (1 / 1.1, 1.0), (1.0, 1.3), (1.0, 1 / 1.3)):
        other = KrigingModel(corr=corr, poly="linear", normalize=False, nugget=fs["lambda"] * dl, likelihood="ml",
                             hyperparameters=[np.log(fs["range"] * da)]).fit(x, y)
        assert other.spatialSummary()["logRestrictedProfile"] < fs["logRestrictedProfile"]
