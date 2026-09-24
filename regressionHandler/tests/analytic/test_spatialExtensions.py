"""LOO, variograms, block prediction, space-time, cokriging, scalable and non-stationary Kriging vs closed forms."""
import numpy as np
import pytest

from pythonLibs.regressionHandler import (CokrigingModel, KrigingModel, ScalableKrigingModel, crossValidate)
from pythonLibs.regressionHandler.evaluation import EmpiricalVariogram, empiricalVariogram, fitVariogram
from pythonLibs.regressionHandler.kernels import (Gneiting, Matern, Matern52, NonstationaryKernel, ProductKernel,
                                                  Spherical, SquaredExponential, WarpedKernel, buildKernel)
from pythonLibs.regressionHandler.models.ScalableKrigingModel import maxminOrder, previousNeighbors


def _data(n=30, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 3, (n, 2))
    return x, np.sin(x[:, 0]) + 0.3 * x[:, 1] ** 2 + 0.05 * rng.standard_normal(n)


def _fixed(x, y, **kw):
    opts = dict(corr="matern52", poly="linear", normalize=False, nugget=1e-2, likelihood="ml",
                hyperparameters=[np.log(1.1), np.log(0.8)])
    opts.update(kw)
    return KrigingModel(**opts).fit(x, y)


# ---------------------------------------------------------------- leave-one-out
def test_krigingLooEqualsRefits():
    x, y = _data()
    m = _fixed(x, y)
    d = m.looDiagnostics()
    for i in range(x.shape[0]):
        keep = np.arange(x.shape[0]) != i
        mi = _fixed(x[keep], y[keep])
        pred = float(np.ravel(mi.predict(x[i:i + 1]))[0])
        assert d["residuals"][i] == pytest.approx(y[i] - pred, abs=1e-9)
        # prediction variance of y_i with sigma2 held at the full-data value
        vi = float(np.ravel(mi.predictVariances(x[i:i + 1], "prediction"))[0]) / mi._sigma2 * m._sigma2
        assert d["variances"][i] == pytest.approx(vi, rel=1e-8)
    np.testing.assert_allclose(crossValidate(m, x, y, method="analytic").predictions.ravel(), d["predictions"],
                               atol=1e-12)
    np.testing.assert_allclose(d["standardized"], d["residuals"] / np.sqrt(d["variances"]))


# ---------------------------------------------------------------- variograms
def test_empiricalVariogramBruteForce():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, (40, 2))
    y = rng.standard_normal(40)
    edges = np.linspace(0, 0.8, 6)
    ev = empiricalVariogram(x, y, binEdges=edges, chunkSize=7)
    i, j = np.triu_indices(40, 1)
    d = np.linalg.norm(x[i] - x[j], axis=1)
    dy = y[i] - y[j]
    for k, b in enumerate(ev.extra["binIndex"]):
        sel = (d > edges[b]) & (d <= edges[b + 1])
        assert ev.counts[k] == sel.sum()
        assert ev.gamma[k] == pytest.approx(0.5 * np.mean(dy[sel] ** 2), rel=1e-12)
        assert ev.distance[k] == pytest.approx(d[sel].mean(), rel=1e-12)
    rob = empiricalVariogram(x, y, binEdges=edges, estimator="robust")
    b = rob.extra["binIndex"][0]
    sel = (d > edges[b]) & (d <= edges[b + 1])
    nh = sel.sum()
    assert rob.gamma[0] == pytest.approx(0.5 * np.mean(np.sqrt(np.abs(dy[sel]))) ** 4 / (0.457 + 0.494 / nh + 0.045 / nh ** 2),
                                         rel=1e-12)


def test_directionalVariogramOnGrid():
    g = np.arange(6.0)
    x = np.array([[a, b] for a in g for b in g])
    y = x[:, 0] * 2.0                          # varies only along the first axis
    along = empiricalVariogram(x, y, binEdges=[0.5, 1.5], direction=0.0, tolerance=1.0)
    across = empiricalVariogram(x, y, binEdges=[0.5, 1.5], direction=90.0, tolerance=1.0)
    assert along.gamma[0] == pytest.approx(0.5 * 4.0) and along.counts[0] == 30
    assert across.gamma[0] == pytest.approx(0.0) and across.counts[0] == 30


@pytest.mark.parametrize("model, nu", [("exponential", None), ("gaussian", None), ("spherical", None),
                                       ("matern", 1.5), ("wendland", None), ("cubic", None)])
def test_fitVariogramRecoversNoiseFreeModel(model, nu):
    h = np.linspace(0.05, 2.0, 25)
    true = dict(nugget=0.1, partialSill=0.9, range=0.6)
    from pythonLibs.regressionHandler.evaluation.Variogram import VARIOGRAM_MODELS
    g = true["nugget"] + true["partialSill"] * (1 - VARIOGRAM_MODELS[model](h / true["range"], nu))
    ev = EmpiricalVariogram(binEdges=np.r_[0, h], distance=h, gamma=g, counts=np.full(h.size, 50), estimator="classical")
    fit = fitVariogram(ev, model, nu=nu)
    assert fit.nugget == pytest.approx(0.1, abs=1e-5)
    assert fit.partialSill == pytest.approx(0.9, rel=1e-5)
    assert fit.range == pytest.approx(0.6, rel=1e-5)
    np.testing.assert_allclose(fit.gamma(h), g, rtol=1e-6)
    assert fit.covariance(np.array([0.0]))[0] == pytest.approx(1.0, rel=1e-5)


@pytest.mark.parametrize("model, nu", [("exponential", None), ("matern", 1.5), ("gaussian", None), ("spherical", None),
                                       ("wendland", None)])
def test_variogramKrigingOptionsReproduceCovariance(model, nu):
    from pythonLibs.regressionHandler.evaluation.Variogram import VariogramFit
    fit = VariogramFit(model=model, nugget=0.2, partialSill=1.5, range=0.7, nu=nu, sse=0.0, weights="equal")
    opts = fit.krigingOptions()
    k = buildKernel(opts["corr"]).setup(1)
    h = np.array([0.05, 0.3, 0.6, 1.2])
    kv = k.matrix(np.zeros((1, 1)), h[:, None], np.array(opts["hyperparameters"]))[0]
    np.testing.assert_allclose(fit.partialSill * kv, fit.covariance(h), rtol=1e-12, atol=1e-15)
    assert opts["nugget"] == pytest.approx(0.2 / 1.5)


def test_sphericalKernelClosedFormAndGradient():
    k = Spherical(ard=False).setup(1)
    for d in (0.0, 0.3, 0.9, 1.0, 1.4):
        r = d / 0.8
        val = k.matrix(np.zeros((1, 1)), np.array([[d]]), np.array([np.log(0.8)]))[0, 0]
        assert val == pytest.approx(1 - 1.5 * r + 0.5 * r ** 3 if r < 1 else 0.0, abs=1e-15)
    x = np.random.default_rng(2).uniform(0, 2, (10, 2))
    k2 = Spherical(ard=True).setup(2)
    p = np.log([1.5, 2.0])
    g = k2.gradients(x, p)
    for i in range(2):
        e = np.zeros(2)
        e[i] = 1e-6
        np.testing.assert_allclose(g[i], (k2.matrix(x, x, p + e) - k2.matrix(x, x, p - e)) / 2e-6, atol=1e-7)


# ---------------------------------------------------------------- block prediction
def test_blockPredictionClosedForm():
    x, y = _data()
    m = _fixed(x, y)
    pts = np.array([[0.5, 0.5], [1.0, 2.0], [2.5, 1.5]])
    w = np.array([1.0, 2.0, 1.0])
    out = m.predictBlock([pts, pts[:1]], weights=[w, None])
    wn = w / w.sum()
    cov = m.predictCovariance(pts)[0]
    assert out["mean"][0, 0] == pytest.approx(wn @ m.predict(pts).ravel(), rel=1e-12)
    assert out["variance"][0, 0] == pytest.approx(wn @ cov @ wn, rel=1e-10)
    assert out["mean"][1, 0] == pytest.approx(float(np.ravel(m.predict(pts[:1]))[0]), rel=1e-12)
    assert out["variance"][1, 0] == pytest.approx(float(np.ravel(m.predictVariances(pts[:1]))[0]), rel=1e-10)
    box = m.predictBlock([{"lower": [0, 0], "upper": [1, 2], "n": 4}])
    assert box["points"][0].shape == (16, 2)
    np.testing.assert_allclose(np.sort(np.unique(box["points"][0][:, 0])), [0.125, 0.375, 0.625, 0.875])
    # averaging reduces the variance below the mean point variance
    assert box["variance"][0, 0] < np.mean(m.predictVariances(box["points"][0]))


# ---------------------------------------------------------------- space-time kernels
def test_productColumnsEqualsManualProduct():
    rng = np.random.default_rng(3)
    xa, xb = rng.uniform(0, 2, (6, 3)), rng.uniform(0, 2, (5, 3))
    kern = buildKernel({"type": "product", "kernels": [{"type": "matern52", "ard": True}, "squaredExponential"],
                        "columns": [[0, 1], [2]]}).setup(3)
    p = np.array([0.1, -0.2, 0.3])
    s, t = Matern52(ard=True).setup(2), SquaredExponential().setup(1)
    ref = s.matrix(xa[:, :2], xb[:, :2], p[:2]) * t.matrix(xa[:, 2:], xb[:, 2:], p[2:])
    np.testing.assert_allclose(kern.matrix(xa, xb, p), ref, rtol=1e-14)
    np.testing.assert_allclose(kern.batchMatrix(xa[None], xb[None], p)[0], ref, rtol=1e-14)
    assert kern.lengthscaleColumns() == [0, 1, 2]
    for kx in range(3):
        h = 1e-6
        up, dn = xa.copy(), xa.copy()
        up[:, kx] += h
        dn[:, kx] -= h
        np.testing.assert_allclose(kern.dx(xa, xb, p, kx), (kern.matrix(up, xb, p) - kern.matrix(dn, xb, p)) / (2 * h),
                                   atol=1e-8)
    x = rng.uniform(0, 2, (8, 3))
    for g, i in zip(kern.gradients(x, p), range(3)):
        e = np.zeros(3)
        e[i] = 1e-6
        np.testing.assert_allclose(g, (kern.matrix(x, x, p + e) - kern.matrix(x, x, p - e)) / 2e-6, atol=1e-7)


def test_gneitingClosedForm():
    rng = np.random.default_rng(4)
    xa, xb = rng.uniform(0, 2, (5, 3)), rng.uniform(0, 2, (4, 3))
    ls, lt, a, g = 0.7, 1.3, 0.8, 0.6
    h = np.sqrt(((xa[:, None, :2] - xb[None, :, :2]) ** 2).sum(-1))
    u = np.abs(xa[:, None, 2] - xb[None, :, 2])
    psi = (u / lt) ** (2 * a) + 1
    for beta in (0.0, 0.4, 1.0):
        k = Gneiting(alpha=a, gamma=g, beta=beta).setup(3)
        ref = psi ** -1.0 * np.exp(-(h / ls) ** (2 * g) / psi ** (beta * g))
        np.testing.assert_allclose(k.matrix(xa, xb, np.log([ls, lt])), ref, rtol=1e-13)
    sep = Gneiting(alpha=a, gamma=g, beta=0.0).setup(3)
    np.testing.assert_allclose(sep.matrix(xa, xb, np.log([ls, lt])), np.exp(-(h / ls) ** (2 * g)) / psi, rtol=1e-13)
    est = Gneiting().setup(3)
    x = rng.uniform(0, 3, (40, 3))
    assert np.linalg.eigvalsh(est.matrix(x, x, np.array([0.0, 0.2, 0.5]))).min() > -1e-10


# ---------------------------------------------------------------- cokriging
def test_cokrigingSingleOutputEqualsKriging():
    x, y = _data()
    s, tau2, ell = 1.7, 0.02, 0.9
    cok = CokrigingModel(corr={"type": "matern52", "ard": False}, poly="linear", normalize=False, likelihood="ml",
                         hyperparameters=[np.log(ell), s, np.log(tau2)]).fit(x, y)
    kri = KrigingModel(corr={"type": "matern52", "ard": False}, poly="linear", normalize=False, likelihood="ml",
                       nugget=tau2 / s ** 2, hyperparameters=[np.log(ell)]).fit(x, y)
    t = np.random.default_rng(5).uniform(0, 3, (10, 2))
    np.testing.assert_allclose(cok.predict(t).ravel(), kri.predict(t).ravel(), rtol=1e-9, atol=1e-11)
    ratio = cok.predictVariances(t).ravel() / kri.predictVariances(t).ravel()
    np.testing.assert_allclose(ratio, s ** 2 / kri._sigma2, rtol=1e-8)


def test_cokrigingIndependentOutputsEqualSeparateKriging():
    rng = np.random.default_rng(6)
    x = rng.uniform(0, 3, (25, 2))
    y = np.column_stack([np.sin(x[:, 0]), np.cos(x[:, 1])]) + 0.02 * rng.standard_normal((25, 2))
    y[15:, 0] = np.nan
    ell, s0, s1, t0, t1 = 0.8, 1.2, 0.7, 0.01, 0.03
    params = [np.log(ell), s0, 0.0, 0.0, s1, np.log(t0), np.log(t1)]          # L = diag(s0, s1)
    cok = CokrigingModel(corr={"type": "matern52", "ard": False}, poly="constant", normalize=False,
                         hyperparameters=params).fit(x, y)
    t = rng.uniform(0, 3, (8, 2))
    for j, (sj, tj) in enumerate(((s0, t0), (s1, t1))):
        obs = np.isfinite(y[:, j])
        kri = KrigingModel(corr={"type": "matern52", "ard": False}, poly="constant", normalize=False,
                           nugget=tj / sj ** 2, hyperparameters=[np.log(ell)]).fit(x[obs], y[obs, j])
        np.testing.assert_allclose(cok.predict(t)[:, j], kri.predict(t).ravel(), rtol=1e-9, atol=1e-11)


def test_cokrigingLikelihoodGradient():
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 3, (30, 2))
    y = np.column_stack([np.sin(x[:, 0]), np.sin(x[:, 0]) + x[:, 1]]) + 0.1 * rng.standard_normal((30, 2))
    y[20:, 0] = np.nan
    for likelihood in ("reml", "ml"):
        m = CokrigingModel(corr=["matern32", "squaredExponential"], poly="linear", likelihood=likelihood,
                           nStart=1, maxIter=3).fit(x, y)
        p = m._params + 0.05
        _, g = m._negLogLikelihood(p)
        num = [(m._negLogLikelihood(p + 1e-6 * e)[0] - m._negLogLikelihood(p - 1e-6 * e)[0]) / 2e-6
               for e in np.eye(p.size)]
        np.testing.assert_allclose(g, num, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------- scalable Kriging
def test_vecchiaWithAllNeighboursIsExact():
    x, y = _data(40)
    hp = [np.log(1.1), np.log(0.8), np.log(1e-2)]
    ex = KrigingModel(corr="matern52", poly="linear", likelihood="ml", hyperparameters=hp).fit(x, y)
    ve = ScalableKrigingModel(corr="matern52", poly="linear", hyperparameters=hp, neighbors=39,
                              predictionNeighbors=40, ordering="random").fit(x, y)
    t = np.random.default_rng(8).uniform(0, 3, (10, 2))
    assert ve._logLik == pytest.approx(ex._logLik, rel=1e-9)
    np.testing.assert_allclose(ve.predict(t), ex.predict(t), rtol=1e-8, atol=1e-10)


def test_fitcWithAllPointsInducingIsExact():
    x, y = _data(40)
    hp = [np.log(1.1), np.log(0.8), np.log(1e-2)]
    ex = KrigingModel(corr="matern52", poly="linear", likelihood="ml", hyperparameters=hp).fit(x, y)
    fi = ScalableKrigingModel(corr="matern52", poly="linear", hyperparameters=hp, approximation="fitc",
                              nInducing=40).fit(x, y)
    t = np.random.default_rng(9).uniform(0, 3, (10, 2))
    assert fi._logLik == pytest.approx(ex._logLik, rel=1e-5)
    np.testing.assert_allclose(fi.predict(t), ex.predict(t), rtol=1e-5, atol=1e-6)


def test_previousNeighborsExactAndMaxminOrder():
    rng = np.random.default_rng(10)
    x = rng.uniform(0, 1, (3000, 2))
    m = 8
    nb = previousNeighbors(x, m)
    agree = 0
    for i in range(m, 3000, 7):
        d = np.sum((x[:i] - x[i]) ** 2, axis=1)
        agree += set(np.argsort(d)[:m]) == set(nb[i])
    assert agree / len(range(m, 3000, 7)) > 0.99
    assert np.all(nb[np.arange(3000)[:, None] > np.arange(m)[None, :]] >= 0)
    o = maxminOrder(x[:200])
    xs = x[:200][o]
    for k in range(1, 20):
        dmin = np.min(np.sum((xs[k:, None] - xs[None, :k]) ** 2, axis=2), axis=1)
        assert dmin[0] == pytest.approx(dmin.max())


# ---------------------------------------------------------------- non-stationary kernels
def test_nonstationaryAndWarpedReduceToStationary():
    x = np.random.default_rng(11).uniform(-1, 1, (9, 2))
    s = Matern52(ard=False).setup(2)
    ns = NonstationaryKernel(kernel="matern52").setup(2)
    np.testing.assert_allclose(ns.matrix(x, x, np.array([0.3, 0.0, 0.0])), s.matrix(x, x, np.array([0.3])), atol=1e-15)
    w = WarpedKernel(kernel={"type": "matern52", "ard": False}).setup(2)
    np.testing.assert_allclose(w.matrix(x, x, np.array([0.3, 0, 0, 0, 0])), s.matrix(x, x, np.array([0.3])), atol=1e-15)


def test_nonstationaryClosedFormAndPositiveDefinite():
    rng = np.random.default_rng(12)
    x = rng.uniform(-1, 1, (30, 2))
    ns = NonstationaryKernel(kernel={"type": "squaredExponential"}).setup(2)
    p = np.array([0.2, 0.7, -0.4])
    l = np.exp(0.2 + x @ p[1:])
    mean = 0.5 * (l[:, None] ** 2 + l[None, :] ** 2)
    d2 = ((x[:, None] - x[None]) ** 2).sum(-1)
    ref = (l[:, None] * l[None, :] / mean) * np.exp(-0.5 * d2 / mean)
    np.testing.assert_allclose(ns.matrix(x, x, p), ref, rtol=1e-12)
    assert np.linalg.eigvalsh(ns.matrix(x, x, p)).min() > -1e-10
    np.testing.assert_allclose(ns.batchMatrix(x[None], x[None], p)[0], ref, rtol=1e-12)
    w = WarpedKernel(kernel="matern52").setup(2)
    pw = np.array([0.1, 0.2, 0.3, -0.2, 0.5, 0.1])
    u = np.sinh(np.exp(pw[2:4]) * np.arcsinh(x) - pw[4:])
    np.testing.assert_allclose(w.matrix(x, x, pw), Matern52().setup(2).matrix(u, u, pw[:2]), rtol=1e-13)
