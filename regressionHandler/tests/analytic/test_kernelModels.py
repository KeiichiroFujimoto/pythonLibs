"""Kriging, RBF and IDW against the closed-form interpolation / posterior formulas."""
import numpy as np
import pytest

from pythonLibs.regressionHandler import IdwModel, KrigingModel, RbfModel
from pythonLibs.regressionHandler.bases.RadialBasis import rbfValue
from pythonLibs.regressionHandler.kernels import Matern32, Matern52, SquaredExponential


def _data(n=25, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 2, (n, 2))
    return x, np.sin(2 * x[:, 0]) + x[:, 1] ** 2


def test_kernelClosedFormValues():
    x = np.zeros((1, 1))
    r = np.array([[0.7]])
    for kernel, f in ((SquaredExponential(), lambda d: np.exp(-d * d / 2)),
                      (Matern32(), lambda d: (1 + np.sqrt(3) * d) * np.exp(-np.sqrt(3) * d)),
                      (Matern52(), lambda d: (1 + np.sqrt(5) * d + 5 * d * d / 3) * np.exp(-np.sqrt(5) * d))):
        kernel.setup(1)
        assert kernel.matrix(x, r, np.array([0.0]))[0, 0] == pytest.approx(f(0.7), rel=1e-14)
        # lengthscale l rescales the distance: k(d; l) = k(d / l; 1)
        assert kernel.matrix(x, r, np.array([np.log(2.0)]))[0, 0] == pytest.approx(f(0.35), rel=1e-14)


@pytest.mark.parametrize("poly", ["none", "constant", "linear"])
def test_krigingEqualsGaussianProcessPosterior(poly):
    """Universal Kriging mean and variance with fixed hyperparameters, written out explicitly."""
    x, y = _data()
    ell, eta = np.array([0.6, 1.3]), 1e-4
    m = KrigingModel(poly=poly, nugget=eta, likelihood="ml", hyperparameters=list(np.log(ell))).fit(x, y)
    mx, sx, my, sy = x.mean(0), x.std(0), y.mean(), y.std()
    xs, ys = (x - mx) / sx, (y - my) / sy
    t = np.random.default_rng(1).uniform(-1, 2, (15, 2))
    ts = (t - mx) / sx

    def k(a, b):
        d2 = sum(((a[:, [j]] - b[:, j][None, :]) / ell[j]) ** 2 for j in range(2))
        return np.exp(-0.5 * d2)

    def f(a):
        cols = {"none": [], "constant": [np.ones(len(a))], "linear": [np.ones(len(a)), a[:, 0], a[:, 1]]}[poly]
        return np.column_stack(cols) if cols else np.zeros((len(a), 0))

    r = k(xs, xs) + eta * np.eye(len(xs))
    ri = np.linalg.inv(r)
    fx, ft, kt = f(xs), f(ts), k(ts, xs)
    if fx.shape[1]:
        a = fx.T @ ri @ fx
        beta = np.linalg.solve(a, fx.T @ ri @ ys)
    else:
        a, beta = np.zeros((0, 0)), np.zeros(0)
    resid = ys - fx @ beta
    mean = ft @ beta + kt @ ri @ resid
    sigma2 = resid @ ri @ resid / len(ys)
    u = fx.T @ ri @ kt.T - ft.T
    var = 1 - np.sum(kt @ ri * kt, axis=1)
    if fx.shape[1]:
        var = var + np.sum(u * np.linalg.solve(a, u), axis=0)
    np.testing.assert_allclose(m.predict(t), my + sy * mean, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(m.predictVariances(t)[:, 0], sigma2 * np.maximum(var, 0) * sy ** 2, rtol=1e-6,
                               atol=1e-12)
    np.testing.assert_allclose(m.predictVariances(t, "prediction")[:, 0],
                               sigma2 * (np.maximum(var, 0) + eta) * sy ** 2, rtol=1e-6, atol=1e-12)


def test_krigingInterpolatesAndHasZeroVarianceAtData():
    x, y = _data()
    m = KrigingModel(nugget=1e-12, hyperparameters=[np.log(0.5), np.log(0.5)]).fit(x, y)
    np.testing.assert_allclose(m.predict(x), y, atol=1e-7)
    assert np.max(m.predictVariances(x)) < 1e-8 * np.var(y)


def test_rbfSolvesTheAugmentedSystem():
    x, y = _data(30)
    for kernel, eps, degree, smoothing in (("thinPlateSpline", 1.0, 1, 0.0), ("gaussian", 1.7, -1, 0.0),
                                           ("multiquadric", 0.8, 0, 1e-3), ("cubic", 1.0, 1, 0.1)):
        m = RbfModel(kernel=kernel, epsilon=eps, degree=degree, smoothing=smoothing).fit(x, y)
        mx, sx = x.mean(0), x.std(0)
        xs = (x - mx) / sx
        dist = np.sqrt(((xs[:, None, :] - xs[None, :, :]) ** 2).sum(-1))
        p = {-1: np.zeros((30, 0)), 0: np.ones((30, 1)), 1: np.column_stack([np.ones(30), xs])}[degree]
        q = p.shape[1]
        a = np.block([[rbfValue(kernel, dist, eps) + smoothing * np.eye(30), p], [p.T, np.zeros((q, q))]])
        sol = np.linalg.solve(a, np.r_[y, np.zeros(q)])
        t = np.random.default_rng(2).uniform(-1, 2, (10, 2))
        ts = (t - mx) / sx
        dt = np.sqrt(((ts[:, None, :] - xs[None, :, :]) ** 2).sum(-1))
        pt = {-1: np.zeros((10, 0)), 0: np.ones((10, 1)), 1: np.column_stack([np.ones(10), ts])}[degree]
        np.testing.assert_allclose(m.predict(t), rbfValue(kernel, dt, eps) @ sol[:30] + pt @ sol[30:], atol=1e-9)
        if smoothing == 0.0:
            np.testing.assert_allclose(m.predict(x), y, atol=1e-9)


def test_rbfReproducesItsPolynomialTail():
    x, _ = _data(40)
    plane = 1 + 2 * x[:, 0] - x[:, 1]
    t = np.random.default_rng(3).uniform(-1, 2, (30, 2))
    exact = 1 + 2 * t[:, 0] - t[:, 1]
    np.testing.assert_allclose(RbfModel().fit(x, plane).predict(t), exact, atol=1e-9)
    np.testing.assert_allclose(RbfModel(neighbors=12).fit(x, plane).predict(t), exact, atol=1e-9)


def test_idwIsAnExactConvexInterpolator():
    x, y = _data(40)
    m = IdwModel(p=2.0).fit(x, y)
    np.testing.assert_allclose(m.predict(x), y, atol=1e-12)
    t = np.array([[0.3, 0.4]])
    xs, ts = (x - x.mean(0)) / x.std(0), (t - x.mean(0)) / x.std(0)
    w = 1.0 / np.sum((xs - ts) ** 2, axis=1)                    # |d|^-2
    assert m.predict(t)[0] == pytest.approx(w @ y / w.sum(), rel=1e-12)
    np.testing.assert_allclose(IdwModel().fit(x, np.full(40, 3.2)).predict(t), 3.2)   # constants are exact
