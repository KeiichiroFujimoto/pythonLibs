"""Closed-form consistency of the Kriging family: trend uncertainty, likelihood values, input scaling."""
import warnings

import numpy as np
import pytest

from pythonLibs.regressionHandler import GradientKrigingModel, KrigingModel, ScalableKrigingModel


def _data(n=40, seed=2):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, 2))
    y = 5 * np.sin(4 * x[:, 0]) + x[:, 1] ** 2 + 0.1 * rng.standard_normal(n)
    return x, y


@pytest.mark.parametrize("approximation", ["vecchia", "fitc"])
def test_scalableWithAllNeighboursIsUniversalKriging(approximation):
    x, y = _data()
    n = x.shape[0]
    exact = KrigingModel(poly="linear", likelihood="reml").fit(x, y)
    extra = {"nInducing": n} if approximation == "fitc" else {}
    s = ScalableKrigingModel(approximation=approximation, poly="linear", likelihood="reml",
                             hyperparameters=exact._params.tolist(), neighbors=n - 1, predictionNeighbors=n,
                             **extra).fit(x, y)
    xp = np.random.default_rng(3).uniform(-0.2, 1.2, (6, 2))       # includes extrapolation: large trend term
    tol = 1e-8 if approximation == "vecchia" else 2e-3            # FITC keeps a tiny inducing-point jitter
    np.testing.assert_allclose(s.predictVariances(xp), exact.predictVariances(xp), rtol=tol)
    assert s.hyperparameters["logLikelihood"] == pytest.approx(exact.hyperparameters["logLikelihood"], rel=tol)
    back = ScalableKrigingModel.fromDict(s.toDict())
    np.testing.assert_allclose(back.predictVariances(xp), s.predictVariances(xp), rtol=1e-12)


def test_reportedLogLikelihoodIsTheMaximizedLikelihood():
    x, y = _data()
    n = x.shape[0]
    for crit in ("ml", "reml"):
        m = KrigingModel(poly="linear", likelihood=crit).fit(x, y)
        # dense Gaussian (restricted) log-likelihood on the output scale at the fitted parameters
        xs, ys = m._xs, m._ys * m._yStd
        c = m._kernel.matrix(xs, xs, m._kp) + m._eta * np.eye(n)
        f = m._f
        ci = np.linalg.inv(c)
        a = f.T @ ci @ f
        beta = np.linalg.solve(a, f.T @ ci @ ys)
        r = ys - f @ beta
        q = f.shape[1]
        dof = n - q if crit == "reml" else n
        s2 = float(r @ ci @ r) / dof
        ll = -0.5 * (dof * np.log(2 * np.pi * s2) + np.linalg.slogdet(c)[1] + dof
                     + (np.linalg.slogdet(a)[1] if crit == "reml" else 0.0))
        assert m.hyperparameters["logLikelihood"] == pytest.approx(ll, rel=1e-9)
        if crit == "ml":
            assert m.spatialSummary()["logLikelihood"] == pytest.approx(ll, rel=1e-9)


@pytest.mark.parametrize("opts", [{"poly": "linear"}, {"spatialColumns": [0]}])
def test_constantTrendColumnIsDropped(opts):
    x1 = np.linspace(0, 1, 8)
    x = np.column_stack([x1, np.ones(8)])
    m = KrigingModel(**opts).fit(x, np.sin(x1))
    ref = KrigingModel(**({"poly": "linear"} if "poly" in opts else {})).fit(x1[:, None], np.sin(x1))
    xp = [[0.33, 1.0]]
    assert np.ravel(m.predict(xp))[0] == pytest.approx(np.ravel(ref.predict([[0.33]]))[0], abs=1e-6)
    np.testing.assert_allclose(KrigingModel.fromDict(m.toDict()).predictVariances(xp), m.predictVariances(xp))


def test_greatCircleChildSeesDegrees():
    rng = np.random.default_rng(4)
    lonlat = np.column_stack([rng.uniform(10, 20, 50), rng.uniform(50, 55, 50)])
    t = rng.uniform(0, 30, 50)
    x = np.column_stack([lonlat, t])
    y = np.sin(lonlat[:, 0] / 3) + np.cos(lonlat[:, 1]) + 0.05 * t
    corr = {"type": "product", "kernels": [{"type": "matern52", "distance": "greatCircle"}, "matern52"],
            "columns": [[0, 1], [2]]}
    m = KrigingModel(corr=corr).fit(x, y)
    np.testing.assert_array_equal(m._xs[:, :2], lonlat)                     # degrees, not z-scores
    assert m._xStd[2] == pytest.approx(t.std())
    child = m._kernel._children()[0]
    d = child.greatCircleDistance(lonlat[:1], lonlat[1:2])[0, 0]
    k = m._kernel.matrix(m._xs[:2], m._xs[:2], m._kp)[0, 1]
    kt = m._kernel._children()[1].matrix(m._xs[:1, 2:], m._xs[1:2, 2:], m._kp[1:])[0, 0]
    ell = np.exp(m._kp[0])
    r = np.sqrt(5.0) * d / ell
    assert k == pytest.approx((1 + r + r * r / 3) * np.exp(-r) * kt, rel=1e-10)
    np.testing.assert_allclose(KrigingModel.fromDict(m.toDict()).predict(x[:3]), m.predict(x[:3]), rtol=1e-12)


def test_isotropicLengthscaleReportedPerColumn():
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 1, (30, 3)) * np.array([1.0, 10.0, 100.0])
    y = np.sin(3 * x[:, 0]) + 0.1 * x[:, 1]
    m = KrigingModel(corr={"type": "matern52", "ard": False}).fit(x, y)
    ls = m.spatialSummary()["lengthscales"] if "lengthscales" in m.spatialSummary() else m.spatialSummary()["range"]
    assert len(ls) == 3
    np.testing.assert_allclose(np.array(ls) / m._xStd, np.exp(m._kp[0]), rtol=1e-12)


def test_nonstationaryFieldFollowsTheDataRange():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 100, (60, 1))
    f = lambda v: np.sin(v / 10 * (1 + v / 100))                                    # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for basis in ("linear", "rbf"):
            m = KrigingModel(corr={"type": "nonstationary", "basis": basis}, normalize=False).fit(x, f(x[:, 0]))
            xt = np.linspace(0, 100, 200)
            assert np.sqrt(np.mean((m.predict(xt[:, None]).ravel() - f(xt)) ** 2)) < 5e-3
            np.testing.assert_allclose(KrigingModel.fromDict(m.toDict()).predict(x[:5]), m.predict(x[:5]))


@pytest.mark.parametrize("corr", ["periodic", {"type": "warped"}, {"type": "nonstationary"},
                                  {"type": "gneiting", "timeColumn": 3}])
def test_plsIsRejectedWhereUnsupported(corr):
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError, match="PLS"):
        KrigingModel(corr=corr, plsComponents=1).fit(rng.uniform(size=(20, 4)), rng.uniform(size=20))


def test_gradientKrigingValidation():
    x = np.linspace(0, 1, 5)[:, None]
    y = np.column_stack([np.sin(3 * x[:, 0]), 3 * np.cos(3 * x[:, 0])])
    with pytest.raises(ValueError, match="nu > 1"):
        GradientKrigingModel(corr={"type": "matern", "nu": 0.5}).fit(x, y)
    with pytest.raises(ValueError, match="hyperparameters needs"):
        GradientKrigingModel(corr="matern52", hyperparameters=[0.0, 0.0, 5.0]).fit(x, y)
    m = GradientKrigingModel(corr={"type": "matern", "nu": "estimate"}).fit(x, y)
    assert m._kernel.options["nuBounds"][0] > 1.0
