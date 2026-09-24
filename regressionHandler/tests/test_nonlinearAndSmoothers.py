"""NonlinearModel + ModelLibrary, SplineModel and LOESS."""
import json

import numpy as np
import pytest

from pythonLibs.regressionHandler import (LocalRegressionModel, NonlinearModel, RegressionHandler, SplineModel,
                                          SurrogateModelBase, createModel, registerForm)
from pythonLibs.regressionHandler.models.ModelLibrary import LIBRARY

TRUTH = {
    "linear": ([1, 2], (0, 5)), "powerLaw": ([2, 1.5], (0.5, 5)), "powerLawOffset": ([2, 1.5, 3], (0.5, 5)),
    "exponential": ([2, -0.7], (0, 5)), "exponentialDecay": ([5, 1.2, 1], (0, 6)),
    "exponentialRise": ([4, 0.8, 1], (0, 5)), "inverseExponential": ([2e-3, 1800], (280, 900)),
    "logistic": ([3, 2, 1, 0.5], (-3, 5)), "saturation": ([5, 2], (0.1, 10)),
    "powerSaturation": ([5, 2, 2.5], (0.1, 6)), "gaussianPeak": ([4, 2, 0.5, 1], (0, 4)),
    "weibullCdf": ([3, 2], (0.2, 8)), "sinusoid": ([2, 0.7, 0.4, 1], (0, 10)),
    "rationalPower": ([3.0, 1.5, 4.0], (0.5, 40)), "linearPlusPower": ([2.0, 0.05, 3.0], (0.1, 10)),
    "logarithmic": ([1, 2], (0.5, 10)), "inverse": ([1, 2], (0.5, 10)),
}


def _evaluate(name, params, x):
    m = NonlinearModel(library=name, p0=list(params)).fit(x, np.ones_like(x))
    m._p = np.array(params, dtype=float)
    return m.predict(x)


def test_libraryCoversEveryForm():
    assert set(TRUTH) == set(LIBRARY)


@pytest.mark.parametrize("name", sorted(TRUTH))
def test_libraryRecoversParametersFromAutomaticGuess(name):
    params, (lo, hi) = TRUTH[name]
    x = np.linspace(lo, hi, 60 if name == "sinusoid" else 40)
    y = _evaluate(name, params, x)
    m = NonlinearModel(library=name).fit(x, y)
    est = np.array(list(m.parameters.values()))
    np.testing.assert_allclose(est, params, rtol=1e-5)


def test_expressionCallableBoundsAndMultiOutput():
    rng = np.random.default_rng(1)
    x = rng.uniform(1, 3, (50, 2))
    y = 2 * x[:, 0] ** 1.3 * x[:, 1] ** -0.5
    m = NonlinearModel(expression="C*T**a*P**b", params=["C", "a", "b"], variables=["T", "P"], p0=[1, 1, 0]).fit(x, y)
    np.testing.assert_allclose(list(m.parameters.values()), [2, 1.3, -0.5], rtol=1e-8)
    bounded = NonlinearModel(expression="C*T**a*P**b", params=["C", "a", "b"], variables=["T", "P"],
                             p0={"C": 1, "a": 1, "b": 0}, bounds={"a": [0, 1.0]}).fit(x, y)
    assert bounded.parameters["a"] == pytest.approx(1.0)
    fn = NonlinearModel(function=lambda v, a, b: a * np.exp(b * v[:, 0]) + v[:, 1], params=["a", "b"]).fit(
        x, 2 * np.exp(0.3 * x[:, 0]) + x[:, 1])
    np.testing.assert_allclose(list(fn.parameters.values()), [2, 0.3], rtol=1e-8)
    with pytest.raises(NotImplementedError):
        fn.toDict()
    lin = createModel("logarithmic").fit(np.linspace(1, 5, 20),
                                                               np.column_stack([np.log(np.linspace(1, 5, 20)),
                                                                                2 * np.log(np.linspace(1, 5, 20))]))
    assert lin.result.params.shape == (2, 2)


def test_robustLossAndMultiStart():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 6, 60)
    y = 5 * np.exp(-x / 1.2) + 1 + rng.normal(0, 0.02, 60)
    y[[5, 25, 45]] += 2.0
    plain = createModel("exponentialDecay").fit(x, y)
    robust = NonlinearModel(library="exponentialDecay", loss="soft_l1", fScale=0.05, nStart=3).fit(x, y)
    truth = np.array([5, 1.2, 1])
    assert np.abs(np.array(list(robust.parameters.values())) - truth).max() < \
        0.3 * np.abs(np.array(list(plain.parameters.values())) - truth).max()


def test_badSpecs():
    with pytest.raises(ValueError):
        NonlinearModel(expression="a*x", params=["a", "unused"])
    with pytest.raises(ValueError):
        NonlinearModel(expression="a*x")
    with pytest.raises(KeyError):
        NonlinearModel(library="noSuchForm")
    with pytest.raises(ValueError):
        NonlinearModel(expression="__import__('os').system('x')", params=["a"])


def test_registerCustomForm():
    registerForm("doubleExp", "a*exp(-x/t1) + b*exp(-x/t2)", ["a", "t1", "b", "t2"],
                 guess=lambda x, y: [1.0, 0.5, 1.0, 3.0], description="two-term decay")
    x = np.linspace(0, 10, 80)
    y = 3 * np.exp(-x / 0.4) + 1.5 * np.exp(-x / 4.0)
    m = createModel("doubleExp").fit(x, y)
    np.testing.assert_allclose(sorted(m.parameters.values()), sorted([3, 0.4, 1.5, 4.0]), rtol=1e-6)
    LIBRARY.pop("doubleExp")


def test_nonlinearRoundTrip():
    x = np.linspace(0.5, 40, 30)
    m = createModel("rationalPower").fit(x, 3.0 * x ** 1.5 / (x + 4.0) * (1 + 0.001 * np.sin(x)))
    loaded = SurrogateModelBase.fromDict(json.loads(json.dumps(m.toDict())))
    np.testing.assert_allclose(loaded.predictInterval(x).upper, m.predictInterval(x).upper)


def test_splineModel():
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 10, 300))
    y = np.sin(x) + rng.normal(0, 0.2, 300)
    s = SplineModel(nSegments=30).fit(x, y)
    t = np.linspace(0.2, 9.8, 200)
    assert np.sqrt(np.mean((s.predict(t) - np.sin(t)) ** 2)) < 0.06
    loaded = SurrogateModelBase.fromDict(json.loads(json.dumps(s.toDict())))
    assert type(loaded) is SplineModel
    np.testing.assert_allclose(loaded.predict(t), s.predict(t))
    x2 = rng.uniform(0, 1, (400, 2))
    s2 = SplineModel(nSegments=[8, 8]).fit(x2, np.sin(4 * x2[:, 0]) * x2[:, 1])
    assert s2.metrics[0].rSquared > 0.999


def test_loessIntervalsRobustnessAndND():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 10, 300))
    y = np.sin(x) + rng.normal(0, 0.2, 300)
    t = np.linspace(0.5, 9.5, 300)
    m = LocalRegressionModel(span=0.15, degree=2).fit(x, y)
    band = m.predictInterval(t, 0.95, "confidence")
    assert np.mean((band.lower[:, 0] <= np.sin(t)) & (np.sin(t) <= band.upper[:, 0])) > 0.9
    assert np.sqrt(m._sigma2) == pytest.approx(0.2, rel=0.15)
    yo = y.copy()
    yo[::25] += 4
    robust = LocalRegressionModel(span=0.2, iterations=3).fit(x, yo)
    plain = LocalRegressionModel(span=0.2).fit(x, yo)
    assert np.mean((robust.predict(t) - np.sin(t)) ** 2) < 0.5 * np.mean((plain.predict(t) - np.sin(t)) ** 2)
    x3 = rng.uniform(-1, 1, (2000, 3))
    m3 = LocalRegressionModel(span=0.05, degree=2).fit(x3, np.sin(2 * x3[:, 0]) + x3[:, 1] * x3[:, 2])
    t3 = rng.uniform(-0.8, 0.8, (200, 3))
    assert np.sqrt(np.mean((m3.predict(t3) - np.sin(2 * t3[:, 0]) - t3[:, 1] * t3[:, 2]) ** 2)) < 0.02
    loaded = SurrogateModelBase.fromDict(json.loads(json.dumps(m.toDict())))
    np.testing.assert_allclose(loaded.predictInterval(t).upper, m.predictInterval(t).upper)


def test_handlerLibraryWorkflow():
    rh = RegressionHandler()
    x = np.linspace(1, 5, 30)
    rh.invoke("setData", x=x.tolist(), y=(2.0 * np.exp(-3.0 / x)).tolist(), featureNames=["u"])
    rep = rh.invoke("fitModel", modelName="k", model="inverseExponential")
    assert rep["parameters"]["y0"][1]["estimate"] == pytest.approx(-3.0, rel=1e-6)
    assert len(rh.invoke("listLibraryForms")["forms"]) == len(LIBRARY)


def test_inferenceMatchesClosedForm():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 6, 40)
    y = 5 * np.exp(-x / 1.2) + 1 + rng.normal(0, 0.05, 40)
    m = createModel("exponentialDecay").fit(x, y)
    a, tau, c = m.parameters.values()
    fitted = a * np.exp(-x / tau) + c
    # the optimum satisfies the normal equations J^T r = 0
    jac = np.column_stack([np.exp(-x / tau), a * x / tau ** 2 * np.exp(-x / tau), np.ones_like(x)])
    np.testing.assert_allclose(jac.T @ (y - fitted), 0.0, atol=1e-6)
    s2 = np.sum((y - fitted) ** 2) / (40 - 3)
    np.testing.assert_allclose(m.result.stdErrors[:, 0], np.sqrt(np.diag(s2 * np.linalg.inv(jac.T @ jac))),
                               rtol=1e-4)


def test_loessReproducesPolynomialsExactly():
    x = np.linspace(-2, 3, 80)
    for degree, y in ((1, 2 - 3 * x), (2, 1 + x - 0.5 * x ** 2)):
        for span in (0.1, 0.5):
            m = LocalRegressionModel(span=span, degree=degree).fit(x, y)
            np.testing.assert_allclose(m.predict(x), y, atol=1e-6)
