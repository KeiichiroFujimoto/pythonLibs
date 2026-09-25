"""SmoothingSplineModel (FITPACK curfit port)."""
import warnings

import numpy as np
import pytest

from pythonLibs.regressionHandler import SmoothingSplineModel, loadModel


def _data(m=60, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 10.0, m))
    return x, np.sin(x) + rng.normal(0.0, noise, m)


def test_largeSmoothingGivesLeastSquaresPolynomial():
    x = np.linspace(0.0, 1.0, 30)
    y = 1.0 - 2.0 * x + 3.0 * x ** 3
    m = SmoothingSplineModel(smoothing=1.0).fit(x, y)
    assert m.ier == -2
    q = np.linspace(-0.5, 1.5, 21)
    assert m(q) == pytest.approx(1.0 - 2.0 * q + 3.0 * q ** 3, abs=1e-10)


def test_zeroSmoothingInterpolates():
    x, y = _data(20)
    m = SmoothingSplineModel(smoothing=0.0).fit(x, y)
    assert m.ier == -1 and m(x) == pytest.approx(y, abs=1e-10)


def test_residualMeetsSmoothingFactor():
    x, y = _data(80)
    m = SmoothingSplineModel(smoothing=0.5).fit(x, y)
    assert m.ier == 0 and abs(m.residual - 0.5) <= 0.001 * 0.5
    assert m.residual == pytest.approx(float(np.sum((m(x) - y) ** 2)), rel=1e-9)


def test_derivativeMatchesSplineDerivative():
    x, y = _data(50)
    m = SmoothingSplineModel(smoothing=1.0).fit(x, y)
    q = np.linspace(0.5, 9.5, 40)
    h = 1e-6
    assert m.predictDerivatives(q[:, None], 0)[:, 0] == pytest.approx((m(q + h) - m(q - h)) / (2 * h), abs=1e-5)


def test_saveLoadKeepsContinuationState(tmp_path):
    x, y = _data(90)
    m = SmoothingSplineModel().fit(x, y)
    m.save(str(tmp_path / "s.json"))
    back = loadModel(str(tmp_path / "s.json"))
    q = np.linspace(-1.0, 11.0, 50)
    assert back(q) == pytest.approx(m(q), abs=0.0)
    assert back.setSmoothingFactor(0.3)(q) == pytest.approx(m.setSmoothingFactor(0.3)(q), abs=0.0)


def test_matchesScipySplrep():
    interpolate = pytest.importorskip("scipy.interpolate")
    rng = np.random.default_rng(1)
    for trial in range(20):
        x, y = _data(int(rng.integers(10, 120)), noise=float(rng.choice([0.01, 0.2])), seed=trial)
        k = int(rng.choice([1, 3, 5]))
        s = float(rng.choice([0.05, 1.0, 0.1 * x.size]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t, c, _ = interpolate.splrep(x, y, k=k, s=s)
            m = SmoothingSplineModel(degree=k, smoothing=s, nest=max(x.size + k + 1, 2 * k + 3)).fit(x, y)
        assert np.array_equal(m.knots, t)
        assert m.coefficients == pytest.approx(c[:t.size - k - 1], rel=1e-8, abs=1e-8 * np.max(np.abs(c)))


def test_matchesScipyUnivariateSplineIncludingContinuation():
    interpolate = pytest.importorskip("scipy.interpolate")
    grown = 0
    for trial in range(15):
        x, y = _data(int(np.random.default_rng(trial).integers(20, 150)), noise=0.1, seed=100 + trial)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = interpolate.UnivariateSpline(x, y)
            m = SmoothingSplineModel().fit(x, y)
            assert np.array_equal(ref.get_knots(), m.knots[3:-3])
            ref.set_smoothing_factor(0.2)
            m.setSmoothingFactor(0.2)
        grown += m._state.nest == x.size + 4
        assert np.array_equal(ref.get_knots(), m.knots[3:-3])
        q = np.linspace(x[0], x[-1], 200)
        assert m(q) == pytest.approx(ref(q), rel=1e-7, abs=1e-7)
    assert grown > 0      # the enlarged-nest continuation was exercised
