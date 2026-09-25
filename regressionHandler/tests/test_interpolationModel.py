import numpy as np
import pytest

from pythonLibs.regressionHandler import InterpolationModel, createModel, loadModel


def _cubic(x):
    return 0.5 * x ** 3 - 2.0 * x ** 2 + x - 3.0


def test_linearReproducesLinesAndExtrapolates():
    x = np.array([0.0, 1.0, 2.5, 4.0])
    m = InterpolationModel(kind="linear").fit(x, 2.0 * x + 1.0)
    q = np.array([-2.0, 0.3, 3.9, 6.0])
    assert m(q) == pytest.approx(2.0 * q + 1.0, abs=1e-14)
    assert m.predictDerivatives(q[:, None], 0)[:, 0] == pytest.approx(np.full(4, 2.0))


def test_notAKnotCubicReproducesAnyCubicExactly():
    x = np.array([-1.0, -0.2, 0.7, 1.1, 2.9, 3.0, 4.5])
    m = InterpolationModel(kind="cubic").fit(x, _cubic(x))
    q = np.linspace(-3.0, 6.0, 101)
    assert m(q) == pytest.approx(_cubic(q), rel=1e-12, abs=1e-11)
    dq = 1.5 * q ** 2 - 4.0 * q + 1.0
    assert m.predictDerivatives(q[:, None], 0)[:, 0] == pytest.approx(dq, rel=1e-11, abs=1e-10)


@pytest.mark.parametrize("n,degree", [(2, 1), (3, 2)])
def test_fewPointsGiveLineOrParabola(n, degree):
    x = np.linspace(0.0, 2.0, n)
    coef = [0.7, -1.2, 2.0][:degree + 1]
    m = InterpolationModel(kind="cubic").fit(x, np.polyval(coef, x))
    q = np.linspace(-1.0, 3.0, 9)
    assert m(q) == pytest.approx(np.polyval(coef, q), abs=1e-12)


def test_extrapolationModes():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 3.0, 2.0])
    q = np.array([-1.0, 0.5, 3.0])
    assert InterpolationModel(extrapolation="clamp").fit(x, y)(q) == pytest.approx([1.0, 2.0, 2.0])
    filled = InterpolationModel(extrapolation="fill").fit(x, y)(q)
    assert np.isnan(filled[0]) and filled[1] == 2.0 and np.isnan(filled[2])
    assert InterpolationModel(extrapolation="fill", fillValue=-9.0).fit(x, y)(q) == pytest.approx([-9.0, 2.0, -9.0])


def test_unsortedInputMultiOutputAndCallShape():
    x = np.array([3.0, 0.0, 2.0, 1.0])
    y = np.column_stack([x ** 2, -x])
    m = createModel({"type": "interpolation", "kind": "cubic"}).fit(x, y)
    assert m.predict([[1.5]])[0] == pytest.approx([2.25, -1.5])
    single = InterpolationModel().fit(x, x)
    assert single(0.5).shape == () and single(np.ones((2, 3))).shape == (2, 3)


def test_cubicRejectsDuplicateX():
    with pytest.raises(ValueError):
        InterpolationModel(kind="cubic").fit([0.0, 1.0, 1.0, 2.0], [0.0, 1.0, 2.0, 3.0])


def test_saveLoadRoundTrip(tmp_path):
    x = np.linspace(0.0, 5.0, 12)
    m = InterpolationModel(kind="cubic", extrapolation="fill").fit(x, np.cos(x))
    path = str(tmp_path / "interp.json")
    m.save(path)
    back = loadModel(path)
    q = np.linspace(-1.0, 6.0, 30)
    assert np.array_equal(np.isnan(back(q)), np.isnan(m(q)))
    assert back(q)[np.isfinite(m(q))] == pytest.approx(m(q)[np.isfinite(m(q))], abs=0.0)


def test_matchesScipyReference():
    interpolate = pytest.importorskip("scipy.interpolate")
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(-3.0, 7.0, 40))
    y = np.sin(x) + rng.normal(0.0, 0.3, x.size)
    q = rng.uniform(-5.0, 9.0, 400)
    for kind in ("linear", "cubic"):
        ref = interpolate.interp1d(x, y, kind=kind, bounds_error=False, fill_value="extrapolate")(q)
        got = InterpolationModel(kind=kind).fit(x, y)(q)
        assert np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1.0)) < 1e-11
    ref = interpolate.splev(q, interpolate.splrep(x, y, s=0))
    assert np.max(np.abs(InterpolationModel(kind="cubic").fit(x, y)(q) - ref) / np.maximum(np.abs(ref), 1.0)) < 1e-11


@pytest.mark.parametrize("kind", ["linear", "cubic"])
@pytest.mark.parametrize("extrapolation", ["extend", "clamp", "fill"])
def test_singlePointCallMatchesArrayPathBitwise(kind, extrapolation):
    rng = np.random.default_rng(5)
    x = np.sort(rng.uniform(0.0, 60.0, 40))
    m = InterpolationModel(kind=kind, extrapolation=extrapolation, fillValue=-2.0).fit(x, np.sin(x / 5.0))
    q = np.concatenate([rng.uniform(-20.0, 80.0, 500), x])
    assert np.array_equal(np.array([float(m(v)) for v in q]), m(q))
