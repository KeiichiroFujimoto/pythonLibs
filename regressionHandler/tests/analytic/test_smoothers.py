"""B-splines and LOESS against their defining recursions and reproduction properties."""
import numpy as np

from pythonLibs.regressionHandler import LocalRegressionModel
from pythonLibs.regressionHandler.bases import BSplineBasis


def _coxDeBoor(t, i, k, v, hi):
    if k == 0:
        inside = (t[i] <= v) & (v < t[i + 1]) & (v < hi)
        return np.where(inside | ((v == hi) & np.isclose(t[i + 1], hi)), 1.0, 0.0)
    left = (v - t[i]) / (t[i + k] - t[i]) * _coxDeBoor(t, i, k - 1, v, hi)
    right = (t[i + k + 1] - v) / (t[i + k + 1] - t[i + 1]) * _coxDeBoor(t, i + 1, k - 1, v, hi)
    return left + right


def test_bsplineMatchesCoxDeBoorRecursion():
    x = np.sort(np.random.default_rng(0).uniform(0, 3, 50))
    for nSeg, deg in ((7, 3), (4, 2), (10, 1)):
        b = BSplineBasis(nSegments=nSeg, degree=deg).fit(x[:, None])
        lo, hi = x.min(), x.max()
        t = lo + np.arange(-deg, nSeg + deg + 1) * (hi - lo) / nSeg
        ref = np.column_stack([_coxDeBoor(t, i, deg, x, hi) for i in range(nSeg + deg)])
        np.testing.assert_allclose(b.transform(x[:, None]), ref, atol=1e-13)


def test_bsplineDerivativeIdentity():
    # d/dx B_{i,p} = p/(t_{i+p} - t_i) B_{i,p-1} - p/(t_{i+p+1} - t_{i+1}) B_{i+1,p-1}
    x = np.linspace(0.05, 2.95, 40)
    nSeg, deg = 6, 3
    b = BSplineBasis(nSegments=nSeg, degree=deg).fit(np.r_[0.0, x, 3.0][:, None])
    t = np.arange(-deg, nSeg + deg + 1) * 3.0 / nSeg
    ref = np.column_stack([
        deg / (t[i + deg] - t[i]) * _coxDeBoor(t, i, deg - 1, x, 3.0)
        - deg / (t[i + deg + 1] - t[i + 1]) * _coxDeBoor(t, i + 1, deg - 1, x, 3.0)
        for i in range(nSeg + deg)])
    np.testing.assert_allclose(b.derivative(x[:, None], 0), ref, atol=1e-12)
    np.testing.assert_allclose(b.transform(x[:, None]).sum(axis=1), 1.0, atol=1e-14)   # partition of unity


def test_loessReproducesPolynomialsOfItsDegree():
    x = np.linspace(-2, 3, 80)
    for degree, y in ((0, np.full_like(x, 4.2)), (1, 2 - 3 * x), (2, 1 + x - 0.5 * x ** 2)):
        for span in (0.1, 0.5):
            np.testing.assert_allclose(LocalRegressionModel(span=span, degree=degree).fit(x, y).predict(x), y,
                                       atol=1e-6)
    x2 = np.random.default_rng(0).uniform(-1, 1, (300, 2))
    plane = 0.5 + x2 @ np.array([1.5, -2.0])
    np.testing.assert_allclose(LocalRegressionModel(span=0.1).fit(x2, plane).predict(x2), plane, atol=1e-6)
