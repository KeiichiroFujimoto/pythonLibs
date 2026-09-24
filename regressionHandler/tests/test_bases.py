"""Basis components: structure, derivatives, serialization."""
import numpy as np
import pytest

from pythonLibs.regressionHandler.bases import (BSplineBasis, CombinedBasis, ExpressionBasis, OrthogonalPolynomialBasis,
                                                PolynomialBasis, RadialBasis, polynomialExponents)
from pythonLibs.regressionHandler.core.Registry import buildComponent


def test_polynomialTermCounts():
    assert polynomialExponents(3, 2).shape == (10, 3)
    assert polynomialExponents(3, 2, interactionOrder=1).shape == (7, 3)
    assert polynomialExponents(2, 2, mode="tensor").shape == (9, 2)
    assert polynomialExponents(2, 3, includeBias=False).shape == (9, 2)


def test_termNames():
    b = PolynomialBasis(degree=2).fit(np.zeros((3, 2)))
    assert b.termNames(["T", "P"]) == ["1", "T", "P", "T^2", "T*P", "P^2"]


def test_bsplinePartitionOfUnity():
    x = np.linspace(-2, 5, 101)[:, None]
    b = BSplineBasis(nSegments=9, degree=3).fit(x)
    phi = b.transform(x)
    assert phi.shape == (101, 12)
    np.testing.assert_allclose(phi.sum(axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(b.derivative(x, 0).sum(axis=1), 0.0, atol=1e-12)


@pytest.mark.parametrize("basis", [
    PolynomialBasis(degree=3), PolynomialBasis(degree=3, scaling="minmax"),
    OrthogonalPolynomialBasis(degree=4), OrthogonalPolynomialBasis(degree=4, family="chebyshev"),
    RadialBasis(kernel="gaussian"), RadialBasis(kernel="multiquadric"), RadialBasis(kernel="cubic"),
    RadialBasis(kernel="thinPlateSpline"), BSplineBasis(nSegments=4),
    CombinedBasis(bases=[{"type": "polynomial", "degree": 1}, {"type": "radial", "polyDegree": -1}]),
])
def test_analyticDerivativesMatchFiniteDifferences(basis):
    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 2, (40, 3))
    basis.fit(x)
    inner = rng.uniform(-0.8, 1.8, (25, 3))
    for k in range(3):
        xp, xm = inner.copy(), inner.copy()
        xp[:, k] += 1e-6
        xm[:, k] -= 1e-6
        fd = (basis.transform(xp) - basis.transform(xm)) / 2e-6
        np.testing.assert_allclose(basis.derivative(inner, k), fd, atol=1e-6 * max(1.0, np.abs(fd).max()))


def test_basisRoundTrip():
    x = np.random.default_rng(0).uniform(size=(30, 2))
    for b in (PolynomialBasis(degree=3, scaling="minmax"), RadialBasis(), BSplineBasis(nSegments=5),
              ExpressionBasis(terms=["a*b", "exp(-a)"], variables=["a", "b"])):
        b.fit(x)
        clone = buildComponent("basis", b.toDict())
        np.testing.assert_allclose(clone.transform(x), b.transform(x))


def test_expressionBasisRejectsCode():
    with pytest.raises(ValueError):
        ExpressionBasis(terms=["__import__('os')"]).fit(np.zeros((2, 1)))
    with pytest.raises(ValueError):
        ExpressionBasis(terms=["x0.real"]).fit(np.zeros((2, 1)))


def test_bsplineCoxDeBoorReference():
    """Compare with the textbook Cox-de Boor recursion on the same uniform knots."""
    x = np.sort(np.random.default_rng(0).uniform(0, 3, 50))
    nSeg, deg = 7, 3
    b = BSplineBasis(nSegments=nSeg, degree=deg).fit(x[:, None])
    lo, hi = x.min(), x.max()
    t = lo + np.arange(-deg, nSeg + deg + 1) * (hi - lo) / nSeg

    def coxDeBoor(i, k, v):
        if k == 0:
            # half-open intervals; the right end of the data range belongs to the last one
            inside = (t[i] <= v) & (v < t[i + 1]) & (v < hi)
            return np.where(inside | ((v == hi) & np.isclose(t[i + 1], hi)), 1.0, 0.0)
        left = (v - t[i]) / (t[i + k] - t[i]) * coxDeBoor(i, k - 1, v)
        right = (t[i + k + 1] - v) / (t[i + k + 1] - t[i + 1]) * coxDeBoor(i + 1, k - 1, v)
        return left + right

    ref = np.column_stack([coxDeBoor(i, deg, x) for i in range(nSeg + deg)])
    np.testing.assert_allclose(b.transform(x[:, None]), ref, atol=1e-13)
