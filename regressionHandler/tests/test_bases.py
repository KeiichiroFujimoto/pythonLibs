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
