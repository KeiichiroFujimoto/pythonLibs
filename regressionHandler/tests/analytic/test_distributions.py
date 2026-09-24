"""Special functions and distributions against closed-form identities.

Every expected value below is an exact mathematical result: special cases
of the incomplete gamma / beta functions and distributions whose CDF and
quantile have elementary closed forms.
"""
import math

import numpy as np
import pytest

from pythonLibs.regressionHandler.numerics import SpecialFunctions as sf
from pythonLibs.regressionHandler.numerics.Distributions import ChiSquared, FDistribution, Normal, StudentT

X_POS = np.array([1e-6, 0.01, 0.3, 1.0, 2.5, 7.0, 20.0, 60.0])
P = np.array([1e-10, 1e-4, 0.025, 0.2, 0.5, 0.8, 0.975, 1 - 1e-6])


def test_gammalnFactorialsAndHalfIntegers():
    for n in range(1, 30):
        assert sf.gammaln(float(n)) == pytest.approx(math.log(math.factorial(n - 1)), rel=1e-13, abs=1e-13)
    # Gamma(1/2) = sqrt(pi), Gamma(n + 1/2) = (2n)! sqrt(pi) / (4^n n!)
    for n in range(0, 15):
        exact = math.log(math.factorial(2 * n) * math.sqrt(math.pi) / (4 ** n * math.factorial(n)))
        assert sf.gammaln(n + 0.5) == pytest.approx(exact, rel=1e-13, abs=1e-13)
    x = np.linspace(0.05, 40, 50)
    np.testing.assert_allclose(sf.gammaln(x + 1) - sf.gammaln(x), np.log(x), atol=1e-12)   # Gamma(x+1) = x Gamma(x)


def test_incompleteGammaClosedForms():
    np.testing.assert_allclose(sf.gammainc(1.0, X_POS), -np.expm1(-X_POS), rtol=1e-13)          # P(1, x) = 1 - e^-x
    np.testing.assert_allclose(sf.gammaincc(1.0, X_POS), np.exp(-X_POS), rtol=1e-13)
    for n in (2, 3, 5, 8):                                                                          # integer shape
        series = sum(X_POS ** k / math.factorial(k) for k in range(n))
        np.testing.assert_allclose(sf.gammaincc(float(n), X_POS), np.exp(-X_POS) * series, rtol=1e-12)
    erf = np.array([math.erf(math.sqrt(v)) for v in X_POS])                                         # P(1/2, x) = erf(sqrt x)
    np.testing.assert_allclose(sf.gammainc(0.5, X_POS), erf, rtol=1e-13)


def test_incompleteBetaClosedForms():
    x = np.linspace(0.001, 0.999, 60)
    for a in (0.3, 1.0, 2.5, 17.0):
        np.testing.assert_allclose(sf.betainc(a, 1.0, x), x ** a, rtol=1e-12)                      # I_x(a, 1) = x^a
        np.testing.assert_allclose(sf.betainc(1.0, a, x), 1 - (1 - x) ** a, rtol=1e-12)             # I_x(1, b)
    np.testing.assert_allclose(sf.betainc(0.5, 0.5, x), 2 / np.pi * np.arcsin(np.sqrt(x)), rtol=1e-12)
    a, b = 3.7, 0.8
    np.testing.assert_allclose(sf.betainc(a, b, x), 1 - sf.betainc(b, a, 1 - x), atol=1e-14)        # symmetry
    # integer a, b: I_x(a, b) = sum_{j=a}^{a+b-1} C(a+b-1, j) x^j (1-x)^(a+b-1-j)
    m = 7
    exact = sum(math.comb(m - 1, j) * x ** j * (1 - x) ** (m - 1 - j) for j in range(3, m))
    np.testing.assert_allclose(sf.betainc(3.0, 4.0, x), exact, rtol=1e-12)


def test_errorFunctionAndNormal():
    z = np.linspace(-6, 6, 97)
    np.testing.assert_allclose(sf.erf(z), [math.erf(v) for v in z], atol=1e-15)
    np.testing.assert_allclose(sf.erfc(z), [math.erfc(v) for v in z], rtol=1e-12)
    np.testing.assert_allclose(Normal().cdf(z), [0.5 * math.erfc(-v / math.sqrt(2)) for v in z], rtol=1e-12)
    np.testing.assert_allclose(Normal().cdf(Normal().ppf(P)), P, rtol=1e-12)                          # exact inverse
    np.testing.assert_allclose(Normal(2.0, 3.0).ppf(P), 2.0 + 3.0 * Normal().ppf(P), rtol=1e-14)       # location-scale


def test_studentTSpecialCases():
    t = np.linspace(-50, 50, 101)
    np.testing.assert_allclose(StudentT(1).cdf(t), 0.5 + np.arctan(t) / np.pi, atol=1e-14)            # Cauchy
    np.testing.assert_allclose(StudentT(1).ppf(P), -1.0 / np.tan(np.pi * P), rtol=1e-10, atol=1e-15)  # -cot(pi p)
    np.testing.assert_allclose(StudentT(2).cdf(t), 0.5 + t / (2 * np.sqrt(2 + t * t)), atol=1e-14)
    np.testing.assert_allclose(StudentT(2).ppf(P), (2 * P - 1) / np.sqrt(2 * P * (1 - P)), rtol=1e-10)
    # large df: Cornish-Fisher expansion t_p = z + (z^3 + z) / (4 nu) + O(nu^-2)
    nu, z = 1e6, Normal().ppf(P)
    np.testing.assert_allclose(StudentT(nu).ppf(P), z + (z ** 3 + z) / (4 * nu), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(StudentT(4.3).cdf(-t), StudentT(4.3).sf(t), atol=1e-15)                 # symmetry


def test_chiSquaredAndFSpecialCases():
    x = np.linspace(0.01, 40, 80)
    np.testing.assert_allclose(ChiSquared(2).cdf(x), -np.expm1(-x / 2), rtol=1e-13)                   # exponential
    np.testing.assert_allclose(ChiSquared(2).ppf(P), -2 * np.log1p(-P), rtol=1e-12)
    np.testing.assert_allclose(ChiSquared(1).cdf(x), [math.erf(math.sqrt(v / 2)) for v in x], rtol=1e-12)
    np.testing.assert_allclose(FDistribution(2, 2).cdf(x), x / (1 + x), rtol=1e-13)                   # F(2, 2)
    np.testing.assert_allclose(FDistribution(2, 2).ppf(P), P / (1 - P), rtol=1e-10)
    t = np.linspace(0.1, 6, 30)                                                                        # F(1, v) = t_v^2
    np.testing.assert_allclose(FDistribution(1, 7).cdf(t * t), 2 * StudentT(7).cdf(t) - 1, atol=1e-14)
    np.testing.assert_allclose(FDistribution(3, 5).cdf(x), 1 - FDistribution(5, 3).cdf(1 / x), atol=1e-14)


def test_densitiesIntegrateToCdf():
    for dist, lo in ((StudentT(3.5), -40.0), (ChiSquared(4.0), 0.0), (FDistribution(4, 9), 0.0)):
        grid = np.linspace(lo, 6.0, 20001)
        integral = np.concatenate([[0.0], np.cumsum(0.5 * (dist.pdf(grid[1:]) + dist.pdf(grid[:-1])) * np.diff(grid))])
        np.testing.assert_allclose(integral[-1], dist.cdf(6.0) - dist.cdf(lo), atol=2e-6)


def test_quantileRoundTripEverywhere():
    for dist in (StudentT(0.8), StudentT(7.5), StudentT(400), ChiSquared(0.6), ChiSquared(35), FDistribution(2, 9)):
        np.testing.assert_allclose(dist.cdf(dist.ppf(P[1:-1])), P[1:-1], rtol=1e-9)
