"""Continuous distributions used for inference.

Every distribution is an object with ``pdf``, ``cdf``, ``sf`` (survival function), ``ppf``
(quantile) and ``isf`` (inverse survival). Tail probabilities are computed
directly from the incomplete gamma / beta functions rather than as ``1 - cdf``,
so p-values of 1e-20 keep full relative precision.

Quantiles are obtained by a vectorized safeguarded Newton iteration: every
step is checked against a shrinking bracket and replaced by bisection when it
leaves the bracket, which guarantees convergence for any monotone CDF.
"""
from __future__ import annotations

import numpy as np

from pythonLibs.regressionHandler.numerics import SpecialFunctions as sf


def _solveMonotone(func, deriv, target, lo, hi, increasing: bool, x0=None,
                   maxIter: int = 600, rtol: float = 1e-14) -> np.ndarray:
    """Solve func(x) = target elementwise on brackets [lo, hi].

    Only unconverged elements are re-evaluated each sweep, so a few hard
    elements do not make the whole array pay for extra iterations.
    """
    target = np.atleast_1d(np.array(target, dtype=float))
    lo = np.atleast_1d(np.array(lo, dtype=float))
    hi = np.atleast_1d(np.array(hi, dtype=float))
    x = 0.5 * (lo + hi) if x0 is None else np.clip(np.atleast_1d(np.array(x0, dtype=float)), lo, hi)
    if x.size == 1:
        return np.array([_solveMonotoneScalar(func, deriv, float(target[0]), float(lo[0]), float(hi[0]),
                                              increasing, float(x[0]), maxIter, rtol)])
    idx = np.arange(x.size)
    for _ in range(maxIter):
        xa, la, ha = x[idx], lo[idx], hi[idx]
        f = func(xa) - target[idx]
        belowRoot = (f < 0.0) if increasing else (f > 0.0)
        la = np.where(belowRoot, xa, la)
        ha = np.where(belowRoot, ha, xa)
        with np.errstate(divide="ignore", invalid="ignore"):
            xNew = xa - f / deriv(xa)
        converged = np.isfinite(xNew) & (np.abs(xNew - xa) <= rtol * np.abs(xa) + 1e-300)
        bad = ~converged & (~np.isfinite(xNew) | (xNew <= la) | (xNew >= ha))
        # Geometric bisection when the bracket is positive so quantiles that
        # are many decades below 1 (chi2 / F with small df) converge quickly.
        with np.errstate(invalid="ignore", divide="ignore"):
            geo = np.where((la > 0.0) & (ha > 4.0 * la), np.sqrt(la * ha), 0.5 * (la + ha))
        xNew = np.where(bad, geo, xNew)
        done = converged | (ha - la <= rtol * np.abs(xa))
        x[idx], lo[idx], hi[idx] = xNew, la, ha
        idx = idx[~done]
        if idx.size == 0:
            break
    return x


def _solveMonotoneScalar(func, deriv, target, lo, hi, increasing, x, maxIter, rtol) -> float:
    """Scalar version of the safeguarded Newton iteration (plain floats)."""
    for _ in range(maxIter):
        f = float(func(x)) - target
        if (f < 0.0) == increasing:
            lo = x
        else:
            hi = x
        d = float(deriv(x))
        xNew = x - f / d if d != 0.0 else float("nan")
        if np.isfinite(xNew) and abs(xNew - x) <= rtol * abs(x) + 1e-300:
            return xNew
        if not np.isfinite(xNew) or xNew <= lo or xNew >= hi:
            xNew = np.sqrt(lo * hi) if (lo > 0.0 and hi > 4.0 * lo) else 0.5 * (lo + hi)
        if hi - lo <= rtol * abs(x):
            return xNew
        x = xNew
    return x


def _expandUpper(func, target, start, increasing: bool) -> np.ndarray:
    """Grow an upper bracket until func(hi) passes target."""
    hi = np.array(start, dtype=float)
    for _ in range(2000):
        f = func(hi)
        need = (f < target) if increasing else (f > target)
        if not np.any(need):
            break
        hi = np.where(need, hi * 2.0, hi)
    return hi


def _asArray(v):
    arr = np.asarray(v, dtype=float)
    return arr, arr.ndim == 0


def _ret(arr, scalar):
    return float(arr) if scalar else arr


class ContinuousDistribution:
    """Base with the public pdf / cdf / sf / ppf / isf methods."""

    def pdf(self, x):
        raise NotImplementedError

    def cdf(self, x):
        raise NotImplementedError

    def sf(self, x):
        raise NotImplementedError

    def ppf(self, q):
        raise NotImplementedError

    def isf(self, q):
        q, scalar = _asArray(q)
        return _ret(np.asarray(self.ppf(1.0 - q)), scalar)

    def interval(self, confidence: float):
        """Equal-tail interval containing ``confidence`` probability mass."""
        a = 0.5 * (1.0 - confidence)
        return self.ppf(a), self.isf(a)


class Normal(ContinuousDistribution):

    def __init__(self, loc: float = 0.0, scale: float = 1.0):
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.loc = float(loc)
        self.scale = float(scale)

    def _z(self, x):
        return (np.asarray(x, dtype=float) - self.loc) / self.scale

    def pdf(self, x):
        z = self._z(x)
        return np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * self.scale)

    def cdf(self, x):
        return sf.ndtr(self._z(x))

    def sf(self, x):
        return sf.ndtr(-self._z(x))

    def ppf(self, q):
        return self.loc + self.scale * np.asarray(sf.ndtri(q)) if np.ndim(q) else \
            self.loc + self.scale * sf.ndtri(q)

    def isf(self, q):
        return self.loc - self.scale * np.asarray(sf.ndtri(q)) if np.ndim(q) else \
            self.loc - self.scale * sf.ndtri(q)


class StudentT(ContinuousDistribution):

    def __init__(self, df: float):
        if not df > 0:
            raise ValueError("df must be positive")
        self.df = float(df)
        self._logNorm = float(sf.gammaln(0.5 * (self.df + 1.0)) - sf.gammaln(0.5 * self.df)
                              - 0.5 * np.log(self.df * np.pi))

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.exp(self._logNorm - 0.5 * (self.df + 1.0) * np.log1p(x * x / self.df))

    def _upperTail(self, t):
        """P(T > t) for t >= 0."""
        return 0.5 * np.asarray(sf.betainc(0.5 * self.df, 0.5, self.df / (self.df + t * t)))

    def sf(self, x):
        x, scalar = _asArray(x)
        tail = self._upperTail(np.abs(x))
        return _ret(np.where(x >= 0.0, tail, 1.0 - tail), scalar)

    def cdf(self, x):
        x, scalar = _asArray(x)
        tail = self._upperTail(np.abs(x))
        return _ret(np.where(x >= 0.0, 1.0 - tail, tail), scalar)

    def ppf(self, q):
        q, scalar = _asArray(q)
        q = np.atleast_1d(q)
        out = np.full(q.shape, np.nan)
        out[q == 0.0] = -np.inf
        out[q == 1.0] = np.inf
        ok = (q > 0.0) & (q < 1.0)
        if np.any(ok):
            qq = q[ok]
            tail = np.minimum(qq, 1.0 - qq)
            z = np.abs(np.atleast_1d(sf.ndtri(tail)))
            # Cornish-Fisher expansion of the t quantile around the normal one.
            n = self.df
            x0 = z + (z ** 3 + z) / (4.0 * n) + (5.0 * z ** 5 + 16.0 * z ** 3 + 3.0 * z) / (96.0 * n * n)
            hi = _expandUpper(self._upperTail, tail, np.maximum(2.0 * x0, 1.0), increasing=False)
            t = _solveMonotone(self._upperTail, lambda v: -self.pdf(v), tail,
                               np.zeros_like(tail), hi, increasing=False, x0=x0)
            out[ok] = np.where(qq < 0.5, -t, t)
        return _ret(out[0] if scalar else out, scalar)

    def isf(self, q):
        q, scalar = _asArray(q)
        return _ret(-np.asarray(self.ppf(q)), scalar)


class ChiSquared(ContinuousDistribution):

    def __init__(self, df: float):
        if not df > 0:
            raise ValueError("df must be positive")
        self.df = float(df)
        self._k = 0.5 * self.df
        self._logNorm = float(-self._k * np.log(2.0) - sf.gammaln(self._k))

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.exp(self._logNorm + (self._k - 1.0) * np.log(x) - 0.5 * x)
        return np.where(x > 0.0, out, 0.0)

    def cdf(self, x):
        x = np.maximum(np.asarray(x, dtype=float), 0.0)
        return sf.gammainc(self._k, 0.5 * x)

    def sf(self, x):
        x = np.maximum(np.asarray(x, dtype=float), 0.0)
        return sf.gammaincc(self._k, 0.5 * x)

    def ppf(self, q):
        return _invertPositive(self, q)


class FDistribution(ContinuousDistribution):

    def __init__(self, dfn: float, dfd: float):
        if not (dfn > 0 and dfd > 0):
            raise ValueError("degrees of freedom must be positive")
        self.dfn = float(dfn)
        self.dfd = float(dfd)
        self._logNorm = float(0.5 * self.dfn * np.log(self.dfn / self.dfd)
                              - sf.betaln(0.5 * self.dfn, 0.5 * self.dfd))

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.exp(self._logNorm + (0.5 * self.dfn - 1.0) * np.log(x)
                         - 0.5 * (self.dfn + self.dfd) * np.log1p(self.dfn * x / self.dfd))
        return np.where(x > 0.0, out, 0.0)

    def cdf(self, x):
        x = np.maximum(np.asarray(x, dtype=float), 0.0)
        return sf.betainc(0.5 * self.dfn, 0.5 * self.dfd, self.dfn * x / (self.dfn * x + self.dfd))

    def sf(self, x):
        x = np.maximum(np.asarray(x, dtype=float), 0.0)
        return sf.betainc(0.5 * self.dfd, 0.5 * self.dfn, self.dfd / (self.dfn * x + self.dfd))

    def ppf(self, q):
        return _invertPositive(self, q)


def _invertPositive(dist: ContinuousDistribution, q):
    """Quantile of a distribution supported on [0, inf)."""
    q, scalar = _asArray(q)
    q = np.atleast_1d(q)
    out = np.full(q.shape, np.nan)
    out[q == 0.0] = 0.0
    out[q == 1.0] = np.inf
    ok = (q > 0.0) & (q < 1.0)
    lower = ok & (q <= 0.5)
    upper = ok & (q > 0.5)
    if np.any(lower):
        target = q[lower]
        hi = _expandUpper(dist.cdf, target, np.ones_like(target), increasing=True)
        out[lower] = _solveMonotone(dist.cdf, dist.pdf, target, np.zeros_like(target), hi, True)
    if np.any(upper):
        target = 1.0 - q[upper]
        hi = _expandUpper(dist.sf, target, np.ones_like(target), increasing=False)
        out[upper] = _solveMonotone(dist.sf, lambda v: -dist.pdf(v), target,
                                    np.zeros_like(target), hi, False)
    return _ret(out[0] if scalar else out, scalar)
