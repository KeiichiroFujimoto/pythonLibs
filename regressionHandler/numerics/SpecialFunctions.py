"""Vectorized special functions.

Implemented with numpy only and accurate to roughly 1e-14 relative error in
the ranges used by regression statistics:

- ``gammaln``          Lanczos approximation (g=7, n=9)
- ``gammainc``/``gammaincc``  regularized incomplete gamma P(a, x), Q(a, x)
  (power series for x < a+1, modified Lentz continued fraction otherwise)
- ``betainc``          regularized incomplete beta I_x(a, b)
  (modified Lentz continued fraction with the usual symmetry switch)
- ``erf``/``erfc``     via Q(1/2, x^2), so the tails keep full relative precision
- ``ndtri``            inverse standard-normal CDF (Acklam's rational
  approximation followed by two Halley refinement step)

References:
    Press, Teukolsky, Vetterling & Flannery, *Numerical Recipes* 3rd ed.,
    sections 6.1-6.4.
    Lentz, W.J. (1976) Applied Optics 15(3), 668-671.
    Acklam, P.J. (2003) "An algorithm for computing the inverse normal
    cumulative distribution function".
"""
from __future__ import annotations

import math

import numpy as np

_EPS = np.finfo(float).eps
_TINY = 1e-300
_MAX_ITER = 500

_LANCZOS_G = 7.0
_LANCZOS_COEF = np.array([
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
])
_HALF_LOG_2PI = 0.5 * np.log(2.0 * np.pi)


def gammaln(x):
    """log|Gamma(x)| for x > 0 (reflection formula used for x < 0.5)."""
    if _isScalar(x):
        return _gammalnScalar(float(x))
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = x < 0.5
    if np.any(small):
        xs = x[small]
        # Gamma(x) Gamma(1-x) = pi / sin(pi x)
        out[small] = np.log(np.pi / np.abs(np.sin(np.pi * xs))) - _gammalnLanczos(1.0 - xs)
    if np.any(~small):
        out[~small] = _gammalnLanczos(x[~small])
    return out if out.ndim else float(out)


def _gammalnLanczos(x):
    z = x - 1.0
    series = np.full_like(z, _LANCZOS_COEF[0])
    for i in range(1, _LANCZOS_COEF.size):
        series = series + _LANCZOS_COEF[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return _HALF_LOG_2PI + (z + 0.5) * np.log(t) - t + np.log(series)


def betaln(a, b):
    return gammaln(a) + gammaln(b) - gammaln(np.asarray(a) + np.asarray(b))


# ---------------------------------------------------------------- incomplete gamma
def gammainc(a, x):
    """Regularized lower incomplete gamma P(a, x), a > 0, x >= 0."""
    p, _ = _gammaincPair(a, x)
    return p


def gammaincc(a, x):
    """Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x)."""
    _, q = _gammaincPair(a, x)
    return q


def _gammaincPair(a, x):
    if _isScalar(a, x):
        return _gammaincPairScalar(float(a), float(x))
    if np.size(a) == 1 and np.size(x) == 1:
        shape = np.broadcast(np.asarray(a), np.asarray(x)).shape
        p, q = _gammaincPairScalar(float(np.ravel(a)[0]), float(np.ravel(x)[0]))
        return np.full(shape, p), np.full(shape, q)
    a, x = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(x, dtype=float))
    scalar = a.ndim == 0
    shape = a.shape
    a = np.atleast_1d(a).astype(float).ravel()
    x = np.atleast_1d(x).astype(float).ravel()
    p = np.zeros_like(x)
    q = np.ones_like(x)
    pos = x > 0.0
    useSeries = pos & (x < a + 1.0)
    useCf = pos & ~useSeries
    if np.any(useSeries):
        ps = _gammaincSeries(a[useSeries], x[useSeries])
        p[useSeries] = ps
        q[useSeries] = 1.0 - ps
    if np.any(useCf):
        qc = _gammainccContinuedFraction(a[useCf], x[useCf])
        q[useCf] = qc
        p[useCf] = 1.0 - qc
    if scalar:
        return float(p[0]), float(q[0])
    return p.reshape(shape), q.reshape(shape)


def _gammaincSeries(a, x):
    ap = a.copy()
    term = 1.0 / a
    total = term.copy()
    active = np.ones(a.shape, dtype=bool)
    for _ in range(_MAX_ITER):
        ap[active] += 1.0
        term[active] *= x[active] / ap[active]
        total[active] += term[active]
        active &= np.abs(term) > np.abs(total) * _EPS
        if not np.any(active):
            break
    return total * np.exp(-x + a * np.log(x) - gammaln(a))


def _gammainccContinuedFraction(a, x):
    b = x + 1.0 - a
    c = np.full_like(x, 1.0 / _TINY)
    d = 1.0 / b
    h = d.copy()
    active = np.ones(a.shape, dtype=bool)
    for i in range(1, _MAX_ITER):
        an = -i * (i - a)
        b = b + 2.0
        d = an * d + b
        d = np.where(np.abs(d) < _TINY, _TINY, d)
        c = b + an / c
        c = np.where(np.abs(c) < _TINY, _TINY, c)
        d = 1.0 / d
        delta = d * c
        h = np.where(active, h * delta, h)
        active &= np.abs(delta - 1.0) > _EPS
        if not np.any(active):
            break
    return np.exp(-x + a * np.log(x) - gammaln(a)) * h


# ---------------------------------------------------------------- incomplete beta
def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b), a, b > 0, 0 <= x <= 1."""
    if _isScalar(a, b, x):
        return _betaincScalar(float(a), float(b), float(x))
    if np.size(a) == 1 and np.size(b) == 1 and np.size(x) == 1:
        shape = np.broadcast(np.asarray(a), np.asarray(b), np.asarray(x)).shape
        return np.full(shape, _betaincScalar(float(np.ravel(a)[0]), float(np.ravel(b)[0]),
                                             float(np.ravel(x)[0])))
    a, b, x = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                                  np.asarray(x, dtype=float))
    shape = a.shape
    a = a.ravel().astype(float)
    b = b.ravel().astype(float)
    x = np.clip(x.ravel().astype(float), 0.0, 1.0)
    out = np.where(x >= 1.0, 1.0, 0.0)
    inner = (x > 0.0) & (x < 1.0)
    if np.any(inner):
        ai, bi, xi = a[inner], b[inner], x[inner]
        with np.errstate(divide="ignore"):
            logFront = ai * np.log(xi) + bi * np.log1p(-xi) - betaln(ai, bi)
        swap = xi > (ai + 1.0) / (ai + bi + 2.0)
        res = np.empty_like(xi)
        if np.any(~swap):
            k = ~swap
            res[k] = np.exp(logFront[k]) * _betaContinuedFraction(ai[k], bi[k], xi[k]) / ai[k]
        if np.any(swap):
            k = swap
            res[k] = 1.0 - np.exp(logFront[k]) * _betaContinuedFraction(bi[k], ai[k], 1.0 - xi[k]) / bi[k]
        out[inner] = res
    out = out.reshape(shape)
    return out if out.ndim else float(out)


def _betaContinuedFraction(a, b, x):
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = np.ones_like(x)
    d = 1.0 - qab * x / qap
    d = np.where(np.abs(d) < _TINY, _TINY, d)
    d = 1.0 / d
    h = d.copy()
    active = np.ones(x.shape, dtype=bool)
    for m in range(1, _MAX_ITER):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < _TINY, _TINY, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < _TINY, _TINY, c)
        d = 1.0 / d
        h = np.where(active, h * d * c, h)
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < _TINY, _TINY, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < _TINY, _TINY, c)
        d = 1.0 / d
        delta = d * c
        h = np.where(active, h * delta, h)
        active &= np.abs(delta - 1.0) > _EPS
        if not np.any(active):
            break
    return h


# ---------------------------------------------------------------- error function
def erfc(x):
    x = np.asarray(x, dtype=float)
    q = np.asarray(gammaincc(0.5, x * x), dtype=float)
    out = np.where(x >= 0.0, q, 2.0 - q)
    return out if out.ndim else float(out)


def erf(x):
    x = np.asarray(x, dtype=float)
    out = 1.0 - np.asarray(erfc(x))
    return out if out.ndim else float(out)


def ndtr(x):
    """Standard normal CDF."""
    x = np.asarray(x, dtype=float)
    out = 0.5 * np.asarray(erfc(-x / np.sqrt(2.0)))
    return out if out.ndim else float(out)


_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)


def ndtri(p):
    """Inverse standard normal CDF (quantile function)."""
    p = np.asarray(p, dtype=float)
    scalar = p.ndim == 0
    p = np.atleast_1d(p).astype(float)
    x = np.full_like(p, np.nan)
    x[p == 0.0] = -np.inf
    x[p == 1.0] = np.inf
    ok = (p > 0.0) & (p < 1.0)
    pLow = 0.02425
    low = ok & (p < pLow)
    high = ok & (p > 1.0 - pLow)
    mid = ok & ~low & ~high
    if np.any(low):
        q = np.sqrt(-2.0 * np.log(p[low]))
        x[low] = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
                 ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    if np.any(high):
        q = np.sqrt(-2.0 * np.log1p(-p[high]))
        x[high] = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
                  ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    if np.any(mid):
        q = p[mid] - 0.5
        r = q * q
        x[mid] = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
                 (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
    if np.any(ok):
        # Two Halley steps against the high-accuracy CDF.
        for _ in range(2):
            xo = x[ok]
            e = np.asarray(ndtr(xo)) - p[ok]
            u = e * np.sqrt(2.0 * np.pi) * np.exp(0.5 * xo * xo)
            x[ok] = xo - u / (1.0 + 0.5 * xo * u)
    return float(x[0]) if scalar else x


# ---------------------------------------------------------------- scalar fast paths
# Statistical inference mostly asks for single quantiles / p-values; plain
# float arithmetic avoids numpy's per-call overhead (~50x faster for scalars).
def _gammalnScalar(x: float) -> float:
    if x < 0.5:
        return math.log(math.pi / abs(math.sin(math.pi * x))) - _gammalnScalar(1.0 - x)
    z = x - 1.0
    series = _LANCZOS_COEF[0]
    for i in range(1, 9):
        series += _LANCZOS_COEF[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return _HALF_LOG_2PI + (z + 0.5) * math.log(t) - t + math.log(series)


def _betaCfScalar(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) >= _TINY else _TINY)
    h = d
    for m in range(1, _MAX_ITER):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) >= _TINY else _TINY)
        c = 1.0 + aa / c
        c = c if abs(c) >= _TINY else _TINY
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) >= _TINY else _TINY)
        c = 1.0 + aa / c
        c = c if abs(c) >= _TINY else _TINY
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= _EPS:
            break
    return h


def _betaincScalar(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    logFront = (a * math.log(x) + b * math.log1p(-x)
                - (_gammalnScalar(a) + _gammalnScalar(b) - _gammalnScalar(a + b)))
    if x <= (a + 1.0) / (a + b + 2.0):
        return math.exp(logFront) * _betaCfScalar(a, b, x) / a
    return 1.0 - math.exp(logFront) * _betaCfScalar(b, a, 1.0 - x) / b


def _gammaincPairScalar(a: float, x: float) -> tuple[float, float]:
    if x <= 0.0:
        return 0.0, 1.0
    logFront = -x + a * math.log(x) - _gammalnScalar(a)
    if x < a + 1.0:
        ap, term = a, 1.0 / a
        total = term
        for _ in range(_MAX_ITER):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) <= abs(total) * _EPS:
                break
        p = total * math.exp(logFront)
        return p, 1.0 - p
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_ITER):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = d if abs(d) >= _TINY else _TINY
        c = b + an / c
        c = c if abs(c) >= _TINY else _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= _EPS:
            break
    q = math.exp(logFront) * h
    return 1.0 - q, q


def _isScalar(*values) -> bool:
    return all(np.ndim(v) == 0 for v in values)
