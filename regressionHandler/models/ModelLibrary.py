"""Common nonlinear model forms for ``NonlinearModel`` with data-driven initial guesses.

Each entry defines a safe expression in ``x`` (single input), its parameter
names and a guess function that linearizes the model where possible (log
transforms, Lineweaver-type reciprocals, periodograms) so the nonlinear fit
starts close to the optimum. Custom forms can be added with ``registerForm``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

def _linfit(u, v, w=None):
    """Slope and intercept of v ~ a + b u."""
    a = np.column_stack([np.ones_like(u), u])
    coef, *_ = np.linalg.lstsq(a, v, rcond=None)
    return float(coef[1]), float(coef[0])


def _positive(v) -> bool:
    return bool(np.all(np.asarray(v) > 0))


def _order(x, y):
    o = np.argsort(x)
    return x[o], y[o]


def _gLinear(x, y):
    b, a = _linfit(x, y)
    return [a, b]


def _gPowerLaw(x, y):
    if _positive(x) and _positive(np.abs(y)):
        b, lna = _linfit(np.log(x), np.log(np.abs(y)))
        return [np.sign(np.mean(y)) * np.exp(lna), b]
    return [float(np.mean(y)), 1.0]


def _gPowerLawOffset(x, y):
    c = float(np.min(y) - 0.1 * np.ptp(y)) if np.ptp(y) > 0 else 0.0
    a, b = _gPowerLaw(x, y - c)
    return [a, b, c]


def _gExponential(x, y):
    if _positive(np.abs(y)):
        b, lna = _linfit(x, np.log(np.abs(y)))
        return [np.sign(np.mean(y)) * np.exp(lna), b]
    return [float(np.mean(y)), 0.0]


def _gExponentialDecay(x, y):
    xs, ys = _order(x, y)
    c = float(ys[-1])
    a = float(ys[0] - c) or float(np.ptp(y)) or 1.0
    tau = float(np.ptp(x) / 3.0) or 1.0
    return [a, tau, c]


def _gExponentialRise(x, y):
    xs, ys = _order(x, y)
    c = float(ys[0])
    a = float(ys[-1] - c) or 1.0
    tau = float(np.ptp(x) / 3.0) or 1.0
    return [a, tau, c]


def _gInverseExponential(x, y):
    if np.all(x != 0) and _positive(np.abs(y)):
        slope, lnA = _linfit(1.0 / x, np.log(np.abs(y)))
        return [np.sign(np.mean(y)) * np.exp(lnA), slope]
    return [float(np.mean(y)), 0.0]


def _gLogistic(x, y):
    xs, ys = _order(x, y)
    c = float(np.min(y))
    lam = float(np.ptp(y)) or 1.0
    mid = c + 0.5 * lam
    x0 = float(xs[np.argmin(np.abs(ys - mid))])
    direction = np.sign(np.corrcoef(xs, ys)[0, 1]) if np.ptp(xs) > 0 and np.ptp(ys) > 0 else 1.0
    k = 4.0 * direction / (float(np.ptp(x)) or 1.0)
    return [lam, k, x0, c]


def _gSaturation(x, y):
    a = float(np.max(y)) * 1.2 or 1.0
    xs, ys = _order(x, y)
    half = xs[np.argmin(np.abs(ys - 0.5 * np.max(y)))]
    return [a, float(abs(half)) or 1.0]


def _gPowerSaturation(x, y):
    a, b = _gSaturation(x, y)
    return [a, b, 1.0]


def _gGaussianPeak(x, y):
    c = float(np.min(y))
    i = int(np.argmax(y))
    a = float(y[i] - c) or 1.0
    above = x[y > c + 0.5 * a]
    sigma = float(np.ptp(above) / 2.355) if above.size > 1 else float(np.ptp(x) / 6.0)
    return [a, float(x[i]), sigma or 1.0, c]


def _gWeibullCdf(x, y):
    ok = (x > 0) & (y > 0) & (y < 1)
    if np.sum(ok) >= 2:
        k, icpt = _linfit(np.log(x[ok]), np.log(-np.log1p(-y[ok])))
        return [np.exp(-icpt / k) if k != 0 else float(np.median(x)), k]
    return [float(np.median(x)) or 1.0, 1.5]


def _gSinusoid(x, y):
    """Best frequency of a least-squares periodogram (works on uneven sampling)."""
    span = float(np.ptp(x)) or 1.0
    dxs = np.diff(np.sort(x))
    dx = float(np.median(dxs[dxs > 0])) if np.any(dxs > 0) else span
    freqs = np.linspace(0.5 / span, 0.5 / dx, 600)
    best, bestSse = None, np.inf
    for f in freqs:
        a = np.column_stack([np.ones_like(x), np.sin(2 * np.pi * f * x), np.cos(2 * np.pi * f * x)])
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        sse = float(np.sum((a @ coef - y) ** 2))
        if sse < bestSse:
            best, bestSse = (f, coef), sse
    f, (c, s, co) = best
    return [float(np.hypot(s, co)), float(f), float(np.arctan2(co, s)), float(c)]


def _slope(u, v):
    return _linfit(np.log(u), np.log(v))[0] if u.size >= 2 else 1.0


def _gRationalPower(x, y):
    """a*x**n/(x + b): log-log slope is ~n well below b and ~n-1 well above b."""
    if not (_positive(x) and _positive(np.abs(y))):
        return [float(np.mean(y)), 1.0, float(np.median(np.abs(x))) or 1.0]
    xs, ys = _order(x, np.abs(y))
    k = max(2, xs.size // 3)
    n = 0.5 * (_slope(xs[:k], ys[:k]) + _slope(xs[-k:], ys[-k:]) + 1.0)
    b = float(np.median(xs))
    a = float(np.mean(y * (x + b) / x ** n))
    return [a, n, b]


def _gLinearPlusPower(x, y):
    """a*x + b*x**n: a from the low-x slope, (b, n) from the remaining growth."""
    xs, ys = _order(x, y)
    m = max(3, xs.size // 5)
    den = float(np.sum(xs[:m] ** 2))
    a = float(np.sum(xs[:m] * ys[:m]) / den) if den > 0 else 1.0
    rest = ys - a * xs
    upper = (xs > 0) & (rest > 0) & (np.arange(xs.size) >= xs.size // 2)
    if np.sum(upper) >= 2:
        n, lnb = _linfit(np.log(xs[upper]), np.log(rest[upper]))
        return [a, float(np.exp(lnb)), float(n)]
    return [a, 0.0, 2.0]


def _gLogarithmic(x, y):
    b, a = _linfit(np.log(x), y) if _positive(x) else (1.0, float(np.mean(y)))
    return [a, b]


def _gInverse(x, y):
    b, a = _linfit(1.0 / x, y) if np.all(x != 0) else (1.0, float(np.mean(y)))
    return [a, b]


LIBRARY: dict[str, dict] = {
    "linear": dict(expression="a + b*x", params=["a", "b"], guess=_gLinear, description="Straight line"),
    "powerLaw": dict(expression="a*x**b", params=["a", "b"], guess=_gPowerLaw, description="Power law"),
    "powerLawOffset": dict(expression="a*x**b + c", params=["a", "b", "c"], guess=_gPowerLawOffset,
                           description="Power law with offset"),
    "exponential": dict(expression="a*exp(b*x)", params=["a", "b"], guess=_gExponential,
                        description="Exponential growth / decay"),
    "exponentialDecay": dict(expression="a*exp(-x/tau) + c", params=["a", "tau", "c"], guess=_gExponentialDecay,
                             description="Exponential decay to an asymptote"),
    "exponentialRise": dict(expression="a*(1 - exp(-x/tau)) + c", params=["a", "tau", "c"],
                            guess=_gExponentialRise, description="Exponential approach from below"),
    "inverseExponential": dict(expression="a*exp(b/x)", params=["a", "b"], guess=_gInverseExponential,
                               description="Exponential of the reciprocal, a*exp(b/x)"),
    "logistic": dict(expression="L/(1 + exp(-k*(x - x0))) + c", params=["L", "k", "x0", "c"], guess=_gLogistic,
                     description="Logistic sigmoid with offset"),
    "saturation": dict(expression="a*x/(b + x)", params=["a", "b"], guess=_gSaturation,
                       description="Rectangular-hyperbola saturation"),
    "powerSaturation": dict(expression="a*x**n/(b**n + x**n)", params=["a", "b", "n"], guess=_gPowerSaturation,
                            description="Saturation with a power-law transition"),
    "gaussianPeak": dict(expression="a*exp(-(x - mu)**2/(2*sigma**2)) + c", params=["a", "mu", "sigma", "c"],
                         guess=_gGaussianPeak, description="Gaussian peak on a baseline"),
    "weibullCdf": dict(expression="1 - exp(-(x/lam)**k)", params=["lam", "k"], guess=_gWeibullCdf,
                       description="Weibull cumulative distribution function"),
    "sinusoid": dict(expression="a*sin(2*pi*f*x + phi) + c", params=["a", "f", "phi", "c"], guess=_gSinusoid,
                     description="Sinusoid (periodogram initial frequency)"),
    "rationalPower": dict(expression="a*x**n/(x + b)", params=["a", "n", "b"], guess=_gRationalPower,
                          description="Power law divided by a linear term"),
    "linearPlusPower": dict(expression="a*x + b*x**n", params=["a", "b", "n"], guess=_gLinearPlusPower,
                            description="Linear term plus a power-law term"),
    "logarithmic": dict(expression="a + b*log(x)", params=["a", "b"], guess=_gLogarithmic,
                        description="Logarithmic trend"),
    "inverse": dict(expression="a + b/x", params=["a", "b"], guess=_gInverse, description="Hyperbolic trend"),
}


def libraryEntry(name: str) -> dict:
    try:
        return LIBRARY[name]
    except KeyError:
        raise KeyError(f"unknown library form {name!r}; available: {', '.join(sorted(LIBRARY))}") from None


def registerForm(name: str, expression: str, params: list[str], guess: Callable, description: str = "",
                 constants: dict | None = None, variables: list[str] | None = None) -> None:
    """Add a custom model form to the library."""
    LIBRARY[name] = dict(expression=expression, params=list(params), guess=guess, description=description,
                         constants=dict(constants or {}), variables=list(variables or ["x"]))


def libraryCatalog() -> list[dict]:
    return [{"name": k, "expression": v["expression"], "params": v["params"], "description": v["description"],
             "constants": v.get("constants", {})} for k, v in sorted(LIBRARY.items())]
