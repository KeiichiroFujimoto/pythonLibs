"""Exponential-family distributions and link functions for generalized linear / additive models.

A ``Family`` supplies the variance function V(mu), the unit deviance d(y, mu),
the log-likelihood (for AIC), starting values and the valid ranges; a
``Link`` supplies eta = g(mu), mu = g^-1(eta) and dmu/deta. Families are
built from a name or a dict::

    buildFamily("poisson")                                  # canonical log link
    buildFamily({"type": "binomial", "link": "probit"})
    buildFamily({"type": "negativeBinomial", "theta": 2.0})  # theta=None: estimated
    buildFamily({"type": "tweedie", "power": 1.5})

Binomial responses are proportions in [0, 1] with the number of trials as
prior weights (a 0/1 response has unit weights).

References:
    McCullagh & Nelder (1989) *Generalized Linear Models*, 2nd ed.
    Jorgensen (1997) *The Theory of Dispersion Models* (Tweedie family).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.numerics import SpecialFunctions as _sf

_EPS = np.finfo(float).eps
_TINY = 1e-300


# ---------------------------------------------------------------- links
class Link:
    """Monotone link g: mean -> linear predictor."""
    name = "link"

    def link(self, mu):
        raise NotImplementedError

    def inverse(self, eta):
        raise NotImplementedError

    def dmuDeta(self, eta):
        raise NotImplementedError

    def validEta(self, eta) -> bool:
        return bool(np.all(np.isfinite(eta)))

    def toDict(self) -> dict:
        return {"type": self.name}


class IdentityLink(Link):
    name = "identity"

    def link(self, mu):
        return np.asarray(mu, dtype=float)

    def inverse(self, eta):
        return np.asarray(eta, dtype=float)

    def dmuDeta(self, eta):
        return np.ones_like(np.asarray(eta, dtype=float))


class LogLink(Link):
    name = "log"

    def link(self, mu):
        return np.log(np.maximum(mu, _TINY))

    def inverse(self, eta):
        return np.exp(np.minimum(eta, 700.0))

    def dmuDeta(self, eta):
        return np.maximum(np.exp(np.minimum(eta, 700.0)), _TINY)


class LogitLink(Link):
    name = "logit"

    def link(self, mu):
        mu = np.clip(mu, _TINY, 1.0 - _EPS)
        return np.log(mu / (1.0 - mu))

    def inverse(self, eta):
        eta = np.clip(eta, -700.0, 700.0)
        return 1.0 / (1.0 + np.exp(-eta))

    def dmuDeta(self, eta):
        eta = np.clip(np.abs(eta), None, 700.0)
        e = np.exp(-eta)
        return np.maximum(e / (1.0 + e) ** 2, _EPS)


class ProbitLink(Link):
    name = "probit"

    def link(self, mu):
        return _sf.ndtri(np.clip(mu, _EPS, 1.0 - _EPS))

    def inverse(self, eta):
        return _sf.ndtr(np.asarray(eta, dtype=float))

    def dmuDeta(self, eta):
        eta = np.asarray(eta, dtype=float)
        return np.maximum(np.exp(-0.5 * eta * eta) / np.sqrt(2.0 * np.pi), _EPS)


class CloglogLink(Link):
    name = "cloglog"

    def link(self, mu):
        mu = np.clip(mu, _TINY, 1.0 - _EPS)
        return np.log(-np.log1p(-mu))

    def inverse(self, eta):
        eta = np.minimum(eta, 700.0)
        return np.clip(-np.expm1(-np.exp(eta)), _EPS, 1.0 - _EPS)

    def dmuDeta(self, eta):
        eta = np.minimum(eta, 700.0)
        return np.maximum(np.exp(eta - np.exp(eta)), _EPS)


class InverseLink(Link):
    name = "inverse"

    def link(self, mu):
        return 1.0 / np.asarray(mu, dtype=float)

    def inverse(self, eta):
        return 1.0 / np.asarray(eta, dtype=float)

    def dmuDeta(self, eta):
        return -1.0 / np.asarray(eta, dtype=float) ** 2

    def validEta(self, eta) -> bool:
        return bool(np.all(np.isfinite(eta)) and (np.all(eta > 0) or np.all(eta < 0)))


class InverseSquaredLink(Link):
    name = "inverseSquared"

    def link(self, mu):
        return 1.0 / np.asarray(mu, dtype=float) ** 2

    def inverse(self, eta):
        return 1.0 / np.sqrt(eta)

    def dmuDeta(self, eta):
        return -0.5 * np.asarray(eta, dtype=float) ** -1.5

    def validEta(self, eta) -> bool:
        return bool(np.all(np.isfinite(eta)) and np.all(eta > 0))


class SqrtLink(Link):
    name = "sqrt"

    def link(self, mu):
        return np.sqrt(mu)

    def inverse(self, eta):
        return np.asarray(eta, dtype=float) ** 2

    def dmuDeta(self, eta):
        return 2.0 * np.asarray(eta, dtype=float)

    def validEta(self, eta) -> bool:
        return bool(np.all(np.isfinite(eta)) and np.all(eta > 0))


class PowerLink(Link):
    """eta = mu^power (power 0 is the log link)."""
    name = "power"

    def __init__(self, power: float) -> None:
        self.power = float(power)

    def link(self, mu):
        return np.asarray(mu, dtype=float) ** self.power

    def inverse(self, eta):
        return np.maximum(eta, _TINY) ** (1.0 / self.power)

    def dmuDeta(self, eta):
        return (1.0 / self.power) * np.maximum(eta, _TINY) ** (1.0 / self.power - 1.0)

    def validEta(self, eta) -> bool:
        return bool(np.all(np.isfinite(eta)) and np.all(eta > 0))

    def toDict(self) -> dict:
        return {"type": "power", "power": self.power}


LINKS = {"identity": IdentityLink, "log": LogLink, "logit": LogitLink, "probit": ProbitLink,
         "cloglog": CloglogLink, "inverse": InverseLink, "inverseSquared": InverseSquaredLink, "sqrt": SqrtLink}


def buildLink(spec) -> Link:
    if isinstance(spec, Link):
        return spec
    if isinstance(spec, dict):
        if spec.get("type") == "power":
            p = float(spec.get("power", 1.0))
            return LogLink() if p == 0.0 else PowerLink(p)
        spec = spec.get("type")
    if spec not in LINKS:
        raise ValueError(f"unknown link {spec!r}; choose from {sorted(LINKS) + ['power']}")
    return LINKS[spec]()


# ---------------------------------------------------------------- families
def _xlogy(x, y):
    """x log(y) with 0 log 0 = 0."""
    x = np.asarray(x, dtype=float)
    return np.where(x == 0.0, 0.0, x * np.log(np.where(x == 0.0, 1.0, y)))


class Family:
    """Exponential dispersion family: Var(y) = phi V(mu) / w."""
    name = "family"
    canonicalLink = "identity"
    scaleKnown = False            # phi fixed at 1 (binomial, poisson)
    extraParams = 0               # dispersion-like parameters counted in AIC

    def __init__(self, link=None) -> None:
        self.link = buildLink(link or self.canonicalLink)

    def variance(self, mu):
        raise NotImplementedError

    def dVariance(self, mu):
        """dV/dmu (used by the expected-information derivatives)."""
        raise NotImplementedError

    def unitDeviance(self, y, mu):
        raise NotImplementedError

    def deviance(self, y, mu, w) -> float:
        return float(np.sum(w * self.unitDeviance(y, mu)))

    def logLikelihood(self, y, mu, w, phi: float) -> float:
        raise NotImplementedError

    def initialize(self, y, w) -> np.ndarray:
        return np.asarray(y, dtype=float).copy()

    def validMu(self, mu) -> bool:
        return bool(np.all(np.isfinite(mu)))

    def checkResponse(self, y) -> None:
        pass

    def toDict(self) -> dict:
        return {"type": self.name, "link": self.link.toDict()}

    def describe(self) -> str:
        return f"{self.name}({self.link.name})"


class Gaussian(Family):
    name = "gaussian"
    canonicalLink = "identity"
    extraParams = 1

    def variance(self, mu):
        return np.ones_like(np.asarray(mu, dtype=float))

    def dVariance(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=float))

    def unitDeviance(self, y, mu):
        return (y - mu) ** 2

    def logLikelihood(self, y, mu, w, phi):
        return float(-0.5 * np.sum(w * (y - mu) ** 2 / phi + np.log(2.0 * np.pi * phi / w)))


class Binomial(Family):
    name = "binomial"
    canonicalLink = "logit"
    scaleKnown = True

    def variance(self, mu):
        return np.maximum(mu * (1.0 - mu), _TINY)

    def dVariance(self, mu):
        return 1.0 - 2.0 * mu

    def unitDeviance(self, y, mu):
        mu = np.clip(mu, _TINY, 1.0 - _EPS)
        return 2.0 * (_xlogy(y, y / mu) + _xlogy(1.0 - y, (1.0 - y) / (1.0 - mu)))

    def logLikelihood(self, y, mu, w, phi):
        mu = np.clip(mu, _TINY, 1.0 - _EPS)
        m = np.round(w)
        k = np.round(m * y)
        comb = _sf.gammaln(m + 1.0) - _sf.gammaln(k + 1.0) - _sf.gammaln(m - k + 1.0)
        return float(np.sum(comb + _xlogy(k, mu) + _xlogy(m - k, 1.0 - mu)))

    def initialize(self, y, w):
        return (w * y + 0.5) / (w + 1.0)

    def validMu(self, mu):
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0) and np.all(mu < 1))

    def checkResponse(self, y):
        if np.any(y < 0) or np.any(y > 1):
            raise ValueError("binomial responses must be proportions in [0, 1] (trials as weights)")


class Poisson(Family):
    name = "poisson"
    canonicalLink = "log"
    scaleKnown = True

    def variance(self, mu):
        return np.maximum(mu, _TINY)

    def dVariance(self, mu):
        return np.ones_like(np.asarray(mu, dtype=float))

    def unitDeviance(self, y, mu):
        mu = np.maximum(mu, _TINY)
        return 2.0 * (_xlogy(y, y / mu) - (y - mu))

    def logLikelihood(self, y, mu, w, phi):
        mu = np.maximum(mu, _TINY)
        return float(np.sum(w * (_xlogy(y, mu) - mu - _sf.gammaln(y + 1.0))))

    def initialize(self, y, w):
        return y + 0.1

    def validMu(self, mu):
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0))

    def checkResponse(self, y):
        if np.any(y < 0):
            raise ValueError("poisson responses must be non-negative")


class Gamma(Family):
    name = "gamma"
    canonicalLink = "inverse"
    extraParams = 1

    def variance(self, mu):
        return np.asarray(mu, dtype=float) ** 2

    def dVariance(self, mu):
        return 2.0 * np.asarray(mu, dtype=float)

    def unitDeviance(self, y, mu):
        return 2.0 * (-np.log(y / mu) + (y - mu) / mu)

    def logLikelihood(self, y, mu, w, phi):
        shape = 1.0 / phi
        return float(np.sum(w * (shape * np.log(shape * y / mu) - shape * y / mu - np.log(y) - _sf.gammaln(shape))))

    def validMu(self, mu):
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0))

    def checkResponse(self, y):
        if np.any(y <= 0):
            raise ValueError("gamma responses must be positive")


class InverseGaussian(Family):
    name = "inverseGaussian"
    canonicalLink = "inverseSquared"
    extraParams = 1

    def variance(self, mu):
        return np.asarray(mu, dtype=float) ** 3

    def dVariance(self, mu):
        return 3.0 * np.asarray(mu, dtype=float) ** 2

    def unitDeviance(self, y, mu):
        return (y - mu) ** 2 / (mu * mu * y)

    def logLikelihood(self, y, mu, w, phi):
        return float(-0.5 * np.sum(w * ((y - mu) ** 2 / (phi * mu * mu * y) + np.log(2.0 * np.pi * phi * y ** 3 / w))))

    def validMu(self, mu):
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0))

    def checkResponse(self, y):
        if np.any(y <= 0):
            raise ValueError("inverse Gaussian responses must be positive")


class NegativeBinomial(Family):
    """Var = mu + mu^2 / theta; theta fixed or estimated by maximum likelihood (theta=None)."""
    name = "negativeBinomial"
    canonicalLink = "log"
    scaleKnown = True

    def __init__(self, link=None, theta: Optional[float] = None) -> None:
        super().__init__(link)
        self.estimateTheta = theta is None
        self.theta = 1.0 if theta is None else float(theta)
        self.extraParams = 1 if self.estimateTheta else 0

    def variance(self, mu):
        return np.maximum(mu + mu * mu / self.theta, _TINY)

    def dVariance(self, mu):
        return 1.0 + 2.0 * mu / self.theta

    def unitDeviance(self, y, mu):
        mu = np.maximum(mu, _TINY)
        t = self.theta
        return 2.0 * (_xlogy(y, y / mu) - (y + t) * np.log((y + t) / (mu + t)))

    def logLikelihood(self, y, mu, w, phi):
        t = self.theta
        mu = np.maximum(mu, _TINY)
        ll = (_sf.gammaln(y + t) - _sf.gammaln(t) - _sf.gammaln(y + 1.0) + t * np.log(t / (t + mu))
              + _xlogy(y, mu / (t + mu)))
        return float(np.sum(w * ll))

    def updateTheta(self, y, mu, w, maxIter: int = 50) -> None:
        """Maximum-likelihood theta for fixed mu (Newton on log theta)."""
        if not self.estimateTheta:
            return
        mu = np.maximum(mu, _TINY)
        logT = np.log(self.theta)
        for _ in range(maxIter):
            t = np.exp(logT)
            score = np.sum(w * (_sf.digamma(y + t) - _sf.digamma(t) + np.log(t) + 1.0 - np.log(t + mu)
                                - (y + t) / (t + mu)))
            info = np.sum(w * (_sf.trigamma(y + t) - _sf.trigamma(t) + 1.0 / t - 2.0 / (t + mu)
                               + (y + t) / (t + mu) ** 2))
            # d/dlogT: score*t ; d2/dlogT2: t*score + t^2*info
            g = score * t
            h = t * score + t * t * info
            step = -g / h if h < 0 else np.sign(g) * 0.5
            step = float(np.clip(step, -2.0, 2.0))
            logT = float(np.clip(logT + step, np.log(1e-6), np.log(1e8)))
            if abs(step) < 1e-9:
                break
        self.theta = float(np.exp(logT))

    def validMu(self, mu):
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0))

    def initialize(self, y, w):
        return y + 0.1

    def checkResponse(self, y):
        if np.any(y < 0):
            raise ValueError("negative binomial responses must be non-negative")

    def toDict(self) -> dict:
        return {"type": self.name, "link": self.link.toDict(), "theta": self.theta,
                "estimateTheta": self.estimateTheta}

    def describe(self) -> str:
        return f"negativeBinomial(theta={self.theta:.4g}, {self.link.name})"


class Tweedie(Family):
    """Var = mu^power, 1 < power < 2 (compound Poisson-gamma: exact zeros and positive values)."""
    name = "tweedie"
    canonicalLink = "log"

    def __init__(self, link=None, power: float = 1.5) -> None:
        if not 1.0 < power < 2.0:
            raise ValueError("tweedie power must be in (1, 2)")
        self.power = float(power)
        super().__init__(link)

    def variance(self, mu):
        return np.maximum(mu, _TINY) ** self.power

    def dVariance(self, mu):
        return self.power * np.maximum(mu, _TINY) ** (self.power - 1.0)

    def unitDeviance(self, y, mu):
        p = self.power
        mu = np.maximum(mu, _TINY)
        return 2.0 * (np.maximum(y, 0.0) ** (2 - p) / ((1 - p) * (2 - p)) - y * mu ** (1 - p) / (1 - p)
                      + mu ** (2 - p) / (2 - p))

    def logLikelihood(self, y, mu, w, phi):
        return float("nan")            # needs series evaluation of the density; AIC is not reported

    def initialize(self, y, w):
        return y + 0.1 * max(float(np.mean(y)), 1e-3)

    def validMu(self, mu):
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0))

    def checkResponse(self, y):
        if np.any(y < 0):
            raise ValueError("tweedie responses must be non-negative")

    def toDict(self) -> dict:
        return {"type": self.name, "link": self.link.toDict(), "power": self.power}


FAMILIES = {"gaussian": Gaussian, "binomial": Binomial, "poisson": Poisson, "gamma": Gamma,
            "inverseGaussian": InverseGaussian, "negativeBinomial": NegativeBinomial, "tweedie": Tweedie}


def buildFamily(spec) -> Family:
    """Family from a name, a dict {"type", "link", ...} or an instance."""
    if isinstance(spec, Family):
        return spec
    if isinstance(spec, str):
        spec = {"type": spec}
    spec = dict(spec)
    name = spec.pop("type")
    if name not in FAMILIES:
        raise ValueError(f"unknown family {name!r}; choose from {sorted(FAMILIES)}")
    link = spec.pop("link", None)
    spec.pop("estimateTheta", None)
    return FAMILIES[name](link=link, **spec)
