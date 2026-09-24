"""Empirical semivariograms and variogram model fitting.

    gamma(h) = 1/2 E[(Z(s + h) - Z(s))^2]

``empiricalVariogram`` bins the pairs by distance (optionally only pairs in
a direction cone) with the classical (Matheron) or the robust
(Cressie-Hawkins) estimator. ``fitVariogram`` fits

    gamma(h) = nugget * [h > 0] + partialSill * (1 - rho(h / range))

by weighted least squares (Cressie weights N_h / gamma(h)^2 by default).
``VariogramFit.krigingOptions()`` turns the fit into ``KrigingModel``
options (raw units), e.g. as fixed hyperparameters or a starting point.

References:
    Matheron (1962) Traite de geostatistique appliquee.
    Cressie & Hawkins (1980) Mathematical Geology 12(2).
    Cressie (1985) Mathematical Geology 17(5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.numerics import SpecialFunctions as _sf
from pythonLibs.regressionHandler.numerics.Optimizers import leastSquares

EARTH_RADIUS = {"km": 6378.388, "miles": 3963.34}


# ---------------------------------------------------------------- correlation functions rho(t), t = h / range
def _rhoExponential(t, nu):
    return np.exp(-t)


def _rhoGaussian(t, nu):
    return np.exp(-t * t)


def _rhoSpherical(t, nu):
    return np.where(t < 1.0, 1.0 - 1.5 * t + 0.5 * t ** 3, 0.0)


def _rhoCubic(t, nu):
    tt = np.minimum(t, 1.0)
    return np.where(t < 1.0, 1.0 - 7 * tt ** 2 + 8.75 * tt ** 3 - 3.5 * tt ** 5 + 0.75 * tt ** 7, 0.0)


def _rhoMatern(t, nu):
    t = np.asarray(t, dtype=float)
    out = np.ones_like(t)
    pos = t > 0
    if np.any(pos):
        tp = t[pos]
        with np.errstate(under="ignore"):
            out[pos] = np.exp((1.0 - nu) * np.log(2.0) - _sf.gammaln(nu) + nu * np.log(tp)) * _sf.besselK(nu, tp)
    return out


def _rhoWendland(t, nu):
    # Wendland phi_{3,1} (positive definite up to 3-D): (1 - t)^4 (4 t + 1)
    tt = np.minimum(t, 1.0)
    return (1.0 - tt) ** 4 * (4.0 * tt + 1.0)


VARIOGRAM_MODELS = {"exponential": _rhoExponential, "gaussian": _rhoGaussian, "spherical": _rhoSpherical,
                    "cubic": _rhoCubic, "matern": _rhoMatern, "wendland": _rhoWendland}


# ---------------------------------------------------------------- empirical variogram
@dataclass
class EmpiricalVariogram:
    """Binned semivariogram: bin centers, mean pair distance, gamma and pair counts per bin."""
    binEdges: np.ndarray
    distance: np.ndarray
    gamma: np.ndarray
    counts: np.ndarray
    estimator: str
    direction: Optional[np.ndarray] = None
    extra: dict = field(default_factory=dict)

    def toDict(self) -> dict:
        return {"binEdges": self.binEdges.tolist(), "distance": self.distance.tolist(), "gamma": self.gamma.tolist(),
                "counts": self.counts.tolist(), "estimator": self.estimator,
                "direction": None if self.direction is None else self.direction.tolist()}


def _pairDistances(xa, xb, distance: str, radiusUnit: str):
    if distance == "greatCircle":
        rad = np.pi / 180.0
        lon1, lat1 = xa[:, [0]] * rad, xa[:, [1]] * rad
        lon2, lat2 = xb[:, 0][None, :] * rad, xb[:, 1][None, :] * rad
        h = np.sin(0.5 * (lat1 - lat2)) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * (lon1 - lon2)) ** 2
        return 2.0 * EARTH_RADIUS[radiusUnit] * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
    diff2 = np.zeros((xa.shape[0], xb.shape[0]))
    for k in range(xa.shape[1]):
        d = xa[:, k][:, None] - xb[:, k][None, :]
        diff2 += d * d
    return np.sqrt(diff2)


def _directionVector(direction, nx: int) -> np.ndarray:
    if np.ndim(direction) == 0:
        if nx != 2:
            raise ValueError("an angle direction needs 2-D inputs; give a direction vector otherwise")
        a = np.deg2rad(float(direction))
        v = np.array([np.cos(a), np.sin(a)])
    else:
        v = np.asarray(direction, dtype=float)
        if v.size != nx:
            raise ValueError("direction vector must have one entry per input")
    return v / np.linalg.norm(v)


def empiricalVariogram(x, y, nBins: int = 15, maxDistance: Optional[float] = None, binEdges=None,
                       estimator: str = "classical", direction=None, tolerance: float = 22.5,
                       distance: str = "euclidean", radiusUnit: str = "km",
                       chunkSize: int = 2048) -> EmpiricalVariogram:
    """Binned empirical semivariogram of y(x).

    Args:
        x:            (n, nx) locations (longitude, latitude in degrees for greatCircle)
        y:            (n,) values (residuals of a trend fit for a non-constant mean)
        nBins:        number of equal-width bins on (0, maxDistance]
        maxDistance:  largest lag (default: half the largest pairwise distance)
        binEdges:     explicit bin edges (overrides nBins / maxDistance)
        estimator:    "classical" (Matheron) or "robust" (Cressie-Hawkins)
        direction:    None (omnidirectional), an angle in degrees from the first axis (2-D)
                      or a direction vector; pairs within ``tolerance`` degrees are used
        distance:     "euclidean" or "greatCircle"
    """
    x = np.asarray(x, dtype=float)
    x = x[:, None] if x.ndim == 1 else x
    y = np.asarray(y, dtype=float).ravel()
    n, nx = x.shape
    if y.size != n:
        raise ValueError("x and y must have the same number of rows")
    if estimator not in ("classical", "robust"):
        raise ValueError("estimator must be 'classical' or 'robust'")
    if direction is not None and distance == "greatCircle":
        raise ValueError("directional variograms need euclidean distance")
    dirVec = None if direction is None else _directionVector(direction, nx)
    cosTol = np.cos(np.deg2rad(tolerance))

    if binEdges is None:
        if maxDistance is None:
            span = np.ptp(x, axis=0)
            maxDistance = 0.5 * (float(np.linalg.norm(span)) if distance == "euclidean"
                                 else float(_pairDistances(x, x, distance, radiusUnit).max()))
        edges = np.linspace(0.0, float(maxDistance), nBins + 1)
    else:
        edges = np.asarray(binEdges, dtype=float)
    nb = edges.size - 1
    sumSq, sumRoot, sumDist, counts = np.zeros(nb), np.zeros(nb), np.zeros(nb), np.zeros(nb)

    for start in range(0, n, chunkSize):
        rows = np.arange(start, min(start + chunkSize, n))
        d = _pairDistances(x[rows], x, distance, radiusUnit)
        mask = np.arange(n)[None, :] > rows[:, None]                     # each pair once
        mask &= (d > edges[0]) & (d <= edges[-1])
        if dirVec is not None:
            proj = np.abs(np.tensordot(x[rows][:, None, :] - x[None, :, :], dirVec, axes=([2], [0])))
            with np.errstate(invalid="ignore", divide="ignore"):
                mask &= proj >= cosTol * d
        i, j = np.nonzero(mask)
        if i.size == 0:
            continue
        dd = d[i, j]
        dy = y[rows][i] - y[j]
        b = np.clip(np.searchsorted(edges, dd, side="left") - 1, 0, nb - 1)
        counts += np.bincount(b, minlength=nb)
        sumDist += np.bincount(b, weights=dd, minlength=nb)
        sumSq += np.bincount(b, weights=dy * dy, minlength=nb)
        sumRoot += np.bincount(b, weights=np.sqrt(np.abs(dy)), minlength=nb)

    ok = counts > 0
    gamma = np.full(nb, np.nan)
    dist = np.full(nb, np.nan)
    dist[ok] = sumDist[ok] / counts[ok]
    if estimator == "classical":
        gamma[ok] = 0.5 * sumSq[ok] / counts[ok]
    else:
        m = counts[ok]
        gamma[ok] = 0.5 * (sumRoot[ok] / m) ** 4 / (0.457 + 0.494 / m + 0.045 / m ** 2)
    return EmpiricalVariogram(binEdges=edges, distance=dist[ok], gamma=gamma[ok],
                              counts=counts[ok].astype(int), estimator=estimator, direction=dirVec,
                              extra={"binIndex": np.flatnonzero(ok)})


# ---------------------------------------------------------------- model fitting
@dataclass
class VariogramFit:
    """Fitted variogram model gamma(h) = nugget [h > 0] + partialSill (1 - rho(h / range))."""
    model: str
    nugget: float
    partialSill: float
    range: float
    nu: Optional[float]
    sse: float
    weights: str

    @property
    def sill(self) -> float:
        return self.nugget + self.partialSill

    def gamma(self, h) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        rho = VARIOGRAM_MODELS[self.model](h / self.range, self.nu)
        return np.where(h > 0, self.nugget + self.partialSill * (1.0 - rho), 0.0)

    def covariance(self, h) -> np.ndarray:
        """C(h) = sill - gamma(h) (nugget only at h = 0)."""
        return self.sill - self.gamma(h)

    def practicalRange(self) -> float:
        """Distance at which gamma reaches 95 % of the partial sill (the range itself for compact models)."""
        if self.model in ("spherical", "cubic", "wendland"):
            return self.range
        lo, hi = 0.0, self.range
        f = lambda t: 1.0 - VARIOGRAM_MODELS[self.model](np.array([t]), self.nu)[0] - 0.95
        while f(hi / self.range) < 0:
            hi *= 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid / self.range) < 0 else (lo, mid)
        return 0.5 * (lo + hi)

    def krigingOptions(self) -> dict:
        """KrigingModel options reproducing this covariance (use with ``normalize=False``, isotropic).

        The Kriging nugget is the ratio lambda = nugget / partialSill.
        """
        spec = {"exponential": {"type": "matern", "nu": 0.5, "parameterization": "range", "ard": False},
                "matern": {"type": "matern", "nu": self.nu, "parameterization": "range", "ard": False},
                "gaussian": {"type": "squaredExponential", "ard": False},
                "spherical": {"type": "spherical", "ard": False},
                "wendland": {"type": "wendland", "k": 1, "dimension": 3, "ard": False}}
        if self.model not in spec:
            raise ValueError(f"no Kriging kernel for the {self.model} variogram")
        length = self.range / np.sqrt(2.0) if self.model == "gaussian" else self.range
        lam = max(self.nugget / self.partialSill, 1e-10)
        return {"corr": spec[self.model], "nugget": lam, "hyperparameters": [float(np.log(length))],
                "normalize": False}

    def toDict(self) -> dict:
        return {"model": self.model, "nugget": self.nugget, "partialSill": self.partialSill, "range": self.range,
                "nu": self.nu, "sse": self.sse, "weights": self.weights, "sill": self.sill}


def fitVariogram(empirical: EmpiricalVariogram, model: str = "exponential", nu: Optional[float] = 0.5,
                 weights: str = "cressie", nugget: Optional[float] = None, fitNu: bool = False,
                 start: Optional[dict] = None) -> VariogramFit:
    """Weighted least-squares fit of a variogram model to an empirical variogram.

    Args:
        model:    exponential, gaussian, spherical, cubic, matern, wendland
        nu:       Matern smoothness (fixed unless ``fitNu``)
        weights:  "cressie" (N_h / gamma(h)^2, iterated), "counts" (N_h / h^2) or "equal"
        nugget:   fixed nugget value (default: estimated, >= 0)
        start:    optional initial {"nugget", "partialSill", "range"}
    """
    if model not in VARIOGRAM_MODELS:
        raise ValueError(f"model must be one of {sorted(VARIOGRAM_MODELS)}")
    if weights not in ("cressie", "counts", "equal"):
        raise ValueError("weights must be 'cressie', 'counts' or 'equal'")
    h, g, nh = empirical.distance, empirical.gamma, empirical.counts.astype(float)
    if h.size < 3:
        raise ValueError("need at least three non-empty bins")
    rho = VARIOGRAM_MODELS[model]
    useNu = model == "matern"
    estNugget = nugget is None
    gMax = float(np.max(g))
    s = start or {}
    p0 = [np.log(max(s.get("partialSill", 0.8 * gMax), 1e-12 * gMax)),
          np.log(max(s.get("range", 0.3 * float(np.max(h))), 1e-12))]
    lb, ub = [np.log(1e-8 * gMax), np.log(1e-6 * float(np.min(h)) + 1e-300)], [np.log(100 * gMax), np.log(1e3 * float(np.max(h)))]
    if estNugget:
        p0.append(np.log(max(s.get("nugget", 0.1 * float(np.min(g))), 1e-6 * gMax)))
        lb.append(np.log(1e-10 * gMax))
        ub.append(np.log(10 * gMax))
    if useNu and fitNu:
        p0.append(np.log(nu or 0.5))
        lb.append(np.log(0.05))
        ub.append(np.log(10.0))

    def unpack(p):
        ps, rg = np.exp(p[0]), np.exp(p[1])
        k = 2
        nug = float(nugget) if not estNugget else float(np.exp(p[k]))
        k += estNugget
        v = float(np.exp(p[k])) if (useNu and fitNu) else nu
        return ps, rg, nug, v

    def modelGamma(p):
        ps, rg, nug, v = unpack(p)
        return nug + ps * (1.0 - rho(h / rg, v))

    w = {"equal": np.ones_like(h), "counts": nh / np.maximum(h, 1e-300) ** 2}.get(weights)
    p = np.array(p0)
    for _ in range(1 if weights != "cressie" else 5):
        if weights == "cressie":
            wCur = nh / np.maximum(modelGamma(p), 1e-12 * gMax) ** 2
        else:
            wCur = w
        sw = np.sqrt(wCur / np.mean(wCur))
        res = leastSquares(lambda q: sw * (modelGamma(q) - g) / gMax, p, bounds=(np.array(lb), np.array(ub)))
        if np.allclose(res.x, p, rtol=1e-6, atol=1e-8):
            p = res.x
            break
        p = res.x
    ps, rg, nug, v = unpack(p)
    sse = float(np.sum(wCur * (modelGamma(p) - g) ** 2))
    return VariogramFit(model=model, nugget=nug, partialSill=float(ps), range=float(rg),
                        nu=float(v) if useNu else None, sse=sse, weights=weights)


def variogramCloud(x, y, maxPairs: int = 20000, seed: int = 0, distance: str = "euclidean",
                   radiusUnit: str = "km") -> tuple[np.ndarray, np.ndarray]:
    """(distances, half squared differences) of up to ``maxPairs`` random pairs."""
    x = np.asarray(x, dtype=float)
    x = x[:, None] if x.ndim == 1 else x
    y = np.asarray(y, dtype=float).ravel()
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    total = n * (n - 1) // 2
    if total <= maxPairs:
        i, j = np.triu_indices(n, 1)
    else:
        i = rng.integers(0, n, maxPairs * 2)
        j = rng.integers(0, n, maxPairs * 2)
        keep = i < j
        i, j = i[keep][:maxPairs], j[keep][:maxPairs]
    d = np.array([_pairDistances(x[[a]], x[[b]], distance, radiusUnit)[0, 0] for a, b in zip(i, j)]) \
        if distance == "greatCircle" else np.linalg.norm(x[i] - x[j], axis=1)
    return d, 0.5 * (y[i] - y[j]) ** 2
