"""Correlation kernels for Kriging / Gaussian-process models.

Kernels compose with ``+`` and ``*``.

Every kernel works on (already standardized) inputs and provides
    matrix(xa, xb, p)           correlation matrix
    gradients(x, p)             [dK/dp_i] on the training set (analytic)
    dx(xa, xb, p, kx)           dK(xa, xb)/dxa_kx for prediction gradients
with hyperparameters p in log space and box bounds.

Stationary kernels are functions of the scaled squared distance
``r2 = sum_k invL2_k (xa_k - xb_k)^2``. The inverse squared lengthscales
invL2 come from one of three parameterizations:
    isotropic   one lengthscale
    ARD         one lengthscale per input (default)
    PLS         nComp parameters mapped through PLS weights W (nx, nComp):
                invL2_k = sum_l W_kl^2 / l_l^2 (KPLS construction)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.core.Registry import ComponentBase, buildComponent, registry
from pythonLibs.regressionHandler.numerics import SpecialFunctions as _sf


def weightedSqDist(xa: np.ndarray, xb: np.ndarray, c: np.ndarray) -> np.ndarray:
    """sum_k c_k (xa_ik - xb_jk)^2 for c >= 0.

    Accumulated dimension by dimension (exact zeros on coincident points, no
    cancellation); for many inputs the Gram identity (BLAS-3) is used instead.
    """
    c = np.asarray(c, dtype=float)
    if xa.shape[1] <= 16:
        out = np.zeros((xa.shape[0], xb.shape[0]))
        for k in np.flatnonzero(c):
            d = xa[:, k][:, None] - xb[:, k][None, :]
            out += c[k] * d * d
        return out
    s = np.sqrt(c)
    a = xa * s
    b = xb * s
    d2 = np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - 2.0 * (a @ b.T)
    return np.maximum(d2, 0.0)


def signedSqDist(xa: np.ndarray, xb: np.ndarray, c: np.ndarray) -> np.ndarray:
    pos = np.maximum(c, 0.0)
    neg = np.maximum(-c, 0.0)
    out = weightedSqDist(xa, xb, pos) if np.any(pos) else 0.0
    if np.any(neg):
        out = out - weightedSqDist(xa, xb, neg)
    return out


class KernelBase(ComponentBase):
    componentKind = "kernel"

    def __init__(self, **options) -> None:
        super().__init__(**options)
        self.nx: Optional[int] = None

    def setup(self, nx: int, plsWeights: Optional[np.ndarray] = None, unitScale: float = 1.0) -> "KernelBase":
        self.nx = nx
        return self

    def lengthscaleIndices(self) -> list[int]:
        """Indices of log-lengthscale parameters (tied together for the tied-lengthscale start)."""
        return list(range(self.nParams()))

    def nParams(self) -> int:
        raise NotImplementedError

    def initialParams(self) -> np.ndarray:
        raise NotImplementedError

    def bounds(self) -> np.ndarray:
        """(nParams, 2) bounds in log space."""
        raise NotImplementedError

    def paramNames(self) -> list[str]:
        return [f"p{i}" for i in range(self.nParams())]

    def matrix(self, xa, xb, p) -> np.ndarray:
        raise NotImplementedError

    def gradients(self, x, p) -> list[np.ndarray]:
        raise NotImplementedError

    def dx(self, xa, xb, p, kx: int) -> np.ndarray:
        raise NotImplementedError

    def trainingCache(self, x):
        """Precomputed data reused by every likelihood evaluation on the same inputs."""
        return None

    def matrixAndGradients(self, x, p, cache=None) -> tuple[np.ndarray, list[np.ndarray]]:
        """Training correlation matrix and its hyperparameter gradients in one pass."""
        return self.matrix(x, x, p), self.gradients(x, p)

    def __add__(self, other: "KernelBase") -> "KernelBase":
        return SumKernel(kernels=[self, other])

    def __mul__(self, other: "KernelBase") -> "KernelBase":
        return ProductKernel(kernels=[self, other])


EARTH_RADIUS = {"km": 6378.388, "miles": 3963.34}   # radii used by the great-circle distance


class StationaryKernel(KernelBase):
    """Kernel k(r2) of a squared, scaled distance r2 between inputs.

    ``distance`` selects how r2 is formed:
        "scaled"       sum_k (dx_k / l_k)^2 with ARD, isotropic or PLS lengthscales
        "anisotropic"  d^T L L^T d with an estimated lower-triangular L (full
                       geometric anisotropy); with a fixed ``V`` matrix the inputs
                       are first transformed x -> x V^-T and then scaled
        "greatCircle"  (great-circle distance / l)^2 for (longitude, latitude)
                       inputs in degrees; l is in ``radiusUnit`` units
    Subclasses may add shape parameters (e.g. the Matern smoothness), which
    follow the distance parameters in the parameter vector.
    """

    def _declareOptions(self, declare) -> None:
        declare("ard", True, types=bool, desc="One lengthscale per input (False: isotropic)")
        declare("lengthscale0", 1.0, types=(int, float), lower=0.0, desc="Initial lengthscale")
        declare("lengthscaleBounds", [1e-3, 1e3], types=list, desc="Lengthscale bounds")
        declare("distance", "scaled", values=("scaled", "anisotropic", "greatCircle"), desc="Distance definition")
        declare("V", None, types=list, desc="Fixed anisotropy matrix (inputs transformed by V^-T), nx x nx")
        declare("radiusUnit", "km", values=tuple(EARTH_RADIUS), desc="Earth radius unit for greatCircle")

    def setup(self, nx: int, plsWeights: Optional[np.ndarray] = None, unitScale: float = 1.0) -> "KernelBase":
        self.nx = nx
        self._pls = None if plsWeights is None else np.asarray(plsWeights, dtype=float)
        self._logUnit = float(np.log(unitScale)) if unitScale > 0 else 0.0
        dist = self.options["distance"]
        if dist == "greatCircle" and nx != 2:
            raise ValueError("greatCircle distance needs exactly two inputs (longitude, latitude in degrees)")
        v = self.options["V"]
        self._vInvT = None if v is None else np.linalg.inv(np.asarray(v, dtype=float)).T
        if self._vInvT is not None and self._vInvT.shape != (nx, nx):
            raise ValueError("V must be an nx x nx matrix")
        self._tril = np.tril_indices(nx)
        return self

    # -- parameter bookkeeping ----------------------------------------------------
    def _nShape(self) -> int:
        return 0

    def _shapeInitial(self) -> np.ndarray:
        return np.zeros(0)

    def _shapeBounds(self) -> np.ndarray:
        return np.zeros((0, 2))

    def _shapeNames(self) -> list[str]:
        return []

    def _nDistance(self) -> int:
        dist = self.options["distance"]
        if dist == "greatCircle":
            return 1
        if dist == "anisotropic" and self._vInvT is None:
            return self.nx * (self.nx + 1) // 2
        if self._pls is not None:
            return self._pls.shape[1]
        return self.nx if self.options["ard"] else 1

    def nParams(self) -> int:
        return self._nDistance() + self._nShape()

    def _isCholesky(self) -> bool:
        return self.options["distance"] == "anisotropic" and self._vInvT is None

    def _cholDiagMask(self) -> np.ndarray:
        rows, cols = self._tril
        return rows == cols

    def lengthscaleIndices(self) -> list[int]:
        return [] if self._isCholesky() else list(range(self._nDistance()))

    def initialParams(self) -> np.ndarray:
        l0 = np.log(self.options["lengthscale0"]) + self._logUnit
        if self._isCholesky():
            p = np.zeros(self._nDistance())
            p[self._cholDiagMask()] = -l0          # log of L_ii = 1 / lengthscale
        else:
            p = np.full(self._nDistance(), l0)
        return np.concatenate([p, self._shapeInitial()])

    def bounds(self) -> np.ndarray:
        lo, hi = self.options["lengthscaleBounds"]
        lo, hi = np.log(lo) + self._logUnit, np.log(hi) + self._logUnit
        if self._isCholesky():
            b = np.tile([-50.0, 50.0], (self._nDistance(), 1))
            b[self._cholDiagMask()] = [-hi, -lo]
        else:
            b = np.tile([lo, hi], (self._nDistance(), 1))
        return np.vstack([b, self._shapeBounds()])

    def paramNames(self) -> list[str]:
        if self._isCholesky():
            rows, cols = self._tril
            names = [f"log_L{i}{j}" if i == j else f"L{i}{j}" for i, j in zip(rows, cols)]
        elif self.options["distance"] == "greatCircle":
            names = [f"log_lengthscale_{self.options['radiusUnit']}"]
        else:
            tag = "plsLengthscale" if self._pls is not None else "lengthscale"
            names = [f"log_{tag}{i}" for i in range(self._nDistance())]
        return names + self._shapeNames()

    def _split(self, p):
        p = np.asarray(p, dtype=float)
        nd = self._nDistance()
        return p[:nd], p[nd:]

    # -- scaled distance (ARD / isotropic / PLS) -----------------------------------
    def invL2(self, p) -> np.ndarray:
        pd = np.asarray(p, dtype=float)[: self._nDistance()]
        e = np.exp(-2.0 * pd)
        if self._pls is not None:
            return (self._pls ** 2) @ e
        return e if self.options["ard"] else np.full(self.nx, e[0])

    def _dInvL2(self, pd) -> np.ndarray:
        e = np.exp(-2.0 * np.asarray(pd, dtype=float))
        if self._pls is not None:
            return (-2.0 * e)[:, None] * (self._pls ** 2).T
        if self.options["ard"]:
            return np.diag(-2.0 * e)
        return np.full((1, self.nx), -2.0 * e[0])

    def _cholesky(self, pd) -> np.ndarray:
        l = np.zeros((self.nx, self.nx))
        vals = np.array(pd, dtype=float)
        diag = self._cholDiagMask()
        vals[diag] = np.exp(vals[diag])
        l[self._tril] = vals
        return l

    def _transform(self, x):
        return x if self._vInvT is None else x @ self._vInvT

    # -- great-circle distance --------------------------------------------------------
    def _haversine(self, xa, xb):
        rad = np.pi / 180.0
        lon1, lat1 = xa[:, [0]] * rad, xa[:, [1]] * rad
        lon2, lat2 = xb[:, 0][None, :] * rad, xb[:, 1][None, :] * rad
        h = np.sin(0.5 * (lat1 - lat2)) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * (lon1 - lon2)) ** 2
        return np.clip(h, 0.0, 1.0), (lon1, lat1, lon2, lat2)

    def greatCircleDistance(self, xa, xb) -> np.ndarray:
        h, _ = self._haversine(xa, xb)
        return 2.0 * EARTH_RADIUS[self.options["radiusUnit"]] * np.arcsin(np.sqrt(h))

    # -- r2 and its derivatives -------------------------------------------------------
    def _r2(self, xa, xb, pd) -> np.ndarray:
        dist = self.options["distance"]
        if dist == "greatCircle":
            return (self.greatCircleDistance(xa, xb) * np.exp(-pd[0])) ** 2
        if self._isCholesky():
            l = self._cholesky(pd)
            return weightedSqDist(xa @ l, xb @ l, np.ones(self.nx))
        return weightedSqDist(self._transform(xa), self._transform(xb), self.invL2(pd))

    def _r2WithGrads(self, x, pd, cache):
        """Training r2 and [d r2 / d pd_i]."""
        dist = self.options["distance"]
        if dist == "greatCircle":
            r2 = self._r2(x, x, pd)
            return r2, [-2.0 * r2]
        if self._isCholesky():
            diff = cache if cache is not None and cache.ndim == 3 and cache.shape[0] == self.nx and \
                self._cacheSigned else np.stack([x[:, k][:, None] - x[:, k][None, :] for k in range(self.nx)])
            l = self._cholesky(pd)
            u = np.tensordot(l.T, diff, axes=1)                  # u_j = sum_i L_ij d_i
            r2 = np.sum(u * u, axis=0)
            rows, cols = self._tril
            grads = []
            for a, b in zip(rows, cols):
                g = 2.0 * u[b] * diff[a]
                grads.append(g * l[a, b] if a == b else g)          # log-diagonal chain rule
            return r2, grads
        xt = self._transform(x)
        inv = self.invL2(pd)
        dInv = self._dInvL2(pd)
        if cache is not None and not self._cacheSigned:
            r2 = np.tensordot(inv, cache, axes=1)
            grads = []
            for i in range(dInv.shape[0]):
                nz = np.flatnonzero(dInv[i])
                grads.append(dInv[i, nz[0]] * cache[nz[0]] if nz.size == 1 else np.tensordot(dInv[i], cache, axes=1))
            return r2, grads
        r2 = weightedSqDist(xt, xt, inv)
        grads = []
        for i in range(dInv.shape[0]):
            nz = np.flatnonzero(dInv[i])
            if nz.size == 1:
                k = nz[0]
                d = xt[:, k][:, None] - xt[:, k][None, :]
                grads.append(dInv[i, k] * d * d)
            else:
                grads.append(signedSqDist(xt, xt, dInv[i]))
        return r2, grads

    def _dr2dx(self, xa, xb, pd, kx: int) -> np.ndarray:
        dist = self.options["distance"]
        if dist == "greatCircle":
            h, (lon1, lat1, lon2, lat2) = self._haversine(xa, xb)
            rad = np.pi / 180.0
            radius = EARTH_RADIUS[self.options["radiusUnit"]]
            scale = np.exp(-2.0 * pd[0]) * 4.0 * radius * radius
            sq = np.sqrt(h)
            with np.errstate(divide="ignore", invalid="ignore"):
                dr2dh = scale * np.where(sq > 1e-12, np.arcsin(sq) / (sq * np.sqrt(np.maximum(1.0 - h, 1e-300))), 1.0)
            if kx == 0:
                dh = np.cos(lat1) * np.cos(lat2) * 0.5 * np.sin(lon1 - lon2)
            else:
                dh = 0.5 * np.sin(lat1 - lat2) - np.sin(lat1) * np.cos(lat2) * np.sin(0.5 * (lon1 - lon2)) ** 2
            return dr2dh * dh * rad
        if self._isCholesky():
            m = self._cholesky(pd) @ self._cholesky(pd).T
            diff = np.stack([xa[:, k][:, None] - xb[:, k][None, :] for k in range(self.nx)])
            return 2.0 * np.tensordot(m[kx], diff, axes=1)
        inv = self.invL2(pd)
        xta, xtb = self._transform(xa), self._transform(xb)
        dr2dxt = [2.0 * inv[j] * (xta[:, j][:, None] - xtb[:, j][None, :]) for j in range(self.nx)]
        if self._vInvT is None:
            return dr2dxt[kx]
        return sum(dr2dxt[j] * self._vInvT[kx, j] for j in range(self.nx))

    # -- kernel profile -------------------------------------------------------------
    def f(self, r2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fPrime(self, r2: np.ndarray) -> np.ndarray:
        """dk / d(r2); where singular at r2 = 0 it is multiplied by zero distances, return 0 there."""
        raise NotImplementedError

    def _f(self, r2, ps):
        return self.f(r2)

    def _fPrime(self, r2, ps):
        return self.fPrime(r2)

    def _shapeGradients(self, r2, ps) -> list[np.ndarray]:
        """d k / d ps_i by central differences (shape parameters are few)."""
        out = []
        for i in range(ps.size):
            h = 1e-5 * max(1.0, abs(ps[i]))
            up, dn = ps.copy(), ps.copy()
            up[i] += h
            dn[i] -= h
            out.append((self._f(r2, up) - self._f(r2, dn)) / (2.0 * h))
        return out

    # -- public ---------------------------------------------------------------
    def matrix(self, xa, xb, p) -> np.ndarray:
        pd, ps = self._split(p)
        return self._f(self._r2(xa, xb, pd), ps)

    def gradients(self, x, p) -> list[np.ndarray]:
        return self.matrixAndGradients(x, p, None)[1]

    _CACHE_LIMIT = 60_000_000  # elements of the (nx, n, n) difference cache

    @property
    def _cacheSigned(self) -> bool:
        return self._isCholesky()

    def trainingCache(self, x):
        n, nx = x.shape
        if n * n * nx > self._CACHE_LIMIT or self.options["distance"] == "greatCircle":
            return None
        if self._isCholesky():
            return np.stack([x[:, k][:, None] - x[:, k][None, :] for k in range(nx)])
        xt = self._transform(x)
        return np.stack([(xt[:, k][:, None] - xt[:, k][None, :]) ** 2 for k in range(nx)])

    def matrixAndGradients(self, x, p, cache=None):
        pd, ps = self._split(p)
        r2, dr2 = self._r2WithGrads(x, pd, cache)
        k = self._f(r2, ps)
        fp = self._fPrime(r2, ps)
        grads = [fp * g for g in dr2] + self._shapeGradients(r2, ps)
        return k, grads

    def dx(self, xa, xb, p, kx: int) -> np.ndarray:
        pd, ps = self._split(p)
        return self._fPrime(self._r2(xa, xb, pd), ps) * self._dr2dx(xa, xb, pd, kx)


@registry("kernel").register("squaredExponential")
class SquaredExponential(StationaryKernel):
    """exp(-r2 / 2) (Gaussian / RBF)."""

    def f(self, r2):
        return np.exp(-0.5 * r2)

    def fPrime(self, r2):
        return -0.5 * np.exp(-0.5 * r2)


@registry("kernel").register("absoluteExponential")
class AbsoluteExponential(StationaryKernel):
    """exp(-r) (Matern 1/2, Ornstein-Uhlenbeck)."""

    def f(self, r2):
        return np.exp(-np.sqrt(r2))

    def fPrime(self, r2):
        r = np.sqrt(r2)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r > 0.0, -np.exp(-r) / (2.0 * np.where(r > 0.0, r, 1.0)), 0.0)


@registry("kernel").register("matern32")
class Matern32(StationaryKernel):

    def f(self, r2):
        s = np.sqrt(3.0 * r2)
        return (1.0 + s) * np.exp(-s)

    def fPrime(self, r2):
        return -1.5 * np.exp(-np.sqrt(3.0 * r2))


@registry("kernel").register("matern52")
class Matern52(StationaryKernel):

    def f(self, r2):
        s = np.sqrt(5.0 * r2)
        return (1.0 + s + s * s / 3.0) * np.exp(-s)

    def fPrime(self, r2):
        s = np.sqrt(5.0 * r2)
        return -(5.0 / 6.0) * (1.0 + s) * np.exp(-s)


@registry("kernel").register("powerExponential")
class PowerExponential(StationaryKernel):
    """exp(-(r2)^(power/2)); power in (0, 2] on the scaled distance."""

    def _declareOptions(self, declare) -> None:
        super()._declareOptions(declare)
        declare("power", 1.9, types=(int, float), lower=1e-6, upper=2.0, desc="Exponent of the distance")

    def f(self, r2):
        return np.exp(-(r2 ** (0.5 * self.options["power"])))

    def fPrime(self, r2):
        q = 0.5 * self.options["power"]
        with np.errstate(divide="ignore", invalid="ignore"):
            val = -q * r2 ** (q - 1.0) * np.exp(-(r2 ** q))
        return np.where(r2 > 0.0, val, -1.0 if q == 1.0 else 0.0)


@registry("kernel").register("rationalQuadratic")
class RationalQuadratic(StationaryKernel):
    """(1 + r2 / (2 alpha))^-alpha; a scale mixture of squared exponentials."""

    def _declareOptions(self, declare) -> None:
        super()._declareOptions(declare)
        declare("alpha", 1.0, types=(int, float), lower=1e-6, desc="Shape parameter (large -> squared exponential)")

    def f(self, r2):
        a = self.options["alpha"]
        return (1.0 + r2 / (2.0 * a)) ** (-a)

    def fPrime(self, r2):
        a = self.options["alpha"]
        return -0.5 * (1.0 + r2 / (2.0 * a)) ** (-a - 1.0)


@registry("kernel").register("matern")
class Matern(StationaryKernel):
    """Matern correlation with general smoothness nu (fixed or estimated).

        k = 2^(1-nu) / Gamma(nu) * z^nu * K_nu(z)
        z = sqrt(2 nu) * r   (parameterization="standard": nu = 1/2, 3/2, 5/2 equal the
                               absoluteExponential / matern32 / matern52 kernels)
        z = r                (parameterization="range": the lengthscale is the range parameter)
    """

    def _declareOptions(self, declare) -> None:
        super()._declareOptions(declare)
        declare("nu", 1.5, values=("estimate",), types=(int, float), lower=1e-3,
                desc="Smoothness; 'estimate' adds log(nu) to the hyperparameters")
        declare("nu0", 1.5, types=(int, float), lower=1e-3, desc="Initial smoothness when estimated")
        declare("nuBounds", [0.25, 8.0], types=list, desc="Smoothness bounds when estimated")
        declare("parameterization", "standard", values=("standard", "range"), desc="Distance scaling convention")

    def _nShape(self) -> int:
        return 1 if self.options["nu"] == "estimate" else 0

    def _shapeInitial(self):
        return np.log([self.options["nu0"]]) if self._nShape() else np.zeros(0)

    def _shapeBounds(self):
        return np.log([self.options["nuBounds"]]) if self._nShape() else np.zeros((0, 2))

    def _shapeNames(self):
        return ["log_nu"] if self._nShape() else []

    def _nu(self, ps) -> float:
        return float(np.exp(ps[0])) if ps.size else float(self.options["nu"])

    def _scale2(self, nu: float) -> float:
        return 2.0 * nu if self.options["parameterization"] == "standard" else 1.0

    def _f(self, r2, ps):
        nu = self._nu(ps)
        z = np.sqrt(self._scale2(nu) * np.asarray(r2, dtype=float))
        out = np.ones_like(z)
        pos = z > 0
        if np.any(pos):
            zp = z[pos]
            logc = (1.0 - nu) * np.log(2.0) - _sf.gammaln(nu)
            with np.errstate(under="ignore"):
                out[pos] = np.exp(logc + nu * np.log(zp)) * _sf.besselK(nu, zp)
        return out

    def _fPrime(self, r2, ps):
        # dk/dr2 = -c * (s2 / 2) * z^(nu-1) K_(nu-1)(z);  at z = 0 the limit is -s2 / (4 (nu - 1)) for nu > 1
        nu = self._nu(ps)
        s2 = self._scale2(nu)
        z = np.sqrt(s2 * np.asarray(r2, dtype=float))
        out = np.full(z.shape, -s2 / (4.0 * (nu - 1.0)) if nu > 1.0 else 0.0)
        pos = z > 0
        if np.any(pos):
            zp = z[pos]
            logc = (1.0 - nu) * np.log(2.0) - _sf.gammaln(nu)
            with np.errstate(under="ignore"):
                out[pos] = -0.5 * s2 * np.exp(logc + (nu - 1.0) * np.log(zp)) * _sf.besselK(abs(nu - 1.0), zp)
        return out

    def rangeParameters(self, p) -> np.ndarray:
        """Lengthscales expressed as range parameters a (distance divisor inside z = d / a)."""
        pd, ps = self._split(p)
        ell = np.exp(pd) if self.options["distance"] != "anisotropic" else np.full(1, np.nan)
        return ell / np.sqrt(self._scale2(self._nu(ps)))


@registry("kernel").register("wendland")
class Wendland(StationaryKernel):
    """Compactly supported Wendland correlation (zero beyond one lengthscale).

    phi_{d,k}(r), r = distance / lengthscale, with l = floor(d/2) + k + 1:
        k = 0: (1-r)^l
        k = 1: (1-r)^(l+1) ((l+1) r + 1)
        k = 2: (1-r)^(l+2) ((l^2+4l+3) r^2 + (3l+6) r + 3) / 3
    positive definite in up to ``dimension`` inputs (default: number of inputs).
    """

    def _declareOptions(self, declare) -> None:
        super()._declareOptions(declare)
        declare("k", 2, values=(0, 1, 2), desc="Smoothness order (2k continuous derivatives at 0)")
        declare("dimension", None, types=int, lower=1, desc="Dimension of positive definiteness (default nx)")

    def _l(self) -> int:
        d = self.options["dimension"] or self.nx
        return d // 2 + self.options["k"] + 1

    def f(self, r2):
        r = np.sqrt(np.asarray(r2, dtype=float))
        t = np.clip(1.0 - r, 0.0, None)
        l, k = self._l(), self.options["k"]
        if k == 0:
            return t ** l
        if k == 1:
            return t ** (l + 1) * ((l + 1) * r + 1.0)
        a, b = l * l + 4 * l + 3, 3 * l + 6
        return t ** (l + 2) * (a * r * r + b * r + 3.0) / 3.0

    def fPrime(self, r2):
        r = np.sqrt(np.asarray(r2, dtype=float))
        t = np.clip(1.0 - r, 0.0, None)
        l, k = self._l(), self.options["k"]
        if k == 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(r > 0, -l * t ** (l - 1) / (2.0 * np.where(r > 0, r, 1.0)), 0.0)
        if k == 1:
            return -0.5 * (l + 1) * (l + 2) * t ** l
        m = l + 2
        a, b = l * l + 4 * l + 3, 3 * l + 6
        nOverR = (2 * a - b * (m + 1)) - a * (m + 2) * r      # N(r) / r, N = -m P + (1-r) P'
        return t ** (m - 1) * nOverR / 6.0


@registry("kernel").register("periodic")
class Periodic(KernelBase):
    """exp(-sum_k 2 sin^2(pi d_k / P_k) / l_k^2), ARD lengthscales and periods."""

    def _declareOptions(self, declare) -> None:
        declare("lengthscale0", 1.0, types=(int, float), lower=0.0, desc="Initial lengthscale")
        declare("period0", 1.0, types=(int, float), lower=0.0, desc="Initial period (standardized units)")
        declare("lengthscaleBounds", [1e-3, 1e3], types=list, desc="Lengthscale bounds")
        declare("periodBounds", [1e-2, 1e2], types=list, desc="Period bounds")

    def setup(self, nx: int, plsWeights: Optional[np.ndarray] = None, unitScale: float = 1.0) -> "KernelBase":
        self.nx = nx
        self._logUnit = float(np.log(unitScale)) if unitScale > 0 else 0.0
        return self

    def nParams(self) -> int:
        return 2 * self.nx

    def lengthscaleIndices(self) -> list[int]:
        return []

    def initialParams(self) -> np.ndarray:
        u = getattr(self, "_logUnit", 0.0)
        return np.concatenate([np.full(self.nx, np.log(self.options["lengthscale0"])),
                               np.full(self.nx, np.log(self.options["period0"]) + u)])

    def bounds(self) -> np.ndarray:
        u = getattr(self, "_logUnit", 0.0)
        lb, ub = self.options["lengthscaleBounds"]
        pb, pu = self.options["periodBounds"]
        return np.vstack([np.tile([np.log(lb), np.log(ub)], (self.nx, 1)),
                          np.tile([np.log(pb) + u, np.log(pu) + u], (self.nx, 1))])

    def paramNames(self) -> list[str]:
        return [f"log_lengthscale{i}" for i in range(self.nx)] + [f"log_period{i}" for i in range(self.nx)]

    def _parts(self, xa, xb, p):
        ell2 = np.exp(2.0 * p[: self.nx])
        per = np.exp(p[self.nx:])
        u = [np.pi * (xa[:, k][:, None] - xb[:, k][None, :]) / per[k] for k in range(self.nx)]
        return ell2, per, u

    def matrix(self, xa, xb, p) -> np.ndarray:
        ell2, _, u = self._parts(xa, xb, np.asarray(p, float))
        return np.exp(-sum(2.0 * np.sin(uk) ** 2 / ell2[k] for k, uk in enumerate(u)))

    def gradients(self, x, p) -> list[np.ndarray]:
        p = np.asarray(p, float)
        ell2, _, u = self._parts(x, x, p)
        k = np.exp(-sum(2.0 * np.sin(uk) ** 2 / ell2[i] for i, uk in enumerate(u)))
        gl = [k * 4.0 * np.sin(uk) ** 2 / ell2[i] for i, uk in enumerate(u)]
        gp = [k * 2.0 * uk * np.sin(2.0 * uk) / ell2[i] for i, uk in enumerate(u)]
        return gl + gp

    def dx(self, xa, xb, p, kx: int) -> np.ndarray:
        p = np.asarray(p, float)
        ell2, per, u = self._parts(xa, xb, p)
        k = np.exp(-sum(2.0 * np.sin(uk) ** 2 / ell2[i] for i, uk in enumerate(u)))
        return k * (-2.0 / ell2[kx]) * np.sin(2.0 * u[kx]) * np.pi / per[kx]


class _CompositeKernel(KernelBase):

    def _declareOptions(self, declare) -> None:
        declare("kernels", [], types=list, desc="Child kernel specs or instances")

    def _children(self) -> list[KernelBase]:
        if not hasattr(self, "_built"):
            self._built = [buildComponent("kernel", k) for k in self.options["kernels"]]
            self.options["kernels"] = self._built
        return self._built

    def setup(self, nx: int, plsWeights: Optional[np.ndarray] = None, unitScale: float = 1.0) -> "KernelBase":
        self.nx = nx
        for k in self._children():
            k.setup(nx, plsWeights, unitScale)
        return self

    def lengthscaleIndices(self) -> list[int]:
        out = []
        for k, sl in zip(self._children(), self._slices()):
            out.extend(sl.start + i for i in k.lengthscaleIndices())
        return out

    def _slices(self):
        out, start = [], 0
        for k in self._children():
            out.append(slice(start, start + k.nParams()))
            start += k.nParams()
        return out

    def nParams(self) -> int:
        return sum(k.nParams() for k in self._children())

    def initialParams(self) -> np.ndarray:
        return np.concatenate([k.initialParams() for k in self._children()])

    def bounds(self) -> np.ndarray:
        return np.vstack([k.bounds() for k in self._children()])

    def paramNames(self) -> list[str]:
        return [f"k{i}.{n}" for i, k in enumerate(self._children()) for n in k.paramNames()]

    def trainingCache(self, x):
        return [k.trainingCache(x) for k in self._children()]


@registry("kernel").register("sum")
class SumKernel(_CompositeKernel):
    """Average of child correlations (keeps a unit diagonal)."""

    def matrix(self, xa, xb, p):
        ks = self._children()
        return sum(k.matrix(xa, xb, p[s]) for k, s in zip(ks, self._slices())) / len(ks)

    def gradients(self, x, p):
        ks = self._children()
        return [g / len(ks) for k, s in zip(ks, self._slices()) for g in k.gradients(x, p[s])]

    def matrixAndGradients(self, x, p, cache=None):
        ks = self._children()
        cache = cache or [None] * len(ks)
        mats, grads = [], []
        for k, s, c in zip(ks, self._slices(), cache):
            m, g = k.matrixAndGradients(x, p[s], c)
            mats.append(m)
            grads.extend(gi / len(ks) for gi in g)
        return sum(mats) / len(ks), grads

    def dx(self, xa, xb, p, kx):
        ks = self._children()
        return sum(k.dx(xa, xb, p[s], kx) for k, s in zip(ks, self._slices())) / len(ks)


@registry("kernel").register("product")
class ProductKernel(_CompositeKernel):

    def matrix(self, xa, xb, p):
        out = 1.0
        for k, s in zip(self._children(), self._slices()):
            out = out * k.matrix(xa, xb, p[s])
        return out

    def gradients(self, x, p):
        ks, sl = self._children(), self._slices()
        mats = [k.matrix(x, x, p[s]) for k, s in zip(ks, sl)]
        out = []
        for i, (k, s) in enumerate(zip(ks, sl)):
            others = 1.0
            for j, m in enumerate(mats):
                if j != i:
                    others = others * m
            out.extend(g * others for g in k.gradients(x, p[s]))
        return out

    def matrixAndGradients(self, x, p, cache=None):
        ks, sl = self._children(), self._slices()
        cache = cache or [None] * len(ks)
        parts = [k.matrixAndGradients(x, p[s], c) for k, s, c in zip(ks, sl, cache)]
        total = 1.0
        for m, _ in parts:
            total = total * m
        grads = []
        for i, (m, g) in enumerate(parts):
            others = 1.0
            for j, (mj, _) in enumerate(parts):
                if j != i:
                    others = others * mj
            grads.extend(gi * others for gi in g)
        return total, grads

    def dx(self, xa, xb, p, kx):
        ks, sl = self._children(), self._slices()
        mats = [k.matrix(xa, xb, p[s]) for k, s in zip(ks, sl)]
        total = 0.0
        for i, (k, s) in enumerate(zip(ks, sl)):
            term = k.dx(xa, xb, p[s], kx)
            for j, m in enumerate(mats):
                if j != i:
                    term = term * m
            total = total + term
        return total


KERNEL_ALIASES = {"rbf": "squaredExponential", "gaussian": "squaredExponential", "exponential": "absoluteExponential"}


def buildKernel(spec) -> KernelBase:
    if isinstance(spec, str):
        spec = KERNEL_ALIASES.get(spec, spec)
    elif isinstance(spec, dict) and spec.get("type") in KERNEL_ALIASES:
        spec = {**spec, "type": KERNEL_ALIASES[spec["type"]]}
    return buildComponent("kernel", spec)
