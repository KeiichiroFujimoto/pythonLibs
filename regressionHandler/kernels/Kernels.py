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

    def lengthscaleColumns(self) -> list[int]:
        """Input column (of this kernel's inputs) scaling each entry of ``lengthscaleIndices``."""
        return [0] * len(self.lengthscaleIndices())

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

    def batchMatrix(self, xa, xb, p) -> np.ndarray:
        """Stacked correlation matrices: xa (B, ma, d), xb (B, mb, d) -> (B, ma, mb)."""
        return np.stack([self.matrix(a, b, p) for a, b in zip(xa, xb)])

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

    def lengthscaleColumns(self) -> list[int]:
        n = len(self.lengthscaleIndices())
        if self._pls is None and self.options["ard"] and self.options["distance"] == "scaled" and n == self.nx:
            return list(range(self.nx))
        return [0] * n

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

    def batchMatrix(self, xa, xb, p) -> np.ndarray:
        if self.options["distance"] == "greatCircle":
            return super().batchMatrix(xa, xb, p)
        pd, ps = self._split(p)
        if self._isCholesky():
            l = self._cholesky(pd)
            ta, tb, w = xa @ l, xb @ l, np.ones(self.nx)
        else:
            ta, tb, w = self._transform(xa), self._transform(xb), self.invL2(pd)
        r2 = np.zeros((xa.shape[0], xa.shape[1], xb.shape[1]))
        for k in np.flatnonzero(w):
            d = ta[:, :, k][:, :, None] - tb[:, :, k][:, None, :]
            r2 += w[k] * d * d
        return self._f(r2, ps)

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

    def fPrime2(self, r2, ps=None):
        return 0.25 * np.exp(-0.5 * r2)


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

    def fPrime2(self, r2, ps=None):
        return (25.0 / 12.0) * np.exp(-np.sqrt(5.0 * r2))


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

    def fPrime2(self, r2, ps):
        """d^2 k / d(r2)^2 = c (s2^2 / 4) z^(nu-2) K_(nu-2)(z) for r2 > 0 (twice differentiable for nu > 1).

        Where it diverges (r2 = 0, nu <= 2) it only multiplies zero distances in
        second derivatives of the kernel; 0 is returned there.
        """
        nu = self._nu(ps)
        s2 = self._scale2(nu)
        z = np.sqrt(s2 * np.asarray(r2, dtype=float))
        out = np.zeros_like(z)
        if nu > 2.0:
            out[:] = s2 * s2 / (16.0 * (nu - 1.0) * (nu - 2.0))       # limit at z = 0
        pos = z > 0
        if np.any(pos):
            zp = z[pos]
            logc = (1.0 - nu) * np.log(2.0) - _sf.gammaln(nu)
            with np.errstate(under="ignore"):
                out[pos] = 0.25 * s2 * s2 * np.exp(logc + (nu - 2.0) * np.log(zp)) * _sf.besselK(abs(nu - 2.0), zp)
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


@registry("kernel").register("spherical")
class Spherical(StationaryKernel):
    """Spherical correlation 1 - 1.5 r + 0.5 r^3 for r < 1, zero beyond (positive definite up to 3-D)."""

    def f(self, r2):
        r = np.sqrt(np.asarray(r2, dtype=float))
        return np.where(r < 1.0, 1.0 - 1.5 * r + 0.5 * r ** 3, 0.0)

    def fPrime(self, r2):
        # dk/dr2 = (-1.5 + 1.5 r^2) / (2 r); the r -> 0 singularity is multiplied by zero distances
        r = np.sqrt(np.asarray(r2, dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((r > 0) & (r < 1.0), 0.75 * (r * r - 1.0) / np.where(r > 0, r, 1.0), 0.0)


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


@registry("kernel").register("gneiting")
class Gneiting(KernelBase):
    """Non-separable space-time correlation (Gneiting 2002, eq. 14)::

        psi(u) = (|u| / l_t)^(2 alpha) + 1
        k(h, u) = psi^-tau * exp(-(||h|| / l_s)^(2 gamma) / psi^(beta gamma)),   tau = d / 2

    h is the spatial lag (``spaceColumns``), u the time lag (``timeColumn``);
    beta in [0, 1] sets the space-time interaction (0: separable). Estimated
    parameters: log l_s, log l_t and, with beta="estimate", logit(beta).
    """

    def _declareOptions(self, declare) -> None:
        declare("spaceColumns", None, types=list, desc="Spatial input columns (default: all but the time column)")
        declare("timeColumn", -1, types=int, desc="Time input column")
        declare("alpha", 1.0, types=(int, float), lower=1e-6, upper=1.0, desc="Temporal smoothness, 0 < alpha <= 1")
        declare("gamma", 0.5, types=(int, float), lower=1e-6, upper=1.0, desc="Spatial smoothness, 0 < gamma <= 1")
        declare("beta", "estimate", values=("estimate",), types=(int, float), lower=0.0, upper=1.0,
                desc="Space-time interaction in [0, 1] or 'estimate'")
        declare("lengthscale0", 1.0, types=(int, float), lower=0.0, desc="Initial space / time lengthscale")
        declare("lengthscaleBounds", [1e-3, 1e3], types=list, desc="Lengthscale bounds")

    def setup(self, nx: int, plsWeights: Optional[np.ndarray] = None, unitScale: float = 1.0) -> "KernelBase":
        if nx < 2:
            raise ValueError("gneiting needs at least one space and one time input")
        self.nx = nx
        self._t = self.options["timeColumn"] % nx
        sc = self.options["spaceColumns"]
        self._s = np.array([j for j in range(nx) if j != self._t]) if sc is None else np.asarray(sc, dtype=int)
        self._logUnit = float(np.log(unitScale)) if unitScale > 0 else 0.0
        return self

    def _estBeta(self) -> bool:
        return self.options["beta"] == "estimate"

    def nParams(self) -> int:
        return 2 + self._estBeta()

    def lengthscaleIndices(self) -> list[int]:
        return [0, 1]

    def lengthscaleColumns(self) -> list[int]:
        return [int(self._s[0]), int(self._t)]

    def initialParams(self) -> np.ndarray:
        l0 = np.log(self.options["lengthscale0"]) + self._logUnit
        return np.array([l0, l0] + ([0.0] if self._estBeta() else []))

    def bounds(self) -> np.ndarray:
        lo, hi = np.log(self.options["lengthscaleBounds"]) + self._logUnit
        b = [[lo, hi], [lo, hi]] + ([[-8.0, 8.0]] if self._estBeta() else [])
        return np.array(b)

    def paramNames(self) -> list[str]:
        return ["log_lengthscaleSpace", "log_lengthscaleTime"] + (["logit_beta"] if self._estBeta() else [])

    def _beta(self, p) -> float:
        return float(1.0 / (1.0 + np.exp(-p[2]))) if self._estBeta() else float(self.options["beta"])

    def matrix(self, xa, xb, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        a, g = float(self.options["alpha"]), float(self.options["gamma"])
        d = self._s.size
        h2 = weightedSqDist(xa[:, self._s], xb[:, self._s], np.full(d, np.exp(-2.0 * p[0])))
        u = np.abs(xa[:, self._t][:, None] - xb[:, self._t][None, :]) * np.exp(-p[1])
        psi = u ** (2.0 * a) + 1.0
        beta = self._beta(p)
        return psi ** (-0.5 * d) * np.exp(-(h2 ** g) / psi ** (beta * g))

    def gradients(self, x, p) -> list[np.ndarray]:
        p = np.asarray(p, dtype=float)
        out = []
        for i in range(p.size):
            step = 1e-6 * max(1.0, abs(p[i]))
            up, dn = p.copy(), p.copy()
            up[i] += step
            dn[i] -= step
            out.append((self.matrix(x, x, up) - self.matrix(x, x, dn)) / (2.0 * step))
        return out

    def dx(self, xa, xb, p, kx: int) -> np.ndarray:
        scale = max(1.0, float(np.max(np.abs(xa[:, kx])))) if xa.size else 1.0
        step = 1e-6 * scale
        up, dn = xa.copy(), xa.copy()
        up[:, kx] += step
        dn[:, kx] -= step
        return (self.matrix(up, xb, p) - self.matrix(dn, xb, p)) / (2.0 * step)


class _WrappedKernel(KernelBase):
    """Kernel built around one child kernel; gradients and input derivatives by central differences."""

    def _child(self) -> KernelBase:
        if not hasattr(self, "_built"):
            self._built = buildComponent("kernel", self.options["kernel"])
            self.options["kernel"] = self._built
        return self._built

    def gradients(self, x, p) -> list[np.ndarray]:
        p = np.asarray(p, dtype=float)
        out = []
        for i in range(p.size):
            step = 1e-6 * max(1.0, abs(p[i]))
            up, dn = p.copy(), p.copy()
            up[i] += step
            dn[i] -= step
            out.append((self.matrix(x, x, up) - self.matrix(x, x, dn)) / (2.0 * step))
        return out

    def dx(self, xa, xb, p, kx: int) -> np.ndarray:
        step = 1e-6 * max(1.0, float(np.max(np.abs(xa[:, kx])))) if xa.size else 1e-6
        up, dn = xa.copy(), xa.copy()
        up[:, kx] += step
        dn[:, kx] -= step
        return (self.matrix(up, xb, p) - self.matrix(dn, xb, p)) / (2.0 * step)


@registry("kernel").register("warped")
class WarpedKernel(_WrappedKernel):
    """Stationary kernel on monotonically warped inputs (non-stationarity through input warping).

    Each input is mapped by the sinh-arcsinh transform
        w(u) = sinh(delta * asinh(u) - epsilon)
    (identity at delta = 1, epsilon = 0) with log(delta) and epsilon estimated
    per input, then passed to the child kernel.
    """

    def _declareOptions(self, declare) -> None:
        declare("kernel", "matern52", types=(str, dict, object), desc="Child kernel on the warped inputs")
        declare("warpBounds", [-2.0, 2.0], types=list, desc="Bounds of log(delta) and epsilon")

    def setup(self, nx: int, plsWeights: Optional[np.ndarray] = None, unitScale: float = 1.0) -> "KernelBase":
        self.nx = nx
        self._child().setup(nx, None, unitScale)
        return self

    def nParams(self) -> int:
        return self._child().nParams() + 2 * self.nx

    def lengthscaleIndices(self) -> list[int]:
        return self._child().lengthscaleIndices()

    def lengthscaleColumns(self) -> list[int]:
        return self._child().lengthscaleColumns()

    def initialParams(self) -> np.ndarray:
        return np.concatenate([self._child().initialParams(), np.zeros(2 * self.nx)])

    def bounds(self) -> np.ndarray:
        lo, hi = self.options["warpBounds"]
        return np.vstack([self._child().bounds(), np.tile([lo, hi], (2 * self.nx, 1))])

    def paramNames(self) -> list[str]:
        return self._child().paramNames() + [f"log_warpDelta{i}" for i in range(self.nx)] + \
            [f"warpEpsilon{i}" for i in range(self.nx)]

    def _warp(self, x, p):
        nc = self._child().nParams()
        delta = np.exp(p[nc:nc + self.nx])
        eps = p[nc + self.nx:]
        return np.sinh(delta * np.arcsinh(x) - eps)

    def warp(self, x, p) -> np.ndarray:
        """Warped inputs (for inspection)."""
        return self._warp(np.asarray(x, dtype=float), np.asarray(p, dtype=float))

    def matrix(self, xa, xb, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        nc = self._child().nParams()
        return self._child().matrix(self._warp(xa, p), self._warp(xb, p), p[:nc])

    def batchMatrix(self, xa, xb, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        nc = self._child().nParams()
        return self._child().batchMatrix(self._warp(xa, p), self._warp(xb, p), p[:nc])


@registry("kernel").register("nonstationary")
class NonstationaryKernel(_WrappedKernel):
    """Isotropic kernel with a spatially varying lengthscale (Paciorek & Schervish 2006)::

        k(x, x') = (2 l l' / (l^2 + l'^2))^(d/2) * rho(|x - x'|^2 / ((l^2 + l'^2) / 2))
        log l(x) = c_0 + sum_j c_j phi_j(x)

    rho is the correlation profile of the child stationary kernel. phi:
    "linear" (the inputs) or "rbf" (Gaussian bumps on a grid of ``nCenters``
    centres over the standardized input box [-2, 2]^d).
    """

    def _declareOptions(self, declare) -> None:
        declare("kernel", "matern52", types=(str, dict, object), desc="Stationary child (its correlation profile)")
        declare("basis", "linear", values=("linear", "rbf"), desc="Basis of the log-lengthscale field")
        declare("nCenters", 9, types=int, lower=1, desc="RBF centres (rounded to a grid)")
        declare("lengthscale0", 1.0, types=(int, float), lower=0.0, desc="Initial lengthscale")
        declare("lengthscaleBounds", [1e-3, 1e3], types=list, desc="Bounds of the base lengthscale")
        declare("coefficientBounds", [-3.0, 3.0], types=list, desc="Bounds of the field coefficients")

    def setup(self, nx: int, plsWeights: Optional[np.ndarray] = None, unitScale: float = 1.0) -> "KernelBase":
        self.nx = nx
        child = self._child()
        if not isinstance(child, StationaryKernel):
            raise ValueError("nonstationary needs a stationary child kernel")
        child.options["ard"] = False
        child.setup(nx)
        self._nShapeChild = child.nParams() - 1
        self._logUnit = float(np.log(unitScale)) if unitScale > 0 else 0.0
        if self.options["basis"] == "rbf":
            k = max(2, int(round(self.options["nCenters"] ** (1.0 / nx))))
            axes = [np.linspace(-2.0, 2.0, k)] * nx
            grids = np.meshgrid(*axes, indexing="ij")
            self._centers = np.column_stack([g.ravel() for g in grids])
            self._width = 4.0 / (k - 1)
        return self

    def _nBasis(self) -> int:
        return self.nx if self.options["basis"] == "linear" else self._centers.shape[0]

    def nParams(self) -> int:
        return 1 + self._nBasis() + self._nShapeChild

    def lengthscaleIndices(self) -> list[int]:
        return [0]

    def initialParams(self) -> np.ndarray:
        child = self._child()
        shape = child.initialParams()[1:]
        return np.concatenate([[np.log(self.options["lengthscale0"]) + self._logUnit], np.zeros(self._nBasis()), shape])

    def bounds(self) -> np.ndarray:
        lo, hi = np.log(self.options["lengthscaleBounds"]) + self._logUnit
        clo, chi = self.options["coefficientBounds"]
        return np.vstack([[[lo, hi]], np.tile([clo, chi], (self._nBasis(), 1)), self._child().bounds()[1:]])

    def paramNames(self) -> list[str]:
        return ["log_lengthscale0"] + [f"field{j}" for j in range(self._nBasis())] + self._child().paramNames()[1:]

    def _phi(self, x) -> np.ndarray:
        if self.options["basis"] == "linear":
            return x
        d2 = np.sum((x[..., None, :] - self._centers) ** 2, axis=-1)
        return np.exp(-0.5 * d2 / self._width ** 2)

    def lengthscale(self, x, p) -> np.ndarray:
        """l(x) on standardized inputs."""
        p = np.asarray(p, dtype=float)
        nb = self._nBasis()
        return np.exp(p[0] + self._phi(np.asarray(x, dtype=float)) @ p[1:1 + nb])

    def _core(self, xa, xb, p, batched: bool):
        p = np.asarray(p, dtype=float)
        la, lb = self.lengthscale(xa, p), self.lengthscale(xb, p)
        l2a, l2b = (la ** 2)[..., :, None], (lb ** 2)[..., None, :]
        mean = 0.5 * (l2a + l2b)
        if batched:
            d2 = np.sum((xa[:, :, None, :] - xb[:, None, :, :]) ** 2, axis=-1)
        else:
            d2 = weightedSqDist(xa, xb, np.ones(self.nx))
        pre = (np.sqrt(l2a * l2b) / mean) ** (0.5 * self.nx)
        child = self._child()
        ps = p[1 + self._nBasis():]
        return pre * child._f(d2 / mean, ps)

    def matrix(self, xa, xb, p) -> np.ndarray:
        return self._core(xa, xb, p, False)

    def batchMatrix(self, xa, xb, p) -> np.ndarray:
        return self._core(xa, xb, p, True)


class _CompositeKernel(KernelBase):
    """Base of sum / product kernels.

    ``columns`` (optional, one entry per child) restricts each child to a
    subset of the input columns, e.g. a space x time product::

        {"type": "product", "kernels": ["matern52", "absoluteExponential"], "columns": [[0, 1], [2]]}
    """

    def _declareOptions(self, declare) -> None:
        declare("kernels", [], types=list, desc="Child kernel specs or instances")
        declare("columns", None, types=list, desc="Input columns of each child (default: all columns)")

    def _children(self) -> list[KernelBase]:
        if not hasattr(self, "_built"):
            self._built = [buildComponent("kernel", k) for k in self.options["kernels"]]
            self.options["kernels"] = self._built
        return self._built

    def setup(self, nx: int, plsWeights: Optional[np.ndarray] = None, unitScale: float = 1.0) -> "KernelBase":
        self.nx = nx
        cols = self.options["columns"]
        kids = self._children()
        if cols is None:
            self._cols = [None] * len(kids)
        else:
            if len(cols) != len(kids):
                raise ValueError("columns needs one entry per child kernel")
            self._cols = [None if c is None else np.asarray(c, dtype=int) for c in cols]
            for c in self._cols:
                if c is not None and (c.size == 0 or c.min() < 0 or c.max() >= nx):
                    raise ValueError("child columns must be valid input indices")
            if plsWeights is not None and any(c is not None for c in self._cols):
                raise ValueError("per-child columns cannot be combined with PLS weights")
        for k, c in zip(kids, self._cols):
            k.setup(nx if c is None else c.size, plsWeights, unitScale)
        return self

    def _sub(self, x, i):
        c = self._cols[i]
        return x if c is None else x[..., c]

    def _localColumn(self, i, kx):
        """Index of input kx within child i, or None when the child does not use it."""
        c = self._cols[i]
        if c is None:
            return kx
        where = np.flatnonzero(c == kx)
        return int(where[0]) if where.size else None

    def _childDx(self, i, k, xa, xb, p, kx):
        local = self._localColumn(i, kx)
        if local is None:
            return np.zeros((xa.shape[0], xb.shape[0]))
        return k.dx(self._sub(xa, i), self._sub(xb, i), p, local)

    def lengthscaleIndices(self) -> list[int]:
        out = []
        for k, sl in zip(self._children(), self._slices()):
            out.extend(sl.start + i for i in k.lengthscaleIndices())
        return out

    def lengthscaleColumns(self) -> list[int]:
        out = []
        for i, k in enumerate(self._children()):
            c = self._cols[i] if hasattr(self, "_cols") else None
            out.extend(j if c is None else int(c[j]) for j in k.lengthscaleColumns())
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
        return [k.trainingCache(self._sub(x, i)) for i, k in enumerate(self._children())]


@registry("kernel").register("sum")
class SumKernel(_CompositeKernel):
    """Average of child correlations (keeps a unit diagonal)."""

    def matrix(self, xa, xb, p):
        ks = self._children()
        return sum(k.matrix(self._sub(xa, i), self._sub(xb, i), p[s])
                   for i, (k, s) in enumerate(zip(ks, self._slices()))) / len(ks)

    def gradients(self, x, p):
        return self.matrixAndGradients(x, p)[1]

    def matrixAndGradients(self, x, p, cache=None):
        ks = self._children()
        cache = cache or [None] * len(ks)
        mats, grads = [], []
        for i, (k, s, c) in enumerate(zip(ks, self._slices(), cache)):
            m, g = k.matrixAndGradients(self._sub(x, i), p[s], c)
            mats.append(m)
            grads.extend(gi / len(ks) for gi in g)
        return sum(mats) / len(ks), grads

    def dx(self, xa, xb, p, kx):
        ks = self._children()
        return sum(self._childDx(i, k, xa, xb, p[s], kx) for i, (k, s) in enumerate(zip(ks, self._slices()))) / len(ks)

    def batchMatrix(self, xa, xb, p):
        ks = self._children()
        return sum(k.batchMatrix(self._sub(xa, i), self._sub(xb, i), p[s])
                   for i, (k, s) in enumerate(zip(ks, self._slices()))) / len(ks)


@registry("kernel").register("product")
class ProductKernel(_CompositeKernel):
    """Product of child correlations (e.g. separable space x time)."""

    def matrix(self, xa, xb, p):
        out = 1.0
        for i, (k, s) in enumerate(zip(self._children(), self._slices())):
            out = out * k.matrix(self._sub(xa, i), self._sub(xb, i), p[s])
        return out

    def gradients(self, x, p):
        return self.matrixAndGradients(x, p)[1]

    def batchMatrix(self, xa, xb, p):
        out = 1.0
        for i, (k, s) in enumerate(zip(self._children(), self._slices())):
            out = out * k.batchMatrix(self._sub(xa, i), self._sub(xb, i), p[s])
        return out

    def matrixAndGradients(self, x, p, cache=None):
        ks, sl = self._children(), self._slices()
        cache = cache or [None] * len(ks)
        parts = [k.matrixAndGradients(self._sub(x, i), p[s], c) for i, (k, s, c) in enumerate(zip(ks, sl, cache))]
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
        mats = [k.matrix(self._sub(xa, i), self._sub(xb, i), p[s]) for i, (k, s) in enumerate(zip(ks, sl))]
        total = 0.0
        for i, (k, s) in enumerate(zip(ks, sl)):
            term = self._childDx(i, k, xa, xb, p[s], kx)
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
