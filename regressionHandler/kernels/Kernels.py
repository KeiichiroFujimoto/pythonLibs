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

    def setup(self, nx: int, plsWeights: Optional[np.ndarray] = None) -> "KernelBase":
        self.nx = nx
        return self

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


class StationaryKernel(KernelBase):
    """Kernel k(r2) of the scaled squared distance."""

    def _declareOptions(self, declare) -> None:
        declare("ard", True, types=bool, desc="One lengthscale per input (False: isotropic)")
        declare("lengthscale0", 1.0, types=(int, float), lower=0.0, desc="Initial lengthscale (standardized units)")
        declare("lengthscaleBounds", [1e-3, 1e3], types=list, desc="Lengthscale bounds (standardized units)")

    def setup(self, nx: int, plsWeights: Optional[np.ndarray] = None) -> "KernelBase":
        self.nx = nx
        self._pls = None if plsWeights is None else np.asarray(plsWeights, dtype=float)
        return self

    # -- parameterization ------------------------------------------------------
    def nParams(self) -> int:
        if self._pls is not None:
            return self._pls.shape[1]
        return self.nx if self.options["ard"] else 1

    def initialParams(self) -> np.ndarray:
        return np.full(self.nParams(), np.log(self.options["lengthscale0"]))

    def bounds(self) -> np.ndarray:
        lo, hi = self.options["lengthscaleBounds"]
        return np.tile([np.log(lo), np.log(hi)], (self.nParams(), 1))

    def paramNames(self) -> list[str]:
        tag = "plsLengthscale" if self._pls is not None else "lengthscale"
        return [f"log_{tag}{i}" for i in range(self.nParams())]

    def invL2(self, p) -> np.ndarray:
        e = np.exp(-2.0 * np.asarray(p, dtype=float))
        if self._pls is not None:
            return (self._pls ** 2) @ e
        return e if self.options["ard"] else np.full(self.nx, e[0])

    def _dInvL2(self, p) -> np.ndarray:
        """(nParams, nx): d invL2 / d p_i."""
        e = np.exp(-2.0 * np.asarray(p, dtype=float))
        if self._pls is not None:
            return (-2.0 * e)[:, None] * (self._pls ** 2).T
        if self.options["ard"]:
            return np.diag(-2.0 * e)
        return np.full((1, self.nx), -2.0 * e[0])

    # -- kernel profile -------------------------------------------------------
    def f(self, r2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fPrime(self, r2: np.ndarray) -> np.ndarray:
        """dk / d(r2); where singular at r2 = 0 it is multiplied by zero distances, return 0 there."""
        raise NotImplementedError

    # -- public ---------------------------------------------------------------
    def matrix(self, xa, xb, p) -> np.ndarray:
        return self.f(weightedSqDist(xa, xb, self.invL2(p)))

    def gradients(self, x, p) -> list[np.ndarray]:
        r2 = weightedSqDist(x, x, self.invL2(p))
        fp = self.fPrime(r2)
        dInv = self._dInvL2(p)
        out = []
        for i in range(dInv.shape[0]):
            c = dInv[i]
            nz = np.flatnonzero(c)
            if nz.size == 1:
                k = nz[0]
                diff = x[:, k][:, None] - x[:, k][None, :]
                out.append(fp * c[k] * diff * diff)
            else:
                out.append(fp * signedSqDist(x, x, c))
        return out

    _CACHE_LIMIT = 60_000_000  # elements of the (nx, n, n) squared-difference cache

    def trainingCache(self, x):
        n, nx = x.shape
        if n * n * nx > self._CACHE_LIMIT:
            return None
        return np.stack([(x[:, k][:, None] - x[:, k][None, :]) ** 2 for k in range(nx)])

    def matrixAndGradients(self, x, p, cache=None):
        if cache is None:
            return self.matrix(x, x, p), self.gradients(x, p)
        inv = self.invL2(p)
        r2 = np.tensordot(inv, cache, axes=1)
        k = self.f(r2)
        fp = self.fPrime(r2)
        dInv = self._dInvL2(p)
        grads = []
        for i in range(dInv.shape[0]):
            nz = np.flatnonzero(dInv[i])
            if nz.size == 1:
                grads.append(fp * (dInv[i, nz[0]] * cache[nz[0]]))
            else:
                grads.append(fp * np.tensordot(dInv[i], cache, axes=1))
        return k, grads

    def dx(self, xa, xb, p, kx: int) -> np.ndarray:
        inv = self.invL2(p)
        fp = self.fPrime(weightedSqDist(xa, xb, inv))
        return fp * 2.0 * inv[kx] * (xa[:, kx][:, None] - xb[:, kx][None, :])


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


@registry("kernel").register("periodic")
class Periodic(KernelBase):
    """exp(-sum_k 2 sin^2(pi d_k / P_k) / l_k^2), ARD lengthscales and periods."""

    def _declareOptions(self, declare) -> None:
        declare("lengthscale0", 1.0, types=(int, float), lower=0.0, desc="Initial lengthscale")
        declare("period0", 1.0, types=(int, float), lower=0.0, desc="Initial period (standardized units)")
        declare("lengthscaleBounds", [1e-3, 1e3], types=list, desc="Lengthscale bounds")
        declare("periodBounds", [1e-2, 1e2], types=list, desc="Period bounds")

    def nParams(self) -> int:
        return 2 * self.nx

    def initialParams(self) -> np.ndarray:
        return np.concatenate([np.full(self.nx, np.log(self.options["lengthscale0"])),
                               np.full(self.nx, np.log(self.options["period0"]))])

    def bounds(self) -> np.ndarray:
        lb, ub = self.options["lengthscaleBounds"]
        pb, pu = self.options["periodBounds"]
        return np.vstack([np.tile([np.log(lb), np.log(ub)], (self.nx, 1)),
                          np.tile([np.log(pb), np.log(pu)], (self.nx, 1))])

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

    def setup(self, nx: int, plsWeights: Optional[np.ndarray] = None) -> "KernelBase":
        self.nx = nx
        for k in self._children():
            k.setup(nx, plsWeights)
        return self

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


KERNEL_ALIASES = {"rbf": "squaredExponential", "gaussian": "squaredExponential"}


def buildKernel(spec) -> KernelBase:
    if isinstance(spec, str):
        spec = KERNEL_ALIASES.get(spec, spec)
    elif isinstance(spec, dict) and spec.get("type") in KERNEL_ALIASES:
        spec = {**spec, "type": KERNEL_ALIASES[spec["type"]]}
    return buildComponent("kernel", spec)
