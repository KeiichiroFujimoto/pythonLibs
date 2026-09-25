"""Epsilon support vector regression (epsilon-SVR), solved like LIBSVM.

    min  1/2 (a - a*)^T K (a - a*) + eps sum (a + a*) - y^T (a - a*)
    s.t. sum (a - a*) = 0,  0 <= a, a* <= C
    f(x) = sum (a_i - a*_i) k(x_i, x) - rho

The dual is solved by sequential minimal optimization with LIBSVM's
second-order working-set selection (Fan, Chen & Lin 2005) on the 2n
variables [a, a*], the same two-variable update and clipping, the same
stopping rule (maximal violation < tol) and the same rho (mean over free
variables, else the midpoint of the bounds). Kernel values enter the
updates in single precision as in LIBSVM's kernel cache; shrinking is not
used, which changes only the path within the tolerance.

Reproducibility against a compiled LIBSVM: the SMO path is identical when the
gradient update is rounded the same way. Builds that contract
``G += Q_i dA_i + Q_j dA_j`` into ``fma(Q_i, dA_i, Q_j dA_j)`` (clang on
arm64, e.g. scikit-learn wheels on Apple silicon) round differently from the
separate numpy operations used here, which on flat-curvature problems (rbf)
can flip a working-set choice and move the solution within ``tol``;
``gradientUpdate`` lets a caller supply the fused rounding.

Kernels: rbf exp(-gamma |x - x'|^2), linear x.x', poly (gamma x.x' + coef0)^degree,
sigmoid tanh(gamma x.x' + coef0); gamma "scale" = 1 / (nx var(X)), "auto" = 1 / nx.

References:
    R.-E. Fan, P.-H. Chen, C.-J. Lin, "Working set selection using second order
    information for training support vector machines", JMLR 6 (2005).
    C.-C. Chang, C.-J. Lin, "LIBSVM: a library for support vector machines" (2011).
"""
from __future__ import annotations

import numpy as np

from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase

SVR_KERNELS = ("rbf", "linear", "poly", "sigmoid")
_TAU = 1e-12


def svrKernel(kernel: str, a: np.ndarray, b: np.ndarray, gamma: float, degree: int, coef0: float) -> np.ndarray:
    if kernel == "linear":
        return a @ b.T
    if kernel == "rbf":
        d2 = np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - 2.0 * (a @ b.T)
        return np.exp(-gamma * np.maximum(d2, 0.0))
    if kernel == "poly":
        return (gamma * (a @ b.T) + coef0) ** degree
    if kernel == "sigmoid":
        return np.tanh(gamma * (a @ b.T) + coef0)
    raise ValueError(f"unknown kernel {kernel!r}; use one of {SVR_KERNELS}")


def solveSvrDual(k: np.ndarray, y: np.ndarray, c: float, eps: float, tol: float, maxIter: int,
                 gradientUpdate=None):
    """LIBSVM Solver on [a, a*]; returns (coef = a - a*, rho, iterations).

    ``gradientUpdate(qi, dI, qj, dJ)`` returns the increment of the gradient (default
    ``qi * dI + qj * dJ`` with separate roundings).
    """
    n = y.size
    ll = 2 * n
    sign = np.r_[np.ones(n), -np.ones(n)]               # LIBSVM y of the 2n variables
    idx = np.r_[np.arange(n), np.arange(n)]
    kf = k.astype(np.float32).astype(np.float64)          # single-precision kernel cache
    qd = np.diag(k)[idx].copy()                          # QD kept in double
    alpha = np.zeros(ll)
    grad = np.r_[eps - y, eps + y]                       # G = p at alpha = 0
    upper = np.zeros(ll, dtype=bool)
    lower = np.ones(ll, dtype=bool)

    def qRow(i):
        return sign[i] * sign * kf[idx[i], idx]

    it = 0
    while it < maxIter:
        # --- working set selection (WSS2); ">=" / "<=" keep LIBSVM's last-index tie rule
        candI = np.where(sign > 0, ~upper, ~lower)
        score = np.where(sign > 0, -grad, grad)
        score = np.where(candI, score, -np.inf)
        gMax = float(np.max(score))
        i = ll - 1 - int(np.argmax(score[::-1])) if np.isfinite(gMax) else -1
        candJ = np.where(sign > 0, ~lower, ~upper)
        gMax2 = float(np.max(np.where(candJ, np.where(sign > 0, grad, -grad), -np.inf)))
        if i < 0 or gMax + gMax2 < tol:
            break
        qi = qRow(i)
        gradDiff = np.where(sign > 0, gMax + grad, gMax - grad)
        # y_i Q_ij y_j = K(i, j) in both LIBSVM branches.
        quad = qd[i] + qd - 2.0 * sign[i] * qi * sign
        quad = np.where(quad > 0, quad, _TAU)
        objDiff = np.where(candJ & (gradDiff > 0), -(gradDiff * gradDiff) / quad, np.inf)
        best = float(np.min(objDiff))
        if not np.isfinite(best):
            break
        j = int(np.flatnonzero(objDiff == best)[-1])
        it += 1
        # --- two-variable update (LIBSVM Solver::Solve)
        qj = qRow(j)
        oldI, oldJ = alpha[i], alpha[j]
        if sign[i] != sign[j]:
            q = qd[i] + qd[j] + 2.0 * qi[j]
            q = q if q > 0 else _TAU
            delta = (-grad[i] - grad[j]) / q
            diff = alpha[i] - alpha[j]
            alpha[i] += delta
            alpha[j] += delta
            if diff > 0:
                if alpha[j] < 0:
                    alpha[j] = 0.0
                    alpha[i] = diff
            elif alpha[i] < 0:
                alpha[i] = 0.0
                alpha[j] = -diff
            if diff > 0.0:                               # C_i - C_j = 0
                if alpha[i] > c:
                    alpha[i] = c
                    alpha[j] = c - diff
            elif alpha[j] > c:
                alpha[j] = c
                alpha[i] = c + diff
        else:
            q = qd[i] + qd[j] - 2.0 * qi[j]
            q = q if q > 0 else _TAU
            delta = (grad[i] - grad[j]) / q
            total = alpha[i] + alpha[j]
            alpha[i] -= delta
            alpha[j] += delta
            if total > c:
                if alpha[i] > c:
                    alpha[i] = c
                    alpha[j] = total - c
            elif alpha[j] < 0:
                alpha[j] = 0.0
                alpha[i] = total
            if total > c:
                if alpha[j] > c:
                    alpha[j] = c
                    alpha[i] = total - c
            elif alpha[i] < 0:
                alpha[i] = 0.0
                alpha[j] = total
        dI, dJ = alpha[i] - oldI, alpha[j] - oldJ
        grad += qi * dI + qj * dJ if gradientUpdate is None else gradientUpdate(qi, dI, qj, dJ)
        for t in (i, j):
            upper[t] = alpha[t] >= c
            lower[t] = alpha[t] <= 0.0
    # --- rho (Solver::calculate_rho)
    yg = sign * grad
    free = ~upper & ~lower
    if free.any():
        rho = float(np.sum(yg[free]) / np.count_nonzero(free))
    else:
        toUb = (upper & (sign < 0)) | (lower & (sign > 0))
        toLb = (upper & (sign > 0)) | (lower & (sign < 0))
        ub = float(np.min(yg[toUb])) if toUb.any() else np.inf
        lb = float(np.max(yg[toLb])) if toLb.any() else -np.inf
        rho = 0.5 * (ub + lb)
    return alpha[:n] - alpha[n:], rho, it


@registry("model").register("svr")
class SupportVectorModel(SurrogateModelBase):
    """Epsilon support vector regression (LIBSVM-equivalent SMO)."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("kernel", "rbf", values=SVR_KERNELS, desc="Kernel function")
        d("C", 1.0, types=(int, float), lower=0.0, desc="Box constraint (penalty on errors beyond epsilon)")
        d("epsilon", 0.1, types=(int, float), lower=0.0, desc="Half-width of the insensitive tube")
        d("gamma", "scale", values=("scale", "auto"), types=(int, float),
          desc="Kernel coefficient: 'scale' 1/(nx var X), 'auto' 1/nx, or a value")
        d("degree", 3, types=int, lower=1, desc="Degree of the poly kernel")
        d("coef0", 0.0, types=(int, float), desc="Offset of the poly / sigmoid kernels")
        d("tol", 1e-3, types=float, lower=0.0, desc="Stopping tolerance on the maximal KKT violation")
        d("maxIter", None, types=int, lower=1, desc="SMO iterations (None: max(1e7, 100 n))")
        self.supports.update(multiOutput=False, weights=False, variances=False, derivatives=False,
                             parameterInference=False)

    def _train(self) -> None:
        x, y = self.xt, self.yt[:, 0]
        g = self.options["gamma"]
        if g == "scale":
            v = float(x.var())
            g = 1.0 / (x.shape[1] * v) if v > 0 else 1.0
        elif g == "auto":
            g = 1.0 / x.shape[1]
        self._gamma = float(g)
        k = self._kernel(x, x)
        maxIter = self.options["maxIter"] or max(10_000_000, 100 * y.size)
        coef, rho, iters = solveSvrDual(k, y, float(self.options["C"]), float(self.options["epsilon"]),
                                        float(self.options["tol"]), maxIter)
        keep = coef != 0.0
        self._sv, self._coef, self._rho, self._iterations = x[keep], coef[keep], rho, iters

    def _kernel(self, a, b):
        o = self.options
        return svrKernel(o["kernel"], a, b, self._gamma, o["degree"], float(o["coef0"]))

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        if self._sv.shape[0] == 0:
            return np.full((x.shape[0], 1), -self._rho)
        return (self._kernel(x, self._sv) @ self._coef - self._rho)[:, None]

    @property
    def supportVectors(self) -> np.ndarray:
        return self._sv.copy()

    @property
    def dualCoefficients(self) -> np.ndarray:
        return self._coef.copy()

    @property
    def intercept(self) -> float:
        return -self._rho

    def _effectiveParams(self):
        return float(self._sv.shape[0] + 1)

    def _stateToDict(self) -> dict:
        return {"sv": self._sv.tolist(), "coef": self._coef.tolist(), "rho": self._rho, "gamma": self._gamma}

    def _stateFromDict(self, state: dict) -> None:
        self._sv = np.array(state["sv"], dtype=float).reshape(-1, self.nx)
        self._coef = np.array(state["coef"], dtype=float)
        self._rho, self._gamma = float(state["rho"]), float(state["gamma"])
