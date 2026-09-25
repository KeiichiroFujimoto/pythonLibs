"""Self-contained optimizers.

- ``leastSquares``   Levenberg-Marquardt with box bounds (projected /
  active-set steps), robust losses (``linear``, ``soft_l1``, ``huber``,
  ``cauchy``, ``arctan``) with residual scale ``fScale``, analytic or
  finite-difference Jacobian ('2-point' / '3-point')
- ``minimize``       projected limited-memory BFGS (``L-BFGS-B``) or
  ``Nelder-Mead`` with bounds
- ``minimizeScalar`` Brent's bounded golden-section / parabolic search
- ``OptimizeResult`` common result object

References:
    Nielsen, H.B. (1999) "Damping parameter in Marquardt's method", IMM DTU.
    Moré, J.J. (1978) "The Levenberg-Marquardt algorithm: implementation and
    theory", Lecture Notes in Mathematics 630.
    Nocedal & Wright (2006) *Numerical Optimization*, 2nd ed., ch. 7 (L-BFGS).
    Brent, R.P. (1973) *Algorithms for Minimization without Derivatives*, ch. 5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

_EPS = np.finfo(float).eps


@dataclass
class OptimizeResult:
    """Outcome of an optimization run.

    Attributes:
        x:        solution vector
        fun:      objective value (``cost`` = 0.5 * sum(rho) for least squares)
        success:  whether a convergence criterion was met
        status:   integer code (see ``message``)
        message:  human readable termination reason
        nfev:     number of objective evaluations
        njev:     number of Jacobian / gradient evaluations
        nit:      number of iterations
        jac:      Jacobian of residuals (least squares) or gradient (minimize)
        residuals: residual vector at the solution (least squares only)
    """
    x: np.ndarray
    fun: float
    success: bool
    status: int
    message: str
    nfev: int
    njev: int
    nit: int
    jac: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    extra: dict = field(default_factory=dict)

    @property
    def cost(self) -> float:
        return self.fun


# ---------------------------------------------------------------- bounds helpers
def _prepareBounds(bounds, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalize bounds to (lb, ub) arrays.

    Accepted forms: None (unbounded); a *tuple* ``(lb, ub)`` of scalars or
    length-n arrays; a *list* of ``(lo, hi)``
    pairs with None for an open side (``minimize`` style).
    """
    if bounds is None:
        return np.full(n, -np.inf), np.full(n, np.inf)
    if isinstance(bounds, tuple):
        if len(bounds) != 2:
            raise ValueError("tuple bounds must be (lb, ub)")
        lb = np.broadcast_to(np.asarray(bounds[0], dtype=float), (n,)).copy()
        ub = np.broadcast_to(np.asarray(bounds[1], dtype=float), (n,)).copy()
    else:
        if len(bounds) != n:
            raise ValueError(f"expected {n} (lo, hi) pairs, got {len(bounds)}")
        lb, ub = boundsFromPairs(bounds)
    if np.any(lb >= ub):
        raise ValueError("each lower bound must be strictly less than its upper bound")
    return lb, ub


def boundsFromPairs(pairs) -> tuple[np.ndarray, np.ndarray]:
    """Convert [(lo, hi), ...] into (lb, ub) arrays (None means unbounded)."""
    arr = np.array([[(-np.inf if lo is None else lo), (np.inf if hi is None else hi)]
                    for lo, hi in pairs], dtype=float)
    return arr[:, 0], arr[:, 1]


# ---------------------------------------------------------------- finite differences
def approxJacobian(fun: Callable, x: np.ndarray, f0: Optional[np.ndarray] = None,
                   method: str = "2-point", lb=None, ub=None) -> tuple[np.ndarray, int]:
    """Finite-difference Jacobian of a vector function. Returns (J, nfev)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    lb = np.full(n, -np.inf) if lb is None else lb
    ub = np.full(n, np.inf) if ub is None else ub
    nfev = 0
    if f0 is None and method == "2-point":
        f0 = np.atleast_1d(fun(x))
        nfev += 1
    if method == "2-point":
        h = np.sqrt(_EPS) * np.maximum(1.0, np.abs(x))
        h = np.where(x + h > ub, -h, h)
        jac = np.empty((np.size(f0), n))
        for j in range(n):
            xp = x.copy()
            xp[j] += h[j]
            dx = xp[j] - x[j]
            jac[:, j] = (np.atleast_1d(fun(xp)) - f0) / dx
            nfev += 1
        return jac, nfev
    if method == "3-point":
        h = _EPS ** (1.0 / 3.0) * np.maximum(1.0, np.abs(x))
        cols = []
        for j in range(n):
            hp = h[j] if x[j] + h[j] <= ub[j] else 0.0
            hm = h[j] if x[j] - h[j] >= lb[j] else 0.0
            xp = x.copy()
            xm = x.copy()
            xp[j] += hp
            xm[j] -= hm
            if hp == 0.0 and hm == 0.0:
                raise ValueError("bounds too tight for finite differences")
            cols.append((np.atleast_1d(fun(xp)) - np.atleast_1d(fun(xm))) / (hp + hm))
            nfev += 2
        return np.column_stack(cols), nfev
    raise ValueError("method must be '2-point' or '3-point'")


def approxGradient(fun: Callable, x: np.ndarray, f0: Optional[float] = None,
                   lb=None, ub=None) -> tuple[np.ndarray, int]:
    jac, nfev = approxJacobian(lambda v: np.array([fun(v)]), x,
                               None if f0 is None else np.array([f0]), "2-point", lb, ub)
    return jac[0], nfev


# ---------------------------------------------------------------- robust losses
def _lossFunction(name: str) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return z -> (rho(z), rho'(z)) with z = (r / fScale)^2."""
    if name == "linear":
        return lambda z: (z, np.ones_like(z))
    if name == "soft_l1":
        def softL1(z):
            t = np.sqrt(1.0 + z)
            return 2.0 * (t - 1.0), 1.0 / t
        return softL1
    if name == "huber":
        def huber(z):
            small = z <= 1.0
            sq = np.sqrt(np.where(small, 1.0, z))
            return np.where(small, z, 2.0 * sq - 1.0), np.where(small, 1.0, 1.0 / sq)
        return huber
    if name == "cauchy":
        return lambda z: (np.log1p(z), 1.0 / (1.0 + z))
    if name == "arctan":
        return lambda z: (np.arctan(z), 1.0 / (1.0 + z * z))
    raise ValueError(f"unknown loss {name!r}; use linear, soft_l1, huber, cauchy or arctan")


LOSSES = ("linear", "soft_l1", "huber", "cauchy", "arctan")


# ---------------------------------------------------------------- least squares
def leastSquares(fun: Callable, x0, jac="2-point", bounds=None, loss: str = "linear",
                 fScale: float = 1.0, xScale="jac", ftol: float = 1e-10, xtol: float = 1e-10,
                 gtol: float = 1e-10, maxNfev: Optional[int] = None, args: tuple = ()) -> OptimizeResult:
    """Minimize 0.5 * sum(rho((f_i(x) / fScale)^2)) * fScale^2 subject to bounds.

    Args:
        fun:      residual function f(x, *args) -> (m,)
        x0:       initial guess (projected into the bounds)
        jac:      callable J(x, *args) -> (m, n), '2-point' or '3-point'
        bounds:   tuple (lb, ub) of scalars/arrays, or list of (lo, hi); None = unbounded
        loss:     robust loss name (see ``LOSSES``)
        fScale:   residual scale separating inliers from outliers for robust losses
        xScale:   'jac' (adaptive, MINPACK style) or an array of variable scales
        ftol, xtol, gtol: relative cost, step and gradient tolerances
        maxNfev:  evaluation budget (default 100 * n * (n + 1))
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    n = x.size
    lb, ub = _prepareBounds(bounds, n)
    x = np.clip(x, lb, ub)
    rhoFun = _lossFunction(loss)
    fs2 = fScale * fScale
    maxNfev = maxNfev or 100 * n * (n + 1)

    def residuals(v):
        return np.atleast_1d(np.asarray(fun(v, *args), dtype=float))

    def jacobian(v, f):
        if callable(jac):
            return np.atleast_2d(np.asarray(jac(v, *args), dtype=float)), 0
        return approxJacobian(lambda u: residuals(u), v, f, jac, lb, ub)

    def robustCost(f):
        z = (f * f) / fs2
        rho, rho1 = rhoFun(z)
        return 0.5 * fs2 * float(np.sum(rho)), rho1

    f = residuals(x)
    nfev, njev = 1, 0
    if not np.all(np.isfinite(f)):
        raise ValueError("residuals are not finite at the initial point")
    cost, rho1 = robustCost(f)
    jMat, extra = jacobian(x, f)
    nfev += extra
    njev += 1

    fixedScale = not (isinstance(xScale, str) and xScale == "jac")
    diag = None if not fixedScale else 1.0 / np.broadcast_to(np.asarray(xScale, float), (n,))
    mu = None
    nu = 2.0
    gScale = None
    status, message = 0, "maximum number of function evaluations exceeded"
    nit = 0
    while nfev < maxNfev:
        nit += 1
        # IRLS weighting of residuals and Jacobian for the robust loss.
        sw = np.sqrt(rho1)
        jw = jMat * sw[:, None]
        fw = f * sw
        g = jw.T @ fw
        a = jw.T @ jw
        # Projected gradient: components pushing into an active bound vanish.
        atLower = (x <= lb) & (g > 0.0)
        atUpper = (x >= ub) & (g < 0.0)
        pg = np.where(atLower | atUpper, 0.0, g)
        if gScale is None:
            gScale = max(1.0, float(np.max(np.abs(g))))
        if np.max(np.abs(pg)) <= gtol * gScale:
            status, message = 1, "gtol termination: projected gradient is small"
            break
        colNorm = np.sqrt(np.diag(a))
        if not fixedScale:
            diag = colNorm.copy() if diag is None else np.maximum(diag, colNorm)
            diag[diag == 0.0] = 1.0
        d2 = diag * diag
        if mu is None:
            mu = 1e-3 * float(np.max(np.diag(a) / d2)) if np.max(np.diag(a)) > 0 else 1e-3

        accepted = False
        while nfev < maxNfev:
            # Active-set step: variables held at a bound by the gradient stay fixed.
            freeIdx = np.flatnonzero(~(atLower | atUpper))
            step = np.zeros(n)
            try:
                sub = a[np.ix_(freeIdx, freeIdx)] + mu * np.diag(d2[freeIdx])
                step[freeIdx] = np.linalg.solve(sub, -g[freeIdx])
            except np.linalg.LinAlgError:
                mu *= nu
                nu *= 2.0
                continue
            xNew = np.clip(x + step, lb, ub)
            step = xNew - x
            fNew = residuals(xNew)
            nfev += 1
            if not np.all(np.isfinite(fNew)):
                mu *= nu
                nu *= 2.0
                continue
            costNew, rho1New = robustCost(fNew)
            predicted = -(g @ step + 0.5 * step @ (a @ step))
            actual = cost - costNew
            ratio = actual / predicted if predicted > 0 else -1.0
            if ratio > 0.0:
                mu *= max(1.0 / 3.0, 1.0 - (2.0 * ratio - 1.0) ** 3)
                nu = 2.0
                stepNorm = float(np.linalg.norm(step * diag))
                xNorm = float(np.linalg.norm(x * diag))
                x, f, cost, rho1 = xNew, fNew, costNew, rho1New
                accepted = True
                if actual <= ftol * max(cost, _EPS) and ratio > 0.25:
                    status, message = 2, "ftol termination: cost reduction is small"
                elif stepNorm <= xtol * (xtol + xNorm):
                    status, message = 3, "xtol termination: step is small"
                break
            mu *= nu
            nu *= 2.0
            if float(np.linalg.norm(step * diag)) <= xtol * (xtol + float(np.linalg.norm(x * diag))):
                status, message = 3, "xtol termination: step is small"
                break
        if status in (2, 3):
            if accepted:
                jMat, extra = jacobian(x, f)
                nfev += extra
                njev += 1
            break
        if not accepted:
            break
        jMat, extra = jacobian(x, f)
        nfev += extra
        njev += 1

    return OptimizeResult(x=x, fun=cost, success=status > 0, status=status, message=message,
                          nfev=nfev, njev=njev, nit=nit, jac=jMat, residuals=f)


# ---------------------------------------------------------------- scalar minimization
def minimizeScalar(fun: Callable[[float], float], bounds: tuple[float, float],
                   xatol: float = 1e-8, maxIter: int = 500) -> OptimizeResult:
    """Brent's bounded minimization of a scalar function on [a, b]."""
    a, b = map(float, bounds)
    if not a < b:
        raise ValueError("bounds must satisfy a < b")
    golden = 0.5 * (3.0 - np.sqrt(5.0))
    x = w = v = a + golden * (b - a)
    fx = fw = fv = float(fun(x))
    nfev = 1
    d = e = 0.0
    it = 0
    for it in range(1, maxIter + 1):
        m = 0.5 * (a + b)
        tol1 = np.sqrt(_EPS) * abs(x) + xatol / 3.0
        tol2 = 2.0 * tol1
        if abs(x - m) <= tol2 - 0.5 * (b - a):
            break
        useGolden = True
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            # accept the parabolic step only if it lands inside (a, b): q (a - x) < p < q (b - x)
            if abs(p) < abs(0.5 * q * e) and q * (a - x) < p < q * (b - x):
                e, d = d, p / q
                u = x + d
                if u - a < tol2 or b - u < tol2:
                    d = tol1 if x < m else -tol1
                useGolden = False
        if useGolden:
            e = (b - x) if x < m else (a - x)
            d = golden * e
        u = x + (d if abs(d) >= tol1 else (tol1 if d > 0 else -tol1))
        fu = float(fun(u))
        nfev += 1
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, fv, w, fw, x, fx = w, fw, x, fx, u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv, w, fw = w, fw, u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return OptimizeResult(x=np.array([x]), fun=fx, success=True, status=1,
                          message="converged", nfev=nfev, njev=0, nit=it)


# ---------------------------------------------------------------- general minimization
def minimize(fun: Callable, x0, jac=None, bounds=None, method: str = "L-BFGS-B",
             tol: float = 1e-8, maxIter: int = 1000, memory: int = 10, args: tuple = ()) -> OptimizeResult:
    """Minimize a scalar function of several variables.

    Args:
        fun:     f(x, *args) -> float, or (float, grad) when ``jac is True``
        jac:     True (fun returns gradient), callable grad(x, *args), or None (finite differences)
        bounds:  list of (lo, hi) pairs or tuple (lb, ub); honoured by both methods
        method:  "L-BFGS-B" or "Nelder-Mead"
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    lb, ub = _prepareBounds(bounds, x.size)
    x = np.clip(x, lb, ub)
    if method.upper() == "NELDER-MEAD":
        return _nelderMead(lambda v: float(fun(v, *args) if jac is not True else fun(v, *args)[0]),
                           x, lb, ub, tol, maxIter * max(1, x.size))
    if method.upper() != "L-BFGS-B":
        raise ValueError("method must be 'L-BFGS-B' or 'Nelder-Mead'")
    return _projectedLbfgs(fun, jac, x, lb, ub, tol, maxIter, memory, args)


def _projectedLbfgs(fun, jac, x, lb, ub, tol, maxIter, memory, args) -> OptimizeResult:
    counters = {"nfev": 0, "njev": 0}

    def evaluate(v):
        counters["nfev"] += 1
        if jac is True:
            fv, gv = fun(v, *args)
            counters["njev"] += 1
            return float(fv), np.asarray(gv, dtype=float)
        fv = float(fun(v, *args))
        if callable(jac):
            counters["njev"] += 1
            return fv, np.asarray(jac(v, *args), dtype=float)
        gv, extra = approxGradient(lambda u: float(fun(u, *args)), v, fv, lb, ub)
        counters["nfev"] += extra
        return fv, gv

    def projectedGradient(v, g):
        pg = g.copy()
        pg[(v <= lb) & (g > 0.0)] = 0.0
        pg[(v >= ub) & (g < 0.0)] = 0.0
        return pg

    f, g = evaluate(x)
    if not np.isfinite(f):
        raise ValueError("objective is not finite at the initial point")
    sList: list[np.ndarray] = []
    yList: list[np.ndarray] = []
    status, message = 0, "maximum number of iterations reached"
    it = 0
    for it in range(1, maxIter + 1):
        pg = projectedGradient(x, g)
        if np.max(np.abs(pg)) <= tol:
            status, message = 1, "projected gradient below tolerance"
            break
        free = pg != 0.0
        # Two-loop recursion on the free variables.
        q = np.where(free, g, 0.0)
        alphas = []
        for s, y in zip(reversed(sList), reversed(yList)):
            rho = 1.0 / (y @ s)
            alpha = rho * (s @ q)
            alphas.append((alpha, rho, s, y))
            q = q - alpha * np.where(free, y, 0.0)
        if yList:
            gamma = (sList[-1] @ yList[-1]) / (yList[-1] @ yList[-1])
        else:
            gamma = 1.0 / max(np.linalg.norm(pg), 1.0)
        r = gamma * q
        for alpha, rho, s, y in reversed(alphas):
            beta = rho * (y @ r)
            r = r + np.where(free, s, 0.0) * (alpha - beta)
        direction = -np.where(free, r, 0.0)
        if direction @ g >= 0.0:
            direction = -pg
            sList.clear()
            yList.clear()
        # Projected backtracking (Armijo) line search.
        step = 1.0
        accepted = False
        for _ in range(60):
            xNew = np.clip(x + step * direction, lb, ub)
            fNew, gNew = evaluate(xNew)
            if np.isfinite(fNew) and fNew <= f + 1e-4 * (g @ (xNew - x)):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            status, message = 2, "line search could not reduce the objective"
            break
        s = xNew - x
        y = gNew - g
        fOld = f
        x, f, g = xNew, fNew, gNew
        if s @ y > 1e-10 * np.linalg.norm(s) * np.linalg.norm(y):
            sList.append(s)
            yList.append(y)
            if len(sList) > memory:
                sList.pop(0)
                yList.pop(0)
        if abs(fOld - f) <= tol * max(1.0, abs(f), abs(fOld)) * 1e-2:
            status, message = 3, "relative reduction of objective below tolerance"
            break
    return OptimizeResult(x=x, fun=f, success=status in (1, 3), status=status, message=message,
                          nfev=counters["nfev"], njev=counters["njev"], nit=it, jac=g)


def _nelderMead(f, x0, lb, ub, tol, maxFev) -> OptimizeResult:
    n = x0.size
    simplex = [x0]
    for i in range(n):
        v = x0.copy()
        v[i] = v[i] + (0.05 * v[i] if v[i] != 0 else 0.00025)
        simplex.append(np.clip(v, lb, ub))
    simplex = np.array(simplex)
    values = np.array([f(v) for v in simplex])
    nfev = n + 1
    it = 0
    while nfev < maxFev:
        it += 1
        order = np.argsort(values)
        simplex, values = simplex[order], values[order]
        if np.max(np.abs(simplex[1:] - simplex[0])) <= tol and np.max(np.abs(values[1:] - values[0])) <= tol:
            break
        centroid = simplex[:-1].mean(axis=0)
        xr = np.clip(centroid + (centroid - simplex[-1]), lb, ub)
        fr = f(xr)
        nfev += 1
        if fr < values[0]:
            xe = np.clip(centroid + 2.0 * (centroid - simplex[-1]), lb, ub)
            fe = f(xe)
            nfev += 1
            simplex[-1], values[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < values[-2]:
            simplex[-1], values[-1] = xr, fr
        else:
            outside = fr < values[-1]
            xc = centroid + (0.5 if outside else -0.5) * (centroid - simplex[-1])
            fc = f(xc)
            nfev += 1
            if fc < (fr if outside else values[-1]):
                simplex[-1], values[-1] = xc, fc
            else:
                simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                values[1:] = [f(v) for v in simplex[1:]]
                nfev += n
    best = int(np.argmin(values))
    return OptimizeResult(x=simplex[best], fun=float(values[best]), success=nfev < maxFev,
                          status=1 if nfev < maxFev else 0,
                          message="converged" if nfev < maxFev else "maximum evaluations reached",
                          nfev=nfev, njev=0, nit=it)


def multiStart(optimizer: Callable[[np.ndarray], OptimizeResult], starts: np.ndarray) -> OptimizeResult:
    """Run ``optimizer`` from each start point and keep the lowest objective."""
    best = None
    failures = []
    for s in np.atleast_2d(starts):
        try:
            res = optimizer(np.asarray(s, dtype=float))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            failures.append(str(exc))
            continue
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("all optimizer starts failed: " + "; ".join(failures[:3]))
    best.extra["nStarts"] = len(np.atleast_2d(starts))
    best.extra["nFailedStarts"] = len(failures)
    return best
