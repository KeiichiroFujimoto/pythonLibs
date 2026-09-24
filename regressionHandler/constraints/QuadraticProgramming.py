"""Small dense quadratic programs with simple bounds.

    minimize  0.5 x^T H x - g^T x   subject to  lb <= x <= ub        (H symmetric positive semidefinite)

solved exactly by a primal active-set method: the iterate stays feasible, each
step solves the equality problem on the free variables, a blocking bound is
added by a ratio test and the bound with the most violated multiplier is
released at a stationary point. This is the dual problem of projecting onto
linear equality / inequality constraints in any positive definite metric
(equality multipliers are free, inequality multipliers sign-constrained).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def boundedQp(h: np.ndarray, g: np.ndarray, lb: Optional[np.ndarray] = None, ub: Optional[np.ndarray] = None,
              maxIter: Optional[int] = None, ridge: float = 1e-12) -> np.ndarray:
    """Solution of min 0.5 x^T H x - g^T x on the box [lb, ub] (bounds may be infinite).

    A relative ``ridge`` (times the mean diagonal of H) makes semidefinite H
    (redundant constraints in the primal) uniquely solvable.
    """
    h = 0.5 * (np.asarray(h, dtype=float) + np.asarray(h, dtype=float).T)
    g = np.asarray(g, dtype=float)
    n = g.size
    if n == 0:
        return np.zeros(0)
    lb = np.full(n, -np.inf) if lb is None else np.asarray(lb, dtype=float)
    ub = np.full(n, np.inf) if ub is None else np.asarray(ub, dtype=float)
    if np.any(lb > ub):
        raise ValueError("lower bounds exceed upper bounds")
    scale = max(float(np.mean(np.abs(np.diag(h)))), 1e-300)
    h = h + ridge * scale * np.eye(n)
    x = np.clip(np.zeros(n), lb, ub)
    atLower = np.isfinite(lb) & (x == lb)
    atUpper = np.isfinite(ub) & (x == ub) & ~atLower
    gScale = max(float(np.max(np.abs(g))), scale * max(1.0, float(np.max(np.abs(np.where(np.isfinite(x), x, 0))))),
                 1e-300)
    tol = 1e-12 * gScale
    for _ in range(maxIter or 20 * n + 100):
        free = ~(atLower | atUpper)
        target = x.copy()
        if free.any():
            fixed = ~free
            rhs = g[free] - h[np.ix_(free, fixed)] @ x[fixed]
            target[free] = np.linalg.solve(h[np.ix_(free, free)], rhs)
        d = target - x
        # ratio test against the bounds of the free variables
        with np.errstate(divide="ignore", invalid="ignore"):
            toLower = np.where(free & (d < 0) & np.isfinite(lb), (lb - x) / d, np.inf)
            toUpper = np.where(free & (d > 0) & np.isfinite(ub), (ub - x) / d, np.inf)
        steps = np.minimum(toLower, toUpper)
        k = int(np.argmin(steps))
        if steps[k] < 1.0:
            x = x + max(steps[k], 0.0) * d
            if toLower[k] <= toUpper[k]:
                x[k], atLower[k] = lb[k], True
            else:
                x[k], atUpper[k] = ub[k], True
            continue
        x = target
        grad = h @ x - g
        # multiplier sign: at a lower bound the gradient must be >= 0, at an upper bound <= 0
        viol = np.where(atLower, -grad, np.where(atUpper, grad, -np.inf))
        k = int(np.argmax(viol))
        if viol[k] <= tol:
            return x
        atLower[k] = atUpper[k] = False
    raise RuntimeError("bounded QP did not converge")
