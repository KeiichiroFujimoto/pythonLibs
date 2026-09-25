"""Conservative projection: the smallest change of a field that satisfies conservation laws.

    minimize  0.5 sum_k w_k (x_k - x0_k)^2
    subject to  C x = d            (m linear conservation constraints, m << n)
                lb <= x <= ub      (positivity / physical range)

With a diagonal metric w (lumped mass, heat capacity, area, ...) the solution
is x(lambda) = clip(x0 - W^-1 C^T lambda, lb, ub) and only the m multipliers
are unknown: the piecewise-linear equations C x(lambda) = d are solved by a
semismooth Newton method with the generalized Jacobian
-C diag(free / w) C^T and a backtracking line search. Without bounds this is
one exact step. Cost per iteration: O(nnz(C) + m^3), so arrays with millions
of values and a few thousand constraints are cheap.

Nonlinear conservation laws g(x) = d (e.g. energy with temperature-dependent
heat capacity) are handled by sequential linearization: each step projects
x0 onto {g(x_k) + J_k (x - x_k) = d, lb <= x <= ub} (the objective is exactly
quadratic, so this is an SQP method); a residual-based backtracking between
successive iterates keeps it robust.

``feasibility`` reports whether each constraint can be met within the bounds
(for constraints on disjoint groups of values that is exact).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from pythonLibs.regressionHandler.numerics.Sparse import SparseMatrix


@dataclass
class ProjectionResult:
    """Projected field and diagnostics.

    x:           projected values (n,)
    multipliers: Lagrange multipliers of the constraints (m,)
    residual:    constraint residual C x - d (or g(x) - d) (m,)
    atBounds:    boolean mask of values held at a bound
    iterations:  Newton (or SQP) iterations
    converged:   residual below tolerance
    """
    x: np.ndarray
    multipliers: np.ndarray
    residual: np.ndarray
    atBounds: np.ndarray
    iterations: int
    converged: bool
    info: dict = field(default_factory=dict)

    def summary(self) -> dict:
        r = np.abs(self.residual)
        return {"converged": self.converged, "iterations": self.iterations,
                "maxAbsResidual": float(r.max()) if r.size else 0.0, "nAtBounds": int(self.atBounds.sum()),
                **self.info}


def _asSparse(c) -> SparseMatrix:
    if isinstance(c, SparseMatrix):
        return c
    return SparseMatrix.fromDense(np.atleast_2d(np.asarray(c, dtype=float)))


def projectAffine(x0, w, c, d, lb=None, ub=None, tol: float = 1e-12, maxIter: int = 100) -> ProjectionResult:
    """Weighted projection of x0 onto {C x = d, lb <= x <= ub}; see the module docstring.

    Args:
        x0:  (n,) field to correct
        w:   (n,) positive weights of the metric (scalar allowed)
        c:   (m, n) constraint matrix (SparseMatrix or dense)
        d:   (m,) constraint values
        lb, ub: optional bounds (scalars or (n,), may be infinite)
        tol: relative tolerance on |C x - d| (relative to sum |C| |x| per row)
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size
    w = np.broadcast_to(np.asarray(w, dtype=float), (n,)).copy()
    if np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("metric weights must be positive and finite")
    c = _asSparse(c)
    if c.shape[1] != n:
        raise ValueError(f"constraint matrix has {c.shape[1]} columns, field has {n} values")
    d = np.asarray(d, dtype=float).ravel()
    m = c.shape[0]
    if d.size != m:
        raise ValueError("one constraint value per row of C is needed")
    lb = np.broadcast_to(-np.inf if lb is None else np.asarray(lb, dtype=float), (n,))
    ub = np.broadcast_to(np.inf if ub is None else np.asarray(ub, dtype=float), (n,))
    if np.any(lb > ub):
        raise ValueError("lower bounds exceed upper bounds")
    if m == 0:
        x = np.clip(x0, lb, ub)
        return ProjectionResult(x, np.zeros(0), np.zeros(0), (x == lb) | (x == ub), 0, True)
    absC = SparseMatrix(c.row, c.col, np.abs(c.data), c.shape)
    winv = 1.0 / w

    def xOf(lam):
        return np.clip(x0 - winv * c.rmatvec(lam), lb, ub)

    def scaleOf(x):
        return np.maximum(absC @ np.abs(x) + np.abs(d), 1e-300)

    lam = np.zeros(m)
    x = xOf(lam)
    res = c @ x - d
    it = 0
    converged = bool(np.all(np.abs(res) <= tol * scaleOf(x)))
    while not converged and it < maxIter:
        it += 1
        u = x0 - winv * c.rmatvec(lam)                                # unclipped values
        free = (u >= lb) & (u <= ub) & (lb < ub)                    # generalized derivative of the clip
        jac = c.gram(np.where(free, winv, 0.0))                     # -dF/dlambda
        # regularize rows whose values are all at bounds (no local control)
        diag = np.diag(jac).copy()
        dead = diag <= 1e-14 * max(float(diag.max()), 1e-300)
        jac[dead, dead] = max(float(diag.max()), 1.0)
        try:
            step = np.linalg.solve(jac, res)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(jac, res, rcond=None)[0]
        norm0 = float(np.linalg.norm(res))
        t = 1.0
        for _ in range(40):
            trial = lam + t * step
            xt = xOf(trial)
            rt = c @ xt - d
            if np.linalg.norm(rt) <= (1.0 - 1e-4 * t) * norm0:
                break
            t *= 0.5
        else:
            break                                                   # no progress: infeasible within bounds
        lam, x, res = trial, xt, rt
        converged = bool(np.all(np.abs(res) <= tol * scaleOf(x)))
    atBounds = (x <= lb) | (x >= ub)
    return ProjectionResult(x, lam, res, atBounds, it, converged)


def projectNonlinear(x0, w, constraint: Callable[[np.ndarray], tuple[np.ndarray, SparseMatrix]], d, lb=None,
                     ub=None, tol: float = 1e-10, maxIter: int = 50, start=None, scale=None) -> ProjectionResult:
    """Weighted projection of x0 onto {g(x) = d, lb <= x <= ub} by sequential linearization.

    ``constraint(x)`` returns (g(x) (m,), Jacobian dg/dx as SparseMatrix (m, n)).
    ``scale`` (m,) sets the size of each constraint for the tolerance (default |d|);
    give it when a target can be zero.
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size
    d = np.asarray(d, dtype=float).ravel()
    lbA = np.broadcast_to(-np.inf if lb is None else np.asarray(lb, dtype=float), (n,))
    ubA = np.broadcast_to(np.inf if ub is None else np.asarray(ub, dtype=float), (n,))
    x = np.clip(x0 if start is None else np.asarray(start, dtype=float), lbA, ubA)
    g, jac = constraint(x)
    scale = np.maximum(np.abs(d), 1e-300) if scale is None else \
        np.maximum(np.broadcast_to(np.asarray(scale, dtype=float), d.shape), 1e-300)
    lam = np.zeros(d.size)
    it = 0
    xScale = max(float(np.max(np.abs(x0))), 1e-300)
    feasible = bool(np.all(np.abs(g - d) <= tol * scale))
    stationary = False
    # stop only when the constraints hold AND the linearized projection no longer moves the point
    # (a KKT point of the nonlinear problem); feasibility alone is reached after one step
    while not (feasible and stationary) and it < maxIter:
        it += 1
        rhs = d - g + jac @ x
        sub = projectAffine(x0, w, jac, rhs, lbA, ubA, tol=min(tol, 1e-12))
        merit0 = float(np.linalg.norm((g - d) / scale))
        t = 1.0
        for _ in range(30):
            xt = x + t * (sub.x - x)
            gt, jt = constraint(xt)
            merit = float(np.linalg.norm((gt - d) / scale))
            if merit <= max(merit0, 10.0 * tol) or t < 1e-6:
                break
            t *= 0.5
        stationary = float(np.max(np.abs(xt - x))) <= 1e3 * np.finfo(float).eps * xScale + tol * xScale
        x, g, jac, lam = xt, gt, jt, sub.multipliers
        feasible = bool(np.all(np.abs(g - d) <= tol * scale))
    return ProjectionResult(x, lam, g - d, (x <= lbA) | (x >= ubA), it, feasible and stationary)


def feasibility(c, d, lb, ub) -> dict:
    """Range of each constraint row over the box and whether d lies inside it.

    Exact when the rows involve disjoint sets of values (conservation groups);
    otherwise a necessary condition.
    """
    c = _asSparse(c)
    d = np.asarray(d, dtype=float)
    lo = np.where(c.data > 0, np.asarray(lb, dtype=float)[c.col] if np.ndim(lb) else lb,
                  np.asarray(ub, dtype=float)[c.col] if np.ndim(ub) else ub) * c.data
    hi = np.where(c.data > 0, np.asarray(ub, dtype=float)[c.col] if np.ndim(ub) else ub,
                  np.asarray(lb, dtype=float)[c.col] if np.ndim(lb) else lb) * c.data
    low = np.bincount(c.row, weights=np.nan_to_num(lo, nan=0.0, neginf=-1e300), minlength=c.shape[0])
    high = np.bincount(c.row, weights=np.nan_to_num(hi, nan=0.0, posinf=1e300), minlength=c.shape[0])
    ok = (d >= low - 1e-12 * np.abs(d)) & (d <= high + 1e-12 * np.abs(d))
    return {"feasible": bool(ok.all()), "rowFeasible": ok, "lower": low, "upper": high}
