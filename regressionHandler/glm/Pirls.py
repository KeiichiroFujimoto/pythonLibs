"""Penalized iteratively reweighted least squares (P-IRLS) and smoothing-parameter selection.

Solves

    minimize  deviance(y, g^-1(X beta + offset)) + beta^T S_lambda beta,   S_lambda = sum_j lambda_j S_j

by Fisher scoring on the working response z = eta - offset + (y - mu) g'(mu)
with weights W = w (dmu/deta)^2 / V(mu). Each step is solved as the augmented
least-squares problem [sqrt(W) X; R] beta ~ [sqrt(W) z; 0] with R^T R =
S_lambda (column-equilibrated SVD), which stays accurate for rank-deficient
or strongly penalized designs. Step halving keeps the penalized deviance
decreasing and mu valid.

Smoothing parameters minimize GCV = n D / (n - edf)^2 (unknown scale) or
UBRE = D / n + 2 phi edf / n - phi (known scale) by an outer quasi-Newton
search in log lambda (Wood 2006, "performance-oriented" outer iteration).

References:
    Green & Silverman (1994) *Nonparametric Regression and Generalized Linear Models*.
    Wood (2017) *Generalized Additive Models*, 2nd ed., ch. 3, 6.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.glm.Families import Family
from pythonLibs.regressionHandler.numerics.LinearAlgebra import leastSquaresSvd
from pythonLibs.regressionHandler.numerics.Optimizers import minimize, minimizeScalar


@dataclass
class PirlsResult:
    """Converged P-IRLS fit.

    covUnscaled = (X^T W X + S)^-1 (multiply by phi); hatDiag and edf come from
    the influence matrix A = X (X^T W X + S)^-1 X^T W; edfTerms splits the edf by
    coefficient (diag of F = (X^T W X + S)^-1 X^T W X).
    """
    coef: np.ndarray
    eta: np.ndarray
    mu: np.ndarray
    weights: np.ndarray
    deviance: float
    penalty: float
    covUnscaled: np.ndarray
    hatDiag: np.ndarray
    edf: float
    edfTerms: np.ndarray
    iterations: int
    converged: bool
    rank: int
    info: dict = field(default_factory=dict)


def _sqrtPenalty(s: np.ndarray) -> np.ndarray:
    """R with R^T R = S for a symmetric PSD S (rows with zero eigenvalue dropped)."""
    w, v = np.linalg.eigh(0.5 * (s + s.T))
    keep = w > max(float(w.max()), 0.0) * 1e-13 if w.size else np.zeros(0, bool)
    return (v[:, keep] * np.sqrt(w[keep])).T


def pirls(x: np.ndarray, y: np.ndarray, w: np.ndarray, family: Family, penalty: Optional[np.ndarray] = None,
          offset: Optional[np.ndarray] = None, start: Optional[np.ndarray] = None, maxIter: int = 100,
          tol: float = 1e-10) -> PirlsResult:
    """Fit one (penalized) GLM by Fisher scoring; see the module docstring."""
    n, p = x.shape
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    link = family.link
    rPen = _sqrtPenalty(penalty) if penalty is not None and np.any(penalty) else np.zeros((0, p))
    sPen = rPen.T @ rPen

    def penalizedDeviance(beta, mu):
        return family.deviance(y, mu, w) + float(beta @ sPen @ beta)

    if start is not None:
        beta = np.asarray(start, dtype=float)
        eta = x @ beta + off
        mu = link.inverse(eta)
        if not (family.validMu(mu) and link.validEta(eta)):
            start, beta = None, None
    if start is None:
        mu = family.initialize(y, w)
        eta = link.link(mu)
        beta = None
        # least-squares projection of the starting eta: a fallback for step halving at the first step
        sw0 = np.sqrt(w)
        beta0 = leastSquaresSvd(x * sw0[:, None], (eta - off) * sw0).coef
        eta0 = x @ beta0 + off
        mu0 = link.inverse(eta0)
        if family.validMu(mu0) and link.validEta(eta0):
            beta, eta, mu = beta0, eta0, mu0
    obj = penalizedDeviance(beta, mu) if beta is not None else np.inf
    converged = False
    it = 0
    sol = None
    for it in range(1, maxIter + 1):
        dmu = link.dmuDeta(eta)
        var = family.variance(mu)
        z = eta - off + (y - mu) / dmu
        wIrls = np.maximum(w * dmu * dmu / var, 0.0)
        sw = np.sqrt(wIrls)
        a = np.vstack([x * sw[:, None], rPen])
        b = np.concatenate([z * sw, np.zeros(rPen.shape[0])])
        sol = leastSquaresSvd(a, b)
        betaNew = sol.coef
        # step halving towards the previous iterate
        for _ in range(40):
            etaNew = x @ betaNew + off
            muNew = link.inverse(etaNew)
            if family.validMu(muNew) and link.validEta(etaNew):
                objNew = penalizedDeviance(betaNew, muNew)
                if beta is None or objNew <= obj * (1.0 + 1e-12) + 1e-12:
                    break
            if beta is None:
                raise np.linalg.LinAlgError("P-IRLS produced an invalid mean at the first step")
            betaNew = 0.5 * (betaNew + beta)
        else:
            break
        change = abs(objNew - obj) / (abs(objNew) + 0.1) if np.isfinite(obj) else np.inf
        etaChange = float(np.max(np.abs(etaNew - eta))) / (1.0 + float(np.max(np.abs(etaNew)))) if eta.size else 0.0
        beta, eta, mu, obj = betaNew, etaNew, muNew, objNew
        # the deviance change is quadratic in the gradient; also require a negligible step
        if change < tol and etaChange < 1e-11:
            converged = True
            break
    # final quantities at the converged beta
    dmu = link.dmuDeta(eta)
    wIrls = np.maximum(w * dmu * dmu / family.variance(mu), 0.0)
    sw = np.sqrt(wIrls)
    a = np.vstack([x * sw[:, None], rPen])
    sol = leastSquaresSvd(a, np.zeros(a.shape[0]))
    cov = sol.inverseGram()
    xw = x * sw[:, None]
    hat = np.einsum("ij,jk,ik->i", xw, cov, xw)
    f = cov @ (xw.T @ xw)
    edfTerms = np.diag(f).copy()
    return PirlsResult(coef=beta, eta=eta, mu=mu, weights=wIrls, deviance=family.deviance(y, mu, w),
                       penalty=float(beta @ sPen @ beta), covUnscaled=cov, hatDiag=hat, edf=float(edfTerms.sum()),
                       edfTerms=edfTerms, iterations=it, converged=converged, rank=sol.rank)


def smoothingCriterion(res: PirlsResult, family: Family, n: int, criterion: str, gamma: float = 1.0) -> float:
    """GCV = n D / (n - gamma edf)^2, or UBRE = D / n - 1 + 2 gamma edf / n (phi = 1)."""
    if criterion == "gcv":
        return n * res.deviance / max(n - gamma * res.edf, 1e-8) ** 2
    if criterion == "ubre":
        return res.deviance / n - 1.0 + 2.0 * gamma * res.edf / n
    raise ValueError("criterion must be 'gcv' or 'ubre'")


def selectSmoothing(x, y, w, family: Family, penalties: list[np.ndarray], criterion: str = "auto",
                    offset=None, logLambdaBounds=(-12.0, 12.0), gamma: float = 1.0, maxIter: int = 100,
                    fixed: Optional[list] = None) -> tuple[np.ndarray, PirlsResult, float]:
    """Choose lambda_j for S = sum lambda_j S_j by GCV / UBRE; returns (lambdas, fit, score).

    ``fixed`` may give a value per penalty (None entries are optimized).
    """
    n = x.shape[0]
    if criterion == "auto":
        criterion = "ubre" if family.scaleKnown else "gcv"
    k = len(penalties)
    fixed = [None] * k if fixed is None else list(fixed)
    free = [j for j in range(k) if fixed[j] is None]
    scales = [max(float(np.trace(s)), 1e-300) / max(np.linalg.matrix_rank(s), 1) for s in penalties]
    xScale = float(np.sum(w[:, None] * x * x)) / max(x.shape[1], 1)
    cache = {"start": None}

    def lambdas(logL):
        lam = np.array([float(fixed[j]) if fixed[j] is not None else 0.0 for j in range(k)])
        for i, j in enumerate(free):
            lam[j] = np.exp(logL[i]) * xScale / scales[j]
        return lam

    def fitAt(logL):
        lam = lambdas(logL)
        s = sum(l * sj for l, sj in zip(lam, penalties)) if k else None
        res = pirls(x, y, w, family, s, offset, cache["start"], maxIter)
        cache["start"] = res.coef
        return res

    def score(logL):
        try:
            return smoothingCriterion(fitAt(np.atleast_1d(logL)), family, n, criterion, gamma)
        except np.linalg.LinAlgError:
            return np.inf

    lo, hi = logLambdaBounds
    if not free:
        res = fitAt(np.zeros(0))
        return lambdas(np.zeros(0)), res, smoothingCriterion(res, family, n, criterion, gamma)
    if len(free) == 1:
        grid = np.linspace(lo, hi, 25)
        vals = np.array([score(np.array([g])) for g in grid])
        j = int(np.argmin(vals))
        best = minimizeScalar(lambda t: score(np.array([t])), (grid[max(j - 1, 0)], grid[min(j + 1, grid.size - 1)]),
                              xatol=1e-4)
        logL = np.array([best.x[0] if best.fun <= vals[j] else grid[j]])
    else:
        # coarse common-lambda search, then all lambdas jointly
        grid = np.linspace(lo, hi, 13)
        vals = np.array([score(np.full(len(free), g)) for g in grid])
        start = np.full(len(free), grid[int(np.argmin(vals))])
        res = minimize(score, start, bounds=[(lo, hi)] * len(free), maxIter=200, tol=1e-10)
        logL = res.x.copy()
        best = score(logL)
        # polish: the criterion is often flat in some log lambda (terms heading to their null space);
        # coordinate-wise Brent searches settle those directions precisely
        for _ in range(4):
            before = best
            for i in range(len(free)):
                def along(t, i=i):
                    trial = logL.copy()
                    trial[i] = t
                    return score(trial)
                a, b = max(lo, logL[i] - 3.0), min(hi, logL[i] + 3.0)
                r = minimizeScalar(along, (a, b), xatol=1e-6)
                if r.fun < best:
                    logL[i], best = float(r.x[0]), float(r.fun)
            if before - best <= 1e-12 * abs(before):
                break
    final = fitAt(logL)
    return lambdas(logL), final, smoothingCriterion(final, family, n, criterion, gamma)
