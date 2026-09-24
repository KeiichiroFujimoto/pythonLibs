"""Solvers for models that are linear in their coefficients: y ~ Phi @ c.

All solvers take the design matrix ``phi (n, p)``, outputs ``y (n, ny)``,
positive sample weights ``w (n,)``, the basis roughness penalty (or None)
and the basis intercept mask, and return a ``SolveResult`` holding, per
output, the coefficients, the unscaled coefficient covariance
``(Phi^T W Phi + alpha P)^-1`` (None when the solver gives no inference),
leverages, effective degrees of freedom and the residual variance.

Solvers:
    ols         SVD least squares with column equilibration and rank detection
    ridge       L2 / smoothness penalty; alpha fixed or chosen by GCV / LOO
                (Reinsch eigen-decomposition: every trial alpha costs O(p))
    elasticNet  L1 + L2 penalty by covariance-updating coordinate descent
    robust      IRLS M-estimation (Huber, Tukey bisquare, Cauchy)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.core.Registry import ComponentBase, registry
from pythonLibs.regressionHandler.numerics.LinearAlgebra import leastSquaresSvd
from pythonLibs.regressionHandler.numerics.Optimizers import minimizeScalar


@dataclass
class SolveResult:
    """Per-output solution of a linear least-squares problem.

    Shapes: coef (p, ny); covUnscaled (ny, p, p) or None; hatDiag (n, ny);
    edf, sse, sigma2, alpha (ny,); weights (n, ny) = final effective weights.
    """
    coef: np.ndarray
    covUnscaled: Optional[np.ndarray]
    hatDiag: Optional[np.ndarray]
    edf: np.ndarray
    sse: np.ndarray
    sigma2: np.ndarray
    rank: int
    conditionNumber: float
    alpha: np.ndarray
    weights: np.ndarray
    info: dict = field(default_factory=dict)

    @property
    def dofResid(self) -> np.ndarray:
        return self.weights.shape[0] - self.edf


class LinearSolverBase(ComponentBase):
    componentKind = "solver"
    usesPenalty = False

    def solve(self, phi: np.ndarray, y: np.ndarray, w: np.ndarray, penalty: Optional[np.ndarray],
              biasMask: np.ndarray) -> SolveResult:
        raise NotImplementedError


def _sigma2(sse: np.ndarray, n: int, edf: np.ndarray) -> np.ndarray:
    dof = n - edf
    return np.where(dof > 0, sse / np.maximum(dof, 1e-12), np.nan)


@registry("solver").register("ols")
class OrdinaryLeastSquares(LinearSolverBase):

    def _declareOptions(self, declare) -> None:
        declare("rcond", None, types=float, lower=0.0, desc="Relative singular-value cutoff (None: eps*max(n,p))")

    def solve(self, phi, y, w, penalty, biasMask) -> SolveResult:
        return olsSolve(phi, y, w, self.options["rcond"])


def olsSolve(phi, y, w, rcond=None) -> SolveResult:
    sw = np.sqrt(w)
    sol = leastSquaresSvd(phi * sw[:, None], y * sw[:, None], rcond)
    coef = sol.coef
    resid = y - phi @ coef
    sse = np.sum(w[:, None] * resid * resid, axis=0)
    hat = np.sum(sol.u * sol.u, axis=1)
    ny = y.shape[1]
    edf = np.full(ny, float(sol.rank))
    cov = np.broadcast_to(sol.inverseGram(), (ny,) + (phi.shape[1],) * 2).copy()
    return SolveResult(coef=coef, covUnscaled=cov, hatDiag=np.repeat(hat[:, None], ny, axis=1), edf=edf,
                       sse=sse, sigma2=_sigma2(sse, phi.shape[0], edf), rank=sol.rank,
                       conditionNumber=sol.conditionNumber, alpha=np.zeros(ny),
                       weights=np.repeat(w[:, None], ny, axis=1))


@registry("solver").register("ridge")
class RidgeSolver(LinearSolverBase):
    """Penalized least squares ``min |W^1/2 (y - Phi c)|^2 + alpha c^T P c``.

    ``penalty="ridge"``: P is diagonal with the (weighted) column variances,
    intercept excluded, so alpha is scale free. ``penalty="smoothness"``: P
    is the basis roughness penalty (P-splines), normalized to the Gram trace.
    ``alpha="gcv"`` / ``"loo"`` minimizes generalized / exact leave-one-out
    cross-validation over log10(alpha) in ``alphaRange`` (grid + Brent).

    The Gram matrix is whitened once and the penalty diagonalized in that
    metric (Reinsch form), after which each trial alpha costs O(n p) instead
    of a new factorization.
    """
    usesPenalty = True

    def _declareOptions(self, declare) -> None:
        declare("alpha", "gcv", values=("gcv", "loo"), types=(int, float),
                desc="Penalty weight (>= 0) or 'gcv' / 'loo' for automatic selection")
        declare("penalty", "ridge", values=("ridge", "smoothness"), desc="Penalty matrix type")
        declare("alphaRange", [-10.0, 6.0], types=list, desc="log10(alpha) search interval")

    def solve(self, phi, y, w, penalty, biasMask) -> SolveResult:
        n, p = phi.shape
        ny = y.shape[1]
        sw = np.sqrt(w)
        b = phi * sw[:, None]
        yw = y * sw[:, None]
        gram = b.T @ b
        pen = self._penalty(gram, phi, w, penalty, biasMask)
        lam, vec = np.linalg.eigh(gram)
        floor = max(float(lam.max()), 1e-300) * 1e-13
        rank = int(np.sum(lam > floor))
        lamC = np.maximum(lam, floor)
        whiten = vec / np.sqrt(lamC)
        m = whiten.T @ pen @ whiten
        s, u = np.linalg.eigh(0.5 * (m + m.T))
        s = np.maximum(s, 0.0)
        t = whiten @ u
        bOrth = b @ t
        bo2 = bOrth * bOrth
        z = bOrth.T @ yw
        coef = np.empty((p, ny))
        cov = np.empty((ny, p, p))
        hat = np.empty((n, ny))
        alphas = np.empty(ny)
        for j in range(ny):
            alphas[j] = self._chooseAlpha(s, z[:, j], yw[:, j], bOrth, bo2)
            d = 1.0 / (1.0 + alphas[j] * s)
            coef[:, j] = t @ (d * z[:, j])
            cov[j] = (t * d) @ t.T
            hat[:, j] = bo2 @ d
        edf = hat.sum(axis=0)
        resid = y - phi @ coef
        sse = np.sum(w[:, None] * resid * resid, axis=0)
        return SolveResult(coef=coef, covUnscaled=cov, hatDiag=hat, edf=edf, sse=sse,
                           sigma2=_sigma2(sse, n, edf), rank=rank,
                           conditionNumber=float(np.sqrt(lamC.max() / lamC.min())), alpha=alphas,
                           weights=np.repeat(w[:, None], ny, axis=1), info={"penalty": self.options["penalty"]})

    def _penalty(self, gram, phi, w, penalty, biasMask) -> np.ndarray:
        if self.options["penalty"] == "smoothness":
            if penalty is None:
                raise ValueError("penalty='smoothness' needs a basis with a roughness penalty (e.g. bspline)")
            tr = np.trace(penalty)
            return penalty * (np.trace(gram) / tr if tr > 0 else 1.0)
        wn = w / w.sum()
        mean = wn @ phi
        var = wn @ (phi - mean) ** 2
        # Constant columns have no variance; keep them unpenalized like the intercept.
        diag = np.where(biasMask | (var <= 0.0), 0.0, var * w.sum())
        return np.diag(diag)

    def _chooseAlpha(self, s, zj, ywj, bOrth, bo2) -> float:
        a = self.options["alpha"]
        if not isinstance(a, str):
            if a < 0:
                raise ValueError("alpha must be non-negative")
            return float(a)
        n = ywj.size

        def score(logA: float) -> float:
            d = 1.0 / (1.0 + (10.0 ** logA) * s)
            r = ywj - bOrth @ (d * zj)
            h = bo2 @ d
            if a == "gcv":
                return float(n * (r @ r) / max(n - h.sum(), 1e-8) ** 2)
            return float(np.mean((r / np.maximum(1.0 - h, 1e-10)) ** 2))

        lo, hi = map(float, self.options["alphaRange"])
        grid = np.linspace(lo, hi, 65)
        values = np.array([score(g) for g in grid])
        k = int(np.nanargmin(values))
        left, right = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
        best = minimizeScalar(score, (left, right), xatol=1e-4)
        return float(10.0 ** (best.x[0] if best.fun <= values[k] else grid[k]))


def _splitIntercept(phi, biasMask):
    """Index of a single constant intercept column, or None."""
    const = [j for j in np.flatnonzero(biasMask) if np.ptp(phi[:, j]) == 0.0]
    return const[0] if len(const) == 1 else None


@registry("solver").register("elasticNet")
class ElasticNetSolver(LinearSolverBase):
    """``min 1/(2 sum w) |W^1/2 (y - Phi c)|^2 + alpha (l1Ratio |c|_1 + (1-l1Ratio)/2 |c|^2)``.

    Columns are standardized internally (weighted), the intercept is left
    unpenalized, and coordinate descent runs on the Gram matrix (O(p^2) per
    sweep) with the usual (alpha, l1Ratio) parameterization. ``l1Ratio=1`` is
    the Lasso. No covariance is reported (post-selection inference is not
    valid); refit the selected terms with ``ols`` for intervals.
    """

    def _declareOptions(self, declare) -> None:
        declare("alpha", 1e-3, types=(int, float), lower=0.0, desc="Overall penalty strength")
        declare("l1Ratio", 1.0, types=(int, float), lower=0.0, upper=1.0, desc="L1 share (1 = Lasso)")
        declare("maxIter", 5000, types=int, lower=1, desc="Maximum coordinate-descent sweeps")
        declare("tol", 1e-9, types=float, lower=0.0, desc="Convergence tolerance on coefficient change")

    def solve(self, phi, y, w, penalty, biasMask) -> SolveResult:
        n, p = phi.shape
        ny = y.shape[1]
        wn = w / w.sum()
        icpt = _splitIntercept(phi, biasMask)
        cols = np.array([j for j in range(p) if j != icpt], dtype=int)
        x = phi[:, cols]
        if icpt is not None:
            xMean = wn @ x
            yMean = wn @ y
        else:
            xMean = np.zeros(x.shape[1])
            yMean = np.zeros(ny)
        xc = x - xMean
        scale = np.sqrt(wn @ (xc * xc))
        scale[scale == 0.0] = 1.0
        xs = xc / scale
        gram = xs.T @ (xs * wn[:, None])
        coef = np.zeros((p, ny))
        a1 = self.options["alpha"] * self.options["l1Ratio"]
        a2 = self.options["alpha"] * (1.0 - self.options["l1Ratio"])
        nIters = []
        for j in range(ny):
            yc = y[:, j] - yMean[j]
            rho = xs.T @ (wn * yc)
            beta = np.zeros(x.shape[1])
            diag = np.diag(gram)
            it = 0
            for it in range(1, self.options["maxIter"] + 1):
                maxDelta = 0.0
                for k in range(beta.size):
                    old = beta[k]
                    r = rho[k] - gram[k] @ beta + diag[k] * old
                    new = np.sign(r) * max(abs(r) - a1, 0.0) / (diag[k] + a2)
                    if new != old:
                        beta[k] = new
                        maxDelta = max(maxDelta, abs(new - old))
                if maxDelta <= self.options["tol"] * max(1.0, float(np.max(np.abs(beta)))):
                    break
            nIters.append(it)
            b = beta / scale
            coef[cols, j] = b
            if icpt is not None:
                coef[icpt, j] = (yMean[j] - xMean @ b) / phi[0, icpt]
        resid = y - phi @ coef
        sse = np.sum(w[:, None] * resid * resid, axis=0)
        edf = np.sum(coef != 0.0, axis=0).astype(float)
        return SolveResult(coef=coef, covUnscaled=None, hatDiag=None, edf=edf, sse=sse,
                           sigma2=_sigma2(sse, n, edf), rank=int(edf.max()), conditionNumber=float("nan"),
                           alpha=np.full(ny, self.options["alpha"]), weights=np.repeat(w[:, None], ny, axis=1),
                           info={"iterations": nIters})


_ROBUST_DEFAULT_C = {"huber": 1.345, "bisquare": 4.685, "cauchy": 2.385}


def robustWeights(u: np.ndarray, loss: str) -> np.ndarray:
    """IRLS weights psi(u)/u for standardized residuals u (tuning constant applied)."""
    a = np.abs(u)
    if loss == "huber":
        return np.where(a <= 1.0, 1.0, 1.0 / np.maximum(a, 1e-300))
    if loss == "bisquare":
        return np.where(a < 1.0, (1.0 - a * a) ** 2, 0.0)
    if loss == "cauchy":
        return 1.0 / (1.0 + a * a)
    raise ValueError("loss must be huber, bisquare or cauchy")


@registry("solver").register("robust")
class RobustSolver(LinearSolverBase):
    """M-estimation by iteratively reweighted least squares.

    Residual scale is re-estimated each iteration by the normalized median
    absolute deviation; tuning constants default to 95 % Gaussian efficiency
    (Huber 1.345, Tukey bisquare 4.685, Cauchy 2.385). Bisquare starts from
    the Huber solution because its objective is non-convex.
    """

    def _declareOptions(self, declare) -> None:
        declare("loss", "huber", values=tuple(_ROBUST_DEFAULT_C), desc="M-estimator")
        declare("c", None, types=(int, float), lower=0.0, desc="Tuning constant (None: 95 % efficiency default)")
        declare("maxIter", 100, types=int, lower=1, desc="Maximum IRLS iterations")
        declare("tol", 1e-10, types=float, lower=0.0, desc="Relative coefficient-change tolerance")
        declare("outlierThreshold", 3.0, types=(int, float), lower=0.0,
                desc="|residual| / robust scale above which a point is flagged as outlier")

    def solve(self, phi, y, w, penalty, biasMask) -> SolveResult:
        n, p = phi.shape
        ny = y.shape[1]
        loss = self.options["loss"]
        c = self.options["c"] or _ROBUST_DEFAULT_C[loss]
        parts = []
        effW = np.empty((n, ny))
        scales = np.empty(ny)
        iters = []
        for j in range(ny):
            yj = y[:, [j]]
            res = olsSolve(phi, yj, w)
            stages = ["huber", loss] if loss == "bisquare" else [loss]
            it = 0
            wEff = w
            for stage in stages:
                cStage = c if stage == loss else _ROBUST_DEFAULT_C["huber"]
                for it in range(1, self.options["maxIter"] + 1):
                    r = (yj[:, 0] - phi @ res.coef[:, 0]) * np.sqrt(w)
                    scale = 1.4826 * np.median(np.abs(r - np.median(r)))
                    if scale <= 0.0:
                        break
                    wEff = w * robustWeights(r / (cStage * scale), stage)
                    wEff = np.maximum(wEff, 1e-12 * w)
                    new = olsSolve(phi, yj, wEff)
                    delta = np.max(np.abs(new.coef - res.coef)) / max(1.0, float(np.max(np.abs(new.coef))))
                    res = new
                    if delta <= self.options["tol"]:
                        break
            r = (yj[:, 0] - phi @ res.coef[:, 0]) * np.sqrt(w)
            scales[j] = 1.4826 * np.median(np.abs(r - np.median(r)))
            effW[:, j] = wEff
            iters.append(it)
            parts.append(res)
        coef = np.hstack([pt.coef for pt in parts])
        resid = y - phi @ coef
        sse = np.sum(w[:, None] * resid * resid, axis=0)
        edf = np.array([pt.edf[0] for pt in parts])
        # Inference from the final weighted fit (asymptotic M-estimation covariance
        # approximated by the weighted least-squares covariance).
        sigma2 = np.array([pt.sigma2[0] for pt in parts])
        outliers = np.abs(resid * np.sqrt(w)[:, None]) > self.options["outlierThreshold"] * np.maximum(scales, 1e-300)
        return SolveResult(coef=coef, covUnscaled=np.concatenate([pt.covUnscaled for pt in parts]),
                           hatDiag=np.hstack([pt.hatDiag for pt in parts]), edf=edf, sse=sse, sigma2=sigma2,
                           rank=min(pt.rank for pt in parts),
                           conditionNumber=max(pt.conditionNumber for pt in parts), alpha=np.zeros(ny),
                           weights=effW, info={"iterations": iters, "robustScale": scales.tolist(),
                                               "outlierMask": outliers})


def larsOrder(x: np.ndarray, y: np.ndarray, maxSteps: int) -> list[int]:
    """Order in which columns enter the least-angle regression path (Efron et al. 2004).

    x must be centred with unit-norm columns and y centred.
    """
    n, p = x.shape
    active: list[int] = []
    mu = np.zeros(n)
    limit = min(maxSteps, p, n - 1)
    while len(active) < limit:
        c = x.T @ (y - mu)
        inactive = np.array([j for j in range(p) if j not in active], dtype=int)
        if inactive.size == 0:
            break
        if not active:
            active.append(int(inactive[np.argmax(np.abs(c[inactive]))]))
            if len(active) >= limit:
                break
        cMax = float(np.max(np.abs(c[active])))
        if cMax <= 1e-14 * max(1.0, float(np.linalg.norm(y))):
            break
        s = np.sign(c[active])
        xa = x[:, active] * s
        g = xa.T @ xa
        try:
            gi1 = np.linalg.solve(g, np.ones(len(active)))
        except np.linalg.LinAlgError:
            break
        aa = 1.0 / np.sqrt(max(float(np.sum(gi1)), 1e-300))
        u = xa @ (aa * gi1)
        a = x.T @ u
        inactive = np.array([j for j in range(p) if j not in active], dtype=int)
        if inactive.size == 0:
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            g1 = (cMax - c[inactive]) / (aa - a[inactive])
            g2 = (cMax + c[inactive]) / (aa + a[inactive])
        cand = np.concatenate([g1, g2])
        idx = np.concatenate([inactive, inactive])
        ok = np.isfinite(cand) & (cand > 1e-14)
        if not np.any(ok):
            break
        k = int(np.argmin(np.where(ok, cand, np.inf)))
        mu = mu + cand[k] * u
        active.append(int(idx[k]))
    return active


@registry("solver").register("lars")
class LarsSolver(LinearSolverBase):
    """Sparse least squares: LARS variable ordering + OLS refits, best subset by (corrected) LOO error.

    The least-angle path fixes the order in which terms enter; each prefix of
    that order is refitted by ordinary least squares (hybrid LARS) and scored
    by its exact leave-one-out error, optionally with the small-sample
    correction T = n / (n - k) (1 + tr((Phi^T Phi / n)^-1) / n) (Blatman &
    Sudret 2011). The intercept is always kept. This is the standard way to
    fit sparse polynomial chaos expansions with fewer samples than terms.
    """

    def _declareOptions(self, declare) -> None:
        declare("maxTerms", None, types=int, lower=1, desc="Largest number of selected terms (default n - 2)")
        declare("criterion", "looCorrected", values=("loo", "looCorrected"), desc="Model-size criterion")

    def solve(self, phi, y, w, penalty, biasMask) -> SolveResult:
        n, p = phi.shape
        ny = y.shape[1]
        sw = np.sqrt(w)
        icpt = _splitIntercept(phi, biasMask)
        cols = np.array([j for j in range(p) if j != icpt], dtype=int)
        coef = np.zeros((p, ny))
        cov = np.zeros((ny, p, p))
        hat = np.zeros((n, ny))
        edf, sse = np.zeros(ny), np.zeros(ny)
        selected, paths = [], []
        maxK = self.options["maxTerms"] or max(1, n - 2 - (icpt is not None))
        for j in range(ny):
            xw = phi[:, cols] * sw[:, None]
            yw = y[:, j] * sw
            if icpt is not None:
                ones = sw / np.linalg.norm(sw)
                xc = xw - np.outer(ones, ones @ xw)
                yc = yw - ones * (ones @ yw)
            else:
                xc, yc = xw, yw
            norms = np.linalg.norm(xc, axis=0)
            usable = norms > 1e-12 * max(1.0, float(norms.max()) if norms.size else 1.0)
            xs = np.where(usable, xc / np.where(usable, norms, 1.0), 0.0)
            order = [int(cols[k]) for k in larsOrder(xs, yc, maxK) if usable[k]]
            base = [icpt] if icpt is not None else []
            best, bestScore, path, fits = None, np.inf, [], []
            for k in range(0, len(order) + 1):
                terms = base + order[:k]
                if not terms or len(terms) >= n:
                    continue
                sol = leastSquaresSvd(phi[:, terms] * sw[:, None], yw)
                if sol.rank < len(terms):
                    continue
                r = yw - (phi[:, terms] * sw[:, None]) @ sol.coef
                h = np.sum(sol.u * sol.u, axis=1)
                if np.any(h > 1.0 - 1e-10):
                    continue
                loo = float(np.mean((r / (1.0 - h)) ** 2))
                if self.options["criterion"] == "looCorrected":
                    gram = (phi[:, terms] * sw[:, None]).T @ (phi[:, terms] * sw[:, None]) / n
                    corr = n / (n - len(terms)) * (1.0 + float(np.trace(np.linalg.pinv(gram))) / n)
                    loo *= corr
                path.append(loo)
                fits.append((loo, (terms, sol, h, r)))
            if not fits:
                raise np.linalg.LinAlgError("LARS found no well-posed subset")
            # smallest model whose score is within round-off of the best (exact fits tie at ~0)
            bestScore = min(f[0] for f in fits)
            tolScore = bestScore + 1e-10 * float(np.mean(yc * yc) + 1e-300)
            terms, sol, h, r = next(f[1] for f in fits if f[0] <= tolScore)
            coef[terms, j] = sol.coef
            g = sol.inverseGram()
            cov[j][np.ix_(terms, terms)] = g
            hat[:, j] = h
            edf[j] = len(terms)
            sse[j] = float(r @ r)
            selected.append(terms)
            paths.append(path)
        return SolveResult(coef=coef, covUnscaled=cov, hatDiag=hat, edf=edf, sse=sse, sigma2=_sigma2(sse, n, edf),
                           rank=int(edf.max()), conditionNumber=float("nan"), alpha=np.zeros(ny),
                           weights=np.repeat(w[:, None], ny, axis=1), info={"selected": selected, "looPath": paths})
