"""Linear quantile regression on any basis.

    beta(tau) = argmin sum_i w_i rho_tau(y_i - Phi(x_i) beta),   rho_tau(u) = u (tau - 1[u < 0])

The problem is solved exactly as the linear program (dual form)

    max  y^T a   s.t.  Phi^T a = (1 - tau) Phi^T 1,   0 <= a <= 1

by a primal-dual interior-point method with Mehrotra predictor-corrector
steps (Frisch-Newton; Portnoy & Koenker 1997); beta is minus the equality
multiplier, finally moved to the exact vertex (the p interpolated points)
as a simplex method would return. Each iteration solves one p x p system,
so n = 10^5 is fast.

Standard errors (``se``):
    "nid"        Hendricks-Koenker local sparsity from fits at tau +/- h (default)
    "iid"        Koenker-Bassett, sparsity from the ranked residuals nearest zero
    "ker"        Powell kernel sandwich
    "bootstrap"  xy-pair bootstrap
with the Hall-Sheather bandwidth h.

References:
    Koenker & Bassett (1978) Econometrica 46(1).
    Portnoy & Koenker (1997) Statistical Science 12(4).
    Koenker (2005) *Quantile Regression*, ch. 3, 6.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.core.FitResult import FitResult
from pythonLibs.regressionHandler.core.Registry import buildComponent, registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase
from pythonLibs.regressionHandler.numerics import SpecialFunctions as _sf

import pythonLibs.regressionHandler.bases  # noqa: F401  (registers bases)


# ---------------------------------------------------------------- solver
def quantileFit(x: np.ndarray, y: np.ndarray, tau: float, weights: Optional[np.ndarray] = None,
                tol: float = 1e-10, maxIter: int = 100) -> tuple[np.ndarray, dict]:
    """Exact weighted linear quantile regression by the Frisch-Newton interior-point method.

    Returns (beta, info) with info = {"iterations", "gap", "dual": a}.
    """
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must be in (0, 1)")
    n, p = x.shape
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    a = (x * w[:, None]).T                                # A (p, n)
    c = -(y * w)
    u = np.ones(n)
    b = (1.0 - tau) * a.sum(axis=1)
    # starting point: feasible primal x = 1 - tau, least-squares multipliers
    xv = np.full(n, 1.0 - tau)
    s = u - xv
    lam = np.linalg.lstsq(a.T, c, rcond=None)[0]
    r = c - a.T @ lam
    delta = max(1e-3, 0.1 * float(np.max(np.abs(r))) if r.size else 1.0)
    z = np.maximum(r, 0.0) + delta
    ww = np.maximum(-r, 0.0) + delta
    gap = float(xv @ z + s @ ww)
    it = 0

    def steplength(v, dv):
        neg = dv < 0
        return float(min(1.0, np.min(-v[neg] / dv[neg]))) if np.any(neg) else 1.0

    for it in range(1, maxIter + 1):
        rp = b - a @ xv
        rd = c - a.T @ lam - z + ww
        theta = 1.0 / (z / xv + ww / s)
        m = (a * theta) @ a.T
        try:
            lower = np.linalg.cholesky(m)
        except np.linalg.LinAlgError:
            lower = np.linalg.cholesky(m + 1e-12 * np.trace(m) / p * np.eye(p))

        def solve(rxz, rsw):
            rho = rd - rxz / xv + rsw / s
            rhs = rp + a @ (theta * rho)
            dlam = np.linalg.solve(lower.T, np.linalg.solve(lower, rhs))
            dx = theta * (a.T @ dlam - rho)
            dz = (rxz - z * dx) / xv
            dw = (rsw + ww * dx) / s
            return dx, dlam, dz, dw

        # predictor (affine scaling)
        dx, dlam, dz, dw = solve(-xv * z, -s * ww)
        ds = -dx
        ap = min(steplength(xv, dx), steplength(s, ds))
        ad = min(steplength(z, dz), steplength(ww, dw))
        mu = gap / (2.0 * n)
        gapAff = float((xv + ap * dx) @ (z + ad * dz) + (s + ap * ds) @ (ww + ad * dw))
        sigma = (gapAff / gap) ** 3 if gap > 0 else 0.0
        # corrector
        dx, dlam, dz, dw = solve(sigma * mu - xv * z - dx * dz, sigma * mu - s * ww - ds * dw)
        ds = -dx
        ap = min(1.0, 0.99995 * min(steplength(xv, dx), steplength(s, ds)))
        ad = min(1.0, 0.99995 * min(steplength(z, dz), steplength(ww, dw)))
        xv = xv + ap * dx
        s = s + ap * ds
        lam = lam + ad * dlam
        z = z + ad * dz
        ww = ww + ad * dw
        gap = float(xv @ z + s @ ww)
        primalObj = float(c @ xv)
        if gap < tol * (1.0 + abs(primalObj)):
            break
    beta = -lam
    # crossover to the exact vertex: the p observations the solution interpolates
    r = y - x @ beta
    basis = np.argsort(np.abs(r))[:p]
    try:
        vertex = np.linalg.solve(x[basis], y[basis])
        loss = lambda b: float(np.sum(w * checkLoss(y - x @ b, tau)))
        if loss(vertex) <= loss(beta) * (1.0 + 1e-12) + 1e-12:
            beta = vertex
    except np.linalg.LinAlgError:
        pass
    return beta, {"iterations": it, "gap": gap, "dual": xv}


def checkLoss(r: np.ndarray, tau: float) -> np.ndarray:
    return r * (tau - (r < 0))


def hallSheather(n: int, tau: float, alpha: float = 0.05) -> float:
    """Hall-Sheather (1988) bandwidth in probability units."""
    q = float(_sf.ndtri(np.array(tau)))
    dens = float(np.exp(-0.5 * q * q) / np.sqrt(2.0 * np.pi))
    z = float(_sf.ndtri(np.array(1.0 - alpha / 2.0)))
    return n ** (-1.0 / 3.0) * z ** (2.0 / 3.0) * (1.5 * dens ** 2 / (2.0 * q * q + 1.0)) ** (1.0 / 3.0)


# ---------------------------------------------------------------- model
@registry("model").register("quantile")
class QuantileModel(SurrogateModelBase):
    """Linear quantile regression (any basis), exact interior-point solution and Koenker-style inference."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("basis", {"type": "polynomial", "degree": 1}, types=(str, dict, object), desc="Basis spec")
        d("tau", 0.5, types=float, lower=1e-6, upper=1 - 1e-6, desc="Quantile level")
        d("se", "nid", values=("nid", "iid", "ker", "bootstrap", "none"), desc="Standard-error method")
        d("nBoot", 200, types=int, lower=10, desc="Bootstrap replicates for se='bootstrap'")
        d("seed", 0, types=int, desc="Bootstrap seed")
        d("tol", 1e-10, types=float, lower=0.0, desc="Relative duality-gap tolerance")
        self.supports.update(variances=True, derivatives=True, parameterInference=True)

    def _validateOptions(self) -> None:
        buildComponent("basis", copy.deepcopy(self.options["basis"]))

    def _train(self) -> None:
        y, w = self.yt[:, 0], self.wt
        tau = float(self.options["tau"])
        self._basis = copy.deepcopy(buildComponent("basis", self.options["basis"])).fit(self.xt)
        phi = self._basis.transform(self.xt)
        n, p = phi.shape
        if p >= n:
            raise ValueError(f"{p} basis terms but only {n} samples")
        self._coef, self._info = quantileFit(phi, y, tau, w, self.options["tol"])
        self._cov = self._covariance(phi, y, w, tau)
        self.result = None if self._cov is None else FitResult(
            parameterNames=self.termNames, params=self._coef[:, None], covariance=self._cov[None],
            dofResid=float(n - p), sigma2=np.array([np.nan]), outputNames=self.outputNames)

    def _covariance(self, phi, y, w, tau) -> Optional[np.ndarray]:
        method = self.options["se"]
        n, p = phi.shape
        if method == "none":
            return None
        resid = y - phi @ self._coef
        xtx = (phi * w[:, None]).T @ phi
        h = hallSheather(n, tau)
        eps = np.finfo(float).eps ** (2.0 / 3.0)
        if np.all(np.abs(resid) <= eps * max(1.0, float(np.max(np.abs(y))))):
            # exact fit (e.g. constant response): the sparsity 1/f(0) is zero / undefined
            self._info["seWarning"] = "all residuals are zero; standard errors are undefined"
            return np.full((p, p), np.nan)
        if method == "bootstrap":
            rng = np.random.default_rng(self.options["seed"])
            draws = []
            for _ in range(self.options["nBoot"]):
                idx = rng.integers(0, n, n)
                try:
                    draws.append(quantileFit(phi[idx], y[idx], tau, w[idx], 1e-8)[0])
                except np.linalg.LinAlgError:
                    continue
            return np.cov(np.array(draws).T).reshape(p, p)
        if method == "iid":
            # sparsity 1/f(0): median-regression slope of the residuals closest to zero on their ranks
            pz = int(np.sum(np.abs(resid) < eps))                        # interpolated points
            hh = max(p + 1, int(np.ceil(n * h)))
            ir = np.arange(pz + 1, min(hh + pz + 1, n) + 1)
            ordResid = np.sort(resid[np.argsort(np.abs(resid), kind="stable")][ir - 1])
            xt = ir / (n - p)
            slope = quantileFit(np.column_stack([np.ones(ir.size), xt]), ordResid, 0.5)[0][1]
            return slope ** 2 * tau * (1.0 - tau) * np.linalg.inv(xtx)
        if method == "ker":
            qlo, qhi = _sf.ndtri(np.array([max(tau - h, 1e-6), min(tau + h, 1 - 1e-6)]))
            iqr = np.quantile(resid, 0.75) - np.quantile(resid, 0.25)
            hn = (qhi - qlo) * min(np.std(resid, ddof=1), iqr / 1.34)
            f = np.exp(-0.5 * (resid / hn) ** 2) / (np.sqrt(2.0 * np.pi) * hn)
        else:                                                   # nid
            lo, hi = max(tau - h, 1e-6), min(tau + h, 1 - 1e-6)
            bHi = quantileFit(phi, y, hi, w, 1e-9)[0]
            bLo = quantileFit(phi, y, lo, w, 1e-9)[0]
            dq = phi @ (bHi - bLo)
            f = np.maximum(0.0, (hi - lo) / (dq - eps))
        hmat = (phi * (w * f)[:, None]).T @ phi
        if np.linalg.matrix_rank(hmat) < np.linalg.matrix_rank(xtx):
            # too few points with a positive density estimate (e.g. nid at an extreme tau, where the
            # fits at tau +/- h coincide): the sandwich would silently return zero standard errors
            self._info["seWarning"] = (f"se='{method}': the local density estimate is degenerate "
                                       "(tau too extreme for this sample size); standard errors are undefined")
            return np.full((p, p), np.nan)
        hinv = np.linalg.pinv(hmat)
        return tau * (1.0 - tau) * hinv @ xtx @ hinv

    # ------------------------------------------------------------------ prediction
    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        return (self._basis.transform(x) @ self._coef)[:, None]

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        if kind != "confidence":
            raise ValueError("quantile regression provides confidence bands of the quantile curve only")
        if self._cov is None:
            raise RuntimeError("fitted with se='none'")
        phi = self._basis.transform(x)
        return np.einsum("ij,jk,ik->i", phi, self._cov, phi)[:, None]

    def predictInterval(self, x, level: float = 0.95, kind: str = "confidence"):
        return super().predictInterval(x, level, kind)

    def _predictDerivatives(self, x: np.ndarray, kx: int) -> np.ndarray:
        return (self._basis.derivative(x, kx) @ self._coef)[:, None]

    def _effectiveParams(self):
        return float(self._coef.size)

    def _intervalDof(self) -> Optional[float]:
        return float(self.nTrain - self._coef.size)

    # ------------------------------------------------------------------ extras
    @property
    def termNames(self) -> list[str]:
        return self._basis.termNames(self.featureNames)

    @property
    def coefficients(self) -> np.ndarray:
        """(p,) coefficients, or {outputName: (p,)} with several outputs."""
        self._checkTrained()
        if self._subModels is not None:
            return {n: m.coefficients for n, m in zip(self.outputNames, self._subModels)}
        return self._coef.copy()

    def objective(self) -> float:
        """Weighted check loss at the solution ({outputName: value} with several outputs)."""
        self._checkTrained()
        if self._subModels is not None:
            return self._eachOutput("objective")
        self._requireTrainingData()
        r = self.yt[:, 0] - self._predictValues(self.xt)[:, 0]
        return float(np.sum(self.wt * checkLoss(r, float(self.options["tau"]))))

    def pseudoR2(self) -> float:
        """Koenker-Machado R1 = 1 - V(tau) / V~(tau) (intercept-only reference)."""
        self._checkTrained()
        if self._subModels is not None:
            return self._eachOutput("pseudoR2")
        self._requireTrainingData()
        y, w, tau = self.yt[:, 0], self.wt, float(self.options["tau"])
        b0, _ = quantileFit(np.ones((y.size, 1)), y, tau, w)
        v0 = float(np.sum(w * checkLoss(y - b0[0], tau)))
        return 1.0 - self.objective() / v0 if v0 > 0 else float("nan")

    # ------------------------------------------------------------------ persistence
    def _stateToDict(self) -> dict:
        return {"basis": self._basis.toDict(), "coef": self._coef.tolist(),
                "cov": None if self._cov is None else self._cov.tolist(),
                "info": {k: self._info[k] for k in ("iterations", "gap", "seWarning") if k in self._info}}

    def _stateFromDict(self, state: dict) -> None:
        self._basis = buildComponent("basis", state["basis"])
        self._coef = np.array(state["coef"], dtype=float)
        self._cov = None if state["cov"] is None else np.array(state["cov"], dtype=float)
        self._info = dict(state["info"])


def quantileProcess(x, y, taus, basis=None, weights=None) -> dict:
    """Coefficients beta(tau) for several quantile levels: {"taus", "coef" (nTau, p), "termNames"}."""
    m = QuantileModel(basis=basis or {"type": "polynomial", "degree": 1}, se="none")
    m.setTrainingValues(x, y, weights)
    m.train()
    phi = m._basis.transform(m.xt)
    coefs = [quantileFit(phi, m.yt[:, 0], float(t), m.wt)[0] for t in taus]
    return {"taus": [float(t) for t in taus], "coef": np.array(coefs), "termNames": m.termNames}
