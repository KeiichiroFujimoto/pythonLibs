"""Orthogonal distance regression: nonlinear fits with errors in the inputs as well as the output.

    minimize  sum_i [ (y_i - f(x_i + delta_i; beta))^2 / sy^2 + sum_k delta_ik^2 / sx_k^2 ]

over beta and the input corrections delta. For fixed beta every delta_i is
an independent nx-dimensional problem, solved for all points at once by
Gauss-Newton with the closed-form (Sherman-Morrison) inverse of
D + g g^T / sy^2; beta then minimizes the profiled sum of squares by
Levenberg-Marquardt. Memory and time are O(n nx) per evaluation, so large n
is fine. A straight line f = a + b x with sx / sy fixed is Deming regression
(sx = sy: orthogonal / total least squares).

Standard errors follow the ODRPACK convention: the Gauss-Newton information
of beta with the corrections eliminated (effective-variance weights) scaled
by s^2 = SS / (n - p), SS the minimized weighted sum of squares.

References:
    Boggs, Byrd & Schnabel (1987) SIAM J. Sci. Stat. Comput. 8(6).
    Deming (1943) *Statistical Adjustment of Data*.
"""
from __future__ import annotations

import numpy as np

from pythonLibs.regressionHandler.core.FitResult import FitResult
from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.models.NonlinearModel import NonlinearModel
from pythonLibs.regressionHandler.numerics.LinearAlgebra import leastSquaresSvd
from pythonLibs.regressionHandler.numerics.Optimizers import leastSquares


@registry("model").register("odr")
class OdrModel(NonlinearModel):
    """Nonlinear orthogonal distance regression (errors in x and y); expression / library / callable models."""

    def _initialize(self) -> None:
        super()._initialize()
        d = self.options.declare
        d("xSigma", 1.0, types=(int, float, list), desc="Standard deviation of the input errors (scalar or per input)")
        d("ySigma", 1.0, types=(int, float), lower=0.0, desc="Standard deviation of the output errors")
        d("innerIter", 30, types=int, lower=1, desc="Gauss-Newton iterations for the input corrections")
        self.supports.update(weights=True)

    # ------------------------------------------------------------------ inner problem
    def _xSig(self) -> np.ndarray:
        s = np.asarray(self.options["xSigma"], dtype=float)
        s = np.full(self.nx, float(s)) if s.ndim == 0 else s
        if s.size != self.nx or np.any(s <= 0):
            raise ValueError("xSigma must be positive, scalar or one value per input")
        return s

    def _gradX(self, x, p) -> np.ndarray:
        """df/dx (n, nx) by central differences."""
        g = np.empty_like(x)
        for k in range(self.nx):
            h = 1e-6 * np.maximum(np.abs(x[:, k]), 1.0)
            xp, xm = x.copy(), x.copy()
            xp[:, k] += h
            xm[:, k] -= h
            g[:, k] = (self._evaluate(xp, p) - self._evaluate(xm, p)) / (2.0 * h)
        return g

    def _corrections(self, x, y, w, p, delta0=None) -> np.ndarray:
        """Optimal delta (n, nx) for fixed parameters."""
        dInv2 = 1.0 / self._xSig() ** 2                       # D = diag(1 / sx^2)
        wy = w / float(self.options["ySigma"]) ** 2
        delta = np.zeros_like(x) if delta0 is None else delta0.copy()
        for _ in range(self.options["innerIter"]):
            xd = x + delta
            e = self._evaluate(xd, p) - y
            g = self._gradX(xd, p)
            # solve (D + wy g g^T) step = -(wy e g + D delta) per point (Sherman-Morrison)
            rhs = -(wy[:, None] * e[:, None] * g + dInv2 * delta)
            dInvRhs = rhs / dInv2
            dInvG = g / dInv2
            denom = 1.0 + wy * np.sum(g * dInvG, axis=1)
            step = dInvRhs - dInvG * (wy * np.sum(g * dInvRhs, axis=1) / denom)[:, None]
            delta = delta + step
            if float(np.max(np.abs(step))) <= 1e-12 * (1.0 + float(np.max(np.abs(delta)))):
                break
        return delta

    def _profiledResiduals(self, x, y, w, p):
        delta = self._corrections(x, y, w, p, getattr(self, "_deltaWarm", None))
        self._deltaWarm = delta
        ry = (self._evaluate(x + delta, p) - y) * np.sqrt(w) / float(self.options["ySigma"])
        rx = (delta / self._xSig()).ravel()
        return np.concatenate([ry, rx])

    # ------------------------------------------------------------------ training
    def _train(self) -> None:
        x, y, w = self.xt, self.yt[:, 0], self.wt
        self._varNames = self._variableNames()
        if self.options["expression"] is not None and self.options["function"] is None:
            self._compile(self._varNames)
        lb, ub = self._boundsArrays()
        p0 = np.clip(self._initialGuess(x, y, lb, ub), lb, ub)
        # ordinary least squares first: a good start for the orthogonal fit
        sw = np.sqrt(w)
        olsFit = leastSquares(lambda p: (self._evaluate(x, p) - y) * sw, p0, bounds=(lb, ub), ftol=1e-12,
                              xtol=1e-12, gtol=1e-12)
        self._deltaWarm = None
        best = leastSquares(lambda p: self._profiledResiduals(x, y, w, p), olsFit.x, bounds=(lb, ub),
                            ftol=1e-13, xtol=1e-13, gtol=1e-13)
        self._opt = best
        self._p = best.x.copy()
        self._delta = self._corrections(x, y, w, self._p)
        self._inference(x, y, w)

    def _inference(self, x, y, w) -> None:
        """Gauss-Newton information of beta with the corrections profiled out (Schur complement).

        Per point the delta block is eliminated in closed form, which leaves the
        effective-variance weights w_eff = wy / (1 + wy sum_k g_k^2 sx_k^2):
        Info = sum_i w_eff,i j_i j_i^T with j_i = df/dbeta at x_i + delta_i.
        """
        n, k = x.shape[0], self._p.size
        xd = x + self._delta
        wy = w / float(self.options["ySigma"]) ** 2
        g = self._gradX(xd, self._p)
        wEff = wy / (1.0 + wy * np.sum(g * g * self._xSig() ** 2, axis=1))
        jac = self._paramJacobian(xd, self._p) * np.sqrt(wEff)[:, None]
        sol = leastSquaresSvd(jac, np.zeros(n))
        self._rank = sol.rank
        res = self._profiledResiduals(x, y, w, self._p)
        ss = float(res @ res)
        self._dof = n - k
        self._residualVariance = ss / self._dof if self._dof > 0 else float("nan")
        self._covUnscaled = sol.inverseGram()
        self._sigma2 = self._residualVariance
        self.result = FitResult(parameterNames=list(self.options["params"]), params=self._p.reshape(-1, 1),
                                covariance=(self._covUnscaled * self._sigma2)[None], dofResid=float(self._dof),
                                sigma2=np.array([self._sigma2]), outputNames=self.outputNames)

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        g = self._paramJacobian(x, self._p)
        var = np.einsum("ij,jk,ik->i", g, self._covUnscaled, g) * self._sigma2
        if kind == "prediction":
            var = var + self._sigma2 * float(self.options["ySigma"]) ** 2
        return var[:, None]

    @property
    def inputCorrections(self) -> np.ndarray:
        """Estimated input errors delta (n, nx) of the training points."""
        self._checkTrained()
        return self._delta.copy()

    def _stateToDict(self) -> dict:
        state = super()._stateToDict()
        state["delta"] = self._delta.tolist()
        return state

    def _stateFromDict(self, state: dict) -> None:
        super()._stateFromDict(state)
        self._delta = np.array(state["delta"], dtype=float)
