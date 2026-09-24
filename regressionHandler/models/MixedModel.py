"""Linear mixed-effects model with one grouping factor (random intercepts and slopes).

    y_ij = Phi(x_ij) beta + z_ij^T b_i + eps_ij,   b_i ~ N(0, sigma^2 L L^T),   eps ~ N(0, sigma^2)

z_ij = [1, x_ij[randomSlopes]]. The (restricted) likelihood is profiled over
beta and sigma^2; per group the Woodbury identity reduces V_i^-1 to a q x q
solve, and all groups are handled as one batched solve, so an evaluation
costs O(n p^2 + G q^3). The relative Cholesky factor L is optimized with
bounds (diagonal >= 0, allowing a zero variance component on the boundary).

Predictions for rows whose group was seen in training include the group's
BLUP; unseen groups get the population mean. Variances of a known group are
Henderson's prediction-error variance (conditional covariance of the group
effects plus the fixed-effect uncertainty including the beta / BLUP cross
term); a new group adds the prior sigma^2 z^T L L^T z; prediction variances
also add sigma^2.

References:
    Laird & Ware (1982) Biometrics 38(4).
    Bates, Maechler, Bolker & Walker (2015) J. Statistical Software 67(1).
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.core.FitResult import FitResult
from pythonLibs.regressionHandler.core.Registry import buildComponent, registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase
from pythonLibs.regressionHandler.numerics.Optimizers import minimize, multiStart

import pythonLibs.regressionHandler.bases  # noqa: F401  (registers bases)


@registry("model").register("mixed")
class MixedModel(SurrogateModelBase):
    """Linear mixed model: fixed-effect basis + random intercept / slopes per group (REML or ML)."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("groupColumn", -1, types=int, desc="Input column with the group label (integer or float codes)")
        d("basis", {"type": "polynomial", "degree": 1}, types=(str, dict, object),
          desc="Fixed-effect basis on the other inputs")
        d("randomSlopes", [], types=list, desc="Input columns with random slopes (random intercept always)")
        d("reml", True, types=bool, desc="Restricted maximum likelihood (else ML)")
        d("nStart", 3, types=int, lower=1, desc="Optimizer starts")
        self.supports.update(variances=True, parameterInference=True, weights=False, derivatives=False)

    def _validateOptions(self) -> None:
        buildComponent("basis", copy.deepcopy(self.options["basis"]))

    # ------------------------------------------------------------------ layout
    def _gc(self) -> int:
        return self.options["groupColumn"] % self.nx

    def _fixedInputs(self, x):
        gc = self._gc()
        return x[:, [j for j in range(self.nx) if j != gc]]

    def _zRows(self, x):
        return np.column_stack([np.ones(x.shape[0])] + [x[:, int(c) % self.nx] for c in self.options["randomSlopes"]])

    # ------------------------------------------------------------------ likelihood
    def _lower(self, theta) -> np.ndarray:
        l = np.zeros((self._q, self._q))
        l[np.tril_indices(self._q)] = theta
        return l

    def _core(self, theta):
        """Profiled pieces for relative factor theta."""
        l = self._lower(theta)
        zl = self._ztz @ l                                            # (G, q, q)
        m = np.einsum("ij,gjk->gik", l.T, zl) + np.eye(self._q)      # I + L^T Z^T Z L
        sign, logdet = np.linalg.slogdet(m)
        a = np.einsum("ij,gjp->gip", l.T, self._ztx)                  # L^T Z^T X   (G, q, p)
        c = np.einsum("ij,gj->gi", l.T, self._zty)                    # L^T Z^T y   (G, q)
        mInvA = np.linalg.solve(m, a)
        mInvC = np.linalg.solve(m, c[:, :, None])[:, :, 0]
        xvx = self._xtx - np.einsum("gip,giq->pq", a, mInvA)
        xvy = self._xty - np.einsum("gip,gi->p", a, mInvC)
        yvy = self._yty - float(np.einsum("gi,gi->", c, mInvC))
        return l, m, logdet, xvx, xvy, yvy

    def _objective(self, theta) -> float:
        try:
            l, m, logdet, xvx, xvy, yvy = self._core(np.asarray(theta, dtype=float))
            ch = np.linalg.cholesky(xvx)
        except np.linalg.LinAlgError:
            return 1e20
        beta = np.linalg.solve(xvx, xvy)
        r = max(yvy - float(beta @ xvy), 1e-300)
        n, p = self._n, self._p
        if self.options["reml"]:
            s2 = r / (n - p)
            return float((n - p) * np.log(2 * np.pi * s2) + np.sum(logdet) + 2.0 * np.sum(np.log(np.diag(ch)))
                         + (n - p))
        s2 = r / n
        return float(n * np.log(2 * np.pi * s2) + np.sum(logdet) + n)

    # ------------------------------------------------------------------ training
    def _train(self) -> None:
        x, y = self.xt, self.yt[:, 0]
        groups = x[:, self._gc()]
        self._levels, gidx = np.unique(groups, return_inverse=True)
        gidx = gidx.ravel()
        fixed = self._fixedInputs(x)
        # only the group column given: intercept-only fixed part
        self._basis = copy.deepcopy(buildComponent("basis", self.options["basis"])).fit(fixed) if fixed.shape[1] else None
        phi = self._fixedDesign(x)
        z = self._zRows(x)
        self._n, self._p = phi.shape
        self._q = z.shape[1]
        g = self._levels.size
        if g < 2:
            raise ValueError("a mixed model needs at least two groups")
        self._ztz = np.zeros((g, self._q, self._q))
        np.add.at(self._ztz, gidx, z[:, :, None] * z[:, None, :])
        self._ztx = np.zeros((g, self._q, self._p))
        np.add.at(self._ztx, gidx, z[:, :, None] * phi[:, None, :])
        self._zty = np.zeros((g, self._q))
        np.add.at(self._zty, gidx, z * y[:, None])
        self._xtx, self._xty, self._yty = phi.T @ phi, phi.T @ y, float(y @ y)
        nt = self._q * (self._q + 1) // 2
        rows, cols = np.tril_indices(self._q)
        bounds = [(0.0, 1e3) if r == c else (-1e3, 1e3) for r, c in zip(rows, cols)]
        starts = []
        for s in (1.0, 0.3, 3.0)[: self.options["nStart"]]:
            t0 = np.zeros(nt)
            t0[rows == cols] = s
            starts.append(t0)
        self._opt = multiStart(lambda t: minimize(self._objective, t, bounds=bounds, maxIter=500, tol=1e-12),
                               np.array(starts))
        self._theta = self._opt.x
        self._finalize(phi, z, y, gidx)

    def _finalize(self, phi, z, y, gidx) -> None:
        l, m, logdet, xvx, xvy, yvy = self._core(self._theta)
        self._beta = np.linalg.solve(xvx, xvy)
        r = yvy - float(self._beta @ xvy)
        n, p = self._n, self._p
        self._sigma2 = r / (n - p if self.options["reml"] else n)
        self._covBeta = self._sigma2 * np.linalg.inv(xvx)
        self._psi = self._sigma2 * (l @ l.T)                         # random-effect covariance
        resid = y - phi @ self._beta
        ztr = np.zeros((self._levels.size, self._q))
        np.add.at(ztr, gidx, z * resid[:, None])
        c = np.einsum("ij,gj->gi", l.T, ztr)
        self._blup = np.einsum("ij,gj->gi", l, np.linalg.solve(m, c[:, :, None])[:, :, 0])
        # conditional covariance of each group's effects given the data: sigma2 L M_g^-1 L^T
        self._condCov = self._sigma2 * np.einsum("ij,gjk,lk->gil", l, np.linalg.inv(m), l)
        # BLUP sensitivity to beta, d b_g / d beta = -L M_g^-1 L^T Z_g^T X_g: gives the beta / b_g cross covariance
        self._blupX = np.einsum("ij,gjp->gip", l, np.linalg.solve(m, np.einsum("ij,gjp->gip", l.T, self._ztx)))
        self._logLik = -0.5 * self._objective(self._theta)
        self.result = FitResult(parameterNames=self._termNames(), params=self._beta[:, None],
            covariance=self._covBeta[None], dofResid=float(n - p), sigma2=np.array([self._sigma2]),
            outputNames=self.outputNames)

    # ------------------------------------------------------------------ prediction
    def _groupIndex(self, x):
        g = x[:, self._gc()]
        pos = np.searchsorted(self._levels, g)
        pos = np.clip(pos, 0, self._levels.size - 1)
        known = self._levels[pos] == g
        return pos, known

    def _fixedDesign(self, x) -> np.ndarray:
        if self._basis is None:
            return np.ones((x.shape[0], 1))
        return self._basis.transform(self._fixedInputs(x))

    def _termNames(self) -> list:
        if self._basis is None:
            return ["1"]
        return self._basis.termNames([f for j, f in enumerate(self.featureNames) if j != self._gc()])

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        mean = self._fixedDesign(x) @ self._beta
        pos, known = self._groupIndex(x)
        z = self._zRows(x)
        mean = mean + np.where(known, np.sum(z * self._blup[pos], axis=1), 0.0)
        return mean[:, None]

    def predictPopulation(self, x) -> np.ndarray:
        """Fixed-effect (population-level) prediction, ignoring group effects."""
        self._checkTrained()
        xv = self._validX(x)
        return self._fixedDesign(xv) @ self._beta

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        phi = self._fixedDesign(x)
        pos, known = self._groupIndex(x)
        z = self._zRows(x)
        # known groups: Henderson prediction-error variance of x^T beta + z^T b_g, which includes the
        # beta / BLUP cross covariance, (x - A_g^T z)^T Cov(beta) (x - A_g^T z) + z^T condCov_g z;
        # new groups: Cov(beta) plus the full prior of b
        phiEff = np.where(known[:, None], phi - np.einsum("ij,ijp->ip", z, self._blupX[pos]), phi)
        var = np.einsum("ij,jk,ik->i", phiEff, self._covBeta, phiEff)
        groupVar = np.where(known, np.einsum("ij,ijk,ik->i", z, self._condCov[pos], z),
                            np.einsum("ij,jk,ik->i", z, self._psi, z))
        if kind == "prediction":
            var = var + self._sigma2 + groupVar
        else:
            var = var + np.where(known, groupVar, 0.0)
        return var[:, None]

    def _effectiveParams(self):
        return float(self._p)

    def _intervalDof(self) -> Optional[float]:
        return float(self._n - self._p)

    # ------------------------------------------------------------------ reporting
    def varianceComponents(self) -> dict:
        """Random-effect covariance, residual variance, intra-class correlation and log-likelihood."""
        self._checkTrained()
        names = ["(Intercept)"] + [self.featureNames[int(c) % self.nx] for c in self.options["randomSlopes"]]
        return {"randomEffects": names, "covariance": self._psi.tolist(),
                "std": np.sqrt(np.diag(self._psi)).tolist(), "residualVariance": self._sigma2,
                "icc": float(self._psi[0, 0] / (self._psi[0, 0] + self._sigma2)),
                "logLikelihood": self._logLik, "criterion": "REML" if self.options["reml"] else "ML",
                "nGroups": int(self._levels.size)}

    def randomEffects(self) -> dict:
        """BLUPs per group: {group: [intercept, slopes...]}."""
        self._checkTrained()
        return {float(g): b.tolist() for g, b in zip(self._levels, self._blup)}

    def _stateToDict(self) -> dict:
        return {"basis": None if self._basis is None else self._basis.toDict(), "condCov": self._condCov.tolist(),
                "blupX": self._blupX.tolist(),
                "levels": self._levels.tolist(), "beta": self._beta.tolist(),
                "covBeta": self._covBeta.tolist(), "psi": self._psi.tolist(), "sigma2": self._sigma2,
                "blup": self._blup.tolist(), "theta": self._theta.tolist(), "logLik": self._logLik,
                "n": self._n, "p": self._p, "q": self._q}

    def _stateFromDict(self, state: dict) -> None:
        self._basis = None if state["basis"] is None else buildComponent("basis", state["basis"])
        for k in ("levels", "beta", "covBeta", "psi", "blup", "theta", "condCov", "blupX"):
            setattr(self, "_" + k, np.array(state[k], dtype=float))
        self._sigma2, self._logLik = float(state["sigma2"]), float(state["logLik"])
        self._n, self._p, self._q = int(state["n"]), int(state["p"]), int(state["q"])
