"""Physics-preserving regression: any model made to satisfy linear physical constraints exactly.

The fitted model f is corrected by the smallest change, measured in the
metric of its own uncertainty, that satisfies the constraints (see
``constraints.Functionals``: point values, derivatives, integrals / means,
balances across outputs, bounds, monotonicity, convexity):

    minimize (f - m)^T Sigma^-1 (f - m)   subject to   L_E f = c_E,  L_I f >= c_I

m and Sigma are the model's mean and joint covariance. The dual is a small
bounded QP in the multipliers (one per constraint), solved exactly; the
corrected mean is m(x) + Sigma(x, P) L^T lambda. Three routes:

    "linear"    linear-basis models: the correction stays in the basis. It is
                the constrained generalized least-squares estimator
                beta_c = beta + Cov L^T lambda; derivatives are exact, the
                coefficient covariance is projected on the active constraints
                (Cov - Cov G^T (G Cov G^T)^-1 G Cov) and intervals shrink
                accordingly.
    "gaussian"  models with a joint posterior covariance (Kriging family):
                conditioning of the posterior on the active constraints, i.e.
                the constrained posterior mean / mode with exact conditional
                variances.
    "kernel"    any other model: minimum-norm correction in a squared-
                exponential reproducing-kernel space (``correctionKernel``);
                the model's own variances are passed through.

Equalities hold exactly (to round-off); inequalities hold at their points
(use a grid dense enough for the shape you need). ``constraintReport()`` lists
each constraint's value, residual, multiplier and whether it is active.

    ConstrainedModel(model="quadratic", constraints=[
        {"type": "integral", "value": 1.0, "box": {"lower": [0], "upper": [1]}},
        {"type": "bound", "points": grid, "lower": 0.0}])
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.constraints.Functionals import buildConstraints, expand
from pythonLibs.regressionHandler.constraints.QuadraticProgramming import boundedQp
from pythonLibs.regressionHandler.core.FitResult import FitResult
from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase

_CHUNK = 512


def _pinvSym(a: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    w, v = np.linalg.eigh(0.5 * (a + a.T))
    keep = w > rtol * max(float(w.max()) if w.size else 0.0, 1e-300)
    return (v[:, keep] / w[keep]) @ v[:, keep].T


@registry("model").register("constrained")
class ConstrainedModel(SurrogateModelBase):
    """Any regression model corrected to satisfy linear physical constraints exactly."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("model", "quadratic", types=(str, dict, object), desc="Base model spec (any model)")
        d("constraints", [], types=list, desc="Constraint specs (see constraints.Functionals)")
        d("route", "auto", values=("auto", "linear", "gaussian", "kernel"),
          desc="Correction route (auto: linear for linear-basis models, gaussian with a joint covariance, else kernel)")
        d("correctionKernel", {"lengthscale": 0.3}, types=dict,
          desc="Squared-exponential correction kernel for the kernel route (lengthscale in input-range units)")
        d("stepFraction", 1e-3, types=float, lower=1e-8, upper=0.1,
          desc="Finite-difference step of derivative constraints, relative to the input range")
        self.supports.update(multiOutput=True, weights=True, variances=True, derivatives=True, covariance=True,
                             parameterInference=True)

    def _validateOptions(self) -> None:
        buildConstraints(self.options["constraints"])

    # ------------------------------------------------------------------ training
    def _train(self) -> None:
        from pythonLibs.regressionHandler.models.ModelFactory import createModel
        base = createModel(copy.deepcopy(self.options["model"]))
        w = None if np.all(self.wt == 1.0) else self.wt
        base.fit(self.xt, self.yt, w, self.featureNames, self.outputNames)
        self._base = base
        self._setup()

    def _setup(self) -> None:
        base = self._base
        self._constraints = buildConstraints(self.options["constraints"])
        span = np.ptp(base.xt, axis=0) if base.xt is not None else base._xRange
        span = np.where(np.asarray(span) > 0, span, 1.0)
        self._span = np.asarray(span, dtype=float)
        route = self.options["route"]
        from pythonLibs.regressionHandler.models.LinearBasisModel import LinearBasisModel
        isLinear = isinstance(base, LinearBasisModel) and base.supports["covariance"]
        hasCov = base.supports.get("covariance", False)
        if route == "auto":
            route = "linear" if isLinear else "gaussian" if hasCov else "kernel"
        if route == "linear" and not isLinear:
            raise ValueError("route 'linear' needs a linear-basis model with coefficient covariance")
        if route == "gaussian" and not hasCov:
            raise ValueError("route 'gaussian' needs a model with a joint predictive covariance")
        self._route = route
        self.supports["variances"] = route != "kernel" or base.supports["variances"]
        self.supports["covariance"] = route != "kernel"
        self.supports["parameterInference"] = route == "linear"
        self.supports["derivatives"] = route == "linear" and base.supports["derivatives"]
        self.supports["missingOutputs"] = base.supports.get("missingOutputs", False)
        if route == "linear":
            self._solveLinear()
        else:
            self._solveGaussian()

    def _bounds(self):
        kinds = [c.kind for c in self._constraints]
        lb = np.array([0.0 if k == "lower" else -np.inf for k in kinds])
        ub = np.array([0.0 if k == "upper" else np.inf for k in kinds])
        return lb, ub

    def _steps(self, order: int = 1) -> np.ndarray:
        return self.options["stepFraction"] * self._span

    # ------------------------------------------------------------------ linear route
    def _solveLinear(self) -> None:
        base = self._base
        basis = base.basis
        beta = base.coefficients                                  # (p, ny)
        p, ny = beta.shape
        cov = base.result.covariance                              # (ny, p, p)
        exact = bool(basis.hasDerivative)
        points, rows = expand(self._constraints, ny, self._stepsFor(), exactDerivative=exact)
        g = np.zeros((len(rows), ny * p))
        cache = {}
        for i, entries in enumerate(rows):
            for o, k, coef, der in entries:
                key = (o, der)
                if key not in cache:
                    pts = points[o]
                    cache[key] = basis.transform(pts) if not der else basis.derivative(pts, der[0])
                g[i, o * p:(o + 1) * p] += coef * cache[key][k]
        covFull = np.zeros((ny * p, ny * p))
        for o in range(ny):
            covFull[o * p:(o + 1) * p, o * p:(o + 1) * p] = cov[o]
        b = beta.T.ravel()
        target = np.array([c.value for c in self._constraints])
        h = g @ covFull @ g.T
        lb, ub = self._bounds()
        lam = boundedQp(h, target - g @ b, lb, ub) if len(rows) else np.zeros(0)
        bc = b + covFull @ g.T @ lam
        active = self._activeMask(lam)
        ga = g[active]
        covC = covFull - covFull @ ga.T @ _pinvSym(ga @ covFull @ ga.T) @ ga @ covFull if active.any() else covFull
        self._g, self._lam, self._active = g, lam, active
        self._beta = bc.reshape(ny, p).T
        self._covC = covC
        self._constraintValues = g @ bc
        self._p = p
        self.result = FitResult(parameterNames=base.termNames, params=self._beta,
                                covariance=np.stack([covC[o * p:(o + 1) * p, o * p:(o + 1) * p] for o in range(ny)]),
                                dofResid=base.result.dofResid + float(active.sum()) / ny,
                                sigma2=base.result.sigma2, outputNames=self.outputNames)

    def _stepsFor(self) -> np.ndarray:
        return self._steps()

    def _activeMask(self, lam) -> np.ndarray:
        kinds = np.array([c.kind for c in self._constraints])
        scale = max(float(np.max(np.abs(lam))) if lam.size else 0.0, 1e-300)
        return (kinds == "equal") | (np.abs(lam) > 1e-10 * scale)

    # ------------------------------------------------------------------ gaussian / kernel route
    def _priorCov(self, o: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Joint covariance of output o between point sets a and b (model or correction kernel)."""
        if self._route == "gaussian":
            both = np.vstack([a, b])
            cov = self._base.predictCovariance(both, "confidence")
            cov = cov[o] if cov.shape[0] > 1 else cov[0]
            return cov[:a.shape[0], a.shape[0]:]
        ell = float(self.options["correctionKernel"].get("lengthscale", 0.3)) * self._span
        d2 = np.sum(((a[:, None, :] - b[None, :, :]) / ell) ** 2, axis=-1)
        return self._ySd[o] ** 2 * np.exp(-0.5 * d2)

    def _solveGaussian(self) -> None:
        base = self._base
        ny = base.ny
        yt = base.yt
        self._ySd = np.array([np.nanstd(yt[:, o]) if np.nanstd(yt[:, o]) > 0 else 1.0 for o in range(ny)])
        points, rows = expand(self._constraints, ny, self._stepsFor(), exactDerivative=False)
        self._points = points
        offs = np.cumsum([0] + [p.shape[0] for p in points])
        nTot = int(offs[-1])
        lmat = np.zeros((len(rows), nTot))
        for i, entries in enumerate(rows):
            for o, k, coef, _ in entries:
                lmat[i, offs[o] + k] += coef
        mean = np.zeros(nTot)
        sig = np.zeros((nTot, nTot))
        for o in range(ny):
            if points[o].shape[0]:
                mean[offs[o]:offs[o + 1]] = base.predictValues(points[o])[:, o]
                sig[offs[o]:offs[o + 1], offs[o]:offs[o + 1]] = self._priorCov(o, points[o], points[o])
        target = np.array([c.value for c in self._constraints])
        h = lmat @ sig @ lmat.T
        lb, ub = self._bounds()
        lam = boundedQp(h, target - lmat @ mean, lb, ub) if len(rows) else np.zeros(0)
        active = self._activeMask(lam)
        a = lmat.T @ lam
        self._offs, self._lmat, self._lam, self._active = offs, lmat, lam, active
        self._coefOut = [a[offs[o]:offs[o + 1]] for o in range(ny)]
        la = lmat[active]
        self._hInvActive = _pinvSym(la @ sig @ la.T) if active.any() else np.zeros((0, 0))
        self._constraintValues = lmat @ (mean + sig @ a)

    def _cross(self, x: np.ndarray) -> list:
        """Sigma(x, P_o) per output."""
        return [self._priorCov(o, x, self._points[o]) if self._points[o].shape[0] else np.zeros((x.shape[0], 0))
                for o in range(self._base.ny)]

    # ------------------------------------------------------------------ prediction
    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        if self._route == "linear":
            return self._base.basis.transform(x) @ self._beta
        out = np.empty((x.shape[0], self._base.ny))
        for s in range(0, x.shape[0], _CHUNK):
            xs = x[s:s + _CHUNK]
            m = self._base.predictValues(xs)
            for o, c in enumerate(self._cross(xs)):
                m[:, o] = m[:, o] + c @ self._coefOut[o]
            out[s:s + _CHUNK] = m
        return out

    def _reduction(self, x: np.ndarray):
        """(n, nA) per output: L_A Sigma(P, x) (the part of the variance explained by the constraints)."""
        cross = self._cross(x)
        la = self._lmat[self._active]
        return [c @ la[:, self._offs[o]:self._offs[o + 1]].T for o, c in enumerate(cross)]

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        base = self._base
        if self._route == "linear":
            phi = base.basis.transform(x)
            p = self._p
            out = np.stack([np.einsum("ij,jk,ik->i", phi, self._covC[o * p:(o + 1) * p, o * p:(o + 1) * p], phi)
                            for o in range(base.ny)], axis=1)
            if kind == "prediction":
                out = out + base.result.sigma2[None, :]
            return out
        var = base.predictVariances(x, kind)
        if self._route == "kernel" or not self._active.any():
            return var
        for o, v in enumerate(self._reduction(x)):
            var[:, o] = var[:, o] - np.einsum("ia,ab,ib->i", v, self._hInvActive, v)
        return np.maximum(var, 0.0)

    def _predictCovariance(self, x: np.ndarray, kind: str) -> np.ndarray:
        base = self._base
        if self._route == "linear":
            phi = base.basis.transform(x)
            p = self._p
            out = np.stack([phi @ self._covC[o * p:(o + 1) * p, o * p:(o + 1) * p] @ phi.T for o in range(base.ny)])
            if kind == "prediction":
                out = out + base.result.sigma2[:, None, None] * np.eye(x.shape[0])[None]
            return out
        cov = base.predictCovariance(x, kind).copy()
        if self._active.any():
            for o, v in enumerate(self._reduction(x)):
                cov[o] = cov[o] - v @ self._hInvActive @ v.T
        return cov

    def _predictDerivatives(self, x: np.ndarray, kx: int) -> np.ndarray:
        return self._base.basis.derivative(x, kx) @ self._beta

    def _effectiveParams(self):
        base = np.broadcast_to(np.asarray(self._base.nEffectiveParams, dtype=float), (self._base.ny,)).copy()
        for c, act in zip(self._constraints, self._active):
            if act and c.kind == "equal":
                outs = {t.output for t in c.terms}
                for o in outs:
                    base[o] -= 1.0 / len(outs)
        return np.maximum(base, 1.0)

    def _intervalDof(self) -> Optional[float]:
        return self._base.intervalDof

    # ------------------------------------------------------------------ reporting
    @property
    def baseModel(self) -> SurrogateModelBase:
        """The unconstrained model."""
        self._checkTrained()
        return self._base

    @property
    def coefficients(self) -> np.ndarray:
        """Constrained coefficients (p, ny) (linear route)."""
        self._checkTrained()
        if self._route != "linear":
            raise ValueError("coefficients exist for the linear route only")
        return self._beta.copy()

    def constraintReport(self) -> list[dict]:
        """Per constraint: target, achieved value, residual, multiplier and active flag."""
        self._checkTrained()
        out = []
        for c, v, lam, act in zip(self._constraints, self._constraintValues, self._lam, self._active):
            out.append({"name": c.name, "kind": c.kind, "target": c.value, "value": float(v),
                        "residual": float(v - c.value), "multiplier": float(lam), "active": bool(act)})
        return out

    def summary(self) -> str:
        text = super().summary()
        rep = self.constraintReport()
        worst = max((abs(r["residual"]) for r in rep if r["kind"] == "equal"), default=0.0)
        viol = sum(1 for r in rep if (r["kind"] == "lower" and r["residual"] < -1e-9 * max(1.0, abs(r["target"])))
                   or (r["kind"] == "upper" and r["residual"] > 1e-9 * max(1.0, abs(r["target"]))))
        return (text + f"\n\nConstraints ({self._route} route): {len(rep)} total, "
                f"{sum(r['active'] for r in rep)} active, max |equality residual| = {worst:.3g}, "
                f"violated inequalities = {viol}")

    # ------------------------------------------------------------------ persistence
    def _stateToDict(self) -> dict:
        return {"base": self._base._toDictPlain(True), "route": self._route}

    def _stateFromDict(self, state: dict) -> None:
        self._base = SurrogateModelBase.fromDict(state["base"])
        self.options["route"] = state["route"]
        self._setup()
