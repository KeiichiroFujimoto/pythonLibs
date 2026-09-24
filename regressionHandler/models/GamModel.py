"""Generalized additive models with penalized regression splines.

    g(E[y | x]) = beta_0 + sum_j f_j(x_j) + f_jk(x_j, x_k) + ... + offset

Terms (``terms`` option; default one smooth per input)::

    {"type": "smooth", "column": j, "nSegments": 10, "degree": 3, "penaltyOrder": 2}   # P-spline f(x_j)
    {"type": "tensor", "columns": [j, k], "nSegments": [6, 6]}     # f(x_j, x_k), one penalty per margin
    {"type": "linear", "columns": [j, ...]}                         # unpenalized linear terms
    {"type": "factor", "column": j}                                 # categorical codes, treatment contrasts
    {"type": "random", "column": j}                                 # i.i.d. random intercepts (ridge penalty)

Smooth terms carry a sum-to-zero constraint over the training data (so the
intercept is identifiable and every term is a centred partial effect). All
smoothing parameters are chosen jointly by GCV (unknown scale) or UBRE
(known scale) through penalized IRLS, for any family / link of ``GlmModel``.
Coefficient uncertainty uses the Bayesian covariance (X^T W X + S)^-1 phi.

References:
    Eilers & Marx (1996) Statistical Science 11(2).
    Wood (2017) *Generalized Additive Models: An Introduction with R*, 2nd ed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.bases.BSplineBasis import BSplineBasis, differenceMatrix
from pythonLibs.regressionHandler.core.FitResult import FitResult
from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.glm.Pirls import PirlsResult, selectSmoothing
from pythonLibs.regressionHandler.models.GlmModel import GlmModel
from pythonLibs.regressionHandler.numerics.Distributions import ChiSquared, FDistribution


# ---------------------------------------------------------------- terms
def _sumToZero(b: np.ndarray) -> np.ndarray:
    """Z (k, k-1) spanning {c : 1^T B c = 0} (Householder / QR null space)."""
    c = b.sum(axis=0)[:, None]
    q, _ = np.linalg.qr(c, mode="complete")
    return q[:, 1:]


class _Term:
    kind = "term"
    penalized = False

    def __init__(self, spec: dict) -> None:
        self.spec = dict(spec)

    def columns(self) -> list[int]:
        cols = self.spec.get("columns", [self.spec.get("column")])
        return [int(c) for c in cols]

    def fit(self, x: np.ndarray) -> "_Term":
        return self

    def design(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, x: np.ndarray, kx: int) -> np.ndarray:
        return np.zeros((x.shape[0], self.size))

    def penalties(self) -> list[np.ndarray]:
        return []

    def label(self, names) -> str:
        return f"{self.kind}(" + ",".join(names[c] for c in self.columns()) + ")"

    def coefNames(self, names) -> list[str]:
        return [f"{self.label(names)}.{i}" for i in range(self.size)]

    def state(self) -> dict:
        return {}

    def load(self, state: dict) -> None:
        pass


class _SplineTerm(_Term):
    """1-D smooth or 2-D tensor-product P-spline with a sum-to-zero constraint."""
    penalized = True

    def __init__(self, spec: dict) -> None:
        super().__init__(spec)
        self.kind = "s" if spec["type"] == "smooth" else "te"

    def _basis(self) -> BSplineBasis:
        seg = self.spec.get("nSegments", 10 if self.kind == "s" else 6)
        return BSplineBasis(nSegments=seg, degree=int(self.spec.get("degree", 3)),
                            penaltyOrder=int(self.spec.get("penaltyOrder", 2)))

    def fit(self, x):
        cols = self.columns()
        self._b = self._basis().fit(x[:, cols])
        self._z = _sumToZero(self._b.transform(x[:, cols]))
        self.size = self._z.shape[1]
        return self

    def design(self, x):
        return self._b.transform(x[:, self.columns()]) @ self._z

    def derivative(self, x, kx):
        cols = self.columns()
        if kx not in cols:
            return np.zeros((x.shape[0], self.size))
        return self._b.derivative(x[:, cols], cols.index(kx)) @ self._z

    def penalties(self):
        order = int(self.spec.get("penaltyOrder", 2))
        sizes = self._b._sizes
        out = []
        for j in range(len(sizes)):                  # one penalty per margin (anisotropic tensor smoothing)
            d = differenceMatrix(sizes[j], min(order, sizes[j] - 1))
            mats = [np.eye(s) for s in sizes]
            mats[j] = d.T @ d
            k = mats[0]
            for m in mats[1:]:
                k = np.kron(k, m)
            out.append(self._z.T @ k @ self._z)
        return out

    def state(self):
        return {"basis": self._b.toDict(), "z": self._z.tolist()}

    def load(self, state):
        from pythonLibs.regressionHandler.core.Registry import buildComponent
        self._b = buildComponent("basis", state["basis"])
        self._z = np.array(state["z"], dtype=float)
        self.size = self._z.shape[1]


class _LinearTerm(_Term):
    kind = "linear"

    def fit(self, x):
        self.size = len(self.columns())
        return self

    def design(self, x):
        return x[:, self.columns()]

    def derivative(self, x, kx):
        return np.array([[1.0 if c == kx else 0.0 for c in self.columns()]]).repeat(x.shape[0], axis=0)

    def coefNames(self, names):
        return [names[c] for c in self.columns()]

    def load(self, state):
        self.size = len(self.columns())


class _FactorTerm(_Term):
    """Categorical column (integer / float codes): dummies for all levels but the first, or all
    levels with an identity penalty for random intercepts."""

    def __init__(self, spec: dict) -> None:
        super().__init__(spec)
        self.kind = "factor" if spec["type"] == "factor" else "re"
        self.penalized = self.kind == "re"

    def fit(self, x):
        self._levels = np.unique(x[:, self.columns()[0]])
        if self._levels.size < 2:
            raise ValueError("a factor term needs at least two levels")
        self._use = self._levels[1:] if self.kind == "factor" else self._levels
        self.size = self._use.size
        return self

    def design(self, x):
        col = x[:, self.columns()[0]]
        unknown = ~np.isin(col, self._levels)
        if self.kind == "factor" and np.any(unknown):
            raise ValueError("factor term received levels not seen in training")
        return (col[:, None] == self._use[None, :]).astype(float)

    def penalties(self):
        return [np.eye(self.size)] if self.penalized else []

    def coefNames(self, names):
        return [f"{names[self.columns()[0]]}[{v:g}]" for v in self._use]

    def state(self):
        return {"levels": self._levels.tolist()}

    def load(self, state):
        self._levels = np.array(state["levels"], dtype=float)
        self._use = self._levels[1:] if self.kind == "factor" else self._levels
        self.size = self._use.size


_TERMS = {"smooth": _SplineTerm, "tensor": _SplineTerm, "linear": _LinearTerm, "factor": _FactorTerm,
          "random": _FactorTerm}


def _makeTerm(spec) -> _Term:
    spec = dict(spec)
    if spec.get("type") not in _TERMS:
        raise ValueError(f"term type must be one of {sorted(_TERMS)}")
    return _TERMS[spec["type"]](spec)


# ---------------------------------------------------------------- model
@registry("model").register("gam")
class GamModel(GlmModel):
    """Generalized additive model: penalized spline, tensor, linear, factor and random-effect terms."""

    def _initialize(self) -> None:
        super()._initialize()
        d = self.options.declare
        d("terms", None, types=list, desc="Term specs (default: one smooth per non-offset input)")
        d("method", "auto", values=("auto", "gcv", "ubre"), desc="Smoothing-parameter criterion")
        d("lambdas", None, types=list, desc="Fixed smoothing parameters (None entries are estimated)")
        d("gamma", 1.0, types=(int, float), lower=1.0, desc="GCV / UBRE edf inflation (1.4 gives smoother fits)")
        d("logLambdaBounds", [-12.0, 12.0], types=list, desc="Search bounds of log smoothing parameters")

    def _validateOptions(self) -> None:
        self._makeFamily()
        for spec in self.options["terms"] or []:
            _makeTerm(spec)

    # ------------------------------------------------------------------ design
    def _termSpecs(self) -> list[dict]:
        if self.options["terms"] is not None:
            return [dict(t) for t in self.options["terms"]]
        oc = self.options["offsetColumn"]
        return [{"type": "smooth", "column": j} for j in range(self.nx) if oc is None or j != oc % self.nx]

    def _design(self, x):
        oc = self.options["offsetColumn"]
        off = np.zeros(x.shape[0]) if oc is None else x[:, oc % self.nx]
        blocks = [np.ones((x.shape[0], 1))] + [t.design(x) for t in self._terms]
        return np.hstack(blocks), off

    def _blockSlices(self) -> list[slice]:
        out, start = [], 1
        for t in self._terms:
            out.append(slice(start, start + t.size))
            start += t.size
        return out

    # ------------------------------------------------------------------ training
    def _train(self) -> None:
        y, w = self.yt[:, 0], self.wt
        self._family = self._makeFamily()
        self._family.checkResponse(y)
        self._terms = [_makeTerm(s).fit(self.xt) for s in self._termSpecs()]
        x, off = self._design(self.xt)
        p = x.shape[1]
        if p >= y.size:
            raise ValueError(f"{p} coefficients but only {y.size} observations; use fewer segments")
        penalties, owners = [], []
        for ti, (t, sl) in enumerate(zip(self._terms, self._blockSlices())):
            for s in t.penalties():
                full = np.zeros((p, p))
                full[sl, sl] = s
                penalties.append(full)
                owners.append(ti)
        self._owners = owners
        fixed = self.options["lambdas"]
        if fixed is not None and len(fixed) != len(penalties):
            raise ValueError(f"lambdas needs {len(penalties)} entries (one per penalty)")
        for _ in range(50):
            lam, res, score = selectSmoothing(x, y, w, self._family, penalties, self.options["method"], off,
                                              tuple(self.options["logLambdaBounds"]), float(self.options["gamma"]),
                                              self.options["maxIter"], fixed)
            before = getattr(self._family, "theta", None)
            if getattr(self._family, "estimateTheta", False):
                self._family.updateTheta(y, res.mu, w)
            if before is None or abs(self._family.theta - before) <= 1e-7 * max(before, 1.0):
                break
        self._lambdas, self._res, self._score = lam, res, float(score)
        self._alpha = float(np.sum(lam)) if lam.size else 0.0
        n = y.size
        fam = self._family
        if fam.scaleKnown:
            self._phi = 1.0
        elif not isinstance(self.options["dispersion"], str):
            self._phi = float(self.options["dispersion"])
        elif self.options["dispersion"] == "pearson":
            self._phi = float(np.sum(w * (y - res.mu) ** 2 / fam.variance(res.mu)) / max(n - res.edf, 1.0))
        else:
            self._phi = res.deviance / max(n - res.edf, 1.0)
        self._nullDeviance = self._nullFit(y, w, off)
        self._logLik = self._logLikelihood(y, w)
        self.result = FitResult(parameterNames=self.termNames, params=res.coef[:, None],
                                covariance=(self._phi * res.covUnscaled)[None],
                                dofResid=float("inf") if fam.scaleKnown else max(n - res.edf, 1.0),
                                sigma2=np.array([self._phi]), outputNames=self.outputNames)

    # ------------------------------------------------------------------ prediction extras
    def _predictDerivatives(self, x: np.ndarray, kx: int) -> np.ndarray:
        design, off = self._design(x)
        eta = design @ self._res.coef + off
        oc = self.options["offsetColumn"]
        if oc is not None and kx == oc % self.nx:
            deta = np.ones(x.shape[0])
        else:
            d = np.hstack([np.zeros((x.shape[0], 1))] + [t.derivative(x, kx) for t in self._terms])
            deta = d @ self._res.coef
        return (self._family.link.dmuDeta(eta) * deta)[:, None]

    def predictTerms(self, x) -> dict:
        """Centred partial effects on the link scale: {label: {"fit": (m,), "se": (m,)}}."""
        self._checkTrained()
        xv = self._validX(x)
        out = {}
        for t, sl in zip(self._terms, self._blockSlices()):
            b = t.design(xv)
            cov = self._res.covUnscaled[sl, sl] * self._phi
            out[t.label(self.featureNames)] = {"fit": b @ self._res.coef[sl],
                                               "se": np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", b, cov, b), 0.0))}
        return out

    # ------------------------------------------------------------------ reporting
    @property
    def termNames(self) -> list[str]:
        names = ["(Intercept)"]
        for t in self._terms:
            names += t.coefNames(self.featureNames)
        return names

    def termTable(self) -> list[dict]:
        """Per term: edf, smoothing parameters and an approximate Wald test of 'term = 0'.

        The statistic beta_j^T V_j^+ beta_j uses a rank-r pseudo-inverse with
        r = max(1, round(edf_j)); p-values are approximate (chi^2_r, or F_{r, n-edf}
        for unknown scale), in the spirit of Wood (2006).
        """
        self._checkTrained()
        n = self.xt.shape[0]
        rows = []
        for ti, (t, sl) in enumerate(zip(self._terms, self._blockSlices())):
            beta = self._res.coef[sl]
            v = self._res.covUnscaled[sl, sl] * self._phi
            edf = float(np.sum(self._res.edfTerms[sl])) if self._res.edfTerms.size else float("nan")
            r = max(1, int(round(edf))) if np.isfinite(edf) else t.size
            wv, vv = np.linalg.eigh(0.5 * (v + v.T))
            idx = np.argsort(wv)[::-1][:min(r, t.size)]
            keep = idx[wv[idx] > wv.max() * 1e-12] if wv.size else idx
            stat = float(np.sum((vv[:, keep].T @ beta) ** 2 / wv[keep])) if keep.size else 0.0
            rank = keep.size
            if self._family.scaleKnown:
                pv = float(ChiSquared(max(rank, 1)).sf(stat))
                test = {"chi2": stat}
            else:
                fstat = stat / max(rank, 1)
                pv = float(FDistribution(max(rank, 1), max(n - self._res.edf, 1.0)).sf(fstat))
                test = {"F": fstat}
            lam = [float(self._lambdas[k]) for k, o in enumerate(self._owners) if o == ti]
            rows.append({"term": t.label(self.featureNames), "edf": edf, "refDf": rank, **test, "pValue": pv,
                         "lambdas": lam})
        return rows

    def gamSummary(self) -> dict:
        out = self.glmSummary()
        out.pop("penaltyWeight", None)
        crit = self.options["method"]
        if crit == "auto":
            crit = "ubre" if self._family.scaleKnown else "gcv"
        out.update({"criterion": crit, "score": self._score, "lambdas": self._lambdas.tolist(),
                    "devianceExplained": 1.0 - self._res.deviance / self._nullDeviance
                    if self._nullDeviance and np.isfinite(self._nullDeviance) else float("nan")})
        return out

    def summary(self) -> str:
        self._checkTrained()
        lines = [f"Model: {self.describe()}", f"Training points: {self.xt.shape[0]}, inputs: {self.nx}"]
        m = self.metrics[0]
        lines.append(f"  {self.outputNames[0]}: R2={m.rSquared:.6g}  RMSE={m.rmse:.6g}  edf={self._res.edf:.4g}")
        se = self.result.stdErrors[:, 0]
        paramIdx = [0] + [i for t, sl in zip(self._terms, self._blockSlices()) if not t.penalized
                          for i in range(sl.start, sl.stop)]
        lines.append("\nParametric coefficients:")
        names = self.termNames
        for i in paramIdx:
            lines.append(f"  {names[i]:>16}  {self._res.coef[i]: .6g}  (se {se[i]:.4g})")
        lines.append("\nSmooth / random terms:")
        for row in self.termTable():
            test = f"chi2={row['chi2']:.4g}" if "chi2" in row else f"F={row['F']:.4g}"
            if any(t.penalized and t.label(self.featureNames) == row["term"] for t in self._terms):
                lines.append(f"  {row['term']:>16}  edf={row['edf']:.3f}  {test}  p={row['pValue']:.3g}")
        g = self.gamSummary()
        lines.append(f"\n{g['criterion'].upper()} = {g['score']:.6g}   deviance explained = "
                     f"{100 * g['devianceExplained']:.1f} %   scale = {g['dispersion']:.4g}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ persistence
    def _stateToDict(self) -> dict:
        state = super()._stateToDict()
        state.pop("basis", None)
        state.update({"terms": [t.state() for t in self._terms], "termSpecs": self._termSpecs(),
                      "lambdas": self._lambdas.tolist(), "score": self._score, "owners": self._owners,
                      "edfTerms": self._res.edfTerms.tolist()})
        return state

    def _stateFromDict(self, state: dict) -> None:
        from pythonLibs.regressionHandler.glm.Families import buildFamily
        self._terms = []
        for spec, st in zip(state["termSpecs"], state["terms"]):
            t = _makeTerm(spec)
            t.load(st)
            self._terms.append(t)
        fam = dict(state["family"])
        est = fam.pop("estimateTheta", None)
        self._family = buildFamily(fam)
        if est is not None:
            self._family.estimateTheta = bool(est)
        coef = np.array(state["coef"], dtype=float)
        self._res = PirlsResult(coef=coef, eta=np.zeros(0), mu=np.zeros(0), weights=np.zeros(0),
                                deviance=float(state["deviance"]), penalty=0.0,
                                covUnscaled=np.array(state["cov"], dtype=float), hatDiag=np.zeros(0),
                                edf=float(state["edf"]), edfTerms=np.array(state["edfTerms"], dtype=float),
                                iterations=int(state["iterations"]), converged=bool(state["converged"]),
                                rank=coef.size)
        self._phi = float(state["phi"])
        self._nullDeviance = float(state["nullDeviance"])
        self._logLik = float(state["logLik"])
        self._alpha = float(state["alpha"])
        self._lambdas = np.array(state["lambdas"], dtype=float)
        self._score = float(state["score"])
        self._owners = list(state["owners"])
