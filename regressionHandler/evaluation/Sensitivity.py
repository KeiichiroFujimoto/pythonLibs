"""Global sensitivity analysis: Sobol indices for independent inputs.

    Var[y] = sum_i V_i + sum_{i<j} V_ij + ...,   S_i = V_i / Var[y],   S_Ti = E[Var(y | x_~i)] / Var[y]

``sobolIndices(model)`` chooses the method:

    "pce"         exact indices of a polynomial chaos expansion: a ``LinearBasisModel``
                  with a Legendre ``orthogonalPolynomial`` basis, inputs taken as
                  independent uniforms over the basis box (the training range)
    "montecarlo"  Saltelli (2010) first-order and Jansen (1999) total-effect estimators on
                  a randomly shifted low-discrepancy design, N (d + 2) model runs,
                  bootstrap confidence intervals; works for any model or callable

Inputs are independent uniforms over ``xlimits`` (default: the training range)
unless ``sampler`` draws them from another distribution.

References:
    Sobol' (1993) Mathematical Modelling and Computational Experiments 1(4).
    Saltelli et al. (2010) Computer Physics Communications 181(2).
    Sudret (2008) Reliability Engineering & System Safety 93(7).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from pythonLibs.regressionHandler.sampling.Sampling import sobolLike


@dataclass
class SobolResult:
    """First-order, total and (PCE only) second-order indices of one output."""
    names: list
    first: np.ndarray
    total: np.ndarray
    variance: float
    method: str
    second: Optional[np.ndarray] = None
    firstInterval: Optional[np.ndarray] = None
    totalInterval: Optional[np.ndarray] = None
    extra: dict = field(default_factory=dict)

    def table(self) -> list[dict]:
        rows = []
        for i, n in enumerate(self.names):
            row = {"input": n, "first": float(self.first[i]), "total": float(self.total[i])}
            if self.firstInterval is not None:
                row["firstInterval"] = self.firstInterval[i].tolist()
                row["totalInterval"] = self.totalInterval[i].tolist()
            rows.append(row)
        return rows

    def summary(self) -> str:
        lines = [f"Sobol indices ({self.method}), Var[y] = {self.variance:.6g}", f"{'input':>12}  {'S_i':>9}  {'S_Ti':>9}"]
        for r in self.table():
            lines.append(f"{r['input']:>12}  {r['first']:9.4f}  {r['total']:9.4f}")
        return "\n".join(lines)

    def toDict(self) -> dict:
        d = {"names": self.names, "first": self.first.tolist(), "total": self.total.tolist(),
             "variance": self.variance, "method": self.method}
        if self.second is not None:
            d["second"] = self.second.tolist()
        if self.firstInterval is not None:
            d["firstInterval"] = self.firstInterval.tolist()
            d["totalInterval"] = self.totalInterval.tolist()
        return d


def _isLegendrePce(model) -> bool:
    from pythonLibs.regressionHandler.bases.PolynomialBasis import OrthogonalPolynomialBasis
    from pythonLibs.regressionHandler.models.LinearBasisModel import LinearBasisModel
    return isinstance(model, LinearBasisModel) and isinstance(model.basis, OrthogonalPolynomialBasis) and \
        model.basis.options["family"] == "legendre"


def pceSobol(model, output: int = 0) -> SobolResult:
    """Exact Sobol indices of a Legendre polynomial chaos expansion (uniform inputs on the basis box).

    With P_k orthogonal on [-1, 1] and E[P_k^2] = 1 / (2k + 1) under the uniform
    measure, a term c_a prod_j P_(a_j) contributes c_a^2 prod_j 1 / (2 a_j + 1).
    """
    if not _isLegendrePce(model):
        raise ValueError("pceSobol needs a LinearBasisModel with a Legendre orthogonalPolynomial basis")
    exps = model.basis._exps
    c = model.coefficients[:, output]
    norm = np.prod(1.0 / (2.0 * exps + 1.0), axis=1)
    contrib = c * c * norm
    nonconst = exps.sum(axis=1) > 0
    var = float(np.sum(contrib[nonconst]))
    d = exps.shape[1]
    active = exps > 0
    nActive = active.sum(axis=1)
    first = np.array([np.sum(contrib[(nActive == 1) & active[:, i]]) for i in range(d)]) / var
    total = np.array([np.sum(contrib[active[:, i]]) for i in range(d)]) / var
    second = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            second[i, j] = second[j, i] = np.sum(contrib[(nActive == 2) & active[:, i] & active[:, j]]) / var
    const = contrib[~nonconst]
    mean = float(c[~nonconst][0]) if const.size else 0.0
    return SobolResult(names=list(model.featureNames), first=first, total=total, variance=var, method="pce",
                       second=second, extra={"mean": mean})


def monteCarloSobol(fun: Callable[[np.ndarray], np.ndarray], xlimits, nSamples: int = 4096, seed: int = 0,
                    names=None, nBoot: int = 200, level: float = 0.95,
                    sampler: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> SobolResult:
    """Saltelli / Jansen estimators for a function of independent inputs.

    Args:
        fun:       vectorized model (m, d) -> (m,)
        xlimits:   (d, 2) box of the uniform inputs (or the unit box when ``sampler`` is given)
        nSamples:  base sample size N; the cost is N (d + 2) evaluations
        sampler:   optional map from unit-cube points (m, d) to the input distribution
                   (e.g. inverse CDFs); replaces the uniform box scaling
    """
    lim = np.asarray(xlimits, dtype=float)
    d = lim.shape[0]
    unit = sobolLike(nSamples, np.tile([0.0, 1.0], (2 * d, 1)), seed)
    ua, ub = unit[:, :d], unit[:, d:]
    toX = sampler if sampler is not None else (lambda u: lim[:, 0] + u * (lim[:, 1] - lim[:, 0]))
    a, b = toX(ua), toX(ub)
    ab = []
    for i in range(d):
        m = ua.copy()
        m[:, i] = ub[:, i]
        ab.append(toX(m))
    fa = np.asarray(fun(a), dtype=float).ravel()
    fb = np.asarray(fun(b), dtype=float).ravel()
    fab = np.column_stack([np.asarray(fun(m), dtype=float).ravel() for m in ab])

    def estimate(idx):
        fA, fB, fAB = fa[idx], fb[idx], fab[idx]
        var = float(np.var(np.concatenate([fA, fB])))
        s1 = np.mean(fB[:, None] * (fAB - fA[:, None]), axis=0) / var
        st = 0.5 * np.mean((fA[:, None] - fAB) ** 2, axis=0) / var
        return s1, st, var

    s1, st, var = estimate(np.arange(nSamples))
    rng = np.random.default_rng(seed + 1)
    boots = [estimate(rng.integers(0, nSamples, nSamples)) for _ in range(nBoot)]
    q = [(1.0 - level) / 2.0, (1.0 + level) / 2.0]
    s1b = np.array([b[0] for b in boots])
    stb = np.array([b[1] for b in boots])
    return SobolResult(names=list(names) if names is not None else [f"x{i}" for i in range(d)], first=s1, total=st,
                       variance=var, method="montecarlo", firstInterval=np.quantile(s1b, q, axis=0).T,
                       totalInterval=np.quantile(stb, q, axis=0).T, extra={"nEvaluations": nSamples * (d + 2)})


def sobolIndices(model, xlimits=None, method: str = "auto", nSamples: int = 4096, seed: int = 0, output: int = 0,
                 nBoot: int = 200, sampler=None) -> SobolResult:
    """Sobol indices of a trained model (see the module docstring for the methods)."""
    if callable(model) and not hasattr(model, "predictValues"):
        if xlimits is None:
            raise ValueError("xlimits are required for a plain function")
        return monteCarloSobol(model, xlimits, nSamples, seed, None, nBoot, sampler=sampler)
    model._checkTrained()
    if method == "auto":
        method = "pce" if _isLegendrePce(model) and xlimits is None and sampler is None else "montecarlo"
    if method == "pce":
        return pceSobol(model, output)
    if method != "montecarlo":
        raise ValueError("method must be auto, pce or montecarlo")
    lim = np.column_stack([model.xt.min(axis=0), model.xt.max(axis=0)]) if xlimits is None else xlimits
    return monteCarloSobol(lambda x: model.predictValues(x)[:, output], lim, nSamples, seed, model.featureNames,
                           nBoot, sampler=sampler)


def ishigamiSobol(a: float = 7.0, b: float = 0.1) -> dict:
    """Analytic Sobol indices of the Ishigami function (reference values)."""
    v1 = 0.5 * (1 + b * np.pi ** 4 / 5) ** 2
    v2 = a * a / 8
    v13 = b * b * np.pi ** 8 * (1 / 18 - 1 / 50)
    var = v1 + v2 + v13
    return {"first": np.array([v1, v2, 0.0]) / var, "total": np.array([v1 + v13, v2, v13]) / var, "variance": var}
