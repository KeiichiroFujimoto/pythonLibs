"""Parameter inference of a fitted parametric model.

Holds, per output, the estimates, their covariance and the residual degrees
of freedom, and derives standard errors, t statistics, two-sided p-values and
confidence intervals from the (self-implemented) Student-t distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.numerics.Distributions import StudentT, Normal


@dataclass
class FitResult:
    """Estimates for ``ny`` outputs sharing the same parameter names.

    Attributes:
        parameterNames: names of the p parameters
        params:         (p, ny) estimates
        covariance:     (ny, p, p) covariance of the estimates
        dofResid:       residual degrees of freedom (inf -> normal quantiles)
        sigma2:         (ny,) residual variance estimates
        outputNames:    names of the outputs
    """
    parameterNames: list[str]
    params: np.ndarray
    covariance: np.ndarray
    dofResid: float
    sigma2: np.ndarray
    outputNames: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.params = np.asarray(self.params, dtype=float)
        if self.params.ndim == 1:
            self.params = self.params.reshape(-1, 1)
        self.covariance = np.asarray(self.covariance, dtype=float).reshape(
            self.params.shape[1], len(self.parameterNames), len(self.parameterNames))
        self.sigma2 = np.atleast_1d(np.asarray(self.sigma2, dtype=float))
        if not self.outputNames:
            self.outputNames = [f"y{j}" for j in range(self.params.shape[1])]

    def _dist(self):
        return StudentT(self.dofResid) if np.isfinite(self.dofResid) and self.dofResid > 0 else Normal()

    @property
    def stdErrors(self) -> np.ndarray:
        """(p, ny) standard errors."""
        d = np.diagonal(self.covariance, axis1=1, axis2=2).T
        return np.sqrt(np.maximum(d, 0.0))

    @property
    def tValues(self) -> np.ndarray:
        se = self.stdErrors
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(se > 0, self.params / se, np.inf * np.sign(self.params))

    @property
    def pValues(self) -> np.ndarray:
        t = np.abs(self.tValues)
        dist = self._dist()
        out = np.zeros_like(t)
        finite = np.isfinite(t)
        out[finite] = 2.0 * np.asarray(dist.sf(t[finite]))
        return out

    def confInt(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper), each (p, ny)."""
        q = float(self._dist().ppf(0.5 + level / 2.0))
        se = self.stdErrors
        return self.params - q * se, self.params + q * se

    def correlation(self, output: int = 0) -> np.ndarray:
        c = self.covariance[output]
        s = np.sqrt(np.maximum(np.diag(c), 1e-300))
        return c / np.outer(s, s)

    def table(self, output: int = 0, level: float = 0.95) -> list[dict]:
        lo, hi = self.confInt(level)
        se, t, p = self.stdErrors, self.tValues, self.pValues
        return [{"name": n, "estimate": float(self.params[i, output]), "stdError": float(se[i, output]),
                 "tValue": float(t[i, output]), "pValue": float(p[i, output]),
                 "lower": float(lo[i, output]), "upper": float(hi[i, output])}
                for i, n in enumerate(self.parameterNames)]

    def summary(self, output: Optional[int] = None, level: float = 0.95) -> str:
        outputs = range(self.params.shape[1]) if output is None else [output]
        lines = []
        pct = f"{level * 100:g}%"
        for j in outputs:
            lines.append(f"Output {self.outputNames[j]}  (residual dof = {self.dofResid:g}, "
                         f"sigma = {np.sqrt(self.sigma2[j]):.6g})")
            header = f"{'term':>16} {'estimate':>14} {'std err':>12} {'t':>9} {'p':>10} {pct + ' CI':>30}"
            lines.append(header)
            lines.append("-" * len(header))
            for row in self.table(j, level):
                ci = f"[{row['lower']:.6g}, {row['upper']:.6g}]"
                lines.append(f"{row['name']:>16} {row['estimate']:>14.6g} {row['stdError']:>12.4g} "
                             f"{row['tValue']:>9.3g} {row['pValue']:>10.3g} {ci:>30}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def toDict(self) -> dict:
        return {"parameterNames": list(self.parameterNames), "params": self.params.tolist(),
                "covariance": self.covariance.tolist(), "dofResid": float(self.dofResid),
                "sigma2": self.sigma2.tolist(), "outputNames": list(self.outputNames)}

    @classmethod
    def fromDict(cls, d: dict) -> "FitResult":
        return cls(parameterNames=d["parameterNames"], params=np.array(d["params"]),
                   covariance=np.array(d["covariance"]), dofResid=float(d["dofResid"]),
                   sigma2=np.array(d["sigma2"]), outputNames=d.get("outputNames", []))
