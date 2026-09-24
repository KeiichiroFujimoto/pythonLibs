"""Residual diagnostics for any fitted model.

Per output:
    - residuals, standardized residuals; for linear models with leverages
      also internally / externally studentized residuals and Cook's distance
    - normality: D'Agostino-Pearson K^2 (skewness and kurtosis tests) and
      Jarque-Bera
    - heteroscedasticity: Koenker's studentized Breusch-Pagan test
      (squared residuals regressed on the inputs)
    - lack of fit / structure: Wald-Wolfowitz runs test of residual signs
      ordered by the fitted value, Durbin-Watson for residuals in data order
    - collinearity: variance inflation factors and condition number
      (linear-basis models)
and a plain-language list of warnings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.core.InputValidation import asFeatureMatrix, asOutputMatrix, asWeights, toJsonable
from pythonLibs.regressionHandler.numerics.Distributions import ChiSquared, Normal


# ---------------------------------------------------------------- statistical tests
def skewTest(r: np.ndarray) -> tuple[float, float]:
    n = r.size
    if n < 8:
        return np.nan, np.nan
    d = r - r.mean()
    m2 = np.mean(d ** 2)
    b2 = np.mean(d ** 3) / m2 ** 1.5 if m2 > 0 else 0.0
    y = b2 * np.sqrt(((n + 1) * (n + 3)) / (6.0 * (n - 2)))
    beta2 = 3.0 * (n * n + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2.0) * (n + 5) * (n + 7) * (n + 9))
    w2 = -1.0 + np.sqrt(2.0 * (beta2 - 1.0))
    delta = 1.0 / np.sqrt(0.5 * np.log(w2))
    alpha = np.sqrt(2.0 / (w2 - 1.0))
    y = 1.0 if y == 0 else y
    z = delta * np.log(y / alpha + np.sqrt((y / alpha) ** 2 + 1.0))
    return float(z), float(2.0 * Normal().sf(abs(z)))


def kurtosisTest(r: np.ndarray) -> tuple[float, float]:
    n = r.size
    if n < 5:
        return np.nan, np.nan
    d = r - r.mean()
    m2 = np.mean(d ** 2)
    b2 = np.mean(d ** 4) / m2 ** 2 if m2 > 0 else 3.0
    e = 3.0 * (n - 1) / (n + 1)
    varb2 = 24.0 * n * (n - 2) * (n - 3) / ((n + 1.0) ** 2 * (n + 3) * (n + 5))
    x = (b2 - e) / np.sqrt(varb2)
    sqrtBeta1 = 6.0 * (n * n - 5 * n + 2) / ((n + 7.0) * (n + 9)) * np.sqrt(6.0 * (n + 3) * (n + 5) / (n * (n - 2.0) * (n - 3)))
    a = 6.0 + 8.0 / sqrtBeta1 * (2.0 / sqrtBeta1 + np.sqrt(1.0 + 4.0 / sqrtBeta1 ** 2))
    term1 = 1.0 - 2.0 / (9.0 * a)
    denom = 1.0 + x * np.sqrt(2.0 / (a - 4.0))
    if denom == 0:
        return np.nan, np.nan
    term2 = np.sign(denom) * ((1.0 - 2.0 / a) / abs(denom)) ** (1.0 / 3.0)
    z = (term1 - term2) / np.sqrt(2.0 / (9.0 * a))
    return float(z), float(2.0 * Normal().sf(abs(z)))


def normalTest(r: np.ndarray) -> tuple[float, float]:
    """D'Agostino-Pearson omnibus K^2 (n >= 8)."""
    zs, _ = skewTest(r)
    zk, _ = kurtosisTest(r)
    if not (np.isfinite(zs) and np.isfinite(zk)):
        return np.nan, np.nan
    k2 = zs * zs + zk * zk
    return float(k2), float(ChiSquared(2).sf(k2))


def jarqueBera(r: np.ndarray) -> tuple[float, float]:
    n = r.size
    d = r - r.mean()
    m2 = np.mean(d ** 2)
    if n < 3 or m2 <= 0:
        return np.nan, np.nan
    s = np.mean(d ** 3) / m2 ** 1.5
    k = np.mean(d ** 4) / m2 ** 2
    jb = n / 6.0 * (s * s + 0.25 * (k - 3.0) ** 2)
    return float(jb), float(ChiSquared(2).sf(jb))


def breuschPagan(r: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Koenker's studentized Breusch-Pagan LM = n R^2 of r^2 on [1, x]."""
    n = r.size
    e2 = r * r
    a = np.column_stack([np.ones(n), x])
    if n <= a.shape[1] + 1 or np.ptp(e2) == 0:
        return np.nan, np.nan
    coef, *_ = np.linalg.lstsq(a, e2, rcond=None)
    fitted = a @ coef
    sst = np.sum((e2 - e2.mean()) ** 2)
    r2 = 1.0 - np.sum((e2 - fitted) ** 2) / sst if sst > 0 else 0.0
    lm = n * r2
    return float(lm), float(ChiSquared(x.shape[1]).sf(lm))


def runsTest(r: np.ndarray, order: np.ndarray) -> tuple[float, float]:
    """Wald-Wolfowitz runs test of residual signs taken in ``order``."""
    s = np.sign(r[order])
    s = s[s != 0]
    n1, n2 = int(np.sum(s > 0)), int(np.sum(s < 0))
    n = n1 + n2
    if n1 == 0 or n2 == 0 or n < 3:
        return np.nan, np.nan
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    mu = 2.0 * n1 * n2 / n + 1.0
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1.0))
    if var <= 0:
        return np.nan, np.nan
    z = (runs - mu) / np.sqrt(var)
    return float(z), float(2.0 * Normal().sf(abs(z)))


def durbinWatson(r: np.ndarray) -> float:
    den = float(np.sum(r * r))
    return float(np.sum(np.diff(r) ** 2) / den) if den > 0 else np.nan


def varianceInflation(phi: np.ndarray, biasMask: np.ndarray) -> Optional[np.ndarray]:
    cols = [j for j in range(phi.shape[1]) if not biasMask[j] and np.ptp(phi[:, j]) > 0]
    if len(cols) < 2 or len(cols) > 300:
        return None
    c = np.corrcoef(phi[:, cols], rowvar=False)
    try:
        inv = np.linalg.inv(c)
    except np.linalg.LinAlgError:
        return np.full(len(cols), np.inf)
    vif = np.full(phi.shape[1], np.nan)
    vif[cols] = np.diag(inv)
    return vif


# ---------------------------------------------------------------- report
@dataclass
class DiagnosticsReport:
    outputs: list[dict]
    warnings: list[str] = field(default_factory=list)
    modelDescription: str = ""

    def toDict(self, includeArrays: bool = False) -> dict:
        outs = []
        for o in self.outputs:
            d = {k: v for k, v in o.items() if includeArrays or not isinstance(v, np.ndarray)}
            outs.append(d)
        return toJsonable({"model": self.modelDescription, "outputs": outs, "warnings": self.warnings})

    def summary(self) -> str:
        lines = [f"Residual diagnostics: {self.modelDescription}"]
        for o in self.outputs:
            lines.append(f"  [{o['output']}] n={o['n']} sigma={o['sigma']:.4g} "
                         f"normality p={o['normalityP']:.3g} (JB p={o['jarqueBeraP']:.3g}) "
                         f"Breusch-Pagan p={o['breuschPaganP']:.3g} runs p={o['runsP']:.3g} DW={o['durbinWatson']:.3g}")
            if o["outliers"]:
                lines.append(f"      outliers (|t| > 3): {o['outliers']}")
            if o["influential"]:
                lines.append(f"      influential (Cook > 4/n): {o['influential']}")
        lines.extend(f"  ! {w}" for w in self.warnings) if self.warnings else lines.append("  no warnings")
        return "\n".join(lines)


def diagnose(model, x=None, y=None, weights=None, alpha: float = 0.01) -> DiagnosticsReport:
    """Diagnose a trained model on (x, y) (defaults to its training data)."""
    if not model.isTrained:
        raise RuntimeError("model must be trained")
    xa = model.xt if x is None else asFeatureMatrix(x, model.nx)
    ya = model.yt if y is None else asOutputMatrix(y, xa.shape[0])
    if xa is None or ya is None:
        raise ValueError("no data: pass x and y (the model was stored without training data)")
    wa = (model.wt if x is None else None) if weights is None else asWeights(weights, xa.shape[0])
    wa = np.ones(xa.shape[0]) if wa is None else wa
    n = xa.shape[0]
    fitted = model.predictValues(xa)
    resid = ya - fitted
    sw = np.sqrt(wa)
    onTraining = x is None
    lev = None
    solution = getattr(model, "_solution", None)
    if onTraining and solution is not None and solution.hatDiag is not None:
        lev = solution.hatDiag
    edf = np.broadcast_to(np.asarray(model.nEffectiveParams, dtype=float), (ya.shape[1],))
    vif = cond = None
    if solution is not None and getattr(model, "basis", None) is not None:
        vif = varianceInflation(model.basis.transform(xa), model.basis.biasMask)
        cond = solution.conditionNumber
    outputs, warnings = [], []
    for j in range(ya.shape[1]):
        name = model.outputNames[j]
        r = resid[:, j] * sw
        dof = n - edf[j] if onTraining else n
        sigma = float(np.sqrt(np.sum(r * r) / dof)) if dof > 0 else np.nan
        out = {"output": name, "n": n, "sigma": sigma, "residuals": resid[:, j].copy(),
               "standardized": r / sigma if sigma > 0 else np.full(n, np.nan)}
        if lev is not None:
            h = np.clip(lev[:, j], 0.0, 1.0 - 1e-12)
            p = edf[j]
            internal = r / (sigma * np.sqrt(1.0 - h))
            with np.errstate(invalid="ignore"):
                external = internal * np.sqrt(np.maximum(n - p - 1.0, 1e-12) /
                                              np.maximum(n - p - internal ** 2, 1e-12))
            cooks = internal ** 2 * h / (p * (1.0 - h))
            out.update(leverage=h, studentized=external, cooksDistance=cooks)
            screen = np.abs(external)
            out["influential"] = [int(i) for i in np.flatnonzero(cooks > 4.0 / n)]
            out["maxCooksDistance"] = float(np.max(cooks))
        else:
            screen = np.abs(out["standardized"])
            out["influential"] = []
        out["outliers"] = [int(i) for i in np.flatnonzero(screen > 3.0)]
        out["normalityStatistic"], out["normalityP"] = normalTest(r)
        out["jarqueBera"], out["jarqueBeraP"] = jarqueBera(r)
        out["breuschPagan"], out["breuschPaganP"] = breuschPagan(r, xa)
        out["runsZ"], out["runsP"] = runsTest(r, np.argsort(fitted[:, j], kind="stable"))
        out["durbinWatson"] = durbinWatson(r)
        outputs.append(out)

        tag = f"[{name}] "
        if np.isfinite(out["normalityP"]) and out["normalityP"] < alpha:
            warnings.append(tag + f"residuals are not normal (K^2 p={out['normalityP']:.2g}); "
                                  "prediction intervals may be miscalibrated")
        if np.isfinite(out["breuschPaganP"]) and out["breuschPaganP"] < alpha:
            warnings.append(tag + f"variance depends on the inputs (Breusch-Pagan p={out['breuschPaganP']:.2g}); "
                                  "consider weights or a transformed response")
        if onTraining and np.isfinite(out["runsP"]) and out["runsP"] < alpha:
            warnings.append(tag + f"residual signs cluster along the fit (runs p={out['runsP']:.2g}); "
                                  "the model form may be missing structure")
        if out["outliers"]:
            warnings.append(tag + f"{len(out['outliers'])} outlier(s) with |studentized residual| > 3: "
                                  f"{out['outliers'][:10]}")
        if out.get("maxCooksDistance", 0.0) > 1.0:
            warnings.append(tag + f"highly influential point(s) (max Cook's distance {out['maxCooksDistance']:.2g})")
        if onTraining and edf[j] > 0.5 * n:
            warnings.append(tag + f"{edf[j]:.3g} effective parameters for {n} points; check cross-validation "
                                  "for over-fitting")
    if vif is not None and np.nanmax(vif) > 10:
        warnings.append(f"collinear terms (max VIF {np.nanmax(vif):.3g}); coefficients are poorly determined")
    if cond is not None and np.isfinite(cond) and cond > 1e8:
        warnings.append(f"ill-conditioned design (condition number {cond:.2g}); use an orthogonal basis or ridge")
    for o in outputs:
        if vif is not None:
            o["vif"] = vif
        if cond is not None:
            o["conditionNumber"] = cond
    return DiagnosticsReport(outputs=outputs, warnings=warnings, modelDescription=model.describe())
