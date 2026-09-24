"""Markdown report of a fitted model (summary, parameters, CV, diagnostics, comparison)."""
from __future__ import annotations

from typing import Optional

import numpy as np


def _fmt(v, digits: int = 5) -> str:
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "nan" if np.isnan(f) else f"{f:.{digits}g}"


def modelReport(model, title: Optional[str] = None, cv=None, diagnostics=None, selection=None,
                dataSummary: Optional[dict] = None) -> str:
    """Build a Markdown report; every section is optional except the model summary."""
    lines = [f"# {title or 'Regression report'}", ""]
    lines += ["## Model", "", f"- Type: `{model.registryName}`", f"- Specification: `{model.describe()}`",
              f"- Inputs ({model.nx}): {', '.join(model.featureNames)}",
              f"- Outputs ({model.ny}): {', '.join(model.outputNames)}", ""]
    if dataSummary:
        lines += ["## Data", "", f"{dataSummary['nSamples']} samples.", "",
                  "| variable | min | max | mean | std |", "|---|---|---|---|---|"]
        for row in dataSummary["inputs"] + dataSummary["outputs"]:
            lines.append(f"| {row['name']} | {_fmt(row['min'])} | {_fmt(row['max'])} | {_fmt(row['mean'])} | "
                         f"{_fmt(row['std'])} |")
        lines.append("")
    lines += ["## Goodness of fit", "", "| output | R2 | adj. R2 | RMSE | max abs err | eff. params | AICc | BIC |",
              "|---|---|---|---|---|---|---|---|"]
    for name, m in zip(model.outputNames, model.metrics):
        lines.append(f"| {name} | {_fmt(m.rSquared, 6)} | {_fmt(m.adjRSquared, 6)} | {_fmt(m.rmse)} | "
                     f"{_fmt(m.maxAbsError)} | {_fmt(m.nParams, 4)} | {_fmt(m.aicc)} | {_fmt(m.bic)} |")
    lines.append("")
    equation = getattr(model, "equation", None)
    if callable(equation):
        try:
            lines += ["## Equation", ""] + [f"    {equation(j)}" for j in range(model.ny)] + [""]
        except Exception:
            pass
    if model.result is not None:
        lines += ["## Parameters (95 % confidence)", ""]
        for j, name in enumerate(model.outputNames):
            if model.ny > 1:
                lines += [f"### {name}", ""]
            lines += ["| term | estimate | std. error | t | p | lower | upper |", "|---|---|---|---|---|---|---|"]
            for row in model.result.table(j):
                lines.append(f"| `{row['name']}` | {_fmt(row['estimate'], 6)} | {_fmt(row['stdError'], 4)} | "
                             f"{_fmt(row['tValue'], 4)} | {_fmt(row['pValue'], 3)} | {_fmt(row['lower'], 6)} | "
                             f"{_fmt(row['upper'], 6)} |")
            lines.append("")
    hyper = getattr(type(model), "hyperparameters", None)
    if isinstance(hyper, property):
        h = model.hyperparameters
        lines += ["## Hyperparameters", "", "```", _pretty(h), "```", ""]
    if cv is not None:
        lines += [f"## Cross-validation ({cv.method}, {cv.nFolds} folds)", "",
                  "| output | RMSE | MAE | max abs err | Q2 |", "|---|---|---|---|---|"]
        for j, name in enumerate(model.outputNames):
            lines.append(f"| {name} | {_fmt(cv.rmse[j])} | {_fmt(cv.mae[j])} | {_fmt(cv.maxAbsError[j])} | "
                         f"{_fmt(cv.q2[j], 6)} |")
        lines.append("")
    if diagnostics is not None:
        lines += ["## Residual diagnostics", "",
                  "| output | sigma | normality p | Breusch-Pagan p | runs p | Durbin-Watson | outliers |",
                  "|---|---|---|---|---|---|---|"]
        for o in diagnostics.outputs:
            lines.append(f"| {o['output']} | {_fmt(o['sigma'])} | {_fmt(o['normalityP'], 3)} | "
                         f"{_fmt(o['breuschPaganP'], 3)} | {_fmt(o['runsP'], 3)} | {_fmt(o['durbinWatson'], 3)} | "
                         f"{len(o['outliers'])} |")
        lines.append("")
        if diagnostics.warnings:
            lines += ["**Warnings**", ""] + [f"- {w}" for w in diagnostics.warnings] + [""]
        else:
            lines += ["No diagnostic warnings.", ""]
    if selection is not None:
        lines += [f"## Model comparison ({selection.criterion})", "",
                  "| rank | model | score | CV NRMSE | AICc | eff. params | seconds | note |",
                  "|---|---|---|---|---|---|---|---|"]
        for i, s in enumerate(selection.scores):
            note = "selected" if i == selection.bestIndex else (s.error or "")
            lines.append(f"| {i + 1} | `{s.name}` | {_fmt(s.score)} | {_fmt(s.cvNrmse, 4)} | {_fmt(s.aicc)} | "
                         f"{_fmt(s.nParams, 4)} | {_fmt(s.seconds, 3)} | {note[:80]} |")
        lines += [""] + [f"_{n}_" for n in selection.notes]
    return "\n".join(lines).rstrip() + "\n"


def _pretty(d, indent: int = 0) -> str:
    pad = "  " * indent
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out.append(f"{pad}{k}:")
            out.append(_pretty(v, indent + 1))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            out.append(f"{pad}{k}:")
            for i, item in enumerate(v):
                out.append(f"{pad}  [{i}]")
                out.append(_pretty(item, indent + 2))
        elif isinstance(v, (list, tuple)):
            out.append(f"{pad}{k}: [{', '.join(_fmt(e) for e in v)}]")
        else:
            out.append(f"{pad}{k}: {_fmt(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v}")
    return "\n".join(out)
