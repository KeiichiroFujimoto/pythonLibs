"""Create models from compact specs.

Accepted specs:
    - a registered model name:            "linearBasis", "kriging", ...
    - a dict:                              {"type": "linearBasis", "basis": {...}, "solver": "ridge"}
    - a shorthand (see ``SHORTHANDS``):    "linear", "quadratic", "poly3", "ortho6", "pspline",
                                           "robust-poly2", "lasso-poly3", "rbf", "kriging", ...
    - an existing model instance (returned as an untrained clone)
"""
from __future__ import annotations

import re
from typing import Any, Optional

from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase

import pythonLibs.regressionHandler.models.LinearBasisModel  # noqa: F401  (registers linearBasis)
import pythonLibs.regressionHandler.models.KrigingModel  # noqa: F401  (registers kriging, kpls)
import pythonLibs.regressionHandler.models.RbfModel  # noqa: F401  (registers rbf)
import pythonLibs.regressionHandler.models.IdwModel  # noqa: F401  (registers idw)
import pythonLibs.regressionHandler.models.NonlinearModel  # noqa: F401  (registers nonlinear)
import pythonLibs.regressionHandler.models.SplineModel  # noqa: F401  (registers spline)
import pythonLibs.regressionHandler.models.LocalRegressionModel  # noqa: F401  (registers loess)
from pythonLibs.regressionHandler.models.ModelLibrary import LIBRARY, libraryCatalog

SHORTHANDS = {
    "linear": "Linear least squares",
    "quadratic": "Full quadratic response surface",
    "poly<N>": "Polynomial of total degree N, ordinary least squares",
    "ortho<N>": "Legendre polynomial of degree N with GCV ridge (stable at high N)",
    "ridge-poly<N>": "Polynomial degree N with GCV-tuned ridge",
    "robust-poly<N>": "Polynomial degree N, Huber IRLS (outlier resistant)",
    "lasso-poly<N>": "Polynomial degree N, Lasso term selection (alpha=1e-3)",
    "pspline": "Penalized cubic B-spline, smoothing chosen by GCV (nx <= 3)",
    "rbf-ridge": "Radial-basis regression (k-means centers, GCV ridge)",
    "gp": "Gaussian process / Kriging, Matern 5/2 ARD kernel, estimated noise",
    "rbf-smooth": "RBF (thin-plate spline) with leave-one-out smoothing",
    "<libraryForm>": "Any ModelLibrary form name (powerLaw, logistic, exponentialDecay, ...) as a nonlinear model",
}

_POLY = re.compile(r"^(?:(ridge|robust|lasso)-)?poly(\d+)$")
_ORTHO = re.compile(r"^ortho(\d+)$")


def expandShorthand(name: str) -> Optional[dict]:
    """Return the dict spec for a shorthand name, or None if it is not one."""
    if name == "linear":
        return {"type": "linearBasis", "basis": {"type": "polynomial", "degree": 1}, "solver": "ols"}
    if name == "quadratic":
        return {"type": "linearBasis", "basis": {"type": "polynomial", "degree": 2}, "solver": "ols"}
    m = _POLY.match(name)
    if m:
        solver = {"ridge": {"type": "ridge", "alpha": "gcv"}, "robust": {"type": "robust", "loss": "huber"},
                  "lasso": {"type": "elasticNet", "alpha": 1e-3, "l1Ratio": 1.0}, None: "ols"}[m.group(1)]
        return {"type": "linearBasis", "basis": {"type": "polynomial", "degree": int(m.group(2))}, "solver": solver}
    m = _ORTHO.match(name)
    if m:
        return {"type": "linearBasis", "basis": {"type": "orthogonalPolynomial", "degree": int(m.group(1))},
                "solver": {"type": "ridge", "alpha": "gcv"}}
    if name == "pspline":
        return {"type": "linearBasis", "basis": {"type": "bspline", "nSegments": 20, "degree": 3},
                "solver": {"type": "ridge", "penalty": "smoothness", "alpha": "gcv"}}
    if name == "gp":
        return {"type": "kriging", "corr": "matern52", "nugget": "auto"}
    if name == "rbf-smooth":
        return {"type": "rbf", "kernel": "thinPlateSpline", "smoothing": "loo"}
    if name == "rbf-ridge":
        return {"type": "linearBasis", "basis": {"type": "radial", "kernel": "thinPlateSpline"},
                "solver": {"type": "ridge", "alpha": "gcv"}}
    return None


def createModel(spec: Any, **overrides: Any) -> SurrogateModelBase:
    """Build an untrained model from a spec (see module docstring)."""
    if isinstance(spec, SurrogateModelBase):
        model = spec.clone()
        model.options.update(overrides)
        return model
    if isinstance(spec, str):
        expanded = expandShorthand(spec)
        if expanded is None and spec in LIBRARY and spec not in registry("model"):
            expanded = {"type": "nonlinear", "library": spec}
        spec = expanded if expanded is not None else {"type": spec}
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError(f"model spec must be a name, shorthand or dict with 'type': {spec!r}")
    options = {k: v for k, v in spec.items() if k != "type"}
    options.update(overrides)
    return registry("model").get(spec["type"])(**options)


def availableModels() -> list[dict]:
    """Registered model types with their declared options, plus the shorthands."""
    out = []
    for name, cls in registry("model").items():
        proto = cls(library="linear") if name == "nonlinear" else cls()
        out.append({"type": name, "description": (cls.__doc__ or "").strip().split("\n")[0],
                    "supports": dict(proto.supports), "options": proto.options.describe()})
    out.extend({"type": k, "description": v, "shorthand": True} for k, v in SHORTHANDS.items())
    out.extend({"type": f["name"], "description": f["description"], "expression": f["expression"],
                "params": f["params"], "libraryForm": True} for f in libraryCatalog())
    return out
