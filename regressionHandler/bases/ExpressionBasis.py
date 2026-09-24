"""User-defined basis terms written as safe text expressions.

Example: y = a + b*x + c/x^2 + d*log(x) (linear in a, b, c, d)::

    ExpressionBasis(terms=["x", "1/x**2", "log(x)"], variables=["x"])

Terms are parsed by ``SafeExpression`` (no arbitrary code), so the basis
serializes to JSON. Derivatives are left to the model's finite differences.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.bases.BasisBase import BasisBase
from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.core.SafeExpression import SafeExpression


@registry("basis").register("expression")
class ExpressionBasis(BasisBase):

    def _declareOptions(self, declare) -> None:
        declare("terms", [], types=list, desc="Term expressions, e.g. ['x0', '1/x0', 'x0*x1']")
        declare("variables", None, types=list, desc="Names bound to input columns (default x0, x1, ...)")
        declare("includeBias", True, types=bool, desc="Prepend a constant term")

    def _variables(self, nx: int) -> list[str]:
        names = self.options["variables"] or [f"x{i}" for i in range(nx)]
        if len(names) != nx:
            raise ValueError(f"{len(names)} variable names given for {nx} inputs")
        return list(names)

    def _fit(self, x: np.ndarray) -> None:
        if not self.options["terms"]:
            raise ValueError("ExpressionBasis needs at least one term")
        names = self._variables(self.nx)
        self._exprs = [SafeExpression(t, names) for t in self.options["terms"]]

    @property
    def nTerms(self) -> int:
        return len(self.options["terms"]) + int(self.options["includeBias"])

    def transform(self, x: np.ndarray) -> np.ndarray:
        env = {n: x[:, i] for i, n in enumerate(self._variables(x.shape[1]))}
        cols = [np.ones(x.shape[0])] if self.options["includeBias"] else []
        for e in self._exprs:
            cols.append(np.broadcast_to(np.asarray(e.evaluate(env), dtype=float), (x.shape[0],)))
        phi = np.column_stack(cols)
        if not np.all(np.isfinite(phi)):
            bad = [t for t, c in zip(self.options["terms"], phi.T[int(self.options["includeBias"]):])
                   if not np.all(np.isfinite(c))]
            raise ValueError(f"basis terms are not finite on these inputs: {bad}")
        return phi

    @property
    def biasMask(self) -> np.ndarray:
        mask = np.zeros(self.nTerms, dtype=bool)
        if self.options["includeBias"]:
            mask[0] = True
        return mask

    def termNames(self, featureNames: Optional[list[str]] = None) -> list[str]:
        return (["1"] if self.options["includeBias"] else []) + list(self.options["terms"])

    def _stateFromDict(self, state: dict) -> None:
        self.nx = state.get("nx")
        if self.nx is not None:
            self._exprs = [SafeExpression(t, self._variables(self.nx)) for t in self.options["terms"]]
