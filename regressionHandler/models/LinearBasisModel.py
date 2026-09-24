"""Linear-in-parameter regression: y = Phi(x) @ c for any basis and solver.

This one class covers linear and quadratic least squares, response surfaces, polynomial regression, P-splines, tensor splines and
RBF regression::

    LinearBasisModel(basis={"type": "polynomial", "degree": 2})                 # quadratic RSM
    LinearBasisModel(basis={"type": "orthogonalPolynomial", "degree": 6},
                     solver={"type": "ridge", "alpha": "gcv"})                  # stable high-order RSM
    LinearBasisModel(basis={"type": "bspline", "nSegments": 20},
                     solver={"type": "ridge", "penalty": "smoothness"})          # P-spline, GCV
    LinearBasisModel(basis="radial", solver="ridge")                            # RBF regression
    LinearBasisModel(basis={"type": "polynomial", "degree": 1}, solver="robust")  # outlier resistant

Outputs share the design matrix, so all ny outputs are solved together.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.core.FitResult import FitResult
from pythonLibs.regressionHandler.core.Registry import buildComponent, registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase
from pythonLibs.regressionHandler.solvers.LinearSolvers import SolveResult

import pythonLibs.regressionHandler.bases  # noqa: F401  (registers bases)
import pythonLibs.regressionHandler.solvers  # noqa: F401  (registers solvers)


@registry("model").register("linearBasis")
class LinearBasisModel(SurrogateModelBase):
    """Linear-in-parameter model: any basis x any linear solver (polynomial, RSM, P-spline, RBF regression)."""

    def _initialize(self) -> None:
        declare = self.options.declare
        declare("basis", {"type": "polynomial", "degree": 1}, types=(str, dict, object),
                desc="Basis spec: name, {'type': name, **options} or a BasisBase instance")
        declare("solver", "ols", types=(str, dict, object),
                desc="Solver spec: 'ols', 'ridge', 'elasticNet', 'robust' or {'type': ..., **options}")
        self.supports.update(multiOutput=True, variances=True, derivatives=True, parameterInference=True)
        self._basis = None
        self._solution: Optional[SolveResult] = None

    def _validateOptions(self) -> None:
        buildComponent("basis", copy.deepcopy(self.options["basis"]))
        buildComponent("solver", copy.deepcopy(self.options["solver"]))

    # ------------------------------------------------------------------ training
    def _train(self) -> None:
        self._basis = copy.deepcopy(buildComponent("basis", self.options["basis"]))
        solver = buildComponent("solver", self.options["solver"])
        phi = self._basis.fitTransform(self.xt)
        if phi.shape[1] == 0:
            raise ValueError("basis produced no terms")
        if phi.shape[1] > phi.shape[0] and solver.registryName == "ols":
            raise ValueError(f"{phi.shape[1]} basis terms but only {phi.shape[0]} samples; "
                             "use a smaller basis or solver='ridge'")
        penalty = self._basis.penaltyMatrix() if solver.usesPenalty else None
        self._solution = solver.solve(phi, self.yt, self.wt, penalty, self._basis.biasMask)
        self._updateSupports()
        self.result = self._buildResult()

    def _updateSupports(self) -> None:
        self.supports["variances"] = self._solution.covUnscaled is not None
        self.supports["parameterInference"] = self._solution.covUnscaled is not None
        self.supports["derivatives"] = bool(self._basis.hasDerivative)

    def _buildResult(self) -> Optional[FitResult]:
        sol = self._solution
        if sol.covUnscaled is None:
            return None
        cov = sol.covUnscaled * sol.sigma2[:, None, None]
        return FitResult(parameterNames=self.termNames, params=sol.coef, covariance=cov,
                         dofResid=float(np.min(sol.dofResid)), sigma2=sol.sigma2, outputNames=self.outputNames)

    # ------------------------------------------------------------------ prediction
    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        return self._basis.transform(x) @ self._solution.coef

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        phi = self._basis.transform(x)
        sol = self._solution
        out = np.empty((x.shape[0], sol.coef.shape[1]))
        for j in range(out.shape[1]):
            out[:, j] = np.einsum("ij,jk,ik->i", phi, sol.covUnscaled[j], phi) * sol.sigma2[j]
        if kind == "prediction":
            out = out + sol.sigma2[None, :]
        return out

    def _predictDerivatives(self, x: np.ndarray, kx: int) -> np.ndarray:
        return self._basis.derivative(x, kx) @ self._solution.coef

    def _effectiveParams(self):
        return self._solution.edf.copy()

    def _intervalDof(self) -> Optional[float]:
        dof = float(np.min(self._solution.dofResid))
        return dof if dof > 0 else None

    # ------------------------------------------------------------------ extras
    @property
    def basis(self):
        return self._basis

    @property
    def coefficients(self) -> np.ndarray:
        """(p, ny) fitted coefficients."""
        self._checkTrained()
        return self._solution.coef.copy()

    @property
    def termNames(self) -> list[str]:
        return self._basis.termNames(self.featureNames)

    @property
    def solution(self) -> SolveResult:
        self._checkTrained()
        return self._solution

    def looResiduals(self) -> np.ndarray:
        """Exact leave-one-out residuals e_i / (1 - h_ii), shape (n, ny).

        Available for solvers that report leverages (ols, ridge with fixed
        penalty, robust); the ridge penalty weight is held fixed.
        """
        self._checkTrained()
        sol = self._solution
        if sol.hatDiag is None:
            raise NotImplementedError("this solver does not report leverages")
        resid = self.yt - self._predictValues(self.xt)
        return resid / np.maximum(1.0 - sol.hatDiag, 1e-12)

    @property
    def leverage(self) -> Optional[np.ndarray]:
        return None if self._solution.hatDiag is None else self._solution.hatDiag.copy()

    def equation(self, output: int = 0, precision: int = 6) -> str:
        """Human readable fitted equation for one output."""
        terms = []
        for name, c in zip(self.termNames, self._solution.coef[:, output]):
            if c == 0.0:
                continue
            coef = f"{c:.{precision}g}"
            terms.append(coef if name == "1" else f"{coef}*{name}")
        return f"{self.outputNames[output]} = " + (" + ".join(terms).replace("+ -", "- ") if terms else "0")

    # ------------------------------------------------------------------ persistence
    def _stateToDict(self) -> dict:
        sol = self._solution
        return {
            "basis": self._basis.toDict(),
            "coef": sol.coef.tolist(),
            "covUnscaled": None if sol.covUnscaled is None else sol.covUnscaled.tolist(),
            "edf": sol.edf.tolist(), "sse": sol.sse.tolist(), "sigma2": sol.sigma2.tolist(),
            "rank": sol.rank, "conditionNumber": sol.conditionNumber, "alpha": sol.alpha.tolist(),
            "nSamples": int(self.xt.shape[0]) if self.xt is not None else int(self.metrics[0].nSamples),
        }

    def _stateFromDict(self, state: dict) -> None:
        self._basis = buildComponent("basis", state["basis"])
        n = int(state["nSamples"])
        coef = np.array(state["coef"], dtype=float)
        ny = coef.shape[1]
        self._solution = SolveResult(
            coef=coef,
            covUnscaled=None if state["covUnscaled"] is None else np.array(state["covUnscaled"], dtype=float),
            hatDiag=None, edf=np.array(state["edf"]), sse=np.array(state["sse"]),
            sigma2=np.array(state["sigma2"], dtype=float), rank=int(state["rank"]),
            conditionNumber=float(state["conditionNumber"]), alpha=np.array(state["alpha"]),
            weights=np.ones((n, ny)))
        self._updateSupports()
