"""Dimensionally homogeneous regression through Buckingham pi groups.

    y = prod_i x_i^b_i * g(pi_1(x), ..., pi_k(x)),   pi_j(x) = prod_i x_i^a_ji

The exponents come from the dimension matrix of the inputs (``inputDimensions``)
and the output (``outputDimension``) by exact rational arithmetic
(``constraints.DimensionalAnalysis``). Any base model learns g on the k < nx
dimensionless groups (optionally in log space), so every prediction is
dimensionally consistent by construction, the model is invariant under a
change of units, and it extrapolates along similarity lines. When no group
remains (k = 0), g is a constant fitted by least squares on the scaled output.

    DimensionlessModel(inputDimensions=[{"L": 1}, {"L": 1, "T": -2}], outputDimension={"T": 1},
                       model="linear")          # pendulum: period = sqrt(L / g) * C

Inputs involved in groups must be positive.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.constraints.DimensionalAnalysis import (buckinghamPi, dimensionMatrix,
                                                                          scalingExponents)
from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase


@registry("model").register("dimensionless")
class DimensionlessModel(SurrogateModelBase):
    """Regression on Buckingham pi groups (dimensionally homogeneous by construction)."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("inputDimensions", [], types=list, desc="Per input: {'M': .., 'L': .., 'T': .., 'Theta': ..} or exponent list")
        d("outputDimension", {}, types=(dict, list), desc="Dimension of the output (same form)")
        d("repeating", None, types=list, desc="Repeating input indices (default: automatic)")
        d("model", "linear", types=(str, dict, object), desc="Base model on the dimensionless groups")
        d("logGroups", True, types=bool, desc="Fit the base model on log(pi) (power laws become linear)")
        d("logOutput", False, types=bool, desc="Fit log of the scaled output (multiplicative errors)")
        self.supports.update(multiOutput=False, weights=True, variances=True, derivatives=False)

    def _validateOptions(self) -> None:
        if self.options["inputDimensions"]:
            self._analyse()

    def _analyse(self) -> None:
        dims = list(self.options["inputDimensions"])
        allDims = dims + [self.options["outputDimension"]]
        full = dimensionMatrix(allDims)
        dmat, target = full[:, :-1], full[:, -1]
        pi = buckinghamPi(dmat, self.options["repeating"])
        self._groups = pi["groups"].astype(float)
        self._repeating = pi["repeating"]
        self._scale = scalingExponents(dmat, target, self._repeating)
        self._dmat = dmat

    # ------------------------------------------------------------------ transforms
    def _pis(self, x) -> np.ndarray:
        used = np.any(self._groups != 0, axis=0) | (self._scale != 0)
        if np.any(x[:, used] <= 0):
            raise ValueError("inputs entering dimensionless groups must be positive")
        logx = np.log(np.where(x > 0, x, 1.0))
        z = logx @ self._groups.T
        return z if self.options["logGroups"] else np.exp(z)

    def _yScale(self, x) -> np.ndarray:
        return np.exp(np.log(np.where(x > 0, x, 1.0)) @ self._scale)

    # ------------------------------------------------------------------ training
    def _train(self) -> None:
        if len(self.options["inputDimensions"]) != self.nx:
            raise ValueError(f"inputDimensions needs one entry per input ({self.nx})")
        self._analyse()
        x, y, w = self.xt, self.yt[:, 0], self.wt
        s = self._yScale(x)
        pi = self._pis(x)
        target = y / s
        if self.options["logOutput"]:
            if np.any(target <= 0):
                raise ValueError("logOutput needs a positive output")
            target = np.log(target)
        self._k = pi.shape[1]
        if self._k == 0:
            self._const = float(np.sum(w * target) / np.sum(w))
            r = target - self._const
            self._constVar = float(np.sum(w * r * r) / max(np.sum(w) - 1.0, 1.0))
            self._base = None
            return
        from pythonLibs.regressionHandler.models.ModelFactory import createModel
        self._base = createModel(copy.deepcopy(self.options["model"]))
        names = ["pi%d" % (j + 1) for j in range(self._k)]
        self._base.fit(pi, target, None if np.all(w == 1.0) else w, names, [self.outputNames[0]])
        self.supports["variances"] = self._base.supports["variances"]

    # ------------------------------------------------------------------ prediction
    def _g(self, x):
        if self._base is None:
            return np.full(x.shape[0], self._const)
        return self._base.predictValues(self._pis(x))[:, 0]

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        g = self._g(x)
        if self.options["logOutput"]:
            g = np.exp(g)
        return (self._yScale(x) * g)[:, None]

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        s = self._yScale(x)
        if self._base is None:
            v = np.full(x.shape[0], self._constVar / self.xt.shape[0] + (self._constVar if kind == "prediction" else 0))
        else:
            v = self._base.predictVariances(self._pis(x), kind)[:, 0]
        if self.options["logOutput"]:
            v = v * np.exp(self._g(x)) ** 2                     # delta method
        return (s * s * v)[:, None]

    def _effectiveParams(self):
        return 1.0 if self._base is None else float(np.ravel(self._base.nEffectiveParams)[0])

    # ------------------------------------------------------------------ reporting
    def groups(self) -> dict:
        """Dimensionless groups as exponent tables and readable products."""
        names = self.featureNames or [f"x{i}" for i in range(self._groups.shape[1])]

        def text(vec):
            parts = [f"{n}^{int(e) if float(e).is_integer() else e}" for n, e in zip(names, vec) if e != 0]
            return " * ".join(parts) if parts else "1"
        return {"groups": self._groups.tolist(), "products": [text(g) for g in self._groups],
                "outputScale": self._scale.tolist(), "outputScaleProduct": text(self._scale),
                "repeating": [names[i] for i in self._repeating]}

    @property
    def baseModel(self) -> Optional[SurrogateModelBase]:
        return self._base

    def _stateToDict(self) -> dict:
        return {"base": None if self._base is None else self._base._toDictPlain(True),
                "const": getattr(self, "_const", None), "constVar": getattr(self, "_constVar", None)}

    def _stateFromDict(self, state: dict) -> None:
        self._analyse()
        self._k = self._groups.shape[0]
        self._base = None if state["base"] is None else SurrogateModelBase.fromDict(state["base"])
        self._const, self._constVar = state["const"], state["constVar"]
