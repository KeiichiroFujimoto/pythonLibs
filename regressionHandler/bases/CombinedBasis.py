"""Concatenation of several bases (e.g. polynomial trend + radial terms)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.bases.BasisBase import BasisBase
from pythonLibs.regressionHandler.core.Registry import buildComponent, registry


@registry("basis").register("combined")
class CombinedBasis(BasisBase):
    """Columns of every child basis side by side.

    Only one child should carry an intercept (set ``includeBias=False`` on
    the others) or the design matrix will be rank deficient.
    """

    def _declareOptions(self, declare) -> None:
        declare("bases", [], types=list, desc="Child basis specs or instances")

    def _children(self) -> list[BasisBase]:
        if not hasattr(self, "_built"):
            self._built = [buildComponent("basis", b) for b in self.options["bases"]]
            self.options["bases"] = self._built
        return self._built

    def _fit(self, x: np.ndarray) -> None:
        if not self.options["bases"]:
            raise ValueError("CombinedBasis needs at least one child basis")
        for b in self._children():
            b.fit(x)

    @property
    def nTerms(self) -> int:
        return sum(b.nTerms for b in self._children())

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.hstack([b.transform(x) for b in self._children()])

    @property
    def hasDerivative(self) -> bool:
        return all(b.hasDerivative for b in self._children())

    def derivative(self, x: np.ndarray, kx: int) -> np.ndarray:
        return np.hstack([b.derivative(x, kx) for b in self._children()])

    @property
    def biasMask(self) -> np.ndarray:
        return np.concatenate([b.biasMask for b in self._children()])

    def penaltyMatrix(self) -> Optional[np.ndarray]:
        blocks = [b.penaltyMatrix() for b in self._children()]
        if all(p is None for p in blocks):
            return None
        sizes = [b.nTerms for b in self._children()]
        out = np.zeros((sum(sizes), sum(sizes)))
        start = 0
        for p, s in zip(blocks, sizes):
            if p is not None:
                out[start:start + s, start:start + s] = p
            start += s
        return out

    def termNames(self, featureNames: Optional[list[str]] = None) -> list[str]:
        return [n for b in self._children() for n in b.termNames(featureNames)]
