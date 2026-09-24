"""Basis-function components: X (n, nx) -> design matrix Phi (n, p).

A basis is fitted to the training inputs (to learn ranges, knots, centers)
and then maps any input to its design matrix. Bases optionally provide
analytic derivatives dPhi/dx_k (so linear-in-parameter models get exact
gradients) and a roughness penalty matrix (for penalized splines).
"""
from __future__ import annotations

import itertools
from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.core.Registry import ComponentBase


class BasisBase(ComponentBase):
    componentKind = "basis"

    def __init__(self, **options) -> None:
        super().__init__(**options)
        self.nx: Optional[int] = None

    # -- contract -----------------------------------------------------------
    def fit(self, x: np.ndarray) -> "BasisBase":
        self.nx = x.shape[1]
        self._fit(x)
        return self

    def _fit(self, x: np.ndarray) -> None:
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, x: np.ndarray, kx: int) -> np.ndarray:
        """dPhi/dx_kx (n, p)."""
        raise NotImplementedError

    @property
    def hasDerivative(self) -> bool:
        return type(self).derivative is not BasisBase.derivative

    def termNames(self, featureNames: Optional[list[str]] = None) -> list[str]:
        return [f"b{i}" for i in range(self.nTerms)]

    @property
    def nTerms(self) -> int:
        raise NotImplementedError

    @property
    def biasMask(self) -> np.ndarray:
        """True for columns that must not be shrunk by ridge penalties (intercept)."""
        return np.zeros(self.nTerms, dtype=bool)

    def penaltyMatrix(self) -> Optional[np.ndarray]:
        """Roughness penalty (p, p) or None when the basis has none."""
        return None

    def fitTransform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def _stateToDict(self) -> dict:
        return {"nx": self.nx} if self.nx is not None else {}

    def _stateFromDict(self, state: dict) -> None:
        self.nx = state.get("nx")


def polynomialExponents(nx: int, degree: int, interactionOrder: Optional[int] = None,
                        mode: str = "total", includeBias: bool = True) -> np.ndarray:
    """Exponent matrix (p, nx) of a multivariate polynomial basis.

    mode "total":  all monomials with total degree <= degree
    mode "tensor": all monomials with each exponent <= degree
    interactionOrder limits the number of distinct variables in one term
    (1 = additive model without cross terms).
    """
    if mode == "total":
        rows = []
        for total in range(0, degree + 1):
            for combo in itertools.combinations_with_replacement(range(nx), total):
                e = np.zeros(nx, dtype=int)
                for c in combo:
                    e[c] += 1
                rows.append(e)
        exps = np.array(rows, dtype=int).reshape(-1, nx)
    elif mode == "tensor":
        grids = np.meshgrid(*[np.arange(degree + 1)] * nx, indexing="ij")
        exps = np.stack([g.ravel() for g in grids], axis=1)
        order = np.lexsort(tuple(exps[:, ::-1].T) + (exps.sum(axis=1),))
        exps = exps[order]
    else:
        raise ValueError("mode must be 'total' or 'tensor'")
    if interactionOrder is not None:
        exps = exps[(exps > 0).sum(axis=1) <= interactionOrder]
    if not includeBias:
        exps = exps[exps.sum(axis=1) > 0]
    return exps


def monomialName(exponent: np.ndarray, featureNames: list[str]) -> str:
    parts = []
    for name, e in zip(featureNames, exponent):
        if e == 1:
            parts.append(name)
        elif e > 1:
            parts.append(f"{name}^{e}")
    return "*".join(parts) if parts else "1"


def productOfFactors(factors: list[np.ndarray], exps: np.ndarray) -> np.ndarray:
    """Phi[:, t] = prod_j factors[j][:, exps[t, j]] for per-feature factor tables."""
    phi = np.ones((factors[0].shape[0], exps.shape[0]))
    for j, table in enumerate(factors):
        col = exps[:, j]
        if np.any(col):
            phi *= table[:, col]
    return phi
