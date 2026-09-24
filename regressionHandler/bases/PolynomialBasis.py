"""Multivariate monomial and orthogonal-polynomial bases."""
from __future__ import annotations

from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.bases.BasisBase import (BasisBase, monomialName, polynomialExponents,
                                                          productOfFactors)
from pythonLibs.regressionHandler.core.Registry import registry


def _declarePolynomial(declare) -> None:
    declare("degree", 2, types=int, lower=0, desc="Maximum polynomial degree")
    declare("mode", "total", values=("total", "tensor"),
            desc="'total': total degree <= degree; 'tensor': every exponent <= degree")
    declare("interactionOrder", None, types=int, lower=1,
            desc="Max number of distinct variables per term (1 = no cross terms); None = unlimited")
    declare("includeBias", True, types=bool, desc="Include the constant term")


@registry("basis").register("polynomial")
class PolynomialBasis(BasisBase):
    """Monomials x0^a * x1^b * ... (degree 1: linear model, degree 2: quadratic response surface).

    With ``scaling="minmax"`` inputs are mapped to [-1, 1] first, which keeps
    high degrees well conditioned; coefficients then refer to the scaled
    variables (named ``z0, z1, ...``). With ``scaling="none"`` coefficients are
    in physical units and the least-squares solver's column equilibration
    handles moderate degrees.
    """

    def _declareOptions(self, declare) -> None:
        _declarePolynomial(declare)
        declare("scaling", "none", values=("none", "minmax"), desc="Input scaling before building terms")

    def _fit(self, x: np.ndarray) -> None:
        o = self.options
        self._exps = polynomialExponents(self.nx, o["degree"], o["interactionOrder"], o["mode"], o["includeBias"])
        if o["scaling"] == "minmax":
            self._lo = x.min(axis=0)
            span = x.max(axis=0) - self._lo
            self._span = np.where(span > 0, span, 1.0)
        else:
            self._lo = np.zeros(self.nx)
            self._span = np.full(self.nx, 2.0)

    def _scale(self, x: np.ndarray) -> np.ndarray:
        if self.options["scaling"] == "minmax":
            return 2.0 * (x - self._lo) / self._span - 1.0
        return x

    @property
    def nTerms(self) -> int:
        return self._exps.shape[0]

    @property
    def exponents(self) -> np.ndarray:
        return self._exps.copy()

    def _powerTables(self, z: np.ndarray) -> list[np.ndarray]:
        deg = int(self._exps.max()) if self._exps.size else 0
        return [z[:, [j]] ** np.arange(deg + 1) for j in range(self.nx)]

    def transform(self, x: np.ndarray) -> np.ndarray:
        return productOfFactors(self._powerTables(self._scale(x)), self._exps)

    def derivative(self, x: np.ndarray, kx: int) -> np.ndarray:
        z = self._scale(x)
        tables = self._powerTables(z)
        deg = tables[kx].shape[1] - 1
        powers = np.arange(deg + 1)
        dTable = np.zeros_like(tables[kx])
        dTable[:, 1:] = powers[1:] * tables[kx][:, :-1]
        tables[kx] = dTable
        exps = self._exps
        phi = np.ones((x.shape[0], exps.shape[0]))
        for j, table in enumerate(tables):
            phi *= table[:, exps[:, j]]
        chain = 2.0 / self._span[kx] if self.options["scaling"] == "minmax" else 1.0
        return phi * chain

    @property
    def biasMask(self) -> np.ndarray:
        return self._exps.sum(axis=1) == 0

    def termNames(self, featureNames: Optional[list[str]] = None) -> list[str]:
        names = featureNames or [f"x{i}" for i in range(self.nx)]
        if self.options["scaling"] == "minmax":
            names = [f"z({n})" for n in names]
        return [monomialName(e, names) for e in self._exps]

    def _stateToDict(self) -> dict:
        if self.nx is None:
            return {}
        return {"nx": self.nx, "exps": self._exps.tolist(), "lo": self._lo.tolist(), "span": self._span.tolist()}

    def _stateFromDict(self, state: dict) -> None:
        self.nx = state["nx"]
        self._exps = np.array(state["exps"], dtype=int).reshape(-1, self.nx)
        self._lo = np.array(state["lo"], dtype=float)
        self._span = np.array(state["span"], dtype=float)


def _legendreTables(t: np.ndarray, deg: int) -> tuple[np.ndarray, np.ndarray]:
    """P_k(t) and P_k'(t) for k = 0..deg (Bonnet recurrence)."""
    n = t.size
    p = np.zeros((n, deg + 1))
    dp = np.zeros((n, deg + 1))
    p[:, 0] = 1.0
    if deg >= 1:
        p[:, 1] = t
        dp[:, 1] = 1.0
    for k in range(1, deg):
        p[:, k + 1] = ((2 * k + 1) * t * p[:, k] - k * p[:, k - 1]) / (k + 1)
        dp[:, k + 1] = dp[:, k - 1] + (2 * k + 1) * p[:, k]
    return p, dp


def _chebyshevTables(t: np.ndarray, deg: int) -> tuple[np.ndarray, np.ndarray]:
    """T_k(t) and T_k'(t) (with U_{k-1} for the derivative)."""
    n = t.size
    tt = np.zeros((n, deg + 1))
    u = np.zeros((n, deg + 1))
    tt[:, 0] = 1.0
    u[:, 0] = 1.0
    if deg >= 1:
        tt[:, 1] = t
        u[:, 1] = 2.0 * t
    for k in range(1, deg):
        tt[:, k + 1] = 2.0 * t * tt[:, k] - tt[:, k - 1]
        u[:, k + 1] = 2.0 * t * u[:, k] - u[:, k - 1]
    dt = np.zeros_like(tt)
    dt[:, 1:] = np.arange(1, deg + 1) * u[:, :-1]
    return tt, dt


@registry("basis").register("orthogonalPolynomial")
class OrthogonalPolynomialBasis(BasisBase):
    """Products of Legendre or Chebyshev polynomials on inputs mapped to [-1, 1].

    Numerically the most robust polynomial basis for high degrees and many
    inputs (response surfaces, polynomial chaos style expansions).
    """

    def _declareOptions(self, declare) -> None:
        _declarePolynomial(declare)
        declare("family", "legendre", values=("legendre", "chebyshev"), desc="Polynomial family")

    def _fit(self, x: np.ndarray) -> None:
        o = self.options
        self._exps = polynomialExponents(self.nx, o["degree"], o["interactionOrder"], o["mode"], o["includeBias"])
        self._lo = x.min(axis=0)
        span = x.max(axis=0) - self._lo
        self._span = np.where(span > 0, span, 1.0)

    @property
    def nTerms(self) -> int:
        return self._exps.shape[0]

    def _tables(self, x: np.ndarray):
        z = 2.0 * (x - self._lo) / self._span - 1.0
        deg = int(self._exps.max()) if self._exps.size else 0
        maker = _legendreTables if self.options["family"] == "legendre" else _chebyshevTables
        return [maker(z[:, j], deg) for j in range(self.nx)]

    def transform(self, x: np.ndarray) -> np.ndarray:
        return productOfFactors([t[0] for t in self._tables(x)], self._exps)

    def derivative(self, x: np.ndarray, kx: int) -> np.ndarray:
        tables = self._tables(x)
        factors = [t[0] for t in tables]
        factors[kx] = tables[kx][1] * (2.0 / self._span[kx])
        phi = np.ones((x.shape[0], self._exps.shape[0]))
        for j, table in enumerate(factors):
            phi *= table[:, self._exps[:, j]]
        return phi

    @property
    def biasMask(self) -> np.ndarray:
        return self._exps.sum(axis=1) == 0

    def termNames(self, featureNames: Optional[list[str]] = None) -> list[str]:
        names = featureNames or [f"x{i}" for i in range(self.nx)]
        letter = "P" if self.options["family"] == "legendre" else "T"
        out = []
        for e in self._exps:
            parts = [f"{letter}{k}({n})" for n, k in zip(names, e) if k > 0]
            out.append("*".join(parts) if parts else "1")
        return out

    _stateToDict = PolynomialBasis._stateToDict
    _stateFromDict = PolynomialBasis._stateFromDict
