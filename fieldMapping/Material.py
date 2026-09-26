"""Thermal material: density and temperature-dependent specific heat with exact energy integrals.

    e(T) = integral from T0 to T of cp(t) dt        (J / kg, relative to the table's first temperature)

cp is a constant or a piecewise-linear table [[T, cp], ...] (constant beyond
its ends); e(T) is then piecewise quadratic and evaluated exactly, as is its
derivative cp(T), so energy constraints and their Jacobians are exact.
"""
from __future__ import annotations

from typing import Union

import numpy as np


class Material:
    """rho [kg/m^3] and cp(T) [J/(kg K)] with exact sensible energy per unit volume rho * e(T)."""

    def __init__(self, density: float, cp: Union[float, list, np.ndarray], name: str = "") -> None:
        self.density = float(density)
        if self.density <= 0:
            raise ValueError("density must be positive")
        tab = np.atleast_2d(np.asarray(cp, dtype=float))
        if tab.size == 1:
            tab = np.array([[0.0, float(tab.ravel()[0])], [1.0, float(tab.ravel()[0])]])
        if tab.shape[1] != 2 or tab.shape[0] < 1:
            raise ValueError("cp must be a number or a table [[T, cp], ...]")
        if tab.shape[0] == 1:
            tab = np.vstack([tab, tab + [1.0, 0.0]])
        if np.any(np.diff(tab[:, 0]) <= 0):
            raise ValueError("cp table temperatures must increase")
        if np.any(tab[:, 1] <= 0):
            raise ValueError("cp must be positive")
        self.table = tab
        self.name = name
        t, c = tab[:, 0], tab[:, 1]
        seg = 0.5 * (c[1:] + c[:-1]) * np.diff(t)
        self._eKnot = np.concatenate([[0.0], np.cumsum(seg)])               # e at the table temperatures

    @classmethod
    def fromSpec(cls, spec) -> "Material":
        """Material from ``{"density", "cp", "name"?}`` (cp a number or [[T, cp], ...]); instances pass through."""
        if isinstance(spec, Material):
            return spec
        return cls(spec["density"], spec["cp"], spec.get("name", ""))

    def toDict(self) -> dict:
        """``{"density", "cp" table, "name"}``, accepted by ``fromSpec``."""
        return {"density": self.density, "cp": self.table.tolist(), "name": self.name}

    def cp(self, temperature) -> np.ndarray:
        """Specific heat [J/(kg K)] at ``temperature`` (piecewise linear, constant beyond the table)."""
        t = np.asarray(temperature, dtype=float)
        return np.interp(t, self.table[:, 0], self.table[:, 1])

    def energy(self, temperature) -> np.ndarray:
        """Sensible energy per unit mass e(T) (exact for the piecewise-linear cp)."""
        T = np.asarray(temperature, dtype=float)
        t, c = self.table[:, 0], self.table[:, 1]
        k = np.clip(np.searchsorted(t, T, side="right") - 1, 0, t.size - 2)
        below = T < t[0]
        above = T > t[-1]
        t0, c0 = t[k], c[k]
        slope = (c[k + 1] - c0) / (t[k + 1] - t0)
        dt = T - t0
        inside = self._eKnot[k] + c0 * dt + 0.5 * slope * dt * dt
        return np.where(below, c[0] * (T - t[0]), np.where(above, self._eKnot[-1] + c[-1] * (T - t[-1]), inside))

    def inverseEnergy(self, energy, tol: float = 1e-12) -> np.ndarray:
        """Temperature with e(T) = energy (Newton; e is strictly increasing)."""
        e = np.asarray(energy, dtype=float)
        T = self.table[0, 0] + e / self.table[0, 1]
        for _ in range(60):
            step = (self.energy(T) - e) / self.cp(T)
            T = T - step
            if np.all(np.abs(step) <= tol * np.maximum(1.0, np.abs(T))):
                break
        return T

    def volumetricEnergy(self, temperature) -> np.ndarray:
        """Sensible energy per unit volume rho * e(T) [J/m^3]."""
        return self.density * self.energy(temperature)

    def volumetricCapacity(self, temperature) -> np.ndarray:
        """Volumetric heat capacity rho * cp(T) [J/(m^3 K)]."""
        return self.density * self.cp(temperature)

    def __repr__(self) -> str:
        return f"<Material {self.name or ''} rho={self.density:g}, cp table {self.table.shape[0]} points>"
