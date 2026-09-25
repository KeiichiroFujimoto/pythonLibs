"""Multilinear interpolation on a full rectilinear grid, any number of inputs.

The training points must cover every node of a tensor grid (the unique
coordinates per input, in any order); the model interpolates linearly in each
input between the enclosing nodes, which reproduces scipy
``RegularGridInterpolator(method="linear")``.

Outside the grid the model raises (``extrapolation="error"``), continues the
edge cells linearly (``"extend"``), holds the edge values (``"clamp"``) or
returns ``fillValue`` (``"fill"``). The instance is callable, ``model(xi)``
with ``xi`` of shape (..., nx), returning (...) for a single output.
"""
from __future__ import annotations

import itertools

import numpy as np

from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase

GRID_EXTRAPOLATION_MODES = ("error", "extend", "clamp", "fill")


@registry("model").register("gridInterpolation")
class GridInterpolationModel(SurrogateModelBase):
    """Multilinear interpolation on a full rectilinear grid."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("extrapolation", "extend", values=GRID_EXTRAPOLATION_MODES,
          desc="Outside the grid: raise, continue the edge cells, hold the edge values or return fillValue")
        d("fillValue", float("nan"), types=(int, float), desc="Value returned outside the grid for 'fill'")
        self.supports.update(multiOutput=True, weights=False, variances=False, derivatives=False,
                             parameterInference=False)

    def _train(self) -> None:
        axes = [np.unique(self.xt[:, j]) for j in range(self.nx)]
        if any(a.size < 2 for a in axes):
            raise ValueError("every input needs at least 2 distinct grid coordinates")
        shape = tuple(a.size for a in axes)
        if self.xt.shape[0] != int(np.prod(shape)):
            raise ValueError(f"training points do not form a full {shape} grid")
        idx = tuple(np.searchsorted(a, self.xt[:, j]) for j, a in enumerate(axes))
        values = np.full(shape + (self.ny,), np.nan)
        values[idx] = self.yt
        if np.isnan(values).any():
            raise ValueError("training points do not cover every grid node (duplicates or gaps)")
        self._axes = axes
        self._values = values

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        mode = self.options["extrapolation"]
        lo = np.array([a[0] for a in self._axes])
        hi = np.array([a[-1] for a in self._axes])
        outside = np.any((x < lo) | (x > hi), axis=1)
        if mode == "error" and outside.any():
            j = int(np.argmax(np.any((x < lo) | (x > hi), axis=0)))
            raise ValueError(f"a requested point is out of bounds in input {j}")
        q = np.clip(x, lo, hi) if mode == "clamp" else x
        cells, fracs = [], []
        for j, a in enumerate(self._axes):
            i = np.clip(np.searchsorted(a, q[:, j], side="right") - 1, 0, a.size - 2)
            cells.append(i)
            fracs.append((q[:, j] - a[i]) / (a[i + 1] - a[i]))
        out = np.zeros((x.shape[0], self.ny))
        for corner in itertools.product((0, 1), repeat=self.nx):
            weight = np.ones(x.shape[0])
            for j, c in enumerate(corner):
                weight = weight * (fracs[j] if c else 1.0 - fracs[j])
            out += weight[:, None] * self._values[tuple(cells[j] + c for j, c in enumerate(corner))]
        if mode == "fill":
            out[outside] = float(self.options["fillValue"])
        return out

    def __call__(self, xi) -> np.ndarray:
        """Evaluation at ``xi`` (..., nx); (...) for a single output, (..., ny) otherwise."""
        self._checkTrained()
        xa = np.asarray(xi, dtype=float)
        y = self._predictValues(xa.reshape(-1, self.nx))
        lead = xa.shape[:-1] if self.nx > 1 or (xa.ndim and xa.shape[-1] == 1) else xa.shape
        return y[:, 0].reshape(lead) if self.ny == 1 else y.reshape(lead + (self.ny,))

    def _effectiveParams(self):
        return float(self._values[..., 0].size)

    def _stateToDict(self) -> dict:
        return {"axes": [a.tolist() for a in self._axes], "values": self._values.ravel().tolist()}

    def _stateFromDict(self, state: dict) -> None:
        self._axes = [np.array(a, dtype=float) for a in state["axes"]]
        shape = tuple(a.size for a in self._axes) + (self.ny,)
        self._values = np.array(state["values"], dtype=float).reshape(shape)
