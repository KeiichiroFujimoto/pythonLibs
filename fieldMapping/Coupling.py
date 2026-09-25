"""Partitioned (staggered) coupling iteration: fixed-point relaxation with Aitken acceleration.

The interface unknown x (e.g. a wall temperature history carried between two solvers with
``SurfaceFluxMapper.map`` / ``mapBack``) is updated from the solvers' answer x~ = F(x):

    r_k = x~_k - x_k,        x_{k+1} = x_k + w_k r_k

with a constant ``omega`` or Aitken's dynamic factor (Irons-Tuck / Kuettler-Wall)

    w_k = -w_{k-1} r_{k-1} . (r_k - r_{k-1}) / |r_k - r_{k-1}|^2,

clipped to [omegaMin, omegaMax]. Only finite entries take part (NaN = not on the interface).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FixedPointRelaxation:
    omega: float = 0.5                    # first (or constant) relaxation factor
    aitken: bool = True
    omegaMin: float = 0.05
    omegaMax: float = 1.0
    tolerance: float = 1.0                # converged when max |r| <= tolerance (units of x)
    history: list = field(default_factory=list)
    _prevResidual: np.ndarray | None = None
    _omega: float | None = None

    def update(self, x: np.ndarray, xTilde: np.ndarray) -> tuple[np.ndarray, bool]:
        """One relaxation step; returns (x_next, converged). Entries where x~ is NaN keep x."""
        x = np.asarray(x, float)
        xTilde = np.asarray(xTilde, float)
        live = np.isfinite(xTilde) & np.isfinite(x)
        r = np.where(live, xTilde - x, 0.0)
        norm = float(np.max(np.abs(r))) if live.any() else 0.0
        rms = float(np.sqrt(np.mean(r[live] ** 2))) if live.any() else 0.0
        w = self.omega if self._omega is None else self._omega
        if self.aitken and self._prevResidual is not None:
            dr = r - self._prevResidual
            den = float(np.sum(dr * dr))
            if den > 0.0:
                w = float(np.clip(-w * np.sum(self._prevResidual * dr) / den, self.omegaMin, self.omegaMax))
        converged = norm <= self.tolerance
        self.history.append({"iteration": len(self.history), "maxResidual": norm, "rmsResidual": rms,
                             "omega": w, "liveEntries": int(live.sum())})
        self._prevResidual, self._omega = r, w
        return (x if converged else x + w * r), converged
