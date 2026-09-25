"""Exact 1-D interpolation through the data points.

    kind="linear":  piecewise linear through (x_i, y_i)
    kind="cubic":   C2 cubic spline with not-a-knot end conditions
                    (the interpolant of scipy interp1d(kind="cubic") and
                    splrep(s=0); 2 points give a line, 3 points a parabola)

Outside [x_0, x_{n-1}] the model either continues the end piece
(``extrapolation="extend"``), holds the end value (``"clamp"``) or returns
``fillValue`` (``"fill"``). The instance is also callable, ``model(x)``,
returning an array with the shape of ``x`` (single output).

The cubic spline is stored by its knot second derivatives M_i, obtained from
the tridiagonal system that remains after eliminating M_0 and M_{n-1} with
the not-a-knot conditions (continuous third derivative at x_1 and x_{n-2}).
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right

import numpy as np

from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase

INTERPOLATION_KINDS = ("linear", "cubic")
EXTRAPOLATION_MODES = ("extend", "clamp", "fill")


def _solveTridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Thomas algorithm; lower[0] and upper[-1] are unused, rhs is (m, ny)."""
    m = diag.shape[0]
    c = np.empty(m)
    d = np.empty_like(rhs)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, m):
        denom = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / denom if i < m - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / denom
    for i in range(m - 2, -1, -1):
        d[i] -= c[i] * d[i + 1]
    return d


def notAKnotSecondDerivatives(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Knot second derivatives M (n, ny) of the not-a-knot cubic spline through (x, y (n, ny))."""
    n = x.shape[0]
    if n == 2:
        return np.zeros_like(y)
    h = np.diff(x)
    slope = np.diff(y, axis=0) / h[:, None]
    if n == 3:
        # One parabola through three points: constant second derivative.
        m = 2.0 * (slope[1] - slope[0]) / (h[0] + h[1])
        return np.tile(m, (3, 1))
    rhs = 6.0 * (slope[1:] - slope[:-1])            # rows i = 1 .. n-2
    lower = h[:-1].copy()
    diag = 2.0 * (h[:-1] + h[1:])
    upper = h[1:].copy()
    # Eliminate M_0 = ((h0 + h1) M_1 - h0 M_2) / h1 from row 1.
    h0, h1 = h[0], h[1]
    diag[0] = (h0 + h1) * (h0 + 2.0 * h1) / h1
    upper[0] = (h1 * h1 - h0 * h0) / h1
    # Eliminate M_{n-1} = ((a + b) M_{n-2} - b M_{n-3}) / a from row n-2 (a = h_{n-3}, b = h_{n-2}).
    a, b = h[-2], h[-1]
    diag[-1] = (a + b) * (2.0 * a + b) / a
    lower[-1] = (a * a - b * b) / a
    inner = _solveTridiagonal(lower, diag, upper, rhs)
    first = ((h0 + h1) * inner[0] - h0 * inner[1]) / h1
    last = ((a + b) * inner[-1] - b * inner[-2]) / a
    return np.vstack([first[None, :], inner, last[None, :]])


@registry("model").register("interpolation")
class InterpolationModel(SurrogateModelBase):
    """Exact 1-D interpolation (piecewise linear or not-a-knot cubic spline)."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("kind", "linear", values=INTERPOLATION_KINDS, desc="Interpolant between the data points")
        d("extrapolation", "extend", values=EXTRAPOLATION_MODES,
          desc="Outside the data: continue the end piece, hold the end value or return fillValue")
        d("fillValue", float("nan"), types=(int, float), desc="Value returned outside the data for 'fill'")
        self.supports.update(multiOutput=True, weights=False, variances=False, derivatives=True,
                             parameterInference=False)

    def _train(self) -> None:
        if self.nx != 1:
            raise ValueError("InterpolationModel is one-dimensional (nx must be 1)")
        order = np.argsort(self.xt[:, 0], kind="mergesort")
        self._x = self.xt[order, 0].copy()
        self._y = self.yt[order].copy()
        if self._x.shape[0] < 2:
            raise ValueError("InterpolationModel needs at least 2 points")
        self._m = None
        if self.options["kind"] == "cubic":
            if np.any(np.diff(self._x) <= 0.0):
                raise ValueError("cubic interpolation needs distinct x values")
            self._m = notAKnotSecondDerivatives(self._x, self._y)
        self._buildPieces()

    def _buildPieces(self) -> None:
        """Per-segment polynomial y_i + b t + c t^2 + d t^3 (t = x - x_i), precomputed for fast evaluation."""
        self._b = self._c = self._d = None
        if self._m is not None:
            h = np.diff(self._x)[:, None]
            mi, mj = self._m[:-1], self._m[1:]
            self._b = np.diff(self._y, axis=0) / h - h * (2.0 * mi + mj) / 6.0
            self._c = 0.5 * mi
            self._d = (mj - mi) / (6.0 * h)
        # Plain-float copies for single-point calls, where per-call numpy overhead dominates.
        self._scalar = None
        if self._y.shape[1] == 1:
            pieces = (None,) * 3 if self._m is None else (self._b[:, 0].tolist(), self._c[:, 0].tolist(),
                                                           self._d[:, 0].tolist())
            self._scalar = (self._x.tolist(), self._y[:, 0].tolist()) + pieces

    # ---------------------------------------------------------------- evaluation
    def _prepare(self, x: np.ndarray):
        """Query coordinates after clamping and the mask of points outside the data."""
        q = x[:, 0]
        outside = (q < self._x[0]) | (q > self._x[-1])
        if self.options["extrapolation"] == "clamp":
            q = np.clip(q, self._x[0], self._x[-1])
        return q, outside

    def _linear(self, q: np.ndarray, derivative: bool) -> np.ndarray:
        # Same segment choice as scipy interp1d(kind="linear") for bit-level agreement.
        hi = np.clip(np.searchsorted(self._x, q), 1, self._x.shape[0] - 1)
        lo = hi - 1
        xLo, xHi = self._x[lo], self._x[hi]
        yLo, yHi = self._y[lo], self._y[hi]
        slope = (yHi - yLo) / (xHi - xLo)[:, None]
        if derivative:
            return slope
        return slope * (q - xLo)[:, None] + yLo

    def _cubic(self, q: np.ndarray, derivative: bool) -> np.ndarray:
        i = np.searchsorted(self._x, q, side="right") - 1
        np.minimum(i, self._x.shape[0] - 2, out=i)
        np.maximum(i, 0, out=i)
        t = (q - self._x[i])[:, None]
        b, c, d = self._b[i], self._c[i], self._d[i]
        if derivative:
            return b + t * (2.0 * c + 3.0 * d * t)
        return self._y[i] + t * (b + t * (c + d * t))

    def _evaluate(self, x: np.ndarray, derivative: bool) -> np.ndarray:
        q, outside = self._prepare(x)
        out = (self._linear if self.options["kind"] == "linear" else self._cubic)(q, derivative)
        mode = self.options["extrapolation"]
        if mode == "fill":
            out[outside] = 0.0 if derivative else float(self.options["fillValue"])
        elif mode == "clamp" and derivative:
            out[outside] = 0.0
        return out

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        return self._evaluate(x, derivative=False)

    def _predictDerivatives(self, x: np.ndarray, kx: int) -> np.ndarray:
        return self._evaluate(x, derivative=True)

    def _scalarValue(self, q: float) -> float:
        """One query point of a single-output model, same operations as the array path."""
        xs, ys, bs, cs, ds = self._scalar
        if q < xs[0] or q > xs[-1]:
            mode = self.options["extrapolation"]
            if mode == "fill":
                return float(self.options["fillValue"])
            if mode == "clamp":
                q = xs[0] if q < xs[0] else xs[-1]
        last = len(xs) - 1
        if bs is None:
            hi = min(max(bisect_left(xs, q), 1), last)
            lo = hi - 1
            return (ys[hi] - ys[lo]) / (xs[hi] - xs[lo]) * (q - xs[lo]) + ys[lo]
        i = min(max(bisect_right(xs, q) - 1, 0), last - 1)
        t = q - xs[i]
        return ys[i] + t * (bs[i] + t * (cs[i] + ds[i] * t))

    def __call__(self, x) -> np.ndarray:
        """Single-output evaluation with the shape of ``x`` (scalar in, 0-d array out)."""
        self._checkTrained()
        xa = np.asarray(x, dtype=float)
        if xa.size == 1 and self._scalar is not None:
            return np.asarray(self._scalarValue(float(xa.flat[0]))).reshape(xa.shape)
        y = self._evaluate(xa.reshape(-1, 1), derivative=False)
        return (y[:, 0] if self.ny == 1 else y).reshape(xa.shape + (() if self.ny == 1 else (self.ny,)))

    # ---------------------------------------------------------------- bookkeeping
    def _effectiveParams(self):
        return float(self._x.shape[0])

    def _stateToDict(self) -> dict:
        return {"x": self._x.tolist(), "y": self._y.tolist(),
                "m": None if self._m is None else self._m.tolist()}

    def _stateFromDict(self, state: dict) -> None:
        self._x = np.array(state["x"], dtype=float)
        self._y = np.array(state["y"], dtype=float).reshape(self._x.shape[0], -1)
        self._m = None if state["m"] is None else np.array(state["m"], dtype=float).reshape(self._y.shape)
        self._buildPieces()
