"""FITPACK smoothing spline (Dierckx curfit): fewest knots with sum (w (y - s(x)))^2 <= s.

1-D spline of degree k (1..5) whose knots are added where the residuals are
largest until the weighted residual sum of squares fp can reach the
smoothing factor s, then fp = s is met (to 0.1 %) by a smoothing parameter.
s = 0 interpolates; a large s returns the least-squares polynomial.
This is the spline of scipy ``splrep(x, y, s=s)`` and ``UnivariateSpline``
(port in ``numerics/Fitpack.py``); ``setSmoothingFactor`` continues from the
knots found so far like ``UnivariateSpline.set_smoothing_factor``.

Options:
    degree       spline degree k
    smoothing    s; None = sum of the sample weights (UnivariateSpline default, = n for unit weights)
    nest         knot storage; None = FITPACK / UnivariateSpline choice (m + k + 1 for s = 0,
                 else max(m // 2, 2 k + 2), enlarged to m + k + 1 when exhausted)
    extrapolation  extend (continue the end polynomials), clamp, fill (fillValue), error

Sample weights ``weights`` enter as FITPACK weights w_i = sqrt(weight_i).
The instance is callable, ``model(x)``, returning the shape of ``x``.
"""
from __future__ import annotations

import warnings

import numpy as np

from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase
from pythonLibs.regressionHandler.numerics.Fitpack import (CURFIT_MESSAGES, CurfitState, fpcurf, splder,
                                                            splev)

_EXT = {"extend": 0, "fill": 1, "error": 2, "clamp": 3}


@registry("model").register("smoothingSpline")
class SmoothingSplineModel(SurrogateModelBase):
    """FITPACK (Dierckx) smoothing spline with automatic knots, 1 input."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("degree", 3, types=int, lower=1, upper=5, desc="Spline degree k")
        d("smoothing", None, types=(int, float), lower=0.0,
          desc="Smoothing factor s (None: sum of the sample weights)")
        d("nest", None, types=int, lower=4, desc="Knot storage (None: FITPACK choice, enlarged when exhausted)")
        d("extrapolation", "extend", values=tuple(_EXT), desc="Outside the data range")
        d("fillValue", 0.0, types=(int, float), desc="Value returned outside the data for 'fill'")
        self.supports.update(multiOutput=False, weights=True, variances=False, derivatives=True,
                             parameterInference=False)

    # ---------------------------------------------------------------- training
    def _train(self) -> None:
        if self.nx != 1:
            raise ValueError("SmoothingSplineModel is one-dimensional (nx must be 1)")
        order = np.argsort(self.xt[:, 0], kind="mergesort")
        self._xData = self.xt[order, 0].copy()
        self._yData = self.yt[order, 0].copy()
        self._wData = np.sqrt(self.wt[order])
        k = self.options["degree"]
        m = self._xData.size
        if m <= k:
            raise ValueError(f"degree {k} needs more than {k} points")
        s = self.options["smoothing"]
        s = float(np.sum(self.wt)) if s is None else float(s)
        self._autoNest = self.options["nest"] is None
        nest = (m + k + 1 if s == 0.0 else max(m // 2, 2 * (k + 1))) if self._autoNest else self.options["nest"]
        if nest < 2 * (k + 1):
            raise ValueError(f"nest must be at least {2 * (k + 1)}")
        self._state = CurfitState.empty(nest)
        self._run(0, s)

    def _run(self, iopt: int, s: float) -> None:
        k = self.options["degree"]
        x, y, w = self._xData, self._yData, self._wData
        st = fpcurf(iopt, x, y, w, x[0], x[-1], k, s, self._state)
        if st.ier == 1 and self._autoNest and st.nest < x.size + k + 1:
            # UnivariateSpline: nest too small -> continue with the maximum bound.
            self._state = st.resized(x.size + k + 1)
            st = fpcurf(1, x, y, w, x[0], x[-1], k, s, self._state)
        self._state = st
        self._s = s
        if st.ier > 0:
            warnings.warn(CURFIT_MESSAGES.get(st.ier, f"ier={st.ier}"), RuntimeWarning, stacklevel=3)
        self._setPieces()

    def _setPieces(self) -> None:
        k = self.options["degree"]
        self._t, self._c = self._state.tck(k)
        self._tD, self._cD, self._kD = splder(self._t, self._c, k)

    def setSmoothingFactor(self, s: float) -> "SurrogateModelBase":
        """Refit with a new smoothing factor, continuing from the current knots (iopt = 1)."""
        self._checkTrained()
        if s < 0.0:
            raise ValueError("smoothing factor must be non-negative")
        self.options["smoothing"] = float(s)
        self._run(1, float(s))
        return self

    # ---------------------------------------------------------------- results
    @property
    def knots(self) -> np.ndarray:
        return self._t.copy()

    @property
    def coefficients(self) -> np.ndarray:
        return self._c[:self._t.size - self.options["degree"] - 1].copy()

    @property
    def residual(self) -> float:
        """Weighted residual sum of squares fp."""
        return self._state.fp

    @property
    def ier(self) -> int:
        """FITPACK status: 0 fp = s, -1 interpolating, -2 least-squares polynomial, > 0 see warnings."""
        return self._state.ier

    # ---------------------------------------------------------------- evaluation
    def _evaluate(self, q: np.ndarray, derivative: bool) -> np.ndarray:
        mode = self.options["extrapolation"]
        if derivative:
            out = splev(self._tD, self._cD, self._kD, q, _EXT[mode])
            if mode == "clamp":
                out[(q < self._t[0]) | (q > self._t[-1])] = 0.0
            return out
        out = splev(self._t, self._c, self.options["degree"], q, _EXT[mode])
        if mode == "fill" and self.options["fillValue"] != 0.0:
            out[(q < self._t[0]) | (q > self._t[-1])] = float(self.options["fillValue"])
        return out

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        return self._evaluate(x[:, 0], False)[:, None]

    def _predictDerivatives(self, x: np.ndarray, kx: int) -> np.ndarray:
        return self._evaluate(x[:, 0], True)[:, None]

    def __call__(self, x) -> np.ndarray:
        """Evaluation with the shape of ``x`` (scalar in, 0-d array out)."""
        self._checkTrained()
        xa = np.asarray(x, dtype=float)
        return self._evaluate(xa.ravel(), False).reshape(xa.shape)

    # ---------------------------------------------------------------- bookkeeping
    def _effectiveParams(self):
        return float(self._t.size - self.options["degree"] - 1)

    def _stateToDict(self) -> dict:
        st = self._state
        return {"x": self._xData.tolist(), "y": self._yData.tolist(), "w": self._wData.tolist(), "s": self._s,
                "autoNest": self._autoNest, "nest": st.nest, "n": st.n, "fp": st.fp, "ier": st.ier,
                "t": st.t, "c": st.c, "fpint": st.fpint, "nrdata": st.nrdata}

    def _stateFromDict(self, state: dict) -> None:
        self._xData, self._yData = np.array(state["x"], dtype=float), np.array(state["y"], dtype=float)
        self._wData, self._s, self._autoNest = np.array(state["w"], dtype=float), state["s"], state["autoNest"]
        self._state = CurfitState(nest=state["nest"], n=state["n"], fp=state["fp"], ier=state["ier"],
                                  t=[float(v) for v in state["t"]], c=[float(v) for v in state["c"]],
                                  fpint=[float(v) for v in state["fpint"]], nrdata=[int(v) for v in state["nrdata"]])
        self._setPieces()
