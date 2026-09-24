"""Penalized B-spline smoother (P-spline) for 1 to 3 inputs.

A convenience front end over ``LinearBasisModel`` with a uniform B-spline
basis and a difference penalty whose weight is chosen by GCV (default) or
exact leave-one-out: a smoothing spline in 1D and a regularized
tensor-product spline in 2D/3D.
"""
from __future__ import annotations

from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.models.LinearBasisModel import LinearBasisModel


@registry("model").register("spline")
class SplineModel(LinearBasisModel):
    """Penalized (P-)spline smoother with smoothing chosen by GCV / LOO (1-3 inputs)."""

    def _initialize(self) -> None:
        super()._initialize()
        d = self.options.declare
        d("nSegments", 20, types=(int, list), desc="Knot intervals per input")
        d("degree", 3, types=int, lower=0, upper=5, desc="Spline degree")
        d("penaltyOrder", 2, types=int, lower=0, upper=4, desc="Difference order of the roughness penalty")
        d("alpha", "gcv", values=("gcv", "loo"), types=(int, float), desc="Smoothing weight or selection rule")

    def _validateOptions(self) -> None:
        o = self.options
        o["basis"] = {"type": "bspline", "nSegments": o["nSegments"], "degree": o["degree"],
                      "penaltyOrder": o["penaltyOrder"]}
        o["solver"] = {"type": "ridge", "penalty": "smoothness", "alpha": o["alpha"]}
        super()._validateOptions()
