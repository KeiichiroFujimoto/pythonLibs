"""regressionHandler - regression and surrogate modelling, 1D to N-D.

Numerics (distributions, optimizers, splines, kernels) are implemented in
this package on top of numpy only. ``RegressionHandler`` exposes the toolkit
as a toolBaseSecured service; the model classes can also be used directly::

    from pythonLibs.regressionHandler import createModel
    model = createModel("quadratic").fit(x, y)
    band = model.predictInterval(xNew, level=0.95)
"""
from pythonLibs.regressionHandler.core import (FitMetrics, FitResult, OptionsDictionary, PredictionInterval,
                                               SurrogateModelBase, buildComponent, registry)
from pythonLibs.regressionHandler.evaluation import (ModelSelector, bootstrap, crossValidate, defaultCandidates,
                                                    diagnose, modelReport, regressionMetrics, stepwiseSelect,
                                                    tuneHyperparameters)
from pythonLibs.regressionHandler.sampling import PROBLEMS, getProblem, latinHypercube
from pythonLibs.regressionHandler.models import (CokrigingModel, ConstrainedModel, DimensionlessModel, GamModel, GlmModel, GridInterpolationModel, InterpolationModel, QuantileModel, MultiFidelityKrigingModel, GradientKrigingModel, OdrModel, HeteroscedasticModel, TransformedTargetModel, RandomForestModel, GradientBoostingModel, NeuralNetworkModel, MixedModel, IsotonicModel, ShapeSplineModel, BayesianLinearModel, SmoothingSplineModel, SupportVectorModel, IdwModel, KplsModel, KrigingModel, LinearBasisModel,
                                                LocalRegressionModel, NonlinearModel, RbfModel, ScalableKrigingModel,
                                                SplineModel,
                                                availableModels, createModel, registerForm)
from pythonLibs.regressionHandler.RegressionHandler import RegressionDataset, RegressionHandler

loadModel = SurrogateModelBase.load

__all__ = ["RegressionHandler", "RegressionDataset", "SurrogateModelBase", "ConstrainedModel", "DimensionlessModel", "LinearBasisModel", "KrigingModel", "KplsModel", "CokrigingModel", "ScalableKrigingModel", "GlmModel", "GamModel", "QuantileModel", "MultiFidelityKrigingModel", "GradientKrigingModel", "OdrModel", "HeteroscedasticModel", "TransformedTargetModel", "RandomForestModel", "GradientBoostingModel", "NeuralNetworkModel", "MixedModel", "IsotonicModel", "ShapeSplineModel", "BayesianLinearModel", "RbfModel", "IdwModel", "InterpolationModel", "GridInterpolationModel", "SmoothingSplineModel", "SupportVectorModel", "NonlinearModel", "SplineModel",
           "LocalRegressionModel", "registerForm", "createModel",
           "availableModels", "loadModel", "crossValidate", "regressionMetrics", "ModelSelector",
           "defaultCandidates", "stepwiseSelect", "tuneHyperparameters", "diagnose", "bootstrap", "modelReport",
           "PROBLEMS", "getProblem", "latinHypercube", "FitMetrics", "FitResult",
           "OptionsDictionary", "PredictionInterval", "buildComponent", "registry"]
