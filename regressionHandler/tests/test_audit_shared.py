"""Shared-code fixes from the core audit: distances, undefined inference, option persistence."""
import json

import numpy as np
import pytest

from pythonLibs.regressionHandler import LinearBasisModel, RbfModel
from pythonLibs.regressionHandler.bases.RadialBasis import pairwiseDistances
from pythonLibs.regressionHandler.core.Registry import _optionsToDict


def test_pairwiseDistancesExactForNearPointsOfOffsetData():
    rng = np.random.default_rng(0)
    x = 1000.0 + rng.uniform(0.0, 1.0, (40, 2))
    d = pairwiseDistances(x, x)
    assert np.all(np.diag(d) == 0.0)
    near = x + np.array([1e-6, 0.0])
    assert np.diag(pairwiseDistances(near, x)) == pytest.approx(np.full(40, 1e-6), rel=1e-6)
    direct = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    assert d == pytest.approx(direct, rel=1e-12, abs=1e-12)


def test_linearRbfDerivativeNearNodesOfOffsetData():
    rng = np.random.default_rng(0)
    x = 1000.0 + rng.uniform(0.0, 1.0, (60, 2))
    m = RbfModel(kernel="linear", normalize="none").fit(x, np.sin(3.0 * x[:, 0]) + x[:, 1])
    q = x[:20] + np.array([1e-4, 0.0])
    h = 1e-7
    fd = (m.predictValues(q + [h, 0.0])[:, 0] - m.predictValues(q - [h, 0.0])[:, 0]) / (2.0 * h)
    assert np.max(np.abs(m.predictDerivatives(q, 0)[:, 0] - fd)) < 1e-4     # was ~13 with the plain Gram identity


def test_undefinedStandardErrorsGiveUndefinedPValues():
    x = np.array([0.0, 1.0, 2.0])
    m = LinearBasisModel(basis={"type": "polynomial", "degree": 2}, solver="ols").fit(x, 1.0 + x ** 2)
    assert np.all(np.isnan(m.result.tValues)) and np.all(np.isnan(m.result.pValues))


def test_optionsWithNumpyValuesAreJsonReady():
    plain = _optionsToDict({"a": np.arange(3), "b": [np.float64(1.5), {"c": np.ones(2)}], "d": (np.int64(2),)})
    assert json.loads(json.dumps(plain)) == {"a": [0, 1, 2], "b": [1.5, {"c": [1.0, 1.0]}], "d": [2]}
