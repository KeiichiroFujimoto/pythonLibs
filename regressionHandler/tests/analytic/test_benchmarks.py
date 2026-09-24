"""Benchmark functions at their known optima and sampling designs by construction."""
import numpy as np
import pytest

from pythonLibs.regressionHandler.sampling import PROBLEMS, getProblem, latinHypercube
from pythonLibs.regressionHandler.sampling.Sampling import _radicalInverse


@pytest.mark.parametrize("name,point,value", [
    ("branin", [-np.pi, 12.275], 0.39788735772973816),
    ("branin", [np.pi, 2.275], 0.39788735772973816),
    ("branin", [9.42478, 2.475], 0.39788735772973816),
    ("rosenbrock", [1.0, 1.0], 0.0),
    ("sphere", [0.0, 0.0, 0.0], 0.0),
    ("ackley", [0.0, 0.0], 0.0),
    ("hartmann3", [0.114614, 0.555649, 0.852547], -3.86278),
    ("hartmann6", [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573], -3.32237),
    ("ishigami", [0.0, 0.0, 1.0], 0.0),
    ("friedman", [0.5, 0.0, 0.5, 0.0, 0.0], 0.0),
])
def test_knownOptima(name, point, value):
    assert getProblem(name)(np.array([point]))[0] == pytest.approx(value, abs=1e-5)


def test_ishigamiClosedForm():
    x = np.array([[np.pi / 2, np.pi / 2, 1.0]])
    assert getProblem("ishigami")(x)[0] == pytest.approx(1 + 7 + 0.1)


def test_everyProblemIsFiniteOnItsDomain():
    for prob in PROBLEMS.values():
        x, y = prob.sample(30)
        assert np.all((x >= prob.xlimits[:, 0]) & (x <= prob.xlimits[:, 1])) and np.all(np.isfinite(y))


def test_radicalInverseDigits():
    np.testing.assert_allclose(_radicalInverse(np.arange(1, 8), 2), [1 / 2, 1 / 4, 3 / 4, 1 / 8, 5 / 8, 3 / 8, 7 / 8])
    np.testing.assert_allclose(_radicalInverse(np.arange(1, 5), 3), [1 / 3, 2 / 3, 1 / 9, 4 / 9])


def test_latinHypercubeStratification():
    lim = np.array([[0.0, 1.0], [10.0, 20.0], [-1.0, 1.0]])
    for crit in ("random", "center", "maximin", "ese"):
        s = latinHypercube(16, lim, criterion=crit, seed=0, iterations=5)
        strata = np.floor((s - lim[:, 0]) / (lim[:, 1] - lim[:, 0]) * 16).astype(int)
        for k in range(3):
            assert sorted(strata[:, k]) == list(range(16))        # exactly one point per stratum
