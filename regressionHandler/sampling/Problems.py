"""Benchmark functions for surrogate validation.

Each problem is callable on an (n, nx) array and carries its input box
``xlimits`` so ``problem.sample(n)`` gives a Latin-hypercube training set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pythonLibs.regressionHandler.sampling.Sampling import latinHypercube


@dataclass(frozen=True)
class Problem:
    name: str
    xlimits: np.ndarray
    function: Callable[[np.ndarray], np.ndarray]
    description: str = ""

    @property
    def nx(self) -> int:
        return self.xlimits.shape[0]

    def __call__(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[1] != self.nx:
            raise ValueError(f"{self.name} expects {self.nx} inputs")
        return self.function(x)

    def sample(self, n: int, seed: int = 0, criterion: str = "maximin") -> tuple[np.ndarray, np.ndarray]:
        x = latinHypercube(n, self.xlimits, criterion=criterion, seed=seed)
        return x, self(x)


def _branin(x):
    a, b, c, r, s, t = 1.0, 5.1 / (4 * np.pi ** 2), 5 / np.pi, 6.0, 10.0, 1 / (8 * np.pi)
    return a * (x[:, 1] - b * x[:, 0] ** 2 + c * x[:, 0] - r) ** 2 + s * (1 - t) * np.cos(x[:, 0]) + s


def _rosenbrock(x):
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1.0 - x[:, :-1]) ** 2, axis=1)


def _sphere(x):
    return np.sum(x * x, axis=1)


def _ackley(x):
    n = x.shape[1]
    return (-20.0 * np.exp(-0.2 * np.sqrt(np.sum(x * x, axis=1) / n))
            - np.exp(np.sum(np.cos(2 * np.pi * x), axis=1) / n) + 20.0 + np.e)


_HARTMANN_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H3_A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
_H3_P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])
_H6_A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
_H6_P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])


def _hartmann(a, p):
    def f(x):
        inner = np.einsum("ij,nij->ni", a, (x[:, None, :] - p[None, :, :]) ** 2)
        return -np.sum(_HARTMANN_ALPHA * np.exp(-inner), axis=1)
    return f


def _ishigami(x):
    return np.sin(x[:, 0]) + 7.0 * np.sin(x[:, 1]) ** 2 + 0.1 * x[:, 2] ** 4 * np.sin(x[:, 0])


def _friedman(x):
    return 10 * np.sin(np.pi * x[:, 0] * x[:, 1]) + 20 * (x[:, 2] - 0.5) ** 2 + 10 * x[:, 3] + 5 * x[:, 4]


def rosenbrock(nx: int = 2) -> Problem:
    return Problem("rosenbrock", np.tile([-2.0, 2.0], (nx, 1)), _rosenbrock, "Rosenbrock valley")


def sphere(nx: int = 2) -> Problem:
    return Problem("sphere", np.tile([-10.0, 10.0], (nx, 1)), _sphere, "Sum of squares")


def ackley(nx: int = 2) -> Problem:
    return Problem("ackley", np.tile([-5.0, 5.0], (nx, 1)), _ackley, "Ackley (multimodal)")


PROBLEMS = {
    "branin": Problem("branin", np.array([[-5.0, 10.0], [0.0, 15.0]]), _branin, "Branin-Hoo (2D)"),
    "rosenbrock": rosenbrock(2),
    "sphere": sphere(3),
    "ackley": ackley(2),
    "hartmann3": Problem("hartmann3", np.tile([0.0, 1.0], (3, 1)), _hartmann(_H3_A, _H3_P), "Hartmann (3D)"),
    "hartmann6": Problem("hartmann6", np.tile([0.0, 1.0], (6, 1)), _hartmann(_H6_A, _H6_P), "Hartmann (6D)"),
    "ishigami": Problem("ishigami", np.tile([-np.pi, np.pi], (3, 1)), _ishigami,
                        "Ishigami (3D, strongly nonlinear and non-monotonic)"),
    "friedman": Problem("friedman", np.tile([0.0, 1.0], (5, 1)), _friedman, "Friedman #1 (5D)"),
}


def getProblem(name: str) -> Problem:
    try:
        return PROBLEMS[name]
    except KeyError:
        raise KeyError(f"unknown problem {name!r}; available: {', '.join(sorted(PROBLEMS))}") from None
