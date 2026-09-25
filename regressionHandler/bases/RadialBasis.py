"""Radial basis functions with an optional low-order polynomial tail.

Kernels are selected by name and scaled by the ``epsilon`` shape
parameter. Centers are either the training
points themselves (interpolation-style, ``centers="data"``) or a smaller set
found by k-means (regression-style, ``centers="kmeans"``). Inputs are
standardized before distances are taken so every dimension counts equally.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.bases.BasisBase import BasisBase, monomialName, polynomialExponents
from pythonLibs.regressionHandler.core.Registry import registry

RBF_KERNELS = ("gaussian", "multiquadric", "inverseMultiquadric", "inverseQuadratic",
               "thinPlateSpline", "cubic", "linear", "quintic")
_SCALE_FREE = ("thinPlateSpline", "cubic", "linear", "quintic")


def rbfValue(kernel: str, r: np.ndarray, eps: float) -> np.ndarray:
    er = eps * r
    if kernel == "gaussian":
        return np.exp(-er * er)
    if kernel == "multiquadric":
        return np.sqrt(1.0 + er * er)
    if kernel == "inverseMultiquadric":
        return 1.0 / np.sqrt(1.0 + er * er)
    if kernel == "inverseQuadratic":
        return 1.0 / (1.0 + er * er)
    if kernel == "thinPlateSpline":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r > 0.0, r * r * np.log(np.where(r > 0.0, r, 1.0)), 0.0)
    if kernel == "cubic":
        return r ** 3
    if kernel == "linear":
        return r
    if kernel == "quintic":
        return r ** 5
    raise ValueError(f"unknown RBF kernel {kernel!r}; use one of {RBF_KERNELS}")


def rbfDerivativeOverR(kernel: str, r: np.ndarray, eps: float) -> np.ndarray:
    """(d phi / d r) / r, finite at r = 0 wherever the derivative is."""
    e2 = eps * eps
    q = e2 * r * r
    if kernel == "gaussian":
        return -2.0 * e2 * np.exp(-q)
    if kernel == "multiquadric":
        return e2 / np.sqrt(1.0 + q)
    if kernel == "inverseMultiquadric":
        return -e2 * (1.0 + q) ** -1.5
    if kernel == "inverseQuadratic":
        return -2.0 * e2 / (1.0 + q) ** 2
    if kernel == "thinPlateSpline":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r > 0.0, 2.0 * np.log(np.where(r > 0.0, r, 1.0)) + 1.0, 0.0)
    if kernel == "cubic":
        return 3.0 * r
    if kernel == "linear":
        with np.errstate(divide="ignore"):
            return np.where(r > 0.0, 1.0 / np.where(r > 0.0, r, 1.0), 0.0)
    if kernel == "quintic":
        return 5.0 * r ** 3
    raise ValueError(f"unknown RBF kernel {kernel!r}")


def pairwiseDistances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix via the Gram identity (BLAS-3).

    Points are centred first, and entries where the identity cancels (distance small
    against the norms) are recomputed from coordinate differences, so near and
    coincident points get exact distances even for offset, un-normalized inputs.
    """
    c = b.mean(axis=0) if b.shape[0] else np.zeros(b.shape[1])
    a, b = a - c, b - c
    aa, bb = np.sum(a * a, axis=1), np.sum(b * b, axis=1)
    d2 = aa[:, None] + bb[None, :] - 2.0 * (a @ b.T)
    i, j = np.nonzero(d2 <= 1e-4 * (aa[:, None] + bb[None, :]))
    if i.size:
        diff = a[i] - b[j]
        d2[i, j] = np.einsum("ij,ij->i", diff, diff)
    return np.sqrt(np.maximum(d2, 0.0))


def kMeans(x: np.ndarray, k: int, seed: int = 0, iterations: int = 50) -> np.ndarray:
    """k-means++ initialization followed by Lloyd iterations."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    centers = [x[rng.integers(n)]]
    d2 = np.sum((x - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        total = d2.sum()
        idx = rng.choice(n, p=d2 / total) if total > 0 else rng.integers(n)
        centers.append(x[idx])
        d2 = np.minimum(d2, np.sum((x - x[idx]) ** 2, axis=1))
    c = np.array(centers)
    for _ in range(iterations):
        labels = np.argmin(pairwiseDistances(x, c), axis=1)
        newC = c.copy()
        for j in range(k):
            members = labels == j
            if np.any(members):
                newC[j] = x[members].mean(axis=0)
        if np.allclose(newC, c):
            break
        c = newC
    return c


@registry("basis").register("radial")
class RadialBasis(BasisBase):

    def _declareOptions(self, declare) -> None:
        declare("kernel", "thinPlateSpline", values=RBF_KERNELS, desc="Radial function")
        declare("epsilon", None, types=(int, float), lower=0.0,
                desc="Shape parameter (None: 1 / mean nearest-center spacing); unused by scale-free kernels")
        declare("centers", "kmeans", values=("kmeans", "data"), desc="How centers are chosen")
        declare("nCenters", None, types=int, lower=1, desc="Number of k-means centers (None: min(n, 10 + n/5))")
        declare("polyDegree", 1, values=(-1, 0, 1), desc="Polynomial tail degree (-1: none)")
        declare("seed", 0, types=int, desc="Random seed for k-means")

    def _fit(self, x: np.ndarray) -> None:
        o = self.options
        self._mean = x.mean(axis=0)
        std = x.std(axis=0)
        self._std = np.where(std > 0, std, 1.0)
        z = (x - self._mean) / self._std
        n = z.shape[0]
        if o["centers"] == "data":
            self._centers = z.copy()
        else:
            k = o["nCenters"] or int(min(n, 10 + n // 5))
            self._centers = kMeans(z, min(k, n), seed=o["seed"])
        if o["epsilon"] is not None:
            self._eps = float(o["epsilon"])
        elif self._centers.shape[0] > 1:
            d = pairwiseDistances(self._centers, self._centers)
            np.fill_diagonal(d, np.inf)
            spacing = float(np.mean(np.min(d, axis=1)))
            self._eps = 1.0 / spacing if spacing > 0 else 1.0
        else:
            self._eps = 1.0
        deg = o["polyDegree"]
        self._tail = polynomialExponents(self.nx, deg) if deg >= 0 else np.zeros((0, self.nx), dtype=int)

    @property
    def nTerms(self) -> int:
        return self._tail.shape[0] + self._centers.shape[0]

    def _z(self, x):
        return (x - self._mean) / self._std

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = self._z(x)
        r = pairwiseDistances(z, self._centers)
        tail = np.ones((z.shape[0], self._tail.shape[0]))
        for t, e in enumerate(self._tail):
            j = np.flatnonzero(e)
            if j.size:
                tail[:, t] = z[:, j[0]]
        return np.hstack([tail, rbfValue(self.options["kernel"], r, self._eps)])

    def derivative(self, x: np.ndarray, kx: int) -> np.ndarray:
        z = self._z(x)
        r = pairwiseDistances(z, self._centers)
        g = rbfDerivativeOverR(self.options["kernel"], r, self._eps) * (z[:, [kx]] - self._centers[:, kx][None, :])
        tail = np.zeros((z.shape[0], self._tail.shape[0]))
        for t, e in enumerate(self._tail):
            if e[kx] == 1:
                tail[:, t] = 1.0
        return np.hstack([tail, g]) / self._std[kx]

    @property
    def biasMask(self) -> np.ndarray:
        mask = np.zeros(self.nTerms, dtype=bool)
        mask[: self._tail.shape[0]] = True
        return mask

    def termNames(self, featureNames: Optional[list[str]] = None) -> list[str]:
        names = [f"z({n})" for n in (featureNames or [f"x{i}" for i in range(self.nx)])]
        return [monomialName(e, names) for e in self._tail] + \
               [f"rbf{c}" for c in range(self._centers.shape[0])]

    def _stateToDict(self) -> dict:
        if self.nx is None:
            return {}
        return {"nx": self.nx, "mean": self._mean.tolist(), "std": self._std.tolist(),
                "centers": self._centers.tolist(), "eps": self._eps, "tail": self._tail.tolist()}

    def _stateFromDict(self, state: dict) -> None:
        self.nx = state["nx"]
        self._mean = np.array(state["mean"])
        self._std = np.array(state["std"])
        self._centers = np.array(state["centers"], dtype=float).reshape(-1, self.nx)
        self._eps = float(state["eps"])
        self._tail = np.array(state["tail"], dtype=int).reshape(-1, self.nx)
