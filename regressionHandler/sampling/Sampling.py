"""Design-of-experiments sampling.

All samplers return points in the box given by ``xlimits`` (nx, 2); the
unit-hypercube versions are used internally for optimizer multi-starts.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _asLimits(xlimits) -> np.ndarray:
    lim = np.asarray(xlimits, dtype=float)
    if lim.ndim != 2 or lim.shape[1] != 2 or np.any(lim[:, 0] >= lim[:, 1]):
        raise ValueError("xlimits must be (nx, 2) with lower < upper")
    return lim


def _scale(unit: np.ndarray, lim: np.ndarray) -> np.ndarray:
    return lim[:, 0] + unit * (lim[:, 1] - lim[:, 0])


def _minDistance(points: np.ndarray) -> float:
    g = points @ points.T
    sq = np.diag(g)
    d2 = sq[:, None] + sq[None, :] - 2.0 * g
    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(max(d2.min(), 0.0)))


def latinHypercube(nSamples: int, xlimits, criterion: str = "maximin", seed: Optional[int] = 0,
                   iterations: int = 50) -> np.ndarray:
    """Latin hypercube sample.

    criterion:
        "random"  - one random point per stratum
        "center"  - stratum centers, random permutation
        "maximin" - best of ``iterations`` random LHS designs by minimum pairwise distance
        "ese"     - enhanced stochastic evolutionary improvement of the maximin distance
                    by pairwise swaps within a column (Jin, Chen & Sudjianto 2005)
    """
    lim = _asLimits(xlimits)
    nx = lim.shape[0]
    rng = np.random.default_rng(seed)

    def oneDesign(center: bool) -> np.ndarray:
        u = 0.5 * np.ones((nSamples, nx)) if center else rng.random((nSamples, nx))
        strata = np.column_stack([rng.permutation(nSamples) for _ in range(nx)])
        return (strata + u) / nSamples

    if criterion == "random":
        unit = oneDesign(False)
    elif criterion == "center":
        unit = oneDesign(True)
    elif criterion == "maximin":
        best, bestD = None, -1.0
        for _ in range(max(1, iterations)):
            cand = oneDesign(False)
            d = _minDistance(cand) if nSamples > 1 else 0.0
            if d > bestD:
                best, bestD = cand, d
        unit = best
    elif criterion == "ese":
        unit = oneDesign(False)
        bestD = _minDistance(unit) if nSamples > 1 else 0.0
        best = unit
        threshold = 0.005 * bestD
        for _ in range(max(1, iterations) * nx * 10):
            if nSamples < 2:
                break
            col = rng.integers(nx)
            i, j = rng.choice(nSamples, 2, replace=False)
            cand = unit.copy()
            cand[[i, j], col] = cand[[j, i], col]
            d = _minDistance(cand)
            if d > bestD - threshold * rng.random():
                unit = cand
                if d > bestD:
                    # the walk may accept slightly worse designs; the best one visited is returned
                    best, bestD = cand, d
        unit = best
    else:
        raise ValueError("criterion must be random, center, maximin or ese")
    return _scale(unit, lim)


def fullFactorial(levels, xlimits) -> np.ndarray:
    """Tensor grid with ``levels`` points per dimension (int or list)."""
    lim = _asLimits(xlimits)
    nx = lim.shape[0]
    lv = [int(levels)] * nx if np.ndim(levels) == 0 else [int(v) for v in levels]
    axes = [np.linspace(lim[k, 0], lim[k, 1], lv[k]) if lv[k] > 1 else np.array([lim[k].mean()])
            for k in range(nx)]
    grids = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([g.ravel() for g in grids])


def randomSampling(nSamples: int, xlimits, seed: Optional[int] = 0) -> np.ndarray:
    lim = _asLimits(xlimits)
    return _scale(np.random.default_rng(seed).random((nSamples, lim.shape[0])), lim)


def sobolLike(nSamples: int, xlimits, seed: Optional[int] = 0) -> np.ndarray:
    """Randomly shifted Halton sequence (low-discrepancy, any dimension up to 50)."""
    lim = _asLimits(xlimits)
    nx = lim.shape[0]
    primes = _firstPrimes(nx)
    idx = np.arange(1, nSamples + 1)
    unit = np.column_stack([_radicalInverse(idx, b) for b in primes])
    shift = np.random.default_rng(seed).random(nx)
    return _scale((unit + shift) % 1.0, lim)


def _radicalInverse(idx: np.ndarray, base: int) -> np.ndarray:
    out = np.zeros(idx.size)
    f = 1.0 / base
    i = idx.copy()
    while np.any(i > 0):
        out += f * (i % base)
        i //= base
        f /= base
    return out


def _firstPrimes(k: int) -> list[int]:
    primes = []
    c = 2
    while len(primes) < k:
        if all(c % p for p in primes if p * p <= c):
            primes.append(c)
        c += 1
    return primes
