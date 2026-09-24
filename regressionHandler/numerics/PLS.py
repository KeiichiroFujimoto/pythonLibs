"""Partial least squares (NIPALS) for dimension reduction in KPLS."""
from __future__ import annotations

import numpy as np


def plsRotations(x: np.ndarray, y: np.ndarray, nComponents: int) -> np.ndarray:
    """X rotations R = W (P^T W)^-1 (nx, nComponents) of PLS1/PLS2 by NIPALS.

    Inputs are expected centered/scaled. Scores are ``T = X @ R``; the
    squared entries of R measure how much each input drives each latent
    direction, which is what KPLS uses to build ARD lengthscales from only
    ``nComponents`` hyperparameters.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float).reshape(x.shape[0], -1)
    nx = x.shape[1]
    k = min(nComponents, nx, x.shape[0] - 1)
    if k < 1:
        raise ValueError("PLS needs at least one component and two samples")
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    w = np.zeros((nx, k))
    p = np.zeros((nx, k))
    for c in range(k):
        # Dominant left singular vector of X^T Y (exact NIPALS fixed point).
        cross = x.T @ y
        u, s, _ = np.linalg.svd(cross, full_matrices=False)
        wc = u[:, 0]
        if s[0] == 0.0:
            wc = np.eye(nx)[:, c % nx]
        t = x @ wc
        tt = float(t @ t)
        if tt == 0.0:
            w[:, c] = wc
            p[:, c] = wc
            continue
        pc = x.T @ t / tt
        qc = y.T @ t / tt
        x = x - np.outer(t, pc)
        y = y - np.outer(t, qc)
        w[:, c] = wc
        p[:, c] = pc
    return w @ np.linalg.pinv(p.T @ w)
