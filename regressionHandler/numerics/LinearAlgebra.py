"""Dense linear algebra helpers built on numpy's LAPACK-backed primitives.

numpy provides factorizations (SVD, Cholesky, QR) but not triangular solves
or Cholesky-based solves; those are implemented here with a blocked
recursive algorithm whose work is dominated by matrix-matrix products
(BLAS-3), so they run at roughly the speed of the factorization itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_BLOCK = 64


def solveTriangular(t: np.ndarray, b: np.ndarray, lower: bool = True, trans: bool = False) -> np.ndarray:
    """Solve ``T x = b`` (or ``T^T x = b`` when ``trans``) for triangular T.

    ``b`` may be a vector or a matrix of right-hand sides.
    """
    if trans:
        t = t.T
        lower = not lower
    vector = b.ndim == 1
    rhs = b.reshape(-1, 1) if vector else b
    x = _solveTriangularBlocked(np.asarray(t, dtype=float), np.array(rhs, dtype=float), lower)
    return x.ravel() if vector else x


def _solveTriangularBlocked(t: np.ndarray, b: np.ndarray, lower: bool) -> np.ndarray:
    n = t.shape[0]
    if n <= _BLOCK:
        return np.linalg.solve(t, b)
    h = n // 2
    if lower:
        x1 = _solveTriangularBlocked(t[:h, :h], b[:h], True)
        x2 = _solveTriangularBlocked(t[h:, h:], b[h:] - t[h:, :h] @ x1, True)
    else:
        x2 = _solveTriangularBlocked(t[h:, h:], b[h:], False)
        x1 = _solveTriangularBlocked(t[:h, :h], b[:h] - t[:h, h:] @ x2, False)
    return np.vstack([x1, x2])


def choleskyWithJitter(k: np.ndarray, maxTries: int = 8) -> tuple[np.ndarray, float]:
    """Lower Cholesky factor of a symmetric PSD matrix, adding diagonal jitter if needed.

    Returns (L, jitterAdded). Raises ``np.linalg.LinAlgError`` if the matrix
    is not positive definite even after ``maxTries`` escalations.
    """
    try:
        return np.linalg.cholesky(k), 0.0
    except np.linalg.LinAlgError:
        pass
    scale = float(np.mean(np.diag(k))) if k.size else 1.0
    jitter = max(scale, 1e-300) * 1e-10
    eye = np.eye(k.shape[0])
    for _ in range(maxTries):
        try:
            return np.linalg.cholesky(k + jitter * eye), jitter
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError("matrix is not positive definite")


def choSolve(lower: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``(L L^T) x = b`` given the lower Cholesky factor L."""
    return solveTriangular(lower, solveTriangular(lower, b, lower=True), lower=True, trans=True)


def inverseFromCholesky(lower: np.ndarray) -> np.ndarray:
    """``(L L^T)^-1`` from the lower Cholesky factor."""
    lInv = solveTriangular(lower, np.eye(lower.shape[0]), lower=True)
    return lInv.T @ lInv


def logDetFromCholesky(lower: np.ndarray) -> float:
    return 2.0 * float(np.sum(np.log(np.diag(lower))))


@dataclass
class LeastSquaresSolution:
    """Result of an SVD least-squares solve of ``A x = B``.

    Attributes:
        coef:          (p, k) solution (k right-hand sides)
        rank:          numerical rank of A
        singular:      singular values of the column-equilibrated A
        colScale:      column scaling applied before the SVD
        vt, u:         SVD factors of the equilibrated matrix (rank-truncated)
    """
    coef: np.ndarray
    rank: int
    singular: np.ndarray
    colScale: np.ndarray
    u: np.ndarray
    vt: np.ndarray

    @property
    def conditionNumber(self) -> float:
        s = self.singular
        return float(s[0] / s[self.rank - 1]) if self.rank > 0 else float("inf")

    def inverseGram(self) -> np.ndarray:
        """(A^T A)^+ in the original (unscaled) coordinates."""
        r = self.rank
        vs = self.vt[:r].T / self.singular[:r]
        g = vs @ vs.T
        return g / np.outer(self.colScale, self.colScale)


def leastSquaresSvd(a: np.ndarray, b: np.ndarray, rcond: float | None = None) -> LeastSquaresSolution:
    """Minimum-norm least squares via SVD of the column-equilibrated matrix.

    Column equilibration (dividing every column by its norm) removes the
    scale disparity of raw polynomial terms (1, T, T^2 ... with T ~ 1000),
    which otherwise inflates the condition number by many orders of magnitude.
    """
    a = np.asarray(a, dtype=float)
    b2 = np.asarray(b, dtype=float)
    vector = b2.ndim == 1
    if vector:
        b2 = b2.reshape(-1, 1)
    colScale = np.linalg.norm(a, axis=0)
    colScale[colScale == 0.0] = 1.0
    aScaled = a / colScale
    if aScaled.shape[0] > 2 * aScaled.shape[1]:
        # Tall matrix: QR first, then SVD of the small triangular factor.
        q, r = np.linalg.qr(aScaled)
        ur, s, vt = np.linalg.svd(r)
        u = q @ ur
    else:
        u, s, vt = np.linalg.svd(aScaled, full_matrices=False)
    if rcond is None:
        rcond = np.finfo(float).eps * max(a.shape)
    rank = int(np.sum(s > rcond * (s[0] if s.size else 0.0)))
    coefScaled = vt[:rank].T @ ((u[:, :rank].T @ b2) / s[:rank, None])
    coef = coefScaled / colScale[:, None]
    return LeastSquaresSolution(coef=coef[:, 0] if vector else coef, rank=rank, singular=s,
                                colScale=colScale, u=u[:, :rank], vt=vt)
