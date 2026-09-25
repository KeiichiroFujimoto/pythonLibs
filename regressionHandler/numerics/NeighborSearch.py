"""Exact k-nearest-neighbour search.

For up to four dimensions points are bucketed into a uniform grid whose
cells hold about ``k`` points; queries sharing a cell are answered together
against the surrounding block of cells, and the block grows ring by ring
until the k-th distance is provably inside it. Higher dimensions use chunked
brute force (Gram-identity distances), which is already BLAS bound.
"""
from __future__ import annotations

import itertools

import numpy as np


def _bruteForce(points: np.ndarray, queries: np.ndarray, k: int, chunk: int = 1024):
    dist = np.empty((queries.shape[0], k))
    idx = np.empty((queries.shape[0], k), dtype=np.int64)
    # the Gram identity cancels badly for points far from the origin: centre first, and take the
    # returned k distances from coordinate differences (exact, e.g. 0 for a query at a data point)
    centre = points.mean(axis=0)
    points = points - centre
    queries = queries - centre
    p2 = np.sum(points * points, axis=1)
    chunk = max(1, min(chunk, 4_000_000 // max(k * points.shape[1], 1)))
    for s in range(0, queries.shape[0], chunk):
        q = queries[s:s + chunk]
        d2 = np.maximum(np.sum(q * q, axis=1)[:, None] + p2[None, :] - 2.0 * (q @ points.T), 0.0)
        part = np.argpartition(d2, k - 1, axis=1)[:, :k] if k < points.shape[0] else \
            np.tile(np.arange(points.shape[0]), (q.shape[0], 1))
        diff = points[part] - q[:, None, :]
        dd = np.einsum("mkd,mkd->mk", diff, diff)
        order = np.argsort(dd, axis=1, kind="stable")
        idx[s:s + chunk] = np.take_along_axis(part, order, axis=1)
        dist[s:s + chunk] = np.sqrt(np.take_along_axis(dd, order, axis=1))
    return dist, idx


class NeighborSearch:
    """Build once on ``points`` (n, d), then ``query(queries, k)`` -> (distances, indices) sorted."""

    def __init__(self, points: np.ndarray, gridMaxDim: int = 4, targetPerCell: int = 16) -> None:
        self.points = np.asarray(points, dtype=float)
        n, d = self.points.shape
        self._useGrid = d <= gridMaxDim and n > 256
        if not self._useGrid:
            return
        self._lo = self.points.min(axis=0)
        span = np.maximum(self.points.max(axis=0) - self._lo, 1e-12)
        volume = float(np.prod(span))
        h = (volume * targetPerCell / n) ** (1.0 / d)
        self._h = max(h, float(span.max()) / 1e6)
        self._shape = np.maximum(np.ceil(span / self._h).astype(np.int64), 1)
        cells = self._cellOf(self.points)
        keys = np.ravel_multi_index(cells.T, self._shape)
        order = np.argsort(keys, kind="stable")
        self._order = order
        self._sortedKeys = keys[order]

    def _cellOf(self, x: np.ndarray) -> np.ndarray:
        c = np.floor((x - self._lo) / self._h).astype(np.int64)
        return np.clip(c, 0, self._shape - 1)

    def _pointsInCells(self, cellList: np.ndarray) -> np.ndarray:
        keys = np.ravel_multi_index(cellList.T, self._shape)
        starts = np.searchsorted(self._sortedKeys, keys, side="left")
        ends = np.searchsorted(self._sortedKeys, keys, side="right")
        if not np.any(ends > starts):
            return np.zeros(0, dtype=np.int64)
        return np.concatenate([self._order[s:e] for s, e in zip(starts, ends) if e > s])

    def query(self, queries: np.ndarray, k: int):
        queries = np.asarray(queries, dtype=float)
        n, d = self.points.shape
        if not 1 <= k <= n:
            raise ValueError(f"k must be in [1, {n}]")
        if not self._useGrid or queries.shape[0] * n <= 2e7:
            return _bruteForce(self.points, queries, k)
        dist = np.empty((queries.shape[0], k))
        idx = np.empty((queries.shape[0], k), dtype=np.int64)
        qCells = self._cellOf(queries)
        qKeys = np.ravel_multi_index(qCells.T, self._shape)
        order = np.argsort(qKeys, kind="stable")
        uniq, starts = np.unique(qKeys[order], return_index=True)
        bounds = list(starts) + [order.size]
        maxRing = int(self._shape.max())
        for g in range(uniq.size):
            members = order[bounds[g]:bounds[g + 1]]
            cell = qCells[members[0]]
            pending = members
            ring = 1
            while pending.size:
                offsets = np.array(list(itertools.product(range(-ring, ring + 1), repeat=d)), dtype=np.int64)
                block = cell + offsets
                valid = np.all((block >= 0) & (block < self._shape), axis=1)
                cand = self._pointsInCells(block[valid])
                covered = ring >= maxRing
                if cand.size >= k:
                    q = queries[pending]
                    pts = self.points[cand]
                    diff = q[:, None, :] - pts[None, :, :]
                    d2 = np.einsum("mcd,mcd->mc", diff, diff)
                    part = np.argpartition(d2, k - 1, axis=1)[:, :k] if k < cand.size else \
                        np.tile(np.arange(cand.size), (q.shape[0], 1))
                    dd = np.take_along_axis(d2, part, axis=1)
                    srt = np.argsort(dd, axis=1)
                    part = np.take_along_axis(part, srt, axis=1)
                    dd = np.sqrt(np.take_along_axis(dd, srt, axis=1))
                    lowEdge = self._lo + (cell - ring) * self._h
                    highEdge = self._lo + (cell + ring + 1) * self._h
                    safe = np.min(np.minimum(q - lowEdge, highEdge - q), axis=1)
                    ok = covered | (dd[:, -1] <= safe)
                    dist[pending[ok]] = dd[ok]
                    idx[pending[ok]] = cand[part[ok]]
                    pending = pending[~ok]
                ring += 1
        return dist, idx
