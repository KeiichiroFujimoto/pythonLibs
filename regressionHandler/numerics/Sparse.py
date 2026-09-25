"""Minimal sparse matrix (coordinate storage) for conservation constraints and mapping operators.

Only what the projection and mapping code needs, in pure numpy:

    y = A @ x, A.T @ y (``rmatvec``), products with diagonal matrices, row / column
    slicing, A diag(v) A^T as a dense (m, m) matrix for the few-rows case, and
    conversion to / from dense.

Entries with the same (row, col) are summed on construction; the entries are
kept sorted by row, then column.
"""
from __future__ import annotations

import numpy as np

DENSE_GRAM_LIMIT = 20_000_000          # m * n up to which A diag(v) A^T is formed densely


class SparseMatrix:
    """Sparse (m, n) matrix in sorted coordinate form."""

    __slots__ = ("shape", "row", "col", "data")

    def __init__(self, row, col, data, shape) -> None:
        row = np.asarray(row, dtype=np.int64).ravel()
        col = np.asarray(col, dtype=np.int64).ravel()
        data = np.asarray(data, dtype=float).ravel()
        m, n = int(shape[0]), int(shape[1])
        if not (row.size == col.size == data.size):
            raise ValueError("row, col and data must have the same length")
        if row.size and (row.min() < 0 or row.max() >= m or col.min() < 0 or col.max() >= n):
            raise ValueError("sparse index out of range")
        key = row * n + col
        if key.size and np.any(key[1:] < key[:-1]):
            order = np.argsort(key, kind="stable")
            key, data = key[order], data[order]
        if key.size:
            start = np.flatnonzero(np.concatenate([[True], key[1:] != key[:-1]]))
            uniq = key[start]
            summed = np.add.reduceat(data, start) if start.size < key.size else data
        else:
            uniq, summed = key, data
        self.shape = (m, n)
        self.row = (uniq // n).astype(np.int64) if n else uniq
        self.col = (uniq % n).astype(np.int64) if n else uniq
        self.data = summed

    # ------------------------------------------------------------------ construction
    @classmethod
    def raw(cls, row, col, data, shape) -> "SparseMatrix":
        """Fast construction for operators built row by row: entries are only ordered by row
        (stable), duplicates are kept (they add up in every product)."""
        out = cls.__new__(cls)
        row = np.asarray(row, dtype=np.int64).ravel()
        col = np.asarray(col, dtype=np.int64).ravel()
        data = np.asarray(data, dtype=float).ravel()
        if row.size and np.any(row[1:] < row[:-1]):
            order = np.argsort(row, kind="stable")
            row, col, data = row[order], col[order], data[order]
        out.shape = (int(shape[0]), int(shape[1]))
        out.row, out.col, out.data = row, col, data
        return out

    @classmethod
    def fromDense(cls, a, tol: float = 0.0) -> "SparseMatrix":
        a = np.atleast_2d(np.asarray(a, dtype=float))
        r, c = np.nonzero(np.abs(a) > tol)
        return cls(r, c, a[r, c], a.shape)

    @classmethod
    def identity(cls, n: int) -> "SparseMatrix":
        i = np.arange(n)
        return cls(i, i, np.ones(n), (n, n))

    @classmethod
    def vstack(cls, blocks: list["SparseMatrix"]) -> "SparseMatrix":
        n = blocks[0].shape[1]
        if any(b.shape[1] != n for b in blocks):
            raise ValueError("blocks need the same number of columns")
        offs = np.cumsum([0] + [b.shape[0] for b in blocks])
        return cls(np.concatenate([b.row + o for b, o in zip(blocks, offs)]),
                   np.concatenate([b.col for b in blocks]), np.concatenate([b.data for b in blocks]),
                   (int(offs[-1]), n))

    # ------------------------------------------------------------------ products
    @property
    def nnz(self) -> int:
        return int(self.data.size)

    @property
    def T(self) -> "SparseMatrix":
        return SparseMatrix(self.col, self.row, self.data, (self.shape[1], self.shape[0]))

    def __matmul__(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return np.bincount(self.row, weights=self.data * x[self.col], minlength=self.shape[0])
        out = np.zeros((self.shape[0],) + x.shape[1:])
        if self.nnz:
            # entries are sorted by row: segment sums
            start = np.flatnonzero(np.concatenate([[True], self.row[1:] != self.row[:-1]]))
            out[self.row[start]] = np.add.reduceat(self.data[:, None] * x[self.col], start, axis=0)
        return out

    def rmatvec(self, y) -> np.ndarray:
        """A^T y."""
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            return np.bincount(self.col, weights=self.data * y[self.row], minlength=self.shape[1])
        out = np.zeros((self.shape[1],) + y.shape[1:])
        if self.nnz:
            order = np.argsort(self.col, kind="stable")
            c = self.col[order]
            start = np.flatnonzero(np.concatenate([[True], c[1:] != c[:-1]]))
            out[c[start]] = np.add.reduceat((self.data[:, None] * y[self.row])[order], start, axis=0)
        return out

    def scaleColumns(self, v) -> "SparseMatrix":
        """A diag(v)."""
        v = np.asarray(v, dtype=float)
        return SparseMatrix(self.row, self.col, self.data * v[self.col], self.shape)

    def scaleRows(self, v) -> "SparseMatrix":
        """diag(v) A."""
        v = np.asarray(v, dtype=float)
        return SparseMatrix(self.row, self.col, self.data * v[self.row], self.shape)

    def gram(self, v=None) -> np.ndarray:
        """Dense A diag(v) A^T (m, m); meant for matrices with few rows."""
        m, n = self.shape
        v = np.ones(n) if v is None else np.asarray(v, dtype=float)
        if m * n <= DENSE_GRAM_LIMIT:
            a = self.toDense()
            return (a * v) @ a.T
        # pairs of entries sharing a column
        order = np.argsort(self.col, kind="stable")
        col, row, dat = self.col[order], self.row[order], self.data[order]
        counts = np.bincount(col, minlength=n)
        start = np.concatenate([[0], np.cumsum(counts)[:-1]])
        per = counts[col]
        a = np.repeat(np.arange(col.size), per)
        offset = np.arange(a.size) - np.repeat(np.cumsum(per) - per, per)
        b = start[col[a]] + offset
        out = np.zeros((m, m))
        np.add.at(out, (row[a], row[b]), dat[a] * dat[b] * v[col[a]])
        return out

    def rowSums(self) -> np.ndarray:
        return np.bincount(self.row, weights=self.data, minlength=self.shape[0])

    def columnSums(self) -> np.ndarray:
        return np.bincount(self.col, weights=self.data, minlength=self.shape[1])

    def selectColumns(self, mask) -> "SparseMatrix":
        """Zero every column outside ``mask`` (shape kept)."""
        keep = np.asarray(mask, dtype=bool)[self.col]
        return SparseMatrix(self.row[keep], self.col[keep], self.data[keep], self.shape)

    def toDense(self) -> np.ndarray:
        out = np.zeros(self.shape)
        np.add.at(out, (self.row, self.col), self.data)
        return out

    def toDict(self) -> dict:
        return {"shape": list(self.shape), "row": self.row.tolist(), "col": self.col.tolist(),
                "data": self.data.tolist()}

    @classmethod
    def fromDict(cls, d: dict) -> "SparseMatrix":
        return cls(d["row"], d["col"], d["data"], d["shape"])

    def __repr__(self) -> str:
        return f"<SparseMatrix {self.shape[0]}x{self.shape[1]}, nnz={self.nnz}>"
