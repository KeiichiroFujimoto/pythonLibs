"""Buckingham pi theorem: dimensionless groups from a dimension matrix (exact rational arithmetic).

The dimension matrix D (nDimensions x nVariables) holds the exponents of the
base dimensions (e.g. M, L, T, Theta) of each variable. Every null-space
vector a of D (D a = 0) defines a dimensionless product prod_i x_i^a_i; the
reduced row echelon form gives one group per non-pivot ("non-repeating")
variable, with integer exponents. An output with dimension vector e is made
dimensionless by prod_i x_i^b_i with D b = e, using the repeating (pivot)
variables only.
"""
from __future__ import annotations

from fractions import Fraction
from math import lcm
from typing import Optional, Sequence

import numpy as np

BASE_DIMENSIONS = ("M", "L", "T", "Theta", "N", "I", "J")


def _rref(rows: list[list[Fraction]]) -> tuple[list[list[Fraction]], list[int]]:
    a = [r[:] for r in rows]
    m, n = len(a), len(a[0]) if a else 0
    pivots, r = [], 0
    for c in range(n):
        piv = next((i for i in range(r, m) if a[i][c] != 0), None)
        if piv is None:
            continue
        a[r], a[piv] = a[piv], a[r]
        p = a[r][c]
        a[r] = [v / p for v in a[r]]
        for i in range(m):
            if i != r and a[i][c] != 0:
                f = a[i][c]
                a[i] = [vi - f * vr for vi, vr in zip(a[i], a[r])]
        pivots.append(c)
        r += 1
        if r == m:
            break
    return a[:r], pivots


def dimensionMatrix(dimensions: Sequence, names: Optional[Sequence[str]] = None) -> np.ndarray:
    """(nDim, nVar) exponent matrix from per-variable specs: lists of exponents or dicts {"L": 1, "T": -1}."""
    cols = []
    for d in dimensions:
        if isinstance(d, dict):
            unknown = set(d) - set(BASE_DIMENSIONS)
            if unknown:
                raise ValueError(f"unknown base dimensions {sorted(unknown)}; use {BASE_DIMENSIONS}")
            cols.append([d.get(k, 0) for k in BASE_DIMENSIONS])
        else:
            v = list(d)
            cols.append(v + [0] * (len(BASE_DIMENSIONS) - len(v)))
    mat = np.array(cols, dtype=float).T
    used = np.any(mat != 0, axis=1)
    return mat[used] if used.any() else mat[:1] * 0


def buckinghamPi(dmat, repeating: Optional[Sequence[int]] = None) -> dict:
    """Dimensionless groups of the variables with dimension matrix ``dmat``.

    Args:
        dmat:      (nDim, nVar) exponents (numbers convertible to exact fractions)
        repeating: optional variable indices to use as repeating variables (they
                   are moved first so they become the pivots)
    Returns {"groups": (nGroups, nVar) integer exponents, "repeating": [...], "rank": r}.
    """
    d = np.atleast_2d(np.asarray(dmat, dtype=float))
    nVar = d.shape[1]
    order = list(range(nVar))
    if repeating is not None:
        rep = [int(i) for i in repeating]
        order = rep + [i for i in order if i not in rep]
    rows = [[Fraction(v).limit_denominator(1000) for v in d[i, order]] for i in range(d.shape[0])]
    red, piv = _rref(rows)
    if repeating is not None and len(piv) < len(repeating):
        raise ValueError("the repeating variables are not dimensionally independent")
    free = [c for c in range(nVar) if c not in piv]
    groups = []
    for f in free:
        vec = [Fraction(0)] * nVar
        vec[f] = Fraction(1)
        for r, p in enumerate(piv):
            vec[p] = -red[r][f]
        den = lcm(*[v.denominator for v in vec])
        ints = [int(v * den) for v in vec]
        g = np.zeros(nVar, dtype=int)
        g[order] = ints
        groups.append(g)
    return {"groups": np.array(groups, dtype=int).reshape(len(groups), nVar),
            "repeating": [order[p] for p in piv], "rank": len(piv)}


def scalingExponents(dmat, target, repeating: Sequence[int]) -> np.ndarray:
    """Exponents b (nVar,) with D b = target, nonzero only on the repeating variables."""
    d = np.atleast_2d(np.asarray(dmat, dtype=float))
    t = np.asarray(target, dtype=float).ravel()
    t = np.concatenate([t, np.zeros(max(0, d.shape[0] - t.size))])[:d.shape[0]]
    rep = list(repeating)
    rows = [[Fraction(v).limit_denominator(1000) for v in list(d[i, rep]) + [t[i]]] for i in range(d.shape[0])]
    red, piv = _rref(rows)
    if len(rep) in piv:
        raise ValueError("the output dimension cannot be formed from the repeating variables")
    b = np.zeros(d.shape[1])
    for r, p in enumerate(piv):
        b[rep[p]] = float(red[r][-1])
    if not np.allclose(d @ b, t):
        raise ValueError("the output dimension cannot be formed from the input variables")
    return b
