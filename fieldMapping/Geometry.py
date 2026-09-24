"""Vectorized computational geometry for field transfer between non-matching meshes.

    boxPairs            candidate pairs of overlapping axis-aligned boxes (uniform-grid hashing)
    triangleFrames      orthonormal frames (origin, e1, e2, normal) of triangles
    clipPolygons        Sutherland-Hodgman clipping of many convex polygons by triangles (2-D)
    clipHalfPlane       clipping of polygons by the half plane {a.x + c >= 0} (sign splitting)
    polygonMoments      area and centroid of many polygons
    closestPointOnTriangles   closest point, distance and barycentric coordinates (Ericson)

Polygons are stored as fixed-width arrays (P, maxVertices, d) with a vertex
count per polygon, so every operation runs as numpy array code over all
polygons at once.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- candidate search
def _sortedUnique(a: np.ndarray):
    """(unique values, first index, counts) of a sorted 1-D array (faster than hashing for large int arrays)."""
    if a.size == 0:
        return a, np.zeros(0, np.int64), np.zeros(0, np.int64)
    first = np.concatenate([[True], a[1:] != a[:-1]])
    start = np.flatnonzero(first)
    counts = np.diff(np.concatenate([start, [a.size]]))
    return a[start], start, counts


def boxPairs(loA: np.ndarray, hiA: np.ndarray, loB: np.ndarray, hiB: np.ndarray, cellSize=None,
             maxCellsPerBox: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """All (i, j) with box A_i overlapping box B_j (closed boxes), by uniform-grid hashing.

    The grid spacing defaults to the median extent of the B boxes; boxes
    spanning more than ``maxCellsPerBox`` cells fall back to a direct test.
    """
    loA, hiA, loB, hiB = (np.asarray(a, dtype=float) for a in (loA, hiA, loB, hiB))
    if loA.shape[0] == 0 or loB.shape[0] == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    d = loA.shape[1]
    ext = np.max(hiB - loB, axis=1)
    h = float(cellSize) if cellSize else float(np.median(ext[ext > 0])) if np.any(ext > 0) else 1.0
    h = max(h, 1e-12 * max(1.0, float(np.max(np.abs(np.vstack([loB, hiB]))))))
    origin = np.minimum(loA.min(axis=0), loB.min(axis=0))

    def cellsOf(lo, hi):
        a = np.floor((lo - origin) / h).astype(np.int64)
        b = np.floor((hi - origin) / h).astype(np.int64)
        return a, b

    def expand(lo, hi):
        a, b = cellsOf(lo, hi)
        span = b - a + 1
        count = np.prod(span, axis=1)
        big = count > maxCellsPerBox
        idx = np.flatnonzero(~big)
        reps = count[idx]
        owner = np.repeat(idx, reps)
        local = np.arange(owner.size) - np.repeat(np.cumsum(reps) - reps, reps)
        cells = np.empty((owner.size, d), dtype=np.int64)
        rem = local.copy()
        for k in range(d - 1, -1, -1):
            s = span[owner, k]
            cells[:, k] = a[owner, k] + rem % s
            rem //= s
        return owner, cells, np.flatnonzero(big)

    ownB, cellB, bigB = expand(loB, hiB)
    ownA, cellA, bigA = expand(loA, hiA)
    # hash cells to scalar keys
    base = np.max(np.vstack([cellA, cellB]), axis=0) + 1 if cellA.size or cellB.size else np.ones(d, np.int64)
    mult = np.cumprod(np.concatenate([[1], base[:-1]]))
    keyB, keyA = cellB @ mult, cellA @ mult
    order = np.argsort(keyB, kind="stable")
    keyB, ownB = keyB[order], ownB[order]
    uniq, start, counts = _sortedUnique(keyB)
    pos = np.searchsorted(uniq, keyA)
    pos = np.clip(pos, 0, max(uniq.size - 1, 0))
    hit = uniq.size > 0
    found = (uniq[pos] == keyA) if hit else np.zeros(keyA.size, bool)
    ia, pa = ownA[found], pos[found]
    n = counts[pa]
    ii = np.repeat(ia, n)
    jj = ownB[np.repeat(start[pa], n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))]
    pairs = [(ii, jj)]
    # oversized boxes: direct overlap tests
    for big, other, bigIsA in ((bigA, None, True), (bigB, None, False)):
        for i in big:
            if bigIsA:
                ok = np.all((loB <= hiA[i]) & (hiB >= loA[i]), axis=1)
                pairs.append((np.full(int(ok.sum()), i), np.flatnonzero(ok)))
            else:
                ok = np.all((loA <= hiB[i]) & (hiA >= loB[i]), axis=1)
                pairs.append((np.flatnonzero(ok), np.full(int(ok.sum()), i)))
    ii = np.concatenate([p[0] for p in pairs])
    jj = np.concatenate([p[1] for p in pairs])
    key = np.sort(ii.astype(np.int64) * loB.shape[0] + jj, kind="stable")
    key = _sortedUnique(key)[0]
    ii, jj = key // loB.shape[0], key % loB.shape[0]
    ok = np.all((loA[ii] <= hiB[jj]) & (hiA[ii] >= loB[jj]), axis=1)
    return ii[ok], jj[ok]


# ---------------------------------------------------------------- frames
def triangleFrames(tri: np.ndarray):
    """(origin, e1, e2, unit normal, area) of triangles (m, 3, 3)."""
    o = tri[:, 0]
    a, b = tri[:, 1] - o, tri[:, 2] - o
    n = np.cross(a, b)
    area2 = np.linalg.norm(n, axis=1)
    safe = np.where(area2 > 0, area2, 1.0)
    nrm = n / safe[:, None]
    la = np.linalg.norm(a, axis=1)
    e1 = a / np.where(la > 0, la, 1.0)[:, None]
    e2 = np.cross(nrm, e1)
    return o, e1, e2, nrm, 0.5 * area2


# ---------------------------------------------------------------- polygons
def polygonMoments(poly: np.ndarray, count: np.ndarray):
    """Signed area and centroid of polygons (P, V, 2) with ``count`` vertices (fan from vertex 0)."""
    p0 = poly[:, :1, :]
    a = poly[:, 1:-1, :] - p0
    b = poly[:, 2:, :] - p0
    cr = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]                      # (P, V-2) twice the fan areas
    valid = np.arange(poly.shape[1] - 2)[None, :] < (count[:, None] - 2)
    cr = np.where(valid, cr, 0.0)
    area = 0.5 * cr.sum(axis=1)
    cen = (poly[:, :1, :] + poly[:, 1:-1, :] + poly[:, 2:, :]) / 3.0          # fan triangle centroids
    cx = 0.5 * np.einsum("pv,pvd->pd", cr, cen)
    safe = np.where(np.abs(area) > 0, area, 1.0)
    return area, cx / safe[:, None]


def clipHalfPlane(poly: np.ndarray, count: np.ndarray, attrs: np.ndarray, value: np.ndarray):
    """Clip polygons to {value >= 0}, value linear along edges (given at the vertices).

    poly (P, V, 2), attrs (P, V, k) carried linearly (e.g. barycentric
    coordinates), value (P, V). Returns (poly, count, attrs, value) with
    capacity V + 1.
    """
    P, V, _ = poly.shape
    k = attrs.shape[2]
    outP = np.zeros((P, V + 1, 2))
    outA = np.zeros((P, V + 1, k))
    outV = np.zeros((P, V + 1))
    outN = np.zeros(P, dtype=np.int64)
    rows = np.arange(P)
    for i in range(V):
        j = (i + 1) % np.maximum(count, 1)
        active = i < count
        vi, vj = value[rows, i], value[rows, j]
        inI, inJ = vi >= 0, vj >= 0
        # keep vertex i if inside
        keep = active & inI
        idx = outN[keep]
        outP[rows[keep], idx] = poly[rows[keep], i]
        outA[rows[keep], idx] = attrs[rows[keep], i]
        outV[rows[keep], idx] = vi[keep]
        outN[keep] += 1
        # add the crossing point on a strict sign change (a zero vertex is itself on the line)
        cross = active & (((vi > 0) & (vj < 0)) | ((vi < 0) & (vj > 0)))
        denom = np.where(cross, vi - vj, 1.0)
        t = np.where(cross, vi / denom, 0.0)
        r = rows[cross]
        idx = outN[cross]
        tt = t[cross][:, None]
        outP[r, idx] = poly[r, i] + tt * (poly[r, j[cross]] - poly[r, i])
        outA[r, idx] = attrs[r, i] + tt * (attrs[r, j[cross]] - attrs[r, i])
        outV[r, idx] = 0.0
        outN[cross] += 1
    return outP, outN, outA, outV


def clipPolygons(poly: np.ndarray, count: np.ndarray, attrs: np.ndarray, clip: np.ndarray):
    """Clip convex polygons (P, V, 2) by counter-clockwise triangles clip (P, 3, 2).

    attrs (P, V, k) are interpolated linearly onto the new vertices. Returns
    (poly, count, attrs) with capacity V + 3.
    """
    cur, curA, n = poly, attrs, count.copy()
    for e in range(3):
        a, b = clip[:, e], clip[:, (e + 1) % 3]
        edge = b - a
        # inside: left of the edge (counter-clockwise clip polygon); a convex polygon gains at most one vertex
        val = edge[:, None, 0] * (cur[..., 1] - a[:, None, 1]) - edge[:, None, 1] * (cur[..., 0] - a[:, None, 0])
        cur, n, curA, _ = clipHalfPlane(cur, n, curA, val)
    return cur, n, curA


# ---------------------------------------------------------------- closest points
def closestPointOnTriangles(p: np.ndarray, tri: np.ndarray):
    """Closest point on each triangle (m, 3, 3) to p (m, 3): (point, distance, barycentric (m, 3))."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    bary = np.zeros((p.shape[0], 3))
    done = np.zeros(p.shape[0], dtype=bool)

    def setb(mask, w):
        nonlocal done
        m = mask & ~done
        bary[m] = w[m]
        done |= m
    one = np.ones(p.shape[0])
    zero = np.zeros(p.shape[0])
    setb((d1 <= 0) & (d2 <= 0), np.column_stack([one, zero, zero]))
    setb((d3 >= 0) & (d4 <= d3), np.column_stack([zero, one, zero]))
    setb((d6 >= 0) & (d5 <= d6), np.column_stack([zero, zero, one]))
    with np.errstate(divide="ignore", invalid="ignore"):
        v = d1 / (d1 - d3)
        setb((vc <= 0) & (d1 >= 0) & (d3 <= 0), np.column_stack([1 - v, v, zero]))
        w = d2 / (d2 - d6)
        setb((vb <= 0) & (d2 >= 0) & (d6 <= 0), np.column_stack([1 - w, zero, w]))
        w2 = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        setb((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), np.column_stack([zero, 1 - w2, w2]))
        den = va + vb + vc
        den = np.where(den != 0, den, 1.0)
        vv, ww = vb / den, vc / den
    setb(np.ones(p.shape[0], dtype=bool), np.column_stack([1 - vv - ww, vv, ww]))
    q = np.einsum("mk,mkd->md", bary, tri)
    return q, np.linalg.norm(p - q, axis=1), bary
