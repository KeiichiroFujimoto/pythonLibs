"""Histogram-based regression trees (building block of random forests and gradient boosting).

Inputs are quantile-binned once (at most ``maxBins`` codes per feature); a
node's split search then only needs the per-bin sums of gradients g and
hessians h, which one ``bincount`` gives for all features at once. The
split gain is the second-order (Newton) gain

    gain = G_L^2 / (H_L + lambda) + G_R^2 / (H_R + lambda) - G^2 / (H + lambda)

and a leaf predicts -G / (H + lambda). With g = -y and h = 1 (lambda = 0)
this is the ordinary variance-reduction CART used by random forests. The
larger child's histogram is obtained by subtraction from the parent's.

References:
    Breiman, Friedman, Olshen & Stone (1984) *Classification and Regression Trees*.
    Chen & Guestrin (2016) KDD (second-order gains).
    Ke et al. (2017) NIPS (histogram-based trees).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


class FeatureBinner:
    """Quantile bins per feature: codes 0..nBins-1 (uint8 for <= 256 bins)."""

    def __init__(self, maxBins: int = 255) -> None:
        if not 2 <= maxBins <= 256:
            raise ValueError("maxBins must be in [2, 256]")
        self.maxBins = maxBins

    def fit(self, x: np.ndarray) -> "FeatureBinner":
        self.edges = []
        for j in range(x.shape[1]):
            u = np.unique(x[:, j])
            if u.size <= self.maxBins:
                e = 0.5 * (u[1:] + u[:-1])
            else:
                q = np.quantile(x[:, j], np.linspace(0, 1, self.maxBins + 1)[1:-1])
                e = np.unique(q)
            self.edges.append(e)
        return self

    @property
    def nBins(self) -> np.ndarray:
        return np.array([e.size + 1 for e in self.edges])

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(x.shape, dtype=np.uint8 if self.maxBins <= 256 else np.int32)
        for j, e in enumerate(self.edges):
            out[:, j] = np.searchsorted(e, x[:, j], side="left")
        return out

    def threshold(self, feature: int, code: int) -> float:
        """Split value: codes <= code go left, i.e. x <= edges[code]."""
        return float(self.edges[feature][code])

    def toDict(self) -> dict:
        return {"maxBins": self.maxBins, "edges": [e.tolist() for e in self.edges]}

    @classmethod
    def fromDict(cls, d: dict) -> "FeatureBinner":
        b = cls(int(d["maxBins"]))
        b.edges = [np.array(e, dtype=float) for e in d["edges"]]
        return b


@dataclass
class Tree:
    """Flat binary tree: internal nodes have feature >= 0; leaves have feature = -1."""
    feature: np.ndarray
    threshold: np.ndarray        # raw-value threshold (x <= t goes left)
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray
    gain: np.ndarray             # split gain per internal node (importance)
    cover: np.ndarray            # hessian sum per node

    def predict(self, x: np.ndarray) -> np.ndarray:
        node = np.zeros(x.shape[0], dtype=np.int64)
        active = self.feature[node] >= 0
        while np.any(active):
            idx = np.flatnonzero(active)
            n = node[idx]
            f = self.feature[n]
            goLeft = x[idx, f] <= self.threshold[n]
            node[idx] = np.where(goLeft, self.left[n], self.right[n])
            active[idx] = self.feature[node[idx]] >= 0
        return self.value[node]

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Leaf index of every row."""
        node = np.zeros(x.shape[0], dtype=np.int64)
        active = self.feature[node] >= 0
        while np.any(active):
            idx = np.flatnonzero(active)
            n = node[idx]
            goLeft = x[idx, self.feature[n]] <= self.threshold[n]
            node[idx] = np.where(goLeft, self.left[n], self.right[n])
            active[idx] = self.feature[node[idx]] >= 0
        return node

    @property
    def depth(self) -> int:
        d = np.zeros(self.feature.size, dtype=int)
        for i in range(self.feature.size):
            if self.feature[i] >= 0:
                d[self.left[i]] = d[i] + 1
                d[self.right[i]] = d[i] + 1
        return int(d.max())

    def toDict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("feature", "threshold", "left", "right", "value", "gain",
                                                         "cover")}

    @classmethod
    def fromDict(cls, d: dict) -> "Tree":
        return cls(feature=np.array(d["feature"], dtype=np.int64), threshold=np.array(d["threshold"], dtype=float),
                   left=np.array(d["left"], dtype=np.int64), right=np.array(d["right"], dtype=np.int64),
                   value=np.array(d["value"], dtype=float), gain=np.array(d["gain"], dtype=float),
                   cover=np.array(d["cover"], dtype=float))


def buildTree(codes: np.ndarray, binner: FeatureBinner, g: np.ndarray, h: np.ndarray,
              rows: Optional[np.ndarray] = None, maxDepth: Optional[int] = None, maxLeaves: Optional[int] = None,
              minSamplesLeaf: int = 1, minHessian: float = 1e-3, lam: float = 0.0, minGain: float = 0.0,
              maxFeatures: Optional[int] = None, rng: Optional[np.random.Generator] = None,
              sampleCount: Optional[np.ndarray] = None) -> Tree:
    """Grow one tree on the rows ``rows`` (default all) of the binned matrix.

    ``sampleCount`` holds per-row multiplicities (bootstrap) used for minSamplesLeaf.
    Growth is best-first (largest gain first) until maxLeaves / maxDepth / no gain.
    """
    n, nf = codes.shape
    rows = np.arange(n) if rows is None else rows
    cnt = np.ones(n) if sampleCount is None else sampleCount.astype(float)
    nb = int(binner.nBins.max())
    offsets = np.arange(nf) * nb
    maxDepth = maxDepth if maxDepth is not None else 64
    maxLeaves = maxLeaves if maxLeaves is not None else 1 << 30
    rng = rng or np.random.default_rng(0)

    def histogram(r):
        flat = (codes[r].astype(np.int64) + offsets[None, :]).ravel()
        gg = np.bincount(flat, weights=np.repeat(g[r], nf), minlength=nf * nb).reshape(nf, nb)
        hh = np.bincount(flat, weights=np.repeat(h[r], nf), minlength=nf * nb).reshape(nf, nb)
        cc = np.bincount(flat, weights=np.repeat(cnt[r], nf), minlength=nf * nb).reshape(nf, nb)
        return gg, hh, cc

    def bestSplit(hist, depth):
        gg, hh, cc = hist
        if depth >= maxDepth:
            return None
        gTot, hTot, cTot = gg[0].sum(), hh[0].sum(), cc[0].sum()
        if cTot < 2 * minSamplesLeaf:
            return None
        gl, hl, cl = np.cumsum(gg, axis=1)[:, :-1], np.cumsum(hh, axis=1)[:, :-1], np.cumsum(cc, axis=1)[:, :-1]
        gr, hr, cr = gTot - gl, hTot - hl, cTot - cl
        ok = (cl >= minSamplesLeaf) & (cr >= minSamplesLeaf) & (hl >= minHessian) & (hr >= minHessian)
        for j, e in enumerate(binner.edges):                   # only existing edges are valid split points
            ok[j, e.size:] = False
        if maxFeatures is not None and maxFeatures < nf:
            keep = np.zeros(nf, dtype=bool)
            keep[rng.choice(nf, maxFeatures, replace=False)] = True
            ok &= keep[:, None]
        if not np.any(ok):
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = gl ** 2 / (hl + lam) + gr ** 2 / (hr + lam) - gTot ** 2 / (hTot + lam)
        gain = np.where(ok, gain, -np.inf)
        f, b = np.unravel_index(int(np.argmax(gain)), gain.shape)
        if not np.isfinite(gain[f, b]) or gain[f, b] <= minGain:
            return None
        return float(gain[f, b]), int(f), int(b)

    feature, threshold, left, right, value, gainArr, cover = [], [], [], [], [], [], []

    def newNode(r, hist):
        gg, hh, _ = hist
        gS, hS = gg[0].sum(), hh[0].sum()
        feature.append(-1)
        threshold.append(0.0)
        left.append(-1)
        right.append(-1)
        value.append(-gS / (hS + lam) if hS + lam > 0 else 0.0)
        gainArr.append(0.0)
        cover.append(hS)
        return len(feature) - 1

    rootHist = histogram(rows)
    root = newNode(rows, rootHist)
    frontier = []                                              # (-gain, node, rows, hist, depth, split)
    s = bestSplit(rootHist, 0)
    if s is not None:
        frontier.append((s, root, rows, rootHist, 0))
    leaves = 1
    while frontier and leaves < maxLeaves:
        k = int(np.argmax([fr[0][0] for fr in frontier]))
        (gain, f, b), node, r, hist, depth = frontier.pop(k)
        goLeft = codes[r, f] <= b
        rl, rr = r[goLeft], r[~goLeft]
        small, large = (rl, rr) if rl.size <= rr.size else (rr, rl)
        hSmall = histogram(small)
        hLarge = tuple(p - c for p, c in zip(hist, hSmall))
        hl_, hr_ = (hSmall, hLarge) if rl.size <= rr.size else (hLarge, hSmall)
        li = newNode(rl, hl_)
        ri = newNode(rr, hr_)
        feature[node], threshold[node] = f, binner.threshold(f, b)
        left[node], right[node], gainArr[node] = li, ri, gain
        leaves += 1
        for child, cr_, ch in ((li, rl, hl_), (ri, rr, hr_)):
            sp = bestSplit(ch, depth + 1)
            if sp is not None:
                frontier.append((sp, child, cr_, ch, depth + 1))
    return Tree(feature=np.array(feature, dtype=np.int64), threshold=np.array(threshold, dtype=float),
                left=np.array(left, dtype=np.int64), right=np.array(right, dtype=np.int64),
                value=np.array(value, dtype=float), gain=np.array(gainArr, dtype=float),
                cover=np.array(cover, dtype=float))


def buildTreeLevelwise(codes: np.ndarray, binner: FeatureBinner, g: np.ndarray, h: np.ndarray,
                       rows: Optional[np.ndarray] = None, maxDepth: Optional[int] = None, minSamplesLeaf: int = 1,
                       lam: float = 0.0, maxFeatures: Optional[int] = None,
                       rng: Optional[np.random.Generator] = None, maxCells: int = 8_000_000) -> Tree:
    """Depth-wise growth with all nodes of a level processed together.

    One ``bincount`` per level builds the (node, feature, bin) histograms of
    every open node, and the split search is vectorized over nodes, so deep
    trees (random forests) cost a few dozen numpy calls instead of one per node.
    ``rows`` may contain repeats (bootstrap multiplicities).
    """
    n, nf = codes.shape
    rows = np.arange(n) if rows is None else np.asarray(rows)
    rng = rng or np.random.default_rng(0)
    nb = int(binner.nBins.max())
    validBin = np.zeros((nf, nb - 1), dtype=bool)
    for j, e in enumerate(binner.edges):
        validBin[j, :e.size] = True
    maxDepth = maxDepth if maxDepth is not None else 64
    gs, hs = g[rows], h[rows]
    cs = codes[rows].astype(np.int64)
    node = np.zeros(rows.size, dtype=np.int64)            # node id of every sample occurrence
    feature, threshold, left, right, value, gainArr, cover = [-1], [0.0], [-1], [-1], [0.0], [0.0], [0.0]
    open_ = np.array([0])
    depth = 0
    while open_.size and depth <= maxDepth:
        # local index of each open node; samples in closed nodes are dropped
        local = np.full(len(feature), -1, dtype=np.int64)
        local[open_] = np.arange(open_.size)
        ln = local[node]
        act = ln >= 0
        cs, gs, hs, node, ln = cs[act], gs[act], hs[act], node[act], ln[act]
        k = open_.size
        gTot = np.bincount(ln, weights=gs, minlength=k)
        hTot = np.bincount(ln, weights=hs, minlength=k)
        cTot = np.bincount(ln, minlength=k).astype(float)
        for i, nd in enumerate(open_):
            value[nd] = -gTot[i] / (hTot[i] + lam) if hTot[i] + lam > 0 else 0.0
            cover[nd] = hTot[i]
        if depth == maxDepth:
            break
        bestGain = np.full(k, -np.inf)
        bestF = np.zeros(k, dtype=np.int64)
        bestB = np.zeros(k, dtype=np.int64)
        chunk = max(1, maxCells // (nf * nb))
        for c0 in range(0, k, chunk):
            c1 = min(k, c0 + chunk)
            sel = (ln >= c0) & (ln < c1)
            idx = (((ln[sel] - c0)[:, None] * nf + np.arange(nf)[None, :]) * nb + cs[sel]).ravel()
            size = (c1 - c0) * nf * nb
            gg = np.bincount(idx, weights=np.repeat(gs[sel], nf), minlength=size).reshape(c1 - c0, nf, nb)
            hh = np.bincount(idx, weights=np.repeat(hs[sel], nf), minlength=size).reshape(c1 - c0, nf, nb)
            cc = np.bincount(idx, minlength=size).reshape(c1 - c0, nf, nb).astype(float)
            gl, hl, cl = np.cumsum(gg, 2)[:, :, :-1], np.cumsum(hh, 2)[:, :, :-1], np.cumsum(cc, 2)[:, :, :-1]
            G, H, C = gTot[c0:c1, None, None], hTot[c0:c1, None, None], cTot[c0:c1, None, None]
            gr, hr, cr = G - gl, H - hl, C - cl
            ok = (cl >= minSamplesLeaf) & (cr >= minSamplesLeaf) & (hl > 0) & (hr > 0) & validBin[None]
            if maxFeatures is not None and maxFeatures < nf:
                keys = rng.random((c1 - c0, nf))
                thresh = np.sort(keys, axis=1)[:, maxFeatures - 1:maxFeatures]
                ok &= (keys <= thresh)[:, :, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                gain = gl ** 2 / (hl + lam) + gr ** 2 / (hr + lam) - G ** 2 / (H + lam)
            gain = np.where(ok, gain, -np.inf).reshape(c1 - c0, -1)
            arg = np.argmax(gain, axis=1)
            bestGain[c0:c1] = gain[np.arange(c1 - c0), arg]
            bestF[c0:c1], bestB[c0:c1] = np.divmod(arg, nb - 1)
        split = np.isfinite(bestGain) & (bestGain > 1e-12 * np.maximum(np.abs(gTot) ** 2 / np.maximum(hTot, 1e-300),
                                                                       1e-300))
        if not np.any(split):
            break
        childOf = np.full((k, 2), -1, dtype=np.int64)
        for i in np.flatnonzero(split):
            nd = open_[i]
            feature[nd], threshold[nd] = int(bestF[i]), binner.threshold(int(bestF[i]), int(bestB[i]))
            gainArr[nd] = float(bestGain[i])
            for side in range(2):
                feature.append(-1)
                threshold.append(0.0)
                left.append(-1)
                right.append(-1)
                value.append(0.0)
                gainArr.append(0.0)
                cover.append(0.0)
                childOf[i, side] = len(feature) - 1
            left[nd], right[nd] = int(childOf[i, 0]), int(childOf[i, 1])
        movers = split[ln]
        f = bestF[ln[movers]]
        goLeft = cs[movers, :][np.arange(f.size), f] <= bestB[ln[movers]]
        node[movers] = np.where(goLeft, childOf[ln[movers], 0], childOf[ln[movers], 1])
        node[~movers] = -1                                   # samples in new leaves leave the frontier
        keep = node >= 0
        cs, gs, hs, node = cs[keep], gs[keep], hs[keep], node[keep]
        open_ = childOf[split].ravel()
        depth += 1
    return Tree(feature=np.array(feature, dtype=np.int64), threshold=np.array(threshold, dtype=float),
                left=np.array(left, dtype=np.int64), right=np.array(right, dtype=np.int64),
                value=np.array(value, dtype=float), gain=np.array(gainArr, dtype=float),
                cover=np.array(cover, dtype=float))
