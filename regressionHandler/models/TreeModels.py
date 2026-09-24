"""Tree ensembles: random forest and gradient boosting (histogram trees, numpy only).

RandomForestModel
    Bootstrap aggregation of CART trees with feature subsampling per split.
    Out-of-bag predictions give an honest R^2 / RMSE; ``predictVariances``
    uses the bias-corrected infinitesimal jackknife (Wager, Hastie & Efron
    2014) for the confidence of the forest mean, plus the out-of-bag residual
    variance for ``kind="prediction"``.

GradientBoostingModel
    Stage-wise additive trees on the gradient / hessian of the loss
    (squared, absolute, huber, quantile) with shrinkage, row / column
    subsampling, L2 leaf regularization and early stopping on a validation
    split. Absolute / quantile / huber leaves are re-estimated as the exact
    loss minimizers (Friedman 2001).

References:
    Breiman (2001) Machine Learning 45(1).
    Friedman (2001) Annals of Statistics 29(5).
    Wager, Hastie & Efron (2014) JMLR 15.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase
from pythonLibs.regressionHandler.numerics.Trees import FeatureBinner, Tree, buildForestLevelwise, buildTree


def _nFeatures(spec, nx: int) -> Optional[int]:
    if spec is None or spec == "all":
        return None
    if spec == "sqrt":
        return max(1, int(np.sqrt(nx)))
    if spec == "third":
        return max(1, nx // 3)
    if isinstance(spec, float):
        return max(1, int(round(spec * nx)))
    return max(1, min(int(spec), nx))


class _TreeEnsemble(SurrogateModelBase):

    def _importance(self) -> np.ndarray:
        imp = np.zeros(self.nx)
        for t in self._trees:
            internal = t.feature >= 0
            np.add.at(imp, t.feature[internal], t.gain[internal])
        s = imp.sum()
        return imp / s if s > 0 else imp

    def featureImportance(self) -> dict:
        """Total split gain per input, normalized to 1."""
        self._checkTrained()
        return dict(zip(self.featureNames, self._importance().tolist()))

    def _effectiveParams(self):
        return float(np.mean([np.sum(t.feature < 0) for t in self._trees]))

    def _packTrees(self) -> dict:
        """All trees as concatenated arrays + offsets (compact persistence)."""
        ts = self._trees
        cat = lambda name: np.concatenate([getattr(t, name) for t in ts]) if ts else np.zeros(0)
        feat, thr = cat("feature").astype(np.int64), cat("threshold")
        # thresholds are bin edges: store the bin code (small integers) instead of the float
        code = np.full(feat.size, -1, dtype=np.int64)
        for j, e in enumerate(self._binner.edges):
            sel = feat == j
            code[sel] = np.searchsorted(e, thr[sel])
        return {"offsets": np.cumsum([0] + [t.feature.size for t in ts]).tolist(),
                "feature": feat.astype(np.int32).tolist(), "thresholdBin": code.astype(np.int32).tolist(),
                "left": cat("left").astype(np.int32).tolist(), "right": cat("right").astype(np.int32).tolist(),
                "value": cat("value").tolist(),
                "gain": cat("gain").astype(np.float32).astype(float).tolist(),
                "cover": cat("cover").astype(np.float32).astype(float).tolist()}

    def _unpackTrees(self, state: dict) -> list:
        if "trees" in state:                                   # older layout: one dict per tree
            return [Tree.fromDict(t) for t in state["trees"]]
        p = state["forest"]
        off = p["offsets"]
        arr = {k: np.asarray(p[k]) for k in ("feature", "left", "right", "value", "gain", "cover")}
        code = np.asarray(p["thresholdBin"], dtype=np.int64)
        thr = np.zeros(code.size)
        for j, e in enumerate(self._binner.edges):
            sel = arr["feature"] == j
            thr[sel] = e[code[sel]]
        arr["threshold"] = thr
        return [Tree(feature=arr["feature"][a:b].astype(np.int64), threshold=arr["threshold"][a:b].astype(float),
                     left=arr["left"][a:b].astype(np.int64), right=arr["right"][a:b].astype(np.int64),
                     value=arr["value"][a:b].astype(float), gain=arr["gain"][a:b].astype(float),
                     cover=arr["cover"][a:b].astype(float)) for a, b in zip(off[:-1], off[1:])]


@registry("model").register("randomForest")
class RandomForestModel(_TreeEnsemble):
    """Random forest regression with out-of-bag error and infinitesimal-jackknife uncertainty."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("nTrees", 200, types=int, lower=1, desc="Number of trees")
        d("maxFeatures", "third", types=(str, int, float),
          desc="Features tried per split: 'sqrt', 'third', 'all', a fraction or a count")
        d("minSamplesLeaf", 5, types=int, lower=1, desc="Minimum samples per leaf")
        d("maxDepth", None, types=int, lower=1, desc="Maximum depth (None: unlimited)")
        d("bootstrap", True, types=bool, desc="Bootstrap rows per tree")
        d("maxBins", 255, types=int, lower=2, upper=256, desc="Histogram bins per feature")
        d("seed", 0, types=int, desc="Random seed")
        d("nJobs", 1, types=int, lower=1, desc="Threads building trees (numpy-bound; >1 helps only with large n)")
        self.supports.update(multiOutput=False, variances=True, weights=False)

    def _train(self) -> None:
        x, y = self.xt, self.yt[:, 0]
        n = y.size
        self._binner = FeatureBinner(self.options["maxBins"]).fit(x)
        codes = self._binner.transform(x)
        mf = _nFeatures(self.options["maxFeatures"], self.nx)
        g, h = -y, np.ones(n)
        nTrees = self.options["nTrees"]
        rngRows = np.random.default_rng(self.options["seed"])
        rowSets = [rngRows.integers(0, n, n) if self.options["bootstrap"] else np.arange(n) for _ in range(nTrees)]
        # grow trees in batches (all trees of a batch share every level's numpy calls)
        batch = max(1, min(nTrees, 4_000_000 // max(n * self.nx, 1)))
        groups = [list(range(s, min(nTrees, s + batch))) for s in range(0, nTrees, batch)]

        def grow(group):
            rng = np.random.default_rng([self.options["seed"], group[0]])     # independent of thread order
            return buildForestLevelwise(codes, self._binner, g, h, [rowSets[b] for b in group],
                                        self.options["maxDepth"], self.options["minSamplesLeaf"], 0.0, mf, rng)

        jobs = self.options["nJobs"]
        if jobs > 1 and len(groups) > 1:
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                parts = list(pool.map(grow, groups))
        else:
            parts = [grow(gr) for gr in groups]
        self._trees = [t for part in parts for t in part]
        self._inbag = np.array([np.bincount(r, minlength=n) for r in rowSets], dtype=np.int32)
        inbag = self._inbag
        preds = np.array([t.predict(x) for t in self._trees])            # (B, n)
        oob = inbag == 0
        cnt = oob.sum(axis=0)
        self._oob = np.where(cnt > 0, np.sum(preds * oob, axis=0) / np.maximum(cnt, 1), np.nan)
        ok = np.isfinite(self._oob)
        r = y[ok] - self._oob[ok]
        self._oobMse = float(np.mean(r * r)) if ok.any() else float("nan")
        self._oobR2 = 1.0 - self._oobMse / float(np.var(y[ok])) if ok.any() and np.var(y[ok]) > 0 else float("nan")

    def _treePredictions(self, x) -> np.ndarray:
        return np.array([t.predict(x) for t in self._trees])

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        return self._treePredictions(x).mean(axis=0)[:, None]

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        preds = self._treePredictions(x)                                  # (B, m)
        nTrees, n = self._inbag.shape
        tc = preds - preds.mean(axis=0)
        nc = self._inbag - self._inbag.mean(axis=0)
        var = np.empty(x.shape[0])
        for s in range(0, x.shape[0], 512):
            cov = nc.T @ tc[:, s:s + 512] / nTrees                         # (n, m)
            ij = np.sum(cov * cov, axis=0)
            correction = n / nTrees ** 2 * np.sum(tc[:, s:s + 512] ** 2, axis=0)
            # bias-corrected IJ; its correction is noisy when nTrees << n, so it is floored at the
            # Monte Carlo variance of the tree average (use nTrees of order n for a sharp estimate)
            var[s:s + 512] = np.maximum(ij - correction, np.sum(tc[:, s:s + 512] ** 2, axis=0) / nTrees ** 2)
        if kind == "prediction":
            var = var + self._oobMse
        return var[:, None]

    @property
    def outOfBag(self) -> dict:
        """Out-of-bag predictions, MSE and R^2."""
        self._checkTrained()
        return {"predictions": self._oob, "mse": self._oobMse, "rSquared": self._oobR2}

    def _stateToDict(self) -> dict:
        return {"binner": self._binner.toDict(), "forest": self._packTrees(),
                "inbag": self._inbag.tolist(), "oob": [None if not np.isfinite(v) else float(v) for v in self._oob],
                "oobMse": self._oobMse, "oobR2": self._oobR2}

    def _stateFromDict(self, state: dict) -> None:
        self._binner = FeatureBinner.fromDict(state["binner"])
        self._trees = self._unpackTrees(state)
        self._inbag = np.array(state["inbag"], dtype=np.int32)
        self._oob = np.array([np.nan if v is None else v for v in state["oob"]], dtype=float)
        self._oobMse, self._oobR2 = float(state["oobMse"]), float(state["oobR2"])


_LOSSES = ("squared", "absolute", "huber", "quantile")


@registry("model").register("gradientBoosting")
class GradientBoostingModel(_TreeEnsemble):
    """Gradient-boosted histogram trees (squared, absolute, huber or quantile loss) with early stopping."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("loss", "squared", values=_LOSSES, desc="Loss function")
        d("tau", 0.5, types=float, lower=1e-6, upper=1 - 1e-6, desc="Quantile level for loss='quantile'")
        d("huberAlpha", 0.9, types=float, lower=0.0, upper=1.0, desc="Residual quantile defining the huber delta")
        d("nEstimators", 500, types=int, lower=1, desc="Maximum number of trees")
        d("learningRate", 0.1, types=float, lower=0.0, desc="Shrinkage")
        d("maxDepth", None, types=int, lower=1, desc="Maximum depth")
        d("maxLeaves", 31, types=int, lower=2, desc="Maximum leaves per tree (best-first growth)")
        d("minSamplesLeaf", 10, types=int, lower=1, desc="Minimum samples per leaf")
        d("l2", 1.0, types=(int, float), lower=0.0, desc="L2 regularization of leaf values")
        d("subsample", 1.0, types=float, lower=0.0, upper=1.0, desc="Row fraction per tree")
        d("colsample", 1.0, types=float, lower=0.0, upper=1.0, desc="Feature fraction per split")
        d("validationFraction", 0.1, types=float, lower=0.0, upper=0.9,
          desc="Held-out fraction for early stopping (0 disables)")
        d("patience", 30, types=int, lower=1, desc="Rounds without validation improvement before stopping")
        d("refit", True, types=bool, desc="After early stopping, refit on all rows with the selected number of trees")
        d("maxBins", 255, types=int, lower=2, upper=256, desc="Histogram bins per feature")
        d("seed", 0, types=int, desc="Random seed")
        self.supports.update(multiOutput=False, variances=False, weights=False)

    # ------------------------------------------------------------------ loss pieces
    def _init(self, y) -> float:
        loss = self.options["loss"]
        if loss == "squared":
            return float(np.mean(y))
        if loss == "quantile":
            return float(np.quantile(y, self.options["tau"]))
        return float(np.median(y))

    def _gradHess(self, y, f):
        r = y - f
        loss = self.options["loss"]
        if loss == "squared":
            return -r, np.ones_like(r)
        if loss == "absolute":
            return -np.sign(r), np.ones_like(r)
        if loss == "quantile":
            tau = self.options["tau"]
            return np.where(r >= 0, -tau, 1.0 - tau), np.ones_like(r)
        self._delta = float(np.quantile(np.abs(r), self.options["huberAlpha"]))
        return -np.where(np.abs(r) <= self._delta, r, self._delta * np.sign(r)), np.ones_like(r)

    def _lossValue(self, y, f) -> float:
        r = y - f
        loss = self.options["loss"]
        if loss == "squared":
            return float(np.mean(r * r))
        if loss == "absolute":
            return float(np.mean(np.abs(r)))
        if loss == "quantile":
            tau = self.options["tau"]
            return float(np.mean(r * (tau - (r < 0))))
        d = getattr(self, "_delta", float(np.quantile(np.abs(r), self.options["huberAlpha"])))
        a = np.abs(r)
        return float(np.mean(np.where(a <= d, 0.5 * r * r, d * (a - 0.5 * d))))

    def _refitLeaves(self, tree: Tree, x, y, f, rows) -> None:
        """Exact leaf values for non-quadratic losses (median / quantile / huber step of the residuals)."""
        loss = self.options["loss"]
        if loss == "squared":
            return
        leaf = tree.apply(x[rows])
        r = y[rows] - f[rows]
        for node in np.unique(leaf):
            rr = r[leaf == node]
            if loss == "absolute":
                v = np.median(rr)
            elif loss == "quantile":
                v = np.quantile(rr, self.options["tau"])
            else:
                med = np.median(rr)
                dev = rr - med
                v = med + np.mean(np.sign(dev) * np.minimum(np.abs(dev), self._delta))
            tree.value[node] = v

    # ------------------------------------------------------------------ training
    def _train(self) -> None:
        x, y = self.xt, self.yt[:, 0]
        n = y.size
        rng = np.random.default_rng(self.options["seed"])
        vf = self.options["validationFraction"]
        perm = rng.permutation(n)
        nVal = int(round(vf * n)) if vf > 0 and n >= 20 else 0
        self._boost(x, y, perm[:nVal], perm[nVal:], self.options["nEstimators"], rng)
        if nVal and self.options["refit"]:
            history, nTrees = self._history, len(self._trees)
            self._boost(x, y, perm[:0], np.arange(n), max(nTrees, 1), np.random.default_rng(self.options["seed"]))
            self._history, self._nVal, self._selectedByValidation = history, nVal, nTrees

    def _boost(self, x, y, val, tr, nEstimators, rng) -> None:
        n = y.size
        nVal = val.size
        self._binner = FeatureBinner(self.options["maxBins"]).fit(x[tr])
        codes = self._binner.transform(x)
        mf = _nFeatures(self.options["colsample"], self.nx) if self.options["colsample"] < 1.0 else None
        self._f0 = self._init(y[tr])
        f = np.full(n, self._f0)
        lr = self.options["learningRate"]
        self._trees, history = [], []
        best, bestIter, stall = np.inf, 0, 0
        for it in range(nEstimators):
            g, h = self._gradHess(y, f)
            rows = tr if self.options["subsample"] >= 1.0 else \
                np.sort(rng.choice(tr, max(2, int(self.options["subsample"] * tr.size)), replace=False))
            tree = buildTree(codes, self._binner, g, h, rows, self.options["maxDepth"], self.options["maxLeaves"],
                             self.options["minSamplesLeaf"], 1e-12, float(self.options["l2"]), 0.0, mf, rng)
            self._refitLeaves(tree, x, y, f, rows)
            tree.value *= lr
            f = f + tree.predict(x)
            self._trees.append(tree)
            if nVal:
                score = self._lossValue(y[val], f[val])
                history.append(score)
                if not np.isfinite(best) or score < best - 1e-12 * abs(best):
                    best, bestIter, stall = score, it + 1, 0
                else:
                    stall += 1
                    if stall >= self.options["patience"]:
                        break
            else:
                history.append(self._lossValue(y[tr], f[tr]))
        if nVal:
            self._trees = self._trees[:bestIter]
        self._history = history
        self._nVal = nVal

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        f = np.full(x.shape[0], self._f0)
        for t in self._trees:
            f += t.predict(x)
        return f[:, None]

    def stagedPredict(self, x, every: int = 1) -> np.ndarray:
        """Predictions after every ``every`` trees, shape (nStages, m)."""
        self._checkTrained()
        xv = self._validX(x)
        f = np.full(xv.shape[0], self._f0)
        out = []
        for i, t in enumerate(self._trees, 1):
            f = f + t.predict(xv)
            if i % every == 0:
                out.append(f.copy())
        return np.array(out)

    @property
    def trainingHistory(self) -> dict:
        self._checkTrained()
        return {"loss": list(self._history), "nTrees": len(self._trees), "validationSize": self._nVal}

    def _stateToDict(self) -> dict:
        return {"binner": self._binner.toDict(), "forest": self._packTrees(), "f0": self._f0,
                "history": self._history, "nVal": self._nVal}

    def _stateFromDict(self, state: dict) -> None:
        self._binner = FeatureBinner.fromDict(state["binner"])
        self._trees = self._unpackTrees(state)
        self._f0 = float(state["f0"])
        self._history = list(state["history"])
        self._nVal = int(state["nVal"])
