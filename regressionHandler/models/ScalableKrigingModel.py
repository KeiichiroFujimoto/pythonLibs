"""Kriging for large data sets: nearest-neighbour (Vecchia) and inducing-point (FITC) approximations.

Same model and options as ``KrigingModel`` (kernel, trend, covariates,
nugget, normalization); only the likelihood and the predictor are
approximated so that n = 10^5 and more points fit in memory and time.

approximation="vecchia" (default)::

    p(y) ~= prod_i p(y_i | y_N(i)),   N(i) = the m nearest points preceding i in an ordering

  Ordering: max-min distance (n <= ``maxminLimit``) or random. The
  conditionals are solved in batches of (m+1) x (m+1) systems, so a
  likelihood evaluation costs O(n m^3) time and O(n m) memory. With ARD
  lengthscales the neighbours are recomputed once in the fitted metric and
  the parameters re-optimized. Prediction conditions on the m nearest
  training points. m = n - 1 reproduces exact Kriging.

approximation="fitc"::

    Cov(y) ~= K_nz K_zz^-1 K_zn + diag(K - K_nz K_zz^-1 K_zn) + lambda I

  with ``nInducing`` inducing points chosen as a max-min subset of the data;
  O(n z^2) per evaluation.

References:
    Vecchia (1988) JRSS B 50(2).
    Datta, Banerjee, Finley & Gelfand (2016) JASA 111(514).
    Guinness (2018) Technometrics 60(4).
    Snelson & Ghahramani (2006) NIPS 18.
"""
from __future__ import annotations

import numpy as np

from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.models.KrigingModel import KrigingModel
from pythonLibs.regressionHandler.numerics.NeighborSearch import NeighborSearch, _bruteForce
from pythonLibs.regressionHandler.numerics.Optimizers import minimize, multiStart
from pythonLibs.regressionHandler.sampling.Sampling import latinHypercube


# ---------------------------------------------------------------- ordering and neighbour sets
def maxminOrder(x: np.ndarray, seed: int = 0) -> np.ndarray:
    """Max-min distance ordering (each next point is the farthest from the ones already chosen)."""
    n = x.shape[0]
    first = int(np.argmin(np.sum((x - x.mean(axis=0)) ** 2, axis=1)))
    order = np.empty(n, dtype=np.int64)
    order[0] = first
    dmin = np.sum((x - x[first]) ** 2, axis=1)
    dmin[first] = -1.0
    for k in range(1, n):
        j = int(np.argmax(dmin))
        order[k] = j
        np.minimum(dmin, np.sum((x - x[j]) ** 2, axis=1), out=dmin)
        dmin[j] = -1.0
    return order


def previousNeighbors(x: np.ndarray, m: int) -> np.ndarray:
    """(n, m) indices of the m nearest *preceding* points of each point (x already ordered).

    Rows i < m hold i valid entries followed by -1. The prefix is indexed by a
    grid built on doubling blocks, so the search costs about O(n log n).
    """
    n = x.shape[0]
    out = np.full((n, m), -1, dtype=np.int64)
    start = min(n, max(4 * m, 256))
    # first block: brute force among preceding points
    for i in range(1, start):
        d = np.sum((x[:i] - x[i]) ** 2, axis=1)
        k = min(m, i)
        idx = np.argpartition(d, k - 1)[:k] if k < i else np.arange(i)
        out[i, :k] = idx[np.argsort(d[idx])]
    s = start
    while s < n:
        e = min(n, 2 * s)
        prefix = NeighborSearch(x[:s])
        dPre, iPre = prefix.query(x[s:e], m)
        # preceding points inside the block: query the block itself, keep earlier ones
        kb = min(e - s, 2 * m + 1)
        dBlk, iBlk = (NeighborSearch(x[s:e]).query(x[s:e], kb) if e - s > kb else
                      _bruteForce(x[s:e], x[s:e], kb))
        iBlk = iBlk + s
        rows = np.arange(s, e)[:, None]
        valid = iBlk < rows
        dBlk = np.where(valid, dBlk, np.inf)
        dAll = np.hstack([dPre, dBlk])
        iAll = np.hstack([iPre, iBlk])
        pick = np.argsort(dAll, axis=1)[:, :m]
        out[s:e] = np.take_along_axis(iAll, pick, axis=1)
        s = e
    return out


def _chordal(lonlat: np.ndarray) -> np.ndarray:
    rad = np.pi / 180.0
    lon, lat = lonlat[:, 0] * rad, lonlat[:, 1] * rad
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


@registry("model").register("scalableKriging")
class ScalableKrigingModel(KrigingModel):
    """Kriging for large n: Vecchia nearest-neighbour or FITC inducing-point likelihood and predictor."""

    def _initialize(self) -> None:
        super()._initialize()
        d = self.options.declare
        d("approximation", "vecchia", values=("vecchia", "fitc"), desc="Likelihood / predictor approximation")
        d("neighbors", 20, types=int, lower=1, desc="Vecchia conditioning set size m")
        d("predictionNeighbors", None, types=int, lower=1,
          desc="Training points conditioned on per prediction (default 3 x neighbors)")
        d("ordering", "auto", values=("auto", "maxmin", "random"), desc="Vecchia ordering (auto: maxmin if small)")
        d("maxminLimit", 20000, types=int, lower=1, desc="Largest n ordered by max-min in 'auto'")
        d("refineNeighbors", True, types=bool, desc="Recompute neighbours in the fitted metric and refit once")
        d("nInducing", 200, types=int, lower=2, desc="FITC inducing points")
        d("subsampleSize", 800, types=int, lower=50,
          desc="Above this n the optimizer starts from an exact fit on a random subsample of this size")
        d("chunkSize", 4096, types=int, lower=64, desc="Points per batched block")
        self.options["likelihood"] = "ml"
        self.options["nStart"] = 3
        self.supports.update(covariance=False, derivatives=False, weights=False)

    def _validateOptions(self) -> None:
        super()._validateOptions()
        if self.options["likelihood"] not in ("ml", "reml"):
            raise ValueError("scalable Kriging supports likelihood 'ml' or 'reml'")
        if self.options["plsComponents"]:
            raise ValueError("plsComponents is not supported by scalable Kriging")

    # ------------------------------------------------------------------ geometry
    def _metricCoords(self, xs: np.ndarray, kp=None) -> np.ndarray:
        """Coordinates in which Euclidean distance orders neighbours like the kernel does."""
        s = xs[:, self._sc]
        k = self._kernel
        if k.options.get("distance") == "greatCircle":
            return _chordal(s)
        if kp is None or not hasattr(k, "invL2"):
            return s
        if k._isCholesky():
            return s @ k._cholesky(kp[: k._nDistance()])
        return k._transform(s) * np.sqrt(k.invL2(kp))

    def _setGeometry(self, kp=None) -> None:
        self._geometryParams = None if kp is None else np.asarray(kp, dtype=float)
        n = self._xs.shape[0]
        coords = self._metricCoords(self._xs, kp)
        if self.options["approximation"] == "fitc":
            self._order = np.arange(n)
            nz = min(self.options["nInducing"], n)
            rng = np.random.default_rng(self.options["seed"])
            sub = rng.choice(n, min(n, 20 * nz), replace=False)
            self._inducing = self._xs[sub[maxminOrder(coords[sub])[:nz]]]
            return
        ordering = self.options["ordering"]
        if ordering == "auto":
            ordering = "maxmin" if n <= self.options["maxminLimit"] else "random"
        self._order = maxminOrder(coords, self.options["seed"]) if ordering == "maxmin" else \
            np.random.default_rng(self.options["seed"]).permutation(n)
        m = min(self.options["neighbors"], n - 1)
        self._nbr = previousNeighbors(coords[self._order], m) if m > 0 else np.zeros((n, 0), dtype=np.int64)

    # ------------------------------------------------------------------ Vecchia likelihood
    def _vecchiaTerms(self, p):
        """Whitened responses and trend rows, and sum log r_i, for parameters p."""
        kp, eta = self._split(p)
        order = self._order
        xs = self._xs[order][:, self._sc]
        y = self._ys[order]
        f = self._f[order]
        n, m = self._nbr.shape
        q = f.shape[1]
        yt, ft, logR = np.empty(n), np.empty((n, q)), 0.0
        k = self._kernel
        # points with an incomplete conditioning set (i < m)
        for i in range(min(m, n)):
            nb = self._nbr[i, :i]
            idx = np.concatenate([nb, [i]])
            c = k.matrix(xs[idx], xs[idx], kp) + eta * np.eye(idx.size)
            if i == 0:
                r, b = c[0, 0], np.zeros(0)
            else:
                b = np.linalg.solve(c[:i, :i], c[:i, i])
                r = c[i, i] - c[:i, i] @ b
            r = max(r, 1e-12)
            sr = np.sqrt(r)
            yt[i] = (y[i] - b @ y[nb]) / sr
            ft[i] = (f[i] - b @ f[nb]) / sr
            logR += np.log(r)
        eye = eta * np.eye(m + 1)
        chunk = self.options["chunkSize"]
        for s in range(m, n, chunk):
            e = min(n, s + chunk)
            rows = np.arange(s, e)
            idx = np.hstack([self._nbr[s:e], rows[:, None]])            # (B, m+1), point last
            pts = xs[idx]
            c = k.batchMatrix(pts, pts, kp) + eye
            a, cv = c[:, :m, :m], c[:, :m, m]
            b = np.linalg.solve(a, cv[:, :, None])[:, :, 0]
            r = np.maximum(c[:, m, m] - np.sum(cv * b, axis=1), 1e-12)
            sr = np.sqrt(r)
            nb = self._nbr[s:e]
            yt[s:e] = (y[s:e] - np.sum(b * y[nb], axis=1)) / sr
            if q:
                ft[s:e] = (f[s:e] - np.einsum("bm,bmq->bq", b, f[nb])) / sr[:, None]
            logR += float(np.sum(np.log(r)))
        return yt, ft, logR

    # ------------------------------------------------------------------ FITC likelihood
    def _fitcParts(self, p):
        kp, eta = self._split(p)
        k = self._kernel
        xs = self._xs[:, self._sc]
        z = self._inducing[:, self._sc]
        kzz = k.matrix(z, z, kp)
        kzz = kzz + 1e-8 * np.eye(z.shape[0])
        lz = np.linalg.cholesky(kzz)
        kzn = k.matrix(z, xs, kp)
        v = np.linalg.solve(lz, kzn)                                    # Lz^-1 Kzn
        kdiag = np.diag(k.matrix(xs[:1], xs[:1], kp))[0]
        lam = np.maximum(kdiag - np.sum(v * v, axis=0), 0.0) + eta      # FITC diagonal
        vl = v / np.sqrt(lam)
        inner = np.eye(z.shape[0]) + vl @ vl.T
        li = np.linalg.cholesky(inner)
        return lz, v, lam, li

    def _fitcSolve(self, parts, rhs):
        """C^-1 rhs by the Woodbury identity."""
        lz, v, lam, li = parts
        r = rhs / (lam if rhs.ndim == 1 else lam[:, None])
        t = np.linalg.solve(li, v @ r)
        t = np.linalg.solve(li.T, t)
        corr = v.T @ t
        return r - corr / (lam if rhs.ndim == 1 else lam[:, None])

    # ------------------------------------------------------------------ profile likelihood
    def _profile(self, p):
        """(negative log-likelihood, beta, sigma2) with beta and sigma2 profiled out."""
        n = self._xs.shape[0]
        q = self._f.shape[1]
        reml = self.options["likelihood"] == "reml"
        if self.options["approximation"] == "vecchia":
            yt, ft, logDet = self._vecchiaTerms(p)
            if q:
                beta, *_ = np.linalg.lstsq(ft, yt, rcond=None)
                resid = yt - ft @ beta
                logDetA = float(np.linalg.slogdet(ft.T @ ft)[1])
            else:
                beta, resid, logDetA = np.zeros(0), yt, 0.0
            rss = float(resid @ resid)
        else:
            parts = self._fitcParts(p)
            lz, v, lam, li = parts
            logDet = float(np.sum(np.log(lam)) + 2.0 * np.sum(np.log(np.diag(li))))
            y, f = self._ys, self._f
            cy = self._fitcSolve(parts, y)
            if q:
                cf = self._fitcSolve(parts, f)
                a = f.T @ cf
                beta = np.linalg.solve(a, f.T @ cy)
                logDetA = float(np.linalg.slogdet(a)[1])
            else:
                beta, logDetA = np.zeros(0), 0.0
            resid = y - f @ beta
            rss = float(resid @ self._fitcSolve(parts, resid))
        dof = n - q if reml else n
        sigma2 = max(rss / dof, 1e-300)
        nll = 0.5 * (dof * np.log(sigma2) + logDet + (logDetA if reml else 0.0))
        return nll, beta, sigma2

    def _nll(self, p) -> float:
        try:
            return float(self._profile(np.asarray(p, dtype=float))[0])
        except np.linalg.LinAlgError:
            return 1e20

    # ------------------------------------------------------------------ training
    def _train(self) -> None:
        self._prepare()
        self._setGeometry()
        fixed = self.options["hyperparameters"]
        if fixed is not None:
            p = np.asarray(fixed, dtype=float)
            if p.size != self._kernel.nParams() + self._autoNugget:
                raise ValueError("wrong number of hyperparameters")
            self._optResult = None
        else:
            p = self._optimize(None)
            if self.options["refineNeighbors"] and self.options["approximation"] == "vecchia" and \
                    len(self._kernel.lengthscaleIndices()) > 1:
                self._setGeometry(self._split(p)[0])
                p = self._optimize(p)
        self._finalize(p)

    def _optimize(self, start):
        bounds = self._bounds()
        p0 = np.concatenate([self._kernel.initialParams(), [np.log(1e-2)] if self._autoNugget else []])
        n = self._xs.shape[0]
        if start is None and n > self.options["subsampleSize"]:
            start = self._subsampleStart()
        starts = [np.clip(p0 if start is None else start, bounds[:, 0], bounds[:, 1])]
        if start is None and self.options["nStart"] > 1:
            mid = bounds.mean(axis=1)
            half = 0.3 * (bounds[:, 1] - bounds[:, 0])
            box = np.column_stack([np.maximum(bounds[:, 0], mid - half), np.minimum(bounds[:, 1], mid + half)])
            starts.extend(latinHypercube(self.options["nStart"] - 1, box, criterion="maximin", seed=self.options["seed"]))
        pairs = [tuple(b) for b in bounds]
        self._optResult = multiStart(
            lambda s: minimize(self._nll, s, bounds=pairs, maxIter=self.options["maxIter"], tol=1e-6),
            np.array(starts))
        return self._optResult.x

    def _subsampleStart(self):
        """Exact Kriging fit on a random subsample as the optimizer start.

        The subsample is given the already standardized data with normalize=False,
        so its log-parameters mean the same as this model's.
        """
        n = self._xs.shape[0]
        idx = np.random.default_rng(self.options["seed"]).choice(n, self.options["subsampleSize"], replace=False)
        opts = {k: v for k, v in self.options.toDict().items() if k in KrigingModel().options.toDict()}
        opts.update(normalize=False, hyperparameters=None, nStart=2, likelihood=self.options["likelihood"])
        sub = KrigingModel(**opts)
        sub.setTrainingValues(self._xs[idx], self._ys[idx])
        try:
            sub.train()
        except (ValueError, np.linalg.LinAlgError):
            return None
        return sub._params

    def _finalize(self, p) -> None:
        p = np.asarray(p, dtype=float)
        self._params = p
        self._kp, self._eta = self._split(p)
        nll, self._beta, self._sigma2 = self._profile(p)
        self._logLik = -nll
        self._search = NeighborSearch(self._metricCoords(self._xs, self._kp)) \
            if self.options["approximation"] == "vecchia" else None
        self._fitc = self._fitcParts(p) if self.options["approximation"] == "fitc" else None
        if self._fitc is not None:
            resid = self._ys - self._f @ self._beta
            self._fitcAlpha = self._fitcSolve(self._fitc, resid)

    # ------------------------------------------------------------------ prediction
    def _predictParts(self, x: np.ndarray):
        """Latent mean (standardized) and latent variance / sigma2 at x."""
        xs = self._scaled(x)
        kp, eta = self._kp, self._eta
        k = self._kernel
        trend = self._trendMatrix(xs) @ self._beta if self._f.shape[1] else np.zeros(x.shape[0])
        if self.options["approximation"] == "fitc":
            lz, v, lam, li = self._fitc
            z = self._inducing[:, self._sc]
            kzx = k.matrix(z, xs[:, self._sc], kp)
            vx = np.linalg.solve(lz, kzx)                                # Lz^-1 K_zx
            mean = trend + vx.T @ (v @ self._fitcAlpha)                   # Q_xn C^-1 (y - F beta)
            # predictive latent variance of FITC: k - Q + Q_xn C^-1 Q_nx
            w = np.linalg.solve(li, vx)
            var = np.diag(k.matrix(xs[:1, self._sc], xs[:1, self._sc], kp))[0] - np.sum(vx * vx, axis=0) \
                + np.sum(w * w, axis=0)
            return mean, np.maximum(var, 0.0)
        m = min(self.options["predictionNeighbors"] or 3 * self.options["neighbors"], self._xs.shape[0])
        _, nb = self._search.query(self._metricCoords(xs, kp), m)
        xsc = self._xs[:, self._sc]
        mean = np.empty(x.shape[0])
        var = np.empty(x.shape[0])
        resid = self._ys - (self._f @ self._beta if self._f.shape[1] else 0.0)
        chunk = self.options["chunkSize"]
        for s in range(0, x.shape[0], chunk):
            e = min(x.shape[0], s + chunk)
            pts = np.concatenate([xsc[nb[s:e]], xs[s:e, self._sc][:, None, :]], axis=1)
            c = k.batchMatrix(pts, pts, kp)
            a = c[:, :m, :m] + eta * np.eye(m)
            cv = c[:, :m, m]
            b = np.linalg.solve(a, cv[:, :, None])[:, :, 0]
            mean[s:e] = np.sum(b * resid[nb[s:e]], axis=1)
            var[s:e] = c[:, m, m] - np.sum(cv * b, axis=1)
        return trend + mean, np.maximum(var, 0.0)

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        mean, _ = self._predictParts(x)
        return (self._yMean + self._yStd * mean)[:, None]

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        _, var = self._predictParts(x)
        var = var * self._sigma2
        if kind == "prediction":
            var = var + self._sigma2 * self._eta
        return (var * self._yStd ** 2)[:, None]

    def _predictDerivatives(self, x: np.ndarray, kx: int) -> np.ndarray:
        h = 1e-5 * max(1.0, float(self._xStd[kx]))
        xp, xm = x.copy(), x.copy()
        xp[:, kx] += h
        xm[:, kx] -= h
        return (self._predictValues(xp) - self._predictValues(xm)) / (2.0 * h)

    def _effectiveParams(self):
        return float(self._f.shape[1])

    # ------------------------------------------------------------------ reporting / unsupported dense features
    def spatialSummary(self) -> dict:
        self._checkTrained()
        out = {"lambda": self._eta, "sigma2": self._sigma2 * self._yStd ** 2,
               "tau": float(np.sqrt(self._eta * self._sigma2)) * self._yStd,
               "logLikelihood": self._logLik, "n": int(self._xs.shape[0]),
               "approximation": self.options["approximation"]}
        report = self._paramReport(self._params)
        lengths = [v for k, v in report.items() if "engthscale" in k]
        if lengths:
            out["lengthscales"] = lengths if len(lengths) > 1 else lengths[0]
        if "nu" in report:
            out["nu"] = report["nu"]
        return out

    @property
    def hyperparameters(self) -> dict:
        self._checkTrained()
        names = self._kernel.paramNames() + (["log_nugget"] if self._autoNugget else [])
        return {"logParams": dict(zip(names, self._params.tolist())), "processVariance": self._sigma2 * self._yStd ** 2,
                "nugget": self._eta, "logLikelihood": self._logLik, "trendCoefficients": self._beta.tolist(),
                "spatial": self.spatialSummary()}

    def _unsupported(self, *args, **kwargs):
        raise NotImplementedError("not available for the scalable (approximate) Kriging model; use KrigingModel")

    parameterIntervals = looDiagnostics = looResiduals = _unsupported

    def summary(self) -> str:
        text = super(KrigingModel, self).summary()
        rows = [f"  {k:>16}: {v}" for k, v in self.spatialSummary().items()]
        return text + "\n\nCovariance model (" + self.options["approximation"] + "):\n" + "\n".join(rows)

    # ------------------------------------------------------------------ persistence
    def _stateToDict(self) -> dict:
        geo = self._geometryParams
        return {"params": self._params.tolist(), "x": self.xt.tolist(), "y": self.yt[:, 0].tolist(),
                "geometryParams": None if geo is None else geo.tolist()}

    def _stateFromDict(self, state: dict) -> None:
        self.xt = np.array(state["x"], dtype=float).reshape(-1, self.nx)
        self.yt = np.array(state["y"], dtype=float)[:, None]
        self.wt = np.ones(self.xt.shape[0])
        self._prepare()
        params = np.array(state["params"], dtype=float)
        geo = state.get("geometryParams")
        self._setGeometry(None if geo is None else np.array(geo, dtype=float))
        self._optResult = None
        self._finalize(params)
