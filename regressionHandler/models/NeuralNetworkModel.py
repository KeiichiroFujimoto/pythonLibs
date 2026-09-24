"""Multilayer perceptron regression (numpy only), with deep-ensemble uncertainty.

    yhat = W_L sigma(... sigma(x W_1 + b_1) ...) + b_L      (inputs and outputs standardized)

Training minimizes 0.5 mean ||yhat - y||^2 + 0.5 weightDecay |W|^2 / n with
exact back-propagated gradients:

    solver="lbfgs"   full-batch projected L-BFGS (smooth, small / medium data; default up to 5000 rows)
    solver="adam"    mini-batch Adam with early stopping on a validation split (large data)

``nNetworks > 1`` trains an ensemble from different initializations; its
spread is the confidence variance and the out-of-sample residual variance is
added for ``kind="prediction"`` (Lakshminarayanan et al. 2017). Input
gradients are exact (forward-mode tangents), so ``predictGradient`` is cheap
and smooth for tanh / softplus activations.

References:
    Rumelhart, Hinton & Williams (1986) Nature 323.
    Kingma & Ba (2015) ICLR (Adam).
    Glorot & Bengio (2010) AISTATS (initialization).
"""
from __future__ import annotations

import numpy as np

from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase
from pythonLibs.regressionHandler.numerics.Optimizers import minimize

_ACT = ("tanh", "relu", "softplus", "sigmoid")


def _act(name, z):
    if name == "tanh":
        return np.tanh(z)
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "softplus":
        return np.logaddexp(0.0, z)
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _dact(name, z, a):
    if name == "tanh":
        return 1.0 - a * a
    if name == "relu":
        return (z > 0).astype(float)
    if name == "softplus":
        return 0.5 * (1.0 + np.tanh(0.5 * z))
    return a * (1.0 - a)


class _Net:
    """Weights of one network as a flat vector with (W, b) views."""

    def __init__(self, sizes: list[int], activation: str) -> None:
        self.sizes = sizes
        self.activation = activation
        self.shapes = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.nParams = sum(a * b + b for a, b in self.shapes)

    def init(self, rng) -> np.ndarray:
        parts = []
        for i, (a, b) in enumerate(self.shapes):
            last = i == len(self.shapes) - 1
            scale = np.sqrt(2.0 / a) if self.activation == "relu" and not last else np.sqrt(6.0 / (a + b))
            w = rng.normal(0.0, scale, (a, b)) if self.activation == "relu" and not last else \
                rng.uniform(-scale, scale, (a, b))
            parts += [w.ravel(), np.zeros(b)]
        return np.concatenate(parts)

    def unpack(self, theta):
        out, k = [], 0
        for a, b in self.shapes:
            w = theta[k:k + a * b].reshape(a, b)
            k += a * b
            out.append((w, theta[k:k + b]))
            k += b
        return out

    def forward(self, theta, x):
        layers = self.unpack(theta)
        acts, pre = [x], []
        a = x
        for i, (w, b) in enumerate(layers):
            z = a @ w + b
            pre.append(z)
            a = z if i == len(layers) - 1 else _act(self.activation, z)
            acts.append(a)
        return acts, pre

    def lossGrad(self, theta, x, y, weightDecay: float, nTotal: int):
        acts, pre = self.forward(theta, x)
        layers = self.unpack(theta)
        m = x.shape[0]
        diff = acts[-1] - y
        loss = 0.5 * float(np.sum(diff * diff)) / m
        wd = weightDecay / nTotal
        loss += 0.5 * wd * sum(float(np.sum(w * w)) for w, _ in layers)
        grads = []
        delta = diff / m
        for i in range(len(layers) - 1, -1, -1):
            w, _ = layers[i]
            gw = acts[i].T @ delta + wd * w
            gb = delta.sum(axis=0)
            grads.append((gw, gb))
            if i:
                delta = (delta @ w.T) * _dact(self.activation, pre[i - 1], acts[i])
        grads.reverse()
        return loss, np.concatenate([np.concatenate([gw.ravel(), gb]) for gw, gb in grads])

    def predict(self, theta, x):
        return self.forward(theta, x)[0][-1]

    def tangent(self, theta, x, direction):
        """d output / d x along ``direction`` (nx,), forward mode: (m, ny)."""
        layers = self.unpack(theta)
        a, t = x, np.broadcast_to(direction, x.shape).astype(float)
        for i, (w, b) in enumerate(layers):
            z = a @ w + b
            dz = t @ w
            if i == len(layers) - 1:
                return dz
            a = _act(self.activation, z)
            t = dz * _dact(self.activation, z, a)
        return t


@registry("model").register("neuralNetwork")
class NeuralNetworkModel(SurrogateModelBase):
    """Multilayer perceptron (tanh / relu / softplus), L-BFGS or Adam, optional deep ensemble."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("hidden", [32, 32], types=list, desc="Hidden layer widths")
        d("activation", "tanh", values=_ACT, desc="Hidden activation")
        d("solver", "auto", values=("auto", "lbfgs", "adam"), desc="Optimizer (auto: lbfgs up to 5000 rows)")
        d("weightDecay", 1e-4, types=(int, float), lower=0.0, desc="L2 penalty on weights")
        d("maxIter", 3000, types=int, lower=1, desc="L-BFGS iterations")
        d("epochs", 500, types=int, lower=1, desc="Adam epochs")
        d("batchSize", 128, types=int, lower=1, desc="Adam mini-batch size")
        d("learningRate", 1e-3, types=float, lower=0.0, desc="Adam step size")
        d("validationFraction", 0.1, types=float, lower=0.0, upper=0.9, desc="Adam early-stopping split")
        d("patience", 30, types=int, lower=1, desc="Adam epochs without validation improvement")
        d("nNetworks", 1, types=int, lower=1, desc="Ensemble size (> 1 enables predictive variances)")
        d("seed", 0, types=int, desc="Random seed")
        self.supports.update(multiOutput=True, derivatives=True, variances=False, weights=False)

    # ------------------------------------------------------------------ training
    def _train(self) -> None:
        x, y = self.xt, self.yt
        self._xMean, sx = x.mean(axis=0), x.std(axis=0)
        self._xStd = np.where(sx > 0, sx, 1.0)
        self._yMean, sy = y.mean(axis=0), y.std(axis=0)
        self._yStd = np.where(sy > 0, sy, 1.0)
        xs, ys = (x - self._xMean) / self._xStd, (y - self._yMean) / self._yStd
        self._net = _Net([self.nx] + [int(h) for h in self.options["hidden"]] + [self.ny], self.options["activation"])
        solver = self.options["solver"]
        if solver == "auto":
            solver = "lbfgs" if x.shape[0] <= 5000 else "adam"
        self._solver = solver
        self._thetas, self._histories = [], []
        for k in range(self.options["nNetworks"]):
            rng = np.random.default_rng(self.options["seed"] + 1000 * k)
            theta = self._net.init(rng)
            theta, hist = (self._lbfgs if solver == "lbfgs" else self._adam)(theta, xs, ys, rng)
            self._thetas.append(theta)
            self._histories.append(hist)
        resid = ys - self._ensemble(xs).mean(axis=0)
        self._residVar = np.mean(resid * resid, axis=0) * self._yStd ** 2
        self.supports["variances"] = self.options["nNetworks"] > 1

    def _lbfgs(self, theta, xs, ys, rng):
        n = xs.shape[0]
        res = minimize(lambda t: self._net.lossGrad(t, xs, ys, float(self.options["weightDecay"]), n), theta,
                       jac=True, maxIter=self.options["maxIter"], tol=1e-12)
        return res.x, {"loss": float(res.fun), "iterations": res.nit if hasattr(res, "nit") else None}

    def _adam(self, theta, xs, ys, rng):
        n = xs.shape[0]
        vf = self.options["validationFraction"]
        perm = rng.permutation(n)
        nVal = int(round(vf * n)) if vf > 0 and n >= 50 else 0
        val, tr = perm[:nVal], perm[nVal:]
        lr, b1, b2, eps = float(self.options["learningRate"]), 0.9, 0.999, 1e-8
        m1, m2 = np.zeros_like(theta), np.zeros_like(theta)
        step, best, bestTheta, stall, history = 0, np.inf, theta.copy(), 0, []
        bs = min(self.options["batchSize"], tr.size)
        wd = float(self.options["weightDecay"])
        for epoch in range(self.options["epochs"]):
            order = rng.permutation(tr)
            for s in range(0, order.size, bs):
                idx = order[s:s + bs]
                # the mini-batch objective 0.5 mean|r|^2 + 0.5 wd |W|^2 / n is an unbiased estimate of the full one
                _, g = self._net.lossGrad(theta, xs[idx], ys[idx], wd, tr.size)
                step += 1
                m1 = b1 * m1 + (1 - b1) * g
                m2 = b2 * m2 + (1 - b2) * g * g
                theta = theta - lr * (m1 / (1 - b1 ** step)) / (np.sqrt(m2 / (1 - b2 ** step)) + eps)
            if nVal:
                d = self._net.predict(theta, xs[val]) - ys[val]
                score = float(np.mean(d * d))
                history.append(score)
                if score < best:
                    best, bestTheta, stall = score, theta.copy(), 0
                else:
                    stall += 1
                    if stall >= self.options["patience"]:
                        break
        return (bestTheta if nVal else theta), {"validationLoss": history, "epochs": epoch + 1}

    # ------------------------------------------------------------------ prediction
    def _ensemble(self, xs) -> np.ndarray:
        return np.array([self._net.predict(t, xs) for t in self._thetas])

    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        xs = (x - self._xMean) / self._xStd
        return self._yMean + self._yStd * self._ensemble(xs).mean(axis=0)

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        xs = (x - self._xMean) / self._xStd
        preds = self._ensemble(xs) * self._yStd
        var = preds.var(axis=0, ddof=1) if preds.shape[0] > 1 else np.zeros(preds.shape[1:])
        if kind == "prediction":
            var = var + self._residVar
        return var

    def _predictDerivatives(self, x: np.ndarray, kx: int) -> np.ndarray:
        xs = (x - self._xMean) / self._xStd
        direction = np.zeros(self.nx)
        direction[kx] = 1.0 / self._xStd[kx]
        t = np.mean([self._net.tangent(th, xs, direction) for th in self._thetas], axis=0)
        return t * self._yStd

    def _effectiveParams(self):
        return float(self._net.nParams)

    @property
    def trainingHistory(self) -> dict:
        self._checkTrained()
        return {"solver": self._solver, "networks": self._histories}

    def _stateToDict(self) -> dict:
        return {"sizes": self._net.sizes, "thetas": [t.tolist() for t in self._thetas],
                "xMean": self._xMean.tolist(), "xStd": self._xStd.tolist(), "yMean": self._yMean.tolist(),
                "yStd": self._yStd.tolist(), "residVar": self._residVar.tolist(), "solver": self._solver}

    def _stateFromDict(self, state: dict) -> None:
        self._net = _Net(list(state["sizes"]), self.options["activation"])
        self._thetas = [np.array(t, dtype=float) for t in state["thetas"]]
        for k in ("xMean", "xStd", "yMean", "yStd", "residVar"):
            setattr(self, "_" + k, np.array(state[k], dtype=float))
        self._solver = state["solver"]
        self._histories = []
        self.supports["variances"] = len(self._thetas) > 1
