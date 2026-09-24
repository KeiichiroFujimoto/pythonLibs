"""Nonlinear least-squares regression.

The model function is a safe text expression over input variables and
parameters (serializable), or a Python callable ``f(x, *params)`` (not
serializable). Fitting uses the package's Levenberg-Marquardt with bounds,
robust losses and optional multi-start; parameter inference follows
the usual asymptotic form: Cov = s^2 (J^T W J)^-1 with
s^2 = SSE_w / (n - p). Confidence / prediction bands use the delta method.

Common model forms with automatic initial guesses live in
``ModelLibrary`` (``NonlinearModel(library="exponentialDecay")``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from pythonLibs.regressionHandler.core.FitResult import FitResult
from pythonLibs.regressionHandler.core.Registry import registry
from pythonLibs.regressionHandler.core.SafeExpression import SafeExpression
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase
from pythonLibs.regressionHandler.numerics.LinearAlgebra import leastSquaresSvd
from pythonLibs.regressionHandler.numerics.Optimizers import LOSSES, leastSquares


def _freeNames(text: str) -> set[str]:
    """Names referenced in an expression other than whitelisted functions / constants."""
    import ast
    from pythonLibs.regressionHandler.core.SafeExpression import _CONSTANTS, _FUNCTIONS
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression {text!r}: {exc.msg}") from None
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - set(_FUNCTIONS) - set(_CONSTANTS)


@registry("model").register("nonlinear")
class NonlinearModel(SurrogateModelBase):
    """Nonlinear least squares on an expression or callable, with parameter inference."""

    def _initialize(self) -> None:
        d = self.options.declare
        d("expression", None, types=str, desc="Model expression, e.g. 'a*exp(-b/x) + c' (safe subset)")
        d("function", None, types=object, desc="Python callable f(x, *params) instead of an expression "
                                                "(x is (n,) for one input, (n, nx) otherwise); not serializable")
        d("params", None, types=list, desc="Parameter names in order")
        d("variables", None, types=list, desc="Input variable names used by the expression (default x or x0..)")
        d("constants", {}, types=dict, desc="Named constants available to the expression, e.g. {'k': 2.0}")
        d("p0", None, types=(list, dict), desc="Initial parameter values (list or {name: value})")
        d("bounds", None, types=(list, dict), desc="Parameter bounds: {name: [lo, hi]} or [[lo, hi], ...]")
        d("loss", "linear", values=LOSSES, desc="Robust loss: linear, soft_l1, huber, cauchy or arctan")
        d("fScale", 1.0, types=(int, float), lower=0.0, desc="Inlier residual scale for robust losses")
        d("nStart", 1, types=int, lower=1, desc="Number of starts (random log-perturbations of p0)")
        d("seed", 0, types=int, desc="Seed for multi-start perturbations")
        d("library", None, types=str, desc="Name of a ModelLibrary form (fills expression, params, guesses)")
        self.supports.update(multiOutput=False, variances=True, derivatives=False, parameterInference=True)

    def _validateOptions(self) -> None:
        if self.options["library"]:
            from pythonLibs.regressionHandler.models.ModelLibrary import libraryEntry
            entry = libraryEntry(self.options["library"])
            if self.options["expression"] is None and self.options["function"] is None:
                self.options["expression"] = entry["expression"]
            if self.options["params"] is None:
                self.options["params"] = list(entry["params"])
            if self.options["variables"] is None:
                self.options["variables"] = list(entry.get("variables", ["x"]))
            if not self.options["constants"] and entry.get("constants"):
                self.options["constants"] = dict(entry["constants"])
        if self.options["expression"] is None and self.options["function"] is None:
            raise ValueError("NonlinearModel needs an expression, a function or a library form")
        if not self.options["params"]:
            raise ValueError("NonlinearModel needs parameter names ('params')")
        if self.options["expression"] is not None:
            variables = self.options["variables"]
            if variables is None:
                known = set(self.options["params"]) | set(self.options["constants"])
                variables = sorted(_freeNames(self.options["expression"]) - known)
            self._compile(variables)

    # ------------------------------------------------------------------ evaluation
    def _variableNames(self) -> list[str]:
        v = self.options["variables"]
        if v is not None:
            if len(v) != self.nx:
                raise ValueError(f"{len(v)} variables declared for {self.nx} inputs")
            return list(v)
        return ["x"] if self.nx == 1 else [f"x{i}" for i in range(self.nx)]

    def _compile(self, variables: list[str]) -> None:
        symbols = list(variables) + list(self.options["params"]) + list(self.options["constants"])
        self._expr = SafeExpression(self.options["expression"], symbols)
        unused = set(self.options["params"]) - self._expr.usedSymbols
        if unused:
            raise ValueError(f"parameters not used in the expression: {sorted(unused)}")

    def _evaluate(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        if self.options["function"] is not None:
            xa = x[:, 0] if x.shape[1] == 1 else x
            out = np.asarray(self.options["function"](xa, *p), dtype=float)
        else:
            env = dict(self.options["constants"])
            env.update(zip(self.options["params"], p))
            env.update({n: x[:, i] for i, n in enumerate(self._varNames)})
            out = np.asarray(self._expr.evaluate(env), dtype=float)
        return np.broadcast_to(out, (x.shape[0],)).astype(float)

    def _paramJacobian(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """d f / d p (n, nParams) by central differences."""
        jac = np.empty((x.shape[0], p.size))
        for j in range(p.size):
            h = 1e-6 * max(abs(p[j]), 1e-8)
            pp, pm = p.copy(), p.copy()
            pp[j] += h
            pm[j] -= h
            jac[:, j] = (self._evaluate(x, pp) - self._evaluate(x, pm)) / (2.0 * h)
        return jac

    # ------------------------------------------------------------------ training
    def _vector(self, spec, default: float) -> np.ndarray:
        names = self.options["params"]
        if spec is None:
            return np.full(len(names), default)
        if isinstance(spec, dict):
            return np.array([float(spec.get(n, default)) for n in names])
        arr = np.asarray(spec, dtype=float)
        if arr.size != len(names):
            raise ValueError(f"expected {len(names)} values, got {arr.size}")
        return arr

    def _boundsArrays(self) -> tuple[np.ndarray, np.ndarray]:
        names = self.options["params"]
        b = self.options["bounds"]
        lb, ub = np.full(len(names), -np.inf), np.full(len(names), np.inf)
        if b is None:
            return lb, ub
        items = b.items() if isinstance(b, dict) else zip(names, b)
        for name, pair in items:
            if name not in names:
                raise ValueError(f"bounds given for unknown parameter {name!r}")
            i = names.index(name)
            lo, hi = pair
            lb[i] = -np.inf if lo is None else float(lo)
            ub[i] = np.inf if hi is None else float(hi)
        return lb, ub

    def _initialGuess(self, x, y, lb, ub) -> np.ndarray:
        if self.options["p0"] is not None:
            return self._vector(self.options["p0"], 1.0)
        if self.options["library"]:
            from pythonLibs.regressionHandler.models.ModelLibrary import libraryEntry
            guess = libraryEntry(self.options["library"])["guess"](x[:, 0] if x.shape[1] == 1 else x, y)
            return np.asarray(guess, dtype=float)
        p = np.ones(len(self.options["params"]))
        return np.clip(p, np.where(np.isfinite(lb), lb, -np.inf), np.where(np.isfinite(ub), ub, np.inf))

    def _train(self) -> None:
        x, y, w = self.xt, self.yt[:, 0], self.wt
        self._varNames = self._variableNames()
        if self.options["expression"] is not None and self.options["function"] is None:
            self._compile(self._varNames)
        lb, ub = self._boundsArrays()
        p0 = np.clip(self._initialGuess(x, y, lb, ub), lb, ub)
        sw = np.sqrt(w)

        def resid(p):
            return (self._evaluate(x, p) - y) * sw

        if not np.all(np.isfinite(resid(p0))):
            raise ValueError("model is not finite at the initial guess; provide p0 or bounds")
        starts = [p0]
        rng = np.random.default_rng(self.options["seed"])
        for _ in range(self.options["nStart"] - 1):
            scale = np.where(p0 != 0.0, np.abs(p0), 1.0)
            cand = p0 + scale * rng.normal(0.0, 1.0, p0.size) * np.exp(rng.uniform(-2, 1, p0.size))
            starts.append(np.clip(cand, lb, ub))
        best = None
        for s in starts:
            try:
                res = leastSquares(resid, s, bounds=(lb, ub), loss=self.options["loss"], fScale=self.options["fScale"],
                                   ftol=1e-12, xtol=1e-12, gtol=1e-12)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if best is None or res.cost < best.cost:
                best = res
        if best is None:
            raise RuntimeError("nonlinear fit failed from every start")
        self._opt = best
        self._p = best.x.copy()
        self._inference(x, y, w)

    def _inference(self, x, y, w) -> None:
        n, k = x.shape[0], self._p.size
        jac = self._paramJacobian(x, self._p) * np.sqrt(w)[:, None]
        sol = leastSquaresSvd(jac, np.zeros(n))
        self._rank = sol.rank
        self._covUnscaled = sol.inverseGram()
        r = y - self._evaluate(x, self._p)
        sse = float(np.sum(w * r * r))
        self._dof = n - k
        self._sigma2 = sse / self._dof if self._dof > 0 else float("nan")
        cov = self._covUnscaled * self._sigma2
        self.result = FitResult(parameterNames=list(self.options["params"]), params=self._p.reshape(-1, 1),
                                covariance=cov[None], dofResid=float(self._dof), sigma2=np.array([self._sigma2]),
                                outputNames=self.outputNames)

    # ------------------------------------------------------------------ prediction
    def _predictValues(self, x: np.ndarray) -> np.ndarray:
        return self._evaluate(x, self._p)[:, None]

    def _predictVariances(self, x: np.ndarray, kind: str) -> np.ndarray:
        g = self._paramJacobian(x, self._p)
        var = np.einsum("ij,jk,ik->i", g, self._covUnscaled, g) * self._sigma2
        if kind == "prediction":
            var = var + self._sigma2
        return var[:, None]

    def _effectiveParams(self):
        return float(self._p.size)

    def _intervalDof(self) -> Optional[float]:
        return float(self._dof) if self._dof > 0 else None

    @property
    def parameters(self) -> dict:
        self._checkTrained()
        if self._subModels is not None:
            return {"outputs": [m.parameters for m in self._subModels]}
        return dict(zip(self.options["params"], self._p.tolist()))

    @property
    def optimizer(self):
        return self._opt

    def equation(self, output: int = 0, precision: int = 6) -> str:
        self._checkTrained()
        if self._subModels is not None:
            return self._subModels[output].equation(0, precision)
        body = self.options["expression"] or getattr(self.options["function"], "__name__", "f")
        values = ", ".join(f"{n}={v:.{precision}g}" for n, v in zip(self.options["params"], self._p))
        return f"{self.outputNames[output]} = {body}   [{values}]"

    # ------------------------------------------------------------------ persistence
    def _stateToDict(self) -> dict:
        if self.options["function"] is not None:
            raise NotImplementedError("models built from Python callables cannot be serialized; use an expression")
        return {"p": self._p.tolist(), "covUnscaled": self._covUnscaled.tolist(), "sigma2": self._sigma2,
                "dof": self._dof, "varNames": self._varNames, "rank": self._rank}

    def _stateFromDict(self, state: dict) -> None:
        self._p = np.array(state["p"], dtype=float)
        self._covUnscaled = np.array(state["covUnscaled"], dtype=float)
        self._sigma2 = float(state["sigma2"])
        self._dof = int(state["dof"])
        self._rank = int(state["rank"])
        self._varNames = list(state["varNames"])
        self._compile(self._varNames)
        self._opt = None
