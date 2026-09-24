"""Linear functionals of a regression function, used as physical constraints.

A constraint is

    sum_terms  sum_j  w_j  D^alpha f_o(x_j)   {=, >=, <=}   value

i.e. a weighted sum of values (or derivatives D^alpha, alpha a list of input
indices) of output o at points x_j. This covers

    point values            f(x) = v                 (boundary / symmetry / known states)
    derivatives             df/dx_k (x) = v          (fluxes, zero-gradient conditions)
    integrals and means     sum_j w_j f(x_j) = Q     (quadrature: conservation of a total)
    output balances         sum_o a_o f_o(x) = v     (fractions summing to one, energy balance)
    bounds and shape        f(x) >= 0, df/dx >= 0, d2f/dx2 >= 0 on a set of points

Derivatives are evaluated with central-difference stencils (step relative to
the input range) unless the model supplies exact derivatives (linear-basis
models). Builders return lists of ``LinearConstraint``; specs are plain dicts
for JSON / toolBase use::

    {"type": "integral", "value": 2.0, "box": {"lower": [0], "upper": [1]}, "n": 16}
    {"type": "value", "points": [[0.0]], "values": [0.0]}
    {"type": "derivative", "points": [[1.0]], "kx": 0, "values": [0.0]}
    {"type": "bound", "points": grid, "lower": 0.0}
    {"type": "monotone", "points": grid, "kx": 0, "increasing": true}
    {"type": "convex", "points": grid, "kx": 0}
    {"type": "outputSum", "points": grid, "coefficients": [1, 1, 1], "value": 1.0}
    {"type": "general", "kind": "equal", "value": 3.0, "terms": [{"output": 0, "points": [...], "weights": [...],
                                                                   "derivative": [0]}]}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

KINDS = ("equal", "lower", "upper")


@dataclass
class Term:
    """sum_j weights_j * D^derivative f_output(points_j)."""
    points: np.ndarray
    weights: np.ndarray
    output: int = 0
    derivative: tuple = ()

    def __post_init__(self) -> None:
        self.points = np.atleast_2d(np.asarray(self.points, dtype=float))
        self.weights = np.broadcast_to(np.asarray(self.weights, dtype=float), (self.points.shape[0],)).copy()
        self.output = int(self.output)
        self.derivative = tuple(int(k) for k in (self.derivative or ()))

    def toDict(self) -> dict:
        return {"points": self.points.tolist(), "weights": self.weights.tolist(), "output": self.output,
                "derivative": list(self.derivative)}


@dataclass
class LinearConstraint:
    """sum of terms  (kind)  value, kind in {"equal", "lower" (>=), "upper" (<=)}."""
    terms: list
    value: float
    kind: str = "equal"
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"constraint kind must be one of {KINDS}")
        self.terms = [t if isinstance(t, Term) else Term(**t) for t in self.terms]
        self.value = float(self.value)

    def toDict(self) -> dict:
        return {"type": "general", "kind": self.kind, "value": self.value, "name": self.name,
                "terms": [t.toDict() for t in self.terms]}


# ---------------------------------------------------------------- quadrature
def gaussLegendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights on [-1, 1] (Golub-Welsch)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    k = np.arange(1, n)
    beta = k / np.sqrt(4.0 * k * k - 1.0)
    w, v = np.linalg.eigh(np.diag(beta, 1) + np.diag(beta, -1))
    return w, 2.0 * v[0] ** 2


def boxQuadrature(lower, upper, n=8) -> tuple[np.ndarray, np.ndarray]:
    """Tensor Gauss-Legendre points and weights on a box (exact for degree 2n-1 per input)."""
    lo = np.atleast_1d(np.asarray(lower, dtype=float))
    hi = np.atleast_1d(np.asarray(upper, dtype=float))
    if lo.shape != hi.shape or np.any(hi < lo):
        raise ValueError("box needs lower <= upper with one entry per input")
    ns = np.broadcast_to(np.asarray(n, dtype=int), lo.shape)
    axes, wts = [], []
    for a, b, k in zip(lo, hi, ns):
        t, w = gaussLegendre(int(k))
        axes.append(0.5 * (a + b) + 0.5 * (b - a) * t)
        wts.append(0.5 * (b - a) * w)
    grids = np.meshgrid(*axes, indexing="ij")
    wgrid = np.meshgrid(*wts, indexing="ij")
    return np.column_stack([g.ravel() for g in grids]), np.prod([g.ravel() for g in wgrid], axis=0)


# ---------------------------------------------------------------- builders
def _pts(points) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    return p[:, None] if p.ndim == 1 else np.atleast_2d(p)


def valueConstraints(points, values, output: int = 0, kind: str = "equal") -> list[LinearConstraint]:
    """f_output(x_i) (kind) values_i, one constraint per point."""
    p = _pts(points)
    v = np.broadcast_to(np.asarray(values, dtype=float), (p.shape[0],))
    return [LinearConstraint([Term(p[i:i + 1], [1.0], output)], v[i], kind, f"value[{i}]") for i in range(p.shape[0])]


def derivativeConstraints(points, kx, values=0.0, output: int = 0, kind: str = "equal",
                          order: int = 1) -> list[LinearConstraint]:
    """D^order f / dx_kx (x_i) (kind) values_i."""
    p = _pts(points)
    v = np.broadcast_to(np.asarray(values, dtype=float), (p.shape[0],))
    der = (int(kx),) * int(order)
    return [LinearConstraint([Term(p[i:i + 1], [1.0], output, der)], v[i], kind, f"d{order}/dx{kx}[{i}]")
            for i in range(p.shape[0])]


def integralConstraint(value, box=None, points=None, weights=None, n=8, output: int = 0, kind: str = "equal",
                       mean: bool = False) -> list[LinearConstraint]:
    """Integral (or mean when ``mean``) of f_output over a box (Gauss-Legendre) or given quadrature."""
    if box is not None:
        p, w = boxQuadrature(box["lower"], box["upper"], box.get("n", n))
    elif points is not None and weights is not None:
        p, w = _pts(points), np.asarray(weights, dtype=float)
    else:
        raise ValueError("give a box or quadrature points and weights")
    if mean:
        w = w / np.sum(w)
    return [LinearConstraint([Term(p, w, output)], value, kind, "mean" if mean else "integral")]


def boundConstraints(points, lower=None, upper=None, output: int = 0) -> list[LinearConstraint]:
    """lower <= f_output(x_i) <= upper at every point."""
    out = []
    if lower is not None:
        out += valueConstraints(points, lower, output, "lower")
    if upper is not None:
        out += valueConstraints(points, upper, output, "upper")
    return out


def monotoneConstraints(points, kx, increasing: bool = True, output: int = 0) -> list[LinearConstraint]:
    return derivativeConstraints(points, kx, 0.0, output, "lower" if increasing else "upper")


def convexConstraints(points, kx, convex: bool = True, output: int = 0) -> list[LinearConstraint]:
    return derivativeConstraints(points, kx, 0.0, output, "lower" if convex else "upper", order=2)


def outputSumConstraints(points, coefficients, value, kind: str = "equal") -> list[LinearConstraint]:
    """sum_o a_o f_o(x_i) (kind) value at every point (balances across outputs)."""
    p = _pts(points)
    a = np.asarray(coefficients, dtype=float)
    v = np.broadcast_to(np.asarray(value, dtype=float), (p.shape[0],))
    return [LinearConstraint([Term(p[i:i + 1], [a[o]], o) for o in range(a.size) if a[o] != 0.0], v[i], kind,
                             f"outputSum[{i}]") for i in range(p.shape[0])]


_BUILDERS = {
    "value": lambda s: valueConstraints(s["points"], s["values"], s.get("output", 0), s.get("kind", "equal")),
    "derivative": lambda s: derivativeConstraints(s["points"], s["kx"], s.get("values", 0.0), s.get("output", 0),
                                                  s.get("kind", "equal"), s.get("order", 1)),
    "integral": lambda s: integralConstraint(s["value"], s.get("box"), s.get("points"), s.get("weights"),
                                             s.get("n", 8), s.get("output", 0), s.get("kind", "equal"),
                                             s.get("mean", False)),
    "mean": lambda s: integralConstraint(s["value"], s.get("box"), s.get("points"), s.get("weights"), s.get("n", 8),
                                         s.get("output", 0), s.get("kind", "equal"), True),
    "bound": lambda s: boundConstraints(s["points"], s.get("lower"), s.get("upper"), s.get("output", 0)),
    "monotone": lambda s: monotoneConstraints(s["points"], s["kx"], s.get("increasing", True), s.get("output", 0)),
    "convex": lambda s: convexConstraints(s["points"], s["kx"], s.get("convex", True), s.get("output", 0)),
    "outputSum": lambda s: outputSumConstraints(s["points"], s["coefficients"], s["value"], s.get("kind", "equal")),
    "general": lambda s: [LinearConstraint(s["terms"], s["value"], s.get("kind", "equal"), s.get("name", ""))],
}


def buildConstraints(specs: Optional[Sequence]) -> list[LinearConstraint]:
    """Expand constraint specs (dicts or LinearConstraint objects) into a flat list."""
    out = []
    for s in specs or []:
        if isinstance(s, LinearConstraint):
            out.append(s)
            continue
        s = dict(s)
        kind = s.get("type")
        if kind not in _BUILDERS:
            raise ValueError(f"constraint type must be one of {sorted(_BUILDERS)}")
        out.extend(_BUILDERS[kind](s))
    return out


# ---------------------------------------------------------------- evaluation
def stencil(point: np.ndarray, derivative: tuple, steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference points and weights for D^derivative at ``point``."""
    pts, wts = point[None, :].copy(), np.array([1.0])
    grow = 10.0 ** (len(derivative) - 1)            # larger steps for higher orders (round-off)
    for k in derivative:
        h = steps[k] * grow
        shift = np.zeros(point.size)
        shift[k] = h
        pts = np.vstack([pts + shift, pts - shift])
        wts = np.concatenate([wts, -wts]) / (2.0 * h)
    return pts, wts


def expand(constraints: list[LinearConstraint], ny: int, steps: np.ndarray, exactDerivative: bool = False):
    """Point sets and the constraint matrix over them.

    Returns (points[o] (k_o, nx) per output, rows) where rows is a list of
    (output, pointIndex, coefficient, derivative) entries per constraint. With
    ``exactDerivative`` first derivatives are kept symbolic (derivative=(k,))
    for models with analytic derivatives; otherwise all derivatives are
    expanded into value stencils.
    """
    pts = [[] for _ in range(ny)]
    counts = [0] * ny
    rows = []
    for c in constraints:
        entries = []
        for t in c.terms:
            if not 0 <= t.output < ny:
                raise ValueError(f"constraint refers to output {t.output}, model has {ny}")
            for p, w in zip(t.points, t.weights):
                if not t.derivative or (exactDerivative and len(t.derivative) == 1):
                    sp, sw, der = p[None, :], np.array([w]), t.derivative
                else:
                    sp, sw = stencil(p, t.derivative, steps)
                    sw, der = sw * w, ()
                for q, v in zip(sp, sw):
                    pts[t.output].append(q)
                    entries.append((t.output, counts[t.output], float(v), der))
                    counts[t.output] += 1
        rows.append(entries)
    points = [np.array(p) if p else np.zeros((0, steps.size)) for p in pts]
    return points, rows
