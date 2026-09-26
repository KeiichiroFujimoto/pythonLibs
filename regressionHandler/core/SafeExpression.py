"""Whitelisted arithmetic expressions evaluated on numpy arrays.

Expressions let model forms and basis functions be written as text
(``"a*exp(-b/x)"``) so fitted models can be saved to JSON and loaded
again without pickling Python callables. The text is parsed with ``ast`` and
only numeric literals, known symbols, arithmetic operators and a fixed set of
numpy functions are accepted; anything else (attribute access, subscripts,
lambdas, comprehensions, keyword arguments ...) is rejected at parse time, so
loading an expression from an untrusted file cannot execute arbitrary code.
"""
from __future__ import annotations

import ast
import operator
from typing import Iterable, Mapping

import numpy as np

_FUNCTIONS = {
    "exp": np.exp, "expm1": np.expm1, "log": np.log, "log10": np.log10,
    "log2": np.log2, "log1p": np.log1p, "sqrt": np.sqrt, "cbrt": np.cbrt,
    "abs": np.abs, "sign": np.sign,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan, "arctan2": np.arctan2,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "minimum": np.minimum, "maximum": np.maximum, "power": np.power,
    "heaviside": lambda v: np.heaviside(v, 0.5),
}

_CONSTANTS = {"pi": np.pi, "e": np.e}

_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
}

_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class SafeExpression:
    """Parsed, validated expression over a fixed set of symbols.

    Example:
        >>> expr = SafeExpression("a*x**b", symbols=["a", "b", "x"])
        >>> expr.evaluate({"a": 2.0, "b": 2.0, "x": np.array([1.0, 3.0])})
        array([ 2., 18.])
    """

    def __init__(self, text: str, symbols: Iterable[str]):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("expression must be a non-empty string")
        self.text = text.strip()
        self.symbols = list(symbols)
        overlap = set(self.symbols) & (set(_FUNCTIONS) | set(_CONSTANTS))
        if overlap:
            raise ValueError(f"symbol names collide with reserved names: {sorted(overlap)}")
        try:
            tree = ast.parse(self.text, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"invalid expression {self.text!r}: {exc.msg}") from None
        self._body = tree.body
        self.usedSymbols: set[str] = set()
        self._validate(self._body)

    @staticmethod
    def availableFunctions() -> list[str]:
        """Names of the functions an expression may call (sin, exp, log, sqrt, ...)."""
        return sorted(_FUNCTIONS)

    def _validate(self, node: ast.AST) -> None:
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _BINARY:
                raise ValueError(f"operator {type(node.op).__name__} not allowed in {self.text!r}")
            self._validate(node.left)
            self._validate(node.right)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY:
                raise ValueError(f"operator {type(node.op).__name__} not allowed in {self.text!r}")
            self._validate(node.operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
                raise ValueError(
                    f"only these functions are allowed: {', '.join(sorted(_FUNCTIONS))}"
                )
            if node.keywords:
                raise ValueError("keyword arguments are not allowed in expressions")
            for arg in node.args:
                self._validate(arg)
        elif isinstance(node, ast.Name):
            if node.id in _CONSTANTS:
                return
            if node.id not in self.symbols:
                raise ValueError(f"unknown symbol {node.id!r} in {self.text!r}")
            self.usedSymbols.add(node.id)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"only numeric literals are allowed, got {node.value!r}")
        else:
            raise ValueError(f"syntax element {type(node).__name__} not allowed in {self.text!r}")

    def evaluate(self, env: Mapping[str, object]):
        """Evaluate with values (scalars or numpy arrays) for every symbol; raises if one is missing."""
        missing = self.usedSymbols - set(env)
        if missing:
            raise ValueError(f"missing values for symbols: {sorted(missing)}")
        with np.errstate(all="ignore"):
            return self._eval(self._body, env)

    def _eval(self, node: ast.AST, env: Mapping[str, object]):
        if isinstance(node, ast.BinOp):
            return _BINARY[type(node.op)](self._eval(node.left, env), self._eval(node.right, env))
        if isinstance(node, ast.UnaryOp):
            return _UNARY[type(node.op)](self._eval(node.operand, env))
        if isinstance(node, ast.Call):
            return _FUNCTIONS[node.func.id](*[self._eval(a, env) for a in node.args])
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            return _CONSTANTS[node.id]
        return float(node.value)

    def __repr__(self) -> str:
        return f"SafeExpression({self.text!r})"
