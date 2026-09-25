"""Name -> class registries and the pluggable-component base class.

Models and their building blocks (bases, solvers, kernels, transforms) are
registered under short names so they can be created from JSON-like specs::

    buildComponent("basis", {"type": "polynomial", "degree": 3})
    buildComponent("kernel", "matern52")
    createModel({"type": "kriging", "corr": "matern52"})

Every component carries its options in an ``OptionsDictionary`` and
round-trips through ``toDict`` / ``fromDict``.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, ClassVar, Dict, Type

import numpy as np

from pythonLibs.regressionHandler.core.OptionsDictionary import OptionsDictionary


class Registry:
    """A named collection of classes of one kind."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._classes: Dict[str, Type] = {}

    def register(self, name: str) -> Callable[[Type], Type]:
        def decorator(cls: Type) -> Type:
            if name in self._classes and self._classes[name] is not cls:
                raise ValueError(f"{self.kind} {name!r} is already registered to {self._classes[name].__name__}")
            self._classes[name] = cls
            cls.registryName = name
            return cls
        return decorator

    def get(self, name: str) -> Type:
        try:
            return self._classes[name]
        except KeyError:
            raise KeyError(f"unknown {self.kind} {name!r}; available: {', '.join(self.names())}") from None

    def names(self) -> list[str]:
        return sorted(self._classes)

    def __contains__(self, name: str) -> bool:
        return name in self._classes

    def items(self):
        return sorted(self._classes.items())


REGISTRIES: Dict[str, Registry] = {
    kind: Registry(kind) for kind in ("model", "basis", "solver", "kernel", "transform")
}


def registry(kind: str) -> Registry:
    return REGISTRIES[kind]


class ComponentBase:
    """Configurable, serializable building block (basis, solver, kernel, ...).

    Subclasses implement ``_declareOptions(declare)`` and may keep fitted
    state, exported with ``_stateToDict`` / restored with ``_stateFromDict``.
    """

    componentKind: ClassVar[str] = ""
    registryName: ClassVar[str] = ""

    def __init__(self, **options: Any) -> None:
        self.options = OptionsDictionary()
        self._declareOptions(self.options.declare)
        self.options.update(options)

    def _declareOptions(self, declare) -> None:
        pass

    def _stateToDict(self) -> dict:
        return {}

    def _stateFromDict(self, state: dict) -> None:
        pass

    def toDict(self) -> dict:
        d = {"type": self.registryName, **_optionsToDict(self.options.toDict())}
        state = self._stateToDict()
        if state:
            d["_state"] = state
        return d

    @classmethod
    def fromDict(cls, d: dict) -> "ComponentBase":
        d = dict(d)
        d.pop("type", None)
        state = d.pop("_state", None)
        obj = cls(**_optionsFromDict(d))
        if state:
            obj._stateFromDict(state)
        return obj

    def copyUnfitted(self) -> "ComponentBase":
        return type(self)(**copy.deepcopy(self.options.toDict()))

    def __repr__(self) -> str:
        opts = ", ".join(f"{k}={v!r}" for k, v in self.options.nonDefault().items())
        return f"{type(self).__name__}({opts})"


def buildComponent(kind: str, spec: Any) -> ComponentBase:
    """Create a component from a name, a ``{"type": name, **options}`` dict or an instance."""
    if isinstance(spec, ComponentBase):
        if spec.componentKind != kind:
            raise TypeError(f"expected a {kind}, got a {spec.componentKind}")
        return spec
    if isinstance(spec, str):
        return registry(kind).get(spec)()
    if isinstance(spec, dict):
        if "type" not in spec:
            raise ValueError(f"{kind} spec needs a 'type' key: {spec}")
        return registry(kind).get(spec["type"]).fromDict(spec)
    raise TypeError(f"cannot build a {kind} from {type(spec).__name__}")


def _plainValue(v):
    """numpy arrays / scalars (also inside lists and dicts) as JSON-ready Python values."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return type(v)(_plainValue(e) for e in v)
    if isinstance(v, dict):
        return {k: _plainValue(e) for k, e in v.items()}
    return v


def _optionsToDict(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if isinstance(v, ComponentBase):
            out[k] = {"__component__": v.componentKind, **v.toDict()}
        elif isinstance(v, (list, tuple)) and v and all(isinstance(e, ComponentBase) for e in v):
            out[k] = [{"__component__": e.componentKind, **e.toDict()} for e in v]
        else:
            out[k] = _plainValue(v)
    return out


def _optionsFromDict(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if isinstance(v, dict) and "__component__" in v:
            spec = dict(v)
            kind = spec.pop("__component__")
            out[k] = buildComponent(kind, spec)
        elif isinstance(v, list) and v and all(isinstance(e, dict) and "__component__" in e for e in v):
            out[k] = [buildComponent(e["__component__"], {kk: vv for kk, vv in e.items() if kk != "__component__"})
                      for e in v]
        else:
            out[k] = v
    return out
