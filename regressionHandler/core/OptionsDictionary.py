"""Declared, validated options.

Every model and component declares its options up front with a default,
allowed values/types, optional numeric range and a description. Assignment
is validated immediately, unknown names are rejected, and ``describe()``
returns the declarations as plain dicts so catalogs, CLI help and LLM tool
schemas can be generated from the same source.
"""
from __future__ import annotations

import copy
from typing import Any, Iterable, Optional


class OptionsDictionary:

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._declared: dict[str, dict] = {}

    def declare(self, name: str, default: Any = None, values: Optional[Iterable] = None,
                types=None, desc: str = "", lower: Optional[float] = None,
                upper: Optional[float] = None, allowNone: bool = False) -> None:
        """Declare an option.

        Args:
            name:      option name
            default:   default value (validated like any other value)
            values:    admissible discrete values (checked before ``types``)
            types:     admissible type or tuple of types
            desc:      human-readable description
            lower, upper: inclusive numeric range for numeric values
            allowNone: accept None in addition to the constraints above
        """
        self._declared[name] = {
            "values": tuple(values) if values is not None else None,
            "types": types,
            "desc": desc,
            "lower": lower,
            "upper": upper,
            "allowNone": allowNone or default is None,
            "default": default,
        }
        self._values[name] = copy.deepcopy(default)

    def _validate(self, name: str, value: Any) -> None:
        if name not in self._declared:
            raise KeyError(f"option {name!r} is not declared; known options: {sorted(self._declared)}")
        spec = self._declared[name]
        if value is None:
            if spec["allowNone"]:
                return
            raise ValueError(f"option {name!r} does not accept None")
        values, types = spec["values"], spec["types"]
        inValues = values is not None and _safeIn(value, values)
        inTypes = types is not None and isinstance(value, types) and not (
            isinstance(value, bool) and not _acceptsBool(types))
        if values is not None and types is not None:
            if not (inValues or inTypes):
                raise ValueError(f"option {name!r}: {value!r} must be one of {values} or of type {types}")
        elif values is not None and not inValues:
            raise ValueError(f"option {name!r}: {value!r} must be one of {values}")
        elif types is not None and not inTypes:
            raise TypeError(f"option {name!r}: {value!r} must be of type {types}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if spec["lower"] is not None and value < spec["lower"]:
                raise ValueError(f"option {name!r}: {value} is below the minimum {spec['lower']}")
            if spec["upper"] is not None and value > spec["upper"]:
                raise ValueError(f"option {name!r}: {value} is above the maximum {spec['upper']}")

    def __getitem__(self, name: str) -> Any:
        if name not in self._declared:
            raise KeyError(f"option {name!r} is not declared")
        return self._values[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self._validate(name, value)
        self._values[name] = value

    def __contains__(self, name: str) -> bool:
        return name in self._declared

    def __iter__(self):
        return iter(self._declared)

    def update(self, values: dict) -> None:
        for k, v in values.items():
            self[k] = v

    def get(self, name: str, default: Any = None) -> Any:
        return self._values.get(name, default)

    def toDict(self) -> dict:
        return {k: self._values[k] for k in self._declared}

    def nonDefault(self) -> dict:
        """Options whose value differs from the declared default."""
        out = {}
        for k, spec in self._declared.items():
            try:
                same = self._values[k] == spec["default"]
                same = bool(same) if not hasattr(same, "all") else bool(same.all())
            except Exception:
                same = False
            if not same:
                out[k] = self._values[k]
        return out

    def describe(self) -> list[dict]:
        """Declarations as JSON-friendly dicts (for catalogs and help)."""
        out = []
        for k, spec in self._declared.items():
            types = spec["types"]
            if types is not None:
                types = [t.__name__ for t in (types if isinstance(types, tuple) else (types,))]
            out.append({
                "name": k, "default": _plain(spec["default"]), "values": list(spec["values"]) if spec["values"] else None,
                "types": types, "lower": spec["lower"], "upper": spec["upper"], "desc": spec["desc"],
            })
        return out

    def clone(self) -> "OptionsDictionary":
        return copy.deepcopy(self)


def _safeIn(value, values) -> bool:
    try:
        return any(value is v or (type(value) is type(v) and value == v) for v in values)
    except Exception:
        return False


def _acceptsBool(types) -> bool:
    ts = types if isinstance(types, tuple) else (types,)
    return bool in ts


def _plain(value):
    toDict = getattr(value, "toDict", None)
    return toDict() if callable(toDict) else value
