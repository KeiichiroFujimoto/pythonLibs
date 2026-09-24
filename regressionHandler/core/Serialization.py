"""Compact JSON encoding of large numeric arrays.

Model dictionaries stay plain JSON, but every numeric list with at least
``minSize`` elements (rectangular, possibly with None entries) is replaced by

    {"__ndarray__": base64(zlib(raw bytes)), "dtype": "float64", "shape": [...], "none": base64(bitmask) | None}

which is exact (binary floats), several times smaller and much faster to
parse than decimal text. ``expandArrays`` restores the nested lists, so model
code sees the same structure either way. Integer and boolean arrays keep their
type (also with None entries); integers outside the int64 range stay plain.

Strict JSON has no NaN / Infinity, so files go through ``encodeNonFinite``
(each non-finite float becomes {"__float__": "nan" | "inf" | "-inf"}) and
``decodeNonFinite`` on the way back.
"""
from __future__ import annotations

import base64
import zlib
from numbers import Number

import numpy as np

MARKER = "__ndarray__"
FLOAT_MARKER = "__float__"
_INT64 = (-(2 ** 63), 2 ** 63 - 1)
_DTYPES = {"float64", "int64", "int32", "uint8", "bool"}


def _numericArray(value):
    """ndarray for a rectangular list of numbers / None, or None if it is not one."""
    if not value or isinstance(value[0], (str, dict)):
        return None, None
    try:
        arr = np.array(value, dtype=object)
    except (ValueError, TypeError):
        return None, None
    if arr.dtype != object or arr.ndim == 0:
        return None, None
    flat = arr.ravel()
    if not all(v is None or (isinstance(v, Number) and not isinstance(v, complex)) for v in flat):
        return None, None
    noneMask = np.array([v is None for v in flat])
    if noneMask.all():
        return None, None
    values = [v for v in flat if v is not None]
    isBool = [isinstance(v, (bool, np.bool_)) for v in values]
    if all(isBool):
        out = np.array([bool(v) if v is not None else False for v in flat], dtype=bool)
    elif any(isBool):
        return None, None                                   # mixed bool / number: keep plain
    elif all(isinstance(v, (int, np.integer)) for v in values):
        if any(not _INT64[0] <= int(v) <= _INT64[1] for v in values):
            return None, None                               # beyond int64: keep plain (exact Python ints)
        out = np.array([0 if v is None else int(v) for v in flat], dtype=np.int64)
    else:
        out = np.array([np.nan if v is None else float(v) for v in flat], dtype=float)
    return out.reshape(arr.shape), (noneMask if noneMask.any() else None)


def _encode(arr: np.ndarray, noneMask) -> dict:
    return {MARKER: base64.b64encode(zlib.compress(np.ascontiguousarray(arr).tobytes(), 6)).decode("ascii"),
            "dtype": str(arr.dtype), "shape": list(arr.shape),
            "none": None if noneMask is None else base64.b64encode(np.packbits(noneMask).tobytes()).decode("ascii")}


def compactArrays(obj, minSize: int = 64):
    """Copy of a JSON-like structure with large numeric lists encoded (see the module docstring)."""
    if isinstance(obj, dict):
        return {k: compactArrays(v, minSize) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) and not isinstance(obj[0], dict):
            size = len(obj) * (len(obj[0]) if isinstance(obj[0], list) else 1)
            if size >= minSize:
                arr, mask = _numericArray(obj)
                if arr is not None:
                    return _encode(arr, mask)
        return [compactArrays(v, minSize) for v in obj]
    return obj


def expandArrays(obj):
    """Inverse of ``compactArrays`` (plain structures pass through unchanged)."""
    if isinstance(obj, dict):
        if MARKER in obj:
            dtype = obj["dtype"]
            if dtype not in _DTYPES:
                raise ValueError(f"unsupported encoded dtype {dtype!r}")
            raw = zlib.decompress(base64.b64decode(obj[MARKER]))
            arr = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(obj["shape"])
            out = arr.tolist()
            if obj.get("none"):
                mask = np.unpackbits(np.frombuffer(base64.b64decode(obj["none"]), dtype=np.uint8))[:arr.size]
                flat = arr.ravel().tolist()
                flat = [None if m else v for v, m in zip(flat, mask)]
                out = np.array(flat, dtype=object).reshape(arr.shape).tolist()
            return out
        return {k: expandArrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expandArrays(v) for v in obj]
    return obj


def encodeNonFinite(obj):
    """Copy with every non-finite float replaced by {"__float__": "nan" | "inf" | "-inf"} (strict JSON)."""
    if isinstance(obj, float) or isinstance(obj, np.floating):
        v = float(obj)
        if v != v:
            return {FLOAT_MARKER: "nan"}
        if v in (np.inf, -np.inf):
            return {FLOAT_MARKER: "inf" if v > 0 else "-inf"}
        return v
    if isinstance(obj, dict):
        return {k: encodeNonFinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encodeNonFinite(v) for v in obj]
    return obj


def decodeNonFinite(obj):
    """Inverse of ``encodeNonFinite``."""
    if isinstance(obj, dict):
        if len(obj) == 1 and FLOAT_MARKER in obj:
            return float(obj[FLOAT_MARKER])
        return {k: decodeNonFinite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decodeNonFinite(v) for v in obj]
    return obj
