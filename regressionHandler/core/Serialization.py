"""Compact JSON encoding of large numeric arrays.

Model dictionaries stay plain JSON, but every numeric list with at least
``minSize`` elements (rectangular, possibly with None entries) is replaced by

    {"__ndarray__": base64(zlib(raw bytes)), "dtype": "float64", "shape": [...], "none": base64(bitmask) | None}

which is exact (binary floats), several times smaller and much faster to
parse than decimal text. ``expandArrays`` restores the nested lists, so model
code sees the same structure either way.
"""
from __future__ import annotations

import base64
import zlib
from numbers import Number

import numpy as np

MARKER = "__ndarray__"
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
    ints = all(v is None or isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in flat)
    bools = all(isinstance(v, (bool, np.bool_)) for v in flat)
    if bools:
        out = flat.astype(bool)
    elif ints and not noneMask.any():
        out = flat.astype(np.int64)
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
