"""Input coercion shared by every regression model.

All models work on a 2D feature matrix ``(nSamples, nFeatures)`` and a 1D
target ``(nSamples,)``. These helpers turn user input (lists, 1D arrays,
scalars) into that canonical form and reject non-finite values early, so the
numerical code never has to second-guess its inputs.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def asFeatureMatrix(x, nFeatures: Optional[int] = None, name: str = "x") -> np.ndarray:
    """Return ``x`` as a finite float array of shape (nSamples, nFeatures).

    A 1D input is read as one feature per sample, except when the model is
    known to have ``nFeatures > 1`` and the input length equals it; then it
    is read as a single multi-feature sample.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        if nFeatures is not None and nFeatures > 1 and arr.size == nFeatures:
            arr = arr.reshape(1, -1)
        else:
            arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got {arr.ndim}D")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} is empty")
    if nFeatures is not None and arr.shape[1] != nFeatures:
        raise ValueError(f"{name} has {arr.shape[1]} features, model expects {nFeatures}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def asOutputMatrix(y, nSamples: int, name: str = "y") -> np.ndarray:
    """Return ``y`` as a finite float array of shape (nSamples, nOutputs)."""
    arr = np.asarray(y, dtype=float)
    if arr.ndim <= 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got {arr.ndim}D")
    if arr.shape[0] != nSamples:
        raise ValueError(f"{name} has {arr.shape[0]} rows, expected {nSamples}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def asTarget(y, nSamples: int, name: str = "y") -> np.ndarray:
    """Return ``y`` as a finite 1D float array of length ``nSamples``."""
    arr = np.asarray(y, dtype=float).ravel()
    if arr.size != nSamples:
        raise ValueError(f"{name} has {arr.size} values, expected {nSamples}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def asWeights(weights, nSamples: int) -> np.ndarray:
    """Return strictly positive sample weights (all ones when ``weights`` is None)."""
    if weights is None:
        return np.ones(nSamples)
    arr = np.asarray(weights, dtype=float).ravel()
    if arr.size != nSamples:
        raise ValueError(f"weights has {arr.size} values, expected {nSamples}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("weights must be finite and strictly positive")
    return arr


def toJsonable(value):
    """Recursively convert numpy containers/scalars to plain Python types."""
    if isinstance(value, dict):
        return {str(k): toJsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [toJsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return toJsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if np.isnan(v):
            return None
        if np.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    return value
