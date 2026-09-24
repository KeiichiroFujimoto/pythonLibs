"""Unit conversion convenience API.

Thin wrapper around :class:`~pythonLibs.dataHandler.UnitHandler.UnitHandler`
providing a clean functional interface for unit conversion.

Usage::

    from pythonLibs.dataHandler.unit_converter import convert

    temp_c = convert(300.0, "K", "C")       # 26.85
    pressure = convert(1.0, "atm", "Pa")    # 101325.0
"""
from __future__ import annotations

from typing import Any

from pythonLibs.dataHandler.UnitHandler import UnitHandler


def convert(value: Any, from_unit: str, to_unit: str) -> Any:
    """
    Convert *value* from *from_unit* to *to_unit*.

    Delegates to :class:`~pythonLibs.dataHandler.UnitHandler.UnitHandler`.

    Parameters
    ----------
    value : int | float
        Numerical value to convert.
    from_unit : str
        Original unit string, e.g. "m", "kg", "K".
    to_unit : str
        Desired target unit.

    Returns
    -------
    int | float
        Converted magnitude, preserving the original numeric type when possible.

    Raises
    ------
    ValueError
        If conversion cannot be performed.
    """
    if value is None:
        raise ValueError("Cannot convert None")
    return UnitHandler.convertUnit(value=value, unitNameIn=from_unit, unitNameOut=to_unit)
