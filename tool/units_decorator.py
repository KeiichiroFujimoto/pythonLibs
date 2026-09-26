"""units_decorator - decorator for automatic unit conversion of function arguments using Pint.

Example:
    @with_units(radius="m", thickness="mm->m", pressure="MPa->Pa")
    def evaluate(radius, thickness, pressure):
        # radius stays in m, thickness is converted to m, pressure to Pa
        ...

    @with_units(pressure_mean="MPa->Pa", pressure_std="MPa->Pa",
                temperature_mean="K", temperature_std="K")
    def monteCarlo(pressure_mean, pressure_std, temperature_mean, temperature_std):
        ...

Format:
    "unit"          -> keep that unit (metadata only, no conversion)
    "src->dst"      -> receive in src units, convert to dst units and pass to the function
"""

from __future__ import annotations

from functools import wraps
from typing import Callable, Dict

try:
    import pint
    _ureg = pint.UnitRegistry()
    _PINT_AVAILABLE = True
except ImportError:
    _PINT_AVAILABLE = False


def _convert(value: float, src_unit: str, dst_unit: str) -> float:
    """Unit conversion with Pint. If Pint is unavailable, convert with known fixed factors."""
    if _PINT_AVAILABLE:
        q = value * _ureg(src_unit)
        return q.to(dst_unit).magnitude

    # Fallback: common unit conversions
    _factors = {
        ("mm", "m"): 1e-3,
        ("m", "mm"): 1e3,
        ("MPa", "Pa"): 1e6,
        ("Pa", "MPa"): 1e-6,
        ("kPa", "Pa"): 1e3,
        ("Pa", "kPa"): 1e-3,
        ("km", "m"): 1e3,
        ("m", "km"): 1e-3,
        ("g", "kg"): 1e-3,
        ("kg", "g"): 1e3,
        ("degC", "K"): None,  # Handled separately
        ("K", "degC"): None,
    }
    if (src_unit, dst_unit) in _factors:
        factor = _factors[(src_unit, dst_unit)]
        if factor is not None:
            return value * factor
    # Degrees Celsius ↔ Kelvin
    if src_unit == "degC" and dst_unit == "K":
        return value + 273.15
    if src_unit == "K" and dst_unit == "degC":
        return value - 273.15
    if src_unit == dst_unit:
        return value
    raise ValueError(f"Unknown unit conversion: {src_unit} -> {dst_unit}")


def with_units(**unit_specs: str) -> Callable:
    """Decorator for automatic unit conversion of function arguments.

    Args:
        **unit_specs: {param_name: "unit" or "src->dst"}
                      "m"           -> metadata only (no conversion)
                      "mm->m"       -> mm input, converted to m before being passed to the function
    """
    # Parse specs
    conversions: Dict[str, tuple] = {}
    for param, spec in unit_specs.items():
        if "->" in spec:
            src, dst = spec.split("->")
            conversions[param] = (src.strip(), dst.strip())
        # Else: metadata only, no conversion

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for param, (src, dst) in conversions.items():
                if param in kwargs and kwargs[param] is not None:
                    kwargs[param] = _convert(kwargs[param], src, dst)
            return func(*args, **kwargs)

        # Attach unit metadata for documentation / introspection
        wrapper.__units__ = unit_specs  # type: ignore[attr-defined]
        return wrapper

    return decorator
