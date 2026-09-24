# decorators.py
from __future__ import annotations
from functools import wraps
from typing import Any, Callable, Iterable, Optional

__all__ = ["secure_expose", "no_auth"]


def secure_expose(_func: Optional[Callable] = None, *, alias: Optional[str] = None,
                  category: Optional[str] = None,
                  required_roles: Optional[Iterable[str]] = None,
                  precondition: Optional[dict[str, Any]] = None,
                  effect: Optional[dict[str, Any]] = None) -> Callable:
    """
    Mark a method as *securely exposed* (eligible for auto-wrapping and auth checks).
    - `alias`: optional human-friendly alias (used by resolve/invoke helpers)
    - `category`: optional category for catalog grouping (e.g. "entity", "persistence")
    - `required_roles`: reserved for future role checks (not enforced by default)
    - `precondition`: declarative preconditions — world-state propositions that must hold
    - `effect`: declarative effects — how world-state changes after execution

    Precondition/effect follow STRIPS-like semantics:
      keys = proposition names (string identifiers for world-state predicates)
      values = True/False (boolean), string with {param} substitution, or None (deleted)

    Usage:
        @secure_expose
        def run(self): ...

        @secure_expose(alias="createEntity", category="entity",
                       precondition={"project_loaded": True},
                       effect={"entity_exists": "{entity_id}", "diagram_dirty": True})
        def create_entity(self, entity_id: str): ...
    """
    def decorator(func: Callable) -> Callable:
        setattr(func, "_secure_expose", True)
        if alias:
            setattr(func, "_secure_alias", alias)
        if category:
            setattr(func, "_secure_category", category)
        if required_roles is not None:
            setattr(func, "_secure_required_roles", tuple(required_roles))
        if precondition is not None:
            setattr(func, "_secure_precondition", precondition)
        if effect is not None:
            setattr(func, "_secure_effect", effect)
        return func

    if _func is None:
        return decorator
    return decorator(_func)


def no_auth(func: Callable) -> Callable:
    """
    Mark a method as *exposed without auth*. It will be auto-wrapped but the wrapper
    will bypass token verification. Use sparingly for safe, non-sensitive operations
    (e.g., health checks, local debug helpers).
    """
    setattr(func, "_no_auth", True)
    return func
