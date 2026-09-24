"""Auto-discovery registry for toolBaseSecured-derived service classes.

When a subclass of ``toolBaseSecured`` is defined, ``__init_subclass__`` calls
``maybeRegister(cls)`` here. Classes that match the convention
(name ends in ``Service``, no required ``__init__`` args) are recorded in a
process-wide registry under a derived short name.

The registry is **inert at import time** — it just records class references.
Services are only instantiated when ``collectAll()`` is called explicitly,
so importing a service module never triggers heavy work elsewhere.

Convention
----------
- ``class FooService(toolBaseSecured)`` → registered under name ``"foo"``
  (auto-derived: strip ``Service`` suffix, lowercase first letter).
- ``class FooBarService(toolBaseSecured)`` → registered under ``"fooBar"``.
- ``class FooOps(toolBaseSecured)`` → **skipped** (does not end in ``Service``).
- Override with ``LAB_NAME = "myname"`` class attribute.
- Force opt-in with ``LAB_AUTOREGISTER = True``; opt-out with
  ``LAB_AUTOREGISTER = False``.

Usage
-----
::

    from pythonLibs.tool.LabRegistry import collectAll, listRegistered

    # After importing any modules that define *Service classes:
    services = collectAll()  # name -> instantiated service
    print(listRegistered())  # name -> class
"""
from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from typing import Any, Dict, Iterable, List, Optional, Type

logger = logging.getLogger(__name__)


_REGISTRY: Dict[str, Type[Any]] = {}


def _deriveName(className: str) -> str:
    """Convert ``"VehicleSpecService"`` → ``"vehicleSpec"``."""
    if className.endswith("Service"):
        className = className[: -len("Service")]
    if not className:
        return ""
    return className[:1].lower() + className[1:]


def _instantiableNoArgs(cls: Type[Any]) -> bool:
    """True iff ``cls()`` can be called with no arguments."""
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return False
    for p in list(sig.parameters.values())[1:]:  # skip self
        if p.default is inspect.Parameter.empty and p.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return False
    return True


def maybeRegister(cls: Type[Any]) -> None:
    """Decide whether ``cls`` should be auto-registered, and do so if yes.

    Failures are logged at DEBUG and never raised — the toolBaseSecured
    ``__init_subclass__`` hook calls this and must not break tool imports.
    """
    try:
        autoreg = getattr(cls, "LAB_AUTOREGISTER", None)
        if autoreg is False:
            return
        if autoreg is None:
            # Convention: only classes ending in "Service" auto-register.
            if not cls.__name__.endswith("Service"):
                return
        # Auto-instantiable required (no required args other than self).
        if not _instantiableNoArgs(cls):
            return
        name = getattr(cls, "LAB_NAME", None) or _deriveName(cls.__name__)
        if not name:
            return
        # Last definition wins on collision — log a debug note.
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            logger.debug(
                "LabRegistry: name %r already mapped to %s; replacing with %s",
                name, _REGISTRY[name].__module__ + "." + _REGISTRY[name].__name__,
                cls.__module__ + "." + cls.__name__,
            )
        _REGISTRY[name] = cls
    except Exception:
        # Never let registry failures break tool imports.
        logger.debug("LabRegistry.maybeRegister failed for %r", cls, exc_info=True)


def listRegistered() -> Dict[str, Type[Any]]:
    """Return the current name → class mapping (a defensive copy)."""
    return dict(_REGISTRY)


def _isLikelyServiceModule(fullName: str) -> bool:
    """Heuristic: limit auto-discovery imports to modules whose leaf name
    ends in ``Service`` (case-insensitive).

    Skips test packages (``tests``/``test`` segment) and ``test_*`` files.
    The ``endswith('service')`` rule is strong enough that we don't need
    additional script/demo prefix exclusions — a plain ``Demo.py`` or
    ``RunFoo.py`` won't match, while real services like ``PlotService.py``
    or ``RemoteToolProxyService.py`` will.
    """
    leaf = fullName.rsplit(".", 1)[-1]
    parts = fullName.split(".")
    if any(seg in {"tests", "test"} for seg in parts):
        return False
    if leaf.startswith("test_") or leaf.startswith("Test"):
        return False
    return leaf.lower().endswith("service")


def discoverInPackages(
    packages: Iterable[str],
    *,
    moduleFilter: Optional[Any] = None,
) -> List[str]:
    """Walk ``packages`` and import service-looking submodules so their
    ``__init_subclass__`` hooks fire.

    Parameters
    ----------
    packages : top-level package names to walk (e.g. ``["pythonLibs.rocketDesign"]``)
    moduleFilter : callable ``(fullName: str) -> bool``. If None, defaults to
        :func:`_isLikelyServiceModule` which limits imports to modules whose
        leaf name ends in ``Service``.

    Returns
    -------
    list of fully-qualified module names that were imported.
    """
    if moduleFilter is None:
        moduleFilter = _isLikelyServiceModule

    imported: List[str] = []
    for pkg_name in packages:
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception:
            logger.debug("discoverInPackages: cannot import %r", pkg_name, exc_info=True)
            continue
        if not hasattr(pkg, "__path__"):
            imported.append(pkg_name)
            continue
        for finder, name, ispkg in pkgutil.walk_packages(pkg.__path__, prefix=pkg_name + "."):
            if not moduleFilter(name):
                continue
            try:
                importlib.import_module(name)
                imported.append(name)
            except Exception:
                logger.debug("discoverInPackages: cannot import %r", name, exc_info=True)
    return imported


def collectAll(
    *,
    includeNames: Optional[Iterable[str]] = None,
    excludeNames: Optional[Iterable[str]] = None,
    includePackagePrefixes: Optional[Iterable[str]] = None,
    excludePackagePrefixes: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Instantiate registered services and return ``name -> instance`` dict.

    Parameters
    ----------
    includeNames : if provided, only return these names
    excludeNames : exclude these names (applied after include)
    includePackagePrefixes : only include classes whose ``__module__`` starts
        with one of these prefixes
    excludePackagePrefixes : exclude classes whose ``__module__`` starts with
        one of these prefixes (applied after include)
    """
    out: Dict[str, Any] = {}
    inc_names = set(includeNames) if includeNames else None
    exc_names = set(excludeNames or [])
    inc_prefixes = tuple(includePackagePrefixes) if includePackagePrefixes else None
    exc_prefixes = tuple(excludePackagePrefixes or [])
    for name, cls in _REGISTRY.items():
        if inc_names is not None and name not in inc_names:
            continue
        if name in exc_names:
            continue
        mod = cls.__module__ or ""
        if inc_prefixes is not None and not mod.startswith(inc_prefixes):
            continue
        if exc_prefixes and mod.startswith(exc_prefixes):
            continue
        instance = None
        # Attempt 1: zero-arg constructor.
        try:
            instance = cls()
        except Exception:
            # Attempt 2: many toolBaseSecured services default to
            # secure_enabled=True which tries to load secret.json. Retry
            # with secure_enabled=False so the lab can still grab them.
            try:
                instance = cls(secure_enabled=False)
            except Exception:
                logger.debug(
                    "LabRegistry.collectAll: instantiation failed for %s.%s",
                    cls.__module__, cls.__name__, exc_info=True,
                )
                continue
        out[name] = instance
    return out


def filterByInputType(inputType: str) -> Dict[str, Type[Any]]:
    """Return ``name -> class`` mapping for services whose ``inputType`` ClassVar matches.

    Used by knowledgeLink.dispatcher.LensDispatcher to find all Lens services
    that subscribe to a given Description type (e.g. "tmtc.envelope").

    Lens-as-Service contract (knowledgeLink/DESIGN.md §6):
        class FooLensService(toolBaseSecured):
            inputType: ClassVar[str]   # "tmtc.envelope"
            outputType: ClassVar[str]  # "anomaly.crossLayer"

    Classes without an ``inputType`` attribute are silently skipped.

    Parameters
    ----------
    inputType : exact-match Description.descriptionType string to filter by.

    Returns
    -------
    dict[str, type] : registered short name → class. Empty if no match.
    """
    return {
        name: cls
        for name, cls in _REGISTRY.items()
        if getattr(cls, "inputType", None) == inputType
    }


def listInputTypes() -> List[str]:
    """Return sorted list of distinct ``inputType`` values across registered services."""
    types: set[str] = set()
    for cls in _REGISTRY.values():
        t = getattr(cls, "inputType", None)
        if isinstance(t, str) and t:
            types.add(t)
    return sorted(types)


def reset() -> None:
    """Clear the registry. Test-only helper."""
    _REGISTRY.clear()
