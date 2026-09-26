from __future__ import annotations

import os, sys
if not os.environ.get('NEXUS_PATH_CONFIGURED'): sys.path.append(os.environ.get('PYTHON_PATH_PYTHONLIBS', '') + '/')
from pythonLibs.tool.ToolSecurityManager import ToolSecurityManager, SecurityVerificationError

# tool_base_secured.py
import fnmatch
import functools
import os
import re
import time
import traceback
from typing import Any, Callable, Dict, Optional, Set

# Import toolBase from your library layout; provide a fallback if needed
try:
    from pythonLibs.toolBase import toolBase
except Exception:  # pragma: no cover
    from pythonLibs.tool import toolBase

from pythonLibs.tool.decorators import secure_expose, no_auth  # re-exported in README for clarity

import inspect as _kwarg_inspect
import re as _kwarg_re

_CAMEL_BOUNDARY_RE = _kwarg_re.compile(r"([a-z0-9])([A-Z])")


def _camelToSnake(name: str) -> str:
    return _CAMEL_BOUNDARY_RE.sub(r"\1_\2", name).lower()


def _snakeToCamel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])


def _adaptKwargsToSignature(method, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Re-key ``kwargs`` so each key matches a parameter of ``method``.

    Tolerates camelCase ↔ snake_case mismatches between caller and target.
    If the underlying method declares ``**kwargs`` we leave the dict alone.
    Unknown keys that don't match either form are passed through (so the
    method can decide to raise its own TypeError).
    """
    if not kwargs:
        return kwargs
    try:
        sig = _kwarg_inspect.signature(method)
    except (TypeError, ValueError):
        return kwargs
    params = sig.parameters
    if any(p.kind is _kwarg_inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    valid = set(params)
    out: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in valid:
            out[k] = v
            continue
        snake = _camelToSnake(k)
        if snake in valid:
            out[snake] = v
            continue
        camel = _snakeToCamel(k)
        if camel in valid:
            out[camel] = v
            continue
        out[k] = v  # leave it; method will raise if truly unknown
    return out

class toolBaseSecured(toolBase):
    """
    Exposed-method extension of toolBase with pluggable verification.

    Modes:
      - auto_all_public (default): expose & wrap all subclass public methods; @no_auth bypasses verification.
      - expose_only: expose ONLY methods decorated with @secure_expose (closed-by-default API).

    Strict base wrap:
      - If `strict_base_wrap=True`, selected base-class methods in STRICT_BASE_WRAP_ALLOWLIST are wrapped too.

    This public-domain package does not include a login implementation. If
    access control is needed, inject an application-owned verifier through
    ``auth_handler``.
    """

    # Optional alias map: human-friendly name -> actual method name
    EXPOSE_ALIASES: Dict[str, str] = {}

    # Base methods you may want to guard in strict mode (adjust for your base)
    STRICT_BASE_WRAP_ALLOWLIST: Set[str] = {
        "runPreprocess", "runPostprocess", "executeWithFileStatusCheck"
    }

    # Default root for user work directories (can be overridden per instance)
    DEFAULT_WORKDIR_ROOT = os.environ.get("TOOL_WORKDIR_ROOT", "./work")

    # ── Lab auto-registration (LabRegistry) ────────────────────────────────
    # ``__init_subclass__`` calls maybeRegister(cls). The registry is inert
    # at import time — it just records the class reference. Override these
    # to control registration:
    #   LAB_AUTOREGISTER = True/False/None (None = auto by name convention)
    #   LAB_NAME = "myname" (None = derived from class name)
    LAB_AUTOREGISTER: Optional[bool] = None
    LAB_NAME: Optional[str] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        try:
            from pythonLibs.tool.LabRegistry import maybeRegister
            maybeRegister(cls)
        except Exception:
            # Never let registry failures break tool imports.
            pass

    def __init__(
        self,
        instance_subcls=None,
        variablesDict=None,
        filePathConfigToolExecutor=None,
        caseName=None,
        *,
        secure_enabled: Optional[bool] = None,
        auth_handler: Optional[object] = None,
        token: Optional[str] = None,
        exposure_mode: str = "auto_all_public",  # or "expose_only"
        strict_base_wrap: bool = False,
        workdir_root: Optional[str] = None,
    ) -> None:
        super().__init__(instance_subcls, variablesDict, filePathConfigToolExecutor, caseName)

        # Priority: explicit arg > env var > default(True)
        if secure_enabled is None:
            env = os.environ.get("TOOLBASE_SECURED_ENABLED", "1").strip()
            secure_enabled = (env != "0" and env.lower() != "false")

        self._security = ToolSecurityManager(handler=auth_handler, secure_enabled=secure_enabled)
        if token:
            self._security.set_token(token)

        self._exposure_mode = exposure_mode
        self._strict_base_wrap = bool(strict_base_wrap)

        # Username cached at this layer as well (mirrors security manager)
        self._username: Optional[str] = None

        # Root for user-specific working dirs
        self._workdir_root = workdir_root or self.DEFAULT_WORKDIR_ROOT

        # Wrap methods according to policy
        self._wrap_methods()

        # Populate username (if token already provided)
        self._username = self._security.get_username()

    # ---------- Public API: tokens & username ----------
    def set_access_token(self, token: Optional[str]) -> None:
        """Set the access token used for calls that do not pass their own ``token=``."""
        self._security.set_token(token)

    def enable_security(self) -> None:
        """Turn token verification on for all exposed methods."""
        self._security.enable()

    def disable_security(self) -> None:
        """Turn token verification off (local use, tests)."""
        self._security.disable()

    def set_username(self, username: Optional[str]) -> None:
        """Manually set username (overrides cached value)."""
        self._username = username
        self._security.set_username(username)

    def get_username(self) -> Optional[str]:
        """Return the cached username (None until verified or manually set)."""
        # Keep in sync with security manager
        return self._username or self._security.get_username()

    # ---------- Work directory helpers ----------
    @staticmethod
    def _sanitize_component(s: str) -> str:
        """Sanitize a path component (letters, digits, _.- allowed; others -> '_')."""
        return re.sub(r"[^A-Za-z0-9_.-]", "_", s)

    def set_workdir_root(self, root: str) -> None:
        """Set the root directory of the per-user work directories (see get_user_workdir)."""
        self._workdir_root = root

    def get_workdir_root(self) -> str:
        """Return the root directory of the per-user work directories."""
        return self._workdir_root

    def get_user_workdir(self, case_name: Optional[str] = None, create: bool = True) -> str:
        """
        Compute a per-user working directory path like: <root>/<username>/<case>
        - Username defaults to 'anonymous' if not known yet.
        - Case name defaults to self.caseName or 'default'.
        - Optionally create the directory.
        """
        uname = self.get_username() or "anonymous"
        case = case_name or getattr(self, "caseName", None) or "default"
        path = os.path.join(self._workdir_root, self._sanitize_component(uname), self._sanitize_component(case))
        if create:
            os.makedirs(path, exist_ok=True)
        return path

    # ---------- Discovery & dispatch ----------
    def list_exposed_methods(self, pattern: Optional[str] = None) -> list[str]:
        """Return the sorted names of the exposed methods.

        Args:
            pattern (str): Optional shell-style filter, e.g. ``"get*"``.
        """
        if not hasattr(self, "_exposed_methods"):
            return []
        items = sorted(self._exposed_methods)
        if pattern:
            items = [m for m in items if fnmatch.fnmatch(m, pattern)]
        return items

    def resolve_method_name(self, name: str) -> str:
        """Map an alias or method name to the exposed method name.

        Raises:
            AttributeError: the name is neither an exposed method nor an alias of one.
        """
        actual = self.EXPOSE_ALIASES.get(name, name)
        if not hasattr(self, "_exposed_methods") or actual not in self._exposed_methods:
            raise AttributeError(f"method '{name}' is not exposed")
        return actual

    def invoke(self, method_name: str, *args, token: Optional[str] = None, **kwargs) -> Any:
        """Call an exposed method by name or ``@secure_expose`` alias.

        The call goes through the same checks as every other interface: the token is
        verified when security is enabled, required roles are checked when enforced, and
        camelCase / snake_case keyword names are matched to the method signature. The
        result is stored in ``paramDict["executionResult"][methodName]`` together with the
        arguments, and recorded as a job when a ToolTrace is attached. Exceptions are
        recorded and re-raised.

        Args:
            method_name (str): Method name or alias, e.g. ``"tipDeflection"``.
            token (str): Access token for this call; defaults to the instance token.

        Returns:
            Whatever the method returns.

        Raises:
            AttributeError: the name is not exposed.
            SecurityVerificationError: security is enabled and the token is missing or rejected.
            PermissionError: role enforcement is on and the actor lacks a required role.
        """
        actual = self.resolve_method_name(method_name)
        if token is not None:
            kwargs.setdefault("token", token)
        method = getattr(self, actual)

        # Role-based access control: check required_roles if enforced
        required_roles = getattr(method, "_secure_required_roles", None)
        if required_roles and getattr(self, "_enforce_roles", False):
            actor_roles = getattr(self, "_actor_roles", set())
            if not actor_roles.intersection(required_roles):
                raise PermissionError(
                    f"Role required: {required_roles}. Actor has: {actor_roles}"
                )
        tool_trace = getattr(self, "_tool_trace", None)
        previous_job_context = getattr(self, "_active_tool_job_context", None)
        job_context = previous_job_context
        owns_job_context = False
        if tool_trace is not None and previous_job_context is None:
            try:
                job_context = tool_trace.begin_job(tool=self, method_name=actual)
                setattr(self, "_active_tool_job_context", job_context)
                owns_job_context = True
            except Exception:
                job_context = None
        exec_result_before = None
        had_exec_entry_before = False
        try:
            exec_results = self.paramDict.get("executionResult", {})
            if isinstance(exec_results, dict):
                had_exec_entry_before = actual in exec_results
                exec_result_before = exec_results.get(actual)
        except Exception:
            exec_results = {}

        def _execution_context_snapshot() -> Dict[str, Any]:
            execution_context = {}
            get_execution_context = getattr(self, "getExecutionContext", None)
            if callable(get_execution_context):
                try:
                    execution_context = get_execution_context()
                except Exception:
                    execution_context = {}
            return execution_context

        def _should_auto_record(exec_results_after: Any) -> bool:
            should_auto_record = True
            if isinstance(exec_results_after, dict):
                has_exec_entry_after = actual in exec_results_after
                exec_result_after = exec_results_after.get(actual)
                if has_exec_entry_after and (not had_exec_entry_before or exec_result_after is not exec_result_before):
                    should_auto_record = False
            return should_auto_record

        started_at = time.perf_counter()
        # Adapt camelCase ↔ snake_case kwargs to the bound method's signature
        # so callers don't have to remember each service's local naming choice.
        kwargs = _adaptKwargsToSignature(method, kwargs)
        try:
            result = method(*args, **kwargs)
        except Exception as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000.0
            try:
                exec_results_after = self.paramDict.get("executionResult", {})
            except Exception:
                exec_results_after = {}
            if _should_auto_record(exec_results_after):
                execution_context = _execution_context_snapshot()
                error_info = {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                if "executionResult" not in self.paramDict:
                    self.paramDict["executionResult"] = {}
                self.paramDict["executionResult"][actual] = {
                    "args": args,
                    "kwargs": dict(kwargs or {}),
                    "result": None,
                    "error": error_info,
                    "filePathListSaved": None,
                    "executionContext": execution_context if execution_context else None,
                }
                tool_trace = getattr(self, "_tool_trace", None)
                if tool_trace is not None:
                    try:
                        tool_trace.record(
                            tool=self,
                            method_name=actual,
                            args=args,
                            kwargs=kwargs,
                            result=None,
                            file_paths=None,
                            execution_context=execution_context,
                            duration_ms=duration_ms,
                            status="failed",
                            error=error_info,
                            job_id=(job_context or {}).get("jobId"),
                            job_started_at=(job_context or {}).get("startedAt"),
                            parent_job_id=(job_context or {}).get("parentJobId"),
                        )
                    except Exception:
                        pass
            raise
        finally:
            if owns_job_context:
                try:
                    delattr(self, "_active_tool_job_context")
                except AttributeError:
                    pass
        duration_ms = (time.perf_counter() - started_at) * 1000.0

        try:
            exec_results_after = self.paramDict.get("executionResult", {})
        except Exception:
            exec_results_after = {}

        if _should_auto_record(exec_results_after):
            execution_context = _execution_context_snapshot()

            if "executionResult" not in self.paramDict:
                self.paramDict["executionResult"] = {}
            self.paramDict["executionResult"][actual] = {
                "args": args,
                "kwargs": dict(kwargs or {}),
                "result": result,
                "filePathListSaved": None,
                "executionContext": execution_context if execution_context else None,
            }

            if tool_trace is not None:
                try:
                    tool_trace.record(
                        tool=self,
                        method_name=actual,
                        args=args,
                        kwargs=kwargs,
                        result=result,
                        file_paths=None,
                        execution_context=execution_context,
                        duration_ms=duration_ms,
                        job_id=(job_context or {}).get("jobId"),
                        job_started_at=(job_context or {}).get("startedAt"),
                        parent_job_id=(job_context or {}).get("parentJobId"),
                    )
                except Exception:
                    pass

        return result

    def refresh_exposed(self) -> None:
        """Re-scan the class for exposed methods, e.g. after adding methods at runtime."""
        self._wrap_methods()

    # ---------- Internal: wrapping policy ----------
    def _wrap_methods(self) -> None:
        self._exposed_methods: Set[str] = set()

        # Per-instance alias map. Previously EXPOSE_ALIASES was a single
        # mutable dict on toolBaseSecured shared by EVERY subclass instance;
        # _make_wrapped_callable mutates it via setdefault(), so the first
        # tool in the process to register a given alias string won globally
        # and any later tool reusing that alias string resolved to the wrong
        # method -> "method 'X' is not exposed" (e.g. entity/hub tools then
        # MetaFrameTool.getRelations). Shadow it with an instance dict seeded
        # from the most-derived class-declared map (PlotService etc. keep
        # their hand-written aliases); the class-level dict is never mutated
        # again, so it stays pristine and tools no longer cross-contaminate.
        self.EXPOSE_ALIASES = dict(type(self).EXPOSE_ALIASES)

        base_names: Set[str] = set(dir(toolBaseSecured)) | set(dir(toolBase))
        for name in dir(self):
            if name.startswith("_"):
                continue
            attr = getattr(self, name)
            if not callable(attr):
                continue

            is_base_member = name in base_names
            try:
                owner = getattr(attr, "__qualname__", "").split(".")[0]
                if owner and owner not in (toolBase.__name__, toolBaseSecured.__name__):
                    is_base_member = False
            except Exception:
                pass
            if is_base_member and not self._strict_base_wrap:
                continue

            if not self._should_expose(name, attr, is_base_member):
                continue

            if not getattr(attr, "_secured_wrapped", False):
                wrapped = self._make_wrapped_callable(attr)
                setattr(self, name, wrapped)
                attr = wrapped

            self._exposed_methods.add(name)

        if self._strict_base_wrap:
            for name in self.STRICT_BASE_WRAP_ALLOWLIST:
                if hasattr(self, name):
                    attr = getattr(self, name)
                    if callable(attr) and not getattr(attr, "_secured_wrapped", False):
                        wrapped = self._make_wrapped_callable(attr)
                        setattr(self, name, wrapped)
                        self._exposed_methods.add(name)

    def _should_expose(self, name: str, attr: Callable, is_base_member: bool) -> bool:
        wants_expose = bool(getattr(attr, "_secure_expose", False))
        if self._exposure_mode == "expose_only":
            return wants_expose or (is_base_member and name in self.STRICT_BASE_WRAP_ALLOWLIST)
        else:
            return (not is_base_member) or (is_base_member and name in self.STRICT_BASE_WRAP_ALLOWLIST)

    def _make_wrapped_callable(self, func: Callable) -> Callable:
        noauth = bool(getattr(func, "_no_auth", False))
        accepts_token = False
        accepts_kwargs = False
        try:
            import inspect
            sig = inspect.signature(func)
            params = sig.parameters.values()
            accepts_token = "token" in sig.parameters
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        except Exception:
            accepts_token = False
            accepts_kwargs = True

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            token = kwargs.get("token")
            if not noauth:
                self._security.verify_or_raise(token)
                # After successful verification, sync username from security manager
                self._username = self._security.get_username()
            if not (accepts_token or accepts_kwargs):
                kwargs.pop("token", None)
            return func(*args, **kwargs)

        setattr(wrapper, "_secured_wrapped", True)
        alias = getattr(func, "_secure_alias", None)
        if alias:
            self.EXPOSE_ALIASES.setdefault(alias, func.__name__)
        return wrapper

    # ---------- Parallel execution ----------
    def executeParallel(self, method_name, params_list, max_workers=None):
        """Parallel execution via invoke(), applying alias resolution + security.

        Args:
            method_name: Name or alias of the method to execute
            params_list: List of parameter dicts [{...}, {...}, ...]
            max_workers: Maximum number of threads (None=auto)

        Returns:
            List of results ordered as the input. Failed tasks contain {"__error": str, "__traceback": str}.
        """
        actual = self.resolve_method_name(method_name)
        return super().executeParallel(actual, params_list, max_workers=max_workers)

    # ---------- LangChain integration ----------
    def toLangchainTools(self) -> list:
        """Generate LangChain StructuredTool list from @secure_expose methods.

        Each exposed alias becomes a StructuredTool with:
        - name: the alias (e.g. "createEntity")
        - description: first line of the method docstring
        - args_schema: dynamically created Pydantic model from type hints + parameterMap
        - func: wrapper that calls self.invoke(alias, **kwargs)
        """
        from langchain.tools import StructuredTool
        from pydantic import create_model, Field
        import inspect
        from typing import get_origin, get_args, Optional as TypingOptional

        _TYPE_MAP = {
            str: str, int: int, float: float, bool: bool,
            dict: dict, list: list,
        }

        tools = []
        for alias, method_name in self.EXPOSE_ALIASES.items():
            method = getattr(self, method_name, None)
            if method is None:
                continue

            # Unwrap functools.wraps to get original signature
            unwrapped = getattr(method, "__wrapped__", method)
            try:
                sig = inspect.signature(unwrapped)
            except (ValueError, TypeError):
                continue

            # Build Pydantic fields from signature
            fields = {}
            for pname, param in sig.parameters.items():
                if pname in ("self", "token"):
                    continue

                annotation = param.annotation
                if annotation is inspect.Parameter.empty:
                    py_type = str
                else:
                    py_type = annotation
                    # Handle Optional[X] / X | None
                    origin = get_origin(py_type)
                    if origin is not None:
                        type_args = get_args(py_type)
                        non_none = [a for a in type_args if a is not type(None)]
                        if non_none:
                            py_type = non_none[0]

                # Normalize to basic Python types for Pydantic
                py_type = _TYPE_MAP.get(py_type, str)

                desc = self.parameterMap.get(pname, "")
                if param.default is not inspect.Parameter.empty:
                    fields[pname] = (TypingOptional[py_type], Field(default=param.default, description=desc))
                else:
                    fields[pname] = (py_type, Field(description=desc))

            InputModel = create_model(f"{alias}_Input", **fields)

            # Closure to capture alias correctly
            def _make_func(a):
                def _tool_func(**kwargs):
                    return self.invoke(a, **kwargs)
                return _tool_func

            doc = (unwrapped.__doc__ or "").strip().split("\n")[0]
            # Append precondition/effect to description for LLM reasoning
            precondition = getattr(unwrapped, "_secure_precondition", None)
            effect_meta = getattr(unwrapped, "_secure_effect", None)
            tool_desc = doc or alias
            if precondition:
                tool_desc += f"\nPrecondition: {precondition}"
            if effect_meta:
                tool_desc += f"\nEffect: {effect_meta}"
            tool = StructuredTool.from_function(
                name=alias,
                func=_make_func(alias),
                description=tool_desc,
                args_schema=InputModel,
            )
            # Store structured metadata for programmatic access by AI planners
            tool.metadata = {
                **(getattr(tool, "metadata", None) or {}),
                "precondition": precondition,
                "effect": effect_meta,
            }
            tools.append(tool)

        return tools

    # ---------- Catalog generation ----------
    def buildCatalog(self, category_map: dict = None) -> list:
        """Build a command catalog from @secure_expose methods.

        Each entry: {id, method, description, category, params: [{name, type, required, description, default?}]}

        Category resolution order:
          1. @secure_expose(category="...") decorator attribute
          2. category_map argument (command_id -> category)
          3. "other" fallback

        This is the introspection counterpart to toLangchainTools().
        """
        import inspect

        _TYPE_MAP = {
            "str": "string", "int": "integer", "float": "number",
            "bool": "boolean", "dict": "object", "list": "array",
            "NoneType": "null",
        }
        category_map = category_map or {}
        commands = []
        param_descriptions = getattr(self, "parameterMap", {})
        param_completions = getattr(self, "completionMap", {})

        for method_name in self.list_exposed_methods():
            method = getattr(self, method_name, None)
            if method is None:
                continue

            # Resolve alias (reverse lookup)
            alias = None
            for a, m in self.EXPOSE_ALIASES.items():
                if m == method_name:
                    alias = a
                    break
            command_id = alias or method_name

            # Introspect signature
            try:
                sig = inspect.signature(method)
            except (ValueError, TypeError):
                continue

            params = []
            for pname, param in sig.parameters.items():
                if pname in ("self", "token"):
                    continue

                annotation = param.annotation
                if annotation is inspect.Parameter.empty:
                    ptype = "string"
                else:
                    type_name = getattr(annotation, "__name__", str(annotation))
                    origin = getattr(annotation, "__origin__", None)
                    if origin is not None:
                        args = getattr(annotation, "__args__", ())
                        non_none = [a for a in args if a is not type(None)]
                        if non_none:
                            type_name = getattr(non_none[0], "__name__", str(non_none[0]))
                    ptype = _TYPE_MAP.get(type_name, "string")

                has_default = param.default is not inspect.Parameter.empty
                if has_default and param.default == "__UNSET__":
                    has_default = True

                param_info = {
                    "name": pname,
                    "type": ptype,
                    "required": not has_default,
                    "description": param_descriptions.get(pname, ""),
                }
                if has_default and param.default is not None and param.default != "__UNSET__":
                    param_info["default"] = param.default

                # Completion hint: check command-specific key first, then param-name default
                comp = param_completions.get(f"{command_id}.{pname}")
                if comp is None:
                    comp = param_completions.get(pname)
                if comp is not None:
                    param_info["completion"] = comp

                params.append(param_info)

            # Resolve category: decorator > external map > "other"
            unwrapped = getattr(method, "__wrapped__", method)
            decorator_cat = getattr(unwrapped, "_secure_category", None)
            cat = decorator_cat or category_map.get(command_id, "other")

            # Precondition / Effect (World Chunk boundary metadata)
            precondition = getattr(unwrapped, "_secure_precondition", None)
            effect_meta = getattr(unwrapped, "_secure_effect", None)

            doc = (method.__doc__ or "").strip()
            if not doc:
                uw = getattr(method, "__wrapped__", None)
                if uw:
                    doc = (uw.__doc__ or "").strip()

            entry = {
                "id": command_id,
                "method": method_name,
                "description": doc,
                "category": cat,
                "params": params,
            }
            if precondition is not None:
                entry["precondition"] = precondition
            if effect_meta is not None:
                entry["effect"] = effect_meta
            commands.append(entry)

        commands.sort(key=lambda c: (c["category"], c["id"]))
        return commands

    # ---------- World Chunk boundary ----------
    def toBoundarySpec(self, category_map: dict = None):
        """Generate a BoundarySpec from this service's @secure_expose catalog."""
        from pythonLibs.tool.BoundarySpec import BoundarySpec
        catalog = self.buildCatalog(category_map=category_map)
        name = getattr(self, "service_name", self.__class__.__name__)
        return BoundarySpec.from_catalog(catalog, service_name=name)

    def toWorldChunk(self, initial_state: dict = None, category_map: dict = None):
        """Create a WorldChunk with this service's boundary and optional initial state."""
        from pythonLibs.tool.BoundarySpec import WorldChunk
        boundary = self.toBoundarySpec(category_map=category_map)
        return WorldChunk(boundary=boundary, state=dict(initial_state or {}))

    # ---------- CLI generation ----------
    def toCLI(self, **kwargs):
        """Create a ServiceREPL with a LocalBackend for this service.

        Usage:
            svc = MyService(state)
            svc.toCLI().run()   # opens interactive REPL
        """
        from pythonLibs.tool.ServiceREPL import ServiceREPL, LocalBackend
        backend = LocalBackend(self)
        return ServiceREPL(backend, **kwargs)

    # ---- Execution flow (adds light-weight audit info) ----
    def execute(self):
        """Run the classic ``functionList`` pipeline after verifying the token; records the executing user."""
        # Early verification before executing functionList (unless globally disabled)
        try:
            self._security.verify_or_raise()
            self._username = self._security.get_username()
        except SecurityVerificationError:
            raise

        # Optionally add executor username into paramDict for audit
        try:
            if "executionResult" not in self.paramDict:
                self.paramDict["executionResult"] = {}
            self.paramDict["executionResult"].update({"_executed_by": self._username})
        except Exception:
            pass

        return super().execute()
