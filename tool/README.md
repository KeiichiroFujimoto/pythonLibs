# tool — toolBase / toolBaseSecured

Write an operation once as a Python method; use it from Python, a JSON command
catalog, REST, an interactive CLI or an LLM agent, with parallel runs, execution
traces and optional access control.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../docs/images/toolBase-dark.svg">
  <img alt="toolBase: one decorated method is exposed on six interfaces" src="../docs/images/toolBase-light.svg">
</picture>

## Quick start

```python
from pythonLibs.tool.decorators import secure_expose
from pythonLibs.tool.toolBaseSecured import toolBaseSecured


class GreetingService(toolBaseSecured):
    """Services are plain classes; expose methods with @secure_expose."""

    def __init__(self):
        super().__init__(instance_subcls=self, exposure_mode="expose_only", secure_enabled=False)

    @secure_expose(alias="greet", category="social")
    def greet(self, name: str = "World") -> dict:
        """Say hello.

        Args:
            name (str): Who to greet.
        """
        return {"greeting": f"Hello, {name}!"}

    @secure_expose(alias="add", category="math")
    def add_numbers(self, x: float, y: float) -> dict:
        """Add two numbers.

        Args:
            x (float): First term.
            y (float): Second term.
        """
        return {"result": x + y}

    def helper(self):
        """Not decorated: not part of the command surface in expose_only mode."""


svc = GreetingService()
print(svc.invoke("greet", name="Ada"))                      # {'greeting': 'Hello, Ada!'}
print(svc.invoke("add", x=2, y=5))                          # {'result': 7}
print(svc.list_exposed_methods())                           # ['add_numbers', 'greet']
print(svc.paramDict["executionResult"]["greet"]["result"])  # last result per method, with its arguments
```

## How a method becomes a command

| Source in your code | Becomes |
|---|---|
| `@secure_expose(alias="greet")` | command id (`invoke("greet")`, CLI command, REST command, tool name) |
| `category="social"` | grouping in the catalog, the CLI `commands` listing and GUIs |
| first docstring line | command description |
| type hints (`str`, `int`, `float`, `bool`, `dict`, `list`) | JSON-schema types `string`, `integer`, `number`, ... |
| default values | optional parameters with defaults |
| `Args:` lines `name (type): text` | parameter descriptions (with `instance_subcls=self`) |
| `precondition=` / `effect=` | STRIPS-style planning metadata (`toBoundarySpec()`) |

Keyword arguments of `invoke` may be written in camelCase or snake_case; they are
matched to the method signature (`eModulus=` reaches `e_modulus`).

Avoid method names that the base class already uses as attributes (`status`, `name`,
`paramDict`, `invoke`, `execute`, ...): the base class assigns them in `__init__`, so
such a method would not be exposed. Choose names like `get_status` instead.

## Interfaces

```python
catalog = svc.buildCatalog()           # list of {id, method, description, category, params: [...]}
results = svc.executeParallel("add", [{"x": 1, "y": 2}, {"x": 3, "y": 4}])   # thread pool, one result per case
spec = svc.toBoundarySpec()            # declarative actions with preconditions / effects
print(len(catalog), results, [a.id for a in spec.actions])
```

| Interface | Call | Notes |
|---|---|---|
| Python | `svc.invoke(alias, **kwargs)` | records `paramDict["executionResult"]` and `executionContext` |
| Catalog | `svc.buildCatalog()` | JSON-ready; `category_map=` regroups commands |
| REST | [examples/restServer.py](examples/restServer.py) | `GET /api/commands`, `POST /api/commands/invoke {"command", "params"}` |
| CLI / REPL | `svc.toCLI().run()` | `commands`, `describe <cmd>`, `cmd key=value ...`, history, completion |
| Remote CLI | `python -m pythonLibs.tool http://host:8322` | any server speaking the REST protocol above |
| LLM agent | `svc.toLangchainTools()` | LangChain `StructuredTool`s with a Pydantic argument model (`llm` extra) |
| Parallel | `svc.executeParallel(alias, paramsList, max_workers=None)` | `ThreadRunner` / `ProcessRunner` in `ToolRunner.py` for task lists |
| Trace | `ToolTrace().attach(svc)` | jobs, events, errors; `to_json()`, `toRuntimeEvents()` |

The CLI is documented in detail in [docs/ServiceREPL.md](docs/ServiceREPL.md)
(custom prompts and commands, combining several services with `CompositeBackend`).

## Execution traces

```python
from pythonLibs.tool.ToolTrace import ToolTrace

trace = ToolTrace()
trace.attach(svc)                      # every invoke of svc is recorded from now on
svc.invoke("add", x=1, y=1)
data = trace.to_dict()                 # {"session_id", "tools", "jobs", "events", ...}
job = data["jobs"][0]
print(job["method"], job["status"], round(job["durationMs"], 3) >= 0)
```

Failed calls are recorded with their error and re-raised. `trace.to_json()` writes
the same structure as a string.

## Access control

Security is off with `secure_enabled=False` (or the environment variable
`TOOLBASE_SECURED_ENABLED=0`). With it on, every exposed call needs a token that
your own verifier accepts; no login or password code ships with this package.

```python
from pythonLibs.tool.ToolSecurityManager import SecurityVerificationError


class ApiKeyVerifier:
    """Any object with verify(token) -> bool; get_username(token) is optional."""

    def verify(self, token):
        return token == "secret-key"

    def get_username(self, token):
        return "analyst"


class SecureService(toolBaseSecured):
    def __init__(self, token=None):
        super().__init__(exposure_mode="expose_only", secure_enabled=True,
                         auth_handler=ApiKeyVerifier(), token=token)

    @secure_expose(alias="getStatus")
    def get_status(self) -> dict:
        """Report the service status."""
        return {"ok": True}


print(SecureService(token="secret-key").invoke("getStatus"))      # {'ok': True}
try:
    SecureService(token="wrong").invoke("getStatus")
except SecurityVerificationError as exc:
    print("rejected:", exc)
```

- `@no_auth` (from `decorators`) exposes a method without token checks, e.g. a health check.
- `exposure_mode="auto_all_public"` (the default) exposes every public method of the
  subclass; `"expose_only"` exposes only `@secure_expose` methods (closed by default).
- `svc.get_user_workdir(caseName)` gives a per-user working directory under
  `workdir_root` (default `./work`, or `TOOL_WORKDIR_ROOT`).

## Units and settings

```python
from pythonLibs.tool.units_decorator import with_units


@with_units(thickness="mm->m", pressure="MPa->Pa")
def hoopStress(radius, thickness, pressure):
    """Arguments arrive converted to SI; the caller keeps engineering units."""
    return pressure * radius / thickness


print(hoopStress(radius=0.5, thickness=5.0, pressure=2.0))     # 2.0e8
```

`SettingsParams.SettingsParamsMixin` turns the parameters a service declares in
`USER_PARAMS` into a settings surface (`paramsSpec`, `getParams`, `setParams`,
`saveParams`, `loadParams` as JSON / TOML / XML) persisted per user by
`ParamConfigStore`.

## Module map

| Module | Contents |
|---|---|
| `toolBaseSecured.py` | `toolBaseSecured`: exposure, verification, `invoke`, catalog, LLM tools, CLI, parallel runs |
| `executionBaseSecured.py`, `executionBase.py` | the same classes under the newer names (preferred for new code) |
| `toolBase.py` | base class: `paramDict`, docstring-driven `parameterMap`, file-status checks, work directories |
| `decorators.py` | `@secure_expose`, `@no_auth` |
| `ToolSecurityManager.py` | adapter around the injected verifier, `SecurityVerificationError` |
| `ServiceREPL.py`, `__main__.py` | interactive CLI, local / remote / composite backends |
| `ToolTrace.py` | execution trace (jobs, events, runtime-event export) |
| `ToolRunner.py` | `ToolTask`, `ThreadRunner`, `ProcessRunner` |
| `BoundarySpec.py` | `BoundarySpec`, `ActionSpec`, `WorldChunk` from catalog preconditions / effects |
| `LabRegistry.py` | process-wide auto-registry of `*Service` subclasses (`collectAll`, `listRegistered`) |
| `SettingsParams.py`, `ParamConfigStore.py` | user settings surface and its per-user persistence |
| `units_decorator.py` | `@with_units` (Pint) |
| `toolChain.py`, `toolFlowBase.py`, `toolMap.py`, `toolInterface*.py`, `toolExecutor.py` | JSON-configured chains of tools and file-based interfaces between them (classic pipeline API) |

## Examples

- [examples/restServer.py](examples/restServer.py) — FastAPI server for any service, usable by the remote CLI
- [examples/demo_service_repl.py](examples/demo_service_repl.py) — the three CLI patterns
- [../regressionHandler/RegressionHandler.py](../regressionHandler/RegressionHandler.py),
  [../fieldMapping/FieldMappingHandler.py](../fieldMapping/FieldMappingHandler.py) — full services built on `toolBaseSecured`
