# ServiceREPL — Usage

## Overview

`ServiceREPL` is a framework that generates an interactive CLI, at **zero additional cost**, for any `executionBaseSecured` service whose methods are exposed with the `@secure_expose` decorator.

`toolBaseSecured` still works as a compatibility name, but `executionBaseSecured` is recommended for new code.

```
@secure_expose  ->  five outlets generated automatically
  ├── Web API        buildCatalog()  -> /api/commands
  ├── LLM Agent      toLangchainTools() -> StructuredTool[]
  ├── CLI/REPL       toCLI()         -> ServiceREPL        <- this one
  ├── GUI            Command Palette (Cmd+K)
  └── Multi-Service  CompositeBackend -> several services combined
```

---

## Quick start

### 1. Zero-cost CLI (0 lines of additional code)

```python
from pythonLibs.tool.executionBaseSecured import executionBaseSecured
from pythonLibs.tool.decorators import secure_expose

class MyService(executionBaseSecured):
    class _NoOpAuth:
        pass

    def __init__(self):
        super().__init__(
            secure_enabled=False,
            auth_handler=self._NoOpAuth(),
            exposure_mode="expose_only",
        )

    @secure_expose(alias="greet", category="social")
    def greet(self, name: str = "World") -> dict:
        """Say hello."""
        return {"greeting": f"Hello, {name}!"}

    @secure_expose(alias="add", category="math")
    def add_numbers(self, x: float, y: float) -> dict:
        """Add two numbers."""
        return {"result": x + y}

# This alone gives a working CLI
svc = MyService()
svc.toCLI().run()
```

Run:
```
$ python my_service.py
  2 commands available. Type 'help' for usage.

service> commands
  [math] (1 commands)
    add                            Add two numbers.  (2/2 params)

  [social] (1 commands)
    greet                          Say hello.  (0/1 params)

service> greet name=Alice
{"greeting": "Hello, Alice!"}

service> add x=3 y=4
{"result": 7.0}

service> quit
Bye.
```

### 2. Connect to a remote server (standalone CLI)

```bash
# Connect to the FiTsZ server (port 8322)
python3.12 -m pythonLibs.tool http://localhost:8322

# Customize the prompt
python3.12 -m pythonLibs.tool http://localhost:8322 --prompt "fitsz> "
```

### 3. Combine several services

```python
from pythonLibs.tool import ServiceREPL, CompositeBackend

svc_a = ServiceA()
svc_b = ServiceB()

backend = CompositeBackend([svc_a, svc_b])
ServiceREPL(backend, prompt="combined> ").run()
# -> the commands of both services are available in one REPL
```

---

## Architecture

### Backend Strategy Pattern

```
ServiceBackend (Protocol)
  ├── LocalBackend(service)      direct Python calls
  ├── RemoteBackend(url)         HTTP /api/commands
  └── CompositeBackend([svc...]) merged catalog + automatic dispatch
```

| Backend | Use | Dependencies |
|---------|------|------|
| `LocalBackend` | tests, scripts, no server needed | none |
| `RemoteBackend` | connect to a running server | urllib only |
| `CompositeBackend` | several services in one REPL | none |

### The catalog is the intermediate representation

```python
# Output of buildCatalog() (the common source of every interface)
{
    "id": "createEntity",
    "method": "create_entity",
    "description": "Create a new entity.",
    "category": "entity",
    "params": [
        {"name": "name", "type": "string", "required": True, "description": ""},
        {"name": "stereotype", "type": "string", "required": False, "default": "Block", "description": ""}
    ]
}
```

---

## Built-in commands

| Command | Description |
|----------|------|
| `commands [category]` | List commands (optionally filtered by category) |
| `describe <command>` | Show parameter names, types, required flags, defaults and descriptions |
| `<command> key=value ...` | Run any command |
| `help` | Show help |
| `refresh` | Fetch the catalog again |
| `quit` / `exit` | Exit |

---

## key=value parser

### Basic syntax

```
command name=value stereotype=Block count=5
```

### Type conversion (catalog-driven)

| Catalog type | Input | Result |
|-----------|--------|---------|
| `string` | `name=Sensor` | `"Sensor"` |
| `string` | `name="Heat Exchanger"` | `"Heat Exchanger"` (quotes removed) |
| `integer` | `count=42` | `42` (int) |
| `number` | `ratio=3.14` | `3.14` (float) |
| `boolean` | `flag=true` | `True` |
| `object` | `pos={"x":100,"y":200}` | `{"x": 100, "y": 200}` (dict) |
| `array` | `ids=[1,2,3]` | `[1, 2, 3]` (list) |

### JSON handling

`shlex.split()` breaks JSON quoting, so a custom `_tokenize_kv()` is used.
It tracks brace depth and keeps spaces and quotes inside `{...}` / `[...]`.

---

## Tab completion

| Cursor position | Candidates |
|----------|---------|
| start of line | command names (builtin + catalog + custom) |
| `commands ` | category names |
| `describe ` | command names |
| `createEntity ` | parameter names (`name=`, `stereotype=`) |
| (subclass) | extend freely with `extra_completions()` |

Works with both readline and libedit (macOS libedit is detected automatically).

---

## Extending with a subclass

Six extension hooks:

```python
class MyREPL(ServiceREPL):

    def on_connect(self) -> str | None:
        """Called after the catalog is loaded. Returns a banner string."""
        return f"Connected! {len(self._catalog)} operations."

    def on_refresh(self) -> str | None:
        """Called after refresh."""
        return None

    def custom_commands(self) -> dict[str, Callable]:
        """Register domain-specific commands."""
        return {
            "show": self._cmd_show,
            "list": self._cmd_list,
        }

    def extra_completions(self, line: str, text: str) -> list[str] | None:
        """Domain-specific tab completion. None falls through to the generic one."""
        if line.startswith("show "):
            return [n for n in self.entity_names if n.startswith(text)]
        return None  # fall through

    def format_result(self, command: str, result: Any) -> str | None:
        """Customize the output format. None gives the default JSON."""
        if command == "getItems":
            return "\n".join(f"  {k}: {v}" for k, v in result["items"].items())
        return None

    def custom_prompt(self) -> str:
        """Dynamic prompt."""
        return f"{self.current_project}> "

    def implicit_command(self, line: str) -> bool:
        """Handle unknown input. True = handled, False = show an error."""
        if "=" in line and not line.split()[0] in self._cmd_index:
            # implicit export syntax
            self._handle_export(line)
            return True
        return False
```

---

## API reference

### Classes

| Class | Description |
|--------|------|
| `ServiceREPL(backend, *, prompt=, banner=)` | Generic REPL |
| `LocalBackend(service, category_map=)` | Direct-call backend |
| `RemoteBackend(base_url)` | HTTP backend |
| `CompositeBackend(services, category_maps=)` | Several services combined |

### executionBaseSecured methods

| Method | Description |
|----------|------|
| `svc.buildCatalog(category_map=)` | Build the command catalog |
| `svc.toCLI(**kwargs)` | Return `ServiceREPL(LocalBackend(self))` |

### Parser functions

| Function | Description |
|------|------|
| `_tokenize_kv(s)` | Token splitting aware of JSON and quotes |
| `_strip_quotes(s)` | Remove the outer quotes |
| `_coerce_value(raw, type)` | Convert a value according to the catalog type |
| `_parse_kv_args(args, specs)` | Turn key=value strings into a dict |

### Category precedence

How `buildCatalog()` resolves the category:

1. The `@secure_expose(category="...")` decorator attribute (**highest priority**)
2. The `category_map` argument (when there is no decorator)
3. The `"other"` fallback

---

## Files

```
pythonLibs/tool/
  ├── ServiceREPL.py          main framework (~680 lines)
  ├── executionBaseSecured.py buildCatalog() + toCLI()
  ├── decorators.py           @secure_expose(category=) added
  ├── __init__.py             re-export
  ├── __main__.py             python -m pythonLibs.tool URL
  ├── docs/
  │   └── ServiceREPL.md      this document
  ├── tests/
  │   ├── test_service_repl.py  101 tests
  │   └── RESULTS.md           verification results
  └── examples/
      └── demo_service_repl.py  InventoryService demo
```

---

## Examples

### Demo service

```bash
python3.12 pythonLibs/tool/examples/demo_service_repl.py
python3.12 pythonLibs/tool/examples/demo_service_repl.py --extended
```

### Connect to the FiTsZ server

```bash
# Start the server
python3.12 -m pythonLibs.FiTsZ.backend.server &

# Connect with the CLI
python3.12 -m pythonLibs.tool http://localhost:8322
```

```
localhost:8322> commands entity
  [entity] (7 commands)
    createEntity                   Create a new entity  (1/4 params)
    deleteEntity                   Delete an entity     (1/1 params)
    ...

localhost:8322> describe createEntity
  createEntity  [entity]
  Create a new entity

  Parameters:
    * name                 string
      stereotype           string   = Block
      parent_id            string
      position             object

localhost:8322> createEntity name="Rocket Engine" stereotype=Block
{"entity_id": "Block_1738...", "name": "Rocket Engine", ...}
```

### Running the tests

```bash
python3.12 -m pytest pythonLibs/tool/tests/test_service_repl.py -v
```
