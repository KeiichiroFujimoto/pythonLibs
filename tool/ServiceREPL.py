"""ServiceREPL — Generic interactive CLI for any toolBaseSecured service.

Zero-cost REPL generation from @secure_expose command catalogs.
Works in two modes:
  - Local: ServiceREPL(LocalBackend(service)) — direct Python calls
  - Remote: ServiceREPL(RemoteBackend(url)) — HTTP /api/commands

Usage:
    # Local (via toCLI convenience):
    svc = MyService(state)
    svc.toCLI().run()

    # Remote:
    repl = ServiceREPL(RemoteBackend("http://localhost:8322"))
    repl.run()

    # Subclass for domain-specific features:
    class MyREPL(ServiceREPL):
        def custom_commands(self):
            return {"show": self._cmd_show}
        def extra_completions(self, line, text):
            ...  # domain-aware tab completion
"""
from __future__ import annotations

import json
import sys
from typing import Any, Callable, Protocol


# ---------------------------------------------------------------------------
# Backend Protocol (Strategy pattern)
# ---------------------------------------------------------------------------
class ServiceBackend(Protocol):
    """Abstract interface for REPL backends."""

    def catalog(self) -> list[dict]:
        """Return the command catalog."""
        ...

    def invoke(self, command: str, params: dict) -> Any:
        """Invoke a command with parameters."""
        ...


class LocalBackend:
    """Direct Python backend using a toolBaseSecured instance."""

    def __init__(self, service, category_map: dict = None):
        self._svc = service
        self._category_map = category_map or {}

    def catalog(self) -> list[dict]:
        """Return the service's command catalog (``buildCatalog``)."""
        return self._svc.buildCatalog(category_map=self._category_map)

    def invoke(self, command: str, params: dict) -> Any:
        """Call ``command`` on the service with keyword ``params``."""
        return self._svc.invoke(command, **params)


class CompositeBackend:
    """Merge multiple services into one REPL.

    Usage:
        backend = CompositeBackend([svc_a, svc_b, svc_c])
        ServiceREPL(backend).run()
    """

    def __init__(self, services: list, category_maps: dict = None):
        category_maps = category_maps or {}
        self._backends: list[LocalBackend] = []
        self._dispatch: dict[str, int] = {}
        for svc in services:
            cmap = category_maps.get(type(svc).__name__, {})
            self._backends.append(LocalBackend(svc, category_map=cmap))

    def catalog(self) -> list[dict]:
        """Return the commands of all services and remember which service owns each command."""
        merged = []
        for i, b in enumerate(self._backends):
            for cmd in b.catalog():
                self._dispatch[cmd["id"]] = i
                merged.append(cmd)
        return merged

    def invoke(self, command: str, params: dict) -> Any:
        """Route ``command`` to its service; ``catalog()`` must have been called once before."""
        idx = self._dispatch.get(command)
        if idx is None:
            raise AttributeError(f"Unknown command: {command}")
        return self._backends[idx].invoke(command, params)

class RemoteBackend:
    """HTTP backend calling /api/commands endpoints (urllib only)."""

    def __init__(self, base_url: str):
        from urllib.request import Request, urlopen
        self._base = base_url.rstrip("/")
        self._Request = Request
        self._urlopen = urlopen

    def _headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def _request(self, method: str, path: str, data: dict = None) -> Any:
        url = f"{self._base}{path}"
        body = json.dumps(data).encode() if data else None
        req = self._Request(url, data=body, headers=self._headers(), method=method)
        with self._urlopen(req) as resp:
            return json.loads(resp.read())

    def catalog(self) -> list[dict]:
        """GET ``/api/commands`` and return its ``commands`` list."""
        result = self._request("GET", "/api/commands")
        return result.get("commands", [])

    def invoke(self, command: str, params: dict) -> Any:
        """POST ``{"command", "params"}`` to ``/api/commands/invoke``; return ``result`` or raise on failure."""
        result = self._request("POST", "/api/commands/invoke",
                               {"command": command, "params": params})
        if not result.get("success"):
            raise RuntimeError(f"Server error: {result}")
        return result["result"]

# ---------------------------------------------------------------------------
# Key=Value parser
# ---------------------------------------------------------------------------
def _coerce_value(raw: str, expected_type: str) -> Any:
    """Coerce a string value based on expected JSON schema type."""
    if not raw:
        return raw

    # JSON objects/arrays/literals
    if raw.startswith(("{", "[")) or raw in ("null", "true", "false"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    if expected_type == "integer":
        try:
            return int(raw)
        except ValueError:
            pass
    elif expected_type == "number":
        try:
            return float(raw)
        except ValueError:
            pass
    elif expected_type == "boolean":
        if raw.lower() in ("true", "1", "yes"):
            return True
        if raw.lower() in ("false", "0", "no"):
            return False

    return raw


def _tokenize_kv(s: str) -> list[str]:
    """Split key=value pairs, respecting quotes and JSON braces.

    Unlike shlex.split(), preserves quotes inside {}/[] so JSON objects
    survive intact: position={"x":100} stays as one token.
    """
    tokens = []
    i = 0
    n = len(s)
    current: list[str] = []
    in_quote: str | None = None
    brace_depth = 0

    while i < n:
        ch = s[i]
        if in_quote:
            current.append(ch)
            if ch == in_quote and (i == 0 or s[i - 1] != "\\"):
                in_quote = None
        elif brace_depth > 0:
            current.append(ch)
            if ch in ("{", "["):
                brace_depth += 1
            elif ch in ("}", "]"):
                brace_depth -= 1
        elif ch in ("{", "["):
            brace_depth += 1
            current.append(ch)
        elif ch in ('"', "'") and brace_depth == 0:
            in_quote = ch
            # Skip opening quote for simple "value" unwrapping
            # but keep it if we're building key=value (shlex-like)
            current.append(ch)
        elif ch in (" ", "\t") and brace_depth == 0:
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(ch)
        i += 1

    if current:
        tokens.append("".join(current))
    return tokens


def _strip_quotes(s: str) -> str:
    """Strip matching outer quotes from a string."""
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def _parse_kv_args(args_str: str, param_specs: list[dict] = None) -> dict:
    """Parse 'key=value ...' into {key: value} with type coercion.

    Handles quoted strings, JSON objects, and catalog-driven type coercion.
    Examples:
        name=Sensor stereotype=Block       → {"name": "Sensor", "stereotype": "Block"}
        name="Heat Exchanger"              → {"name": "Heat Exchanger"}
        position={"x":100,"y":200}         → {"position": {"x": 100, "y": 200}}
        max_rows=50                        → {"max_rows": 50}  (if catalog says integer)
    """
    if not args_str.strip():
        return {}

    type_map = {}
    if param_specs:
        for p in param_specs:
            type_map[p["name"]] = p["type"]

    params = {}
    tokens = _tokenize_kv(args_str)

    for token in tokens:
        if "=" not in token:
            continue
        key, _, raw_value = token.partition("=")
        raw_value = _strip_quotes(raw_value)
        params[key] = _coerce_value(raw_value, type_map.get(key, "string"))

    return params


# ---------------------------------------------------------------------------
# Tab Completion
# ---------------------------------------------------------------------------
class _CatalogCompleter:
    """Readline completer backed by ServiceREPL catalog."""

    def __init__(self, repl: ServiceREPL):
        self._repl = repl
        self._matches: list[str] = []

    def complete(self, text: str, state: int) -> str | None:
        if state == 0:
            try:
                import readline
                line = readline.get_line_buffer()
                self._matches = self._generate(line, text)
            except Exception:
                self._matches = []
        try:
            return self._matches[state]
        except IndexError:
            return None

    def _generate(self, line: str, text: str) -> list[str]:
        stripped = line.lstrip()
        parts = stripped.split(None, 1)

        # Check domain-specific completions first
        extra = self._repl.extra_completions(line, text)
        if extra is not None:
            return extra

        # No command yet → complete command names
        if len(parts) == 0 or (len(parts) == 1 and not stripped.endswith(" ")):
            prefix = text.lower()
            return [n + " " for n in self._repl._all_command_names()
                    if n.lower().startswith(prefix)]

        cmd = parts[0]

        # After 'commands' → complete category names
        if cmd == "commands":
            cats = sorted(set(c["category"] for c in self._repl._catalog))
            prefix = text.lower()
            return [c for c in cats if c.lower().startswith(prefix)]

        # After 'describe' → complete command names
        if cmd == "describe":
            prefix = text.lower()
            return [n for n in self._repl._cmd_index.keys()
                    if n.lower().startswith(prefix)]

        # After a catalog command → complete parameter names with '='
        cmd_entry = self._repl._cmd_index.get(cmd)
        if cmd_entry:
            used = set()
            arg_str = parts[1] if len(parts) > 1 else ""
            for token in arg_str.split():
                if "=" in token:
                    used.add(token.split("=")[0])

            prefix = text.lower()
            return [
                p["name"] + "="
                for p in cmd_entry["params"]
                if p["name"].lower().startswith(prefix) and p["name"] not in used
            ]

        return []


# ---------------------------------------------------------------------------
# ServiceREPL
# ---------------------------------------------------------------------------
class ServiceREPL:
    """Generic interactive REPL for any toolBaseSecured service.

    Provides:
      - Auto-generated command names from catalog
      - Tab completion for commands + parameter names
      - Key=value parser with catalog-driven type coercion
      - Extension hooks for domain-specific behavior
    """

    def __init__(self, backend: ServiceBackend, *, prompt: str = None,
                 banner: str = None):
        self._backend = backend
        self._catalog: list[dict] = []
        self._cmd_index: dict[str, dict] = {}
        self._prompt = prompt or "service> "
        self._banner = banner
        self._custom_cmds: dict[str, Callable] = {}
        self._completer = None

    # ---- Catalog Management ----

    def _load_catalog(self) -> None:
        """Fetch catalog from backend and build lookup index."""
        self._catalog = self._backend.catalog()
        self._cmd_index = {cmd["id"]: cmd for cmd in self._catalog}

    def _all_command_names(self) -> list[str]:
        """All completable command names: builtin + catalog + custom."""
        builtin = ["commands", "describe", "help", "refresh", "quit", "exit"]
        catalog_ids = list(self._cmd_index.keys())
        custom = list(self._custom_cmds.keys())
        return sorted(set(builtin + catalog_ids + custom))

    # ---- Built-in Commands ----

    def _cmd_commands(self, args: str) -> None:
        """List available operations, optionally filtered by category."""
        category = args.strip() or None
        groups: dict[str, list] = {}
        for cmd in self._catalog:
            cat = cmd["category"]
            if category and cat != category:
                continue
            groups.setdefault(cat, []).append(cmd)

        if not groups:
            if category:
                print(f"  No commands in category '{category}'.")
                cats = sorted(set(c["category"] for c in self._catalog))
                print(f"  Available categories: {', '.join(cats)}")
            else:
                print("  No commands available.")
            return

        for cat in sorted(groups.keys()):
            cmds = groups[cat]
            print(f"\n  [{cat}] ({len(cmds)} commands)")
            for cmd in cmds:
                desc = cmd["description"].split("\n")[0][:60] if cmd["description"] else ""
                n_params = len(cmd["params"])
                req = sum(1 for p in cmd["params"] if p["required"])
                print(f"    {cmd['id']:<30s} {desc}  ({req}/{n_params} params)")
        print()

    def _cmd_describe(self, args: str) -> None:
        """Show detailed info for a command."""
        cmd_id = args.strip()
        if not cmd_id:
            print("  Usage: describe <command>")
            return

        cmd = self._cmd_index.get(cmd_id)
        if not cmd:
            print(f"  Unknown command: '{cmd_id}'")
            from difflib import get_close_matches
            close = get_close_matches(cmd_id, self._cmd_index.keys(), n=3, cutoff=0.5)
            if close:
                print(f"  Did you mean: {', '.join(close)}?")
            return

        print(f"\n  {cmd['id']}  [{cmd['category']}]")
        if cmd["description"]:
            for line in cmd["description"].split("\n"):
                print(f"  {line}")
        print()
        if cmd["params"]:
            print("  Parameters:")
            for p in cmd["params"]:
                req = "*" if p["required"] else " "
                default = f" = {p.get('default', '')}" if "default" in p else ""
                desc = f"  {p['description']}" if p["description"] else ""
                print(f"    {req} {p['name']:<20s} {p['type']:<8s}{default}{desc}")
        else:
            print("  (no parameters)")

        # Precondition / Effect (World Chunk boundary metadata)
        if cmd.get("precondition"):
            print("  Precondition:")
            for prop, expected in cmd["precondition"].items():
                print(f"    {prop} = {expected}")
        if cmd.get("effect"):
            print("  Effect:")
            for prop, val in cmd["effect"].items():
                print(f"    {prop} → {val}")
        print()

    def _cmd_help(self, args: str) -> None:
        """Show help text."""
        print("\n  Built-in commands:")
        print("    commands [category]      List available operations")
        print("    describe <command>       Show command parameters and types")
        print("    <command> key=value ...  Invoke any catalog command")
        print("    help                     Show this help")
        print("    refresh                  Re-fetch command catalog")
        print("    quit / exit              Exit")

        if self._custom_cmds:
            print("\n  Domain commands:")
            for name, fn in sorted(self._custom_cmds.items()):
                doc = (fn.__doc__ or "").strip().split("\n")[0]
                print(f"    {name:<26s} {doc}")
        print()

    def _cmd_refresh(self, args: str) -> None:
        """Re-fetch the catalog."""
        self._load_catalog()
        self._custom_cmds = self.custom_commands()
        banner = self.on_refresh()
        if banner:
            print(banner)
        print(f"  Refreshed. {len(self._catalog)} commands available.")

    # ---- Invoke a catalog command ----

    def _invoke_command(self, cmd_id: str, args_str: str) -> None:
        """Parse key=value args and invoke a catalog command."""
        cmd = self._cmd_index.get(cmd_id)
        if not cmd:
            return

        params = _parse_kv_args(args_str, cmd.get("params", []))

        try:
            result = self._backend.invoke(cmd_id, params)
        except Exception as e:
            print(f"  Error: {e}")
            return

        # Allow subclass to format
        formatted = self.format_result(cmd_id, result)
        if formatted is not None:
            print(formatted)
        else:
            if isinstance(result, (dict, list)):
                print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
            else:
                print(f"  {result}")

    # ---- Extension Hooks (for subclasses) ----

    def on_connect(self) -> str | None:
        """Called after catalog is loaded. Return a banner string or None."""
        return None

    def on_refresh(self) -> str | None:
        """Called after catalog is refreshed. Return a status string or None."""
        return None

    def custom_commands(self) -> dict[str, Callable]:
        """Return dict of domain-specific commands: name -> handler(args_str)."""
        return {}

    def extra_completions(self, line: str, text: str) -> list[str] | None:
        """Return domain-specific completions, or None to fall through to defaults."""
        return None

    def format_result(self, command: str, result: Any) -> str | None:
        """Return a formatted string for the result, or None for default JSON."""
        return None

    def custom_prompt(self) -> str:
        """Return the prompt string. Override to make it dynamic."""
        return self._prompt

    def implicit_command(self, line: str) -> bool:
        """Handle unrecognized input. Return True if handled, False to show error."""
        return False

    # ---- Readline Setup ----

    def _setup_readline(self) -> None:
        """Configure readline/libedit for tab completion."""
        try:
            import readline
            self._completer = _CatalogCompleter(self)
            readline.set_completer(self._completer.complete)
            readline.set_completer_delims(" \t")
            if "libedit" in (readline.__doc__ or ""):
                readline.parse_and_bind("bind ^I rl_complete")
            else:
                readline.parse_and_bind("tab: complete")
        except Exception:
            self._completer = None

    # ---- Main Loop ----

    def run(self) -> None:
        """Start the interactive REPL loop."""
        from urllib.error import URLError

        # Load catalog
        try:
            self._load_catalog()
        except (URLError, ConnectionError, OSError) as e:
            print(f"Cannot connect: {e}")
            sys.exit(1)

        # Register custom commands
        self._custom_cmds = self.custom_commands()

        # Banner
        if self._banner:
            print(self._banner)
        hook_banner = self.on_connect()
        if hook_banner:
            print(hook_banner)
        print(f"  {len(self._catalog)} commands available. Type 'help' for usage.\n")

        # Setup readline
        self._setup_readline()

        # REPL loop
        while True:
            try:
                line = input(self.custom_prompt()).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not line:
                continue

            parts = line.split(None, 1)
            cmd = parts[0]
            args_str = parts[1] if len(parts) > 1 else ""

            if cmd in ("quit", "exit"):
                print("Bye.")
                break
            elif cmd == "commands":
                self._cmd_commands(args_str)
            elif cmd == "describe":
                self._cmd_describe(args_str)
            elif cmd == "help":
                self._cmd_help(args_str)
            elif cmd == "refresh":
                self._cmd_refresh(args_str)
            elif cmd in self._custom_cmds:
                try:
                    self._custom_cmds[cmd](args_str)
                except Exception as e:
                    print(f"  Error: {e}")
            elif cmd in self._cmd_index:
                self._invoke_command(cmd, args_str)
            else:
                if not self.implicit_command(line):
                    from difflib import get_close_matches
                    all_names = self._all_command_names()
                    close = get_close_matches(cmd, all_names, n=3, cutoff=0.4)
                    if close:
                        print(f"  Unknown command: '{cmd}'. Did you mean: {', '.join(close)}?")
                    else:
                        print(f"  Unknown command: '{cmd}'. Type 'help' for usage.")


# ---------------------------------------------------------------------------
# Standalone CLI entry point
# ---------------------------------------------------------------------------
def main():
    """Run ServiceREPL as a standalone CLI tool.

    Usage:
        python -m pythonLibs.tool.ServiceREPL http://localhost:8322
        python -m pythonLibs.tool.ServiceREPL --url http://localhost:8322 --prompt "beam> "
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generic REPL client for any toolBaseSecured service",
    )
    parser.add_argument("url", nargs="?", default=None,
                        help="Service URL (e.g. http://localhost:8322)")
    parser.add_argument("--url", dest="url_opt", default=None,
                        help="Service URL (alternative to positional)")
    parser.add_argument("--prompt", default=None,
                        help="REPL prompt string (default: derived from URL)")
    args = parser.parse_args()

    url = args.url or args.url_opt
    if not url:
        parser.error("Service URL is required (positional or --url)")

    # Derive prompt from URL if not specified
    prompt = args.prompt
    if not prompt:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = parsed.hostname or "service"
        prompt = f"{host}:{parsed.port}> " if parsed.port else f"{host}> "

    backend = RemoteBackend(url)
    repl = ServiceREPL(backend, prompt=prompt)
    repl.run()


if __name__ == "__main__":
    main()
