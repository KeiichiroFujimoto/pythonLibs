# Tool Registry Architecture

## Overview
The Tool Registry is the architecture for automatically discovering, registering and looking up tools (methods). It is designed so that agents can select and run tools dynamically.

## Design principles

### 1. **Discovery-Driven Design**
- Methods decorated with `@secure_expose` inside modules and classes are discovered automatically
- Both manual registration and automatic discovery are supported

### 2. **Registry Pattern**
- Tools are managed with the tool name as key and the tool class as value
- Looking up and retrieving tools happens in one place

### 3. **Name Resolution**
- Maps the human-readable names agents use to the actual method names
- Aliases allow flexible tool names

## Class structure

### `ToolRegistry` class

**Main features:**

| Method | Description |
|---------|------|
| `register(name, tool_class)` | Register a tool |
| `discover_from_module(module)` | Discover the tools in a module automatically |
| `get_tool(name)` | Get a tool class |
| `list_tools(pattern)` | Get the list of tool names (with pattern matching) |
| `invoke(name, **kwargs)` | Instantiate a tool and run it |

## Usage

### Basic usage

```python
from pythonLibs.tool.ToolRegistry import ToolRegistry

# Create the registry
registry = ToolRegistry()

# Discover a module's tools automatically
registry.discover_from_module(your_tool_module)

# Get a tool
tool_class = registry.get_tool("analyze_data")

# List tools
available_tools = registry.list_tools("*data*")
# -> ["analyze_data", "process_data", "extract_data"]
```

### Custom registration

```python
# Register an existing class manually
registry.register("custom_tool", CustomToolClass)

# Set an alias
# Give CustomToolClass @secure_alias="my_custom_tool";
# it can then also be retrieved with registry.get_tool("my_custom_tool")
```

### Pattern matching

```python
# Get all tools
all_tools = registry.list_tools("*")

# Get only the tools whose names match a pattern
data_tools = registry.list_tools("*data*")
io_tools = registry.list_tools("*io*")
```

## Implementation best practices

### 1. **Automatic module discovery**

```python
def discover_from_module(self, module) -> None:
    """Discover methods decorated with @secure_expose automatically."""
    for name in dir(module):
        if name.startswith("_"):
            continue
        attr = getattr(module, name)
        if hasattr(attr, "_secure_expose"):
            self.register(name, module)
```

### 2. **Error handling in name resolution**

```python
def resolve_method_name(self, name: str) -> str:
    actual = self.EXPOSE_ALIASES.get(name, name)
    if not hasattr(self, "_exposed_methods") or actual not in self._exposed_methods:
        raise AttributeError(f"method '{name}' is not exposed")
    return actual
```

### 3. **Tool execution flow**

```python
def invoke(self, method_name: str, *args, token: Optional[str] = None, **kwargs) -> Any:
    actual = self.resolve_method_name(method_name)
    if token is not None:
        kwargs.setdefault("token", token)
    method = getattr(self, actual)
    return method(*args, **kwargs)
```

## Extensibility

### Custom discovery logic

```python
def discover_from_custom_location(self, location) -> None:
    """Discover tools from a custom location."""
    # Example: discovery from the file system
    # ...
```

### Custom search filters

```python
def list_tools_filtered(self, pattern: str, category: str = None) -> List[str]:
    """Filter by pattern and category."""
    # Example: filtering on a category attribute
    # ...
```

## Related classes

- `toolBaseSecured`: base class of secured tools
- `ToolSecurityManager`: token validation and security management
- `secure_expose` decorator: declares a method as exposed

## See also

- [Tool Selection Concepts](../concepts/tool-selection.md)
